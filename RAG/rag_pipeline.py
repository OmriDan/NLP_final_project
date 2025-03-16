import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import faiss
import pickle


# RAG components
class Document:
    def __init__(self, doc_id, text, metadata=None):
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata or {}


class RAGRetriever:
    def __init__(self, documents, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.documents = documents
        self.embedding_model = SentenceTransformer(embedding_model)

        # Create FAISS index for fast similarity search
        document_texts = [doc.text for doc in documents]
        self.document_embeddings = self.embedding_model.encode(document_texts)

        # Create FAISS index
        self.dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.array(self.document_embeddings).astype('float32'))

    def retrieve(self, query, k=3):
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.index.search(np.array(query_embedding).astype('float32'), k)

        return [self.documents[idx] for idx in indices[0]]


# Dataset class for RAG-augmented data
class RAGAugmentedDataset(Dataset):
    def __init__(self, questions, answers, retriever, labels=None, tokenizer=None, max_length=512, k=3):
        self.questions = questions
        self.answers = answers
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.retriever = retriever
        self.k = k

        # Pre-compute augmented inputs
        self.augmented_inputs = self._create_augmented_inputs()

    def _create_augmented_inputs(self):
        augmented_inputs = []

        for question, answer in zip(self.questions, self.answers):
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(question, k=self.k)

            # Extract context from retrieved documents
            context = " ".join([doc.text for doc in retrieved_docs])

            # Create augmented input
            augmented_input = {
                "question": question,
                "answer": answer,
                "context": context
            }

            augmented_inputs.append(augmented_input)

        return augmented_inputs

    def __len__(self):
        return len(self.augmented_inputs)

    def __getitem__(self, idx):
        augmented_input = self.augmented_inputs[idx]

        # Combine question, retrieved context, and answer
        text = f"Context: {augmented_input['context']} Question: {augmented_input['question']} Answer: {augmented_input['answer']}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Convert dict of tensors to tensors and remove batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])

        return item


# Model with classification head
class RAGQuestionDifficultyClassifier(torch.nn.Module):
    def __init__(self, encoder_model, num_labels):
        super().__init__()
        self.encoder = encoder_model
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# Main pipeline function
def build_rag_difficulty_classifier(train_df, valid_df, knowledge_corpus, model_name="microsoft/deberta-v3-base",
                                    num_labels=3):
    # Step 1: Prepare the RAG retriever with the knowledge corpus
    documents = [Document(i, text) for i, text in enumerate(knowledge_corpus)]
    retriever = RAGRetriever(documents)

    # Step 2: Initialize tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # Step 3: Create RAG-augmented datasets
    train_dataset = RAGAugmentedDataset(
        questions=train_df['question'].tolist(),
        answers=train_df['answer'].tolist(),
        labels=train_df['difficulty'].tolist(),
        tokenizer=tokenizer,
        retriever=retriever,
        k=3  # Retrieve top 3 documents
    )

    valid_dataset = RAGAugmentedDataset(
        questions=valid_df['question'].tolist(),
        answers=valid_df['answer'].tolist(),
        labels=valid_df['difficulty'].tolist(),
        tokenizer=tokenizer,
        retriever=retriever,
        k=3
    )

    # Step 4: Initialize the model
    model = RAGQuestionDifficultyClassifier(base_model, num_labels)

    # Step 5: Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    # Step 6: Define evaluation metrics
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro")
        }

    # Step 7: Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    # Step 8: Train the model
    trainer.train()

    # Step 9: Save the trained model, tokenizer, and retriever
    model_artifacts = {
        "model": model,
        "tokenizer": tokenizer,
        "retriever": retriever,
        "id2label": {i: label for i, label in enumerate(sorted(set(train_df['difficulty'])))}
    }

    with open("difficulty_classifier_artifacts.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)

    return model_artifacts


# Function to predict difficulty with explanation using the RAG-augmented model
def predict_difficulty_with_rag(question, answer, model_artifacts):
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    retriever = model_artifacts["retriever"]
    id2label = model_artifacts["id2label"]

    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(question, k=3)
    context = " ".join([doc.text for doc in retrieved_docs])

    # Create augmented input
    augmented_input = f"Context: {context} Question: {question} Answer: {answer}"

    # Tokenize
    inputs = tokenizer(
        augmented_input,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs["logits"]
    probabilities = torch.softmax(logits, dim=1)[0]
    pred_class = torch.argmax(logits, dim=1).item()

    difficulty = id2label[pred_class]
    confidence = probabilities[pred_class].item()

    # Generate explanation that includes RAG context
    explanation = generate_rag_explanation(question, answer, context, difficulty, confidence)

    return {
        "difficulty": difficulty,
        "confidence": confidence,
        "explanation": explanation,
        "context_used": context  # Return the context used for transparency
    }


def generate_rag_explanation(question, answer, context, difficulty, confidence):
    # This could be further enhanced with feature importance
    question_length = len(question.split())
    answer_length = len(answer.split())

    # Extract key phrases from the context that might have influenced the prediction
    context_keywords = extract_key_phrases(context, question)

    explanation = f"This question was classified as {difficulty} (confidence: {confidence:.2f}).\n\n"

    # Include context-specific information
    explanation += f"The classification was informed by similar questions in our knowledge base that cover: {', '.join(context_keywords[:3])}.\n\n"

    if difficulty == "easy":
        explanation += f"The question is {question_length} words long and uses straightforward language. "
        explanation += f"The answer is concise ({answer_length} words) and direct."
    elif difficulty == "medium":
        explanation += f"The question contains {question_length} words with moderate complexity. "
        explanation += f"The {answer_length}-word answer requires some domain knowledge."
    else:  # hard
        explanation += f"This {question_length}-word question uses complex concepts or requires deep understanding. "
        explanation += f"The detailed answer ({answer_length} words) demonstrates advanced reasoning."

    return explanation


def extract_key_phrases(context, question, n=5):
    # Simplified key phrase extraction
    # In a real implementation, you might use NLP techniques like TF-IDF or keyword extraction
    all_words = set(context.lower().split())
    question_words = set(question.lower().split())

    # Find common words between context and question
    common_words = all_words.intersection(question_words)

    # If not enough common words, just take some words from the context
    if len(common_words) < n:
        return list(common_words) + list(all_words - common_words)[:n - len(common_words)]

    return list(common_words)[:n]


def enhance_explanation(raw_explanation, difficulty, question, answer):
    # You could use an LLM API call here
    prompt = f"""
    Question: {question}
    Answer: {answer}
    Predicted difficulty: {difficulty}

    Raw assessment: {raw_explanation}

    Please provide a natural, helpful explanation of why this question is classified as {difficulty} difficulty.
    Focus on specific characteristics of the question and answer that indicate this difficulty level.
    """

    # Call your LLM of choice with this prompt
    enhanced_explanation = llm_call(prompt)
    return enhanced_explanation


# Example usage
def main():
    # Load your data
    train_df = pd.read_csv("train_difficulty_data.csv")
    valid_df = pd.read_csv("valid_difficulty_data.csv")

    # Load your knowledge corpus - this is what RAG retrieves from
    # This could be a large set of QA pairs, textbooks, or other relevant documents
    with open("knowledge_corpus.txt", "r") as f:
        knowledge_corpus = f.readlines()

    # Build and train the model
    model_artifacts = build_rag_difficulty_classifier(
        train_df,
        valid_df,
        knowledge_corpus,
        model_name="microsoft/deberta-v3-base",
        num_labels=3
    )

    # Example prediction
    question = "What is the Pythagorean theorem and how is it used?"
    answer = "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse equals the sum of the squares of the other two sides (a² + b² = c²). It's used to find the length of any side of a right triangle when the other two are known."

    result = predict_difficulty_with_rag(question, answer, model_artifacts)
    print(f"Predicted difficulty: {result['difficulty']}")
    print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()