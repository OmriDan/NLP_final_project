import os
import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, pipeline
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
from typing import List, Dict, Any

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = None


class RAGRetriever:
    def __init__(self, documents: List[Document], embedding_model: str = "BAAI/bge-base-en-v1.5"):
        self.documents = documents
        self.embedding_model = SentenceTransformer(embedding_model)

        # Generate embeddings for all documents
        print(f"Generating embeddings for {len(documents)} documents...")
        texts = [doc.content for doc in documents]
        self.document_embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Create FAISS index
        vector_dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        self.index.add(self.document_embeddings)

    def retrieve(self, query: str, k: int = 3):
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([query])

        # Search for similar documents
        scores, indices = self.index.search(query_embedding, k)

        # Return top-k documents
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        return retrieved_docs


def prepare_knowledge_corpus(dataset_name, split):
    """Load and prepare a dataset as a knowledge corpus"""
    print(f"Loading dataset: {dataset_name}, split: {split}")
    try:
        dataset = load_dataset(dataset_name, split=split)
        corpus = []

        for item in dataset:
            # Handle different dataset structures
            if isinstance(item, dict):
                text = ""
                if "text" in item:
                    text = item["text"]
                elif "problem" in item:
                    text = item["problem"]
                elif "question" in item:
                    text = item["question"]
                elif "content" in item:
                    text = item["content"]

                if text:
                    corpus.append(Document(content=text, metadata={"source": dataset_name}))

        print(f"Added {len(corpus)} documents from {dataset_name}")
        return corpus
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return []


def precompute_rag_retrievals(problems, solutions, retriever, k=3):
    """Precompute retrievals for all problems"""
    retrievals = []
    for problem, solution in zip(problems, solutions):
        query = f"Question: {problem} Answer: {solution}"
        retrieved_docs = retriever.retrieve(query, k=k)
        retrievals.append(retrieved_docs)
    return retrievals


# Load the CSV dataset
leetcode_df = load_dataset("csv",
                           data_files="/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/classification_leetcode_df.csv")

# Define label mappings for the three classes
id2label = {0: "easy", 1: "medium", 2: "hard"}
label2id = {"easy": 0, "medium": 1, "hard": 2}

# Load accuracy metric
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_score = accuracy.compute(predictions=predictions, references=labels)

    # Add F1 score calculation
    f1_metric = evaluate.load("f1")
    # Use macro average for multi-class classification
    f1_scores = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    # Add per-class F1 scores
    per_class_f1 = f1_metric.compute(predictions=predictions, references=labels, average=None)

    # Combine metrics
    results = {
        "accuracy": accuracy_score["accuracy"],
        "f1_macro": f1_scores["f1"],
        "f1_easy": per_class_f1["f1"][0],
        "f1_medium": per_class_f1["f1"][1],
        "f1_hard": per_class_f1["f1"][2],
    }
    return results


def preprocess_function(examples, tokenizer, retrievals=None, use_rag=False):
    # Combine question and answer into a single text
    texts = []
    for i, (q, a) in enumerate(zip(examples["content"], examples["python"])):
        if use_rag and retrievals is not None:
            # Include retrieved documents as context
            retrieval_texts = [doc.content for doc in retrievals[i]]
            context = " ".join(retrieval_texts)
            texts.append(f"Context: {context} Question: {q} Answer: {a}")
        else:
            texts.append(f"Question: {q} Answer: {a}")

    # Map difficulty labels to IDs with None handling
    labels = []
    for label in examples["difficulty"]:
        if label is None:
            print(f"Warning: Found None difficulty label, using default")
            # Assign a default label or you could choose to skip this example
            labels.append(0)  # Default to "easy"
        else:
            labels.append(label2id[label.lower()])

    # Tokenize the texts
    tokenized = tokenizer(texts, truncation=True)
    tokenized["labels"] = labels

    return tokenized


def train_and_evaluate_model(model_name, filtered_dataset, train_retrievals=None, val_retrievals=None, use_rag=False):
    """Train and evaluate a model from a given checkpoint"""
    print(f"\n===== Training model: {model_name} with RAG={use_rag} =====")

    # Load tokenizer for the specific model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create train/test split
    dataset_split = filtered_dataset["train"].train_test_split(test_size=0.2)

    # Tokenize and prepare the dataset with the specific tokenizer
    if use_rag and train_retrievals is not None and val_retrievals is not None:
        train_dataset = dataset_split["train"].map(
            lambda examples, idx: preprocess_function(
                examples,
                tokenizer,
                retrievals=[train_retrievals[i] for i in idx],
                use_rag=True
            ),
            batched=True,
            with_indices=True
        )

        test_dataset = dataset_split["test"].map(
            lambda examples, idx: preprocess_function(
                examples,
                tokenizer,
                retrievals=[val_retrievals[i] for i in idx],
                use_rag=True
            ),
            batched=True,
            with_indices=True
        )
    else:
        train_dataset = dataset_split["train"].map(
            lambda examples: preprocess_function(examples, tokenizer, use_rag=False),
            batched=True
        )

        test_dataset = dataset_split["test"].map(
            lambda examples: preprocess_function(examples, tokenizer, use_rag=False),
            batched=True
        )

    # Create model for 3-class classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=id2label, label2id=label2id
    )

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define model save path
    rag_suffix = "_rag" if use_rag else ""
    model_save_path = f"text_classification/text_classifications_models/{model_name.replace('/', '_')}_leetcode{rag_suffix}"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb reporting
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Save the model
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Test with a simple example
    sample_text = "Question: Write a function to find the maximum element in an array Answer: def find_max(arr): return max(arr)"
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    prediction = classifier(sample_text)

    print(f"Model: {model_name}")
    print(f"Evaluation results: {eval_results}")
    print(f"Sample prediction: {prediction}")

    return eval_results, model_save_path


def main(use_rag=True):
    # Filter out entries with None difficulty
    filtered_leetcode_df = leetcode_df.filter(lambda example: example["difficulty"] is not None)

    # List of models to test
    small_models = [
        "distilbert-base-uncased",  # Small and fast
        "google/mobilebert-uncased",  # Very small and fast
        "microsoft/deberta-v3-small",  # DeBERTa small
        "prajjwal1/bert-tiny",  # Tiny BERT (4 layers)
    ]

    # Large models to test
    large_models = [
        "bert-large-uncased",  # Larger BERT
        "roberta-large",  # Larger RoBERTa
        "microsoft/deberta-v3-large",  # Large DeBERTa
        "microsoft/deberta-v3-base",  # Base DeBERTa
    ]
    models_to_test = small_models + large_models

    # Create knowledge corpus from programming datasets
    knowledge_corpus = []
    programming_datasets = [
        ("codeparrot/apps", "train[:500]"),
        ("open-r1/OpenR1-Math-220k", "train[:500]"),
        ("sciq", "train[:500]"),
        ("NeelNanda/pile-10k", "train[:500]"),
        ("miike-ai/mathqa", "train[:500]"),
        ("squad_v2", "train[:500]"),
        ("nlile/hendrycks-MATH-benchmark", "train[:500]"),
    ]

    # Only load corpus if using RAG
    if use_rag:
        for dataset_name, split in programming_datasets:
            dataset_corpus = prepare_knowledge_corpus(dataset_name=dataset_name, split=split)
            knowledge_corpus.extend(dataset_corpus)
        print(f'Length of knowledge corpus: {len(knowledge_corpus)}')

        # Check for precomputed retrievals
        retrievals_file = "precomputed_leetcode_retrievals.pt"
        if os.path.exists(retrievals_file):
            print(f"Loading precomputed retrievals from {retrievals_file}...")
            torch.serialization.add_safe_globals([Document])
            retrievals_data = torch.load(retrievals_file)
            train_retrievals = retrievals_data["train"]
            val_retrievals = retrievals_data["val"]
        else:
            # Create retriever
            print('Creating retriever...')
            retriever = RAGRetriever(knowledge_corpus, embedding_model="BAAI/bge-base-en-v1.5")

            # Split dataset for computing retrievals
            dataset_split = filtered_leetcode_df["train"].train_test_split(test_size=0.2)

            # Precompute retrievals
            print("Precomputing retrievals for training set...")
            train_retrievals = precompute_rag_retrievals(
                problems=dataset_split["train"]["content"],
                solutions=dataset_split["train"]["python"],
                retriever=retriever,
                k=3
            )

            print("Precomputing retrievals for validation set...")
            val_retrievals = precompute_rag_retrievals(
                problems=dataset_split["test"]["content"],
                solutions=dataset_split["test"]["python"],
                retriever=retriever,
                k=3
            )

            # Save retrievals to disk
            print(f"Saving precomputed retrievals to {retrievals_file}...")
            torch.save({"train": train_retrievals, "val": val_retrievals}, retrievals_file)
    else:
        train_retrievals = None
        val_retrievals = None

    # Store results for comparison
    all_results = {}

    # Train and evaluate each model
    for model_name in models_to_test:
        try:
            results, model_path = train_and_evaluate_model(
                model_name,
                filtered_leetcode_df,
                train_retrievals,
                val_retrievals,
                use_rag=use_rag
            )
            model_size = "Small" if model_name in small_models else "Large"
            all_results[model_name] = {
                "metrics": results,
                "path": model_path,
                "size": model_size
            }
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    # Print comparison of all models
    print("\n===== MODEL COMPARISON =====")
    comparison_data = []
    for model_name, result in all_results.items():
        metrics = result["metrics"]
        model_size = result["size"]

        comparison_data.append({
            "model": model_name,
            "model_path": result["path"],
            "size": model_size,
            "accuracy": metrics.get("eval_accuracy", "N/A"),
            "f1_macro": metrics.get("eval_f1_macro", "N/A"),
            "f1_easy": metrics.get("eval_f1_easy", "N/A"),
            "f1_medium": metrics.get("eval_f1_medium", "N/A"),
            "f1_hard": metrics.get("eval_f1_hard", "N/A")
        })

        print(f"{model_name} ({model_size}):")
        print(f"  Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
        print(f"  F1-macro: {metrics.get('eval_f1_macro', 'N/A'):.4f}")
        print(f"  F1-easy: {metrics.get('eval_f1_easy', 'N/A'):.4f}")
        print(f"  F1-medium: {metrics.get('eval_f1_medium', 'N/A'):.4f}")
        print(f"  F1-hard: {metrics.get('eval_f1_hard', 'N/A'):.4f}")

    # Save results to CSV for easier comparison
    rag_suffix = "_rag" if use_rag else ""
    results_df = pd.DataFrame(comparison_data)
    results_df.to_csv(f"leetcode_classification_comparison{rag_suffix}.csv", index=False)
    print(f"Comparison results saved to leetcode_classification_comparison{rag_suffix}.csv")

    # Visualize results (can be added similar to numina_classification.py if needed)


if __name__ == "__main__":
    main(use_rag=True)