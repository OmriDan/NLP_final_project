import torch
import pickle
import torch.nn.functional as F


def predict_difficulty_with_rag(question, answer, model_artifacts):
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    retriever = model_artifacts["retriever"]

    # Determine model's device
    device = next(model.parameters()).device

    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(question, k=3)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Create augmented input
    augmented_input = f"Context: {context} Question: {question} Answer: {answer}"

    # Tokenize
    inputs = tokenizer(
        augmented_input,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    difficulty_score = outputs["score"][0].item()

    # Generate explanation that includes RAG context
    explanation = generate_rag_explanation(question, answer, retriever, difficulty_score)

    return {
        "difficulty_score": difficulty_score,
        "explanation": explanation,
        "context_used": context
    }


def generate_rag_explanation(question, answer, retriever, difficulty_score):
    question_length = len(question.split())
    answer_length = len(answer.split())
    context_keywords = extract_key_phrases(retriever=retriever, question=question)

    # Convert continuous score to descriptive difficulty
    if difficulty_score < 0.33:
        difficulty_level = "easy"
    elif difficulty_score < 0.67:
        difficulty_level = "medium"
    else:
        difficulty_level = "hard"

    raw_explanation = f"This question was rated at {difficulty_score:.2f} on a scale of 0 (very easy) to 1 (very hard).\n\n"
    raw_explanation += f"The assessment was informed by similar questions that cover: {', '.join(context_keywords[:3])}.\n\n"

    if difficulty_score < 0.33:
        raw_explanation += f"The question is {question_length} words long and uses straightforward language. "
        raw_explanation += f"The answer is concise ({answer_length} words) and direct."
    elif difficulty_score < 0.67:
        raw_explanation += f"The question contains {question_length} words with moderate complexity. "
        raw_explanation += f"The {answer_length}-word answer requires some domain knowledge."
    else:
        raw_explanation += f"This {question_length}-word question uses complex concepts or requires deep understanding. "
        raw_explanation += f"The detailed answer ({answer_length} words) demonstrates advanced reasoning."

    # Call enhance_explanation to improve the raw explanation
    enhanced_explanation = enhance_explanation(raw_explanation, difficulty_level, question, answer)

    return enhanced_explanation


def llm_call(prompt,
             model_artifacts_path=r'/media/omridan/data/work/msc/NLP/NLP_final_project/difficulty_regressor_artifacts.pkl'):
    """
    Generate enhanced explanations using your trained RAG model.
    Takes prompt and model_artifacts as input and returns explanation text.
    """
    if not model_artifacts_path:
        # Fallback to simple template if no model available
        difficulty = prompt.split("Predicted difficulty: ")[1].split("\n")[0]
        return f"Based on analysis, this question is {difficulty} difficulty because of its complexity level and knowledge requirements."

    with open(model_artifacts_path, "rb") as f:
        model_artifacts = pickle.load(f)

    # Extract question and answer from the prompt
    prompt_lines = prompt.strip().split("\n")
    question = prompt_lines[0].replace("Question: ", "")
    answer = prompt_lines[1].replace("Answer: ", "")
    difficulty = prompt_lines[2].replace("Predicted difficulty: ", "")

    # Use our trained model to analyze the content
    tokenizer = model_artifacts["tokenizer"]
    model = model_artifacts["model"]
    retriever = model_artifacts["retriever"]

    # Get relevant documents for context
    retrieved_docs = retriever.retrieve(question, k=3)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Create input for the model
    augmented_input = f"Context: {context} Question: {question} Answer: {answer}"
    inputs = tokenizer(augmented_input, return_tensors="pt", truncation=True, max_length=512)

    # Move to correct device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model's features
    with torch.no_grad():
        outputs = model.encoder(**inputs)
        question_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

    # Compute similarity between question embedding and document embeddings
    doc_embeddings = []
    for doc in retrieved_docs:
        doc_input = tokenizer(doc.page_content, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            doc_output = model.encoder(**doc_input)
            doc_embedding = doc_output.last_hidden_state[:, 0, :]  # [CLS] token
            doc_embeddings.append(doc_embedding)

    similarities = [F.cosine_similarity(question_embedding, doc_embedding).item() for doc_embedding in doc_embeddings]
    most_similar_doc = retrieved_docs[similarities.index(max(similarities))]

    # Generate an explanation based on the most similar document
    question_length = len(question.split())
    answer_length = len(answer.split())

    explanation = f"This {difficulty} difficulty question ({question_length} words) "

    if difficulty.lower() == "easy":
        explanation += f"uses straightforward language and concepts. The {answer_length}-word answer requires basic knowledge and minimal problem-solving."
    elif difficulty.lower() == "medium":
        explanation += f"introduces moderate complexity and requires some domain knowledge. The {answer_length}-word answer demonstrates intermediate reasoning skills."
    else:  # hard
        explanation += f"contains complex concepts that require deep understanding. The {answer_length}-word answer showcases advanced reasoning and specialized knowledge."

    explanation += f"\n\nThe assessment was informed by similar content from the retrieved document: {most_similar_doc.page_content[:200]}..."

    return explanation


def extract_key_phrases(retriever, question, n=5):
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(question, k=n)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Simplified key phrase extraction
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


def load_model_and_run_inference(model_path="difficulty_regressor_artifacts.pkl"):
    # Load saved model artifacts
    with open(model_path, "rb") as f:
        model_artifacts = pickle.load(f)

    # Get model and move to available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_artifacts["model"] = model_artifacts["model"].to(device)
    model_artifacts["model"].eval()  # Set to evaluation mode

    def predict(question, answer):
        return predict_difficulty_with_rag(question, answer, model_artifacts)

    return predict