import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import r2_score, mean_squared_error
from datasets import load_dataset

from retriever import RAGRetriever
from model import RAGQuestionDifficultyRegressor
from dataset import RAGAugmentedDataset
from corpus_utils import prepare_knowledge_corpus


def setup_rag_regressor(model_name="microsoft/deberta-v3-base",
                        embedding_model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                        knowledge_corpus=None):
    """
    Set up the RAG-based difficulty regressor without fine-tuning
    """
    print(f"Loading models:\n- Base: {model_name}\n- Embedding: {embedding_model_name}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # Initialize retriever with knowledge corpus
    if knowledge_corpus is None:
        # Default small corpus if none provided
        default_docs = [
            "Algorithms are step-by-step procedures for solving problems.",
            "Data structures are specialized formats for organizing and storing data.",
            "Time complexity measures how an algorithm's execution time grows with input size.",
            "Dynamic programming solves problems by breaking them down into simpler subproblems.",
            "Graph algorithms process relationships between pairs of objects."
        ]
        from langchain.schema.document import Document
        knowledge_corpus = [Document(page_content=doc) for doc in default_docs]
        print("Using default small knowledge corpus. For better results, provide a domain-specific corpus.")

    retriever = RAGRetriever(knowledge_corpus, embedding_model=embedding_model_name)
    print(f"Initialized retriever with {len(knowledge_corpus)} documents")

    # Create regressor model
    model = RAGQuestionDifficultyRegressor(base_model)

    return model, tokenizer, retriever


def predict_difficulties(questions, answers=None, model=None, tokenizer=None, retriever=None, k=5):
    """
    Predict difficulty scores for a list of coding questions
    """
    if answers is None:
        # Create empty answers if not provided
        answers = [""] * len(questions)

    # Create a dataset for prediction
    dataset = RAGAugmentedDataset(
        questions=questions,
        answers=answers,
        labels=None, # For prediction, labels are not needed
        tokenizer=tokenizer,
        retriever=retriever,
        k=k
    )

    # Set model to evaluation mode
    model.eval()
    # Get the device of the model
    device = next(model.parameters()).device
    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            inputs = dataset[i]
            # Convert to batch format
            batch_inputs = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v]
                            for k, v in inputs.items() if k != 'labels'}

            # Get prediction
            outputs = model(**batch_inputs)
            difficulty_score = outputs[0].item()
            results.append({
                'question': questions[i],
                'answer': answers[i],
                'predicted_difficulty': difficulty_score
            })

    return results


def evaluate_on_dataset(test_df, model, tokenizer, retriever, k=5):
    """
    Evaluate the model on a test dataset with known difficulty scores
    """
    questions = test_df['question'].tolist()
    answers = test_df['answer'].tolist()
    true_scores = test_df['difficulty'].tolist()

    # Make predictions
    predictions = predict_difficulties(questions, answers, model, tokenizer, retriever, k)
    pred_scores = [p['predicted_difficulty'] for p in predictions]

    # Calculate metrics
    mse = mean_squared_error(true_scores, pred_scores)
    rmse = mse ** 0.5
    r2 = r2_score(true_scores, pred_scores)

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG difficulty regressor with pretrained models")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-large",
                        help="HuggingFace model to use")
    parser.add_argument("--embedding_model", type=str,
                        default="BAAI/bge-large-en-v1.5",
                        help="Embedding model for retriever")
    parser.add_argument("--corpus_path", type=str, default=None,
                        help="Path to knowledge corpus file")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Path to test data CSV (with question, answer, difficulty columns)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of documents to retrieve")

    args = parser.parse_args()

    # Build knowledge corpus using the imported function
    knowledge_corpus = []

    # Add corpus from file if provided
    if args.corpus_path:
        file_corpus = prepare_knowledge_corpus(file_path=args.corpus_path)
        knowledge_corpus.extend(file_corpus)

    # Add programming-specific datasets
    programming_datasets = [
        ("codeparrot/apps", "train[:2000]"),  # Programming problems
        ("codeparrot/github-jupyter-code-to-text", "train[:500]"),  # Code documentation
        ("open-r1/verifiable-coding-problems-python-10k", "train[:2000]"),  # Python exercises
        ("sahil2801/CodeAlpaca-20k", "train[:500]"),  # Code instruction data
    ]

    # Add CS knowledge and QA datasets
    cs_qa_datasets = [
        ("Kaeyze/computer-science-synthetic-dataset", "train[:6000]"),  # CS-specific QA
        ("squad", "train[:4000]"),  # General QA format
        ("habedi/stack-exchange-dataset", "train[:4000]"),  # CS-specific QA from Stack Exchange
        ("ajibawa-2023/WikiHow", "train[:300]"),  # Step-by-step guides
    ]

    # Combine all datasets
    hf_datasets = programming_datasets + cs_qa_datasets

    # Load each dataset using the imported function
    for dataset_name, split in hf_datasets:
        print(f"Loading dataset: {dataset_name}")
        dataset_corpus = prepare_knowledge_corpus(
            dataset_name=dataset_name,
            split=split
        )
        knowledge_corpus.extend(dataset_corpus)

    print(f"Total knowledge corpus documents: {len(knowledge_corpus)}")

    if not knowledge_corpus:
        knowledge_corpus = None  # Will use default small corpus

    # Set up the regressor
    model, tokenizer, retriever = setup_rag_regressor(
        model_name=args.model,
        embedding_model_name=args.embedding_model,
        knowledge_corpus=knowledge_corpus
    )

    # Example questions to test
    test_questions = [
        "Print Hello World in Python.",
        "Write a function to find the maximum subarray sum.",
        "Implement a function to check if a string is a palindrome.",
        "Create a function to find the nth Fibonacci number using dynamic programming.",
        "Implement a depth-first search algorithm for a graph."
    ]
    # Example answers
    test_answers = [
        "print('Hello World')",
        "def max_subarray_sum(arr): return max(sum(arr[i:j]) for i in range(len(arr)) for j in range(i+1, len(arr)+1))",
        "def is_palindrome(s): return s == s[::-1]",
        "def fibonacci(n): if n <= 1: return n; return fibonacci(n-1) + fibonacci(n-2)",
        "def dfs(graph, start): visited = set(); stack = [start]; while stack: vertex = stack.pop(); if vertex not in visited: visited.add(vertex); stack.extend(set(graph[vertex]) - visited)"
    ]

    # Run predictions on examples
    print("\nRunning predictions on example questions:")
    results = predict_difficulties(test_questions, test_answers, model=model, tokenizer=tokenizer, retriever=retriever, k=args.k)

    # Print results
    for result in results:
        print(f"\nQuestion: {result['question']}")
        print(f"Predicted difficulty: {result['predicted_difficulty']:.4f}")

    # If test data is provided, evaluate on it
    if args.test_path and os.path.exists(args.test_path):
        print("\n\nEvaluating on test dataset:")
        test_df = pd.read_csv(args.test_path)
        metrics = evaluate_on_dataset(test_df, model, tokenizer, retriever, k=args.k)

        print(f"Test MSE: {metrics['mse']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        print(f"Test R²: {metrics['r2']:.4f}")