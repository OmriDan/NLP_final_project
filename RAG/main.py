import os
import pytorch_lightning
import pandas as pd
from corpus_utils import prepare_knowledge_corpus
from pipeline import build_rag_difficulty_regressor
from inference import predict_difficulty_with_rag

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    pytorch_lightning.utilities.seed.seed_everything(42)
    # Wandb configuration
    wandb_project = "rag-difficulty-regressor"
    wandb_run_name = "modernBERT-base-embedding"

    # Load your data
    train_df = pd.read_csv("/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/leetcode_train.csv")
    valid_df = pd.read_csv("/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/leetcode_val.csv")

    # Enhanced knowledge corpus with CS and programming datasets
    knowledge_corpus = []

    # Load from local file if exists
    # local_path = "RAG/knowledge_corpus.txt"
    local_path = r''
    if os.path.exists(local_path):
        local_corpus = prepare_knowledge_corpus(file_path=local_path)
        knowledge_corpus.extend(local_corpus)

    # Add programming-specific datasets
    programming_datasets = [
        ("codeparrot/apps", "train[:200]"),  # Programming problems
        ("codeparrot/github-jupyter-code-to-text", "train[:200]"),  # Code documentation
        ("open-r1/verifiable-coding-problems-python-10k", "train[:200]"),  # Python exercises
        ("sahil2801/CodeAlpaca-20k", "train[:100]"),  # Code instruction data
    ]

    # Add CS knowledge and QA datasets
    cs_qa_datasets = [
        ("squad", "train[:200]"),  # General QA format
        ("Kaeyze/computer-science-synthetic-dataset", "train[:300]"),  # CS-specific QA
        ("habedi/stack-exchange-dataset", "train[:200]"),  # CS-specific QA from Stack Exchange
        ("ajibawa-2023/WikiHow", "train[:100]"),  # Step-by-step guides
    ]

    # Combine all datasets
    all_datasets = programming_datasets + cs_qa_datasets

    # Load each dataset
    for dataset_name, split in all_datasets:
        print(f"Loading dataset: {dataset_name}")
        dataset_corpus = prepare_knowledge_corpus(
            dataset_name=dataset_name,
            split=split
        )
        knowledge_corpus.extend(dataset_corpus)

    print(f"Total knowledge corpus documents: {len(knowledge_corpus)}")
    embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1" # A good general-purpose embedding model for retrieval tasks
    embedding_model_name = "BAAI/bge-large-en-v1.5" # Currently one of the highest-performing embedding models for retrieval tasks.
    # It provides better semantic understanding of technical content and
    # would likely improve your RAG system's ability to find relevant context for difficulty estimation
    """
    If computational efficiency is a concern, consider:
    BAAI/bge-small-en-v1.5 - Smaller version with good performance
    ssentence-transformers/multi-qa-distilbert-cos-v1 - More lightweight option still optimized for QA
    """
    # Build and train the model with wandb integration
    model_artifacts = build_rag_difficulty_regressor(
        train_df,
        valid_df,
        knowledge_corpus,
        model_name="answerdotai/ModernBERT-base",
        embedding_model_name=embedding_model_name,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name
    )

    # Example prediction
    question = "What is the list comprehensions and how are they used in Python?"
    answer = "List comprehensions in Python are a concise way to create lists based on existing lists or iterables. The syntax is [expression for item in iterable if condition]. They provide a more readable and efficient alternative to using for loops and append() methods. For example, [x**2 for x in range(10) if x % 2 == 0] creates a list of squares of even numbers from 0 to 9."
    result = predict_difficulty_with_rag(question, answer, model_artifacts)
    print(f"Predicted difficulty: {result['difficulty_score']}")
    print(f"Explanation: {result['explanation']}")


def inference_demo(model_dir=None):
    """Demo function for inference with a trained model"""
    try:
        # If model_dir is not specified, use most recent model
        if model_dir is None:
            model_dirs = sorted([d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))],
                                key=lambda x: os.path.getmtime(os.path.join("models", x)),
                                reverse=True)
            if model_dirs:
                model_dir = os.path.join("models", model_dirs[0])
            else:
                raise FileNotFoundError("No model directories found")

        # Load artifacts from the specified directory
        artifacts_path = os.path.join(model_dir, "difficulty_regressor_artifacts.pkl")

        import pickle
        with open(artifacts_path, "rb") as f:
            model_artifacts = pickle.load(f)

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
        for i in range(len(test_questions)):
            question = test_questions[i]
            answer = test_answers[i]
            result = predict_difficulty_with_rag(question, answer, model_artifacts)

            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Difficulty Score: {result['difficulty_score']:.2f}")
            # print(f"Explanation: {result['explanation']}")

    except FileNotFoundError:
        print(f"Model artifacts not found. Please check the path: {model_dir}")

if __name__ == "__main__":
    main()
    inference_demo()