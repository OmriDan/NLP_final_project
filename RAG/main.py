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
    wandb_run_name = "rag-training-run-1"

    # Load your data
    train_df = pd.read_csv("./data/leetcode/leetcode_train.csv")
    valid_df = pd.read_csv("./data/leetcode/leetcode_val.csv")

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
        ("codeparrot/apps", "train[:2000]"),  # Programming problems
        ("codeparrot/github-jupyter-code-to-text", "train[:500]"),  # Code documentation
        ("open-r1/verifiable-coding-problems-python-10k", "train[:1000]"),  # Python exercises
        ("sahil2801/CodeAlpaca-20k", "train[:500]"),  # Code instruction data
    ]

    # Add CS knowledge and QA datasets
    cs_qa_datasets = [
        ("squad", "train[:1000]"),  # General QA format
        ("habedi/stack-exchange-dataset", "train[:4000]"),  # CS-specific QA from Stack Exchange
        ("ajibawa-2023/WikiHow", "train[:300]"),  # Step-by-step guides
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
        model_name="microsoft/deberta-v3-large",
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


# Example inference-only usage
def inference_demo():
    """Demo function for inference with a trained model"""
    try:
        # Load saved model artifacts
        import pickle
        with open("difficulty_regressor_artifacts.pkl", "rb") as f:
            model_artifacts = pickle.load(f)

        # Example question and answer
        question = "Given an array of integers, find two numbers such that they add up to a specific target."
        answer = "Use a hash map to store values and check for target-num in the map."
        # Predict difficulty
        result = predict_difficulty_with_rag(question, answer, model_artifacts)

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Difficulty Score: {result['difficulty_score']:.2f}")
        print(f"Explanation: {result['explanation']}")

    except FileNotFoundError:
        print("Model artifacts not found. Please train the model first.")


if __name__ == "__main__":
    main()
    inference_demo()