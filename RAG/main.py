import os
import pandas as pd
import torch
import wandb
from corpus_utils import prepare_knowledge_corpus
from pipeline import build_rag_difficulty_regressor
from inference import predict_difficulty_with_rag

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # Wandb configuration
    wandb_project = "rag-difficulty-regressor"
    wandb_run_name = "rag-training-run-1"

    # Optional: Set your wandb API key if not already set via environment variable
    # wandb.login(key="your-api-key-here")

    # Load your data
    train_df = pd.read_csv("./data/leetcode/leetcode_train.csv")
    valid_df = pd.read_csv("./data/leetcode/leetcode_val.csv")

    # Load your knowledge corpus - this is what RAG retrieves from
    # This could be a large set of QA pairs, textbooks, or other relevant documents
    with open("RAG/knowledge_corpus.txt", "r") as f:
        knowledge_corpus = f.readlines()

    # Build and train the model with wandb integration
    model_artifacts = build_rag_difficulty_regressor(
        train_df,
        valid_df,
        knowledge_corpus,
        model_name="microsoft/deberta-v3-base",
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name
    )

    # Example prediction
    question = "What is the list comprehensions and how are they used in Python?"
    answer = "List comprehensions in Python are a concise way to create lists based on existing lists or iterables. The syntax is [expression for item in iterable if condition]. They provide a more readable and efficient alternative to using for loops and append() methods. For example, [x**2 for x in range(10) if x % 2 == 0] creates a list of squares of even numbers from 0 to 9."
    result = predict_difficulty_with_rag(question, answer, model_artifacts)
    print(f"Predicted difficulty: {result['difficulty_score']}")
    print(f"Explanation: {result['explanation']}")

# Rest of the code remains the same