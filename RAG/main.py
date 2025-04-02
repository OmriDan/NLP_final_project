import os
import pandas as pd
import torch
from corpus_utils import prepare_knowledge_corpus
from pipeline import build_rag_difficulty_regressor
from inference import predict_difficulty_with_rag

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # Load your data
    train_df = pd.read_csv("./data/leetcode/leetcode_train.csv")
    valid_df = pd.read_csv("./data/leetcode/leetcode_val.csv")

    # Load your knowledge corpus - this is what RAG retrieves from
    # This could be a large set of QA pairs, textbooks, or other relevant documents
    with open("RAG/knowledge_corpus.txt", "r") as f:
        knowledge_corpus = f.readlines()

    # Build and train the model
    model_artifacts = build_rag_difficulty_regressor(
        train_df,
        valid_df,
        knowledge_corpus,
        model_name="microsoft/deberta-v3-base"
    )

    # Example prediction
    question = "What is the Pythagorean theorem and how is it used?"
    answer = "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse equals the sum of the squares of the other two sides (a² + b² = c²). It's used to find the length of any side of a right triangle when the other two are known."

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