import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from RAG.retriever import RAGRetriever


def load_knowledge_corpus():
    """Load knowledge corpus from datasets used in training"""
    from RAG.corpus_utils import prepare_knowledge_corpus

    knowledge_corpus = []

    # Programming datasets (same as used in numina_classification)
    programming_datasets = [
        ("codeparrot/apps", "train[:500]"),
        ("open-r1/OpenR1-Math-220k", "train[:500]"),
        ("sciq", "train[:500]"),
        ("NeelNanda/pile-10k", "train[:500]"),
        ("miike-ai/mathqa", "train[:500]"),
        ("squad_v2", "train[:500]"),
        ("nlile/hendrycks-MATH-benchmark", "train[:500]"),
    ]

    # Load and prepare corpus
    for dataset_name, split in tqdm(programming_datasets, desc="Loading datasets"):
        print(f"Loading dataset: {dataset_name}")
        dataset_corpus = prepare_knowledge_corpus(dataset_name=dataset_name, split=split)
        knowledge_corpus.extend(dataset_corpus)

    print(f"Length of knowledge corpus: {len(knowledge_corpus)}")
    return knowledge_corpus

def predict_difficulty(model_folder, csv_path, use_rag=False, batch_size=16, difficulty_mode='numeric'):
    """Run inference using a fine-tuned Numina classification model

    Args:
        model_folder: Path to the model folder
        csv_path: Path to CSV with problem, solution and difficulty columns
        use_rag: Whether to use retrieval augmented generation
        batch_size: Batch size for inference
        difficulty_mode: 'numeric' for 1-10 scale or 'categorical' for easy/medium/hard
    """
    # Set up the model and tokenizer
    print(f"Loading model from {model_folder}")

    try:
        model_abs_path = os.path.abspath(model_folder)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_abs_path,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_abs_path,
            local_files_only=True
        )

        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device= 0 if torch.cuda.is_available() else -1
        )

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


    # Define label mappings based on difficulty mode
    if difficulty_mode == 'numeric':
        # For numeric mode (1-10)
        id2label = {i: str(i + 1) for i in range(10)}
        label2id = {str(i + 1): i for i in range(1, 11)}
    else:
        # For categorical mode (easy, medium, hard)
        id2label = {0: 'easy', 1: 'medium', 2: 'hard'}
        label2id = {'easy': 0, 'medium': 1, 'hard': 2}

    # Load data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names if needed
    if "question" in df.columns and "problem" not in df.columns:
        df = df.rename(columns={"question": "problem"})
    if "answer" in df.columns and "solution" not in df.columns:
        df = df.rename(columns={"answer": "solution"})

    # Initialize RAG retriever if needed
    if use_rag:
        print("Initializing RAG retriever")
        corpus = load_knowledge_corpus()
        retriever = RAGRetriever(corpus, embedding_model="BAAI/bge-large-en-v1.5")

    # Prepare input texts
    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing inputs"):
        if use_rag:
            # Get relevant context
            query = f"Problem: {row['problem']} Solution: {row['solution']}"
            retrieved_docs = retriever.retrieve(query, k=3)
            context = " ".join([doc.page_content for doc in retrieved_docs])
            text = f"Problem: {row['problem']} Solution: {row['solution']} Additional Context: {context}"
        else:
            text = f"Problem: {row['problem']} Solution: {row['solution']}"
        texts.append(text)

    # Run inference using the pipeline
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
        batch_texts = texts[i:i + batch_size]
        batch_results = classifier(batch_texts)
        results.extend(batch_results)

    # Extract predictions and scores
    if difficulty_mode == 'numeric':
        predicted_labels = [int(res['label'].split('_')[-1]) for res in results]
    else:
        # For categorical model, we need to map the label IDs to category names
        predicted_labels = [int(label2id[res['label']]) for res in results]

    confidence_scores = [res['score'] for res in results]

    # Create results dataframe
    results_df = df.copy()
    results_df["predicted_difficulty"] = predicted_labels
    results_df["confidence"] = confidence_scores

    # Calculate metrics if ground truth is available
    if "difficulty" in df.columns:
        true_labels = []
        for label in df["difficulty"]:
            if pd.isna(label):
                true_labels.append(None)
            else:
                if difficulty_mode == 'numeric':
                    # Handle integer input (expected 1-10 in data, model expects 0-9)
                    try:
                        numeric_label = int(float(label))
                        # Convert to 0-9 range for comparing with model output
                        true_labels.append(numeric_label - 1)
                    except:
                        true_labels.append(None)
                else:
                    # Handle categorical input (expected 'easy', 'medium', 'hard')
                    try:
                        if isinstance(label, str) and label.lower() in label2id:
                            true_labels.append(label.lower())
                        else:
                            true_labels.append(None)
                    except:
                        true_labels.append(None)

        def map_to_three_levels(score):
            """Map 0-9 difficulty scores to three levels:
            0-3 -> 1, 4-7 -> 2, 8-9 -> 3"""
            if 0 <= score <= 2:
                return 1
            elif 3 <= score <= 5:
                return 2
            else:  # 8-9
                return 3

        # Filter out None values
        valid_indices = [i for i, label in enumerate(true_labels) if label is not None]
        # Then modify the filtered_true and filtered_pred section
        if difficulty_mode == 'numeric':
            # First get the raw 0-indexed integers
            raw_filtered_true = [true_labels[i] - 1 for i in valid_indices]
            raw_filtered_pred = [predicted_labels[i] for i in valid_indices]

            # Then map them to the 3-level scale
            filtered_true = [map_to_three_levels(score) for score in raw_filtered_true]
            filtered_pred = [map_to_three_levels(score) for score in raw_filtered_pred]

            # Store the original values for reference
            results_df["original_predicted"] = predicted_labels

            # Update the display logic for metrics
            display_true = filtered_true
            display_pred = filtered_pred
            target_names = ["Level 1 (Easy)", "Level 2 (Medium)", "Level 3 (Hard)"]
        else:
            # Categorical mode remains unchanged
            filtered_true = [label2id[true_labels[i]] for i in valid_indices]
            filtered_pred = [predicted_labels[i] for i in valid_indices]

        if filtered_true:
            # Calculate metrics
            if difficulty_mode == 'numeric':
                # For display, convert back to 1-10
                display_true = [l + 1 for l in filtered_true]
                display_pred = [l for l in filtered_pred]
                target_names = [f"Difficulty {i}" for i in range(1, 11)]
            else:
                display_true = filtered_true
                display_pred = filtered_pred
                target_names = ['easy', 'medium', 'hard']

            acc = accuracy_score(filtered_true, filtered_pred)
            f1_macro = f1_score(filtered_true, filtered_pred, average="macro", labels=list(set(filtered_true)))
            f1_weighted = f1_score(filtered_true, filtered_pred, average="weighted", labels=list(set(filtered_true)))

            print("\nPerformance Metrics:")
            print(f"Difficulty Mode: {difficulty_mode}")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Macro: {f1_macro:.4f}")
            print(f"F1 Weighted: {f1_weighted:.4f}")

            # Add detailed classification report
            print("\nClassification Report:")
            print(classification_report(filtered_true, filtered_pred,
                                        target_names=target_names,
                                        labels=list(set(filtered_true))))

            # Store metrics
            metrics = {
                "difficulty_mode": difficulty_mode,
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted
            }

            # Save metrics to a separate file
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f"metrics_{os.path.basename(csv_path)}_{difficulty_mode}_rag_{use_rag}.csv", index=False)

    # Save predictions to CSV
    output_path = f"predictions_{os.path.basename(csv_path)}_{difficulty_mode}_rag_{use_rag}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Numina classification models")
    parser.add_argument("--model_folder", type=str,
                        help="Path to the fine-tuned model folder")
    parser.add_argument("--csv_path", type=str,
                        help="Path to CSV with problem, solution and difficulty columns")
    parser.add_argument("--use_rag", action="store_true",
                        help="Enable RAG for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--difficulty_mode", type=str, choices=['numeric', 'categorical'],
                        help="Difficulty scale: 'numeric' for 1-10 or 'categorical' for easy/medium/hard")

    args = parser.parse_args()

    # Default values for development
    if not args.model_folder:
        args.model_folder = r'/media/omridan/data/work/msc/NLP/NLP_final_project/text_classification/text_classification/text_classifications_models/leetcode_rag/answerdotai_ModernBERT-large_leetcode_rag'
    if not args.csv_path:
        args.csv_path = r'/media/omridan/data/work/msc/NLP/NLP_final_project/data/CS_course/DS_tests_with_difficulty_categorical.csv'
    if not args.batch_size:
        args.batch_size = 4
    if not args.difficulty_mode:
        args.difficulty_mode = 'categorical'

    predict_difficulty(
        model_folder=args.model_folder,
        csv_path=args.csv_path,
        use_rag=True,
        batch_size=args.batch_size,
        difficulty_mode=args.difficulty_mode
    )