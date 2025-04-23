import os
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
    knowledge_corpus = []

    # Programming datasets (same as used in numina_classification)
    programming_datasets = [
        ("codeparrot/apps", "train[:500]"),
        ("sciq", "train[:500]"),
        ("squad_v2", "train[:500]"),
        ("nlile/hendrycks-MATH-benchmark", "train[:500]")
    ]

    # Load and prepare corpus
    for dataset_name, split in tqdm(programming_datasets, desc="Loading datasets"):
        try:
            dataset = load_dataset(dataset_name, split=split)

            # Extract text based on dataset structure
            if "text" in dataset.column_names:
                knowledge_corpus.extend(dataset["text"])
            elif "question" in dataset.column_names and "answer" in dataset.column_names:
                knowledge_corpus.extend([f"Q: {q} A: {a}" for q, a in zip(dataset["question"], dataset["answer"])])
            elif "content" in dataset.column_names:
                knowledge_corpus.extend(dataset["content"])
            elif "code" in dataset.column_names:
                knowledge_corpus.extend(dataset["code"])
            elif "instruction" in dataset.column_names and "output" in dataset.column_names:
                knowledge_corpus.extend(
                    [f"{inst} {out}" for inst, out in zip(dataset["instruction"], dataset["output"])])

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")

    print(f"Loaded {len(knowledge_corpus)} documents for knowledge corpus")
    return knowledge_corpus


def predict_difficulty(model_folder, csv_path, use_rag=False, batch_size=16):
    """Run inference using a fine-tuned Numina classification model"""
    # Set up the transformers pipeline
    print(f"Loading model from {model_folder}")
    classifier = pipeline(
        "text-classification",
        model=model_folder,
        device=0 if 'cuda' in [str(it).lower() for it in range(0, 8)] else -1
    )

    # Get tokenizer for RAG context preparation
    tokenizer = AutoTokenizer.from_pretrained(model_folder)

    # Map numeric labels (0-9) to difficulty levels (1-10)
    id2label = {i: str(i + 1) for i in range(10)}
    label2id = {str(i + 1): i - 1 for i in range(1, 11)}

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
    predicted_labels = [int(res['label'].split('_')[-1]) for res in results]
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
                # Handle integer or string input
                try:
                    numeric_label = int(float(label))
                    # Numina model expects 0-9 for 1-10
                    true_labels.append(numeric_label - 1)
                except:
                    true_labels.append(None)

        # Filter out None values
        valid_indices = [i for i, label in enumerate(true_labels) if label is not None]
        filtered_true = [true_labels[i] for i in valid_indices]
        filtered_pred = [predicted_labels[i] for i in valid_indices]

        if filtered_true:
            # Calculate metrics
            # Convert to 1-10 for reporting
            display_true = [l + 1 for l in filtered_true]
            display_pred = [l for l in filtered_pred]

            acc = accuracy_score(filtered_true, filtered_pred)
            f1_macro = f1_score(filtered_true, filtered_pred, average="macro")
            f1_weighted = f1_score(filtered_true, filtered_pred, average="weighted")

            print("\nPerformance Metrics:")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Macro: {f1_macro:.4f}")
            print(f"F1 Weighted: {f1_weighted:.4f}")

            # Add detailed classification report
            print("\nClassification Report:")
            target_names = [f"Difficulty {i}" for i in range(1, 11)]
            print(classification_report(filtered_true, filtered_pred,
                                        target_names=target_names))

            # Store metrics
            metrics = {
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted
            }

            # Save metrics to a separate file
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f"metrics_{os.path.basename(csv_path)}_rag_{use_rag}.csv", index=False)

    # Save predictions to CSV
    output_path = f"predictions_{os.path.basename(csv_path)}_rag_{use_rag}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Numina classification models")
    parser.add_argument("--model_folder", required=True,
                        help="Path to the fine-tuned model folder")
    parser.add_argument("--csv_path", required=True,
                        help="Path to CSV with problem, solution and difficulty columns")
    parser.add_argument("--use_rag", action="store_true",
                        help="Enable RAG for inference")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")

    args = parser.parse_args()

    predict_difficulty(
        model_folder=args.model_folder,
        csv_path=args.csv_path,
        use_rag=args.use_rag,
        batch_size=args.batch_size
    )