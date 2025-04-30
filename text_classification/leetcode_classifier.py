import os
import evaluate
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
from transformers.trainer_callback import EarlyStoppingCallback
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from RAG.retriever import RAGRetriever
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define label mappings
id2label = {0: "easy", 1: "medium", 2: "hard"}
label2id = {"easy": 0, "medium": 1, "hard": 2}

# Load accuracy metric
accuracy = evaluate.load("accuracy")


def prepare_knowledge_corpus(dataset_name=None, split=None):
    """Load and prepare documents for the knowledge corpus"""
    if dataset_name and split:
        try:
            dataset = load_dataset(dataset_name, split=split)
            corpus = []

            # Handle different dataset formats
            if "text" in dataset.column_names:
                corpus.extend(dataset["text"])
            elif "question" in dataset.column_names and "answer" in dataset.column_names:
                corpus.extend([f"Q: {q} A: {a}" for q, a in zip(dataset["question"], dataset["answer"])])
            elif "content" in dataset.column_names:
                corpus.extend(dataset["content"])
            elif "code" in dataset.column_names:
                corpus.extend(dataset["code"])
            elif "instruction" in dataset.column_names and "output" in dataset.column_names:
                corpus.extend([f"{inst} {out}" for inst, out in zip(dataset["instruction"], dataset["output"])])

            return corpus
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []
    return []


def precompute_rag_retrievals(problems, solutions, retriever, k=3):
    """Precompute RAG retrievals for the dataset"""
    retrievals = []
    for problem, solution in zip(problems, solutions):
        query = f"Question: {problem} Answer: {solution}"
        retrieved_docs = retriever.retrieve(query, k=k)
        retrievals.append(retrieved_docs)
    return retrievals


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
        if use_rag and retrievals and i < len(retrievals):
            # Extract text content from Document objects
            instructions = f"Classify the difficulty of the following problem and solution pair. "
            problem_solution = f"Problem: {q} Solution: {a} "
            fixed_text = instructions + problem_solution
            context = " ".join([doc.page_content for doc in retrievals[i][:1]])
            text = fixed_text + "**END OF PAIR** Additional Context: " + context
        else:
            text = f"Question: {q} Answer: {a}"
        texts.append(text)

    # Map difficulty labels to IDs with None handling
    labels = []
    for label in examples["difficulty"]:
        if label is None:
            print(f"Warning: Found None difficulty label, using default")
            labels.append(0)  # Default to "easy"
        else:
            labels.append(label2id[label.lower()])

    # Tokenize the texts
    tokenized = tokenizer(texts, truncation=True, max_length=2048)
    tokenized["labels"] = labels

    return tokenized


def calculate_weighted_f1(dataset=None, class_distribution=None, f1_scores_list=None):
    """
    Calculate weighted F1 scores based on class distribution

    Args:
        dataset: Optional - Dataset containing difficulty labels
        class_distribution: Optional - Tuple (easy_prop, medium_prop, hard_prop)
        f1_scores_list: List of F1 score tuples (easy, medium, hard)

    Returns:
        List of weighted F1 scores
    """
    # Get class distribution either from dataset or use provided distribution
    if class_distribution is None and dataset is not None:
        # Count occurrences of each difficulty
        difficulties = [example["difficulty"].lower() if example["difficulty"] else None
                        for example in dataset["train"]]
        difficulties = [d for d in difficulties if d is not None]

        total = len(difficulties)
        easy_count = difficulties.count("easy")
        medium_count = difficulties.count("medium")
        hard_count = difficulties.count("hard")

        # Calculate proportions
        easy_prop = easy_count / total
        medium_prop = medium_count / total
        hard_prop = hard_count / total

        print(f"Class distribution: easy={easy_prop:.3f}, medium={medium_prop:.3f}, hard={hard_prop:.3f}")
    elif class_distribution is not None:
        easy_prop, medium_prop, hard_prop = class_distribution
    else:
        raise ValueError("Either dataset or class_distribution must be provided")

    # Calculate weighted F1 for each set of F1 scores
    weighted_f1_scores = []
    for f1_scores in f1_scores_list:
        f1_easy, f1_medium, f1_hard = f1_scores
        weighted_f1 = (easy_prop * f1_easy) + (medium_prop * f1_medium) + (hard_prop * f1_hard)
        weighted_f1_scores.append(weighted_f1)

    return weighted_f1_scores

def train_and_evaluate_model(model_name, filtered_dataset, train_retrievals=None, val_retrievals=None, use_rag=False):
    """Train and evaluate a model from a given checkpoint"""
    print(f"\n===== Training model: {model_name} with RAG={use_rag} =====")

    # Load tokenizer for the specific model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create train/test split
    split_dataset = filtered_dataset["train"].train_test_split(test_size=0.2)

    # Tokenize and prepare the dataset with the specific tokenizer
    train_dataset = split_dataset["train"].map(
        lambda examples: preprocess_function(
            examples,
            tokenizer,
            retrievals=train_retrievals if use_rag else None,
            use_rag=use_rag
        ),
        batched=True
    )

    val_dataset = split_dataset["test"].map(
        lambda examples: preprocess_function(
            examples,
            tokenizer,
            retrievals=val_retrievals if use_rag else None,
            use_rag=use_rag
        ),
        batched=True
    )

    # Create model for 3-class classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=id2label, label2id=label2id
    )

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define model save path
    rag_suffix = "_rag" if use_rag else "_no_rag"
    model_save_path = f"text_classification/text_classifications_models/{model_name.replace('/', '_')}_leetcode{rag_suffix}"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb reporting
        metric_for_best_model="f1_macro",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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


def main(use_rag=False):
    # Load the CSV dataset
    leetcode_df = load_dataset("csv",
                               data_files="/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/classification_leetcode_df.csv")

    # Filter out entries with None difficulty
    filtered_leetcode_df = leetcode_df.filter(lambda example: example["difficulty"] is not None)
    f1_scores_list = [
        (0.519, 0.62, 0.347),
        (0.537, 0.645, 0.408),
        (0.526, 0.636, 0.348),
        (0.567, 0.697, 0.43)
    ]
    weighted_f1 = calculate_weighted_f1(dataset=filtered_leetcode_df, f1_scores_list=f1_scores_list)
    print("Weighted F1 scores:", [f"{score:.4f}" for score in weighted_f1])
    # List of models to test
    models_to_test = [
        # "distilbert-base-uncased",  # Small and fast (66M parameters)
        # "bert-base-uncased",  # Classic BERT (110M parameters)
        # "roberta-base",  # Improved BERT variant (125M parameters)
        "answerdotai/ModernBERT-base",  # (110M parameters)
        # "bert-large-uncased",  # Large BERT variant (345M parameters)
        # "roberta-large",  # Large RoBERTa variant (355M parameters)
        "answerdotai/ModernBERT-large",  # Large variant of ModernBERT (355M parameters)
    ]

    knowledge_corpus = []

    # Programming datasets
    programming_datasets = [
        ("codeparrot/apps", "train[:2000]"),  # Programming problems
        ("codeparrot/github-jupyter-code-to-text", "train[:500]"),  # Code documentation
        ("open-r1/verifiable-coding-problems-python-10k", "train[:2000]"),  # Python exercises
        ("sahil2801/CodeAlpaca-20k", "train[:500]"),  # Code instruction data
    ]

    # CS knowledge and QA datasets
    cs_qa_datasets = [
        ("squad", "train[:4000]"),  # General QA format
        ("Kaeyze/computer-science-synthetic-dataset", "train[:6000]"),  # CS-specific QA
        ("habedi/stack-exchange-dataset", "train[:4000]"),  # CS-specific QA from Stack Exchange
        ("ajibawa-2023/WikiHow", "train[:300]"),  # Step-by-step guides
    ]

    # Only load datasets if RAG is enabled
    if use_rag:
        for dataset_name, split in programming_datasets + cs_qa_datasets:
            print(f"Loading dataset: {dataset_name}")
            dataset_corpus = prepare_knowledge_corpus(dataset_name=dataset_name, split=split)
            knowledge_corpus.extend(dataset_corpus)
        print(f'Length of knowledge corpus: {len(knowledge_corpus)}')

        # Precompute retrievals
        retrievals_file = "leetcode_precomputed_retrievals.pt"
        if os.path.exists(retrievals_file):
            print(f"Loading precomputed retrievals from {retrievals_file}...")
            torch.serialization.add_safe_globals([Document])
            retrievals_data = torch.load(retrievals_file)
            train_retrievals = retrievals_data["train"]
            val_retrievals = retrievals_data["val"]
        else:
            # Create retriever once
            print('Creating retriever...')
            retriever = RAGRetriever(knowledge_corpus, embedding_model="BAAI/bge-large-en-v1.5")

            # Split dataset for precomputing retrievals
            split_data = filtered_leetcode_df["train"].train_test_split(test_size=0.2)

            # Precompute retrievals for training and validation sets
            print("Precomputing retrievals for training set...")
            train_retrievals = precompute_rag_retrievals(
                problems=split_data["train"]["content"],
                solutions=split_data["train"]["python"],
                retriever=retriever,
                k=3
            )

            print("Precomputing retrievals for validation set...")
            val_retrievals = precompute_rag_retrievals(
                problems=split_data["test"]["content"],
                solutions=split_data["test"]["python"],
                retriever=retriever,
                k=3
            )

            # Save retrievals to disk
            print(f"Saving precomputed retrievals to {retrievals_file}...")
            torch.save({"train": train_retrievals, "val": val_retrievals}, retrievals_file)
    else:
        train_retrievals = None
        val_retrievals = None

    # Store results
    all_results = {}
    os.makedirs("leetcode_temporal_results", exist_ok=True)

    for model_name in models_to_test:
        try:
            results, model_path = train_and_evaluate_model(
                model_name,
                filtered_leetcode_df,
                train_retrievals,
                val_retrievals,
                use_rag=use_rag,
            )
            all_results[model_name] = {
                "metrics": results,
                "path": model_path
            }
            # Save the current all_results to a text file
            with open(f"leetcode_temporal_results/all_results_{model_name.replace('/', '_')}_rag_{use_rag}.txt",
                      "w") as f:
                f.write(str(all_results))
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    # Print comparison of all models
    print("\n===== MODEL COMPARISON =====")
    comparison_data = []
    for model_name, result in all_results.items():
        metrics = result["metrics"]
        comparison_data.append({
            "model": model_name,
            "model_path": result["path"],
            "rag": "Yes" if use_rag else "No",
            "accuracy": metrics.get("eval_accuracy", "N/A"),
            "f1_macro": metrics.get("eval_f1_macro", "N/A"),
            "f1_easy": metrics.get("eval_f1_easy", "N/A"),
            "f1_medium": metrics.get("eval_f1_medium", "N/A"),
            "f1_hard": metrics.get("eval_f1_hard", "N/A")
        })

        print(f"{model_name}:")
        print(f"  Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
        print(f"  F1-macro: {metrics.get('eval_f1_macro', 'N/A'):.4f}")
        print(f"  F1-easy: {metrics.get('eval_f1_easy', 'N/A'):.4f}")
        print(f"  F1-medium: {metrics.get('eval_f1_medium', 'N/A'):.4f}")
        print(f"  F1-hard: {metrics.get('eval_f1_hard', 'N/A'):.4f}")

    # Save results to CSV
    results_df = pd.DataFrame(comparison_data)
    results_df.to_csv(f"classification_comparison_rag_{use_rag}.csv", index=False)
    print(f"Comparison results saved to classification_comparison_rag_{use_rag}.csv")


if __name__ == "__main__":
    main(use_rag=False)  # Set to True to enable RAG, False to disable