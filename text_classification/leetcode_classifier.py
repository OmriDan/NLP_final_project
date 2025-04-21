import os
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, pipeline
import numpy as np
import pandas as pd
from transformers.trainer_callback import EarlyStoppingCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def preprocess_function(examples, tokenizer):
    # Combine question and answer into a single text
    texts = [f"Question: {q} Answer: {a}" for q, a in zip(examples["content"], examples["python"])]

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


def train_and_evaluate_model(model_name, filtered_dataset):
    """Train and evaluate a model from a given checkpoint"""
    print(f"\n===== Training model: {model_name} =====")

    # Load tokenizer for the specific model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and prepare the dataset with the specific tokenizer
    tokenized_dataset = filtered_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True
    )

    # Create train/test split
    tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)

    # Create model for 3-class classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=id2label, label2id=label2id
    )

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define model save path
    model_save_path = f"text_classification/text_classifications_models/{model_name.replace('/', '_')}_leetcode_no_rag"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb reporting
        metric_for_best_model="f1_macro",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
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


# Filter out entries with None difficulty
filtered_leetcode_df = leetcode_df.filter(lambda example: example["difficulty"] is not None)

# List of models to test
models_to_test = [
    "roberta-large",  # Large RoBERTa variant (355M parameters)
    "bert-large-uncased",  # Large BERT variant (345M parameters)
    "distilbert-base-uncased",  # Small and fast (66M parameters)
    "bert-base-uncased",  # Classic BERT (110M parameters)
    "roberta-base",  # Improved BERT variant (125M parameters)
]

# Store results for comparison
all_results = {}

# Train and evaluate each model
for model_name in models_to_test:
    try:
        results, model_path = train_and_evaluate_model(model_name, filtered_leetcode_df)
        all_results[model_name] = {
            "metrics": results,
            "path": model_path
        }
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Print comparison of all models
print("\n===== MODEL COMPARISON =====")
comparison_data = []
for model_name, result in all_results.items():
    metrics = result["metrics"]
    comparison_data.append({
        "model": model_name,
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

# Save results to CSV for easier comparison
results_df = pd.DataFrame(comparison_data)
results_df.to_csv("classification_comparison.csv", index=False)
print(f"Comparison results saved to classification_comparison.csv")