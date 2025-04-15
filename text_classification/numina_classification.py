import os
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, pipeline
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the dataset from Hugging Face
print("Loading dataset...")
novadata = load_dataset("NovaSky-AI/labeled_numina_difficulty_859K")
# Define the number of samples you want
N = 30000  # Your desired sample size
total_size = len(novadata["train"])
sample_percentage = N / total_size


# Load random subset directly
sampled_dataset = load_dataset(
    "NovaSky-AI/labeled_numina_difficulty_859K",
    split=f"train[:{N}]"  # e.g., "train[:3.49%]"
)

sampled_data_dict = {
    "train": sampled_dataset
}

# Create a validation split
splits = sampled_dataset.train_test_split(test_size=0.2, seed=42)
sampled_data_dict = {
    "train": splits["train"],
    "validation": splits["test"]
}

print(f"Training set size: {len(sampled_data_dict['train'])}")
print(f"Validation set size: {len(sampled_data_dict['validation'])}")


# Define label mappings (1-10 difficulty levels)
id2label = {i: str(i + 1) for i in range(10)}
label2id = {str(i + 1): i for i in range(10)}

# Load metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy_score = accuracy.compute(predictions=predictions, references=labels)

    # Calculate F1 scores
    f1_macro = f1.compute(predictions=predictions, references=labels, average="macro")
    f1_weighted = f1.compute(predictions=predictions, references=labels, average="weighted")

    # Per-class F1 scores
    per_class_f1 = f1.compute(predictions=predictions, references=labels, average=None)

    # Format per-class metrics for each difficulty level
    per_difficulty_f1 = {}
    for i in range(10):
        level = i + 1
        per_difficulty_f1[f"f1_level_{level}"] = per_class_f1["f1"][i]

    # Combine metrics
    results = {
        "accuracy": accuracy_score["accuracy"],
        "f1_macro": f1_macro["f1"],
        "f1_weighted": f1_weighted["f1"],
        **per_difficulty_f1
    }
    return results


def preprocess_function(examples, tokenizer):
    # Combine problem and solution into a single text
    texts = [f"Problem: {p} Solution: {s}" for p, s in
             zip(examples["problem"], examples["solution"])]

    # Convert difficulties to integers (0-9) for classification
    # First round to nearest integer, then subtract 1 to get 0-9 range
    difficulties = [int(round(float(d))) - 1 for d in examples["gpt_difficulty_parsed"]]

    # Handle any out of range values
    difficulties = [min(9, max(0, d)) for d in difficulties]

    # Tokenize the texts
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = difficulties

    return tokenized


def train_and_evaluate_model(model_name, dataset):
    """Train and evaluate a classification model"""
    print(f"\n===== Training model: {model_name} =====")

    # Load tokenizer for the specific model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and prepare the dataset - process each split separately
    tokenized_dataset = {}
    for split_name, split_data in dataset.items():
        tokenized_dataset[split_name] = split_data.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=split_data.column_names
        )

    # Create model for classification (10 classes)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=10, id2label=id2label, label2id=label2id
    )

    # Define model save path
    model_save_path = f"difficulty_classification/models/{model_name.replace('/', '_')}_nova"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Save the model
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Test with a sample
    sample_text = "Problem: Write a function to find the maximum element in an array Solution: def find_max(arr): return max(arr)"

    # Create classifier pipeline
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    prediction = classifier(sample_text)

    print(f"Model: {model_name}")
    print(f"Evaluation results: {eval_results}")
    print(f"Sample prediction (difficulty 1-10): {prediction}")

    return eval_results, model_save_path


# Small models to test
small_models = [
    "distilbert-base-uncased",  # Small and fast
    "google/mobilebert-uncased",  # Very small and fast
    "microsoft/deberta-v3-small",  # DeBERTa small
    "prajjwal1/bert-tiny",  # Tiny BERT (4 layers)
]

# Large models to test
large_models = [
    "bert-large-uncased",  # Larger BERT
    "roberta-large",  # Larger RoBERTa
    "microsoft/deberta-v3-base",  # Base DeBERTa
    "microsoft/deberta-v3-large",  # Large DeBERTa
]

# Store results
all_results = {}

# Evaluate small models
print("\n===== EVALUATING SMALL MODELS =====")
for model_name in small_models:
    try:
        results, model_path = train_and_evaluate_model(model_name, sampled_data_dict)
        all_results[model_name] = {
            "metrics": results,
            "path": model_path,
            "size": "Small"
        }
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Evaluate large models
print("\n===== EVALUATING LARGE MODELS =====")
for model_name in large_models:
    try:
        results, model_path = train_and_evaluate_model(model_name, sampled_data_dict)
        all_results[model_name] = {
            "metrics": results,
            "path": model_path,
            "size": "Large"
        }
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Build comparison data
print("\n===== MODEL COMPARISON =====")
comparison_data = []
for model_name, result in all_results.items():
    metrics = result["metrics"]
    model_size = result["size"]

    comparison_data.append({
        "model": model_name,
        "size": model_size,
        "accuracy": metrics.get("eval_accuracy", "N/A"),
        "f1_macro": metrics.get("eval_f1_macro", "N/A"),
        "f1_weighted": metrics.get("eval_f1_weighted", "N/A")
    })

    print(f"{model_name} ({model_size}):")
    print(f"  Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
    print(f"  F1 Macro: {metrics.get('eval_f1_macro', 'N/A'):.4f}")
    print(f"  F1 Weighted: {metrics.get('eval_f1_weighted', 'N/A'):.4f}")

# Save results to CSV
results_df = pd.DataFrame(comparison_data)
results_df.to_csv("difficulty_classification_comparison.csv", index=False)
print(f"Comparison results saved to difficulty_classification_comparison.csv")

# Visualize results
plt.figure(figsize=(12, 8))
results_df = results_df.sort_values(by=["size", "f1_macro"])
sns.barplot(x="model", y="f1_macro", hue="size", data=results_df)
plt.title("Model Comparison: F1 Macro Score")
plt.xlabel("Model")
plt.ylabel("F1 Macro (higher is better)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("model_comparison_f1.png")

# Use only numeric columns for mean calculation
numeric_columns = ['accuracy', 'f1_macro', 'f1_weighted']
size_comparison = results_df.groupby("size")[numeric_columns].mean()

print("\n===== SMALL VS LARGE MODELS =====")
print(f"Average metrics by model size:")
print(size_comparison)

# Print best model
best_model = results_df.loc[results_df["f1_macro"].idxmax()]
print(f"\nBest model: {best_model['model']} (F1 Macro: {best_model['f1_macro']:.4f})")