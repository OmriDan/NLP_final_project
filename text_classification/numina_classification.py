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
from transformers.trainer_callback import EarlyStoppingCallback
from RAG.retriever import RAGRetriever
from RAG.corpus_utils import prepare_knowledge_corpus

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class RAGAugmentedClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, problems, solutions, labels, tokenizer, retriever, k=5):
        self.problems = problems
        self.solutions = solutions
        self.labels = labels
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.k = k

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        solution = self.solutions[idx]

        # Create classification-specific prompt for NovaSky-AI math problems dataset
        task_prompt = (
            "Task: Classify this mathematical problem's difficulty on a scale from 1 to 10.\n\n"
            "Difficulty Scale Guide:\n"
            "- Level 1-2: Very easy problems requiring only basic arithmetic and simple calculations\n"
            "- Level 3-4: Easy problems using elementary algebra, geometry, or number theory concepts\n"
            "- Level 5-6: Medium difficulty requiring multiple mathematical concepts or standard techniques\n"
            "- Level 7-8: Challenging problems at AMC level requiring insight and mathematical maturity\n"
            "- Level 9-10: Very difficult problems at olympiad level requiring deep mathematical knowledge and creativity\n\n"
            "Evaluation factors: conceptual depth, solution complexity, required background knowledge, insight needed"
        )

        # Step 1: First prioritize the problem (highest priority)
        problem_text = f"Problem: {problem}"
        problem_encoding = self.tokenizer(
            problem_text,
            truncation=True,
            max_length=self.tokenizer.model_max_length // 3,
            return_length=True
        )
        problem_length = problem_encoding["length"][0]
        problem_truncated = self.tokenizer.decode(
            problem_encoding["input_ids"],
            skip_special_tokens=True
        )

        # Step 2: Allocate tokens for the solution (second priority)
        remaining_tokens = self.tokenizer.model_max_length - problem_length - 2
        solution_text = f"Solution: {solution}"
        solution_encoding = self.tokenizer(
            solution_text,
            truncation=True,
            max_length=remaining_tokens // 2,
            return_length=True
        )
        solution_length = solution_encoding["length"][0]
        solution_truncated = self.tokenizer.decode(
            solution_encoding["input_ids"],
            skip_special_tokens=True
        )

        # Step 3: Retrieve context and allocate remaining tokens
        query = f"{problem} {solution}"
        retrieved_docs = self.retriever.retrieve(query, k=self.k)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        # Calculate remaining tokens, including space for the task prompt
        prompt_encoding = self.tokenizer(task_prompt, return_length=True)
        prompt_length = prompt_encoding["length"][0]

        # Use remaining tokens for context
        remaining_tokens = self.tokenizer.model_max_length - problem_length - solution_length - prompt_length - 5
        context_text = f"Retrieved Examples: {context}"
        context_encoding = self.tokenizer(
            context_text,
            truncation=True,
            max_length=remaining_tokens,
            return_length=True
        )
        context_truncated = self.tokenizer.decode(
            context_encoding["input_ids"],
            skip_special_tokens=True
        )

        # Combine components with task prompt first for better instruction following
        text = f"{task_prompt}\n{context_truncated}\n{problem_truncated}\n{solution_truncated}"

        # Final encoding
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        # Convert to appropriate format
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item

# Load the dataset from Hugging Face
print("Loading dataset...")
novadata = load_dataset("NovaSky-AI/labeled_numina_difficulty_859K")
# Define the number of samples you want
N = 50000  # Your desired sample size
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


def train_and_evaluate_model(model_name, dataset, knowledge_corpus=None):
    """Train and evaluate a classification model with RAG if knowledge_corpus is provided"""
    print(f"\n===== Training model: {model_name} =====")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if knowledge_corpus:
        # Using RAG approach
        print("Using RAG augmentation...")
        retriever = RAGRetriever(knowledge_corpus, embedding_model="BAAI/bge-large-en-v1.5")

        # Create RAG-augmented datasets
        train_dataset = RAGAugmentedClassificationDataset(
            problems=dataset["train"]["problem"],
            solutions=dataset["train"]["solution"],
            labels=[int(round(float(d))) - 1 for d in dataset["train"]["gpt_difficulty_parsed"]],
            tokenizer=tokenizer,
            retriever=retriever,
            k=3  # Number of documents to retrieve
        )

        val_dataset = RAGAugmentedClassificationDataset(
            problems=dataset["validation"]["problem"],
            solutions=dataset["validation"]["solution"],
            labels=[int(round(float(d))) - 1 for d in dataset["validation"]["gpt_difficulty_parsed"]],
            tokenizer=tokenizer,
            retriever=retriever,
            k=3
        )
    else:
        # Original approach (without RAG)
        tokenized_dataset = {}
        for split_name, split_data in dataset.items():
            tokenized_dataset[split_name] = split_data.map(
                lambda examples: preprocess_function(examples, tokenizer),
                batched=True,
                remove_columns=split_data.column_names
            )
        train_dataset = tokenized_dataset["train"]
        val_dataset = tokenized_dataset["validation"]

    # Create model for classification (10 classes)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=10, id2label=id2label, label2id=label2id
    )

    # Define model save path
    model_save_path = f"difficulty_classification/models/{model_name.replace('/', '_')}_rag"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=6,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="wandb",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
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
    # "BAAI/bge-small-en-v1.5",       # Small BGE embedding model. less relevant for classification
    # "sentence-transformers/all-MiniLM-L6-v2",  # Compact sentence transformer needs import modification
    # "cross-encoder/ms-marco-MiniLM-L-6-v2",    # QA-specific small model needs import modification
    # "facebook/bart-base",           # Smaller BART model for text understanding

    "distilbert-base-uncased",  # Small and fast
    "google/mobilebert-uncased",  # Very small and fast
    "microsoft/deberta-v3-small",  # DeBERTa small
    "prajjwal1/bert-tiny",  # Tiny BERT (4 layers) bad results (keep)
]

# Large models to test
large_models = [
    # "BAAI/bge-large-en-v1.5",  # Large BGE embedding model. less relevant for classification
    # "sentence-transformers/all-mpnet-base-v2",  # Strong sentence transformer. less relevant for classification
    # "facebook/bart-large",  # Good for text understanding too large
    "google/electra-large-discriminator",  # Strong discriminator model. very bad results
    "bert-large-uncased",  # Larger BERT
    "roberta-large",  # Larger RoBERTa
    "microsoft/deberta-v3-base",  # Base DeBERTa
    "microsoft/deberta-v3-large",  # Large DeBERTa
]

knowledge_corpus = []
programming_datasets = [
    ("codeparrot/apps", "train[:1000]"),                      # Works correctly
    ("open-r1/OpenR1-Math-220k", "train[:1000]"),
    # ("deepmind/math_dataset", "algebra__linear_1d[:1000]"), # Specify config
    ("sciq", "train[:1000]"),                               # Alternative science dataset
    ("NeelNanda/pile-10k", "train[:1000]"),  # Smaller subset of Pile, faster loading
    ("miike-ai/mathqa", "train[:1000]"),  # Math QA dataset without config needs
    ("squad_v2", "train[:1000]"),  # Well-maintained QA dataset
    ("nlile/hendrycks-MATH-benchmark", "train[:1000]"), # Math problems across various subjects
]

for dataset_name, split in programming_datasets:
    print(f"Loading dataset: {dataset_name}")
    dataset_corpus = prepare_knowledge_corpus(dataset_name=dataset_name, split=split)
    knowledge_corpus.extend(dataset_corpus)
print(f'Length of knowledge corpus: {len(knowledge_corpus)}')

# Store results
all_results = {}

# Evaluate small models
print("\n===== EVALUATING SMALL MODELS =====")
for model_name in small_models:
    try:
        results, model_path = train_and_evaluate_model(model_name, sampled_data_dict, knowledge_corpus)
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