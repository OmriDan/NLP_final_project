import os
import re
import wandb
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, pipeline
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers.trainer_callback import EarlyStoppingCallback
from RAG.retriever import RAGRetriever
from RAG.corpus_utils import prepare_knowledge_corpus
from langchain_core.documents.base import Document

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def precompute_rag_retrievals(problems, solutions, retriever, k=2):
    precomputed_retrievals = []
    for problem, solution in tqdm(zip(problems, solutions)):
        query = f"Problem:{problem} Solution:{solution}"
        retrieved_docs = retriever.retrieve(query, k=k)
        precomputed_retrievals.append(retrieved_docs)
    return precomputed_retrievals


class RAGAugmentedClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, problems, solutions, labels, tokenizer, precomputed_retrievals=None, k=3, retriever=None):
        self.problems = problems
        self.solutions = solutions
        self.labels = labels
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.precomputed_retrievals = precomputed_retrievals
        self.k = k

        if precomputed_retrievals is None and retriever is None:
            raise ValueError("Either precomputed_retrievals or retriever must be provided")

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        solution = self.solutions[idx]

        # Get retrieved context
        if self.precomputed_retrievals is not None:
            retrieved_docs = self.precomputed_retrievals[idx]
        else:
            query = f"{problem} {solution}"
            retrieved_docs = self.retriever.retrieve(query, k=self.k)

        # Process context with dynamic allocation based on this specific problem+solution
        instructions = f"Classify the difficulty of the following problem-solution pair. "
        problem_solution = f"Problem: {problem} Solution: {solution} "
        fixed_text = instructions + problem_solution + "Additional Context: "

        # Calculate available tokens
        # fixed_tokens = len(self.tokenizer.encode(fixed_text))
        # Get context from retrieved documents
        context = " ".join([doc.page_content for doc in retrieved_docs[:1]])
        # Create final text
        text = instructions + problem_solution + "**END OF PAIR** Additional Context: " + context
        # total_tokens = len(self.tokenizer.encode(text))
        # Tokenize with truncation as a fallback
        encodings = self.tokenizer(text, truncation=True, max_length=2048)
        # Add labels directly to the encodings dictionary
        encodings["labels"] = torch.tensor(self.labels[idx])
        return encodings


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
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
    tokenized = tokenizer(texts, truncation=True, max_length=1024)
    tokenized["labels"] = difficulties

    return tokenized


def train_and_evaluate_model(model_name, dataset, train_retrievals=None, val_retrievals=None, use_rag=True, weights=None):
    # Define label mappings (1-10 difficulty levels)
    id2label = {i: str(i + 1) for i in range(10)}
    label2id = {str(i + 1): i for i in range(10)}

    # Load metrics
    """Train and evaluate a classification model with optional RAG retrievals"""
    print(f"\n===== Training model: {model_name} =====")
    print(f"Using RAG: {use_rag}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_rag and train_retrievals and val_retrievals:
        # Use RAG-augmented datasets
        train_dataset = RAGAugmentedClassificationDataset(
            problems=dataset["train"]["problem"],
            solutions=dataset["train"]["solution"],
            labels=[int(round(float(d))) - 1 for d in dataset["train"]["gpt_difficulty_parsed"]],
            tokenizer=tokenizer,
            precomputed_retrievals=train_retrievals,
            k=3
        )

        val_dataset = RAGAugmentedClassificationDataset(
            problems=dataset["validation"]["problem"],
            solutions=dataset["validation"]["solution"],
            labels=[int(round(float(d))) - 1 for d in dataset["validation"]["gpt_difficulty_parsed"]],
            tokenizer=tokenizer,
            precomputed_retrievals=val_retrievals,
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
    model_save_path = f"difficulty_classification/models/numina_{model_name.replace('/', '_')}_rag_{use_rag}"

    class WeightedLossTrainer(Trainer):
        def __init__(self, weight=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight = weight

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Added **kwargs to handle extra parameters like num_items_in_batch
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.weight)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=6,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        report_to='none',
        greater_is_better=True,
    )

    # Initialize trainer
    trainer = WeightedLossTrainer(
        weight=weights,
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

def main(use_rag=True):
    print("Loading dataset...")
    # Define the number of samples you want
    N = 800000  # Your desired sample size

    # First load the dataset
    sampled_dataset = load_dataset(
        "NovaSky-AI/labeled_numina_difficulty_859K",
        split=f"train[:]"
    )

    # Then filter based on length criteria
    filtered_dataset = sampled_dataset.filter(
        lambda example: (
                example["problem"] is not None and
                example["solution"] is not None and
                len(example["problem"]) < 1500 and
                len(example["solution"]) < 1500
        )
    )

    # Continue with the filtered dataset instead of the original
    splits = filtered_dataset.train_test_split(test_size=0.2, seed=42)

    # Calculate class weights inversely proportional to frequency
    counts = dict(filtered_dataset.to_pandas()["gpt_difficulty_parsed"].value_counts())
    total_samples = sum(counts.values())
    # Convert to 0-indexed for the model
    class_weights = {i - 1: total_samples / (count * len(counts)) for i, count in counts.items()}

    # Convert to tensor for loss function
    weight = torch.tensor([class_weights[i] for i in range(10)], dtype=torch.float32).to('cuda')
    # Create a validation split
    sampled_data_dict = {
        "train": splits["train"],
        "validation": splits["test"]
    }

    print(f"Training set size: {len(sampled_data_dict['train'])}")
    print(f"Validation set size: {len(sampled_data_dict['validation'])}")

# Small models to test
    small_models = [
        "answerdotai/ModernBERT-base" #(110M parameters)
        # "distilbert-base-uncased",  # Small and fast (66M parameters)
        # "roberta-base",  # Improved BERT variant (125M parameters)
        # "bert-base-uncased",  # Classic BERT (110M parameters)
    ]

    # Large models to test
    large_models = [
        # "bert-large-uncased",  # Large BERT variant (345M parameters)
        # "roberta-large",  # Large RoBERTa variant (355M parameters)
        "answerdotai/ModernBERT-large",  # Large variant of ModernBERT (355M parameters)
    ]

    knowledge_corpus = []
    programming_datasets = [
        ("codeparrot/apps", "train[:500]"),                      # Works correctly
        ("open-r1/OpenR1-Math-220k", "train[:500]"),
        ("sciq", "train[:500]"),                               # Alternative science dataset
        ("NeelNanda/pile-10k", "train[:500]"),  # Smaller subset of Pile, faster loading
        ("miike-ai/mathqa", "train[:500]"),  # Math QA dataset without config needs
        ("squad_v2", "train[:500]"),  # Well-maintained QA dataset
        ("nlile/hendrycks-MATH-benchmark", "train[:500]"), # Math problems across various subjects
    ]

    for dataset_name, split in programming_datasets:
        print(f"Loading dataset: {dataset_name}")
        dataset_corpus = prepare_knowledge_corpus(dataset_name=dataset_name, split=split)
        knowledge_corpus.extend(dataset_corpus)
    print(f'Length of knowledge corpus: {len(knowledge_corpus)}')

    retrievals_file = "precomputed_retrievals.pt"
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

        # Precompute retrievals once before training any models
        print("Precomputing retrievals for training set...")
        train_retrievals = precompute_rag_retrievals(
            problems=sampled_data_dict["train"]["problem"],
            solutions=sampled_data_dict["train"]["solution"],
            retriever=retriever,
            k=3
        )

        print("Precomputing retrievals for validation set...")
        val_retrievals = precompute_rag_retrievals(
            problems=sampled_data_dict["validation"]["problem"],
            solutions=sampled_data_dict["validation"]["solution"],
            retriever=retriever,
            k=3
        )

        # Save retrievals to disk
        print(f"Saving precomputed retrievals to {retrievals_file}...")
        torch.save({"train": train_retrievals, "val": val_retrievals}, retrievals_file)



    # Store results
    all_results = {}
    os.makedirs("numina_temporal_results", exist_ok=True)
    for model_name in large_models + small_models:
        try:
            results, model_path = train_and_evaluate_model(
                model_name,
                sampled_data_dict,
                train_retrievals,
                val_retrievals,
                use_rag=use_rag,
                weights=weight,
            )
            model_size = "Small" if model_name in small_models else "Large"
            all_results[model_name] = {
                "metrics": results,
                "path": model_path,
                "size": model_size
            }
            # Save the current all_results to a text file
            with open(f"numina_temporal_results/all_results_{model_name.replace('/', '_')}_rag_{use_rag}.txt", "w") as f:
                f.write(str(all_results))
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
            "model_path": result["path"],
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

if __name__ == "__main__":
    main(use_rag=False)