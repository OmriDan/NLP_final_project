import os
import torch
import pandas as pd
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import EarlyStoppingCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_and_preprocess_data(csv_path):
    """
    Load and preprocess CSV data for classification
    """
    # Load the CSV file
    df = pd.read_csv(csv_path).dropna(subset=['difficulty'])
    # Verify required columns exist
    required_cols = ['question', 'answer', 'difficulty']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")

    # Map difficulty to numeric labels
    label2id = {"easy": 0, "medium": 1, "hard": 2}
    id2label = {0: "easy", 1: "medium", 2: "hard"}
    # Add validation to ensure all difficulty values are valid
    valid_difficulties = set(label2id.keys())
    invalid_difficulties = set(df['difficulty'].unique()) - valid_difficulties
    if invalid_difficulties:
        raise ValueError(f"Found invalid difficulty values: {invalid_difficulties}")

    # Map difficulty to labels with validation
    df['label'] = df['difficulty'].apply(lambda x: label2id.get(x.lower(), -1))
    if (df['label'] < 0).any():
        raise ValueError("Found rows with invalid difficulty values")

    # Debug print some label information
    print(f"Label value counts: {df['label'].value_counts().to_dict()}")
    print(f"Label dtype: {df['label'].dtype}")
    # Ensure difficulty values are lowercase
    df['difficulty'] = df['difficulty'].str.lower()
    df['label'] = df['difficulty'].map(label2id)

    # Combine question and answer into one text field
    df['text'] = df['question'] + " " + df['answer']

    print(f"Loaded {len(df)} examples with label distribution: {df['difficulty'].value_counts().to_dict()}")
    return df, label2id, id2label

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        self.class_weights = class_weights
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def create_and_train_model(df, label2id, id2label, model_name="roberta-base",
                           output_dir="./results", num_epochs=3, batch_size=16):
    """# try "microsoft/deberta-v3-base"
    Create, train and evaluate a text classification model
    """
    # Create a Hugging Face Dataset
    dataset = Dataset.from_pandas(df[['text', 'label']])

    # Split into training and testing
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['label']),
        y=df['label']
    )
    class_weights = torch.FloatTensor(class_weights).to('cuda')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Debug what columns we have
    print(f"Columns after tokenization: {tokenized_train.column_names}")

    train_dataset = tokenized_train.remove_columns(["text"])
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"],
                             dtype=torch.long)  # Use actual torch.long type, not string

    test_dataset = tokenized_test.remove_columns(["text"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"],
                            dtype=torch.long)  # Use actual torch.long type, not string
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True  # Add this for model changes
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,  # Increased learning rate
        per_device_train_batch_size=16,  # Larger batch size if GPU memory allows
        per_device_eval_batch_size=16,
        num_train_epochs=15,  # More epochs
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        save_strategy="epoch",  # Save each epoch
        load_best_model_at_end=True,  # Load best model after training
        metric_for_best_model="f1",  # Optimize for F1
        greater_is_better=True,
        warmup_ratio=0.2,  # Add warmup
    )
    # Define trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # Train model
    print("Training model...")
    trainer.train()

    # Evaluate model
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    return model, tokenizer


def test_classification_examples(model, tokenizer, examples):
    """
    Test the model on specific examples
    """
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    for example in examples:
        results = classifier(example)
        print(f"Input: {example[:100]}...")
        print(f"Prediction: {results}")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classifier for coding problems")
    parser.add_argument("--csv_path", type=str, default="/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/modified_classification_leetcode_df.csv",
                        help="Path to CSV file with question, answer, difficulty columns")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                        help="Base model to use")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save model output")

    args = parser.parse_args()

    # Load and preprocess data
    df, label2id, id2label = load_and_preprocess_data(args.csv_path)

    # Create and train model
    model, tokenizer = create_and_train_model(
        df,
        label2id,
        id2label,
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs
    )

    # Test with some examples
    test_examples = [
        "Question: Write a function to print 'Hello World'. Answer: print('Hello World')",
        "Question: Implement a binary search tree. Answer: class Node: def __init__(self, key): self.left = None self.right = None self.val = key"
    ]

    test_classification_examples(model, tokenizer, test_examples)