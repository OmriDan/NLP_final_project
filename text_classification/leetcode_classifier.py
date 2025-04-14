import os
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the CSV dataset
leetcode_df = load_dataset("csv", data_files="/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/classification_leetcode_df.csv")

# Define label mappings for the three classes
id2label = {0: "easy", 1: "medium", 2: "hard"}
label2id = {"easy": 0, "medium": 1, "hard": 2}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
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


# Tokenize and prepare the dataset
filtered_leetcode_df = leetcode_df.filter(lambda example: example["difficulty"] is not None)
tokenized_leetcode = filtered_leetcode_df.map(preprocess_function, batched=True)
# Always create a train/test split from the train dataset
tokenized_leetcode = tokenized_leetcode["train"].train_test_split(test_size=0.2)

# Split the dataset into train and test if it doesn't already have splits
if "train" not in tokenized_leetcode:
    tokenized_leetcode = tokenized_leetcode["train"].train_test_split(test_size=0.2)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Create model for 3-class classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="leetcode_difficulty_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_leetcode["train"],
    eval_dataset=tokenized_leetcode["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
model_path = r'text_classification/text_classifications_models/leetcode_difficulty_model'
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example prediction
sample_text = "Question: Write a function to find the maximum element in an array Answer: def find_max(arr): return max(arr)"
print(classifier(sample_text))

# Load the model from the saved directory with the base architecture specified
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

saved_model = AutoModelForSequenceClassification.from_pretrained(model_path)
saved_tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline("text-classification", model=saved_model, tokenizer=saved_tokenizer)
print(classifier(sample_text))