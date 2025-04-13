import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Verifying that torch detects the GPU and CUDA
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Define constants
BERT_MODEL_NAME = "bert-base-uncased"
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load your dataset
df = pd.read_csv("../data/raw/merged_leetcode_df.csv")  # Adjust path as needed
df["combined_text"] = df["content"] + " " + df["python"]
questions = df["combined_text"].tolist()
difficulties = df["normalized_difficulty"].tolist()

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    questions, difficulties, test_size=0.2, random_state=SEED
)

# Define a custom dataset class
class LeetCodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }

# Function to fine-tune a model
def fine_tune_model(model_name, train_texts, train_labels, test_texts, test_labels, is_codebert=False):
    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name) if is_codebert else BertTokenizer.from_pretrained(model_name)
    train_dataset = LeetCodeDataset(train_texts, train_labels, tokenizer)
    test_dataset = LeetCodeDataset(test_texts, test_labels, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1) if is_codebert else BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model.to(device)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Evaluation loop
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.squeeze(-1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    mse = mean_squared_error(true_labels, predictions)
    print(f"Mean Squared Error on Test Set: {mse:.4f}")
    
    # Save the model
    model.save_pretrained(f"../models/{model_name.replace('/', '_')}_fine_tuned")
    tokenizer.save_pretrained(f"../models/{model_name.replace('/', '_')}_fine_tuned")
    return model

# Fine-tune BERT
print("[INFO] Fine-tuning BERT...")
bert_model = fine_tune_model(BERT_MODEL_NAME, train_texts, train_labels, test_texts, test_labels)

# Fine-tune CodeBERT
print("[INFO] Fine-tuning CodeBERT...")
codebert_model = fine_tune_model(CODEBERT_MODEL_NAME, train_texts, train_labels, test_texts, test_labels, is_codebert=True)