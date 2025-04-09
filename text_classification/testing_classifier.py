import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from sklearn.model_selection import train_test_split

# === Step 1: Load and Preprocess Data ===

# Replace 'data.csv' with your CSV file path.
df = pd.read_csv('/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/modified_classification_leetcode_df.csv')

# Assume your CSV has columns: 'question', 'answer', 'difficulty'
# Map difficulty to numeric labels.
label2id = {"easy": 0, "medium": 1, "hard": 2}
df['label'] = df['difficulty'].map(label2id)

# Option A: Combine question and answer into one text field.
df['text'] = df['question'] + " " + df['answer']

# Option B: Alternatively, you can keep separate fields and later modify your model
# to use custom inputs. Here we simply combine.

# Create a Hugging Face Dataset from the Pandas DataFrame.
dataset = Dataset.from_pandas(df[['text', 'label']])

# Split into training and testing subsets.
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# === Step 2: Tokenization ===

# Choose one of the suggested models, e.g., DistilBERT.
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

# Tokenize the datasets.
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove columns that are not model inputs.
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

# Set format for PyTorch.
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# === Step 3: Model Initialization ===

# Initialize a model for sequence classification with 3 labels.
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    problem_type="single_label_classification"  # Explicitly specify the problem type
)
# === Step 4: Fine-tuning with Trainer ===

# Define training arguments.
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",         # Evaluate at the end of each epoch.
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,                  # Increase this number based on your dataset size.
    weight_decay=0.01,
    logging_dir='./logs',                # Directory for storing logs.
    logging_steps=10,
)

# Define a Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model.
trainer.train()

# Evaluate the model.
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# === Step 5: Testing with a Pretrained Model Using Pipeline ===

# You can test your (pre-trained or fine-tuned) model using Hugging Face's pipeline.
# If you want to test before fine-tuning, load the pipeline with the base model.
text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example input to classify.
sample_input = "Question: Print 'hello world' in python. Answer: print('hello world')"
results = text_classifier(sample_input)
print("Pipeline output:", results)
