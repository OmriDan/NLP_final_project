import torch
import parse_data
from regression import run_regression
from transformers import AutoTokenizer, T5ForConditionalGeneration
import nltk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from torch.cuda.amp import autocast
import csv

# Download tokenizer models
nltk.download('punkt')

# Model setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "google/flan-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
# TEST: loading model with mixed precision inference
model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device).half()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model.eval()

loss_fn = torch.nn.CrossEntropyLoss()
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.get_device_name(0)}")

def compute_losses(df_subset, batch_size=24):
    """Compute QA cross-entropy losses on a DataFrame subset using Flan-T5."""
    qa_losses = []

    for i in tqdm(range(0, len(df_subset), batch_size), desc="Flan-T5 QA pairs"):
        batch = df_subset.iloc[i:i+batch_size]
        prompts = [f"Write an answer to the following question:\nQuestion: {q}\nAnswer:" for q in batch['question_text']]
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=1024, padding='longest').to(device)
        labels = tokenizer(list(batch['answer_text']), return_tensors="pt", truncation=True, max_length=1024, padding='longest')['input_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
            logits = outputs.logits

        for j in range(len(batch)):
            num_tokens = torch.count_nonzero(labels[j] != tokenizer.pad_token_id)
            if num_tokens <= 0:
                # Assign NaN for empty generations
                qa_losses.append(float('nan'))
            else:
                good_logit = logits[j][:num_tokens]
                good_label = labels[j][:num_tokens]
                qa_losses.append(loss_fn(good_logit, good_label).item())

    df_out = df_subset.copy()
    df_out['loss'] = qa_losses
    mean_loss = df_out['loss'].mean()
    df_out['loss'] = df_out['loss'].fillna(mean_loss)  # Fill NaN with mean loss
    return df_out


def process_and_regress(dataset_type, source, dataset_name, estimator_params={'max_depth': 5, 'n_estimators': 250}):
    """
    Load data (HF dataset or CSV), compute QA losses, perform train/test split, and run regression.

    :param dataset_type: 'dataset' or 'csv'
    :param source: name of the HF dataset or path to CSV
    :param dataset_name: label for outputs
    """
    if dataset_type == 'dataset':
        df = parse_data.load_dataset_to_df(source)
    else:
        df = parse_data.load_csv_to_df(source)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_loss = compute_losses(train_df)
    test_loss = compute_losses(test_df)

    train_loss.to_csv(f"qa_losses_Flan-T5_{dataset_name}_train.csv", quoting=csv.QUOTE_MINIMAL, escapechar="\\", index=False)
    test_loss.to_csv(f"qa_losses_Flan-T5_{dataset_name}_test.csv", quoting=csv.QUOTE_MINIMAL, escapechar="\\", index=False)

    run_regression(
        dataset_name=f"Flan-T5_{dataset_name}",
        estimator=RandomForestRegressor,
        estimator_params=estimator_params,
        train_df=train_loss,
        test_df=test_loss
    )

if __name__ == "__main__":
    #process_and_regress('csv', 'DS_tests_with_difficulty.csv', 'DS')
    #process_and_regress('csv', 'merged_leetcode_df.csv', 'Leetcode')
    process_and_regress('dataset', 'NovaSky-AI/labeled_numina_difficulty_162K', 'NovaSky')