import torch
from torch.cuda.amp import autocast
import parse_data
from regression import run_regression
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import nltk

# FAIRLY CERTAIN THIS FILE DOES NOT WORK PROPERLY

# Download NLTK tokenizer data
nltk.download('punkt')

# Model and tokenizer setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# Load model in half precision for faster inference
model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device).half()
model.eval()
torch.backends.cudnn.benchmark = True

# Loss function
torch.set_grad_enabled(False)
loss_fn = torch.nn.CrossEntropyLoss()


def compute_losses(indices, encodings, label_ids, batch_size=24):
    """
    Compute QA cross-entropy losses for examples indexed by `indices` using pre-tokenized data.
    Returns a list of per-example losses.
    """
    qa_losses = []
    for start in tqdm(range(0, len(indices), batch_size), desc="Flan-T5 QA pairs"):
        batch_idx = indices[start:start+batch_size]
        input_ids = encodings['input_ids'][batch_idx].to(device)
        attention_mask = encodings['attention_mask'][batch_idx].to(device)
        labels = label_ids[batch_idx].to(device)

        with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits

        # Flatten logits and labels for batch cross-entropy
        B, S, V = logits.shape
        flat_logits = logits.view(-1, V)
        flat_labels = labels.view(-1)
        loss = loss_fn(flat_logits, flat_labels)
        per_loss = loss.item()
        qa_losses.extend([per_loss] * len(batch_idx))

    return qa_losses


def process_and_regress(dataset_type, source, dataset_name,
                        batch_size=24, max_length=1024):
    """
    Load data, compute QA losses, and run regression.

    :param dataset_type: 'dataset' for HF datasets or 'csv' for CSV files
    :param source: HF dataset name or path to CSV file
    :param dataset_name: label used in output filenames and metrics
    """
    # 1. Load DataFrame
    if dataset_type == 'dataset':
        df = parse_data.load_dataset_to_df(source)
    else:
        df = parse_data.load_csv_to_df(source)

    # 2. Pre-tokenize questions & answers
    questions = df['question_text'].tolist()
    answers = df['answer_text'].tolist()
    encodings = tokenizer(
        questions,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )
    label_ids = tokenizer(
        answers,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )['input_ids']

    # 3. Split indices for train/test
    indices = list(range(len(df)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # 4. Compute losses
    train_losses = compute_losses(train_idx, encodings, label_ids, batch_size=batch_size)
    test_losses  = compute_losses(test_idx,  encodings, label_ids, batch_size=batch_size)

    # 5. Attach losses and save CSVs
    train_df = df.iloc[train_idx].copy()
    train_df['loss'] = train_losses
    test_df  = df.iloc[test_idx].copy()
    test_df['loss']  = test_losses

    train_csv = f"qa_losses_Flan-T5_{dataset_name}_train.csv"
    test_csv  = f"qa_losses_Flan-T5_{dataset_name}_test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Saved {train_csv} and {test_csv}")

    # 6. Run regression with split DataFrames
    run_regression(
        dataset_name=f"Flan-T5_{dataset_name}",
        train_df=train_df,
        test_df=test_df,
        estimator=RandomForestRegressor,
        estimator_params={ 'max_depth':5, 'n_estimators':250 }
    )


def main():
    # Orchestrate for multiple datasets
    datasets = {
        #'DS':      ('csv',     'DS_tests_with_difficulty.csv'),
        #'Leetcode':('csv',     'merged_leetcode_df.csv'),
        'NovaSky': ('dataset', 'NovaSky-AI/labeled_numina_difficulty_162K'),
    }
    for name, (dtype, src) in datasets.items():
        print(f"Processing {name} with Flan-T5")
        process_and_regress(dtype, src, name)


if __name__ == '__main__':
    main()