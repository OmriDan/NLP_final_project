import torch
import parse_data
from regression import run_regression
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import csv

# Download tokenizer models
nltk.download('punkt')

# Model setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token
# Load model with half-precision for faster inference
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
loss_fn = torch.nn.CrossEntropyLoss()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.get_device_name(0)}")

def compute_losses(df_subset, batch_size=4):
    """
    Compute QA cross-entropy losses on a DataFrame subset using Qwen.
    
    Args:
        df_subset (pd.DataFrame): DataFrame containing 'question_text' and 'answer_text' columns.
        batch_size (int): Number of samples to process in each batch.
    
    Returns:
        pd.DataFrame: Copy of df_subset with an additional 'loss' column containing the computed losses.
    """
    qa_losses = []

    # Process in batches
    for i in tqdm(range(0, len(df_subset), batch_size), desc="Qwen QA pairs"):
        batch = df_subset.iloc[i:i+batch_size]
        questions = batch['question_text'].tolist()
        answers = batch['answer_text'].tolist()

        # Prepare prompts and tokezine inputs/labels
        prompts = [f"Write an answer to the following question:\nQuestion: {q}" for q in questions]
        full_convs = [
            f"Write an answer to the following question:\nQuestion: {q}\nAnswer: {a}"
            for q, a in zip(questions, answers)
        ]
        inp = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)
        whole = tokenizer(full_convs, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)

        # Forward pass without gradient tracking
        with torch.no_grad():
            outputs = model(input_ids=whole['input_ids'], attention_mask=whole['attention_mask'])
            logits = outputs.logits

        # Compute loss per example
        for j in range(len(batch)):
            inp_ids = inp['input_ids'][j]
            whole_ids = whole['input_ids'][j]
            inp_ids = inp_ids[inp_ids != tokenizer.pad_token_id]
            whole_ids = whole_ids[whole_ids != tokenizer.pad_token_id]
            logit_seq = logits[j][:-1]
            gen_len = len(whole_ids) - len(inp_ids)
            if gen_len <= 0:
                # Assign NaN for empty generations
                qa_losses.append(float('nan'))
            else:
                good_logit = logit_seq[-gen_len:]
                good_label = whole_ids[len(inp_ids):]
                qa_losses.append(loss_fn(good_logit, good_label).item())

    df_out = df_subset.copy()
    df_out['loss'] = qa_losses
    mean_loss = df_out['loss'].mean(skipna=True)
    df_out['loss'] = df_out['loss'].fillna(mean_loss)  # Fill NaN with mean loss
    return df_out


def process_and_regress(dataset_type, source, dataset_name, estimator = None, estimator_params=None):
    """
    Load data (HF dataset or CSV), compute QA losses, perform train/test split, and run regression.

    Args:
        dataset_type (str): 'dataset' for Hugging Face dataset or 'csv' for CSV file input
        source (str): Name of the HF dataset or path to CSV
        dataset_name (str): Label used in output files
        estimator (callable): Regression estimator class (e.g., RandomForestRegressor)
        estimator_params (dict): Parameters for the regression estimator
    """
    if dataset_type == 'dataset':
        df = parse_data.load_dataset_to_df(source)
    else:
        df = parse_data.load_csv_to_df(source)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_loss = compute_losses(train_df)
    test_loss = compute_losses(test_df)

    train_loss.to_csv(f"qa_losses_Qwen_{dataset_name}_train.csv", quoting=csv.QUOTE_MINIMAL, escapechar="\\", index=False)
    test_loss.to_csv(f"qa_losses_Qwen_{dataset_name}_test.csv", quoting=csv.QUOTE_MINIMAL, escapechar="\\", index=False)

    run_regression(
        dataset_name=f"Qwen_{dataset_name}",
        estimator=estimator,
        estimator_params=estimator_params,
        train_df=train_loss,
        test_df=test_loss
    )


if __name__ == "__main__":
    # process_and_regress('csv', 'DS_tests_with_difficulty.csv', 'DS', estimator=RandomForestRegressor ,estimator_params={'max_depth': 5, 'n_estimators': 250})
    # process_and_regress('csv', 'merged_leetcode_df.csv', 'Leetcode', estimator=RandomForestRegressor ,estimator_params={'max_depth': 5, 'n_estimators': 250})
    process_and_regress('dataset', 'NovaSky-AI/labeled_numina_difficulty_162K', 'NovaSky',estimator=RandomForestRegressor ,estimator_params={'max_depth': 5, 'n_estimators': 250})
