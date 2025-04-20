import torch
import parse_data
from regression import run_regression
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Set device and model parameters.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "Qwen/Qwen1.5-0.5B-Chat"

# Configure the tokenizer for a causal, chat-style model.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the causal language model.
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Define the loss function.
loss_fn = torch.nn.CrossEntropyLoss()

# Load your DataFrame.
# df = parse_data.load_csv_to_df("./DS_tests_with_difficulty.csv")
df = parse_data.load_dataset_to_df("NovaSky-AI/labeled_numina_difficulty_162K")

# List to store computed QA losses.
qa_losses = []

batch_size = 4  # Number of samples to process at once
# Process the DataFrame in batches.
for i in tqdm(range(0, len(df), batch_size), desc="Processing QA pairs"):
    batch = df.iloc[i:i+batch_size]
    questions = batch.iloc[:, 1].tolist()  # Second column: questions
    answers = batch.iloc[:, 2].tolist()    # Third column: answers

    # Create prompts and full conversations for the batch.
    prompts = [f"Write an answer to the following question:\nQuestion: {q}" for q in questions]
    full_conversations = [
        f"Write an answer to the following question:\nQuestion: {q}\nAnswer: {a}"
        for q, a in zip(questions, answers)
    ]

    # Tokenize the prompts and full conversations.
    input_prompts = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)
    whole_prompts = tokenizer(full_conversations, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)

    with torch.no_grad():
        # Run the model on the full prompts.
        outputs = model(input_ids=whole_prompts['input_ids'], attention_mask=whole_prompts['attention_mask'])
        logits = outputs.logits

    # Compute losses for the batch.
    for j in range(len(batch)):
        # Remove padding tokens.
        input_seq = input_prompts['input_ids'][j]
        whole_seq = whole_prompts['input_ids'][j]
        input_seq = input_seq[input_seq != tokenizer.pad_token_id]
        whole_seq = whole_seq[whole_seq != tokenizer.pad_token_id]

        # Remove the last logit (the model automatically adds one extra token).
        logit = logits[j][:-1]  # Shape: (seq_len - 1, vocab_size)

        # Determine the start position of the assistant response.
        generation_length = len(whole_seq) - len(input_seq)
        good_logit = logit[-generation_length:]
        good_label = whole_seq[len(input_seq):]

        # Compute the cross-entropy loss for the answer portion.
        loss = loss_fn(good_logit, good_label)
        qa_losses.append(loss.item())


# Add the computed QA loss to the DataFrame and save to a CSV.
df['loss'] = qa_losses
df.to_csv("qa_loss_qwen.csv", index=False)

# Run the regression
run_regression("qa_loss_qwen.csv", "Qwen", estimator=RandomForestRegressor, estimator_params={'max_depth': 5, 'n_estimators': 250})
