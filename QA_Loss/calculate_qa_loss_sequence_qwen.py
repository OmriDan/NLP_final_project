import torch
import parse_data
from regression import run_regression
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# Set device and model parameters.
device = 'cuda'
model_name = "Qwen/Qwen1.5-0.5B-Chat"

# Configure the tokenizer for a causal, chat-style model.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the causal language model.
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Define the loss function.
loss_fn = torch.nn.CrossEntropyLoss()

# Load your DataFrame.
df = parse_data.load_csv_to_df("./merged_leetcode_df.csv")

# List to store computed QA losses.
qa_losses = []

# Process each (question, answer) pair in the DataFrame.
for idx, row in df.iterrows():
    question = row[1]  # Second column: question.
    answer = row[2]  # Third column: answer.

    # Create the prompt: a user message asking the question.
    prompt_content = f"Write a Python program that answers the following question.\nQuestion: {question}"

    # Build the chat template for the input (user prompt).
    text_i = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt_content}],
        tokenize=False,
        add_generation_prompt=True
    )

    # Build the full conversation with the assistant's response appended.
    text_l = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt_content}, {'role': 'assistant', 'content': answer}],
        tokenize=False,
        add_generation_prompt=False,
    ) + "<|im_end|>"

    # Tokenize both the input prompt and the full conversation.
    input_prompts = tokenizer(text_i, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(
        device)
    whole_prompts = tokenizer(text_l, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(
        device)

    # Run the model on the full prompt.
    with torch.no_grad():
        outputs = model(input_ids=whole_prompts['input_ids'], attention_mask=whole_prompts['attention_mask'])
        logits = outputs.logits

    # Process a single example
    # Remove left padding from both the input and whole conversation.
    input_seq = input_prompts['input_ids'][0]
    whole_seq = whole_prompts['input_ids'][0]
    input_seq = input_seq[input_seq != tokenizer.pad_token_id]
    whole_seq = whole_seq[whole_seq != tokenizer.pad_token_id]

    # Remove the last logit (the model automatically adds one extra token).
    logit = logits[0][:-1]  # Shape: (seq_len - 1, vocab_size)

    # Determine the start position of the assistant response.
    # The answer tokens start right after the tokens corresponding to the input prompt.
    generation_length = len(whole_seq) - len(input_seq)
    good_logit = logit[-generation_length:]
    good_label = whole_seq[len(input_seq):]

    # Compute the cross-entropy loss for the answer portion.
    loss = loss_fn(good_logit, good_label)
    qa_losses.append(loss.item())

# Add the computed QA loss to the DataFrame and save to a CSV.
df['qa_loss'] = qa_losses
df.to_csv("qa_loss_qwen.csv", index=False)
run_regression("qa_loss_quen.csv")
