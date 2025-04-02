import torch
import parse_data
from regression import run_regression
from transformers import AutoTokenizer, T5ForConditionalGeneration
import nltk
nltk.download('punkt')


device = 'cuda'
model_name = "google/flan-t5-xl"  # Path to your model directory

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 8

# Load your dataframe; adjust the file name/path as needed.
# Assuming the first column is "question" and the second column is "answer"
df = parse_data.load_csv_to_df('./merged_leetcode_df.csv')

# List to store the computed QA loss for each question-answer pair.
qa_losses = []

# Iterate through each row in the dataframe.
for idx, row in df.iterrows():
    question = row[1]  # second column: question
    answer = row[2]    # third column: answer

    # Create a prompt that asks the model to answer the question.
    prompt = f"Write a python program that answers the following question:\nQuestion: {question}\nProgram:"

    # Tokenize the prompt and the answer separately.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding='longest').to(device)
    labels = tokenizer(answer, return_tensors="pt", truncation=True, max_length=1024, padding='longest')['input_ids'].to(device)

    with torch.no_grad():
        # Get the model outputs (logits) and compute the loss.
        outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=labels)
        logits = outputs.logits

    # Calculate loss only over the non-padding tokens.
    num_elements = torch.count_nonzero(labels != tokenizer.pad_token_id)
    good_logit = logits[0][:num_elements]
    good_label = labels[0][:num_elements]
    loss = loss_fn(good_logit, good_label)

    qa_losses.append(loss.item())

# Add the computed loss as a new column in your dataframe.
df['qa_loss'] = qa_losses
# Save the updated dataframe to a CSV file.
df.to_csv("qa_loss.csv", index=False)
# Run the regression
run_regression("qa_loss.csv")
