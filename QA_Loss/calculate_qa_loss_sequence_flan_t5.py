import torch
import parse_data
from regression import run_regression
from transformers import AutoTokenizer, T5ForConditionalGeneration
import nltk
from tqdm import tqdm
nltk.download('punkt')
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda) 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "google/flan-t5-base"  
print("device: ", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

# Load dataframe; adjust the file name/path as needed.
# df = parse_data.load_csv_to_df('./merged_leetcode_df.csv')
df = parse_data.load_dataset_to_df("NovaSky-AI/labeled_numina_difficulty_162K")

# List to store the computed QA loss for each question-answer pair.
qa_losses = []

batch_size = 16  # Number of samples to process at once
# Iterate through each row in the dataframe.
for i in tqdm(range(0, len(df), batch_size), desc="Processing QA pairs"):
    batch = df.iloc[i:i+batch_size]
    prompts = [f"Write an answer to the following question:\nQuestion: {row[1]}\nAnswer:" for row in batch.itertuples()]
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=1024, padding='longest').to(device)
    labels = tokenizer(list(batch.iloc[:, 2]), return_tensors="pt", truncation=True, max_length=1024, padding='longest')['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
        logits = outputs.logits
    
    # Compute losses for the batch
    for j in range(len(batch)):
        num_elements = torch.count_nonzero(labels[j] != tokenizer.pad_token_id)
        good_logit = logits[j][:num_elements]
        good_label = labels[j][:num_elements]
        loss = loss_fn(good_logit, good_label)
        qa_losses.append(loss.item())

# Add the computed loss as a new column in your dataframe.
df['loss'] = qa_losses
# Save the updated dataframe to a CSV file.
df.to_csv("qa_loss.csv", index=False)
# Run the regression
run_regression("qa_loss.csv", "Flan-T5")
