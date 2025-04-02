# Add to pipeline.py or create a new wandb_utils.py file

def log_prediction_examples(model, dataset, tokenizer, num_examples=5):
    """Log prediction examples to wandb"""
    if wandb.run is None:
        return

    examples_table = wandb.Table(columns=["Question", "Answer", "True Difficulty", "Predicted Difficulty", "Error"])

    # Get random samples
    import random
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))

    device = next(model.parameters()).device

    for idx in indices:
        item = dataset[idx]
        inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != 'labels'}

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = outputs["score"].item()
        true_value = item["labels"].item() if "labels" in item else 0.0

        # Decode the input to get the original text
        decoded_text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)

        # Extract question and answer from the decoded text
        # This is approximate and depends on your exact formatting
        parts = decoded_text.split("Question:")
        if len(parts) > 1:
            context = parts[0].replace("Context:", "").strip()
            qa_parts = parts[1].split("Answer:")
            question = qa_parts[0].strip()
            answer = qa_parts[1].strip() if len(qa_parts) > 1 else ""
        else:
            context, question, answer = "", decoded_text, ""

        examples_table.add_data(
            question,
            answer,
            true_value,
            prediction,
            abs(true_value - prediction)
        )

    wandb.log({"prediction_examples": examples_table})