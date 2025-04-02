import os
import torch
import pickle
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from dataset import RAGAugmentedDataset
from model import RAGQuestionDifficultyRegressor

def build_rag_difficulty_regressor(train_df, valid_df, knowledge_corpus, model_name="microsoft/deberta-v3-base"):
    # Step 1: Prepare the RAG retriever with the knowledge corpus
    from retriever import RAGRetriever
    retriever = RAGRetriever(knowledge_corpus)

    # Step 2: Initialize tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # Step 3: Create RAG-augmented datasets
    train_dataset = RAGAugmentedDataset(
        questions=train_df['question'].tolist(),
        answers=train_df['answer'].tolist(),
        labels=train_df['difficulty'].tolist(),
        tokenizer=tokenizer,
        retriever=retriever,
        k=3
    )

    valid_dataset = RAGAugmentedDataset(
        questions=valid_df['question'].tolist(),
        answers=valid_df['answer'].tolist(),
        labels=valid_df['difficulty'].tolist(),
        tokenizer=tokenizer,
        retriever=retriever,
        k=3
    )

    # Step 4: Initialize the regressor model
    model = RAGQuestionDifficultyRegressor(base_model)

    # Step 5: Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=100,
        weight_decay=0.01,
        push_to_hub=False,
        fp16=True,
        lr_scheduler_type="linear",
    )

    # Step 6: Define evaluation metrics for regression
    def compute_metrics(eval_preds):
        scores, labels = eval_preds
        mse = ((scores - labels) ** 2).mean().item()
        mae = abs(scores - labels).mean().item()
        return {
            "mse": mse,
            "mae": mae,
            "rmse": mse ** 0.5
        }

    # Step 7: Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    # Step 8: Train the model
    trainer.train()

    # Step 9: Save the trained model, tokenizer, and retriever
    model_artifacts = {
        "model": model,
        "tokenizer": tokenizer,
        "retriever": retriever
    }

    with open("difficulty_regressor_artifacts.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)

    return model_artifacts