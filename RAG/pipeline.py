import os
import torch
import pickle
import wandb
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from dataset import RAGAugmentedDataset
from model import RAGQuestionDifficultyRegressor
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback
from wandb_utils import log_prediction_examples

class CustomWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        # Log additional custom metrics
        if wandb.run is not None and logs is not None:
            # Add any custom metrics you want to track
            wandb.log({
                "custom/learning_rate": logs.get("learning_rate", 0),
                "custom/epoch": logs.get("epoch", 0),
                "custom/step": state.global_step
            })

    def on_train_end(self, args, state, control, **kwargs):
        if wandb.run is not None:
            # Log a summary of the training
            wandb.run.summary["final_loss"] = state.log_history[-1].get("loss", 0)
            wandb.run.summary["total_steps"] = state.global_step


def build_rag_difficulty_regressor(train_df, valid_df, knowledge_corpus, model_name="microsoft/deberta-v3-base",
                                   wandb_project="rag-difficulty-regressor", wandb_run_name=None):
    # Initialize wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

    # Log dataset info
    wandb.config.update({
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "model_name": model_name,
        "knowledge_corpus_size": len(knowledge_corpus)
    })

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

    # Step 5: Set up training arguments with wandb integration
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        report_to="wandb",
        logging_steps=50
    )

    # Step 6: Define evaluation metrics for regression
    def compute_metrics(eval_preds):
        scores, labels = eval_preds
        mse = ((scores - labels) ** 2).mean().item()
        mae = abs(scores - labels).mean().item()
        rmse = mse ** 0.5

        # Log a few example predictions to wandb
        if wandb.run is not None:
            wandb.log({"eval/mse": mse, "eval/mae": mae, "eval/rmse": rmse})

            # Log some example predictions
            if len(scores) > 5:
                example_table = wandb.Table(columns=["Predicted", "Actual", "Error"])
                for i in range(5):  # Log first 5 examples
                    example_table.add_data(scores[i], labels[i], abs(scores[i] - labels[i]))
                wandb.log({"eval_examples": example_table})

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse
        }

    # Step 7: Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[CustomWandbCallback(), WandbCallback()]
    )

    # Step 8: Train the model
    trainer.train()

    # Log final model as an artifact
    model_dir = "./final_model"
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)

    if wandb.run is not None:
        artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
        artifact.add_dir(model_dir)
        wandb.log_artifact(artifact)

        # Log a sample prediction
        if len(valid_dataset) > 0:
            sample_inputs = valid_dataset[0]
            sample_inputs = {k: v.unsqueeze(0) for k, v in sample_inputs.items() if k != 'labels'}
            with torch.no_grad():
                prediction = model(**sample_inputs)["score"].item()
            wandb.log({"sample_prediction": prediction})
            log_prediction_examples(model, dataset, tokenizer, num_examples=5)
    # Finish the wandb run
    wandb.finish()

    # Step 9: Save the trained model, tokenizer, and retriever
    model_artifacts = {
        "model": model,
        "tokenizer": tokenizer,
        "retriever": retriever
    }

    with open("difficulty_regressor_artifacts.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)

    return model_artifacts