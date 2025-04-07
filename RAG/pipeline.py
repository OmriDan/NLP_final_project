import datetime
import os
import torch
import pickle
import wandb
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from wandb.cli.cli import offline, online

from dataset import RAGAugmentedDataset
from model import RAGQuestionDifficultyRegressor
from retriever import RAGRetriever

from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback
from wandb_utils import log_prediction_examples


def setup_wandb_run_directory(project_name, run_name=None):
    """Setup wandb run and create a custom directory for model artifacts"""
    # Generate meaningful name if not provided
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"difficulty_model_{timestamp}"

    # Initialize wandb run
    run = wandb.init(project=project_name, name=run_name, mode="online")

    # Create directory structure based on run name
    save_dir = os.path.join("models", run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save directory path to wandb config
    wandb.config.update({"save_dir": save_dir})

    print(f"Model will be saved to: {save_dir}")
    return run, save_dir

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

def compute_metrics(eval_preds):
    # Handle the complex output structure from the model
    predictions, labels = eval_preds

    # Extract scores from predictions (which may be a tuple of nested arrays)
    if isinstance(predictions, tuple) and len(predictions) > 0:
        # If predictions is a tuple, extract the first element (likely scores)
        scores = predictions[0].flatten()
    else:
        scores = predictions.flatten()

    # Ensure labels are flattened too
    labels = labels.flatten()

    # Compute metrics
    mse = ((scores - labels) ** 2).mean().item()
    mae = abs(scores - labels).mean().item()
    rmse = mse ** 0.5

    # Log to wandb
    if wandb.run is not None:
        wandb.log({"eval/mse": mse, "eval/mae": mae, "eval/rmse": rmse})

        # Log example predictions
        if len(scores) > 5:
            example_table = wandb.Table(columns=["Predicted", "Actual", "Error"])
            for i in range(5):
                example_table.add_data(float(scores[i]), float(labels[i]),
                                       float(abs(scores[i] - labels[i])))
            wandb.log({"eval_examples": example_table})

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse
    }


def build_rag_difficulty_regressor(train_df, valid_df, knowledge_corpus, model_name="microsoft/deberta-v3-large",
                                   embedding_model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                                   wandb_project="rag-difficulty-regressor", wandb_run_name=None):
    """Build and train a RAG-based difficulty regressor with wandb integration."""
    import datetime

    # Initialize wandb and get model directory
    run, model_dir = setup_wandb_run_directory(wandb_project, wandb_run_name)

    # Configure output paths
    output_dir = os.path.join(model_dir, "checkpoints")
    final_model_dir = os.path.join(model_dir, "final_model")
    artifacts_path = os.path.join(model_dir, "difficulty_regressor_artifacts.pkl")

    # Log dataset info
    wandb.config.update({
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "model_name": model_name,
        "knowledge_corpus_size": len(knowledge_corpus),
        "model_path": final_model_dir,
        "artifacts_path": artifacts_path
    })

    # Step 1: Prepare the RAG retriever with the knowledge corpus
    retriever = RAGRetriever(knowledge_corpus, embedding_model=embedding_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # Create datasets
    train_dataset = RAGAugmentedDataset(
        questions=train_df['question'].tolist(),
        answers=train_df['answer'].tolist(),
        labels=train_df['difficulty'].tolist(),
        tokenizer=tokenizer,
        retriever=retriever,
        k=5
    )

    valid_dataset = RAGAugmentedDataset(
        questions=valid_df['question'].tolist(),
        answers=valid_df['answer'].tolist(),
        labels=valid_df['difficulty'].tolist(),
        tokenizer=tokenizer,
        retriever=retriever,
        k=5
    )

    # Initialize model
    model = RAGQuestionDifficultyRegressor(base_model)

    # Update training arguments to use the new output directory
    training_args = TrainingArguments(
        output_dir=output_dir,  # Updated to use run-specific directory
        # Other arguments remain the same...
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=100,
        weight_decay=0.001,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        report_to="wandb",
        logging_steps=50
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[CustomWandbCallback(), WandbCallback()]
    )

    # Train model
    trainer.train()

    # Save model to the run-specific directory
    os.makedirs(final_model_dir, exist_ok=True)
    trainer.save_model(final_model_dir)

    # Log as wandb artifact
    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
    artifact.add_dir(final_model_dir)
    wandb.log_artifact(artifact)

    # Save artifacts to the run-specific path
    model_artifacts = {
        "model": model,
        "tokenizer": tokenizer,
        "retriever": retriever
    }

    with open(artifacts_path, "wb") as f:
        pickle.dump(model_artifacts, f)

    # Create a README file with run information
    with open(os.path.join(model_dir, "README.md"), "w") as f:
        f.write(f"# Model: {wandb_run_name or 'Unnamed Run'}\n\n")
        f.write(f"* Run ID: {wandb.run.id}\n")
        f.write(f"* Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"* Base model: {model_name}\n")
        f.write(f"* Embedding model: {embedding_model_name}\n")

    return model_artifacts