import datetime
import os
import torch
import pickle
import wandb
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from wandb.cli.cli import offline, online
from sklearn.metrics import r2_score

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
        if not state.is_world_process_zero or logs is None or wandb.run is None:
            return

        # Log training vs validation loss difference to track overfitting
        if 'loss' in logs and 'eval_loss' in logs:
            wandb.log({
                "overfitting/train_val_diff": logs['eval_loss'] - logs['loss'],
                "custom/learning_rate": logs.get("learning_rate", 0),
                "custom/epoch": logs.get("epoch", 0),
                "custom/step": state.global_step
            })

    def on_train_end(self, args, state, control, **kwargs):
        if wandb.run is not None:
            # Log a summary of the training
            wandb.run.summary["final_loss"] = state.log_history[-1].get("loss", 0)
            wandb.run.summary["total_steps"] = state.global_step


class R2MonitorCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_r2" not in metrics:
            return control

        # Log to console
        print(f"Eval R2: {metrics['eval_r2']:.4f}")

        # Update best R2 in summary
        if wandb.run is not None:
            try:
                # Safely check and update best R2
                current_r2 = metrics["eval_r2"]
                best_r2 = None

                try:
                    # Use .get() method with default instead of direct access
                    best_r2 = wandb.run.summary.get("best_r2", -float('inf'))
                except Exception as e:
                    print(f"Warning: Could not access best_r2 in wandb summary: {e}")
                    best_r2 = -float('inf')

                if best_r2 is None or current_r2 > best_r2:
                    try:
                        # Set summary value
                        wandb.run.summary["best_r2"] = current_r2
                        print(f"New best R2: {current_r2:.4f}")
                    except Exception as e:
                        print(f"Warning: Could not update best_r2 in wandb summary: {e}")

            except Exception as e:
                print(f"Error in R2 monitoring: {e}")

        return control

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
    r2 = r2_score(labels, scores)  # Add R2 score calculation

    # Log to wandb
    if wandb.run is not None:
        wandb.log({
            "eval/mse": mse,
            "eval/mae": mae,
            "eval/rmse": rmse,
            "eval/r2": r2  # Log R2 score
        })

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
        "rmse": rmse,
        "r2": r2  # Return R2 score in metrics
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
        gradient_accumulation_steps=2,
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=3e-6,  # Reduced from 2e-5
        max_grad_norm=1.0,  # Add gradient clipping
        lr_scheduler_type="linear",  # Changed from cosine
        warmup_ratio=0.01,  # Reduced warmup
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,  # Reduced from 100
        weight_decay=0.02,  # Increased from 0.001
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        report_to="wandb",
        logging_steps=50,

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
    trainer.add_callback(R2MonitorCallback())
    # Train model
    trainer.train()

    # Get the best metric value
    best_metric_value = trainer.state.best_metric
    best_mse = round(float(best_metric_value), 4) if best_metric_value is not None else 0.0

    # Update model names to include the metric
    metric_suffix = f"_mse{best_mse}"
    final_model_dir = os.path.join(model_dir, f"final_model{metric_suffix}")
    artifacts_path = os.path.join(model_dir, f"difficulty_regressor_artifacts{metric_suffix}.pkl")

    # Create wandb directory for model
    wandb_dir = os.path.join("wandb", wandb_run_name + metric_suffix)
    os.makedirs(wandb_dir, exist_ok=True)
    wandb_model_path = os.path.join(wandb_dir, f"model{metric_suffix}")

    # Save model to all directories
    os.makedirs(final_model_dir, exist_ok=True)
    trainer.save_model(final_model_dir)
    trainer.save_model(wandb_model_path)

    # Log as wandb artifact with metric in name
    artifact = wandb.Artifact(name=f"model-mse{best_mse}-{wandb.run.id}", type="model")
    artifact.add_dir(final_model_dir)
    wandb.log_artifact(artifact)

    # Update model_artifacts to include best metric
    model_artifacts = {
        "model": model,
        "tokenizer": tokenizer,
        "retriever": retriever,
        "best_mse": best_mse
    }

    # Save artifacts
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