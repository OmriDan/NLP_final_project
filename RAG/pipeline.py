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


def build_rag_difficulty_regressor(train_df, valid_df, knowledge_corpus, model_name="microsoft/deberta-v3-large",embedding_model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                                   wandb_project="rag-difficulty-regressor", wandb_run_name=None):
    """
    Build and train a RAG-based difficulty regressor with wandb integration.
    Another encoder model option is neulab/codebert-base
    Specifically pre-trained on code, Better understanding of programming concepts,
     Would understand programming difficulty better than general-purpose models
    :param train_df:
    :param valid_df:
    :param knowledge_corpus:
    :param model_name:
    :param wandb_project:
    :param wandb_run_name:
    :return:
    """
    # Initialize wandb
    wandb.init(project=wandb_project, name=wandb_run_name, mode="online")

    # Log dataset info
    wandb.config.update({
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "model_name": model_name,
        "knowledge_corpus_size": len(knowledge_corpus)
    })

    # Step 1: Prepare the RAG retriever with the knowledge corpus
    retriever = RAGRetriever(knowledge_corpus, embedding_model=embedding_model_name)

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

    # Step 4: Initialize the regressor model
    model = RAGQuestionDifficultyRegressor(base_model)

    # Step 5: Set up training arguments with wandb integration
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=2e-4,  # Increase from 5e-5 to help escape local minimum
        lr_scheduler_type="linear",  # Try linear instead of cosine
        warmup_ratio=0.2,  # Longer warmup period
        per_device_train_batch_size=8,  # Larger batch for more stable gradients
        per_device_eval_batch_size=16,
        num_train_epochs=30,  # Potentially train longer
        weight_decay=0.001,  # Reduce from 0.01 to allow more flexibility
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        report_to="wandb",
        logging_steps=50
    )

    # Step 6: Define evaluation metrics for regression
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

        if len(valid_dataset) > 0:
            sample_inputs = valid_dataset[0]
            sample_inputs = {k: v.unsqueeze(0) for k, v in sample_inputs.items() if k != 'labels'}
            with torch.no_grad():
                prediction = model(**sample_inputs)
                # Check if prediction is a tuple (loss, score) or just score
                if isinstance(prediction, tuple):
                    prediction = prediction[1].item()
                else:
                    prediction = prediction.item()
            wandb.log({"sample_prediction": prediction})

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