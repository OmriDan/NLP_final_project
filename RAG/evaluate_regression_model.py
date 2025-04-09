import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from inference import predict_difficulty_with_rag
from tqdm import tqdm
import argparse


def calculate_prediction_metrics(model_path, csv_path):
    # Load model artifacts
    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model_artifacts = pickle.load(f)

    # Load CSV data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Verify required columns are present
    required_columns = ['question', 'answer', 'difficulty']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Make predictions
    print("Making predictions...")
    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        answer = row['answer']
        result = predict_difficulty_with_rag(question, answer, model_artifacts)
        predictions.append(result['difficulty_score'])

    # Calculate metrics
    actual = df['difficulty'].values
    predicted = np.array(predictions)
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    # Add predictions to dataframe
    df['predicted_difficulty'] = predictions
    df['squared_error'] = (df['difficulty'] - df['predicted_difficulty']) ** 2
    df['absolute_error'] = abs(df['difficulty'] - df['predicted_difficulty'])

    return mse, mae, r2, df


def train_and_evaluate_linear_baseline(df):
    """Train a simple linear regression model as a baseline and evaluate it."""
    print("\n--- Linear Regression Baseline ---")

    # Prepare text features using TF-IDF
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=1000)

    # Combine question and answer for feature extraction
    combined_text = df['question'] + " " + df['answer']
    X = tfidf.fit_transform(combined_text)

    # Use 80% of data for training, 20% for evaluation
    train_size = int(0.8 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = df['difficulty'].values[:train_size], df['difficulty'].values[train_size:]

    # Train linear regression model
    print("Training linear regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions
    y_pred = lr_model.predict(X_test)

    # Calculate metrics
    lr_mse = mean_squared_error(y_test, y_pred)
    lr_mae = mean_absolute_error(y_test, y_pred)
    lr_r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression MSE: {lr_mse:.4f}")
    print(f"Linear Regression MAE: {lr_mae:.4f}")
    print(f"Linear Regression R² Score: {lr_r2:.4f}")

    return lr_mse, lr_mae, lr_r2


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics between model predictions and actual difficulty')
    parser.add_argument('--model', required=True, help='Path to the pickle file containing model artifacts')
    parser.add_argument('--data', required=True, help='Path to the CSV file with question, answer, difficulty columns')
    parser.add_argument('--output', help='Path to save results CSV (optional)')
    parser.add_argument('--baseline', action='store_true', help='Run a linear regression baseline for comparison')
    args = parser.parse_args()

    print("--- RAG-based Model Evaluation ---")
    mse, mae, r2, results_df = calculate_prediction_metrics(args.model, args.data)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Run linear regression baseline if requested
    if args.baseline:
        lr_mse, lr_mae, lr_r2 = train_and_evaluate_linear_baseline(results_df)

        # Show improvement over baseline
        mse_improvement = ((lr_mse - mse) / lr_mse) * 100
        mae_improvement = ((lr_mae - mae) / lr_mae) * 100
        r2_improvement = ((r2 - lr_r2) / max(0.001, abs(lr_r2))) * 100

        print("\n--- Comparison to Baseline ---")
        print(f"MSE: {mse_improvement:.2f}% improvement over baseline")
        print(f"MAE: {mae_improvement:.2f}% improvement over baseline")
        print(f"R²: {r2_improvement:.2f}% improvement over baseline")

    # Save results if output path specified
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

    # Print additional statistics
    print("\nStatistics:")
    print(f"Mean Actual Difficulty: {results_df['difficulty'].mean():.4f}")
    print(f"Mean Predicted Difficulty: {results_df['predicted_difficulty'].mean():.4f}")
    print(f"Std Dev Actual: {results_df['difficulty'].std():.4f}")
    print(f"Std Dev Predicted: {results_df['predicted_difficulty'].std():.4f}")


if __name__ == "__main__":
    main()