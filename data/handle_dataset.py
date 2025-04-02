import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_dataset(input_file='leetcode/merged_leetcode_df.csv', train_size=0.8, random_state=42):
    """
    Split the merged LeetCode dataset into training and validation sets

    Args:
        input_file: Path to the input CSV file
        train_size: Proportion of data to use for training (default: 0.8 or 80%)
        random_state: Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(input_file) or '.'

    # Read the dataset
    print(f"Reading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Dataset loaded: {len(df)} records")

    # Split the dataset
    train_df, val_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df.get('Difficulty') if 'Difficulty' in df.columns else None
    )

    print(f"Split completed: {len(train_df)} training samples, {len(val_df)} validation samples")

    # Save the splits
    train_output = os.path.join(output_dir, 'leetcode_train.csv')
    val_output = os.path.join(output_dir, 'leetcode_val.csv')

    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

    print(f"Training data saved to {train_output}")
    print(f"Validation data saved to {val_output}")


if __name__ == "__main__":
    split_dataset()