import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def split_dataset(input_file='/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/leetcode_data_for_model.csv', train_size=0.8,
                  random_state=42, sampling_strategy='stratify',
                  output_dir=None, visualize=True):
    """
    Split the dataset with continuous difficulty scores into training and validation sets
    with options for handling score distribution imbalance.
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(input_file) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # Read the dataset
    print(f"Reading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Dataset loaded: {len(df)} records")

    # Use fixed-width bins to capture the real distribution (not quantile-based)
    # Approximately matches typical Leetcode difficulty: Easy (0-0.33), Medium (0.33-0.67), Hard (0.67-1.0)
    bins = [0, 0.33, 0.67, 1.0]
    labels = ['Easy', 'Medium', 'Hard']
    df['difficulty_bins'] = pd.cut(df['difficulty'], bins=bins, labels=labels)

    # Display the natural bin distribution
    bin_counts = df['difficulty_bins'].value_counts().sort_index()
    print(f"Natural difficulty distribution:\n{bin_counts}")

    if visualize:
        # Plot difficulty distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['difficulty'], bins=15, edgecolor='black')
        plt.title('Difficulty Distribution')
        plt.xlabel('Difficulty Score')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'difficulty_distribution.png'))
        plt.close()

    # Create stratified split to maintain relative proportions
    train_df, val_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df['difficulty_bins']
    )

    print(f"Initial split: {len(train_df)} training, {len(val_df)} validation")

    # Now the bin distribution should show the natural imbalance
    print(f"Original training distribution: {train_df['difficulty_bins'].value_counts().to_dict()}")

    # Handle class imbalance if requested
    if sampling_strategy != 'none':
        if sampling_strategy == 'undersample':
            # Undersample majority classes
            undersampler = RandomUnderSampler(random_state=random_state)
            X = train_df.drop(columns=['difficulty_bins'])
            y = train_df['difficulty_bins']
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            train_df = pd.concat([X_resampled, pd.Series(y_resampled, name='difficulty_bins')], axis=1)
            print("Applied undersampling to training data")

        elif sampling_strategy == 'oversample':
            # Oversample minority classes
            oversampler = RandomOverSampler(random_state=random_state)
            X = train_df.drop(columns=['difficulty_bins'])
            y = train_df['difficulty_bins']
            X_resampled, y_resampled = oversampler.fit_resample(X, y)
            train_df = pd.concat([X_resampled, pd.Series(y_resampled, name='difficulty_bins')], axis=1)
            print("Applied oversampling to training data")

        print(f"New training distribution: {train_df['difficulty_bins'].value_counts().to_dict()}")

        if visualize:
            # Plot resampled distribution
            plt.figure(figsize=(10, 6))
            plt.hist(train_df['difficulty'], bins=15, edgecolor='black')
            plt.title(f'Difficulty Distribution After {sampling_strategy.capitalize()}')
            plt.xlabel('Difficulty Score')
            plt.ylabel('Count')
            plt.savefig(os.path.join(output_dir, f'difficulty_after_{sampling_strategy}.png'))
            plt.close()

    # Save the splits (drop the temporary binning column)
    train_output = os.path.join(output_dir, f'leetcode_train_{sampling_strategy}.csv')
    val_output = os.path.join(output_dir, 'leetcode_val.csv')

    # Drop the binning column before saving
    if 'difficulty_bins' in train_df.columns:
        train_df = train_df.drop(columns=['difficulty_bins'])
    if 'difficulty_bins' in val_df.columns:
        val_df = val_df.drop(columns=['difficulty_bins'])

    # Calculate and display difficulty statistics
    print(
        f"Training difficulty statistics: mean={train_df['difficulty'].mean():.3f}, std={train_df['difficulty'].std():.3f}")
    print(
        f"Validation difficulty statistics: mean={val_df['difficulty'].mean():.3f}, std={val_df['difficulty'].std():.3f}")

    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

    print(f"Training data saved to {train_output}")
    print(f"Validation data saved to {val_output}")

    return train_df, val_df


if __name__ == "__main__":
    # You can choose which balancing strategy to use
    split_dataset(sampling_strategy='oversample')  # Options: 'stratify', 'undersample', 'oversample', 'none'