import pandas as pd
from datasets import load_dataset


def load_csv_to_df(filename: str) -> pd.DataFrame:
    """
    Parse a CSV file into a standardized DataFrame for QA difficulty tasks.

    Supports two CSV schema types:
      1) Columns: ['title', 'difficulty', 'content', 'python', 'Acceptance_rate',
         'Topic_tags', 'normalized_difficulty']
      2) Columns: ['question_type', 'points', 'question_translated_to_English',
         'multiple_choice_answer', 'answer_translated_to_English', 'Difficulty', 'source_file']

    Args:
        filename (str): Path to the input CSV file matching one of the supported schemas.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - question_type (str): 'open question' or multiple choice labels.
            - question_text (str): Text of the question.
            - answer_text (str): Ground-truth answer text.
            - difficulty (float): Difficulty score (normalized if provided).

    Raises:
        ValueError: If the CSV structure does not match any supported schema.
    """
    # Read the CSV file
    df = pd.read_csv(filename)

    # Identify CSV type based on expected columns
    # For type1, expected header columns
    type1_cols = ["title", "difficulty", "content", "python", "Acceptance_rate", "Topic_tags", "normalized_difficulty"]
    type2_cols = ["question_type", "points", "question_translated_to_English", "multiple_choice_answer",
                  "answer_translated_to_English", "Difficulty", "source_file"]

    # Check if the CSV columns match type1 structure
    if list(df.columns) == type1_cols:
        # Create new DataFrame with the required columns
        new_df = pd.DataFrame({
            "question_type": "open question",
            "question_text": df["content"],
            "answer_text": df["python"],
            "difficulty": df["normalized_difficulty"]
        })

    elif list(df.columns) == type2_cols:
        # Process type2 CSV
        # We'll first create a copy of the DataFrame to work on
        df_copy = df.copy()

        # Adjust the question_type: if it's not one of the multiple choice types, set to "open question"
        df_copy["question_type"] = df_copy["question_type"].apply(
            lambda qt: qt if qt in ["Multiple Choice", "Multiple choice with explanations"] else "open question"
        )

        # Determine the answer_text column based on the original question_type in the CSV
        # Use a lambda that checks the original question_type (from df) to decide which column to pick.
        df_copy["answer_text"] = df_copy.apply(
            lambda row: row["multiple_choice_answer"] if row["question_type"] == "Multiple Choice" else row["answer_translated_to_English"],
            axis=1
)

        # Build the final DataFrame with mapped columns
        new_df = pd.DataFrame({
            "question_type": df_copy["question_type"],
            "question_text": df["question_translated_to_English"],
            "answer_text": df_copy["answer_text"],
            "difficulty": df["Difficulty"]
        })
    else:
        raise ValueError("CSV file structure is not recognized. Please ensure the CSV has the expected columns.")

    return new_df

def load_dataset_to_df(dataset_name: str) -> pd.DataFrame:
    """
    Load a Hugging Face dataset and convert to a standardized DataFrame.

    Fetches the specified dataset, selects the 'train' split or first available,
    filters out invalid or missing entries, and normalizes difficulty to [0,1].

    Args:
        dataset_name (str): Name or path of the HF dataset (e.g., 'NovaSky-AI/...').

    Returns:
        pd.DataFrame: DataFrame with columns:
            - question_type (str): Always 'open question'.
            - question_text (str): Problem description as string.
            - answer_text (str): Solution text as string.
            - difficulty (float): Normalized difficulty in [0,1].

    Raises:
        KeyError: If no valid difficulty column is found in the dataset.
    """
    # Load the dataset
    ds = load_dataset(dataset_name)
    split = "train" if "train" in ds else next(iter(ds))
    hf = ds[split]

    # Convert to pandas
    df = hf.to_pandas()

    # Filter out any rows where difficulty is invalid
    if "gpt_difficulty_parsed" in df:
        df = df[df["gpt_difficulty_parsed"] != -1]
        diff_col = "gpt_difficulty_parsed"
    elif "gpt_difficulty" in df:
        df = df[df["gpt_difficulty"] != -1]
        diff_col = "gpt_difficulty"
    else:
        raise KeyError("Could not find a difficulty column in the dataset.")

    # Drop any rows missing question or answer text
    df = df.dropna(subset=["problem", "solution"])

    # Build the new DataFrame in the "correct format"
    out = pd.DataFrame({
        "question_type": "open question",
        "question_text": df["problem"].astype(str),
        "answer_text":   df["solution"].astype(str),
        "difficulty":    df[diff_col].astype(float),
    })

    # Normalize difficulty to be between 0 and 1
    out['difficulty'] = (out['difficulty'].astype(float) - 1.0) / 9.0

    return out
