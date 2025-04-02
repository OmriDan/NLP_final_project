import pandas as pd


def load_csv_to_df(filename: str) -> pd.DataFrame:
    """
    Load a CSV file and convert it to a pandas DataFrame with standardized columns:
    'question_type', 'question_text', 'answer_text', and 'difficulty'.

    The function handles two types of CSV files:

    1. Type 1:
       Columns (in order): title, difficulty, content, python, Acceptance, Topic_tags, normalized_difficulty.
       Mapping:
         - question_type: "open question" (constant)
         - question_text: content
         - answer_text: python
         - difficulty: normalized_difficulty

    2. Type 2:
       Columns (in order): question_type, points, question_translation_latex, multiple_choice_answer, answer_translation_latex, topics_covered, difficulty.
       Mapping:
         - question_type: if the csv row's question_type is "Multiple Choice" or "Multiple choice with explanations",
           then keep it; otherwise, set to "open question".
         - question_text: question_translation_latex
         - answer_text: for rows with question_type exactly "Multiple Choice", use multiple_choice_answer;
           for all other rows, use answer_translation_latex.
         - difficulty: difficulty

    Parameters:
        filename (str): The path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with standardized columns.
    """
    # Read the CSV file
    df = pd.read_csv(filename)

    # Identify CSV type based on expected columns
    # For type1, expected header columns
    type1_cols = ["title", "difficulty", "content", "python", "Acceptance", "Topic_tags", "normalized_difficulty"]
    type2_cols = ["question_type", "points", "question_translation_latex", "multiple_choice_answer",
                  "answer_translation_latex", "topics_covered", "difficulty"]

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
        df_copy["answer_text"] = df["question_type"].apply(
            lambda qt: df["multiple_choice_answer"] if qt == "Multiple Choice" else df["answer_translation_latex"]
        )

        # Build the final DataFrame with mapped columns
        new_df = pd.DataFrame({
            "question_type": df_copy["question_type"],
            "question_text": df["question_translation_latex"],
            "answer_text": df_copy["answer_text"],
            "difficulty": df["difficulty"]
        })
    else:
        raise ValueError("CSV file structure is not recognized. Please ensure the CSV has the expected columns.")

    return new_df
