import pandas as pd
import text2props.constants as const
import pickle


def parse_data(filename: str, leetcode: bool, save_latent_traits: bool, output_path: str = None) -> pd.DataFrame:
    """
    Parse a CSV file and convert it into a DataFrames questions+answers. And save latent traits as a pickle file.
    :param filename: Path to csv
    :param leetcode: If True, parse as LeetCode-style CSV; otherwise, parse as Data Structure course format.
    :return: A DataFrame with columns Q_ID, Q_TEXT, CORRECT_TEXTS, WRONG_TEXTS, and DIFFICULTY.
    """
    df_raw = pd.read_csv(filename)
    # Pick a sample row containing Python code from the 'python' column

    if leetcode:
        # LeetCode-style CSV format
        # Remove rows where 'content' or 'python' is empty or just whitespace.
        df_raw = df_raw[df_raw['content'].astype(str).str.strip() != ""]
        df_raw = df_raw[df_raw['python'].astype(str).str.strip() != ""]

        df_questions = pd.DataFrame()
        df_questions[const.Q_ID] = df_raw.index.astype(str)
        df_questions[const.Q_TEXT] = df_raw['content']
        df_questions[const.CORRECT_TEXTS] = df_raw['python'].apply(lambda x: [x] if pd.notnull(x) else [])
        df_questions[const.WRONG_TEXTS] = [[] for _ in range(len(df_raw))]
        df_questions[const.DIFFICULTY] = df_raw['normalized_difficulty']

    else:
        # Data Structures course format
        # Remove rows where 'questions_translated_to_English' is empty or just whitespace.
        df_raw = df_raw[df_raw['question_translated_to_English'].astype(str).str.strip() != ""]

        df_questions = pd.DataFrame()
        df_questions[const.Q_ID] = df_raw.index.astype(str)
        df_questions[const.Q_TEXT] = df_raw['question_translated_to_English']
        df_questions[const.CORRECT_TEXTS] = df_raw['answer_translated_to_English'].apply(lambda x: [x] if pd.notnull(x) else [])
        df_questions[const.WRONG_TEXTS] = [[] for _ in range(len(df_raw))]
        df_questions[const.DIFFICULTY] = df_raw['Difficulty']

    # Save latent traits as a pickle file
    if save_latent_traits:
        if output_path is None:
            raise ValueError("output_path must be specified if save_latent_traits is True.")
        
        # Extract latent traits
        dict_latent_traits = {
            const.DIFFICULTY: dict(zip(df_questions[const.Q_ID], df_questions[const.DIFFICULTY]))
        }

        # Save to pickle
        with open(output_path, 'wb') as f:
            pickle.dump(dict_latent_traits, f)
        print(f"[INFO] Latent traits saved to {output_path}")

    return df_questions
