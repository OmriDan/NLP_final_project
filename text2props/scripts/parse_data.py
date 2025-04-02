import pandas as pd


def parse_data(filename: str, leetcode: bool) -> pd.DataFrame:
    """
    Parse a CSV file and convert it into 2 DataFrames: One for questions+answers and the other for latent traits.
    :param filename: Path to csv
    :param leetcode: If True, parse as LeetCode-style CSV; otherwise, parse as Data Structure course format.
    :return: A DataFrame with columns Q_ID, Q_TEXT, CORRECT_TEXTS, WRONG_TEXTS, and DIFFICULTY.
    """
    df_raw = pd.read_csv(filename)

    if leetcode:
        # LeetCode-style CSV format
        df_questions = pd.DataFrame()
        df_questions['Q_ID'] = df_raw.index.astype(str)
        df_questions['Q_TEXT'] = df_raw['content']
        df_questions['CORRECT_TEXTS'] = df_raw['python'].apply(lambda x: [x] if pd.notnull(x) else [])
        df_questions['WRONG_TEXTS'] = [[] for _ in range(len(df_raw))]
        df_questions['DIFFICULTY'] = df_raw['normalized_difficulty']

    else:
        # Data Structures course format
        df_questions = pd.DataFrame()
        df_questions['Q_ID'] = df_raw.index.astype(str)
        df_questions['Q_TEXT'] = df_raw['question_translated_to_English']
        df_questions['CORRECT_TEXTS'] = df_raw['answer_translated_to_English'].apply(lambda x: [x] if pd.notnull(x) else [])
        df_questions['WRONG_TEXTS'] = [[] for _ in range(len(df_raw))]
        df_questions['DIFFICULTY'] = df_raw['Difficulty']

    return df_questions
