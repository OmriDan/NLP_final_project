from ..constants import (
    QUESTION_LATENT_TRAITS_DF_COLS,
    ANSWERS_DF_COLS,
    QUESTION_DF_COLS,
)


def check_questions_lt_columns(input_df):
    if not set(QUESTION_LATENT_TRAITS_DF_COLS).issubset(set(input_df.columns)):
        raise ValueError("The input dataframe contains the wrong columns")


def check_answers_df_columns(input_df):
    if not set(ANSWERS_DF_COLS).issubset(set(input_df.columns)):
        raise ValueError("The input dataframe contains the wrong columns")


def check_question_df_columns(input_df):
    if not set(QUESTION_DF_COLS).issubset(set(input_df.columns)):
        raise ValueError("The input dataframe contains the wrong columns")
