import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
from text2props.constants import Q_TEXT
from ..utils import gen_wrong_answers_dict, gen_correct_answers_dict, concatenate_answers_text_into_question_text_df
from . import BaseFeatEngComponent


class IRFeaturesComponent(BaseFeatEngComponent):

    def __init__(self, vectorizer: CountVectorizer, concatenate_correct: bool = None, concatenate_wrong: bool = None):
        self.vectorizer = vectorizer
        self.correct_text_dict = None
        self.wrong_text_dict = None
        self.concatenate_correct = concatenate_correct
        self.concatenate_wrong = concatenate_wrong

    def fit_transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        Fit the IRFeaturesComponent and transforms the input data. Returns a sparse matrix.
        :param input_df:
        :return:
        """
        if Q_TEXT not in input_df.columns:
            raise ValueError("Q_TEXT should be in input_df.columns")
        self.correct_text_dict = gen_correct_answers_dict(input_df)
        self.wrong_text_dict = gen_wrong_answers_dict(input_df)
        local_df = input_df.copy()
        local_df[Q_TEXT] = concatenate_answers_text_into_question_text_df(
            local_df, self.correct_text_dict, self.wrong_text_dict, self.concatenate_correct, self.concatenate_wrong)
        return coo_matrix(self.vectorizer.fit_transform(local_df[Q_TEXT].values))

    def transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        Transforms the input data using the trained vectorizer. Returns a sparse matrix.
        :param input_df:
        :return:
        """
        if Q_TEXT not in input_df.columns:
            raise ValueError("Q_TEXT should be in input_df.columns")
        self.correct_text_dict.update(gen_correct_answers_dict(input_df))
        self.wrong_text_dict.update(gen_wrong_answers_dict(input_df))
        local_df = input_df.copy()
        local_df[Q_TEXT] = concatenate_answers_text_into_question_text_df(
            local_df, self.correct_text_dict, self.wrong_text_dict, self.concatenate_correct, self.concatenate_wrong)
        return coo_matrix(self.vectorizer.transform(local_df[Q_TEXT].values))
