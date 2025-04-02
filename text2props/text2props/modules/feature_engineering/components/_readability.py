import pandas as pd
import textstat
from scipy.sparse import coo_matrix

from text2props.constants import Q_TEXT
from . import BaseFeatEngComponent


class ReadabilityFeaturesComponent(BaseFeatEngComponent):

    def __init__(self, use_smog: bool = True, version: int = 1):
        self.use_smog = use_smog
        self.version = version

    def fit_transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        Transforms the input data and returns a sparse matrix. It returns a sparse matrix, although these features are
        not sparse for consistency reasons with the other components.
        :param input_df:
        :return:
        """
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        It receives a DF as parameter and computes the readability features, returning them in a feature array.
        The DF must have Q_TEXT in its columns. Returns a sparse matrix for consistency with the type returned by the
        other components.
        :param input_df:
        :return:
        """
        if Q_TEXT not in input_df.columns:
            raise ValueError("Q_TEXT should be in input_df.columns")
        df = pd.DataFrame()
        df['flesch_reading_ease'] = input_df.apply(lambda r: textstat.flesch_reading_ease(r[Q_TEXT]), axis=1)
        df['flesch_kincaid_grade_level'] = input_df.apply(lambda r: textstat.flesch_kincaid_grade(r[Q_TEXT]), axis=1)
        df['automated_readability_index'] = input_df.apply(
            lambda r: textstat.automated_readability_index(r[Q_TEXT]), axis=1)
        df['gunning_fog_index'] = input_df.apply(lambda r: textstat.gunning_fog(r[Q_TEXT]), axis=1)
        df['coleman_liau'] = input_df.apply(lambda r: textstat.coleman_liau_index(r[Q_TEXT]), axis=1)
        if self.use_smog:
            df['smog_index'] = input_df.apply(lambda r: textstat.smog_index(r[Q_TEXT]), axis=1)
        if self.version > 1:
            df['linsear_write_formula'] = input_df.apply(lambda r: textstat.linsear_write_formula(r[Q_TEXT]), axis=1)
            df['dale_chall'] = input_df.apply(lambda r: textstat.dale_chall_readability_score(r[Q_TEXT]), axis=1)
        return coo_matrix(df.values)
