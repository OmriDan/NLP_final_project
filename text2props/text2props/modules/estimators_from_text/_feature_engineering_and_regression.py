from typing import Dict, List
from ._base import BaseEstimatorFromText
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.regression import RegressionModule
from text2props.constants import Q_ID
import pandas as pd


class FeatureEngAndRegressionPipeline(object):
    def __init__(self, feature_engineering: FeatureEngineeringModule, regression: RegressionModule):
        self.feat_eng_module = feature_engineering
        self.regression_module = regression

    def train(self, input_df: pd.DataFrame, y: List[float]):
        partial_results = self.feat_eng_module.fit_transform(input_df)
        self.regression_module.train(partial_results, y)

    def predict(self, input_df: pd.DataFrame) -> List[float]:
        partial_results = self.feat_eng_module.transform(input_df)
        return self.regression_module.predict(partial_results)

    def randomized_cv_train(
            self,
            param_distributions: List[Dict[str, List[float]]],
            df_train: pd.DataFrame,
            y_train: List,
            n_iter: int,
            n_jobs: int,
            cv: int,
            random_state: int
    ) -> float:
        partial_results = self.feat_eng_module.fit_transform(df_train)
        score = self.regression_module.randomized_cv_train(
            partial_results, y_train, param_distributions=param_distributions, n_iter=n_iter, cv=cv, n_jobs=n_jobs,
            random_state=random_state
        )
        return score


class FeatureEngAndRegressionEstimatorFromText(BaseEstimatorFromText):
    """
    The FeatureEngAndRegressionEstimatorFromText object is made of one or more pipelines, which work in parallel, each
    working on one latent trait. Each pipeline is made of two building blocks: the FeatureEngineering module and the
    Regression module.
    """

    def __init__(self, pipelines: Dict[str, FeatureEngAndRegressionPipeline]):
        super().__init__()
        self.pipelines = pipelines

    def train(self, df_train: pd.DataFrame, ground_truth_latent_traits: Dict[str, Dict[str, float]]):
        """
        Trains the EstimatorFromText object, training all the pipelines contained in the object.
        :param df_train:
        :param ground_truth_latent_traits:
        :return:
        """
        for latent_trait in self.pipelines.keys():
            local_y = [ground_truth_latent_traits[latent_trait][q_id] for q_id in df_train[Q_ID].values]
            self.pipelines[latent_trait].train(df_train, local_y)

    def predict(self, input_df: pd.DataFrame) -> Dict[str, List]:
        """
        Performs the prediction. The returned object is a dictionary whose keys are the names of the latent traits.
        :param input_df:
        :return:
        """
        predictions = dict()
        for latent_trait in self.pipelines.keys():
            predictions[latent_trait] = self.pipelines[latent_trait].predict(input_df)
        return predictions

    def randomized_cv_train(
            self,
            param_distributions: Dict[str, List[Dict[str, List[float]]]],
            df_train: pd.DataFrame,
            ground_truth_latent_traits: Dict[str, Dict[str, float]],
            n_iter: int = 10,
            n_jobs: int = None,
            cv: int = None,
            random_state: int = None,
    ) -> Dict[str, float]:
        """
        Trains the EstimatorFromText object with RandomizedCV, bu training all the pipelines, and returns the scores so
        that they can be compared with the results obtained with other models.
        :param param_distributions: distribution of parameters for randomized CV. The key is the name of the latent
          train, the values are lists of dictionaries (one list for each component of the regression module)
        :param df_train:
        :param ground_truth_latent_traits:
        :param n_iter:
        :param n_jobs:
        :param cv:
        :param random_state:
        :return:
        """
        scores = dict()
        for latent_trait in self.pipelines.keys():
            local_y = [ground_truth_latent_traits[latent_trait][str(q_id)] for q_id in df_train[Q_ID].values]
            scores[latent_trait] = self.pipelines[latent_trait].randomized_cv_train(
                param_distributions=param_distributions[latent_trait],
                df_train=df_train,
                y_train=local_y,
                n_iter=n_iter,
                n_jobs=n_jobs,
                cv=cv,
                random_state=random_state
            )
        return scores
