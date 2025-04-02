from ..constants import Q_ID, DATA_PATH
from ..data_validation import check_question_df_columns, check_answers_df_columns
from ..evaluation.latent_traits_estimation import (
    compute_error_metrics_latent_traits_estimation_regression,
    compute_eval_metrics_latent_traits_estimation_classification,
)
from ..modules.estimators_from_text import BaseEstimatorFromText
from ..modules.latent_traits_calibration import BaseLatentTraitsCalibrator
import os
import pandas as pd
import pickle
from typing import Dict, List, Optional


class Text2PropsModel(object):
    def __init__(
            self,
            latent_traits_calibrator: BaseLatentTraitsCalibrator,
            estimator_from_text: BaseEstimatorFromText
    ):
        self.latent_traits_calibrator = latent_traits_calibrator
        self.n_latent_traits = latent_traits_calibrator.get_n_latent_traits()
        self.latent_traits = latent_traits_calibrator.get_name_latent_traits()
        self.estimator_from_text = estimator_from_text
        self.ground_truth_latent_traits = None

    def calibrate_latent_traits(self, df_gte: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Performs the initial calibration of latent traits. These latent traits are the ones that will later be used as
        ground truth while training the model that performs the estimation of latent traits from text.
        :param df_gte: dataframe containing the answers given by students to questions
        :return:
        """
        if self.latent_traits_calibrator is None:
            raise ValueError("No LatentTraitsCalibrator defined")
        self.ground_truth_latent_traits = self.latent_traits_calibrator.calibrate_latent_traits(df_gte)
        return self.ground_truth_latent_traits

    def train(self, df_train: pd.DataFrame, df_gte: pd.DataFrame = None):
        """
        Train the model, by training the latent traits calibrator and estimator from text objects. The df_gte dataframe
        might be None for compatibility with the KnownParametersCalibrator, which does not require the calibration of
        latent traits as they are known in advance.
        :param df_train: dataframe containing the textual information about the questions
        :param df_gte: (optional) dataframe containing the interactions between students and questions, to be used for
          estimating the ground truth latent traits.
        :return:
        """
        self.calibrate_latent_traits(df_gte)
        self.estimator_from_text.train(df_train, self.ground_truth_latent_traits)

    def predict(self, input_df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Receives as input the dataframe containing the textual information and returns the estimated latent traits.
        :param input_df:
        :return:
        """
        return self.estimator_from_text.predict(input_df)

    def randomized_cv_train(
            self,
            param_distributions: Dict[str, List[Dict[str, List[float]]]],
            df_train: pd.DataFrame,
            df_gte: pd.DataFrame = None,
            n_iter: int = 10,
            n_jobs: int = None,
            cv: int = None,
            random_state: int = None,
            perform_calibration: bool = False
    ) -> Dict[str, float]:
        """
        Performs the training with RandomizedCV of the EstimatorFromText object of the Text2PropsModel object. Then, it
        returns the scores. It can work both with already calibrated latent traits and with latent traits to be
        calibrated, it can be specified by means of the perform_calibration parameter.
        :param param_distributions: probability distribution of the parameters for the Randomized CV
        :param df_train: training dataset, it needs to have the columns specified in QUESTION_DF_COLS
        :param df_gte: dataset to perform the calibration of ground truth latent traits, it needs to have the columns
          specified in ANSWERS_DF_COLS
        :param n_iter: number of iteration of RandomizedCV
        :param n_jobs: number of jobs for parallelization
        :param cv:
        :param random_state: for reproducibility
        :param perform_calibration: boolean to select if ground truth latent have to be calibrated
        :return: The scores obtained by the best performing model in the RandomizedCV
        """
        if perform_calibration:
            check_answers_df_columns(df_gte)
        check_question_df_columns(df_train)
        if perform_calibration:
            self.calibrate_latent_traits(df_gte)
        scores = self.estimator_from_text.randomized_cv_train(
            param_distributions=param_distributions,
            df_train=df_train,
            ground_truth_latent_traits=self.ground_truth_latent_traits,
            n_iter=n_iter,
            n_jobs=n_jobs,
            cv=cv,
            random_state=random_state
        )
        return scores

    def compute_error_metrics_latent_traits_estimation(
        self,
        input_df: pd.DataFrame,
        classification: bool = False,
        int_to_class_mapper=None,
        metrics: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Performs the prediction from the input dataframe, and compute the error metrics for the estimation of each
        latent trait. The results are returned in a dictionary whose keys are the name of the latent traits.
        :param input_df:
        :param classification:
        :param int_to_class_mapper:
        :param metrics:
        :return:
        """
        if classification and int_to_class_mapper is None:
            raise ValueError("If classification, you must pass the mapping method")
        results = dict()
        predictions = self.predict(input_df)
        for lt in self.latent_traits:
            results[lt] = dict()
            y_pred = predictions[lt]
            y_true = [self.ground_truth_latent_traits[lt][q_id] for q_id in input_df[Q_ID].values]
            if classification:
                y_pred = [int_to_class_mapper(x) for x in y_pred]
                results[lt] = compute_eval_metrics_latent_traits_estimation_classification(y_true, y_pred, metrics)
            else:
                results[lt] = compute_error_metrics_latent_traits_estimation_regression(y_true, y_pred, metrics)
            return results

    def get_calibrated_latent_traits(self) -> Dict[str, Dict[str, float]]:
        """
        Returns the calibrated latent traits using the get_calibrated_latent_traits of the latent traits calibrator obj
        :return:
        """
        return self.latent_traits_calibrator.get_calibrated_latent_traits()

    def store_calibrated_latent_traits(
            self,
            output_data_path: str = DATA_PATH,
            output_filename: str = 'latent_traits.p'
    ):
        """
        Stores, using pickle.dump, the calibrated latent traits in the file specified as parameter (optional)
        :param output_data_path: output directory to store the data in
        :param output_filename: the name of the output file
        :return:
        """
        latent_traits = self.latent_traits_calibrator.get_calibrated_latent_traits()
        pickle.dump(latent_traits, open(os.path.join(output_data_path, output_filename), "wb"))
