import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from text2props.utils.math import inverse_item_response_function, item_response_function, information_function
from text2props.constants import (
    S_ID,
    CORRECT,
    Q_ID,
    DEFAULT_GUESS,
    DEFAULT_DISCRIMINATION,
    DEFAULT_SLIP,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
    DIFFICULTY_RANGE,
    DIFFICULTY,
    DISCRIMINATION,
)


def perform_user_irt_prediction(
        interactions_df: pd.DataFrame,
        difficulty_dict: Dict[str, float],
        discrimination_dict: Dict[str, float],
        difficulty_range: Tuple[float, float] = DIFFICULTY_RANGE,
        theta_increment: float = 0.1,
        initial_theta: float = (DIFFICULTY_MAX+DIFFICULTY_MIN)/2,
        guess: float = DEFAULT_GUESS,
        slip: float = DEFAULT_SLIP,
) -> List[float]:
    """
    :param interactions_df: dataframe containing all the interactions between users and items
    :param difficulty_dict:
    :param discrimination_dict:
    :param difficulty_range: tuple containing min and max difficulty
    :param theta_increment: the granularity of the skill level we are interested in
    :param initial_theta: starting skill level for the estimation
    :param guess: guess factor to use in the IRT model
    :param slip: slip factor to use in the IRT model
    :return: the list containing the predicted results for the interactions in the input dataframe
    """
    predicted_result = []
    estimated_theta = initial_theta
    thetas = np.arange(difficulty_range[0], difficulty_range[1] + theta_increment, theta_increment)
    log_likelihood = np.zeros(len(thetas), dtype=np.float)
    information_func = np.zeros(len(thetas), dtype=np.float)
    list_loglikelihood = np.zeros(len(thetas), dtype=object)
    list_information_function = np.zeros(len(thetas), dtype=object)
    for idx, theta in enumerate(thetas):
        list_loglikelihood[idx] = []
        list_information_function[idx] = []

    for true_result, item_id in interactions_df[[CORRECT, Q_ID]].values:
        if item_id in difficulty_dict.keys() and item_id in discrimination_dict.keys():
            difficulty, discrimination = difficulty_dict[item_id], discrimination_dict[item_id]
        else:
            print("[INFO] Question with ID %s was not known. Manually set latent traits" % item_id)
            difficulty, discrimination = (DIFFICULTY_MAX+DIFFICULTY_MIN)/2, DEFAULT_DISCRIMINATION

        predicted_result.append(item_response_function(difficulty, estimated_theta, discrimination, guess, slip))

        func = item_response_function if true_result == 1 else inverse_item_response_function
        for idx, theta in enumerate(thetas):
            list_loglikelihood[idx].append(np.log(func(difficulty, theta, discrimination, guess, slip)))
            log_likelihood[idx] = np.sum(list_loglikelihood[idx])
            list_information_function[idx].append(information_function(difficulty, theta, discrimination, guess, slip))
            information_func[idx] = np.sum(list_information_function[idx])

        estimated_theta = thetas[np.argmax(log_likelihood)]

    return predicted_result


def irt_prediction_with_update(
        interactions_df: pd.DataFrame,
        latent_traits_dict: Dict[str, Dict[str, float]],
        difficulty_range: Tuple[float, float] = DIFFICULTY_RANGE,
        theta_increment: float = 0.1,
        initial_theta: float = (DIFFICULTY_MAX + DIFFICULTY_MIN) / 2,
        guess: float = DEFAULT_GUESS,
        slip: float = DEFAULT_SLIP,
) -> Dict[str, List[float]]:
    """
    Performs the task of students' performance prediction for all the students in the interactions_df. It does so by
    calling the perform_user_irt_prediction method, which works on one students only.
    :param interactions_df: input dataframe containing the answers given by the students to the questions.
    :param latent_traits_dict: dictionary containing the latent traits of the questions
    :param difficulty_range: range of difficulties to consider
    :param theta_increment: granularity of the estimated theta
    :param initial_theta: initial estimation of the difficulty for new items (default: middle point of difficulty range)
    :param guess: guess factor
    :param slip: slip factor
    :return: a dictionary containing the predicted answers of each student (student IDs are the keys). The predicted
      answer is a float representing the probability that the student answer that question correctly.
    """
    predicted_result = dict()
    for user_id in interactions_df[S_ID].unique():
        predicted_result[user_id] = perform_user_irt_prediction(
            interactions_df=interactions_df[interactions_df[S_ID] == user_id],
            difficulty_dict=latent_traits_dict[DIFFICULTY],
            discrimination_dict=latent_traits_dict[DISCRIMINATION],
            difficulty_range=difficulty_range,
            theta_increment=theta_increment,
            initial_theta=initial_theta,
            guess=guess,
            slip=slip,
            )
    return predicted_result
