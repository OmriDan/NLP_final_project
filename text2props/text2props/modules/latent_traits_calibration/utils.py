import pandas as pd
from text2props.constants import (
    ANSWERS_DF_COLS,
    Q_ID,
    CORRECT,
    S_ID,
    TIMESTAMP,
    DIFFICULTY_RANGE,
    DEFAULT_GUESS,
    DEFAULT_DISCRIMINATION,
    DIFFICULTY,
    DISCRIMINATION,
)
from pyirt import irt


def irt_estimation(
        df_gte: pd.DataFrame,
        difficulty_range=DIFFICULTY_RANGE,
        discrimination_range=(DEFAULT_DISCRIMINATION, DEFAULT_DISCRIMINATION),
        guess=DEFAULT_GUESS
) -> (dict, dict):
    """
    This method performs the IRT estimation of the items from the interactions stored in DF GTE. If necessary, adds some
    artificial answers, since pyirt requires all the questions to have at least one wrong and one correct answer.
    :param df_gte:
    :param difficulty_range:
    :param discrimination_range:
    :param guess:
    :return:
    """
    # Add perfectly bad and perfectly good students, in order for pyirt to work
    q_cnt_per_correctness = df_gte.groupby([Q_ID, CORRECT]).size().reset_index().groupby(Q_ID).size().reset_index()
    questions_to_add = list(q_cnt_per_correctness[q_cnt_per_correctness[0] < 2][Q_ID])
    num_q_to_add = len(questions_to_add)
    print('[INFO] %d questions to fill in out of %d' % (num_q_to_add, len(df_gte[Q_ID].unique())))
    questions_to_add_df = pd.DataFrame(
        {
            S_ID: ['P_GOOD'] * num_q_to_add + ['P_BAD'] * num_q_to_add,
            TIMESTAMP: [None] * 2 * num_q_to_add,
            CORRECT: [True] * num_q_to_add + [False] * num_q_to_add,
            Q_ID: questions_to_add + questions_to_add,
        }
    )
    df = pd.concat([df_gte[ANSWERS_DF_COLS], questions_to_add_df[ANSWERS_DF_COLS]], ignore_index=True)

    interactions_list = [(user, item, correctness) for user, item, correctness in df[[S_ID, Q_ID, CORRECT]].values]
    try:
        item_params, user_params = irt(
            interactions_list,
            theta_bnds=difficulty_range,
            beta_bnds=difficulty_range,
            alpha_bnds=discrimination_range,
            in_guess_param={q: guess for q in df[Q_ID].unique()},
            max_iter=100
        )
    except Exception:
        raise ValueError("Problem in IRTCalibrator. Check if there are items with only correct/wrong answers.")

    question_dict = dict()
    question_dict[DIFFICULTY] = dict()
    question_dict[DISCRIMINATION] = dict()
    for question, question_params in item_params.items():
        question_dict[DIFFICULTY][question] = -question_params['beta']
        question_dict[DISCRIMINATION][question] = question_params["alpha"]

    return question_dict
