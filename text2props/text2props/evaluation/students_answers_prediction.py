from text2props.constants import S_ID, TIMESTAMP, CORRECT, Q_ID, DIFFICULTY, DISCRIMINATION
from ._prediction_methods import irt_prediction_with_update
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from typing import Dict, Iterable


def evaluate_students_answers_prediction(
        df_sap: pd.DataFrame,
        df_train: pd.DataFrame,
        dict_real_latent_traits: Dict[str, Dict[str, float]],
        dict_estimated_latent_traits: Dict[str, Dict[str, float]],
) -> None:
    df_sap = df_sap.sort_values([S_ID, TIMESTAMP])
    true_results = df_sap[CORRECT].values
    user_id_list = list(df_sap[S_ID].unique())

    list_train_questions = list(df_train[Q_ID].unique())
    for q_id in list_train_questions:
        dict_estimated_latent_traits[DIFFICULTY][q_id] = dict_real_latent_traits[DIFFICULTY][q_id]
        dict_estimated_latent_traits[DISCRIMINATION][q_id] = dict_real_latent_traits[DISCRIMINATION][q_id]

    print("[INFO] Doing prediction with ground truth IRT latent traits...")
    dict_irt_predicted_results = irt_prediction_with_update(df_sap, dict_real_latent_traits)
    irt_predicted_results = []
    for user_id in user_id_list:
        irt_predicted_results.extend(dict_irt_predicted_results[user_id])
    print('IRT estimation: ' + gen_output(irt_predicted_results, true_results))

    print("[INFO] Doing prediction with predicted latent traits...")
    dict_nlp_predicted_results = irt_prediction_with_update(df_sap, dict_estimated_latent_traits)
    nlp_predicted_results = []
    for user_id in user_id_list:
        nlp_predicted_results.extend(dict_nlp_predicted_results[user_id])
    print('NLP estimation: ' + gen_output(nlp_predicted_results, true_results))

    print("[INFO] Doing majority prediction...")
    majority_prediction = np.ones(len(true_results)) * np.mean(true_results)
    print('Majority: ' + gen_output(majority_prediction, true_results))


def gen_output(predicted_results: Iterable[float], true_results: Iterable[bool]) -> str:
    predicted_results_bool = np.array(predicted_results) >= 0.5
    tn, fp, fn, tp = confusion_matrix(true_results, predicted_results_bool).ravel()
    output_str = ''
    output_str += 'acc: %.3f | ' % ((tp+tn)/(tp+tn+fp+fn))
    output_str += 'prec correct: %.3f | ' % (tp/(tp+fp))
    output_str += 'rec correct: %.3f | ' % (tp/(tp+fn))
    output_str += 'prec wrong: %.3f | ' % (tn/(tn+fn))
    output_str += 'rec wrong: %.3f | ' % (tn/(tn+fp))
    output_str += 'auc: %.5f' % roc_auc_score(true_results, predicted_results)
    return output_str
