from typing import Dict, List

from .constants import *
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error,
    ndcg_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_error_metrics_latent_traits_estimation(
        y_true: List[float],
        y_pred: List[float],
        metrics: List[str] = None
) -> Dict[str, float]:
    # this is kept for backward compatibility, since scripts written before November 2022 could use this instead of the
    # one with "regression" in the name.
    return compute_error_metrics_latent_traits_estimation_regression(y_true, y_pred, metrics)


def compute_error_metrics_latent_traits_estimation_regression(
        y_true: List[float],
        y_pred: List[float],
        metrics: List[str] = None
) -> Dict[str, float]:
    dict_errors = dict()
    if metrics is None:
        metrics = [MAE, MSE, RMSE, R2, MAX_ERROR, MIN_ERROR, NDCG]
    if MAE in metrics:
        dict_errors[MAE] = mean_absolute_error(y_true, y_pred)
    if MSE in metrics:
        dict_errors[MSE] = mean_squared_error(y_true, y_pred)
    if RMSE in metrics:
        dict_errors[RMSE] = np.sqrt(mean_squared_error(y_true, y_pred))
    if R2 in metrics:
        dict_errors[R2] = r2_score(y_true, y_pred)
    if MAX_ERROR in metrics:
        dict_errors[MAX_ERROR] = max_error(y_true, y_pred)
    if MIN_ERROR in metrics:
        dict_errors[MIN_ERROR] = np.min([np.abs(y_true[idx] - y_pred[idx]) for idx in range(len(y_true))])
    if NDCG in metrics:
        dict_errors[NDCG] = ndcg_score([[x + 5 for x in y_true]], [[x + 5 for x in y_pred]])
    return dict_errors


def compute_eval_metrics_latent_traits_estimation_classification(
        y_true: List[bool],
        y_pred: List[bool],
        metrics: List[str] = None
    ) -> Dict[str, float]:
    dict_errors = dict()
    if metrics is None:
        metrics = [ACC, PREC, REC, F1]
    if ACC in metrics:
        dict_errors[ACC] = accuracy_score(y_true, y_pred)
    if PREC in metrics:
        dict_errors[PREC] = precision_score(y_true, y_pred)
    if REC in metrics:
        dict_errors[REC] = recall_score(y_true, y_pred)
    if F1 in metrics:
        dict_errors[F1] = f1_score(y_true, y_pred)
    return dict_errors
