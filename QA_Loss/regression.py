import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from evaluation import compute_error_metrics


def run_regression(
    dataset_name: str,
    estimator=None,
    estimator_params: dict = None,
    filename: str = None,
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
) -> None:
    """
    Train and evaluate regression models on loss vs. difficulty data.

    Loads or accepts train/test splits, fits a provided estimator or performs
    grid search over multiple model types, and saves evaluation metrics.

    Args:
        dataset_name (str): Identifier for output filenames and logs.
        estimator (type, optional): A regressor class (e.g., RandomForestRegressor). Defaults to None.
        estimator_params (dict, optional): Initialization parameters for the estimator. Defaults to None.
        filename (str, optional): CSV path with 'loss' and 'difficulty' columns. Required if train_df/test_df not provided.
        train_df (pd.DataFrame, optional): Pre-split training data. Defaults to None.
        test_df (pd.DataFrame, optional): Pre-split testing data. Defaults to None.

    Raises:
        ValueError: If neither filename nor both train_df and test_df are provided.

    Returns:
        None: Prints metrics and writes a metrics CSV.
    """
    # Load or split data
    if train_df is None or test_df is None:
        if filename is None:
            raise ValueError("Either filename or both train_df and test_df must be provided.")

        data = pd.read_csv(filename)
        X = data[['loss']]
        y = data['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:   # Use provided splits
        X_train = train_df[['loss']]
        y_train = train_df['difficulty']
        X_test = test_df[['loss']]
        y_test = test_df['difficulty']

    # If a single estimator is given, train and evaluate it directly
    if estimator is not None:
        print(f"Training with provided estimator: {estimator.__name__}")
        reg = estimator(**(estimator_params or {}))
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', reg)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE for {estimator.__name__}: {mse:.4f}")

        metrics = compute_error_metrics(y_test.tolist(), y_pred.tolist())
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"metrics_{dataset_name}.csv", index=False)
        return

    # Otherwise, evaluate multiple models via grid search
    models_to_evaluate = [
        RandomForestRegressor(),
        DecisionTreeRegressor(),
        LinearRegression(),
        SVR()
    ]
    model_names = [
        'RF Regressor', 'DT Regressor', 'Linear Regressor', 'SVR'
    ]
    parameters_for_model_evaluation = [
        {
            'regressor__n_estimators': [10, 25, 50, 100, 150, 200, 250],
            'regressor__max_depth': [2, 5, 10, 15, 25, 50]
        },
        {
            'regressor__max_features': [1, 2, None],
            'regressor__max_depth': [2, 5, 10, 20, 50]
        },
        {
            'regressor__fit_intercept': [True, False]
        },
        {
            'regressor__kernel': ['linear', 'poly', 'rbf'],
            'regressor__gamma': ['auto', 'scale'],
            'regressor__shrinking': [True, False],
            'regressor__degree': [1, 2, 3, 4]
        }
    ]
    best_models = {}
    # Grid search over each model
    for model, params, name in zip(models_to_evaluate, parameters_for_model_evaluation, model_names):
        print(f"Training and tuning {name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        grid_search = GridSearchCV(
            pipeline,
            param_grid=params,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_models[name] = best_model

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Test MSE for {name}: {mse:.4f}\n")

    best_model_name = min(
        best_models,
        key=lambda k: mean_squared_error(
            y_test,
            best_models[k].predict(X_test)
        )
    )
    final_model = best_models[best_model_name]
    print(f"Best overall model: {best_model_name}")

    y_test_pred = final_model.predict(X_test)
    metrics = compute_error_metrics(y_test.tolist(), y_test_pred.tolist())
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"metrics_{dataset_name}.csv", index=False)

if __name__ == "__main__":
    run_regression("Flan-T5", estimator=RandomForestRegressor, estimator_params={'max_depth': 5, 'n_estimators': 250}, filename="qa_loss.csv")