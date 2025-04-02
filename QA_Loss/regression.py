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


def run_regression(filename: str) -> None:
    # dataset - Change the csv name to point to the correct data
    data = pd.read_csv(filename)

    # Prepare input and target variables
    X = data[['loss']]
    y = data['difficulty']

    # Split dataset into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and hyperparameter grids
    models_to_evaluate = [RandomForestRegressor(), DecisionTreeRegressor(), LinearRegression(), SVR()]
    model_names = ['RF Regressor', 'DT Regressor', 'Linear Regressor', 'SVR']
    parameters_for_model_evaluation = [
        {
            'regressor__n_estimators': [10, 25, 50, 100, 150, 200, 250],
            'regressor__max_depth': [2, 5, 10, 15, 25, 50]
        },  # RF
        {
            'regressor__max_features': [1, 2, None],
            'regressor__max_depth': [2, 5, 10, 20, 50]
        },  # DT
        {
            'regressor__fit_intercept': [True, False]
        },  # LR (Note: 'normalize' is deprecated in newer versions of sklearn)
        {
            'regressor__kernel': ['linear', 'poly', 'rbf'],
            'regressor__gamma': ['auto', 'scale'],
            'regressor__shrinking': [True, False],
            'regressor__degree': [1, 2, 3, 4]
        },  # SVR
    ]

    # Iterate through models, perform GridSearchCV, and evaluate performance
    best_models = {}

    for model, params, name in zip(models_to_evaluate, parameters_for_model_evaluation, model_names):
        print(f"Training and tuning {name}...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalize features
            ('regressor', model)
        ])

        grid_search = GridSearchCV(pipeline, param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_models[name] = best_model

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Test MSE for {name}: {mse}\n")

    # Use the best model (lowest test MSE) for inference
    best_model_name = min(best_models, key=lambda k: mean_squared_error(y_test, best_models[k].predict(X_test)))
    final_model = best_models[best_model_name]

    print(f"Best overall model: {best_model_name}")

    # Predict difficulty for new loss values
    new_loss = np.array([[0.35], [0.65]])
    predicted_difficulty = final_model.predict(new_loss)
    print("Predicted difficulty scores:", predicted_difficulty)


if __name__ == "__main__":
    run_regression()
