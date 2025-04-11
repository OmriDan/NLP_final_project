import pyximport

pyximport.install(language_level=3)
import parse_data
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# Import text2props modules
from text2props.constants import DATA_PATH, DIFFICULTY, DIFFICULTY_RANGE, Q_ID
from text2props.model import Text2PropsModel
from text2props.modules.estimators_from_text import FeatureEngAndRegressionPipeline, \
    FeatureEngAndRegressionEstimatorFromText
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.feature_engineering.utils import vectorizer_text_preprocessor
from text2props.modules.feature_engineering.components import IRFeaturesComponent, ReadabilityFeaturesComponent
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator
from text2props.modules.regression import RegressionModule
from text2props.modules.regression.components import SklearnRegressionComponent
from text2props.evaluation.latent_traits_estimation import compute_error_metrics_latent_traits_estimation_regression

# Import custom vectorizers
from custom_vectorizers import Word2VecVectorizer, BERTVectorizer, CodeBERTVectorizer


# Define your custom python_preprocessor
def python_preprocessor(text):
    """
    Custom preprocessor to retain Python syntax. Basically, minimal preprocessing.
    """
    return text.strip()


# Set random seed and file suffix
SEED = 42
suffix = "leetcode"

# Load data and split
df_questions = parse_data.parse_data(f"../data/raw/merged_leetcode_df.csv", True, True,
                                     f"../data/processed/known_latent_traits_{suffix}.pickle")
df_train, df_test = train_test_split(df_questions, test_size=0.2, random_state=SEED)

# Get known difficulties from training data (a dictionary mapping q_id to difficulty)
dict_difficulty = df_train.set_index(Q_ID)[DIFFICULTY].to_dict()

# Define the experiment configurations.
# For BERT and CodeBERT we iterate over use_python_preproc in {True, False}.
# For all vectorizers, we test with and without readability features.
experiments = []

# TF-IDF experiment: no preproc variant option; readability on/off.
for use_readability in [True, False]:
    experiments.append({
        "vectorizer": "tfidf",
        "use_readability": use_readability,
        "use_python_preproc": True  # not really used; we always use python_preprocessor for tfidf in this experiment
    })

# Word2Vec experiment: similar treatment.
for use_readability in [True, False]:
    experiments.append({
        "vectorizer": "word2vec",
        "use_readability": use_readability,
        "use_python_preproc": True  # assuming Word2VecVectorizer is always instantiated with python_preprocessor
    })

# BERT experiments: with and without python_preprocessor, and readability on/off.
for use_preproc in [True, False]:
    for use_readability in [True, False]:
        experiments.append({
            "vectorizer": "bert",
            "use_readability": use_readability,
            "use_python_preproc": use_preproc
        })

# CodeBERT experiments: with and without python_preprocessor, and readability on/off.
for use_preproc in [True, False]:
    for use_readability in [True, False]:
        experiments.append({
            "vectorizer": "codebert",
            "use_readability": use_readability,
            "use_python_preproc": use_preproc
        })

# Loop over all experiments
for exp in experiments:
    vectorizer_type = exp["vectorizer"]
    use_readability = exp["use_readability"]
    use_python_preproc = exp["use_python_preproc"]

    # Create a name string for the experiment to use in saving files.
    exp_name = vectorizer_type
    if vectorizer_type in ["bert", "codebert"]:
        exp_name += "_python_preproc" if use_python_preproc else "_default_preproc"
    if use_readability:
        exp_name += "_readability"
    else:
        exp_name += "_no_readability"

    print(f"[INFO] Running experiment: {exp_name}")

    # Set up the vectorizer instance
    if vectorizer_type == "tfidf":
        vectorizer_instance = TfidfVectorizer(
            stop_words='english',
            preprocessor=python_preprocessor,  # fixed preprocessor for tfidf
            min_df=0.02,
            max_df=0.92
        )
    elif vectorizer_type == "word2vec":
        # Adjust the model path if needed.
        vectorizer_instance = Word2VecVectorizer(model_name="word2vec-google-news-300")
    elif vectorizer_type == "bert":
        # For BERT, choose preprocessor based on the experiment flag.
        chosen_preproc = python_preprocessor if use_python_preproc else vectorizer_text_preprocessor
        vectorizer_instance = BERTVectorizer(model_name="bert-base-uncased", preprocessor=chosen_preproc)
    elif vectorizer_type == "codebert":
        # For CodeBERT, similarly choose the preprocessor.
        chosen_preproc = python_preprocessor if use_python_preproc else vectorizer_text_preprocessor
        vectorizer_instance = CodeBERTVectorizer(model_name="microsoft/codebert-base", preprocessor=chosen_preproc)
    else:
        raise ValueError("Unknown vectorizer type.")

    # Build the IR component with the chosen vectorizer
    ir_component = IRFeaturesComponent(
        vectorizer_instance,
        concatenate_correct=True,  # include answer options as in your original code
        concatenate_wrong=False
    )

    # Build a list of feature engineering components.
    components = [ir_component]
    if use_readability:
        components.append(ReadabilityFeaturesComponent())

    # Build the full pipeline using the chosen features and regression module.
    pipeline_difficulty = FeatureEngAndRegressionPipeline(
        FeatureEngineeringModule(components),
        RegressionModule([
            SklearnRegressionComponent(
                RFRegressor(n_estimators=100, max_depth=20, random_state=SEED),
                latent_trait_range=DIFFICULTY_RANGE
            )
        ])
    )

    # Create the model using the known difficulty calibrator.
    model = Text2PropsModel(
        KnownParametersCalibrator({DIFFICULTY: dict_difficulty}),
        FeatureEngAndRegressionEstimatorFromText({DIFFICULTY: pipeline_difficulty})
    )

    # Train the model on the training set.
    model.train(df_train)
    print(f"[INFO] Model trained for experiment: {exp_name}")

    # Predict difficulties on the test set.
    dict_predictions = model.predict(df_test)

    # Build a dictionary mapping question IDs to the predicted difficulty.
    predicted_difficulty = {}
    for idx, q_id in enumerate(df_test[Q_ID].values):
        predicted_difficulty[q_id] = dict_predictions[DIFFICULTY][idx]

    # Save the predicted latent traits to a pickle file.
    pred_file = f"../data/processed/predicted_latent_traits_{suffix}_{exp_name}.pickle"
    with open(pred_file, "wb") as f:
        pickle.dump(predicted_difficulty, f)
    print(f"[INFO] Saved predicted latent traits to {pred_file}")

    # To compute metrics, we compare the true difficulty (from test set) with predicted values.
    # Here we assume the true difficulties are in the test set's DIFFICULTY column.
    true_difficulties = df_test[DIFFICULTY].values
    predicted_difficulties = [predicted_difficulty[q_id] for q_id in df_test[Q_ID].values]

    # Compute variance and standard deviation for the training and test set difficulties.
    true_difficulties_train = df_train[DIFFICULTY].values
    variance_train = np.var(true_difficulties_train)
    std_dev_train = np.sqrt(variance_train)

    variance_test = np.var(true_difficulties)
    std_dev_test = np.sqrt(variance_test)

    # Compute error metrics for the prediction.
    error_metrics = compute_error_metrics_latent_traits_estimation_regression(true_difficulties, predicted_difficulties)

    # Combine all metrics into one dictionary.
    metrics_dict = {
        "error_metrics": error_metrics,
        "variance_train": variance_train,
        "std_dev_train": std_dev_train,
        "variance_test": variance_test,
        "std_dev_test": std_dev_test,
        "experiment": exp_name
    }

    # Save the metrics to a pickle file.
    metrics_file = f"../data/processed/evaluation_metrics_{suffix}_{exp_name}.pickle"
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics_dict, f)
    print(f"[INFO] Saved evaluation metrics to {metrics_file}\n")

print("[INFO] All experiments completed.")
