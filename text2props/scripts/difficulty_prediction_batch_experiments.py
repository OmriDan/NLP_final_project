import pyximport
pyximport.install(language_level=3)
import parse_data
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

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
import text2props.constants as const

# Import custom vectorizers
from custom_vectorizers import Word2VecVectorizer, BERTVectorizer, CodeBERTVectorizer

# Define your custom python_preprocessor
def python_preprocessor(text):
    """
    Custom preprocessor to retain Python syntax. Basically, minimal preprocessing.
    """
    return text.strip()

# Configuration Flags
DO_TFIDF = True
DO_WORD2VEC = False
DO_BERT = True
DO_CODEBERT = True

USE_READABILITY = False    # Run experiments with and without readability features
USE_PREPROC = False        # For BERT/CodeBERT: run with default and python_preprocessor
USE_FINETUNED = True      # For BERT/CodeBERT: include pre-trained and fine-tuned models

# Paths for fine-tuned models
fine_tuned_bert_path = "../models/bert-base-uncased_fine_tuned"
fine_tuned_codebert_path = "../models/microsoft_codebert-base_fine_tuned"

# Define custom python_preprocessor
def python_preprocessor(text):
    return text.strip()

num_jobs = -1  # Number of jobs for parallel processing of CPU bound processes

# Set seed and dataset suffix
SEED = 42
suffix = "NovaSky" 

# DS and Leetcode dataset code:
# database_name = "DS_tests_with_difficulty"
# suffix = "DS-test"
# df_questions = parse_data.parse_data(f"../data/raw/{database_name}.csv", False, True, f"../data/processed/known_latent_traits_{suffix}.pickle")

# NovaSky dataset code:
# Load and preprocess NovaSky dataset
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

ds = load_dataset("NovaSky-AI/labeled_numina_difficulty_162K")
df_raw = ds["train"].to_pandas()
# Filter invalid rows
df_filtered = df_raw[(df_raw['gpt_difficulty'] != -1)
                     & df_raw['problem'].notnull()
                     & (df_raw['problem'].astype(str).str.strip() != "")]
# Build questions DataFrame
df_questions = pd.DataFrame({
    const.Q_ID: df_filtered.index.astype(str),
    const.Q_TEXT: df_filtered['problem'],
    const.CORRECT_TEXTS: df_filtered['solution'].apply(lambda x: [x] if pd.notnull(x) and str(x).strip() != "" else []),
    const.WRONG_TEXTS: [[] for _ in range(len(df_filtered))],
    const.DIFFICULTY: df_filtered['gpt_difficulty_parsed'].apply(lambda x: (x - 1) / 9 if pd.notnull(x) else None)
})
# Save known latent traits
known_traits = {const.DIFFICULTY: dict(zip(df_questions[const.Q_ID], df_questions[const.DIFFICULTY]))}
with open(f"../data/processed/known_latent_traits_{suffix}.pickle", 'wb') as f:
    pickle.dump(known_traits, f)
# End of NovaSky dataset code.

# Split into train/test
df_train, df_test = train_test_split(df_questions, test_size=0.2, random_state=SEED)
# True difficulties map
dict_difficulty = df_train.set_index(Q_ID)[DIFFICULTY].to_dict()

# Build experiment configurations dynamically
experiments = []
for vec in ['tfidf', 'word2vec', 'bert', 'codebert']:
    if vec == 'tfidf' and not DO_TFIDF:
        continue
    if vec == 'word2vec' and not DO_WORD2VEC:
        continue
    if vec == 'bert' and not DO_BERT:
        continue
    if vec == 'codebert' and not DO_CODEBERT:
        continue

    # Preprocessor options
    preproc_opts = [False]
    if vec in ['bert', 'codebert'] and USE_PREPROC:
        preproc_opts = [False, True]

    # Fine-tuning options
    finetune_opts = [False]
    if vec in ['bert', 'codebert'] and USE_FINETUNED:
        finetune_opts = [False, True]

    # Readability options
    readability_opts = [False, True] if USE_READABILITY else [False]

    for use_preproc in preproc_opts:
        for use_readability in readability_opts:
            for use_finetuned in finetune_opts:
                exp = {
                    'vectorizer': vec,
                    'use_python_preproc': use_preproc,
                    'use_readability': use_readability
                }
                if vec in ['bert', 'codebert']:
                    exp['model_path'] = (fine_tuned_bert_path if vec=='bert' else fine_tuned_codebert_path) if use_finetuned else None
                experiments.append(exp)

# Run experiments
for exp in experiments:
    vectorizer_type = exp['vectorizer']
    use_readability = exp['use_readability']
    use_python_preproc = exp['use_python_preproc']
    model_path = exp.get('model_path', None)

    # Initialize vectorizer instance
    if vectorizer_type == 'tfidf':
        vectorizer_instance = TfidfVectorizer(stop_words='english', preprocessor=vectorizer_text_preprocessor, min_df=0.02, max_df=0.92)
    elif vectorizer_type == 'word2vec':
        vectorizer_instance = Word2VecVectorizer(model_name="word2vec-google-news-300")
    elif vectorizer_type == 'bert':
        chosen_preproc = python_preprocessor if use_python_preproc else vectorizer_text_preprocessor
        vectorizer_instance = BERTVectorizer(model_name="bert-base-uncased", preprocessor=chosen_preproc, model_path=model_path)
    elif vectorizer_type == 'codebert':
        chosen_preproc = python_preprocessor if use_python_preproc else vectorizer_text_preprocessor
        vectorizer_instance = CodeBERTVectorizer(model_name="microsoft/codebert-base", preprocessor=chosen_preproc, model_path=model_path)
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")

    # Name experiment
    exp_name = vectorizer_type
    if vectorizer_type in ['bert', 'codebert']:
        exp_name += '_python_preproc' if use_python_preproc else '_default_preproc'
    exp_name += '_readability' if use_readability else '_no_readability'
    if model_path:
        exp_name += '_finetuned'

    print(f"[INFO] Running experiment: {exp_name}")
    print(f"[INFO] Using model path: {model_path or 'pre-trained model'}")

    # Feature engineering components
    ir_comp = IRFeaturesComponent(vectorizer_instance, concatenate_correct=True, concatenate_wrong=False)
    components = [ir_comp]
    if use_readability:
        components.append(ReadabilityFeaturesComponent())

    pipeline = FeatureEngAndRegressionPipeline(
        FeatureEngineeringModule(components),
        RegressionModule([
            SklearnRegressionComponent(
                RFRegressor(n_estimators=100, max_depth=20, random_state=SEED, n_jobs=num_jobs),
                latent_trait_range=DIFFICULTY_RANGE
            )
        ])
    )

    model = Text2PropsModel(
        KnownParametersCalibrator({DIFFICULTY: dict_difficulty}),
        FeatureEngAndRegressionEstimatorFromText({DIFFICULTY: pipeline})
    )

    model.train(df_train)
    dict_predictions = model.predict(df_test)

    predicted = {q_id: dict_predictions[DIFFICULTY][i] for i, q_id in enumerate(df_test[Q_ID].values)}
    with open(f"../data/processed/predicted_latent_traits_{suffix}_{exp_name}.pickle", 'wb') as f:
        pickle.dump(predicted, f)

    true = df_test[DIFFICULTY].values
    preds = [predicted[q_id] for q_id in df_test[Q_ID].values]

    # Compute and save metrics
    error_metrics = compute_error_metrics_latent_traits_estimation_regression(true, preds)
    metrics = {
        'error_metrics': error_metrics,
        'variance_train': np.var(df_train[DIFFICULTY].values),
        'std_dev_train': np.sqrt(np.var(df_train[DIFFICULTY].values)),
        'variance_test': np.var(true),
        'std_dev_test': np.sqrt(np.var(true)),
        'experiment': exp_name
    }
    with open(f"../data/processed/evaluation_metrics_{suffix}_{exp_name}.pickle", 'wb') as f:
        pickle.dump(metrics, f)
    print(f"[INFO] Completed and saved metrics for {exp_name}\n")

print("[INFO] All experiments completed.")
