import pyximport
pyximport.install(language_level=3)
import parse_data
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Import text2props modules
from text2props.constants import DATA_PATH, DIFFICULTY, DIFFICULTY_RANGE, Q_ID
from text2props.model import Text2PropsModel
from text2props.modules.estimators_from_text import FeatureEngAndRegressionPipeline, FeatureEngAndRegressionEstimatorFromText
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.feature_engineering.utils import vectorizer_text_preprocessor
from text2props.modules.feature_engineering.components import IRFeaturesComponent, LinguisticFeaturesComponent, ReadabilityFeaturesComponent
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator
from text2props.modules.regression import RegressionModule
from text2props.modules.regression.components import SklearnRegressionComponent
from text2props.evaluation.latent_traits_estimation import compute_error_metrics_latent_traits_estimation_regression
from text2props.modules.feature_engineering.components import IRFeaturesComponent
from custom_vectorizers import Word2VecVectorizer, BERTVectorizer


SEED = 42
database_name = "merged_leetcode_df"
suffix = "leetcode"
df_questions = parse_data.parse_data(f"../data/raw/{database_name}.csv", True, True, f"../data/processed/known_latent_traits_{suffix}.pickle")

# Split into training and testing sets
df_train, df_test = train_test_split(df_questions, test_size=0.2, random_state=SEED)


# # 2. Build the Pipeline for Difficulty Estimation Only - Vectorizer = TF-IDF
# print("[INFO] Vectorizer = TF-IDF")
# pipeline_difficulty = FeatureEngAndRegressionPipeline(
#     FeatureEngineeringModule([
#         IRFeaturesComponent(
#             TfidfVectorizer(stop_words='english',
#                             preprocessor=vectorizer_text_preprocessor,
#                             min_df=0.02,
#                             max_df=0.92),
#             concatenate_correct=True,  # Include answer options, change this to False later and see what happens
#             concatenate_wrong=False
#         ),
#         #LinguisticFeaturesComponent(),
#         ReadabilityFeaturesComponent(),
#     ]),
#     RegressionModule([
#         SklearnRegressionComponent(
#             RFRegressor(n_estimators=100, max_depth=20, random_state=SEED),
#             latent_trait_range=DIFFICULTY_RANGE
#         )
#     ])
# )
# # 2. Build the Pipeline for Difficulty Estimation Only - Vectorizer = word2vec
# word2vec_vectorizer = Word2VecVectorizer(model_path="path_to_word2vec_model.bin")
# print("[INFO] Vectorizer = Word2Vec")
# pipeline_difficulty = FeatureEngAndRegressionPipeline(
#     FeatureEngineeringModule([
#         IRFeaturesComponent(
#             word2vec_vectorizer,
#             concatenate_correct=True,  # Include answer options, change this to False later and see what happens
#             concatenate_wrong=False
#         ),
#         #LinguisticFeaturesComponent(),
#         ReadabilityFeaturesComponent(),
#     ]),
#     RegressionModule([
#         SklearnRegressionComponent(
#             RFRegressor(n_estimators=100, max_depth=20, random_state=SEED),
#             latent_trait_range=DIFFICULTY_RANGE
#         )
#     ])
# )
# 2. Build the Pipeline for Difficulty Estimation Only - Vectorizer = BERT
print("[INFO] Vectorizer = BERT")
bert_vectorizer = BERTVectorizer(model_name="distilbert-base-uncased")
pipeline_difficulty = FeatureEngAndRegressionPipeline(
    FeatureEngineeringModule([
        IRFeaturesComponent(
            bert_vectorizer,
            concatenate_correct=True,  # Include answer options, change this to False later and see what happens
            concatenate_wrong=False
        ),
        #LinguisticFeaturesComponent(),
        ReadabilityFeaturesComponent(),
    ]),
    RegressionModule([
        SklearnRegressionComponent(
            RFRegressor(n_estimators=100, max_depth=20, random_state=SEED),
            latent_trait_range=DIFFICULTY_RANGE
        )
    ])
)


# 3. Create a Calibrator Using Lecturer-Provided Difficulty Ratings
# Here we create a dictionary mapping question IDs to their known difficulty ratings from the training set.
dict_difficulty = df_train.set_index(Q_ID)[DIFFICULTY].to_dict()

# 4. Create and Train the Model
model = Text2PropsModel(
    KnownParametersCalibrator({DIFFICULTY: dict_difficulty}),
    FeatureEngAndRegressionEstimatorFromText({DIFFICULTY: pipeline_difficulty})
)

model.train(df_train)
print("[INFO] Model trained on open-ended questions.")

# 5. Predict on the Test Set and Save the Predictions
dict_predictions = model.predict(df_test)

# Convert predictions to a dictionary mapping q_id to the predicted difficulty
predicted_difficulty = {}
for idx, q_id in enumerate(df_test[Q_ID].values):
    predicted_difficulty[q_id] = dict_predictions[DIFFICULTY][idx]

# Save the predictions to a pickle file
with open(f"../data/processed/predicted_latent_traits_{suffix}.pickle", "wb") as f:
    pickle.dump(predicted_difficulty, f)

print("[INFO] Predicted difficulty ratings saved.")

# Load true and predicted latent traits
dict_true_latent_traits = pickle.load(open(f'../data/processed/predicted_latent_traits_{suffix}.pickle', "rb"))
dict_predicted_latent_traits = pickle.load(open(f'../data/processed/known_latent_traits_{suffix}.pickle', 'rb'))


# Extract true and predicted difficulties
#true_difficulties = [dict_true_latent_traits[DIFFICULTY][q_id] for q_id in df_test[Q_ID].values]
true_difficulties = [dict_true_latent_traits[q_id] for q_id in df_test[Q_ID].values]
predicted_difficulties = [dict_predicted_latent_traits[DIFFICULTY][q_id] for q_id in df_test[Q_ID].values]

# Extract true difficulties from the training and test sets
true_difficulties_train = df_train[DIFFICULTY].values
true_difficulties_test = df_test[DIFFICULTY].values

# Calculate variance and standard deviation for training set
variance_train = np.var(true_difficulties_train)
std_dev_train = np.sqrt(variance_train)

# Calculate variance and standard deviation for test set
variance_test = np.var(true_difficulties_test)
std_dev_test = np.sqrt(variance_test)

# Print the results
print(f"Training Set - Variance: {variance_train:.4f}, Standard Deviation: {std_dev_train:.4f}")
print(f"Test Set - Variance: {variance_test:.4f}, Standard Deviation: {std_dev_test:.4f}")

# Evaluate the predicted difficulties
metrics = compute_error_metrics_latent_traits_estimation_regression(true_difficulties, predicted_difficulties)
print("[INFO] Evaluation of Predicted Difficulties:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")
