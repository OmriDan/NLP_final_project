import parse_data
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

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

SEED = 42

df_questions = parse_data.parse_data("../../merged_leetcode_df.csv", True)

# # Optionally save dataframes to CSV:
# df_questions.to_csv("q_questions.csv", index=False)
# df_latent_traits.to_csv("q_latent_traits.csv", index=False)

# Split into training and testing sets
df_train, df_test = train_test_split(df_questions, test_size=0.2, random_state=SEED)


# 2. Build the Pipeline for Difficulty Estimation Only
# For open-ended questions, we use only the question text.
pipeline_difficulty = FeatureEngAndRegressionPipeline(
    FeatureEngineeringModule([
        IRFeaturesComponent(
            TfidfVectorizer(stop_words='english',
                            preprocessor=vectorizer_text_preprocessor,
                            min_df=0.02,
                            max_df=0.92),
            concatenate_correct=True,  # Include answer options, change this to False later and see what happens
            concatenate_wrong=False
        ),
        LinguisticFeaturesComponent(),
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
    KnownParametersCalibrator(dict_difficulty),
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
with open("predicted_difficulty.pickle", "wb") as f:
    pickle.dump(predicted_difficulty, f)

print("[INFO] Predicted difficulty ratings saved.")
