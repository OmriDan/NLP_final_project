from text2props.constants import DATA_PATH, DIFFICULTY, DISCRIMINATION, DIFFICULTY_RANGE, DISCRIMINATION_RANGE, Q_ID
from text2props.model import Text2PropsModel
from text2props.modules.estimators_from_text import (
    FeatureEngAndRegressionPipeline, FeatureEngAndRegressionEstimatorFromText)
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.feature_engineering.utils import vectorizer_text_preprocessor
from text2props.modules.feature_engineering.components import (
    IRFeaturesComponent, LinguisticFeaturesComponent, ReadabilityFeaturesComponent)
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator
from text2props.modules.regression import RegressionModule
from text2props.modules.regression.components import SklearnRegressionComponent

import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42

# load data - TODO: in order to run this script you have to provide your own data
df_gte = pd.read_csv(os.path.join(DATA_PATH, 'a_gte.csv'))
df_sap = pd.read_csv(os.path.join(DATA_PATH, 'a_sap.csv'))
df_train = pd.read_csv(os.path.join(DATA_PATH, 'q_train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'q_test.csv'))
dict_latent_traits = pickle.load(open(os.path.join(DATA_PATH, 'known_latent_traits.p'), "rb"))

# This first section is to train the model - I assume that the parameters used here are the ones of the best performing
#   model, as obtained in the scripts 5_1_*
pipeline_difficulty = FeatureEngAndRegressionPipeline(
    FeatureEngineeringModule([
        IRFeaturesComponent(
            TfidfVectorizer(stop_words='english', preprocessor=vectorizer_text_preprocessor, min_df=0.02, max_df=0.92),
            concatenate_correct=True,
            concatenate_wrong=True
        ),
        LinguisticFeaturesComponent(),
        ReadabilityFeaturesComponent(),
    ]),
    RegressionModule([
        SklearnRegressionComponent(
            RFRegressor(n_estimators=100, max_depth=20, random_state=SEED), latent_trait_range=DIFFICULTY_RANGE
        )
    ])
)
pipeline_discrimination = FeatureEngAndRegressionPipeline(
    FeatureEngineeringModule([
        IRFeaturesComponent(
            TfidfVectorizer(stop_words='english', preprocessor=vectorizer_text_preprocessor, min_df=0.02, max_df=0.96),
            concatenate_correct=True,
            concatenate_wrong=True
        ),
        LinguisticFeaturesComponent(),
        ReadabilityFeaturesComponent(),
    ]),
    RegressionModule([
        SklearnRegressionComponent(
            RFRegressor(n_estimators=100, max_depth=20, random_state=SEED), latent_trait_range=DISCRIMINATION_RANGE
        )
    ])
)
model = Text2PropsModel(
    KnownParametersCalibrator(dict_latent_traits),
    FeatureEngAndRegressionEstimatorFromText({DIFFICULTY: pipeline_difficulty, DISCRIMINATION: pipeline_discrimination})
)
model.train(df_train)
print('[INFO] model trained')


# Here I estimate the latent traits for the test set
dict_predictions_test_set = model.predict(df_test)
# I have to convert the dictionary of the prediction in the right format as model.predict returns a dict of lists
#   (one list for each latent trait)
dict_predicted_latent_traits = dict()
dict_predicted_latent_traits[DIFFICULTY], dict_predicted_latent_traits[DISCRIMINATION] = dict(), dict()
for idx, q_id in enumerate(df_test[Q_ID].values):
    dict_predicted_latent_traits[DIFFICULTY][q_id] = dict_predictions_test_set[DIFFICULTY][idx]
    dict_predicted_latent_traits[DISCRIMINATION][q_id] = dict_predictions_test_set[DISCRIMINATION][idx]
pickle.dump(dict_predicted_latent_traits, open(os.path.join(DATA_PATH, 'predicted_latent_traits.p'), 'wb'))
