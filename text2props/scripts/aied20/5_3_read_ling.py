from text2props.constants import (
    DATA_PATH,
    DIFFICULTY,
    DISCRIMINATION,
    DIFFICULTY_RANGE as B_RANGE,
    DISCRIMINATION_RANGE as A_RANGE
)
from text2props.model import Text2PropsModel
from text2props.modules.estimators_from_text import (
    FeatureEngAndRegressionPipeline,
    FeatureEngAndRegressionEstimatorFromText,
)
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.feature_engineering.components import ReadabilityFeaturesComponent, LinguisticFeaturesComponent
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator
from text2props.modules.regression import RegressionModule
from text2props.modules.regression.components import SklearnRegressionComponent

import os
import pandas as pd
import pickle
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor

print("[INFO] Start of 5_3_read_ling.py - ablation study")

SEED = 42

# load data - TODO: in order to run this script you have to provide your own data
df_gte = pd.read_csv(os.path.join(DATA_PATH, 'a_gte.csv'))
df_train = pd.read_csv(os.path.join(DATA_PATH, 'q_train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'q_test.csv'))
df_test = df_test.drop(df_test.head(100).index)  # Not to use the validation data used in 5.1 for model selection
dict_latent_traits = pickle.load(open(os.path.join(DATA_PATH, 'known_latent_traits.p'), "rb"))

# define latent traits calibrator (known latent traits)
latent_traits_calibrator = KnownParametersCalibrator(dict_latent_traits)

file = open("outputs/5_3_read_ling.txt", 'w')

# pipeline difficulty
pipe_b = FeatureEngAndRegressionPipeline(
    FeatureEngineeringModule([ReadabilityFeaturesComponent(), LinguisticFeaturesComponent()]),
    RegressionModule([SklearnRegressionComponent(RandomForestRegressor(random_state=SEED), latent_trait_range=B_RANGE)])
)
# pipeline discrimination
pipe_a = FeatureEngAndRegressionPipeline(
    FeatureEngineeringModule([ReadabilityFeaturesComponent(), LinguisticFeaturesComponent()]),
    RegressionModule([SklearnRegressionComponent(RandomForestRegressor(random_state=SEED), latent_trait_range=A_RANGE)])
)
# create estimator from text form the previous pipelines
estimator_from_text = FeatureEngAndRegressionEstimatorFromText({DIFFICULTY: pipe_b, DISCRIMINATION: pipe_a})
model = Text2PropsModel(latent_traits_calibrator, estimator_from_text)
model.calibrate_latent_traits(None)

# define parameters for randomized CV
dict_params = {
    DIFFICULTY: [{'regressor__n_estimators': randint(20, 200), 'regressor__max_depth': randint(2, 50)}],
    DISCRIMINATION: [{'regressor__n_estimators': randint(20, 200), 'regressor__max_depth': randint(2, 50)}],
}
scores = model.randomized_cv_train(
    param_distributions=dict_params,
    n_iter=10,
    cv=5,
    n_jobs=-1,
    random_state=SEED,
    df_train=df_train
)

# write to file
file.write("CV scores =\n" + str(scores))
file.write("\nERROR METRICS ON TEST:\n" + str(model.compute_error_metrics_latent_traits_estimation(df_test)))
file.write("\nERROR METRICS ON TRAINING:\n"+str(model.compute_error_metrics_latent_traits_estimation(df_train)))
file.close()
