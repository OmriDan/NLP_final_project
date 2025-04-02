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
from text2props.modules.feature_engineering.utils import vectorizer_text_preprocessor as preproc
from text2props.modules.feature_engineering.components import IRFeaturesComponent, LinguisticFeaturesComponent
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator
from text2props.modules.regression import RegressionModule
from text2props.modules.regression.components import SklearnRegressionComponent

import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

print("[INFO] Start of 5_3_ir_ling.py - ablation study")

SEED = 42

# load data - TODO: in order to run this script you have to provide your own data
df_gte = pd.read_csv(os.path.join(DATA_PATH, 'a_gte.csv'))
df_train = pd.read_csv(os.path.join(DATA_PATH, 'q_train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'q_test.csv'))
df_test = df_test.drop(df_test.head(100).index)  # Not to use the validation data used in 5.1 for model selection
dict_latent_traits = pickle.load(open(os.path.join(DATA_PATH, 'known_latent_traits.p'), "rb"))

# define latent traits calibrator (known latent traits)
latent_traits_calibrator = KnownParametersCalibrator(dict_latent_traits)

for min_df in np.arange(0.00, 0.11, 0.02):
    for max_df in np.arange(0.90, 1.01, 0.02):

        file = open("outputs/5_3_ir_ling_mindf_%.2f_maxdf_%.2f.txt" % (min_df, max_df), 'w')
        file.write("MIN_DF = %.2f - MAX DF = %.2f" % (min_df, max_df))

        # pipeline difficulty
        vec_b = TfidfVectorizer(stop_words='english', preprocessor=preproc, min_df=min_df, max_df=max_df)
        pipe_b = FeatureEngAndRegressionPipeline(
            FeatureEngineeringModule([
                IRFeaturesComponent(vec_b, concatenate_correct=True, concatenate_wrong=True),
                LinguisticFeaturesComponent(),
            ]),
            RegressionModule([
                SklearnRegressionComponent(RandomForestRegressor(random_state=SEED), latent_trait_range=B_RANGE)
            ])
        )
        # pipeline discrimination
        vec_a = TfidfVectorizer(stop_words='english', preprocessor=preproc, min_df=min_df, max_df=max_df)
        pipe_a = FeatureEngAndRegressionPipeline(
            FeatureEngineeringModule([
                IRFeaturesComponent(vec_a, concatenate_correct=True, concatenate_wrong=True),
                LinguisticFeaturesComponent(),
            ]),
            RegressionModule([
                SklearnRegressionComponent(RandomForestRegressor(random_state=SEED), latent_trait_range=A_RANGE)
            ])
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
