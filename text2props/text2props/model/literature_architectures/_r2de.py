from typing import Dict

from .._model import Text2PropsModel
from text2props.modules.latent_traits_calibration import IRTCalibrator
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator
from text2props.modules.estimators_from_text import (
    FeatureEngAndRegressionEstimatorFromText,
    FeatureEngAndRegressionPipeline,
)
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.feature_engineering.components import IRFeaturesComponent
from text2props.constants import (
    DIFFICULTY_RANGE,
    DISCRIMINATION_RANGE,
    DIFFICULTY,
    DISCRIMINATION,
)
from text2props.modules.feature_engineering.utils import vectorizer_text_preprocessor
from text2props.modules.regression import RegressionModule
from text2props.modules.regression.components import SklearnRegressionComponent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor


class R2DEText2PropsModel(Text2PropsModel):
    """
    This class implements the R2DE model, proposed in "R2DE: a NLP approach to estimating IRT parameters of newly
    generated questions" by L. Benedetto, A. Cappelli, R. Turrin and P. Cremonesi (https://arxiv.org/pdf/2001.07569.pdf)
    The paper was presented at the 10th conference on Learning Analytics and Knowledge (LAK20).

    The model is made of:
    - IRTCalibrator that performs the estimation of difficulty and discrimination
    - Estimator from Text that made of two FeatEng (both IRFeatures) + Regression (both Sklearn RandomForests) pipelines
    """
    def __init__(self, random_state: int = None, known_latent_traits: Dict[str, Dict[str, float]] = None):
        if known_latent_traits is not None:
            latent_traits_calibrator = KnownParametersCalibrator(latent_traits=known_latent_traits)
            if set(known_latent_traits.keys()) != {DIFFICULTY, DISCRIMINATION}:
                raise ValueError("wrong keys in known_latent_traits dictionary")
        else:
            latent_traits_calibrator = IRTCalibrator(DIFFICULTY_RANGE, DISCRIMINATION_RANGE)
        vec_diff = TfidfVectorizer(stop_words='english', preprocessor=vectorizer_text_preprocessor, max_features=1000)
        feat_eng_regression_pipeline_difficulty = FeatureEngAndRegressionPipeline(
            FeatureEngineeringModule([IRFeaturesComponent(vec_diff, concatenate_correct=True, concatenate_wrong=True)]),
            RegressionModule([
                SklearnRegressionComponent(
                    RandomForestRegressor(n_estimators=250, max_depth=50, random_state=random_state),
                    latent_trait_range=DIFFICULTY_RANGE
                )
            ])
        )
        vec_disc = TfidfVectorizer(stop_words='english', preprocessor=vectorizer_text_preprocessor, max_features=800)
        feat_eng_regression_pipeline_discrimination = FeatureEngAndRegressionPipeline(
            FeatureEngineeringModule([IRFeaturesComponent(vec_disc, concatenate_correct=True, concatenate_wrong=True)]),
            RegressionModule([
                SklearnRegressionComponent(
                    RandomForestRegressor(n_estimators=100, max_depth=25, random_state=random_state),
                    latent_trait_range=DISCRIMINATION_RANGE
                )
            ])
        )
        estimator_from_text = FeatureEngAndRegressionEstimatorFromText(
            {
                DIFFICULTY: feat_eng_regression_pipeline_difficulty,
                DISCRIMINATION: feat_eng_regression_pipeline_discrimination
            }
        )
        super().__init__(latent_traits_calibrator, estimator_from_text)
