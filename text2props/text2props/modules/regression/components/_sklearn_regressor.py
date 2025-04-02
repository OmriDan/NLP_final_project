from typing import Tuple, List, Iterable, Dict

from scipy.sparse import coo_matrix

from . import BaseRegressionComponent
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


class SklearnRegressionComponent(BaseRegressionComponent):

    def __init__(self, regressor, latent_trait_range: Tuple[float, float]):
        self.regressor = regressor
        self.min_latent_trait, self.max_latent_trait = latent_trait_range

    def train(self, x: coo_matrix, y: List[float]):
        self.regressor.fit(x, y)

    def predict(self, x: coo_matrix) -> Iterable[float]:
        predictions = self.regressor.predict(x)
        for idx, x in enumerate(predictions):
            predictions[idx] = min(x, self.max_latent_trait)
            predictions[idx] = max(x, self.min_latent_trait)
        return predictions

    def randomized_cv_train(
            self,
            x: coo_matrix,
            y: List[float],
            param_distributions: Dict[str, List[float]] = None,
            n_iter: int = None,
            cv: int = None,
            n_jobs: int = None,
            random_state: int = None
    ) -> float:
        """
        Performs the training of the regressor with RandomizedSearchCV, and returns the score obtained with the best
        estimator.
        :param x:
        :param y:
        :param param_distributions:
        :param n_iter:
        :param cv:
        :param n_jobs:
        :param random_state:
        :return:
        """
        random_search = RandomizedSearchCV(
            Pipeline(steps=[('regressor', self.regressor)], verbose=False),
            param_distributions, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state
        )
        random_search = random_search.fit(x, y)
        self.regressor = random_search.best_estimator_['regressor']
        return random_search.best_score_
