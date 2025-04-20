from typing import List, Dict, Iterable

import numpy as np
from scipy.sparse import coo_matrix

from .components import BaseRegressionComponent

from tqdm import tqdm


class RegressionModule(object):

    def __init__(self, components: List[BaseRegressionComponent] = None):
        """
        Initialize RegressionModule object. If available components are initialized, otherwise creates sn empty list
        :param components: the list of RegressionComponent objects that the module is made of (possibly None).
        """
        if components is None:
            self.components = []
        else:
            self.components = components

    def add_component(self, component: BaseRegressionComponent):
        """
        Add a RegressionComponent to the RegressionModule object
        :param component:
        :return:
        """
        self.components.append(component)

    def add_list_components(self, list_components: List[BaseRegressionComponent]):
        """
        Add a list of RegressionComponent objects to the RegressionModule
        :param list_components:
        :return:
        """
        for component in list_components:
            self.components.append(component)

    def train(self, x: coo_matrix, y: List[float]):
        """
        Train the RegressionModule, by training all the RegressionComponent objects
        :param x: sparse matrix (obtained from the FeatureEngineering module) containing the features obtained from text
        :param y: iterable containing target values
        :return:
        """
        for component in tqdm(self.components, desc="Regression: Training components"):
            component.train(x, y)

    def predict(self, x: coo_matrix) -> List[float]:
        """
        Performs the prediction for the sparse matrix given as input. The matrix is obtained from the FeatureEngineering
          module and contains the features obtained transforming the input text. The final prediction for each input
          sample is obtained by averaging the prediction of all the components of the RegressionModule.
        :param x:
        :return:
        """
        partial_results = []
        for component in tqdm(self.components, desc="Regression: Predicting with components"):
            partial_results.append(component.predict(x))
        return np.mean(partial_results, axis=0)

    def randomized_cv_train(
            self,
            x_train: coo_matrix,
            y_train: List[float],
            param_distributions: List[Dict[str, List[float]]],
            n_iter: int,
            cv: int,
            n_jobs: int,
            random_state: int
    ) -> float:
        """
        Performs the training with randomized cross validation of all the components in the RegressionModule. For each
        RegressionComponent, computes the R2 score of the best performing configuration. In the end returns the R2 score
        obtained as the average of the R2 scores of all the components.
        :param x_train:
        :param y_train:
        :param param_distributions:
        :param n_iter:
        :param cv:
        :param n_jobs:
        :param random_state:
        :return:
        """
        scores = []
        for idx, component in enumerate(self.components):
            component_score = component.randomized_cv_train(
                x_train,
                y_train,
                param_distributions=param_distributions[idx],
                n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state)
            scores.append(component_score)
        return np.float(np.mean(scores))
