from abc import ABCMeta, abstractmethod
from typing import List, Iterable

from scipy.sparse import coo_matrix


class BaseRegressionComponent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, x: coo_matrix, y: List[float]):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: coo_matrix) -> Iterable[float]:
        raise NotImplementedError

    @abstractmethod
    def randomized_cv_train(self, x: coo_matrix, y: List[float], **params) -> float:
        raise NotImplementedError
