from scipy.sparse import coo_matrix
from abc import ABCMeta
from abc import abstractmethod


class BaseFeatEngComponent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **params):
        raise NotImplementedError

    @abstractmethod
    def transform(self, input_df) -> coo_matrix:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, input_df) -> coo_matrix:
        raise NotImplementedError
