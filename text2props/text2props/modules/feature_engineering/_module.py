from typing import List

from scipy.sparse import hstack, coo_matrix
from .components import BaseFeatEngComponent
import pandas as pd


class FeatureEngineeringModule(object):

    def __init__(self, components: List[BaseFeatEngComponent] = None, normalize_method = None):
        if components is None:
            self.components = []
        else:
            self.components = components
        self.normalize_method = normalize_method

    def add_component(self, component: BaseFeatEngComponent):
        """
        Add a component to a existing feature engineering module.
        :param component:
        :return:
        """
        self.components.append(component)

    def add_list_components(self, list_components: List[BaseFeatEngComponent]):
        """
        Add components to a existing feature engineering module.
        :param list_components:
        :return:
        """
        for component in list_components:
            self.components.append(component)

    def fit_transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        Call the fit_transform method on all the components of the module and stack the results.
        :param input_df:
        :return:
        """
        partial_results = []
        for component in self.components:
            partial_results.append(component.fit_transform(input_df))
        partial_results = hstack(partial_results)
        if self.normalize_method is not None:
            partial_results = self.normalize_method(partial_results)
        return partial_results

    def transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        Call the transform method on all the components of the module and stack the results.
        :param input_df:
        :return:
        """
        partial_results = []
        for component in self.components:
            partial_results.append(component.transform(input_df))
        partial_results = hstack(partial_results)
        if self.normalize_method is not None:
            partial_results = self.normalize_method(partial_results)
        return partial_results
