from typing import Dict, List

from text2props.constants import DATA_PATH
import pandas as pd
from abc import abstractmethod
import os
import pickle


class BaseLatentTraitsCalibrator(object):

    @abstractmethod
    def __init__(self, **params):
        pass

    def get_n_latent_traits(self) -> int:
        return self.n_latent_traits

    def get_name_latent_traits(self) -> List[str]:
        return self.name_latent_traits

    def get_calibrated_latent_traits(self) -> Dict[str, Dict[str, float]]:
        return self.estimated_latent_traits

    @abstractmethod
    def calibrate_latent_traits(self, df_gte: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError

    def store_calibrated_latent_traits(self, output_data_path: str = DATA_PATH, output_filename: str = 'lt.p'):
        pickle.dump(self.estimated_latent_traits, open(os.path.join(output_data_path, output_filename), "wb"))
