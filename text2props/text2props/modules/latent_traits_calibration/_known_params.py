from typing import Dict, List

from . import BaseLatentTraitsCalibrator


class KnownParametersCalibrator(BaseLatentTraitsCalibrator):

    def __init__(self, latent_traits: Dict[str, Dict[str, float]]):
        """
        Initializes a KnownParametersCalibrator object. This is a degenerate estimator object, which does not perform an
        estimation but receives already-estimated latent traits.
        :param latent_traits: a dictionary whose keys are the name of the estimated latent traits and whose values are
          dictionaries containing the actual latent traits in the form <Q_ID: latent trait value>.
          e.g.: If I have one latent trait named wrongness: latent_traits = {'wrongness': {Q_0: 0.5, Q_1: 0.2, ...}}
          e.g.: If I have two latent traits, named difficulty and discrimination, I have to pass:
            {'DIFFICULTY': {Q_0: 0.8, Q_1: -0.1, ...}, 'DISCRIMINATION': {Q_0: 0.9, Q_1: 1.0, ...}}
        """
        self.n_latent_traits = len(latent_traits.keys())
        self.name_latent_traits = list(latent_traits.keys())
        self.estimated_latent_traits = latent_traits.copy()

    def get_n_latent_traits(self) -> int:
        return self.n_latent_traits

    def get_name_latent_traits(self) -> List[str]:
        return self.name_latent_traits

    def get_calibrated_latent_traits(self) -> Dict[str, Dict[str, float]]:
        return self.estimated_latent_traits

    def calibrate_latent_traits(self, df_gte) -> Dict[str, Dict[str, float]]:
        return self.get_calibrated_latent_traits()
