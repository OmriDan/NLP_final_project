import unittest
import pandas as pd
from text2props.modules.latent_traits_calibration import WrongnessCalibrator
from text2props.constants import (
    WRONGNESS,
    S_ID,
    TIMESTAMP,
    CORRECT,
    Q_ID,
)


class WrongnessCalibratorTestCase(unittest.TestCase):

    def test_calibration(self):
        estimator = WrongnessCalibrator()
        self.assertEqual(estimator.get_n_latent_traits(), 1)
        self.assertEqual(estimator.get_name_latent_traits()[0], WRONGNESS)
        answers_df = pd.DataFrame({
            S_ID: ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9'],
            TIMESTAMP: [None] * 10,
            CORRECT: [True, True, True, True, False, False, False, True, False, True],
            Q_ID: ['q1', 'q1', 'q1', 'q1', 'q1', 'q2', 'q2', 'q3', 'q3', 'q4'],
        })
        estimated_latent_traits = estimator.calibrate_latent_traits(answers_df)
        self.assertEqual(estimated_latent_traits[WRONGNESS]['q1'], 0.2)
        self.assertEqual(estimated_latent_traits[WRONGNESS]['q2'], 1.0)
        self.assertEqual(estimated_latent_traits[WRONGNESS]['q3'], 0.5)
        self.assertEqual(estimated_latent_traits[WRONGNESS]['q4'], 0.0)

        estimated_latent_traits = estimator.get_calibrated_latent_traits()
        self.assertEqual(estimated_latent_traits[WRONGNESS]['q1'], 0.2)
        self.assertEqual(estimated_latent_traits[WRONGNESS]['q2'], 1.0)
        self.assertEqual(estimated_latent_traits[WRONGNESS]['q3'], 0.5)
        self.assertEqual(estimated_latent_traits[WRONGNESS]['q4'], 0.0)
