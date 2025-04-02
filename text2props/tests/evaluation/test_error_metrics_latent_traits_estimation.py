import unittest
from text2props.evaluation.constants import MAE, MSE, MAX_ERROR, MIN_ERROR
from text2props.evaluation import compute_error_metrics_latent_traits_estimation


class ErrorMetricsLatentTraitsEstimationTestCase(unittest.TestCase):

    def test_error_measurement(self):
        y_true = [1.0, 2.0, -5.0, 4.0, 0.5]
        y_pred = [1.2, 3.0, 0.0, 4.0, -0.3]
        results = compute_error_metrics_latent_traits_estimation(y_true, y_pred)
        self.assertEqual(results[MAE], (0.2 + 1.0 + 5.0 + 0.0 + 0.8)/5.0)
        self.assertEqual(results[MSE], (0.2**2 + 1.0 + 25.0 + 0.0 + 0.8**2)/5.0)
        self.assertEqual(results[MAX_ERROR], 5.0)
        self.assertEqual(results[MIN_ERROR], 0.0)

        y_true = [1.8, 4.0, 42., 4.0, 0.5]
        y_pred = [1.2, 3.0, 0.0, 4.0, -0.3]
        results = compute_error_metrics_latent_traits_estimation(y_true, y_pred)
        self.assertEqual(results[MAE], (0.6 + 1.0 + 42 + 0.0 + 0.8)/5.0)
        self.assertEqual(results[MSE], (0.6**2 + 1.0 + 42**2 + 0.0 + 0.8**2)/5.0)
        self.assertEqual(results[MAX_ERROR], 42.0)
        self.assertEqual(results[MIN_ERROR], 0.0)
