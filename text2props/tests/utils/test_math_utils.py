import unittest
from text2props.utils.math import (
    item_response_function as irf,
)


class MathUtilsTestCase(unittest.TestCase):

    def test_item_response_function(self):
        self.assertEqual(irf(difficulty=0.0, skill=0.0, discrimination=1.0, guess=0.0, slip=0.0), 0.5)
        self.assertEqual(irf(difficulty=2.0, skill=2.0, discrimination=1.0, guess=0.0, slip=0.0), 0.5)
        self.assertEqual(irf(difficulty=-3.0, skill=-3.0, discrimination=1.0, guess=0.0, slip=0.0), 0.5)
        self.assertEqual(irf(difficulty=-2.0, skill=2.0, discrimination=0.0, guess=0.0, slip=0.0), 0.5)
        self.assertAlmostEqual(irf(difficulty=50.0, skill=0.0, discrimination=1.0, guess=0.0, slip=0.0), 0.0, places=4)
        self.assertAlmostEqual(irf(difficulty=0.0, skill=50.0, discrimination=1.0, guess=0.0, slip=0.0), 1.0, places=4)
