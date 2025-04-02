import unittest
import pandas as pd
from text2props.constants import (
    WRONGNESS,
    S_ID,
    TIMESTAMP,
    CORRECT,
    Q_ID,
    Q_TEXT,
    CORRECT_TEXTS,
    WRONG_TEXTS,
)
from text2props.model.literature_architectures import MajorityWrongnessText2PropsModel


class WrongnessMajorityModelTestCase(unittest.TestCase):

    def test_model(self):

        df_gte = pd.DataFrame({
            S_ID: ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9'],
            TIMESTAMP: [None] * 10,
            CORRECT: [True, True, True, True, False, False, False, True, False, True],
            Q_ID: ['q1', 'q1', 'q1', 'q1', 'q1', 'q2', 'q2', 'q3', 'q3', 'q4']
        })
        df_train = pd.DataFrame({
            Q_ID: ['q1', 'q2', 'q3', 'q4'],
            Q_TEXT: ['test1', 'test2', 'test3', 'test4'],
            CORRECT_TEXTS: [['.'], ['.'], ['.'], ['.']],
            WRONG_TEXTS: [['.'], ['.'], ['.'], ['.']],
        })
        df_test = pd.DataFrame({
            Q_ID: ['q5', 'q6'], Q_TEXT: ['asd', 'qwe'], CORRECT_TEXTS: [['.'], ['.']], WRONG_TEXTS: [['.'], ['.']]
        })
        model = MajorityWrongnessText2PropsModel()
        model.train(df_train=df_train, df_gte=df_gte)
        predictions = model.predict(df_test)

        self.assertEqual(len(predictions.keys()), 1)
        self.assertEqual(list(predictions.keys())[0], WRONGNESS)
        self.assertEqual(len(predictions[WRONGNESS]), 2)
        self.assertEqual(predictions[WRONGNESS][0], 0.425)
        self.assertEqual(predictions[WRONGNESS][1], 0.425)
