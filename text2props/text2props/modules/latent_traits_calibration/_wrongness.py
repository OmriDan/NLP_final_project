from typing import Dict

import pandas as pd
from text2props.modules.latent_traits_calibration import BaseLatentTraitsCalibrator
from text2props.constants import WRONGNESS, Q_ID, CORRECT
from text2props.data_validation import check_answers_df_columns


class WrongnessCalibrator(BaseLatentTraitsCalibrator):

    def __init__(self):
        self.n_latent_traits = 1
        self.name_latent_traits = [WRONGNESS]
        self.estimated_latent_traits = dict()

    def calibrate_latent_traits(self, df_gte: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        check_answers_df_columns(df_gte)
        df_count_total = df_gte.groupby(Q_ID).size().reset_index()
        dict_count_total = {q_id: count for q_id, count in df_count_total[[Q_ID, 0]].values}
        df_count_wrong = df_gte[~df_gte[CORRECT]].groupby(Q_ID).size().reset_index()
        dict_count_wrong = {q_id: count for q_id, count in df_count_wrong[[Q_ID, 0]].values}
        dict_wrongness = dict()
        for q_id in dict_count_total.keys():
            if q_id not in dict_count_wrong.keys():
                dict_wrongness[q_id] = 0
            else:
                dict_wrongness[q_id] = dict_count_wrong[q_id] / dict_count_total[q_id]
        self.estimated_latent_traits[WRONGNESS] = dict_wrongness.copy()
        return self.estimated_latent_traits
