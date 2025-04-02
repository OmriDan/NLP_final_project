from text2props.constants import DATA_PATH
from text2props.model.literature_architectures import MajorityWrongnessText2PropsModel
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator

import os
import pandas as pd
import pickle

print("[INFO] Start of 5_3_majority.py - ablation study")

# load data - TODO: in order to run this script you have to provide your own data
df_gte = pd.read_csv(os.path.join(DATA_PATH, 'a_gte.csv'))
df_train = pd.read_csv(os.path.join(DATA_PATH, 'q_train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'q_test.csv'))
df_test = df_test.drop(df_test.head(100).index)  # Not to use the validation data used in 5.1 for model selection
dict_latent_traits = pickle.load(open(os.path.join(DATA_PATH, 'known_latent_traits.p'), "rb"))

# define latent traits calibrator (known latent traits)
latent_traits_calibrator = KnownParametersCalibrator(dict_latent_traits)

file = open("outputs/5_3_majority.txt", 'w')

# define and train the text2props_model
model = MajorityWrongnessText2PropsModel(known_latent_traits=dict_latent_traits)
model.train(df_train=df_train, df_gte=df_gte)

# write to file
file.write("\nERROR METRICS ON TEST:\n" + str(model.compute_error_metrics_latent_traits_estimation(df_test)))
file.write("\nERROR METRICS ON TRAINING:\n"+str(model.compute_error_metrics_latent_traits_estimation(df_train)))
file.close()
