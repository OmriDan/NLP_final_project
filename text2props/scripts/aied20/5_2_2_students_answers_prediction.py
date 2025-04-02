from text2props.evaluation.students_answers_prediction import evaluate_students_answers_prediction
from text2props.constants import DATA_PATH
import pandas as pd
import os
import pickle

# TODO IMPORTANT: in order to run this script you have to provide your own data
df_sap = pd.read_csv(os.path.join(DATA_PATH, 'a_sap.csv'))
df_train = pd.read_csv(os.path.join(DATA_PATH, 'q_train.csv'))

dict_true_latent_traits = pickle.load(open(os.path.join(DATA_PATH, 'known_latent_traits.p'), "rb"))
dict_predicted_latent_traits = pickle.load(open(os.path.join(DATA_PATH, 'predicted_latent_traits.p'), 'rb'))
evaluate_students_answers_prediction(df_sap, df_train, dict_true_latent_traits, dict_predicted_latent_traits)
