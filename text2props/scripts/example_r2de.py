from text2props.model.literature_architectures import R2DEText2PropsModel
from text2props.constants import DATA_PATH
import os
import pandas as pd

# TODO IMPORTANT: in order to run this script you have to provide your own data
A_GTE = 'a_gte.csv'
Q_TRAIN = 'q_train.csv'
Q_TEST = 'q_test.csv'

# load data
df_gte = pd.read_csv(os.path.join(DATA_PATH, A_GTE))
df_train = pd.read_csv(os.path.join(DATA_PATH, Q_TRAIN))
df_test = pd.read_csv(os.path.join(DATA_PATH, Q_TEST))

# define and train the text2props_model
text2props_model = R2DEText2PropsModel()
text2props_model.train(df_train=df_train, df_gte=df_gte)

# perform predictions
predictions = text2props_model.predict(df_test)

# evaluate model and print results
results = text2props_model.compute_error_metrics_latent_traits_estimation(df_test)
print(results)
