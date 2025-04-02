from text2props.model.literature_architectures import MajorityWrongnessText2PropsModel
from text2props.constants import DATA_PATH
import os
import pandas as pd
import pickle

# TODO IMPORTANT: in order to run this script you have to provide your own data
# load data
df_gte = pd.read_csv(os.path.join(DATA_PATH, 'a_gte.csv'))
df_train = pd.read_csv(os.path.join(DATA_PATH, 'q_train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'q_test.csv'))
df_test = df_test.drop(df_test.head(100).index)  # Not to use the validation data used in 5.1 for model selection
dict_latent_traits = pickle.load(open(os.path.join(DATA_PATH, 'known_latent_traits.p'), "rb"))

# define and train the text2props_model
text2props_model = MajorityWrongnessText2PropsModel(known_latent_traits=dict_latent_traits)
text2props_model.train(df_train=df_train, df_gte=df_gte)

# perform predictions
predictions = text2props_model.predict(df_test)

# evaluate model and print results
results = text2props_model.compute_error_metrics_latent_traits_estimation(df_test)
print(results)
