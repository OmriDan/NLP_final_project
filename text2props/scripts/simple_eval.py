import pickle
import os
import numpy as np
from sklearn.metrics import mean_squared_error

# Set your data path appropriately
DATA_PATH = "path_to_your_data_directory"

# Load the dictionaries of known and predicted latent traits.
dict_true = pickle.load(open(os.path.join(DATA_PATH, 'known_latent_traits.p'), "rb"))
dict_pred = pickle.load(open(os.path.join(DATA_PATH, 'predicted_latent_traits.p'), "rb"))

# Assuming the difficulty latent trait is stored under a key like "difficulty"
true_values = []
pred_values = []
for q_id, true_val in dict_true["difficulty"].items():
    # Only compare if the question ID exists in the predictions
    if q_id in dict_pred["difficulty"]:
        true_values.append(true_val)
        pred_values.append(dict_pred["difficulty"][q_id])

rmse = np.sqrt(mean_squared_error(true_values, pred_values))
print("RMSE for difficulty predictions:", rmse)