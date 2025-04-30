import os
import glob
import pickle
import pandas as pd

# Folder where the evaluation_metrics pickle files are located.
data_folder = "../data/processed/"

# Pattern to find all evaluation_metrics pickle files.
pattern = os.path.join(data_folder, "evaluation_metrics_*.pickle")
metric_files = glob.glob(pattern)

# List to hold the flattened data from each pickle.
data = []

for file_path in metric_files:
    with open(file_path, "rb") as f:
        metrics = pickle.load(f)
    # Create a flat dictionary for each experiment.
    flat_record = {
        "experiment": metrics.get("experiment", ""),
        "variance_train": metrics.get("variance_train", None),
        "std_dev_train": metrics.get("std_dev_train", None),
        "variance_test": metrics.get("variance_test", None),
        "std_dev_test": metrics.get("std_dev_test", None)
    }

    # Flatten the error_metrics dictionary into individual key-value pairs.
    error_metrics = metrics.get("error_metrics", {})
    for key, value in error_metrics.items():
        flat_record[f"error_{key}"] = value

    data.append(flat_record)

# Create a DataFrame from the list of dictionaries.
df = pd.DataFrame(data)

# Define output file paths.
csv_file = os.path.join(data_folder, "combined_evaluation_metrics.csv")
txt_file = os.path.join(data_folder, "combined_evaluation_metrics.txt")

# Save the DataFrame as a CSV file.
df.to_csv(csv_file, index=False)
print(f"Combined evaluation metrics saved as CSV to {csv_file}")

# Also save as a human-readable text file.
with open(txt_file, "w") as f:
    f.write(df.to_string(index=False))
print(f"Combined evaluation metrics saved as text to {txt_file}")
