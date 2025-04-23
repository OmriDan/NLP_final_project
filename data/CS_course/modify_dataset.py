import pandas as pd
import sys


def process_csv(input_file):
    # Read the input CSV
    df = pd.read_csv(input_file)

    # Create the first output dataframe with only required columns
    df_output1 = df[['question', 'answer', 'difficulty']]

    # Create the second output dataframe with classified difficulty
    df_output2 = df_output1.copy()

    # Convert difficulty (0-1) to classes (1-10)
    # Class 1 is easiest (0-0.1), Class 10 is hardest (0.9-1.0)
    df_output2['difficulty'] = df_output2['difficulty'].apply(lambda x: min(10, int(x * 10) + 1))

    # Write to output files
    base_name = input_file.rsplit('.', 1)[0]  # Remove extension
    df_output1.to_csv(f"{base_name}_continuous.csv", index=False)
    df_output2.to_csv(f"{base_name}_classified.csv", index=False)

    print(f"Created {base_name}_continuous.csv and {base_name}_classified.csv")


if __name__ == "__main__":
    input_file = r'DS_tests_with_difficulty.csv'
    process_csv(input_file)