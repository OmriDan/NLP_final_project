import pandas as pd


def process_csv(input_file):
    # Read the input CSV
    df = pd.read_csv(input_file)

    # Create the first output dataframe with only required columns
    df_output1 = df[['question', 'answer', 'difficulty']]

    # Create the second output dataframe with classified difficulty (1-10)
    df_output2 = df_output1.copy()
    df_output2['difficulty'] = df_output2['difficulty'].apply(lambda x: min(10, int(x * 10) + 1))

    # Create the third output dataframe with categorical difficulty (easy, medium, hard)
    df_output3 = df_output1.copy()

    # Map difficulty values to categories
    def categorize_difficulty(value):
        if 0 <= value < 0.33:
            return 'easy'
        elif 0.33 <= value < 0.67:
            return 'medium'
        else:  # 0.75 <= value <= 1
            return 'hard'

    df_output3['difficulty'] = df_output3['difficulty'].apply(categorize_difficulty)

    # Write to output files
    base_name = input_file.rsplit('.', 1)[0]  # Remove extension
    df_output1.to_csv(f"{base_name}_continuous.csv", index=False)
    df_output2.to_csv(f"{base_name}_classified.csv", index=False)
    df_output3.to_csv(f"{base_name}_categorical.csv", index=False)

    print(f"Created {base_name}_continuous.csv, {base_name}_classified.csv, and {base_name}_categorical.csv")


if __name__ == "__main__":
    input_file = r'DS_tests_with_difficulty.csv'
    process_csv(input_file)