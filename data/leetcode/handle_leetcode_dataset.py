import pandas as pd
import kagglehub


def merge_leetcode_dataframes(df_with_acceptance_rate, df_with_py_answers):
    """
    Merge LeetCode DataFrames containing questions and Python solutions.

    Args:
        df_with_acceptance_rate: DataFrame with acceptance rate info (from CSV)
        df_with_py_answers: DataFrame with Python solutions (from JSONL)

    Returns:
        DataFrame with merged data containing specified columns from both sources
    """
    # Create normalized title fields for better matching
    df_accept = df_with_acceptance_rate.copy()
    df_py = df_with_py_answers.copy()

    df_accept['normalized_title'] = df_accept['Question'].str.strip().str.lower()
    df_py['normalized_title'] = df_py['title'].str.strip().str.lower()

    # Merge DataFrames on normalized title
    merged_df = pd.merge(
        df_py[['title', 'difficulty', 'content', 'python', 'normalized_title']],
        df_accept[['normalized_title', 'Acceptance_rate', 'Topic_tags']],
        on='normalized_title',
        how='inner'
    )

    # Select only the required columns and remove the temporary key
    result_df = merged_df[['title', 'difficulty', 'content', 'python', 'Acceptance_rate', 'Topic_tags']]

    # Print merge statistics
    print(f"Original DataFrames: {len(df_py)} Python solutions, {len(df_accept)} questions with acceptance rate")
    print(f"Merged DataFrame: {len(result_df)} rows")

    return result_df



def calculate_normalized_difficulty(row):
    """
    Calculate normalized difficulty score between 0 (very easy) and 1 (very hard)
    using both categorical difficulty and acceptance rate.

    Args:
        row: DataFrame row with 'difficulty' and 'Acceptance_rate' fields

    Returns:
        float: Normalized difficulty score between 0 and 1
    """
    # Convert categorical difficulty to base score
    difficulty_base = {
        'Easy': 0.0,
        'Medium': 0.33,
        'Hard': 0.66
    }

    # Define category range width
    category_width = 0.33

    # Extract data
    difficulty = row['difficulty']
    # Remove '%' character if present and convert to float
    acceptance_rate = float(row['Acceptance_rate'].replace('%', '')) if isinstance(row['Acceptance_rate'], str) else \
    row['Acceptance_rate']

    # Calculate base score from difficulty category
    base_score = difficulty_base.get(difficulty, 0.5)  # Default to middle if unknown

    # Calculate position within range based on acceptance rate (inverse relationship)
    # Lower acceptance rate = higher position in range = harder problem
    position_in_range = (1 - acceptance_rate / 100) * category_width

    # Calculate final normalized score
    normalized_score = base_score + position_in_range

    return round(normalized_score, 3)


if __name__ == "__main__":
    leetcode_df_with_acceptance_rate = pd.read_csv(
        "/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/leetcode_data2/Leetcode_Questions_updated (2024-11-02).csv")

    leetcode_df_with_py_answers = pd.read_json("hf://datasets/greengerong/leetcode/leetcode-train.jsonl", lines=True)
    merged_leetcode_df = merge_leetcode_dataframes(leetcode_df_with_acceptance_rate, leetcode_df_with_py_answers)
    merged_leetcode_df['normalized_difficulty'] = merged_leetcode_df.apply(calculate_normalized_difficulty, axis=1)
    print(merged_leetcode_df.head(10))
    merged_leetcode_df.to_csv('/media/omridan/data/work/msc/NLP/NLP_final_project/data/leetcode/merged_leetcode_df.csv', index=False)

