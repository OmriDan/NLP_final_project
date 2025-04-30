import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Set style for better visualizations
plt.style.use('ggplot')
sns.set(font_scale=1.2)

def load_and_explore_data(file_path):
    """Load the dataset and perform initial exploration"""
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")

    print("\nMissing values:")
    print(df.isnull().sum())

    return df

def analyze_difficulty_distribution(df):
    """Analyze the distribution of difficulty scores"""
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Numeric difficulty distribution
    if 'difficulty' in df.columns:
        sns.histplot(df['difficulty'], kde=True, ax=ax[0])
        ax[0].set_title('Distribution of Difficulty Scores')
        ax[0].set_xlabel('Difficulty Score')

        # Show statistics
        print("\nDifficulty Score Statistics:")
        print(df['difficulty'].describe())

    # Category difficulty distribution
    category_counts = df['category_difficulty'].value_counts()
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax[1])
    ax[1].set_title('Distribution of Category Difficulty')
    ax[1].set_xlabel('Category')
    ax[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('difficulty_distribution.png')

    return fig

def analyze_acceptance_vs_difficulty(df):
    """Analyze relationship between acceptance rate and difficulty"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert acceptance rate to numeric if needed
    if df['Acceptance_rate'].dtype == 'object':
        df['Acceptance_rate_numeric'] = df['Acceptance_rate'].str.rstrip('%').astype(float)
    else:
        df['Acceptance_rate_numeric'] = df['Acceptance_rate']

    # Plot by category difficulty
    colors = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}
    for category in df['category_difficulty'].unique():
        subset = df[df['category_difficulty'] == category]
        ax.scatter(subset['Acceptance_rate_numeric'], subset['difficulty'],
                  alpha=0.7, label=category, color=colors.get(category, 'blue'))

    ax.set_title('Acceptance Rate vs Difficulty')
    ax.set_xlabel('Acceptance Rate')
    ax.set_ylabel('Difficulty Score')
    ax.legend()

    # Calculate correlation
    corr = df['Acceptance_rate_numeric'].corr(df['difficulty'])
    print(f"\nCorrelation between Acceptance Rate and Difficulty: {corr:.3f}")

    plt.tight_layout()
    plt.savefig('acceptance_vs_difficulty.png')

    return fig

def analyze_question_complexity(df):
    """Analyze question complexity metrics and their relation to difficulty"""
    # Extract metrics like question length, code length, etc.
    df['question_length'] = df['question'].apply(lambda x: len(x))
    df['answer_length'] = df['answer'].apply(lambda x: len(x))

    # Calculate reading complexity metrics (simplified Flesch-Kincaid)
    def count_sentences(text):
        return len(re.split(r'[.!?]+', text))

    def count_words(text):
        return len(re.findall(r'\w+', text))

    df['question_sentences'] = df['question'].apply(count_sentences)
    df['question_words'] = df['question'].apply(count_words)
    df['words_per_sentence'] = df['question_words'] / df['question_sentences'].replace(0, 1)

    # Plot correlations
    metrics = ['question_length', 'answer_length', 'question_words', 'words_per_sentence']
    correlations = [df[metric].corr(df['difficulty']) for metric in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, correlations)

    # Add correlation values on top of bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.annotate(f'{corr:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')

    ax.set_title('Correlation of Text Metrics with Difficulty')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig('complexity_correlations.png')

    return fig

def analyze_topics(df):
    """Analyze topic tags and their relationship with difficulty"""
    # Extract all tags
    all_tags = []
    for tags in df['Topic_tags'].str.split(','):
        if isinstance(tags, list):
            all_tags.extend([tag.strip() for tag in tags])

    tag_counter = Counter(all_tags)

    # Get top tags
    top_tags = [tag for tag, count in tag_counter.most_common(10)]

    # Create binary features for top tags
    for tag in top_tags:
        df[f'tag_{tag}'] = df['Topic_tags'].str.contains(tag).astype(int)

    # Calculate average difficulty by tag
    tag_difficulty = {}
    for tag in top_tags:
        tag_difficulty[tag] = df.loc[df[f'tag_{tag}'] == 1, 'difficulty'].mean()

    # Sort tags by difficulty
    sorted_tags = sorted(tag_difficulty.items(), key=lambda x: x[1])
    tags, difficulties = zip(*sorted_tags)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(tags, difficulties)

    # Add values on bars
    for bar, diff in zip(bars, difficulties):
        width = bar.get_width()
        ax.annotate(f'{diff:.2f}',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(3, 0),  # 3 points horizontal offset
                   textcoords="offset points",
                   ha='left', va='center')

    ax.set_title('Average Difficulty by Topic Tag')
    ax.set_xlabel('Average Difficulty')

    plt.tight_layout()
    plt.savefig('topic_difficulty.png')

    return fig

def generate_text_embeddings(df):
    """Generate text embeddings for questions and visualize clusters"""
    # Create TF-IDF vectors from questions
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    question_vectors = vectorizer.fit_transform(df['question'])

    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    question_2d = pca.fit_transform(question_vectors.toarray())

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color points by category difficulty
    categories = df['category_difficulty'].unique()
    colors = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}

    for category in categories:
        mask = df['category_difficulty'] == category
        ax.scatter(question_2d[mask, 0], question_2d[mask, 1],
                  label=category, alpha=0.7, color=colors.get(category, 'blue'))

    ax.set_title('Question Text Embedding Visualization')
    ax.legend()

    plt.tight_layout()
    plt.savefig('question_embeddings.png')

    return fig

def main():
    # Load the data
    file_path = 'data/leetcode/leetcode_val.csv'
    df = load_and_explore_data(file_path)

    # Run analyses
    analyze_difficulty_distribution(df)
    analyze_acceptance_vs_difficulty(df)
    analyze_question_complexity(df)
    # analyze_topics(df)
    generate_text_embeddings(df)

    # Print summary insights
    print("\n=== SUMMARY INSIGHTS ===")
    print("1. Difficulty Distribution: See difficulty_distribution.png")
    print("2. Acceptance Rate vs Difficulty: See acceptance_vs_difficulty.png")
    print("3. Question Complexity Metrics: See complexity_correlations.png")
    print("4. Topic Tag Analysis: See topic_difficulty.png")
    print("5. Question Text Embeddings: See question_embeddings.png")

    print("\nAnalysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()