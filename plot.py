import os
import re
import dotenv
import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load environment variables
dotenv.load_dotenv()


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the dataset from the given path and convert it to a Pandas DataFrame.

    Args:
        dataset_path (str): Path to the dataset on disk.

    Returns:
        pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    return datasets.load_from_disk(dataset_path).to_pandas()


def extract_rating(text: str) -> int:
    """
    Extract the numeric rating from judge output text.

    Args:
        text (str): Judge output text containing the rating.

    Returns:
        int: Extracted rating or None if not found.
    """
    pattern = re.compile(r"\[RESULT\]\s*(\d+)")
    if not isinstance(text, str):
        return None
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def add_rating_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ratings from judge output columns and add them as separate columns.

    Args:
        df (pd.DataFrame): Input DataFrame with judge output columns.

    Returns:
        pd.DataFrame: DataFrame with additional rating columns.
    """
    for col in df.columns:
        if col.startswith("judge_output_"):
            judge_model_name = col[len("judge_output_"):]
            rating_col = f"rating_{judge_model_name}"
            df[rating_col] = df[col].apply(extract_rating)
    return df


def generate_correlation_heatmaps(df: pd.DataFrame, rating_cols: list, output_dir: str):
    """
    Generate and save correlation heatmaps for different categories.

    Args:
        df (pd.DataFrame): DataFrame containing rating columns.
        rating_cols (list): List of rating column names.
        output_dir (str): Directory to save plots.
    """
    categories_to_plot = [None, 'explanation', 'coding', 'math', 'summarization']
    os.makedirs(output_dir, exist_ok=True)

    for category in categories_to_plot:
        df_subset = df if category is None else df[df['categories'] == category]

        if df_subset.empty:
            print(f"No data for category: {category}, skipping.")
            continue

        title = f"Correlation - {category.capitalize() if category else 'All Categories'}"
        corr_matrix = df_subset[rating_cols].corr()

        sorted_labels = sorted(corr_matrix.index, key=lambda x: x.lower())
        corr_matrix = corr_matrix.loc[sorted_labels, sorted_labels]

        plt.figure(figsize=(9, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn", center=0, vmin=-1, vmax=1)
        plt.xticks(rotation=45, ha='right')
        plt.title(title)
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f"corr_{category or 'all'}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved correlation heatmap: {plot_filename}")


def generate_model_grading_heatmaps(df: pd.DataFrame, rating_cols: list, output_dir: str):
    """
    Generate and save model grading heatmaps.

    Args:
        df (pd.DataFrame): DataFrame containing rating columns.
        rating_cols (list): List of rating column names.
        output_dir (str): Directory to save plots.
    """
    categories_to_plot = [None, 'explanation', 'coding', 'math', 'summarization']
    os.makedirs(output_dir, exist_ok=True)

    def lower_sort(labels):
        """Helper function to sort labels case-insensitively."""
        return sorted(labels, key=str.lower)

    for cat in categories_to_plot:
        cat_df = df if cat is None else df[df["categories"] == cat]
        cat_label = "All answers" if cat is None else f"Category: {cat}"

        if cat_df.empty:
            print(f"No data for {cat_label}, skipping.")
            continue

        # Extract model ID and compute mean ratings per model
        cat_df['model_id'] = cat_df['model_id'].apply(lambda x: x.split('/')[-1])
        group_means = cat_df.groupby("model_id")[rating_cols].mean()

        # Rename columns to remove "rating_" prefix
        cleaned_columns = [c.replace("rating_", "").split('/')[-1] for c in group_means.columns]
        group_means.columns = cleaned_columns

        # Sort row index & columns alphabetically (case-insensitive)
        group_means = group_means.sort_index(key=lambda idx: idx.str.lower())
        sorted_cols = lower_sort(group_means.columns)
        group_means = group_means.reindex(columns=sorted_cols)

        # Plot heatmap
        plt.figure(figsize=(9, 6))
        sns.heatmap(group_means, annot=True, cmap="RdYlGn")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Average Grades by Model and Judge\n{cat_label}")
        plt.xlabel("Judge Model")
        plt.ylabel("Answer-Producing Model")
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f"grading_{cat or 'all'}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved grading heatmap: {plot_filename}")


if __name__ == '__main__':
    # Load dataset
    dataset_path = 'data/answers_multiple_models_rated'
    output_dir = 'plots'
    
    df = load_dataset(dataset_path)

    # Extract ratings
    df = add_rating_columns(df)
    rating_cols = [c for c in df.columns if c.startswith("rating_")]

    # Generate and save plots
    generate_correlation_heatmaps(df, rating_cols, output_dir)
    generate_model_grading_heatmaps(df, rating_cols, output_dir)

