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
    return datasets.load_from_disk(dataset_path).to_pandas()

def extract_rating(text: str) -> int:
    pattern = re.compile(r"\[RESULT\]\s*(\d+)")
    if not isinstance(text, str):
        return None
    match = pattern.search(text)
    return int(match.group(1)) if match else None

def add_rating_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col.startswith("judge_output_"):
            judge_model_name = col[len("judge_output_"):]
            rating_col = f"rating_{judge_model_name}"
            df[rating_col] = df[col].apply(extract_rating)
    return df

def generate_correlation_heatmaps(df: pd.DataFrame, rating_cols: list, output_dir: str):
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
    categories_to_plot = [None, 'explanation', 'coding', 'math', 'summarization']
    os.makedirs(output_dir, exist_ok=True)

    def lower_sort(labels):
        return sorted(labels, key=str.lower)

    for cat in categories_to_plot:
        cat_df = df if cat is None else df[df["categories"] == cat]
        cat_label = "All answers" if cat is None else f"Category: {cat}"
        if cat_df.empty:
            print(f"No data for {cat_label}, skipping.")
            continue

        cat_df['model_id'] = cat_df['model_id'].apply(lambda x: x.split('/')[-1])
        group_means = cat_df.groupby("model_id")[rating_cols].mean()

        cleaned_columns = [c.replace("rating_", "").split('/')[-1] for c in group_means.columns]
        group_means.columns = cleaned_columns

        group_means = group_means.sort_index(key=lambda idx: idx.str.lower())
        sorted_cols = lower_sort(group_means.columns)
        group_means = group_means.reindex(columns=sorted_cols)

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

def plot_rating_histograms(df: pd.DataFrame, rating_cols: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    n_cols, n_rows = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()

    for i, rating_col in enumerate(sorted(rating_cols, key=lambda x: x.lower())):
        ax = axes[i]
        sns.histplot(df[rating_col].dropna(), bins=range(1, 10), discrete=True, ax=ax)
        ax.set_title(f"Histogram of {rating_col.replace('rating_', '')}")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Frequency")
        ax.set_xticks(range(1, 9))

    for j in range(len(rating_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Rating Distribution Histograms (Ratings 1-8)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    histogram_filename = os.path.join(output_dir, "rating_histograms.png")
    plt.savefig(histogram_filename)
    plt.close()
    print(f"Saved rating histograms plot: {histogram_filename}")

if __name__ == '__main__':
    dataset_path = 'data/answers_multiple_models_rated'
    output_dir = 'plots'

    df = load_dataset(dataset_path)
    df = add_rating_columns(df)
    rating_cols = [c for c in df.columns if c.startswith("rating_")]

    generate_correlation_heatmaps(df, rating_cols, output_dir)
    generate_model_grading_heatmaps(df, rating_cols, output_dir)
    plot_rating_histograms(df, rating_cols, output_dir)

