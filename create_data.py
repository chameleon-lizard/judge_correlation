import os
import datasets
from tqdm import tqdm


def load_dataset(dataset_name: str, split: str = 'train'):
    """
    Load the specified dataset and convert it to a pandas DataFrame.
    
    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The dataset split to load (default: 'train').
    
    Returns:
        pd.DataFrame: The dataset converted to a pandas DataFrame.
    """
    return datasets.load_dataset(dataset_name, split=split).to_pandas()


def filter_dataset(data, categories, max_prompt_length=8192, avg_response_length=2048, max_samples_per_category=100):
    """
    Filter the dataset based on category and text length constraints.
    
    Args:
        data (pd.DataFrame): The input dataset.
        categories (list): List of categories to filter.
        max_prompt_length (int): Maximum allowed prompt length.
        avg_response_length (int): Maximum average response length.
        max_samples_per_category (int): Maximum samples per category.
    
    Returns:
        dict: Filtered dataset with prompts and categories.
    """
    prompts, filtered_categories = [], []
    
    for idx, cat in enumerate(categories):
        count = 0
        for _, row in tqdm(data[data.category == cat].iterrows(), desc=f"Filtering {cat}"):
            prompt = row['conversations'][0]['value']
            deepseek_response = row['deepseek_response']['value']
            phi3_response = row['phi-3-mini_response']['value']
            avg_length = (len(deepseek_response) + len(phi3_response)) / 2
            
            if (
                len(prompt) <= max_prompt_length and
                avg_length <= avg_response_length
            ):
                prompts.append(prompt)
                filtered_categories.append(cat)
                count += 1
            
            if count >= max_samples_per_category:
                break
    
    return {'prompts': prompts, 'categories': filtered_categories}


def save_dataset(data_dict, output_path):
    """
    Save the filtered dataset to disk.
    
    Args:
        data_dict (dict): Dictionary containing prompts and categories.
        output_path (str): Path to save the dataset.
    """
    dataset = datasets.Dataset.from_dict(data_dict)
    dataset.save_to_disk(output_path)
    print(f"Dataset saved at {output_path}")


def main():
    """
    Main function to execute the dataset filtering process.
    """
    dataset_name = 'OpenLeecher/lmsys_chat_1m_clean'
    output_folder = 'data/lmsys_subset'
    categories_of_interest = ['explanation', 'coding', 'math', 'summarization']
    
    os.makedirs('data', exist_ok=True)
    
    data = load_dataset(dataset_name)
    filtered_data = filter_dataset(data, categories_of_interest)
    save_dataset(filtered_data, output_folder)


if __name__ == '__main__':
    main()
