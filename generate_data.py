import os
import dotenv
import datasets
import pandas as pd
import concurrent.futures
import httpx
from httpx_socks import SyncProxyTransport
from tqdm import tqdm
import openai

# Load environment variables
dotenv.load_dotenv('.env')

API_LINK = os.getenv("API_LINK")
PROXY_URL = os.getenv("PROXY_URL")
API_TOKEN = os.getenv("API_TOKEN")

DATASET_PATH = "data/lmsys_subset"
OUTPUT_PATH = "data/answers_multiple_models"


def send_question(messages, model, temperature, max_tokens):
    """
    Sends a prompt to the OpenAI-compatible model via the provided API link.
    
    Args:
        messages (list): List of message dictionaries following OpenAI format.
        model (str): Model identifier.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum response tokens.
    
    Returns:
        str: Model-generated response or error message.
    """
    transport = SyncProxyTransport.from_url(PROXY_URL)
    http_client = httpx.Client(transport=transport)
    client = openai.OpenAI(http_client=http_client, api_key=API_TOKEN, base_url=API_LINK)

    response = None
    attempt = 0
    while response is None and attempt < 10:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=1,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Error during API call: {e}")
            attempt += 1
            continue
    
    return response.choices[0].message.content if response else "Model did not return a response."


def process_prompt(text, model, item_no, data_len, temperature, max_tokens):
    """
    Processes a single text prompt using a given model.
    
    Args:
        text (str): Input prompt.
        model (str): Model identifier.
        item_no (int): Index of the prompt in the dataset.
        data_len (int): Total number of prompts.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum response tokens.
    
    Returns:
        tuple: (item_no, response)
    """
    messages = [{"role": "user", "content": text}]
    output = send_question(messages, model, temperature, max_tokens)
    print(f"Processed {item_no + 1}/{data_len}")
    return item_no, output


def get_answers(data_list, model, num_threads=10, temperature=0.0, max_tokens=512):
    """
    Sends a batch of prompts to a model using parallel execution.
    
    Args:
        data_list (list): List of prompts.
        model (str): Model identifier.
        num_threads (int): Number of parallel threads.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum response tokens.
    
    Returns:
        list: Model-generated responses, maintaining original order.
    """
    data_len = len(data_list)
    results = [None] * data_len

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_item = {
            executor.submit(process_prompt, text, model, i, data_len, temperature, max_tokens): i
            for i, text in enumerate(data_list)
        }
        for future in concurrent.futures.as_completed(future_to_item):
            idx = future_to_item[future]
            item_no, output = future.result()
            results[item_no] = output
    
    return results


def save_results(records, output_path):
    """
    Saves results to a Hugging Face dataset format.
    
    Args:
        records (list): List of dictionaries containing model results.
        output_path (str): Path to save the dataset.
    """
    df = pd.DataFrame(records)
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


def main():
    """
    Main function to load dataset, process prompts using multiple models, and save results.
    """
    # Load dataset
    data = datasets.load_from_disk(DATASET_PATH)
    prompts = [row["prompts"] for row in data]
    categories = [row["categories"] for row in data]
    
    # Define models
    models_to_evaluate = [
        "microsoft/phi-4",
        "mistralai/mistral-small-24b-instruct-2501",
        "google/gemini-2.0-flash-001",
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "openai/chatgpt-4o-latest",
        "anthropic/claude-3.5-sonnet",
    ]
    
    records = []
    for model_id in tqdm(models_to_evaluate, desc="Evaluating Models"):
        answers = get_answers(prompts, model_id, num_threads=10, temperature=0.0, max_tokens=512)
        for idx, (prompt, category) in enumerate(zip(prompts, categories)):
            records.append({
                "model_id": model_id,
                "prompt": prompt,
                "answer": answers[idx],
                "categories": category,
            })
        save_results(records, OUTPUT_PATH)
    
    # Save results
    save_results(records, OUTPUT_PATH)


if __name__ == "__main__":
    main()

