import os
import dotenv
import datasets
import openai
import pandas as pd
import httpx
import concurrent.futures
from tqdm import tqdm
from httpx_socks import SyncProxyTransport

# Load environment variables
dotenv.load_dotenv('.env')

API_LINK = os.getenv("API_LINK")
API_TOKEN = os.getenv("API_TOKEN")
PROXY_URL = os.getenv("PROXY_URL")

DATA_PATH = "data/answers_multiple_models"
OUTPUT_PATH = "data/answers_multiple_models_rated"


def create_judge_prompt(instruction: str, answer: str, category: str) -> str:
    return f"""###Task Description:

You will be given an instruction, a category of the question and the answer. Your task is to:

1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 8. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 8}}"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{answer}

###Category:
{category}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect and incoherent, consisting of broken language or incomplete.
Score 2: The response is coherent, but the answer is unrelated to the instruction.
Score 3: The response is coherent, but it is a refusal to answer the instruction.
Score 4: The response is coherent and relevant to the instruction, but the result contradicts the instruction.
Score 5: The response is coherent and relevant to the instruction, but the resulting answer is incomplete in the sense that it does not fully follow the instruction, only doing part of the job.
Score 6: The response is coherent and relevant to the instruction, the resulting answer is complete, but the response contains some additional information, which was not explicitly asked in the instruction.
Score 7: The response is coherent, relevant to the instruction, the resulting answer is complete and without any additional unneeded information, but it contains odd or broken formatting, imperfect use of language and anything else, which makes this answer not ideal.
Score 8: The response is coherent, relevant to the instruction, the resulting answer is complete, without any additional information, it is perfectly formatted, uses perfect language and overall a perfect answer to the instruction query.

###Feedback:
"""


def send_question(messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    """
    Sends a prompt to the OpenAI-compatible model via an API.

    Args:
        messages (list[dict[str, str]]): The conversation history.
        model (str): The model name.
        temperature (float): Sampling temperature for randomness.
        max_tokens (int): Maximum token count for response.

    Returns:
        str: The model's response.
    """
    transport = SyncProxyTransport.from_url(PROXY_URL) if PROXY_URL else None
    http_client = httpx.Client(transport=transport) if transport else httpx.Client()

    client = openai.OpenAI(
        http_client=http_client,
        api_key=API_TOKEN,
        base_url=API_LINK,
    )

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
            print(f"Error during API call (attempt {attempt + 1}): {e}")
            attempt += 1

    return response.choices[0].message.content if response else "Model did not return a response."


def process_prompt(text: str, model: str, item_no: int, data_len: int, temperature: float, max_tokens: int) -> tuple:
    """
    Processes a single prompt using the given model.

    Args:
        text (str): The prompt to send.
        model (str): The model name.
        item_no (int): The index of the prompt in the dataset.
        data_len (int): Total number of prompts.
        temperature (float): Sampling temperature for randomness.
        max_tokens (int): Maximum token count for response.

    Returns:
        tuple: (item_no, model response).
    """
    messages = [{"role": "user", "content": text}]
    output = send_question(messages, model, temperature, max_tokens)
    print(f"Processed {item_no + 1}/{data_len}")
    return item_no, output


def get_answers(data_list: list[str], model: str, num_threads: int = 10, temperature: float = 0.0, max_tokens: int = 512) -> list[str]:
    """
    Fetches answers from the model for a list of prompts using parallel processing.

    Args:
        data_list (list[str]): List of prompts.
        model (str): The model name.
        num_threads (int, optional): Number of parallel threads. Defaults to 10.
        temperature (float, optional): Sampling temperature. Defaults to 0.0.
        max_tokens (int, optional): Maximum token count. Defaults to 512.

    Returns:
        list[str]: The model's responses.
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


if __name__ == '__main__':
    # Load dataset
    dataset = datasets.load_from_disk(DATA_PATH).select(range(5))
    df = dataset.to_pandas()

    # Generate judge prompts
    judge_prompts = [
        create_judge_prompt(row["prompt"], row["answer"], row["categories"])
        for _, row in df.iterrows()
    ]

    # Models to evaluate
    models_to_evaluate = [
        'microsoft/phi-4',
        'mistralai/mistral-small-24b-instruct-2501',
        'google/gemini-2.0-flash-001',
        'openai/gpt-4o-mini',
        'meta-llama/llama-3.3-70b-instruct',
        'qwen/qwen-2.5-72b-instruct',
        'openai/chatgpt-4o-latest',
        'anthropic/claude-3.5-sonnet',
    ]

    # Evaluate responses for each model
    for judge_model in tqdm(models_to_evaluate, desc="Evaluating models"):
        df[f"judge_output_{judge_model.split('/')[-1]}"] = get_answers(judge_prompts, judge_model)

        final_dataset = datasets.Dataset.from_pandas(df)
        final_dataset.save_to_disk(OUTPUT_PATH)

    # Save results
    final_dataset = datasets.Dataset.from_pandas(df)
    final_dataset.save_to_disk(OUTPUT_PATH)

    print("Final dataset saved at:", OUTPUT_PATH)
    print("Total samples:", len(final_dataset))
    print("Columns:", final_dataset.column_names)

