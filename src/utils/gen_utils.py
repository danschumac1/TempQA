import re
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer
import random

def gemma_generation_formatter(df: pd.DataFrame, context_type: str) -> list:
    prompts = []
    # Iterate over the DataFrame rows
    for question, context in zip(df['question'], df[context_type]):
        prompt = f'''<bos><start_of_turn>user\nIn as few words as possible, answer the following question given the context.\nQuestion: {question}\nContext: {context}<end_of_turn>\n<start_of_turn>model\n'''
        prompts.append(prompt)
    return prompts

def llama_generation_formatter(df: pd.DataFrame, context_type: str) -> list:
    prompts = []
    for question, context in zip(df['question'], df[context_type]):
        system = "You are an expert in answering time related questions. Please provide consistent, brief answers in the style of 'The answer is X'."
        prompt  = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        prompts.append(prompt)
    return prompts

def mistral_generation_formatter(df: pd.DataFrame, context_type: str) -> list:
    prompts = []
    for question, context in zip(df['question'], df[context_type]):
        prompt = f'''<s>[INST] In as few words as possible, answer the following question given the context.\nQuestion:{question}\nContext:{context}[/INST]\n'''
        prompts.append(prompt)
    return prompts


def get_format_function(model:str) -> callable:
    # Define mappings for instruction-tuned (IT) and non-instruction-tuned (NIT) models
    format_functions = {
        'gemma': gemma_generation_formatter,
        'llama': llama_generation_formatter,
        'mistral': mistral_generation_formatter
    }
    
    return format_functions.get(model)

def map_tokens_over_data(df: pd.DataFrame, tokenizer: PreTrainedTokenizer, context: str) -> Dataset:
    """
    Maps tokens over the given data using the provided tokenizer.
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
        context (str): The context to extract from the DataFrame.
    Returns:
        Dataset: The tokenized dataset.
    """
    pass
    # convert from pandas to dataset obj
    dataset = Dataset.from_pandas(df)
    
    # TOKENIZE DATASET
    dataset = dataset.map(lambda samples: tokenizer(samples[context]), batched=True)
    # limit the length
    # dataset = dataset.map(lambda x: tokenizer(x[context], max_length=512, truncation=True), batched=True)

    return dataset

def assign_mixed_context(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Assigns a mixed context based on the given random seed.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing context options.
    random_seed (int): Seed for the random generator. Default is 42.

    Returns:
    pd.DataFrame: The same DataFrame with an added 'mixed_context' column.
    """
    random.seed(random_seed)
    df['no_context'] = ''  # Add no_context column as None initially
    options = ['no_context', 'relevant_context', 'random_context', 'wrong_date_context']

    # Create 'mixed_context' column by directly modifying the DataFrame using .loc[]
    df['mixed_context'] = df.apply(lambda row: row[random.choice(options)], axis=1)

    # Drop the no_context column after use
    df.drop(columns=['no_context'], inplace=True)

    return df

def preprocess_text(text:str) -> str:
    """
    Preprocesses the given text by converting it to lowercase and removing non-alphanumeric characters.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.

    """
    text = ' '.join(re.findall(r"\w+", text.lower()))
    return text 