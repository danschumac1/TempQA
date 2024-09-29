from typing import List
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

from utils.training_utils import extract_answer_from_list


def format_IT_c_first(df:pd.DataFrame, context:bool=False) -> List[str]:
    """
    Formats the input DataFrame into a list of tokenized instructions for Italian conversational AI.

    Args:
        df (pd.DataFrame): The input DataFrame containing the questions and optional context.
        context (bool, optional): Flag indicating whether to include context in the tokenized instructions. 
                                    Defaults to False.

    Returns:
        List[str]: A list of tokenized instructions for Italian conversational AI.

    Raises:
        None
    """
    tokenized_instructions = []
    ft_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", add_bos_token=True, add_eos_token=False)
    ft_tokenizer.padding_side = 'left'

    if context == 'no_context':
        context = False
    if context:
        for question, context in zip(df['question'], df[context]):
            message = f"<start_of_turn>user\nGiven the following context, answer the question. Here is the context: {context[:500]} Here is the question: {question} <end_of_turn>\n<start_of_turn>model\nThe answer is:"
            tokenized_instructions.append(message)
    else:
        for question in df['question']:
            message = f"<start_of_turn>user\Answer the question. Here is the question: {question} <end_of_turn>\n<start_of_turn>model\nThe answer is:"
            tokenized_instructions.append(message)
    return tokenized_instructions


def IT_formatter(df:pd.DataFrame, context_type:str) -> list:
    tokenized_instructions = []

    fi_tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b-it", add_bos_token=True, add_eos_token=True, max_length=512, truncation=True
        )
    fi_tokenizer.padding_side = 'right'

    for _, (question, answer, context) in df[['question', 'answer', context_type]].iterrows():
        answer = extract_answer_from_list(answer)

        if context_type == 'no_context' or context == '':
            message = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f'The answer is: {answer}.'}
            ]
        else:
            message = [
                {"role": "user", "content": f'{question} Here is the context: {context[:500]}'},
                {"role": "assistant", "content": f'The answer is: {answer}.'}
            ]

        encodeds = fi_tokenizer.apply_chat_template(message, return_tensors="pt")
        output = fi_tokenizer.batch_decode(encodeds, special_tokens=True)
        tokenized_instructions.append(output[0] + '<eos>')
    return tokenized_instructions

def format_gemma_generations(df: pd.DataFrame, context_type: str) -> List[str]:
    tokenized_instructions = []
    ft_tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b-it", add_bos_token=True, add_eos_token=False
    )
    ft_tokenizer.padding_side = 'left'

    for _, row in df[['question', context_type]].iterrows():
        question, context = row['question'], row[context_type]
        
        if context_type == 'no_context' or context == '':
            message = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\nThe answer is: "
            tokenized_instructions.append(message)
        else:
            # Consistent formatting of the message, regardless of context presence
            message = [
                {"role": "user", "content": f"{question}\nHere is the context: {context}"},
            ]
            encodeds = ft_tokenizer.apply_chat_template(message, return_tensors="pt")
            output = ft_tokenizer.batch_decode(encodeds, skip_special_tokens=True)
            tokenized_instructions.append(output[0] + '<start_of_turn>model\nThe answer is: ')
    
    return tokenized_instructions

def map_tokens_over_data(df: pd.DataFrame, tokenizer: AutoTokenizer, context: str) -> Dataset:
    """
    Maps tokens over the data in a DataFrame using a tokenizer.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
        context (str): The context to extract from the DataFrame.

    Returns:
        Dataset: The dataset with mapped tokens.

    """
    dataset = df.map(lambda samples: tokenizer(samples[f"{context.split('_')[0]}_prompt"]))
    return dataset

def format_NIT_c_first(df:pd.DataFrame, context:bool=False) -> List[str]:
    """
    Formats the given DataFrame `df` into a list of tokenized instructions.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        context (bool, optional): Whether to include context in the tokenized instructions. Defaults to False.
    Returns:
        List[str]: The list of tokenized instructions.
    Raises:
        None
    """
    tokenized_instructions = []
    
    if context == 'no_context':
        context = False
    if context:
        for question, context in zip(df['question'], df[context]):
            message = f'<bos>Here is the context: {context[:500]} Here is the question: {question}  The answer is: '
            tokenized_instructions.append(message)
    else:
        for question in df['question']:
            message = f'<bos>{question} The answer is: '
            tokenized_instructions.append(message)

    return tokenized_instructions

def format_NIT_q_first(df:pd.DataFrame, context:bool=False) -> List[str]:
    """
    Formats the questions from a DataFrame into a list of tokenized instructions.
    Args:
        df (pd.DataFrame): The DataFrame containing the questions.
        context (bool, optional): Indicates whether to include the context in the tokenized instructions. 
                                    Defaults to False.
    Returns:
        List[str]: The list of tokenized instructions.
    """
    tokenized_instructions = []
 
    if context:
        for question, context in zip(df['question'], df[context]):

            message = f'<bos>Answer the following question using the context.\nQuestion: {question}\nHere is the context: {context}\nThe answer is: '
            tokenized_instructions.append(message)
    else:
        for question in df['question']:
            
            message = f'<bos>Answer the following question.\nQuestion: {question}\nThe answer is: '
            tokenized_instructions.append(message)

    return tokenized_instructions


#endregion
#region # TRAINING
# =============================================================================
# TRAINING
# =============================================================================
def IT_training_formatter(df:pd.DataFrame, context_type:str) -> list:
    tokenized_instructions = []

    fi_tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b-it", add_bos_token=True, add_eos_token=True, max_length=512, truncation=True
        )
    fi_tokenizer.padding_side = 'right'

    for _, (question, answer, context) in df[['question', 'answer', context_type]].iterrows():
        answer = extract_answer_from_list(answer)

        if context_type == 'no_context' or context == '':
            message = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f'The answer is: {answer}.'}
            ]
        else:
            message = [
                {"role": "user", "content": f'{question} Here is the context: {context[:500]}'},
                {"role": "assistant", "content": f'The answer is: {answer}.'}
            ]

        encodeds = fi_tokenizer.apply_chat_template(message, return_tensors="pt")
        output = fi_tokenizer.batch_decode(encodeds, special_tokens=True)
        tokenized_instructions.append(output[0] + '<eos>')
    return tokenized_instructions

def NIT_training_formatter(df:pd.DataFrame, context_type:str) -> list:
    tokenized_instructions = []

    for question, answer, context in zip(df['question'], df['answer'], df[context_type]):
        answer = extract_answer_from_list(answer)

        if context_type == 'no_context' or context == '':
            message = f'<bos>{question} The answer is: {answer}<eos>'
        else:
            message = f'<bos>{question} Here is the context: {context[:500]} The answer is: {answer}<eos>'

        tokenized_instructions.append(message)
    return tokenized_instructions