import string
import json
from typing import List
from utils.gen_utils import preprocess_text
import pandas as pd

def load_funky_json(file_path):
    """
    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries loaded from the JSONL file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                pass
                # print(f"Error decoding JSON: {e} - Line skipped")
        return data
    
def extract_generations(gen_list: List[dict], key_name:str = 'OUTPUT') -> list:
    """
    Extracts the generations from a list of dictionaries.

    Args:
        gen_list (List[dict]): A list of dictionaries containing the generations.
        key_name (str, optional): The key name to extract the generations from. Defaults to 'PREDICTION'.

    Returns:
        list: A list of cleaned generations.
        
    Raises:
        Exception: If an error occurs while extracting the generations.

    """
    cleaned_list = []
    for gen in gen_list:
        cleaned_list.append(preprocess_text(gen[key_name].split('he answer is ')[1]))
        # cleaned_list.append(gen[key_name].split('he answer is ')[1])

    return cleaned_list

def extract_actual_answers(actual_df: pd.DataFrame, answer_key: str = 'answer') -> List[List[str]]:
    """
    Extracts the actual answers from a DataFrame and preprocesses them.

    Args:
        actual_df (pd.DataFrame): The DataFrame containing the actual answers.
        output_key (str, optional): The column name in the DataFrame that contains the answers. Defaults to 'answer'.

    Returns:
        List[List[str]]: A list of lists, where each inner list contains the preprocessed actual answers.

    """
    cleaned_answers_lists = []
    for answer in actual_df[answer_key]:
        if isinstance(answer, list):  # Check if it's a list
            cleaned_answers_lists.append([preprocess_text(ans) for ans in answer])  # Preprocess each answer and wrap in a list
            # cleaned_answers_lists.append([ans for ans in answer])  # Preprocess each answer and wrap in a list

        elif isinstance(answer, str):  # Check if it's a string
            cleaned_answers_lists.append([preprocess_text(answer)])  # Wrap the string in a list
            # cleaned_answers_lists.append(answer)  # Wrap the string in a list

        else:
            print(f"Answer is neither a list nor a string: {answer}")
            cleaned_answers_lists.append([])  # Append an empty list or handle as needed
    return cleaned_answers_lists

def calc_f1(pred: str, answer_list: List[str]) -> float:
    """
    Calculates the F1 score for a given prediction and a list of answers.
    Args:
        pred (str): The prediction string.
        answer_list (List[str]): The list of answer strings.
    Returns:
        float: The F1 score.
    """
    # Rest of the code...
    pred_words = set(pred.lower().split())  # Convert the prediction into a set of words for faster operations
    best_f1 = 0  # Initialize best F1 score

    for answer in answer_list:
        answer_words = set(answer.lower().split())  # Convert answer into a set of words
        TP = len(answer_words.intersection(pred_words))
        FP = len(pred_words.difference(answer_words))
        FN = len(answer_words.difference(pred_words))
        
        if TP == 0:
            f1 = 0
        else:
            prec = TP / (TP + FP) if TP + FP > 0 else 0
            rec = TP / (TP + FN) if TP + FN > 0 else 0
            if (prec + rec) > 0:
                f1 = 2 * ((prec * rec) / (prec + rec))
            else:
                f1 = 0

        if f1 > best_f1:
            best_f1 = f1

    return best_f1

def calc_contains_acc(pred:str, answer_list:List[str]) -> int:
    """
    Checks if any answer in the list is contained within the prediction after removing punctuation
    and converting to lowercase.

    Parameters:
    - pred (str): The prediction string to be evaluated.
    - answer_list (list of str): A list of answer strings against which the prediction is evaluated.

    Returns:
    - bool: True if any answer is contained within the prediction, False otherwise.
    """
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    normalized_pred = pred.lower().translate(translator)

    for answer in answer_list:
        # Normalize each answer
        normalized_answer = answer.lower().translate(translator)
        # Check if the normalized answer is contained within the normalized prediction
        if normalized_answer in normalized_pred:
            return 1

    return 0