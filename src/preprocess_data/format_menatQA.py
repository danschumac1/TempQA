"""
Created on 09/15/2024

@author: Dan Schumacher
How to run this script:
python ./src/preprocess_data/format_menatQA.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from multiprocessing import cpu_count
import pandas as pd
from utils.format_data_utils import assign_no_context, generate_wd_context, generate_random_context_Menat, assign_mixed_context

from typing import List, Dict

if __name__ == "__main__":
    # HOW RIOS DO
    # ADD INDEX FOR ROW WHERE IT CAME FROM
    # IF CF AND COMING FROM ROW 3 THEN
    # THEN ALL YOU NEED TO DO IS RANDOMLY A CONTEXT THAT IS NOT FROM ROW 3
    
    # Load the dataset
    with open('data/datasets/MenatQA/MenatQA.json') as f:
        data = json.load(f)

    # Splitting data into train/dev/test
    train_data = data[:int(len(data) * 0.7)]
    dev_data = data[int(len(data) * 0.7):int(len(data) * 0.80)]
    test_data = data[int(len(data) * 0.80):]

    data_dict = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    final_dataframes: Dict[str, pd.DataFrame] = {}  # Dictionary to store DataFrames

    for split, data in data_dict.items():
        return_data: List[Dict] = []  # List to store all the rows for the current split
        for ids, example in enumerate(data, start=1):
            # Process contexts and categorize into order, counterfactual, or scope
            for x in example['context']:
                if x['updated_text'] != '':
                    order_row = {
                        'question': example['updated_question'],
                        'relevant_context': x['updated_text'],
                        'answer': example['updated_answer'],
                        'time_scope': example.get('time_scope', []),
                        'type': 'order',
                        'id': ids
                    }
                    return_data.append(order_row)
                    break  # Break once relevant context is found for 'order'

            if example['type'] == "counterfactual":
                counter_row = {
                    'question': example['updated_question'],
                    'relevant_context': x['text'],
                    'answer': example['updated_answer'],
                    'time_scope': example.get('time_scope', []),
                    'type': 'counterfactual',
                    'id': ids
                }
                return_data.append(counter_row)

                scope_row = {
                    'question': example['question'],
                    'relevant_context': x['text'],
                    'answer': example['updated_answer'],
                    'time_scope': example.get('time_scope', []),
                    'type': 'scope',
                    'id': ids
                }
                return_data.append(scope_row)

            if example['type'] in ["narrow", "expand", "granularity"]:
                narrow_row = {
                    'question': example['question'],
                    'relevant_context': x['text'],
                    'answer': example['updated_answer'],
                    'time_scope': example.get('time_scope', []),
                    'type': 'scope_expand',
                    'id': ids
                }
                return_data.append(narrow_row)

                wide_row = {
                    'question': example['updated_question'],
                    'relevant_context': x['updated_text'],
                    'answer': example['updated_answer'],
                    'time_scope': example.get('time_scope', []),
                    'type': 'scope_expand',
                    'id': ids
                }
                return_data.append(wide_row)

        # Convert the return_data list of dictionaries into a DataFrame
        final_dataframes[split] = pd.DataFrame(return_data)

    # Example: Counting the number of rows for each type
    num_workers = cpu_count()
    for split, df in final_dataframes.items():
        df['no_context'] = ''
        generate_random_context_Menat(df)
        generate_wd_context(df,num_workers)
        assign_mixed_context(df)
        assign_no_context(df)

        # export split
        df.to_json(
            f'data/datasets/MenatQA/final/{split}.jsonl', orient='records', lines=True
            )

    


