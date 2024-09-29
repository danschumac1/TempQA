"""
Created on 09/15/2024

@author: Dan Schumacher

# How to run this script:
#python src/format_l1_tempReason.py

"""

import os
import pandas as pd
from utils.format_data_utils import temp_reason_basic_processing, assign_mixed_context

def main():
    paths = {
        'train_l1': './data/datasets/TempReason/original/train_l1.json',
        'dev_l1': './data/datasets/TempReason/original/val_l1.json',
        'test_l1': './data/datasets/TempReason/original/test_l1.json'
    }

    output_dir = './data/datasets/TempReason/final'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, path in paths.items():
        print(f"Processing {name}...")

        # Load the main dataset
        df = pd.read_json(path, lines=True)
        df = temp_reason_basic_processing(df)

        # Determine the type of data (train/dev/test)
        train_dev_test = name.split('_')[0]

        # Load the corresponding context dataset
        context_path = path.split('original')[0] + f'final/{train_dev_test}_l2.jsonl'
        context_df = pd.read_json(context_path, lines=True)

        # Get lengths of the dfs
        len_con_data = len(context_df)
        len_data = len(df)

        # Handle length mismatch by extending the smaller context df
        if len_con_data < len_data:
            repetitions = len_data // len_con_data + 1
            context_df_extended = context_df.loc[context_df.index.repeat(repetitions)].reset_index(drop=True)
            context_df_extended = context_df_extended.iloc[:len_data]
        else:
            context_df_extended = context_df

        # Add the extended context columns to the main df
        df['relevant_context'] = context_df_extended['relevant_context']
        df['random_context'] = context_df_extended['random_context']
        df['wrong_date_context'] = context_df_extended['wrong_date_context']
        assign_mixed_context(df)

        # Save the processed df
        output_path = os.path.join(output_dir, f'{name}.jsonl')
        df.to_json(output_path, orient='records', lines=True)
        print(f'{name} saved successfully to {output_path}')

if __name__ == '__main__':
    main()
