"""
Created on 09/29/2024

@author: Dan Schumacher
How to use:
python ./src/preprocess_data/format_timeQA.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multiprocessing import cpu_count
import pandas as pd
from utils.format_data_utils import generate_wd_context, generate_random_context_TimeQA, assign_mixed_context

# =============================================================================
# FUNCTIONS
# =============================================================================
def preprocess_data(file_path):
    """Preprocess the dataset and add the necessary columns and contexts."""
    num_workers = cpu_count()

    # Load data
    df = pd.read_json(file_path, lines=True)
    
    # Create 'answer' column and rename 'context' to 'relevant_context'
    df['answer'] = df['targets']
    df['relevant_context'] = df['context']
    
    # Drop unnecessary columns
    df.drop(columns=['targets', 'context', 'paragraphs'], inplace=True)
    
    # Replace empty answer (['']) with ['unanswerable']
    df['answer'] = df['answer'].apply(lambda x: ['unanswerable'] if x == [''] else x)

    # Add split column based on the file name
    split_name = file_path.split('/')[-1].split('.json')[0].replace('.', '_')

    df['split'] = split_name
    
    # Generate random contexts
    generate_random_context_TimeQA(df) 

    # Generate wrong date contexts
    generate_wd_context(df, num_workers)

    assign_mixed_context(df)

    return df

def main():
    # Define file paths for datasets
    datasets = {
        'train_easy': './data/datasets/TimeQA/easy/train.easy.json',
        'train_hard': './data/datasets/TimeQA/hard/train.hard.json',
        'dev_easy': './data/datasets/TimeQA/easy/dev.easy.json',
        'dev_hard': './data/datasets/TimeQA/hard/dev.hard.json',
        'test_easy': './data/datasets/TimeQA/easy/test.easy.json',
        'test_hard': './data/datasets/TimeQA/hard/test.hard.json'
    }

    # Define output directory
    easy_output_dir = './data/datasets/TimeQAEasy/final'
    hard_output_dir = './data/datasets/TimeQAHard/final'
    for output_dir in [easy_output_dir, hard_output_dir]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Loop through each dataset, preprocess, and save
    for dataset_name, file_path in datasets.items():
        print(f'Processing {dataset_name}...')
        data = preprocess_data(file_path)

        # Define output path for saving
        if 'easy' in dataset_name:
            output_path = os.path.join(easy_output_dir, f'{dataset_name}.jsonl')

        elif 'hard' in dataset_name:
            output_path = os.path.join(hard_output_dir, f'{dataset_name}.jsonl')

        data.to_json(output_path, orient='records', lines=True)
        print(f'{dataset_name} saved to {output_path}.')

if __name__ == '__main__':
    main()
