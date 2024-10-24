"""
Created on 09/15/2024

@author: Dan Schumacher
How to run this script:
python ./src/preprocess_data/format_l2l3_tempReason.py
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import cpu_count
import pandas as pd
from utils.format_data_utils import generate_wd_context, temp_reason_basic_processing, generate_random_context_TimeQA, assign_mixed_context

# =============================================================================
# FUNCTIONS
# =============================================================================
def process_dataset(df: pd.DataFrame, num_workers: int) -> pd.DataFrame:
    """Process a single dataset by generating contexts and extracting relevant fields."""
    df = temp_reason_basic_processing(df)
    df['no_context'] = ''
    df = generate_random_context_TimeQA(df)
    generate_wd_context(df, num_workers)
    assign_mixed_context(df)
    return df

def load_process_and_save_dataset(dataset_name: str, path: str, output_dir: str, num_workers: int):
    """Load, process, and save a dataset one by one."""
    print(f'Loading {dataset_name}...')
    dataset =  pd.read_json(path, lines=True)
    print(f'Processing {dataset_name}...')
    dataset = process_dataset(dataset, num_workers)
    output_path = os.path.join(output_dir, f'{dataset_name}.jsonl')
    dataset.to_json(output_path, orient='records', lines=True)
    print(f'{dataset_name} saved successfully.')

# =============================================================================
# MAIN FUNCTION
# =============================================================================

if __name__ == "__main__":
    paths = {
        'train_l2': './data/datasets/TempReason/original/train_l2.json',
        'train_l3': './data/datasets/TempReason/original/train_l3.json',
        'dev_l2': './data/datasets/TempReason/original/val_l2.json',
        'dev_l3': './data/datasets/TempReason/original/val_l3.json',
        'test_l2': './data/datasets/TempReason/original/test_l2.json',
        'test_l3': './data/datasets/TempReason/original/test_l3.json'
    }
    l2_output_dir = './data/datasets/TR_l2/final'
    l3_output_dir = './data/datasets/TR_l3/final'

    for path in [l2_output_dir, l3_output_dir]:
        os.makedirs(path, exist_ok=True)

    num_workers = cpu_count()

    # Load, process, and save datasets one at a time
    for dataset_name, path in paths.items():
        if 'l2' in dataset_name:
            output_dir = l2_output_dir
        else:
            output_dir = l3_output_dir
        dataset_name = dataset_name.split('_')[0]
        load_process_and_save_dataset(dataset_name, path, output_dir, num_workers)