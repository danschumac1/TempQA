
"""
Created on 09/15/2024

@author: Dan Schumacher
How to use:
    python ./src/format_AQA.py
"""
from utils.format_data_utils import assign_mixed_context

import pandas as pd


def main():
    paths = {
    'train': './data/datasets/AQA/original/train.jsonl',
    'dev': './data/datasets/AQA/original/dev.jsonl',
    'test': './data/datasets/AQA/original/test.jsonl'
    }
    for path in paths.values():
        df = pd.read_json(path, lines=True)
        assign_mixed_context(df)
        df.to_json(path.replace('original', 'final'), lines=True, orient='records')

if __name__ == "__main__":
    main()
