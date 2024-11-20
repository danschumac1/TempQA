"""
Created on 11/09/2024

@author: Dan Schumacher
How to use:
python ./src/preprocess_data/combine_scopeExpand_and_scope.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

def main():
    # read in the two dataframes
    scope = pd.read_json('./data/datasets/MenatQA/final/scope_test.jsonl', lines=True)
    scope_expand = pd.read_json('./data/datasets/MenatQA/final/scope_expand_test.jsonl', lines=True)
    # combine the two dataframes
    combined = pd.concat([scope, scope_expand], ignore_index=True)
    # shuffle
    combined = combined.sample(frac=1).reset_index(drop=True)
    # export
    combined.to_json('./data/datasets/MenatQA/final/SCOPE_test.jsonl', orient='records', lines=True)

if __name__ == "__main__":
    main()