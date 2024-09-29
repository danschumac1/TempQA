"""
Created on 09/20/2024

@author: Dan Schumacher
How to use:
python src/preprocess_data/format_menatQA_split_test.py
"""
# LOCAL IMPORTS

# STANDARD LIBRARIES
import pandas as pd

def main():
    # load menatQA test data
    og_test_df = pd.read_json('./data/datasets/MenatQA/final/test.jsonl', lines=True)
    order_df = og_test_df[og_test_df['type'] == 'order']
    scope_expand_df = og_test_df[og_test_df['type'] == 'scope_expand']
    counterfactual_df = og_test_df[og_test_df['type'] == 'counterfactual']
    scope_df = og_test_df[og_test_df['type'] == 'scope']
    
    # export
    order_df.to_json('./data/datasets/MenatQA/final/order_test.jsonl', orient='records', lines=True)
    scope_expand_df.to_json('./data/datasets/MenatQA/final/scope_expand_test.jsonl', orient='records', lines=True)
    counterfactual_df.to_json('./data/datasets/MenatQA/final/counterfactual_test.jsonl', orient='records', lines=True)
    scope_df.to_json('./data/datasets/MenatQA/final/scope_test.jsonl', orient='records', lines=True)

if __name__ == "__main__":
    main()