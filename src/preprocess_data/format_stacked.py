import pandas as pd
import argparse
"""
Created on 11/08/2024

@author: Dan Schumacher
How to use:
python ./src/preprocess_data/format_stacked.py --dataset TimeQAEasy --train_file train_easy.jsonl --dev_file dev_easy.jsonl
python ./src/preprocess_data/format_stacked.py --dataset TimeQAHard --train_file train_hard.jsonl --dev_file dev_hard.jsonl
python ./src/preprocess_data/format_stacked.py --dataset TR_l2 --train_file train.jsonl --dev_file dev.jsonl
python ./src/preprocess_data/format_stacked.py --dataset TR_l3 --train_file train.jsonl --dev_file dev.jsonl
python ./src/preprocess_data/format_stacked.py --dataset MenatQA --train_file train.jsonl --dev_file dev.jsonl
python ./src/preprocess_data/format_stacked.py --dataset AQA --train_file train.jsonl --dev_file dev.jsonl
python ./src/preprocess_data/format_stacked.py --dataset TQE --train_file train.jsonl --dev_file dev.jsonl
"""

def parse_args():
    """Argument parsing function"""
    # REQUIRED ARGUMENTS
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Stacking Script")
    parser.add_argument('--dataset', type=str, required=True, help='Where do the train/dev files live?')
    parser.add_argument('--train_file', type=str, required=True, help='What is the train file name?')
    parser.add_argument('--dev_file', type=str, required=True, help='What is the dev file name?')
    return parser.parse_args()

def reshape_context(df: pd.DataFrame, col_list: list[str]) -> pd.DataFrame:
    
    mixed_df = pd.DataFrame()

    for col in col_list:
        new_df = df[['question','answers', col]].copy()
        new_df.rename(columns={col: 'mixed_context'}, inplace=True)
        mixed_df = pd.concat([mixed_df, new_df], ignore_index=True)
    
    return mixed_df

def main():
    # Load datasets
    args = parse_args()
    train_path = f"./data/datasets/{args.dataset}/final/{args.train_file}"
    dev_path = f"./data/datasets/{args.dataset}/final/{args.dev_file}"
    
    # Load JSON Lines files
    train = pd.read_json(train_path, lines=True)
    dev = pd.read_json(dev_path, lines=True)
    
    # this may be redundant but to be safe...
    train['no_context'] = ''
    dev['no_context'] = ''

    # Reshape the context for both train and dev datasets
    col_list = ['relevant_context', 'random_context', 'wrong_date_context', 'no_context']
    train_mixed = reshape_context(train, col_list)
    dev_mixed = reshape_context(dev, col_list)

    # Save the new datasets
    train_mixed.to_json(f"./data/datasets/{args.dataset}/final/train_mixed_stacked.jsonl", lines=True, orient='records')
    dev_mixed.to_json(f"./data/datasets/{args.dataset}/final/dev_mixed_stacked.jsonl", lines=True, orient='records')

    print("Mixed context datasets created and saved.")

if __name__ == "__main__":
    main()
