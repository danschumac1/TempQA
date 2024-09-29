import pandas as pd

def reshape_context(df: pd.DataFrame, col_list: list[str]) -> pd.DataFrame:
    mixed_df = pd.DataFrame()

    for col in col_list:
        new_df = df[['question','answer', col]].copy()
        new_df.rename(columns={col: 'mixed_context'}, inplace=True)
        mixed_df = pd.concat([mixed_df, new_df], ignore_index=True)
    
    return mixed_df

def main():
    # Load datasets
    train_path = "./data/datasets/MenatQA/final/train.jsonl"
    dev_path = "./data/datasets/MenatQA/final/dev.jsonl"
    
    # Load JSON Lines files
    train = pd.read_json(train_path, lines=True)
    dev = pd.read_json(dev_path, lines=True)
    
    train['no_context'] = ''
    dev['no_context'] = ''

    # Reshape the context for both train and dev datasets
    col_list = ['relevant_context', 'random_context', 'wrong_date_context', 'no_context']
    train_mixed = reshape_context(train, col_list)
    dev_mixed = reshape_context(dev, col_list)

    # Save the new datasets
    train_mixed.to_json("./data/datasets/MenatQA/final/train_mixed_stacked.jsonl", lines=True, orient='records')
    dev_mixed.to_json("./data/datasets/MenatQA/final/dev_mixed_stacked.jsonl", lines=True, orient='records')

    print("Mixed context datasets created and saved.")

if __name__ == "__main__":
    main()
