"""
Created on 11/02/2024

@author: Dan Schumacher
How to use:
python sketch_pad.py
"""

import json
import os
import pandas as pd

def main():
    folder_path = './data/datasets'
    results = {'answer': '', 'max_length': 0}

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Skip if not a directory
        
        for final_folder in os.listdir(subfolder_path):
            final_folder_path = os.path.join(subfolder_path, final_folder)
            if not os.path.isdir(final_folder_path):
                continue  # Skip if not a directory
            
            for file in os.listdir(final_folder_path):
                if file.endswith('.jsonl'):
                    file_path = os.path.join(final_folder_path, file)
                    data = pd.read_json(file_path, lines=True)
                    if 'answer' in data.columns:
                        print(f"{subfolder}, {file}, : 'answer': {type(data['answer'].iloc[0])}")
                    elif 'answers' in data.columns:
                        print(f"{subfolder}, {file}, : 'answers': {type(data['answers'].iloc[0])}")
                    else:
                        print('butts')
                
def main2():
    '''take see the unique columns for all files'''
    folder_path = './data/datasets'
    unique_cols = set()
    for subfolder in os.listdir(folder_path):
        print(f'Processing {subfolder}')
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Skip if not a directory
        
        for final_folder in os.listdir(subfolder_path):
            final_folder_path = os.path.join(subfolder_path, final_folder)
            if not os.path.isdir(final_folder_path):
                continue  # Skip if not a directory
            
            for file in os.listdir(final_folder_path):
                if file.endswith('.jsonl'):
                    print(json.dumps(f'Processing {file}', indent=4))
                    file_path = os.path.join(final_folder_path, file)
                    data = pd.read_json(file_path, lines=True)
                    print(json.dumps(data.columns.tolist(), indent=8))
                    # print the data type of each column
                    for col in data.columns:
                        print(f'{col}: {data[col].dtype}')
                    unique_cols.update(data.columns)
    print(unique_cols)

if __name__ == '__main__':
    main()
