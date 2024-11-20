"""
Created on 11/02/2024

@author: Dan Schumacher
How to use:
python ./src/sketch_pad.py
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    folder_path = './data/datasets'
    results = {'answer': '', 'max_length': 0, 'file': '', 'subfolder': '', 'final_folder': ''}

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Skip if not a directory
        if subfolder != 'MenatQA':
            continue
        
        for final_folder in os.listdir(subfolder_path):
            final_folder_path = os.path.join(subfolder_path, final_folder)
            if not os.path.isdir(final_folder_path):
                continue  # Skip if not a directory
            
            for file in os.listdir(final_folder_path):
                if file.endswith('.jsonl'):
                    file_path = os.path.join(final_folder_path, file)
                    try:
                        data = pd.read_json(file_path, lines=True)
                        col = 'answer' if 'answer' in data.columns else 'answers'
                        for _, row in data.iterrows():
                            if isinstance(row[col], list):  # Ensure it's a list
                                for answer in row[col]:
                                    if isinstance(answer, str) and len(answer) > results['max_length']:
                                        results['answer'] = answer
                                        results['max_length'] = len(answer)
                                        results['file'] = file
                                        results['subfolder'] = subfolder
                                        results['final_folder'] = final_folder
                    except Exception as e:
                        print(f'Error processing {file_path}: {e}')

    print(results)

def main2():
    print(len("'Minister Without Portfolio Responsible for the Planning, Building and Putting into Operation of the Two New Blocks of the Paks Nuclear Plant".split()))

def main3():
    folder_path = './data/datasets'
    total_word_count = 0
    total_answers = 0

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Skip if not a directory
        if subfolder != 'MenatQA':
            continue

        for final_folder in os.listdir(subfolder_path):
            final_folder_path = os.path.join(subfolder_path, final_folder)
            if not os.path.isdir(final_folder_path):
                continue  # Skip if not a directory

            for file in os.listdir(final_folder_path):
                if file.endswith('.jsonl'):
                    file_path = os.path.join(final_folder_path, file)
                    try:
                        data = pd.read_json(file_path, lines=True)
                        col = 'answer' if 'answer' in data.columns else 'answers'
                        for _, row in data.iterrows():
                            if isinstance(row[col], list):  # Ensure it's a list
                                for answer in row[col]:
                                    if isinstance(answer, str):
                                        word_count = len(answer.split())
                                        total_word_count += word_count
                                        total_answers += 1
                    except Exception as e:
                        print(f'Error processing {file_path}: {e}')

    if total_answers > 0:
        avg_num_words = total_word_count / total_answers
        print(f'Average number of words: {avg_num_words:.2f}')
    else:
        print('No answers found to calculate the average.')


def main4():
    folder_path = './data/datasets'
    word_counts = []  # List to store the word count for each answer

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Skip if not a directory
        if subfolder != 'MenatQA':
            continue

        for final_folder in os.listdir(subfolder_path):
            final_folder_path = os.path.join(subfolder_path, final_folder)
            if not os.path.isdir(final_folder_path):
                continue  # Skip if not a directory

            for file in os.listdir(final_folder_path):
                if file.endswith('.jsonl'):
                    file_path = os.path.join(final_folder_path, file)
                    try:
                        data = pd.read_json(file_path, lines=True)
                        col = 'answer' if 'answer' in data.columns else 'answers'
                        for _, row in data.iterrows():
                            if isinstance(row[col], list):  # Ensure it's a list
                                for answer in row[col]:
                                    if isinstance(answer, str):
                                        word_count = len(answer.split())
                                        word_counts.append(word_count)
                    except Exception as e:
                        print(f'Error processing {file_path}: {e}')

    # Plotting the distribution
    if word_counts:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(word_counts, kde=True, bins=30)
        plt.title('Distribution of Number of Words in Answers')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')

        # Save the plot
        os.makedirs('./figures', exist_ok=True)
        plt.savefig('./figures/word_count_distribution.png')
        plt.close()
        print('Distribution plot saved to ./figures/word_count_distribution.png')
    else:
        print('No word counts available to plot.')


if __name__ == '__main__':
    main4()
