"""
Created on 09/14/2024

@author: Dan Schumacher

ABOUT eval.py:
    This script is used to evaluate the generated answers from a model. It takes in the path to the generated answers \
    and the path to the actual answers. It then calculates the F1 score and the contains accuracy for each generated answer. \
    It calculates the average F1 score and the average contains accuracy for all the generated answers, \
    then saves the results to a jsonl file.

HOW TO RUN:

python ./src/eval.py \
    --gen_path "./data/generations/gemma/MenatQA/relevant_context_trained/scope_test_relevant_context_evaluated.jsonl" \
    --actual_path "./data/datasets/MenatQA/final/scope_test.jsonl" \
    --model "gemma" \
    --trained_on "relevant_context" \
    --dataset "MenatQA" \
    --eval_test_set "scope_test" \
    --eval_context "relevant_context" > test1.jsonl
"""

# LOCAL IMPORTS
from datetime import datetime
import json
from utils.eval_utils import load_funky_json, extract_generations, extract_actual_answers, calc_contains_acc, calc_f1
import pandas as pd
import argparse
import os

from utils.logging_utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate generated answers.')
    # REQUIRED arguments
    parser.add_argument('--gen_path', type=str, help='Path to the generated answers file', required=True)
    parser.add_argument('--actual_path', type=str, help='Path to the actual answers file', required=True)
    parser.add_argument('--dataset', type=str, help='Dataset used for evaluation', required=True)
    parser.add_argument('--model', type=str, help='Model name', required=True)
    parser.add_argument('--trained_on', type=str, help='Dataset used for training', required=True)
    parser.add_argument('--eval_test_set', type=str, help='Test-set used for evaluation', required=True)
    parser.add_argument('--eval_context', type=str, help='Context used for evaluation', required=True)

    # OPTIONAL arguments
    parser.add_argument('--answer_key', type=str, help='Key to extract the answer from the data dictionaries', default='answer')
    parser.add_argument('--question_key', type=str, help='Key to extract the question from the data dictionaries', default='question')
    parser.add_argument('--key_name', type=str, default='OUTPUT', help='Key name to extract the generations', required=False)

    return parser.parse_args()

def main():
    logger = setup_logging("evaluation_logger")  # Initialize logger

  # Argument Parsing
    args = parse_args()
    # Debugging print
    logger.info(f"Model: {args.model}, Trained on: {args.trained_on}, Eval testset: {args.eval_test_set}, eval Context: {args.eval_context}")

    # Ensure output directory exists
    # output_dir = os.path.dirname(args.output_path)
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # Load actual answers
    actual_df = pd.read_json(args.actual_path, lines=True)
    # Load generated answers
    gen_dict = load_funky_json(args.gen_path)

    gen_list = extract_generations(gen_dict)

    

    # Extract actual answers
    actual_answers_list = extract_actual_answers(actual_df, answer_key=args.answer_key)

    # Calculate metrics
    f1s = []
    contains_acc = []
    for pred, ans_list in zip(gen_list, actual_answers_list):
        # print nicely
        print(f"Prediction: {pred}", f"\nActual: {ans_list}", "\n\n")
        f1s.append(calc_f1(pred, ans_list))
        contains_acc.append(calc_contains_acc(pred, ans_list))
    avg_contains = 'devide by zero'
    avg_f1 = 'devide by zero'
    if len(f1s) != 0:
        avg_f1 = sum(f1s) / len(f1s)
    if len(contains_acc) != 0:
        avg_contains = sum(contains_acc) / len(contains_acc)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare output
    output_dict = {
        'model': args.model,
        'dataset': args.dataset,
        'trained_on': args.trained_on,
        'eval_test_set': args.eval_test_set,
        'eval_context': args.eval_context,
        'f1': avg_f1,
        'acc': avg_contains,
        'timestamp': current_time
    }
    print('butts')
    print(json.dumps(output_dict))

if __name__ == "__main__":
    main()
  