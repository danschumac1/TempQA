"""
Created on 09/14/2024

@author: Dan Schumacher
HOW TO RUN:
python ./src/eval.py \
--gen_path "dummy_generations.jsonl" \
--actual_path "./data/datasets/dummy/dummy_test.jsonl" \
--output_path "dummy_eval.jsonl" \
--answer_key "answer" \
--question_key "question" \
--model "dummy" \
--trained_on "mixed_context" \
--eval_on "wrong_date_context" \
--key_name "key_name_to_extract_generations"
"""

# LOCAL IMPORTS
from datetime import datetime
import json
from utils.eval_utils import load_funky_json, extract_generations, extract_actual_answers, calc_contains_acc, calc_f1
import pandas as pd
import argparse
import os

from utils.logging_utils import gen_logger
from utils.training_utils import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate generated answers.')
    # REQUIRED arguments
    parser.add_argument('--gen_path', type=str, help='Path to the generated answers file', required=True)
    parser.add_argument('--actual_path', type=str, help='Path to the actual answers file', required=True)
    parser.add_argument('--dataset', type=str, help='Dataset used for evaluation', required=True)
    parser.add_argument('--model', type=str, help='Model name', required=True)
    parser.add_argument('--trained_on', type=str, help='Dataset used for training', required=True)
    parser.add_argument('--test_file', type=str, help='Test-set used for evaluation', required=True)
    parser.add_argument('--eval_context', type=str, help='Context used for evaluation', required=True)
    parser.add_argument('--config_type', type=str, help='Type of generation config', required=True)

    # OPTIONAL arguments
    parser.add_argument(
        '--splitter', type=str, help='Splitter to split the generations', default='\nmodel\n', required=False)
    parser.add_argument(
        '--answer_key', type=str, help='Key to extract the answer from the data dictionaries', default='answers')
    parser.add_argument(
        '--question_key', type=str, help='Key to extract the question from the data dictionaries', default='question')
    parser.add_argument(
        '--key_name', type=str, default='OUTPUT', help='Key name to extract the generations', required=False)

    return parser.parse_args()

def main():
    log_file = os.path.join(os.getcwd(), 'logs', 'eval.log')
    gen_logger(init=True, log_file=log_file)  # Initialize logger

  # Argument Parsing
    args = parse_args()

    generation_params = load_config('./resources/generator_config.json', args.config_type)

    # Debugging print
    gen_logger(
        message=f"Model: {args.model}, Trained on: {args.trained_on},"
        f"Eval testset: {args.test_file.split('_test')[0]}, eval Context: {args.eval_context}",
        log_file=log_file)

    # Load actual answers
    actual_df = pd.read_json(args.actual_path, lines=True)

    # Load generated answers
    gen_dict = load_funky_json(args.gen_path)
    gen_list = extract_generations(gen_dict, splitter=args.splitter)

    # Extract actual answers
    actual_answers_list = extract_actual_answers(actual_df, answer_key=args.answer_key)

    # Calculate metrics
    f1_scores, acc_scores = [], []
    for pred, actual_answers in zip(gen_list, actual_answers_list):
        acc = calc_contains_acc(pred, actual_answers)
        f1 = calc_f1(pred, actual_answers)

        acc_scores.append(acc)
        f1_scores.append(f1)

        # print(json.dumps({"pred": pred, "actual":actual_answers, "acc":acc, "f1":f1}), flush=True)
            
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0

    # Log and print evaluation results
    result = {

        'trained_on': args.trained_on,
        'eval_on': args.eval_context,
        'accuracy': avg_acc,
        'f1': avg_f1,
        'model': args.model,
        'dataset': args.dataset,
        'subset': args.test_file.split('_test')[0],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'gen_params': generation_params
    }
    gen_logger(f"Evaluation Results: {result}")
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()