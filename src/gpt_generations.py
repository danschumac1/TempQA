"""
Created on Mon 11 25 2024

@author:
Dan Schumacher

TO RUN:
python ./src/gpt_generations.py \
    --dataset 'MenatQA' \
    --test_file 'counterfactual_test.jsonl' \
    --eval_context 'relevant_context'
"""

# IMPORTS
from utils.api_tools import api_config
from utils.logging_utils import gen_logger
from utils.gen_utils import get_format_function
import os
import pandas as pd
import json
import argparse
import sys


def parse_args():
    """Argument parsing function"""
    parser = argparse.ArgumentParser(description="GPT-4o Generations Script")
    parser.add_argument('--dataset', type=str, required=True, help='Which dataset to evaluate on?')
    parser.add_argument('--test_file', type=str, required=True, help='What is the test file name?')
    parser.add_argument(
        '--eval_context', type=str, required=True, choices=['no_context', 'random_context', 'relevant_context', 
        'wrong_date_context', 'mixed_context'], help='Select context to evaluate')
    return parser.parse_args()


def load_data(file_path):
    """Load JSONL data as a Pandas DataFrame"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist!")
    return pd.read_json(file_path, lines=True)


def main():
    # Setup logging
    args = parse_args()

    log_file = './logs/gpt_generations.log'
    gen_logger(init=True, log_file=log_file)
    gen_logger(message="GPT-4o Generations Script Initialized", log_file=log_file)

    # Set up output file
    pre_path = os.path.join(".", "data", "generations", "gpt", args.dataset, "base_trained")
    os.makedirs(pre_path, exist_ok=True)
    output_path = os.path.join(pre_path, f'{args.eval_context}_evaluated.jsonl')

    # Load the dataset
    file_path = os.path.join(".", "data", "datasets", args.dataset, "final", args.test_file)
    test_data = load_data(file_path)
    gen_logger(message=f"Loaded test data from {file_path}", log_file=log_file)

    # Set up the format function for prompts
    try:
        format_function = get_format_function('gpt')
        system, prompts = format_function(test_data, context_type=args.eval_context)
    except Exception as e:
        gen_logger(message=f"Error formatting prompts: {e}", log_file=log_file)
        sys.exit(1)

    # Initialize OpenAI GPT client
    client = api_config()

    # Prepare output file
    with open(output_path, 'w') as output_file:
        for idx, prompt in enumerate(prompts):
            if idx % 100 == 0:
                gen_logger(message=f"Generating response for index {idx}", log_file=log_file)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=150,
                    top_p=1
                )
                generated_output = response.choices[0].message.content.strip()
            except Exception as e:
                gen_logger(message=f"Error generating response for index {idx}: {e}", log_file=log_file)
                generated_output = ""

            # Write to output file
            output_file.write(json.dumps({'INDEX': idx, 'OUTPUT': generated_output}) + '\n')
            output_file.flush()
    gen_logger(message=f"All generations completed. Outputs saved to {output_path}", log_file=log_file)


if __name__ == "__main__":
    main()