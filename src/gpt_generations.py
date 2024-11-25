"""
Created on Mon 11 25 2024

@author: Dan Schumacher

TO RUN:
python ./src/gpt_generations.py \
    --dataset_folder './data/datasets/MenatQA/final' \
    --test_file 'test.jsonl' \
    --eval_context 'relevant_context'
"""

# IMPORTS
import os
from utils.api_tools import api_config
from utils.logging_utils import setup_logging
from utils.gen_utils import get_format_function
import pandas as pd
import json
import argparse

def parse_args():
    """Argument parsing function"""
    parser = argparse.ArgumentParser(description="GPT-4o Generations Script")
    parser.add_argument('--dataset_folder', type=str, required=True, help='Where do the train/dev files live?')
    parser.add_argument('--test_file', type=str, required=True, help='What is the test file name?')
    parser.add_argument(
        '--eval_context', type=str, required=True, choices=['no_context', 'random_context', 'relevant_context', 
        'wrong_date_context', 'mixed_context'], help='Select context to evaluate')
    parser.add_argument(
        '--output_file', type=str, required=False, default='gpt_outputs.jsonl', help='Output file for GPT generations')
    return parser.parse_args()


def load_data(file_path):
    """Load JSONL data as a Pandas DataFrame"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist!")
    return pd.read_json(file_path, lines=True)


def main():
    # Setup logging
    logger = setup_logging("gpt_generation_logger")
    logger.info("Logging setup complete.")

    # Parse arguments
    args = parse_args()

    # Load the dataset
    file_path = f'{args.dataset_folder}/{args.test_file}'
    test_data = load_data(file_path)
    logger.info(f"Loaded dataset from {file_path}")

    # Set up the format function for prompts
    format_function = get_format_function('gpt')  # Assumes 'gpt' is a valid key for format function
    test_data[args.eval_context] = format_function(test_data, context_type=args.eval_context)

    # Initialize OpenAI GPT client
    client = api_config()

    # Prepare output file
    output_path = args.output_file
    with open(output_path, 'w') as output_file:
        # Loop through data and generate responses
        for idx, row in test_data.iterrows():
            prompt = row[args.eval_context]

            # Prepare API request
            logger.info(f"Generating response for index {idx}")
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant providing answers for temporal reasoning tasks."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=150,
                    top_p=1
                )
                generated_output = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error generating response for index {idx}: {e}")
                generated_output = ""

            # Write to output file
            output_file.write(json.dumps({'INDEX': idx, 'OUTPUT': generated_output}) + '\n')
            output_file.flush()
            logger.info(f"Saved response for index {idx}")

    logger.info(f"All generations completed. Outputs saved to {output_path}")


if __name__ == "__main__":
    main()
