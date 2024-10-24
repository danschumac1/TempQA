"""
Created on 06/09/2024

@author: Dan Schumacher

HOW TO RUN
python ./src/generations.py \
    --dataset_folder './data/datasets/MenatQA/final' \
    --test_file 'test.jsonl' \
    --dataset MenatQA \
    --eval_context relevant_context \
    --model_path 'models/gemma/MenatQA/relevant_context_trained' \
    --model gemma> dummy_generations.jsonl
"""

# TODO CHECK THAT WORKS FOR NO CONTEXT!
# Local Libraries
from utils.training_utils import load_config
from utils.gen_utils import map_tokens_over_data, get_format_function
from utils.logging_utils import setup_logging

# Standard Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import login

import json
import argparse
import sys

def parse_args():
    """Argument parsing function"""
    # REQUIRED ARGUMENTS
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")
    parser.add_argument('--dataset_folder', type=str, required=True, help='Where do the train/dev files live?')
    parser.add_argument('--test_file', type=str, required=True, help='What is the train file name?')
    parser.add_argument('--dataset', type=str, required=True, choices=['dummy','AQA','TQE', 'MenatQA','TimeQAEasy', 'TimeQAHard','TempReason'], help='Select the dataset to use')
    parser.add_argument('--eval_context', type=str, required=True, choices=['no_context','random_context','relevant_context','wrong_date_context','mixed_context'], help='Select context to evaluate')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--model', type=str, required=True, choices=['gemma','mistral','llama'], help='What model (for grabbing generation configs)')

    # OPTIONAL ARGUMENTS
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='Batch size for model inference')
    parser.add_argument('--data_file', type=str, required=False, default='test.jsonl', help='Name of the data file to evaluate')
    return parser.parse_args()

def main(): 
    """Main function to run model generations"""
    # Setup Torch and clear cache
    torch.cuda.empty_cache()

    # Initialize logging from JSON configuration
    print("Starting logging setup...")
    logger = setup_logging("generation_logger")
    logger.info("Logging setup complete.")
    # Argument Parsing
    args = parse_args()
    
    # Login to Hugging Face Hub using token (token stored locally)
    with open('./resources/token.txt', 'r') as file:
        token = file.read().strip()
    login(token=token)
    
    # Check if file exists
    file_path = f'{args.dataset_folder}/{args.test_file}'
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist!")
        sys.exit(1)
    
    # Load the dataset from file
    test = pd.read_json(file_path, lines=True)
    
    # Set up the format function for prompts
    format_function = get_format_function(args.model)
    
    # Determine the type of prompt to format
    test[args.eval_context] = format_function(test, context_type=args.eval_context)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_bos_token=True, add_eos_token=False)

    test_Dataset = map_tokens_over_data(test, tokenizer, args.eval_context)

    # Tokenize the dataset
    logger.info(f"Tokenizing data using model: {args.model_path}")
    tokenizer.padding_side = 'left'

    # Tokenize the prompts in the test dataset
    inputs = tokenizer(list(test_Dataset[args.eval_context]), return_tensors="pt", max_length=1800, padding=True, truncation=True)
    
    # Create a TensorDataset and DataLoader for batch processing
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    loader = DataLoader(dataset, batch_size=args.batch_size)

    # Load the base model and adapter model
    model_path = f"{args.model_path}"
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    adapter_model = PeftModel.from_pretrained(base_model, model_path).to('cuda')
    logger.info(f"model on device: {adapter_model.device}")
    generation_params = load_config('./resources/generator_config.json', args.model)
    logger.info("Loaded generation parameters from configuration file.")

    # Generate outputs in batches
    for i, batch in enumerate(loader):
        logger.info(f"Processing batch {i+1}/{len(loader)}")
        input_ids, attention_mask = [b.to('cuda') for b in batch]

        # Use the parameters in the generate function
        generated_ids = adapter_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_params
        )
        logger.info(f"Generated output for batch {i+1}")

        # Decode responses and write them to stdout in JSONL format
        decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logger.info(f"Decoded responses for batch {i+1}")

        for j, item in enumerate(decoded_responses):
            idx = i * len(decoded_responses) + j
            sys.stdout.write(json.dumps({'INDEX': idx, 'OUTPUT': item}) + "\n")
            sys.stdout.flush()
        
        logger.info(f"Written responses for batch {i+1} to stdout")

        # Clear memory after each batch
        del input_ids, attention_mask, generated_ids, decoded_responses
        torch.cuda.empty_cache()
        logger.info(f"Cleared memory for batch {i+1}")

if __name__ == '__main__':
    main()