# COMBINED GENERATION AND EVALUATION SCRIPT
# Created on 11/01/2024
# @author: Dan Schumacher

import os
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.training_utils import load_config
from utils.gen_utils import map_tokens_over_data, get_format_function
from utils.eval_utils import extract_generations, extract_actual_answers, calc_contains_acc, calc_f1
from utils.logging_utils import gen_logger
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser(description='Combine generation and evaluation.')
    # Required arguments
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--test_file', type=str, required=True, help='Name of the test file')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--eval_context', type=str, required=True, choices=['no_context','random_context','relevant_context','wrong_date_context','mixed_context'], help='Evaluation context')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model', type=str, required=True, choices=['gemma','mistral','llama'], help='Model type')
    parser.add_argument('--config_type', type=str, required=True, help='Type of generation config')
    parser.add_argument('--splitter', type=str, default='\nmodel\n', help='Splitter for generation output')
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
    parser.add_argument('--num_rows', type=int, default=10, help='Number of rows to process')
    parser.add_argument('--answer_key', type=str, default='answers', help='Key for actual answers in data')
    parser.add_argument('--key_name', type=str, default='OUTPUT', help='Key for extracting generated outputs')

    return parser.parse_args()

def main():
    # Argument parsing
    args = parse_args()
    gen_logger(init=True)

    # Login to Hugging Face Hub
    gen_logger('loading token.txt')
    with open('./resources/token.txt', 'r') as file:
        token = file.read().strip()
    login(token=token)

    # Load dataset
    file_path = f"{args.dataset_folder}/{args.test_file}"
    if not os.path.exists(file_path):
        gen_logger(log_type="ERROR", message=f"File {file_path} does not exist!", )
        exit(1)
    
    test_data = pd.read_json(file_path, lines=True).iloc[:args.num_rows]
    gen_logger("test_data loaded")

    # Set up prompt formatting function
    format_function = get_format_function(args.model)
    gen_logger("formatted_instructions")
    test_data[args.eval_context] = format_function(test_data, context_type=args.eval_context)

    gen_logger('INFO', f"Loading model from {args.model_path}")

    base_model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model = PeftModel.from_pretrained(base_model, args.model_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_bos_token=True, add_eos_token=False)
    gen_logger("INFO", f"Loading Tokenizer from {args.model_path}")

    # Initialize tokenizer and model
    gen_logger("tokenizing data")
    test_dataset = map_tokens_over_data(test_data, tokenizer, args.eval_context)

    inputs = tokenizer(list(test_dataset[args.eval_context]), return_tensors="pt", max_length=1800, padding=True, truncation=True)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    loader = DataLoader(dataset, batch_size=args.batch_size)

    generation_params = load_config('./resources/generator_config.json', args.config_type)
    generated_outputs = []

    gen_logger("beginning generations")
    for i, batch in enumerate(loader):
        input_ids, attention_mask = [b.to('cuda') for b in batch]
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_params
        )
        decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for j, item in enumerate(decoded_responses):
            idx = i * args.batch_size + j
            generated_outputs.append({'INDEX': idx, args.key_name: item})

        gen_logger(f"Processed batch {i+1}/{len(loader)}")
        del input_ids, attention_mask, generated_ids
        torch.cuda.empty_cache()

    # Save generated responses
    output_path = './data/temp/temp_generated_responses.jsonl'
    with open(output_path, 'w') as f:
        for item in generated_outputs:
            f.write(json.dumps(item) + '\n')
    gen_logger(f"Generated responses saved to {output_path}")
    
    # Load actual answers and evaluate
    gen_logger("beginning eval")
    actual_answers = extract_actual_answers(test_data, answer_key=args.answer_key)
    gen_list = extract_generations(generated_outputs, splitter=args.splitter)

    f1_scores, acc_scores = [], []
    for pred, actual in zip(gen_list, actual_answers):
        acc = calc_contains_acc(pred, actual)
        f1 = calc_f1(pred, actual)

        acc_scores.append(acc)
        f1_scores.append(f1)

        print(json.dumps({"pred": pred, "actual":actual, "acc":acc, "f1":f1}), flush=True)
            
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0

    # Log and print evaluation results
    result = {
        'model': args.model,
        'dataset': args.dataset,
        'trained_on': args.model_path.split('/')[-1],
        'eval_context': args.eval_context,
        'f1': avg_f1,
        'accuracy': avg_acc,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'gen_params': generation_params
    }
    gen_logger(f"Evaluation Results: {result}")
    print(json.dumps(result, indent=8), flush=True)

if __name__ == "__main__":
    main()