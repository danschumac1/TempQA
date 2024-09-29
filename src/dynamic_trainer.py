"""
Created on 09/19/2024

@author: Dan
How to use:
CUDA_VISIBLE_DEVICES=0 nohup python src/dynamic_trainer.py \
    --model_type gemma \
    --train_file_path './data/datasets/MenatQA/final/train.jsonl' \
    --dev_file_path './data/datasets/MenatQA/final/dev.jsonl' \
    --save_path './models/gemma/MenatQA/mixed_context_trained' \
    --training_context mixed_context \
    --batch_size 16 \
    --lr 2e-5 \
    --epochs 6 &
"""
# /home/dan/TemporalQA/data/datasets/dummy/dummy_train.jsonl

# =============================================================================
# IMPORTS AND SET UP
# =============================================================================
# LOCAL LIBRARIES
from utils.training_utils import make_tokenized_prompt_column, save_model_and_tokenizer, selection_formatter
from utils.logging_utils import setup_logging
from utils.training_utils import load_config, get_trainer  # Load your external config and trainer logic

# STANDARD LIBRARIES
import pandas as pd
import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import argparse
import wandb 
import warnings

def parse_args():
    # REQUIRED ARGUMENTS
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")
    parser.add_argument('--model_type', type=str, required=True, choices=['gemma', 'mistral', 'llama'], help="Which model to use: gemma, mistral, or llama?")
    parser.add_argument('--training_context', type=str, required=True, help='Model context', choices=['no_context', 'random_context', 'relevant_context', 'wrong_date_context', 'mixed_context'])
    parser.add_argument('--save_path', type = str, required=True, help = 'Where to save the model?')
    parser.add_argument('--train_file_path', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--dev_file_path', type=str, required=True, help='Path to the dev dataset')
    # OPTIONAL ARGUMENTS
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='How many pass at model at once?')
    parser.add_argument('--lr', type = float, required=False, default=2e-5, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, required=False, default=6, help = 'How many training epochs?')
    
    # args.epoch, args.lr, args.batch_size, args.save_path
    return parser.parse_args()
def main():
    logger = setup_logging("trainer_logger")  # Initialize logger
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()

    dataset = args.train_file_path.split('/')[-3]
    logger.info(f'Beginning training ||| training context {args.training_context}, model {args.model_type}, dataset {dataset}')
    
    # PRINT CUDA INFO
    logger.info(torch.cuda.is_available())  # This should print True if CUDA is available
    logger.info(torch.cuda.current_device())  # Check the current CUDA device
    logger.info(torch.cuda.get_device_name(0))  # Print the name of the GPU

    # PARSE ARGUMENT

    # Load the configuration from JSON file based on the selected model type
    config_path = './resources/trainer_config.json'
    config = load_config(config_path, args.model_type)

    # Initialize wandb
    logger.info('WANDB initialized')
    wandb.init(project="temporal_revamp", config=config)
    
    logger.info('Logging into Hugging Face Hub')
    # LOG INTO HUGGING FACE
    with open('./resources/token.txt', 'r') as file:
        token = file.read().strip()
    login(token=token)

    bnb_config = BitsAndBytesConfig(load_in_8bit=False)
    logger.info('Bits and Bytes Config set up')

    # LOAD MODEL, TOKENIZER, AND DATASET
    logger.info(f'Loading model: {config["base_model"]}')
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, use_cache=False)
    logger.info(f"model on device: {model.device}")
    # model.to("cuda")

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], add_bos_token=False, add_eos_token=False, truncation=True, max_length=512)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.get("padding_side", "right")

    train_df = pd.read_json(args.train_file_path, lines=True)
    dev_df = pd.read_json(args.dev_file_path, lines=True)
    logger.info('Data loaded successfully')

    format_instruction = selection_formatter(config["formatter"])
    logger.info('Instruction formatter set up')

    # CREATE PROMPTS
    train_df[args.training_context] = format_instruction(train_df, context_type=args.training_context)
    dev_df[args.training_context] = format_instruction(dev_df, context_type=args.training_context)
    logger.info('Prompts created successfully')

    train_Dataset = make_tokenized_prompt_column(train_df, tokenizer, args.training_context)
    dev_Dataset = make_tokenized_prompt_column(dev_df, tokenizer, args.training_context)
    logger.info('Datasets tokenized successfully')

    # SET UP TRAINER
    trainer = get_trainer(model, tokenizer, train_Dataset, dev_Dataset, config, args.epochs, args.lr, args.batch_size, args.save_path)
    logger.info('Trainer set up successfully')

    # TRAINING PROCESS
    logger.info("Training started")
    trainer.train()
    wandb.finish()
    logger.info("Training complete")

    # SAVE MODEL
    logger.info("Saving model")
    save_model_and_tokenizer(trainer, args.save_path)
    logger.info("Model saved successfully")
if __name__ == '__main__':
    main()