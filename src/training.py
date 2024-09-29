"""
Created on 05/14/2024

@author: Dan Schumacher

ABOUT:
This code takes a GEMMA model and a dataset, processes the data, fine-tunes, and saves a model for evaluation.

HOW TO USE:
python src/training.py \
    --dataset_folder './data/datasets/dummy' \
    --train_file dummy_train.jsonl \
    --dev_file dummy_dev.jsonl \
    --save_path './models/dummy' \
    --formatter NIT \
    --training_context mixed_context \
    --batch_size 32 \
    --base_model gemma-2b \
    --lr 2e-5 \
    --epochs 1
"""
# /home/dan/TemporalQA/data/datasets/dummy/dummy_train.jsonl
#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================
# LOCAL LIBRARIES
from utils.training_utils import make_tokenized_prompt_column, selection_formatter
from utils.logging_utils import setup_logging
from utils.gen_utils import assign_mixed_context

# STANDARD LIBARIES
import os
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
import transformers
import argparse
import wandb 
import warnings

def parse_args():
    # REQUIRED ARGUMENTS
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")
    parser.add_argument('--dataset_folder', type=str, required=True, help='Where do the train/dev files live?')
    parser.add_argument('--train_file', type=str, required=True, help='What is the train file name?')
    parser.add_argument('--dev_file', type=str, required=True, help='What is the dev file name?')
    parser.add_argument('--base_model', type = str, required=True, choices= ['gemma-2b','gemma-1.1-2b-it','gemma-7b','gemma-1.1-7b-it'])
    parser.add_argument('--formatter', type=str, required=True, choices=['NIT', 'IT'], help='Format for IT or NIT?')
    parser.add_argument('--training_context', type=str, required=True, choices= ['no_context','random_context','relevant_context','wrong_date_context', 'mixed_context'], help='Model context')
    parser.add_argument('--save_path', type = str, required=True, help = 'Where to save the model?')

    # OPTIONAL ARGUMENTS
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='How many pass at model at once?')
    parser.add_argument('--lr', type = float, required=False, default=2e-5, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, required=False, default=6, help = 'How many training epochs?')
    return parser.parse_args()

def main():
    logger = setup_logging()  # Initialize logger
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
    
    # PARSE ARGUMENT
    args = parse_args()

    # Initialize wandb
    logger.info('WANDB initialized')
    wandb.init(project="temporal_understanding", config=args)
    
    logger.info('Logging into Hugging Face Hub')
    # LOG INTO HUGGING FACE
    with open('./resources/token.txt', 'r') as file:
        token = file.read().strip()
    login(token=token)

    bnb_config = BitsAndBytesConfig(load_in_8bit=False)  # Example setting, adjust as needed

    # LOAD MODEL, TOKENIZER, AND DATASET
    logger.info('Loading model')
    model_id = f"google/{args.base_model}"
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, use_cache=False)

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=False, add_eos_token=False, truncation=True, max_length=512)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    logger.info('Loading data')
    train = pd.read_json(f'{args.dataset_folder}/{args.train_file}', lines=True)
    dev = pd.read_json(f'{args.dataset_folder}/{args.dev_file}', lines=True)
    logger.info('Data loaded successfully')

    format_instruction = selection_formatter(args.formatter)

    # CREATE PROMPTS
    train[args.training_context] = format_instruction(train, context_type=args.training_context)
    dev[args.training_context] = format_instruction(dev, context_type=args.training_context)

    train_Dataset = make_tokenized_prompt_column(train, tokenizer, args.training_context)
    dev_Dataset = make_tokenized_prompt_column(dev, tokenizer, args.training_context)


    # ADVANCED FORMATTING WITH LORA
    logger.info('Preparing model with LoRA')
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    logger.info('Model prepared for training with LoRA')

# TODO train for X epoch, after each epoch, eval (F1 / contains Acc) on dev set, if better than best, save model

    # SET UP TRAINER
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_Dataset,
        eval_dataset=dev_Dataset,
        dataset_text_field="prompt",
        peft_config=lora_config,
        packing=True,
        args=transformers.TrainingArguments(
            run_name=f'{args.base_model}-{args.training_context}-trained',
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=8,
            warmup_steps=1,  # Example setting, adjust as needed
            gradient_checkpointing=True,
            fp16=True,
            optim="paged_adamw_8bit",
            learning_rate=args.lr,
            logging_steps=25,
            eval_steps=500,
            output_dir="outputs",
            save_strategy="no", # TODO Mayber here? 
            evaluation_strategy='steps',
            report_to="wandb"
        ),
        max_seq_length=512,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    logger.info('Trainer set up successfully')

    # TRAINING PROCESS
    trainer.train()
    logger.info("Training complete")

    # Save the model
    os.makedirs(args.save_path, exist_ok=True)
    trainer.model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    logger.info(f"Model and tokenizer saved at {args.save_path}")

if __name__ == '__main__':
    main()
