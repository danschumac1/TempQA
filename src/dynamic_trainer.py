# =============================================================================
# DYNAMIC TRAINER
# =============================================================================
# LOCAL LIBRARIES
from utils.training_utils import make_tokenized_prompt_column, save_model_and_tokenizer, selection_formatter
from utils.logging_utils import gen_logger
from utils.training_utils import load_config, get_trainer  # Load your external config and trainer logic

# STANDARD LIBRARIES
import torch
import pandas as pd
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
    parser.add_argument('--gpu', type=int, required=True, help='Which GPU to use?')    
    parser.add_argument('--epochs', type=int, required=True, help='How many epochs to train for?')
    return parser.parse_args()

def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore")

    args = parse_args()
    log_file = f'./logs/dt_{args.model_type}_{args.training_context}_{args.epochs}-epochs.log'
    gen_logger(init=True, log_file=log_file)  # Initialize gen_logger
    gen_logger(
        f'Beginning training ||| training context: {args.training_context}, '
        f'model: {args.model_type}, dataset: {args.train_file_path.split("/")[-3]}', 
        log_file=log_file
    )
    
    # PRINT CUDA INFO
    gen_logger(torch.cuda.is_available(), log_file=log_file)  # This should print True if CUDA is available
    gen_logger(torch.cuda.current_device(), log_file=log_file)  # Check the current CUDA device
    gen_logger(torch.cuda.get_device_name(0), log_file=log_file)  # Print the name of the GPU

    # PARSE ARGUMENT

    # Load the configuration from JSON file based on the selected model type
    config_path = 'resources/trainer_config.json'
    config = load_config(config_path, args.model_type)
    config['trainer_config']['output_dir'] = args.save_path
    config['trainer_config']['num_train_epochs'] = args.epochs

    # Initialize wandb
    gen_logger('WANDB initialized', log_file=log_file)
    wandb.init(project="temporal_revamp", config=config)
    
    gen_logger('Logging into Hugging Face Hub', log_file=log_file)
    # LOG INTO HUGGING FACE
    with open('./resources/token.txt', 'r') as file:
        token = file.read().strip()
    login(token=token)

    bnb_config = BitsAndBytesConfig(load_in_8bit=False)
    gen_logger('Bits and Bytes Config set up', log_file=log_file)

    # LOAD MODEL, TOKENIZER, AND DATASET
    gen_logger(f'Loading model: {config["base_model"]}', log_file=log_file)
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, use_cache=False)
    gen_logger(f"model on device: {model.device}", log_file=log_file)
    # model.to("cuda")

    gen_logger('Loading tokenizer', log_file=log_file)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], add_bos_token=False, add_eos_token=False, truncation=True, max_length=512)
    tokenizer.padding_side = config.get("padding_side", "right")
    tokenizer.pad_token = tokenizer.eos_token

    train_df = pd.read_json(args.train_file_path, lines=True)
    dev_df = pd.read_json(args.dev_file_path, lines=True)
    gen_logger('Data loaded successfully', log_file=log_file)

    format_instruction = selection_formatter(config["formatter"])
    gen_logger('Instruction formatter set up', log_file=log_file)

    # CREATE PROMPTS
    train_df[args.training_context] = format_instruction(train_df, context_type=args.training_context)
    dev_df[args.training_context] = format_instruction(dev_df, context_type=args.training_context)
    gen_logger('Prompts created successfully', log_file=log_file)

    train_Dataset = make_tokenized_prompt_column(train_df, tokenizer, args.training_context)
    dev_Dataset = make_tokenized_prompt_column(dev_df, tokenizer, args.training_context)
    gen_logger('Datasets tokenized successfully', log_file=log_file)

    # SET UP TRAINER
    trainer = get_trainer(model, tokenizer, train_Dataset, dev_Dataset, config, args.save_path)
    gen_logger('Trainer set up successfully', log_file=log_file)

    # TRAINING PROCESS
    gen_logger("Training started", log_file=log_file)
    trainer.train()
    wandb.finish()
    gen_logger("Training complete", log_file=log_file)

    # SAVE MODEL
    gen_logger("Saving model", log_file=log_file)
    save_model_and_tokenizer(trainer, args.save_path)
    gen_logger("Model saved successfully", log_file=log_file)
if __name__ == '__main__':
    main()