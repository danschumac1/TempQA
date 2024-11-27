"""
Created on 05/14/2024

@author: Dan Schumacher
This code takes a GEMMA model and a dataset, processes the data, fine-tunes, and saves a model for evaluation.
"""

#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================
# print('IMPORTS AND SET UP')
import os
os.chdir('/home/dan/mini_temporal')

#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import json
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # TrainingArguments,
    # pipeline,
    # logging,
)
from trl import SFTTrainer
import transformers
import argparse
import wandb 
import warnings

#torch.cuda.set_per_process_memory_fraction(0.95)
#torch.cuda.set_device(1)

device = 'cuda'
#endregion
#region # ARGPARSE
# =============================================================================
# ARGPARSE
# =============================================================================
# print('ARGPARSE')
def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")
    parser.add_argument('--dataset', type=str, required=True, choices= ['AQA','TQE'], help='Do you want to use AQA dataset or TQE')
    parser.add_argument('--model_context', type=str, required=True, choices= ['no_context','random_context','relevant_context','wrong_date_context'], help='Model context')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='How many pass at model at once?')
    parser.add_argument('--base_model', type = str, required=True, choices= ['gemma-2b','gemma-1.1-2b-it','gemma-7b','gemma-1.1-7b-it'])
    parser.add_argument('--lr', type = float, required=False, default=2e-5, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, required=False, default=6, help = 'How many training epochs?')
    return parser.parse_args()

# print('Argument parser defined')  # @$@
# =============================================================================
# FUNCTIONS
# =============================================================================
# Some of our answers in our TQE dataset have multiple correct answers (exmp: US, USA)
# these are seperated with the __or__ characters
# we will just grab the first answer during training. 
def check_or(answer):
    if '__or__' in answer:
        answer = answer.split('__or__')[0]
    return answer

def format_instruction(df, model_path, context=False):
    tokenized_instructions = []
    # USING AN INSTRUCTION TUNED TRAINED MODEL

    # all IT models use the same special tokens, so I will use 2b-it for both 2b-it and 7b-it
    fi_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", add_bos_token=True, add_eos_token=True, max_length=512,truncation=True)
    fi_tokenizer.padding_side = 'right'

    # Does this matte? I don't think so during training.
    # fi_tokenizer.padding_side = 'right'
    
    if context: # IT MODEL WITH CONTEXT
        for question, context, answer in zip(df['question'], df[context], df['answer']):

            # Make sure only 1 answer and no __or__ tokens
            answer = check_or(answer)

            # Plug in the question, context, and answer.
            message = [
                    {"role": "user", "content": f'{question} Here is the context: {context[:500]}'},
                {"role": "assistant", "content": f'The answer is: {answer}.'},
            ]

            # encode the model
            encodeds = fi_tokenizer.apply_chat_template(message, return_tensors="pt")

            # decode the model to get special tokens
            output = fi_tokenizer.batch_decode(encodeds, special_tokens=True)

            # apply eos to end of the string
            tokenized_instructions.append(output[0] + '<eos>')

    else: # IT MODEL WITHOUT CONTEXT
        for question, answer in zip(df['question'], df['answer']):

            # Make sure only 1 answer and no __or__ tokens
            answer = check_or(answer)

            # Plug in the question and answer.
            message = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f'The answer is: {answer}.'}
            ]

            # encode the model
            encodeds = fi_tokenizer.apply_chat_template(message, return_tensors="pt")

            # decode the model to get special tokens
            output = fi_tokenizer.batch_decode(encodeds, special_tokens=True)

            # apply eos to end of the string
            tokenized_instructions.append(output[0] + '<eos>')

    # during our training, lets print one to make sure it looks good.
    # print(tokenized_instructions[-1])
    return tokenized_instructions


def other_preprocessing(df, tokenizer, context):

    # convert from pandas to dataset obj
    df2 = df.copy()

    # shuffle

    # TOKENIZE DATASET
    if context == 'random_context':
        context_plug = 'rand'
    elif context == 'relevant_context':
        context_plug = 'rel'
    elif context == 'wrong_date_context':
        context_plug = 'wd'
    elif context == 'no_context':
        context_plug = 'no'
    df2 = df2[[context_plug+"_prompt"]]
    dataset = Dataset.from_pandas(df2)
    #print(df['no_prompt'].iloc[0])
    #print(type(df['no_prompt'].iloc[0]))
    dataset = dataset.map(lambda x: tokenizer(x[f'{context_plug}_prompt'], max_length=512, truncation=True), batched=True)
    dataset = dataset.remove_columns(f'{context_plug}_prompt')
    dataset = dataset.shuffle(seed=1234)
    print(dataset)
    # generated_text = tokenizer.decode(dataset[0][0], skip_special_tokens=False)
    # print('\n\n',f'GENERATED TEXT: {generated_text}', '\n\n')
    return dataset

def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
    # PARSE ARGUMENT
    args = parse_args()

    # Initialize wandb
    print('WANDB')
    wandb.init(project="temporal_understanding", config=args)
    print('LOG INTO HUGGING FACE')
    # LOG INTO HUGGING FACE
    with open('./training/token.txt', 'r') as file:
        token = file.read().strip()
    login(token=token)

    # print('BNB CONFIG')

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False, # Example setting, adjust as needed
    )
    '''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    '''

    #endregion
    #region # LOAD MODEL, TOKENIZER, AND DATASET
    # =============================================================================
    # LOAD MODEL, TOKENIZER, AND DATASET
    # =============================================================================
    print('load model')
    # LOAD MODEL
    model_id = f"google/{args.base_model}"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, use_cache=False)
    #model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0}, use_cache=False)
    #model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False)

    print('tokenizer')
    # SET UP TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=False, add_eos_token=False, truncation=True,  max_length=512)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    print('load data')
    # LOAD DATA
    train = pd.read_json(f'./data/{args.dataset}/train.jsonl', lines=True)
    dev = pd.read_json(f'./data/{args.dataset}/dev.jsonl', lines=True)

    print('Data loaded')  # @$@
    #endregion
    #region # CREATE PROMPTS
    # =============================================================================
    # CREATE PROMPTS
    # =============================================================================
    train['no_prompt'] = format_instruction(train, model_path = args.base_model)
    train['rel_prompt'] = format_instruction(train, model_path = args.base_model, context = 'relevant_context')
    train['rand_prompt'] = format_instruction(train, model_path = args.base_model, context = 'random_context')
    train['wd_prompt'] = format_instruction(train, model_path = args.base_model, context = 'wrong_date_context')

    dev['no_prompt'] = format_instruction(dev, model_path = args.base_model)
    dev['rel_prompt'] = format_instruction(dev, model_path = args.base_model, context = 'relevant_context')
    dev['rand_prompt'] = format_instruction(dev, model_path = args.base_model, context = 'random_context')
    dev['wd_prompt'] = format_instruction(dev, model_path = args.base_model, context = 'wrong_date_context')

    #endregion


    #region # OTHER PREPROCESSING
    train_data = other_preprocessing(train, tokenizer, args.model_context)
    dev_data = other_preprocessing(dev, tokenizer, args.model_context)
    print('Data preprocessed')  # @$@
    #endregion
    #region # ADVANCED FORMATTING
    # =============================================================================
    # LORA
    # =============================================================================
    print('lora')
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    #model = get_peft_model(model, lora_config).to(device)
    #model.gradient_checkpointing_enable()
    #model.enable_input_require_grads()

    # CALCULATE THE NUMBER OF TRAINABLE PARAMS
    #trainable, total = model.get_nb_trainable_parameters()
    #print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")
    print('Model prepared for training')
    
    #endregion
    #region # SET UP TRAINER
    # =============================================================================
    # SET UP TRAINER
    # =============================================================================
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=dev_data,
        dataset_text_field="prompt",
        peft_config=lora_config,
        packing=True,
        args=transformers.TrainingArguments(
            run_name = f'{args.base_model}-{args.model_context}-trained',
            num_train_epochs= args.epochs,
            per_device_train_batch_size= args.batch_size,
            per_device_eval_batch_size = args.batch_size,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=8,
            warmup_steps=1, # FATEMEH DID 1
            gradient_checkpointing=True,
            fp16=True,
            optim="paged_adamw_8bit",
            learning_rate=args.lr,
            logging_steps=25,
            eval_steps=500,
            output_dir="outputs",
            save_strategy="no",
            evaluation_strategy= 'steps',
            report_to="wandb"  # Integrate with wandb
        ),
        max_seq_length=512,  # Specify the desired max sequence length
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    print('Trainer set up')
    #endregion
    #region # TRAINING PROCESS
    # =============================================================================
    # TRAINING PROCESS
    # =============================================================================
    # EXECUTE THE FINE TUNING PROCESS
    trainer.train()
    print("THIS IS A TEST")

    # Specify save location of new model
    new_model = f"./training/models/{args.dataset}/{args.base_model.replace('.','_')}/{args.model_context}"

    # make sure that location exists
    os.makedirs(new_model, exist_ok=True)

    # save the model
    trainer.model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)
    print(f"Model and tokenizer saved at {new_model}")

if __name__ == '__main__':
    main()
