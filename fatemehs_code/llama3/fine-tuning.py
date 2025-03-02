from unsloth import FastLanguageModel
import argparse
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from datetime import datetime
import json
import pandas as pd
import os
import warnings

parser = argparse.ArgumentParser(description="Fine-tune model script")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
parser.add_argument("--base_model_id", type=str, required=True, help="Base model ID")
parser.add_argument("--run_name", type=str, required=True, help="Run name for the model")
parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")
parser.add_argument("--fine_tune_version", type=str, required=True, help="Version of fine-tuning instructions")
parser.add_argument("--warmup_steps", type=int, required=True, help="Number of warmup steps")
parser.add_argument("--logging_steps", type=int, required=True, help="Number of logging steps")
parser.add_argument("--eval_steps", type=int, required=True, help="Number of steps between evaluations")
parser.add_argument("--r", type=int, required=True, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, required=True, help="LoRA Alpha")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size per device")
parser.add_argument("--accumulation", type=int, required=True, help="Gradient accumulation steps")
parser.add_argument("--epochs", type=int, required=True, help="Number of Epochs")
args = parser.parse_args()

warnings.filterwarnings("ignore")

os.environ['http_proxy'] = 'http://xa-proxy.utsarr.net:80'
os.environ['https_proxy'] = 'http://xa-proxy.utsarr.net:80'

import wandb
wandb.login()

wandb_project = args.wandb_project
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project


# region LOAD DATA 
# =============================================================================
# LOAD DATA 
# =============================================================================
train_path = os.path.join(args.data_path, 'train.jsonl')
dev_path = os.path.join(args.data_path, 'dev.jsonl')
test_path = os.path.join(args.data_path, 'test.jsonl')

def load_jsonl_to_df(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Load data from jsons
if "AQA" in args.data_path: # For AQA dataset
    train_df = load_jsonl_to_df(train_path)
    dev_df = load_jsonl_to_df(dev_path)
else: # For TQE dataset
    train_df = load_jsonl_to_df(train_path).drop(columns=['i'])
    dev_df = load_jsonl_to_df(dev_path).drop(columns=['i'])

def check_or(answer):
    if '__or__' in answer:
        answer = answer.split('__or__')[0]
    return answer

train_df['answer'] = train_df['answer'].apply(lambda x: check_or(x))
dev_df['answer'] = dev_df['answer'].apply(lambda x: check_or(x))

print(f"THERE ARE {len(train_df)} TRAIN DATA.")
print(f"THERE ARE {len(dev_df)} VALIDATION DATA.")

# endregion

# region LOAD MODEL + PREAPRE DATA
# =============================================================================
# LOAD MODEL + PREAPRE DATA
# =============================================================================
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.base_model_id,
    max_seq_length = max_seq_length,
    dtype = None,                                       # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False,                               # Use 4bit quantization to reduce memory usage. Can be False
)

model = FastLanguageModel.get_peft_model(
    model,
    r = args.r,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = args.lora_alpha,
    lora_dropout = 0,                                   # Dropout = 0 is currently optimized
    bias = "none",                                      # Bias = "none" is currently optimized
)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(row, context_column, context=True):
    if context:
        return f"[INST] {row['question']} Here is the context: {row[context_column]} [/INST] \nThe answer is: {row['answer']} " + EOS_TOKEN
    else:
        return f"[INST] {row['question']} [/INST] \nThe answer is: {row['answer']} " + EOS_TOKEN


# Apply the function to create instruction-tuning data for each context
train_df['relevant_instructions'] = train_df.apply(formatting_prompts_func, axis=1, context_column='relevant_context').tolist()
train_df['wrong_date_instructions'] = train_df.apply(formatting_prompts_func, axis=1, context_column='wrong_date_context').tolist()
train_df['random_instructions'] = train_df.apply(formatting_prompts_func, axis=1, context_column='random_context').tolist()
train_df['no_context_instructions'] = train_df.apply(formatting_prompts_func, axis=1, context_column='', context=False).tolist()

dev_df['relevant_instructions'] = dev_df.apply(formatting_prompts_func, axis=1, context_column='relevant_context').tolist()
dev_df['wrong_date_instructions'] = dev_df.apply(formatting_prompts_func, axis=1, context_column='wrong_date_context').tolist()
dev_df['random_instructions'] = dev_df.apply(formatting_prompts_func, axis=1, context_column='random_context').tolist()
dev_df['no_context_instructions'] = dev_df.apply(formatting_prompts_func, axis=1, context_column='', context=False).tolist()

if args.fine_tune_version == 'combined_instructions':
    print(f"THIS SCRIPT IS FOR COMBINED INSTRUCTIONS.")

    fine_tune_versions = ['no_context_instructions', 'relevant_instructions', 'random_instructions', 'wrong_date_instructions']

    train_df = train_df[fine_tune_versions].copy()
    dev_df = dev_df[fine_tune_versions].copy()

    import random
    random.seed(42)

    def combine_instructions(row):
        return random.choice([row[col] for col in fine_tune_versions])

    train_df['combined_instructions'] = train_df.apply(combine_instructions, axis=1)
    dev_df['combined_instructions'] = dev_df.apply(combine_instructions, axis=1)


print(train_df.iloc[0]['relevant_instructions'])
print('#'*100)
print(train_df.iloc[0]['wrong_date_instructions'])
print('#'*100)
print(train_df.iloc[0]['random_instructions'])
print('#'*100)
print(train_df.iloc[0]['no_context_instructions'])


from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df[[args.fine_tune_version]])
dev_dataset = Dataset.from_pandas(dev_df[[args.fine_tune_version]])

# endregion

# region TRAIN MODEL
# =============================================================================
# TRAIN MODEL
# =============================================================================

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset=dev_dataset,
    dataset_text_field = args.fine_tune_version,
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,                                    # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.accumulation,
        warmup_steps = args.warmup_steps,
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps =args.logging_steps,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = args.run_name,
        report_to="wandb",
        run_name=f"{args.run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
)

trainer.train()
trainer.model.save_pretrained(args.run_name)
