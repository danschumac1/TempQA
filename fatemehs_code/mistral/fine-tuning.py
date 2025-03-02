import argparse
import warnings
import json
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers
from datetime import datetime
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

parser = argparse.ArgumentParser(description="Fine-tune model script")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
parser.add_argument("--base_model_id", type=str, required=True, help="Base model ID")
parser.add_argument("--run_name", type=str, required=True, help="Run name for the model")
parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")
parser.add_argument("--fine_tune_version", type=str, required=True, help="Version of fine-tuning instructions")
parser.add_argument("--warmup_steps", type=int, required=True, help="Number of warmup steps")
parser.add_argument("--logging_steps", type=int, required=True, help="Number of logging steps")
parser.add_argument("--save_steps", type=int, required=True, help="Number of steps between saving checkpoints")
parser.add_argument("--eval_steps", type=int, required=True, help="Number of steps between evaluations")
parser.add_argument("--r", type=int, required=True, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, required=True, help="LoRA Alpha")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size per device")
parser.add_argument("--accumulation", type=int, required=True, help="Gradient accumulation steps")
parser.add_argument("--epochs", type=int, required=True, help="Number of Epochs")
args = parser.parse_args()

warnings.filterwarnings("ignore")

# Set up accelerator
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Set up proxy
os.environ['http_proxy'] = 'http://xa-proxy.utsarr.net:80'
os.environ['https_proxy'] = 'http://xa-proxy.utsarr.net:80'

# Log in to WandB
import wandb
wandb.login()

os.environ["WANDB_PROJECT"] = args.wandb_project

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


def create_instruction_v01(row, context_column, context=True):
    if context:
        return f"<s> [INST] {row['question']} Here is the context: {row[context_column]} [/INST] \nThe answer is: {row['answer']} </s>"
    else:
        return f"<s> [INST] {row['question']} [/INST] \nThe answer is: {row['answer']} </s>"

def create_instruction_v02(row, context_column, context=True):
    if context:
        return f"<s> [INST] {row['question']} Here is the context: {row[context_column]} [/INST]\nThe answer is: {row['answer']}</s>"
    else:
        return f"<s> [INST] {row['question']} [/INST]\nThe answer is: {row['answer']}</s>"

# Determine which create_instruction function to use based on the base_model_id
if "Instruct-v0.2" in args.base_model_id:
    print("THE MODEL IS FINE_TUNING IS IT.")
    create_instruction = create_instruction_v02
else:
    print("THE MODEL IS FINE_TUNING IS N-IT.")
    create_instruction = create_instruction_v01

# Apply the function to create instruction-tuning data for each context
train_df['relevant_instructions'] = train_df.apply(create_instruction, axis=1, context_column='relevant_context').tolist()
train_df['wrong_date_instructions'] = train_df.apply(create_instruction, axis=1, context_column='wrong_date_context').tolist()
train_df['random_instructions'] = train_df.apply(create_instruction, axis=1, context_column='random_context').tolist()
train_df['no_context_instructions'] = train_df.apply(create_instruction, axis=1, context_column='', context=False).tolist()

dev_df['relevant_instructions'] = dev_df.apply(create_instruction, axis=1, context_column='relevant_context').tolist()
dev_df['wrong_date_instructions'] = dev_df.apply(create_instruction, axis=1, context_column='wrong_date_context').tolist()
dev_df['random_instructions'] = dev_df.apply(create_instruction, axis=1, context_column='random_context').tolist()
dev_df['no_context_instructions'] = dev_df.apply(create_instruction, axis=1, context_column='', context=False).tolist()



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


train_dataset = train_df[[args.fine_tune_version]]
dev_dataset = dev_df[[args.fine_tune_version]]

print(train_df.iloc[0]['relevant_instructions'])
print('#'*100)
print(train_df.iloc[0]['wrong_date_instructions'])
print('#'*100)
print(train_df.iloc[0]['random_instructions'])
print('#'*100)
print(train_df.iloc[0]['no_context_instructions'])

# endregion

# region PREPARE MODEL 
# =============================================================================
# PREPARE MODEL 
# =============================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(args.base_model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(
    args.base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(prompt)

tokenized_train_dataset = train_dataset[args.fine_tune_version].apply(lambda x: generate_and_tokenize_prompt(x))
tokenized_val_dataset = dev_dataset[args.fine_tune_version].apply(lambda x: generate_and_tokenize_prompt(x))

def calculate_99th_percentile_length(df, column):
    lengths = [len(df.iloc[i][column]) for i in range(len(df))]
    percentile_99 = pd.Series(lengths).quantile(0.99)
    return percentile_99

percentile_99_length = calculate_99th_percentile_length(tokenized_train_dataset, 'input_ids')
max_length = int(percentile_99_length)
print(f"99th percentile length of 'input_ids': {max_length}")

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset[args.fine_tune_version].apply(lambda x: generate_and_tokenize_prompt2(x))
tokenized_val_dataset = dev_dataset[args.fine_tune_version].apply(lambda x: generate_and_tokenize_prompt2(x))

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=args.r,
    lora_alpha=args.lora_alpha,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

model = accelerator.prepare_model(model)

# endregion

# region TRAIN MODEL 
# =============================================================================
# TRAIN MODEL 
# =============================================================================

run_name = args.run_name
output_dir = run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=args.warmup_steps,                 # warmup_steps=total_steps×0.05
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=2.5e-5,                           # Want a small lr for finetuning
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,               # When to start reporting loss
        logging_dir="./logs",                           # Directory for storing logs
        save_strategy="steps",                          # Save the model checkpoint every logging step
        save_steps=args.save_steps,                     # Save checkpoints every 50 steps
        evaluation_strategy="steps",                    # Evaluate the model every logging step
        eval_steps=args.eval_steps,                     # Evaluate and save checkpoints every 50 steps
        do_eval=True,                                   # Perform evaluation at the end of training
        report_to="wandb",                              # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False                          # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.model.save_pretrained(run_name+'/model')

# endregion