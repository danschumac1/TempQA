import json
import os
from typing import List, Union
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import transformers

def extract_answer_from_list(answer:Union[str, List[str]]) -> str:
    if isinstance(answer, list):
        answer = answer[0]
    elif '__or__' in answer:
        answer = answer.split('__or__')[0]
    return answer

def gemma_trainer_formatter(df: pd.DataFrame, context_type: str) -> list:
    prompts = []
    # Iterate over the DataFrame rows
    for question, answer, context in zip(df['question'], df['answers'], df[context_type]):
        answer = extract_answer_from_list(answer)
        prompt = f'''<bos><start_of_turn>user\nIn as few words as possible, answer the following question given the context.\nQuestion: {question}\nContext: {context}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn><eos>'''
        prompts.append(prompt)
    return prompts

def llama_trainer_formatter(df: pd.DataFrame, context_type: str) -> list:
    formatted_prompts = []
    for question, context, answer in zip(df['question'], df[context_type], df['answers']):
        system = "You are an expert in answering time related questions. Please provide consistent, brief answers in the style of 'The answer is X'."
        prompt = f"In as few words as possible, answer the following question given the context.\nQuestion: {question}\nContext: {context}"
        formatted_prompt  = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{answer}"
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts

def mistral_trainer_formatter(df: pd.DataFrame, context_type: str) -> list:
    prompts = []
    # Iterate over the DataFrame rows
    for question, answer, context in zip(df['question'], df['answers'], df[context_type]):
        answer = extract_answer_from_list(answer)
        prompt = f'''<s>[INST] In as few words as possible, answer the following question given the context.\nQuestion:{question}\nContext:{context}[/INST]\n{answer}</s>'''
        prompts.append(prompt)
    return prompts
        
def selection_formatter(formatter: str) -> callable:
    if formatter == 'gemma_formatter':
        return gemma_trainer_formatter
    elif formatter == 'mistral_formatter':
        return mistral_trainer_formatter
    elif formatter == 'llama_formatter':
        return llama_trainer_formatter

def make_tokenized_prompt_column(df:pd.DataFrame, tokenizer:AutoTokenizer, context:str) -> Dataset:
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: tokenizer(x[context], max_length=512, truncation=True), batched=True)
    dataset = dataset.shuffle(seed=1234)
    return dataset

def load_config(json_path, config_type):
    """
    Load model-specific configuration from a JSON file.
    
    Args:
        json_path (str): The path to the JSON configuration file.
        config_type (str): The model type to load (e.g., 'gemma', 'mistral', 'llama').

    Returns:
        dict: A dictionary containing the configuration for the specified model type.
    """
    with open(json_path, 'r') as file:
        config = json.load(file)
    return config[config_type]
    
def get_trainer_old(model, tokenizer, train_dataset, dev_dataset, config, epoch, lr, batch_size, save_path):
    """
    Set up and return the SFTTrainer based on the provided configuration.

    Args:
        model: The pre-trained model.
        tokenizer: The tokenizer associated with the model.
        train_dataset: The training dataset.
        dev_dataset: The validation dataset.
        config (dict): The configuration dictionary loaded from the JSON file.

    Returns:
        SFTTrainer: A trainer object configured for fine-tuning the model.
    """
    # Load the LoRA configuration from the config dictionary
    lora_config = LoraConfig(
        **config["lora_config"]
    )
    
    # Set up the trainer using the provided configuration
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        dataset_text_field="prompt",
        peft_config=lora_config,
        packing=True,
        args=TrainingArguments(
            run_name=f'{config["base_model"]}-trained',
            num_train_epochs=epoch,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=8,
            warmup_steps=1,
            gradient_checkpointing=True,
            fp16=True,
            optim="paged_adamw_8bit",
            learning_rate=lr,
            logging_steps=25,
            eval_steps=500,                     # Evaluate every 500 steps, or you can use 'epoch' for evaluation every epoch
            output_dir=save_path,
            save_strategy="epoch",              # Evaluate at each epoch
            evaluation_strategy="epoch",        # Evaluate at the end of every epoch
            save_total_limit=1,                 # Only keep 1 best model to avoid clutter
            load_best_model_at_end=True,        # Load the best model at the end
            metric_for_best_model="eval_loss",  # Save based on evaluation loss
            greater_is_better=False,            # We want to minimize the loss
            report_to="wandb"                   # Report to Weights and Biases
        ),
        max_seq_length=512,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    return trainer

def get_trainer(
        model: str, tokenizer:AutoTokenizer, train_dataset: Dataset, 
        dev_dataset: Dataset, config: dict, save_path: str) -> SFTTrainer:
    """
    Set up and return the SFTTrainer based on the provided configuration.

    Args:
        model: The pre-trained model.
        tokenizer: The tokenizer associated with the model.
        train_dataset: The training dataset.
        dev_dataset: The validation dataset.
        config (dict): The configuration dictionary loaded from the JSON file.

    Returns:
        SFTTrainer: A trainer object configured for fine-tuning the model.
    """
    # Load the LoRA configuration from the config dictionary
    lora_config = LoraConfig(
        **config["lora_config"]
    )

    # Extract the trainer configuration from the config dictionary
    # print the trainer configuration
    for key, value in config["trainer_config"].items():
        print(f"{key}: {value}")
    # Set up the trainer using the provided configuration
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        dataset_text_field="prompt",
        peft_config=lora_config,
        packing=config["packing"],
        max_seq_length=config.get("max_seq_length", 512),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=TrainingArguments(
            run_name=f'{config["base_model"]}-trained',
            **config["trainer_config"],
        ),
    )
    return trainer

def save_model_and_tokenizer(trainer, save_path):
    # Save the fine-tuned model
    trainer.model.save_pretrained(save_path)
    
    # Save the tokenizer
    trainer.tokenizer.save_pretrained(save_path)
    
    # Optionally, save the training arguments (useful for reproducibility)
    with open(os.path.join(save_path, "training_args.json"), "w") as f:
        f.write(trainer.args.to_json_string())