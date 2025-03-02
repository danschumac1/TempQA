import json
import pandas as pd
import argparse
import os
import re
import string
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor
import logging
import transformers
from datasets import load_dataset
from unsloth import FastLanguageModel

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Fine-tuning evaluation script")
parser.add_argument('--is_base', type=str, required=True, help="Is this generation for evaluating base model?")
parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset")
parser.add_argument('--model_path', type=str, required=True, help="Path to the models")
parser.add_argument('--output_path', type=str, required=True, help="Path to outputs folder")
parser.add_argument('--base_model_id', type=str, required=True, help="Base model identifier")
parser.add_argument('--evaluation_versions', nargs='+', required=True, help="Versions we must evaluate the model")
args = parser.parse_args()

# region LOAD DATA 
# =============================================================================
# LOAD DATA 
# =============================================================================
test_path = os.path.join(args.data_path, 'test.jsonl')

def load_jsonl_to_df(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Load data from jsons
if "AQA" in args.data_path: # For AQA dataset
    test_df = load_jsonl_to_df(test_path)
else: # For TQE dataset
    test_df = load_jsonl_to_df(test_path).drop(columns=['i'])

# test_df = test_df.sample(n=500, random_state=42).copy().reset_index(drop=True)
print(f"THERE ARE {len(test_df)} TEST DATA.")


# Determine all acepted answers as correct answer
if "AQA" in args.data_path: # For AQA dataset
    test_df['answer_list'] = test_df['answer'].apply(lambda x: x.split('__or__'))
else: # For TQE dataset
    test_df['answer_list'] = test_df['answer'].apply(lambda x: x[0].split('__or__'))

#endregion

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
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(row, context_column, context=True):
    if context:
        return f"[INST] {row['question']} Here is the context: {row[context_column]} [/INST] \nThe answer is:"
    else:
        return f"[INST] {row['question']} [/INST] \nThe answer is:"


# Apply the function to create instruction-tuning data for each context
test_df['relevant_instructions'] = test_df.apply(formatting_prompts_func, axis=1, context_column='relevant_context').tolist()
test_df['wrong_date_instructions'] = test_df.apply(formatting_prompts_func, axis=1, context_column='wrong_date_context').tolist()
test_df['random_instructions'] = test_df.apply(formatting_prompts_func, axis=1, context_column='random_context').tolist()
test_df['no_context_instructions'] = test_df.apply(formatting_prompts_func, axis=1, context_column='', context=False).tolist()


print(test_df.iloc[0]['relevant_instructions'])
print('#'*100)
print(test_df.iloc[0]['wrong_date_instructions'])
print('#'*100)
print(test_df.iloc[0]['random_instructions'])
print('#'*100)
print(test_df.iloc[0]['no_context_instructions'])

del model, tokenizer
# region LOAD MODEL 
# =============================================================================
# LOAD MODEL 
# =============================================================================

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def preprocess(text):
        return ' '.join(re.findall(r"\w+", text.lower()))

# region # extract_answer function
def extract_answer(generated_answer):
    # Regular expression to find the answer following "The answer is:" until a newline, bracket, period, or end of string
    match = re.search(r'The answer is:\s*(.*?)([\n\[\].]|$)', generated_answer)
    if match:
        return match.group(1).strip()
    return None
# endregion   

# Function to calculate BERT score in batches
def calculate_bert_scores(predictions, labels, tokenizer, model, batch_size=32):
    def compute_bert_embedding(text):
        tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length').to("cuda")
        with torch.no_grad():
            embedding = model(**tokens).last_hidden_state.mean(dim=1).detach().cpu().numpy()
        return embedding

    all_scores = []
    # Compute embeddings for all predictions first
    pred_embeddings = []
    for start in tqdm(range(0, len(predictions), batch_size), desc="Computing BERT Embeddings for Predictions"):
        end = min(start + batch_size, len(predictions))
        batch_predictions = predictions[start:end]
        batch_embeddings = [compute_bert_embedding(pred) for pred in batch_predictions]
        pred_embeddings.extend(batch_embeddings)

    # Compute scores in batches
    for start in tqdm(range(0, len(predictions), batch_size), desc="Calculating BERT Scores"):
        end = min(start + batch_size, len(predictions))
        batch_predictions = predictions[start:end]
        batch_labels = labels[start:end]
        batch_pred_embeddings = pred_embeddings[start:end]

        batch_scores = []
        for pred_embedding, answer_list in zip(batch_pred_embeddings, batch_labels):
            max_score = float('-inf')
            for answer in answer_list:
                answer_embedding = compute_bert_embedding(answer)
                cosine_sim = np.dot(pred_embedding, answer_embedding.T) / (np.linalg.norm(pred_embedding) * np.linalg.norm(answer_embedding))
                cosine_sim = cosine_sim.item()
                if cosine_sim > max_score:
                    max_score = cosine_sim
            batch_scores.append(max_score)
        all_scores.extend(batch_scores)
        
        del batch_predictions, batch_labels, batch_pred_embeddings, batch_scores
        torch.cuda.empty_cache()
    
    return all_scores

# Function to calculate Exact Match F1 score
def exact_match_f1(pred, answer_list):
    pred_words = set(pred.lower().split())
    best_f1 = 0
    for answer in answer_list:
        answer_words = set(answer.lower().split())
        TP = len(answer_words.intersection(pred_words))
        FP = len(pred_words.difference(answer_words))
        FN = len(answer_words.difference(pred_words))
        if TP == 0:
            f1 = 0
        else:
            prec = TP / (TP + FP) if TP + FP > 0 else 0
            rec = TP / (TP + FN) if TP + FN > 0 else 0
            if (prec + rec) > 0:
                f1 = 2 * ((prec * rec) / (prec + rec))
            else:
                f1 = 0
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

# Function to calculate Contains Metric score
def contains_metric(pred, answer_list):
    translator = str.maketrans('', '', string.punctuation)
    normalized_pred = pred.lower().translate(translator)
    for answer in answer_list:
        normalized_answer = answer.lower().translate(translator)
        if normalized_answer in normalized_pred:
            return 1
    return 0

print("Loading models and tokenizers...")
bert_model = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
bert_model = BertModel.from_pretrained(bert_model).to("cuda")

# Function to process in batches for generating responses
def process_batches_for_generation(ft_model, eval_tokenizer, test_dataset, evaluate_version, batch_size=32, max_length=500):
    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token

    results = []
    for start in tqdm(range(0, len(test_dataset), batch_size), desc="Processing Batches"):
        end = min(start + batch_size, len(test_dataset))
        batch = test_dataset.iloc[start:end]
        
        eval_prompts = batch[evaluate_version].tolist()
        model_inputs = eval_tokenizer(eval_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to("cuda")
        
        with torch.no_grad():
            
            
            # =============================================================================
            # MUST BE CHOSEN BASED ON THE MODEL
            # =============================================================================
            
            # Llama3-8B FOR COMBINED INSTRUCTIONS FOR TQE
            # generated_answers = ft_model.generate(
            #     **model_inputs, 
            #     max_new_tokens=16,
            #     do_sample=True, 
            #     temperature=.8,
            #     repetition_penalty=1.15,
            #     num_beams=3,
            #     top_p=0.99,
            #     top_k=0,
            #     length_penalty=.1
            #     )
            
            # Llama3-8B FOR ALL VERSION EXCEPT THE COMBINED INSTRUCTIONS FOR TQE
            # Llama3-8B-NIT FOR AQA
            generated_answers = ft_model.generate(
                **model_inputs, 
                max_new_tokens=16,
                do_sample=True, 
                temperature=.8,
                repetition_penalty=1.15,
                num_beams=2,
                top_p=0.99,
                top_k=0,
                length_penalty=.1
                )


            generated_answers = [eval_tokenizer.decode(g, skip_special_tokens=True) for g in generated_answers]
        
        for idx, row in batch.iterrows():
            eval_prompt = eval_prompts[idx - start]
            label = [preprocess(x) for x in row['answer_list']]
            generated_answer = generated_answers[idx - start]
            extracted_answer = extract_answer(generated_answer)
            
            results.append({
                'index': idx,
                'eval_prompt': eval_prompt,
                'generated_answer': generated_answer,
                'extracted_answer': preprocess(extracted_answer),
                'label': label
            })
        del batch, eval_prompts, model_inputs, generated_answers
        torch.cuda.empty_cache()  # Clear GPU memory
        
    return results


# region START GENERATION
# =============================================================================
# START GENERATION
# =============================================================================

model_dir = args.model_path

if args.is_base == 'true':
    args.model_path = args.base_model_id
    
print(f"Evaluating model: {args.model_path.split('/')[-1]}")

ft_model, eval_tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_dir,
        max_seq_length = max_seq_length,
        dtype = None,                                       # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = False,                               # Use 4bit quantization to reduce memory usage. Can be False
    )
FastLanguageModel.for_inference(ft_model)

for evaluate_version in args.evaluation_versions:
    print(f"Processing evaluation version: {evaluate_version}")
    test_dataset = test_df[[evaluate_version, 'answer_list']]  # Ensure 'answer_list' is included

    results = process_batches_for_generation(ft_model, eval_tokenizer, test_dataset, evaluate_version)

    # Save data with generated answers
    print(f"Saving generated answer data for {args.model_path.split('/')[-1]} with {evaluate_version}...")
    with open(args.output_path + 'files/' + f'{args.model_path.split("/")[-1]}_{evaluate_version}_data_with_generated_answer.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    predictions = [result['extracted_answer'] if result['extracted_answer'] else '' for result in results]
    labels = [result['label'] for result in results]
    
    print(f"Calculating scores for {args.model_path.split('/')[-1]} with evaluation version {evaluate_version}...")
    
    # Calculate Exact Match F1 and Contains Metric in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        exact_match_f1_scores = list(executor.map(exact_match_f1, predictions, labels))
        contains_metric_scores = list(executor.map(contains_metric, predictions, labels))
    
    # Calculate BERT scores in batches
    bem_metric_scores = calculate_bert_scores(predictions, labels, bert_tokenizer, bert_model)

    avg_exact_match_f1 = np.mean(exact_match_f1_scores)
    avg_contains_metric = np.mean(contains_metric_scores)
    avg_bem_metric = np.mean(bem_metric_scores)

    scores = {
        'Average Exact Match F1': avg_exact_match_f1,
        'Average Contains Metric': avg_contains_metric,
        'Average BEM Metric': avg_bem_metric
    }

    # Save scores
    print(f"Saving scores for {args.model_path.split('/')[-1]} with {evaluate_version}...")
    with open(args.output_path + 'scores/' + f'{args.model_path.split("/")[-1]}_{evaluate_version}_scores.json', 'w') as f:
        json.dump(scores, f, indent=4)

    print(f"Scores for {args.model_path.split('/')[-1]} with {evaluate_version}:")
    print(scores)
    torch.cuda.empty_cache()
torch.cuda.empty_cache()

# endregion