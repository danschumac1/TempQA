#!/bin/bash
# ----------------------------------------------------------
# to run: ./bin/gen_eval_sample.sh
# ----------------------------------------------------------
# Bash script to run the combined generation and evaluation Python script

# DATASET PARAMS
dataset="TR_l2"
test_file="test.jsonl"

dataset="MenatQA"
test_file="test.jsonl"

dataset="TR_l2"
test_file="test.jsonl"

# ----------------------------------------------------------
# MODEL PARAMS
training_context="relevant_context"
model_type="llama"
splitter=$'assistant\n' # LLAMA uses this splitter
# model_type="mistral"
# splitter="[/INST]" # Mistral uses this splitter
# model_type="gemma"
# splitter=$'\nmodel\n' # Gemma uses this splitter
# ----------------------------------------------------------

# DON'T CHANGE THESE
eval_context="relevant_context"
num_rows=20
batch_size=2
data_folder="./data/datasets/${dataset}/final"
model_path="models/${model_type}/${dataset}/${training_context}"

# config_type="${dataset}_${model_type}_${training_context%%_context*}"
config_type="MenatQA_${model_type}_${training_context%%_context*}"

# Run the Python script
CUDA_VISIBLE_DEVICES=0 python ./src/gen_eval_sample.py \
    --dataset_folder "$data_folder" \
    --test_file "$test_file" \
    --dataset "$dataset" \
    --eval_context "$eval_context" \
    --model_path "$model_path" \
    --model "$model_type" \
    --config_type "$config_type" \
    --batch_size "$batch_size" \
    --num_rows "$num_rows" \
    --splitter "$splitter" \
     >"./data/temp/dummy_${dataset}_${model}_${training_context%%_context*}_trained.jsonl"

# Print completion message
echo "Generation and evaluation completed."