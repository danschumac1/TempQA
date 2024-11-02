#!/bin/bash
# ----------------------------------------------------------
# to run: ./bin/gen_eval_sample.sh
# ----------------------------------------------------------
# Bash script to run the combined generation and evaluation Python script

# CHANGE THESE
test_file="counterfactual_test.jsonl"
dataset="MenatQA"
training_context="relevant_context"
eval_context="relevant_context"
model_type="gemma"
num_rows=20
batch_size=4

# DON'T CHANGE THESE
data_folder="./data/datasets/${dataset}/final"
model_path="models/${model_type}/${dataset}/${training_context}_trained"
config_type="${dataset}_${model_type}_${training_context%%_context*}"

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
     >"dummy_${dataset}_${model}_${training_context%%_context*}_trained.jsonl"

# Print completion message
echo "Generation and evaluation completed."