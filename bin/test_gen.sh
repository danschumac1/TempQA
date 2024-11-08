#!/bin/bash
# ---------------------------------------------------
# To Run: nohup ./bin/test_gen.sh
# ---------------------------------------------------
model="gemma"                           # Change this
dataset="MenatQA"                       # Change this
training_context="mixed_context-stacked" # Change this 
test_file="counterfactual_test.jsonl"   # Change this
batch_size=1                            # Change this
eval_context="relevant_context"         # Change this
                               
# These stay the same
gpu=0    
dataset_folder="./data/datasets/${dataset}/final"
config_type="${dataset}_${model}_${training_context%%_context*}"

CUDA_VISIBLE_DEVICES=$gpu python ./src/test_gen.py \
    --model="$model" \
    --dataset_folder "$dataset_folder" \
    --test_file "$test_file" \
    --dataset "$dataset" \
    --eval_context "$eval_context" \
    --config_type "$config_type" \
    --model_path "models/${model}/${dataset}/${training_context}_trained/" \
    --batch_size="$batch_size" >"dummy_${dataset}_${model}_${training_context%%_context*}_trained.jsonl"
echo "test_gen.sh: Done"