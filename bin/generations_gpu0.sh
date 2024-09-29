#!/bin/bash
# ---------------------------------------------------------------------------
# To Run: nohup ./bin/generations_gpu0.sh &
# ---------------------------------------------------------------------------
# Set the parameters
test_files=("counterfactual_test.jsonl" "scope_test.jsonl" "scope_expand_test.jsonl" "order_test.jsonl")
dataset="MenatQA"
model="gemma"
training_context="mixed_context_STACKED"  # GPU 0 will run relevant_context trained
dataset_folder="./data/datasets/${dataset}/final"
eval_context="relevant_context"
gpu=0  # GPU 0
error_log="./logs/generation_errors_gpu${gpu}.log"

# Create necessary directories
mkdir -p logs

# Loop through each test file
for test_file in "${test_files[@]}"; do
    # Set up the file paths and names
    save_path="./data/generations/${model}/${dataset}/${training_context}_trained"
    file_name="${test_file%.*}_${eval_context}_evaluated.jsonl"

    mkdir -p "$save_path"

    # Run the generation script on GPU 0
    echo "Running task on GPU $gpu with test file $test_file"
    CUDA_VISIBLE_DEVICES=$gpu python ./src/generations.py \
        --model="$model" \
        --dataset_folder "$dataset_folder" \
        --test_file "$test_file" \
        --dataset "$dataset" \
        --eval_context "$eval_context" \
        --model_path "models/${model}/${dataset}/${training_context}_trained/" \
        --training_context "$training_context" > "${save_path}/${file_name}" 2>> $error_log

done

echo "All tasks on GPU 0 (relevant_context) completed."
