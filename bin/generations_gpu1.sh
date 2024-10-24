#!/bin/bash
# ---------------------------------------------------------------------------
# To Run: nohup ./bin/generations_gpu1.sh &
# ---------------------------------------------------------------------------
# Set the parameters
test_files=("counterfactual_test.jsonl" "scope_test.jsonl" "scope_expand_test.jsonl" "order_test.jsonl") # for MENAT
dataset="MenatQA"
model="gemma"
training_context="relevant_context"  # GPU 1 will run mixed_context trained
dataset_folder="./data/datasets/${dataset}/final"
eval_contexts=("relevant_context" "wrong_date_context" "no_context" "random_context")
gpu=1  # GPU 1
error_log="./logs/generation_errors_gpu${gpu}.log"
batch_size=2
# Create necessary directories
mkdir -p logs || exit 1

# Loop through each test file
for test_file in "${test_files[@]}"; do
    for eval_context in "${eval_contexts[@]}"; do
        # Set up the file paths and names
        save_path="./data/generations/${model}/${dataset}/${training_context}_trained"
        file_name="${test_file%.*}_${eval_context}_evaluated.jsonl"
        
        mkdir -p "$save_path" || exit 1

        # Run the generation script on GPU 1
        echo "Starting generation for $test_file at $(date)" >> "$error_log"
        echo "Running task on GPU $gpu with test file $test_file"
        CUDA_VISIBLE_DEVICES=$gpu python ./src/generations.py \
            --model="$model" \
            --dataset_folder "$dataset_folder" \
            --test_file "$test_file" \
            --dataset "$dataset" \
            --eval_context "$eval_context" \
            --model_path "models/${model}/${dataset}/${training_context}_trained/" \
            --batch_size="$batch_size" > "${save_path}/${file_name}" 2>> "$error_log"
        echo "Finished generation for $test_file at $(date)" >> "$error_log"
    done
done

echo "All tasks on GPU $gpu (relevant_context) completed."
