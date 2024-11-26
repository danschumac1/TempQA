#!/bin/bash
# ----------------------------------------------------------
# to run: ./bin/gpt_gen_eval_sample.sh
# ----------------------------------------------------------
# Bash script to run the combined generation and evaluation Python script using GPT-4o

# CHANGE THESE
test_file="test.jsonl"
dataset="TR_l2"
eval_context="relevant_context"
num_rows=20
batch_size=4
output_file="./data/temp/dummy_${dataset}_gpt_${eval_context}_evaluated.jsonl"

# DON'T CHANGE THESE
data_folder="./data/datasets/${dataset}/final"

# Run the Python script
python ./src/gpt_generations.py \
    --dataset_folder "$data_folder" \
    --test_file "$test_file" \
    --eval_context "$eval_context" \
    --output_file "$output_file"

# Print completion message
echo "GPT-4o generation and evaluation completed. Outputs saved to $output_file."
