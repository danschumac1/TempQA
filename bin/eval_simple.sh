#!/bin/bash
# ---------------------------------------------------------------------------
# To Run: nohup ./bin/eval_simple.sh &
# ---------------------------------------------------------------------------
# Dataset parameters
test_file="counterfactual_test.jsonl"   # Change this
model="gemma"                           # Change this
dataset="MenatQA"                       # Change this
test_file_name="scope_test"             # Change this
training_context="mixed_context"        # Change this 
batch_size=2                            # Change this

# These stay the same
dataset_folder="./data/datasets/${dataset}/final"
eval_contexts=("relevant_context" "wrong_date_context" "random_context" "no_context")
actual_test_file="./data/datasets/${dataset}/final/${test_file_name}.jsonl"
gpu=1  

# create results file if it doesn't exist
if [ ! -f "./data/results/results.jsonl" ]; then
    touch "./data/results/results.jsonl"
fi

# Run the Python evaluation script
python ./src/eval.py \
    --gen_path "$gen_path" \
    --actual_path "$actual_test_file" \
    --model "$model" \
    --trained_on "$trained_on" \
    --dataset "$dataset" \
    --eval_test_set "$test_file_name" \
    --eval_context "$eval_context" \
    >> "./data/results/results.jsonl"