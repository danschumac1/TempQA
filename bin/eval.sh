#!/bin/bash

## ---------------------------------------------------------------------------
# To Run: nohup ./bin/eval.sh &
# ---------------------------------------------------------------------------

# Dataset parameters
dataset="MenatQA"
model="gemma"

# Define the directories
generations_dir="./data/generations/${model}/${dataset}/mixed_context_trained"
test_dir="./data/datasets/${dataset}/final"

# Clear or create results file
> "./data/results/results.jsonl"

# Loop over all generated .jsonl files in the generations directory
find "${generations_dir}" -name "*.jsonl" | while read gen_path; do
    echo "Processing file: $gen_path"

    # Extract the model, dataset, and contexts from the file path
    model=$(echo "$gen_path" | cut -d'/' -f4)  # Extract the model
    dataset=$(echo "$gen_path" | cut -d'/' -f5)  # Extract the dataset

    # Extract the trained context and evaluated context
    trained_on=$(basename "$(dirname "$gen_path")")  # Get directory name for trained context
    trained_on=${trained_on%_trained}  # Remove the "_trained" suffix

    eval_on=$(basename "$gen_path" .jsonl)  # Get file name without extension
    eval_on=${eval_on%_evaluated}  # Remove only the "_evaluated" suffix

    # Split eval_on into eval_test_set and eval_context based on the first occurrence of "test"
    eval_test_set="${eval_on%%_test*}_test"  # Extract everything before "_test" and include "_test"
    eval_context="${eval_on#*_test}"  # Extract everything after "_test"
    eval_context="${eval_context#_}"  # Remove leading underscore, if present

    # Strip off the context part from eval_on (if any context like "_relevant_context", "_random_context", etc. exists)
    eval_on_base=$(echo "$eval_on" | sed -E 's/_(relevant_context|random_context|no_context|wrong_date_context)$//')

    # Map the eval_on_base to the corresponding actual test file
    actual_test_file="${test_dir}/${eval_on_base}.jsonl"
    if [ ! -f "$actual_test_file" ]; then
        echo "Actual test file not found for $eval_on_base: $actual_test_file"
        continue  # Skip this iteration if the actual test file doesn't exist
    fi

    # Run the Python evaluation script
    python ./src/eval.py \
        --gen_path "$gen_path" \
        --actual_path "$actual_test_file" \
        --model "$model" \
        --trained_on "$trained_on" \
        --dataset "$dataset" \
        --eval_test_set "$eval_test_set" \
        --eval_context "$eval_context" >> "./data/results/results.jsonl" 2>> logs/evaluation_errors.log

done
