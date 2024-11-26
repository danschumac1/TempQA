#!/bin/bash
# ---------------------------------------------------------------------------
# To Run: nohup ./bin/gpt_trl2_gen.sh &
# ---------------------------------------------------------------------------
# Set the parameters
# dataset="MenatQA"  # Change this if needed
# test_files=("counterfactual_test.jsonl" "scope_test.jsonl" "scope_expand_test.jsonl" "order_test.jsonl")  # Test files for MENATQA
# ---------------------------------------------------------------------------
# dataset="TimeQAEasy"
# test_files=("test_easy.jsonl")  # For  and TimeQAHard
# ---------------------------------------------------------------------------
# dataset="TimeQAHard"
# test_files=("test.jsonl")  # For everything else
# ---------------------------------------------------------------------------
dataset="TR_l2"
test_files=("test.jsonl")  # For everything else
# ---------------------------------------------------------------------------
# dataset="TR_l3"
# test_files=("test.jsonl")  # For everything else
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# THESE STAY THE SAME
eval_contexts=("relevant_context" "wrong_date_context" "random_context" "no_context")  # Evaluation contexts
gpu=0  # GPU to use for processing

# Create / clear error log
error_log="./logs/gpt_gen.log"
> "$error_log"

# Loop through each test file and evaluation context
for test_file in "${test_files[@]}"; do
    for eval_context in "${eval_contexts[@]}"; do
        # Output directory and file path
        output_dir="./data/generations/gpt/${dataset}/base_trained"
        mkdir -p "$output_dir"  # Ensure output directory exists
        output_file="${output_dir}/${eval_context}_evaluated.jsonl"

        # Log start of task
        echo "Starting generation for $test_file with $eval_context at $(date)" >> "$error_log"
        echo "Running task on GPU $gpu with test file $test_file and context $eval_context"

        # Run the generation script
        CUDA_VISIBLE_DEVICES=$gpu python ./src/gpt_generations.py \
            --dataset "$dataset" \
            --test_file "$test_file" \
            --eval_context "$eval_context" \
            > "$output_file" 2>> "$error_log"

        # Log completion of task
        if [ $? -eq 0 ]; then
            echo "Finished generation for $test_file with $eval_context at $(date)" >> "$error_log"
        else
            echo "Error during generation for $test_file with $eval_context at $(date)" >> "$error_log"
        fi
    done
done

echo "All tasks on GPU $gpu completed."
