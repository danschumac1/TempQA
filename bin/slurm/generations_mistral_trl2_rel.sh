#!/bin/bash
# ---------------------------------------------------------------------------
# To Run: nohup ./bin/generations_mistral_trl2_rel.sh &
# ---------------------------------------------------------------------------
# DATASET PARAMETERS
# dataset="MenatQA"                    
# test_files=("counterfactual_test.jsonl" "scope_test.jsonl" "scope_expand_test.jsonl" "order_test.jsonl") # for MENATQA

# dataset="TimeQAEasy"
# test_files=("test_easy.jsonl") # for TimeQAEasy and TimeQAHard

# dataset="TimeQAHard"
# test_files=("test_hard.jsonl") # for TimeQAEasy and TimeQAHard

dataset="TR_l2"
test_files=("test.jsonl") # for everything else

# ---------------------------------------------------------------------------
# MODEL PARAMETERS
# model="llama"
model="mistral"
# ---------------------------------------------------------------------------

# TRAINING CONTEXT
training_context="relevant_context"    # Change this 
# training_context="mixed_context"

# ---------------------------------------------------------------------------
# These stay the same
dataset_folder="./data/datasets/${dataset}/final"
batch_size=2
eval_contexts=("relevant_context" "wrong_date_context" "random_context" "no_context")
config_type="${dataset}_${model}_${training_context%%_context*}"
gpu=0  

# Create / clear error log
error_log="./logs/gen_${model}_${training_context}.log"
> "$error_log"

# Loop through each test file
for test_file in "${test_files[@]}"; do
    for eval_context in "${eval_contexts[@]}"; do
        # Set up the save path
        save_path="./data/generations/${model}/${dataset}/${training_context}"
        mkdir -p "$save_path" || exit 1

        # set up file name (they change depending on the dataset)
            # this line strips off the test.jsonl part of the test_file and adds the eval_context
            # ie. counterfact_test.jsonl -> counterfactual_relevant_context_evaluated.jsonl
        file_name="${test_file%%test.jsonl*}_${eval_context}_evaluated.jsonl"

        # Check if file_name is empty (ie it was just test.jsonl) get rid of proceeding underscore
            # ie. test.jsonl -> relevant_context_evaluated.jsonl
        if [ -z "$file_name" ]; then
            file_name="${eval_context}_evaluated.jsonl"
        fi

        # Run the generation script on GPU 0
        echo "Starting generation for $test_file at $(date)" >> "$error_log"
        echo "Running task on GPU $gpu with test file $test_file"
        CUDA_VISIBLE_DEVICES=$gpu python ./src/generations.py \
            --model="$model" \
            --dataset_folder "$dataset_folder" \
            --test_file "$test_file" \
            --dataset "$dataset" \
            --eval_context "$eval_context" \
            --config_type "$config_type" \
            --model_path "models/${model}/${dataset}/${training_context}/" \
            --batch_size="$batch_size" > "${save_path}/${file_name}" 2>> "$error_log"
        echo "Finished generation for $test_file at $(date)" >> "$error_log"
    done
done

echo "All tasks on GPU $gpu (relevant_context) completed."
