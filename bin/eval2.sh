#!/bin/bash
# ---------------------------------------------------------------------------
# To Run: nohup ./bin/eval2.sh &
# ---------------------------------------------------------------------------
# DATASET PARAMETERS
# dataset="MenatQA"            
# test_files=("counterfactual_test.jsonl" "order_test.jsonl" "scope_test.jsonl" "scope_expand_test.jsonl") # Change this

dataset="TR_l2"
test_file="test.jsonl"

# dataset="TimeQAEasy"
# test_files=("test_easy.jsonl")

# dataset="TimeQAHard"
# test_files=("test_hard.jsonl")
# ---------------------------------------------------------------------------

# MODEL PARAMETERS
model="llama"      
splitter=$'assistant\n' # LLAMA uses this splitter
# model="mistral" # Change this
# splitter="[/INST]"  

training_contexts="relevant_context" 

# ---------------------------------------------------------------------------
# These stay the same
dataset_folder="./data/datasets/${dataset}/final"
eval_contexts=("relevant_context" "wrong_date_context" "random_context" "no_context")
pre_path="./data/generations/${model}/${dataset}"
if [ ! -f "./data/results/results.jsonl" ]; then
    touch "./data/results/results.jsonl"
fi

# Run the Python evaluation script
actual_path="${dataset_folder}/${test_file}"
for training_context in "${training_contexts[@]}" ; do
    for eval_context in "${eval_contexts[@]}" ; do
        gen_path="${pre_path}/${training_context}_trained/${eval_context}_evaluated.jsonl"
        config_type="${dataset}_${model}_${training_context%%_context*}"
        python src/eval.py \
            --gen_path $gen_path \
            --actual_path $actual_path \
            --dataset $dataset \
            --model $model \
            --trained_on $training_context \
            --test_file $test_file \
            --eval_context $eval_context \
            --config_type $config_type \
                --splitter "$splitter" \
            >> "./data/results/results2.jsonl"
    done
done