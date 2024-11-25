#!/bin/bash
# ---------------------------------------------------------------------------
# To Run: nohup ./bin/eval.sh &
# ---------------------------------------------------------------------------
# DATASET PARAMETERS
dataset="MenatQA"            
test_files=("counterfactual_test.jsonl" "order_test.jsonl" "scope_test.jsonl" "scope_expand_test.jsonl") # Change this

# dataset="TR_l2"
# test_files=("test.jsonl")

# dataset="TimeQAEasy"
# test_files=("test_easy.jsonl")

# dataset="TimeQAHard"
# test_files=("test_hard.jsonl")
# ---------------------------------------------------------------------------

# MODEL PARAMETERS
model="llama"      
splitter=$'assistant\n' # LLAMA uses this splitter

# model="mistral"                                                                            # Change this
# splitter="[/INST]"  
# ---------------------------------------------------------------------------
# These stay the same
training_contexts=("relevant_context" "mixed_context")                                                   
dataset_folder="./data/datasets/${dataset}/final"
eval_contexts=("relevant_context" "wrong_date_context" "random_context" "no_context")
pre_path="./data/generations/${model}/${dataset}"
if [ ! -f "./data/results/results.jsonl" ]; then
    touch "./data/results/results.jsonl"
fi

# Run the Python evaluation script
for test_file in "${test_files[@]}" ; do
    actual_path="${dataset_folder}/${test_file}"
    for eval_context in "${eval_contexts[@]}" ; do
        for training_context in "${training_contexts[@]}" ; do
            gen_path="${pre_path}/${training_context}_trained/${test_file%%_test*}__${eval_context}_evaluated.jsonl"
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
                --splitter $splitter \
                >> "./data/results/results.jsonl"
        done
    done
done