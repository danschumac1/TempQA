#!/bin/bash

# To Run: nohup ./bin/test_gen.sh

CUDA_VISIBLE_DEVICES=0 python ./src/test_gen.py \
    --dataset_folder './data/datasets/MenatQA/final' \
    --test_file 'counterfactual_test.jsonl' \
    --dataset MenatQA \
    --eval_context relevant_context \
    --model_path 'models/gemma/MenatQA/relevant_context_trained' \
    --model gemma > dummy_rel_MenatQA.jsonl

# CUDA_VISIBLE_DEVICES=1 python ./src/test_gen.py \
#     --dataset_folder './data/datasets/TimeQAEasy/final' \
#     --test_file 'test_easy.jsonl' \
#     --dataset TimeQAEasy \
#     --eval_context relevant_context \
#     --model_path 'models/gemma/TimeQAEasy/relevant_context_trained' \
#     --model gemma> dummy_rel_TimeQA.jsonl

# CUDA_VISIBLE_DEVICES=1 python ./src/test_gen.py \
#     --dataset_folder './data/datasets/TimeQAEasy/final' \
#     --test_file 'test_easy.jsonl' \
#     --dataset TimeQAEasy \
#     --eval_context relevant_context \
#     --model_path 'models/gemma/TimeQAEasy/relevant_context_trained' \
#     --model gemma> dummy_generations.jsonl