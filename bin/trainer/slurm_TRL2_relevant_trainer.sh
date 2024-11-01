#!/bin/bash
# Script for training with relevant context

dataset="TR_l2"
model='gemma'
training_context='relevant_context'
pre_path="./data/datasets/${dataset}/final"
train_path="${pre_path}/train.jsonl"
dev_path="${pre_path}/dev.jsonl"
save_path="./models/${model}/${dataset}/${training_context}_trained"
log="logs/trainer_errors_relevant_context.log"

# Ensure that save_path and log directories exist
mkdir -p $save_path
mkdir -p $(dirname $log)

# Use GPU 0 for relevant context training
echo "Starting relevant context training" >> $log
CUDA_VISIBLE_DEVICES=0 python src/dynamic_trainer.py \
    --model_type $model \
    --train_file_path $train_path \
    --dev_file_path $dev_path \
    --save_path $save_path \
    --training_context $training_context \
    --batch_size 8 \
    --lr 2e-5 \
    --epochs 6 2>> $log

echo "Relevant context training complete" >> $log