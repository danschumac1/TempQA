#!/bin/bash
# nohup ./bin/trainer_TimeQAEasy.sh &
#region MIXED CONTEXT
dataset="TimeQAEasy"
model='gemma'
training_context='mixed_context'
pre_path="./data/datasets/${dataset}/final"
train_path="${pre_path}/train_easy.jsonl"
dev_path="${pre_path}/dev_easy.jsonl"
save_path="./models/${model}/${dataset}/${training_context}_trained"
gpu=0
# ensure that save_path exists
mkdir -p $save_path
log="logs/trainer_errors_gpu${gpu}.log"
touch $log

# Run the training script and redirect only errors to trainer_errors.log
CUDA_VISIBLE_DEVICES=0 nohup python src/dynamic_trainer.py \
    --model_type $model \
    --train_file_path $train_path \
    --dev_file_path $dev_path \
    --save_path $save_path \
    --training_context $training_context \
    --batch_size 16 \
    --lr 2e-5 \
    --epochs 6 2>>$log &

echo "Training Script Complete on gpu${gpu}!!!\n">>$log
#endregion
#region RELEVANT CONTEXT
training_context='relevant_context'
save_path="./models/${model}/${dataset}/${training_context}_trained"
gpu=1

# ensure that save_path exists
mkdir -p $save_path
log="logs/trainer_errors_gpu${gpu}.log"
touch $log

# Run the training script and redirect only errors to trainer_errors.log
CUDA_VISIBLE_DEVICES=$gpu nohup python src/dynamic_trainer.py \
    --model_type $model \
    --train_file_path $train_path \
    --dev_file_path $dev_path \
    --save_path $save_path \
    --training_context $training_context \
    --batch_size 16 \
    --lr 2e-5 \
    --epochs 6 2>>$log

echo "Training Script Complete on gpu${gpu}!!!\n">>$log
#endregion