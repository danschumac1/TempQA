#!/bin/bash
# nohup ./bin/trainer/trainer_Menat.sh &
#region MIXED CONTEXT
dataset="TR_l2"
model='llama'
training_context='mixed_context'
pre_path="./data/datasets/${dataset}/final"
train_path="${pre_path}/train_mixed_stacked.jsonl"
dev_path="${pre_path}/dev_mixed_stacked.jsonl"
save_path="./models/${model}/${dataset}/${training_context}"
gpu=0

# ensure that save_path exists
log="logs/tp_${model}_${training_context}.log"
touch $log
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$gpu nohup python ./src/dynamic_trainer.py \
    --model_type $model \
    --train_file_path $train_path \
    --dev_file_path $dev_path \
    --save_path $save_path \
    --training_context $training_context \
    --gpu $gpu \
    --epochs 6 \
    > $log 2>&1

echo "Training Script Complete on gpu${gpu}!!!\n"