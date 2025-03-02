#!/bin/bash

# Define the list of fine-tune versions
# fine_tune_versions=("relevant_instructions" "wrong_date_instructions" "random_instructions" "no_context_instructions")
# fine_tune_versions=("combined_instructions")

# base_model_id="mistralai/Mistral-7B-v0.1"
# run_names=(
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-relevant_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-wrong_date_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-random_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-no_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-combined_context_finetuned-TQE"
# )

# base_model_id="mistralai/Mistral-7B-Instruct-v0.2"
# run_names=(
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-relevant_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-wrong_date_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-random_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-no_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-combined_context_finetuned-TQE"
# )

# Define the common arguments
data_path="/workspace/storage/fatemeh/organized_projects/mini_temporal/clean_for_fatemeh/new_data/"
wandb_project="Mistral-finetune-TQE"
logging_steps=25
eval_steps=25
save_steps=25
warmup_steps=1
r=32
lora_alpha=64
batch_size=2
accumulation=1
epochs=1

# Create or clear the log file
LOG_FILE="finetune_mistral.log"
> $LOG_FILE

# Function to run the Python script and log its output
run_and_log() {
  local version=$1
  local run_name=$2  
  echo "Running fine-tune for version $version..." | tee -a $LOG_FILE
  python3 fine-tuning.py \
    --data_path "$data_path" \
    --base_model_id "$base_model_id" \
    --run_name "$run_name" \
    --wandb_project "$wandb_project" \
    --fine_tune_version "$version" \
    --warmup_steps "$warmup_steps" \
    --logging_steps "$logging_steps" \
    --save_steps "$save_steps" \
    --eval_steps "$eval_steps" \
    --r "$r" \
    --lora_alpha "$lora_alpha" \
    --batch_size "$batch_size" \
    --accumulation "$accumulation" \
    --epochs "$epochs" 2>&1 | tee -a $LOG_FILE
  echo "Finished running fine-tune for version $version" | tee -a $LOG_FILE
}

# Iterate over each fine-tune version
for i in "${!fine_tune_versions[@]}"
do
  run_and_log "${fine_tune_versions[$i]}" "${run_names[$i]}"
done