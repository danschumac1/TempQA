#!/bin/bash

# Define the list of fine-tune versions
fine_tune_versions=("relevant_instructions" "wrong_date_instructions" "random_instructions" "no_context_instructions" "combined_instructions")


# base_model_id="unsloth/llama-3-8b"
# run_names=(
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-relevant_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-wrong_date_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-random_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-no_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-combined_context_finetuned-TQE"
# )

# base_model_id="unsloth/llama-3-8b-Instruct"
# run_names=(
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-relevant_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-wrong_date_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-random_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-no_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-combined_context_finetuned-TQE"
# )

# Define the common arguments
data_path="/workspace/storage/fatemeh/organized_projects/mini_temporal/clean_for_fatemeh/new_data/"
wandb_project="Llama-finetune-TQE"
logging_steps=5
eval_steps=5
warmup_steps=5
r=32
lora_alpha=64
batch_size=2
accumulation=4
epochs=1

# Create or clear the log file
LOG_FILE="finetune_llama.log"
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
