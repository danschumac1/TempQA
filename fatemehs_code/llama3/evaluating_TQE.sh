#!/bin/bash

# Define arguments
IS_BASE=true # Set true if it's a base model
DATA_PATH="/workspace/storage/fatemeh/organized_projects/mini_temporal/clean_for_fatemeh/new_data/"

# BASE_MODEL_ID="unsloth/llama-3-8b"
# MODEL_PATHS=(
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-relevant_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-wrong_date_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-random_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-no_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-combined_context_finetuned-TQE"

# )

# BASE_MODEL_ID="unsloth/llama-3-8b-Instruct"
# MODEL_PATHS=(
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-relevant_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-wrong_date_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-random_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-no_context_finetuned-TQE"
#   "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/models/Llama3-8B-IT-combined_context_finetuned-TQE"
# )

BASE_MODEL_ID="unsloth/llama-3-8b"
MODEL_PATHS=(
    "unsloth/llama-3-8b"
)

# END OF OUTPUT_PATH MUST CONTAIN "/"
OUTPUT_PATH="/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/llama3/outputs/TQE/"
EVALUATION_VERSIONS=("no_context_instructions" "relevant_instructions" "wrong_date_instructions" "random_instructions")

# Loop through each model path
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "Evaluating model: $MODEL_PATH"
    python evaluating.py \
        --data_path "$DATA_PATH" \
        --model_path "$MODEL_PATH" \
        --output_path "$OUTPUT_PATH" \
        --base_model_id "$BASE_MODEL_ID" \
        --evaluation_versions "${EVALUATION_VERSIONS[@]}" \
        --is_base "$IS_BASE" \
        2>&1 | tee -a generation_llama.log
done
