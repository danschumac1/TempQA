#!/bin/bash

# Define arguments
IS_BASE=true # Set true if it's a base model
DATA_PATH="/workspace/storage/fatemeh/organized_projects/mini_temporal/clean_for_fatemeh/new_data/"

# BASE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_PATHS=(
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-relevant_context_finetuned-TQE"
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-wrong_date_context_finetuned-TQE"
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-no_context_finetuned-TQE"
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-random_context_finetuned-TQE"
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-combined_context_finetuned-TQE"
# )

# BASE_MODEL_ID="mistralai/Mistral-7B-v0.1"
# MODEL_PATHS=(
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-relevant_context_finetuned-TQE"
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-wrong_date_context_finetuned-TQE"
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-no_context_finetuned-TQE"
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-random_context_finetuned-TQE"
#     "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-combined_context_finetuned-TQE"
# )

BASE_MODEL_ID="mistralai/Mistral-7B-v0.1"
MODEL_PATHS=(
    "mistralai/Mistral-7B-v0.1"
)

OUTPUT_PATH="/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/outputs/TQE_check/"
EVALUATION_VERSIONS=("no_context_instructions" "relevant_instructions" "wrong_date_instructions" "random_instructions")
CHECKPOINT="175"

# Loop through each model path
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "Evaluating model: $MODEL_PATH"
    python evaluating.py \
        --data_path "$DATA_PATH" \
        --model_path "$MODEL_PATH" \
        --output_path "$OUTPUT_PATH" \
        --base_model_id "$BASE_MODEL_ID" \
        --evaluation_versions "${EVALUATION_VERSIONS[@]}" \
        --checkpoint "$CHECKPOINT" \
        --is_base "$IS_BASE" \
        2>&1 | tee -a generation_mistral.log
done
