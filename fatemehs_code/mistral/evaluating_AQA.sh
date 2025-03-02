# !/bin/bash

DATA_PATH="/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/fine-tuning/data/AQA/"
OUTPUT_PATH="/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/outputs/AQA/"
EVALUATION_VERSIONS=("no_context_instructions" "relevant_instructions" "wrong_date_instructions" "random_instructions")
CHECKPOINT="1875"

# Part 1: Mistral-7B-v0.1 fine-tuned
IS_BASE=false
BASE_MODEL_ID="mistralai/Mistral-7B-v0.1"
MODEL_PATHS=(
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-relevant_context_finetuned-AQA"
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-wrong_date_context_finetuned-AQA"
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-no_context_finetuned-AQA"
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-random_context_finetuned-AQA"
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-combined_context_finetuned-AQA"
)

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

# Part 2: Mistral-7B-Instruct-v0.2 fine-tuned
IS_BASE=false
BASE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_PATHS=(
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-relevant_context_finetuned-AQA"
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-wrong_date_context_finetuned-AQA"
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-no_context_finetuned-AQA"
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-random_context_finetuned-AQA"
    "/workspace/storage/fatemeh/organized_projects/mini_temporal/.fatemeh/EMNLP_2024/fine-tuning/mistral/models/Mistral-7B-IT-combined_context_finetuned-AQA"
)

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

# Part 3: Mistral-7B-v0.1 base
IS_BASE=true
BASE_MODEL_ID="mistralai/Mistral-7B-v0.1"
MODEL_PATHS=(
    "mistralai/Mistral-7B-v0.1"
)

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


# Part 4: Mistral-7B-Instruct-v0.2 base
IS_BASE=true
BASE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_PATHS=(
    "mistralai/Mistral-7B-Instruct-v0.2"
)

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