#!/bin/bash
# ---------------------------------------------------------------------------
# To Run: nohup ./bin/to_tsv.sh &
# ---------------------------------------------------------------------------
result_jsonl="./data/results/results.jsonl"
python src/results_to_tsv.py \
    --input_file $result_jsonl > ./data/results/results.tsv