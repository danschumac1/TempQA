import json
import argparse

def convert_jsonl_to_tsv(jsonl_data):
    # Prepare header for TSV
    header = ["model", "dataset", "trained_on", "eval_on", "f1", "acc", "timestamp"]

    # Prepare output list for TSV
    tsv_output = []

    # Process each line in JSONL data
    for line in jsonl_data:
        data = json.loads(line)
        
        # Filter out rows with 'devide by zero'
        if "devide by zero" in (data["f1"], data["acc"]):
            continue
        
        # Convert to TSV format
        tsv_row = [str(data[col]) for col in header]
        tsv_output.append("\t".join(tsv_row))

    # Add header to the output
    tsv_output.insert(0, "\t".join(header))

    return "\n".join(tsv_output)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert JSONL to TSV format.")
    parser.add_argument("--input_file", help="Path to the input JSONL file.")
    
    args = parser.parse_args()

    # Read the JSONL data from the specified file
    with open(args.input_file, 'r') as f:
        jsonl_data = f.readlines()

    # Convert and print the TSV result
    tsv_result = convert_jsonl_to_tsv(jsonl_data)
    print(tsv_result)
