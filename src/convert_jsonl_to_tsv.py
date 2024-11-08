"""
Created on 11/05/2024

@author: Dan Schumacher
How to use:
python ./src/convert_jsonl_to_tsv.py 
"""

import json
import csv
import os
def main():


    # Input and output file paths
    input_file = "./data/results/results.jsonl"
    output_file = "./data/results/results.tsv"

    # Define the TSV column headers
    headers = ['trained_on', 'eval_on','accuracy','f1','model','dataset','subset','timestamp','gen_params']

    # ensure writepath exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Open the input and output files
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(headers)  # Write the headers to the TSV file

        # Process each line in the JSONL file
        for line in infile:
            # Parse the JSON data
            data = json.loads(line.strip())

            # Extract required fields and assign default values if necessary
            row = [
                data.get("trained_on", ""),
                data.get("eval_on", ""),
                data.get("accuracy", ""),
                data.get("f1", ""),  # assuming subset is used as eval_test_set
                data.get("model", ""),  # assuming eval_on is used as eval_context
                data.get("dataset", ""),
                data.get("subset", ""),
                data.get("timestamp", ""),
                data.get("gen_params", ""),
            ]

            # Write the row to the TSV file
            writer.writerow(row)

    print(f"Data has been successfully written to {output_file}.")


if __name__ == "__main__":
    main()