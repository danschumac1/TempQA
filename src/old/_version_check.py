"""
Created on 09/30/2024

@author: Dan
To run:
python ./src/_version_check.py
"""
import pandas as pd
import torch
import wandb

# Hugging Face packages
import transformers
import huggingface_hub
import datasets

# TRL and PEFT packages
import trl
import peft

# Function to get the version of each package
def print_package_versions():
    print("Package Versions:")
    print(f"pandas: {pd.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"wandb: {wandb.__version__}")
    
    print(f"transformers: {transformers.__version__}")
    print(f"huggingface_hub: {huggingface_hub.__version__}")
    print(f"datasets: {datasets.__version__}")
    
    print(f"trl: {trl.__version__}")
    print(f"peft: {peft.__version__}")

# Call the function to print package versions
if __name__ == "__main__":
    print_package_versions()

