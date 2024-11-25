#!/bin/bash
#SBATCH -J relevant_training                                # Job name
#SBATCH -o ./logs/relevant_training_%j.out         # Standard output file
#SBATCH -e ./logs/relevant_training_%j.err         # Standard error file
#SBATCH -p amdgpu                                           # Partition name
#SBATCH -N 1                                                # Number of nodes
#SBATCH -n 1                                                # Specify 1 task/core
#SBATCH --time=72:00:00                                     # Time limit

# Debugging output to check the environment
echo "Starting job at $(date)"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"

module load anaconda3
conda activate TempQAEnv

# Run the relevant context trainer script
bash ./bin/rel_trainer.sh