#!/bin/bash
#SBATCH -J gen_mistral_trl2_rel               # Job name
#SBATCH -o ./logs/Mistral_Menat_Mixed_gen%j.out  # Standard output file
#SBATCH -e ./logs/Mistral_Menat_Mixed_gen%j.err  # Standard error file
#SBATCH -p amdgpu                    # Partition name
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 1                         # Number of tasks
#SBATCH --time=72:00:00              # Time limit

# Debugging output to check the environment
echo "Starting job at $(date)"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"

module load anaconda3
conda activate TempQAEnv

# Run the mixed context trainer script
bash ./bin/slurm/generations_mistral_trl2_rel.sh