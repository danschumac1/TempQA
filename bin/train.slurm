#!/bin/bash
#SBATCH -J sys
#SBATCH -o ./nohup.out
#SBATCH -p gpu1v100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=72:00:00

module load anaconda3
conda activate ArcTempQA

time bash trainer_Menat.sh  