#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

#SBATCH --job-name=test-project
#SBATCH --output=run/test-project-%j.out
#SBATCH --error=run/test-project-%j.err

module load cuda/12.1
cd /home/alan.masutti/Project
# git stash save "Stashing changes for Job Execution"
# git checkout WMMA_MatMul_v1.0.1
make

srun /home/alan.masutti/Project/build/bin/main

# source scripts/executePython.sh utils/createCharts.py