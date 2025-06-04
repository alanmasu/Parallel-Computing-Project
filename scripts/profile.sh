#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --job-name=test-project
#SBATCH --output=run/test-project-%j.out
#SBATCH --error=run/test-project-%j.err

#SBATCH --mail-user=alan.masutti@studenti.unitn.it
#SBATCH --mail-type=ALL

module load CUDA/12.1.1
cd /home/alan.masutti/Project
# git stash save "Stashing changes for Job Execution"
# git checkout WMMA_MatMul_v1.0.1
# make

sudo /opt/shares/cuda/software/CUDA/12.1.1/bin/ncu --set full -f -o profiling/profile_all.ncu-rep /home/alan.masutti/Project/build/bin/main

# source scripts/executePython.sh utils/createCharts.py