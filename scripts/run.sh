#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test-project
#SBATCH --output=run/test-project-%j.out
#SBATCH --error=run/test-project-%j.err

#SBATCH --mail-user=alan.masutti@studenti.unitn.it
#SBATCH --mail-type=ALL

module load cuda/12.1
cd /home/alan.masutti/Project
git stash save "Stashing changes for Job Execution"
git checkout MatMul_v1.1.1
make
srun /home/alan.masutti/Project/build/bin/main