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
# git stash save "Stashing changes for Job Execution"
# git checkout WMMA_MatMul_v1.0.1
make test

srun /home/alan.masutti/Project/build/bin/test/test_wmma
srun /home/alan.masutti/Project/build/bin/test/test_shared
srun /home/alan.masutti/Project/build/bin/test/test_batched