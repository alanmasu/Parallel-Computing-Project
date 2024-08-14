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
module load cuda/12.1
srun /home/alan.masutti/Project/build/bin/main