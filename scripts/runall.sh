make
srun --partition=edu5 --nodes=1 --tasks=1 --gres=gpu:1 --cpus-per-task=1 --time=00:05:00 build/bin/main