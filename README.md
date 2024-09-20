# Introduction to Parallel Computing - Project
## Abstract
Parallelizzation of a GEMM algorithm using Cuda and tensor cores. The project is developed in C++ and Cuda, and it is tested on a Nvidia Ampere A30 GPU. 

In this project I will compare performances of the GEMM algorithm obtained using the algorithm implemented by me whit and the same algorithm implemented by cuBLAS library. 

## Project structure
The project is structured as follows:
- `src/`: contains the source code of the project
- `lib/`: contains my implementation of the GEMM algorithm and all the utilities functions that I developed.
- `doc/`: contains the documentation of the project 
- `scripts/`: contains the scripts to run the project
- `utils/`: contains the utilities programs to get and analyze the results (such python scripts to plot the results)
- `run/`: contains the outputs of the scheduler
- `results/`: contains the results of the project

## How to run the project
### Prerequisites
First step is to load the cuda module:
```bash
module load cuda/12.1
```

### Compile
Then you can compile the project using the following command:
```bash
make
```

### Run
To run, in an interactive session, the project you can use the following command:
```bash
./scripts/exec.sh <path to the executable> 
```
that will become like this if no changes to Makefile are made:
```bash
./scripts/exec.sh build/bin/main
```

### Schedule
To schedule the project you can use the following command:
```bash
sbatch scripts/scheduler.sh
```

### Clean
To clean the scheduler output you can use the following command:
```bash
make clean
```

If you want to remove the object files you can use the following command:
```bash
make cleanbuild
```

If you want to remove the object files and the executable you can use the following command:
```bash
make cleanall
```

