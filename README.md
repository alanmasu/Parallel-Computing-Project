# Introduction to Parallel Computing - Project
## Abstract
Parallelizzation of a GEMM algorithm using Cuda and tensor cores. The project is developed in C++ and Cuda, and it is tested on a Nvidia Ampere A30 GPU. 

In this project I will compare performances of the GEMM algorithm obtained using the algorithm implemented by me whit and the same algorithm implemented by cuBLAS library. 

## Project structure
The project folder is structured as follows:
```shell
├── build
│   ├── obj
│   │   ├── src
│   │   │   ├── <source_name>.o
│   │   │   └── main.o
│   │   ├── lib
│   │   │   └── <lib_name>
│   │   │       └── <lib_name>.o
│   │   └── test
│   │       └── test_<test_name>
│   │           └── test_<test_name>.o
│   └── bin
│       ├── test
│       │   └── test_<test_name>
│       └── main
├── lib
│   └── <lib_name>
│       ├── <lib_name>.cu
│       └── <lib_name>.h
├── src
│   └── main.cu
├── test
│   └── test_<test_name>
│       └── test_<test_name>.cu
└── Makefile
```

- `src/`: contains the source code of the project
- `lib/`: contains my implementation of the GEMM algorithm and all the utilities functions that I developed.
- `scripts/`: contains the scripts to run the project
- `utils/`: contains the utilities programs to get and analyze the results (such python scripts to plot the results)
- `run/`: contains the outputs of the scheduler
- `results/`: contains the results of the project
- `build/`: contains the object files and the executable of the project
- `test/`: contains the tests files of the project. Those files contains `main()` functions for testing libraries and utilities functions, they will be compiled in the `build/bin/test` folder and linked to libraries object files
- `doc/`: [for future] contains the documentation of the project 

## How to run the project
### Prerequisites
First step is to load the cuda module:
```bash
module load cuda/12.1
```

### Compile
You can compile the project using the following command:
```bash
make
```
This will compile the libraries and the src files and put the executable in the `build/bin` folder as `main`.

#### Compiling tests
If you want to compile the tests you can use the following command:
```bash
make test
```
This will compile the tests and put the executables in the `build/bin/test` folder. It creates a test executable for each test folder in the `test` directory. Test are linked to the libraries object files.

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
sbatch scripts/run.sh
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

