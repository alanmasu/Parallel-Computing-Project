//Simple Matrix Multiplication whit cuBLAS

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>
#include <ctime>

#define N 64
#define BS 32

#warning "Testing mode"

__global__ void simpleKernel(int n, int bs){
    const int N_BLOCKS = n / bs;
    printf("blockDim.y: %d\n", blockDim.y);
    printf("blockDim.x: %d\n", blockDim.x);
    printf("gridDim.y: %d\n", gridDim.y);
    printf("gridDim.x: %d\n", gridDim.x);

    int bRow = blockIdx.y * blockDim.y + threadIdx.y;
    int bCol = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < N_BLOCKS; ++i){
        //Fatto dai tensor core
        printf("c'[%d][%d] = a[%d][%d] X b[%d][%d]\n", bRow, bCol, bRow, i, i, bCol);
    }
    __syncthreads();
    // con i cuda cores accumulo il risultato in c
    


}

int main(int argc, char **argv){
    dim3 threadsPerBlock(BS, BS);
    dim3 numBlocks(N / BS, N / BS);

    simpleKernel<<<numBlocks,threadsPerBlock>>>(N, BS); 
    cudaDeviceSynchronize();
    return 0;
}
