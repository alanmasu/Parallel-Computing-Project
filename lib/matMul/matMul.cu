#include "matMul.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <mma.h>

using namespace nvcuda;

// Dimensione del fragment
#ifndef TILE_SIZE
    #define TILE_SIZE 16
#endif

void serialBatchedMatMul(const float *A, const float *B, float *C, int N, int blockSize){
    //int blockCount = N / blockSize;
    // Initialize the result matrix at zeros
    //memset(C, 0, N * N * sizeof(float));
    // Loop over destination blocks rows
    
}

void serialMatMul(const float *A, const float *B, float *C, int N){
    for(int r = 0; r < N; ++r){
        for(int c = 0; c < N; ++c){
            for(int k = 0; k < N; ++k){
                C[r * N + c] += A[r * N + k] * B[k * N + c];
            }
        }
    }
}

// Funzione helper per il controllo degli errori CUDA
cudaError_t checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        //exit(EXIT_FAILURE);
    }
    return err;
}

// Funzione helper per il controllo degli errori cuBLAS
void checkCublasError(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: %s\n", msg);
        //exit(EXIT_FAILURE);
    }
}

// Funzione per la moltiplicazione di matrici su GPU con cuBLAS
void cublasMatMul(const float *d_A, const float *d_B, float *d_C, int N, float* milliseconds, double* TFLOPS){
    if(d_A != NULL && d_B != NULL && d_C != NULL){
        float alpha = 1.0f, beta = 0.0f;

        // Inizializzazione dell'handle cuBLAS
        cublasHandle_t handle;
        checkCublasError(cublasCreate(&handle), "Inizializzazione cuBLAS");

        // Misurazione del tempo
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Avvia il timer
        cudaEventRecord(start, 0);

        // Esegui la moltiplicazione di matrici (C = alpha * A * B + beta * C) sulla GPU
        checkCublasError(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N),
            "Moltiplicazione di matrici"
        );

        // Ferma il timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calcola il tempo impiegato
        if(milliseconds != NULL){
            *milliseconds = 0;
            cudaEventElapsedTime(milliseconds, start, stop);
        }
        
        // Numero totale di operazioni in virgola mobile (FLOP)
        double FLOPs = 2.0 * N * N * N;

        // Calcolo dei TFLOPS
        if(milliseconds != NULL && TFLOPS != NULL){
            *TFLOPS = (FLOPs / (*milliseconds / 1000.0)) / 1e12;
        }else{
            printf("some pointers are NULL\n");
        }

        // Distruggi l'handle cuBLAS
        cublasDestroy(handle);
    }else{
        printf("unable to perform MatMul caused by NULL pointers\n");
        if(milliseconds != NULL && TFLOPS != NULL){
            *milliseconds = -1;
            *TFLOPS = -1;
        }
    }
}


// Kernel per la moltiplicazione di matrici usando Tensor Cores e WMMA
__global__ void matrixMultiplyTensorCore(const half *a, const half *b, float *c, int M) {
    // Matrici WMMA (warped matrix multiply and accumulate)
    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag;

    // Coordinate di blocco e thread
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Ogni warp calcola una tile C del risultato
    wmma::fill_fragment(c_frag, 0.0f);

    // Itera sui blocchi di K (MATRIX_SIZE / TILE_SIZE)
    for (int tileIdx = 0; tileIdx < M / TILE_SIZE; ++tileIdx) {
        // Carica una tile da A e B
        wmma::load_matrix_sync(a_frag, a + blockRow * TILE_SIZE * M + tileIdx * TILE_SIZE, M);
        wmma::load_matrix_sync(b_frag, b + tileIdx * TILE_SIZE * M + blockCol * TILE_SIZE, M);

        // Esegui la moltiplicazione delle tile
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Scrivi il risultato nella matrice C
    wmma::store_matrix_sync(c + blockRow * TILE_SIZE * M + blockCol * TILE_SIZE, c_frag, M, wmma::mem_row_major);
}


// Funzione per la moltiplicazione di matrici su GPU con Tensor Cores e WMMA
void tensorCoreMatMul(const float *h_A, const float *h_B, float *d_C, int N, float* milliseconds, double* TFLOPS) {
    // Allocazione delle matrici A e B sul device
    int device_matrix_size = N * N * sizeof(half);
    half *A, *B, *h_Ahalf, *h_Bhalf;
    
    cudaError_t err1 = checkCudaError(cudaMalloc((void **)&A, device_matrix_size), "Allocazione matrice A su GPU");
    cudaError_t err2 = checkCudaError(cudaMalloc((void **)&B, device_matrix_size), "Allocazione matrice B su GPU");
    
    //Conversione delle matrici in half
    if( A != NULL && B != NULL && err1 == cudaSuccess && err2 == cudaSuccess){
        for(int i = 0; i < N * N; ++i){
            h_Ahalf[i] = __float2half(h_A[i]);
            h_Bhalf[i] = __float2half(h_B[i]);
        }
        checkCudaError(cudaMemcpy(A, h_Ahalf, device_matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
        checkCudaError(cudaMemcpy(B, h_Bhalf, device_matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");

    }else{
        printf("unable to perform MatMul caused by NULL pointers\n");
        if(milliseconds != NULL && TFLOPS != NULL){
            *milliseconds = -1;
            *TFLOPS = -1;
        }
        return;
    }
    // Configura la griglia e i blocchi per la computazione
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(N / TILE_SIZE, N / TILE_SIZE);

    // Misurazione del tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvia il timer
    cudaEventRecord(start, 0);
    
    // Esegui il kernel per la moltiplicazione di matrici con Tensor Cores e WMMA
    matrixMultiplyTensorCore<<<numBlocks, threadsPerBlock>>>(A, B, d_C, N);
    
    // Ferma il timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calcola il tempo impiegato
    if(milliseconds != NULL){
        *milliseconds = 0;
        cudaEventElapsedTime(milliseconds, start, stop);
    }
    
    // Numero totale di operazioni in virgola mobile (FLOP)
    double FLOPs = 2.0 * N * N * N;

    // Calcolo dei TFLOPS
    if(milliseconds != NULL && TFLOPS != NULL){
        *TFLOPS = (FLOPs / (*milliseconds / 1000.0)) / 1e12;
    }else{
        printf("some pointers are NULL\n");
    }

    // Libera la memoria sul device
    cudaFree(A);
    cudaFree(B);

    // Libera la memoria sull'host
    free(h_Ahalf);
    free(h_Bhalf);

}
