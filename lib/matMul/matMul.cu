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

#define BLOCK_SIZE 32

#ifndef THREADS_PER_BLOCK
    #define THREADS_PER_BLOCK 32
#endif


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
void cublasMatMul(const float *d_A, const float *d_B, float *d_C, int n, float* milliseconds, double* TFLOPS){
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
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n),
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
        double FLOPs = 2.0 * n * n * n;

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
#ifndef WMMA_BATCHED
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
#else
#define WMMA_N 16


/**! 
    @brief Funzione per la moltiplicazione di blocchi BLOCK_SIZE x BLOCK_SIZE
    @details La funzione prende in ingresso i puntatori alle matrici e moltiplica i due blocchi 
    @param a [in] puntatore alla matrice A in shared memory
    @param b [in] puntatore alla matrice B in shared memory
    @param c [out] puntatore alla matrice C in shared memory
    @param n dimensione delle matrici
*/
__device__ void blockMatrixMul(const half *a, const half *b, float *c, int n){
    
    //Creazione dei fragment
    wmma::fragment<wmma::matrix_a, WMMA_N, WMMA_N, WMMA_N, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_N, WMMA_N, WMMA_N, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_N, WMMA_N, WMMA_N, float> acc_frag;
    
    // Carica i fragment
    wmma::load_matrix_sync(a_frag,   a, BLOCK_SIZE);
    wmma::load_matrix_sync(b_frag,   b, BLOCK_SIZE);
    wmma::load_matrix_sync(acc_frag, c, BLOCK_SIZE, wmma::mem_row_major);

    // Moltiplica i fragment
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // Memorizza il risultato
    wmma::store_matrix_sync(c, acc_frag, BLOCK_SIZE, wmma::mem_row_major);
}


__global__ void matrixMultiplyTensorCore(const half *a, const half *b, float *d_c, int n) {
    //TODO: Implementare la gestione della shared memory
#ifdef TESTING_WMMA
    // __shared__ half As [BLOCK_SIZE * BLOCK_SIZE];
    // __shared__ half Bs [BLOCK_SIZE * BLOCK_SIZE];

    // //Copy data to shared memory
    // As[threadIdx.x] = a[threadIdx.x];
    // Bs[threadIdx.x] = b[threadIdx.x];

    blockMatrixMul(a, b, d_c, n);

#else
    int numBlocks = n / BLOCK_SIZE;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    for(int k = 0; k < numBlocks; ++k){
        // Moltiplica ed accumula i blocchi in C, 
        //   k fa muovere il blocco lungo le colonne di A e le righe di B
            // 
                // int threadID = threadIdx.x;
                // int colInsideBlock = threadID % BLOCK_SIZE;
                // int rowInsideBlock = threadID / BLOCK_SIZE * n;
                // int blockColOffset = blockCol * BLOCK_SIZE;
                // int blockRowOffset = blockRow * BLOCK_SIZE * n;

                // int cElement = colInsideBlock + blockColOffset + rowInsideBlock + blockRowOffset;
        blockMatrixMul(a, b, c, n);

        //l'accumulo credo lo si possa fare nel fragment caricandolo con C e non con 0, ammesso che C sia inizializzato a 0
    }
#endif
}
#endif // WMMA_BATCHED

cudaError_t convertFloatToHalf(const float *A, half **B, int N){
    half* h_B = (half*)malloc(N * N * sizeof(half));
    if(B == NULL){
        printf("[ERROR]: unable to convert float to half caused by B NULL pointer\n");
        return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaMalloc((void **)B, N * N * sizeof(half));
    if(h_B != NULL && err == cudaSuccess){
        for(int i = 0; i < N * N; ++i){
            h_B[i] = __float2half(A[i]);
        }
        err = cudaMemcpy(*B, h_B, N * N * sizeof(half), cudaMemcpyHostToDevice);
        free(h_B);
    }else{
        printf("[ERROR]: unable to allocate memory for half matrix\n");
    }
    return err;
}

// Funzione per la moltiplicazione di matrici su GPU con Tensor Cores e WMMA
#ifndef WMMA_BATCHED
void tensorCoreMatMul(const half *d_A, const half *d_B, float *d_C, int n, float* milliseconds, double* TFLOPS) {
    
    if(d_A == NULL || d_B == NULL || d_C == NULL){
        printf("[ERROR]: unable to perform MatMul caused by NULL pointers\n");
        if(milliseconds != NULL && TFLOPS != NULL){
            *milliseconds = -1;
            *TFLOPS = -1;
        }
        return;
    }

    // Configura la griglia e i blocchi per la computazione
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(n / TILE_SIZE, n / TILE_SIZE);

    // Misurazione del tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvia il timer
    cudaEventRecord(start, 0);
    
    // Esegui il kernel per la moltiplicazione di matrici con Tensor Cores e WMMA
    matrixMultiplyTensorCore<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Ferma il timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calcola il tempo impiegato
    if(milliseconds != NULL){
        *milliseconds = 0;
        cudaEventElapsedTime(milliseconds, start, stop);
    }
    
    // Numero totale di operazioni in virgola mobile (FLOP)
    double FLOPs = 2.0 * n * n * n;

    // Calcolo dei TFLOPS
    if(milliseconds != NULL && TFLOPS != NULL){
        *TFLOPS = (FLOPs / (*milliseconds / 1000.0)) / 1e12;
    }else{
        printf("some pointers are NULL\n");
    }
}
#else
void tensorCoreMatMul(const half *d_A, const half *d_B, float *d_C, int n, float* milliseconds, double* TFLOPS) {
    
    if(d_A == NULL || d_B == NULL || d_C == NULL){
        printf("[ERROR]: unable to perform MatMul caused by NULL pointers\n");
        if(milliseconds != NULL && TFLOPS != NULL){
            *milliseconds = -1;
            *TFLOPS = -1;
        }
        return;
    }

    // Configura la griglia e i blocchi per la computazione
    dim3 threadsPerBlock(THREADS_PER_BLOCK * THREADS_PER_BLOCK);
    dim3 numBlocks(n / BLOCK_SIZE, n / BLOCK_SIZE);

    // Misurazione del tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvia il timer
    cudaEventRecord(start, 0);
    
    // Esegui il kernel per la moltiplicazione di matrici con Tensor Cores e WMMA
    matrixMultiplyTensorCore<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Ferma il timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calcola il tempo impiegato
    if(milliseconds != NULL){
        *milliseconds = 0;
        cudaEventElapsedTime(milliseconds, start, stop);
    }
    
    // Numero totale di operazioni in virgola mobile (FLOP)
    double FLOPs = 2.0 * n * n * n;

    // Calcolo dei TFLOPS
    if(milliseconds != NULL && TFLOPS != NULL){
        *TFLOPS = (FLOPs / (*milliseconds / 1000.0)) / 1e12;
    }else{
        printf("[ERROR]: some pointers are NULL\n");
    }
}
#endif // WMMA_BATCHED