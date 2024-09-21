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
__global__ void matrixMultiplyTensorCore(const half *a, const half *b, float *d_c, int n, int bs) {
    //TODO: Implementare la gestione della shared memory

    //Creazione dei fragment
    wmma::fragment<wmma::matrix_a, WMMA_N, WMMA_N, WMMA_N, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_N, WMMA_N, WMMA_N, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_N, WMMA_N, WMMA_N, float> acc_frag;
    // wmma::fragment<wmma::accumulator, WMMA_N, WMMA_N, WMMA_N, float> c_frag;

    //Coordinate di blocco
    int bRow = blockIdx.y * blockDim.y;
    int bCol = blockIdx.x * blockDim.x;

    //Numero di blocchi
    int numBlocks = n / bs;

    //Coordinate di partenza del blocco in C
    int cStartingRow = bRow * n * bs;
    int cStartingCol = bCol * bs;
    //Moltiplico i blocchi BSxBS tra di loro
    for(int k = 0; k < numBlocks; ++k){
        //extern __shared__ float c_temp[];
        //Moltiplico all'interno dei blocchi BSxBS con i tensor cores
        //Simple batched matrix multiplication
        int aStartingCol = k * bs;
        int bStartingRow = k * bs * n;
        // printf("c[%d][%d] = a[%d][%d] * b[%d][%d]\n", cStartingRow, cStartingCol, cStartingRow, aStartingCol, bStartingRow, cStartingCol);
        // //Creo una matrice temporanea per il risultato (è una matrice BS x BS) moltiplicando i 
        // // rispettivi blocchi di matrici di dimensione WMMA_N x WMMA_N
        // extern __shared__ float c_temp[];
        for(int r = 0; r < bs/WMMA_N; ++r){
            for(int c = 0; c < bs/WMMA_N; ++c){
                //Coordinate di partenza del blocco in C
                int cCol = cStartingCol + c * WMMA_N;
                int cRow = cStartingRow + r * n * WMMA_N;
                //Carico il fragment di accumulazione
                wmma::load_matrix_sync(acc_frag, d_c + cRow + cCol, n, wmma::mem_row_major);
                for(int i = 0; i < bs/WMMA_N; ++i){
                    int aCol = aStartingCol + i * WMMA_N;
                    int bRow = bStartingRow + i * n * WMMA_N;
                    // printf("c[%d][%d] = a[%d][%d] * b[%d][%d] -> k=%d\n", cRow/n, cCol, cRow/n, aCol, bRow/n, cCol, k);

                    //Carico le matrici
                    wmma::load_matrix_sync(a_frag, a + cStartingRow + aCol, n);
                    wmma::load_matrix_sync(b_frag, b + bRow + cStartingCol, n);

                    // Moltiplico le matrici
                    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                wmma::store_matrix_sync(d_c + cRow + cCol, acc_frag, n, wmma::mem_row_major);
            }
        }
        // __syncthreads();
        //Copio e accumulo il risultato nella matrice C
        // for(int r = 0; r < bs; ++r){
        //     for(int c = 0; c < bs; ++c){
        //         h_c[(cStartingRow + r * n) + cStartingCol + c] += c_temp[r * bs + c];
        //     }
        // }
    }
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
void tensorCoreMatMul(const half *d_A, const half *d_B, float *d_C, int n, int bs, float* milliseconds, double* TFLOPS) {
    
    if(d_A == NULL || d_B == NULL || d_C == NULL){
        printf("[ERROR]: unable to perform MatMul caused by NULL pointers\n");
        if(milliseconds != NULL && TFLOPS != NULL){
            *milliseconds = -1;
            *TFLOPS = -1;
        }
        return;
    }

    // Configura la griglia e i blocchi per la computazione
    dim3 threadsPerBlock(32);
    dim3 numBlocks(n / bs, n / bs);

    // Misurazione del tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvia il timer
    cudaEventRecord(start, 0);
    
    // Esegui il kernel per la moltiplicazione di matrici con Tensor Cores e WMMA
    matrixMultiplyTensorCore<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n, bs);
    // matrixMultiplyTensorCore<<<numBlocks, 1>>>(d_A, d_B, d_C, n, bs);
    
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