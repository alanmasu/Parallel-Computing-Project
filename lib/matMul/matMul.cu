#include "matMul.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <mma.h>
#include <Utilities.h>

using namespace nvcuda;

void serialMatMul(const float *A, const float *B, float *C, int N){
    for(int r = 0; r < N; ++r){
        for(int c = 0; c < N; ++c){
            for(int k = 0; k < N; ++k){
                C[r * N + c] += A[r * N + k] * B[k * N + c];
            }
        }
    }
}

//Funzione di stampa, oveloaded from template function for half type
__host__ __device__ void printNMat(const half* mat, int rows, int cols, int N) {
    printf("Printing half matrix\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", __half2float(mat[i * N + j]));
        }
        printf("\n");
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


// Kernel per la moltiplicazione di matrici usando i Cuda Cores oppure i Tensor Cores
#ifndef WMMA_BATCHED
    /**!
        @brief Funzione per la moltiplicazione di matrici con Cuda Cores a blocchi di BLOCK_SIZE x BLOCK_SIZE
    */
    __global__ void matrixMultiplyTensorCore(const half *a, const half *b, float *c, int M) {
        #warning "Batched Matrix Multiplication not implemented yet"
    }
#else

#define WMMA_N 16

/**! 
    @brief      Funzione per la moltiplicazione di blocchi BLOCK_SIZE x BLOCK_SIZE
    @details    La funzione prende in ingresso i puntatori alle matrici e moltiplica i due blocchi 
    @param[in]  a puntatore alla matrice A in shared memory
    @param[in]  b puntatore alla matrice B in shared memory
    @param[out] c puntatore a due matrici BLOCK_SIZE x BLOCK_SIZE allocate in shared memory 
                  dove memorizzare i due risultati parziali

    @param n dimensione delle matrici
*/
__device__ void blockMatrixMul(const half *a, const half *b, float *c, int n){
    
    //Creazione dei fragment
    wmma::fragment<wmma::matrix_a, WMMA_N, WMMA_N, WMMA_N, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_N, WMMA_N, WMMA_N, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_N, WMMA_N, WMMA_N, float> acc_frag;
    
    // Calcolo del warp ID e del lane ID
    int warpId = threadIdx.x / 32;  // Warp ID nel blocco
    
    // // Calcolo del blocco
    int blockNumber = warpId / 8;       // Siccome ogni blocco 32x32 richiede 8 WMMA, il blocco lo si ottine dividendo per 8 il warp ID
    // int blockRow    = blockNumber / 2;  // Calcolo della riga del blocco
    // int blockCol    = blockNumber % 2;  // Calcolo della colonna del blocco

    // Calcolo del tile
    int tileNumber = warpId % 4;        // Calcolo del numero di tile, ad ogni warp è assegnato un tile, abbiamo 2 operazioni per tile e 4 tile, da qui l'esigenza di 8 warp
    int tilePage   = (warpId % 8) / 4;  // Calcolo della pagina del tile, per ogni tile abbiamo bisogno di 2 computazioni; 
    int tileRow    = tileNumber / 2;    // Calcolo della riga del tile
    int tileCol    = tileNumber % 2;    // Calcolo della colonna del tile

    #ifdef PRINT_DEBUG
        #warning "Debug print enabled"

        int laneId = threadIdx.x % 32;  // Lane ID nel warp
        // if(laneId == 0){
        //     printf("Warp ID: %d, Block Number: %d, Tile Number: %d, Tile Page: %d, Tile Row: %d, Tile Col: %d\n", warpId, blockNumber, tileNumber, tilePage, tileRow, tileCol);
        // }
    #endif

    if(tileRow * WMMA_N < BLOCK_SIZE && tileCol * WMMA_N < BLOCK_SIZE && blockNumber == 0){
        int cRow = tileRow * WMMA_N * n;
        int cCol = tileCol * WMMA_N;
        int cPage = tilePage * BLOCK_SIZE * BLOCK_SIZE;

        int aCol = tilePage * WMMA_N;
        int bRow = tilePage * WMMA_N * n;

        #ifdef PRINT_DEBUG
            if(laneId == 0){
                // printf("Warp %d of block [%d][%d] is computing: c[%d][%d][%d] = a[%d][%d] * b[%d][%d]\n", warpId, blockIdx.x, blockIdx.y, cPage, cRow, cCol, cRow, aCol, bRow, cCol);
                printf("Warp %d of block [%d][%d] is computing: c[%d][%d][%d] = a[%d][%d] * b[%d][%d] (a[0,0] = %f, b[0,0] = %f)\n", warpId, blockIdx.x, blockIdx.y, cPage, cRow, cCol, cRow, aCol, bRow, cCol, __half2float(a[0]), 0.0);
            }
        #endif

        // Carica i fragment
        wmma::load_matrix_sync(a_frag,   a + cRow + aCol, BLOCK_SIZE);
        wmma::load_matrix_sync(b_frag,   b + bRow + cCol, BLOCK_SIZE);
        // wmma::load_matrix_sync(acc_frag, c + cRow + cCol, BLOCK_SIZE, wmma::mem_row_major);
        wmma::fill_fragment(acc_frag, 0.0f);

        // Moltiplica i fragment
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        // Memorizza il risultato
        wmma::store_matrix_sync(c + cRow + cCol + cPage, acc_frag, BLOCK_SIZE, wmma::mem_row_major);
        if(cRow + cCol + cPage >= 2 * BLOCK_SIZE * BLOCK_SIZE){
            printf("Thread %d is writing after bounds", threadIdx.x);
        }
    }
}

__global__ void matrixMultiplyTensorCore(const half *a, const half *b, float *d_c, int n) {
    const int blockSize = BLOCK_SIZE * BLOCK_SIZE;

    __shared__ half  As [SHARED_PAGE_COUNT * blockSize];
    __shared__ half  Bs [SHARED_PAGE_COUNT * blockSize];
    // __shared__ float Cs [BLOCK_SIZE * BLOCK_SIZE];
    // __shared__ float Acc [BLOCK_SIZE * BLOCK_SIZE];
    extern __shared__ float sharedMem[]; 
    float *Cs = sharedMem; // La matrice Cs è allocata in shared memory
    float *Acc = sharedMem + 2 * (blockSize); // La matrice Acc è allocata in shared memory

    const int numPages = n / (SHARED_PAGE_COUNT * 32) || 1;                   // 32 è il numero di elementi per pagina di shared memory
    const int numBlocks = numPages < 2 ? n / BLOCK_SIZE : SHARED_PAGE_COUNT;  // Se la matrice è piccola non tutti i blocchi di shared memory sono utilizzati
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    
    // __syncthreads();
    // Inizializza la matrice Acc a zero
    clearBlockToShared(Cs);
    clearBlockToShared(Cs + blockSize);
    clearBlockToShared(Acc);
    // Ciclo sulle pagine
    for(int p = 0; p < numPages; ++p){
        // Carica la riga di blocchi della matrice A in shared memory
        const int pageOffset = p * SHARED_PAGE_COUNT;  // Calcola l'offset della pagina in numero di blocchi
        
        // Carica il blocco dalla matrice C in shared memory
        // clearBlockToShared(Cs + blockSize);
        loadBlockToShared(d_c, Cs, blockRow, blockCol, n);
        __syncthreads();
        // Ciclo all'interno della pagina
        for(int k = 0; k < numBlocks; ++k){ 

            // Carica i blocchi in shared memory
            loadBlockToShared(a, As, blockRow, pageOffset + k, n);  
            loadBlockToShared(b, Bs, pageOffset + k, blockCol, n);
            
            __syncthreads();    // Attendi il caricamento dei blocchi in shared memory
            
            if(blockIdx.x ==1 && blockIdx.y == 0 && threadIdx.x == 0){
                printf("As whit k = %d - Block[%d, %d]:\n", k, blockRow, pageOffset + k);
                printNMat(As, 4, 4, BLOCK_SIZE);
                printf("Bs whit k = %d - Block[%d, %d]:\n", k, pageOffset + k, blockCol);
                printNMat(Bs, 4, 4, BLOCK_SIZE);
                printf("\n");
            }
            __syncthreads();    // Attendi la fine della stampa

            blockMatrixMul(As, Bs, Cs, BLOCK_SIZE);
            __syncthreads();    // Attendi i thread dei primi 8 warp per completare la computazione

            if(blockIdx.x ==1 && blockIdx.y == 0 && threadIdx.x == 0){
                printf("Acc prima della somma:\n");
                printNMat(Acc, 4, 4, BLOCK_SIZE);
                printf("Partial Cs whit k = %d:\n", k);
                printNMat(Cs, 4, 4, BLOCK_SIZE);
            }
            __syncthreads();    // Attendi il completamento della stampa

            Acc[threadIdx.x] = Acc[threadIdx.x] + Cs[threadIdx.x] + Cs[threadIdx.x + blockSize];
            __syncthreads();    // Attendi il completamento della somma
            
            clearBlockToShared(Cs);
            __syncthreads();

            if(blockIdx.x ==1 && blockIdx.y == 0 && threadIdx.x == 0){
                printf("Matrice Acc after sum with k = %d:\n", k);
                printNMat(Acc, 4, 4, BLOCK_SIZE);
                printf("\n");
            }
            __syncthreads();
        }

        copyBlockToGlobal(Acc, d_c, blockRow, blockCol, n);
        __syncthreads();

    }
}
#endif // WMMA_BATCHED


cudaError_t convertFloatToHalf(const float *A, half **h_B,  half **B, int N){
    if(h_B == NULL){
        printf("[ERROR]: unable to convert float to half caused by h_B NULL pointer\n");
        return cudaErrorInvalidValue;
    }
    *h_B = (half*)malloc(N * N * sizeof(half));
    if(*h_B == NULL){
        printf("[ERROR]: unable to convert float to half caused by h_B allocation\n");
        return cudaErrorMemoryAllocation;
    }
    if(B == NULL){
        printf("[ERROR]: unable to convert float to half caused by B NULL pointer\n");
        return cudaErrorInvalidValue;
    }
    if(A == NULL){
        printf("[ERROR]: unable to convert float to half caused by A NULL pointer\n");
        return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaMalloc((void **)B, N * N * sizeof(half));
    if(h_B != NULL && err == cudaSuccess){
        for(int i = 0; i < N * N; ++i){
            (*h_B)[i] = __float2half(A[i]);
        }
        err = cudaMemcpy(*B, *h_B, N * N * sizeof(half), cudaMemcpyHostToDevice);
        // free(h_B);
    }else{
        printf("[ERROR]: unable to allocate memory for half matrix\n");
    }
    return err;
}

// Funzione per la moltiplicazione di matrici su GPU con Tensor Cores e WMMA
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
    matrixMultiplyTensorCore<<<numBlocks, threadsPerBlock, 3 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float)>>>(d_A, d_B, d_C, n);
    
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
