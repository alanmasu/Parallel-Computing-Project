#include "matMul.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <mma.h>
#include <Utilities.h>


using namespace nvcuda;

// Dimensione del fragment
#ifndef TILE_SIZE
    #define TILE_SIZE 16
#endif

#define BLOCK_SIZE 32

#ifndef THREADS_PER_BLOCK
    #define THREADS_PER_BLOCK 32
#endif

#define SHARED_PAGE_COUNT 4

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

__device__ void wmmaMatrixMultiply(const half *As, const half *Bs, float *Cs, int N){
    int warpId = threadIdx.x / 32;  // Warp ID nel blocco
    int laneId = threadIdx.x % 32;  // Lane ID nel warp

    int row = (warpId / 2) * 16;  // Calcolo della riga del frammento
    int col = (warpId % 2) * 16;  // Calcolo della colonna del frammento

    // Controllo che l'accesso sia nei bound della matrice N×N
    if (row < N && col < N) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fragA;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fragB;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragC;

        wmma::fill_fragment(fragC, 0.0f);
        wmma::load_matrix_sync(fragA, As + row * N + col, N);
        wmma::load_matrix_sync(fragB, Bs + row * N + col, N);
        wmma::mma_sync(fragC, fragA, fragB, fragC);
        wmma::store_matrix_sync(Cs + row * N + col, fragC, N, wmma::mem_row_major);
    }
}

/**! 
    @brief Funzione per il caricamento di un blocco di matrice in shared memory

*/
template <typename T>
__device__ void loadBlockToShared(const T *a, T *As, int r, int c, int n){
    int threadID = threadIdx.x;
    int colInsideBlock = threadID % BLOCK_SIZE;
    int rowInsideBlock = threadID / BLOCK_SIZE * n;
    int blockColOffset = c * BLOCK_SIZE;
    int blockRowOffset = r * BLOCK_SIZE * n;
    int element = colInsideBlock + blockColOffset + rowInsideBlock + blockRowOffset;
    if(element < n * n){
        As[threadID] = a[element];
    }
}

/**! 
    @brief Funzione per copiare un blocco di matrice in global memory

*/
template <typename T>
__device__ void copyBlockToGlobal(const T *As, T *a, int r, int c, int n){
    int threadID = threadIdx.x;
    int colInsideBlock = threadID % BLOCK_SIZE;
    int rowInsideBlock = threadID / BLOCK_SIZE * n;
    int blockColOffset = c * BLOCK_SIZE;
    int blockRowOffset = r * BLOCK_SIZE * n;
    int element = colInsideBlock + blockColOffset + rowInsideBlock + blockRowOffset;
    if(element < n * n){
        a[element] = As[threadID];
    }
}

__global__ void matrixMultiplyTensorCore(const half *a, const half *b, float *d_c, int n) {
#ifdef TESTING_WMMA
    __shared__ half  As [SHARED_PAGE_COUNT * BLOCK_SIZE * BLOCK_SIZE];
    __shared__ half  Bs [SHARED_PAGE_COUNT * BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Cs [2 * BLOCK_SIZE * BLOCK_SIZE];

    //Copy data to shared memory
    As[threadIdx.x] = a[threadIdx.x];
    Bs[threadIdx.x] = b[threadIdx.x];
    Cs[threadIdx.x] = 0;
    // __syncthreads();

    wmmaMatrixMultiply(As, Bs, Cs, n);
    // blockMatrixMul(As, Bs, Cs, n);
    // blockMatrixMul(a, b, d_c, n);
    // copyBlockToGlobal(Cs, d_c, blockIdx.y, blockIdx.x, n);
    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0){
        printf("Matrice As:\n");
        printNMat(As, 4, 4, n);

        printf("Matrice Bs:\n");
        printNMat(Bs, 4, 4, n);

        printf("Matrice Cs:\n");
        printNMat(Cs, 4, 4, n);

        // //Global memory
        // printf("Matrice A:\n");
        // printNMat(a, 4, 4, n);

        // printf("Matrice B:\n");
        // printNMat(b, 4, 4, n);

        // printf("Matrice C:\n");
        // printNMat(d_c, 4, 4, n);
    }
#else
    __shared__ half  As [SHARED_PAGE_COUNT * BLOCK_SIZE * BLOCK_SIZE];
    __shared__ half  Bs [SHARED_PAGE_COUNT * BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float  Cs [SHARED_PAGE_COUNT * BLOCK_SIZE * BLOCK_SIZE];

    int numPages = n / (SHARED_PAGE_COUNT * 32);                        // 32 è il numero di elementi per pagina di shared memory
    int numBlocks = numPages < 1 ? n / BLOCK_SIZE : SHARED_PAGE_COUNT;  // Se la matrice è piccola non tutti i blocchi di shared memory sono utilizzati
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    for(int p = 0; p < numPages; ++p){
        // Carica la riga di blocchi della matrice A in shared memory
        int pageOffset = p * SHARED_PAGE_COUNT * BLOCK_SIZE * n;
        loadBlockToShared(a, As, pageOffset, blockRow, n);  
        loadBlockToShared(d_c, Cs, pageOffset, blockRow, n); 
        for(int k = 0; k < numBlocks; ++k){
            // Carica i blocchi in shared memory
            loadBlockToShared(b, Bs, blockCol, k, n);
            blockMatrixMul(As, Bs, Cs, n);
        }
        copyBlockToGlobal(Cs, d_c, pageOffset, blockRow, n);
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