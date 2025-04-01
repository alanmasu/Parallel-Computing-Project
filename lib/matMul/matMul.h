#ifndef MATMUL_H
#define MATMUL_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Dimensione del fragment
#ifndef TILE_SIZE
    #define TILE_SIZE 16
#endif

#define BLOCK_SIZE 32

#ifndef THREADS_PER_BLOCK
    #define THREADS_PER_BLOCK 32
#endif

#define SHARED_PAGE_COUNT 8

/*! 
    @brief      Function to perform matrix multiplication of two matrices A and B

    @details    Performing a batched matrix multiplication of two matrices A 
                and B and storing the result in matrix C. The matrices are of size N x N.

    @param      A[in] Pointer to the first matrix
    @param      B[in] Pointer to the second matrix
    @param      C[out] Pointer to the resultant matrix
    @param      N[in] Size of the row/column of the matrices
    @param      blockSize[in] Size of the block to be used for matrix multiplication
*/
void serialBatchedMatMul(const float *A, const float *B, float *C, int N, int blockSize);


/*! 
    @brief      [NOT IMPLEMENTED YET] Function to perform matrix multiplication of two matrices A and B

    @details    Performing a matrix multiplication of two matrices A and B and storing the result in matrix C. 
                The matrices are of size N x N.

    @param      A[in] Pointer to the first matrix
    @param      B[in] Pointer to the second matrix
    @param      C[out] Pointer to the resultant matrix
    @param      N[in] Size of the row/column of the matrices
*/
void serialMatMul(const float *A, const float *B, float *C, int N);


cudaError_t checkCudaError(cudaError_t err, const char *msg);
void checkCublasError(cublasStatus_t status, const char *msg);

/*! 
    @brief      Function to perform matrix multiplication of two matrices A and B using cuBLAS

    @details    Performing a matrix multiplication of two matrices A and B and storing the result in matrix C. 
                The matrices are of size N x N.

    @param      A[in] Pointer to the first matrix in ROW MAJOR format, the matrix is allocated in device memory
    @param      B[in] Pointer to the second matrix in ROW MAJOR format, the matrix is allocated in device memory
    @param      C[out] Pointer to the resultant matrix in ROW MAJOR format, the matrix is allocated in device memory
    @param      N[in] Size of the row/column of the matrices
    @param      milliseconds[out] Time taken to perform the matrix multiplication
    @param      TFLOPS[out] Theoretical peak FLOPS achieved during the matrix multiplication
*/
void cublasMatMul(const float *d_A, const float *d_B, float *d_C, int N, float* milliseconds, double* TFLOPS);


/*! 
    @brief      Function to convert a float matrix to a half matrix

    @details    This function converts a float Matrix A to a half Matrix B on the GPU, allocating 
                memory for B on the device. The matrices are of size N x N.

    @param      A[in] Pointer to the float matrix, allocated in host memory
    @param      h_B[out] Double pointer of the half matrix, will be allocated in host memory
    @param      B[out] Double pointer of the half matrix, will be allocated in device memory
    @param      N[in] Size of the row/column of the matrices

    @return     cudaError_t Error code

*/
cudaError_t convertFloatToHalf(const float *A, half **h_B, half **B, int N);

#ifdef WMMA_BATCHED
/**! 
    @brief Funzione per la moltiplicazione di blocchi BLOCK_SIZE x BLOCK_SIZE
    @details La funzione prende in ingresso i puntatori alle matrici e moltiplica i due blocchi 
    @param a [in] puntatore alla matrice A in shared memory
    @param b [in] puntatore alla matrice B in shared memory
    @param c [out] puntatore alla matrice C in shared memory
    @param n dimensione delle matrici
*/
__device__ void blockMatrixMul(const half *a, const half *b, float *c, int n);
#endif

/**! 
    @brief Funzione per il caricamento di un blocco di matrice in shared memory

    @param a puntatore alla matrice in global memory
    @param As puntatore alla matrice in shared memory
    @param r riga del blocco
    @param c colonna del blocco
    @param n dimensione della matrice

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
    @brief Funzione per azzerare un blocco di matrice in shared memory

    @param a puntatore alla matrice in shared memory
    @param BS dimensione del blocco
*/
template <typename T>
__device__ void clearBlockToShared(T *a, int BS = BLOCK_SIZE){
    int threadID = threadIdx.x;
    
    if(threadID < BS * BS){
        a[threadID] = 0;
    }
}

/**! 
    @brief Funzione per copiare un blocco di matrice in global memory

    @param As puntatore alla matrice in shared memory
    @param a puntatore alla matrice in global memory
    @param r riga del blocco
    @param c colonna del blocco
    @param n dimensione della matrice
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

/*!
    @brief      Function that launch a kernel to perform batched matrix multiplication of two matrices A and B using CUDA tensor cores
    @details    Performing a batched matrix multiplication of two matrices A and B and storing the result in matrix C. 
                The matrices are of size N x N.
    @param      A[in] Pointer to the first matrix in ROW MAJOR format, the matrix is allocated in DEVICE memory
    @param      B[in] Pointer to the second matrix in ROW MAJOR format, the matrix is allocated in DEVICE memory
    @param      C[out] Pointer to the resultant matrix in ROW MAJOR format, the matrix is allocated in DEVICE memory
    @param      N[in] Size of the row/column of the matrices
    @param      milliseconds[out] Time taken to perform the matrix multiplication
    @param      TFLOPS[out] Theoretical peak FLOPS achieved during the matrix multiplication
*/
void tensorCoreMatMul(const half *d_A, const half *d_B, float *d_C, int N, float* milliseconds, double* TFLOPS);

/*!
    @brief      Function to print an half matrix
    @details    This function prints a specified number of rows and culomns of an half matrix on the console 
*/
__host__ __device__ void printNMat(const half* mat, int rows, int cols, int N);

#endif // MATMUL_H