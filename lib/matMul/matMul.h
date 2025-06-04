/**
    @file      matMul.h
    @author    Alan Masutti (@alamasu on GitHub)

    @brief     Header file for matrix multiplication functions
    @details   This file contains the declarations of functions for performing matrix multiplication using CUDA
                and cuBLAS. It includes functions for serial matrix multiplication, cuBLAS matrix multiplication,
                and tensor core matrix multiplication. It also includes utility functions for print and 
                converting matrices between different formats.
*/

#ifndef MATMUL_H
#define MATMUL_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Dimensione del fragment
#ifndef TILE_SIZE
    /// @brief Dimension of the tile used in matrix multiplication
    #define TILE_SIZE 16 
#endif

/// @brief Dimension of the block used in matrix multiplication
#define BLOCK_SIZE 32

#ifndef THREADS_PER_BLOCK
    /// @brief The root of number of threads per block used in matrix multiplication
    #define THREADS_PER_BLOCK 32
#endif

/// @brief Number of blocks loaded in shared memory [NOT EFFECTIVE]
#define SHARED_PAGE_COUNT 8


/** 
    @brief      [NOT IMPLEMENTED YET] Function to perform matrix multiplication of two matrices A and B

    @details    Performing a matrix multiplication of two matrices A and B and storing the result in matrix C. 
                The matrices are of size N x N.

    @param[in]  A Pointer to the first matrix
    @param[in]  B Pointer to the second matrix
    @param[out] C Pointer to the resultant matrix
    @param[in]  N Size of the row/column of the matrices
*/
void serialMatMul(const float *A, const float *B, float *C, int N);

/**
    @brief      Function to get CUDA error message
*/
cudaError_t checkCudaError(cudaError_t err, const char *msg);

/**
    @brief      Function to get cuBLAS error message
*/
void checkCublasError(cublasStatus_t status, const char *msg);

/** 
    @brief      Function to perform matrix multiplication of two matrices A and B using cuBLAS

    @details    Performing a matrix multiplication of two matrices A and B and storing the result in matrix C. 
                The matrices are of size N x N.

    @param[in]      d_A Pointer to the first matrix in ROW MAJOR format, the matrix is allocated in device memory
    @param[in]      d_B Pointer to the second matrix in ROW MAJOR format, the matrix is allocated in device memory
    @param[out]     d_C Pointer to the resultant matrix in ROW MAJOR format, the matrix is allocated in device memory
    @param[in]      N Size of the row/column of the matrices
    @param[out]     milliseconds Time taken to perform the matrix multiplication
    @param[out]     TFLOPS Theoretical peak FLOPS achieved during the matrix multiplication
*/
void cublasMatMul(const float *d_A, const float *d_B, float *d_C, int N, float* milliseconds, double* TFLOPS);


/** 
    @brief      Function to convert a float matrix to a half matrix.
    @details    This function converts a float Matrix A to a half Matrix B on the GPU, allocating 
                memory for B on the device. The matrices are of size N x N.

    @param[in]  A Pointer to the float matrix, allocated in host memory
    @param[out] h_B Double pointer of the half matrix, will be allocated in host memory
    @param[out] B Double pointer of the half matrix, will be allocated in device memory
    @param[in]  N Size of the row/column of the matrices

    @return     cudaError_t Error code

*/
cudaError_t convertFloatToHalf(const float *A, half **h_B, half **B, int N);

#ifdef WMMA_BATCHED
/**
    @brief      Function to multiply blocks of size BLOCK_SIZE x BLOCK_SIZE
    @details    The function takes pointers to matrices A and B and multiplies the two blocks.
    @param[in]  a pointer to the matrix A in shared memory
    @param[in]  b pointer to the matrix B in shared memory
    @param[out] c pointer to two BLOCK_SIZE x BLOCK_SIZE matrices allocated in shared memory
                   where the two partial results are stored
    @param n dimension of the matrices
    @note       The function uses WMMA (Warp Matrix Multiply and Accumulate) to perform the multiplication
                using Tensor Cores. It assumes that the matrices are in half precision (half data type).
*/
__device__ void blockMatrixMul(const half *a, const half *b, float *c, int n);
#endif

/**
    @brief      Function to load a block of matrix from global memory to shared memory

    @details    The function loads a block (BLOCK_SIZE x BLOCK_SIZE) of a matrix from global memory to shared memory.
                The block is specified by the row (r) and column (c) indices. The function assumes that the matrix is in row-major format.
    @param[in]  a Pointer to the matrix in global memory
    @param[out] As Pointer to the matrix in shared memory
    @param[in]  r Row index of the block
    @param[in]  c Column index of the block
    @param[in]  n Size of the row/column of the matrices
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

/**
    @brief Function to clear a block of matrix in shared memory
    @details This function clears a block of matrix in shared memory by setting all elements to zero.

    @param a Pointer to the matrix in shared memory
    @param BS Size of the block. Default is BLOCK_SIZE
*/
template <typename T>
__device__ void clearBlockToShared(T *a, int BS = BLOCK_SIZE){
    int threadID = threadIdx.x;
    
    if(threadID < BS * BS){
        a[threadID] = 0;
    }
}

/**
    @brief Function to copy a block of matrix from shared memory to global memory
    @details This function copies a block of matrix from shared memory to global memory. The block is specified by the row (r) and column (c) indices.
             The function assumes that the matrix is in row-major format.
    @param[in] As Pointer to the matrix in shared memory
    @param[out] a Pointer to the matrix in global memory
    @param[in] r Row index of the block
    @param[in] c Column index of the block
    @param[in] n Size of the row/column of the matrices
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

/**
    @brief      Function that launch a kernel to perform batched matrix multiplication of two matrices A and B using CUDA tensor cores
    @details    Performing a batched matrix multiplication of two matrices A and B and storing the result in matrix C. 
                The matrices are of size N x N.
    @param[in]  d_A Pointer to the first matrix in ROW MAJOR format, the matrix is allocated in DEVICE memory
    @param[in]  d_B Pointer to the second matrix in ROW MAJOR format, the matrix is allocated in DEVICE memory
    @param[out] d_C Pointer to the resultant matrix in ROW MAJOR format, the matrix is allocated in DEVICE memory
    @param[in]  N Size of the row/column of the matrices
    @param[out] milliseconds Time taken to perform the matrix multiplication
    @param[out] TFLOPS Theoretical peak FLOPS achieved during the matrix multiplication
*/
void tensorCoreMatMul(const half *d_A, const half *d_B, float *d_C, int N, float* milliseconds, double* TFLOPS);

/**
    @brief      Function to print an half matrix
    @details    This function prints a specified number of rows and culomns of an half matrix on the console 
*/
__host__ __device__ void printNMat(const half* mat, int rows, int cols, int N);

#endif // MATMUL_H