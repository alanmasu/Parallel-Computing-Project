#ifndef MATMUL_H
#define MATMUL_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    @param      B[out] Double pointer of the half matrix, will be allocated in device memory
    @param      N[in] Size of the row/column of the matrices

    @return     cudaError_t Error code

*/

cudaError_t convertFloatToHalf(const float *A, half **B, int N);

/*!
    @brief      Function to perform batched matrix multiplication of two matrices A and B using CUDA tensor cores
    @details    Performing a batched matrix multiplication of two matrices A and B and storing the result in matrix C. 
                The matrices are of size N x N.
    @param      A[in] Pointer to the first matrix in ROW MAJOR format, the matrix is allocated in DEVICE memory
    @param      B[in] Pointer to the second matrix in ROW MAJOR format, the matrix is allocated in DEVICE memory
    @param      C[out] Pointer to the resultant matrix in ROW MAJOR format, the matrix is allocated in device memory
    @param      N[in] Size of the row/column of the matrices
    @param      milliseconds[out] Time taken to perform the matrix multiplication
    @param      TFLOPS[out] Theoretical peak FLOPS achieved during the matrix multiplication
    @param      bs[in] Size of the block to be used for matrix multiplication
    @note       The 'bs' parameter is only used when WMMA_BATCHED is defined
*/
#ifndef WMMA_BATCHED
    void tensorCoreMatMul(const half *d_A, const half *d_B, float *d_C, int N, float* milliseconds, double* TFLOPS);
#else
    void tensorCoreMatMul(const half *d_A, const half *d_B, float *d_C, int N, int bs, float* milliseconds, double* TFLOPS);
#endif

#endif // MATMUL_H