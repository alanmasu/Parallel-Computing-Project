#ifndef MATMUL_H
#define MATMUL_H
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


/*! 
    @brief      Function to perform matrix multiplication of two matrices A and B using cuBLAS

    @details    Performing a matrix multiplication of two matrices A and B and storing the result in matrix C. 
                The matrices are of size N x N.

    @param      A[in] Pointer to the first matrix in ROW MAJOR format, the matrix is allocated in host memory
    @param      B[in] Pointer to the second matrix in ROW MAJOR format, the matrix is allocated in host memory
    @param      C[out] Pointer to the resultant matrix in ROW MAJOR format, the matrix is allocated in host memory
    @param      N[in] Size of the row/column of the matrices
    @param      milliseconds[out] Time taken to perform the matrix multiplication
    @param      TFLOPS[out] Theoretical peak FLOPS achieved during the matrix multiplication
*/
void cublasMatMul(const float *h_A, const float *h_B, float *h_C, int N, float* milliseconds, double* TFLOPS);

#endif // MATMUL_H