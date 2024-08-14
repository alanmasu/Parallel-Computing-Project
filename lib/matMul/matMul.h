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
    @brief      Function to perform matrix multiplication of two matrices A and B

    @details    Performing a matrix multiplication of two matrices A and B and storing the result in matrix C. 
                The matrices are of size N x N.

    @param      A[in] Pointer to the first matrix
    @param      B[in] Pointer to the second matrix
    @param      C[out] Pointer to the resultant matrix
    @param      N[in] Size of the row/column of the matrices
*/
void serialMatMul(const float *A, const float *B, float *C, int N);


#endif // MATMUL_H