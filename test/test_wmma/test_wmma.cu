/*!
    @file test_wmma.cu

    @brief Test the __device__ function for block multiplication using WMMA
    
    @author alanmasu
    @date 06/03/2025
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>
#include <ctime>
#include <Utilities.h>

#warning "TEST WMMA"

#define N 32

/**!
    @brief Kernel che chiama la funzione 'blockMatrixMul' per moltiplicare due blocchi BLOCK_SIZE x BLOCK_SIZE

    @param a puntatore alla matrice A sul device
    @param b puntatore alla matrice B sul device
    @param d_c puntatore alla matrice C sul device
    @param n dimensione delle matrici (in questo caso BLOCK_SIZE)

*/
__global__ void testBlockMatrixMul(half* a, half* b, float* d_c, int n){
    __shared__ half  As [SHARED_PAGE_COUNT * BLOCK_SIZE * BLOCK_SIZE];
    __shared__ half  Bs [SHARED_PAGE_COUNT * BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Cs [2 * BLOCK_SIZE * BLOCK_SIZE];

    //Copy data to shared memory
    As[threadIdx.x] = a[threadIdx.x];
    Bs[threadIdx.x] = b[threadIdx.x];
    Cs[threadIdx.x] = 0;
    // __syncthreads();

    // wmmaMatrixMultiply(As, Bs, Cs, n);
    blockMatrixMul(As, Bs, Cs, BLOCK_SIZE);

    //Attendi i thread dei primi 8 warp per completare la computazione
    __syncthreads();

    //Somma i risultati parziali nella matrice in global memory
    d_c[threadIdx.x] = Cs[threadIdx.x] + Cs[threadIdx.x + BLOCK_SIZE * BLOCK_SIZE];

    // blockMatrixMul(a, b, d_c, n);
    // copyBlockToGlobal(Cs, d_c, blockIdx.y, blockIdx.x, n);
    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0){
        printf("Matrice As:\n");
        printNMat(As, 4, 4, n);

        printf("\nMatrice Bs:\n");
        printNMat(Bs, 4, 4, n);

        printf("\nMatrice Cs[0]:\n");
        printNMat(Cs, 4, 4, n);

        printf("\nMatrice Cs[1]:\n");
        printNMat(Cs + BLOCK_SIZE * BLOCK_SIZE, 4, 4, n);

        printf("\nMatrice C:\n");
        printNMat(d_c, 4, 4, n);

        printf("\n");
    }
}

int main(int argc, char **argv){
    // Puntatori per le matrici sull'host
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C_cublas = NULL;
    float *h_C_wmma = NULL;

    // // Allocazione delle matrici sull'host (CPU)
    printf("\n--------- TESTING WMMA ---------\n");
    printf("[INFO]: Allocazione delle matrici sull'host\n");
    size_t matrix_size = N * N * sizeof(float);
    h_A = (float *)malloc(matrix_size);
    h_B = (float *)malloc(matrix_size);
    h_C_cublas = (float *)malloc(matrix_size);
    h_C_wmma = (float *)malloc(matrix_size);
    printf("[INFO]: Allocazione delle matrici sull'host completata\n");

    // Inizializza le matrici A e B sull'host
    if(h_A != NULL && h_B != NULL && h_C_wmma != NULL && h_C_cublas != NULL){
        printf("[INFO]: Inizializzazione delle matrici sull'host\n");
        for (int i = 0; i < N * N; ++i) {
            h_A[i] = i;
            h_B[i] = i;
        }
        memset(h_C_wmma, 0, matrix_size);
        memset(h_C_cublas, 0, matrix_size);
    }else{
        printf("[ERR]:Errore nell'allocazione delle matrici sull'host\n");
        return 1;
    }

    // //Allocazione sul device (GPU)
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaError_t err1 = checkCudaError(cudaMalloc((void **)&d_A, matrix_size), "Allocazione matrice A su GPU");
    cudaError_t err2 = checkCudaError(cudaMalloc((void **)&d_B, matrix_size), "Allocazione matrice B su GPU");
    cudaError_t err3 = checkCudaError(cudaMalloc((void **)&d_C, matrix_size), "Allocazione matrice C su GPU");

    // Copia delle matrici dall'host alla GPU
    if(err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess){
        printf("\n[INFO]: Copia delle matrici sulla GPU\n");
        err1 = checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
        err2 = checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");
        err3 = checkCudaError(cudaMemcpy(d_C, h_C_wmma, matrix_size, cudaMemcpyHostToDevice), "Copia matrice C sulla GPU");
        if(!(err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess)){
            printf("[ERR]: Errore nella copia delle matrici sulla GPU\n");
            return 2;
        }
        printf("[INFO]: Matrici copiate sulla GPU\n");
    }else{
        printf("[ERR]: Errore nell'allocazione delle matrici sulla GPU\n");
        return 1;
    }


    //Indicatori di performance
    float cublasMillis = 0;
    double cublasTFLOPS = 0;

    ///////////////////// ALGORHITMs ///////////////////////
    /////// cuBLAS ///////
    // Moltiplicazione di matrici con cuBLAS
    cublasMatMul(d_A, d_B, d_C, N, &cublasMillis, &cublasTFLOPS); 
    // Copia dei risultati dalla GPU all'host
    err1 = checkCudaError(cudaMemcpy(h_C_cublas, d_C, matrix_size, cudaMemcpyDeviceToHost), "Copia matrice C dall'host");
    if(err1 != cudaSuccess){
        printf("[ERR]: Errore nella copia della matrice C dal device\n");
        return 2;
    }
    //Stampa delle matrici
    if(N <= 4){
        printf("Matrice A:\n");
        printMat(h_A, N, N);
        printf("Matrice B:\n");
        printMat(h_B, N, N);
        printf("Matrice C:\n");
        printMat(h_C_cublas, N, N);
    }
    
    //Libero la memoria delle matrici sorgenti
    if(err1 == cudaSuccess && err2 == cudaSuccess){
        cudaFree(d_A);
        cudaFree(d_B);
        d_A = NULL;
        d_B = NULL;
    }

    /////// Custom Kernel ///////
    half* h_A_h = NULL;
    half* h_B_h = NULL;
    half* d_A_h = NULL;
    half* d_B_h = NULL;
    printf("[INFO]: Allocazione delle matrici half A e B sulla GPU\n");

    // Allocazione delle matrici sul device e conversione in half
    err1 = convertFloatToHalf(h_A, &h_A_h, &d_A_h, N);
    err2 = convertFloatToHalf(h_B, &h_B_h, &d_B_h, N);
    if(err1 != cudaSuccess || err2 != cudaSuccess){
        printf("[ERR]: Errore nell'allocazione e conversione delle matrici in half\n");
        d_A_h = NULL;
        d_B_h = NULL;
        return 1;
    }else{
        printf("[INFO]: Allocazione delle matrici half A e B sulla GPU completata\n");
    }

    //Azzeramento matrice C
    checkCudaError(cudaMemcpy(d_C, h_C_wmma, matrix_size, cudaMemcpyHostToDevice), "Copia matrice C sulla GPU");

    // Moltiplicazione di matrici con kernel custom

    dim3 blocks = dim3(N/BLOCK_SIZE, N/BLOCK_SIZE);
    dim3 threads = dim3(THREADS_PER_BLOCK * THREADS_PER_BLOCK);
    testBlockMatrixMul<<<blocks, threads>>>(d_A_h, d_B_h, d_C, N);

    // Copia dei risultati dalla GPU all'host
    checkCudaError(cudaMemcpy(h_C_wmma, d_C, matrix_size, cudaMemcpyDeviceToHost), "Copia matrice C dal device");
    //Stampa delle matrici
    if(N <= 4){
        printf("Matrice A:\n");
        printMat(h_A, N, N);
        printf("Matrice B:\n");
        printMat(h_B, N, N);
        printf("Matrice C:\n");
        printMat(h_C_wmma, N, N);
    }else{
        //Print the first 2x2 matrix
        printf("Matrice A:\n");
        printNMat(h_A, 2, 2, N);
        printf("Matrice B:\n");
        printNMat(h_B, 2, 2, N);
        printf("Matrice C:\n");
        printNMat(h_C_wmma, 2, 2, N);
    }

    //Testing dei risultati e confronto con cuBLAS
    bool success = true;
    for(int i = 0; i < N * N; i++){
        if(h_C_cublas[i] - h_C_wmma[i] > 0.2){
            printf("\n\n[ERRORE]: i risultati non coincidono\n");
            printf("h_C_cublas[%d] != h_C_wmma[%d]\n", i, i);
            printf("%f != %f\n", h_C_cublas[i], h_C_wmma[i]);
            success = false;
            break;
        }
    }

    // Libera la memoria sull'host
    if(h_A != NULL && h_B != NULL && h_C_cublas != NULL && h_C_wmma != NULL){
        free(h_A);
        free(h_B);
        free(h_C_cublas);
        free(h_C_wmma);

        h_A = NULL;
        h_B = NULL;
        h_C_cublas = NULL;
        h_C_wmma = NULL;
    }

    // Libera la memoria sulla GPU
    if(err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess){
        free(h_A_h);
        free(h_B_h);
        cudaFree(d_A_h);
        cudaFree(d_B_h);
        cudaFree(d_C);
        d_A_h = NULL;
        d_B_h = NULL;
        d_C = NULL;
    }
    if(success){
        printf("--------- Test WMMA PASSED ---------\n");
        return 0;
    }
    printf("--------- Test WMMA FAILED ---------\n");
    return -1;
}
