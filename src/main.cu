//Simple Matrix Multiplication whit cuBLAS

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>

// Dimensione delle matrici
#define N 3

// Funzione per la stampa di matrici
void printMat(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    // Punteri per le matrici sull'host
    float *h_A, *h_B, *h_C;

    // Allocazione delle matrici sull'host (CPU)
    size_t matrix_size = N * N * sizeof(float);
    h_A = (float *)malloc(matrix_size);
    h_B = (float *)malloc(matrix_size);
    h_C = (float *)malloc(matrix_size);

    // Inizializza le matrici A e B sull'host
    for (int i = 0; i < N * N; ++i) {
        // h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        // h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_A[i] = i;
        h_B[i] = i;
    }

    //Device allocation
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void **)&d_A, matrix_size), "Allocazione matrice A su GPU");
    checkCudaError(cudaMalloc((void **)&d_B, matrix_size), "Allocazione matrice B su GPU");
    checkCudaError(cudaMalloc((void **)&d_C, matrix_size), "Allocazione matrice C su GPU");

    // Copia delle matrici dall'host alla GPU
    checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
    checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");

    //Indicatori di performance
    float cublasMillis = 0;
    double cublasTFLOPS = 0;

    float myMillis = 0;
    double myTFLOPS = 0;

    ///////////////////// ALGORHITMs ///////////////////////
    /////// cuBLAS ///////
    // Moltiplicazione di matrici con cuBLAS
    cublasMatMul(d_A, d_B, d_C, N, &cublasMillis, &cublasTFLOPS); 
    // Copia dei risultati dalla GPU all'host
    checkCudaError(cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost), "Copia matrice C dall'host");
    //Stampa delle matrici
    if(N <= 3){
        printf("Matrice A:\n");
        printMat(h_A, N, N);
        printf("Matrice B:\n");
        printMat(h_B, N, N);
        printf("Matrice C:\n");
        printMat(h_C, N, N);
    }

    // Stampa dei risultati
    printf("\n\nTempo di esecuzione [cuBLAS]: %f ms\n", milliseconds);
    printf("TFLOPS [cuBLAS]: %f\n", TFLOPS);

    /////// Custom Kernel ///////
    // Moltiplicazione di matrici con kernel custom
    //tensorCoreMatMul(d_A, d_B, d_C, N, blockSize, &myMillis, &myTFLOPS);

    // Libera la memoria sull'host
    free(h_A);
    free(h_B);
    free(h_C);

    // Libera la memoria sulla GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}