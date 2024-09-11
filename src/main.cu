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

    //Indicatori di performance
    float milliseconds = 0;
    double TFLOPS = 0;

    // Moltiplicazione di matrici con cuBLAS
    cublasMatMul(h_A, h_B, h_C, N, &milliseconds, &TFLOPS); 
    
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
    printf("\n\nTempo di esecuzione: %f ms\n", milliseconds);
    printf("TFLOPS: %f\n", TFLOPS);

    // Libera la memoria sull'host
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}