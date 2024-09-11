//Simple Matrix Multiplication whit cuBLAS

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Dimensione delle matrici
#define N 1024

// Funzione helper per il controllo degli errori CUDA
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Funzione helper per il controllo degli errori cuBLAS
void checkCublasError(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %s\n", msg);
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Punteri per le matrici sull'host
    float *h_A, *h_B, *h_C;
    
    // Punteri per le matrici sulla GPU (device)
    float *d_A, *d_B, *d_C;
    float alpha = 1.0f, beta = 0.0f;

    // Allocazione delle matrici sull'host (CPU)
    size_t matrix_size = N * N * sizeof(float);
    h_A = (float *)malloc(matrix_size);
    h_B = (float *)malloc(matrix_size);
    h_C = (float *)malloc(matrix_size);

    // Inizializza le matrici A e B sull'host
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocazione delle matrici sulla GPU (device)
    checkCudaError(cudaMalloc((void **)&d_A, matrix_size), "Allocazione matrice A su GPU");
    checkCudaError(cudaMalloc((void **)&d_B, matrix_size), "Allocazione matrice B su GPU");
    checkCudaError(cudaMalloc((void **)&d_C, matrix_size), "Allocazione matrice C su GPU");

    // Inizializzazione dell'handle cuBLAS
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Inizializzazione cuBLAS");

    // Copia delle matrici dall'host alla GPU
    checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
    checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");

    // Misurazione del tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvia il timer
    cudaEventRecord(start, 0);

    // Esegui la moltiplicazione di matrici (C = alpha * A * B + beta * C) sulla GPU
    checkCublasError(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N),
        "Moltiplicazione di matrici"
    );

    // Ferma il timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calcola il tempo impiegato
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copia dei risultati dalla GPU all'host
    checkCudaError(cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost), "Copia matrice C dall'host");

    // Numero totale di operazioni in virgola mobile (FLOP)
    double FLOPs = 2.0 * N * N * N;

    // Calcolo dei TFLOPS
    double TFLOPS = (FLOPs / (milliseconds / 1000.0)) / 1e12;

    // Stampa dei risultati
    printf("Tempo di esecuzione: %f ms\n", milliseconds);
    printf("TFLOPS: %f\n", TFLOPS);

    // Libera la memoria sulla GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Libera la memoria sull'host
    free(h_A);
    free(h_B);
    free(h_C);

    // Distruggi l'handle cuBLAS
    cublasDestroy(handle);

    return 0;
}