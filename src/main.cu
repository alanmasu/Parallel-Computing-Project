//Simple Matrix Multiplication whit cuBLAS

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>
#include <ctime>
#include <Utilities.h>


int main(int argc, char **argv) {
    // Recupero dell'ora corrente per la creazione del file di output
    time_t curr_time;
    tm * curr_tm;
    char filename[100];
    char descriptionFilename[100];
    time(&curr_time);
    curr_tm = localtime(&curr_time);
    strftime(filename, 99, "results/result-%d_%m_%Y-%H_%M_%S.csv", curr_tm);
    strftime(descriptionFilename, 99, "results/description-%d_%m_%Y-%H_%M_%S.txt", curr_tm);

    //Creazione del file di descrizione
    FILE *descriptionFile = fopen(descriptionFilename, "w");
    if(descriptionFile != NULL){
        fprintf(descriptionFile, "File di descrizione della run:\n");
        if(argc >= 1){
            fprintf(descriptionFile, "%s\n", argv[1]);
        }
        fclose(descriptionFile);
    }else{
        printf("Errore nella creazione del file di descrizione\n");
    }

    //Apertura file dei risultati
    FILE *resultFile = fopen(filename, "r");
    if(resultFile == NULL){
        resultFile = fopen(filename, "w");
        fprintf(resultFile, "Size,cuBLAS_ms,cuBLAS_TFLOPS,blockSize,ms,TFLOPS\n");
    }else{
        fclose(resultFile);
        resultFile = fopen(filename, "a");
    }

    // Puntatori per le matrici sull'host
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C = NULL;

    //Ciclo sulle size delle matrici
    for(int N = 64; N <= 16384; N *= 2){
        printf("\n\nStarting run with size: %d\n", N);
        // Allocazione delle matrici sull'host (CPU)
        size_t matrix_size = N * N * sizeof(float);
        h_A = (float *)malloc(matrix_size);
        h_B = (float *)malloc(matrix_size);
        h_C = (float *)malloc(matrix_size);

        if(h_A == NULL || h_B == NULL || h_C == NULL){
            printf("Errore nell'allocazione delle matrici sull'host [size: %d]\n", N);
            if(resultFile != NULL){
                fclose(resultFile);
            }
            return 1;
        }
        // Inizializza le matrici A e B sull'host
        for (int i = 0; i < N * N; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
            h_C[i] = 0;
            // h_A[i] = i;
            // h_B[i] = i;
        }

        //Allocazione sul device (GPU)
        float *d_A = NULL;
        float *d_B = NULL;
        float *d_C = NULL;
        cudaError_t err1 = checkCudaError(cudaMalloc((void **)&d_A, matrix_size), "Allocazione matrice A su GPU");
        cudaError_t err2 = checkCudaError(cudaMalloc((void **)&d_B, matrix_size), "Allocazione matrice B su GPU");
        cudaError_t err3 = checkCudaError(cudaMalloc((void **)&d_C, matrix_size), "Allocazione matrice C su GPU");

        // Copia delle matrici dall'host alla GPU
        if(err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess){
            checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
            checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");
            checkCudaError(cudaMemcpy(d_C, h_C, matrix_size, cudaMemcpyHostToDevice), "Copia matrice C sulla GPU");
        }
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
        if(N <= 4){
            printf("Matrice A:\n");
            printMat(h_A, N, N);
            printf("Matrice B:\n");
            printMat(h_B, N, N);
            printf("Matrice C:\n");
            printMat(h_C, N, N);
        }

        // Stampa dei risultati
        printf("Tempo di esecuzione [cuBLAS] [size: %d]: %f ms\n", cublasMillis, N);
        printf("TFLOPS [cuBLAS] [size: %d]: %f\n", cublasTFLOPS, N);

        // Libera la memoria sulla GPU
        if(err1 == cudaSuccess && err2 == cudaSuccess){
            cudaFree(d_A);
            cudaFree(d_B);
            d_A = NULL;
            d_B = NULL;
        }

        /////// Custom Kernel ///////
        half* h_A_half = NULL;
        half* h_B_half = NULL;
        half* d_A_half = NULL;
        half* d_B_half = NULL;
        
        // Allocazione delle matrici sul device e conversione in half
        printf("[INFO]: Allocazione delle matrici half A e B sulla GPU\n");
        err1 = convertFloatToHalf(h_A, &h_A_half, &d_A_half, N);
        err2 = convertFloatToHalf(h_B, &h_B_half, &d_B_half, N);
        if(err1 != cudaSuccess || err2 != cudaSuccess){
            printf("[ERR]: Errore nell'allocazione e conversione delle matrici in half\n");
            d_A_half = NULL;
            d_B_half = NULL;
        }else{
            printf("[INFO]: Allocazione delle matrici half A e B sulla GPU completata\n");
        }

        // Moltiplicazione di matrici con kernel custom
        tensorCoreMatMul(d_A_half, d_B_half, d_C, N, &myMillis, &myTFLOPS);
        // Salva i risultati su file
        if(resultFile != NULL){
            fprintf(resultFile, "%d,%f,%f,%d,%f,%f\n", N, cublasMillis, cublasTFLOPS, 16, myMillis, myTFLOPS);
        }else{
            printf("[CSV]:\n");
            printf("%d,%f,%f,%d,%f,%f\n", N, cublasMillis, cublasTFLOPS, 16, myMillis, myTFLOPS);
            printf("[/CSV]\n");
        }
    
        // Libera la memoria sull'host
        if(h_A != NULL && h_B != NULL && h_C != NULL){
            free(h_A);
            free(h_B);
            free(h_C);

            h_A = NULL;
            h_B = NULL;
            h_C = NULL;
        }

        // Libera la memoria sulla GPU
        if(err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess){
            cudaFree(d_A_half);
            cudaFree(d_B_half);
            cudaFree(d_C);
            d_A_half = NULL;
            d_B_half = NULL;
            d_C = NULL;
        }
    }
    if(resultFile != NULL){
        fclose(resultFile);
    }

    return 0;
}
