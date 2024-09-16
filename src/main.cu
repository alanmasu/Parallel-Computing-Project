//Simple Matrix Multiplication whit cuBLAS

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>
#include <ctime>



// Funzione per la stampa di matrici
void printMat(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

#ifndef TESTING
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
    float *h_A, *h_B, *h_C;

    //Ciclo sulle size delle matrici
    for(int N = 16; N <= 16384; N *= 2){
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
            // h_A[i] = i;
            // h_B[i] = i;
        }

        //Allocazione sul device (GPU)
        float *d_A, *d_B, *d_C;
        checkCudaError(cudaMalloc((void **)&d_A, matrix_size), "Allocazione matrice A su GPU");
        checkCudaError(cudaMalloc((void **)&d_B, matrix_size), "Allocazione matrice B su GPU");
        checkCudaError(cudaMalloc((void **)&d_C, matrix_size), "Allocazione matrice C su GPU");

        // Copia delle matrici dall'host alla GPU
        if(d_A != NULL && d_B != NULL && d_C != NULL){
            checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
            checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");
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
        printf("\n\nTempo di esecuzione [cuBLAS] [size: %d]: %f ms\n", cublasMillis, N);
        printf("TFLOPS [cuBLAS] [size: %d]: %f\n", cublasTFLOPS, N);

        /////// Custom Kernel ///////
        for(int bs = 16; bs <= 256; bs *= 2){
            // Moltiplicazione di matrici con kernel custom
            //tensorCoreMatMul(d_A, d_B, d_C, N, blockSize, &myMillis, &myTFLOPS);
            // Salva i risultati su file
            if(resultFile != NULL){
                fprintf(resultFile, "%d,%f,%f,%d,%f,%f\n", N, cublasMillis, cublasTFLOPS, bs, myMillis, myTFLOPS);
            }else{
                printf("[CSV]:\n");
                printf("%d,%f,%f,%d,%f,%f\n", N, cublasMillis, cublasTFLOPS, bs, myMillis, myTFLOPS);
                printf("[/CSV]\n");
            }
        }
        // Libera la memoria sull'host
        free(h_A);
        free(h_B);
        free(h_C);

        // Libera la memoria sulla GPU
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    if(resultFile != NULL){
        fclose(resultFile);
    }

    return 0;
}
#else

#define N 64

#warning "Testing mode"

int main(int argc, char **argv){
    printf("WMMA TEST: Testing mode\n");
    // // Puntatori per le matrici sull'host
    // float *h_A = NULL;
    // float *h_B = NULL;
    // float *h_C_cublas = NULL;
    // float *h_C_wmma = NULL;

    // // Allocazione delle matrici sull'host (CPU)
    // printf("[INFO]: Allocazione delle matrici sull'host\n");
    // size_t matrix_size = N * N * sizeof(float);
    // h_A = (float *)malloc(matrix_size);
    // h_B = (float *)malloc(matrix_size);
    // h_C_cublas = (float *)malloc(matrix_size);
    // h_C_wmma = (float *)malloc(matrix_size);
    // printf("[INFO]: Allocazione delle matrici sull'host completata\n");

    // // Inizializza le matrici A e B sull'host
    // if(h_A != NULL && h_B != NULL){
    //     printf("[INFO]: Inizializzazione delle matrici sull'host\n");
    //     for (int i = 0; i < N * N; ++i) {
    //         h_A[i] = i;
    //         h_B[i] = i;
    //     }
    // }else{
    //     printf("[ERR]:Errore nell'allocazione delle matrici sull'host\n");
    //     return 1;
    // }

    // //Allocazione sul device (GPU)
    // float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    // cudaError_t err1 = checkCudaError(cudaMalloc((void **)&d_A, matrix_size), "Allocazione matrice A su GPU");
    // cudaError_t err2 = checkCudaError(cudaMalloc((void **)&d_B, matrix_size), "Allocazione matrice B su GPU");
    // cudaError_t err3 = checkCudaError(cudaMalloc((void **)&d_C, matrix_size), "Allocazione matrice C su GPU");

    // // Copia delle matrici dall'host alla GPU
    // if(err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess){
    //     printf("\n[INFO]Copia delle matrici sulla GPU\n");
    //     checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
    //     checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");
    //     printf("Matrici copiate sulla GPU\n");
    // }else{
    //     printf("[ERR]: Errore nell'allocazione delle matrici sulla GPU\n");
    //     return 2;
    // }
    // //Indicatori di performance
    // float cublasMillis = 0;
    // double cublasTFLOPS = 0;

    // float myMillis = 0;
    // double myTFLOPS = 0;

    // ///////////////////// ALGORHITMs ///////////////////////
    // /////// cuBLAS ///////
    // // Moltiplicazione di matrici con cuBLAS
    // cublasMatMul(d_A, d_B, d_C, N, &cublasMillis, &cublasTFLOPS); 
    // // Copia dei risultati dalla GPU all'host
    // checkCudaError(cudaMemcpy(h_C_cublas, d_C, matrix_size, cudaMemcpyDeviceToHost), "Copia matrice C dall'host");
    // //Stampa delle matrici
    // if(N <= 4){
    //     printf("Matrice A:\n");
    //     printMat(h_A, N, N);
    //     printf("Matrice B:\n");
    //     printMat(h_B, N, N);
    //     printf("Matrice C:\n");
    //     printMat(h_C_cublas, N, N);
    // }

    // // Stampa dei risultati
    // printf("\n\nTempo di esecuzione [cuBLAS] [size: %d]: %f ms\n", cublasMillis, N);
    // printf("TFLOPS [cuBLAS] [size: %d]: %f\n", cublasTFLOPS, N);
    
    // //Libero la memoria delle matrici sorgenti
    // cudaFree(d_A);
    // cudaFree(d_B);


    // /////// Custom Kernel ///////
    // // Moltiplicazione di matrici con kernel custom
    // tensorCoreMatMul(h_A, h_B, d_C, N, &myMillis, &myTFLOPS);
    // // Copia dei risultati dalla GPU all'host
    // checkCudaError(cudaMemcpy(h_C_wmma, d_C, matrix_size, cudaMemcpyDeviceToHost), "Copia matrice C dal device");
    // //Stampa delle matrici
    // if(N <= 4){
    //     printf("Matrice A:\n");
    //     printMat(h_A, N, N);
    //     printf("Matrice B:\n");
    //     printMat(h_B, N, N);
    //     printf("Matrice C:\n");
    //     printMat(h_C_wmma, N, N);
    // }

    // // Stampa dei risultati
    // printf("\n\nTempo di esecuzione [wmma] [size: %d]: %f ms\n", myMillis, N);
    // printf("TFLOPS [wmma] [size: %d]: %f\n", myTFLOPS, N);

    // //Testing dei risultati e confronto con cuBLAS
    // for(int i = 0; i < N * N; i++){
    //     if(h_C_cublas[i] != h_C_wmma[i]){
    //         printf("Errore: i risultati non coincidono\n");
    //         break;
    //     }
    // }


    // // Libera la memoria sull'host
    // free(h_A);
    // free(h_B);
    // free(h_C_cublas);
    // free(h_C_wmma);

    // // Libera la memoria sulla GPU
    // cudaFree(d_C);
}

#endif