//Simple Matrix Multiplication whit cuBLAS

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>
#include <ctime>
#include <Utilities.h>


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
        half* d_A_half = NULL;
        half* d_B_half = NULL;
        
        // Allocazione delle matrici sul device e conversione in half
        printf("[INFO]: Allocazione delle matrici half A e B sulla GPU\n");
        err1 = convertFloatToHalf(h_A, &d_A_half, N);
        err2 = convertFloatToHalf(h_B, &d_B_half, N);
        if(err1 != cudaSuccess || err2 != cudaSuccess){
            printf("[ERR]: Errore nell'allocazione e conversione delle matrici in half\n");
            d_A_half = NULL;
            d_B_half = NULL;
        }else{
            printf("[INFO]: Allocazione delle matrici half A e B sulla GPU completata\n");
        }

    #ifndef WMMA_BATCHED
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
    #else
        for(int bs = 16; bs <= 256 && bs < N; bs *= 2){
            // const int bs = 32;
            printf("\nStarting run with block size: %d\n", bs);
            // Moltiplicazione di matrici con kernel custom
            tensorCoreMatMul(d_A_half, d_B_half, d_C, N, bs, &myMillis, &myTFLOPS);
            // Salva i risultati su file
            if(resultFile != NULL){
                fprintf(resultFile, "%d,%f,%f,%d,%f,%f\n", N, cublasMillis, cublasTFLOPS, bs, myMillis, myTFLOPS);
            }else{
                printf("[CSV]:\n");
                printf("%d,%f,%f,%d,%f,%f\n", N, cublasMillis, cublasTFLOPS, bs, myMillis, myTFLOPS);
                printf("[/CSV]\n");
            }
        }
    #endif
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
#else

#ifdef TESTING_WMMA
  #warning "WMMA TESTING"
  #define N 32
#endif

#ifndef N
  #define N 64
//   #define N 512
#endif

#warning "Testing mode"

void testShared(){
    printf("\n--------- TESTING SHARED MEMORY ---------\n");


    //Sizes
    const int size = 64;
    const int bs = 32;


    //Testing shared memory
    float* test = NULL;
    float* testDevice = NULL;
    float* destination = NULL;
    float* destination2 = NULL;
    float* destinationHost = NULL;

    
    test = (float*)malloc(size * size * sizeof(float));
    
    if(test == NULL){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on host (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
    }else{
        // printf("
    }

    destinationHost = (float*)malloc(size * size * sizeof(float));
    if(destinationHost == NULL){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on host (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
    }

    cudaError_t err = cudaMalloc((void**)&testDevice, size * size * sizeof(float));
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on device (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
    }else{
        printf("testDevice: %p\n", testDevice);
    }

    err = cudaMalloc((void**)&destination, size * size * sizeof(float));
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on device (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
    }else{
        printf("destination: %p\n", destination);
    }

    err = cudaMalloc((void**)&destination2, size * size * sizeof(float));
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on device (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
    }else{
        printf("destination2: %p\n", destination2);
    }

    for(int i = 0; i < size * size; i++){
        test[i] = i;
    }
    
    err = cudaMemcpy(testDevice, test, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed copy from host to device (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
    }else{
        // memset(test, 0, size * size * sizeof(float));
    }

    //Calling the kernel
    // testSharedMemoryFunctions<<<1, 1024>>>(testDevice, destination, destination2, size);

    cudaDeviceSynchronize();

    //For testing
    bool success = true;
    int rSh = 0;
    int cSh = 0;
    
    //Testing block 0,0
    printf("\n--------- TESTING BLOCK 0,0 ---------\n");
    err = cudaMemcpy(destinationHost, destination, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed copy from device to host (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        printf("\tError: %s\n\tDescription: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    }else{
        rSh = 0;
        for(int r = 0; r < 32; ++r){
            for(int c = 0; c < 32; ++c){
                if(test[r * size + c] != destinationHost[r * size + c]){
                    printf("[ERR]: Test shared memory FAILED -> test[%d] != destinationHost[%d]: %f != %f \t\t(LINE: %d, FILE:%s)\n", r * size + c, rSh * bs + cSh, test[r * size + c], destinationHost[rSh * bs + cSh],__LINE__, __FILE__);
                    success = false;
                    break;
                }
            }
            if(!success){
                break;
            }
        }
    }
    if(success){
        printf("[INFO]: Test shared memory (block 0,0) PASSED\n");
    }else{
        printf("[ERR]: Test shared memory (block 0,0) FAILED\n");
    }

    // //Testing block 1,1
    printf("\n--------- TESTING BLOCK 1,1 ---------\n");
    success = true;
    err = cudaMemcpy(destinationHost, destination2, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed copy from device to host (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        printf("\tError: %s\n\tDescription: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    }else{
        printf("\n");
        rSh = 0;
        for(int r = 32; r < 64; ++r){
            for(int c = 32; c < 64; ++c){
                if(test[r * size + c] != destinationHost[r * size + c]){
                    printf("[ERR]: Test shared memory FAILED -> test[%d] != destinationHost[%d]: %f != %f \t\t(LINE: %d, FILE:%s)\n", r * size + c, rSh * bs + cSh, test[r * size + c], destinationHost[rSh * bs + cSh],__LINE__, __FILE__);
                    success = false;
                    break;
                }
            }
            if(!success){
                break;
            }
        }
    }
    if(success){
        printf("[INFO]: Test shared memory (block 1,1) PASSED\n\n");
    }else{
        printf("[ERR]: Test shared memory (block 1,1) FAILED\n\n");
    }

    if(test != NULL){
        free(test);
    }
    if(testDevice != NULL){
        cudaFree(testDevice);
    }
    if(destination != NULL){
        cudaFree(destination);
    }
}


int main(int argc, char **argv){
    printf("WMMA TEST: Testing mode\n");
    // Puntatori per le matrici sull'host
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C_cublas = NULL;
    float *h_C_wmma = NULL;
    
    testShared();

    // // Allocazione delle matrici sull'host (CPU)
    printf("\n--------- TESTING CODE ---------\n");
    printf("[INFO]: Allocazione delle matrici sull'host\n");
    size_t matrix_size = N * N * sizeof(float);
    h_A = (float *)malloc(matrix_size);
    h_B = (float *)malloc(matrix_size);
    h_C_cublas = (float *)malloc(matrix_size);
    h_C_wmma = (float *)malloc(matrix_size);
    printf("[INFO]: Allocazione delle matrici sull'host completata\n");

    // Inizializza le matrici A e B sull'host
    if(h_A != NULL && h_B != NULL && h_C_wmma != NULL){
        printf("[INFO]: Inizializzazione delle matrici sull'host\n");
        for (int i = 0; i < N; ++i) {
            for(int j = 0; j < N; ++j){
                // h_A[i] = 0.1;
                // h_B[i] = 0.2;                // Blocco [r,c]
                if(i < 32  && j < 32){          // Blocco [0,0] [ERROR]
                    h_A[i + j * N] = i + j * N;
                    h_B[i + j * N] = i + j * N;
                }else if(i >= 32 && j < 32){    // Blocco [0,1] [OK]
                    h_A[i + j * N] = 0;
                    h_B[i + j * N] = 0;
                }else if(i < 32 && j >= 32){    // Blocco [1,0] [ERROR]
                    h_A[i + j * N] = 0;
                    h_B[i + j * N] = 0;
                }else{                          // Blocco [1,1] [OK]
                    h_A[i + j * N] = 0;
                    h_B[i + j * N] = 0;
                }
            }
        }
        printf("A[0,1]: %f\nA[0,32]: %f\nA[32,0]: %f\nA[32,32]: %f\n\n", h_A[1], h_A[32], h_A[32 * N], h_A[32 * N + 32]);
        memset(h_C_wmma, 0, matrix_size);
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
        checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
        checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");
        checkCudaError(cudaMemcpy(d_C, h_C_wmma, matrix_size, cudaMemcpyHostToDevice), "Copia matrice C sulla GPU");
        printf("[INFO]: Matrici copiate sulla GPU\n");
    }else{
        printf("[ERR]: Errore nell'allocazione delle matrici sulla GPU\n");
        return 2;
    }
    //Indicatori di performance
    float cublasMillis = 0;
    double cublasTFLOPS = 0;

    float myMillis = 0;
    double myTFLOPS = 0;

    ///////////////////// ALGORHITMs ///////////////////////
    /////// cuBLAS ///////
    // Moltiplicazione di matrici con cuBLAS
    cublasMatMul(d_A, d_B, d_C, N,&cublasMillis, &cublasTFLOPS); 
    // Copia dei risultati dalla GPU all'host
    checkCudaError(cudaMemcpy(h_C_cublas, d_C, matrix_size, cudaMemcpyDeviceToHost), "Copia matrice C dall'host");
    //Stampa delle matrici
    if(N <= 4){
        printf("Matrice A:\n");
        printMat(h_A, N, N);
        printf("Matrice B:\n");
        printMat(h_B, N, N);
        printf("Matrice C:\n");
        printMat(h_C_cublas, N, N);
    }

    // Stampa dei risultati
    printf("\nTempo di esecuzione [cuBLAS] [size: %d]: %f ms\n", cublasMillis, N);
    printf("TFLOPS [cuBLAS] [size: %d]: %f\n\n", cublasTFLOPS, N);
    
    //Libero la memoria delle matrici sorgenti
    if(err1 == cudaSuccess && err2 == cudaSuccess){
        cudaFree(d_A);
        cudaFree(d_B);
        d_A = NULL;
        d_B = NULL;
    }


    /////// Custom Kernel ///////
    half* d_A_half = NULL;
    half* d_B_half = NULL;
    
    // #ifdef TESTING_BATCHED
    //     #warning "Testing batched"
    //     for(int row = 32; row < 64; ++row){
    //         for(int col = 0; col < 32; ++col){
    //             d_A[row * 64 + col] = row * 64 + col;
    //             d_b[row * 64 + col] = row * 64 + col;
    //         }
    //     }
    // #endif
    // Allocazione delle matrici sul device e conversione in half
    printf("[INFO]: Allocazione delle matrici half A e B sulla GPU\n");
    err1 = convertFloatToHalf(h_A, &d_A_half, N);
    err2 = convertFloatToHalf(h_B, &d_B_half, N);
    if(err1 != cudaSuccess || err2 != cudaSuccess){
        printf("[ERR]: Errore nell'allocazione e conversione delle matrici in half\n");
        d_A_half = NULL;
        d_B_half = NULL;
    }else{
        printf("[INFO]: Allocazione delle matrici half A e B sulla GPU completata\n");
    }

    //Azzeramento matrice C
    checkCudaError(cudaMemcpy(d_C, h_C_wmma, matrix_size, cudaMemcpyHostToDevice), "Copia matrice C sulla GPU");

    // Moltiplicazione di matrici con kernel custom
    tensorCoreMatMul(d_A_half, d_B_half, d_C, N, &myMillis, &myTFLOPS);
    // #ifndef TESTING_BATCHED
    //     tensorCoreMatMul(d_A_half, d_B_half, d_C, N, &myMillis, &myTFLOPS);
    // #else
    //     testBlockMatrixMultiplication(d_A_half, d_B_half, d_C, 0, 0, 64);
    // #endif
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

    // Stampa dei risultati
    printf("\n\nTempo di esecuzione [wmma] [size: %d]: %f ms\n", myMillis, N);
    printf("TFLOPS [wmma] [size: %d]: %f\n\n", myTFLOPS, N);

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
        cudaFree(d_A_half);
        cudaFree(d_B_half);
        cudaFree(d_C);
        d_A_half = NULL;
        d_B_half = NULL;
        d_C = NULL;
    }
    if(success){
        return 0;
    }
    return 2;
}

#endif