/**!
    @file test_batched.cu
    @brief File per il testing delle funzionalità di batched matrix multiplication
    @details Questo file esegue la moltiplicazione a blocchi a mano, per capire dove la funzione matMulTensorCore sbaglia

    @author alanmasu
    @date 06/03/2025
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>
#include <cstdlib>
#include <mma.h>
#include <ctime>
#include <Utilities.h>
#include <math.h>

#warning "TEST BATCHED"

using namespace nvcuda;



/**!
    @brief Funzione per popolare una matrice bs x bs con valori equivalenti ad un blocco (bRow, bCol) di una matrice N x N
    
    @param [out] A puntatore alla matrice da popolare
    @param bRow indice di riga del blocco
    @param bCol indice di colonna del blocco
    @param bs dimensione del blocco
    @param size dimensione della matrice "N x N"
*/
void populateBlockOfMatrix(float *A, int bRow, int bCol, int bs, int size){
    for(int rowInsideBlock = 0; rowInsideBlock < bs; ++rowInsideBlock){
        for(int colInsideBlock = 0; colInsideBlock < bs; ++colInsideBlock){
            int item = (bRow * bs * size) + (bCol * bs) + (rowInsideBlock * size) + colInsideBlock;
            if(rowInsideBlock * bs + colInsideBlock < bs * bs){
                A[rowInsideBlock * bs + colInsideBlock] = item;
            }
        }
    }
}

int testPopulateBlockOfMatrix(){
    float a[2*2] = {-1.0};
    // const float test[2*2] = {10.0, 11.0, 14.0, 15.0};
    populateBlockOfMatrix(a, 1, 1, 2, 4);
    printf("Matrice A:\n");
    printMat(a, 2,2);
    // // For future implementation
    // for (int i = 0; i < 2*2; ++i){
    //     if(abs(a[i] - test[i]) > 0.00001){
    //         printf("[ERR]: Errore nella popolazione della matrice => i was %d, a[i] was %f, test[i] was %f\n", i, a[i], test[i]);
    //         return -1;
    //     }
    // }
    return 0;
}

int testBlockMatrixMultiplication(){
    const int SIZE_COUNT = 4;
    int sizes[SIZE_COUNT] = {32, 64, 256, 2048};
    // Puntatori per le matrici sull'host
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C_cublas = NULL;
    float *h_C_wmma = NULL;
    bool success = true;

    // // Allocazione delle matrici sull'host (CPU)
    printf("\n--------- TESTING BLOCK Marix Multiplication ---------\n");
    
    int smemSize = 0;
    cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("shared memory ammount: %d\n", smemSize);

    for(int size = 0; size < SIZE_COUNT; ++size){
        int N = sizes[size];
        printf("[INFO]: Allocazione delle matrici sull'host -> N: %d\n", N);
        size_t matrix_size = N * N * sizeof(float);
        h_A = (float *)malloc(matrix_size);
        h_B = (float *)malloc(matrix_size);
        h_C_cublas = (float *)malloc(matrix_size);
        h_C_wmma = (float *)malloc(matrix_size);
        if(h_A == NULL || h_B == NULL || h_C_cublas == NULL || h_C_wmma == NULL){
            printf("[ERR]: Errore nell'allocazione delle matrici sull'host\n");
            return 1;
        }
        printf("[INFO]: Allocazione delle matrici sull'host completata\n");

        // //Allocazione sul device (GPU)
        float *d_A = NULL;
        float *d_B = NULL;
        float *d_C = NULL;
        cudaError_t err1 = checkCudaError(cudaMalloc((void **)&d_A, matrix_size), "Allocazione matrice A su GPU");
        cudaError_t err2 = checkCudaError(cudaMalloc((void **)&d_B, matrix_size), "Allocazione matrice B su GPU");
        cudaError_t err3 = checkCudaError(cudaMalloc((void **)&d_C, matrix_size), "Allocazione matrice C su GPU");

        if(err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess){
            printf("[ERR]: Errore nell'allocazione delle matrici sulla GPU\n");
            return 1;
        }

        for (int row = 0; row < N; ++row){
            for(int col = 0; col < N; ++col){
                h_A[row * N + col] = row == col ? 1 : 0;
                h_B[row * N + col] = (row * N + col) / 1000.0;
            }
        }
        memset(h_C_cublas, 0, matrix_size);
        memset(h_C_wmma, 0, matrix_size);
            
        err1 = checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
        err2 = checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");
        err3 = checkCudaError(cudaMemcpy(d_C, h_C_wmma, matrix_size, cudaMemcpyHostToDevice), "Copia matrice C sulla GPU");
        if(err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess){
            printf("[ERR]: Errore nella copia delle matrici sulla GPU\n");
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
        cublasMatMul(d_A, d_B, d_C, N, &cublasMillis, &cublasTFLOPS);
        // Copia dei risultati dalla GPU all'host
        err1 = checkCudaError(cudaMemcpy(h_C_cublas, d_C, matrix_size, cudaMemcpyDeviceToHost), "Copia matrice C dall'host");
        if(err1 != cudaSuccess){
            printf("[ERR]: Errore nella copia della matrice C dall'host\n");
            return 2;
        }
        err1 = checkCudaError(cudaMemcpy(d_C, h_C_wmma, matrix_size, cudaMemcpyHostToDevice), "Copia matrice C sulla GPU");
        if(err1 != cudaSuccess){
            printf("[ERR]: Errore nella copia della matrice C sulla GPU\n");
            return 2;
        }

        // Converto in half
        half* h_A_h = NULL;
        half* h_B_h = NULL;
        half* d_A_h = NULL;
        half* d_B_h = NULL;
        err1 = convertFloatToHalf(h_A, &h_A_h, &d_A_h, N);
        err2 = convertFloatToHalf(h_B, &h_B_h, &d_B_h, N);
        if(err1 != cudaSuccess || err2 != cudaSuccess){
            printf("[ERR]: Errore nella conversione in half\n");
            return 3;
        }
        tensorCoreMatMul(d_A_h, d_B_h, d_C, N, &myMillis, &myTFLOPS);
        err1 = checkCudaError(cudaMemcpy(h_C_wmma, d_C, matrix_size, cudaMemcpyDeviceToHost), "Copia matrice C dall'host");
        if(err1 != cudaSuccess){
            printf("[ERR]: Errore nella copia della matrice C dall'host\n");
            return 2;
        }

        // Controllo dei risultati
        for(int i = 0; i < N * N; ++i){
            // if(abs(h_C_cublas[i] - h_C_wmma[i]) > 0.0001){
            if(__half2float(h_B_h[i]) != h_C_wmma[i]){
                printf("[ERR]: Errore nei risultati => size: %d, h_C_cublas[%d] != h_C_wmma[%d] | %f != %f\n", N, i, i , h_C_cublas[i], h_C_wmma[i]);
                success = false;
                break;
            }
        }

        // Libera la memoria sull'host
        if(h_A != NULL && h_B != NULL && h_C_cublas != NULL && h_C_wmma != NULL && h_A_h != NULL && h_B_h != NULL){
            free(h_A);
            free(h_B);
            free(h_C_cublas);
            free(h_C_wmma);
            free(h_A_h);
            free(h_B_h);

            h_A = NULL;
            h_B = NULL;
            h_C_cublas = NULL;
            h_C_wmma = NULL;
            h_A_h = NULL;
            h_B_h = NULL;
        }

        //Libero la memoria delle matrici sorgenti
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_A_h);
        cudaFree(d_B_h);
        d_A = NULL;
        d_B = NULL;
        d_A_h = NULL;
        d_B_h = NULL;
        if(!success){
            return -1;
        }else{
            printf("[INFO]: Test size %d PASSED\n\n", N);
        }
    }
    return 0;
}

int main(int argc, char **argv){
    int res = 0;
    printf("\n--------- TESTING BATCHED ---------\n");

    res = testPopulateBlockOfMatrix();
    if(res != 0){
        printf("[ERR]: testPopulateBlockOfMatrix FAILED\n");
        printf("\n--------- TESTING BATCHED FAILED---------\n");
        return res;
    }
    res = testBlockMatrixMultiplication();
    if(res != 0){
        printf("[ERR]: testBlockMatrixMultiplication FAILED\n");
        printf("\n--------- TESTING BATCHED FAILED---------\n");
        return res;
    }
    printf("\n--------- TESTING BATCHED PASSED---------\n");
    return 0;

}


// // Inizializza le matrici A e B sull'host
// if(h_A != NULL && h_B != NULL && h_C_wmma != NULL){
//     printf("[INFO]: Inizializzazione delle matrici sull'host\n");
//     for (int i = 0; i < N; ++i) {
//         for(int j = 0; j < N; ++j){
//             // h_A[i] = 0.1;
//             // h_B[i] = 0.2;                // Blocco [r,c]
//             if(i < 32  && j < 32){          // Blocco [0,0] [ERROR]
//                 h_A[i + j * N] = i + j * N;
//                 h_B[i + j * N] = i + j * N;
//             }else if(i >= 32 && j < 32){    // Blocco [0,1] [OK]
//                 h_A[i + j * N] = 0;
//                 h_B[i + j * N] = 0;
//             }else if(i < 32 && j >= 32){    // Blocco [1,0] [ERROR]
//                 h_A[i + j * N] = 0;
//                 h_B[i + j * N] = 0;
//             }else{                          // Blocco [1,1] [OK]
//                 h_A[i + j * N] = 0;
//                 h_B[i + j * N] = 0;
//             }
//         }
//     }
//     printf("A[0,1]: %f\nA[0,32]: %f\nA[32,0]: %f\nA[32,32]: %f\n\n", h_A[1], h_A[32], h_A[32 * N], h_A[32 * N + 32]);
//     memset(h_C_wmma, 0, matrix_size);
// }else{
//     printf("[ERR]:Errore nell'allocazione delle matrici sull'host\n");
//     return 1;
// }


// // Copia delle matrici dall'host alla GPU
// if(err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess){
//     printf("\n[INFO]: Copia delle matrici sulla GPU\n");
//     checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Copia matrice A sulla GPU");
//     checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Copia matrice B sulla GPU");
//     checkCudaError(cudaMemcpy(d_C, h_C_wmma, matrix_size, cudaMemcpyHostToDevice), "Copia matrice C sulla GPU");
//     printf("[INFO]: Matrici copiate sulla GPU\n");
// }else{
//     printf("[ERR]: Errore nell'allocazione delle matrici sulla GPU\n");
//     return 2;
//     }


// //Indicatori di performance
// float cublasMillis = 0;
// double cublasTFLOPS = 0;

// ///////////////////// ALGORHITMs ///////////////////////
// /////// cuBLAS ///////
// // Moltiplicazione di matrici con cuBLAS
// cublasMatMul(d_A, d_B, d_C, N,&cublasMillis, &cublasTFLOPS); 
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
// printf("\nTempo di esecuzione [cuBLAS] [size: %d]: %f ms\n", cublasMillis, N);
// printf("TFLOPS [cuBLAS] [size: %d]: %f\n\n", cublasTFLOPS, N);