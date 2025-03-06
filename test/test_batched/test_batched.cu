//Simple Matrix Multiplication whit cuBLAS

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>
#include <cstdlib>
#include <mma.h>
#include <ctime>
#include <Utilities.h>

#warning "TEST BATCHED"

using namespace nvcuda;

#define N 64
#define BS 32


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
            A[rowInsideBlock * bs + colInsideBlock] = item;
        }
    }
}

void testPopulateBlockOfMatrix(){
    float a[2*2] = {-1.0};
    populateBlockOfMatrix(a, 1, 1, 2, 4);
    printf("Matrice A:\n");
    printMat(a, 2,2);
}

int testBlockMatrixMultiplication(){
    // Puntatori per le matrici sull'host
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C_cublas = NULL;
    float *h_C_wmma = NULL;

    // // Allocazione delle matrici sull'host (CPU)
    printf("\n--------- TESTING BLOC Marix Multiplication ---------\n");
    printf("[INFO]: Allocazione delle matrici sull'host\n");
    size_t matrix_size = N * N * sizeof(float);
    h_A = (float *)malloc(matrix_size);
    h_B = (float *)malloc(matrix_size);
    h_C_cublas = (float *)malloc(matrix_size);
    h_C_wmma = (float *)malloc(matrix_size);
    printf("[INFO]: Allocazione delle matrici sull'host completata\n");

    // //Allocazione sul device (GPU)
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaError_t err1 = checkCudaError(cudaMalloc((void **)&d_A, matrix_size), "Allocazione matrice A su GPU");
    cudaError_t err2 = checkCudaError(cudaMalloc((void **)&d_B, matrix_size), "Allocazione matrice B su GPU");
    cudaError_t err3 = checkCudaError(cudaMalloc((void **)&d_C, matrix_size), "Allocazione matrice C su GPU");

    const int blockNumber = N / BS;
    
    for(int bRow = 0; bRow < blockNumber; ++bRow){
        for(int bCol = 0; bCol < blockNumber; ++bCol){
            printf("Block [%d, %d]\n", bRow, bCol);
            // Inizializza le matrici A e B sull'host
            memset(h_A, 0, matrix_size);
            memset(h_B, 0, matrix_size);
            memset(h_C_cublas, 0, matrix_size);
            
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

    //Libero la memoria delle matrici sorgenti
    if(err1 == cudaSuccess && err2 == cudaSuccess){
        cudaFree(d_A);
        cudaFree(d_B);
        d_A = NULL;
        d_B = NULL;
    }
    return 0;
}

int main(int argc, char **argv){
    testPopulateBlockOfMatrix();
    // testBlockMatrixMultiplication();

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