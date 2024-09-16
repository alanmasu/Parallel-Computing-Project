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

#define N 8

#warning "Testing mode"

int main(int argc, char **argv){
    printf("WMMA TEST: Testing mode\n");
    // Puntatori per le matrici sull'host
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C_wmma = NULL;

    // // Allocazione delle matrici sull'host (CPU)
    printf("[INFO]: Allocazione delle matrici sull'host\n");
    size_t matrix_size = N * N * sizeof(float);
    h_A = (float *)malloc(matrix_size);
    h_B = (float *)malloc(matrix_size);
    h_C_wmma = (float *)malloc(matrix_size);
    printf("[INFO]: Allocazione delle matrici sull'host completata\n");

    // Inizializza le matrici A e B sull'host
    if(h_A != NULL && h_B != NULL){
        printf("[INFO]: Inizializzazione delle matrici sull'host\n");
        for (int i = 0; i < N * N; ++i) {
            h_A[i] = i;
            h_B[i] = i;
        }
    }else{
        printf("[ERR]:Errore nell'allocazione delle matrici sull'host\n");
        return 1;
    }
    
    ///////////////////// ALGORHITMs ///////////////////////
    /////// serialMatMul ///////
    // Moltiplicazione di matrici con cuBLAS
    serialMatMul(h_A, h_B, h_C_wmma, N);
    //Stampa delle matrici
    if(N <= 8){
        printf("Matrice A:\n");
        printMat(h_A, N, N);
        printf("Matrice B:\n");
        printMat(h_B, N, N);
        printf("Matrice C:\n");
        printMat(h_C_wmma, N, N);
    }

    // Libera la memoria sull'host
    free(h_A);
    free(h_B);
    free(h_C_wmma);

}
