#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <iostream>
#include <cstdlib>

// Funzione per la stampa di matrici
template <typename T>
__host__ __device__ void printMat(const T *mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", (float) mat[i * cols + j]);
        }
        printf("\n");
    }
}

template <typename T>
__host__ __device__ void printNMat(const T *mat, int rows, int cols, int N) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", (float) mat[i * N + j]);
        }
        printf("\n");
    }
}

#endif // __UTILITIES_H__
