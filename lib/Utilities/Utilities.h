/*!
    @file Utilities.h
    @brief Header file for utility functions used in matrix operations

    @author Alan Masutti (@alamasu)
*/

#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <iostream>
#include <cstdlib>

/*!
    @brief Function to print a matrix

    @param mat Pointer to the matrix
    @param rows Number of rows in the matrix
    @param cols Number of columns in the matrix
*/
template <typename T>
__host__ __device__ void printMat(const T *mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", (float) mat[i * cols + j]);
        }
        printf("\n");
    }
}

/*!
    @brief Function to print a piece of a matrix
    @param mat Pointer to the matrix
    @param rows Number of rows to print
    @param cols Number of columns to print
    @param N Size of the row/column of the matrix
*/
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
