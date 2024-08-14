#include "matMul.h"
#include <iostream>
#include <cstdlib>

void serialBatchedMatMul(const float *A, const float *B, float *C, int N, int blockSize){
    int blockCount = N / blockSize;
    // Initialize the result matrix at zeros
    memset(C, 0, N * N * sizeof(float));
    // Loop over destination blocks rows
    
}

void serialMatMul(const float *A, const float *B, float *C, int N){
    for(int r = 0; r < N; ++r){
        for(int c = 0; c < N; ++c){
            for(int k = 0; k < N; ++k){
                C[r * N + c] += A[r * N + k] * B[k * N + c];
            }
        }
    }
}