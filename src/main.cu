#include <iostream>
#include "matMul.h"

#define N 4

using namespace std;


int main (int argc, char *argv[]) {

    float A[N * N] = {1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12,
                      13, 14, 15, 16};
    float B[N * N] = {1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12,
                      13, 14, 15, 16};
    float C[N * N] = {0};

    serialMatMul(A, B, C, N);

    cout << endl << "matMul" << endl;
    for(int r = 0; r < N; ++r){
        for(int c = 0; c < N; ++c){
            cout << C[r * N + c] << " ";  
        }
        cout << endl;
    }

    cout << endl << "batchedMatMul" << endl;
    serialBatchedMatMul(A, B, C, N, 2);
    for(int r = 0; r < N; ++r){
        for(int c = 0; c < N; ++c){
            cout << C[r * N + c] << " ";  
        }
        cout << endl;
    }

    return 0;
}