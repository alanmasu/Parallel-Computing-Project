/**!
    @file   test_shared.cu
    @brief  Test load and store from global to shared memory and viceversa

    @author alanmasu
    @date 06/03/2025
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>
#include <ctime>
#include <Utilities.h>

#warning "TESTING SHARED MEMORY"

/**!
    @brief      Function to perform a multiplication between two block of matrices using shared memory
    @details    This function tests the shared memory functions by copying the source matrix to the destination matrix
                using shared memory
*/
__global__ void testSharedMemoryFunctions(float* source, float* destination, int size){
    __shared__ float sharedMem[32 * 32];
    loadBlockToShared(source,       sharedMem,      blockIdx.y, blockIdx.x, size);
    __syncthreads();
    copyBlockToGlobal(sharedMem,    destination,    blockIdx.y, blockIdx.x, size);
    __syncthreads();
}


int testShared(){
    printf("\n--------- TESTING SHARED MEMORY ---------\n");

    //Sizes
    const int size = 64;

    //Testing shared memory
    float* test = NULL;
    float* testDevice = NULL;
    float* destination = NULL;
    float* destinationDevice = NULL;

    
    test = (float*)malloc(size * size * sizeof(float));
    
    if(test == NULL){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on host (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        return 1;
    }

    destination = (float*)malloc(size * size * sizeof(float));
    if(destination == NULL){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on host (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        return 1;
    }

    cudaError_t err = cudaMalloc((void**)&testDevice, size * size * sizeof(float));
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on device (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        return 1;
    }else{
        printf("testDevice: %p\n", testDevice);
    }

    err = cudaMalloc((void**)&destinationDevice, size * size * sizeof(float));
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on device (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        return 1;
    }else{
        printf("destination: %p\n", destinationDevice);
    }

    for(int i = 0; i < size * size; i++){
        test[i] = i;
    }
    
    err = cudaMemcpy(testDevice, test, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed copy from host to device (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        return 2;
    }

    //Calling the kernel
    dim3 blocks(2,2);
    testSharedMemoryFunctions<<<blocks, 1024>>>(testDevice, destinationDevice, size);

    cudaDeviceSynchronize();

    //For testing
    bool success = true;

    printf("\nTESTING COPY\n");
    err = cudaMemcpy(destination, destinationDevice, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed copy from device to host (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        printf("\tError: %s\n\tDescription: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        return 2;
    }else{
        for(int i = 0; i < size * size; i++){
            if(test[i] != destination[i]){
                printf("[ERR]: Test shared memory FAILED -> test[%d] != destination[%d]: %f != %f \t\t(LINE: %d, FILE:%s)\n", i, i, test[i], destination[i],__LINE__, __FILE__);
                success = false;
                break;
            }
        }
    }
    
    if(test != NULL){
        free(test);
    }
    if(testDevice != NULL){
        cudaFree(testDevice);
    }
    if(destination != NULL){
        free(destination);
    }
    if(destinationDevice != NULL){
        cudaFree(destinationDevice);
    }
    if(success){
        printf("[INFO]: Test shared memory PASSED\n");
    }else{
        printf("[ERR]: Test shared memory FAILED\n");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv){
    int res = 0;
    res = testShared();
    if (res == 0){
        printf("--------- Test Shared PASSED ---------\n");
    }else{
        printf("--------- Test Shared FAILED ---------\n");
    }
    return res;
}
