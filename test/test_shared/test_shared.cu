#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <matMul.h>
#include <ctime>
#include <Utilities.h>


/**!
    @brief      Function to perform a multiplication between two block of matrices using shared memory
    @details    This function tests the shared memory functions by copying the source matrix to the destination matrix
                using shared memory
*/
__global__ void testSharedMemoryFunctions(float* source, float* destination00, float* destination11, int size){
    __shared__ float sharedMem[32 * 32];
    loadBlockToShared(source, sharedMem, 0, 0, size);
    copyBlockToGlobal(sharedMem, destination00, 0, 0, size);
    loadBlockToShared(source, sharedMem, 1, 1, size);
    copyBlockToGlobal(sharedMem, destination11, 1, 1, size);
}


int testShared(){
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
        return 1;
    }

    destinationHost = (float*)malloc(size * size * sizeof(float));
    if(destinationHost == NULL){
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

    err = cudaMalloc((void**)&destination, size * size * sizeof(float));
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on device (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        return 1;
    }else{
        printf("destination: %p\n", destination);
    }

    err = cudaMalloc((void**)&destination2, size * size * sizeof(float));
    if(err != cudaSuccess){
        printf("[ERR]: Test shared memory FAILED -> due to failed allocation on device (LINE: %d, FILE:%s)\n", __LINE__, __FILE__);
        return 1;
    }else{
        printf("destination2: %p\n", destination2);
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
    testSharedMemoryFunctions<<<1, 1024>>>(testDevice, destination, destination2, size);

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
        return 2;
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
        return -1;
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
        return -1;
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
