#include "gemvDependency.cuh"

__global__ void setReadyKernel(int * flag, int batch) {
    // use atomic add 
    atomicAdd(flag, batch);
}

__global__ void clearReadyKernel(int * flag) {
    // set to 0
    int k = atomicExch(flag, 0);
    // printf("clearReadyKernel: %d\n", k);
}

__global__ void waitReadyKernel(int * flag, int desired_batch) {
    int t =0;
    t = atomicAdd(flag, 0);
    while (t < desired_batch) {
        // printf("batch %d is not ready\n", t);
        t = atomicAdd(flag, 0);
    }
    // printf("batch %d is ready\n", t);
}