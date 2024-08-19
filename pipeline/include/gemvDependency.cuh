#include "config.h"

#include <cuda_runtime.h>


__global__ void setReadyKernel(int * flag, int batch);

__global__ void clearReadyKernel(int * flag);

__global__ void waitReadyKernel(int * flag, int desired_batch);

class gemvDependency {
public:
    int * device_KQV_ready;
    int * device_GEMV_ready;
    // Constructor
    gemvDependency() {
        cudaMalloc(&device_KQV_ready, sizeof(int));
        cudaMalloc(&device_GEMV_ready, sizeof(int));
        cudaMemset(device_KQV_ready, 0, sizeof(int));
        cudaMemset(device_GEMV_ready, 0, sizeof(int));
    }

    // Destructor
    ~gemvDependency() {
        cudaFree(device_KQV_ready);
        cudaFree(device_GEMV_ready);
    }

    // Method to clear all flags
    void clearAll(cudaStream_t stream) {
        clearReadyKernel<<<1, 1, 0, stream>>>(device_KQV_ready);
        clearReadyKernel<<<1, 1, 0, stream>>>(device_GEMV_ready);
    }

    void incCounter(int* counter, int num, cudaStream_t stream) {
        setReadyKernel<<<1, 1, 0, stream>>>(counter, num);
    }

    // Method to block until GEMV is ready
    void blockUntilGEMVReady(cudaStream_t stream, int desired_batch) {
        waitReadyKernel<<<1, 1, 0, stream>>>(device_GEMV_ready, desired_batch);
    }

private:
    // Disallow copying and assignment
    gemvDependency(const gemvDependency&) = delete;
    gemvDependency& operator=(const gemvDependency&) = delete;


};