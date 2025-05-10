// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cuda_runtime.h>

__global__ void kernel() {}

int main() {
    int cnt;
    cudaError_t err = cudaGetDeviceCount(&cnt);
    if (err != cudaSuccess || cnt == 0) {
        return 1;
    }
    return 0;
}
