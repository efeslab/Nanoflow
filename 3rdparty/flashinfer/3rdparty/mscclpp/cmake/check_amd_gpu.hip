// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <hip/hip_runtime.h>

__global__ void kernel() {}

int main() {
    int cnt;
    hipError_t err = hipGetDeviceCount(&cnt);
    if (err != hipSuccess || cnt == 0) {
        return 1;
    }
    return 0;
}
