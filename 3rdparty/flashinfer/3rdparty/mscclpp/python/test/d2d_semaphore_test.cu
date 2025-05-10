// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/semaphore_device.hpp>

// be careful about using semaphore[my_rank] as it is an invalid semaphore and it is there just for simplicity of
// indexing
extern "C" __global__ void __launch_bounds__(1024, 1)
    d2d_semaphore(mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores, int my_rank, int nranks) {
  int tid = threadIdx.x;
  if (tid < nranks && tid != my_rank) {
    semaphores[tid].signal();
    semaphores[tid].wait();
  }
}
