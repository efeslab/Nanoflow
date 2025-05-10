// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/fifo_device.hpp>
#include <mscclpp/semaphore_device.hpp>

extern "C" __global__ void __launch_bounds__(1024, 1) proxy(int my_rank, int nranks, mscclpp::FifoDeviceHandle fifo,
                                                            mscclpp::Host2DeviceSemaphoreDeviceHandle* semaphores) {
  int tid = threadIdx.x;
  if (tid == 0) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = 123;
    trigger.snd = 0;
    uint64_t currentFifoHead = fifo.push(trigger);
    // wait for the work to be done in cpu side
    fifo.sync(currentFifoHead);
  }
  __syncthreads();
  if (tid < nranks && tid != my_rank) {
    semaphores[tid].wait();
  }
}
