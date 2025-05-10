// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/nvls_device.hpp>
#include <mscclpp/poll_device.hpp>
#include <mscclpp/semaphore_device.hpp>

__device__ mscclpp::DeviceSyncer deviceSyncer;

extern "C" __global__ void __launch_bounds__(1024, 1)
    nvls_test(mscclpp::DeviceMulticastPointerDeviceHandle nvlsPtrs,
              mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores, int my_rank, int nranks, int nbytes) {
  int nelem = nbytes / sizeof(float);
  float* dev_ptr = (float*)nvlsPtrs.devicePtr;
  float* mc_ptr = (float*)nvlsPtrs.mcPtr;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  for (int idx = bid * blockDim.x + tid; idx < nelem; idx += blockDim.x * gridDim.x) {
    dev_ptr[idx] = my_rank;
  }
  deviceSyncer.sync(gridDim.x);
  if (tid == 0 && bid == 0) {
    __threadfence_system();
  }

  if (bid == 0) {
    if (tid < nranks && tid != my_rank) {
      semaphores[tid].signal();
      semaphores[tid].wait();
    }
  }
  deviceSyncer.sync(gridDim.x);

  int my_st = ((int64_t)nelem * (int64_t)my_rank) / (int64_t)nranks;
  int my_en = ((int64_t)nelem * (int64_t)(my_rank + 1)) / (int64_t)nranks;

  int my_offset = (tid + bid * blockDim.x) * 4;
  int my_step = blockDim.x * gridDim.x * 4;

  for (int idx = my_st + my_offset; idx < my_en; idx += my_step) {
    uint4 val;
    DeviceMulticastPointerDeviceHandle::multimemLoad(val, mc_ptr + idx);
    DeviceMulticastPointerDeviceHandle::multimemStore(val, mc_ptr + idx);
  }

  deviceSyncer.sync(gridDim.x);
  if (tid == 0 && bid == 0) {
    __threadfence_system();
  }

  if (bid == 0) {
    if (tid < nranks && tid != my_rank) {
      semaphores[tid].signal();
      semaphores[tid].wait();
    }
  }
  deviceSyncer.sync(gridDim.x);

  for (int idx = bid * blockDim.x + tid; idx < nelem; idx += blockDim.x * gridDim.x) {
    if (dev_ptr[idx] != ((nranks * (nranks - 1)) / 2)) {
      __assert_fail("dev_ptr[idx] != nranks", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
  }
}
