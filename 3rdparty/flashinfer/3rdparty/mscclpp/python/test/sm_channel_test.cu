// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/sm_channel_device.hpp>

// be careful about using channels[my_rank] as it is inavlie and it is there just for simplicity of indexing
extern "C" __global__ void __launch_bounds__(1024, 1)
    sm_channel(mscclpp::SmChannelDeviceHandle* channels, int my_rank, int nranks, int num_elements, int use_packet) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  uint64_t size_per_rank = (num_elements * sizeof(int)) / nranks;
  uint64_t my_offset = size_per_rank * my_rank;
  uint64_t my_nghr_offset = size_per_rank * bid;
  int flag = 123;
  if (bid < nranks && bid != my_rank) {
    if (use_packet) {
      channels[bid].putPackets(2 * my_offset, my_offset, size_per_rank, tid, blockDim.x, flag);
      channels[bid].getPackets(2 * my_nghr_offset, my_nghr_offset, size_per_rank, tid, blockDim.x, flag);
    } else {
      channels[bid].put(my_offset, my_offset, size_per_rank, tid, blockDim.x);
      __syncthreads();
      if (!use_packet && tid == 0) {
        channels[bid].signal();
        channels[bid].wait();
      }
    }
  }
}
