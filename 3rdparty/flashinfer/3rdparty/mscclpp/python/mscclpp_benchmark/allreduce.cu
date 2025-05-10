// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/nvls_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

__device__ mscclpp::DeviceSyncer deviceSyncer;
__device__ mscclpp::DeviceSyncer allGatherDeviceSyncer;
__device__ mscclpp::DeviceSyncer reduceScatterDeviceSyncer;
__device__ mscclpp::DeviceSyncer ibDeviceSyncer;

#ifndef TYPE
#define TYPE float
#endif

#define VECTOR_SIZE (sizeof(int4) / sizeof(TYPE))

template <typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
__forceinline__ __device__ T add_elements(T a, T b) {
  return a + b;
}

template <>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

template <typename T>
__forceinline__ __device__ int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T>
__forceinline__ __device__ int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T>
__forceinline__ __device__ int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int add_vectors<__half>(int a, int b) {
  return add_vectors_helper<__half2>(a, b);
}

__forceinline__ __device__ void vectorSum(TYPE* dst, TYPE* src, size_t nElem, int blockId, int nBlocks) {
  size_t nInt4 = nElem / 4;
  size_t nLastInts = nElem % 4;
  int4* dst4 = (int4*)dst;
  int4* src4 = (int4*)src;
  for (int i = threadIdx.x + blockId * blockDim.x; i < nInt4; i += blockDim.x * nBlocks) {
    dst4[i] = add_vectors<TYPE>(dst4[i], src4[i]);
  }
  if (nLastInts > 0) {
    int* dstLast = ((int*)dst) + nInt4 * 4;
    int* srcLast = ((int*)src) + nInt4 * 4;
    for (int i = threadIdx.x + blockId * blockDim.x; i < nLastInts; i += blockDim.x * nBlocks) {
      dstLast[i] = add_vectors<TYPE>(dstLast[i], srcLast[i]);
    }
  }
}

__forceinline__ __device__ void vectorSum(TYPE* dst, TYPE* src, size_t nElem) {
  vectorSum(dst, src, nElem, blockIdx.x, gridDim.x);
}

// -------------------------------------------
// AllReduce1
// -------------------------------------------

template <int READ_ONLY>
__device__ void allreduce1_helper(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff, int rank, int nranks,
                                  size_t nelems) {
  const size_t chunkSize = nelems / nranks;
  if (nranks == 1) return;
  const int nPeer = nranks - 1;
  const size_t indexOffset = rank * chunkSize;
  const size_t indexOffset4 = indexOffset / VECTOR_SIZE;
  int4* buff4 = (int4*)buff;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // synchronize everyone
  if (tid == 0) {
    __threadfence_system();
  }
  __syncthreads();
  if (tid < nPeer) {
    smChans[tid].relaxedSignal();
  }
  if (tid >= nPeer && tid < nPeer * 2) {
    smChans[tid - nPeer].wait();
  }
  deviceSyncer.sync(gridDim.x);

  // use int4 as much as possible
  const size_t nInt4 = chunkSize / VECTOR_SIZE;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nInt4; idx += blockDim.x * gridDim.x) {
    int4 tmp = buff4[indexOffset4 + idx];
    for (int index = 0; index < nPeer; ++index) {
      int4 val;
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      val = smChans[peerIdx].read<int4>(indexOffset4 + idx);
      tmp = add_vectors<TYPE>(tmp, val);
    }
    if (READ_ONLY == 0) {
      for (int index = 0; index < nPeer; ++index) {
        int peerIdx = (index + rank);
        if (peerIdx >= nPeer) peerIdx -= nPeer;
        smChans[peerIdx].write<int4>(indexOffset4 + idx, tmp);
      }
    }
    buff4[indexOffset4 + idx] = tmp;
  }

  // use the given TYPE for the rest
  size_t processed = nInt4 * VECTOR_SIZE * nranks;
  const size_t nRemElems = nelems - processed;
  const size_t startIdx = processed + (nRemElems * rank) / nranks;
  const size_t endIdx = processed + (nRemElems * (rank + 1)) / nranks;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x + startIdx; idx < endIdx; idx += blockDim.x * gridDim.x) {
    TYPE tmp = buff[idx];
    for (int index = 0; index < nPeer; ++index) {
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      TYPE val = smChans[peerIdx].read<TYPE>(idx);
      tmp += val;
    }
    if (READ_ONLY == 0) {
      for (int index = 0; index < nPeer; ++index) {
        int peerIdx = (index + rank);
        if (peerIdx >= nPeer) peerIdx -= nPeer;
        smChans[peerIdx].write<TYPE>(idx, tmp);
      }
    }
    buff[idx] = tmp;
  }

  // synchronize everyone again
  deviceSyncer.sync(gridDim.x);
  if (tid == 0) {
    __threadfence_system();
  }
  __syncthreads();
  if (tid < nPeer) {
    smChans[tid].relaxedSignal();
  }
  if (tid >= nPeer && tid < nPeer * 2) {
    smChans[tid - nPeer].wait();
  }

  if (READ_ONLY) {
    deviceSyncer.sync(gridDim.x);
    for (int i = 0; i < nPeer; ++i) {
      int peerIdx = (i + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      const int remoteRank = (peerIdx < rank ? peerIdx : peerIdx + 1);
      size_t offset = chunkSize * remoteRank * sizeof(TYPE);
      smChans[peerIdx].get(offset, chunkSize * sizeof(TYPE), tid, blockDim.x * gridDim.x);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024, 1) allreduce1(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff,
                                                                 int rank, int nranks, size_t nelems, int read_only) {
  if (read_only)
    allreduce1_helper<1>(smChans, buff, rank, nranks, nelems);
  else
    allreduce1_helper<0>(smChans, buff, rank, nranks, nelems);
}

// -------------------------------------------
// AllReduce2
// -------------------------------------------

__device__ uint64_t globalFlag = 1;

extern "C" __global__ void __launch_bounds__(1024, 1)
    allreduce2(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff, TYPE* scratch, void* resultBuff, int rank,
               int worldSize, size_t nelems) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));
  // This version of allreduce only works for single nodes
  const int nPeers = worldSize - 1;
  const size_t nPkts = nelems / 2;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank / 2;
  // flag for packets. Initially 1
  const uint32_t flag = (uint32_t)globalFlag;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  mscclpp::SmChannelDeviceHandle smChan = smChans[peerIdx];
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  // double buffering
  size_t scratchBaseOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LLPacket);
  void* scratchBuff = (void*)((char*)scratch + scratchBaseOffset);
  size_t scratchOffset = scratchBaseOffset + rank * nPktsPerRank * sizeof(mscclpp::LLPacket);
  size_t scratchResultOffset =
      (flag & 1) ? 2 * nPkts * sizeof(mscclpp::LLPacket) : 3 * nPkts * sizeof(mscclpp::LLPacket);
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int);
  uint2* src = (uint2*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint2* dst = (uint2*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // step 1: write to scratch buffer
  smChan.putPackets(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid, blockDim.x * nBlocksPerPeer, flag);
  // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
    uint2 data = make_uint2(0, 0);
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratchBuff + remoteRank * nPktsPerRank;
      uint2 val = dstPkt[idx].read(flag);
      data = add_vectors<TYPE>(val, data);
    }
    data = add_vectors<TYPE>(data, src[idx]);
    dst[idx] = data;

    mscclpp::LLPacket packet;
    packet.data1 = data.x;
    packet.flag1 = flag;
    packet.data2 = data.y;
    packet.flag2 = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LLPacket) + (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      smChans[index].write(offset, packet);
    }
  }
  // step 3: get data result from scratch buffer
  mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint2* result = (uint2*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
    uint2 data = dstPkt[idx + dstOffset].read(flag);
    result[idx].x = data.x;
    result[idx].y = data.y;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

// -------------------------------------------
// AllReduce3
// -------------------------------------------

extern "C" __global__ void __launch_bounds__(1024, 1)
    allreduce3(mscclpp::SimpleProxyChannelDeviceHandle* fstRoundChans,
               mscclpp::SimpleProxyChannelDeviceHandle* sndRoundChans, TYPE* buff, TYPE* scratch, int rank,
               int worldSize, size_t nelems) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));

  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int remoteSendRank = (rank + 1) % worldSize;
  int remoteRecvRank = (rank + worldSize - 1) % worldSize;
  int peerSendId = (remoteSendRank < rank) ? remoteSendRank : remoteSendRank - 1;
  int peerRecvId = (remoteRecvRank < rank) ? remoteRecvRank : remoteRecvRank - 1;

  mscclpp::SimpleProxyChannelDeviceHandle& devFstSendChan = fstRoundChans[peerSendId];
  mscclpp::SimpleProxyChannelDeviceHandle& devFstRecvChan = fstRoundChans[peerRecvId];
  mscclpp::SimpleProxyChannelDeviceHandle& devSndSendChan = sndRoundChans[peerSendId];
  mscclpp::SimpleProxyChannelDeviceHandle& devSndRecvChan = sndRoundChans[peerRecvId];

  // Step 1
  size_t chunkIndex = (rank + worldSize - 1) % worldSize;
  size_t chunkNelem = nelems / worldSize;
  size_t offset = chunkIndex * chunkNelem * sizeof(int);
  if (isComm) {
    if (chunkNelem > 1) {
      devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
    }
  }

  // Step 2 ~ Step n-1
  for (int step = 2; step < worldSize; ++step) {
    if (isComm) {
      if (chunkNelem > 1) {
        devFstRecvChan.wait();
        devFstSendChan.flush();
      }
      devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(int), (chunkNelem - chunkNelem / 2) * sizeof(int));
    }
    deviceSyncer.sync(gridDim.x);

    // Reduce
    chunkIndex = (rank + worldSize - step) % worldSize;
    offset = chunkIndex * chunkNelem * sizeof(int);
    int* dst = (int*)((char*)buff + offset);
    int* src = (int*)((char*)scratch + offset);
    vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem / 2);

    if (isComm) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
      if (chunkNelem > 1) {
        devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
      }
    }
    deviceSyncer.sync(gridDim.x);

    dst += chunkNelem / 2;
    src += chunkNelem / 2;
    vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem - chunkNelem / 2);
  }

  // Step n
  if (isComm) {
    if (chunkNelem > 1) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
    }
    devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(int), (chunkNelem - chunkNelem / 2) * sizeof(int));
  }
  deviceSyncer.sync(gridDim.x);

  offset = rank * chunkNelem * sizeof(int);
  int* dst = (int*)((char*)buff + offset);
  int* src = (int*)((char*)scratch + offset);
  vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem / 2);

  if (isComm) {
    devFstRecvChan.wait();
    devFstSendChan.flush();
    if (chunkNelem > 1) {
      devSndSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
    }
  }
  deviceSyncer.sync(gridDim.x);

  dst += chunkNelem / 2;
  src += chunkNelem / 2;
  vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem - chunkNelem / 2);

  if (isComm) {
    if (chunkNelem > 1) {
      devSndRecvChan.wait();
      devSndSendChan.flush();
    }
    devSndSendChan.putWithSignalAndFlush(offset + chunkNelem / 2 * sizeof(int),
                                         (chunkNelem - chunkNelem / 2) * sizeof(int));
  }

  // Step n+1 ~ Step 2n-2
  for (int i = 1; i < worldSize - 1; ++i) {
    if (isComm) {
      devSndRecvChan.wait();
    }
    deviceSyncer.sync(gridDim.x);

    // Copy
    chunkIndex = (rank + worldSize - i) % worldSize;
    if (isComm) {
      devSndSendChan.putWithSignalAndFlush(chunkIndex * chunkNelem * sizeof(int), chunkNelem * sizeof(int));
    }
  }

  // Final receive
  if (isComm) {
    devSndRecvChan.wait();
  }
}

// -------------------------------------------
// AllReduce4
// 2-node
// -------------------------------------------
__device__ void localReduceScatterSm(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff, int rank, int nRanksPerNode,
                                     int startChunkIndex, size_t offsetInChunk, size_t chunkSize, size_t nelems,
                                     int nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;
  const int nPeer = nRanksPerNode - 1;

  const size_t localRankIndexInNode = rank % nRanksPerNode;
  const size_t indexOffset = ((localRankIndexInNode + startChunkIndex) * chunkSize + offsetInChunk);
  const size_t indexOffset4 = indexOffset / 4;

  int4* buff4 = (int4*)buff;

  for (int peerIdx = threadIdx.x + blockIdx.x * blockDim.x; peerIdx < nPeer; peerIdx += blockDim.x * nBlocks) {
    smChans[peerIdx].relaxedSignal();
  }
  for (int peerIdx = threadIdx.x + blockIdx.x * blockDim.x; peerIdx < nPeer; peerIdx += blockDim.x * nBlocks) {
    smChans[peerIdx].wait();
  }
  reduceScatterDeviceSyncer.sync(nBlocks);

  const size_t nInt4 = nelems / 4;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nInt4; idx += blockDim.x * nBlocks) {
    int4 tmp = buff4[indexOffset4 + idx];
    for (int index = 0; index < nPeer; ++index) {
      int4 val;
      int peerIdx = index + localRankIndexInNode;
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      val = smChans[peerIdx].read<int4>(indexOffset4 + idx);
      tmp = add_vectors<TYPE>(tmp, val);
    }
    buff4[indexOffset4 + idx] = tmp;
  }

  // TODO: deal with rest elements
}

// This kernel is the most performant when the number of blocks is a multiple of (nRanksPerNode - 1).
__device__ void localAllGatherSm(mscclpp::SmChannelDeviceHandle* smChans, int rank, int nRanksPerNode,
                                 int startRankChunkIndex, uint64_t offsetInRankChunk, uint64_t rankChunkSize,
                                 uint64_t size, size_t nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t peerIdx = blockIdx.x % nPeer;
  const size_t nBlockForThisPeer = nBlocks / nPeer + (nBlocks % nPeer > peerIdx ? 1 : 0);
  const size_t peerLocalBlockIdx = blockIdx.x / nPeer;
  const size_t rankLocalIndex = rank % nRanksPerNode;
  const int remoteRankLocalIndex = (peerIdx < rankLocalIndex ? peerIdx : peerIdx + 1);

  // Split the data into chunks for aligned data access. Ignore the remainder here and let the last block handle it.
  constexpr size_t chunkBytes = 128;  // heuristic value
  const size_t nChunk = size / chunkBytes;
  const size_t nMinChunkPerBlock = nChunk / nBlockForThisPeer;
  const size_t nRemainderChunk = nChunk % nBlockForThisPeer;

  // Distribute chunks to blocks
  size_t nChunkForThisBlock;
  size_t offsetForThisBlock;
  if (peerLocalBlockIdx < nRemainderChunk) {
    nChunkForThisBlock = nMinChunkPerBlock + 1;
    offsetForThisBlock = (nMinChunkPerBlock + 1) * peerLocalBlockIdx;
  } else {
    nChunkForThisBlock = nMinChunkPerBlock;
    offsetForThisBlock =
        (nMinChunkPerBlock + 1) * nRemainderChunk + (peerLocalBlockIdx - nRemainderChunk) * nMinChunkPerBlock;
  }
  offsetForThisBlock *= chunkBytes;

  // Calculate the size of the data for this block
  size_t sizeForThisBlock = nChunkForThisBlock * chunkBytes;
  const size_t lastChunkSize = size - nChunk * chunkBytes;
  if (lastChunkSize > 0 && peerLocalBlockIdx == nBlockForThisPeer - 1) {
    sizeForThisBlock += lastChunkSize;
  }
  if (threadIdx.x == 0 && peerLocalBlockIdx == 0) {
    smChans[peerIdx].relaxedSignal();
    smChans[peerIdx].wait();
  }
  allGatherDeviceSyncer.sync(nBlocks);
  size_t offset = rankChunkSize * (startRankChunkIndex + remoteRankLocalIndex) + offsetInRankChunk;
  smChans[peerIdx].get(offset + offsetForThisBlock, sizeForThisBlock, threadIdx.x, blockDim.x);
}

__device__ void localAllGatherAllPairsSm(mscclpp::SmChannelDeviceHandle* smChans, int rank, int nRanksPerNode,
                                         uint64_t size, size_t nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nPeer = nRanksPerNode - 1;

  if (tid < nPeer) {
    smChans[tid].signal();
  }
  int waitStart = nBlocks * blockDim.x - nPeer;
  if (tid >= waitStart && tid < nBlocks * blockDim.x) {
    smChans[tid - waitStart].wait();
  }
  allGatherDeviceSyncer.sync(nBlocks);
  for (int i = 0; i < nPeer; ++i) {
    int peerIdx = (i + rank) % nPeer;
    const int remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    size_t offset = size * remoteRankLocalIndex;
    smChans[peerIdx].get(offset, size, tid, blockDim.x * nBlocks);
  }
}

// This is an allgather4 equivalent
__device__ void allGatherSm(mscclpp::SmChannelDeviceHandle* smChans,
                            mscclpp::SimpleProxyChannelDeviceHandle* proxyChans, int rank, int worldSize,
                            int nRanksPerNode, size_t nelemsPerGPU, int pipelineDepth) {
  // this allgather is a pipelined and hierarchical one and only works for two nodes
  // it is implemented as follows:
  // Step 1: each node does a local allgather and concurrently,
  // local GPU i exchange (piplineSize-1)/pipelineSize portion of their data with
  // its cross-node neighbor (local GPU i on the other node) via IB
  // Step 2: each node does a local allgather again with the data just received from its
  // cross-node neighbor in step 1, and concurrently, exchange the rest of the data with
  // its cross-node neighbor
  // Step 3: each node does a local allgather for the last time with the rest of the data

  int pipelineSize = pipelineDepth;
  int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  mscclpp::SimpleProxyChannelDeviceHandle proxyChan = proxyChans[peer];
  const size_t nBlocksForLocalAllGather = gridDim.x / (nRanksPerNode - 1) * (nRanksPerNode - 1);
  const size_t rankChunkSize = nelemsPerGPU * sizeof(int);
  const int startRankIndexInLocalNode = (rank / nRanksPerNode) * nRanksPerNode;
  const int startRankIndexInPeerNode = (peerRank / nRanksPerNode) * nRanksPerNode;

  if (peerNodeId == rank / nRanksPerNode) {
    localAllGatherSm(smChans, rank, nRanksPerNode, 0, 0, rankChunkSize, rankChunkSize, gridDim.x);
    return;
  }

  constexpr size_t alignment = 128;
  size_t step1Bytes = (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int);
  step1Bytes = step1Bytes / alignment * alignment;
  const size_t step2Bytes = nelemsPerGPU * sizeof(int) - step1Bytes;

  // Step 1
  if (threadIdx.x == 0 && blockIdx.x == 0 && step1Bytes > 0) {
    proxyChan.putWithSignal(rank * nelemsPerGPU * sizeof(int), step1Bytes);
  }
  localAllGatherSm(smChans, rank, nRanksPerNode, startRankIndexInLocalNode, 0, rankChunkSize, rankChunkSize,
                   nBlocksForLocalAllGather);
  if (threadIdx.x == 0 && blockIdx.x == 0 && step1Bytes > 0) {
    proxyChan.wait();
    proxyChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 2
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    proxyChan.putWithSignal(rank * nelemsPerGPU * sizeof(int) + step1Bytes, step2Bytes);
  }
  if (step1Bytes > 0)
    localAllGatherSm(smChans, rank, nRanksPerNode, startRankIndexInPeerNode, 0, rankChunkSize, step1Bytes,
                     nBlocksForLocalAllGather);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    proxyChan.wait();
    proxyChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 3
  localAllGatherSm(smChans, rank, nRanksPerNode, startRankIndexInPeerNode, step1Bytes, rankChunkSize, step2Bytes,
                   nBlocksForLocalAllGather);
}

__device__ void reduceScatterSm(mscclpp::SmChannelDeviceHandle* smChans,
                                mscclpp::SimpleProxyChannelDeviceHandle* proxyChans, TYPE* buff, TYPE* scratch,
                                int rank, int nRanksPerNode, int worldSize,
                                size_t nelems,  // must be divisible by 3
                                int pipelineDepth) {
  // this reduce-scatter algorithm works as follows:
  // Step 1: each node does a local reduce-scatter on peer node data chunks with 1/pipeline portion of chunk data. For
  // example, 2 nodes and each node has 2 ranks. rank 0 and rank 1 perform reduce-scatter on chunk 2 and chunk 3, with
  // 1/pipeline portion of the data.
  // Step 2: each node does a local reduce-scatter on peers data chunks with (pipeline-1)/pipeline portion of chunk
  // data. Meanwhile, exchange the reduced data of the previous step with its cross-node neighbor (same local rank
  // number on the other node) via IB. Then performs a reduce operation.
  // Step 3:  each node does a local reduce-scatter on local ranks, meanwhile exchange the reduced data of the previous
  // step with its cross-node neighbor (same local rank number on the other node) via IB. Then performs a reduce
  // operation.
  int pipelineSize = pipelineDepth;
  float nBlocksForReduceScatterRatio = 0.8;
  const size_t chunkSize = nelems / worldSize;
  const int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int nBlocksForReduceScatter =
      (int)(nBlocksForReduceScatterRatio * gridDim.x) / (nRanksPerNode - 1) * (nRanksPerNode - 1);
  int isComm = (threadIdx.x == 0) && (blockIdx.x == nBlocksForReduceScatter);
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  int nBlocksRemain = gridDim.x - nBlocksForReduceScatter;
  mscclpp::SimpleProxyChannelDeviceHandle proxyChan = proxyChans[peer];
  if (peerNodeId == rank / nRanksPerNode) {
    localReduceScatterSm(smChans, buff, rank, nRanksPerNode, 0, 0, chunkSize, chunkSize, gridDim.x);
    return;
  }

  // step 1: local reduce
  int startChunkIndex = peerNodeId * nRanksPerNode;
  localReduceScatterSm(smChans, buff, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize / pipelineSize,
                       nBlocksForReduceScatter);
  deviceSyncer.sync(gridDim.x);

  // step 2: local reduce and exchange data with neighbor
  if (isComm) {
    size_t offset = (peerRank * chunkSize) * sizeof(int);
    // opposite side
    proxyChan.putWithSignal(offset, (chunkSize / pipelineSize * sizeof(int)));
  }
  if (pipelineSize > 1)
    localReduceScatterSm(smChans, buff, rank, nRanksPerNode, startChunkIndex, chunkSize / pipelineSize, chunkSize,
                         (pipelineSize - 1) * chunkSize / pipelineSize, nBlocksForReduceScatter);
  if (isComm) {
    proxyChan.wait();
  }
  if (blockIdx.x >= nBlocksForReduceScatter) {
    ibDeviceSyncer.sync(nBlocksRemain);
    // reduce data received from peer to related rank
    size_t offset = rank * chunkSize * sizeof(int);
    int* dst = (int*)((char*)buff + offset);
    int* src = (int*)((char*)scratch + offset);
    vectorSum((TYPE*)dst, (TYPE*)src, chunkSize / pipelineSize, blockIdx.x - nBlocksForReduceScatter, nBlocksRemain);
  }
  if (isComm) {
    proxyChan.flush();
  }
  deviceSyncer.sync(gridDim.x);

  // step 3: local reduce and exchange data with neighbor
  startChunkIndex = (rank / nRanksPerNode) * nRanksPerNode;
  if (isComm && pipelineSize > 1) {
    size_t offset = (peerRank * chunkSize + chunkSize / pipelineSize) * sizeof(int);
    proxyChan.putWithSignal(offset, (pipelineSize - 1) * chunkSize / pipelineSize * sizeof(int));
  }
  localReduceScatterSm(smChans, buff, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize,
                       nBlocksForReduceScatter);
  if (isComm && pipelineSize > 1) {
    proxyChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  // reduce to related rank, can not overlap since localReduceScatter also calculate the sum
  size_t offset = (rank * chunkSize + chunkSize / pipelineSize) * sizeof(int);
  int* dst = (int*)((char*)buff + offset);
  int* src = (int*)((char*)scratch + offset);
  if (pipelineSize > 1) vectorSum((TYPE*)dst, (TYPE*)src, (pipelineSize - 1) * chunkSize / pipelineSize);
  if (isComm) {
    proxyChan.flush();
  }
}

extern "C" __global__ void __launch_bounds__(1024, 1) __global__
    allreduce4(mscclpp::SmChannelDeviceHandle* smChans,
               mscclpp::SimpleProxyChannelDeviceHandle* reduceScatterProxyChans,
               mscclpp::SimpleProxyChannelDeviceHandle* allGatherProxyChans, TYPE* buff, TYPE* scratch, int rank,
               int nRanksPerNode, int worldSize, size_t nelems, int pipelineDepth) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));
  reduceScatterSm(smChans, reduceScatterProxyChans, buff, scratch, rank, nRanksPerNode, worldSize, nelems,
                  pipelineDepth);
  deviceSyncer.sync(gridDim.x);
  allGatherSm(smChans, allGatherProxyChans, rank, worldSize, nRanksPerNode, nelems / worldSize, pipelineDepth);
}

// allreduce 5 for 2-nodes
extern "C" __global__ void __launch_bounds__(1024, 1)
    allreduce5(mscclpp::SmChannelDeviceHandle* smChans, mscclpp::SimpleProxyChannelDeviceHandle* proxyChans, TYPE* buff,
               TYPE* scratch, TYPE* putBuff, TYPE* resultBuff, int rank, int nRanksPerNode, int worldSize,
               size_t nelems) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));
  // This version of allreduce only works for single nodes
  const int nPeersInNode = nRanksPerNode - 1;
  const int nPkts = nelems / 2;
  const int nelemsPerLocalRank = nelems / nRanksPerNode;
  const int nPktsPerLocalRank = nelemsPerLocalRank / 2;
  const int localRankId = rank % nRanksPerNode;
  // flag for packets. Initially 1
  const uint32_t flag = (uint32_t)globalFlag;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeersInNode;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRankIdx = peerIdx < localRankId ? peerIdx : peerIdx + 1;
  mscclpp::SmChannelDeviceHandle smChan = smChans[peerIdx];
  mscclpp::SimpleProxyChannelDeviceHandle proxyChan = proxyChans[localRankId];
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  // double buffering
  size_t scratchBaseOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LLPacket);
  size_t putBaseOffset = (flag & 1) ? 0 : nPktsPerLocalRank * sizeof(mscclpp::LLPacket);
  void* scratchBuff = (void*)((char*)scratch + scratchBaseOffset);
  size_t scratchOffset = scratchBaseOffset + localRankId * nPktsPerLocalRank * sizeof(mscclpp::LLPacket);
  size_t scratchResultOffset =
      (flag & 1) ? 2 * nPkts * sizeof(mscclpp::LLPacket) : 3 * nPkts * sizeof(mscclpp::LLPacket);
  size_t srcOffset = remoteRankIdx * nelemsPerLocalRank * sizeof(int);
  uint2* src = (uint2*)((char*)buff + localRankId * nelemsPerLocalRank * sizeof(int));
  uint2* dst = (uint2*)((char*)resultBuff + localRankId * nelemsPerLocalRank * sizeof(int));

  // step 1: write to scratch buffer
  if (nRanksPerNode > 1) {
    smChan.putPackets(scratchOffset, srcOffset, nelemsPerLocalRank * sizeof(int), tid, blockDim.x * nBlocksPerPeer,
                      flag);
  }
  // step 2: get data from scratch buffer, do local reduce-scatter in each node.
  mscclpp::LLPacket* putPkt = (mscclpp::LLPacket*)((char*)putBuff + putBaseOffset);
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerLocalRank; idx += blockDim.x * gridDim.x) {
    uint2 data = make_uint2(0, 0);
    for (int index = 0; index < nPeersInNode; index++) {
      const int remoteRank = index < localRankId ? index : index + 1;
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratchBuff + remoteRank * nPktsPerLocalRank;
      uint2 val = dstPkt[idx].read(flag);
      data = add_vectors<TYPE>(val, data);
    }
    data = add_vectors<TYPE>(data, src[idx]);
    putPkt[idx].write(data.x, data.y, flag);
    dst[idx] = data;
  }
  deviceSyncer.sync(gridDim.x);
  // step 3. send local reduced data to remote node.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    proxyChan.put(scratchOffset, putBaseOffset, nPktsPerLocalRank * sizeof(mscclpp::LLPacket));
    if ((flag & 63) == 0) {
      proxyChan.flush();
    }
  }
  // step 4. try to read the data from scratch buffer and write to local peers
  mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratchBuff + localRankId * nPktsPerLocalRank;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerLocalRank; idx += blockDim.x * gridDim.x) {
    uint2 res = dst[idx];
    uint2 val = dstPkt[idx].read(flag);
    res = add_vectors<TYPE>(res, val);

    mscclpp::LLPacket packet;
    packet.data1 = res.x;
    packet.flag1 = flag;
    packet.data2 = res.y;
    packet.flag2 = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LLPacket) + (idx + localRankId * nPktsPerLocalRank);
    for (int index = 0; index < nPeersInNode; index++) {
      smChans[index].write(offset, packet);
    }
    dst[idx] = res;
  }

  // step 5: get data result from scratch buffer
  dstPkt = (mscclpp::LLPacket*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRankIdx * nPktsPerLocalRank;
  uint2* result = (uint2*)((char*)resultBuff + remoteRankIdx * nelemsPerLocalRank * sizeof(int));
  if (nRanksPerNode > 1) {
    for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerLocalRank;
         idx += blockDim.x * nBlocksPerPeer) {
      uint2 data = dstPkt[idx + dstOffset].read(flag);
      result[idx] = data;
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

// -------------------------------------------
// AllReduce6
// NVLS
// -------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

extern "C" __global__ void __launch_bounds__(1024, 1)
    allreduce6(mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores,
               mscclpp::DeviceMulticastPointerDeviceHandle nvlsPtrs, TYPE* buff, int my_rank, int nranks,
               size_t nelem) {
  float* dev_ptr = (float*)nvlsPtrs.devicePtr;
  float* mc_ptr = (float*)nvlsPtrs.mcPtr;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (tid == 0 && bid == 0) {
    __threadfence_system();
  }
  if (bid == 0) {
    if (tid < nranks - 1) {
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
    if (tid < nranks - 1) {
      semaphores[tid].signal();
      semaphores[tid].wait();
    }
  }
  deviceSyncer.sync(gridDim.x);
}
#endif
