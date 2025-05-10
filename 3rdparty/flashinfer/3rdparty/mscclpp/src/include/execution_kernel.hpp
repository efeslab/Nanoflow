// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_KERNEL_HPP_
#define MSCCLPP_EXECUTION_KERNEL_HPP_

#include <mscclpp/executor.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include "execution_common.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include <mscclpp/gpu_data_types.hpp>

#if defined(MSCCLPP_DEVICE_HIP)
#define __synclds() asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
#endif  // defined(MSCCLPP_DEVICE_HIP)

namespace {
template <typename To, typename From>
MSCCLPP_DEVICE_INLINE To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
MSCCLPP_DEVICE_INLINE T add_elements(T a, T b) {
  return a + b;
}

template <>
MSCCLPP_DEVICE_INLINE __half2 add_elements(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

template <typename T>
MSCCLPP_DEVICE_INLINE int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T>
MSCCLPP_DEVICE_INLINE int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
MSCCLPP_DEVICE_INLINE uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T>
MSCCLPP_DEVICE_INLINE uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE __attribute__((unused)) uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
MSCCLPP_DEVICE_INLINE int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T>
MSCCLPP_DEVICE_INLINE int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE __attribute__((unused)) int add_vectors<__half>(int a, int b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
MSCCLPP_DEVICE_INLINE uint32_t add_vectors_helper(uint32_t a, uint32_t b) {
  return bit_cast<uint32_t, T>(add_elements(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

template <typename T>
MSCCLPP_DEVICE_INLINE uint32_t add_vectors(uint32_t a, uint32_t b) {
  return add_vectors_helper<T>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE uint32_t add_vectors<__half>(uint32_t a, uint32_t b) {
  return add_vectors_helper<__half2>(a, b);
}

}  // namespace
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE)

template <typename T>
MSCCLPP_DEVICE_INLINE T* getBuffer(T* input, T* output, T* scratch, BufferType bufferType) {
  if (bufferType == BufferType::INPUT) {
    return input;
  }
  if (bufferType == BufferType::OUTPUT) {
    return output;
  }
  if (bufferType == BufferType::SCRATCH) {
    return scratch;
  }
  return nullptr;
}

MSCCLPP_DEVICE_INLINE void handleSignal(DeviceHandle<SmChannel>* smChannels,
                                        DeviceHandle<SimpleProxyChannel>* proxyChannels, uint8_t* channelIndex,
                                        int nChannels, ChannelType chType) {
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::SM) {
    smChannels[channelIndex[tid]].signal();
    return;
  }
  if (tid < nChannels && chType == ChannelType::PROXY) {
    proxyChannels[channelIndex[threadIdx.x]].signal();
  }
}

MSCCLPP_DEVICE_INLINE void handleWait(DeviceHandle<SmChannel>* smChannels,
                                      DeviceHandle<SimpleProxyChannel>* proxyChannels, uint8_t* channelIndexes,
                                      int nChannels, ChannelType chType) {
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::SM) {
    smChannels[channelIndexes[tid]].wait();
    return;
  }
  if (tid < nChannels && chType == ChannelType::PROXY) {
    proxyChannels[channelIndexes[tid]].wait();
  }
}

MSCCLPP_DEVICE_INLINE void handleGet(DeviceHandle<SmChannel>* smChannel, uint8_t* srcChannelIndexes,
                                     uint32_t* dstOffsets, uint32_t* srcOffsets, int count, uint32_t size) {
  for (int i = 0; i < count; i++) {
    uint32_t dstOffset = dstOffsets[i];
    uint32_t srcOffset = srcOffsets[i];
    smChannel[srcChannelIndexes[i]].get(dstOffset, srcOffset, size, threadIdx.x, blockDim.x);
  }
}

MSCCLPP_DEVICE_INLINE void handlePut(DeviceHandle<SmChannel>* smChannel,
                                     DeviceHandle<SimpleProxyChannel>* proxyChannels, uint8_t* dstChannelIndexes,
                                     uint32_t* dstOffsets, uint32_t* srcOffsets, int count, uint32_t size,
                                     ChannelType chType) {
  if (chType == ChannelType::SM) {
    for (int i = 0; i < count; i++) {
      uint32_t dstOffset = dstOffsets[i];
      uint32_t srcOffset = srcOffsets[i];
      smChannel[dstChannelIndexes[i]].put(dstOffset, srcOffset, size, threadIdx.x, blockDim.x);
    }
    return;
  }
  if (chType == ChannelType::PROXY) {
    for (int i = 0; i < count; i++) {
      uint32_t dstOffset = dstOffsets[i];
      uint32_t srcOffset = srcOffsets[i];
      proxyChannels[dstChannelIndexes[i]].put(dstOffset, srcOffset, size);
    }
  }
}

template <typename T>
MSCCLPP_DEVICE_INLINE void handleReadReduceCopySend(T* output, uint32_t outputOffsetByBytes, T* input,
                                                    uint32_t inputOffsetByBytes, DeviceHandle<SmChannel>* smChannels,
                                                    uint8_t* dstChannelIndexes, uint8_t* srcChannelIndexes,
                                                    uint32_t* dstOffsets, uint32_t* srcOffsets, int nDstChannels,
                                                    int nSrcChannels, uint32_t size, bool sendToRemote = true) {
  const size_t nInt4 = size / sizeof(int4);
  const size_t inputOffset4 = inputOffsetByBytes / sizeof(int4);
  const size_t outputOffset4 = outputOffsetByBytes / sizeof(int4);
  int4* input4 = (int4*)input;
  int4* output4 = (int4*)output;
  for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = input4[inputOffset4 + idx];
    for (int index = 0; index < nSrcChannels; ++index) {
      int4 val;
      size_t srcOffset = srcOffsets[index] / sizeof(int4);
      val = smChannels[srcChannelIndexes[index]].read<int4>(srcOffset + idx);
      tmp = add_vectors<T>(tmp, val);
    }
    output4[outputOffset4 + idx] = tmp;
    if (sendToRemote) {
      for (int index = 0; index < nDstChannels; ++index) {
        size_t dstOffset = dstOffsets[index] / sizeof(int4);
        smChannels[dstChannelIndexes[index]].write<int4>(dstOffset + idx, tmp);
      }
    }
  }
  // handle rest of data
  size_t processed = nInt4 * sizeof(int4);
  const size_t startIdx = (inputOffsetByBytes + processed) / sizeof(T);
  const size_t endIdx = (inputOffsetByBytes + size) / sizeof(T);
  for (size_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    T tmp = input[idx];
    for (int index = 0; index < nSrcChannels; ++index) {
      size_t srcOffset = srcOffsets[index] / sizeof(T);
      tmp += smChannels[srcChannelIndexes[index]].read<T>(srcOffset + idx);
    }
    output[idx] = tmp;
    if (sendToRemote) {
      for (int index = 0; index < nDstChannels; ++index) {
        size_t dstOffset = dstOffsets[index] / sizeof(T);
        smChannels[dstChannelIndexes[index]].write<T>(dstOffset + idx, tmp);
      }
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handlePutPacket(uint32_t inputOffsetByBytes, size_t scratchSize,
                                           DeviceHandle<SmChannel>* smChannels, uint8_t* dstChannelIndexes,
                                           uint32_t* dstOffsets, int nDstChannels, uint32_t size, uint32_t flag) {
  const size_t scratchBaseOffset = flag & 0x1 ? 0 : scratchSize >> 1;
  for (int index = 0; index < nDstChannels; ++index) {
    smChannels[dstChannelIndexes[index]].putPackets<PacketType>(
        scratchBaseOffset + dstOffsets[index] * 2, inputOffsetByBytes, size, threadIdx.x, blockDim.x, flag);
  }
}

template <typename T, typename PacketType>
MSCCLPP_DEVICE_INLINE void handleReduceSendPacket(T* dst, uint32_t dstOffsetByBytes, T* src, uint32_t srcOffsetByBytes,
                                                  T* inputBuff, size_t inputBuffSize, uint32_t* inputOffsets, int nSrcs,
                                                  DeviceHandle<SmChannel>* smChannels, uint8_t* outputChannelIndexes,
                                                  uint32_t* outputOffsets, int nDstChannels, size_t size,
                                                  uint32_t flag) {
  size_t nPackets = size * 2 / sizeof(PacketType);
  const size_t intputBaseOffset = flag & 0x1 ? 0 : inputBuffSize >> 1;
  const uint32_t srcOffset = srcOffsetByBytes / sizeof(PacketPayload<PacketType>);
  const uint32_t dstOffset = dstOffsetByBytes / sizeof(PacketPayload<PacketType>);
  PacketPayload<PacketType>* srcPacketPayload = (PacketPayload<PacketType>*)src + srcOffset;
  PacketPayload<PacketType>* dstPacketPayload = (PacketPayload<PacketType>*)dst + dstOffset;
  for (size_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = {};
    for (int index = 0; index < nSrcs; ++index) {
      PacketType* pkt = (PacketType*)((char*)inputBuff + intputBaseOffset + 2 * inputOffsets[index]);
      PacketPayload<PacketType> val = pkt[idx].read(flag);
      data = add_vectors<T>(data, val);
    }
    data = add_vectors<T>(data, srcPacketPayload[idx]);
    dstPacketPayload[idx] = data;

    PacketType pkt(data, flag);
    for (int index = 0; index < nDstChannels; ++index) {
      size_t offset = (intputBaseOffset + outputOffsets[index] * 2) / sizeof(PacketType);
      smChannels[outputChannelIndexes[index]].write(offset + idx, pkt);
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleCopyPacket(void* dst, void* src, size_t srcSize, uint32_t dstOffset,
                                            uint32_t srcOffset, size_t size, uint32_t flag) {
  const size_t outputScratchBaseOffset = flag & 0x1 ? 0 : srcSize >> 1;
  PacketType* srcPackets = (PacketType*)((char*)src + outputScratchBaseOffset + 2 * srcOffset);
  PacketPayload<PacketType>* result = (PacketPayload<PacketType>*)((char*)dst + dstOffset);
  size_t nPackets = size * 2 / sizeof(PacketType);
  for (size_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = srcPackets[idx].read(flag);
    result[idx] = data;
  }
}

template <typename T>
MSCCLPP_DEVICE_INLINE void handleReduceSend(T* dst, uint32_t dstOffsetByBytes, T* src, uint32_t srcOffsetByBytes,
                                            T* input, uint32_t* inputOffsets, DeviceHandle<SmChannel>* smChannels,
                                            uint8_t* outputChannelIndexes, uint32_t* outputOffsets, int nOutChannels,
                                            uint32_t size) {
  const size_t nInt4 = size / sizeof(int4);
  const size_t srcOffset4 = srcOffsetByBytes / sizeof(int4);
  const size_t dstOffset4 = dstOffsetByBytes / sizeof(int4);
  int4* src4 = (int4*)src;
  int4* dst4 = (int4*)dst;
  int4* input4 = (int4*)input;
  for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = src4[srcOffset4 + idx];
    for (int index = 0; index < nOutChannels; ++index) {
      size_t offset = inputOffsets[index] / sizeof(int4);
      int4 val = input4[offset + idx];
      tmp = add_vectors<T>(tmp, val);
    }
    dst4[dstOffset4 + idx] = tmp;
    for (int index = 0; index < nOutChannels; ++index) {
      size_t offset = outputOffsets[index] / sizeof(int4);
      smChannels[outputChannelIndexes[index]].write<int4>(offset + idx, tmp);
    }
  }
  // handle rest of data
  size_t processed = nInt4 * sizeof(int4);
  const size_t startIdx = (srcOffsetByBytes + processed) / sizeof(T);
  const size_t endIdx = (srcOffsetByBytes + size) / sizeof(T);
  for (size_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    T tmp = src[idx];
    for (int index = 0; index < nOutChannels; ++index) {
      size_t offset = inputOffsets[index] / sizeof(T);
      tmp += input[offset + idx];
    }
    dst[idx] = tmp;
    for (int index = 0; index < nOutChannels; ++index) {
      size_t offset = outputOffsets[index] / sizeof(T);
      smChannels[outputChannelIndexes[index]].write<T>(offset + idx, tmp);
    }
  }
}

template <typename T, typename PacketType = LL16Packet>
__global__ void executionKernel([[maybe_unused]] int rank /*for debug*/, T* input, T* output, T* scratch,
                                size_t scratchSize, DeviceExecutionPlan* plan, uint32_t flag) {
  extern __shared__ int4 sharedMem[];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  DeviceExecutionPlan* localPlan = plan + bid;
  for (size_t i = tid; i < sizeof(DeviceExecutionPlan) / sizeof(int4); i += blockDim.x) {
    sharedMem[i] = ((int4*)localPlan)[i];
  }
#if defined(MSCCLPP_DEVICE_HIP)
  __synclds();
#else   // !defined(MSCCLPP_DEVICE_HIP)
  __syncthreads();
#endif  // !defined(MSCCLPP_DEVICE_HIP)
  localPlan = (DeviceExecutionPlan*)sharedMem;
  int nOperations = localPlan->nOperations;
  Operation* operations = localPlan->operations;
  DeviceHandle<SmChannel>* smChannels = localPlan->channels.smChannels;
  DeviceHandle<SimpleProxyChannel>* proxyChannels = localPlan->channels.proxyChannels;

  for (int i = 0; i < nOperations; i++) {
    Operation& op = operations[i];
    if (op.type == OperationType::BARRIER) {
      __syncthreads();
    } else if (op.type == OperationType::SIGNAL) {
      handleSignal(smChannels, proxyChannels, op.outputChannelIndexes, op.nOutputs, op.channelType);
    } else if (op.type == OperationType::WAIT) {
      handleWait(smChannels, proxyChannels, op.inputChannelIndexes, op.nInputs, op.channelType);
    } else if (op.type == OperationType::PUT) {
      handlePut(smChannels, proxyChannels, op.outputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nOutputs,
                op.size, op.channelType);
    } else if (op.type == OperationType::GET) {
      handleGet(smChannels, op.inputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nInputs, op.size);
    } else if (op.type == OperationType::READ_REDUCE_COPY_SEND) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleReadReduceCopySend(dst, op.dstOffset, src, op.srcOffset, smChannels, op.outputChannelIndexes,
                               op.inputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nOutputs, op.nInputs,
                               op.size);
    } else if (op.type == OperationType::READ_REDUCE_COPY) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleReadReduceCopySend(dst, op.dstOffset, src, op.srcOffset, smChannels, op.outputChannelIndexes,
                               op.inputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nOutputs, op.nInputs,
                               op.size, false);
    } else if (op.type == OperationType::PUT_PACKET) {
      handlePutPacket<PacketType>(op.srcOffset, scratchSize, smChannels, op.outputChannelIndexes, op.outputOffsets,
                                  op.nOutputs, op.size, flag);
    } else if (op.type == OperationType::REDUCE_SEND_PACKET) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleReduceSendPacket<T, PacketType>(dst, op.dstOffset, src, op.srcOffset, scratch, scratchSize, op.inputOffsets,
                                            op.nInputs, smChannels, op.outputChannelIndexes, op.outputOffsets,
                                            op.nOutputs, op.size, flag);
    } else if (op.type == OperationType::COPY_PACKET) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleCopyPacket<PacketType>(dst, src, scratchSize, op.dstOffset, op.srcOffset, op.size, flag);
    } else if (op.type == OperationType::REDUCE_SEND) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      T* tmp = getBuffer(input, output, scratch, op.inputBufferType);
      handleReduceSend(dst, op.dstOffset, src, op.srcOffset, tmp, op.inputOffsets, smChannels, op.outputChannelIndexes,
                       op.outputOffsets, op.nOutputs, op.size);
    }
  }
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

class ExecutionKernel {
 public:
#if defined(MSCCLPP_DEVICE_HIP)
  template <typename PacketType>
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                           size_t scratchSize, DataType dataType, DeviceExecutionPlan* plan, size_t sharedMemSize,
                           cudaStream_t stream, uint32_t flag = 0) {
    switch (dataType) {
      case DataType::INT32:
        executionKernel<int32_t, PacketType><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (int32_t*)src, (int32_t*)dst, (int32_t*)scratch, scratchSize, plan, flag);
        break;
      case DataType::UINT32:
        executionKernel<uint32_t, PacketType><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (uint32_t*)src, (uint32_t*)dst, (uint32_t*)scratch, scratchSize, plan, flag);
        break;
      case DataType::FLOAT16:
        executionKernel<half, PacketType><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (half*)src, (half*)dst, (half*)scratch, scratchSize, plan, flag);
        break;
      case DataType::FLOAT32:
        executionKernel<float, PacketType><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (float*)src, (float*)dst, (float*)scratch, scratchSize, plan, flag);
        break;
    }
  }
#else   // !defined(MSCCLPP_DEVICE_HIP)
  template <typename PacketType>
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                           size_t scratchSize, DataType dataType, DeviceExecutionPlan* plan, size_t sharedMemSize,
                           cudaStream_t stream, uint32_t flag = 0);
#endif  // !defined(MSCCLPP_DEVICE_HIP)
};
}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_KERNEL_HPP_
