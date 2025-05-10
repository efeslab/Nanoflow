// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_CHANNEL_DEVICE_HPP_
#define MSCCLPP_PROXY_CHANNEL_DEVICE_HPP_

#include "fifo_device.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

using SemaphoreId = uint32_t;

/// Numeric ID of @ref RegisteredMemory. @ref ProxyService has an internal array indexed by these handles mapping to the
/// actual.
using MemoryId = uint32_t;

using TriggerType = uint64_t;
const TriggerType TriggerData = 0x1;  // Trigger a data transfer.
const TriggerType TriggerFlag = 0x2;  // Trigger a signaling.
const TriggerType TriggerSync = 0x4;  // Trigger a flush.

#define MSCCLPP_BITS_SIZE 32
#define MSCCLPP_BITS_OFFSET 32
#define MSCCLPP_BITS_REGMEM_HANDLE 9
#define MSCCLPP_BITS_TYPE 3
#define MSCCLPP_BITS_CONNID 10
#define MSCCLPP_BITS_FIFO_RESERVED 1

/// Basic structure of each work element in the FIFO.
union ChannelTrigger {
  ProxyTrigger value;
  // The summation of number of bits must be 128 or less.
  struct {
    // First 64 bits: value[0]
    uint64_t size : MSCCLPP_BITS_SIZE;
    uint64_t srcOffset : MSCCLPP_BITS_OFFSET;
    uint64_t : (64 - MSCCLPP_BITS_SIZE - MSCCLPP_BITS_OFFSET);  // ensure 64-bit alignment
    // Second 64 bits: value[1]
    uint64_t dstOffset : MSCCLPP_BITS_OFFSET;
    uint64_t srcMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t dstMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t type : MSCCLPP_BITS_TYPE;
    uint64_t chanId : MSCCLPP_BITS_CONNID;
    uint64_t : (64 - MSCCLPP_BITS_OFFSET - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_TYPE -
                MSCCLPP_BITS_CONNID - MSCCLPP_BITS_FIFO_RESERVED);  // ensure 64-bit alignment
    uint64_t reserved : MSCCLPP_BITS_FIFO_RESERVED;
  } fields;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Default constructor.
  MSCCLPP_DEVICE_INLINE ChannelTrigger() {}

  /// Copy constructor.
  MSCCLPP_DEVICE_INLINE ChannelTrigger(ProxyTrigger value) : value(value) {}

  /// Constructor.
  /// @param type The type of the trigger.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param bytes The bytes of the transfer.
  /// @param semaphoreId The ID of the semaphore.
  MSCCLPP_DEVICE_INLINE ChannelTrigger(TriggerType type, MemoryId dst, uint64_t dstOffset, MemoryId src,
                                       uint64_t srcOffset, uint64_t bytes, int semaphoreId) {
    constexpr uint64_t maskSize = (1ULL << MSCCLPP_BITS_SIZE) - 1;
    constexpr uint64_t maskSrcOffset = (1ULL << MSCCLPP_BITS_OFFSET) - 1;
    constexpr uint64_t maskDstOffset = (1ULL << MSCCLPP_BITS_OFFSET) - 1;
    constexpr uint64_t maskSrcMemoryId = (1ULL << MSCCLPP_BITS_REGMEM_HANDLE) - 1;
    constexpr uint64_t maskDstMemoryId = (1ULL << MSCCLPP_BITS_REGMEM_HANDLE) - 1;
    constexpr uint64_t maskType = (1ULL << MSCCLPP_BITS_TYPE) - 1;
    constexpr uint64_t maskChanId = (1ULL << MSCCLPP_BITS_CONNID) - 1;
    value.fst = (((srcOffset & maskSrcOffset) << MSCCLPP_BITS_SIZE) + (bytes & maskSize));
    value.snd = (((((((((semaphoreId & maskChanId) << MSCCLPP_BITS_TYPE) + ((uint64_t)type & maskType))
                      << MSCCLPP_BITS_REGMEM_HANDLE) +
                     (dst & maskDstMemoryId))
                    << MSCCLPP_BITS_REGMEM_HANDLE) +
                   (src & maskSrcMemoryId))
                  << MSCCLPP_BITS_OFFSET) +
                 (dstOffset & maskDstOffset));
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

struct ProxyChannelDeviceHandle {
  SemaphoreId semaphoreId_;

  Host2DeviceSemaphoreDeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  FifoDeviceHandle fifo_;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a @ref TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset, uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  /// Push a @ref TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    put(dst, offset, src, offset, size);
  }

  /// Push a @ref TriggerFlag to the FIFO.
  MSCCLPP_DEVICE_INLINE void signal() { fifo_.push(ChannelTrigger(TriggerFlag, 0, 0, 0, 0, 1, semaphoreId_).value); }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                           uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData | TriggerFlag, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignal(dst, offset, src, offset, size);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                   uint64_t size) {
    uint64_t curFifoHead = fifo_.push(
        ChannelTrigger(TriggerData | TriggerFlag | TriggerSync, dst, dstOffset, src, srcOffset, size, semaphoreId_)
            .value);
    fifo_.sync(curFifoHead);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(dst, offset, src, offset, size);
  }

  /// Push a @ref TriggerSync to the FIFO.
  MSCCLPP_DEVICE_INLINE void flush() {
    uint64_t curFifoHead = fifo_.push(ChannelTrigger(TriggerSync, 0, 0, 0, 0, 1, semaphoreId_).value);
    fifo_.sync(curFifoHead);
  }

  /// Check if the proxy channel has been signaled.
  /// @return true if the proxy channel has been signaled.
  MSCCLPP_DEVICE_INLINE bool poll() { return semaphore_.poll(); }

  /// Wait for the proxy channel to be signaled.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) { semaphore_.wait(maxSpinCount); }

#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

struct SimpleProxyChannelDeviceHandle {
  ProxyChannelDeviceHandle proxyChan_;
  MemoryId dst_;
  MemoryId src_;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a @ref TriggerData to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.put(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a @ref TriggerData to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t size) { put(offset, offset, size); }

  /// Push a @ref TriggerFlag to the FIFO.
  MSCCLPP_DEVICE_INLINE void signal() { proxyChan_.signal(); }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.putWithSignal(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) { putWithSignal(offset, offset, size); }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(offset, offset, size);
  }

  /// Push a @ref TriggerSync to the FIFO.
  MSCCLPP_DEVICE_INLINE void flush() { proxyChan_.flush(); }

  /// Check if the proxy channel has been signaled.
  /// @return true if the proxy channel has been signaled.
  MSCCLPP_DEVICE_INLINE bool poll() { return proxyChan_.poll(); }

  /// Wait for the proxy channel to be signaled.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) { proxyChan_.wait(maxSpinCount); }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_CHANNEL_DEVICE_HPP_
