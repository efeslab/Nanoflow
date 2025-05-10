// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SM_CHANNEL_DEVICE_HPP_
#define MSCCLPP_SM_CHANNEL_DEVICE_HPP_

#include "semaphore_device.hpp"
#if defined(MSCCLPP_DEVICE_COMPILE)
#include "packet_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE)

namespace Element {

/// Copy aligned elements from the source memory to the destination memory.
///
/// This function is intended to be collectively called by multiple threads. Each thread copies a part of
/// elements.
///
/// @param dst The destination address.
/// @param src The source address.
/// @param numElems The number of elements to be copied.
/// @param threadId The index of the current thread among all threads running this function. This is different
/// from the `threadIdx` in CUDA.
/// @param numThreads The total number of threads that run this function.
///
template <typename T>
MSCCLPP_DEVICE_INLINE void copy(T* dst, T* src, uint64_t numElems, uint32_t threadId, uint32_t numThreads) {
  T reg;
  for (size_t i = threadId; i < numElems; i += numThreads) {
    // Load to register first.
    reg = src[i];
    // Then store to destination.
    dst[i] = reg;
  }
}

}  // namespace Element

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

/// Channel for accessing peer memory directly from SM.
struct SmChannelDeviceHandle {
  SmDevice2DeviceSemaphoreDeviceHandle semaphore_;
  void* src_;
  void* dst_;
  void* getPacketBuffer_;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Load a value from the remote memory.
  /// @tparam T The type of the value to be loaded.
  /// @param index The index of the value to be loaded. The offset in bytes is calculated as index * sizeof(T).
  /// @return The value loaded.
  template <typename T>
  MSCCLPP_DEVICE_INLINE T read(uint64_t index) {
    return *(reinterpret_cast<T*>(dst_) + index);
  }

  /// Write a value to the remote memory.
  /// @tparam T The type of the value to be written.
  /// @param index The index of the value to be written. The offset in bytes is calculated as index * sizeof(T).
  /// @param v The value to be written.
  template <typename T>
  MSCCLPP_DEVICE_INLINE void write(uint64_t index, const T& v) {
    *(reinterpret_cast<T*>(dst_) + index) = v;
  }

  /// this is a helper for copy function
  template <typename T, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void copy_helper(void* dst, void* src, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
    int* dstInt = reinterpret_cast<int*>(dst);
    int* srcInt = reinterpret_cast<int*>(src);
    const uintptr_t dstPtr = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t srcPtr = reinterpret_cast<uintptr_t>(src);
    const uint64_t numInt = bytes / sizeof(int);
    T* dstElem = reinterpret_cast<T*>((dstPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
    T* srcElem = reinterpret_cast<T*>((srcPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
    uint64_t nFirstInt = (reinterpret_cast<uintptr_t>(dstElem) - dstPtr) / sizeof(int);
    if (CopyRemainder) {
      // Copy the remainder integers at the beginning.
      Element::copy<int>(dstInt, srcInt, nFirstInt, threadId, numThreads);
    }
    // Copy elements.
    constexpr uint64_t nIntPerElem = sizeof(T) / sizeof(int);
    uint64_t nElem = (numInt - nFirstInt) / nIntPerElem;
    Element::copy<T>(dstElem, srcElem, nElem, threadId, numThreads);
    if (CopyRemainder && nIntPerElem > 1) {
      // Copy the remainder integers at the end.
      uint64_t nLastInt = (numInt - nFirstInt) % nIntPerElem;
      Element::copy<int>(dstInt + nFirstInt + nElem * nIntPerElem, srcInt + nFirstInt + nElem * nIntPerElem, nLastInt,
                         threadId, numThreads);
    }
  }

  /// Copy aligned data from the source memory to the destination memory.
  ///
  /// This function is a warpper of Element<T>::copy(). Unlike Element<T>::copy(), this function can copy remainder
  /// bytes when @p CopyRemainder is true. Still, the  16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param dst The destination address. Should be aligned to @p Alignment in the same way as @p src.
  /// @param src The source address. Should be aligned to @p Alignment in the same way as @p dst.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void copy(void* dst, void* src, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
    if (Alignment == 4) {
      copy_helper<int, CopyRemainder>(dst, src, bytes, threadId, numThreads);
    } else if (Alignment == 8) {
      copy_helper<long long, CopyRemainder>(dst, src, bytes, threadId, numThreads);
    } else if (Alignment == 16) {
      copy_helper<longlong2, CopyRemainder>(dst, src, bytes, threadId, numThreads);
    } else {
      static_assert(Alignment == 4 || Alignment == 8 || Alignment == 16, "Unsupported alignment");
    }
  }

  /// Copy data from the local memory (origin) to the remote memory (target).
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses. Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param targetOffset The offset in bytes of the remote address. Should be a multiple of @p Alignment.
  /// @param originOffset The offset in bytes of the local address. Should be a multiple of @p Alignment.
  /// @param originBytes Bytes of the origin to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void put(uint64_t targetOffset, uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                 uint32_t numThreads) {
    copy<Alignment, CopyRemainder>((char*)dst_ + targetOffset, (char*)src_ + originOffset, originBytes, threadId,
                                   numThreads);
  }

  /// Copy data from the remote memory (target) to the local memory (origin).
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses. Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param targetOffset The offset in bytes of the remote address. Should be a multiple of @p Alignment.
  /// @param originOffset The offset in bytes of the local address. Should be a multiple of @p Alignment.
  /// @param originBytes Bytes of the origin to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void get(uint64_t targetOffset, uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                 uint32_t numThreads) {
    // Note that `dst` and `src` are swapped for `get()`.
    copy<Alignment, CopyRemainder>((char*)src_ + originOffset, (char*)dst_ + targetOffset, originBytes, threadId,
                                   numThreads);
  }

  /// Copy data from the local memory (origin) to the remote memory (target).
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses. Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param offset The offset in bytes of the local and remote addresses. Should be a multiple of @p Alignment.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
    put<Alignment, CopyRemainder>(offset, offset, bytes, threadId, numThreads);
  }

  /// Copy data from the remote memory (target) to the local memory (origin).
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses. Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param offset The offset in bytes of the local and remote addresses. Should be a multiple of @p Alignment.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void get(uint64_t offset, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
    get<Alignment, CopyRemainder>(offset, offset, bytes, threadId, numThreads);
  }

  /// Construct @ref LLPacket from the data in the local memory (origin) and write it on the remote packet buffer
  /// (target).
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of packets.
  ///
  /// @param targetOffset The offset in bytes of the remote packet buffer.
  /// @param originOffset The offset in bytes of the local data.
  /// @param originBytes Bytes of the origin to be copied.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  /// @tparam PacketType The packet type. It should be either @ref LL16Packet or @ref LL8Packet.
  ///
  template <typename PacketType = LL16Packet>
  MSCCLPP_DEVICE_INLINE void putPackets(uint64_t targetOffset, uint64_t originOffset, uint64_t originBytes,
                                        uint32_t threadId, uint32_t numThreads, uint32_t flag) {
    mscclpp::putPackets<PacketType>(dst_, targetOffset, src_, originOffset, originBytes, threadId, numThreads, flag);
  }

  /// Retrieve data from @ref LLPacket in the local packet buffer (target) and write it on the local data (origin).
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @param targetOffset The offset in bytes of the local packet buffer.
  /// @param originOffset The offset in bytes of the local data.
  /// @param originBytes Bytes of the origin to be copied.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  /// @tparam PacketType The packet type. It should be either @ref LL16Packet or @ref LL8Packet.
  ///
  template <typename PacketType = LL16Packet>
  MSCCLPP_DEVICE_INLINE void getPackets(uint64_t targetOffset, uint64_t originOffset, uint64_t originBytes,
                                        uint32_t threadId, uint32_t numThreads, uint32_t flag) {
    mscclpp::getPackets<PacketType>(getPacketBuffer_, targetOffset, src_, originOffset, originBytes, threadId,
                                    numThreads, flag);
  }

  /// Signal the remote semaphore.
  ///
  /// This function guarantees that all the memory operation before this function is completed before the remote
  /// semaphore is signaled.
  ///
  MSCCLPP_DEVICE_INLINE void signal() { semaphore_.signal(); }

  /// Signal the remote semaphore.
  ///
  /// This function is a relaxed version of signal() and provides no guarantee on the completion of memory operations.
  /// User requires to call proper fencing before using this function.
  ///
  MSCCLPP_DEVICE_INLINE void relaxedSignal() { semaphore_.relaxedSignal(); }

  /// Signal the remote semaphore for copied packets.
  ///
  /// Unlike @ref signal(), this function provides no guarantee on the completion of memory operations. This is
  /// intended to be used with @ref putPackets() and @ref getPackets() that use flags inside packets to indicate the
  /// completion of copies.
  ///
  MSCCLPP_DEVICE_INLINE void signalPacket() { semaphore_.signalPacket(); }

  /// Increase the counter of the local semaphore.
  MSCCLPP_DEVICE_INLINE void semaphoreIncrement() { semaphore_.semaphoreIncrement(); }

  /// Read the counter of the local semaphore.
  MSCCLPP_DEVICE_INLINE uint64_t semaphoreGetLocal() const { return semaphore_.semaphoreGetLocal(); }

  /// Check if the remote semaphore has signaled.
  /// @return true if the remote semaphore has signaled.
  MSCCLPP_DEVICE_INLINE bool poll() { return semaphore_.poll(); }

  /// Wait for the remote semaphore to send a signal.
  /// @param maxSpinCount The maximum number of spins before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) { semaphore_.wait(maxSpinCount); }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace mscclpp

#endif  // MSCCLPP_SM_CHANNEL_DEVICE_HPP_
