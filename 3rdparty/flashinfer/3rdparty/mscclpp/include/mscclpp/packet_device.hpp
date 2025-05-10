// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PACKET_DEVICE_HPP_
#define MSCCLPP_PACKET_DEVICE_HPP_

#include <cstdint>
#include <type_traits>

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "atomic_device.hpp"
#include "poll_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {
/// LL (low latency) protocol packet.
union alignas(16) LL16Packet {
  // Assume data is written with an atomicity of 8 bytes (IB/RDMA).
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  using Payload = uint2;

#if defined(MSCCLPP_DEVICE_COMPILE)
  ulonglong2 raw_;

  MSCCLPP_DEVICE_INLINE LL16Packet() {}

  MSCCLPP_DEVICE_INLINE LL16Packet(uint2 val, uint32_t flag) {
    data1 = val.x;
    flag1 = flag;
    data2 = val.y;
    flag2 = flag;
  }

  /// Write 8 bytes of data to the packet.
  /// @param val1 The first 4-byte data to write.
  /// @param val2 The second 4-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE_INLINE void write(uint32_t val1, uint32_t val2, uint32_t flag) {
#if defined(MSCCLPP_DEVICE_CUDA)
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(&raw_), "r"(val1), "r"(flag), "r"(val2),
                 "r"(flag));
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    uint4 reg = make_uint4(val1, flag, val2, flag);
    ulonglong2* p = reinterpret_cast<ulonglong2*>(&reg);
    atomicStore(&(raw_.x), p->x, memoryOrderRelaxed);
    atomicStore(&(raw_.y), p->y, memoryOrderRelaxed);
#endif
  }

  /// Write 8 bytes of data to the packet.
  /// @param val The 8-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE_INLINE void write(uint64_t val, uint32_t flag) { write((uint32_t)val, (uint32_t)(val >> 32), flag); }

  /// Helper of @ref read().
  /// @param flag The flag to read.
  /// @param data The 8-byte data read.
  /// @return True if the flag is not equal to the given flag.
  MSCCLPP_DEVICE_INLINE bool readOnce(uint32_t flag, uint2& data) const {
#if defined(MSCCLPP_DEVICE_CUDA)
    uint32_t flag1, flag2;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data.x), "=r"(flag1), "=r"(data.y), "=r"(flag2)
                 : "l"(&raw_));
    return (flag1 != flag) || (flag2 != flag);
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    ulonglong2 reg;
    reg.x = atomicLoad(&(raw_.x), memoryOrderRelaxed);
    reg.y = atomicLoad(&(raw_.y), memoryOrderRelaxed);
    uint4* ptr = reinterpret_cast<uint4*>(&reg);
    data.x = ptr->x;
    data.y = ptr->z;
    return (ptr->y != flag) || (ptr->w != flag);
#endif
  }

  /// Read 8 bytes of data from the packet.
  /// @param flag The flag to read.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  /// @return The 8-byte data read.
  MSCCLPP_DEVICE_INLINE uint2 read(uint32_t flag, int64_t maxSpinCount = 100000000) const {
    uint2 data;
    POLL_MAYBE_JAILBREAK(readOnce(flag, data), maxSpinCount);
    return data;
  }

  /// Clear the packet.
  MSCCLPP_DEVICE_INLINE void clear() { raw_ = make_ulonglong2(0, 0); }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

union alignas(8) LL8Packet {
  // Assume data is written with an atomicity of 8 bytes (IB/RDMA).
  struct {
    uint32_t data;
    uint32_t flag;
  };
  uint64_t raw_;

  using Payload = uint32_t;
#if defined(MSCCLPP_DEVICE_COMPILE)

  MSCCLPP_DEVICE_INLINE LL8Packet() {}

  MSCCLPP_DEVICE_INLINE LL8Packet(uint32_t val, uint32_t flag) {
    this->data = val;
    this->flag = flag;
  }

  MSCCLPP_DEVICE_INLINE void write(uint32_t val, uint32_t flag) {
#if defined(MSCCLPP_DEVICE_CUDA)
    asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" ::"l"(&raw_), "r"(val), "r"(flag));
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    uint2 reg = make_uint2(val, flag);
    uint64_t* p = reinterpret_cast<uint64_t*>(&reg);
    atomicStore(&(raw_), *p, memoryOrderRelaxed);
#endif
  }

  MSCCLPP_DEVICE_INLINE bool readOnce(uint32_t flag, uint32_t& data) const {
#if defined(MSCCLPP_DEVICE_CUDA)
    uint32_t f;
    asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(data), "=r"(f) : "l"(&raw_));
    return (f != flag);
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    uint64_t reg;
    reg = atomicLoad(&(raw_), memoryOrderRelaxed);
    uint2* ptr = reinterpret_cast<uint2*>(&reg);
    data = ptr->x;
    return (ptr->y != flag);
#endif
  }

  MSCCLPP_DEVICE_INLINE uint32_t read(uint32_t flag, int64_t maxSpinCount = 1000000) const {
    uint32_t data;
    POLL_MAYBE_JAILBREAK(readOnce(flag, data), maxSpinCount);
    return data;
  }

  /// Clear the packet.
  MSCCLPP_DEVICE_INLINE void clear() { raw_ = 0; }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

using LLPacket = LL16Packet;

#if defined(MSCCLPP_DEVICE_COMPILE)
/// Read data from the origin and write LL16Packets to the target buffer.
///
/// @param targetPtr The target buffer.
/// @param targetOffset The offset in the target buffer.
/// @param originPtr The origin buffer.
/// @param originOffset The offset in the origin buffer.
/// @param originBytes The number of bytes to write to the target buffer.
/// @param threadId The thread ID. The thread ID should be less than @p numThreads.
/// @param numThreads The number of threads that call this function.
/// @param flag The flag to write.
///
MSCCLPP_DEVICE_INLINE void putLL16Packets(void* targetPtr, uint64_t targetOffset, const void* originPtr,
                                          uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                          uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  const uint32_t* originBase = (const uint32_t*)((const char*)originPtr + originOffset);
  LL16Packet* targetBase = (LL16Packet*)((char*)targetPtr + targetOffset);
  size_t nElem = originBytes / sizeof(uint64_t);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    LL16Packet* pkt = &targetBase[i];
    pkt->write(originBase[2 * i], originBase[2 * i + 1], flag);
  }
}

/// Read LL16Packets from the target buffer and write retrieved data to the origin.
///
/// @param targetPtr The target buffer.
/// @param targetOffset The offset in the target buffer.
/// @param originPtr The origin buffer.
/// @param originOffset The offset in the origin buffer.
/// @param originBytes The number of bytes to write to the target buffer.
/// @param threadId The thread ID. The thread ID should be less than @p numThreads.
/// @param numThreads The number of threads that call this function.
/// @param flag The flag to write.
///
MSCCLPP_DEVICE_INLINE void getLL16Packets(const void* targetPtr, uint64_t targetOffset, void* originPtr,
                                          uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                          uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  const LL16Packet* targetBase = (const LL16Packet*)((const char*)targetPtr + targetOffset);
  uint2* originBase = (uint2*)((char*)originPtr + originOffset);
  size_t nElem = originBytes / sizeof(uint2);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    const LL16Packet* pkt = &targetBase[i];
    originBase[i] = pkt->read(flag);
  }
}

/// Read data from the origin and write LL8Packets to the target buffer.
///
/// @param targetPtr The target buffer.
/// @param targetOffset The offset in the target buffer.
/// @param originPtr The origin buffer.
/// @param originOffset The offset in the origin buffer.
/// @param originBytes The number of bytes to write to the target buffer.
/// @param threadId The thread ID. The thread ID should be less than @p numThreads.
/// @param numThreads The number of threads that call this function.
/// @param flag The flag to write.
///
MSCCLPP_DEVICE_INLINE void putLL8Packets(void* targetPtr, uint64_t targetOffset, const void* originPtr,
                                         uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                         uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 4 bytes & size should be a multiple of 4 bytes
  const uint32_t* originBase = (const uint32_t*)((const char*)originPtr + originOffset);
  LL8Packet* targetBase = (LL8Packet*)((char*)targetPtr + targetOffset);
  size_t nElem = originBytes / sizeof(uint32_t);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    LL8Packet* pkt = &targetBase[i];
    pkt->write(originBase[i], flag);
  }
}

/// Read LL8Packets from the target buffer and write retrieved data to the origin.
///
/// @param targetPtr The target buffer.
/// @param targetOffset The offset in the target buffer.
/// @param originPtr The origin buffer.
/// @param originOffset The offset in the origin buffer.
/// @param originBytes The number of bytes to write to the target buffer.
/// @param threadId The thread ID. The thread ID should be less than @p numThreads.
/// @param numThreads The number of threads that call this function.
/// @param flag The flag to write.
///
MSCCLPP_DEVICE_INLINE void getLL8Packets(const void* targetPtr, uint64_t targetOffset, void* originPtr,
                                         uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                         uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 4 bytes & size should be a multiple of 4 bytes
  const LL8Packet* targetBase = (const LL8Packet*)((const char*)targetPtr + targetOffset);
  uint32_t* originBase = (uint32_t*)((char*)originPtr + originOffset);
  size_t nElem = originBytes / sizeof(uint32_t);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    const LL8Packet* pkt = &targetBase[i];
    originBase[i] = pkt->read(flag);
  }
}

/// Read data from the origin and write packets to the target buffer.
///
/// @param targetPtr The target buffer.
/// @param targetOffset The offset in the target buffer.
/// @param originPtr The origin buffer.
/// @param originOffset The offset in the origin buffer.
/// @param originBytes The number of bytes to write to the target buffer.
/// @param threadId The thread ID. The thread ID should be less than @p numThreads.
/// @param numThreads The number of threads that call this function.
/// @param flag The flag to write.
/// @tparam PacketType The packet type. It should be either @ref LL16Packet or @ref LL8Packet.
///
template <typename PacketType = LL16Packet>
MSCCLPP_DEVICE_INLINE void putPackets(void* targetPtr, uint64_t targetOffset, const void* originPtr,
                                      uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                      uint32_t numThreads, uint32_t flag) {
  if constexpr (std::is_same<PacketType, LL16Packet>::value) {
    putLL16Packets(targetPtr, targetOffset, originPtr, originOffset, originBytes, threadId, numThreads, flag);
  } else if constexpr (std::is_same<PacketType, LL8Packet>::value) {
    putLL8Packets(targetPtr, targetOffset, originPtr, originOffset, originBytes, threadId, numThreads, flag);
  } else {
    static_assert(std::is_same<PacketType, LL16Packet>::value || std::is_same<PacketType, LL8Packet>::value,
                  "Unsupported packet type");
  }
}

/// Read packets from the target buffer and write retrieved data to the origin.
///
/// @param targetPtr The target buffer.
/// @param targetOffset The offset in the target buffer.
/// @param originPtr The origin buffer.
/// @param originOffset The offset in the origin buffer.
/// @param originBytes The number of bytes to read from the origin buffer.
/// @param threadId The thread ID. The thread ID should be less than @p numThreads.
/// @param numThreads The number of threads that call this function.
/// @param flag The flag to read.
/// @tparam PacketType The packet type. It should be either @ref LL16Packet or @ref LL8Packet.
///
template <typename PacketType = LL16Packet>
MSCCLPP_DEVICE_INLINE void getPackets(const void* targetPtr, uint64_t targetOffset, void* originPtr,
                                      uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                      uint32_t numThreads, uint32_t flag) {
  if constexpr (std::is_same<PacketType, LL16Packet>::value) {
    getLL16Packets(targetPtr, targetOffset, originPtr, originOffset, originBytes, threadId, numThreads, flag);
  } else if constexpr (std::is_same<PacketType, LL8Packet>::value) {
    getLL8Packets(targetPtr, targetOffset, originPtr, originOffset, originBytes, threadId, numThreads, flag);
  } else {
    static_assert(std::is_same<PacketType, LL16Packet>::value || std::is_same<PacketType, LL8Packet>::value,
                  "Unsupported packet type");
  }
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

};  // namespace mscclpp

#endif  // MSCCLPP_PACKET_DEVICE_HPP_
