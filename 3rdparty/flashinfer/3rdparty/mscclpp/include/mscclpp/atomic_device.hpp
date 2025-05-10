// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ATOMIC_DEVICE_HPP_
#define MSCCLPP_ATOMIC_DEVICE_HPP_

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_CUDA)
#include <cuda/atomic>
#endif  // defined(MSCCLPP_DEVICE_CUDA)

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_CUDA)

constexpr cuda::memory_order memoryOrderRelaxed = cuda::memory_order_relaxed;
constexpr cuda::memory_order memoryOrderAcquire = cuda::memory_order_acquire;
constexpr cuda::memory_order memoryOrderRelease = cuda::memory_order_release;
constexpr cuda::memory_order memoryOrderAcqRel = cuda::memory_order_acq_rel;
constexpr cuda::memory_order memoryOrderSeqCst = cuda::memory_order_seq_cst;

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(T* ptr, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.load(memoryOrder);
}

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, const T& val, cuda::memory_order memoryOrder) {
  cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.store(val, memoryOrder);
}

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.fetch_add(val, memoryOrder);
}

#elif defined(MSCCLPP_DEVICE_HIP)

constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(const T* ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, const T& val, int memoryOrder) {
  __atomic_store_n(ptr, val, memoryOrder);
}

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

#endif  // defined(MSCCLPP_DEVICE_HIP)

}  // namespace mscclpp

#endif  // MSCCLPP_ATOMIC_DEVICE_HPP_
