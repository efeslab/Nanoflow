// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "api.h"
#include "atomic.hpp"

namespace mscclpp {

struct Fifo::Impl {
  UniqueCudaHostPtr<ProxyTrigger[]> triggers;
  UniqueCudaPtr<uint64_t> head;
  UniqueCudaPtr<uint64_t> tailReplica;
  const int size;

  // allocated on the host. Only accessed by the host. This is a copy of the
  // value pointed to by fifoTailDev and the invariant is that
  // *fifoTailDev <= hostTail. Meaning that host's copy of tail is
  // always ahead of the device's copy and host updates the device's copy
  // only when it is needed. Therefore, hostTail is the "true" tail
  // and fifoTailDev is a "stale" tail. See proxy.cc to undertand how
  // these updates are pushed to the device.
  uint64_t hostTail;

  // for transferring fifo tail
  CudaStreamWithFlags stream;

  Impl(int size)
      : triggers(makeUniqueCudaHost<ProxyTrigger[]>(size)),
        head(allocUniqueCuda<uint64_t>()),
        tailReplica(allocUniqueCuda<uint64_t>()),
        size(size),
        hostTail(0),
        stream(cudaStreamNonBlocking) {}
};

MSCCLPP_API_CPP Fifo::Fifo(int size) : pimpl(std::make_unique<Impl>(size)) {}
MSCCLPP_API_CPP Fifo::~Fifo() = default;

MSCCLPP_API_CPP ProxyTrigger Fifo::poll() {
  ProxyTrigger trigger;
  ProxyTrigger* ptr = &pimpl->triggers.get()[pimpl->hostTail % pimpl->size];
  // we are loading fst first. if fst is non-zero then snd is also valid
  trigger.fst = atomicLoad(&(ptr->fst), memoryOrderAcquire);
  trigger.snd = ptr->snd;
  return trigger;
}

MSCCLPP_API_CPP void Fifo::pop() {
  atomicStore(&(pimpl->triggers.get()[pimpl->hostTail % pimpl->size].fst), uint64_t{0}, memoryOrderRelease);
  (pimpl->hostTail)++;
}

MSCCLPP_API_CPP void Fifo::flushTail(bool sync) {
  // Flush the tail to device memory. This is either triggered every ProxyFlushPeriod to make sure that the fifo can
  // make progress even if there is no request mscclppSync. However, mscclppSync type is for flush request.
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(pimpl->tailReplica.get(), &pimpl->hostTail, sizeof(uint64_t),
                                    cudaMemcpyHostToDevice, pimpl->stream));
  if (sync) {
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(pimpl->stream));
  }
}

MSCCLPP_API_CPP int Fifo::size() const { return pimpl->size; }

MSCCLPP_API_CPP FifoDeviceHandle Fifo::deviceHandle() {
  FifoDeviceHandle deviceHandle;
  deviceHandle.triggers = pimpl->triggers.get();
  deviceHandle.head = pimpl->head.get();
  deviceHandle.tailReplica = pimpl->tailReplica.get();
  deviceHandle.size = pimpl->size;
  return deviceHandle;
}

}  // namespace mscclpp
