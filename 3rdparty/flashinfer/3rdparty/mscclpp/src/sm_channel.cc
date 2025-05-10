// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/sm_channel.hpp>

#include "api.h"
#include "debug.h"

namespace mscclpp {

MSCCLPP_API_CPP SmChannel::SmChannel(std::shared_ptr<SmDevice2DeviceSemaphore> semaphore, RegisteredMemory dst,
                                     void* src, void* getPacketBuffer)
    : semaphore_(semaphore), dst_(dst), src_(src), getPacketBuffer_(getPacketBuffer) {
  if (!dst.transports().has(Transport::CudaIpc)) {
    throw Error("SmChannel: dst must be registered with CudaIpc", ErrorCode::InvalidUsage);
  }
}

MSCCLPP_API_CPP SmChannel::DeviceHandle SmChannel::deviceHandle() const {
  return DeviceHandle{.semaphore_ = semaphore_->deviceHandle(),
                      .src_ = src_,
                      .dst_ = dst_.data(),
                      .getPacketBuffer_ = getPacketBuffer_};
}

}  // namespace mscclpp
