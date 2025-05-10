// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/semaphore.hpp>

#include "api.h"
#include "atomic.hpp"
#include "debug.h"

namespace mscclpp {

static NonblockingFuture<RegisteredMemory> setupInboundSemaphoreId(Communicator& communicator, Connection* connection,
                                                                   void* localInboundSemaphoreId) {
  auto localInboundSemaphoreIdsRegMem =
      communicator.registerMemory(localInboundSemaphoreId, sizeof(uint64_t), connection->transport());
  int remoteRank = communicator.remoteRankOf(*connection);
  int tag = communicator.tagOf(*connection);
  communicator.sendMemoryOnSetup(localInboundSemaphoreIdsRegMem, remoteRank, tag);
  return communicator.recvMemoryOnSetup(remoteRank, tag);
}

MSCCLPP_API_CPP Host2DeviceSemaphore::Host2DeviceSemaphore(Communicator& communicator,
                                                           std::shared_ptr<Connection> connection)
    : BaseSemaphore(allocExtUniqueCuda<uint64_t>(), allocExtUniqueCuda<uint64_t>(), std::make_unique<uint64_t>()),
      connection_(connection) {
  INFO(MSCCLPP_INIT, "Creating a Host2Device semaphore for %s transport from %d to %d",
       connection->getTransportName().c_str(), communicator.bootstrap()->getRank(),
       communicator.remoteRankOf(*connection));
  remoteInboundSemaphoreIdsRegMem_ =
      setupInboundSemaphoreId(communicator, connection.get(), localInboundSemaphore_.get());
}

MSCCLPP_API_CPP std::shared_ptr<Connection> Host2DeviceSemaphore::connection() { return connection_; }

MSCCLPP_API_CPP void Host2DeviceSemaphore::signal() {
  connection_->updateAndSync(remoteInboundSemaphoreIdsRegMem_.get(), 0, outboundSemaphore_.get(),
                             *outboundSemaphore_ + 1);
}

MSCCLPP_API_CPP Host2DeviceSemaphore::DeviceHandle Host2DeviceSemaphore::deviceHandle() {
  Host2DeviceSemaphore::DeviceHandle device;
  device.inboundSemaphoreId = localInboundSemaphore_.get();
  device.expectedInboundSemaphoreId = expectedInboundSemaphore_.get();
  return device;
}

MSCCLPP_API_CPP Host2HostSemaphore::Host2HostSemaphore(Communicator& communicator,
                                                       std::shared_ptr<Connection> connection)
    : BaseSemaphore(std::make_unique<uint64_t>(), std::make_unique<uint64_t>(), std::make_unique<uint64_t>()),
      connection_(connection) {
  INFO(MSCCLPP_INIT, "Creating a Host2Host semaphore for %s transport from %d to %d",
       connection->getTransportName().c_str(), communicator.bootstrap()->getRank(),
       communicator.remoteRankOf(*connection));

  if (connection->transport() == Transport::CudaIpc) {
    throw Error("Host2HostSemaphore cannot be used with CudaIpc transport", ErrorCode::InvalidUsage);
  }
  remoteInboundSemaphoreIdsRegMem_ =
      setupInboundSemaphoreId(communicator, connection.get(), localInboundSemaphore_.get());
}

MSCCLPP_API_CPP std::shared_ptr<Connection> Host2HostSemaphore::connection() { return connection_; }

MSCCLPP_API_CPP void Host2HostSemaphore::signal() {
  connection_->updateAndSync(remoteInboundSemaphoreIdsRegMem_.get(), 0, outboundSemaphore_.get(),
                             *outboundSemaphore_ + 1);
}

MSCCLPP_API_CPP bool Host2HostSemaphore::poll() {
  bool signaled =
      (atomicLoad((uint64_t*)localInboundSemaphore_.get(), memoryOrderAcquire) > (*expectedInboundSemaphore_));
  if (signaled) (*expectedInboundSemaphore_) += 1;
  return signaled;
}

MSCCLPP_API_CPP void Host2HostSemaphore::wait(int64_t maxSpinCount) {
  (*expectedInboundSemaphore_) += 1;
  int64_t spinCount = 0;
  while (atomicLoad((uint64_t*)localInboundSemaphore_.get(), memoryOrderAcquire) < (*expectedInboundSemaphore_)) {
    if (maxSpinCount >= 0 && spinCount++ == maxSpinCount) {
      throw Error("Host2HostSemaphore::wait timed out", ErrorCode::Timeout);
    }
  }
}

MSCCLPP_API_CPP SmDevice2DeviceSemaphore::SmDevice2DeviceSemaphore(Communicator& communicator,
                                                                   std::shared_ptr<Connection> connection)
    : BaseSemaphore(allocExtUniqueCuda<uint64_t>(), allocExtUniqueCuda<uint64_t>(), allocExtUniqueCuda<uint64_t>()) {
  INFO(MSCCLPP_INIT, "Creating a Device2Device semaphore for %s transport from %d to %d",
       connection->getTransportName().c_str(), communicator.bootstrap()->getRank(),
       communicator.remoteRankOf(*connection));
  if (connection->transport() == Transport::CudaIpc) {
    remoteInboundSemaphoreIdsRegMem_ =
        setupInboundSemaphoreId(communicator, connection.get(), localInboundSemaphore_.get());
    isRemoteInboundSemaphoreIdSet_ = true;
  } else if (AllIBTransports.has(connection->transport())) {
    // We don't need to really with any of the IB transports, since the values will be local
    isRemoteInboundSemaphoreIdSet_ = false;
  }
}

MSCCLPP_API_CPP SmDevice2DeviceSemaphore::DeviceHandle SmDevice2DeviceSemaphore::deviceHandle() const {
  SmDevice2DeviceSemaphore::DeviceHandle device;
  device.remoteInboundSemaphoreId = isRemoteInboundSemaphoreIdSet_
                                        ? reinterpret_cast<uint64_t*>(remoteInboundSemaphoreIdsRegMem_.get().data())
                                        : nullptr;
  device.inboundSemaphoreId = localInboundSemaphore_.get();
  device.expectedInboundSemaphoreId = expectedInboundSemaphore_.get();
  device.outboundSemaphoreId = outboundSemaphore_.get();
  return device;
};

}  // namespace mscclpp
