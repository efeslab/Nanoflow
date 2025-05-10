// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "context.hpp"

#include "api.h"
#include "connection.hpp"
#include "debug.h"
#include "endpoint.hpp"
#include "registered_memory.hpp"

namespace mscclpp {

Context::Impl::Impl() : ipcStream_(cudaStreamNonBlocking) {}

IbCtx* Context::Impl::getIbContext(Transport ibTransport) {
  // Find IB context or create it
  auto it = ibContexts_.find(ibTransport);
  if (it == ibContexts_.end()) {
    auto ibDev = getIBDeviceName(ibTransport);
    ibContexts_[ibTransport] = std::make_unique<IbCtx>(ibDev);
    return ibContexts_[ibTransport].get();
  } else {
    return it->second.get();
  }
}

MSCCLPP_API_CPP Context::Context() : pimpl_(std::make_unique<Impl>()) {}

MSCCLPP_API_CPP Context::~Context() = default;

MSCCLPP_API_CPP RegisteredMemory Context::registerMemory(void* ptr, size_t size, TransportFlags transports) {
  return RegisteredMemory(std::make_shared<RegisteredMemory::Impl>(ptr, size, transports, *pimpl_));
}

MSCCLPP_API_CPP Endpoint Context::createEndpoint(EndpointConfig config) {
  return Endpoint(std::make_shared<Endpoint::Impl>(config, *pimpl_));
}

MSCCLPP_API_CPP std::shared_ptr<Connection> Context::connect(Endpoint localEndpoint, Endpoint remoteEndpoint) {
  std::shared_ptr<Connection> conn;
  if (localEndpoint.transport() == Transport::CudaIpc) {
    if (remoteEndpoint.transport() != Transport::CudaIpc) {
      throw mscclpp::Error("Local transport is CudaIpc but remote is not", ErrorCode::InvalidUsage);
    }
    conn = std::make_shared<CudaIpcConnection>(localEndpoint, remoteEndpoint, pimpl_->ipcStream_);
  } else if (AllIBTransports.has(localEndpoint.transport())) {
    if (!AllIBTransports.has(remoteEndpoint.transport())) {
      throw mscclpp::Error("Local transport is IB but remote is not", ErrorCode::InvalidUsage);
    }
    conn = std::make_shared<IBConnection>(localEndpoint, remoteEndpoint, *this);
  } else if (localEndpoint.transport() == Transport::Ethernet) {
    if (remoteEndpoint.transport() != Transport::Ethernet) {
      throw mscclpp::Error("Local transport is Ethernet but remote is not", ErrorCode::InvalidUsage);
    }
    conn = std::make_shared<EthernetConnection>(localEndpoint, remoteEndpoint);
  } else {
    throw mscclpp::Error("Unsupported transport", ErrorCode::InternalError);
  }

  pimpl_->connections_.push_back(conn);
  return conn;
}

}  // namespace mscclpp
