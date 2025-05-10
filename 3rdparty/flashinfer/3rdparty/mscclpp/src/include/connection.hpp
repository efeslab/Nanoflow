// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>

#include "communicator.hpp"
#include "context.hpp"
#include "ib.hpp"
#include "registered_memory.hpp"
#include "socket.h"

namespace mscclpp {

class CudaIpcConnection : public Connection {
  cudaStream_t stream_;

 public:
  CudaIpcConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, cudaStream_t stream);

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;
};

class IBConnection : public Connection {
  Transport transport_;
  Transport remoteTransport_;
  IbQp* qp;
  std::unique_ptr<uint64_t> dummyAtomicSource_;  // not used anywhere but IB needs a source
  RegisteredMemory dummyAtomicSourceMem_;
  mscclpp::TransportInfo dstTransportInfo_;

 public:
  IBConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, Context& context);

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;
};

class EthernetConnection : public Connection {
  std::unique_ptr<Socket> sendSocket_;
  std::unique_ptr<Socket> recvSocket_;
  std::thread threadRecvMessages_;
  volatile uint32_t* abortFlag_;
  const uint64_t sendBufferSize_;
  const uint64_t recvBufferSize_;
  std::vector<char> sendBuffer_;
  std::vector<char> recvBuffer_;

 public:
  EthernetConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, uint64_t sendBufferSize = 256 * 1024 * 1024,
                     uint64_t recvBufferSize = 256 * 1024 * 1024);

  ~EthernetConnection();

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;

 private:
  void recvMessages();

  void sendMessage();
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONNECTION_HPP_
