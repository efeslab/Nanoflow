// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "connection.hpp"

#include <mscclpp/utils.hpp>
#include <sstream>
#include <thread>

#include "debug.h"
#include "endpoint.hpp"
#include "infiniband/verbs.h"
#include "npkit/npkit.h"

namespace mscclpp {

void validateTransport(RegisteredMemory mem, Transport transport) {
  if (!mem.transports().has(transport)) {
    throw Error("RegisteredMemory does not support this transport", ErrorCode::InvalidUsage);
  }
}

// Connection

std::shared_ptr<RegisteredMemory::Impl> Connection::getImpl(RegisteredMemory& memory) { return memory.pimpl_; }

std::shared_ptr<Endpoint::Impl> Connection::getImpl(Endpoint& memory) { return memory.pimpl_; }

std::string Connection::getTransportName() {
  return TransportNames[static_cast<int>(this->transport())] + " -> " +
         TransportNames[static_cast<int>(this->remoteTransport())];
}

// CudaIpcConnection

CudaIpcConnection::CudaIpcConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, cudaStream_t stream)
    : stream_(stream) {
  if (localEndpoint.transport() != Transport::CudaIpc) {
    throw mscclpp::Error("Cuda IPC connection can only be made from a Cuda IPC endpoint", ErrorCode::InvalidUsage);
  }
  if (remoteEndpoint.transport() != Transport::CudaIpc) {
    throw mscclpp::Error("Cuda IPC connection can only be made to a Cuda IPC endpoint", ErrorCode::InvalidUsage);
  }
  // sanity check: make sure the IPC connection is being made within a node
  if (getImpl(remoteEndpoint)->hostHash_ != getImpl(localEndpoint)->hostHash_) {
    std::stringstream ss;
    ss << "Cuda IPC connection can only be made within a node: " << std::hex << getImpl(remoteEndpoint)->hostHash_
       << " != " << std::hex << getImpl(localEndpoint)->hostHash_;
    throw mscclpp::Error(ss.str(), ErrorCode::InvalidUsage);
  }
  INFO(MSCCLPP_P2P, "Cuda IPC connection created");
}

Transport CudaIpcConnection::transport() { return Transport::CudaIpc; }

Transport CudaIpcConnection::remoteTransport() { return Transport::CudaIpc; }

void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                              uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  char* dstPtr = (char*)dst.data();
  char* srcPtr = (char*)src.data();

  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dstPtr + dstOffset, srcPtr + srcOffset, size, cudaMemcpyDeviceToDevice, stream_));
  INFO(MSCCLPP_P2P, "CudaIpcConnection write: from %p to %p, size %lu", srcPtr + srcOffset, dstPtr + dstOffset, size);

  // npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)size);
}

void CudaIpcConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
  validateTransport(dst, remoteTransport());
  uint64_t oldValue = *src;
  *src = newValue;
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(dst.data()) + dstOffset);

  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dstPtr, src, sizeof(uint64_t), cudaMemcpyHostToDevice, stream_));
  INFO(MSCCLPP_P2P, "CudaIpcConnection atomic write: from %p to %p, %lu -> %lu", src, dstPtr + dstOffset, oldValue,
       newValue);

  // npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)size);
}

void CudaIpcConnection::flush(int64_t timeoutUsec) {
  if (timeoutUsec >= 0) {
    INFO(MSCCLPP_P2P, "CudaIpcConnection flush: timeout is not supported, ignored");
  }
  AvoidCudaGraphCaptureGuard guard;
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream_));
  // npkitCollectExitEvents(conn, NPKIT_EVENT_DMA_SEND_EXIT);
  INFO(MSCCLPP_P2P, "CudaIpcConnection flushing connection");
}

// IBConnection

IBConnection::IBConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, Context& context)
    : transport_(localEndpoint.transport()),
      remoteTransport_(remoteEndpoint.transport()),
      dummyAtomicSource_(std::make_unique<uint64_t>(0)) {
  qp = getImpl(localEndpoint)->ibQp_;
  qp->rtr(getImpl(remoteEndpoint)->ibQpInfo_);
  qp->rts();
  dummyAtomicSourceMem_ = context.registerMemory(dummyAtomicSource_.get(), sizeof(uint64_t), transport_);
  validateTransport(dummyAtomicSourceMem_, transport_);
  dstTransportInfo_ = getImpl(dummyAtomicSourceMem_)->getTransportInfo(transport_);
  INFO(MSCCLPP_NET, "IB connection via %s created", getIBDeviceName(transport_).c_str());
}

Transport IBConnection::transport() { return transport_; }

Transport IBConnection::remoteTransport() { return remoteTransport_; }

void IBConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                         uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  auto dstTransportInfo = getImpl(dst)->getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    throw Error("dst is local, which is not supported", ErrorCode::InvalidUsage);
  }
  auto srcTransportInfo = getImpl(src)->getTransportInfo(transport());
  if (!srcTransportInfo.ibLocal) {
    throw Error("src is remote, which is not supported", ErrorCode::InvalidUsage);
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  auto srcMr = srcTransportInfo.ibMr;

  qp->stageSend(srcMr, dstMrInfo, (uint32_t)size, /*wrId=*/0, /*srcOffset=*/srcOffset, /*dstOffset=*/dstOffset,
                /*signaled=*/true);

  qp->postSend();
  INFO(MSCCLPP_NET, "IBConnection write: from %p to %p, size %lu", (uint8_t*)srcMr->getBuff() + srcOffset,
       (uint8_t*)dstMrInfo.addr + dstOffset, size);
  // npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_DATA_ENTRY, (uint32_t)size);
}

void IBConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
  validateTransport(dst, remoteTransport());
  auto dstTransportInfo = getImpl(dst)->getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    throw Error("dst is local, which is not supported", ErrorCode::InvalidUsage);
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  // assert that src is on host
  uint64_t oldValue = *src;
  *src = newValue;

  qp->stageAtomicAdd(dstTransportInfo_.ibMr, dstMrInfo, /*wrId=*/0, dstOffset, newValue - oldValue, /*signaled=*/true);

  qp->postSend();
  INFO(MSCCLPP_NET, "IBConnection atomic Write: from %p to %p, %lu -> %lu", src, (uint8_t*)dstMrInfo.addr + dstOffset,
       oldValue, newValue);
}

void IBConnection::flush(int64_t timeoutUsec) {
  Timer timer;
  while (qp->getNumCqItems()) {
    int wcNum = qp->pollCq();
    if (wcNum < 0) {
      throw mscclpp::IbError("pollCq failed: error no " + std::to_string(errno), errno);
    } else if (timeoutUsec >= 0) {
      auto elapsed = timer.elapsed();
      if (elapsed > timeoutUsec) {
        throw Error("pollCq timed out: waited for " + std::to_string(elapsed / 1e6) + " seconds. Expected " +
                        std::to_string(qp->getNumCqItems()) + " signals",
                    ErrorCode::Timeout);
      }
    }
    for (int i = 0; i < wcNum; ++i) {
      const ibv_wc* wc = qp->getWc(i);
      if (wc->status != IBV_WC_SUCCESS) {
        throw mscclpp::IbError("a work item failed: status " + std::to_string(wc->status), wc->status);
      }
    }
  }
  INFO(MSCCLPP_NET, "IBConnection flushing connection");
  // npkitCollectExitEvents(conn, NPKIT_EVENT_IB_SEND_EXIT);
}

// EthernetConnection

EthernetConnection::EthernetConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, uint64_t sendBufferSize,
                                       uint64_t recvBufferSize)
    : abortFlag_(0), sendBufferSize_(sendBufferSize), recvBufferSize_(recvBufferSize) {
  // Validating Transport Protocol
  if (localEndpoint.transport() != Transport::Ethernet || remoteEndpoint.transport() != Transport::Ethernet) {
    throw mscclpp::Error("Ethernet connection can only be made from Ethernet endpoints", ErrorCode::InvalidUsage);
  }

  // Instanciating Buffers
  sendBuffer_.resize(sendBufferSize_);
  recvBuffer_.resize(recvBufferSize_);

  // Creating Thread to Accept the Connection
  auto parameter = (getImpl(localEndpoint)->socket_).get();
  std::thread t([this, parameter]() {
    recvSocket_ = std::make_unique<Socket>(nullptr, MSCCLPP_SOCKET_MAGIC, SocketTypeUnknown, abortFlag_);
    recvSocket_->accept(parameter);
  });

  // Starting Connection
  sendSocket_ = std::make_unique<Socket>(&(getImpl(remoteEndpoint)->socketAddress_), MSCCLPP_SOCKET_MAGIC,
                                         SocketTypeBootstrap, abortFlag_);
  sendSocket_->connect();

  // Ensure the Connection was Established
  t.join();

  // Starting Thread to Receive Messages
  threadRecvMessages_ = std::thread(&EthernetConnection::recvMessages, this);

  INFO(MSCCLPP_NET, "Ethernet connection created");
}

EthernetConnection::~EthernetConnection() {
  sendSocket_->close();
  recvSocket_->close();
  threadRecvMessages_.join();
}

Transport EthernetConnection::transport() { return Transport::Ethernet; }

Transport EthernetConnection::remoteTransport() { return Transport::Ethernet; }

void EthernetConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                               uint64_t size) {
  // Validating Transport Protocol
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  // Initializing Variables
  char* srcPtr = reinterpret_cast<char*>(src.data()) + srcOffset / sizeof(char);
  char* dstPtr = reinterpret_cast<char*>(dst.originalDataPtr()) + dstOffset / sizeof(char);
  uint64_t sentDataSize = 0;
  uint64_t headerSize = 0;

  // Copying Meta Data to Send Buffer
  char* dstPtrBytes = reinterpret_cast<char*>(&dstPtr);
  std::copy(dstPtrBytes, dstPtrBytes + sizeof(dstPtr), sendBuffer_.data() + headerSize / sizeof(char));
  headerSize += sizeof(dstPtr);
  char* sizeBytes = reinterpret_cast<char*>(&size);
  std::copy(sizeBytes, sizeBytes + sizeof(size), sendBuffer_.data() + headerSize / sizeof(char));
  headerSize += sizeof(size);

  // Getting Data From GPU and Sending Message
  while (sentDataSize < size) {
    uint64_t dataSize =
        std::min(sendBufferSize_ - headerSize / sizeof(char), (size - sentDataSize) / sizeof(char)) * sizeof(char);
    uint64_t messageSize = dataSize + headerSize;
    mscclpp::memcpyCuda<char>(sendBuffer_.data() + headerSize / sizeof(char),
                              (char*)srcPtr + (sentDataSize / sizeof(char)), dataSize, cudaMemcpyDeviceToHost);
    sendSocket_->send(sendBuffer_.data(), messageSize);
    sentDataSize += messageSize;
    headerSize = 0;
  }

  INFO(MSCCLPP_NET, "EthernetConnection write: from %p to %p, size %lu", srcPtr, dstPtr, size);
}

void EthernetConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
  // Validating Transport Protocol
  validateTransport(dst, remoteTransport());

  // Initializing Variables
  uint64_t oldValue = *src;
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(dst.originalDataPtr()) + dstOffset);
  uint64_t dataSize = sizeof(uint64_t);
  uint64_t messageSize = 0;
  *src = newValue;

  // Copying Data to Send Buffer
  char* dstPtrBytes = reinterpret_cast<char*>(&dstPtr);
  std::copy(dstPtrBytes, dstPtrBytes + sizeof(dstPtr), sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += sizeof(dstPtr);
  char* sizeBytes = reinterpret_cast<char*>(&dataSize);
  std::copy(sizeBytes, sizeBytes + sizeof(dataSize), sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += sizeof(dataSize);
  char* dataBytes = reinterpret_cast<char*>(src);
  std::copy(dataBytes, dataBytes + dataSize, sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += dataSize;

  // Sending Message
  sendSocket_->send(sendBuffer_.data(), messageSize);

  INFO(MSCCLPP_NET, "EthernetConnection atomic write: from %p to %p, %lu -> %lu", src, dstPtr + dstOffset, oldValue,
       newValue);
}

void EthernetConnection::flush(int64_t) { INFO(MSCCLPP_NET, "EthernetConnection flushing connection"); }

void EthernetConnection::recvMessages() {
  // Declarating Variables
  char* ptr;
  uint64_t size;
  uint64_t recvSize;
  int closed = 0;
  bool received = true;

  // Receiving Messages Until Connection is Closed
  while (recvSocket_->getState() != SocketStateClosed) {
    // Receiving Data Address
    if (closed == 0) recvSocket_->recvUntilEnd(&ptr, sizeof(char*), &closed);
    received &= !closed;

    // Receiving data size
    if (closed == 0) recvSocket_->recvUntilEnd(&size, sizeof(uint64_t), &closed);
    received &= !closed;

    // Receiving Data and Copying Data yo GPU
    recvSize = 0;
    while (recvSize < size && closed == 0) {
      uint64_t messageSize = std::min(recvBufferSize_, (size - recvSize) / sizeof(char)) * sizeof(char);
      recvSocket_->recvUntilEnd(recvBuffer_.data(), messageSize, &closed);
      received &= !closed;

      if (received)
        mscclpp::memcpyCuda<char>((char*)ptr + (recvSize / sizeof(char)), recvBuffer_.data(), messageSize,
                                  cudaMemcpyHostToDevice);
      recvSize += messageSize;
    }
  }
}

}  // namespace mscclpp
