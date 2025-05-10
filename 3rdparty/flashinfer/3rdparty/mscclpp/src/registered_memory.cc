// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "registered_memory.hpp"

#include <algorithm>
#include <mscclpp/gpu_utils.hpp>

#include "api.h"
#include "context.hpp"
#include "debug.h"
#include "utils_internal.hpp"

namespace mscclpp {

RegisteredMemory::Impl::Impl(void* data, size_t size, TransportFlags transports, Context::Impl& contextImpl)
    : data(data),
      originalDataPtr(data),
      size(size),
      hostHash(getHostHash()),
      pidHash(getPidHash()),
      transports(transports) {
  if (transports.has(Transport::CudaIpc)) {
    TransportInfo transportInfo;
    transportInfo.transport = Transport::CudaIpc;
    cudaIpcMemHandle_t handle;

    void* baseDataPtr;
    size_t baseDataSize;  // dummy
    MSCCLPP_CUTHROW(cuMemGetAddressRange((CUdeviceptr*)&baseDataPtr, &baseDataSize, (CUdeviceptr)data));
    MSCCLPP_CUDATHROW(cudaIpcGetMemHandle(&handle, baseDataPtr));
    // TODO: bug with offset of base?
    transportInfo.cudaIpcBaseHandle = handle;
    transportInfo.cudaIpcOffsetFromBase = (char*)data - (char*)baseDataPtr;
    this->transportInfos.push_back(transportInfo);
  }
  if ((transports & AllIBTransports).any()) {
    auto addIb = [&](Transport ibTransport) {
      TransportInfo transportInfo;
      transportInfo.transport = ibTransport;
      const IbMr* mr = contextImpl.getIbContext(ibTransport)->registerMr(data, size);
      transportInfo.ibMr = mr;
      transportInfo.ibLocal = true;
      transportInfo.ibMrInfo = mr->getInfo();
      this->transportInfos.push_back(transportInfo);
      INFO(MSCCLPP_NET, "IB mr for address %p with size %ld is registered", data, size);
    };
    if (transports.has(Transport::IB0)) addIb(Transport::IB0);
    if (transports.has(Transport::IB1)) addIb(Transport::IB1);
    if (transports.has(Transport::IB2)) addIb(Transport::IB2);
    if (transports.has(Transport::IB3)) addIb(Transport::IB3);
    if (transports.has(Transport::IB4)) addIb(Transport::IB4);
    if (transports.has(Transport::IB5)) addIb(Transport::IB5);
    if (transports.has(Transport::IB6)) addIb(Transport::IB6);
    if (transports.has(Transport::IB7)) addIb(Transport::IB7);
  }
}

MSCCLPP_API_CPP RegisteredMemory::RegisteredMemory(std::shared_ptr<Impl> pimpl) : pimpl_(pimpl) {}

MSCCLPP_API_CPP RegisteredMemory::~RegisteredMemory() = default;

MSCCLPP_API_CPP void* RegisteredMemory::data() const { return pimpl_->data; }

MSCCLPP_API_CPP void* RegisteredMemory::originalDataPtr() const { return pimpl_->originalDataPtr; }

MSCCLPP_API_CPP size_t RegisteredMemory::size() { return pimpl_->size; }

MSCCLPP_API_CPP TransportFlags RegisteredMemory::transports() { return pimpl_->transports; }

MSCCLPP_API_CPP std::vector<char> RegisteredMemory::serialize() {
  std::vector<char> result;
  std::copy_n(reinterpret_cast<char*>(&pimpl_->originalDataPtr), sizeof(pimpl_->originalDataPtr),
              std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&pimpl_->size), sizeof(pimpl_->size), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&pimpl_->hostHash), sizeof(pimpl_->hostHash), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&pimpl_->pidHash), sizeof(pimpl_->pidHash), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&pimpl_->transports), sizeof(pimpl_->transports), std::back_inserter(result));
  if (pimpl_->transportInfos.size() > static_cast<size_t>(std::numeric_limits<int8_t>::max())) {
    throw mscclpp::Error("Too many transport info entries", ErrorCode::InternalError);
  }
  int8_t transportCount = pimpl_->transportInfos.size();
  std::copy_n(reinterpret_cast<char*>(&transportCount), sizeof(transportCount), std::back_inserter(result));
  for (auto& entry : pimpl_->transportInfos) {
    std::copy_n(reinterpret_cast<char*>(&entry.transport), sizeof(entry.transport), std::back_inserter(result));
    if (entry.transport == Transport::CudaIpc) {
      std::copy_n(reinterpret_cast<char*>(&entry.cudaIpcBaseHandle), sizeof(entry.cudaIpcBaseHandle),
                  std::back_inserter(result));
      std::copy_n(reinterpret_cast<char*>(&entry.cudaIpcOffsetFromBase), sizeof(entry.cudaIpcOffsetFromBase),
                  std::back_inserter(result));
    } else if (AllIBTransports.has(entry.transport)) {
      std::copy_n(reinterpret_cast<char*>(&entry.ibMrInfo), sizeof(entry.ibMrInfo), std::back_inserter(result));
    } else {
      throw mscclpp::Error("Unknown transport", ErrorCode::InternalError);
    }
  }
  return result;
}

MSCCLPP_API_CPP RegisteredMemory RegisteredMemory::deserialize(const std::vector<char>& data) {
  return RegisteredMemory(std::make_shared<Impl>(data));
}

RegisteredMemory::Impl::Impl(const std::vector<char>& serialization) {
  auto it = serialization.begin();
  std::copy_n(it, sizeof(this->originalDataPtr), reinterpret_cast<char*>(&this->originalDataPtr));
  it += sizeof(this->originalDataPtr);
  std::copy_n(it, sizeof(this->size), reinterpret_cast<char*>(&this->size));
  it += sizeof(this->size);
  std::copy_n(it, sizeof(this->hostHash), reinterpret_cast<char*>(&this->hostHash));
  it += sizeof(this->hostHash);
  std::copy_n(it, sizeof(this->pidHash), reinterpret_cast<char*>(&this->pidHash));
  it += sizeof(this->pidHash);
  std::copy_n(it, sizeof(this->transports), reinterpret_cast<char*>(&this->transports));
  it += sizeof(this->transports);
  int8_t transportCount;
  std::copy_n(it, sizeof(transportCount), reinterpret_cast<char*>(&transportCount));
  it += sizeof(transportCount);
  for (int i = 0; i < transportCount; ++i) {
    TransportInfo transportInfo;
    std::copy_n(it, sizeof(transportInfo.transport), reinterpret_cast<char*>(&transportInfo.transport));
    it += sizeof(transportInfo.transport);
    if (transportInfo.transport == Transport::CudaIpc) {
      std::copy_n(it, sizeof(transportInfo.cudaIpcBaseHandle),
                  reinterpret_cast<char*>(&transportInfo.cudaIpcBaseHandle));
      it += sizeof(transportInfo.cudaIpcBaseHandle);
      std::copy_n(it, sizeof(transportInfo.cudaIpcOffsetFromBase),
                  reinterpret_cast<char*>(&transportInfo.cudaIpcOffsetFromBase));
      it += sizeof(transportInfo.cudaIpcOffsetFromBase);
    } else if (AllIBTransports.has(transportInfo.transport)) {
      std::copy_n(it, sizeof(transportInfo.ibMrInfo), reinterpret_cast<char*>(&transportInfo.ibMrInfo));
      it += sizeof(transportInfo.ibMrInfo);
      transportInfo.ibLocal = false;
    } else {
      throw mscclpp::Error("Unknown transport", ErrorCode::InternalError);
    }
    this->transportInfos.push_back(transportInfo);
  }
  if (it != serialization.end()) {
    throw mscclpp::Error("Serialization failed", ErrorCode::InternalError);
  }

  // Next decide how to set this->data
  if (getHostHash() == this->hostHash && getPidHash() == this->pidHash) {
    // The memory is local to the process, so originalDataPtr is valid as is
    this->data = this->originalDataPtr;
  } else if (transports.has(Transport::CudaIpc) && getHostHash() == this->hostHash) {
    // The memory is local to the machine but not to the process, so we need to open the CUDA IPC handle
    auto entry = getTransportInfo(Transport::CudaIpc);
    void* base;
    MSCCLPP_CUDATHROW(cudaIpcOpenMemHandle(&base, entry.cudaIpcBaseHandle, cudaIpcMemLazyEnablePeerAccess));
    this->data = static_cast<char*>(base) + entry.cudaIpcOffsetFromBase;
    INFO(MSCCLPP_P2P, "Opened CUDA IPC handle at pointer %p", this->data);
  } else {
    // No valid data pointer can be set
    this->data = nullptr;
  }
}

RegisteredMemory::Impl::~Impl() {
  // Close the CUDA IPC handle if it was opened during deserialization
  if (data && transports.has(Transport::CudaIpc) && getHostHash() == this->hostHash && getPidHash() != this->pidHash) {
    void* base = static_cast<char*>(data) - getTransportInfo(Transport::CudaIpc).cudaIpcOffsetFromBase;
    cudaError_t err = cudaIpcCloseMemHandle(base);
    if (err != cudaSuccess) {
      WARN("Failed to close CUDA IPC handle at pointer %p: %s", base, cudaGetErrorString(err));
    } else {
      INFO(MSCCLPP_P2P, "Closed CUDA IPC handle at pointer %p", base);
    }
    data = nullptr;
  }
}

const TransportInfo& RegisteredMemory::Impl::getTransportInfo(Transport transport) const {
  for (auto& entry : transportInfos) {
    if (entry.transport == transport) {
      return entry;
    }
  }
  throw Error("Transport data not found", ErrorCode::InternalError);
}

}  // namespace mscclpp
