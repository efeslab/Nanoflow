// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <mscclpp/core.hpp>
#include <mscclpp/nvls.hpp>
#include <mscclpp/utils.hpp>

#include "api.h"
#include "debug.h"
#include "endpoint.hpp"

namespace mscclpp {

#if (USE_NVLS)
class NvlsConnection::Impl : public std::enable_shared_from_this<NvlsConnection::Impl> {
 public:
  // use this only for the root of the NVLS
  Impl(size_t bufferSize, int numDevices);
  Impl(const std::vector<char>& data);
  ~Impl();

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  size_t getMinMcGran() { return minMcGran_; }
  std::vector<char> serialize();
  void addDevice(int cudaDeviceId);
  size_t allocateBuffer(size_t size);
  void freeBuffer(size_t offset, size_t size) noexcept;
  std::shared_ptr<char> bindMemory(CUmemGenericAllocationHandle memHandle, size_t devBuffSize);

 private:
  friend class NvlsConnection;
  CUmemGenericAllocationHandle mcHandle_;
  CUmulticastObjectProp mcProp_;
  size_t bufferSize_;
  size_t minMcGran_;
  size_t mcGran_;
  // These are only defined for multicast (NVLS) capability
  pid_t rootPid_;
  int mcFileDesc_;

  std::list<std::pair<size_t, size_t>> allocatedRanges_;
  std::list<std::pair<size_t, size_t>> freeRanges_;
};

NvlsConnection::Impl::Impl(size_t bufferSize, int numDevices) {
  minMcGran_ = 0;
  mcGran_ = 0;
  mcProp_ = {};
  mcProp_.size = bufferSize;
  mcProp_.numDevices = numDevices;
  mcProp_.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGran_, &mcProp_, CU_MULTICAST_GRANULARITY_MINIMUM));
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&mcGran_, &mcProp_, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  mcProp_.size = ((mcProp_.size + minMcGran_ - 1) / minMcGran_) * minMcGran_;
  bufferSize_ = mcProp_.size;
  MSCCLPP_CUTHROW(cuMulticastCreate(&mcHandle_, &mcProp_));
  mcFileDesc_ = 0;
  MSCCLPP_CUTHROW(
      cuMemExportToShareableHandle(&mcFileDesc_, mcHandle_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
  freeRanges_.emplace_back(0, bufferSize_);

  rootPid_ = getpid();
  if (rootPid_ < 0) {
    throw mscclpp::SysError("getpid() failed", errno);
  }

  INFO(MSCCLPP_COLL, "NVLS handle created on root with size %ld. minGranularity %ld and recommendedGranularity %ld\n",
       mcProp_.size, minMcGran_, mcGran_);
}

NvlsConnection::Impl::Impl(const std::vector<char>& data) {
  auto it = data.begin();
  std::copy_n(it, sizeof(this->mcHandle_), reinterpret_cast<char*>(&this->mcHandle_));
  it += sizeof(this->mcHandle_);
  std::copy_n(it, sizeof(this->bufferSize_), reinterpret_cast<char*>(&this->bufferSize_));
  it += sizeof(this->bufferSize_);
  std::copy_n(it, sizeof(this->minMcGran_), reinterpret_cast<char*>(&this->minMcGran_));
  it += sizeof(this->minMcGran_);
  std::copy_n(it, sizeof(this->mcGran_), reinterpret_cast<char*>(&this->mcGran_));
  it += sizeof(this->mcGran_);
  std::copy_n(it, sizeof(this->rootPid_), reinterpret_cast<char*>(&this->rootPid_));
  it += sizeof(this->rootPid_);
  std::copy_n(it, sizeof(this->mcFileDesc_), reinterpret_cast<char*>(&this->mcFileDesc_));

  freeRanges_.emplace_back(0, bufferSize_);
  int rootPidFd = syscall(SYS_pidfd_open, rootPid_, 0);
  if (rootPidFd < 0) {
    throw mscclpp::SysError("pidfd_open() failed", errno);
  }
  int mcRootFileDescFd = syscall(SYS_pidfd_getfd, rootPidFd, mcFileDesc_, 0);
  if (mcRootFileDescFd < 0) {
    throw mscclpp::SysError("pidfd_getfd() failed", errno);
  }
  MSCCLPP_CUTHROW(cuMemImportFromShareableHandle(&mcHandle_, reinterpret_cast<void*>(mcRootFileDescFd),
                                                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  close(rootPidFd);
  close(mcRootFileDescFd);

  INFO(MSCCLPP_COLL, "NVLS handle was imported from root");
}

NvlsConnection::Impl::~Impl() {
  // we don't need to free multicast handle object according to NCCL.
  if (rootPid_ == getpid()) {
    close(mcFileDesc_);
  }
}

std::vector<char> NvlsConnection::Impl::serialize() {
  std::vector<char> result;
  std::copy_n(reinterpret_cast<char*>(&mcHandle_), sizeof(mcHandle_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&bufferSize_), sizeof(bufferSize_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&minMcGran_), sizeof(minMcGran_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&mcGran_), sizeof(mcGran_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&rootPid_), sizeof(rootPid_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&mcFileDesc_), sizeof(mcFileDesc_), std::back_inserter(result));
  return result;
}

void NvlsConnection::Impl::addDevice(int cudaDeviceId) {
  MSCCLPP_CUTHROW(cuMulticastAddDevice(mcHandle_, cudaDeviceId));
  INFO(MSCCLPP_COLL, "NVLS connection created");
}

size_t NvlsConnection::Impl::allocateBuffer(size_t size) {
  if (freeRanges_.empty()) {
    throw Error("This NVLS connection mapped more than it was supposed to", ErrorCode::InvalidUsage);
  }
  auto it = std::find_if(freeRanges_.begin(), freeRanges_.end(),
                         [size](const std::pair<size_t, size_t>& range) { return range.second >= size; });
  if (it != freeRanges_.end()) {
    size_t offset = it->first;
    size_t rangeSize = it->second;
    if (rangeSize == size) {
      freeRanges_.erase(it);
    } else {
      it->first += size;
      it->second -= size;
    }
    allocatedRanges_.emplace_back(offset, size);
    INFO(MSCCLPP_COLL, "NVLS connection allocated %ld bytes at offset %ld", size, offset);
    return offset;
  }
  throw Error("This NVLS connection cannot map the requested devBuffSize", ErrorCode::InvalidUsage);
}

void NvlsConnection::Impl::freeBuffer(size_t offset, size_t size) noexcept {
  auto it = std::find_if(
      allocatedRanges_.begin(), allocatedRanges_.end(),
      [offset, size](const std::pair<size_t, size_t>& range) { return range.first == offset && range.second == size; });
  if (it == allocatedRanges_.end()) {
    WARN("NVLS connection tried to free a buffer that was not allocated");
    return;
  }
  allocatedRanges_.erase(it);
  it = std::find_if(freeRanges_.begin(), freeRanges_.end(), [offset, size](const std::pair<size_t, size_t>& range) {
    return range.first + range.second >= offset;
  });
  if (it == freeRanges_.end()) {
    freeRanges_.emplace_back(offset, size);
    return;
  }
  if (it->first + it->second == offset) {
    // merge with the previous free range if possible
    it->second += size;
    // merge with the next free range if possible
    auto nextItr = std::next(it);
    if (nextItr != freeRanges_.end() && it->first + it->second == nextItr->first) {
      it->second += nextItr->second;
      freeRanges_.erase(nextItr);
    }
    return;
  } else if (it->first == offset + size) {
    // merge with the next free range if possible
    it->first -= size;
    it->second += size;
    return;
  } else {
    freeRanges_.emplace(it, offset, size);
    return;
  }
}

std::shared_ptr<char> NvlsConnection::Impl::bindMemory(CUmemGenericAllocationHandle memHandle, size_t devBuffSize) {
  size_t offset = allocateBuffer(devBuffSize);
  MSCCLPP_CUTHROW(cuMulticastBindMem(mcHandle_, offset /*mcOffset*/, memHandle, 0 /*memOffset*/, devBuffSize, 0));

  char* mcPtr;

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  int deviceId = -1;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemAddressReserve((CUdeviceptr*)(&mcPtr), devBuffSize, minMcGran_, 0U, 0));
  MSCCLPP_CUTHROW(cuMemMap((CUdeviceptr)(mcPtr), devBuffSize, 0, mcHandle_, 0));
  MSCCLPP_CUTHROW(cuMemSetAccess((CUdeviceptr)(mcPtr), devBuffSize, &accessDesc, 1));

  auto deleter = [=, self = shared_from_this()](char* ptr) {
    CUdevice device;
    MSCCLPP_CUTHROW(cuDeviceGet(&device, deviceId));
    MSCCLPP_CUTHROW(cuMemUnmap((CUdeviceptr)ptr, devBuffSize));
    MSCCLPP_CUTHROW(cuMemAddressFree((CUdeviceptr)ptr, devBuffSize));
    MSCCLPP_CUTHROW(cuMulticastUnbind(mcHandle_, device, offset, devBuffSize));
    self->freeBuffer(offset, devBuffSize);
  };

  return std::shared_ptr<char>(mcPtr, deleter);
}
#else   // !(USE_NVLS)
class NvlsConnection::Impl {
 public:
  // use this only for the root of the NVLS
  Impl(size_t, int) { throw notSupportedError; }
  Impl(const std::vector<char>&) { throw notSupportedError; }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  std::vector<char> serialize() { throw notSupportedError; }
  size_t allocateBuffer(size_t) { throw notSupportedError; }
  void freeBuffer(size_t, size_t) { throw notSupportedError; }
  std::shared_ptr<char> bindMemory(CUmemGenericAllocationHandle, size_t) { throw notSupportedError; }
  void addDevice(int) { throw notSupportedError; }
  size_t getMinMcGran() { throw notSupportedError; }

 private:
  Error notSupportedError = Error("NVLS is not supported on this CUDA version", ErrorCode::InvalidUsage);
};
#endif  // !(USE_NVLS)

NvlsConnection::NvlsConnection(size_t bufferSize, int numDevices)
    : pimpl_(std::make_shared<Impl>(bufferSize, numDevices)) {}

void NvlsConnection::addDevice() {
  int cudaDeviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDeviceId));
  this->addDevice(cudaDeviceId);
}

void NvlsConnection::addDevice(int cudaDeviceId) { pimpl_->addDevice(cudaDeviceId); }

NvlsConnection::NvlsConnection(const std::vector<char>& data) : pimpl_(std::make_shared<Impl>(data)) {}

std::vector<char> NvlsConnection::serialize() { return pimpl_->serialize(); }

std::shared_ptr<NvlsConnection::DeviceMulticastPointer> NvlsConnection::allocateAndBindCuda(size_t size) {
  auto mem = allocSharedPhysicalCuda<char>(size, pimpl_->getMinMcGran());
  auto mcPtr = pimpl_->bindMemory(mem->memHandle_, size);
  return std::make_shared<DeviceMulticastPointer>(mem, mcPtr, size);
}

std::shared_ptr<char> NvlsConnection::bindAllocatedCuda(CUmemGenericAllocationHandle memHandle, size_t size) {
  return pimpl_->bindMemory(memHandle, size);
}

NvlsConnection::DeviceMulticastPointer::DeviceHandle NvlsConnection::DeviceMulticastPointer::deviceHandle() {
  NvlsConnection::DeviceMulticastPointer::DeviceHandle device;
  device.devicePtr = this->deviceMem_->devicePtr_;
  device.mcPtr = this->mcPtr_.get();
  device.bufferSize = this->bufferSize_;
  return device;
};

char* NvlsConnection::DeviceMulticastPointer::getDevicePtr() { return deviceMem_->devicePtr_; };

size_t NvlsConnection::getMultiCastMinGranularity() { return pimpl_->getMinMcGran(); }

MSCCLPP_API_CPP std::shared_ptr<NvlsConnection> connectNvlsCollective(std::shared_ptr<Communicator> comm,
                                                                      std::vector<int> allRanks, size_t bufferSize) {
  auto bootstrap = comm->bootstrap();
  int rank = bootstrap->getRank();
  bool isRoot = false;
  bool amongAllRanks = false;
  int rootRank = allRanks[0];
  for (auto nvlsRank : allRanks) {
    if (nvlsRank == rank) amongAllRanks = true;
    rootRank = std::min(rootRank, nvlsRank);
  }
  if (amongAllRanks == false) {
    throw Error("rank is not among allRanks", ErrorCode::InvalidUsage);
  }
  if (rootRank == rank) isRoot = true;

  std::shared_ptr<NvlsConnection> conn;
  if (isRoot) {
    conn = std::make_shared<NvlsConnection>(bufferSize, allRanks.size());
    auto serialized = conn->serialize();
    for (auto nvlsRank : allRanks) {
      if (nvlsRank != rank) bootstrap->send(serialized, nvlsRank, 0);
    }
  } else {
    std::vector<char> data;
    bootstrap->recv(data, rootRank, 0);
    conn = std::make_shared<NvlsConnection>(data);
  }

  // Now let's synchronize all ranks
  bootstrap->groupBarrier(allRanks);
  // now it is safe to add my device
  conn->addDevice();

  // sync here to make sure all ranks have added their devices
  bootstrap->groupBarrier(allRanks);
  return conn;
}

}  // namespace mscclpp
