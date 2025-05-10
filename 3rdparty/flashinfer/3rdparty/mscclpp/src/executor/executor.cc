// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/executor.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>
#include <set>

#include "execution_kernel.hpp"
#include "execution_plan.hpp"

namespace mscclpp {
struct ExecutionContextKey {
  void* sendBuff;
  void* recvBuff;
  size_t sendBuffSize;
  size_t recvBuffSize;
  std::string plan;

  bool operator==(const ExecutionContextKey& other) const {
    return sendBuff == other.sendBuff && recvBuff == other.recvBuff && sendBuffSize == other.sendBuffSize &&
           recvBuffSize == other.recvBuffSize && plan == other.plan;
  }
};
}  // namespace mscclpp

namespace std {
template <>
struct hash<std::pair<mscclpp::BufferType, int>> {
  std::size_t operator()(const std::pair<mscclpp::BufferType, int>& key) const {
    return std::hash<int>()(key.second) ^ std::hash<int>()(static_cast<int>(key.first));
  }
};

template <>
struct hash<mscclpp::ExecutionContextKey> {
  std::size_t operator()(const mscclpp::ExecutionContextKey& key) const {
    return std::hash<void*>()(key.sendBuff) ^ std::hash<void*>()(key.recvBuff) ^ std::hash<size_t>()(key.sendBuffSize) ^
           std::hash<size_t>()(key.recvBuffSize) ^ std::hash<std::string>()(key.plan);
  }
};
}  // namespace std

namespace {
auto inSameNode = [](int rank1, int rank2, int nranksPerNode) {
  return rank1 / nranksPerNode == rank2 / nranksPerNode;
};

static const mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                                         mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                                         mscclpp::Transport::IB6, mscclpp::Transport::IB7};
}  // namespace

namespace mscclpp {

struct ExecutionContext {
  std::unordered_map<int, std::shared_ptr<Connection>> connections;
  std::unordered_map<std::pair<BufferType, int>, mscclpp::RegisteredMemory> registeredMemories;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
  std::vector<mscclpp::SemaphoreId> proxySemaphores;
  std::vector<mscclpp::SmChannel> smChannels;
  std::vector<mscclpp::SimpleProxyChannel> proxyChannels;
  std::vector<DeviceExecutionPlan> deviceExecutionPlans;
  std::shared_ptr<char> scratchBuffer;
  size_t scratchBufferSize;
  std::shared_ptr<char> deviceExecutionPlansBuffer;
};

struct Executor::Impl {
  int nranksPerNode;
  std::shared_ptr<Communicator> comm;
  std::shared_ptr<ProxyService> proxyService;
  std::unordered_map<ExecutionContextKey, ExecutionContext> contexts;

  Impl(std::shared_ptr<Communicator> comm) : comm(comm) {
    this->nranksPerNode = comm->bootstrap()->getNranksPerNode();
    this->proxyService = std::make_shared<ProxyService>();
  }
  ~Impl() = default;

  ExecutionContext setupExecutionContext(int rank, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                                         size_t recvBufferSize, const ExecutionPlan& plan) {
    ExecutionContextKey key = {sendbuff, recvbuff, sendBufferSize, recvBufferSize, plan.impl_->name};
    if (this->contexts.find(key) != this->contexts.end()) {
      return this->contexts[key];
    }
    plan.impl_->loadExecutionPlan(sendBufferSize);

    ExecutionContext context;
    size_t scratchBufferSize = plan.impl_->getScratchBufferSize(rank, sendBufferSize);
    std::shared_ptr<char> scratchBuffer = allocExtSharedCuda<char>(scratchBufferSize);
    context.scratchBuffer = scratchBuffer;
    context.scratchBufferSize = scratchBufferSize;
    this->setupConnections(context, rank, plan);
    this->setupRegisteredMemories(context, sendbuff, recvbuff, sendBufferSize, recvBufferSize, rank, plan);
    this->setupChannels(context, sendbuff, recvbuff, sendBufferSize, rank, plan);
    this->setupDeviceExecutionPlan(context, rank, plan);
    context.deviceExecutionPlansBuffer =
        allocExtSharedCuda<char>(context.deviceExecutionPlans.size() * sizeof(DeviceExecutionPlan));
    memcpyCuda(context.deviceExecutionPlansBuffer.get(), (char*)context.deviceExecutionPlans.data(),
               context.deviceExecutionPlans.size() * sizeof(DeviceExecutionPlan), cudaMemcpyHostToDevice);
    this->contexts.insert({key, context});
    return context;
  }

  void setupConnections(ExecutionContext& context, int rank, const ExecutionPlan& plan) {
    std::vector<int> connectedPeers = plan.impl_->getConnectedPeers(rank);
    std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
    for (int peer : connectedPeers) {
      Transport transport =
          inSameNode(rank, peer, this->nranksPerNode) ? Transport::CudaIpc : IBs[rank % this->nranksPerNode];
      connectionFutures.push_back(this->comm->connectOnSetup(peer, 0, transport));
    }
    this->comm->setup();
    for (size_t i = 0; i < connectionFutures.size(); i++) {
      context.connections[connectedPeers[i]] = connectionFutures[i].get();
    }
  }

  void setupRegisteredMemories(ExecutionContext& context, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                               size_t recvBufferSize, int rank, const ExecutionPlan& plan) {
    auto getTransportFlags = [&](std::vector<ChannelInfo>& infos, int rank) {
      TransportFlags flags;
      for (ChannelInfo& info : infos) {
        if (info.channelType == ChannelType::SM) {
          flags |= Transport::CudaIpc;
        } else if (info.channelType == ChannelType::PROXY) {
          for (int peer : info.connectedPeers) {
            if (!inSameNode(rank, peer, this->nranksPerNode)) {
              flags |= IBs[rank % this->nranksPerNode];
            }
          }
        }
      }
      return flags;
    };
    auto getBufferInfo = [&](BufferType type) {
      switch (type) {
        case BufferType::INPUT:
          return std::make_pair(sendbuff, sendBufferSize);
        case BufferType::OUTPUT:
          return std::make_pair(recvbuff, recvBufferSize);
        case BufferType::SCRATCH:
          return std::make_pair((void*)context.scratchBuffer.get(), context.scratchBufferSize);
        default:
          throw Error("Invalid buffer type", ErrorCode::ExecutorError);
      }
    };
    auto getConnectedPeers = [&](std::vector<ChannelInfo>& infos) {
      std::set<int> peers;
      for (ChannelInfo& info : infos) {
        for (int peer : info.connectedPeers) {
          peers.insert(peer);
        }
      }
      return std::vector<int>(peers.begin(), peers.end());
    };

    std::vector<BufferType> bufferTypes = plan.impl_->getConnectedBufferTypes(rank);
    for (BufferType bufferType : bufferTypes) {
      std::vector<ChannelInfo> channelInfos = plan.impl_->getChannelInfos(rank, bufferType);
      TransportFlags transportFlags = getTransportFlags(channelInfos, rank);
      RegisteredMemory memory =
          this->comm->registerMemory(getBufferInfo(bufferType).first, getBufferInfo(bufferType).second, transportFlags);
      std::vector<int> connectedPeers = getConnectedPeers(channelInfos);
      std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemoryFutures;
      for (int peer : connectedPeers) {
        comm->sendMemoryOnSetup(memory, peer, 0);
        remoteRegMemoryFutures.push_back(comm->recvMemoryOnSetup(peer, 0));
      }
      comm->setup();
      for (size_t i = 0; i < remoteRegMemoryFutures.size(); i++) {
        context.registeredMemories[{bufferType, connectedPeers[i]}] = std::move(remoteRegMemoryFutures[i].get());
      }
    }
  }

  void setupChannels(ExecutionContext& context, void* sendbuff, void* recvbuff, size_t sendBufferSize, int rank,
                     const ExecutionPlan& plan) {
    const auto channelTypes = {ChannelType::SM, ChannelType::PROXY};
    std::vector<std::shared_ptr<SmDevice2DeviceSemaphore>> smSemaphores;
    std::vector<mscclpp::SemaphoreId> proxySemaphores;
    for (ChannelType channelType : channelTypes) {
      std::vector<ChannelInfo> channelInfos = plan.impl_->getChannelInfos(rank, channelType);
      for (ChannelInfo& info : channelInfos) {
        for (int peer : info.connectedPeers) {
          if (channelType == ChannelType::SM) {
            smSemaphores.push_back(
                std::make_shared<SmDevice2DeviceSemaphore>(*this->comm, context.connections.at(peer)));
          } else if (channelType == ChannelType::PROXY) {
            proxySemaphores.push_back(
                this->proxyService->buildAndAddSemaphore(*this->comm, context.connections.at(peer)));
          }
        }
      }
    }
    this->comm->setup();
    context.smSemaphores = std::move(smSemaphores);
    context.proxySemaphores = std::move(proxySemaphores);

    auto getBuffer = [&](BufferType type) {
      switch (type) {
        case BufferType::INPUT:
          return sendbuff;
        case BufferType::OUTPUT:
          return recvbuff;
        case BufferType::SCRATCH:
          return (void*)context.scratchBuffer.get();
        default:
          throw Error("Invalid buffer type", ErrorCode::ExecutorError);
      }
    };
    for (ChannelType channelType : channelTypes) {
      std::vector<ChannelInfo> channelInfos = plan.impl_->getChannelInfos(rank, channelType);
      int index = 0;
      for (ChannelInfo& info : channelInfos) {
        void* src = getBuffer(info.srcBufferType);
        TransportFlags transport = context.registeredMemories.begin()->second.transports();
        RegisteredMemory localMemory = this->comm->registerMemory(src, sendBufferSize, transport);
        for (int peer : info.connectedPeers) {
          if (channelType == ChannelType::SM) {
            context.smChannels.emplace_back(context.smSemaphores[index++],
                                            context.registeredMemories[{info.dstBufferType, peer}], src, nullptr);
          } else if (channelType == ChannelType::PROXY) {
            context.proxyChannels.emplace_back(
                this->proxyService->proxyChannel(context.proxySemaphores[index++]),
                this->proxyService->addMemory(context.registeredMemories[{info.dstBufferType, peer}]),
                this->proxyService->addMemory(localMemory));
          }
        }
      }
    }
  }

  void setupDeviceExecutionPlan(ExecutionContext& context, int rank, const ExecutionPlan& plan) {
    std::vector<DeviceExecutionPlan> deviceExecutionPlans;
    for (int threadblock = 0; threadblock < plan.impl_->getThreadblockCount(rank); threadblock++) {
      DeviceExecutionPlan deviceExecutionPlan = {};
      std::vector<Operation> ops = plan.impl_->getOperations(rank, threadblock);
      deviceExecutionPlan.nOperations = ops.size();
      deviceExecutionPlan.nSmChannels = plan.impl_->threadblockSMChannelMap.at(rank).at(threadblock).size();
      deviceExecutionPlan.nProxyChannels = plan.impl_->threadblockProxyChannelMap.at(rank).at(threadblock).size();
      int chanIndex = 0;
      for (const auto& [index, _] : plan.impl_->threadblockSMChannelMap.at(rank).at(threadblock)) {
        deviceExecutionPlan.channels.smChannels[chanIndex++] = mscclpp::deviceHandle(context.smChannels[index]);
      }
      chanIndex = 0;
      for (const auto& [index, _] : plan.impl_->threadblockProxyChannelMap.at(rank).at(threadblock)) {
        deviceExecutionPlan.channels.proxyChannels[chanIndex++] = mscclpp::deviceHandle(context.proxyChannels[index]);
      }
      for (size_t i = 0; i < ops.size(); i++) {
        deviceExecutionPlan.operations[i] = ops[i];
      }
      deviceExecutionPlans.push_back(deviceExecutionPlan);
    }
    context.deviceExecutionPlans = std::move(deviceExecutionPlans);
  }

  void launchKernel(ExecutionContext& context, int rank, int nthreadsPerBlock, void* sendbuff, void* recvbuff,
                    DataType dataType, cudaStream_t stream, PacketType packetType) {
    static uint32_t flag = 0;
    int nthreadblocks = context.deviceExecutionPlans.size();
    size_t sharedMemSize = sizeof(DeviceExecutionPlan);
    switch (packetType) {
      case PacketType::LL16:
        ExecutionKernel::launchKernel<LL16Packet>(
            rank, nthreadblocks, nthreadsPerBlock, sendbuff, recvbuff, (void*)context.scratchBuffer.get(),
            context.scratchBufferSize, dataType, (DeviceExecutionPlan*)context.deviceExecutionPlansBuffer.get(),
            sharedMemSize, stream, ++flag);
        break;
      case PacketType::LL8:
        ExecutionKernel::launchKernel<LL8Packet>(
            rank, nthreadblocks, nthreadsPerBlock, sendbuff, recvbuff, (void*)context.scratchBuffer.get(),
            context.scratchBufferSize, dataType, (DeviceExecutionPlan*)context.deviceExecutionPlansBuffer.get(),
            sharedMemSize, stream, ++flag);
        break;
      default:
        throw Error("Invalid packet type", ErrorCode::ExecutorError);
    }
  }
};

Executor::Executor(std::shared_ptr<Communicator> comm) : impl_(std::make_unique<Impl>(comm)) {}

void Executor::execute(int rank, void* sendbuff, void* recvBuff, size_t sendBuffSize, size_t recvBuffSize,
                       DataType dataType, int nthreads, const ExecutionPlan& plan, cudaStream_t stream,
                       PacketType packetType) {
  ExecutionContext context =
      this->impl_->setupExecutionContext(rank, sendbuff, recvBuff, sendBuffSize, recvBuffSize, plan);
  // TODO(binyli): need to flush proxy channel here this->impl_->proxyService->startProxy();
  this->impl_->launchKernel(context, rank, nthreads, sendbuff, recvBuff, dataType, stream, packetType);
}

Executor::~Executor() = default;

}  // namespace mscclpp
