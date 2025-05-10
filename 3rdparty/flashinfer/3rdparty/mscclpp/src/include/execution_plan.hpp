// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_PLAN_HPP_
#define MSCCLPP_EXECUTOR_PLAN_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "execution_common.hpp"

namespace mscclpp {

struct ChannelKey {
  BufferType srcBufferType;
  BufferType dstBufferType;
  ChannelType channelType;
  bool operator==(const ChannelKey& other) const {
    return srcBufferType == other.srcBufferType && dstBufferType == other.dstBufferType &&
           channelType == other.channelType;
  }
};
}  // namespace mscclpp

namespace std {
template <>
struct hash<mscclpp::ChannelKey> {
  std::size_t operator()(const mscclpp::ChannelKey& key) const {
    return std::hash<int>()(static_cast<int>(key.srcBufferType)) ^
           std::hash<int>()(static_cast<int>(key.dstBufferType)) ^ std::hash<int>()(static_cast<int>(key.channelType));
  }
};
}  // namespace std

namespace mscclpp {

struct ChannelInfo {
  BufferType srcBufferType;
  BufferType dstBufferType;
  ChannelType channelType;
  std::vector<int> connectedPeers;
};

struct ExecutionPlan::Impl {
 public:
  Impl(const std::string name, const std::string planPath);
  ~Impl() = default;

  std::vector<ChannelInfo> getChannelInfos(int rank, ChannelType channelType) const;
  std::vector<ChannelInfo> getChannelInfos(int rank, BufferType bufferType) const;
  std::vector<int> getConnectedPeers(int rank) const;
  std::vector<BufferType> getConnectedBufferTypes(int rank) const;
  size_t getScratchBufferSize(int rank, size_t inputSize) const;
  std::vector<Operation> getOperations(int rank, int threadblock) const;
  int getThreadblockCount(int rank) const;

  void loadExecutionPlan(size_t inputSize);
  void setupChannels(const nlohmann::json& gpus);
  void setupOperations(const nlohmann::json& gpus);

  const std::string name;
  const std::string planPath;
  bool isUsingPacket;
  // operations for [rank][threadblock] = [operations]
  std::unordered_map<int, std::vector<std::vector<Operation>>> operations;
  std::unordered_map<int, std::vector<ChannelInfo>> channelInfos;
  // threadblockChannelMap[rank][threadblock] = [channelIndex, channelKey]
  std::unordered_map<int, std::vector<std::vector<std::pair<int, ChannelKey>>>> threadblockSMChannelMap;
  std::unordered_map<int, std::vector<std::vector<std::pair<int, ChannelKey>>>> threadblockProxyChannelMap;
  std::unordered_map<int, uint32_t> inputChunks;
  std::unordered_map<int, uint32_t> outputChunks;
  std::unordered_map<int, uint32_t> scratchChunks;
  std::unordered_map<int, uint32_t> chunkGroups;
  size_t inputSize;

 private:
  size_t getOffset(int rank, size_t inputSize, uint32_t chunkIndex, uint32_t alignment = 16) const;
  size_t getNChunkSize(int rank, size_t inputSize, uint32_t nChunks, const std::vector<uint32_t> offsets) const;
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_PLAN_HPP_
