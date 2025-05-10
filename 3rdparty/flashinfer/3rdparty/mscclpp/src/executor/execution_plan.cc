// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_plan.hpp"

#include <cassert>
#include <fstream>
#include <set>

namespace {
template <typename T, typename Predicate>
std::vector<T> filter(const std::vector<T>& vec, Predicate pred) {
  std::vector<T> filtered;
  std::copy_if(vec.begin(), vec.end(), std::back_inserter(filtered), pred);
  return filtered;
}

auto getOpType = [](const std::string& str) {
  if (str == "nop") {
    return mscclpp::OperationType::BARRIER;
  } else if (str == "put") {
    return mscclpp::OperationType::PUT;
  } else if (str == "get") {
    return mscclpp::OperationType::GET;
  } else if (str == "copy") {
    return mscclpp::OperationType::COPY;
  } else if (str == "signal") {
    return mscclpp::OperationType::SIGNAL;
  } else if (str == "wait") {
    return mscclpp::OperationType::WAIT;
  } else if (str == "flush") {
    return mscclpp::OperationType::FLUSH;
  } else if (str == "re") {
    return mscclpp::OperationType::REDUCE;
  } else if (str == "rs") {
    return mscclpp::OperationType::REDUCE_SEND;
  } else if (str == "rrc") {
    return mscclpp::OperationType::READ_REDUCE_COPY;
  } else if (str == "rrcs") {
    return mscclpp::OperationType::READ_REDUCE_COPY_SEND;
  } else if (str == "ppkt") {
    return mscclpp::OperationType::PUT_PACKET;
  } else if (str == "rspkt") {
    return mscclpp::OperationType::REDUCE_SEND_PACKET;
  } else if (str == "cpkt") {
    return mscclpp::OperationType::COPY_PACKET;
  } else {
    throw mscclpp::Error("Invalid operation type", mscclpp::ErrorCode::ExecutorError);
  }
};

auto convertToBufferType = [](const std::string& str) {
  if (str == "i") {
    return mscclpp::BufferType::INPUT;
  } else if (str == "o") {
    return mscclpp::BufferType::OUTPUT;
  } else if (str == "s") {
    return mscclpp::BufferType::SCRATCH;
  } else {
    throw mscclpp::Error("Invalid buffer type", mscclpp::ErrorCode::ExecutorError);
  }
};

auto convertToChannelType = [](const std::string& str) {
  if (str == "sm") {
    return mscclpp::ChannelType::SM;
  } else if (str == "proxy") {
    return mscclpp::ChannelType::PROXY;
  } else if (str == "none") {
    return mscclpp::ChannelType::NONE;
  } else {
    throw mscclpp::Error("Invalid channel type", mscclpp::ErrorCode::ExecutorError);
  }
};

}  // namespace

namespace mscclpp {
using json = nlohmann::json;

ExecutionPlan::Impl::Impl(const std::string name, const std::string planPath)
    : name(name), planPath(planPath), isUsingPacket(false) {}

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(int rank, ChannelType channelType) const {
  auto pred = [channelType](const ChannelInfo& info) { return info.channelType == channelType; };
  return filter(this->channelInfos.at(rank), pred);
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(int rank, BufferType dstBufferType) const {
  auto pred = [dstBufferType](const ChannelInfo& info) { return info.dstBufferType == dstBufferType; };
  return filter(this->channelInfos.at(rank), pred);
}

std::vector<int> ExecutionPlan::Impl::getConnectedPeers(int rank) const {
  std::set<int> peers;
  for (const auto& info : this->channelInfos.at(rank)) {
    for (int peer : info.connectedPeers) {
      peers.insert(peer);
    }
  }
  return std::vector<int>(peers.begin(), peers.end());
}

std::vector<BufferType> ExecutionPlan::Impl::getConnectedBufferTypes(int rank) const {
  std::set<BufferType> bufferTypes;
  for (const auto& info : this->channelInfos.at(rank)) {
    bufferTypes.insert(info.dstBufferType);
  }
  return std::vector<BufferType>(bufferTypes.begin(), bufferTypes.end());
}
size_t ExecutionPlan::Impl::getScratchBufferSize(int rank, size_t inputSize) const {
  if (this->isUsingPacket) {
    return inputSize / this->inputChunks.at(rank) * this->scratchChunks.at(rank) * 2 /* data + flag*/ *
           2 /*double buffer*/;
  }
  return inputSize / this->inputChunks.at(rank) * this->scratchChunks.at(rank);
}
std::vector<Operation> ExecutionPlan::Impl::getOperations(int rank, int threadblock) const {
  return this->operations.at(rank)[threadblock];
}

int ExecutionPlan::Impl::getThreadblockCount(int rank) const { return this->operations.at(rank).size(); }

void ExecutionPlan::Impl::loadExecutionPlan(size_t inputSize) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  if (this->name != obj["name"]) {
    throw Error("Plan name does not match", ErrorCode::ExecutorError);
  }
  std::string protocol = obj["protocol"];
  if (protocol == "LL") {
    this->isUsingPacket = true;
  }
  const auto& gpus = obj["gpus"];

  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    this->inputChunks[rank] = gpu["inputChunks"];
    this->outputChunks[rank] = gpu["outputChunks"];
    this->scratchChunks[rank] = gpu["scratchChunks"];
    this->chunkGroups[rank] = gpu["chunkGroups"];
  }
  this->setupChannels(gpus);

  this->inputSize = inputSize;
  this->setupOperations(gpus);
}

// Construct the channel info. Step 1. Flatten SM and PROXY channels into separate vectors.
// Step 2. For each threadblock, construct a vector of channel indexes and keys.
void ExecutionPlan::Impl::setupChannels(const json& gpus) {
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    std::vector<ChannelInfo> channelInfos;
    for (const auto& channel : gpu["channels"]) {
      ChannelInfo info;
      info.srcBufferType = convertToBufferType(channel["srcbuff"]);
      info.dstBufferType = convertToBufferType(channel["dstbuff"]);
      info.channelType = convertToChannelType(channel["type"]);
      for (const auto& peer : channel["connectedTo"]) {
        info.connectedPeers.push_back(peer);
      }
      channelInfos.push_back(info);
    }
    this->channelInfos[rank] = channelInfos;
  }

  // setup threadblockChannelMap
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    auto channelTypes = {ChannelType::SM, ChannelType::PROXY};
    std::unordered_map<ChannelKey, std::vector<int>> channelMap;
    for (auto channelType : channelTypes) {
      const std::vector<ChannelInfo> channelInfos = this->getChannelInfos(rank, channelType);
      int index = 0;
      for (const auto& info : channelInfos) {
        ChannelKey key = {info.srcBufferType, info.dstBufferType, info.channelType};
        for (size_t i = 0; i < info.connectedPeers.size(); i++) {
          channelMap[key].push_back(index++);
        }
      }
    }
    int nthreadblocks = gpu["threadblocks"].size();
    this->threadblockSMChannelMap[rank].resize(nthreadblocks);
    this->threadblockProxyChannelMap[rank].resize(nthreadblocks);
    for (const auto& threadblock : gpu["threadblocks"]) {
      for (const auto& channel : threadblock["channels"]) {
        ChannelType channelType = convertToChannelType(channel["ctype"]);
        ChannelKey key = {convertToBufferType(channel["src"]), convertToBufferType(channel["dst"]), channelType};
        for (int id : channel["cids"]) {
          if (channelType == ChannelType::SM) {
            this->threadblockSMChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id], key);
          } else if (channelType == ChannelType::PROXY) {
            this->threadblockProxyChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id], key);
          }
        }
      }
    }
  }
}

void ExecutionPlan::Impl::setupOperations(const json& gpus) {
  // setup threadblocks and operations
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    for (const auto& threadblock : gpu["threadblocks"]) {
      std::unordered_map<ChannelKey, std::vector<int>> channelIndexes;
      std::vector<Operation> ops;
      int threadblockId = threadblock["id"];
      const auto& smChannels = this->threadblockSMChannelMap[rank][threadblockId];
      const auto& proxyChannels = this->threadblockProxyChannelMap[rank][threadblockId];
      for (size_t i = 0; i < smChannels.size(); i++) {
        const auto& [_, key] = smChannels[i];
        channelIndexes[key].push_back(i);
      }
      for (size_t i = 0; i < proxyChannels.size(); i++) {
        const auto& [_, key] = proxyChannels[i];
        channelIndexes[key].push_back(i);
      }
      for (const auto& op : threadblock["ops"]) {
        Operation operation = {};
        std::vector<uint32_t> chunkIndexes;
        operation.type = static_cast<mscclpp::OperationType>(getOpType(op["name"]));
        if (op.contains("ctype")) {
          operation.channelType = convertToChannelType(op["ctype"]);
        }
        if (op.contains("i_cids")) {
          operation.nInputs = op["i_cids"].size();
          for (int i = 0; i < operation.nInputs; i++) {
            BufferType srcBufferType = convertToBufferType(op["i_buff"]["src"]);
            BufferType dstBufferType = convertToBufferType(op["i_buff"]["dst"]);
            // Get the relevant channel index in rank channelInfos
            operation.inputChannelIndexes[i] =
                channelIndexes[{srcBufferType, dstBufferType, operation.channelType}][op["i_cids"][i]["id"]];
            operation.inputOffsets[i] = this->getOffset(rank, this->inputSize, (uint32_t)op["i_cids"][i]["off"]);
            chunkIndexes.push_back((uint32_t)op["i_cids"][i]["off"]);
          }
        }
        // will have either srcs or i_cids
        if (op.contains("srcs")) {
          operation.nInputs = op["srcs"].size();
          operation.inputBufferType = convertToBufferType(op["srcs"][0]["buff"]);
          for (int i = 0; i < operation.nInputs; i++) {
            operation.inputOffsets[i] = this->getOffset(rank, this->inputSize, (uint32_t)op["srcs"][i]["off"]);
            chunkIndexes.push_back((uint32_t)op["srcs"][i]["off"]);
          }
        }
        if (op.contains("o_cids")) {
          operation.nOutputs = op["o_cids"].size();
          for (int i = 0; i < operation.nOutputs; i++) {
            BufferType srcBufferType = convertToBufferType(op["o_buff"]["src"]);
            BufferType dstBufferType = convertToBufferType(op["o_buff"]["dst"]);
            operation.outputChannelIndexes[i] =
                channelIndexes[{srcBufferType, dstBufferType, operation.channelType}][op["o_cids"][i]["id"]];
            operation.outputOffsets[i] = this->getOffset(rank, this->inputSize, (uint32_t)op["o_cids"][i]["off"]);
            chunkIndexes.push_back((uint32_t)op["o_cids"][i]["off"]);
          }
        }
        // will have either dsts or o_cids
        if (op.contains("dsts")) {
          operation.nOutputs = op["dsts"].size();
          operation.outputBufferType = convertToBufferType(op["dsts"][0]["buff"]);
          for (int i = 0; i < operation.nOutputs; i++) {
            operation.outputOffsets[i] = this->getOffset(rank, this->inputSize, (uint32_t)op["dsts"][i]["off"]);
            chunkIndexes.push_back((uint32_t)op["dsts"][i]["off"]);
          }
        }
        if (op.contains("srcbuff")) {
          operation.srcBufferType = convertToBufferType(op["srcbuff"]);
        }
        if (op.contains("srcoff")) {
          operation.srcOffset = this->getOffset(rank, this->inputSize, (uint32_t)op["srcoff"]);
          chunkIndexes.push_back((uint32_t)op["srcoff"]);
        }
        if (op.contains("dstbuff")) {
          operation.dstBufferType = convertToBufferType(op["dstbuff"]);
        }
        if (op.contains("dstoff")) {
          operation.dstOffset = this->getOffset(rank, this->inputSize, (uint32_t)op["dstoff"]);
          chunkIndexes.push_back((uint32_t)op["dstoff"]);
        }
        if (op.contains("cnt")) {
          operation.size = this->getNChunkSize(rank, this->inputSize, (uint32_t)op["cnt"], chunkIndexes);
        }
        ops.push_back(operation);
      }
      this->operations[rank].push_back(ops);
    }
  }
}

size_t ExecutionPlan::Impl::getOffset(int rank, size_t inputSize, uint32_t chunkIndex, uint32_t alignment) const {
  assert(inputSize % alignment == 0 && "inputSize must be a multiple of alignment");

  const int nGroups = this->chunkGroups.at(rank);
  uint32_t nInputChunks = this->inputChunks.at(rank);
  uint32_t nelems = inputSize / (alignment * sizeof(uint8_t));
  int nelemsPerGroup = nelems / nGroups;
  int nChunksPerGroup = nInputChunks / nGroups;
  uint32_t minNelems = nelemsPerGroup / nChunksPerGroup;
  uint32_t remainder = nelemsPerGroup % nelemsPerGroup;
  uint32_t groupIdx = chunkIndex / nChunksPerGroup;
  uint32_t chunkIndexInGroup = chunkIndex % nChunksPerGroup;
  uint32_t offset = groupIdx * nelemsPerGroup + chunkIndexInGroup * minNelems +
                    (chunkIndexInGroup % nelemsPerGroup < remainder ? chunkIndexInGroup % nelemsPerGroup : remainder);
  return static_cast<size_t>(offset) * alignment;
}

size_t ExecutionPlan::Impl::getNChunkSize(int rank, size_t inputSize, uint32_t nChunks,
                                          const std::vector<uint32_t> chunkIndexes) const {
  size_t nChunkSize = 0;
  for (uint32_t index : chunkIndexes) {
    uint32_t beginOff = getOffset(rank, inputSize, index);
    uint32_t endOff = getOffset(rank, inputSize, index + nChunks);
    if (nChunkSize == 0) {
      nChunkSize = endOff - beginOff;
    } else if (nChunkSize != endOff - beginOff) {
      throw Error("Inconsistent chunk size", ErrorCode::ExecutorError);
    }
  }
  return nChunkSize;
}

ExecutionPlan::ExecutionPlan(const std::string& name, const std::string& planPath)
    : impl_(std::make_shared<Impl>(name, planPath)) {}

}  // namespace mscclpp
