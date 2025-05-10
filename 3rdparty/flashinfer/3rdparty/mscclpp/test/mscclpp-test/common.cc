// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "common.hpp"

#include <getopt.h>
#include <libgen.h>
#include <mpi.h>
#include <numa.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mscclpp/utils.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <type_traits>

int isMainProc = 0;

mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                            mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                            mscclpp::Transport::IB6, mscclpp::Transport::IB7};

#define PRINT(__message)                    \
  do {                                      \
    if (isMainProc) std::cout << __message; \
  } while (0);

#define PRECISION(__val) std::fixed << std::setprecision(2) << __val

namespace {

// Command line parameter defaults
size_t minBytes = 32 * 1024 * 1024;
size_t maxBytes = 32 * 1024 * 1024;
size_t stepBytes = 1 * 1024 * 1024;
size_t stepFactor = 1;
int datacheck = 1;
int warmup_iters = 10;
int iters = 20;
// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
int average = 1;
int kernel_num = 0;
int cudaGraphLaunches = 15;
std::string output_file;

double parseSize(const char* value) {
  std::string valueStr(value);
  std::istringstream iss(valueStr);
  long long int units;
  double size;
  char size_lit = 0;

  if (iss >> size) {
    iss >> std::ws;  // eat whitespace
    iss >> size_lit;
  } else {
    return -1.0;
  }

  if (size_lit != 0 && !std::isspace(size_lit)) {
    switch (size_lit) {
      case 'G':
      case 'g':
        units = 1024 * 1024 * 1024;
        break;
      case 'M':
      case 'm':
        units = 1024 * 1024;
        break;
      case 'K':
      case 'k':
        units = 1024;
        break;
      default:
        return -1.0;
    };
  } else {
    units = 1;
  }
  return size * units;
}

double allreduceTime(int worldSize, double value, int average) {
  double accumulator = value;

  if (average != 0) {
    MPI_Op op;
    if (average == 1) {
      op = MPI_SUM;
    } else if (average == 2) {
      op = MPI_MIN;
    } else if (average == 3) {
      op = MPI_MAX;
    } else if (average == 4) {
      op = MPI_SUM;
    } else {
      throw std::runtime_error("Invalid average type " + std::to_string(average));
    }
    MPI_Allreduce(MPI_IN_PLACE, (void*)&accumulator, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
  }

  if (average == 1) accumulator /= worldSize;
  return accumulator;
}

const std::string getBusId(int cudaDev) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdChar[] = "00000000:00:00.0";
  CUDATHROW(cudaDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), cudaDev));
  // we need the hex in lower case format
  for (size_t i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  return std::string(busIdChar);
}

void validateArgsForDeviceKernel(const std::vector<KernelRestriction>& restrictions, int kernelNum, size_t paramCount,
                                 int worldSize, int nRanksPerNode, int typeSize = 4) {
  auto iter = std::find_if(restrictions.begin(), restrictions.end(), [kernelNum](const KernelRestriction& restriction) {
    return restriction.kernelNum == kernelNum;
  });
  if (iter == restrictions.end()) {
    PRINT("no restriction for kernelNum=" << kernelNum << std::endl);
    return;
  }
  std::stringstream ss;
  bool isOnMultiNodes = worldSize / nRanksPerNode > 1;
  if (isOnMultiNodes && !iter->compatibleWithMultiNodes) {
    ss << "kernel is not compatible with multi nodes, kernelNum=" << kernelNum << ", name=" << iter->kernelName
       << ", worldSize=" << worldSize << ", nRanksPerNode=" << nRanksPerNode;
    throw std::invalid_argument(ss.str());
  }
  if (isOnMultiNodes && paramCount % iter->countDivisorForMultiNodes != 0) {
    ss << "kernel is not compatible with input size, kernelNum=" << kernelNum << ", name=" << iter->kernelName
       << ", paramCount=" << paramCount << ", countDivisorForMultiNodes=" << iter->countDivisorForMultiNodes;
    throw std::invalid_argument(ss.str());
  }
  bool sizeAlignedForMultiNode = (paramCount * typeSize / iter->countDivisorForMultiNodes) % iter->alignedBytes == 0;
  if (((paramCount * typeSize) % iter->alignedBytes != 0) || (isOnMultiNodes && !sizeAlignedForMultiNode)) {
    ss << "kernel is not compatible with alignment restriction, kernelNum=" << kernelNum
       << ", name=" << iter->kernelName << ", paramCount=" << paramCount << ", alignedBytes=" << iter->alignedBytes
       << ", countDivisorForMultiNodes=" << iter->countDivisorForMultiNodes;
    throw std::invalid_argument(ss.str());
  }
}
}  // namespace

int getDeviceNumaNode(int cudaDev) {
  std::string busId = getBusId(cudaDev);
  std::string file_str = "/sys/bus/pci/devices/" + busId + "/numa_node";
  std::ifstream file(file_str);
  int numaNode;
  if (file.is_open()) {
    if (!(file >> numaNode)) {
      throw std::runtime_error("Failed to read NUMA node from file: " + file_str);
    }
  } else {
    throw std::runtime_error("Failed to open file: " + file_str);
  }
  return numaNode;
}

void numaBind(int node) {
  int totalNumNumaNodes = numa_num_configured_nodes();
  if (node < 0 || node >= totalNumNumaNodes) {
    throw std::runtime_error("Invalid NUMA node " + std::to_string(node) + ", must be between 0 and " +
                             std::to_string(totalNumNumaNodes));
  }
  nodemask_t mask;
  nodemask_zero(&mask);
  nodemask_set_compat(&mask, node);
  numa_bind_compat(&mask);
}

BaseTestEngine::BaseTestEngine(const TestArgs& args, const std::string& name)
    : args_(args), name_(name), inPlace_(true), error_(0) {
  this->coll_ = getTestColl();
  CUDATHROW(cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking));
}

BaseTestEngine::~BaseTestEngine() { (void)cudaStreamDestroy(stream_); }

void BaseTestColl::setupCollTest(const TestArgs& args, size_t size) {
  this->worldSize_ = args.totalRanks;
  this->typeSize_ = sizeof(int);
  this->kernelNum_ = args.kernelNum;
  this->setupCollTest(size);
}

double BaseTestEngine::benchTime() {
  // Performance Benchmark
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDATHROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
  mscclpp::Timer timer;
  for (int iter = 0; iter < iters; iter++) {
    coll_->runColl(args_, stream_);
  }
  CUDATHROW(cudaStreamEndCapture(stream_, &graph));
  CUDATHROW(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  this->barrier();
  timer.reset();
  for (int l = 0; l < cudaGraphLaunches; ++l) {
    CUDATHROW(cudaGraphLaunch(graphExec, stream_));
  }
  CUDATHROW(cudaStreamSynchronize(stream_));
  double deltaSec = timer.elapsed() * 1.e-6;
  deltaSec = deltaSec / (iters) / (cudaGraphLaunches);
  // all-reduce to get the average time
  allreduceTime(args_.totalRanks, deltaSec, average);
  CUDATHROW(cudaGraphExecDestroy(graphExec));
  CUDATHROW(cudaGraphDestroy(graph));
  return deltaSec;
}

void BaseTestEngine::barrier() { this->comm_->bootstrap()->barrier(); }

void BaseTestEngine::runTest() {
  // warm-up for large size
  this->coll_->setupCollTest(args_, args_.maxBytes);
  this->barrier();
  validateArgsForDeviceKernel(coll_->getKernelRestrictions(), args_.kernelNum, coll_->getParamBytes() / sizeof(int),
                              args_.totalRanks, args_.nRanksPerNode);
  for (int iter = 0; iter < warmup_iters; iter++) {
    this->coll_->runColl(args_, stream_);
  }
  CUDATHROW(cudaDeviceSynchronize());

  // warm-up for small size
  this->coll_->setupCollTest(args_, args_.minBytes);
  this->barrier();
  validateArgsForDeviceKernel(coll_->getKernelRestrictions(), args_.kernelNum, coll_->getParamBytes() / sizeof(int),
                              args_.totalRanks, args_.nRanksPerNode);
  for (int iter = 0; iter < warmup_iters; iter++) {
    this->coll_->runColl(args_, stream_);
  }
  CUDATHROW(cudaDeviceSynchronize());

  std::stringstream ss;
  ss << "#\n";
  ss << "#                                        in-place                       out-of-place\n";
  ss << "#       size         count     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong\n";
  ss << "#        (B)    (elements)     (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)\n";
  PRINT(ss.str());

  ss.str(std::string());

  // Benchmark
  for (size_t size = args_.minBytes; size <= args_.maxBytes;
       size = ((args_.stepFactor > 1) ? size * args_.stepFactor : size + args_.stepBytes)) {
    coll_->setupCollTest(args_, size);
    this->coll_->initData(this->args_, this->getSendBuff(), this->getExpectedBuff());

    ss << std::setw(12) << std::max(coll_->getSendBytes(), coll_->getExpectedBytes()) << "  " << std::setw(12)
       << coll_->getParamBytes() / sizeof(int);

    validateArgsForDeviceKernel(coll_->getKernelRestrictions(), args_.kernelNum, coll_->getParamBytes() / sizeof(int),
                                args_.totalRanks, args_.nRanksPerNode);
    double deltaSec = benchTime();

    size_t nErrors = 0;
    if (args_.reportErrors) {
      this->coll_->setupCollTest(args_, size);
      this->coll_->initData(this->args_, this->getSendBuff(), this->getExpectedBuff());
      this->barrier();
      this->coll_->runColl(args_, stream_);
      CUDATHROW(cudaDeviceSynchronize());

      nErrors = this->checkData();
      if (nErrors > 0) {
        this->error_++;
      }
      MPI_Allreduce(MPI_IN_PLACE, &nErrors, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    }

    double timeUsec = deltaSec * 1e6;
    char timeStr[100];
    if (timeUsec >= 10000.0) {
      sprintf(timeStr, "%7.0f", timeUsec);
    } else if (timeUsec >= 100.0) {
      sprintf(timeStr, "%7.1f", timeUsec);
    } else {
      sprintf(timeStr, "%7.2f", timeUsec);
    }
    double algBw, busBw;
    this->coll_->getBw(deltaSec, algBw, busBw);
    if (!output_file.empty()) {
      nlohmann::json perfOutput = {{"name", name_},
                                   {"kernel", args_.kernelNum},
                                   {"ranks", args_.totalRanks},
                                   {"ranksPerNode", args_.nRanksPerNode},
                                   {"size", size},
                                   {"time", timeUsec},
                                   {"algBw", algBw},
                                   {"busBw", busBw}};
      std::ofstream out(output_file, std::ios_base::app);
      if (isMainProc) out << perfOutput << std::endl;
    }
    if (!this->inPlace_) {
      ss << "                                 ";
    }
    if (args_.reportErrors) {
      ss << "  " << std::setw(7) << timeStr << "  " << std::setw(6) << PRECISION(algBw) << "  " << std::setw(6)
         << PRECISION(busBw) << "  " << std::setw(5) << nErrors;
    } else {
      ss << "  " << std::setw(7) << timeStr << "  " << std::setw(6) << PRECISION(algBw) << "  " << std::setw(6)
         << PRECISION(busBw);
    }
    ss << "\n";
    PRINT(ss.str());
    ss.str(std::string());
  }
  PRINT("\n");
}

void BaseTestEngine::bootstrap() {
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(args_.rank, args_.totalRanks);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  comm_ = std::make_shared<mscclpp::Communicator>(bootstrap);
}

void BaseTestEngine::setupTest() {
  this->chanService_ = this->createProxyService();
  this->setupConnections();
  this->chanService_->startProxy();
  this->coll_->setChanService(this->chanService_);
}

size_t BaseTestEngine::checkData() {
  size_t nErrors = 0;
  void* recvBuff = this->getRecvBuff();
  void* expectedBuff = this->getExpectedBuff();

  size_t recvBytes = this->coll_->getRecvBytes();
  std::vector<int> recvData(recvBytes / sizeof(int), 0);
  CUDATHROW(cudaMemcpy(recvData.data(), recvBuff, recvBytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < recvData.size(); i++) {
    if (recvData[i] != ((int*)expectedBuff)[i]) {
      nErrors++;
    }
  }
  return nErrors;
}

std::shared_ptr<mscclpp::BaseProxyService> BaseTestEngine::createProxyService() {
  return std::make_shared<mscclpp::ProxyService>();
}

void BaseTestEngine::setupMeshConnectionsInternal(
    std::vector<std::shared_ptr<mscclpp::Connection>>& connections, mscclpp::RegisteredMemory& localRegMemory,
    std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>>& remoteRegMemories, bool addConnections) {
  const int worldSize = args_.totalRanks;
  const int rank = args_.rank;
  const int nRanksPerNode = args_.nRanksPerNode;
  const int thisNode = rank / nRanksPerNode;
  const mscclpp::Transport ibTransport = IBs[args_.gpuNum];
  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;

  auto rankToNode = [&](int rank) { return rank / nRanksPerNode; };
  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    if (addConnections) {
      mscclpp::Transport transport;
      if (rankToNode(r) == thisNode) {
        transport = mscclpp::Transport::CudaIpc;
      } else {
        transport = ibTransport;
      }
      // Connect with all other ranks
      connectionFutures.push_back(comm_->connectOnSetup(r, 0, transport));
    }
    comm_->sendMemoryOnSetup(localRegMemory, r, 0);
    auto remoteMemory = comm_->recvMemoryOnSetup(r, 0);
    remoteRegMemories.push_back(remoteMemory);
  }
  comm_->setup();
  std::transform(
      connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
      [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) { return future.get(); });
}

// Create mesh connections between all ranks. If recvBuff is nullptr, assume in-place.
// TODO(saemal): retrun the actual vector instead of void
void BaseTestEngine::setupMeshConnections(std::vector<DeviceHandle<mscclpp::SimpleProxyChannel>>& proxyChannels,
                                          void* inputBuff, size_t inputBuffBytes, void* outputBuff,
                                          size_t outputBuffBytes, SetupChannelFunc setupChannel) {
  const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc | IBs[args_.gpuNum];
  mscclpp::RegisteredMemory inputBufRegMem = comm_->registerMemory(inputBuff, inputBuffBytes, allTransports);
  mscclpp::RegisteredMemory outputBufRegMem;
  if (outputBuff) {
    outputBufRegMem = comm_->registerMemory(outputBuff, outputBuffBytes, allTransports);
  }

  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
  mscclpp::RegisteredMemory& localRegMemory = (outputBuff) ? outputBufRegMem : inputBufRegMem;

  setupMeshConnectionsInternal(connections, localRegMemory, remoteRegMemories);

  if (setupChannel != nullptr) {
    setupChannel(connections, remoteRegMemories, inputBufRegMem);
  } else {
    auto service = std::dynamic_pointer_cast<mscclpp::ProxyService>(chanService_);
    for (size_t i = 0; i < connections.size(); ++i) {
      proxyChannels.push_back(mscclpp::deviceHandle(mscclpp::SimpleProxyChannel(
          service->proxyChannel(service->buildAndAddSemaphore(*comm_, connections[i])),
          service->addMemory(remoteRegMemories[i].get()), service->addMemory(inputBufRegMem))));
    }
  }

  comm_->setup();
}

void BaseTestEngine::setupMeshConnections(std::vector<mscclpp::SmChannel>& smChannels, void* inputBuff,
                                          size_t inputBuffBytes, void* outputBuff, size_t outputBuffBytes,
                                          ChannelSemantic semantic, size_t nChannelPerConnection) {
  const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc | IBs[args_.gpuNum];
  mscclpp::RegisteredMemory inputBufRegMem = comm_->registerMemory(inputBuff, inputBuffBytes, allTransports);
  mscclpp::RegisteredMemory getPacketBufRegMem;
  mscclpp::RegisteredMemory outputBufRegMem;
  if (outputBuff) {
    outputBufRegMem = comm_->registerMemory(outputBuff, outputBuffBytes, allTransports);
  }

  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
  mscclpp::RegisteredMemory& localRegMemory =
      (outputBuff && semantic == ChannelSemantic::PUT) ? outputBufRegMem : inputBufRegMem;
  setupMeshConnectionsInternal(connections, localRegMemory, remoteRegMemories);

  std::unordered_map<size_t, std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>> smSemaphores;
  for (size_t cid = 0; cid < connections.size(); ++cid) {
    if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
      for (size_t i = 0; i < nChannelPerConnection; ++i) {
        smSemaphores[cid].emplace_back(std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm_, connections[cid]));
      }
    }
  }
  comm_->setup();

  for (size_t i = 0; i < nChannelPerConnection; ++i) {
    for (size_t cid = 0; cid < connections.size(); ++cid) {
      if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        smChannels.emplace_back(smSemaphores[cid][i], remoteRegMemories[cid].get(),
                                (outputBuff && semantic == ChannelSemantic::GET) ? outputBuff : inputBufRegMem.data(),
                                outputBuff);
      }
    }
  }
}

void BaseTestEngine::setupMeshConnections(std::vector<mscclpp::SmChannel>& smChannels,
                                          std::vector<DeviceHandle<mscclpp::SimpleProxyChannel>>& proxyChannels,
                                          void* inputBuff, size_t inputBuffBytes, void* putPacketBuff,
                                          size_t putPacketBuffBytes, void* getPacketBuff, size_t getPacketBuffBytes,
                                          void* outputBuff, size_t outputBuffBytes) {
  const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc | IBs[args_.gpuNum];
  mscclpp::RegisteredMemory inputBufRegMem = comm_->registerMemory(inputBuff, inputBuffBytes, allTransports);
  mscclpp::RegisteredMemory putPacketBufRegMem;
  mscclpp::RegisteredMemory getPacketBufRegMem;
  mscclpp::RegisteredMemory outputBufRegMem;
  if (putPacketBuff) {
    putPacketBufRegMem = comm_->registerMemory(putPacketBuff, putPacketBuffBytes, allTransports);
  }
  if (getPacketBuff) {
    getPacketBufRegMem = comm_->registerMemory(getPacketBuff, getPacketBuffBytes, allTransports);
  }
  if (outputBuff) {
    outputBufRegMem = comm_->registerMemory(outputBuff, outputBuffBytes, allTransports);
  }

  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
  mscclpp::RegisteredMemory& localRegMemory =
      (getPacketBuff) ? getPacketBufRegMem : ((outputBuff) ? outputBufRegMem : inputBufRegMem);

  setupMeshConnectionsInternal(connections, localRegMemory, remoteRegMemories);

  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemoriesOutput;
  if (outputBuff) {
    setupMeshConnectionsInternal(connections, outputBufRegMem, remoteRegMemoriesOutput, false);
  }

  std::unordered_map<size_t, std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
  std::unordered_map<size_t, mscclpp::SemaphoreId> connIdToSemId;
  auto service = std::dynamic_pointer_cast<mscclpp::ProxyService>(chanService_);

  for (size_t cid = 0; cid < connections.size(); ++cid) {
    if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
      smSemaphores.emplace(cid, std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm_, connections[cid]));
    } else {
      connIdToSemId[cid] = service->buildAndAddSemaphore(*comm_, connections[cid]);
    }
  }
  comm_->setup();

  for (size_t cid = 0; cid < connections.size(); ++cid) {
    if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
      smChannels.emplace_back(smSemaphores[cid],
                              (outputBuff) ? remoteRegMemoriesOutput[cid].get() : remoteRegMemories[cid].get(),
                              inputBufRegMem.data(), (outputBuff) ? outputBufRegMem.data() : nullptr);
    } else {
      if (putPacketBuff == nullptr || getPacketBuff == nullptr) {
        throw std::runtime_error("IB transport requires putPacketBuff and getPacketBuff");
      }
      proxyChannels.emplace_back(mscclpp::deviceHandle(mscclpp::SimpleProxyChannel(
          service->proxyChannel(connIdToSemId[cid]), service->addMemory(remoteRegMemories[cid].get()),
          service->addMemory(putPacketBufRegMem))));
    }
  }
}

void run(int argc, char* argv[]);
int main(int argc, char* argv[]) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  // Parse args
  double parsed;
  int longindex;
  static option longopts[] = {{"minbytes", required_argument, 0, 'b'},
                              {"maxbytes", required_argument, 0, 'e'},
                              {"stepbytes", required_argument, 0, 'i'},
                              {"stepfactor", required_argument, 0, 'f'},
                              {"iters", required_argument, 0, 'n'},
                              {"warmup_iters", required_argument, 0, 'w'},
                              {"check", required_argument, 0, 'c'},
                              {"cudagraph", required_argument, 0, 'G'},
                              {"average", required_argument, 0, 'a'},
                              {"kernel_num", required_argument, 0, 'k'},
                              {"output_file", required_argument, 0, 'o'},
                              {"help", no_argument, 0, 'h'},
                              {}};

  while (1) {
    int c;
    c = getopt_long(argc, argv, "b:e:i:f:n:w:c:G:a:k:o:h:", longopts, &longindex);

    if (c == -1) break;

    switch (c) {
      case 'b':
        parsed = parseSize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'minbytes'\n");
          return -1;
        }
        minBytes = (size_t)parsed;
        break;
      case 'e':
        parsed = parseSize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'maxbytes'\n");
          return -1;
        }
        maxBytes = (size_t)parsed;
        break;
      case 'i':
        stepBytes = strtol(optarg, NULL, 0);
        break;
      case 'f':
        stepFactor = strtol(optarg, NULL, 0);
        break;
      case 'n':
        iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'w':
        warmup_iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'c':
        datacheck = (int)strtol(optarg, NULL, 0);
        break;
      case 'G':
        cudaGraphLaunches = strtol(optarg, NULL, 0);
        if (cudaGraphLaunches <= 0) {
          fprintf(stderr, "invalid number for 'cudaGraphLaunches'\n");
          return -1;
        }
        break;
      case 'a':
        average = (int)strtol(optarg, NULL, 0);
        break;
      case 'k':
        kernel_num = (int)strtol(optarg, NULL, 0);
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
        printf(
            "USAGE: %s \n\t"
            "[-b,--minbytes <min size in bytes>] \n\t"
            "[-e,--maxbytes <max size in bytes>] \n\t"
            "[-i,--stepbytes <increment size>] \n\t"
            "[-f,--stepfactor <increment factor>] \n\t"
            "[-n,--iters <iteration count>] \n\t"
            "[-w,--warmup_iters <warmup iteration count>] \n\t"
            "[-c,--check <0/1>] \n\t"
            "[-T,--timeout <time in seconds>] \n\t"
            "[-G,--cudagraph <num graph launches>] \n\t"
            "[-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
            "[-k,--kernel_num <kernel number of commnication primitive>] \n\t"
            "[-o, --output_file <output file name>] \n\t"
            "[-h,--help]\n",
            basename(argv[0]));
        return 0;
    }
  }
  if (minBytes > maxBytes) {
    std::cerr << "invalid sizes for 'minbytes' and 'maxbytes': " << minBytes << " > " << maxBytes << std::endl;
    return -1;
  }
  run(argc, argv);
  return 0;
}

void run(int argc, char* argv[]) {
  int totalRanks = 1, rank = 0;
  int nRanksPerNode = 0, localRank = 0;
  std::string hostname = mscclpp::getHostName(1024, '.');

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_size(shmcomm, &nRanksPerNode);
  MPI_Comm_free(&shmcomm);
  localRank = rank % nRanksPerNode;
  isMainProc = (rank == 0) ? 1 : 0;

  std::stringstream ss;
  ss << "# minBytes " << minBytes << " maxBytes " << maxBytes
     << " step: " << ((stepFactor > 1) ? stepFactor : stepBytes) << "(" << ((stepFactor > 1) ? "factor" : "bytes")
     << ") warmup iters: " << warmup_iters << " iters: " << iters << " validation: " << datacheck
     << " graph: " << cudaGraphLaunches << " kernel num: " << kernel_num << "\n";
  ss << "#\n# Using devices\n";
  PRINT(ss.str());
  ss.str(std::string());

  constexpr int MAX_LINE = 2048;
  char line[MAX_LINE];
  int len = 0;
  size_t maxMem = ~0;

  int cudaDev = localRank;
  cudaDeviceProp prop;
  char busIdChar[] = "00000000:00:00.0";
  CUDATHROW(cudaGetDeviceProperties(&prop, cudaDev));
  CUDATHROW(cudaDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), cudaDev));
  len += snprintf(line + len, MAX_LINE - len, "#  Rank %2d Pid %6d on %10s device %2d [%s] %s\n", rank, getpid(),
                  hostname.c_str(), cudaDev, busIdChar, prop.name);
  maxMem = std::min(maxMem, prop.totalGlobalMem);

  std::shared_ptr<char[]> lines(new char[totalRanks * MAX_LINE]);
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines.get(), MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int r = 0; r < totalRanks; r++) {
      ss << &lines[MAX_LINE * r];
    }
    PRINT(ss.str());
    ss.str(std::string());
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);

  // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 1G for the rest.
  size_t memMaxBytes = (maxMem - (1 << 30)) / (datacheck ? 3 : 2);
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    ss << "#\n# Reducing maxBytes to " << maxBytes << " due to memory limitation\n";
    PRINT(ss.str());
    ss.str(std::string());
  }

  CUDATHROW(cudaSetDevice(cudaDev));
  TestArgs args = {minBytes, maxBytes,  stepBytes,     stepFactor, totalRanks, rank,
                   cudaDev,  localRank, nRanksPerNode, kernel_num, datacheck};

  PRINT("#\n# Initializing MSCCL++\n");

  auto testEngine = getTestEngine(args);
  testEngine->bootstrap();
  testEngine->allocateBuffer();
  PRINT("# Setting up the connection in MSCCL++\n");
  testEngine->setupTest();
  testEngine->barrier();
  testEngine->runTest();

  fflush(stdout);

  int error = testEngine->getTestErrors();
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  ss << "# Out of bounds values : " << error << " " << (error ? "FAILED" : "OK") << "\n#\n";
  PRINT(ss.str());

  MPI_Finalize();
  if (error != 0) {
    exit(1);
  }
}
