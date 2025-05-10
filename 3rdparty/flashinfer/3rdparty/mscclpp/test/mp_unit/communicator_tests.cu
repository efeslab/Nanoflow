// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mpi.h>

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/semaphore.hpp>

#include "mp_unit_tests.hpp"

void CommunicatorTestBase::SetUp() {
  MultiProcessTest::SetUp();

  if (numRanksToUse == -1) {
    numRanksToUse = gEnv->worldSize;
  }
  ASSERT_LE(numRanksToUse, gEnv->worldSize);

  ibTransport = ibIdToTransport(rankToLocalRank(gEnv->rank));
  MSCCLPP_CUDATHROW(cudaSetDevice(rankToLocalRank(gEnv->rank)));

  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  mscclpp::UniqueId id;
  if (gEnv->rank < numRanksToUse) {
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, numRanksToUse);
    if (gEnv->rank == 0) id = bootstrap->createUniqueId();
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (gEnv->rank >= numRanksToUse) {
    return;
  }
  bootstrap->initialize(id);
  communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
}

void CommunicatorTestBase::TearDown() {
  connections.clear();
  communicator.reset();
  MultiProcessTest::TearDown();
}

void CommunicatorTestBase::setNumRanksToUse(int num) { numRanksToUse = num; }

void CommunicatorTestBase::connectMesh(bool useIpc, bool useIb, bool useEthernet) {
  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures(numRanksToUse);
  for (int i = 0; i < numRanksToUse; i++) {
    if (i != gEnv->rank) {
      if ((rankToNode(i) == rankToNode(gEnv->rank)) && useIpc) {
        connectionFutures[i] = communicator->connectOnSetup(i, 0, mscclpp::Transport::CudaIpc);
      } else if (useIb) {
        connectionFutures[i] = communicator->connectOnSetup(i, 0, ibTransport);
      } else if (useEthernet) {
        connectionFutures[i] = communicator->connectOnSetup(i, 0, mscclpp::Transport::Ethernet);
      }
    }
  }
  communicator->setup();
  for (int i = 0; i < numRanksToUse; i++) {
    if (i != gEnv->rank) {
      connections[i] = connectionFutures[i].get();
    }
  }
}

// Register a local memory and receive corresponding remote memories
void CommunicatorTestBase::registerMemoryPairs(void* buff, size_t buffSize, mscclpp::TransportFlags transport, int tag,
                                               const std::vector<int>& remoteRanks,
                                               mscclpp::RegisteredMemory& localMemory,
                                               std::unordered_map<int, mscclpp::RegisteredMemory>& remoteMemories) {
  localMemory = communicator->registerMemory(buff, buffSize, transport);
  std::unordered_map<int, mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> futureRemoteMemories;
  for (int remoteRank : remoteRanks) {
    if (remoteRank != communicator->bootstrap()->getRank()) {
      communicator->sendMemoryOnSetup(localMemory, remoteRank, tag);
      futureRemoteMemories[remoteRank] = communicator->recvMemoryOnSetup(remoteRank, tag);
    }
  }
  communicator->setup();
  for (int remoteRank : remoteRanks) {
    if (remoteRank != communicator->bootstrap()->getRank()) {
      remoteMemories[remoteRank] = futureRemoteMemories[remoteRank].get();
    }
  }
}

// Register a local memory an receive one corresponding remote memory
void CommunicatorTestBase::registerMemoryPair(void* buff, size_t buffSize, mscclpp::TransportFlags transport, int tag,
                                              int remoteRank, mscclpp::RegisteredMemory& localMemory,
                                              mscclpp::RegisteredMemory& remoteMemory) {
  std::vector<int> remoteRanks = {remoteRank};
  std::unordered_map<int, mscclpp::RegisteredMemory> remoteMemories;
  registerMemoryPairs(buff, buffSize, transport, tag, remoteRanks, localMemory, remoteMemories);
  remoteMemory = remoteMemories[remoteRank];
}

void CommunicatorTest::SetUp() {
  CommunicatorTestBase::SetUp();

  ASSERT_EQ((deviceBufferSize / sizeof(int)) % gEnv->worldSize, 0);

  connectMesh(true, true, false);

  devicePtr.resize(numBuffers);
  localMemory.resize(numBuffers);
  remoteMemory.resize(numBuffers);

  std::vector<int> remoteRanks;
  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      remoteRanks.push_back(i);
    }
  }

  for (size_t n = 0; n < numBuffers; n++) {
    devicePtr[n] = mscclpp::allocSharedCuda<int>(deviceBufferSize / sizeof(int));
    registerMemoryPairs(devicePtr[n].get(), deviceBufferSize, mscclpp::Transport::CudaIpc | ibTransport, 0, remoteRanks,
                        localMemory[n], remoteMemory[n]);
  }
}

void CommunicatorTest::TearDown() {
  remoteMemory.clear();
  localMemory.clear();
  devicePtr.clear();
  CommunicatorTestBase::TearDown();
}

void CommunicatorTest::deviceBufferInit() {
  size_t dataCount = deviceBufferSize / sizeof(int);
  for (int n = 0; n < (int)devicePtr.size(); n++) {
    std::vector<int> hostBuffer(dataCount, 0);
    for (size_t i = 0; i < dataCount; i++) {
      hostBuffer[i] = gEnv->rank + n * gEnv->worldSize;
    }
    mscclpp::memcpyCuda<int>(devicePtr[n].get(), hostBuffer.data(), dataCount, cudaMemcpyHostToDevice);
  }
}

void CommunicatorTest::writeToRemote(int dataCountPerRank) {
  for (size_t n = 0; n < numBuffers; n++) {
    for (int i = 0; i < gEnv->worldSize; i++) {
      if (i != gEnv->rank) {
        auto& conn = connections.at(i);
        auto& peerMemory = remoteMemory[n].at(i);
        conn->write(peerMemory, gEnv->rank * dataCountPerRank * sizeof(int), localMemory[n],
                    gEnv->rank * dataCountPerRank * sizeof(int), dataCountPerRank * sizeof(int));
        conn->flush();
      }
    }
  }
}

bool CommunicatorTest::testWriteCorrectness(bool skipLocal) {
  size_t dataCount = deviceBufferSize / sizeof(int);
  for (int n = 0; n < (int)devicePtr.size(); n++) {
    std::vector<int> hostBuffer(dataCount, 0);
    mscclpp::memcpyCuda<int>(hostBuffer.data(), devicePtr[n].get(), dataCount, cudaMemcpyDeviceToHost);
    for (int i = 0; i < gEnv->worldSize; i++) {
      if (((i / gEnv->nRanksPerNode) == (gEnv->rank / gEnv->nRanksPerNode)) && skipLocal) {
        continue;
      }
      for (size_t j = i * dataCount / gEnv->worldSize; j < (i + 1) * dataCount / gEnv->worldSize; j++) {
        if (hostBuffer[j] != i + n * gEnv->worldSize) {
          return false;
        }
      }
    }
  }
  return true;
}

TEST_F(CommunicatorTest, BasicWrite) {
  if (gEnv->rank >= numRanksToUse) return;

  deviceBufferInit();
  communicator->bootstrap()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);
  communicator->bootstrap()->barrier();

  // polling until it becomes ready
  bool ready = false;
  int niter = 0;
  do {
    ready = testWriteCorrectness();
    niter++;
    if (niter == 10000) {
      FAIL() << "Polling is stuck.";
    }
  } while (!ready);
  communicator->bootstrap()->barrier();
}

__global__ void kernelWaitSemaphores(mscclpp::Host2DeviceSemaphore::DeviceHandle* deviceSemaphores, int rank,
                                     int worldSize) {
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize) {
    deviceSemaphores[tid].wait();
  }
}

TEST_F(CommunicatorTest, WriteWithDeviceSemaphores) {
  if (gEnv->rank >= numRanksToUse) return;

  std::unordered_map<int, std::shared_ptr<mscclpp::Host2DeviceSemaphore>> semaphores;
  for (auto entry : connections) {
    auto& conn = entry.second;
    semaphores.insert({entry.first, std::make_shared<mscclpp::Host2DeviceSemaphore>(*communicator.get(), conn)});
  }
  communicator->setup();
  communicator->bootstrap()->barrier();

  deviceBufferInit();
  communicator->bootstrap()->barrier();

  auto deviceSemaphoreHandles = mscclpp::allocSharedCuda<mscclpp::Host2DeviceSemaphore::DeviceHandle>(gEnv->worldSize);
  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      mscclpp::Host2DeviceSemaphore::DeviceHandle deviceHandle = semaphores[i]->deviceHandle();
      mscclpp::memcpyCuda<mscclpp::Host2DeviceSemaphore::DeviceHandle>(deviceSemaphoreHandles.get() + i, &deviceHandle,
                                                                       1, cudaMemcpyHostToDevice);
    }
  }
  communicator->bootstrap()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      semaphores[i]->signal();
    }
  }

  kernelWaitSemaphores<<<1, gEnv->worldSize>>>(deviceSemaphoreHandles.get(), gEnv->rank, gEnv->worldSize);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  ASSERT_TRUE(testWriteCorrectness());
  communicator->bootstrap()->barrier();
}

TEST_F(CommunicatorTest, WriteWithHostSemaphores) {
  if (gEnv->rank >= numRanksToUse) return;

  std::unordered_map<int, std::shared_ptr<mscclpp::Host2HostSemaphore>> semaphores;
  for (auto entry : connections) {
    auto& conn = entry.second;
    // Host2HostSemaphore cannot be used with CudaIpc transport
    if (conn->transport() == mscclpp::Transport::CudaIpc) continue;
    semaphores.insert({entry.first, std::make_shared<mscclpp::Host2HostSemaphore>(*communicator.get(), conn)});
  }
  communicator->setup();
  communicator->bootstrap()->barrier();

  deviceBufferInit();
  communicator->bootstrap()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      semaphores[i]->signal();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      semaphores[i]->wait();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      semaphores[i]->signal();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      semaphores[i]->wait();
    }
  }

  ASSERT_TRUE(testWriteCorrectness());
  communicator->bootstrap()->barrier();
}