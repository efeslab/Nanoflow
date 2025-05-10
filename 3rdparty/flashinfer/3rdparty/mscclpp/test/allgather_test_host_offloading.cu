// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/core.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/semaphore.hpp>

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include "mpi.h"
#endif  // MSCCLPP_USE_MPI_FOR_TESTS
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <unordered_map>

int nranksPerNode;
int rank;
int world_size;

// Measure current time in second.
static double getTime(void) {
  struct timespec tspec;
  if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
    printf("clock_gettime failed\n");
    exit(EXIT_FAILURE);
  }
  return (tspec.tv_nsec / 1.0e9) + tspec.tv_sec;
}

__global__ void kernel(int r, mscclpp::FifoDeviceHandle fifo, mscclpp::Host2DeviceSemaphore::DeviceHandle* handles,
                       int handleIndex) {
  int tid = threadIdx.x;
  __syncthreads();
  // uint64_t tail;
  if (tid == 0) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = handleIndex;
    fifo.push(trigger);
    // tail = fifo.push(trigger);
  }
  if (tid != r) handles[tid].wait();
  // if (tid == 0)
  //   while(*(volatile uint64_t*)fifo.tailReplica < tail) {};
}

int rankToLocalRank(int rank) { return rank % nranksPerNode; }

int rankToNode(int rank) { return rank / nranksPerNode; }

void print_usage(const char* prog) {
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  printf("usage: %s IP:PORT [rank nranks]\n", prog);
#else
  printf("usage: %s IP:PORT rank nranks\n", prog);
#endif
}

void initializeAndAllocateAllGatherData(int rank, int world_size, size_t dataSize, size_t nelemsPerGPU, int** data_h,
                                        int** data_d) {
  MSCCLPP_CUDATHROW(cudaMalloc(data_d, dataSize));
  MSCCLPP_CUDATHROW(cudaMemset(*data_d, 0, dataSize));

  *data_h = new int[nelemsPerGPU * world_size];
  for (size_t i = 0; i < nelemsPerGPU * world_size; i++) {
    int val = i + 1;
    if (i / nelemsPerGPU == (size_t)rank) {
      (*data_h)[i] = val;
    } else {
      (*data_h)[i] = 0;
    }
  }
  MSCCLPP_CUDATHROW(cudaMemcpy(*data_d, *data_h, dataSize, cudaMemcpyHostToDevice));
}

class MyProxyService {
 private:
  int dataSize_;
  std::vector<mscclpp::RegisteredMemory> remoteMemories_;
  mscclpp::RegisteredMemory localMemory_;
  std::vector<std::shared_ptr<mscclpp::Host2HostSemaphore>> hostSemaphores_;
  std::vector<std::shared_ptr<mscclpp::Host2DeviceSemaphore>> deviceSemaphores1_;
  std::vector<std::shared_ptr<mscclpp::Host2DeviceSemaphore>> deviceSemaphores2_;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections_;
  mscclpp::Proxy proxy_;
  int deviceNumaNode_;

 public:
  MyProxyService(mscclpp::Communicator& comm, int* data_d, int dataSize)
      : dataSize_(dataSize),
        remoteMemories_(world_size),
        connections_(world_size),
        proxy_([&](mscclpp::ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }, [&]() { bindThread(); }) {
    int cudaDevice;
    MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
    deviceNumaNode_ = mscclpp::getDeviceNumaNode(cudaDevice);

    int thisNode = rankToNode(rank);
    int cudaNum = rankToLocalRank(rank);
    std::string ibDevStr = "mlx5_ib" + std::to_string(cudaNum);
    mscclpp::Transport ibTransport = mscclpp::getIBTransportByDeviceName(ibDevStr);
    std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionsFuture(world_size);
    std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteMemoriesFuture(world_size);

    localMemory_ = comm.registerMemory(data_d, dataSize, mscclpp::Transport::CudaIpc | ibTransport);
    for (int r = 0; r < world_size; ++r) {
      if (r == rank) {
        hostSemaphores_.emplace_back(nullptr);
        deviceSemaphores1_.emplace_back(nullptr);
        deviceSemaphores2_.emplace_back(nullptr);
        continue;
      }
      mscclpp::Transport transport;
      if (rankToNode(r) == thisNode) {
        transport = mscclpp::Transport::CudaIpc;
      } else {
        transport = ibTransport;
      }
      // Connect with all other ranks
      connectionsFuture[r] = comm.connectOnSetup(r, 0, transport);
      comm.sendMemoryOnSetup(localMemory_, r, 0);

      remoteMemoriesFuture[r] = comm.recvMemoryOnSetup(r, 0);
    }

    comm.setup();

    for (int r = 0; r < world_size; ++r) {
      if (r == rank) {
        continue;
      }
      connections_[r] = connectionsFuture[r].get();
      if (rankToNode(r) == thisNode) {
        hostSemaphores_.emplace_back(nullptr);
      } else {
        hostSemaphores_.emplace_back(std::make_shared<mscclpp::Host2HostSemaphore>(comm, connections_[r]));
      }
      deviceSemaphores1_.emplace_back(std::make_shared<mscclpp::Host2DeviceSemaphore>(comm, connections_[r]));
      deviceSemaphores2_.emplace_back(std::make_shared<mscclpp::Host2DeviceSemaphore>(comm, connections_[r]));
      remoteMemories_[r] = remoteMemoriesFuture[r].get();
    }

    comm.setup();
  }

  void bindThread() {
    if (deviceNumaNode_ >= 0) {
      mscclpp::numaBind(deviceNumaNode_);
    }
  }

  mscclpp::ProxyHandlerResult handleTrigger(mscclpp::ProxyTrigger triggerRaw) {
    static int flusher = 0;
    if (triggerRaw.fst > 0) {
      int dataSizePerRank = dataSize_ / world_size;
      for (int r = 1; r < world_size; ++r) {
        int nghr = (rank + r) % world_size;
        connections_[nghr]->write(remoteMemories_[nghr], rank * dataSizePerRank, localMemory_, rank * dataSizePerRank,
                                  dataSizePerRank);
        if (triggerRaw.fst == 1)
          deviceSemaphores1_[nghr]->signal();
        else
          deviceSemaphores2_[nghr]->signal();
        if ((flusher % 64) == 0 && mscclpp::AllIBTransports.has(connections_[nghr]->transport())) {
          // if we are using IB transport, we need a flush every once in a while
          connections_[nghr]->flush();
        }
      }
      flusher++;
    }
    return mscclpp::ProxyHandlerResult::FlushFifoTailAndContinue;
  }

  void start() { proxy_.start(); }

  void stop() { proxy_.stop(); }

  mscclpp::Fifo& fifo() { return proxy_.fifo(); }

  mscclpp::Host2DeviceSemaphore::DeviceHandle getDeviceHandle1(int r) { return deviceSemaphores1_[r]->deviceHandle(); }

  mscclpp::Host2DeviceSemaphore::DeviceHandle getDeviceHandle2(int r) { return deviceSemaphores2_[r]->deviceHandle(); }
};

std::unordered_map<std::string, std::string> parseArgs(int argc, char* argv[]) {
  std::unordered_map<std::string, std::string> options;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-datasize") {
      if (i + 1 < argc) {
        options["datasize"] = argv[++i];
      } else {
        fprintf(stderr, "Error: -datasize option requires an argument.\n");
        exit(-1);
      }
    } else if (arg == "-help" || arg == "-h") {
      exit(0);
    } else {
      fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
      exit(-1);
    }
  }
  return options;
}

int main(int argc, char* argv[]) {
  // sleep(10);
  MPI_Init(&argc, &argv);
  auto parsedArgs = parseArgs(argc, argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // get the local number of nodes with MPI
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  int shmrank;
  MPI_Comm_size(shmcomm, &shmrank);
  nranksPerNode = shmrank;
  MPI_Comm_free(&shmcomm);

  int cudaNum = rankToLocalRank(rank);
  MSCCLPP_CUDATHROW(cudaSetDevice(cudaNum));

  if (rank == 0) printf("Initializing MSCCL++\n");
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
  mscclpp::UniqueId uniqueId;
  if (rank == 0) uniqueId = bootstrap->createUniqueId();
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(uniqueId);
  mscclpp::Communicator comm(bootstrap);

  int* data_d;
  int* data_h;
  size_t dataSize = 1024 * 1024 * 1024;
  if (parsedArgs.find("datasize") != parsedArgs.end()) {
    dataSize = std::stoul(parsedArgs["datasize"]);
  }
  size_t nelemsPerGPU = dataSize / sizeof(int) / world_size;

  if (rank == 0) printf("Initializing data for allgather test\n");
  initializeAndAllocateAllGatherData(rank, world_size, dataSize, nelemsPerGPU, &data_h, &data_d);

  if (rank == 0) printf("Setting up the connection in MSCCL++\n");

  MyProxyService proxyService(comm, data_d, dataSize);
  // setupProxyService(comm, proxyService, data_d, dataSize);

  if (rank == 0) printf("Launching MSCCL++ proxy threads\n");
  proxyService.start();
  mscclpp::FifoDeviceHandle fifo = proxyService.fifo().deviceHandle();
  if (rank == 0) printf("Testing the correctness of AllGather implementation\n");
  cudaStream_t stream;
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  mscclpp::Host2DeviceSemaphore::DeviceHandle* deviceHandles1;
  mscclpp::Host2DeviceSemaphore::DeviceHandle* deviceHandles2;

  MSCCLPP_CUDATHROW(cudaMalloc(&deviceHandles1, sizeof(mscclpp::Host2DeviceSemaphore::DeviceHandle) * world_size));
  for (int i = 0; i < world_size; ++i) {
    if (i == rank) continue;
    auto handle = proxyService.getDeviceHandle1(i);
    MSCCLPP_CUDATHROW(cudaMemcpy(&deviceHandles1[i], &handle, sizeof(mscclpp::Host2DeviceSemaphore::DeviceHandle),
                                 cudaMemcpyHostToDevice));
  }

  MSCCLPP_CUDATHROW(cudaMalloc(&deviceHandles2, sizeof(mscclpp::Host2DeviceSemaphore::DeviceHandle) * world_size));
  for (int i = 0; i < world_size; ++i) {
    if (i == rank) continue;
    auto handle = proxyService.getDeviceHandle2(i);
    MSCCLPP_CUDATHROW(cudaMemcpy(&deviceHandles2[i], &handle, sizeof(mscclpp::Host2DeviceSemaphore::DeviceHandle),
                                 cudaMemcpyHostToDevice));
  }

  kernel<<<1, world_size, 0, stream>>>(rank, fifo, deviceHandles1, 1);
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  MSCCLPP_CUDATHROW(cudaMemcpy(data_h, data_d, dataSize, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nelemsPerGPU * world_size; i++) {
    int val = i + 1;
    if (data_h[i] != val) {
      printf("oh uh! data_h[%ld] (%d) != val (%d)\n", i, data_h[i], val);
      break;
    }
  }

  bootstrap->barrier();
  if (rank == 0) printf("Correctness test passed!\n");

  double t0, t1, ms, time_in_us;
  int iterwithoutcudagraph = 10;
  if (rank == 0) printf("Running %d iterations of the kernel without CUDA graph\n", iterwithoutcudagraph);
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  bootstrap->barrier();
  t0 = getTime();
  for (int i = 0; i < iterwithoutcudagraph; ++i) {
    kernel<<<1, world_size, 0, stream>>>(rank, fifo, deviceHandles1, 1);
    kernel<<<1, world_size, 0, stream>>>(rank, fifo, deviceHandles2, 2);
  }
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  bootstrap->barrier();
  t1 = getTime();
  ms = (t1 - t0) * 1000.0;
  time_in_us = ms * 1000. / (float)iterwithoutcudagraph / 2;
  printf("No Graph %d report: size %lu time: %f us/iter algBW %f GBps\n", rank, dataSize, time_in_us,
         (double)(dataSize) / 1e9 / (time_in_us / 1e6));

  // cudaGraph Capture
  int cudagraphiter = 10;
  if (rank == 0) printf("Capturing %d iterations of the kernel in a CUDA graph\n", cudagraphiter);
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  MSCCLPP_CUDATHROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < cudagraphiter; ++i) {
    kernel<<<1, world_size, 0, stream>>>(rank, fifo, deviceHandles1, 1);
    kernel<<<1, world_size, 0, stream>>>(rank, fifo, deviceHandles2, 2);
  }
  MSCCLPP_CUDATHROW(cudaStreamEndCapture(stream, &graph));
  MSCCLPP_CUDATHROW(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

  int cudagraphwarmup = 10;
  if (rank == 0)
    printf("Warming up %d iterations of the CUDA graph with %d iterations of the kernel\n", cudagraphwarmup,
           cudagraphiter);
  for (int i = 0; i < cudagraphwarmup; ++i) {
    MSCCLPP_CUDATHROW(cudaGraphLaunch(instance, stream));
  }
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  // measure runtime
  int cudagraphlaunch = 10;
  if (rank == 0)
    printf("Running %d iterations of the CUDA graph with %d iterations of the kernel\n", cudagraphlaunch,
           cudagraphiter);
  bootstrap->barrier();
  t0 = getTime();
  for (int i = 0; i < cudagraphlaunch; ++i) {
    MSCCLPP_CUDATHROW(cudaGraphLaunch(instance, stream));
  }
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  t1 = getTime();
  ms = (t1 - t0) * 1000.0;
  time_in_us = ms * 1000. / (float)cudagraphlaunch / (float)cudagraphiter / 2;
  if (rank == 0)
    printf("Rank %d report: size %lu time: %f us/iter algBW %f GBps\n", rank, dataSize, time_in_us,
           (double)(dataSize) / 1e9 / (time_in_us / 1e6));
  bootstrap->barrier();

  if (rank == 0) printf("Stopping MSCCL++ proxy threads\n");
  proxyService.stop();

  MSCCLPP_CUDATHROW(cudaFree(data_d));
  MSCCLPP_CUDATHROW(cudaFree(deviceHandles1));
  MSCCLPP_CUDATHROW(cudaFree(deviceHandles2));

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Finalize();
#endif
  return 0;
}
