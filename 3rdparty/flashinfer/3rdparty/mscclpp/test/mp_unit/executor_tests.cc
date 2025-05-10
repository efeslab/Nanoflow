// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mpi.h>

#include <filesystem>

#include "mp_unit_tests.hpp"

namespace {
std::string getExecutablePath() {
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count == -1) {
    throw std::runtime_error("Failed to get executable path");
  }
  return std::string(result, count);
}
}  // namespace

void ExecutorTest::SetUp() {
  MultiProcessTest::SetUp();

  MSCCLPP_CUDATHROW(cudaSetDevice(rankToLocalRank(gEnv->rank)));
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  mscclpp::UniqueId id;
  bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  if (gEnv->rank == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  std::shared_ptr<mscclpp::Communicator> communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
  executor = std::make_shared<mscclpp::Executor>(communicator);
}

void ExecutorTest::TearDown() {
  executor.reset();
  MultiProcessTest::TearDown();
}

TEST_F(ExecutorTest, TwoNodesAllreduce) {
  if (gEnv->worldSize != 2 || gEnv->nRanksPerNode != 2) {
    GTEST_SKIP() << "This test requires world size to be 2 and ranks per node to be 2";
    return;
  }
  std::string executablePath = getExecutablePath();
  std::filesystem::path path = executablePath;
  std::filesystem::path executionFilesPath =
      path.parent_path().parent_path().parent_path() / "test/execution-files/allreduce.json";
  mscclpp::ExecutionPlan plan("allreduce_pairs", executionFilesPath.string());
  const int bufferSize = 1024 * 1024;
  std::shared_ptr<char> sendbuff = mscclpp::allocExtSharedCuda<char>(bufferSize);
  mscclpp::CudaStreamWithFlags stream(cudaStreamNonBlocking);
  executor->execute(gEnv->rank, sendbuff.get(), sendbuff.get(), bufferSize, bufferSize, mscclpp::DataType::FLOAT16, 512,
                    plan, stream);
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
}
