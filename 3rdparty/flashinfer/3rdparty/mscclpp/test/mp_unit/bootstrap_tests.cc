// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mpi.h>

#include "mp_unit_tests.hpp"

void BootstrapTest::bootstrapTestAllGather(std::shared_ptr<mscclpp::Bootstrap> bootstrap) {
  std::vector<int> tmp(bootstrap->getNranks(), 0);
  tmp[bootstrap->getRank()] = bootstrap->getRank() + 1;
  bootstrap->allGather(tmp.data(), sizeof(int));
  for (int i = 0; i < bootstrap->getNranks(); ++i) {
    EXPECT_EQ(tmp[i], i + 1);
  }
}

void BootstrapTest::bootstrapTestBarrier(std::shared_ptr<mscclpp::Bootstrap> bootstrap) { bootstrap->barrier(); }

void BootstrapTest::bootstrapTestSendRecv(std::shared_ptr<mscclpp::Bootstrap> bootstrap) {
  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i) continue;
    int msg1 = (bootstrap->getRank() + 1) * 3;
    int msg2 = (bootstrap->getRank() + 1) * 3 + 1;
    int msg3 = (bootstrap->getRank() + 1) * 3 + 2;
    bootstrap->send(&msg1, sizeof(int), i, 0);
    bootstrap->send(&msg2, sizeof(int), i, 1);
    bootstrap->send(&msg3, sizeof(int), i, 2);
  }

  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i) continue;
    int msg1 = 0;
    int msg2 = 0;
    int msg3 = 0;
    // recv them in the opposite order to check correctness
    bootstrap->recv(&msg2, sizeof(int), i, 1);
    bootstrap->recv(&msg3, sizeof(int), i, 2);
    bootstrap->recv(&msg1, sizeof(int), i, 0);
    EXPECT_EQ(msg1, (i + 1) * 3);
    EXPECT_EQ(msg2, (i + 1) * 3 + 1);
    EXPECT_EQ(msg3, (i + 1) * 3 + 2);
  }
}

void BootstrapTest::bootstrapTestAll(std::shared_ptr<mscclpp::Bootstrap> bootstrap) {
  bootstrapTestAllGather(bootstrap);
  bootstrapTestBarrier(bootstrap);
  bootstrapTestSendRecv(bootstrap);
}

TEST_F(BootstrapTest, WithId) {
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  bootstrapTestAll(bootstrap);
}

TEST_F(BootstrapTest, WithIpPortPair) {
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  bootstrap->initialize(gEnv->args["ip_port"]);
  bootstrapTestAll(bootstrap);
}

TEST_F(BootstrapTest, ResumeWithId) {
  // This test may take a few minutes.
  bootstrapTestTimer.set(300);

  for (int i = 0; i < 10; ++i) {
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
    mscclpp::UniqueId id;
    if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    bootstrap->initialize(id, 300);
  }
}

TEST_F(BootstrapTest, ResumeWithIpPortPair) {
  for (int i = 0; i < 5; ++i) {
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
    bootstrap->initialize(gEnv->args["ip_port"]);
  }
}

TEST_F(BootstrapTest, ExitBeforeConnect) {
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  bootstrap->createUniqueId();
}

TEST_F(BootstrapTest, TimeoutWithId) {
  mscclpp::Timer timer;

  // All ranks initialize a bootstrap with their own id (will hang)
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  mscclpp::UniqueId id = bootstrap->createUniqueId();

  try {
    // Set bootstrap timeout to 1 second
    bootstrap->initialize(id, 1);
  } catch (const mscclpp::Error& e) {
    ASSERT_EQ(e.getErrorCode(), mscclpp::ErrorCode::Timeout);
  }

  // Timeout should be sligtly greater than 1 second
  ASSERT_GT(timer.elapsed(), 1000000);
  ASSERT_LT(timer.elapsed(), 1100000);
}

class MPIBootstrap : public mscclpp::Bootstrap {
 public:
  MPIBootstrap() : Bootstrap() {}
  int getRank() override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
  int getNranks() override {
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    return worldSize;
  }
  int getNranksPerNode() override {
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    int shmrank;
    MPI_Comm_size(shmcomm, &shmrank);
    return shmrank;
  }
  void allGather(void* sendbuf, int size) override {
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, sendbuf, size, MPI_BYTE, MPI_COMM_WORLD);
  }
  void barrier() override { MPI_Barrier(MPI_COMM_WORLD); }
  void send(void* sendbuf, int size, int dest, int tag) override {
    MPI_Send(sendbuf, size, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
  }
  void recv(void* recvbuf, int size, int source, int tag) override {
    MPI_Recv(recvbuf, size, MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
};

TEST_F(BootstrapTest, MPIBootstrap) {
  auto bootstrap = std::make_shared<MPIBootstrap>();
  bootstrapTestAll(bootstrap);
}
