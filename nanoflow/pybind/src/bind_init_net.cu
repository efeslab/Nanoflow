#include <iostream>
#include <arpa/inet.h>  // for inet_pton
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>

#include <mscclpp/core.hpp>
#include "netWrapper.cuh"

namespace py = pybind11;

// store bootstrap, communicator, and connection objects in a global variable
// to be used in the init_net function

char* kAddress = "localhost";
int kPort = 12345;

std::shared_ptr<mscclpp::TcpBootstrap> g_bootstrap;
std::shared_ptr<mscclpp::Communicator> g_comm;
std::vector<std::shared_ptr<mscclpp::Connection>> g_connections;
mscclpp::UniqueId g_uniqueId;
int g_rank;
int g_nranks;

void send_unique_id(mscclpp::UniqueId& uid) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(kPort);

    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 16);  // Allow up to 16 connections

    for (int i = 1; i < g_nranks; ++i) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        send(client_fd, &uid, sizeof(uid), 0);
        close(client_fd);
    }

    close(server_fd);
}

void recv_unique_id(mscclpp::UniqueId& uid) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(kPort);
    inet_pton(AF_INET, kAddress, &serv_addr.sin_addr);

    // Wait until rank 0 is listening
    while (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    recv(sock, &uid, sizeof(uid), MSG_WAITALL);
    close(sock);
}

void init_net(int rank, int nranks) {
    if (g_bootstrap) {
        std::cerr << "Unique ID already created" << std::endl;
    }
    // Print off a hello world message
    std::cout << "Hello world from rank " << rank << " out of " << nranks << " ranks" << std::endl;
    cudaSetDevice(rank);

    g_rank = rank;
    g_nranks = nranks;

    g_bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
    if (rank == 0) {
        std::cout << "Creating unique ID" << std::endl;
        g_uniqueId = g_bootstrap->createUniqueId();
        send_unique_id(g_uniqueId);
        std::cout << "Unique ID sent" << std::endl;
    } else {
        std::cout << "Waiting for unique ID" << std::endl;
        recv_unique_id(g_uniqueId);
        std::cout << "Unique ID received" << std::endl;
    }
    g_bootstrap->initialize(g_uniqueId);

    // Initialize Communicator
    g_comm = std::make_shared<mscclpp::Communicator>(g_bootstrap);

    // Initialize Connections
    std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
    for (int r = 0; r < nranks; ++r) {
        if (r == rank) continue;
        mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
        connectionFutures.push_back(g_comm->connectOnSetup(r, 0, transport));
    }
    g_comm->setup();
    std::transform(
        connectionFutures.begin(), connectionFutures.end(), std::back_inserter(g_connections),
        [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) { return future.get(); });
    
    std::cout << "Network initialized for rank " << rank << " out of " << nranks << " ranks" << std::endl;
}

void init_allgather_wrapper() {
    NetAllGather wrapper;
    
}




PYBIND11_MODULE(bind_init_net, m) {
    m.doc() = "Pybind11 bindings for building network";  // Optional module docstring
    m.def("init_net", &init_net, "Initialize the network for distributed training");
}
