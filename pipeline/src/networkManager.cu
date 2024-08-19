#include "networkManager.cuh"
#include "config.h"
#include <iostream>

#ifdef ENABLE_MPI
std::shared_ptr<NetworkManager> netmgr;

void NetworkManager::init(int argc, char** argv) {
#ifdef ENABLE_NETWORK
	// Initialize the MPI environment
	MPI_Init(&argc, &argv);
	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	// Get the rank of the process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	spdlog::info("Hello world from rank {} out of {} ranks", rank, nranks);

	// Print off a hello world message
	std::cout << "Hello world from rank " << rank << " out of " << nranks << " ranks" << std::endl;

	initialized = true;
#endif
}

void NetworkManager::finalize() {
#ifdef ENABLE_NETWORK
	if (initialized) {
		MPI_Finalize();
	}
#endif
}
#else // ENABLE_MPI

SharedState shared_state;
std::unique_ptr<SimpleThreadSync> worker_sync;
std::unique_ptr<SimpleThreadSync> global_sync;

#endif // ENABLE_MPI
