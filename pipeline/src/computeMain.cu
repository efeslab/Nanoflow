#include <computeBound.cuh>


std::vector<vortexOutputData> outputData;
std::vector<vortexInitData> inputData;
std::vector<vortexConfigData> configData;
std::vector<vortexUpdateData> updateData;

inline void print_help() {
	spdlog::warn(
		"should provide nranks as argv[1], vnranks as argv[2]. `nranks` GPUs will be used to simulate executiosn on `vnranks` GPUs. "
		"argv[3] if given will be used to select the pipeline type (default is 0, nanoflow pipeline; 1 means the non-overlap pipeline)");
}

int main(int argc, char** argv) {
	spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] [Th %t] %v");
	
	if(argc < 3) {
		print_help();
		return -1;
	}

	int nranks = atoi(argv[1]);
	int vnranks = atoi(argv[2]);
	if (argc >= 4) {
		int type = atoi(argv[3]);
		Worker::PipeTy = static_cast<Worker::PipelineType>(type);
	}

#ifdef ENABLE_MPI
	netmgr = std::make_shared<NetworkManager>();
	netmgr->init(0, nullptr);
	spdlog::info("Create data");
	CUDA_CHECK(cudaSetDevice(netmgr->rank));
	inputData.resize(1);
	outputData.resize(1);
	createInitData(inputData[0]);
#else
	spdlog::info("Create data");
	inputData.resize(nranks);
	outputData.resize(nranks);
	for (int rank = 0; rank < nranks; ++rank) {
		createInitData(inputData[rank], rank);
	}

	configData.resize(nranks);
	for (int rank = 0; rank < nranks; ++rank) {
		createConfigData(configData[rank], rank);
	}

	updateData.resize(nranks);
	for (int rank = 0; rank < nranks; ++rank) {
		createUpdateData(updateData[rank], rank);
	}

#endif



	init(nranks, vnranks, inputData, outputData, Worker::PipeTy);

	config(nranks, configData);

	update(nranks, updateData);

	run();
	run();
	run();
	run();
	run();

	finalize();
	
	return 0;
}
