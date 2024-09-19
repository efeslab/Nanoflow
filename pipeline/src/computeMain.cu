#include <computeBound.cuh>
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

std::vector<vortexOutputData> outputData;
std::vector<vortexInitData> inputData;
std::vector<vortexConfigData> configData;
std::vector<vortexUpdateData> updateData;

inline void print_help() {
	spdlog::warn(
		"./test_compute [config_path]");
}

int main(int argc, char** argv) {
	spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] [Th %t] %v");
	
	std::string config_path = "../config_all/llama3-8B/fewer_layers/1024.json";
	if (argc >=2)
		config_path = argv[1];
	std::ifstream file(config_path); 
	if (!file.is_open()) { 
		std::cerr << "Could not open the file: " << config_path << std::endl;
		return; 
	} 
    // Read the file contents into a string 
	spdlog::info("read config file");
	json j; 
	file >> j; 
	// Close the file 
	file.close();

	int nranks = j["serve_configs"]["actual_gpu_num"];
	int vnranks = j["model_configs"]["gpu_num"];

	TensorManager::getInstance().init(nranks, vnranks);
	updateVortexModelConfig(config_path);
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
		cudaSetDevice(rank);
		auto weight = vortexModelWeight();
		createModelWeight(weight, rank);
		createInitData(inputData[rank], weight, rank);
	}

	configData.resize(nranks);
	for (int rank = 0; rank < nranks; ++rank) {
		// if (Worker::PipeTy == Worker::PipelineType::PLLM || Worker::PipeTy == Worker::PipelineType::PLLMOFFLOAD) {
		// 	createConfigData(configData[rank], rank, Pipeline::gemmConfig);
		// }
		// else if (Worker::PipeTy == Worker::PipelineType::NONOVERLAP) {
		// 	createConfigData(configData[rank], rank, NonOverlapPipeline::gemmConfig);
		// }
		// else if (Worker::PipeTy == Worker::PipelineType::NANOBATCH) {
		// 	createConfigData(configData[rank], rank, NonOverlapNanoBatchPipeline::gemmConfig);
		// }
		// else if (Worker::PipeTy == Worker::PipelineType::LOCAL) {
		// 	createConfigData(configData[rank], rank, LocalPipeline::gemmConfig);
		// }
		// else
		// 	spdlog::error("Unknown pipeline type {}", static_cast<int>(Worker::PipeTy));

		// readConfigData(configData[rank], rank, "/code/pllm/compute-bound/config/2048.json");
		readConfigData(configData[rank], rank, config_path);
	}

	updateData.resize(nranks);
	int global_batch_size = j["pipeline_configs"]["global_batch_size"];
	for (int rank = 0; rank < nranks; ++rank) {
		createUpdateData(updateData[rank], rank, global_batch_size, 1024, 512);
	}

#endif

	Worker::PipelineType p;
	std::string pipeDesc = j["serve_configs"]["pipeline_type"];
	spdlog::info("pipeline type: {}", pipeDesc);
//pipeline_map = {
//     "LOCAL": pllm_python.PipelineType.LOCAL,
//     "PLLM": pllm_python.PipelineType.PLLM,
//     "NON_OVERLAP": pllm_python.PipelineType.NONOVERLAP,
//     "NANOBATCH": pllm_python.PipelineType.NANOBATCH,
//     "PLLM_OFFLOAD": pllm_python.PipelineType.PLLMOFFLOAD,
//     "NON_OVERLAP_LOCAL": pllm_python.PipelineType.NON_OVERLAP_LOCAL,
//     "NANOBATCH_LOCAL": pllm_python.PipelineType.NANOBATCH_LOCAL,
// }
	if (pipeDesc == "LOCAL") {
		p = Worker::PipelineType::LOCAL;
	}
	else if (pipeDesc == "PLLM") {
		p = Worker::PipelineType::PLLM;
	}
	else if (pipeDesc == "NON_OVERLAP") {
		p = Worker::PipelineType::NONOVERLAP;
	}
	else if (pipeDesc == "NANOBATCH") {
		p = Worker::PipelineType::NANOBATCH;
	}
	else if (pipeDesc == "PLLM_OFFLOAD") {
		p = Worker::PipelineType::PLLMOFFLOAD;
	}
	else if (pipeDesc == "NON_OVERLAP_LOCAL") {
		p = Worker::PipelineType::NON_OVERLAP_LOCAL;
	}
	else if (pipeDesc == "NANOBATCH_LOCAL") {
		p = Worker::PipelineType::NANOBATCH_LOCAL;
	}
	else if (pipeDesc == "KQVBIAS") {
		p = Worker::PipelineType::KQVBIAS;
	}
	else if (pipeDesc == "NONOVERLAP_KQVBIAS") {
		p = Worker::PipelineType::NONOVERLAP_KQVBIAS;
	}
	else if (pipeDesc == "NANOBATCH_KQVBIAS") {
		p = Worker::PipelineType::NANOBATCH_KQVBIAS;
	}
	else {
		spdlog::error("Unknown pipeline type {}", pipeDesc);
	}
	Worker::PipeTy = p;
	init(nranks, vnranks, inputData, outputData, p);

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
