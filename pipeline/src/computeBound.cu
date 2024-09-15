#include <cstdint>
#include <iostream>

#include "computeBound.cuh"

#include "eventManager.cuh"
#include "gemmShape.cuh"
#include "pipeline.h"

#include "config.h"
#include "vortexData.cuh"

#include "networkManager.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_MPI
std::shared_ptr<Worker> worker;
#else
std::vector<std::unique_ptr<Worker>> workers;
#endif

std::vector<vortexOutputData> outputs;
std::vector<int> aggregated_output;


void run_async() {
#ifdef ENABLE_MPI
	worker->run_pipeline();
#else
	shared_state.op = WorkerOp::RUN;
	// issue op command to worker
	global_sync->barrier();
	spdlog::info("All worker threads started pipeline");
#endif
}

void run_async_wait() {
#ifdef ENABLE_MPI
#else
	// wait for worker to complete this op
	global_sync->barrier();
	spdlog::info("All worker threads finished pipeline");
	outputs.clear();
	for (auto& worker : workers) {
		outputs.push_back(*worker->getOutput());
	}
	aggregated_output.clear();
	for (auto& output : outputs) {
		for (int i = 0; i < output.sampled_tokens; ++i) {
			aggregated_output.push_back(output.sampled_token_array[i]);
		}
	}

#endif
}

void run() {
	run_async();
	run_async_wait();
}



void init(int nranks,
		  int vnranks,
		  std::vector<vortexInitData>& input,
		  std::vector<vortexOutputData>& output,
		  Worker::PipelineType pipeTy) {
	spdlog::set_level(spdlog::level::trace);
	spdlog::info("Init network");
#ifdef ENABLE_MPI
	assert(input.size() == 1);
	assert(output.size() == 1);
	worker =
		std::make_shared<Worker>(netmgr->rank, netmgr->nranks, &input.front(), &output.front());
	worker->init();
#else
	assert(input.size() == nranks);
	assert(output.size() == nranks);

	Worker::PipeTy = pipeTy;
	// for (size_t i = 0; i < nranks; i++)
	// {
	// 	allocateKVData(input[i], i);
	// }
	spdlog::info("Init data created");

	worker_sync = std::make_unique<SimpleThreadSync>(nranks);
	global_sync = std::make_unique<SimpleThreadSync>(nranks + 1);

	vortexInitData* persistant_input = new vortexInitData[nranks];
	for (int i = 0; i < nranks; ++i) {
		persistant_input[i] = input[i];
	}

	for(int rank = 0; rank < nranks; ++rank) {
		spdlog::info("Rank is {} ", rank);
		int coreNum =0;
		if (rank < 4){
			coreNum = rank + 10;
		} else{
			coreNum = rank + 20;
		}
		workers
			.emplace_back(std::make_unique<Worker>(
				rank, nranks, vnranks, persistant_input + rank, &output[rank]))
			->as_thread(coreNum);
	}
	global_sync->barrier();
	spdlog::info("All multi-threading workers are initialized");
#endif
}

void update(int nranks, std::vector<vortexUpdateData>& updateData) {
	spdlog::info("Update pipeline");
#ifdef ENABLE_MPI
	assert(updateData.size() == 1);
	worker->run_update(&updateData.front());
#else
	auto &update_d = updateData[0];
	int* complete_input_tokens = new int[update_d.decodePrefillBorder + update_d.prefillTokensNum];
	int input_tokens_count = 0;
	spdlog::info("update_d.keepTokenListLength: {}", update_d.keepTokenListLength);
	spdlog::info("aggregated_output size: {}", aggregated_output.size());
	if (aggregated_output.size() == 0) {
		for (int i = 0; i < update_d.keepTokenListLength; ++i) {
			if (update_d.keep_token_list[i] == 1) {
				complete_input_tokens[input_tokens_count++] = 1;
			}
		}
	}
	else {
		for (int i = 0; i < update_d.keepTokenListLength; ++i) {
			if (update_d.keep_token_list[i] == 1) {
				complete_input_tokens[input_tokens_count++] = aggregated_output[i];
			}
		}
	}
	spdlog::info("input_tokens_count: {}, decodePrefillBorder: {}", input_tokens_count, update_d.decodePrefillBorder);
	spdlog::info("prefillTokensNum: {}", update_d.prefillTokensNum);
	
	for (int i = 0; i < update_d.prefillTokensNum; ++i) {
		complete_input_tokens[input_tokens_count++] = update_d.input_tokens[i];
	}
	spdlog::info("input_tokens_count: {}, total: {}", input_tokens_count, update_d.decodePrefillBorder + update_d.prefillTokensNum);

	for (int i = 0; i < nranks; ++i) {
		updateData[i].input_tokens = complete_input_tokens;
	}

	assert(updateData.size() == nranks);
	shared_state.op = WorkerOp::UPDATE;
	shared_state.updates_ptr = &updateData;
	// issue op command to worker
	global_sync->barrier();
	// wait for worker to complete this op
	global_sync->barrier();
	spdlog::info("All worker threads finished update");
#endif
}

void config(int nranks, std::vector<vortexConfigData>& config) {
	spdlog::info("Config pipeline");
#ifdef ENABLE_MPI
	assert(config.size() == 1);
	worker->config(&config.front());
#else
	assert(config.size() == nranks);
	shared_state.op = WorkerOp::CONFIG;
	shared_state.config_ptr = &config;
	// issue op command to worker
	global_sync->barrier();
	// wait for worker to complete this op
	global_sync->barrier();
	spdlog::info("All worker threads finished config");
#endif
}

void finalize() {
	spdlog::info("Finalize pipeline");
#ifdef ENABLE_MPI
	netmgr->finalize();
	worker.reset();
#else
	shared_state.op = WorkerOp::STOP;
	// issue op command to worker
	global_sync->barrier();
	// wait for worker to complete this op
	global_sync->barrier();
	workers.clear();
#endif
}


Worker::PipelineType Worker::PipeTy = Worker::PipelineType::PLLM;

void Worker::as_thread(int core) {
	thread = std::make_unique<std::thread>(&Worker::thread_entry, this);
	cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);      // Initialize to zero
    CPU_SET(core, &cpuSet);  // Set the desired CPU core

    // Get the native thread handle
    pthread_t nativeHandle = thread->native_handle();

    // Set the thread affinity
    int result = pthread_setaffinity_np(nativeHandle, sizeof(cpuSet), &cpuSet);
    if (result != 0) {
        std::cerr << "Error setting thread affinity: " << strerror(result) << std::endl;
    }
}

void Worker::init() {
	spdlog::info("Rank: {}", rank);
	// get device properties
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, rank));
	cudaUUID_t uuid = prop.uuid;
	std::cout << "UUID: ";
	for(int i = 0; i < 16; ++i) {
		std::cout << std::hex << (uuid.bytes[i] & 0xff) << std::dec;
		if(i < 15) std::cout << "-";
	}
	std::cout << std::endl;

	CUDA_CHECK(cudaSetDevice(rank));

	for(int peer = 0; peer < nranks; ++peer) {
		if(peer != rank) CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
	}

	spdlog::info("Creating pipeline, rank {}, core {}", rank, sched_getcpu());
	if (PipeTy == PipelineType::PLLM || PipeTy == PipelineType::PLLMOFFLOAD || PipeTy == PipelineType::NANOBATCH || PipeTy == PipelineType::KQVBIAS || PipeTy == PipelineType::NANOBATCH_KQVBIAS)
	{
		pipeline = std::make_unique<Pipeline>(input, rank, nranks, vnranks, 
											PipeTy == PipelineType::PLLMOFFLOAD, 
											(PipeTy == PipelineType::NANOBATCH || PipeTy == PipelineType::NANOBATCH_KQVBIAS), 
											(PipeTy == PipelineType::KQVBIAS || PipeTy == PipelineType::NANOBATCH_KQVBIAS));
	}
	else if (PipeTy == PipelineType::NONOVERLAP || PipeTy == PipelineType::NONOVERLAP_KQVBIAS)
		pipeline = std::make_unique<NonOverlapPipeline>(input, rank, nranks, vnranks, PipeTy == PipelineType::NONOVERLAP_KQVBIAS);
	else if (PipeTy == PipelineType::LOCAL || PipeTy == PipelineType::NANOBATCH_LOCAL)
		pipeline = std::make_unique<LocalPipeline>(input, rank, nranks, vnranks, PipeTy == PipelineType::NANOBATCH_LOCAL);
	else if (PipeTy == PipelineType::NON_OVERLAP_LOCAL)
		pipeline = std::make_unique<NonOverlapLocalPipeline>(input, rank, nranks, vnranks);
	else
		spdlog::error("Unknown pipeline type {}", static_cast<int>(PipeTy));
}

void Worker::run_pipeline() {
	*output = pipeline->run();
}

void Worker::thread_entry() {
#ifdef ENABLE_MPI
#else
	init();
	spdlog::info("Pipeline created");

	// this barrier marks the end of init
	global_sync->barrier();

	bool stop = false;
	while(!stop) {
		// this barrier is waiting for a separate invoke to run()
		global_sync->barrier();
		WorkerOp op = shared_state.op;
		switch(op) {
		case WorkerOp::STOP:
			stop = true;
			break;
		case WorkerOp::RUN:
			run_pipeline();
			break;
		case WorkerOp::UPDATE:
			run_update(&(shared_state.updates_ptr->at(rank)));
			break;
		case WorkerOp::CONFIG:
			run_config(&(shared_state.config_ptr->at(rank)));
			break;
		default:
			spdlog::error("Unknown WorkerOp {}", static_cast<int>(op));
			stop = true;
		}
		// this barrier marks the end of pipeline run
		global_sync->barrier();
	}
#endif
}
