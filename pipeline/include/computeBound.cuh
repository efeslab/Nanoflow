#pragma once
#include <cstdint>
#include <iostream>
#include <vector>


#include "gemmShape.cuh"
#include "pipeline.h"
#include "networkManager.cuh"
#include "tensorManager.cuh"


#include "config.h"
#include "vortexData.cuh"

extern std::vector<vortexOutputData> outputs;
extern std::vector<int> aggregated_output;


class Worker {
public:
	enum class PipelineType
	{
		PLLM,
		NONOVERLAP,
		NANOBATCH,
		PLLMOFFLOAD,
		KQVBIAS,
		LOCAL,
		NON_OVERLAP_LOCAL,
		NANOBATCH_LOCAL,
	};
	static PipelineType PipeTy;

private:
	int rank;
	int nranks;
	int vnranks;
	vortexInitData* input;
	vortexOutputData* output;
	std::unique_ptr<PipelineBase> pipeline;
	std::unique_ptr<std::thread> thread;
	void thread_entry();

public:
	Worker(int rank,
				   int nranks,
				   int vnranks,
				   vortexInitData* input,
				   vortexOutputData* output)
		: rank(rank)
		, nranks(nranks)
		, vnranks(vnranks)
		, input(input)
		, output(output) { }
	void init();
	void as_thread(int core);
	void join() { if (thread) thread->join(); }
	void run_pipeline();
	void run_update(vortexUpdateData* update_data) {
		vortexUpdateData& gpu_update_data = TensorManager::getInstance().update_data_to_gpu(*update_data, rank);
		pipeline->update(&gpu_update_data);
	}
	void run_config(vortexConfigData* config_data) {
		pipeline->config(config_data);
	}
	vortexOutputData* getOutput() {
		return output;
	}
	~Worker() { if (thread) thread->join(); }
};


void run();
// vnranks >= nranks. virtualized ranks will not touch any GPU resources, but will take use random data to participate in collective communication buffers.
void init(int nranks, int vnranks, std::vector<vortexInitData>& input, std::vector<vortexOutputData>& output, Worker::PipelineType pipeTy = Worker::PipelineType::PLLM);
inline void init(int nranks, std::vector<vortexInitData>& input, std::vector<vortexOutputData>& output, Worker::PipelineType pipeTy = Worker::PipelineType::PLLM) {
	init(nranks, nranks, input, output, pipeTy);
}

void update(int nranks, std::vector<vortexUpdateData>& update);
void finalize();
void run_async();
void run_async_wait();
void config(int nranks, std::vector<vortexConfigData>& config);