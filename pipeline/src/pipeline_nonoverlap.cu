#include "gemmFactory.cuh"
#include "pipeline.h"
#include "spdlog/spdlog.h"

NonOverlapPipeline::NonOverlapPipeline(vortexInitData* input_data,
									   int nrank,
									   int nranks,
									   int vnranks)
	: PipelineBase(input_data, nrank, nranks, vnranks) {
	// sampled tokens 
	cudaMallocHost(&outputTokens, 4096*sizeof(int));
	init();
}

void NonOverlapPipeline::init() {
	stream_all = cudaStreamPerThread;
	spdlog::info("Init pipeline (non-overlap)");
#ifdef ENABLE_NETWORK
	spdlog::info("Init net");
	NetOpPrepare();
#endif
}

vortexOutputData NonOverlapPipeline::run() {
	constexpr bool enableGraph = false;
	spdlog::info("Start run");
	setWeight(0);

	if(!enableGraph)
		CUDA_CHECK(cudaEventRecord(events[EventManager::GEMM_TIMING_START], stream_all));
	if(enableGraph) cudaStreamBeginCapture(stream_all, cudaStreamCaptureModeGlobal);

	constexpr int totalIter = 10000;
	constexpr int kNetNblocks = 64;

	// TODO: setup phase
	// layerNormPipeStart(stream_gemm);
	// KQV_START->run(stream_gemm);
	// GEMV_START(stream_gemm);


	for (int iter = 1; iter <= RUN_LAYER; ++iter) {
		O->run();
		AG_O(stream_all, kNetNblocks, 1024, true);
		layerNormFFN.run();
		UG->run();
		activation.run();
		D->run();
		AR_D(stream_all, kNetNblocks, 1024, true);
		layerNormAttention.run();

		if (iter == RUN_LAYER) break;
		setWeight(iter%5);
		KQV->run();
		roPEAppend.run();
		GEMV.run();
		// O_TR.skip();
		AG_GEMV(stream_all, kNetNblocks, 1024, true);

		if (update_data.prefillNum > 0)
			prefill.run();
	}
	// cudaMemcpyAsync(
	// 	outputTokens, input_data->tmp_buffer, 2048 * sizeof(int), cudaMemcpyDeviceToHost, stream_all);
	// // End capture
	cudaGraph_t graph;
	if (enableGraph) {
		cudaStreamEndCapture(stream_all, &graph);
		if (graph == NULL) {
			spdlog::error("Failed to create graph");
			exit(1);
		}
		spdlog::info("Graph created");
		cudaGraphExec_t instance;
		cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
		spdlog::info("Graph instantiated");
		CUDA_CHECK(cudaEventRecord(events[EventManager::GEMM_TIMING_START], stream_all));
		for(int i = 0; i < 10; i ++)
			cudaGraphLaunch(instance, stream_all);
		spdlog::info("Graph launched");
	}

	// Record an event when the GEMMs are complete
	CUDA_CHECK(cudaEventRecord(events[EventManager::GEMM_TIMING_END], stream_all));

	// Wait for work on the device to complete.
	CUDA_CHECK(cudaEventSynchronize(events[EventManager::GEMM_TIMING_END]));

	// Measure elapsed runtime
	float runtime_ms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&runtime_ms,
									events[EventManager::GEMM_TIMING_START],
									events[EventManager::GEMM_TIMING_END]));
	// Compute average runtime and GFLOPs.
	runtime_ms = double(runtime_ms);
	double gflops = totalCompute() / runtime_ms / 1e6;
	double bandwidth = sizeof(__half) * (96731136 + 1.25 / 80 * 160 * 1024 * 1024 * 1024 / 2) /
					   (runtime_ms / 1000) / (1 << 30);
	spdlog::info("Total running cost (ms) of one microbatch is {}", runtime_ms);

	vortexOutputData d;
	d.global_batch_size = 0;
	d.partial_num_1 = 0;
	d.partial_num_2 = 0;
	
	return d;
}

void NonOverlapPipeline::update(vortexUpdateData* update_data) {
	this->update_data = *update_data;
	GEMVOpUpdate();
	// log first batch size
	spdlog::info(
		"prefill: {}, decode: {}", update_data->prefillNum, update_data->decodePrefillBorder);
	spdlog::info("Batch size: {}, {}, {}, {}",
				 update_data->gemv_batch_size[0],
				 update_data->gemv_batch_size[1],
				 update_data->gemv_batch_size[2],
				 update_data->gemv_batch_size[3]);
}

void NonOverlapPipeline::config(vortexConfigData* config_data) {
	spdlog::info("Config non-overlap pipeline");
    for (int i = 0; i < gemmNum; ++i) {
        gemms[i] = generateGEMM(gemmConfig[i]);
    }
	this->config_data = *config_data;
    int globalbatch = config_data->global_batch_size;
    O ->set_shape(globalbatch, MODEL_HIDDEN_DIM_PERGPU, MODEL_HIDDEN_DIM);
    UG ->set_shape(globalbatch, MODEL_FF_DIM_GPU + MODEL_FF_DIM_GPU, MODEL_HIDDEN_DIM);
    D ->set_shape(globalbatch, MODEL_HIDDEN_DIM, MODEL_FF_DIM_GPU);
	KQV->set_shape(globalbatch,
				   (MODEL_KV_HEADS + MODEL_KV_HEADS + MODEL_QO_HEADS) * MODEL_HEAD_DIM,
				   MODEL_HIDDEN_DIM);
	KQV_START->set_shape(globalbatch,
						 (MODEL_KV_HEADS + MODEL_KV_HEADS + MODEL_QO_HEADS) * MODEL_HEAD_DIM,
						 MODEL_HIDDEN_DIM);
	spdlog::info("Init schedule");
	ScheduleInit();
	spdlog::info("Init gemm");
	GEMMOpInit();
	spdlog::info("Init other");
	OtherOpInit();
}

void NonOverlapPipeline::setWeight(int layer) {
    AllocationManager<cutlass::half_t> weightM(ptr_cast<cutlass::half_t>(input_data->tmp_buffer), input_data->weight_size);
	const auto& W_O = weightM.allocSpan(O->kn());
	O->set_weight(W_O.data());
	const auto& W_UG = weightM.allocSpan(UG->kn());
	UG->set_weight(W_UG.data());
	const auto& W_D = weightM.allocSpan(UG->kn());
	D->set_weight(W_D.data());
	const auto& W_KQV = weightM.allocSpan(KQV->kn());
	KQV->set_weight(W_KQV.data());
	const auto& W_LN_Attn = weightM.allocSpan(KQV->K);
	layerNormAttention.setWeight(W_LN_Attn.data());
	const auto& W_LN_FFN = weightM.allocSpan(UG->K);
	layerNormFFN.setWeight(W_LN_FFN.data());
    GEMV.setKVData(input_data->kv_data[layer]);
	prefill.setKVData(input_data->kv_data[layer]);
}

void NonOverlapPipeline::ScheduleInit() {
	auto getMajor = [this](GEMM_NAME name, int x) {return getMajorType(config_data.gemm_op_tag[static_cast<int>(name)], x);};
	auto getDim = [this, getMajor](GEMM_NAME name, int x) {return static_cast<PllmDimension>(getMajor(name, x));};
	// (prev)AG_GEMV -> O (col major)
	AG_GEMV.init(comm, connections, rank, nranks, tmpBufferM.allocTensor(O->M, O->K, getMajor(GEMM_NAME::O, 0)));
	O->setA(AG_GEMV.getOutput());
	// O (col major) -> AG_O
	AG_O.init(comm, connections, rank, nranks, tmpBufferM.allocTensor(UG->M, UG->K, getMajor(GEMM_NAME::O, 2)));
	O->setD(AG_O.getInput().getSubTensor(rank, vnranks, getDim(GEMM_NAME::O, 2)));
	// AG_O -> LN_FFN
	layerNormFFN.setInput(AG_O.getOutput()).setOutput(tmpBufferM.allocTensor(UG->M, UG->K, getMajor(GEMM_NAME::UG, 0)));
	// LN_FFN -> UG
	UG->setA(layerNormFFN.getOutput()).setOutput(tmpBufferM.allocTensor(UG->M, UG->K, getMajor(GEMM_NAME::UG, 2)));
	// UG -> activation
	activation.setInput(UG->getD()).setOutput(tmpBufferM.allocTensor(D->M, D->K, getMajor(GEMM_NAME::D, 0)));
	// activation -> D
	D->setA(activation.getOutput());
	// D -> AR_D
	AR_D.init(comm, connections, rank, nranks, tmpBufferM.allocTensor(D->M, D->N, getMajor(GEMM_NAME::D, 2)), tmpBufferM.allocTensor(D->M, D->N, getMajor(GEMM_NAME::D, 2)));
	D->setOutput(AR_D.getInput());
	// AR_D -> LN_Attention
	layerNormAttention.setInput(AR_D.getOutput()).setOutput(tmpBufferM.allocTensor(D->M, D->N, getMajor(GEMM_NAME::O, 2)));
	// LN_Attention -> O residual connection
	O->setC(layerNormAttention.getOutput().getSubTensor(rank, vnranks, getDim(GEMM_NAME::D, 2)));
	KQV->setA(layerNormAttention.getOutput()).setOutput(tmpBufferM.allocTensor(KQV->M, KQV->N, getMajor(GEMM_NAME::KQV, 2)));
	gemvInput = KQV->getD();
	gemvOutput = tmpBufferM.allocTensor(GemvWrapper::getOutputSize(MAX_BATCH_SIZE), KQV->N, getMajor(GEMM_NAME::O, 2));
	// TODO: ropeAppend is not implemented yet
	O_TR.setInput(gemvOutput).setOutput(AG_GEMV.getInput().getSubTensor(rank, vnranks, static_cast<PllmDimension>(!((int)getDim(GEMM_NAME::O, 2)))));
}

void NonOverlapPipeline::GEMMOpInit() {
	cutlass::half_t beta(1);
	setWeight(0);
	O->init(beta);
	UG->init();
	D->init();
	KQV->init();
	for (int i = 0; i < gemmNum; ++i) {
		gemms[i]->setStream(stream_all);
	}

	layerNormAttention.setStream(stream_all);
	layerNormFFN.setStream(stream_all);
	GEMV.setStream(stream_all);
	prefill.setStream(stream_all);
	O_TR.setStream(stream_all);
	activation.setStream(stream_all);
	roPEAppend.setStream(stream_all);
}

void NonOverlapPipeline::GEMVOpUpdate() {
	auto getMajor = [this](GEMM_NAME name, int x) {return static_cast<PllmDimension>((config_data.gemm_op_tag[static_cast<int>(name)], x));};
	spdlog::info("Update GEMV");

	uint32_t decode_batch_size = update_data.decodePrefillBorder;
	const auto& [gemvDec_input, gemvPrefill_input] =
		tensor_cast<cutlass::half_t, half>(gemvInput).splitTensor(getMajor(GEMM_NAME::O, 0),
				  GemvWrapper::getInputSize(decode_batch_size),
				  GemvWrapper::getInputSize(MAX_BATCH_SIZE - decode_batch_size));
	const auto& [gemvDec_output, gemvPrefill_output] =
		tensor_cast<cutlass::half_t, half>(gemvOutput).splitTensor(getMajor(GEMM_NAME::O, 0),
				  GemvWrapper::getOutputSize(decode_batch_size),
				  GemvWrapper::getOutputSize(MAX_BATCH_SIZE - decode_batch_size));



	uint32_t arr[] = {update_data.decodePrefillBorder, update_data.prefillNum};
	std::span<uint32_t, 2> batch_sizes(arr, 2);
	auto total_batch_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0);
	assert (total_batch_size == update_data.decodePrefillBorder + update_data.prefillNum);
	const auto& kv_indptr_split = pllmTensor{update_data.kv_indptr, total_batch_size + 1}.splitTensor(
		PllmDimension::ROW, batch_sizes,/*overlap suffix*/ 1U);
	const auto& kv_last_page_len_split = pllmTensor{update_data.kv_last_page_len, total_batch_size}.
		splitTensor(PllmDimension::ROW, batch_sizes);
	const auto& input_ptr_split = pllmTensor{update_data.input_indptr, total_batch_size + 1}.splitTensor(
		PllmDimension::ROW, batch_sizes,/*overlap suffix*/ 1U);
	
	GEMV.init(decode_batch_size,
			  update_data.gemv_num_blocks[0],
			  kv_indptr_split[0],
			  update_data.kv_indices,
			  kv_last_page_len_split[0],
			  gemvDec_input,
			  gemvDec_output);
	prefill.init(update_data.prefillNum,
				 108,
				 input_ptr_split[1],
				 kv_indptr_split[1],
				 update_data.kv_indices,
				 kv_last_page_len_split[1],
				 tensor_cast<cutlass::half_t, half>(gemvInput),
				 tensor_cast<cutlass::half_t, half>(gemvOutput));
}

double NonOverlapPipeline::totalCompute() {
	double total = 0;
	for(auto gemm : gemms)
		total += gemm->totalCompute();
	return total;
}