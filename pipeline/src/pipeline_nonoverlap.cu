#include "gemmFactory.cuh"
#include "pipeline.h"
#include "spdlog/spdlog.h"

NonOverlapPipeline::NonOverlapPipeline(vortexInitData* input_data,
									   int nrank,
									   int nranks,
									   int vnranks,
									   bool local)
	: PipelineBase(input_data, nrank, nranks, vnranks),
	  local(local)
	{
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

	constexpr int kNetNblocks = 64;

	// TODO: setup phase
	// layerNormPipeStart(stream_gemm);
	// KQV_START->run(stream_gemm);
	// GEMV_START(stream_gemm);
	embedding.run().log(private_logger);

	for (int iter = 0; iter < ModelConfig.run_layer; ++iter) {
		setWeight(iter);
		layerNormAttention.run().log(private_logger);
		KQV->run().log(private_logger);
		roPEAppend.run().log(private_logger);
		GEMV.run().log(private_logger);
		
		if (update_data.prefillNum > 0)
			prefill.run().log(private_logger);
		AG_GEMV.setColumnwise().configRun(kNetNblocks, 1024, true).run().log(private_logger);
		splitTensor.run().log(private_logger);
		O->run().log(private_logger);
		AG_O.setColumnwise().configRun(kNetNblocks, 1024, true).run().log(private_logger);
		layerNormFFN.run().log(private_logger);
		// UG->run().log(private_logger);
		// activation.run().log(private_logger);
		dual.run().log(private_logger);
		D->run().log(private_logger);
		AR_D.configRun(kNetNblocks, 1024, true).run().log(private_logger);
	}
	
	layerNormModel.run().log(private_logger);
	keepToken.run().log(private_logger);
	
	LOGITS->run().log(private_logger);
	maxSampler.run().log(private_logger);
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
	spdlog::info("Total running cost (ms) of one microbatch is {}", runtime_ms);

	// Copy output data back to host
	spdlog::info("sampled_token_array {}, maxSampler.d_argMax.ptr {}, sampled_tokens {}", (size_t) output_data.sampled_token_array, (size_t)maxSampler.d_argMax.ptr, output_data.sampled_tokens);
	CUDA_CHECK(cudaMemcpy(output_data.sampled_token_array, maxSampler.d_argMax.ptr, output_data.sampled_tokens* sizeof(int), cudaMemcpyDeviceToHost));
	
	return output_data;
}

void NonOverlapPipeline::update(vortexUpdateData* update_data_) {
	this->update_data = *update_data_;
	embedding.setInput(pllmTensor<int>(update_data.input_tokens, config_data.global_batch_size, 1, PllmLayout::ROW_MAJOR));
	int req_num = update_data.decodePrefillBorder + update_data.prefillNum;
	keepToken.update(req_num, update_data.input_indptr);
	keepToken.output.dim1 = req_num;
	auto logits_a = keepToken.output.getSubTensor(rank, nranks, PllmDimension::ROW);
	int sample_batch_size = logits_a.dim1;
	if (sample_batch_size == 0) {
		sample_batch_size = 1;
	}
	

	LOGITS->set_shape((sample_batch_size+127)/128*128, ModelConfig.vocab_size, ModelConfig.model_hidden_dim);
	logits_a.dim1 = (sample_batch_size+127)/128*128;
	LOGITS->setA(tensor_cast<half, cutlass::half_t> (logits_a));
	LOGITS->init();
	maxSampler.set_batch_size(sample_batch_size);
	output_data.sampled_tokens = sample_batch_size;
	
	GEMVOpUpdate();
	// log first batch size
	spdlog::info(
		"prefill: {}, decode: {}", update_data.prefillNum, update_data.decodePrefillBorder);
	spdlog::info("Batch size: {}, {}, {}, {}",
				 update_data.gemv_batch_size[0],
				 update_data.gemv_batch_size[1],
				 update_data.gemv_batch_size[2],
				 update_data.gemv_batch_size[3]);
}

void NonOverlapPipeline::config(vortexConfigData* config_data) {
	spdlog::info("Config non-overlap pipeline");
	this->config_data = *config_data;
    for (int i = 0; i < gemmNum; ++i) {
        gemms[i] = generateGEMM(this->config_data.gemm_op_tag[i]);
		spdlog::info("GEMM {} created, tag: {}", i, this->config_data.gemm_op_tag[i]);
    }
    int globalbatch = config_data->global_batch_size;
    O->set_shape(globalbatch, ModelConfig.model_hidden_dim_pergpu, ModelConfig.model_hidden_dim);
    // UG ->set_shape(globalbatch, ModelConfig.model_ff_dim_gpu + ModelConfig.model_ff_dim_gpu, ModelConfig.model_hidden_dim);
    
	D->set_shape(globalbatch, ModelConfig.model_hidden_dim, ModelConfig.model_ff_dim_gpu);
	KQV->set_shape(globalbatch,
				   (ModelConfig.model_kv_heads_gpu + ModelConfig.model_kv_heads_gpu + ModelConfig.model_qo_heads_gpu) * ModelConfig.model_head_dim,
				   ModelConfig.model_hidden_dim);
	LOGITS->set_shape(globalbatch, ModelConfig.vocab_size, ModelConfig.model_hidden_dim);
	dual.set_shape(globalbatch, ModelConfig.model_ff_dim_gpu, ModelConfig.model_hidden_dim);

	spdlog::info("Init schedule");
	ScheduleInit();
	spdlog::info("Init gemm");
	GEMMOpInit();
	spdlog::info("Init gemv");
	GEMVOpInit();
	spdlog::info("Init other");
	OtherOpInit();
	setName();
	// init the output 
	output_data = vortexOutputData();
	output_data.sampled_token_array = new int[config_data->global_batch_size];
	output_data.global_batch_size = config_data->global_batch_size;
}
void NonOverlapPipeline::setName(){
		SET_NAME_PTR(KQV);
		SET_NAME_PTR(LOGITS);
		SET_NAME_PTR(O);
		SET_NAME_REF(dual);
		SET_NAME_PTR(D);

		SET_NAME_REF(embedding);
		SET_NAME_REF(AG_GEMV);
		SET_NAME_REF(AG_O);
		SET_NAME_REF(AR_D);
		SET_NAME_REF(layerNormAttention);
		SET_NAME_REF(layerNormFFN);
		SET_NAME_REF(layerNormModel);
		SET_NAME_REF(GEMV);
		SET_NAME_REF(prefill);
		SET_NAME_REF(roPEAppend);
		SET_NAME_REF(maxSampler);
		SET_NAME_REF(keepToken);
		SET_NAME_REF(splitTensor);
}
void NonOverlapPipeline::setWeight(int layer) {

	bool success = true;
	success &= O->set_weight(input_data->weight.layer_weight[layer].W_O1);

	success &= D->set_weight(input_data->weight.layer_weight[layer].W_D);

	success &= KQV->set_weight(input_data->weight.layer_weight[layer].W_KQV);

	success &= layerNormAttention.setWeight(input_data->weight.layer_weight[layer].W_LN_Attention);
	
	success &= layerNormFFN.setWeight(input_data->weight.layer_weight[layer].W_LN_FFN);
	
	success &= dual.set_weight(input_data->weight.layer_weight[layer].W_G, input_data->weight.layer_weight[layer].W_U);

	if (!success) {
		spdlog::error("Failed to set weight for layer {}", layer);
	}

	GEMV.setKVData(input_data->kv_data[layer]);
	prefill.setKVData(input_data->kv_data[layer]);
	roPEAppend.setKVData(input_data->kv_data[layer]);
	
    // AllocationManager<cutlass::half_t> weightM((cutlass::half_t*)weight_buffer, ((size_t)4)*1024*1024*1024);

	// const auto& W_embedding = weightM.allocSpan((size_t)ModelConfig.vocab_size * ModelConfig.model_hidden_dim);
	// spdlog::info("embedding location {}", (size_t)W_embedding.data());
	// auto vW_embedding = vortexWeight{(half*)W_embedding.data(), ModelConfig.vocab_size, ModelConfig.model_hidden_dim};
	// embedding.setWeight(vW_embedding);


	// const auto& W_KQV = weightM.allocSpan(KQV->kn());
	// KQV->set_weight(W_KQV.data());
	// const auto& W_O = weightM.allocSpan(O->kn());
	// O->set_weight(W_O.data());
	// // const auto& W_UG = weightM.allocSpan(dual.Kn());
	// // UG->set_weight(W_UG.data());
	// auto W_U = weightM.allocSpan(dual.K*dual.N);
	// auto W_G = weightM.allocSpan(dual.K*dual.N);
	// auto vW_U = vortexWeight{(half*)W_U.data(), dual.N, dual.K};
	// auto vW_G = vortexWeight{(half*)W_G.data(), dual.N, dual.K};
	// dual.set_weight(vW_U, vW_G);
	// const auto& W_D = weightM.allocSpan(dual.K*dual.N);
	// D->set_weight(W_D.data());

	// const auto& W_LN_Attn = weightM.allocSpan(KQV->K);
	// layerNormAttention.setWeight(W_LN_Attn.data());
	// // const auto& W_LN_FFN = weightM.allocSpan(dual.K);
	// const auto& W_LN_FFN = weightM.allocSpan(dual.K);
	// layerNormFFN.setWeight(W_LN_FFN.data());
	// const auto& W_LN_Model = weightM.allocSpan(D->N);
	// layerNormModel.setWeight(W_LN_Model.data());
    // GEMV.setKVData(input_data->kv_data[layer]);
	// prefill.setKVData(input_data->kv_data[layer]);
	// roPEAppend.setKVData(input_data->kv_data[layer]);
	
	// const auto& W_LOGITS = weightM.allocSpan(LOGITS->kn());
	// LOGITS->set_weight(W_LOGITS.data());




	// spdlog::info("allocated {} halfs", weightM.getAllocation());

}

void NonOverlapPipeline::ScheduleInit() {
	auto getMajor = [this](GEMM_NAME name, int x) {return getMajorType(config_data.gemm_op_tag[static_cast<int>(name)], x);};
	
	// (prev)AG_GEMV -> O (col major)
	const int qdim = ModelConfig.model_head_dim *  ModelConfig.model_qo_heads_gpu;
	gemvInput = tmpBufferM.allocTensor(config_data.global_batch_size, qdim, PllmLayout::ROW_MAJOR);

	gemvOutput = tmpBufferM.allocTensor(config_data.global_batch_size, qdim, getMajor(GEMM_NAME::O, 2));
	AG_GEMV.init(comm, connections, rank, nranks, gemvOutput, tmpBufferM.allocTensor(O->M, O->K, getMajor(GEMM_NAME::O, 0)));
	O->setA(AG_GEMV.getOutput());
	// O (col major) -> AG_O
	
	const auto& O_output = tmpBufferM.allocTensor(O->M, O->N, PllmLayout::ROW_MAJOR);
	O->setD(O_output);
	AG_O.init(comm, connections, rank, nranks, O_output, tmpBufferM.allocTensor(dual.M, dual.K, getMajor(GEMM_NAME::O, 2)));
	// AG_O -> LN_FFN
	layerNormFFN.setInput(AG_O.getOutput()).setOutput(tmpBufferM.allocTensor(dual.M, dual.K, PllmLayout::ROW_MAJOR));
	// LN_FFN -> UG
	// UG->setA(layerNormFFN.getOutput()).setOutput(tmpBufferM.allocTensor(dual.M, dual.K, getMajor(GEMM_NAME::UG, 2)));
	
	const auto& Dual_output_0 = tmpBufferM.allocTensor(dual.M, dual.N, PllmLayout::ROW_MAJOR);
	const auto& Dual_output_1 = tmpBufferM.allocTensor(dual.M, dual.N, PllmLayout::ROW_MAJOR);
	const auto& activation_output = tmpBufferM.allocTensor(D->M, D->K, PllmLayout::ROW_MAJOR);
	dual.setA(layerNormFFN.getOutput());
	dual.setC(tmpBufferM.allocTensor(dual.M, dual.N, PllmLayout::ROW_MAJOR));
	dual.setD(Dual_output_0, Dual_output_1, activation_output);
	// UG -> activation
	// activation.setInput(UG->getD()).setOutput(tmpBufferM.allocTensor(D->M, D->K, getMajor(GEMM_NAME::D, 0)));
	// activation -> D
	D->setA(activation_output);
	D->setC(AG_O.getOutput());
	// D -> AR_D
	AR_D.init(comm, connections, rank, nranks, tmpBufferM.allocTensor(D->M, D->N, getMajor(GEMM_NAME::D, 2)), tmpBufferM.allocTensor(D->M, D->N, getMajor(GEMM_NAME::D, 2)));
	D->setOutput(AR_D.getInput());
	// AR_D -> LN_Attention
	layerNormAttention.setInput(AR_D.getOutput()).setOutput(tmpBufferM.allocTensor(D->M, D->N, getMajor(GEMM_NAME::O, 2)));
	// LN_Attention -> O residual connection
	splitTensor.init(tensor_cast<cutlass::half_t, half>(AR_D.getOutput()), tensor_cast<cutlass::half_t, half>(tmpBufferM.allocTensor(O->M, O->N, PllmLayout::ROW_MAJOR)), nranks, rank);
	O->setC(tensor_cast<half, cutlass::half_t>(splitTensor.output));
	KQV->setA(layerNormAttention.getOutput()).setOutput(tmpBufferM.allocTensor(KQV->M, KQV->N, getMajor(GEMM_NAME::KQV, 2)));
	

	

	const auto& layerNormModel_output = tmpBufferM.allocTensor(config_data.global_batch_size, D->N, PllmLayout::ROW_MAJOR);
	layerNormModel.setInput(AR_D.getOutput()).setOutput(layerNormModel_output);

	const auto& keepTokenOutput = tmpBufferM.allocTensor(config_data.global_batch_size, D->N, PllmLayout::ROW_MAJOR);
	keepToken.setInput(tensor_cast<cutlass::half_t, half>(layerNormModel_output)).setOutput(tensor_cast<cutlass::half_t, half>(keepTokenOutput));

	const auto& LOGITS_output = tmpBufferM.allocTensor(config_data.global_batch_size, LOGITS->N, PllmLayout::ROW_MAJOR);
	LOGITS->setA(keepTokenOutput);
	LOGITS->setOutput(LOGITS_output);

	int* sample_output_alloc = (int*)tmpBufferM.alloc(config_data.global_batch_size * sizeof(int) / sizeof (half));
	pllmTensor<int> sample_output = {sample_output_alloc, config_data.global_batch_size, 1, PllmLayout::ROW_MAJOR};
	const auto& maxSampler_maxVals = tmpBufferM.allocTensor(config_data.global_batch_size, 1, PllmLayout::ROW_MAJOR);
	maxSampler.init(tensor_cast<cutlass::half_t, half>(LOGITS_output), 
					tensor_cast<cutlass::half_t, half>(maxSampler_maxVals), sample_output);

	embedding.setOutput(tensor_cast<cutlass::half_t, half>(AR_D.getOutput()));
	
	spdlog::info("allocated {} halfs", tmpBufferM.getAllocation());
}

void NonOverlapPipeline::GEMVOpUpdate() {
	auto getMajor = [this](GEMM_NAME name, int x) {return static_cast<PllmDimension>((config_data.gemm_op_tag[static_cast<int>(name)], x));};
	spdlog::info("Update GEMV");

	uint32_t decode_batch_size = update_data.decodePrefillBorder;
	const auto& [gemvDec_input, gemvPrefill_input] =
		tensor_cast<cutlass::half_t, half>(gemvInput).splitTensor(getMajor(GEMM_NAME::O, 0),
				  decode_batch_size,
				  ModelConfig.max_batch_size - decode_batch_size);
	const auto& [gemvDec_output, gemvPrefill_output] =
		tensor_cast<cutlass::half_t, half>(gemvOutput).splitTensor(getMajor(GEMM_NAME::O, 0),
				  decode_batch_size,
				  ModelConfig.max_batch_size - decode_batch_size);



	uint32_t arr[] = {uint32_t(update_data.decodePrefillBorder), uint32_t(update_data.prefillNum)};
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
			  input_ptr_split[0],
			  kv_indptr_split[0],
			  update_data.kv_indices,
			  kv_last_page_len_split[0],
			  tensor_cast<cutlass::half_t, half>(gemvInput),
			  tensor_cast<cutlass::half_t, half>(gemvOutput));
	log_tensor(spdlog::default_logger(), "GEMV kv_indptr_split", pllmTensor{update_data.kv_indptr, total_batch_size + 1}, 1, total_batch_size + 1);
	log_tensor(spdlog::default_logger(), "GEMV kv_last_page_len_split", pllmTensor{update_data.kv_last_page_len, total_batch_size}, 1, total_batch_size);
	log_tensor(spdlog::default_logger(), "GEMV input_ptr_split", pllmTensor{update_data.input_indptr, total_batch_size + 1}, 1, total_batch_size + 1);

	prefill.init(update_data.prefillNum,
				 108,
				 input_ptr_split[1],
				 kv_indptr_split[1],
				 update_data.kv_indices,
				 kv_last_page_len_split[1],
				 tensor_cast<cutlass::half_t, half>(gemvInput),
				 tensor_cast<cutlass::half_t, half>(gemvOutput));

	roPEAppend.update(update_data.decodePrefillBorder+update_data.prefillTokensNum,
						tensor_cast<cutlass::half_t, half>(KQV->getD()),
						tensor_cast<cutlass::half_t, half>(gemvInput), 
						pllmTensor<int>(update_data.rev_input_indptr, config_data.global_batch_size),
						pllmTensor<int>(update_data.per_token_offset, config_data.global_batch_size),
						pllmTensor<int>(update_data.kv_indices, ModelConfig.max_page_num),
						pllmTensor<int>(update_data.kv_indptr, config_data.global_batch_size + 1),
						pllmTensor<int>(update_data.kv_last_page_len, config_data.global_batch_size));
}

void NonOverlapPipeline::GEMMOpInit() {
	cutlass::half_t beta(1);
	setWeight(0);
	O->init(beta);
	// UG->init();
	dual.init();
	D->init(0.125);
	KQV->init();
	for (int i = 0; i < gemmNum; ++i) {
		gemms[i]->setStream(stream_all);
	}
	LOGITS->set_weight(input_data->weight.lm_head); // important

	LOGITS->init();

	LOGITS->set_weight(input_data->weight.lm_head);

	dual.setStream(stream_all);
}

void NonOverlapPipeline::GEMVOpInit(){
	GEMV.setStream(stream_all);
	prefill.setStream(stream_all);
}


void NonOverlapPipeline::OtherOpInit() {
	embedding.setStream(stream_all);
	AG_GEMV.setStream(stream_all);
	AG_O.setStream(stream_all);
	AR_D.setStream(stream_all);
	layerNormAttention.setStream(stream_all);
	layerNormFFN.setStream(stream_all);
	layerNormModel.setStream(stream_all);
	splitTensor.setStream(stream_all);

	roPEAppend.setStream(stream_all);
	maxSampler.setStream(stream_all);
	keepToken.setStream(stream_all);	

	embedding.setWeight(input_data->weight.embedding);
	layerNormModel.setWeight(input_data->weight.model_layernorm);
}

double NonOverlapPipeline::totalCompute() {
	double total = 0;
	for(auto gemm : gemms)
		total += gemm->totalCompute();
	return total;
}