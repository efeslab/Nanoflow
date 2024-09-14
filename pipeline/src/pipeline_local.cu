#include "gemmFactory.cuh"
#include "pipeline.h"
#include "spdlog/spdlog.h"

LocalPipeline::LocalPipeline(vortexInitData* input_data,
									   int nrank,
									   int nranks,
									   int vnranks,
									   bool nanobatch_only)
	: PipelineBase(input_data, nrank, nranks, vnranks)
	, nanobatch_only(nanobatch_only)
{
	init();
}

void LocalPipeline::init() {
	cudaStreamCreate(&stream_gemm);
	cudaStreamCreate(&stream_gemv);
	cudaStreamCreate(&stream_other);
	CUDA_CHECK(cudaMallocHost(&host_rev_input_indptr, ModelConfig.max_batch_size*sizeof(int)));
	spdlog::info("Init pipeline (local)");
}

void LocalPipeline::setName() {
	SET_NAME_PTR(O1);
	SET_NAME_PTR(O2);
	SET_NAME_PTR(D1);
	SET_NAME_PTR(D2);
	SET_NAME_PTR(KQV1);
	SET_NAME_PTR(KQV2);
	SET_NAME_PTR(LOGITS);

	SET_NAME_REF(dual1);
	SET_NAME_REF(dual2);

	SET_NAME_REF(genEmbedding1);
	SET_NAME_REF(genEmbedding2);

	SET_NAME_REF(GEMV1);
	SET_NAME_REF(GEMV2);
	SET_NAME_REF(GEMV3);
	SET_NAME_REF(GEMV4);
	SET_NAME_REF(prefill1);
	SET_NAME_REF(prefill2);

	SET_NAME_REF(layerNormAttention1);
	SET_NAME_REF(layerNormAttention2);
	SET_NAME_REF(layerNormFFN1);
	SET_NAME_REF(layerNormFFN2);
	SET_NAME_REF(layerNormModel);


	SET_NAME_REF(roPEAppends[0]);
	SET_NAME_REF(roPEAppends[1]);


	SET_NAME_REF(maxSampler);

	SET_NAME_REF(keepToken);
}

void LocalPipeline::assignSameStream(){
	for (auto gemv : gemvs) {
		gemv->setStream(stream_gemm);
	}

	prefill1.setStream(stream_gemm);
	prefill2.setStream(stream_gemm);
}


void LocalPipeline::setWeight(int layer) {
	bool success = true;
	success &= O1->set_weight(input_data->weight.layer_weight[layer].W_O1);
	success &= O2->set_weight(input_data->weight.layer_weight[layer].W_O2);


	success &= D1->set_weight(input_data->weight.layer_weight[layer].W_D);
	success &= D2->set_weight(input_data->weight.layer_weight[layer].W_D);

	success &= KQV1->set_weight(input_data->weight.layer_weight[layer].W_KQV);
	success &= KQV2->set_weight(input_data->weight.layer_weight[layer].W_KQV);

	success &= layerNormAttention1.setWeight(input_data->weight.layer_weight[layer].W_LN_Attention);
	success &= layerNormAttention2.setWeight(input_data->weight.layer_weight[layer].W_LN_Attention);
	
	success &= layerNormFFN1.setWeight(input_data->weight.layer_weight[layer].W_LN_FFN);
	success &= layerNormFFN2.setWeight(input_data->weight.layer_weight[layer].W_LN_FFN);
	
	success &= dual1.set_weight(input_data->weight.layer_weight[layer].W_G, input_data->weight.layer_weight[layer].W_U);
	success &= dual2.set_weight(input_data->weight.layer_weight[layer].W_G, input_data->weight.layer_weight[layer].W_U);

	if (!success) {
		spdlog::error("Failed to set weight for layer {}", layer);
	}

	for (auto gemv : gemvs) {
		gemv->setKVData(input_data->kv_data[layer]);
	}
	prefill1.setKVData(input_data->kv_data[layer]);
	prefill2.setKVData(input_data->kv_data[layer]);
	roPEAppends[0].setKVData(input_data->kv_data[layer]);
	roPEAppends[1].setKVData(input_data->kv_data[layer]);
}

void LocalPipeline::setWeight_group_O(int layer) {
	bool success = true;
	success &= O1->set_weight(input_data->weight.layer_weight[layer].W_O1);
	success &= O2->set_weight(input_data->weight.layer_weight[layer].W_O2);
	success &= layerNormFFN1.setWeight(input_data->weight.layer_weight[layer].W_LN_FFN);
	success &= layerNormFFN2.setWeight(input_data->weight.layer_weight[layer].W_LN_FFN);
	success &= dual1.set_weight(input_data->weight.layer_weight[layer].W_G, input_data->weight.layer_weight[layer].W_U);
	success &= dual2.set_weight(input_data->weight.layer_weight[layer].W_G, input_data->weight.layer_weight[layer].W_U);
	success &= D1->set_weight(input_data->weight.layer_weight[layer].W_D);
	success &= D2->set_weight(input_data->weight.layer_weight[layer].W_D);
	if (!success) {
		spdlog::error("Failed to set weight group 1 for layer {}", layer);
	}
	for (auto gemv : gemvs) {
		gemv->setKVData(input_data->kv_data[layer]);
	}
	prefill1.setKVData(input_data->kv_data[layer]);
	prefill2.setKVData(input_data->kv_data[layer]);
}

void LocalPipeline::setWeight_group_LN(int layer) {
	bool success = true;
	success &= layerNormAttention1.setWeight(input_data->weight.layer_weight[layer].W_LN_Attention);
	success &= layerNormAttention2.setWeight(input_data->weight.layer_weight[layer].W_LN_Attention);
	success &= KQV1->set_weight(input_data->weight.layer_weight[layer].W_KQV);
	success &= KQV2->set_weight(input_data->weight.layer_weight[layer].W_KQV);
	if (!success) {
		spdlog::error("Failed to set weight group 2 for layer {}", layer);
	}
	roPEAppends[0].setKVData(input_data->kv_data[layer]);
	roPEAppends[1].setKVData(input_data->kv_data[layer]);
}

void LocalPipeline::GEMMOpInit() {
	cutlass::half_t beta(1);
	setWeight(0);
	O1->init(beta);
	O2->init(beta);
	D1->init(beta);
	D2->init(beta);
	KQV1->init();
	KQV2->init();
	
	LOGITS->set_weight(input_data->weight.lm_head); // important

	LOGITS->init();

	LOGITS->set_weight(input_data->weight.lm_head);

	for (int i = 0; i < gemmNum; ++i) {
		gemms[i]->setStream(stream_gemm);
		gemms[i]->updateEventExistance(false, false);
	}

	dual1.init();
	dual2.init();
	dual1.setStream(stream_gemm);
	dual2.setStream(stream_gemm);
}

void LocalPipeline::GEMVOpInit() {
	for (auto gemv : gemvs) {
		gemv->setStream(stream_gemv);
	}
	prefill1.setStream(stream_gemv);
	prefill2.setStream(stream_gemv);
}

void LocalPipeline::OtherOpInit() {
	// embedding one time set weight
	genEmbedding1.setWeight(input_data->weight.embedding);
	genEmbedding2.setWeight(input_data->weight.embedding);
	layerNormModel.setWeight(input_data->weight.model_layernorm);

	// set other operator to gemm stream
	genEmbedding1.setStream(stream_gemm).updateEventExistance(false, false);
	genEmbedding2.setStream(stream_gemm).updateEventExistance(false, false);
	layerNormAttention1.setStream(stream_gemm).updateEventExistance(false, false);
	layerNormAttention2.setStream(stream_gemm).updateEventExistance(false, false);
	layerNormFFN1.setStream(stream_gemm).updateEventExistance(false, false);
	layerNormFFN2.setStream(stream_gemm).updateEventExistance(false, false);
	layerNormModel.setStream(stream_gemm);
	roPEAppends[0].setStream(stream_gemm);
	roPEAppends[1].setStream(stream_gemm);
	maxSampler.setStream(stream_gemm);
	
	keepToken.setStream(stream_gemm);

	// update event existance

}

void LocalPipeline::ScheduleInit() {
	const int kdim = ModelConfig.model_head_dim * ModelConfig.model_kv_heads_gpu;
	const int qdim = ModelConfig.model_head_dim *  ModelConfig.model_qo_heads_gpu;
	const int vdim = ModelConfig.model_head_dim *  ModelConfig.model_kv_heads_gpu;
	const auto& KQV_shared_output = tmpBufferM.allocTensor(config_data.global_batch_size, kdim + qdim + vdim, PllmLayout::ROW_MAJOR);
	const auto& KQV_output12 = KQV_shared_output.splitTensor(PllmDimension::ROW, O1->M, O2->M);
	KQV_output1 = tensor_cast<cutlass::half_t, half> (KQV_output12[0]);
	KQV_output2 = tensor_cast<cutlass::half_t, half> (KQV_output12[1]);
	
	const auto& Q_shared = tmpBufferM.allocTensor(config_data.global_batch_size, qdim, PllmLayout::ROW_MAJOR);
	const auto& gemvQ = tensor_cast<cutlass::half_t, half>(Q_shared);
	const auto & gemvQ12 = gemvQ.splitTensor(PllmDimension::ROW, O1->M, O2->M);
	gemvQ1 = gemvQ12[0];
	gemvQ2 = gemvQ12[1];

	const auto& GEMV_shared_output = tmpBufferM.allocTensor(config_data.global_batch_size, ModelConfig.model_hidden_dim_pergpu, PllmLayout::ROW_MAJOR);
	const auto& gemvAggregateSharedOutput = GEMV_shared_output;
	const auto& gemvAggregateOutput12 = gemvAggregateSharedOutput.splitTensor(PllmDimension::ROW, O1->M, O2->M);
	gemvAggregateOutput1 = tensor_cast<cutlass::half_t, half>(gemvAggregateOutput12[0]);
	gemvAggregateOutput2 = tensor_cast<cutlass::half_t, half>(gemvAggregateOutput12[1]);

	const auto& D_shared_output = tmpBufferM.allocTensor(config_data.global_batch_size, D1->N, PllmLayout::ROW_MAJOR);
	const auto& D_output12 = D_shared_output.splitTensor(PllmDimension::ROW, D1->M, D2->M);

	// allocate D output buffer
	const auto& D1_output = D_output12[0];
	genEmbedding1.setOutput(tensor_cast<cutlass::half_t, half>(D1_output));

	const auto& layerNormAttention1_output = tmpBufferM.allocTensor(D1->M, D1->N, PllmLayout::ROW_MAJOR);
	layerNormAttention1.setInput(D1_output).setOutput(layerNormAttention1_output);

	KQV1->setA(layerNormAttention1_output).setOutput(KQV_output12[0]);

	// ropeAppend, gemv, prefill initiated in update

	O1->setA(gemvAggregateOutput12[0]);
	const auto& O1_output = tmpBufferM.allocTensor(O1->M, O1->N, PllmLayout::ROW_MAJOR);
	O1->setC(D1_output);
	O1->setD(O1_output);

	layerNormFFN1.setInput(O1_output);
	const auto& LayerNormFFN1_output = tmpBufferM.allocTensor(dual1.M, dual1.K, PllmLayout::ROW_MAJOR);
	layerNormFFN1.setOutput(LayerNormFFN1_output);


	const auto& Dual1_output_0 = tmpBufferM.allocTensor(dual1.M, dual1.N, PllmLayout::ROW_MAJOR);
	const auto& Dual1_output_1 = tmpBufferM.allocTensor(dual1.M, dual1.N, PllmLayout::ROW_MAJOR);
	// const auto& Dual1_output = tmpBufferM.allocTensor(dual1.M, dual1.N, PllmLayout::ROW_MAJOR);

	const auto& activation1_output = tmpBufferM.allocTensor(D1->M, D1->K, PllmLayout::ROW_MAJOR);

	dual1.setA(LayerNormFFN1_output);
	dual1.setC(tmpBufferM.allocTensor(dual1.M, dual1.N, PllmLayout::ROW_MAJOR));
	
	dual1.setD(Dual1_output_0, Dual1_output_1, activation1_output);

	D1->setA(activation1_output);
	D1->setC(O1_output);
	D1->setD(D1_output);



	// nanobatch2
	// allocate D output buffer
	const auto& D2_output = D_output12[1];
	genEmbedding2.setOutput(tensor_cast<cutlass::half_t, half>(D2_output));

	const auto& layerNormAttention2_output = tmpBufferM.allocTensor(D2->M, D2->N, PllmLayout::ROW_MAJOR);
	layerNormAttention2.setInput(D2_output).setOutput(layerNormAttention2_output);

	KQV2->setA(layerNormAttention2_output).setOutput(KQV_output12[1]);

	// ropeAppend, gemv, prefill initiated in update

	O2->setA(gemvAggregateOutput12[1]);
	const auto& O2_output = tmpBufferM.allocTensor(O2->M, O2->N, PllmLayout::ROW_MAJOR);
	O2->setC(D2_output);
	O2->setD(O2_output);

	layerNormFFN2.setInput(O2_output);
	const auto& LayerNormFFN2_output = tmpBufferM.allocTensor(dual2.M, dual2.K, PllmLayout::ROW_MAJOR);
	layerNormFFN2.setOutput(LayerNormFFN2_output);



	const auto& Dual2_output_0 = tmpBufferM.allocTensor(dual2.M, dual2.N, PllmLayout::ROW_MAJOR);
	const auto& Dual2_output_1 = tmpBufferM.allocTensor(dual2.M, dual2.N, PllmLayout::ROW_MAJOR);
	// const auto& Dual2_output = tmpBufferM.allocTensor(dual2.M, dual2.N, PllmLayout::ROW_MAJOR);

	const auto& activation2_output = tmpBufferM.allocTensor(D2->M, D2->K, PllmLayout::ROW_MAJOR);

	dual2.setA(LayerNormFFN2_output);
	dual2.setC(tmpBufferM.allocTensor(dual2.M, dual2.N, PllmLayout::ROW_MAJOR));
	
	dual2.setD(Dual2_output_0, Dual2_output_1, activation2_output);

	D2->setA(activation2_output);
	D2->setC(O2_output);
	D2->setD(D2_output);


// sampling
	const auto& keepTokenOutput = tmpBufferM.allocTensor(config_data.global_batch_size, D2->N, PllmLayout::ROW_MAJOR);
	keepToken.setInput(tensor_cast<cutlass::half_t, half>(D_shared_output)).setOutput(tensor_cast<cutlass::half_t, half>(keepTokenOutput));
	const auto& layerNormModel_output = tmpBufferM.allocTensor(config_data.global_batch_size, D2->N, PllmLayout::ROW_MAJOR);
	layerNormModel.setInput(keepTokenOutput).setOutput(layerNormModel_output);

	const auto& LOGITS_output = tmpBufferM.allocTensor(config_data.global_batch_size, LOGITS->N, PllmLayout::ROW_MAJOR);
	LOGITS->setA(layerNormModel_output);
	LOGITS->setOutput(LOGITS_output);

	int* sample_output_alloc = (int*)tmpBufferM.alloc(config_data.global_batch_size * sizeof(int) / sizeof (half));
	pllmTensor<int> sample_output = {sample_output_alloc, config_data.global_batch_size, 1, PllmLayout::ROW_MAJOR};
	const auto& maxSampler_maxVals = tmpBufferM.allocTensor(config_data.global_batch_size, 1, PllmLayout::ROW_MAJOR);
	maxSampler.init(tensor_cast<cutlass::half_t, half>(LOGITS_output), 
					tensor_cast<cutlass::half_t, half>(maxSampler_maxVals), sample_output);

	spdlog::info("allocated num: {}", tmpBufferM.getAllocation());

}

vortexOutputData LocalPipeline::run() {
	constexpr bool enableGraph = false;
	spdlog::info("Start run");
	setWeight(0);

	if(!enableGraph)
		CUDA_CHECK(cudaEventRecord(events[EventManager::GEMM_TIMING_START], stream_gemm));
	// CUDA_CHECK(cudaEventRecord(events[EventManager::GEMM_TIMING_START], stream_gemm));

	cudaEventRecord(events[EventManager::CAPTURE_GEMM_START], stream_gemm);
	cudaStreamWaitEvent(stream_gemv, events[EventManager::CAPTURE_GEMM_START], 0);

	genEmbedding1.run().log(private_logger);
	genEmbedding2.run().log(private_logger);

	// warmup
	setWeight_group_LN(0);
	layerNormAttention1.run().log(private_logger);
	KQV1->run().log(private_logger);
	if (roPEAppends[0].dense_batch_size > 0) {
		roPEAppends[0].run().log(private_logger);
	} else {
		roPEAppends[0].skip();
	}
	
	layerNormAttention2.run().log(private_logger);
	KQV2->run().log(private_logger);
	if (roPEAppends[1].dense_batch_size > 0) {
		roPEAppends[1].run().log(private_logger);
	} else {
		roPEAppends[1].skip();
	}


	for (int iter = 0; iter < ModelConfig.run_layer; ++iter) {
		private_logger->info(">>>>>>>>>>>>>>>>>>>>>>>>>> layer: {}", iter);
		setWeight_group_O(iter % ModelConfig.model_layer);

		if (GEMV1.batch_size > 0) {
			GEMV1.wait(roPEAppends[0]).run().log(private_logger);
		} else {
			GEMV1.skip();
		}

		if (GEMV2.batch_size > 0) {
			GEMV2.wait(roPEAppends[0]).run().log(private_logger);
		}
		else {
			GEMV2.skip();
		}
		if (prefill1.batch_size > 0) {
			prefill1.wait(roPEAppends[0]).run().log(private_logger);
		}
		else{
			prefill1.skip();
		}

		O1->wait(prefill1).wait(GEMV2).run().log(private_logger);
		layerNormFFN1.run().log(private_logger);
		// UG1->run().log(private_logger);
		// log_tensor(private_logger, "G out", UG1->getD(), 10, 20, 0, ModelConfig.model_ff_dim_gpu);
		// activation1.run().log(private_logger);
		dual1.run().log(private_logger);
		D1->run().log(private_logger);

		if (GEMV3.batch_size > 0) {
			GEMV3.wait(roPEAppends[1]).run().log(private_logger);
		} else {
			GEMV3.skip();
		}

		if (GEMV4.batch_size > 0) {
			GEMV4.wait(roPEAppends[1]).run().log(private_logger);
		}
		else {
			GEMV4.skip();
		}
		if (prefill2.batch_size > 0) {
			prefill2.wait(roPEAppends[1]).wait(dual1).run().log(private_logger); // .wait(O1)
		}
		else{
			prefill2.skip();
		}

		setWeight_group_LN((iter + 1) % ModelConfig.model_layer);

		if (iter != ModelConfig.run_layer - 1) {
			layerNormAttention1.run().log(private_logger);
			KQV1->run().log(private_logger);
			if (roPEAppends[0].dense_batch_size > 0) {
				roPEAppends[0].run().log(private_logger);
			} else {
				roPEAppends[0].skip();
			}
		}

		O2->wait(prefill2).run().log(private_logger);
		layerNormFFN2.run().log(private_logger);
		// UG2->run().log(private_logger);
		// activation2.run().log(private_logger);
		dual2.run().log(private_logger);
		D2->run().log(private_logger);
		
		if (iter != ModelConfig.run_layer - 1) {
			layerNormAttention2.run().log(private_logger);
			KQV2->run().log(private_logger);
			if (roPEAppends[1].dense_batch_size > 0) {
				roPEAppends[1].run().log(private_logger);
			} else {
				roPEAppends[1].skip();
			}
		}
	}


	keepToken.run().log(private_logger);
	layerNormModel.run().log(private_logger);
	LOGITS->run().log(private_logger);
	maxSampler.run().log(private_logger);

	cudaEventRecord(events[EventManager::CAPTURE_GEMV_END], stream_gemv);
	// cudaStreamWaitEvent(stream_gemm, events[EventManager::CAPTURE_GEMV_END], 0);
	// cudaStreamWaitEvent(stream_gemm, events[EventManager::CAPTURE_NET_END], 0);

	// Record an event when the GEMMs are complete
	CUDA_CHECK(cudaEventRecord(events[EventManager::GEMM_TIMING_END], stream_gemm));

	// Wait for work on the device to complete.
	CUDA_CHECK(cudaEventSynchronize(events[EventManager::GEMM_TIMING_END]));

	// Measure elapsed runtime
	float runtime_ms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&runtime_ms,
									events[EventManager::GEMM_TIMING_START],
									events[EventManager::GEMM_TIMING_END]));

	// Compute average runtime and GFLOPs.
	runtime_ms = double(runtime_ms);
	float total_throughput = 0;
	float total_throughput_per_gpu = 0;
	total_throughput = config_data.global_batch_size / (runtime_ms / 1000);
	total_throughput_per_gpu = total_throughput / ModelConfig.gpu_num;

	spdlog::info("Total running cost (ms) of one microbatch is {}", runtime_ms);
	spdlog::info("Total running throughput (token/s) of one microbatch is {}", total_throughput_per_gpu);

	// Copy output data back to host
	spdlog::info("sampled_token_array {}, maxSampler.d_argMax.ptr {}, sampled_tokens {}", (size_t) output_data.sampled_token_array, (size_t)maxSampler.d_argMax.ptr, output_data.sampled_tokens);
	CUDA_CHECK(cudaMemcpy(output_data.sampled_token_array, maxSampler.d_argMax.ptr, output_data.sampled_tokens* sizeof(int), cudaMemcpyDeviceToHost));
	
	return output_data;

	// for (int iter = 1; iter <= ModelConfig.model_layer; ++iter) {
	// 	O->run();
	// 	layerNormFFN.run();
	// 	GEMV.wait(KQV).run();
	// 	if (update_data.prefillNum > 0)
	// 		prefill.run();
	// 	UG->run();
	// 	activation.run();
	// 	D->run();
	// 	layerNormAttention.run();

	// 	if (iter == ModelConfig.model_layer) break;
	// 	setWeight(iter%5);
	// 	KQV->run();
	// 	roPEAppend.skip();
	// }


	// // Record an event when the GEMMs are complete
	// CUDA_CHECK(cudaEventRecord(events[EventManager::GEMM_TIMING_END], stream_gemm));

	// // Wait for work on the device to complete.
	// CUDA_CHECK(cudaEventSynchronize(events[EventManager::GEMM_TIMING_END]));

	// // Measure elapsed runtime
	// float runtime_ms = 0;
	// CUDA_CHECK(cudaEventElapsedTime(&runtime_ms,
	// 								events[EventManager::GEMM_TIMING_START],
	// 								events[EventManager::GEMM_TIMING_END]));
	// // Compute average runtime and GFLOPs.
	// runtime_ms = double(runtime_ms);
	// double gflops = totalCompute() / runtime_ms / 1e6;
	// double bandwidth = sizeof(__half) * (96731136 + 1.25 / 80 * 160 * 1024 * 1024 * 1024 / 2) /
	// 				   (runtime_ms / 1000) / (1 << 30);
	// spdlog::info("Total running cost (ms) of one microbatch is {}", runtime_ms);

	// vortexOutputData d;
	// d.offload_kv_cache = nullptr;
	// d.sampled_token_array = outputTokens;

	// return d;
}

void LocalPipeline::GEMVOpUpdate() {
	
	spdlog::info("Update GEMV");

	bool need_overlap = true;
	if (config_data.nanobatch_1_size >= update_token_num) {
		need_overlap = false;
	} else if (update_data.decodePrefillBorder >= config_data.nanobatch_1_size) {
		need_overlap = false;
	}

	int border_prefill_request;
	if (need_overlap) {
		border_prefill_request = host_rev_input_indptr[config_data.nanobatch_1_size - 1];
	}

	// possible overlap (split overlap) of kv_indptr and input_ptr and last_page_len 
	// request index pointer

	size_t req_num = update_data.decodePrefillBorder + update_data.prefillNum;

	pllmTensor kv_indptr = pllmTensor{update_data.kv_indptr, req_num + 1};
	pllmTensor kv_last_page_len = pllmTensor{update_data.kv_last_page_len, req_num};
	pllmTensor kv_indices = pllmTensor{update_data.kv_indices, ModelConfig.max_page_num};
	pllmTensor input_ptr = pllmTensor{update_data.input_indptr, req_num + 1};
	pllmTensor rev_input_indptr = pllmTensor{update_data.rev_input_indptr, config_data.global_batch_size};
	pllmTensor per_token_offset = pllmTensor{update_data.per_token_offset, config_data.global_batch_size};

	const pllmTensor<half> KQV_split[] = {KQV_output1, KQV_output2};
	const pllmTensor<half> Q_split[] = {gemvQ1, gemvQ2};
	const auto& rev_input_indptr_split = rev_input_indptr.splitTensor(PllmDimension::ROW, KQV1->M, KQV2->M);
	const auto& per_token_offset_split = per_token_offset.splitTensor(PllmDimension::ROW, KQV1->M, KQV2->M);

	if (!need_overlap) {

		int nano1_prefill_size = update_data.prefillNum; // assume token num is small
		int nano2_prefill_size = 0;
		if (config_data.nanobatch_1_size <= update_token_num) {
			assert(config_data.nanobatch_1_size <= update_data.decodePrefillBorder);
			nano1_prefill_size = 0;
			nano2_prefill_size = update_data.prefillNum;
		}

		uint32_t arr[] = {uint32_t(update_data.gemv_batch_size[0]), 
							uint32_t(update_data.gemv_batch_size[1]), 
							uint32_t(nano1_prefill_size),
							uint32_t(update_data.gemv_batch_size[2]), 
							uint32_t(update_data.gemv_batch_size[3]), 
							uint32_t(nano2_prefill_size)};
		spdlog::info("GEMVOpUpdate: local, batch_sizes: {}, {}, {}, {}, {}, {}", arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
		std::span<uint32_t, 6> batch_sizes(arr, 6);
		std::span<int32_t, 4> gemv_num_blocks(update_data.gemv_num_blocks, 4);
		
		// assert(req_num == update_data.decodePrefillBorder + update_data.prefillNum);
		const auto& kv_indptr_split = kv_indptr.splitTensor(PllmDimension::ROW, batch_sizes,/*overlap suffix*/ 1U);
		const auto& kv_last_page_len_split = kv_last_page_len.splitTensor(PllmDimension::ROW, batch_sizes);
		const auto& input_ptr_split = input_ptr.splitTensor(PllmDimension::ROW, batch_sizes,/*overlap suffix*/ 1U);
		// const auto& GEMV_input1 = gemvQ1.splitTensor(PllmDimension::ROW,
		// 								batch_sizes[0],
		// 								batch_sizes[1]);							
		// const auto& GEMV_input2 = gemvQ2.splitTensor(PllmDimension::ROW,
		// 								batch_sizes[2],
		// 								batch_sizes[3],
		// 								ModelConfig.max_batch_size - update_data.decodePrefillBorder);
		// const auto& GEMV_output1 = gemvAggregateOutput1.splitTensor(PllmDimension::ROW,
		// 								batch_sizes[0],
		// 								batch_sizes[1]
		// 								);
		// const auto& GEMV_output2 = gemvAggregateOutput2.splitTensor(PllmDimension::ROW,
		// 								batch_sizes[2],
		// 								batch_sizes[3],
		// 								ModelConfig.max_batch_size - update_data.decodePrefillBorder);
		

		GEMV1.init(batch_sizes[0],
				gemv_num_blocks[0],
				input_ptr_split[0],
				kv_indptr_split[0],
				update_data.kv_indices,
				kv_last_page_len_split[0],
				gemvQ1,
				gemvAggregateOutput1);
		
		GEMV2.init(batch_sizes[1],
				gemv_num_blocks[1],
				input_ptr_split[1],
				kv_indptr_split[1],
				update_data.kv_indices,
				kv_last_page_len_split[1],
				gemvQ1,
				gemvAggregateOutput1);

		prefill1.init(batch_sizes[2],
					108,
					input_ptr_split[2],
					kv_indptr_split[2],
					update_data.kv_indices,
					kv_last_page_len_split[2],
					gemvQ1, // find some input for prefill1
					gemvAggregateOutput1); // find some output for prefill1

		GEMV3.init(batch_sizes[3],
				gemv_num_blocks[2],
				input_ptr_split[3],
				kv_indptr_split[3],
				update_data.kv_indices,
				kv_last_page_len_split[3],
				gemvQ1,
				gemvAggregateOutput1);

		GEMV4.init(batch_sizes[4],
				gemv_num_blocks[3],
				input_ptr_split[4],
				kv_indptr_split[4],
				update_data.kv_indices,
				kv_last_page_len_split[4],
				gemvQ1,
				gemvAggregateOutput1);
		
		// ::log_tensor(spdlog::default_logger(), "input_ptr_split[4]", input_ptr_split[4], 1, 3);
		
		prefill2.init(batch_sizes[5],
					108,
					input_ptr_split[5],
					kv_indptr_split[5],
					update_data.kv_indices,
					kv_last_page_len_split[5],
					gemvQ1, // find some input for prefill2
					gemvAggregateOutput1); // find some output for prefill2
					// use start address because prefill kernel get input use buffer[indptr[i]]
	} else {
		assert(border_prefill_request - update_data.decodePrefillBorder >= 0);
		assert(update_data.prefillNum + update_data.decodePrefillBorder - border_prefill_request >= 0);
		uint32_t arr[] = {uint32_t(update_data.gemv_batch_size[0]), uint32_t(update_data.gemv_batch_size[1]), uint32_t(border_prefill_request - update_data.decodePrefillBorder),  
							0, 0, uint32_t(update_data.prefillNum + update_data.decodePrefillBorder - border_prefill_request)};
		std::span<uint32_t, 6> batch_sizes_for_split(arr, 6);
		std::span<int32_t, 4> gemv_num_blocks(update_data.gemv_num_blocks, 4);
		uint32_t suffix_arr[] = {1, 1, 2, 1, 1, 1};
		std::span<uint32_t, 6> suffix(suffix_arr, 6);
		uint32_t suffix_arr_for_last_page_len[] = {0, 0, 1, 0, 0, 0};
		std::span<uint32_t, 6> suffix_for_last_page_len(suffix_arr_for_last_page_len, 6);

		const auto& kv_indptr_split = kv_indptr.splitTensor(PllmDimension::ROW, batch_sizes_for_split, suffix);
		const auto& kv_last_page_len_split = kv_last_page_len.splitTensor(PllmDimension::ROW, batch_sizes_for_split, suffix_for_last_page_len);
		const auto& input_ptr_split = input_ptr.splitTensor(PllmDimension::ROW, batch_sizes_for_split, suffix);
		// const auto& GEMV_input1 = gemvQ1.splitTensor(PllmDimension::ROW,
		// 								batch_sizes_for_split[0],
		// 								batch_sizes_for_split[1],
		// 								config_data.nanobatch_1_size - update_data.decodePrefillBorder);
		// // no need to split GEMV_input2 since no gemv, while prefill would read data from aggregated input
		// const auto& GEMV_output1 = gemvAggregateOutput1.splitTensor(PllmDimension::ROW,
		// 								batch_sizes_for_split[0],
		// 								batch_sizes_for_split[1],
		// 								config_data.nanobatch_1_size - update_data.decodePrefillBorder);

		// copy input_indptr to prefill1_input_indptr
		CUDA_CHECK(cudaMemcpy(prefill1_update.input_indptr, update_data.input_indptr, 
			req_num*sizeof(int), cudaMemcpyDeviceToDevice));
		// //debug:
		// int* debug_input_indptr = new int[req_num];
		// CUDA_CHECK(cudaMemcpy(debug_input_indptr, prefill1_input_indptr, 
		// 	req_num*sizeof(int), cudaMemcpyDeviceToHost));
		// spdlog::info("last req idx: {}", border_prefill_request);
		// spdlog::info("last req start_row: {}", debug_input_indptr[border_prefill_request]);
		// spdlog::info("last req end_row: {}", debug_input_indptr[border_prefill_request + 1]);
		// spdlog::info("change to: {}", config_data.nanobatch_1_size);
		// delete[] debug_input_indptr;
		// // end debug
		int new_end_row = config_data.nanobatch_1_size;
		// copy new end row to prefill1_input_indptr position border_prefill_request +1
		CUDA_CHECK(cudaMemcpy(prefill1_update.input_indptr + border_prefill_request + 1, &new_end_row, 
			sizeof(int), cudaMemcpyHostToDevice));
		// re-split prefill1_input_indptr
		auto prefill1_input_indptr_tensor = pllmTensor{prefill1_update.input_indptr, req_num + 1};
		const auto& prefill1_input_indptr_split = prefill1_input_indptr_tensor.splitTensor(PllmDimension::ROW, batch_sizes_for_split, suffix);

		int border_offset;
		CUDA_CHECK(cudaMemcpy(&border_offset, per_token_offset.ptr + config_data.nanobatch_1_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
		int seq_len = border_offset + 1;
		int num_page = (seq_len + ModelConfig.frame_page_size-1) / ModelConfig.frame_page_size;
		int last_page_len = (seq_len - 1) % ModelConfig.frame_page_size + 1;
		spdlog::info("border_offset: {}, num_page: {}, last_page_len: {}", border_offset, num_page, last_page_len);

		int kv_indptr_border[2];
		CUDA_CHECK(cudaMemcpy(&kv_indptr_border, kv_indptr.ptr + border_prefill_request, 2*sizeof(int), cudaMemcpyDeviceToHost));
		spdlog::info("kv_indptr_border: {}, {}", kv_indptr_border[0], kv_indptr_border[1]);

		CUDA_CHECK(cudaMemcpy(prefill1_update.kv_indptr, update_data.kv_indptr, 
			(req_num + 1)*sizeof(int), cudaMemcpyDeviceToDevice));
		kv_indptr_border[1] = kv_indptr_border[0] + num_page;
		spdlog::info("change to: {}, {}", kv_indptr_border[0], kv_indptr_border[1]);
		CUDA_CHECK(cudaMemcpy(prefill1_update.kv_indptr + border_prefill_request, &kv_indptr_border, 2*sizeof(int), cudaMemcpyHostToDevice));
		auto prefill1_kv_indptr_tensor = pllmTensor{prefill1_update.kv_indptr, req_num + 1};
		const auto& prefill1_kv_indptr_split = prefill1_kv_indptr_tensor.splitTensor(PllmDimension::ROW, batch_sizes_for_split, suffix);

		int orig_last_page_len;
		CUDA_CHECK(cudaMemcpy(&orig_last_page_len, kv_last_page_len.ptr + border_prefill_request, sizeof(int), cudaMemcpyDeviceToHost));
		spdlog::info("orig_last_page_len: {}", orig_last_page_len);
		spdlog::info("change to: {}", last_page_len);
		CUDA_CHECK(cudaMemcpy(prefill1_update.kv_last_page_len, update_data.kv_last_page_len, 
			req_num*sizeof(int), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(prefill1_update.kv_last_page_len + border_prefill_request, &last_page_len, 
			sizeof(int), cudaMemcpyHostToDevice));
		auto prefill1_last_page_len_tensor = pllmTensor{prefill1_update.kv_last_page_len, req_num};
		const auto& prefill1_last_page_len_split = prefill1_last_page_len_tensor.splitTensor(PllmDimension::ROW, batch_sizes_for_split, suffix_for_last_page_len);

		std::span<uint32_t, 6> actual_batch_size = batch_sizes_for_split;
		actual_batch_size[2]++;
		GEMV1.init(actual_batch_size[0],
				gemv_num_blocks[0],
				input_ptr_split[0],
				kv_indptr_split[0],
				update_data.kv_indices,
				kv_last_page_len_split[0],
				gemvQ1,
				gemvAggregateOutput1);
		
		GEMV2.init(actual_batch_size[1],
				gemv_num_blocks[1],
				input_ptr_split[1],
				kv_indptr_split[1],
				update_data.kv_indices,
				kv_last_page_len_split[1],
				gemvQ1,
				gemvAggregateOutput1);

		prefill1.init(actual_batch_size[2],
				    108,
					prefill1_input_indptr_split[2],
					prefill1_kv_indptr_split[2],
					update_data.kv_indices,
					prefill1_last_page_len_split[2],
					gemvQ1,
					gemvAggregateOutput1);

		GEMV3.init(actual_batch_size[3], // 0
				gemv_num_blocks[3],
				input_ptr_split[3],
				kv_indptr_split[3],
				update_data.kv_indices,
				kv_last_page_len_split[3],
				gemvQ1, // meaningless: no gemv3 and gemv4 input and output
				gemvAggregateOutput1);

		GEMV4.init(actual_batch_size[4], // 0
				gemv_num_blocks[4],
				input_ptr_split[4],
				kv_indptr_split[4],
				update_data.kv_indices,
				kv_last_page_len_split[4],
				gemvQ1,
				gemvAggregateOutput1);

		prefill2.init(actual_batch_size[5],
					108,
					input_ptr_split[5],
					kv_indptr_split[5],
					update_data.kv_indices,
					kv_last_page_len_split[5],
					gemvQ1,
					gemvAggregateOutput1);
	}

	size_t token_remaining = update_token_num;
	spdlog::info("token_remaining: {}", token_remaining);

	for (int i = 0; i < 2; i++)
	{
		if (token_remaining > KQV_ptrs[i] -> M){
			roPEAppends[i].update(KQV_ptrs[i] -> M, KQV_split[i], Q_split[i], rev_input_indptr_split[i], per_token_offset_split[i], 
								kv_indices, kv_indptr, kv_last_page_len);
			spdlog::info("ropeappend[{}] dense batch size {}", i, KQV_ptrs[i]->M);
			token_remaining -= KQV_ptrs[i] -> M;
		}
		else{
			roPEAppends[i].update(token_remaining, KQV_split[i], Q_split[i], rev_input_indptr_split[i], per_token_offset_split[i], 
								kv_indices, kv_indptr, kv_last_page_len);
			spdlog::info("ropeappend[{}] dense batch size {}", i, token_remaining);
			token_remaining = 0;

		}
	}

	spdlog::info("RopeAppend Tokens: {}, {}", roPEAppends[0].dense_batch_size, roPEAppends[1].dense_batch_size);

	
	spdlog::info("finished GEMV update");
}

void LocalPipeline::update(vortexUpdateData* update_data_) {
	this->update_data = *update_data_;
	auto input_span = pllmTensor(update_data.input_tokens, config_data.global_batch_size, 1, PllmLayout::ROW_MAJOR);

	auto input_span1 = input_span.subtensor(0, config_data.nanobatch_1_size);
	auto input_span2 = input_span.subtensor(config_data.nanobatch_1_size, config_data.global_batch_size - config_data.nanobatch_1_size);

	genEmbedding1.setInput(input_span1);
	genEmbedding2.setInput(input_span2);

	update_token_num = update_data.decodePrefillBorder + update_data.prefillTokensNum;
	int req_num = update_data.decodePrefillBorder + update_data.prefillNum;

	maxSampler.set_batch_size(req_num);

	keepToken.update(req_num, update_data.input_indptr);
	LOGITS->set_shape((req_num+127)/128*128, ModelConfig.vocab_size, ModelConfig.model_hidden_dim);
	LOGITS->init();
	cudaMemcpy(host_rev_input_indptr, update_data.rev_input_indptr, 
			(update_token_num)*sizeof(int), cudaMemcpyDeviceToHost);
	output_data.sampled_tokens = req_num;

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


double LocalPipeline::totalCompute() {
	double total = 0;
	for(auto gemm : gemms)
		total += gemm->totalCompute();
	return total;
}

void LocalPipeline::config(vortexConfigData* config_data) {
	spdlog::info("Config local pipeline");
	this->config_data = * config_data;
    for (int i = 0; i < gemmNum; ++i) {
        gemms[i] = generateGEMM(this->config_data.gemm_op_tag[i]);
		spdlog::info("GEMM {} created, tag: {}", i, this->config_data.gemm_op_tag[i]);
    }
	KQV_ptrs[0] = KQV1;
	KQV_ptrs[1] = KQV2;
    int globalbatch = config_data->global_batch_size;
	int nano1 = config_data->nanobatch_1_size;
	int nano2 = globalbatch - nano1;
    O1 ->set_shape(nano1, ModelConfig.model_hidden_dim_pergpu, ModelConfig.model_hidden_dim);
	O2 ->set_shape(nano2, ModelConfig.model_hidden_dim_pergpu, ModelConfig.model_hidden_dim);
    D1 ->set_shape(nano1, ModelConfig.model_hidden_dim, ModelConfig.model_ff_dim_gpu);
	D2 ->set_shape(nano2, ModelConfig.model_hidden_dim, ModelConfig.model_ff_dim_gpu);
	KQV1->set_shape(nano1,
				   (ModelConfig.model_kv_heads_gpu + ModelConfig.model_kv_heads_gpu + ModelConfig.model_qo_heads_gpu) * ModelConfig.model_head_dim,
				   ModelConfig.model_hidden_dim);
	KQV2->set_shape(nano2,
				   (ModelConfig.model_kv_heads_gpu + ModelConfig.model_kv_heads_gpu + ModelConfig.model_qo_heads_gpu) * ModelConfig.model_head_dim,
				   ModelConfig.model_hidden_dim);
	
	LOGITS -> set_shape(globalbatch/nranks, ModelConfig.vocab_size, ModelConfig.model_hidden_dim);

	dual1.set_shape(nano1, ModelConfig.model_ff_dim_gpu, ModelConfig.model_hidden_dim);
	dual2.set_shape(nano2, ModelConfig.model_ff_dim_gpu, ModelConfig.model_hidden_dim);


	setName();

	spdlog::info("Init schedule");
	ScheduleInit();
	spdlog::info("Init gemm");
	GEMMOpInit();
	// No longer need GEMVOpInit and OtherOpInit at this moment.
	spdlog::info("Init gemv");
	GEMVOpInit();
	spdlog::info("Init other");
	OtherOpInit();

	if (nanobatch_only){
		spdlog::info("Init nanobatch only");
		assignSameStream();
	}
	// init the output 
	output_data = vortexOutputData();
	output_data.sampled_token_array = new int[config_data->global_batch_size];
	output_data.global_batch_size = config_data->global_batch_size;

	CUDA_CHECK(cudaMalloc(&prefill1_update.input_indptr, sizeof(int) * config_data->global_batch_size));
	CUDA_CHECK(cudaMalloc(&prefill1_update.kv_indptr, sizeof(int) * (config_data->global_batch_size + 1)));
	CUDA_CHECK(cudaMalloc(&prefill1_update.kv_last_page_len, sizeof(int) * config_data->global_batch_size));
}