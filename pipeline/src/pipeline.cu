#include "netWrapper.cuh"
#include "pipeline.h"
#include "gemmFactory.cuh"
#include <iostream>
#include <string>

PipelineBase::PipelineBase(vortexInitData* input_data, int nrank, int nranks, int vnranks)
	: input_data(input_data)
	, rank(nrank)
	, nranks(nranks)
	, vnranks(vnranks)
	, tmpBufferM(ptr_cast<cutlass::half_t>(input_data->tmp_buffer), input_data->tmp_buffer_size) {
	spdlog::info("rank: {}, nranks: {}", rank, nranks);
	std::string private_file_name = "log_" + std::to_string(rank) + ".txt";
	private_logger = spdlog::basic_logger_mt("private_logger" + std::to_string(rank), private_file_name, true);
	private_logger->set_level(spdlog::level::info);
	private_logger->log(spdlog::level::info, "rank: {}, nranks: {}", rank, nranks);
	private_logger->set_pattern("%v");
	spdlog::info("BaseFinished");
}



Pipeline::Pipeline(vortexInitData* input_data, int rank, int nranks, int vnranks, bool enable_offload)
	: PipelineBase(input_data, rank, nranks, vnranks)
	, enable_offload(enable_offload)
	, gemv_dep() {
	spdlog::info("Pipeline constructor");
	// sampled tokens 
	cudaMallocHost(&outputTokens, 4096*sizeof(int));

	if (enable_offload) {
		// should be equal to gemm batch size 
		// production = consumption
		// possible that one request need two cycles to offload 
		cudaMallocHost(&offloadKVCache, 1024*8192*2*sizeof(half)); // fix
		

		cudaMalloc(&deviceOffloadKVCache, 1024*8192*2*sizeof(half));
		cudaMalloc(&deviceLoadKVCache, 1024*8192*2*sizeof(half));

		
		int32_t * finished_idx_host = new int32_t[2048];
		int32_t * load_idx_host = new int32_t[2048];
		for	(int i = 0; i < 2048; i++) {
			finished_idx_host[i] = i;
			load_idx_host[i] = i;
		}

		cudaMalloc(&finished_idx, 2048*sizeof(int32_t));
		cudaMemcpy(finished_idx, finished_idx_host, 2048*sizeof(int32_t), cudaMemcpyHostToDevice);

		cudaMalloc(&load_idx, 2048*sizeof(int32_t));
		cudaMemcpy(load_idx, load_idx_host, 2048*sizeof(int32_t), cudaMemcpyHostToDevice);
		
	}

	spdlog::info("allocation done");
	init();
}

void Pipeline::StreamInit() {
	cudaStreamCreate(&stream_gemm);
	cudaStreamCreate(&stream_gemv);
	cudaStreamCreate(&stream_net);
	cudaStreamCreate(&stream_other);
	cudaStreamCreate(&stream_cpy);
}

void Pipeline::setName() {
	SET_NAME_PTR(O1);
	SET_NAME_PTR(O2);
	SET_NAME_PTR(UG1);
	SET_NAME_PTR(UG2);
	SET_NAME_PTR(D1);
	SET_NAME_PTR(D2);
	SET_NAME_PTR(KQV1);
	SET_NAME_PTR(KQV2);
	SET_NAME_PTR(KQV3);
	SET_NAME_PTR(KQV4);
	SET_NAME_PTR(LOGITS1);
	SET_NAME_PTR(LOGITS2);

	SET_NAME_REF(AG_O1);
	SET_NAME_REF(AR_O2);
	SET_NAME_REF(AR_D1);
	SET_NAME_REF(AR1_D2);
	SET_NAME_REF(AR2_D2);
	SET_NAME_REF(AG1_GEMV);

	SET_NAME_REF(genEmbedding1);
	SET_NAME_REF(genEmbedding2_1);
	SET_NAME_REF(genEmbedding2_2);
	SET_NAME_REF(genEmbedding2_1_partial);
	SET_NAME_REF(genEmbedding2_2_partial);

	SET_NAME_REF(GEMV1);
	SET_NAME_REF(GEMV2);
	SET_NAME_REF(GEMV3);
	SET_NAME_REF(GEMV4);
	SET_NAME_REF(prefill);

	SET_NAME_REF(layerNormAttention1);
	SET_NAME_REF(layerNormAttention2_1);
	SET_NAME_REF(layerNormAttention2_2);
	SET_NAME_REF(layerNormFFN1);
	SET_NAME_REF(layerNormFFN2);
	SET_NAME_REF(layerNormModel1);

	SET_NAME_REF(activation1);
	SET_NAME_REF(activation2);

	for (int i = 0; i < 4; i++) {
		SET_NAME_REF(roPEAppends[i]);
	}

	SET_NAME_REF(pageAgg);
	SET_NAME_REF(pageDisp);
	SET_NAME_REF(splitTensor);

	SET_NAME_REF(maxSampler1);
	SET_NAME_REF(maxSampler2);

}


void Pipeline::GEMMOpInit() {

	cutlass::half_t beta(1);

	// Need to set the weight matrices of GEMM before initialize them. Otherwise there will be mysterious performance penalty.
	// Seems the GEMM->tensor_b's metadata will only be used during initialization and have no effect if specified later.
	setWeight(0);

	//TODO: fix it
	// KQV_START.init(buffer + offsetM.O1_IN,
	// 		weight_buffer + weight_offset,
	// 		buffer + offsetM.AG_KQV12 + residual_offset,
	// 		buffer + offsetM.O1_UG1,
	// 		KQV_START->M,
	// 		KQV_START->K,
	// 		KQV_START->M,
	// 		beta);

	O1->init(1);
	O2->init(0.125);
	UG1->init();
	UG2->init();
	D1->init(0.125);
	D2->init(0.125);
	KQV1->init();
	KQV2->init();
	KQV3->init();
	KQV4->init();

	LOGITS1->set_weight(input_data->weight.lm_head); // important
	LOGITS2->set_weight(input_data->weight.lm_head);

	LOGITS1->init();
	LOGITS2->init();

	LOGITS1->set_weight(input_data->weight.lm_head);
	LOGITS2->set_weight(input_data->weight.lm_head);

	for (auto gemm : gemms) {
		gemm->setStream(stream_gemm);
	}
	UG2->updateEventExistance(true, true);
}

void Pipeline::GEMVOpInit() {
	for (auto gemv : gemvs) {
		gemv->setStream(stream_gemv);
	}
	prefill.setStream(stream_gemv);

}


void Pipeline::OtherOpInit(){
	genEmbedding1.setWeight(input_data->weight.embedding);
	genEmbedding2_1_partial.setWeight(input_data->weight.embedding);
	genEmbedding2_1.setWeight(input_data->weight.embedding);
	genEmbedding2_2.setWeight(input_data->weight.embedding);
	genEmbedding2_2_partial.setWeight(input_data->weight.embedding);

	layerNormModel1.setWeight(input_data->weight.model_layernorm);
	// layerNormModel2_1.setWeight(input_data->weight.model_layernorm);
	// layerNormModel2_2.setWeight(input_data->weight.model_layernorm);

	genEmbedding1.setStream(stream_other);
	genEmbedding2_1_partial.setStream(stream_other);
	genEmbedding2_1.setStream(stream_other);
	genEmbedding2_2_partial.setStream(stream_other);
	genEmbedding2_2.setStream(stream_other);
	layerNormAttention1.setStream(stream_other);
	layerNormAttention2_1.setStream(stream_other);
	layerNormAttention2_2.setStream(stream_other);
	layerNormFFN1.setStream(stream_other);
	layerNormFFN2.setStream(stream_other);
	activation1.setStream(stream_gemm);
	activation2.setStream(stream_gemm);
	layerNormModel1.setStream(stream_other);
	// layerNormModel2_1.setStream(stream_other);
	// layerNormModel2_2.setStream(stream_other);
	for (int i = 0; i < 4; i++) {
		roPEAppends[i].setStream(stream_gemm);
		roPEAppends[i].updateEventExistance(false, true);
	}
	splitTensor.setStream(stream_net);
	roPEAppends[0].updateEventExistance(false, true);
	roPEAppends[3].updateEventExistance(false, true);
	KQV1->updateEventExistance(false, false);
	KQV2->updateEventExistance(false, true);
	KQV3->updateEventExistance(false, false);

	pageAgg.setStream(stream_cpy);
	pageDisp.setStream(stream_cpy);

	maxSampler1.setStream(stream_gemm);
	maxSampler2.setStream(stream_gemm);
}

void Pipeline::NetOpInit() {
	AG1_GEMV.setStream(stream_net);
	AR1_D2.setStream(stream_net);
	AR2_D2.setStream(stream_net);
	AG_O1.updateEventExistance(true, true);
	AG_O1.setStream(stream_net);
	AR_O2.setStream(stream_net);
	AR_O2.updateEventExistance(true, true);
	AR_D1.setStream(stream_net);
	AR1_D2.setEpsilon(1e-5);
	AR2_D2.setEpsilon(1e-5);
}

void PipelineBase::NetOpPrepare() {
#ifdef ENABLE_NETWORK
	// Initialize Communicator
	spdlog::info("Init communicator");
	std::shared_ptr<mscclpp::TcpBootstrap> bootstrap =
		std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
#ifdef ENABLE_MPI
	mscclpp::UniqueId uniqueId;
	if(rank == 0) uniqueId = bootstrap->createUniqueId();
	MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
	bootstrap->initialize(uniqueId);
#else
	if (rank == 0) shared_state.uniqueId = bootstrap->createUniqueId();
	worker_sync->barrier();
	bootstrap->initialize(shared_state.uniqueId);
#endif
	comm = std::make_shared<mscclpp::Communicator>(bootstrap);

	// Initialize Connections
	spdlog::info("Init connections");
	std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
	for(int r = 0; r < nranks; ++r) {
		if(r == rank) continue;
		mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
		connectionFutures.push_back(comm->connectOnSetup(r, 0, transport));
	}
	comm->setup();
	std::transform(
		connectionFutures.begin(),
		connectionFutures.end(),
		std::back_inserter(connections),
		[](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) {
			return future.get();
		});
	spdlog::info("Connections set up");

#ifdef ENABLE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#else
	worker_sync->barrier();
#endif

#endif

}

void Pipeline::init() {
	spdlog::info("Init stream");
	StreamInit();
	spdlog::info("Init pipeline");
#ifdef ENABLE_NETWORK
	spdlog::info("Init net");
	NetOpPrepare();
#endif
}


void Pipeline::ScheduleInit() {
	auto getMajor = [this](GEMM_NAME name, int x) {return getMajorType(config_data.gemm_op_tag[static_cast<int>(name)], x);};
	auto getDim = [this](GEMM_NAME name, int x) {return static_cast<PllmDimension>(getMajorType(config_data.gemm_op_tag[static_cast<int>(name)], x));};
	// Nanobatch metadata
	// KQV{1,2,3,4} output and AG{1,2}_GEMV both need to live on a contiguous buffer.
	// The KQV{1,2,3,4} => GEMV{1,2,3,4} dependency will be embedded in their kernel implementation
	const int kdim = MODEL_HEAD_DIM * MODEL_KV_HEADS;
	const int qdim = MODEL_HEAD_DIM * MODEL_QO_HEADS;
	const int vdim = MODEL_HEAD_DIM * MODEL_KV_HEADS;
	const auto& KQV_shared_output = tmpBufferM.allocTensor(config_data.global_batch_size, kdim + qdim + vdim, getMajor(GEMM_NAME::KQV1, 2));

	const auto& Q_shared = tmpBufferM.allocTensor(config_data.global_batch_size, qdim, getMajor(GEMM_NAME::KQV1, 2));
	
	gemvQ = tensor_cast<cutlass::half_t, half>(Q_shared);
	
	const auto& KQV_output_cutlass = KQV_shared_output.splitTensor(getDim(GEMM_NAME::KQV1, 2), KQV1->M, KQV2->M, KQV3->M, KQV4->M);
	KQV_output = tensor_cast<cutlass::half_t, half>(KQV_shared_output);

	// GEMV -> AG -> O1 -> UGD1
	// GEMV -> O2 (split K) -> AR -> UGD2
	const auto& GEMV_output_shared = tmpBufferM.allocTensor((O1->M + O2->M), MODEL_HIDDEN_DIM_PERGPU, getMajor(GEMM_NAME::O2, 0));
	spdlog::info(config_data.gemm_op_tag[static_cast<int>(GEMM_NAME::O1)]);
	spdlog::info(config_data.gemm_op_tag[static_cast<int>(GEMM_NAME::O2)]);
	gemvAggregateOutput = tensor_cast<cutlass::half_t, half>(GEMV_output_shared);

	const auto& GEMV_output_O1O2 = GEMV_output_shared.splitTensor(getDim(GEMM_NAME::O2, 0), O1->M, O2->M);
	const auto& AG1_GEMV_buffer = tmpBufferM.allocTensor(O1->M, O1->K, getMajor(GEMM_NAME::O1, 0));
	//spdlog::info("1");
	//spdlog::info("2");spdlog::info(static_cast<int>(getMajor(GEMM_NAME::O1, 1)));
	//spdlog::info("3");spdlog::info(static_cast<int>(getMajor(GEMM_NAME::O1, 2)));
	
	//// Nanoflow 1st batch
	// (prev)AG1_GEMV -> O1
	
	AG1_GEMV.init(comm, connections, rank, nranks, GEMV_output_O1O2[0], AG1_GEMV_buffer);
	//spdlog::info(static_cast<int>(AG1_GEMV.getOutput().layout));
	//spdlog::info(static_cast<int>(AG1_GEMV.getInput().layout));
	O1->setA(AG1_GEMV.getOutput());
	// TODO(gzuo): If I can decouple netwrapper init from setInput and setOutput, I can further simplify this.
	// O1 -> AG_O1
	
	O1->setD(tmpBufferM.allocTensor(O1->M, O1->N, getMajor(GEMM_NAME::O1, 2)));
	AG_O1.init(comm, connections, rank, nranks, O1->getD(), tmpBufferM.allocTensor(UG1->M, UG1->K, getMajor(GEMM_NAME::O1, 2)));
	// BEFORE_LN_FFN_1_TR.setInput(AG_O1.getOutput()).setOutput(tmpBufferM.allocTensor(UG1->M, UG1->K, getMajor(GEMM_NAME::D1, 2)));
	// AG_O1 -> LN_FFN1
	layerNormFFN1.setInput(AG_O1.getOutput()).setOutput(tmpBufferM.allocTensor(UG1->M, UG1->K, getMajor(GEMM_NAME::UG1, 0)));
	// LN_FFN1 -> UG1
	UG1->setA(layerNormFFN1.getOutput()).setOutput(tmpBufferM.allocTensor(UG1->M, UG1->N, getMajor(GEMM_NAME::UG1, 2)));
	// UG1 -> activation1
	activation1.setInput(UG1->getD()).setOutput(tmpBufferM.allocTensor(D1->M, D1->K, getMajor(GEMM_NAME::D1, 0)));
	// activation1 -> D1
	D1->setA(activation1.getOutput());
	// D1 -> AR_D1
	AR_D1.init(comm, connections, rank, nranks, tmpBufferM.allocTensor(D1->M, D1->N, getMajor(GEMM_NAME::D1, 2)), tmpBufferM.allocTensor(D1->M, D1->N, getMajor(GEMM_NAME::D1, 2)));
	D1->setOutput(AR_D1.getInput());
	D1->setC(AG_O1.getOutput()); // TODO same for D2
	// AR_D1 -> LN_Attention1
	layerNormAttention1.setInput(AR_D1.getOutput()).setOutput(tmpBufferM.allocTensor(D1->M, D1->N, getMajor(GEMM_NAME::D1, 2)));
	// LN_AT_1_TR.setInput(AR_D1.getOutput()).setOutput(tmpBufferM.allocTensor(D1->M, D1->N, getMajor(GEMM_NAME::O1, 2)));
	// The residual connection of the self-attention sublayer (before KQV12 and after O1)
	// TODO(gzuo): where is the residual connection of the MLP sublayer? Should add one before O1 and after D1?
	splitTensor.init(tensor_cast<cutlass::half_t, half>(AR_D1.getOutput()), tensor_cast<cutlass::half_t, half>(tmpBufferM.allocTensor(O1->M, O1->N, getMajor(GEMM_NAME::O1, 2))), nranks, rank);
	O1->setC(tensor_cast<half,cutlass::half_t>(splitTensor.output));
	// LN_Attention1 -> KQV1 | KQV2
	// NOTE: KQV1 and KQV2 share a contiguous output buffer because the following GEMV1 and GEMV2 might work with different batch sizes.
	const auto& KQV12_input = layerNormAttention1.getOutput().splitTensor(getDim(GEMM_NAME::KQV1, 0), KQV1->M, KQV2->M);
	KQV1->setA(KQV12_input[0]).setOutput(KQV_output_cutlass[0]);
	KQV2->setA(KQV12_input[1]).setOutput(KQV_output_cutlass[1]);

	// TODO: ropeAppend is not implemented yet

	// O1_TR.setInput(GEMV_output_O1O2[0]).setOutput(AG1_GEMV_buffer.getSubTensor(rank, vnranks, PllmDimension::ROW));


	//// Nanoflow 2nd batch
	// (prev)AG2_GEMV ->O2
	O2->setA(GEMV_output_O1O2[1]);
	// O2 -> AG_O2
	AR_O2.init(comm, connections, rank, nranks, tmpBufferM.allocTensor(UG2->M, UG2->K, getMajor(GEMM_NAME::O2, 2)), tmpBufferM.allocTensor(UG2->M, UG2->K, getMajor(GEMM_NAME::O2, 2)));
	O2->setD(AR_O2.getInput());
	// AG_O2 -> LN_FFN2
	layerNormFFN2.setInput(AR_O2.getOutput()).setOutput(tmpBufferM.allocTensor(UG2->M, UG2->K, getMajor(GEMM_NAME::UG2, 0)));
	// LN_FFN2 -> UG2
	UG2->setA(layerNormFFN2.getOutput()).setOutput(tmpBufferM.allocTensor(UG2->M, UG2->N, getMajor(GEMM_NAME::UG2, 2)));
	// UG2 -> activation2
	activation2.setInput(UG2->getD()).setOutput(tmpBufferM.allocTensor(D2->M, D2->K, getMajor(GEMM_NAME::D2, 0)));
	// activation2 -> D2
	D2->setA(activation2.getOutput()).setOutput(tmpBufferM.allocTensor(D2->M, D2->N, getMajor(GEMM_NAME::D2, 2)));
	D2->setC(AR_O2.getOutput());
	// D2 ->  D2_AR1 | D2_AR2
	const auto& AR12_D2_input = D2->getD().splitTensor(getDim(GEMM_NAME::D2, 2), KQV3->M, KQV4->M);
	const auto& shared_AR_output = tmpBufferM.allocTensor(KQV3->M + KQV4->M, KQV3->K, getMajor(GEMM_NAME::O2, 2));
	const auto& AR12_output = shared_AR_output.splitTensor(getDim(GEMM_NAME::O2, 2), KQV3->M, KQV4->M);

	const auto& AR12_before = tmpBufferM.allocTensor(O2->M, O2->N, getMajor(GEMM_NAME::O2, 2));
	AR12_before.clearContent();
	const auto& AR1_before_split = AR12_before.splitTensor(PllmDimension::ROW, KQV3->M, KQV4->M);

	AR1_D2.init(comm, connections, rank, nranks, AR12_D2_input[0], AR12_output[0], AR1_before_split[0]);
	AR2_D2.init(comm, connections, rank, nranks, AR12_D2_input[1], AR12_output[1], AR1_before_split[1]);
	// LN_Attention2_1/2 needs a shared contiguous buffer for the self-attention residual connection
	const auto& LN_Attn2_shared_output = tmpBufferM.allocTensor(KQV3->M + KQV4->M, KQV3->K, getMajor(GEMM_NAME::LOGITS2, 0));
	const auto& LN_Attn2_output = LN_Attn2_shared_output.splitTensor(getDim(GEMM_NAME::KQV1, 0), KQV3->M, KQV4->M);
	// TODO: O2 should setC(special column slice of LN_Attn2_shared_output)
	// e.g.,
	// const auto& O2_residule = tmpBufferM.allocSpan(O2->mn());
	// ExtractRankAsColumn.setInput(LN_Attn2_shared_output).setOutput(O2_residule);
	O2->setC(AR12_before);
	// D2_AR1 -> LN_Attention2_1 -> KQV3

	genEmbedding2_1.setOutput(tensor_cast<cutlass::half_t, half>(tmpBufferM.allocTensor(KQV3->M, KQV3->K, PllmLayout::ROW_MAJOR)));
	genEmbedding2_1_partial.setOutput(tensor_cast<cutlass::half_t, half>(AR1_before_split[0]
										.subtensor(AR1_before_split[0].dim1*rank/vnranks, AR1_before_split[0].dim1/vnranks)));
	layerNormAttention2_1.setInput(tensor_cast<half, cutlass::half_t>(genEmbedding2_1.getOutput())).setOutput(AR1_D2.getOutput());
	KQV3->setA(layerNormAttention2_1.getOutput());

	// D2_AR2 -> LN_Attention2_2 -> KQV4
	genEmbedding2_2.setOutput(tensor_cast<cutlass::half_t, half>(tmpBufferM.allocTensor(KQV4->M, KQV4->K, PllmLayout::ROW_MAJOR)));
	genEmbedding2_2_partial.setOutput(tensor_cast<cutlass::half_t, half>(AR1_before_split[1]
										.subtensor(AR1_before_split[1].dim1*rank/vnranks, AR1_before_split[1].dim1/vnranks)));
	layerNormAttention2_2.setInput(tensor_cast<half, cutlass::half_t>(genEmbedding2_2.getOutput())).setOutput(AR2_D2.getOutput());
	KQV4->setA(layerNormAttention2_2.getOutput());
	// KQV3 | KQV4 -> GEMV3 | GEMV4
	KQV3->setD(KQV_output_cutlass[2]);
	KQV4->setD(KQV_output_cutlass[3]);

	// connect setup output


	genEmbedding1.setOutput(tensor_cast<cutlass::half_t, half>(AR_D1.getOutput()));


	// connect logits generation

	layerNormModel1.setInput(AR_D1.getOutput()).setOutput(layerNormAttention1.getOutput());
	// layerNormModel2_1.setInput(AR1_D2.getOutput()).setOutput(layerNormAttention2_1.getOutput());
	// layerNormModel2_2.setInput(AR2_D2.getOutput()).setOutput(layerNormAttention2_2.getOutput());

	LOGITS1->setA(layerNormModel1.getOutput().getSubTensor(rank, vnranks, PllmDimension::ROW));
	LOGITS1->setOutput(tmpBufferM.allocTensor(LOGITS1->M, LOGITS1->N, getMajor(GEMM_NAME::LOGITS1, 2)));
	pllmTensor<int> sample_output1 = {(int*)tmpBufferM.alloc(LOGITS1->M * sizeof(int) / sizeof (half)), LOGITS1->M, 1, PllmLayout::ROW_MAJOR};
	maxSampler1.init(tensor_cast<cutlass::half_t, half>(LOGITS1->getD()), 
					 tensor_cast<cutlass::half_t, half>(tmpBufferM.allocTensor(LOGITS1->M, 1, PllmLayout::ROW_MAJOR)),
					 sample_output1); 

	LOGITS2->setA(shared_AR_output.getSubTensor(rank, vnranks, PllmDimension::ROW));
	LOGITS2->setOutput(tmpBufferM.allocTensor(LOGITS2->M, LOGITS2->N, getMajor(GEMM_NAME::LOGITS2, 2)));
	
	pllmTensor<int> sample_output2 = {(int*)tmpBufferM.alloc(LOGITS2->M * sizeof(int) / sizeof (half)), LOGITS2->M, 1, PllmLayout::ROW_MAJOR};
	maxSampler2.init(tensor_cast<cutlass::half_t, half>(LOGITS2->getD()), 
					 tensor_cast<cutlass::half_t, half>(tmpBufferM.allocTensor(LOGITS2->M, 1, PllmLayout::ROW_MAJOR)),
					 sample_output2);
}


void Pipeline::setWeight(int layer) {
	// Set weight before run the layer
	bool success = true;
	success &= O1->set_weight(input_data->weight.layer_weight[layer].W_O1);
	success &= O2->set_weight(input_data->weight.layer_weight[layer].W_O2);

	success &= UG1->set_weight(input_data->weight.layer_weight[layer].W_UG);
	success &= UG2->set_weight(input_data->weight.layer_weight[layer].W_UG);

	success &= D1->set_weight(input_data->weight.layer_weight[layer].W_D);
	success &= D2->set_weight(input_data->weight.layer_weight[layer].W_D);

	success &= KQV1->set_weight(input_data->weight.layer_weight[layer].W_KQV);
	success &= KQV2->set_weight(input_data->weight.layer_weight[layer].W_KQV);
	success &= KQV3->set_weight(input_data->weight.layer_weight[layer].W_KQV);
	success &= KQV4->set_weight(input_data->weight.layer_weight[layer].W_KQV);


	success &= layerNormAttention1.setWeight(input_data->weight.layer_weight[layer].W_LN_Attention);
	success &= layerNormAttention2_1.setWeight(input_data->weight.layer_weight[layer].W_LN_Attention);
	success &= layerNormAttention2_2.setWeight(input_data->weight.layer_weight[layer].W_LN_Attention);
	if (layer < MODEL_LAYER -1){
		success &= AR1_D2.setWeight(input_data->weight.layer_weight[layer+1].W_LN_Attention);
		success &= AR2_D2.setWeight(input_data->weight.layer_weight[layer+1].W_LN_Attention);
	} else{
		success &= AR1_D2.setWeight(input_data->weight.model_layernorm);
		success &= AR2_D2.setWeight(input_data->weight.model_layernorm);
	}


	success &= layerNormFFN1.setWeight(input_data->weight.layer_weight[layer].W_LN_FFN);
	success &= layerNormFFN2.setWeight(input_data->weight.layer_weight[layer].W_LN_FFN);

	if (!success) {
		spdlog::error("Failed to set weight for layer {}", layer);
	}
	if (layer == 0) {
		O2->set_beta(1);
	} else{
		O2->set_beta(1);
	}


	for (auto gemv : gemvs) {
		gemv->setKVData(input_data->kv_data[layer]);
	}
	prefill.setKVData(input_data->kv_data[layer]);
	roPEAppends[0].setKVData(input_data->kv_data[layer]);
	roPEAppends[1].setKVData(input_data->kv_data[layer]);
	roPEAppends[2].setKVData(input_data->kv_data[layer]);
	roPEAppends[3].setKVData(input_data->kv_data[layer]);	
}

double Pipeline::totalCompute() {
	double total = 0;
	for(auto gemm : gemms)
		total += gemm->totalCompute();
	return total;
}

void Pipeline::profileGEMM() {
	for(auto gemm : gemms)
		gemm->profile();
}

vortexOutputData Pipeline::run() {

	
	
	constexpr bool enableGraph = false;
	spdlog::info("Start run");
	setWeight(0);
	if(!enableGraph)
		CUDA_CHECK(cudaEventRecord(events[EventManager::GEMM_TIMING_START], stream_gemm));
	if(enableGraph) cudaStreamBeginCapture(stream_gemm, cudaStreamCaptureModeGlobal);
	cudaEventRecord(events[EventManager::CAPTURE_GEMM_START], stream_gemm);
	cudaStreamWaitEvent(stream_gemv, events[EventManager::CAPTURE_GEMM_START], 0);
	cudaStreamWaitEvent(stream_net, events[EventManager::CAPTURE_GEMM_START], 0);
	constexpr int totalIter = 10000;

	// setup phase
	genEmbedding1.run().log(private_logger);
	genEmbedding2_1.run().log(private_logger);
	genEmbedding2_2.run().log(private_logger);
	genEmbedding2_1_partial.run().log(private_logger);
	genEmbedding2_2_partial.run().log(private_logger);


	splitTensor.run().log(private_logger);

	AR_D1.recordEndEvent();
	AR2_D2.recordEndEvent();
	AR1_D2.recordEndEvent();
	
	for(int iter = 0; iter < RUN_LAYER; ++iter) {
		private_logger->info(">>>>>>>>>>>>>>>>>>>>>>>>>> layer: {}", iter);
		// new starting point

		setWeight(iter%MODEL_LAYER);
							
		layerNormAttention1.wait(AR_D1).wait(genEmbedding1).run().log(private_logger);
		if (iter == 0){				
			layerNormAttention2_1.wait(genEmbedding2_1).run().log(private_logger);
			layerNormAttention2_2.wait(genEmbedding2_2).run().log(private_logger);
		}else{
			layerNormAttention2_1.wait(AR1_D2).skip();
			layerNormAttention2_2.wait(AR2_D2).skip();
		}

		OperatorWrapper* kqv_deps[]={&layerNormAttention1, &layerNormAttention1, &layerNormAttention2_1, &layerNormAttention2_2};
		
		for (int kqv_idx = 0; kqv_idx < 4; kqv_idx++) {
			std::string idx_str = std::to_string(kqv_idx);
			KQV_ptrs[kqv_idx]->wait(*kqv_deps[kqv_idx]).run().log(private_logger);
			if (roPEAppends[kqv_idx].dense_batch_size > 0)
				roPEAppends[kqv_idx].run().log(private_logger);
			else
				roPEAppends[kqv_idx].skip();
			
			if (gemvs[kqv_idx]->batch_size > 0)
				gemvs[kqv_idx]->wait(roPEAppends[kqv_idx]).run().log(private_logger);
			else
				gemvs[kqv_idx]->wait(roPEAppends[kqv_idx]).skip();
		}

		if (update_data.prefillNum > 0)
		{
			prefill.wait(roPEAppends[3]).run().log(private_logger);
			// gemv_dep.incCounter(gemv_dep.device_GEMV_ready, update_token_num - update_data.decodePrefillBorder, stream_gemv);
		}
			
		// int endpoint = update_token_num;
		// if (endpoint > config_data.nanobatch_1_size)
		// 	endpoint = config_data.nanobatch_1_size;

		// gemv_dep.blockUntilGEMVReady(stream_net, endpoint); // need fix
		// cudaStreamWaitEvent(stream_net, events[EventManager::GEMV4_FINISH], 0);
		OperatorWrapper* AG_GEMV_dep;
		if (config_data.nanobatch_1_size == update_data.gemv_batch_size[0] + update_data.gemv_batch_size[1])
			AG_GEMV_dep = &GEMV2;
		else
			AG_GEMV_dep = &prefill;

		AG1_GEMV.setColumnwise().configRun(16, 1024, true).wait(AG_GEMV_dep).run().log(private_logger); // fix around
		

		O1->wait(splitTensor).wait(AG1_GEMV).run().log(private_logger); 
		
		AG_O1.setColumnwise().configRun(8, 1024, true).wait(O1).run().log(private_logger);

		layerNormFFN1.wait(AG_O1).run().log(private_logger);

		O2->wait_for_start(AG_O1).wait(GEMV4).wait(prefill).wait(KQV4).run().log(private_logger);

		AR_O2.configRun(8, 1024, true).wait(O2).run().log(private_logger);

		layerNormFFN2.wait(AR_O2).run().log(private_logger);


		// gemv_dep.clearAll(stream_net);

		UG1->wait_for_start(AR_O2).wait(layerNormFFN1).run().log(private_logger);

		activation1.run().log(private_logger);

		D1->run().log(private_logger);

		if (enable_offload) pageAgg.wait(activation1).run();



		UG2->wait(layerNormFFN2).run().log(private_logger);

		if (enable_offload) {
			cudaStreamWaitEvent(stream_cpy, UG2->start_event);
			int split = 1;
			for (int j = 0; j < split; j++)
				cudaMemcpyAsync(offloadKVCache+ j*2048*2*32*sizeof(half)*4/split, deviceOffloadKVCache, 2048*2*32*sizeof(half)*4/split, cudaMemcpyDeviceToHost, stream_cpy);
			for (int j = 0; j < split; j++)
				cudaMemcpyAsync(deviceLoadKVCache, offloadKVCache+ j*2048*2*32*sizeof(half) *4/split , 2048*2*32*sizeof(half)*4/split, cudaMemcpyHostToDevice, stream_cpy);
		}

		activation2.run().log(private_logger);

		AR_D1.configRun(8, 1024, true).wait_for_start(*UG2).run().log(private_logger);

		splitTensor.run().log(private_logger);
		

		if (enable_offload) cudaStreamWaitEvent(stream_cpy, AR_D1.end_event);
		if (enable_offload) pageDisp.run();

		D2->run().log(private_logger);
		
		AR1_D2.configRun(32, 1024, true).wait(D2).run().log(private_logger);

		AR2_D2.configRun(16, 1024, true).run().log(private_logger);
	}

	layerNormModel1.wait(AR_D1).run().log(private_logger);
	LOGITS1->wait(layerNormModel1).run().log(private_logger);
	maxSampler1.run().log(private_logger);

	LOGITS2->wait(AR1_D2).wait(AR2_D2).run().log(private_logger);
	maxSampler2.run().log(private_logger);

	cudaMemcpyAsync(outputTokens, input_data->tmp_buffer, 2048*sizeof(int), cudaMemcpyDeviceToHost, stream_cpy);
	cudaEventRecord(events[EventManager::CAPTURE_GEMV_END], stream_gemv);
	cudaEventRecord(events[EventManager::CAPTURE_NET_END], stream_net);
	cudaStreamWaitEvent(stream_gemm, events[EventManager::CAPTURE_GEMV_END], 0);
	cudaStreamWaitEvent(stream_gemm, events[EventManager::CAPTURE_NET_END], 0);

	// // End capture
	cudaGraph_t graph;
	if(enableGraph) {
		cudaStreamEndCapture(stream_gemm, &graph);
		if(graph == NULL) {
			spdlog::error("Failed to create graph");
			exit(1);
		}
		spdlog::info("Graph created");
		cudaGraphExec_t instance;
		cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
		spdlog::info("Graph instantiated");
		CUDA_CHECK(cudaEventRecord(events[EventManager::GEMM_TIMING_START], stream_gemm));
		for(int i = 0; i < 10; i ++)
			cudaGraphLaunch(instance, stream_gemm);
		spdlog::info("Graph launched");
	}
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
	double gflops = totalCompute() / runtime_ms / 1e6;
	double bandwidth = sizeof(__half) * (96731136 + 1.25 / 80 * 160 * 1024 * 1024 * 1024 / 2) /
					   (runtime_ms / 1000) / (1 << 30);

	spdlog::info("Total running cost (ms) of one microbatch is {}", runtime_ms);

	// Copy output data back to host
	CUDA_CHECK(cudaMemcpy(output_data.sampled_token_array1, maxSampler1.d_argMax.ptr, output_data.partial_num_1* sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(output_data.sampled_token_array2, maxSampler2.d_argMax.ptr, output_data.partial_num_2 * sizeof(int), cudaMemcpyDeviceToHost));

	return output_data;
}

Pipeline::~Pipeline() {
	cudaStreamDestroy(stream_gemm);
	cudaStreamDestroy(stream_gemv);
	cudaStreamDestroy(stream_net);
	cudaStreamDestroy(stream_other);
}


void Pipeline::GEMVOpUpdate() {

	auto getDim = [this](GEMM_NAME name, int idx) {return static_cast<PllmDimension>(getMajorType(config_data.gemm_op_tag[static_cast<int>(name)], idx));};
	spdlog::info("Update GEMV");

	uint32_t arr[] = {update_data.gemv_batch_size[0], update_data.gemv_batch_size[1], update_data.gemv_batch_size[2], update_data.gemv_batch_size[3], update_data.prefillNum};
	std::span<uint32_t, 5> batch_sizes(arr, 5);
	std::span<int32_t, 4> gemv_num_blocks(update_data.gemv_num_blocks, 4);
	auto total_batch_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0);
	assert (total_batch_size == update_data.decodePrefillBorder + update_data.prefillNum);

	pllmTensor kv_indptr = pllmTensor{update_data.kv_indptr, total_batch_size + 1};
	const auto& kv_indptr_split = kv_indptr.splitTensor(PllmDimension::ROW, batch_sizes,/*overlap suffix*/ 1U);

	pllmTensor kv_last_page_len = pllmTensor{update_data.kv_last_page_len, total_batch_size};
	const auto& kv_last_page_len_split = kv_last_page_len.splitTensor(PllmDimension::ROW, batch_sizes);

	pllmTensor kv_indices = pllmTensor{update_data.kv_indices, MAX_PAGE_NUM};

	pllmTensor input_ptr = pllmTensor{update_data.input_indptr, total_batch_size + 1};
	const auto& input_ptr_split = input_ptr.splitTensor(PllmDimension::ROW, batch_sizes,/*overlap suffix*/ 1U);

	pllmTensor rev_input_indptr = pllmTensor{update_data.rev_input_indptr, config_data.global_batch_size};
	pllmTensor per_token_offset = pllmTensor{update_data.per_token_offset, config_data.global_batch_size};

	const auto& GEMV_input = gemvQ.splitTensor(getDim(GEMM_NAME::KQV1, 2),
									batch_sizes[0],
									batch_sizes[1],
									batch_sizes[2],
									batch_sizes[3],
									MAX_BATCH_SIZE - update_data.decodePrefillBorder);

	const auto& GEMV_output = gemvAggregateOutput.splitTensor(getDim(GEMM_NAME::KQV1, 2),
									batch_sizes[0],
									batch_sizes[1],
									batch_sizes[2],
									batch_sizes[3],
									MAX_BATCH_SIZE - update_data.decodePrefillBorder);

	const int kdim = MODEL_HEAD_DIM * MODEL_KV_HEADS;
	const int qdim = MODEL_HEAD_DIM * MODEL_QO_HEADS;
	const int vdim = MODEL_HEAD_DIM * MODEL_KV_HEADS;

	int req_num = update_data.decodePrefillBorder + update_data.prefillNum;
	
	int* host_input_ptr = new int[req_num + 1]; // each request start from ptr[i] and end before ptr[i+1]
	cudaMemcpy(host_input_ptr, update_data.input_indptr, (req_num + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	update_token_num = host_input_ptr[req_num];
	const auto& Q_split = gemvQ.splitTensor(getDim(GEMM_NAME::KQV1, 2), KQV1->M, KQV2->M, KQV3->M, KQV4->M);
	const auto& KQV_split = KQV_output.splitTensor(getDim(GEMM_NAME::KQV1, 2), KQV1->M, KQV2->M, KQV3->M, KQV4->M);
	const auto& rev_input_indptr_split = rev_input_indptr.splitTensor(getDim(GEMM_NAME::KQV1, 2), KQV1->M, KQV2->M, KQV3->M, KQV4->M);
	const auto& per_token_offset_split = per_token_offset.splitTensor(getDim(GEMM_NAME::KQV1, 2), KQV1->M, KQV2->M, KQV3->M, KQV4->M);

	int token_remaining = update_token_num;
	for (int i = 0; i < 4; i++)
	{
		if (token_remaining > KQV_ptrs[i] -> M){
			roPEAppends[i].update(KQV_ptrs[i] -> M, KQV_split[i], Q_split[i], rev_input_indptr_split[i], per_token_offset_split[i], 
								kv_indices, kv_indptr, kv_last_page_len, gemv_dep.device_KQV_ready);
			token_remaining -= KQV_ptrs[i] -> M;
		}
		else{
			roPEAppends[i].update(token_remaining, KQV_split[i], Q_split[i], rev_input_indptr_split[i], per_token_offset_split[i], 
								kv_indices, kv_indptr, kv_last_page_len, gemv_dep.device_KQV_ready);
			token_remaining = 0;
		}
	}
	spdlog::info("RopeAppend Tokens: {}, {}, {}, {}", roPEAppends[0].dense_batch_size, roPEAppends[1].dense_batch_size, roPEAppends[2].dense_batch_size, roPEAppends[3].dense_batch_size);

	GEMV1.init(batch_sizes[0],
			gemv_num_blocks[0],
			kv_indptr_split[0],
			update_data.kv_indices,
			kv_last_page_len_split[0],
			GEMV_input[0],
			GEMV_output[0],
			gemv_dep.device_KQV_ready,
			gemv_dep.device_GEMV_ready);
	
	GEMV2.init(batch_sizes[1],
			gemv_num_blocks[1],
			kv_indptr_split[1],
			update_data.kv_indices,
			kv_last_page_len_split[1],
			GEMV_input[1],
			GEMV_output[1],
			gemv_dep.device_KQV_ready,
			gemv_dep.device_GEMV_ready);

	GEMV3.init(batch_sizes[2],
			gemv_num_blocks[2],
			kv_indptr_split[2],
			update_data.kv_indices,
			kv_last_page_len_split[2],
			GEMV_input[2],
			GEMV_output[2]);

	GEMV4.init(batch_sizes[3],
			gemv_num_blocks[3],
			kv_indptr_split[3],
			update_data.kv_indices,
			kv_last_page_len_split[3],
			GEMV_input[3],
			GEMV_output[3]);
	
	// ::log_tensor(spdlog::default_logger(), "input_ptr_split[4]", input_ptr_split[4], 1, 3);

	prefill.init(update_data.prefillNum,
				108,
				input_ptr_split[4],
				kv_indptr_split[4],
				update_data.kv_indices,
				kv_last_page_len_split[4],
				gemvQ,
				gemvAggregateOutput); 
				// use start address because prefill kernel get input use buffer[indptr[i]]

	// update page aggregation and page dispatch
	pageAgg.init(
		2,
		finished_idx,
		update_data.kv_indptr,
		update_data.kv_indices,
		deviceOffloadKVCache
	);
	pageAgg.setKVData(input_data->kv_data[0]);

	pageDisp.init(
		2,
		load_idx,
		update_data.kv_indptr,
		update_data.kv_indices,
		deviceLoadKVCache
	);
	pageDisp.setKVData(input_data->kv_data[0]);
	spdlog::info("finished GEMV update");	
}

void Pipeline::update(vortexUpdateData* update_data_) {


	this->update_data = *update_data_;

	// // debug 
	// spdlog::info("decode batch {}, prefill batch {}", update_data.decodePrefillBorder, update_data.prefillNum);
	// int total_batch = update_data.decodePrefillBorder + update_data.prefillNum;
	// int32_t * host_last_page_len = new int32_t[total_batch];
	// int32_t * host_indptr = new int32_t[total_batch + 1];
	// cudaMemcpy(host_last_page_len, update_data.kv_last_page_len, total_batch * sizeof(int32_t), cudaMemcpyDeviceToHost);
	// cudaMemcpy(host_indptr, update_data.kv_indptr, (total_batch + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);
	// int totalPage = host_indptr[total_batch];
	// int * host_indices = new int[totalPage];
	// cudaMemcpy(host_indices, update_data.kv_indices, totalPage * sizeof(int), cudaMemcpyDeviceToHost);
	// spdlog::info("indices: {}", formatCollection(host_indices, totalPage));
	// spdlog::info("last_page_len: {}", formatCollection(host_last_page_len, total_batch));
	// spdlog::info("indptr: {}", formatCollection(host_indptr, total_batch + 1));

	// int32_t * host_input_indptr = new int32_t[total_batch + 1];
	// cudaMemcpy(host_input_indptr, update_data.input_indptr, (total_batch + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);
	// spdlog::info("input_indptr: {}", formatCollection(host_input_indptr, total_batch + 1));
	
	// connect setup
	// split the tokens to 2 nano batch
	auto input_span = pllmTensor(update_data.input_tokens, config_data.global_batch_size, 1, PllmLayout::ROW_MAJOR);

	auto input_span1 = input_span.subtensor(0, config_data.nanobatch_1_size);

	auto input_span2_1 = input_span.subtensor(config_data.nanobatch_1_size, config_data.kqv3_size);
	auto input_span2_2 = input_span.subtensor(config_data.nanobatch_1_size + config_data.kqv3_size);

	auto partial_input_2_1 = input_span2_1.subtensor(input_span2_1.dim1 * rank / nranks, input_span2_1.dim1 / nranks);
	auto partial_input_2_2 = input_span2_2.subtensor(input_span2_2.dim1 * rank / nranks, input_span2_2.dim1 / nranks);

	

	genEmbedding1.setInput(input_span1);
	genEmbedding2_1.setInput(input_span2_1);
	genEmbedding2_2.setInput(input_span2_2);
	genEmbedding2_1_partial.setInput(partial_input_2_1);
	genEmbedding2_2_partial.setInput(partial_input_2_2);

	GEMVOpUpdate();

	// log first batch size
	spdlog::info("prefill: {}, decode: {}", update_data_->prefillNum, update_data_->decodePrefillBorder);
	spdlog::info("Batch size: {}, {}, {}, {}", update_data_->gemv_batch_size[0], update_data_->gemv_batch_size[1], update_data_->gemv_batch_size[2], update_data_->gemv_batch_size[3]);
}

void Pipeline::config(vortexConfigData* config_data){
	spdlog::info("Config pipeline");
	// // print all the members in config_data
	// spdlog::info("Config data: ");
	// spdlog::info("batch_size: {}", config_data->global_batch_size);
	// std::string s{};
	// for(auto i: config_data->gemm_op_tag) {
	// 	s += i + "   ";
	// }
	// spdlog::info("gemm_op_tag: {}", s);
	// spdlog::info("kqv_1: {}", config_data->kqv1_size);
	// spdlog::info("kqv_3: {}", config_data->kqv3_size);
	// spdlog::info("nanobatch: {}", config_data->nanobatch_1_size);

	this->config_data = * config_data;

	for(int i = 0; i < gemmNum; i++) {
		gemms[i] = generateGEMM(this->config_data.gemm_op_tag[i]);
		spdlog::info("GEMM {} created, tag: {}", i, this->config_data.gemm_op_tag[i]);
	}

	KQV_ptrs[0] = KQV1;
	KQV_ptrs[1] = KQV2;
	KQV_ptrs[2] = KQV3;
	KQV_ptrs[3] = KQV4;

	int globalbatch = config_data->global_batch_size;
	int nano1 = config_data->nanobatch_1_size;
	int nano2 = globalbatch - nano1;
	int kqv_batch[] = {config_data->kqv1_size, nano1 - config_data->kqv1_size, config_data->kqv3_size, nano2 - config_data->kqv3_size};

	O1 ->set_shape(nano1, MODEL_HIDDEN_DIM_PERGPU, MODEL_HIDDEN_DIM);
	O2 ->set_shape(nano2, MODEL_HIDDEN_DIM, MODEL_HIDDEN_DIM_PERGPU);
	UG1 ->set_shape(nano1, UG_N, MODEL_HIDDEN_DIM);
	UG2 ->set_shape(nano2, UG_N, MODEL_HIDDEN_DIM);
	D1 ->set_shape(nano1, MODEL_HIDDEN_DIM, MODEL_FF_DIM_GPU);
	D2 ->set_shape(nano2, MODEL_HIDDEN_DIM, MODEL_FF_DIM_GPU);
	KQV1 -> set_shape(kqv_batch[0], KQV_N, MODEL_HIDDEN_DIM);
	KQV2 -> set_shape(kqv_batch[1], KQV_N, MODEL_HIDDEN_DIM);
	KQV3 -> set_shape(kqv_batch[2], KQV_N, MODEL_HIDDEN_DIM);
	KQV4 -> set_shape(kqv_batch[3], KQV_N, MODEL_HIDDEN_DIM);
	LOGITS1 -> set_shape(nano1/nranks, 32000, MODEL_HIDDEN_DIM);
	LOGITS2 -> set_shape(nano2/nranks, 32000, MODEL_HIDDEN_DIM);


	activation1.config(nano1, UG_N/2);
	activation2.config(nano2, UG_N/2);

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

	NetOpInit();

	// init the output 
	output_data = vortexOutputData();
	output_data.partial_num_1 = config_data->nanobatch_1_size / nranks;
	output_data.partial_num_2 = (config_data->global_batch_size - config_data->nanobatch_1_size)/ nranks;
	output_data.sampled_token_array1 = new int[config_data->nanobatch_1_size];
	output_data.sampled_token_array2 = new int[config_data->global_batch_size - config_data->nanobatch_1_size];
	output_data.global_batch_size = config_data->global_batch_size;

	D2->updateEventExistance(true, true);
}