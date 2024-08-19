#include "config.h"
#include "vortexData.cuh"
#include "pipeline.h"
using namespace flashinfer;

void allocateKVData(vortexInitData& data, int rank) {
	cudaSetDevice(rank);
	//init gemv
	constexpr size_t head_dim = 128;
	size_t seqlen = 1024;
	size_t batch_size = ALLOCATE_KV_DATA_BATCH;
	size_t page_size = 16;
	size_t num_qo_heads = 8;
	size_t num_kv_heads = 1;

	// KV cache:
	auto pages_per_seq = (seqlen + page_size - 1) / page_size;
	auto num_pages = pages_per_seq * batch_size;

	half ** kv_datas = new half*[MODEL_LAYER];
	for (int i = 0; i < MODEL_LAYER; i++) {
		CUDA_CHECK(cudaMalloc(&kv_datas[i], num_pages * 2 * num_kv_heads * page_size * head_dim * sizeof(half)));
	}

	data.kv_data = kv_datas;
}

void createInitData(vortexInitData& data, int rank) {
	cudaSetDevice(rank);

	// weight
	constexpr size_t kWeightSize = MODEL_HIDDEN_DIM * MODEL_HIDDEN_DIM * 3;
	auto weight = vortexModelWeight();
	createModelWeight(weight, rank);

	// temp
	half * temp;
	constexpr size_t kTempSize = MODEL_HIDDEN_DIM * MODEL_HIDDEN_DIM * 4;
	CUDA_CHECK(cudaMalloc(&temp, kTempSize * sizeof(half)));
	spdlog::info("allocate temp size = {} GB", kTempSize * sizeof(half) / 1024.0 / 1024.0 / 1024.0);


	data = {
		.kv_data = nullptr,
		.weight = weight,
		.weight_size = kWeightSize,
		.tmp_buffer = temp,
		.tmp_buffer_size = kTempSize
	};
}


void createConfigData(vortexConfigData& data, int rank) {
	cudaSetDevice(rank);
	std::vector<std::string> opname;
	for (auto i : gemmConfig)
	{
		opname.push_back(i);
	}
	data.gemm_op_tag = opname;
	data.global_batch_size = 2048;
	data.nanobatch_1_size = 640;
	data.kqv1_size = 256;
	data.kqv3_size = 768;
}


void createUpdateData(vortexUpdateData& data, int rank) {
	cudaSetDevice(rank);
	int decodePrefillBorder = 1350;
	int prefillNum = 2;
	int denseBatch = 2048;

	int* input_tokens;
	CUDA_CHECK(cudaMalloc(&input_tokens, denseBatch * sizeof(int)));
	int* host_input_tokens = new int[denseBatch];
	for (int i = 0; i < denseBatch; i++)
	{
		host_input_tokens[i] = i;
	}
	CUDA_CHECK(cudaMemcpy(input_tokens, host_input_tokens, denseBatch* sizeof(int), cudaMemcpyHostToDevice));
	

	int prefillLengths[] = { 512, denseBatch - decodePrefillBorder - 512};
	std::vector<int32_t> input_indptr={0};
	for (int i = 0; i < decodePrefillBorder; i++) {
		input_indptr.push_back(i+1);
	}
	for (size_t i = 0; i < prefillNum; i++)
	{
		input_indptr.push_back(input_indptr.back() + prefillLengths[i]);
	}

	int32_t * input_indptr_device;
	CUDA_CHECK(cudaMalloc(&input_indptr_device, (decodePrefillBorder + prefillNum + 1) * sizeof(int32_t)));
	CUDA_CHECK(cudaMemcpy(input_indptr_device, input_indptr.data(), (decodePrefillBorder + prefillNum + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));

	std::vector<int32_t> rev_input_indptr_host;
	for (size_t i = 0; i < decodePrefillBorder + prefillNum; i++)
	{
		for (size_t j = input_indptr[i]; j < input_indptr[i + 1]; j++)
		{
			rev_input_indptr_host.push_back(i);
		}
	}
	if (rev_input_indptr_host.size() != denseBatch)
	{
		spdlog::error("rev_input_indptr_host size is not equal to denseBatch");
	}
	
	int32_t * rev_input_indptr_device;
	CUDA_CHECK(cudaMalloc(&rev_input_indptr_device, rev_input_indptr_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMemcpy(rev_input_indptr_device, rev_input_indptr_host.data(), rev_input_indptr_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

	std::vector<int32_t> per_token_offset_host;
	for (size_t i = 0; i < decodePrefillBorder + prefillNum; i++)
	{
		for (size_t j = input_indptr[i]; j < input_indptr[i + 1]; j++)
		{
			per_token_offset_host.push_back(j - input_indptr[i]);
		}
	}

	int32_t * per_token_offset_device;
	CUDA_CHECK(cudaMalloc(&per_token_offset_device, per_token_offset_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMemcpy(per_token_offset_device, per_token_offset_host.data(), per_token_offset_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));


	//init gemv
	constexpr size_t head_dim = 128;
	size_t seqlen = 1024;
	size_t batch_size = 6000;
	size_t page_size = 16;
	size_t num_qo_heads = 8;
	size_t num_kv_heads = 1;

	// prev len
	std::vector<int32_t> prev_len_host;
	for(size_t i = 0; i < decodePrefillBorder + prefillNum; ++i) {
		prev_len_host.push_back(0);
	}
	int32_t * prev_len_device;
	CUDA_CHECK(cudaMalloc(&prev_len_device, prev_len_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMemcpy(prev_len_device, prev_len_host.data(), prev_len_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

	// KV cache:
	auto pages_per_seq = (seqlen + page_size - 1) / page_size;
	auto num_pages = pages_per_seq * batch_size;
	std::vector<int32_t> kv_indptr_host{0};
	std::vector<int32_t> kv_indicies_host;
	std::vector<int32_t> kv_last_page_len_host;
	for(size_t i = 0; i < batch_size; ++i) {
		for(size_t p = 0; p < pages_per_seq; ++p) {
			kv_indicies_host.push_back(i * pages_per_seq + p);
		}
		kv_indptr_host.push_back(kv_indptr_host.back() + pages_per_seq);
		kv_last_page_len_host.push_back((seqlen - 1) % page_size + 1);
	}

	int32_t * kv_indptr_device;
	int32_t * kv_indices_device;
	int32_t * kv_last_page_len_device;

	CUDA_CHECK(cudaMalloc(&kv_indptr_device, kv_indptr_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMalloc(&kv_indices_device, kv_indicies_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMalloc(&kv_last_page_len_device, kv_last_page_len_host.size() * sizeof(int32_t)));

	CUDA_CHECK(cudaMemcpy(kv_indptr_device, kv_indptr_host.data(), kv_indptr_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(kv_indices_device, kv_indicies_host.data(), kv_indicies_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(kv_last_page_len_device, kv_last_page_len_host.data(), kv_last_page_len_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));


	int32_t * gemv_batch_size = new int32_t[4]{256,384,1350-265-384,0};
	int32_t * gemv_num_blocks = new int32_t[4]{108,108,108,108};
	data = {
		.decodePrefillBorder = decodePrefillBorder,
		.prefillNum = prefillNum,
		.input_tokens = input_tokens,
		.input_indptr = input_indptr_device,
		.rev_input_indptr = rev_input_indptr_device,
		.kv_indptr = kv_indptr_device,
		.kv_indices = kv_indices_device,
		.kv_last_page_len = kv_last_page_len_device,
		.per_token_offset = per_token_offset_device,
		.gemv_batch_size = gemv_batch_size,
		.gemv_num_blocks = gemv_num_blocks,
	};
}

vortexWeight weightToGPU(vortexWeight& weight, int rank)
{
	vortexWeight weightGPU;
	weightGPU = weight;
	cudaSetDevice(rank);
	CUDA_CHECK(cudaMalloc(&weightGPU.ptr, weight.size() * sizeof(half)));
	CUDA_CHECK(cudaMemcpy(weightGPU.ptr, weight.ptr, weight.size() * sizeof(half), cudaMemcpyHostToDevice));
	return weightGPU;
}

vortexModelWeight modelWeightToGPU(vortexModelWeight& modelWeight, int rank)
{
	vortexModelWeight modelWeightGPU;
	cudaSetDevice(rank);
	modelWeightGPU.lm_head = weightToGPU(modelWeight.lm_head, rank);
	modelWeightGPU.embedding = weightToGPU(modelWeight.embedding, rank);
	modelWeightGPU.model_layernorm = weightToGPU(modelWeight.model_layernorm, rank);
	modelWeightGPU.layer_weight.resize(modelWeight.layer_weight.size());
	for (int i = 0; i < modelWeight.layer_weight.size(); i++)
	{
		modelWeightGPU.layer_weight[i].W_UG = weightToGPU(modelWeight.layer_weight[i].W_UG, rank);
		modelWeightGPU.layer_weight[i].W_D = weightToGPU(modelWeight.layer_weight[i].W_D, rank);
		modelWeightGPU.layer_weight[i].W_KQV = weightToGPU(modelWeight.layer_weight[i].W_KQV, rank);
		modelWeightGPU.layer_weight[i].W_O1 = weightToGPU(modelWeight.layer_weight[i].W_O1, rank);
		modelWeightGPU.layer_weight[i].W_O2 = weightToGPU(modelWeight.layer_weight[i].W_O2, rank);
		modelWeightGPU.layer_weight[i].W_LN_Attention = weightToGPU(modelWeight.layer_weight[i].W_LN_Attention, rank);
		modelWeightGPU.layer_weight[i].W_LN_FFN = weightToGPU(modelWeight.layer_weight[i].W_LN_FFN, rank);
		modelWeightGPU.layer_weight[i].W_ROT = weightToGPU(modelWeight.layer_weight[i].W_ROT, rank);
	}
	return modelWeightGPU;
}


vortexWeight createWeight(int N, int K)
{
	vortexWeight weight;
	CUDA_CHECK(cudaMalloc(&weight.ptr, N * K * sizeof(half)));
	weight.N = N;
	weight.K = K;
	return weight;
}

void createModelWeight(vortexModelWeight& modelWeight,int rank)
{
	cudaSetDevice(rank);
	modelWeight.lm_head = createWeight(MODEL_HIDDEN_DIM, 32000);
	modelWeight.embedding = createWeight(32000, MODEL_HIDDEN_DIM);
	modelWeight.model_layernorm = createWeight(MODEL_HIDDEN_DIM, 1);
	modelWeight.layer_weight.resize(MODEL_LAYER);
	for (int i = 0; i < MODEL_LAYER; i++)
	{
		modelWeight.layer_weight[i].W_O1 = createWeight(MODEL_HIDDEN_DIM_PERGPU, MODEL_HIDDEN_DIM);
		modelWeight.layer_weight[i].W_O2 = createWeight(MODEL_HIDDEN_DIM, MODEL_HIDDEN_DIM_PERGPU );
		modelWeight.layer_weight[i].W_UG = createWeight(UG_N, MODEL_HIDDEN_DIM);
		modelWeight.layer_weight[i].W_D = createWeight(MODEL_HIDDEN_DIM, MODEL_FF_DIM_GPU);
		modelWeight.layer_weight[i].W_KQV = createWeight(KQV_N, MODEL_HIDDEN_DIM);
		modelWeight.layer_weight[i].W_LN_Attention = createWeight(MODEL_HIDDEN_DIM, 1);
		modelWeight.layer_weight[i].W_LN_FFN = createWeight(MODEL_HIDDEN_DIM, 1);
		modelWeight.layer_weight[i].W_ROT = createWeight(64, 1);
	}
}