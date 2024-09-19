#include "config.h"
#include "vortexData.cuh"
#include "pipeline.h"
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
using namespace flashinfer;
using json = nlohmann::json;

void allocateKVData(vortexInitData& data, int rank) {
	cudaSetDevice(rank);
	//init gemv
	size_t head_dim = ModelConfig.model_head_dim;
	size_t seqlen = 1024;
	size_t batch_size = ModelConfig.allocate_kv_data_batch;
	size_t page_size = ModelConfig.frame_page_size;
	size_t num_kv_heads = ModelConfig.model_kv_heads_gpu;

	// KV cache:
	auto pages_per_seq = (seqlen + page_size - 1) / page_size;
	auto num_pages = pages_per_seq * batch_size;

	half ** kv_datas = new half*[ModelConfig.model_layer];
	spdlog::info("allocate kv data try to allocate {} GB memory", 
		num_pages * 2 * num_kv_heads * page_size * head_dim * sizeof(half) * ModelConfig.model_layer / 1024.0 / 1024.0 / 1024.0);
	for (int i = 0; i < ModelConfig.model_layer; i++) {
		CUDA_CHECK(cudaMalloc(&kv_datas[i], num_pages * 2 * num_kv_heads * page_size * head_dim * sizeof(half)));
	}

	data.kv_data = kv_datas;
}

void createInitData(vortexInitData& data, vortexModelWeight& weight, int rank) {
	cudaSetDevice(rank);

	// weight
	size_t kWeightSize = ModelConfig.model_hidden_dim * ModelConfig.model_hidden_dim * 3;

	// temp
	half * temp;
	size_t kTempSize = ModelConfig.k_temp_size;
	spdlog::info("[createInitData]: kTempSize try to allocate {} GB memory", kTempSize * sizeof(half) / 1024.0 / 1024.0 / 1024.0);
	CUDA_CHECK(cudaMalloc(&temp, kTempSize * sizeof(half)));
	spdlog::info("allocate temp size = {} GB", kTempSize * sizeof(half) / 1024.0 / 1024.0 / 1024.0);



	data = {
		.weight = weight,
		.weight_size = kWeightSize,
		.tmp_buffer = temp,
		.tmp_buffer_size = kTempSize
	};
	
	allocateKVData(data, rank);
}




void createUpdateData(vortexUpdateData& data, int rank, int global_batch_size, int avg_prefill_length, int avg_decode_length) {
	cudaSetDevice(rank);
	// avg prefill length 1024, avg decode length 512
	spdlog::info("[createUpdateData]: global_batch_size = {}, avg_prefill_length = {}, avg_decode_length = {}", global_batch_size, avg_prefill_length, avg_decode_length);
	int decodePrefillBorder = static_cast<int>(global_batch_size * static_cast<float>(avg_decode_length) / (avg_decode_length + avg_prefill_length)); // 1024 * 1/3 real decode prefill border not nanobatch 1 size
	spdlog::info("decodePrefillBorder = {}", decodePrefillBorder);
	int prefillNum = (global_batch_size - decodePrefillBorder + avg_prefill_length - 1) / avg_prefill_length;// (global_batch_size - decodePrefillBorder + 1023) / 1024; // 1024-340 only contain 1 prefill (avg 1024)
	spdlog::info("prefillNum = {}", prefillNum);
	int denseBatch = global_batch_size; 

	int* input_tokens;
	CUDA_CHECK(cudaMallocHost(&input_tokens, denseBatch * sizeof(int)));
	int* host_input_tokens = new int[denseBatch];
	for (int i = 0; i < denseBatch; i++)
	{
		host_input_tokens[i] = i;
	}
	CUDA_CHECK(cudaMemcpy(input_tokens, host_input_tokens, denseBatch* sizeof(int), cudaMemcpyHostToDevice));
	
	std::vector<int32_t> prefillLengths;
	for (int i = 0; i < prefillNum - 1; i++)
	{
		prefillLengths.push_back(avg_prefill_length);
	}
	prefillLengths.push_back(denseBatch - decodePrefillBorder - (prefillNum - 1) * avg_prefill_length);

	std::vector<int32_t> input_indptr={0};
	for (int i = 0; i < decodePrefillBorder; i++) {
		input_indptr.push_back(i+1);
	}
	for (int i = 0; i < prefillNum; i++)
	{
		input_indptr.push_back(input_indptr.back() + prefillLengths[i]);
	}

	spdlog::info("input_indptr size = {}", input_indptr.size());
	// for (int i = 0; i < input_indptr.size(); i++)
	// {
	// 	spdlog::info("input_indptr[{}] = {}", i, input_indptr[i]);
	// }

	int32_t * input_indptr_device;
	CUDA_CHECK(cudaMallocHost(&input_indptr_device, (decodePrefillBorder + prefillNum + 1) * sizeof(int32_t)));
	CUDA_CHECK(cudaMemcpy(input_indptr_device, input_indptr.data(), (decodePrefillBorder + prefillNum + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));

	std::vector<int32_t> rev_input_indptr_host;
	for (int i = 0; i < decodePrefillBorder + prefillNum; i++)
	{
		for (int j = input_indptr[i]; j < input_indptr[i + 1]; j++)
		{
			rev_input_indptr_host.push_back(i);
		}
	}
	if (rev_input_indptr_host.size() != size_t(denseBatch))
	{
		spdlog::error("rev_input_indptr_host size is not equal to denseBatch");
	}
	
	int32_t * rev_input_indptr_device;
	CUDA_CHECK(cudaMallocHost(&rev_input_indptr_device, rev_input_indptr_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMemcpy(rev_input_indptr_device, rev_input_indptr_host.data(), rev_input_indptr_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

	std::vector<int32_t> per_token_offset_host;
	for (int i = 0; i < decodePrefillBorder + prefillNum; i++)
	{
		for (int j = input_indptr[i]; j < input_indptr[i + 1]; j++)
		{
			per_token_offset_host.push_back(j - input_indptr[i]);
		}
	}

	int32_t * per_token_offset_device;
	CUDA_CHECK(cudaMallocHost(&per_token_offset_device, per_token_offset_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMemcpy(per_token_offset_device, per_token_offset_host.data(), per_token_offset_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

	std::vector<int32_t> keep_token_list_host;
	for (int i = 0; i < decodePrefillBorder; i++)
	{
		keep_token_list_host.push_back(1);
	}

	int32_t * keep_token_list_device;
	CUDA_CHECK(cudaMallocHost(&keep_token_list_device, decodePrefillBorder * sizeof(int32_t)));
	CUDA_CHECK(cudaMemcpy(keep_token_list_device, keep_token_list_host.data(), decodePrefillBorder * sizeof(int32_t), cudaMemcpyHostToDevice));

	//init gemv
	size_t seqlen = 1024;
	size_t batch_size = 6000;
	size_t page_size = 16;

	// prev len
	std::vector<int32_t> prev_len_host;
	for(int i = 0; i < decodePrefillBorder + prefillNum; ++i) {
		prev_len_host.push_back(0);
	}
	int32_t * prev_len_device;
	CUDA_CHECK(cudaMallocHost(&prev_len_device, prev_len_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMemcpy(prev_len_device, prev_len_host.data(), prev_len_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

	// KV cache:
	auto pages_per_seq = (seqlen + page_size - 1) / page_size;
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

	CUDA_CHECK(cudaMallocHost(&kv_indptr_device, kv_indptr_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMallocHost(&kv_indices_device, kv_indicies_host.size() * sizeof(int32_t)));
	CUDA_CHECK(cudaMallocHost(&kv_last_page_len_device, kv_last_page_len_host.size() * sizeof(int32_t)));

	CUDA_CHECK(cudaMemcpy(kv_indptr_device, kv_indptr_host.data(), kv_indptr_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(kv_indices_device, kv_indicies_host.data(), kv_indicies_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(kv_last_page_len_device, kv_last_page_len_host.data(), kv_last_page_len_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));


	int32_t * gemv_batch_size = new int32_t[4]{256,decodePrefillBorder-256,0,0};
	int32_t * gemv_num_blocks = new int32_t[4]{108,108,108,108};
	data = {
		.decodePrefillBorder = decodePrefillBorder,
		.prefillNum = prefillNum,
		.prefillTokensNum = denseBatch - decodePrefillBorder,
		.keepTokenListLength = decodePrefillBorder,
		.input_tokens = input_tokens,
		.input_indptr = input_indptr_device,
		.rev_input_indptr = rev_input_indptr_device,
		.keep_token_list = keep_token_list_device,
		.kv_indptr = kv_indptr_device,
		.kv_indices = kv_indices_device,
		.kv_last_page_len = kv_last_page_len_device,
		.per_token_offset = per_token_offset_device,
		.gemv_batch_size = gemv_batch_size,
		.gemv_num_blocks = gemv_num_blocks,
	};
}

// void createUpdateData(vortexUpdateData& data, int rank) {
// 	cudaSetDevice(rank);
// 	int decodePrefillBorder = 1350;
// 	int prefillNum = 2;
// 	int denseBatch = 2048;

// 	int* input_tokens;
// 	CUDA_CHECK(cudaMallocHost(&input_tokens, denseBatch * sizeof(int)));
// 	int* host_input_tokens = new int[denseBatch];
// 	for (int i = 0; i < denseBatch; i++)
// 	{
// 		host_input_tokens[i] = i;
// 	}
// 	CUDA_CHECK(cudaMemcpy(input_tokens, host_input_tokens, denseBatch* sizeof(int), cudaMemcpyHostToDevice));
	

// 	int prefillLengths[] = { 512, denseBatch - decodePrefillBorder - 512};
// 	std::vector<int32_t> input_indptr={0};
// 	for (int i = 0; i < decodePrefillBorder; i++) {
// 		input_indptr.push_back(i+1);
// 	}
// 	for (size_t i = 0; i < prefillNum; i++)
// 	{
// 		input_indptr.push_back(input_indptr.back() + prefillLengths[i]);
// 	}

// 	int32_t * input_indptr_device;
// 	CUDA_CHECK(cudaMallocHost(&input_indptr_device, (decodePrefillBorder + prefillNum + 1) * sizeof(int32_t)));
// 	CUDA_CHECK(cudaMemcpy(input_indptr_device, input_indptr.data(), (decodePrefillBorder + prefillNum + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));

// 	std::vector<int32_t> rev_input_indptr_host;
// 	for (size_t i = 0; i < decodePrefillBorder + prefillNum; i++)
// 	{
// 		for (size_t j = input_indptr[i]; j < input_indptr[i + 1]; j++)
// 		{
// 			rev_input_indptr_host.push_back(i);
// 		}
// 	}
// 	if (rev_input_indptr_host.size() != denseBatch)
// 	{
// 		spdlog::error("rev_input_indptr_host size is not equal to denseBatch");
// 	}
	
// 	int32_t * rev_input_indptr_device;
// 	CUDA_CHECK(cudaMallocHost(&rev_input_indptr_device, rev_input_indptr_host.size() * sizeof(int32_t)));
// 	CUDA_CHECK(cudaMemcpy(rev_input_indptr_device, rev_input_indptr_host.data(), rev_input_indptr_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

// 	std::vector<int32_t> per_token_offset_host;
// 	for (size_t i = 0; i < decodePrefillBorder + prefillNum; i++)
// 	{
// 		for (size_t j = input_indptr[i]; j < input_indptr[i + 1]; j++)
// 		{
// 			per_token_offset_host.push_back(j - input_indptr[i]);
// 		}
// 	}

// 	int32_t * per_token_offset_device;
// 	CUDA_CHECK(cudaMallocHost(&per_token_offset_device, per_token_offset_host.size() * sizeof(int32_t)));
// 	CUDA_CHECK(cudaMemcpy(per_token_offset_device, per_token_offset_host.data(), per_token_offset_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

// 	std::vector<int32_t> keep_token_list_host;
// 	for (size_t i = 0; i < decodePrefillBorder; i++)
// 	{
// 		keep_token_list_host.push_back(1);
// 	}

// 	int32_t * keep_token_list_device;
// 	CUDA_CHECK(cudaMallocHost(&keep_token_list_device, decodePrefillBorder * sizeof(int32_t)));
// 	CUDA_CHECK(cudaMemcpy(keep_token_list_device, keep_token_list_host.data(), decodePrefillBorder * sizeof(int32_t), cudaMemcpyHostToDevice));

// 	//init gemv
// 	constexpr size_t head_dim = 128;
// 	size_t seqlen = 1024;
// 	size_t batch_size = 6000;
// 	size_t page_size = 16;
// 	size_t num_qo_heads = 8;
// 	size_t num_kv_heads = 1;

// 	// prev len
// 	std::vector<int32_t> prev_len_host;
// 	for(size_t i = 0; i < decodePrefillBorder + prefillNum; ++i) {
// 		prev_len_host.push_back(0);
// 	}
// 	int32_t * prev_len_device;
// 	CUDA_CHECK(cudaMallocHost(&prev_len_device, prev_len_host.size() * sizeof(int32_t)));
// 	CUDA_CHECK(cudaMemcpy(prev_len_device, prev_len_host.data(), prev_len_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

// 	// KV cache:
// 	auto pages_per_seq = (seqlen + page_size - 1) / page_size;
// 	auto num_pages = pages_per_seq * batch_size;
// 	std::vector<int32_t> kv_indptr_host{0};
// 	std::vector<int32_t> kv_indicies_host;
// 	std::vector<int32_t> kv_last_page_len_host;
// 	for(size_t i = 0; i < batch_size; ++i) {
// 		for(size_t p = 0; p < pages_per_seq; ++p) {
// 			kv_indicies_host.push_back(i * pages_per_seq + p);
// 		}
// 		kv_indptr_host.push_back(kv_indptr_host.back() + pages_per_seq);
// 		kv_last_page_len_host.push_back((seqlen - 1) % page_size + 1);
// 	}

// 	int32_t * kv_indptr_device;
// 	int32_t * kv_indices_device;
// 	int32_t * kv_last_page_len_device;

// 	CUDA_CHECK(cudaMallocHost(&kv_indptr_device, kv_indptr_host.size() * sizeof(int32_t)));
// 	CUDA_CHECK(cudaMallocHost(&kv_indices_device, kv_indicies_host.size() * sizeof(int32_t)));
// 	CUDA_CHECK(cudaMallocHost(&kv_last_page_len_device, kv_last_page_len_host.size() * sizeof(int32_t)));

// 	CUDA_CHECK(cudaMemcpy(kv_indptr_device, kv_indptr_host.data(), kv_indptr_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
// 	CUDA_CHECK(cudaMemcpy(kv_indices_device, kv_indicies_host.data(), kv_indicies_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
// 	CUDA_CHECK(cudaMemcpy(kv_last_page_len_device, kv_last_page_len_host.data(), kv_last_page_len_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));


// 	int32_t * gemv_batch_size = new int32_t[4]{256,384,1350-256-384,0};
// 	int32_t * gemv_num_blocks = new int32_t[4]{108,108,108,108};
// 	data = {
// 		.decodePrefillBorder = decodePrefillBorder,
// 		.prefillNum = prefillNum,
// 		.prefillTokensNum = denseBatch - decodePrefillBorder,
// 		.keepTokenListLength = decodePrefillBorder,
// 		.input_tokens = input_tokens,
// 		.input_indptr = input_indptr_device,
// 		.rev_input_indptr = rev_input_indptr_device,
// 		.keep_token_list = keep_token_list_device,
// 		.kv_indptr = kv_indptr_device,
// 		.kv_indices = kv_indices_device,
// 		.kv_last_page_len = kv_last_page_len_device,
// 		.per_token_offset = per_token_offset_device,
// 		.gemv_batch_size = gemv_batch_size,
// 		.gemv_num_blocks = gemv_num_blocks,
// 	};
// }

vortexWeight weightToGPU(vortexWeight& weight, int rank)
{
	vortexWeight weightGPU;
	weightGPU = weight;
	cudaSetDevice(rank);
	CUDA_CHECK(cudaMalloc(&weightGPU.ptr, weight.size() * sizeof(half)));
	CUDA_CHECK(cudaMemcpy(weightGPU.ptr, weight.ptr, weight.size() * sizeof(half), cudaMemcpyHostToDevice));
	return weightGPU;
}

// pybind , pipeline init
vortexModelWeight modelWeightToGPU(vortexModelWeight& modelWeight, int rank)
{
	vortexModelWeight modelWeightGPU;
	cudaSetDevice(rank);
	modelWeightGPU.lm_head = weightToGPU(modelWeight.lm_head, rank);
	modelWeightGPU.embedding = weightToGPU(modelWeight.embedding, rank);
	modelWeightGPU.model_layernorm = weightToGPU(modelWeight.model_layernorm, rank);
	modelWeightGPU.layer_weight.resize(modelWeight.layer_weight.size());
	for (size_t i = 0; i < modelWeight.layer_weight.size(); i++)
	{
		modelWeightGPU.layer_weight[i].W_U = weightToGPU(modelWeight.layer_weight[i].W_U, rank);
		modelWeightGPU.layer_weight[i].W_G = weightToGPU(modelWeight.layer_weight[i].W_G, rank);
		modelWeightGPU.layer_weight[i].W_D = weightToGPU(modelWeight.layer_weight[i].W_D, rank);
		modelWeightGPU.layer_weight[i].W_KQV = weightToGPU(modelWeight.layer_weight[i].W_KQV, rank);
		modelWeightGPU.layer_weight[i].B_KQV = weightToGPU(modelWeight.layer_weight[i].B_KQV, rank);
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

// cuda main compute main
void createModelWeight(vortexModelWeight& modelWeight,int rank)
{
	cudaSetDevice(rank);
	modelWeight.lm_head = createWeight(ModelConfig.model_hidden_dim, ModelConfig.vocab_size);
	modelWeight.embedding = createWeight(ModelConfig.vocab_size, ModelConfig.model_hidden_dim);
	modelWeight.model_layernorm = createWeight(ModelConfig.model_hidden_dim, 1);
	modelWeight.layer_weight.resize(ModelConfig.model_layer);
	for (int i = 0; i < ModelConfig.model_layer; i++)
	{
		modelWeight.layer_weight[i].W_O1 = createWeight(ModelConfig.model_hidden_dim_pergpu, ModelConfig.model_hidden_dim);
		modelWeight.layer_weight[i].W_O2 = createWeight(ModelConfig.model_hidden_dim, ModelConfig.model_hidden_dim_pergpu );
		modelWeight.layer_weight[i].W_U = createWeight(ModelConfig.model_ff_dim_gpu, ModelConfig.model_hidden_dim);
		modelWeight.layer_weight[i].W_G = createWeight(ModelConfig.model_ff_dim_gpu, ModelConfig.model_hidden_dim);
		modelWeight.layer_weight[i].W_D = createWeight(ModelConfig.model_hidden_dim, ModelConfig.model_ff_dim_gpu);
		modelWeight.layer_weight[i].W_KQV = createWeight(ModelConfig.kqv_n, ModelConfig.model_hidden_dim);
		modelWeight.layer_weight[i].B_KQV = createWeight(1, ModelConfig.kqv_n);
		modelWeight.layer_weight[i].W_LN_Attention = createWeight(ModelConfig.model_hidden_dim, 1);
		modelWeight.layer_weight[i].W_LN_FFN = createWeight(ModelConfig.model_hidden_dim, 1);
		modelWeight.layer_weight[i].W_ROT = createWeight(64, 1);
	}
}

vortexWeight createWeightCPU(int N, int K)
{
	vortexWeight weight;
	weight.ptr = new half[N * K];
	weight.N = N;
	weight.K = K;
	return weight;
}

// python
void createModelWeightCPU(vortexModelWeight& modelWeight,int rank)
{
	cudaSetDevice(rank);
	modelWeight.lm_head = createWeightCPU(ModelConfig.model_hidden_dim, ModelConfig.vocab_size);
	modelWeight.embedding = createWeightCPU(ModelConfig.vocab_size, ModelConfig.model_hidden_dim);
	modelWeight.model_layernorm = createWeightCPU(ModelConfig.model_hidden_dim, 1);
	modelWeight.layer_weight.resize(ModelConfig.model_layer);
	for (int i = 0; i < ModelConfig.model_layer; i++)
	{
		modelWeight.layer_weight[i].W_O1 = createWeightCPU(ModelConfig.model_hidden_dim_pergpu, ModelConfig.model_hidden_dim);
		modelWeight.layer_weight[i].W_O2 = createWeightCPU(ModelConfig.model_hidden_dim, ModelConfig.model_hidden_dim_pergpu );
		modelWeight.layer_weight[i].W_U = createWeightCPU(ModelConfig.model_ff_dim_gpu, ModelConfig.model_hidden_dim);
		modelWeight.layer_weight[i].W_G = createWeightCPU(ModelConfig.model_ff_dim_gpu, ModelConfig.model_hidden_dim);
		modelWeight.layer_weight[i].W_D = createWeightCPU(ModelConfig.model_hidden_dim, ModelConfig.model_ff_dim_gpu);
		modelWeight.layer_weight[i].W_KQV = createWeightCPU(ModelConfig.kqv_n, ModelConfig.model_hidden_dim);
		modelWeight.layer_weight[i].B_KQV = createWeightCPU(1, ModelConfig.kqv_n);
		modelWeight.layer_weight[i].W_LN_Attention = createWeightCPU(ModelConfig.model_hidden_dim, 1);
		modelWeight.layer_weight[i].W_LN_FFN = createWeightCPU(ModelConfig.model_hidden_dim, 1);
		modelWeight.layer_weight[i].W_ROT = createWeightCPU(64, 1);
	}
}

void readConfigData(vortexConfigData& data, int rank, const std::string& filename) {
	spdlog::info("enter readConfigData");
	std::ifstream file(filename); 
	if (!file.is_open()) { 
		std::cerr << "Could not open the file: " << filename << std::endl;
		return; 
	} 
    // Read the file contents into a string 
	spdlog::info("read config file");
	json j; 
	file >> j; 
	// Close the file 
	file.close();
	// Access the "gemm_op_tag" array
	spdlog::info("access gemm_op_tag");
    if (j["pipeline_configs"]["gemm_op_tag"].is_array()) {
        std::vector<std::string> gemm_op_tags = j["pipeline_configs"]["gemm_op_tag"].get<std::vector<std::string>>();
		data.gemm_op_tag = gemm_op_tags;
		spdlog::info("tag0 = {}, tag1 = {}, tag2 = {}, tag3 = {}", data.gemm_op_tag[0], data.gemm_op_tag[1], data.gemm_op_tag[2], data.gemm_op_tag[3]);
    } else {
        std::cerr << "\"gemm_op_tag\" not found or not an array!" << std::endl;
    }
	data.global_batch_size = j["pipeline_configs"]["global_batch_size"].get<int>();
	spdlog::info("global_batch_size = {}", data.global_batch_size);
	data.nanobatch_1_size = j["pipeline_configs"]["nanobatch_1_size"].get<int>();
	spdlog::info("nanobatch_1_size = {}", data.nanobatch_1_size);
	data.kqv1_size = j["pipeline_configs"]["kqv1_size"].get<int>();
	spdlog::info("kqv1_size = {}", data.kqv1_size);
	data.kqv3_size = j["pipeline_configs"]["kqv3_size"].get<int>();
	spdlog::info("kqv3_size = {}", data.kqv3_size);
}
	