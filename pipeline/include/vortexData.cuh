#pragma once
#include "cuda_fp16.h"
#include <vector>
#include <string>


struct vortexWeight{
    half* ptr;
    int N;
    int K;
    size_t size() const { return (size_t)N * K; }
};

struct vortexLayerWeight{
    vortexWeight W_O1;
    vortexWeight W_O2;
    vortexWeight W_U;
    vortexWeight W_G;
    vortexWeight W_D;
    vortexWeight W_KQV;
    vortexWeight B_KQV;
    vortexWeight W_LN_Attention;
    vortexWeight W_LN_FFN;
    vortexWeight W_ROT;
};

struct vortexModelWeight{
    vortexWeight lm_head;
    vortexWeight embedding;
    vortexWeight model_layernorm;
    std::vector<vortexLayerWeight> layer_weight;
};


struct vortexInitData
{

// KV cache
    half ** kv_data; // [max_pages, 2, page_size, num_head, head_dimension]
        // continuous KV cache space device pointer

// weight
    vortexModelWeight weight;
        // the device pointer to model weight
    size_t weight_size;
        // The number of element of the weight inside each layer (i.e., weight[i])

// tmp buffer
    half * tmp_buffer;
        // the device pointer to tmp buffer
    size_t tmp_buffer_size;
        // the number of half* elements in the tmp buffer
};

struct vortexUpdateData{
    // input tensor
    int decodePrefillBorder; 
        // [decodePrefillBorder: end] in input matrix is prefill request
    int prefillNum;
        // number of prefill requests
    int prefillTokensNum;
        // number of tokens in prefill requests
    int keepTokenListLength;
        // number of tokens in keep token list
    int *input_tokens; 
        // the device pointer to embedding for first layer
    int32_t * input_indptr;
        // accumulating prefix
    int32_t * rev_input_indptr;
        // reverse accumulating prefix
    int32_t * keep_token_list;
        // request index list of finished requests

    // KV cache
    int32_t * kv_indptr; // [batch_size + 1]
        // accumulating prefix
    int32_t * kv_indices; // [used_pages]
        // pointers to pages
    int32_t * kv_last_page_len; // [batch_size]
        // last page occupied length 
    int32_t * per_token_offset; // [batch_size]
        // previous length of each token 
    int32_t * gemv_batch_size; 
        // the host pointer to batch size
    int32_t * gemv_num_blocks;
        // the host pointer to number of blocks
    
};

struct vortexOutputData{
// output token id
    // int32_t * sampled_token_array1;
    // int32_t * sampled_token_array2;
    int32_t * sampled_token_array;
    int sampled_tokens;
        // the device pointer to sampled token array, one token for each request
    // int partial_num_1;
        // the number of tokens in the first nanobatch
    // int partial_num_2;
        // the device pointer to sampled token array, one token for each request
    half * offload_kv_cache;
    int global_batch_size;
};

struct vortexConfigData{
    std::vector<std::string> gemm_op_tag;
    int global_batch_size;
    int nanobatch_1_size;
    int kqv1_size;
    int kqv3_size;
};
vortexModelWeight modelWeightToGPU(vortexModelWeight& modelWeight, int rank);
void createInitData(vortexInitData& data, vortexModelWeight& weight, int rank);
void createUpdateData(vortexUpdateData& data, int rank, int global_batch_size, int avg_prefill_length, int avg_decode_length);
void createModelWeight(vortexModelWeight& data, int rank);
void createModelWeightCPU(vortexModelWeight& data, int rank);
void allocateKVData(vortexInitData& data, int rank);
void readConfigData(vortexConfigData& data, int rank, const std::string& filename);

template <size_t N>
void createConfigData(vortexConfigData& data, int rank, const std::array<const char*, N>& gemmConfig) {
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