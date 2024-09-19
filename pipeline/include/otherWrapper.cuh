#pragma once
#include <cstdio>
#include <span>
#include "sleep.cuh"
#include <cuda.h>
#include "cutlass/cutlass.h"
#include "cuda_fp16.h"
#include "offloadKernel.cuh"
#include "vortexData.cuh"
#include "small_cuda_operator.cuh"
#include "rms_norm.cuh"
#include "flashinfer/pos_enc.cuh"
#include "config.h"
#include "operatorWrapper.cuh"


class OtherWrapper: public OperatorWrapper
{

};

// input_size / output_size == I_O_Ratio
template <size_t I_O_Ratio = 1>
class VectorOpWrapper : public OtherWrapper {
public:
	using Element = cutlass::half_t;
	Element* input = nullptr;
	Element* output = nullptr;
    pllmTensor<Element> pllm_tensor_input;
    pllmTensor<Element> pllm_tensor_output;
	size_t input_size = 0;
	size_t output_size = 0;

	VectorOpWrapper& setInput(pllmTensor<Element> input_tensor) {
		this->input = input_tensor.ptr;
		this->input_size = input_tensor.size();
        this->pllm_tensor_input = input_tensor;
		assert(output_size == 0 || input_size == output_size * I_O_Ratio);
		return *this;
	}

	VectorOpWrapper& setOutput(pllmTensor<Element> output_tensor) {
		this->output = output_tensor.ptr;
		this->output_size = output_tensor.size();
        this->pllm_tensor_output = output_tensor;
		assert(input_size == 0 || input_size == output_size * I_O_Ratio);
		return *this;
	}

	pllmTensor<Element> getOutput() {
		return pllm_tensor_output;
	}

    pllmTensor<Element> getInput() {
        return pllm_tensor_input;
    }

    virtual OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
        log_tensor(logger, name+" input", getInput(), 10, 20);
        log_tensor(logger, name+" output", getOutput(), 10, 20);
        return *this;
    }
    
};

class LayerNorm: public VectorOpWrapper<1>
{
    public:
    Element* weight = nullptr;
    pllmTensor<half> pllm_tensor_weight;
    // LayerNorm(Element* input, Element* weight, Element* output, int input_size, int output_size): 
    // input(input), weight(weight), output(output), input_size(input_size), output_size(output_size){}
    void init(Element* input, Element* weight, Element* output, int input_size, int output_size)
    {
        this -> input = input;
        this -> weight = weight;
        this -> output = output;
        this -> input_size = input_size;
        this -> output_size = output_size;
    }

    LayerNorm& setWeight(Element* weight)
    {
        this -> weight = weight;
        return *this;
    }

    bool setWeight(vortexWeight weight) {
        this->weight = (cutlass::half_t*)weight.ptr;
        pllm_tensor_weight = pllmTensor<half>(weight.ptr, size_t(1), weight.size(), PllmLayout::ROW_MAJOR);
        return true;
    }

    void work() override{

        assert(this->pllm_tensor_input.layout == PllmLayout::ROW_MAJOR);
        assert(this->pllm_tensor_output.layout == PllmLayout::ROW_MAJOR);
        assert(this->pllm_tensor_input.dimC == ModelConfig.model_hidden_dim);

        // spdlog::info("LayerNorm row {}, col {}", input_size / ModelConfig.model_hidden_dim, ModelConfig.model_hidden_dim);
        rms_norm((half*)output, (half*)input, (half*)weight, input_size / ModelConfig.model_hidden_dim, ModelConfig.model_hidden_dim, ModelConfig.rms_norm_eps, stream);
    }

    OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
        log_tensor(logger, name+" weight", pllm_tensor_input, 1, 20);
        log_tensor(logger, name+" input", pllm_tensor_input, 10, 20);
        log_tensor(logger, name+" output", pllm_tensor_output, 10, 20);
        return *this;
    }


};

class GenEmbedding : public OtherWrapper
{
    public:
    half* weight;
    int* tokens;
    half* output;

    pllmTensor<int> pllm_tensor_tokens;
    pllmTensor<half> pllm_tensor_output;
    pllmTensor<half> pllm_tensor_weight;

    int input_size;
    int output_size;

    void init(int* tokens, half* weight, half* output, int input_size, int output_size)
    {
        this -> tokens = tokens;
        this -> weight = weight;
        this -> output = output;
        this -> input_size = input_size;
        this -> output_size = output_size;
    }

    bool setWeight(vortexWeight weight) {
        this->weight = weight.ptr;
        pllm_tensor_weight = pllmTensor<half>(weight.ptr, weight.K, weight.N, PllmLayout::ROW_MAJOR);
        return true;
    }

    void setInput(pllmTensor<int> tokens_tensor) {
        this->tokens = tokens_tensor.ptr;
        this->input_size = tokens_tensor.size();
        this->pllm_tensor_tokens = tokens_tensor;
        if(output_size) {
            assert(input_size * ModelConfig.model_hidden_dim == output_size);
        } 
        // spdlog::info("GenEmbedding input size {}", input_size);
    }

    pllmTensor<half> getOutput() {
        return this->pllm_tensor_output;
    }

    void setOutput(pllmTensor<half> output) {
        this->output = output.ptr;
        this->output_size = output.size();
        this->pllm_tensor_output = output;
        if(input_size) {
            assert(input_size * ModelConfig.model_hidden_dim == output_size);
        }
    }

    void work(){
        dim3 block(128, 1, 1);
        dim3 grid(input_size, 1, 1);

        genEmbedding<<<grid, block, 0, stream>>>(tokens, weight, output, ModelConfig.model_hidden_dim);
    }

    OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
        log_tensor(logger, name+" weight", pllm_tensor_weight, 1, 20);
		log_tensor(logger, name+" tokens", pllm_tensor_tokens, 1, 20);
		log_tensor(logger, name+" output", pllm_tensor_output, 10, 20);
        return *this;
    }

};


class SplitKQVOutput : public OtherWrapper // row -> row , input and output
{
    public:
    half* kqv;
    half* k;
    half* q;
    half* v;

    pllmTensor<half> pllm_tensor_kqv;
    pllmTensor<half> pllm_tensor_k;
    pllmTensor<half> pllm_tensor_q;
    pllmTensor<half> pllm_tensor_v;

    int k_dim;
    int q_dim;
    int v_dim;
    int batch_size;

    void init(pllmTensor<half> kqv, pllmTensor<half> k, pllmTensor<half> q, pllmTensor<half> v, int batch_size, int k_dim, int q_dim, int v_dim)
    {
        this->pllm_tensor_kqv = kqv;
        this->pllm_tensor_k = k;
        this->pllm_tensor_q = q;
        this->pllm_tensor_v = v;
        assert(k.dimC = k_dim);
        assert(q.dimC = q_dim);
        assert(v.dimC = v_dim);
        
        this -> kqv = kqv.ptr;
        this -> k = k.ptr;
        this -> q = q.ptr;
        this -> v = v.ptr;
        this -> k_dim = k_dim;
        this -> q_dim = q_dim;
        this -> v_dim = v_dim;
        this -> batch_size = batch_size;
    }

    void work(){
        splitKQVOutput(kqv, batch_size, k, q, v, k_dim, q_dim, v_dim, stream);
    }

    OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
        log_tensor(logger, name+" kqv", pllm_tensor_kqv, 10, 20);
        log_tensor(logger, name+" k", pllm_tensor_k, 10, 20);
        log_tensor(logger, name+" q", pllm_tensor_q, 10, 20);
        log_tensor(logger, name+" v", pllm_tensor_v, 10, 20);
        return *this;
    }
};


class SplitTensor : public OtherWrapper // row -> row , input and output
{
    public:
    pllmTensor<half> input;
    pllmTensor<half> output;

    int nranks;
    int rank;
    int N;

    void init(pllmTensor<half> input, pllmTensor<half> output, int nranks, int rank)
    {
        this->input = input;
        this->output = output;
        this->nranks = nranks;
        this->rank = rank;
        this->N = output.dim2;
    }

    void work(){
        extractRankSubmatrixHalfDevice(input.ptr, output.ptr, input.dim1, N, nranks, rank, stream);
    }

    OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
        log_tensor(logger, name+" input", input, 10, 20);
        log_tensor(logger, name+" output", output, 10, 20);
        return *this;
    }
};

// Activation takes two matrices as input : the Up and Gate projection results.
// The output is: self.act_fn(self.gate_proj(x)) * self.up_proj(x))
// Thus the output dimension is half of the input.
class Activation: public VectorOpWrapper<2> 
{
    int M, N;
    public:
    void init(Element* input, Element* output, int input_size, int output_size)
    {
        this -> input = input;
        this -> output = output;
        this -> input_size = input_size;
        this -> output_size = output_size;
    }
    void config(int M, int N)
    {
        this -> M = M;
        this -> N = N;
    }
    void work() override{
        silu_and_multiply((half* )input, (half* )output, M, N, stream);
    }
};


class SleepWrapper: public OtherWrapper
{
    public:
    void work() override{
        cudaSleep<<<1,1, 0 , stream>>>(10);
    }

    void sleep(cudaStream_t stream, int us = 10){
        cudaSleep<<<1,1, 0 , stream>>>(us);
    }
};


class RoPEAppend : public OtherWrapper
{
    public:
    pllmTensor<half> kqv_input;
    pllmTensor<half> q_out;
    pllmTensor<int32_t> rev_input_indptr;
    pllmTensor<int32_t> per_token_offset;
    int dense_batch_size;

    pllmTensor<half> kv_data;
    pllmTensor<int32_t> kv_indices;
    pllmTensor<int32_t> kv_indptr;
    pllmTensor<int32_t> kv_last_page_len;
    int * kqv_ready_counter;

    RoPEAppend(){}

    void setKVData(half* kv_data)
    {
        this -> kv_data = pllmTensor<half>(kv_data, size_t(ModelConfig.max_page_num * ModelConfig.frame_page_size * 2), size_t(ModelConfig.model_head_dim * ModelConfig.model_kv_heads_gpu), PllmLayout::ROW_MAJOR);
    }

    void setKVData(pllmTensor<half> kv_data)
    {
        this -> kv_data = kv_data;
    }

    void update(int dense_batch_size, pllmTensor<half> kqv_input, pllmTensor<half> q_out,
                pllmTensor<int32_t> rev_input_indptr, pllmTensor<int32_t> per_token_offset, 
                pllmTensor<int32_t> kv_indices, pllmTensor<int32_t> kv_indptr, pllmTensor<int32_t> kv_last_page_len, int* kqv_ready_counter = nullptr)
    {
        this -> kqv_input = kqv_input; // sliced
        this -> q_out = q_out; // sliced
        this -> rev_input_indptr = rev_input_indptr; // sliced
        this -> per_token_offset = per_token_offset; // sliced
        this -> dense_batch_size = dense_batch_size;
        this -> kv_indices = kv_indices;
        this -> kv_indptr = kv_indptr;
        this -> kv_last_page_len = kv_last_page_len;
        this -> kqv_ready_counter = kqv_ready_counter;
    }
    
    void work(){
        constexpr flashinfer::QKVLayout kv_layout = flashinfer::QKVLayout::kNHD;
		using namespace flashinfer;

		int num_kv_heads = ModelConfig.model_kv_heads_gpu;
		int head_dim = ModelConfig.model_head_dim;
		int page_size = ModelConfig.frame_page_size;
		int num_qo_heads = ModelConfig.model_qo_heads_gpu;

		// building KV cache structure
		paged_kv_t<PageStorage::kIndices, kv_layout, half, int32_t> paged_kv(
			num_kv_heads,
			page_size,
			head_dim,
			0,  // dense batch_size used, no need for batch_size
			kv_data.ptr,
			kv_indices.ptr,
			kv_indptr.ptr,
			kv_last_page_len.ptr);

        flashinfer::splitRopeAppend(
            paged_kv, 
            kqv_input.ptr, 
            rev_input_indptr.ptr, 
            per_token_offset.ptr, 
            dense_batch_size, 
            num_qo_heads, 
            q_out.ptr,
            kqv_ready_counter, 
            ModelConfig.factor, 
            ModelConfig.rope_theta, 
            ModelConfig.smooth_a,
            ModelConfig.smooth_b,
            stream
        );
    }

    OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
        log_tensor(logger, name+" kqv", kqv_input, 10, 20);
        log_tensor(logger, name+" q_out", q_out, 10, 20);
        log_tensor(logger, name+" kv_data", kv_data, 32, 20, 102 * 32); // start row = 102*32 to show the last page of ropeAppend1
        log_tensor(logger, name+" rev_input_indptr", rev_input_indptr, 1, 512);
        log_tensor(logger, name+" per_token_offset", per_token_offset, 1, 512);
        log_tensor(logger, name+" kv_indices", kv_indices, 1, 512);
        log_tensor(logger, name+" kv_indptr", kv_indptr, 1, 512);
        log_tensor(logger, name+" kv_last_page_len", kv_last_page_len, 1, 512);
        return *this;
    }
};

class Transpose: public VectorOpWrapper<1>
{
    private:
        int rows;
        int cols;
    public:
    void config(int rows, int cols)
    {
        this -> rows = rows;
        this -> cols = cols;
    }

    void work() override{
        assert(static_cast<int>(this->getInput().layout) == 1 - static_cast<int>(this->getOutput().layout));
        transpose((half* )input, (half* )output, rows, cols, stream);
    }
};

class PageAggregator : public OtherWrapper
{
    public:
    int finished_req_num;
    int32_t* finished_idx;
    int32_t* kv_indptr;
    int32_t* kv_indices;
    half* output;
    half* kv_data;

    PageAggregator(){}
    void init(int finished_req_num, int32_t* finished_idx, int32_t* kv_indptr, int32_t* kv_indices, half* output)
    {
        this->finished_req_num = finished_req_num;
        this->finished_idx = finished_idx;
        this->kv_indptr = kv_indptr;
        this->kv_indices = kv_indices;
        this->output = output;
    }
	void setKVData(half* kv_data) {
		this->kv_data = kv_data;
	}

    void work() override{
        dim3 block(256, 1, 1);
        dim3 grid(8, 1, 1);
        moveKVcacheKernel<<<grid, block, 0, stream>>>(finished_req_num, finished_idx, kv_indptr, kv_indices, output, kv_data, ModelConfig.page_mem_size, false);
    }
};

class PageDispatcher : public OtherWrapper
{
    public:
    int load_req_num;
    int32_t* load_idx;
    int32_t* kv_indptr;
    int32_t* kv_indices;
    half* host_input;
    half* kv_data;

    PageDispatcher(){}

    void init(int load_req_num, int32_t* load_idx, int32_t* kv_indptr, int32_t* kv_indices, half* host_input)
    {
        this->load_req_num = load_req_num;
        this->load_idx = load_idx;
        this->kv_indptr = kv_indptr;
        this->kv_indices = kv_indices;
        this->host_input = host_input;
    }
	void setKVData(half* kv_data) {
		this->kv_data = kv_data;
	}

    void work() override{
        dim3 block(256, 1, 1);
        dim3 grid(8, 1, 1);
        moveKVcacheKernel<<<grid, block, 0, stream>>>(load_req_num, load_idx, kv_indptr, kv_indices, host_input, kv_data, ModelConfig.page_mem_size, true);
    }
};

class MaxSampling : public OtherWrapper
{
    public:
    pllmTensor<half> d_matrix;
    pllmTensor<half> d_maxVals;
    pllmTensor<int> d_argMax;
    int batch_size;

    void init(pllmTensor<half> d_matrix, pllmTensor<half> d_maxVals, pllmTensor<int> d_argMax)
    {
        this->d_matrix = d_matrix;
        this->d_maxVals = d_maxVals;
        this->d_argMax = d_argMax;
    }

    void work() override{
        computeRowMax(d_matrix.ptr, d_maxVals.ptr, d_argMax.ptr, batch_size, d_matrix.dim2, stream);
    }

    OperatorWrapper& set_batch_size(int batch_size){
        this->batch_size = batch_size;
        return *this;
    }

    OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
        log_tensor(logger, name+" d_matrix", d_matrix, 10, 20);
        log_tensor(logger, name+" d_maxVals", d_maxVals, 1, 20);
        log_tensor(logger, name+" d_argMax", d_argMax, 1, batch_size);
        return *this;
    }
};

class KeepToken : public OtherWrapper
{
    public:
    int req_num;

    pllmTensor<half> input;
    pllmTensor<half> output;
    int32_t * input_indptr;

    void update(int req_num, int32_t * input_indptr)
    {
        this->req_num = req_num;
        this->input_indptr = input_indptr;
    }

    void work() override{
        copySelectedRows(req_num, input.dim2, input_indptr,  input.ptr,  output.ptr, stream);
    }

    OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
        log_tensor(logger, name+" input", input, 10, 20);
        log_tensor(logger, name+" output", output, 10, 20);
        return *this;
    }

    KeepToken& setOutput(pllmTensor<half> output){
        this->output = output;
        return *this;
    }
    
    KeepToken& setInput(pllmTensor<half> input){
        this->input = input;
        return *this;
    }
};

class CopyTensor : public OtherWrapper
{
    public:
    pllmTensor<half> input;
    pllmTensor<half> output;


    void work() override{
        CUDA_CHECK(cudaMemcpyAsync(output.ptr, input.ptr, input.size() * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    }

    OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
        log_tensor(logger, name+" input", input, 10, 20);
        log_tensor(logger, name+" output", output, 10, 20);
        return *this;
    }

    CopyTensor& setOutput(pllmTensor<half> output){
        this->output = output;
        return *this;
    }
    
    CopyTensor& setInput(pllmTensor<half> input){
        this->input = input;
        return *this;
    }
};