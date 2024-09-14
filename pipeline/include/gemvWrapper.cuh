#pragma once

#include <span>

#include "config.h"
#include "sleep.cuh"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <decode_attention_decl.cuh>
#include <prefill_attention_decl.cuh>
#include "operatorWrapper.cuh"


class GemvWrapper : public OperatorWrapper{
public:
	using Element = half;
	uint32_t batch_size;
	int32_t block_num;

	Element* kv_data; // [max_pages, 2, page_size, num_head, head_dimension]
		// continuous KV cache space device pointer
	int32_t* kv_indptr; // [batch_size + 1]
		// accumulating prefix
	int32_t* kv_indices; // [used_pages]
		// pointers to pages
	int32_t* kv_last_page_len; // [batch_size]
		// last page occupied length
	Element* q;
	Element* o;

	const size_t workspace_size_in_bytes = 32 * 1024 * 1024;
	thrust::device_vector<char> buffer;
	flashinfer::BatchPrefillHandler handler;

	GemvWrapper() {
		buffer = thrust::device_vector<char>(workspace_size_in_bytes);
	}

	void setKVData(Element* kv_data) {
		this->kv_data = kv_data;
	}

	static size_t getInputSize(uint32_t batch_size) {
		return (2 * ModelConfig.model_kv_heads_gpu + ModelConfig.model_qo_heads_gpu) * ModelConfig.model_head_dim * batch_size;
	}
	size_t getInputSize() { return getInputSize(batch_size); }
	static size_t getOutputSize(uint32_t batch_size) {
		return ModelConfig.model_qo_heads_gpu * ModelConfig.model_head_dim * batch_size;
	}
	size_t getOutputSize() { return getOutputSize(batch_size); }
};

class DecodeGemvWrapper : public GemvWrapper {
public:
	std::vector<int32_t> qo_indptr_h{0};
	thrust::device_vector<int32_t> qo_indptr_d;
	pllmTensor<int32_t> qo_indptr_tensor;
	int32_t* qo_indptr;
	int* device_KQV_ready;
	int* device_GEMV_ready;

	pllmTensor<int32_t> kv_indptr_tensor;
	pllmTensor<int32_t> kv_last_page_len_tensor;
	pllmTensor<half> input_tensor;
	pllmTensor<half> output_tensor;
	
	DecodeGemvWrapper() {

		for(int i = 0; i < ModelConfig.max_batch_size; ++i) {
				qo_indptr_h.push_back(qo_indptr_h.back() + 1);
			}
		qo_indptr_d = thrust::device_vector<int32_t>(qo_indptr_h);
	 }


	void init(uint32_t batch_size,
			  int32_t block_num,
			  pllmTensor<int32_t> qo_indptr, // from input
			  pllmTensor<int32_t> kv_indptr, // from input
			  int32_t* kv_indices, // from input
			  pllmTensor<int32_t> kv_last_page_len, // from input
			  pllmTensor<half> input, // chain op
			  pllmTensor<half> output, // chain op
			  int* device_KQV_ready = nullptr, // from gemvDependency
			  int* device_GEMV_ready = nullptr// from gemvDependency
	) {
		using namespace flashinfer;
		this->batch_size = batch_size;
		this->block_num = block_num;

		// spdlog::info("batch_size: {}", batch_size);
		assert(kv_indptr.layout == PllmLayout::ROW_MAJOR);
		assert(kv_indptr.size() == batch_size + 1);

		this->kv_indptr = kv_indptr.ptr;

		this->kv_indices = kv_indices;

		assert(kv_last_page_len.size() == batch_size);
		this->kv_last_page_len = kv_last_page_len.ptr;

		assert(input.layout == PllmLayout::ROW_MAJOR);
		if (input.size() != 0) {
			assert(input.dim2 == ModelConfig.model_qo_heads_gpu * ModelConfig.model_head_dim);
		}
		assert(output.layout == PllmLayout::ROW_MAJOR);
		if (output.size() != 0) {
			assert(input.dim2 == ModelConfig.model_qo_heads_gpu * ModelConfig.model_head_dim);
		}
		this->q = input.ptr;
		this->o = output.ptr;
		this->device_KQV_ready = device_KQV_ready;
		this->device_GEMV_ready = device_GEMV_ready;

		qo_indptr_tensor = qo_indptr;
		this->qo_indptr = qo_indptr.ptr;
		kv_indptr_tensor = kv_indptr;
		kv_last_page_len_tensor = kv_last_page_len;
		input_tensor = input;
		output_tensor = output;

		// Note(Yilong): we only use prefill kernels under GQA Scenarios.
		if (ModelConfig.model_kv_heads_gpu != ModelConfig.model_qo_heads_gpu) {
			// buffer = thrust::device_vector<char>(workspace_size_in_bytes);
			
			handler.BeginForward((void*)thrust::raw_pointer_cast(buffer.data()),
								 workspace_size_in_bytes,
								 qo_indptr_h.data(),
								 batch_size,
								 ModelConfig.model_qo_heads_gpu,
								 ModelConfig.model_kv_heads_gpu,
								 ModelConfig.model_head_dim,
								 LaunchType::SmallBlk); // Preprocess is coupled with the warp layout.
		}
	}

	void work() override {
		constexpr flashinfer::QKVLayout kv_layout = flashinfer::QKVLayout::kNHD;
		using namespace flashinfer;

		int num_kv_heads = ModelConfig.model_kv_heads_gpu;
		int head_dim = ModelConfig.model_head_dim;
		int page_size = ModelConfig.frame_page_size;
		int num_qo_heads = ModelConfig.model_qo_heads_gpu;
		auto rotary_mode = PosEncodingMode::kNone;

		// building KV cache structure
		paged_kv_t<PageStorage::kIndices, kv_layout, half, int32_t> paged_kv(
			num_kv_heads,
			page_size,
			head_dim,
			batch_size,
			thrust::raw_pointer_cast(kv_data),
			thrust::raw_pointer_cast(kv_indices),
			thrust::raw_pointer_cast(kv_indptr),
			thrust::raw_pointer_cast(kv_last_page_len));

		// spdlog::info("Print..");
		// int32_t* host_kv_indices = new int32_t[batch_size];
		// cudaMemcpy(host_kv_indices, kv_indices, sizeof(int32_t) * batch_size, cudaMemcpyDeviceToHost);
		// for (int i = 0; i < batch_size; i++) {
		// 	spdlog::warn("kv_indices[{}]: {}", i, host_kv_indices[i]);
		// }

		// int32_t* host_kv_indptr = new int32_t[batch_size + 1];
		// cudaMemcpy(host_kv_indptr, kv_indptr, sizeof(int32_t) * (batch_size + 1), cudaMemcpyDeviceToHost);
		// for (int i = 0; i < batch_size + 1; i++) {
		// 	spdlog::warn("kv_indptr[{}]: {}", i, host_kv_indptr[i]);
		// }

		if (num_qo_heads == num_kv_heads) {
			// Memory-bound setting: Use default FlashInfer API
			// Note(Yilong): To add cooperative kernel for better performance.
			cudaError_t status =
				BatchDecodeWithPagedKVCache<PageStorage::kIndices, kv_layout, half, half, int32_t>(
					q,
					/*q_offset=*/nullptr,
					paged_kv,
					kv_partition_info_t<int32_t>(),
					o,
					nullptr,
					/*lse=*/nullptr,
					num_qo_heads,
					LaunchType::SmallBlk,
					/*sm_blk=*/block_num,
					rotary_mode,
					std::nullopt,
					1.f,
					1e4,
					stream);

			if(status != cudaSuccess) {
				std::ostringstream err_msg;
				err_msg << "BatchDecodeWithPagedKVCache Failed with error: "
						<< cudaGetErrorString(status) << "\n";
				throw std::invalid_argument(err_msg.str());
			}
		} else {
			// Compute-bound setting
			// Note(Yilong): Here sm_blk should be more general.
			cudaError_t status = BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices,
																	 kv_layout,
																	 half,
																	 half,
																	 int32_t>(
				&handler,
				q,
				// thrust::raw_pointer_cast(qo_indptr_d.data()),
				this->qo_indptr,
				/*q_offset=*/nullptr,
				paged_kv,
				o,
				/*lse=*/nullptr,
				num_qo_heads,
				/*causal=*/false,
				rotary_mode,
				/*fp16 accum=*/false,
				LaunchType::SmallBlk,
				block_num,
				std::nullopt,
				1.f,
				1e4,
				stream,
				device_KQV_ready,
				device_GEMV_ready);

			if(status != cudaSuccess) {
				std::ostringstream err_msg;
				err_msg << "BatchPrefillWithPagedKVCache Failed with error: "
						<< cudaGetErrorString(status) << "\n";
				throw std::invalid_argument(err_msg.str());
			}
		}
	}

	OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override{
		log_tensor(logger, name + " kv_indices", kv_indptr_tensor, 1, 20);
		log_tensor(logger, name + " kv_last_page_len", kv_last_page_len_tensor, 1, 20);
		log_tensor(logger, name + " in", input_tensor, 10, 20);
		log_tensor(logger, name + " out", output_tensor, 10, 20);
		return *this;
	}
};

class PrefillGemvWrapper : public GemvWrapper {
public:
	// Metadata for splitting batched prefill
	pllmTensor<int32_t> qo_indptr_tensor;
	int32_t* qo_indptr;
	pllmTensor<int32_t> kv_indptr_tensor;
	pllmTensor<int32_t> kv_last_page_len_tensor;
	pllmTensor<half> input_tensor;
	pllmTensor<half> output_tensor;
	int* device_KQV_ready;
	int* device_GEMV_ready;
	PrefillGemvWrapper() { }

	void init(int batch_size,
			  int block_num,
			  pllmTensor<int32_t> qo_indptr, // from input
			  pllmTensor<int32_t> kv_indptr, // from input
			  int32_t* kv_indices, // from input
			  pllmTensor<int32_t> kv_last_page_len, // from input
			  pllmTensor<half> input, // chain op
			  pllmTensor<half> output, // chain op
			  int* device_KQV_ready = nullptr, // from gemvDependency
			  int* device_GEMV_ready = nullptr// from gemvDependency
	) {
		using namespace flashinfer;

		qo_indptr_tensor = qo_indptr;
		kv_indptr_tensor = kv_indptr;
		kv_last_page_len_tensor = kv_last_page_len;
		input_tensor = input;
		output_tensor = output;


		this->batch_size = batch_size;
		this->block_num = block_num;

		assert(qo_indptr.layout == PllmLayout::ROW_MAJOR);
		spdlog::info("qo_indptr.size(): {}", qo_indptr.size());
		spdlog::info("batch_size: {}", batch_size);
		assert(qo_indptr.size() == batch_size + 1);
		this->qo_indptr = qo_indptr.ptr;

		assert(kv_indptr.layout == PllmLayout::ROW_MAJOR);
		assert(kv_indptr.size() == batch_size + 1);
		this->kv_indptr = kv_indptr.ptr;

		this->kv_indices = kv_indices;

		assert(kv_last_page_len.size() == batch_size);
		this->kv_last_page_len = kv_last_page_len.ptr;

		this->q = input.ptr;
		this->o = output.ptr;
		// spdlog::info("before handler");
		// spdlog::info("workspace_size_in_bytes: {}", workspace_size_in_bytes);
		// spdlog::info("qo_indptr: {}", (size_t)qo_indptr.ptr);
		// spdlog::info("buffer.data() : {}", (size_t)buffer.data().get());
		// spdlog::info("batch_size: {}", batch_size);

		this->device_KQV_ready = device_KQV_ready;
		this->device_GEMV_ready = device_GEMV_ready;
		
		// buffer = thrust::device_vector<char>(workspace_size_in_bytes);
		handler.BeginForward((void*)thrust::raw_pointer_cast(this->buffer.data()),
							 workspace_size_in_bytes,
							 this->qo_indptr,
							 this->batch_size,
							 ModelConfig.model_qo_heads_gpu,
							 ModelConfig.model_kv_heads_gpu,
							 ModelConfig.model_head_dim,
							 LaunchType::AllBlk);
		spdlog::info("after handler");
	}

	void work() override {
		constexpr flashinfer::QKVLayout kv_layout = flashinfer::QKVLayout::kNHD;
		using namespace flashinfer;

		int num_kv_heads = ModelConfig.model_kv_heads_gpu;
		int head_dim = ModelConfig.model_head_dim;
		int page_size = ModelConfig.frame_page_size;
		int num_qo_heads = ModelConfig.model_qo_heads_gpu;
		auto rotary_mode = PosEncodingMode::kNone;

		// building KV cache structure
		paged_kv_t<PageStorage::kIndices, kv_layout, half, int32_t> paged_kv(
			num_kv_heads,
			page_size,
			head_dim,
			batch_size,
			thrust::raw_pointer_cast(kv_data),
			thrust::raw_pointer_cast(kv_indices),
			thrust::raw_pointer_cast(kv_indptr),
			thrust::raw_pointer_cast(kv_last_page_len));

		cudaError_t status = BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices,
																 kv_layout,
																 half,
																 half,
																 int32_t>(
			&handler,
			q,
			this->qo_indptr,
			/*q_offset=*/nullptr,
			paged_kv,
			o,
			/*lse=*/nullptr,
			num_qo_heads,
			/*causal=*/true,
			rotary_mode,
			/*fp16 accum=*/false,
			LaunchType::AllBlk, // Currently adopts All blocks for all scenarios.
			block_num,
			std::nullopt,
			1.f,
			1e4,
			stream,
			device_KQV_ready,
			device_GEMV_ready);

		if(status != cudaSuccess) {
			std::ostringstream err_msg;
			err_msg << "BatchPrefillWithPagedKVCache Failed with error: "
					<< cudaGetErrorString(status) << "\n";
			throw std::invalid_argument(err_msg.str());
		}
	}

	OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override{
		log_tensor(logger, name + " kv_indptr", kv_indptr_tensor, 1, 20);
		log_tensor(logger, name + " qo_indptr", qo_indptr_tensor, 1, 20);
		log_tensor(logger, name + " kv_last_page_len", kv_last_page_len_tensor, 1, 20);
		log_tensor(logger, name + " in", input_tensor, 10, 20);
		log_tensor(logger, name + " out", output_tensor, 10, 20);
		log_tensor(logger, name+" kv_data", pllmTensor<half>(kv_data, size_t(ModelConfig.max_page_num * ModelConfig.frame_page_size * 2), size_t(ModelConfig.model_head_dim * ModelConfig.model_kv_heads_gpu), PllmLayout::ROW_MAJOR), 32, 20, 102 * 32);
		return *this;
	}
};