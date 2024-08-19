/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <thrust/device_vector.h>

#include <cstdint>
#include <nvbench/nvbench.cuh>
#include <vector>

#include "prefill_attention_decl.cuh"
#include "utils.h"

using utils::vec_bytes;
using namespace flashinfer;

constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename T>
void bench_flashinfer_batch_prefill(nvbench::state& state) {
	constexpr size_t head_dim = 128;
	constexpr auto pos_encoding_mode = PosEncodingMode::kNone;

	size_t kv_seqlen = state.get_int64("kv_seqlen");
	size_t qo_seqlen = state.get_int64("qo_seqlen");
	size_t batch_size = state.get_int64("batch_size");
	size_t page_size = state.get_int64("page_size");
	size_t num_qo_heads = state.get_int64("num_qo_heads");
	size_t num_kv_heads = state.get_int64("num_kv_heads");

	LaunchType lType = LaunchType(state.get_int64("launch_type"));
	size_t sm_blk = state.get_int64("sm_blk");

	if(qo_seqlen < kv_seqlen) {
		state.skip("Append Prefill Kernels should have larger qo_seqlen than kv_seqlen");
		return;
	}

	// KV cache:
	auto pages_per_seq = (kv_seqlen + page_size - 1) / page_size;
	auto num_pages = pages_per_seq * batch_size;

	std::vector<int32_t> kv_indptr_host{0};
	std::vector<int32_t> kv_indicies_host;
	std::vector<int32_t> kv_last_page_len_host;
	for(size_t i = 0; i < batch_size; ++i) {
		for(size_t p = 0; p < pages_per_seq; ++p) {
			kv_indicies_host.push_back(i * pages_per_seq + p);
		}
		kv_indptr_host.push_back(kv_indptr_host.back() + pages_per_seq);
		kv_last_page_len_host.push_back((kv_seqlen - 1) % page_size + 1);
	}
	thrust::device_vector<T> kv_data(num_pages * 2 * num_kv_heads * page_size * head_dim);
	thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);
	thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
	thrust::device_vector<int32_t> kv_last_page_len(kv_last_page_len_host);
	paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		thrust::raw_pointer_cast(kv_data.data()),
		thrust::raw_pointer_cast(kv_indices.data()),
		thrust::raw_pointer_cast(kv_indptr.data()),
		thrust::raw_pointer_cast(kv_last_page_len.data()));

	// qo_seqlen info:
	std::vector<int32_t> q_indptr{0};
	for(uint32_t i = 0; i < batch_size; ++i) {
		q_indptr.push_back(q_indptr.back() + qo_seqlen);
	}
	thrust::device_vector<int32_t> q_indptr_device(q_indptr);

	// Allocate input data:
	thrust::device_vector<T> q(batch_size * num_qo_heads * head_dim * qo_seqlen);
	thrust::device_vector<T> o(batch_size * num_qo_heads * head_dim * qo_seqlen);
	state.add_global_memory_reads<uint8_t>(
		vec_bytes(q) + (num_pages * 2 * num_kv_heads * page_size * head_dim) * sizeof(T) +
			vec_bytes(kv_indptr) + vec_bytes(kv_indices) + vec_bytes(kv_last_page_len) +
			vec_bytes(q_indptr_device),
		"Read");
	state.add_global_memory_writes<uint8_t>(vec_bytes(o), "Write");

	BatchPrefillHandler handler;
	size_t workspace_size_in_bytes = 32 * 1024 * 1024;
	thrust::device_vector<char> buffer(workspace_size_in_bytes);

	handler.BeginForward((void*)thrust::raw_pointer_cast(buffer.data()),
						 workspace_size_in_bytes,
						 thrust::raw_pointer_cast(q_indptr_device.data()),
						 batch_size,
						 num_qo_heads,
						 num_kv_heads,
						 head_dim,
						 lType);

	state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
		auto status =
			BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices, kv_layout, T, T, int32_t>(
				&handler,
				thrust::raw_pointer_cast(q.data()),
				thrust::raw_pointer_cast(q_indptr_device.data()),
				/*q_offset=*/nullptr,
				paged_kv,
				thrust::raw_pointer_cast(o.data()),
				/*lse=*/nullptr,
				num_qo_heads,
				/*casual=*/true,
				pos_encoding_mode,
				/*allow_fp16_accumlate*/ false,
				lType,
				sm_blk);
		if(status != cudaSuccess) {
			std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
		}
	});
	const auto measured_mean = static_cast<nvbench::float32_t>(
		state.get_summary("nv/cold/time/gpu/mean").get_float64("value"));
	auto& summ = state.add_summary("nv/tflops");
	summ.set_string("description", "Achieved TFlops/s");
	summ.set_string("name", "TFlops/s");
	float tflops = batch_size * qo_seqlen * (2 * kv_seqlen - qo_seqlen) * 2 * num_qo_heads *
				   head_dim / measured_mean / 1e12;
	summ.set_float64("value", tflops);
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_BATCH_PREFILL(dtype)                                               \
	auto bench_flashinfer_batch_prefill_##dtype##_ = bench_flashinfer_batch_prefill<dtype>; \
	NVBENCH_BENCH(bench_flashinfer_batch_prefill_##dtype##_)                                \
		.set_name("bench_flashinfer_batch_prefill_" STR(dtype))                             \
		.add_int64_axis("kv_seqlen", {128, 256, 512, 1024, 2048, 4096})                     \
		.add_int64_axis("qo_seqlen", {128, 256, 512, 1024, 2048, 4096})                     \
		.add_int64_axis("batch_size", {1, 10, 14, 128, 160, 192, 512, 1024})                \
		.add_int64_axis("page_size", {16})                                                  \
		.add_int64_axis("num_qo_heads", {32})                                               \
		.add_int64_axis("num_kv_heads", {32, 4})                                            \
		.add_int64_axis("launch_type", {0, 1})                                              \
		.add_int64_axis("sm_blk", {10, 13, 17, 31, 66})

BENCH_FLASHINFER_BATCH_PREFILL(half);