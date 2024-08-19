/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_PREFILL_ATTENTION_DECL_CUH_
#define FLASHINFER_PREFILL_ATTENTION_DECL_CUH_

#include <cuda_runtime.h>

#include <optional>

#include "flashinfer/layout.cuh"
#include "flashinfer/page.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/utils.cuh"

#include "attention/handler.cuh"
#include "small_blk_utils.cuh"

namespace flashinfer
{

template <PageStorage page_storage,
		  QKVLayout kv_layout,
		  uint32_t num_frags_x,
		  uint32_t PAGE_SIZE,
		  uint32_t GROUP_SIZE,
		  uint32_t HEAD_DIM,
		  PosEncodingMode pos_encoding_mode,
		  bool ALLOW_FP16_QK_REDUCTION,
		  bool CAUSAL,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType,
		  LaunchType lType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
	DTypeIn* q,
	IdType* request_indices,
	IdType* tile_indices,
	IdType* qo_indptr,
	IdType* q_offset,
	paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
	DTypeOut* o,
	float* tmp,
	float* lse,
	uint32_t num_qo_tiles,
	size_t sm_blk,
	float sm_scale,
	float rope_scale,
	float rope_theta,
	cudaStream_t stream,
	int* device_KQV_ready,
	int* device_GEMV_ready);

template <PageStorage page_storage,
		  QKVLayout kv_layout,
		  uint32_t GROUP_SIZE,
		  uint32_t HEAD_DIM,
		  PosEncodingMode pos_encoding_mode,
		  bool ALLOW_FP16_QK_REDUCTION,
		  bool CAUSAL,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType,
		  LaunchType lType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
	BatchPrefillHandler* handler,
	DTypeIn* q,
	IdType* qo_indptr,
	IdType* q_offset,
	paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
	DTypeOut* o,
	float* lse,
	size_t sm_blk,
	float sm_scale,
	float rope_scale,
	float rope_theta,
	cudaStream_t stream,
	int* device_KQV_ready,
	int* device_GEMV_ready
	) {
	float* tmp = nullptr;
	IdType* request_indices = nullptr;
	IdType* tile_indices = nullptr;
	uint32_t num_frags_x = 0U;
	uint32_t num_qo_tiles = 0U;
	if(handler->IsForwardStarted()) {
		request_indices = handler->GetRequestIndices<IdType>();
		tile_indices = handler->GetTileIndices<IdType>();
		num_frags_x = handler->GetNumFragsX();
		num_qo_tiles = handler->GetNumQOTiles();
	} else {
		std::ostringstream err_msg;
		err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
				   "BatchPrefillWithPagedKVCacheWrapper()";
		throw std::runtime_error(err_msg.str());
	}

	DISPATCH_NUM_FRAGS_X(
		num_frags_x, NUM_FRAGS_X, {DISPATCH_PAGE_SIZE(paged_kv.page_size, PAGE_SIZE, {
			return BatchPrefillWithPagedKVCacheDispatched<page_storage,
														  kv_layout,
														  NUM_FRAGS_X,
														  PAGE_SIZE,
														  GROUP_SIZE,
														  HEAD_DIM,
														  pos_encoding_mode,
														  ALLOW_FP16_QK_REDUCTION,
														  CAUSAL,
														  DTypeIn,
														  DTypeOut,
														  IdType,
														  lType>(q,
																 request_indices,
																 tile_indices,
																 qo_indptr,
																 q_offset,
																 paged_kv,
																 o,
																 tmp,
																 lse,
																 num_qo_tiles,
																 sm_blk,
																 sm_scale,
																 rope_scale,
																 rope_theta,
																 stream,
																 device_KQV_ready,
																 device_GEMV_ready
																 );
		})});
	return cudaSuccess;
}

template <PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
cudaError_t
BatchPrefillWithPagedKVCacheWrapper(BatchPrefillHandler* handler,
									DTypeIn* q,
									IdType* qo_indptr,
									IdType* q_offset,
									paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
									DTypeOut* o,
									float* lse,
									uint32_t num_qo_heads,
									bool causal = true,
									PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
									bool allow_fp16_qk_reduction = false,
									LaunchType lType = LaunchType::AllBlk,
									std::optional<size_t> maybe_sm_blk = std::nullopt,
									std::optional<float> maybe_sm_scale = std::nullopt,
									float rope_scale = 1.f,
									float rope_theta = 1e4,
									cudaStream_t stream = nullptr,
									int* device_KQV_ready = nullptr,
									int* device_GEMV_ready = nullptr) {
	const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
	const uint32_t num_kv_heads = paged_kv.num_heads;
	const uint32_t head_dim = paged_kv.head_dim;
	const size_t sm_blk = maybe_sm_blk.value_or(0);

	if(lType == LaunchType::SmallBlk && sm_blk == 0) {
		std::ostringstream err_msg;
		err_msg << "sm_blk " << num_qo_heads << " shoule be larger than zero.";
		throw std::invalid_argument(err_msg.str());
	}

	DISPATCH_GQA_GROUP_SIZE(
		num_qo_heads / num_kv_heads,
		GROUP_SIZE,
		{DISPATCH_HEAD_DIM(
			head_dim,
			HEAD_DIM,
			{DISPATCH_CAUSAL(causal,
							 CAUSAL,
							 {DISPATCH_POS_ENCODING_MODE(
								 pos_encoding_mode,
								 pos_encoding_mode,
								 {DISPATCH_ALLOW_FP16_QK_REDUCTION(
									 allow_fp16_qk_reduction,
									 ALLOW_FP16_QK_REDUCTION,
									 {DISPATCH_LAUNCH(lType, LTYPE, {
										 return BatchPrefillWithPagedKVCacheWrapperDispatched<
											 page_storage,
											 kv_layout,
											 GROUP_SIZE,
											 HEAD_DIM,
											 pos_encoding_mode,
											 ALLOW_FP16_QK_REDUCTION,
											 CAUSAL,
											 DTypeIn,
											 DTypeOut,
											 IdType,
											 LTYPE>(handler,
													q,
													qo_indptr,
													q_offset,
													paged_kv,
													o,
													lse,
													sm_blk,
													sm_scale,
													rope_scale,
													rope_theta,
													stream,
													device_KQV_ready,
													device_GEMV_ready);
									 })})})})})});
	return cudaSuccess;
}

} // namespace flashinfer

#endif // FLASHINFER_PREFILL_ATTENTION_DECL_CUH_