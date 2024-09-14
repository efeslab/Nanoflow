#pragma once
#include <cuda.h>
#include "cutlass/cutlass.h"
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/page.cuh"
#include "spdlog/spdlog.h"


__global__ void genEmbedding(int* tokens, half* weights, half* out_embedding, int Hdim);
void splitKQVOutput(half* kqv, int batch_size, half* k, half* q, half* v, int k_dim, int q_dim, int v_dim, cudaStream_t stream);
void transpose(half* d_input, half* d_output, int row, int column, cudaStream_t stream);
void silu_and_multiply(half *d_input, half *d_output, int M, int N, cudaStream_t stream);
void extractRankSubmatrixHalfDevice(half* d_inputMatrix, half* d_outputMatrix, int M, int N, int nrank, int rank, cudaStream_t stream);
void computeRowMax(half *d_matrix, half *d_maxVals, int *d_argMax, int rows, int cols, cudaStream_t stream);
void copySelectedRows(int numKeepRows, int numCols, const int* d_keeplist, const __half* d_input, __half* d_output, cudaStream_t stream);
void replicateKQVBias(const half* d_input, half* d_output, int n, int m, cudaStream_t stream);
namespace flashinfer {

template <uint32_t head_dim, uint32_t bdx, uint32_t vec_size, flashinfer::PageStorage page_storage, flashinfer::QKVLayout layout,
          typename DType, typename IdType> 
__global__ void splitRopeAppendKernel(flashinfer::paged_kv_t<page_storage, layout, DType, IdType> paged_kv,
                                      DType* kqv_input, IdType* rev_input_indptr, IdType* per_token_offset, 
                                      int32_t num_qo_heads, DType* q_out_global, int* device_KQV_ready,  float rope_rcp_scale, float rope_rcp_theta, float smooth_a, float smooth_b) {
    int token_index = blockIdx.x;
    int req_index = rev_input_indptr[token_index];
    int tx = threadIdx.x;
    int num_kv_heads = paged_kv.num_heads;


    size_t kqv_n = head_dim * (num_qo_heads + 2 * paged_kv.num_heads);
    DType* kqv = kqv_input + token_index * kqv_n;  // this block handles this token
    DType* q_out = q_out_global + token_index * head_dim * num_qo_heads; 
    
    vec_t<float, vec_size> freq;
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
        freq[i] =
            __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
        float smooth = freq[i] * smooth_a + smooth_b;
        smooth = max(0.0f, min(1.0f, smooth));  // clamp to [0, 1]
        freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
    int iter = (num_kv_heads * 2 + num_qo_heads + blockDim.y - 1) / blockDim.y;
    for (int i = 0; i < iter; ++i) {
        int ty = i * blockDim.y + threadIdx.y;
        if (ty >= num_kv_heads * 2 + num_qo_heads) break;
        if (ty < num_kv_heads)
        // processing K
        {
            int head_idx = ty;
            int offset = per_token_offset[token_index];
            DType* k_ptr_input = kqv + head_dim * ty;
            vec_t<float, vec_size> k_vec;
            k_vec = vec_apply_llama_rope<vec_size, bdx> (k_ptr_input, freq, offset); // warp size
            // k_vec.cast_store(q_out + head_idx * head_dim + tx * vec_size);
            int token_in_request_index = offset;
            int page_in_indices_index = paged_kv.indptr[req_index] + token_in_request_index / paged_kv.page_size;
            int token_in_page_index = token_in_request_index % paged_kv.page_size;

            DType* k_ptr = paged_kv.get_k_ptr(page_in_indices_index, head_idx, token_in_page_index, tx * vec_size);
            k_vec.cast_store(k_ptr);
        }
        else if (ty < num_kv_heads * 2)
        // processing V
        {
            int head_idx = ty - num_kv_heads;
            int offset = per_token_offset[token_index];
            DType* v_ptr_input = kqv + head_dim * ty;
            int token_in_request_index = offset;
            int page_in_indices_index = paged_kv.indptr[req_index] + token_in_request_index / paged_kv.page_size;
            int token_in_page_index = token_in_request_index % paged_kv.page_size;

            DType* v_ptr = paged_kv.get_k_ptr(page_in_indices_index, head_idx, token_in_page_index, tx * vec_size)
                            + paged_kv.kv_offset_delta();
            vec_t<DType, vec_size>::memcpy(v_ptr, v_ptr_input + tx * vec_size);
        }
        else
        // processing Q
        {
            int head_idx = ty - num_kv_heads * 2;
            int offset = per_token_offset[token_index];
            DType* q_ptr = kqv + head_dim * ty;
            vec_t<float, vec_size> q_vec;
            q_vec = vec_apply_llama_rope<vec_size, bdx> (q_ptr, freq, offset); // warp size
            q_vec.cast_store(q_out + head_idx * head_dim + tx * vec_size);
        }
    }
    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //     atomicAdd(device_KQV_ready, 1);
    // }
}

template <flashinfer::PageStorage page_storage, flashinfer::QKVLayout layout, typename DType, typename IdType>
cudaError_t splitRopeAppend(flashinfer::paged_kv_t<page_storage, layout, DType, IdType> paged_kv, DType* kqv_input,
                                IdType* rev_input_indptr, IdType* per_token_offset, int32_t dense_batch_size, int32_t num_qo_heads,
                                DType* q_out = nullptr, int* device_KQV_ready = nullptr,
                                float rope_scale = 1.f, float rope_theta = 1e4, float smooth_a = 0, float smooth_b = 0, cudaStream_t stream = nullptr){
    float rope_rcp_scale = 1.0f / rope_scale;
    float rope_rcp_theta = 1.0f / rope_theta;
    DISPATCH_HEAD_DIM(paged_kv.head_dim, HEAD_DIM, {
        constexpr uint32_t warp_size = 32;
        constexpr uint32_t vec_size = HEAD_DIM / warp_size;
        uint32_t thread_num_y = num_qo_heads + 2 * paged_kv.num_heads;
        if (thread_num_y > 1024/warp_size) {
            thread_num_y = 8;
        }
        constexpr uint32_t thread_num_x = warp_size; // == HEAD_DIM / vec_size
        dim3 nthreads(thread_num_x, thread_num_y);
        dim3 nblocks(dense_batch_size);
        // spdlog::info("nthreads: ({}, {}), nblocks: {}", thread_num_x, thread_num_y, dense_batch_size);
        auto kernel =
            splitRopeAppendKernel<HEAD_DIM, thread_num_x, vec_size, page_storage, layout, DType, IdType>;
        void* args[] = {
            (void*)&paged_kv,
            (void*)&kqv_input,
            (void*)&rev_input_indptr,
            (void*)&per_token_offset,
            (void*)&num_qo_heads,
            (void*)&q_out,
            (void*)&device_KQV_ready,
            (void*)&rope_rcp_scale,
            (void*)&rope_rcp_theta,
            (void*)&smooth_a,
            (void*)&smooth_b,
        };
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblocks, nthreads, args, 0, stream));
    });
    
    // CUDA_CHECK(cudaDeviceSynchronize());
    return cudaSuccess;
}
}