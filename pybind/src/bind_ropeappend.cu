#include <pybind11/pybind11.h>
// #include <pybind11/torch.h>  // PyTorch tensor support
#include <torch/torch.h>     // LibTorch
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/page.cuh"
#include "cutlass/cutlass.h"
#include "small_cuda_operator.cuh"

#include <cuda.h>
#include <stdio.h>

namespace py = pybind11;

flashinfer::paged_kv_t<half, int32_t>* paged_kv_ptr = nullptr;
int32_t* kv_indptr = nullptr;
int32_t* kv_indices = nullptr;
int32_t* kv_last_page_len = nullptr;


void updateKVCache(torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor kv_last_page_offset, int num_req, int page_size, int num_kv_heads, int num_qo_heads, int head_dim) {
    int32_t* kv_indices_device = reinterpret_cast<int32_t*>(kv_indices.data_ptr<int32_t>());
    int32_t* kv_indptr_device = reinterpret_cast<int32_t*>(kv_indptr.data_ptr<int32_t>());
    int32_t* kv_last_page_len_device = reinterpret_cast<int32_t*>(kv_last_page_offset.data_ptr<int32_t>());

    flashinfer::QKVLayout kv_layout = flashinfer::QKVLayout::kHND;

    paged_kv_ptr = new flashinfer::paged_kv_t<half, int32_t>(
        num_kv_heads,
        page_size,
        head_dim,
        num_req,
        kv_layout,
        nullptr,
        nullptr,
        kv_indices_device,
        kv_indptr_device,
        kv_last_page_len_device
    );
}

void splitRopeAppendWorker(torch::Tensor k_data, torch::Tensor v_data, torch::Tensor kqv_input, torch::Tensor q_global, torch::Tensor rev_input_indptr, torch::Tensor per_token_offset, int num_req, int page_size, int num_kv_heads, int num_qo_heads, int head_dim, float rope_scale, float rope_theta, float smooth_a, float smooth_b) {
    // printf("splitRopeAppendWorker called\n");
    // printf("rope_scale: %f, rope_theta: %f, smooth_a: %f, smooth_b: %f\n", rope_scale, rope_theta, smooth_a, smooth_b);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // nvtxMarkA("Tensor Check");
    // Validate that tensors are on CUDA
    // TORCH_CHECK(k_data.is_cuda(), "k_data must be a CUDA tensor");
    // TORCH_CHECK(v_data.is_cuda(), "v_data must be a CUDA tensor");
    // TORCH_CHECK(kqv_input.is_cuda(), "kqv_input must be a CUDA tensor");
    // TORCH_CHECK(q_global.is_cuda(), "q_global must be a CUDA tensor");
    // TORCH_CHECK(rev_input_indptr.is_cuda(), "rev_input_indptr must be a CUDA tensor");
    // TORCH_CHECK(per_token_offset.is_cuda(), "per_token_offset must be a CUDA tensor");

    // // Validate tensor data types
    // TORCH_CHECK(k_data.dtype() == torch::kHalf, "k_data must be half");
    // TORCH_CHECK(v_data.dtype() == torch::kHalf, "v_data must be half");
    // TORCH_CHECK(kqv_input.dtype() == torch::kHalf, "kqv_input must be half");
    // TORCH_CHECK(q_global.dtype() == torch::kHalf, "q_global must be half");

    // // Validate tensor dimensions
    // TORCH_CHECK(k_data.dim() == 4, "k_data must be 4D");
    // TORCH_CHECK(v_data.dim() == 4, "v_data must be 4D");
    // TORCH_CHECK(kqv_input.dim() == 2, "kqv_input must be 2D");
    // TORCH_CHECK(q_global.dim() == 2, "q_global must be 2D");

    // // nvtxMarkA("Contiguous Check");
    // // Ensure the input tensors are contiguous
    // k_data = k_data.contiguous();
    // v_data = v_data.contiguous();

    // kqv_input = kqv_input.contiguous();
    // q_global = q_global.contiguous();
    // rev_input_indptr = rev_input_indptr.contiguous();
    // per_token_offset = per_token_offset.contiguous();

    // nvtxMarkA("Data Pointer");
    half* k_data_ptr = reinterpret_cast<half*>(k_data.data_ptr<at::Half>());
    half* v_data_ptr = reinterpret_cast<half*>(v_data.data_ptr<at::Half>());

    half* kqv_input_ptr = reinterpret_cast<half*>(kqv_input.data_ptr<at::Half>());
     // q_global
    half* q_global_ptr = reinterpret_cast<half*>(q_global.data_ptr<at::Half>());
    int32_t* rev_input_indptr_ptr = reinterpret_cast<int32_t*>(rev_input_indptr.data_ptr<int32_t>());
    int32_t* per_token_offset_ptr = reinterpret_cast<int32_t*>(per_token_offset.data_ptr<int32_t>());
    int32_t dense_batch_size = rev_input_indptr.size(0);

    paged_kv_ptr->k_data = k_data_ptr;
    paged_kv_ptr->v_data = v_data_ptr;

    splitRopeAppend(
        *paged_kv_ptr,
        kqv_input_ptr,
        rev_input_indptr_ptr,
        per_token_offset_ptr,
        dense_batch_size,
        num_qo_heads,     
        q_global_ptr,
        nullptr,
        rope_scale,
        rope_theta,
        smooth_a,
        smooth_b,
        stream
    );
}

PYBIND11_MODULE(bind_ropeappend, m) {
    m.def("updateKVCache", &updateKVCache, "Configure KV Cache");
    m.def("splitRopeAppend", &splitRopeAppendWorker, "Split Rope Append Worker");
}  // namespace pybind11