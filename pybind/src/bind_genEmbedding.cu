#include <pybind11/pybind11.h>
// #include <pybind11/torch.h>  // PyTorch tensor support
#include <torch/torch.h>     // LibTorch
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>


namespace py = pybind11;

__global__ void genEmbedding(int* tokens, half* weights, half* out_embedding, int Hdim){
    int token_id = blockIdx.x;
    float4* weights_ptr = reinterpret_cast<float4*>(weights);
    float4* out_embedding_ptr = reinterpret_cast<float4*>(out_embedding);
    int token = tokens[token_id];
    int row_offset = token * (Hdim / 8);  

    int per_iter_reads = blockDim.x;  
    int iters = (Hdim / 8) / blockDim.x; 
    
    for (int i = 0; i < iters; i++){
        int idx = i * per_iter_reads + threadIdx.x + row_offset;
        float4 w = weights_ptr[idx];
        out_embedding_ptr[token_id * (Hdim / 8) + idx - row_offset] = w;
    }
}

// write a launcher
void genEmbeddingWorker(torch::Tensor tokens, torch::Tensor weights, torch::Tensor out_embedding){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Validate that tensors are on CUDA
    TORCH_CHECK(tokens.is_cuda(), "tokens must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(out_embedding.is_cuda(), "out_embedding must be a CUDA tensor");

    // Validate tensor data types
    TORCH_CHECK(tokens.dtype() == torch::kInt32, "tokens must be int32");
    TORCH_CHECK(weights.dtype() == torch::kHalf, "weights must be half");
    TORCH_CHECK(out_embedding.dtype() == torch::kHalf, "out_embedding must be half");

    // Validate tensor dimensions
    TORCH_CHECK(tokens.dim() == 1, "tokens must be 1D");
    TORCH_CHECK(weights.dim() == 2, "weights must be 2D");
    TORCH_CHECK(out_embedding.dim() == 2, "out_embedding must be 2D");

    // Ensure the input tensors are contiguous
    tokens = tokens.contiguous();
    weights = weights.contiguous();
    out_embedding = out_embedding.contiguous();

    
    int* tokens_ptr = tokens.data_ptr<int>();
    half* weights_ptr = reinterpret_cast<half*>(weights.data_ptr<at::Half>());
    half* out_embedding_ptr = reinterpret_cast<half*>(out_embedding.data_ptr<at::Half>());
    int num_tokens = tokens.size(0);
    int Hdim = weights.size(1);
    dim3 grid(num_tokens);        // one block per token
    dim3 block(256); // 256 threads per block
    
    // Launch the CUDA kernel
    genEmbedding<<<grid, block, 0, stream>>>(tokens_ptr, weights_ptr, out_embedding_ptr, Hdim);

    // Check for any kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

}

PYBIND11_MODULE(bind_genEmbedding, m) {
    m.doc() = "Pybind11 bindings for small operations in transformer";  // Optional module docstring
    m.def("genEmbedding", &genEmbeddingWorker, "Generate embedding from token ids and weights");
}
