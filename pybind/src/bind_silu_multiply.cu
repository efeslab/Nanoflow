#include <pybind11/pybind11.h>
#include <torch/torch.h>     // LibTorch
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <stdio.h>


namespace py = pybind11;

// CUDA kernel function to apply SiLU to the right half and then multiply by the left half
__global__ void silu_and_multiply_kernel(half *input, half *output, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int idx_left = row * 2 * N + col;
        int idx_right = row * 2 * N + N + col;

        // Apply SiLU to the right half
        float right = __half2float(input[idx_right]);
        float silu = right / (1.0f + expf(-right));

        // Multiply the left half with the SiLU result
        float left = __half2float(input[idx_left]);
        float product = left * silu;

        // Store the result in the output
        int idx_output = row * N + col;
        output[idx_output] = __float2half(product);
    }
}

void silu_and_multiply(torch::Tensor input, torch::Tensor output) {
    // Validate the input and output tensors
    TORCH_CHECK(input.dtype() == torch::kHalf, "Input tensor must be of half data type");
    TORCH_CHECK(output.dtype() == torch::kHalf, "Output tensor must be of half data type");

    // Get the dimensions of the output tensor, denote as M x N
    int M = output.size(0);
    int N = output.size(1);

    input = input.contiguous();
    output = output.contiguous();

    // Get the device pointers
    half *d_input = reinterpret_cast<half *>(input.data_ptr());
    half *d_output = reinterpret_cast<half *>(output.data_ptr());
    
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    silu_and_multiply_kernel<<<gridSize, blockSize>>>(d_input, d_output, M, N);
}

PYBIND11_MODULE(bind_silu_multiply, m) {
    m.doc() = "Pybind11 bindings for small operations in transformer";  // Optional module docstring
    m.def("silu_multiply", &silu_and_multiply, "Apply SiLU to the right half and multiply by the left half");
}
