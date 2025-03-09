#include <pybind11/pybind11.h>
// #include <pybind11/torch.h>  // PyTorch tensor support
#include <torch/torch.h>     // LibTorch
#include <torch/extension.h>
#include <stdio.h>


namespace py = pybind11;

// Kernel to compute the maximum of each row and the corresponding index (argmax)
__global__ void rowMaxKernel(half *d_matrix, half *d_maxVals, int *d_argMax, int cols) {
    extern __shared__ half sdata[];
    int * index_data = (int *)(sdata + blockDim.x);

    int tid = threadIdx.x;
    int row = blockIdx.x;

    half maxVal = -10000;
    int maxIdx = 0;

    // Process multiple elements per thread if necessary
    for (int i = tid; i < cols; i += blockDim.x) {
        half val = d_matrix[row * cols + i];
        if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
        }
    }

    sdata[tid] = maxVal;
    // if (row == 0) printf("tid: %d\n", tid);
    index_data[tid] = maxIdx;
    __syncthreads();

    // Reduction to find the maximum in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                index_data[tid] = index_data[tid + s];
            }
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        d_maxVals[row] = sdata[0];
        d_argMax[row] = index_data[0];
    }
}

// Wrapper function to launch the kernel
void computeRowMax(torch::Tensor matrix, torch::Tensor maxVals, torch::Tensor argMax) {
    // Validate input tensors
    TORCH_CHECK(matrix.is_cuda(), "Input matrix must be a CUDA tensor");
    TORCH_CHECK(maxVals.is_cuda(), "Output max values must be a CUDA tensor");
    TORCH_CHECK(argMax.is_cuda(), "Output argmax must be a CUDA tensor");

    TORCH_CHECK(matrix.dtype() == torch::kHalf, "Input matrix must be of type half");
    TORCH_CHECK(maxVals.dtype() == torch::kHalf, "Output max values must be of type half");
    TORCH_CHECK(argMax.dtype() == torch::kInt, "Output argmax must be of type int");

    matrix = matrix.contiguous();
    maxVals = maxVals.contiguous();
    argMax = argMax.contiguous();

    half *d_matrix = reinterpret_cast<half*>(matrix.data_ptr());
    half *d_maxVals = reinterpret_cast<half*>(maxVals.data_ptr());
    int *d_argMax = reinterpret_cast<int*>(argMax.data_ptr());
    int rows = matrix.size(0);
    int cols = matrix.size(1);

    dim3 blockSize(1024);
    dim3 gridSize(rows);
    size_t sharedMemSize = blockSize.x * (sizeof(half) + sizeof(int));

    // Launch the kernel


    if (gridSize.x > 0) {
        rowMaxKernel<<<gridSize, blockSize, sharedMemSize>>>(d_matrix, d_maxVals, d_argMax, cols);
    }    
}

PYBIND11_MODULE(bind_sample, m) {
    m.def("SampleMax", &computeRowMax, "Compute row max and argmax");
}