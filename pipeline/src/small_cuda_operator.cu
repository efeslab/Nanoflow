#include "small_cuda_operator.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include "helper.h"

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

#define SPLIT_ROW_PER_BLOCK 64
__global__ void splitKQVOutputKernel(half* kqv, half* k, half* q, half* v, int k_dim, int q_dim, int v_dim){
    int row_base = blockIdx.x * SPLIT_ROW_PER_BLOCK;
    int k_dim_f4 = k_dim / (sizeof(float4) / sizeof(half));
    int q_dim_f4 = q_dim / (sizeof(float4) / sizeof(half));
    int v_dim_f4 = v_dim / (sizeof(float4) / sizeof(half));

    float4* k4 = reinterpret_cast<float4*>(k);
    float4* q4 = reinterpret_cast<float4*>(q);
    float4* v4 = reinterpret_cast<float4*>(v);
    float4* kqv4 = reinterpret_cast<float4*>(kqv);

    int kqv_dim_f4 = k_dim_f4 + q_dim_f4 + v_dim_f4;
    
    #pragma unroll 16
    for (int i = 0; i < SPLIT_ROW_PER_BLOCK; i++){
        int row = row_base + i;
        if(threadIdx.x < k_dim_f4){
            k4[row * k_dim_f4 + threadIdx.x] = kqv4[row * kqv_dim_f4 + threadIdx.x];
        } else if (threadIdx.x < k_dim_f4 + q_dim_f4){
            q4[row * q_dim_f4 + threadIdx.x - k_dim_f4] = kqv4[row * kqv_dim_f4 + threadIdx.x];
        } else {
            v4[row * v_dim_f4 + threadIdx.x - k_dim_f4 - q_dim_f4] = kqv4[row * kqv_dim_f4 + threadIdx.x];
        }
    }
}


void splitKQVOutput(half* kqv, int batch_size, half* k, half* q, half* v, int k_dim, int q_dim, int v_dim, cudaStream_t stream){
    int total_dim_f4 = (k_dim + q_dim + v_dim) / (sizeof(float4) / sizeof(half));
    dim3 grid(batch_size/ SPLIT_ROW_PER_BLOCK);
    dim3 block(total_dim_f4);
    splitKQVOutputKernel<<<grid, block, 1000, stream>>>(kqv, k, q, v, k_dim, q_dim, v_dim);
}


#define TRANSPOSE_TILE_DIM 32  

__global__ void transposeKernel(half* input, half* output, int row, int column) {
    __shared__ half tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];  // padding to avoid bank conflicts

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;
    // printf("x: %d, y: %d, input: %f\n", x, y, __half2float(input[y * column + x]));

    if (x < column && y < row) {
        tile[threadIdx.y][threadIdx.x] = input[y * column + x];
    }
    
    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    if (x < row && y < column) {
        output[y * row + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose(half* d_input, half* d_output, int row, int column, cudaStream_t stream) {

    // Define grid and block dimensions
    dim3 dimBlock(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_DIM);
    dim3 dimGrid((column + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM, (row + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);

    // Launch the transpose kernel
    transposeKernel<<<dimGrid, dimBlock, 10000, stream>>>(d_input, d_output, row, column);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in transpose: " << cudaGetErrorString(error) << std::endl;
    }
}

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

void silu_and_multiply(half *d_input, half *d_output, int M, int N, cudaStream_t stream) {
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    silu_and_multiply_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, M, N);
}



__global__ void extractRankSubmatrixHalf(const half* inputMatrix, half* outputMatrix, int M, int N, int nrank, int rank) {
    // Calculate the row index for the current warp
    int row = blockIdx.x * blockDim.y + threadIdx.y;

    // Number of float4 elements per row for the specified rank
    int numFloat4s = N / 8;  // Each float4 can load 8 half values

    if (row < M) {
        // Calculate start index for the columns to copy
        int startCol = rank * N;
        
        // Use float4 for copying
        const float4* inputRow = reinterpret_cast<const float4*>(inputMatrix + row * nrank * N + startCol);
        float4* outputRow = reinterpret_cast<float4*>(outputMatrix + row * N);

        for (int i = threadIdx.x; i < numFloat4s; i += warpSize) {
            outputRow[i] = inputRow[i];
        }
    }
}

void extractRankSubmatrixHalfDevice(half* d_inputMatrix, half* d_outputMatrix, int M, int N, int nrank, int rank, cudaStream_t stream) {
    // Define block and grid dimensions
    dim3 blockSize(32, 8); // 32 threads per warp, 4 warps per block
    int gridSize = (M + blockSize.y - 1) / blockSize.y; // Number of blocks in the grid

    // Launch kernel
    extractRankSubmatrixHalf<<<gridSize, blockSize, 0, stream>>>(d_inputMatrix, d_outputMatrix, M, N, nrank, rank);
}


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
void computeRowMax(half *d_matrix, half *d_maxVals, int *d_argMax, int rows, int cols, cudaStream_t stream) {
    dim3 blockSize(1024);
    dim3 gridSize(rows);
    size_t sharedMemSize = blockSize.x * (sizeof(half) + sizeof(int));

    rowMaxKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(d_matrix, d_maxVals, d_argMax, cols);
}


__global__ void copyLinesHalf(const __half* input, __half* output, const int* keeplist, int numCols) {
    int row = blockIdx.x;
    int threadId = threadIdx.x;

    // Calculate the row to copy based on the keeplist format
    int sourceRow = keeplist[row + 1] - 1;

    int elementsPerThread = sizeof(float4) / sizeof(__half);

    // Each thread processes elements in a loop
    for (int i = threadId * elementsPerThread; i < numCols; i += blockDim.x * elementsPerThread) {
        float4* srcPtr = (float4*)(input + sourceRow * numCols + i);
        float4* dstPtr = (float4*)(output + row * numCols + i);
        
        *dstPtr = *srcPtr;
    }
}

// Wrapper function
void copySelectedRows(int numKeepRows, int numCols, const int* d_keeplist, const __half* d_input, __half* d_output, cudaStream_t stream) {
    // Define block size (number of threads per block)
    int blockSize = 128;  // You can adjust this as needed
    
    // Define grid size (number of blocks in the grid)
    int gridSize = numKeepRows;  // One block per row in the keeplist

    // Launch the kernel
    copyLinesHalf<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, d_keeplist, numCols);
    
    // Synchronize to check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

// Kernel function using float4, each block copies one row
__global__ void replicateKQVBiasKernel(const half* __restrict__ input, half* __restrict__ output, int n) {
    int row = blockIdx.x;     // Each block handles one row
    int tid = threadIdx.x;

    const int elements_per_float4 = 8;  // Each float4 can hold 8 half elements
    int num_float4 = n / elements_per_float4;   // Number of float4 copies needed per row
    int remainder = n % elements_per_float4;    // Remaining elements to process

    const float4* input_f4 = reinterpret_cast<const float4*>(input);
    float4* output_f4 = reinterpret_cast<float4*>(output);

    // Copy using float4
    for (int i = tid; i < num_float4; i += blockDim.x) {
        int idx = i;
        int out_idx = row * num_float4 + i;
        // if (blockIdx.x == 0) {
        //     printf("row: %d, idx: %d, out_idx: %d\n", row, idx, out_idx);
        // }
        output_f4[out_idx] = input_f4[idx];
    }

    // Handle remaining elements
    if (tid == 0 && remainder > 0) {
        int start_idx = num_float4 * elements_per_float4;
        for (int i = 0; i < remainder; ++i) {
            output[row * n + start_idx + i] = input[start_idx + i];
        }
    }
}

// Wrapper function with cudaStream_t parameter
void replicateKQVBias(const half* d_input, half* d_output, int m, int n, cudaStream_t stream) {
    int threads_per_block = 128;
    int blocks_per_grid = m;


    // Launch kernel on the specified stream
    replicateKQVBiasKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_input, d_output, n);

    // Check if kernel launch was successful
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // No device synchronization
}