#include "offloadKernel.cuh"
#include <cstdio>

__device__ void pageCopy(half* input, half* output, int page_mem_size){

    int copyIter = page_mem_size / sizeof(float4) / blockDim.x;
    // printf("copyIter: %d\n", copyIter);
    float4* input4 = (float4*)input;
    float4* output4 = (float4*)output;

    for (int i = 0; i < copyIter; i++){
        output4[i * blockDim.x + threadIdx.x] = input4[i * blockDim.x + threadIdx.x];
    }
}

__global__ void moveKVcacheKernel(int finished_req_num, int32_t * finished_index, 
                                       int32_t* kv_indptr, int32_t* kv_indices, half* host_ptr, half* kv_data, int page_mem_size, bool host_to_gpu){
    page_mem_size /= sizeof(half);
    for (int i = 0; i < finished_req_num; i++){
        int idx = finished_index[i];
        int start = kv_indptr[idx];
        int end = kv_indptr[idx + 1];

        for (int j = start + blockIdx.x; j < end; j += gridDim.x){
            int page_idx = kv_indices[j];
            half* page = kv_data + page_idx * page_mem_size;
            half* host_page = host_ptr + j * page_mem_size;
            // printf("page_idx: %d\n", page_idx);
            if (host_to_gpu)
                pageCopy(host_page, page, page_mem_size);
            else
                pageCopy(page, host_page, page_mem_size);
        }
    }
}