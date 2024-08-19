#include "config.h"
#include <cuda_runtime_api.h>
#include "vortexData.cuh"
#include <vector>

class TensorManager {
    // singleton instance
private:
    TensorManager() {
        // constructor
    }

    std::vector<vortexUpdateData> gpu_update_datas;
    int nranks;
    int vnranks;

public:
    static TensorManager& getInstance() {
        static TensorManager instance;
        return instance;
    }

    void init(int nranks, int vnranks){
        this->nranks = nranks;
        this->vnranks = vnranks;
        gpu_update_datas.resize(nranks);

        for (int i = 0; i < nranks; i++) {
            cudaSetDevice(i);
            cudaMalloc(&gpu_update_datas[i].input_tokens, MAX_BATCH_SIZE * sizeof(int));
            cudaMalloc(&gpu_update_datas[i].input_indptr, (MAX_BATCH_SIZE + 1) * sizeof(int32_t));
            cudaMalloc(&gpu_update_datas[i].kv_indptr, (MAX_BATCH_SIZE + 1) * sizeof(int32_t));
            cudaMalloc(&gpu_update_datas[i].kv_indices, MAX_BATCH_SIZE * MAX_PAGE_NUM * sizeof(int32_t));
            cudaMalloc(&gpu_update_datas[i].kv_last_page_len, MAX_BATCH_SIZE * sizeof(int32_t));
            cudaMalloc(&gpu_update_datas[i].per_token_offset, MAX_BATCH_SIZE * sizeof(int32_t));
            cudaMalloc(&gpu_update_datas[i].rev_input_indptr, MAX_BATCH_SIZE * sizeof(int32_t));

            gpu_update_datas[i].gemv_batch_size = new int32_t[4];
            gpu_update_datas[i].gemv_num_blocks = new int32_t[4];
        }
    }

    template<typename T> 
    void to_gpu(T* host, T* device, size_t size, int rank, bool sync = false) {
        cudaSetDevice(rank);
        if (sync)
            cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
        else
            cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice);
        
    }

    template<typename T>
    void to_gpu_shard(T* host, T* device, size_t size, int rank) {
        size_t partitioned_size = size / vnranks / sizeof(T);
        to_gpu(host + partitioned_size * rank, device, partitioned_size, rank); 
    }

    std::vector<vortexUpdateData> update_data_to_gpu(const std::vector<vortexUpdateData>& update_data) {
        
        for (int i = 0; i < nranks; i++) {
            gpu_update_datas[i].decodePrefillBorder = update_data[i].decodePrefillBorder;
            gpu_update_datas[i].prefillNum = update_data[i].prefillNum;
            int total_token = update_data[i].input_indptr[update_data[i].decodePrefillBorder + update_data[i].prefillNum];
            int total_batch = update_data[i].decodePrefillBorder + update_data[i].prefillNum;
            to_gpu(update_data[i].input_tokens, gpu_update_datas[i].input_tokens, total_token * sizeof(int), i);
            to_gpu(update_data[i].input_indptr, gpu_update_datas[i].input_indptr, (total_batch + 1) * sizeof(int32_t), i);
            to_gpu(update_data[i].kv_indptr, gpu_update_datas[i].kv_indptr, (total_batch + 1) * sizeof(int32_t), i);
            int used_pages = update_data[i].kv_indptr[total_batch];
            to_gpu(update_data[i].kv_indices, gpu_update_datas[i].kv_indices, used_pages * sizeof(int32_t), i);
            to_gpu(update_data[i].kv_last_page_len, gpu_update_datas[i].kv_last_page_len, total_batch * sizeof(int32_t), i);
            memcpy(gpu_update_datas[i].gemv_batch_size, update_data[i].gemv_batch_size, 4 * sizeof(int32_t));
            memcpy(gpu_update_datas[i].gemv_num_blocks, update_data[i].gemv_num_blocks, 4 * sizeof(int32_t));
            to_gpu(update_data[i].per_token_offset, gpu_update_datas[i].per_token_offset, total_token * sizeof(int32_t), i);
            to_gpu(update_data[i].rev_input_indptr, gpu_update_datas[i].rev_input_indptr, total_token * sizeof(int32_t), i);
        }

        spdlog::info("update_data_to_gpu done");
        for (int i = 0; i < nranks; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        spdlog::info("cudaDeviceSynchronize done");
        return gpu_update_datas;
    }
};