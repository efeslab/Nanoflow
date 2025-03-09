#pragma once
// #include "spdlog/sinks/basic_file_sink.h"
// #include "spdlog/spdlog.h"
#include "helper.h"
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <cmath>
#define ENABLE_NETWORK
// #define ENABLE_MPI
struct vortexModelConfig {
    int gpu_num = 8;
    int model_layer = 80;
    int run_layer = 80;
    int allocate_kv_data_batch = 400; // pages for 400 req averaging length 1024 (16*64, 64 will be multiplied later) 16 tokens/page

    int model_head_dim = 128;
    int model_gqa = 8;
    int model_kv_heads = 8;
    int model_qo_heads = 64;

    int model_kv_heads_gpu = model_kv_heads / gpu_num;
    int model_qo_heads_gpu = model_qo_heads / gpu_num;

    int model_ff_dim = 28 * 1024;
    int model_ff_dim_gpu = model_ff_dim / gpu_num;

    int ug_n = model_ff_dim_gpu * 2;

    int model_hidden_dim = 8192;
    int model_hidden_dim_pergpu = model_hidden_dim / gpu_num;

    int kqv_n = model_hidden_dim_pergpu + 2 * model_hidden_dim_pergpu / model_gqa;

    int frame_page_size = 16;

    int max_batch_size = 2048;

    int num_local_experts = 0;
    int num_experts_per_tok = 0;

    float rms_norm_eps = 1e-5;

    float rope_theta = 1e4;
    float factor = 8.0;
    float low_freq_factor = 1.0;
    float high_freq_factor = 4.0;
    int original_max_position_embeddings=8192;
    float smooth_a = 0;
    float smooth_b = 0;
    bool rope_3_1 = false;
    int max_offload_page_num = 0;
    int max_load_page_num = 0;

    size_t vocab_size = 32000;

    size_t gpu_mem = size_t(model_layer) * 1024 * 1024 * 1024;
    size_t page_mem_size = frame_page_size * model_kv_heads_gpu * model_head_dim * 2 * 2;
    size_t max_page_num = size_t(gpu_mem / model_layer / page_mem_size);
    size_t k_temp_size = size_t(max_batch_size * model_hidden_dim * 16 + max_batch_size * vocab_size * 2);
    size_t offload_agg_alloc_size = model_layer * max_offload_page_num * frame_page_size * 2 * model_kv_heads_gpu * model_head_dim * 2;
    size_t offload_agg_alloc_size_per_layer = max_offload_page_num * frame_page_size * 2 * model_kv_heads_gpu * model_head_dim * 2;
    size_t load_agg_alloc_size = model_layer * max_load_page_num * frame_page_size * 2 * model_kv_heads_gpu * model_head_dim * 2;
    size_t load_agg_alloc_size_per_layer = max_load_page_num * frame_page_size * 2 * model_kv_heads_gpu * model_head_dim * 2;

    // Optional: Add a method to display the config
    void printConfig() const {
        std::cout << "GPU_NUM: " << gpu_num << std::endl;
        std::cout << "MODEL_LAYER: " << model_layer << std::endl;
        std::cout << "RUN_LAYER: " << run_layer << std::endl;
        std::cout << "ALLOCATE_KV_DATA_BATCH: " << allocate_kv_data_batch << std::endl;
        std::cout << "MODEL_HEAD_DIM: " << model_head_dim << std::endl;
        std::cout << "MODEL_GQA: " << model_gqa << std::endl;
        std::cout << "MODEL_KV_HEADS: " << model_kv_heads << std::endl;
        std::cout << "MODEL_QO_HEADS: " << model_qo_heads << std::endl;
        std::cout << "MODEL_KV_HEADS_GPU: " << model_kv_heads_gpu << std::endl;
        std::cout << "MODEL_QO_HEADS_GPU: " << model_qo_heads_gpu << std::endl;
        std::cout << "MODEL_FF_DIM: " << model_ff_dim << std::endl;
        std::cout << "MODEL_FF_DIM_GPU: " << model_ff_dim_gpu << std::endl;
        std::cout << "UG_N: " << ug_n << std::endl;
        std::cout << "MODEL_HIDDEN_DIM: " << model_hidden_dim << std::endl;
        std::cout << "MODEL_HIDDEN_DIM_PERGPU: " << model_hidden_dim_pergpu << std::endl;
        std::cout << "KQV_N: " << kqv_n << std::endl;
        std::cout << "FRAME_PAGE_SIZE: " << frame_page_size << std::endl;
        std::cout << "MAX_BATCH_SIZE: " << max_batch_size << std::endl;
        std::cout << "VOCABULARY_SIZE: " << vocab_size << std::endl;
        std::cout << "GPU_MEM: " << gpu_mem << std::endl;
        std::cout << "PAGE_MEM_SIZE: " << page_mem_size << std::endl;
        std::cout << "MAX_PAGE_NUM: " << max_page_num << std::endl;
        std::cout << "ROPE_THETA: " << rope_theta << std::endl;
        std::cout << "FACTOR: " << factor << std::endl;
        std::cout << "LOW_FREQ_FACTOR: " << low_freq_factor << std::endl;
        std::cout << "HIGH_FREQ_FACTOR: " << high_freq_factor << std::endl;
        std::cout << "ORIGINAL_MAX_POSITION_EMBEDDINGS: " << original_max_position_embeddings << std::endl;
        std::cout << "RMS_NORM_EPS: " << rms_norm_eps << std::endl;
        std::cout << "K_TEMP_SIZE: " << k_temp_size << std::endl;
        std::cout << "SMOOTH_A: " << smooth_a << std::endl;
        std::cout << "SMOOTH_B: " << smooth_b << std::endl;
        std::cout << "NUM_LOCAL_EXPERTS: " << num_local_experts << std::endl;
        std::cout << "NUM_EXPERTS_PER_TOK: " << num_experts_per_tok << std::endl;
        std::cout << "MAX_OFFLOAD_PAGE_NUM: " << max_offload_page_num << std::endl;
        std::cout << "MAX_LOAD_PAGE_NUM: " << max_load_page_num << std::endl;
        std::cout << "OFFLOAD_AGG_ALLOC_SIZE: " << offload_agg_alloc_size << std::endl;
        std::cout << "OFFLOAD_AGG_ALLOC_SIZE_PER_LAYER: " << offload_agg_alloc_size_per_layer << std::endl;
        std::cout << "LOAD_AGG_ALLOC_SIZE: " << load_agg_alloc_size << std::endl;
        std::cout << "LOAD_AGG_ALLOC_SIZE_PER_LAYER: " << load_agg_alloc_size_per_layer << std::endl;
    }
    void calculateConfig() {
        model_kv_heads_gpu = model_kv_heads / gpu_num;
        model_qo_heads_gpu = model_qo_heads / gpu_num;
        model_ff_dim_gpu = model_ff_dim / gpu_num;
        ug_n = model_ff_dim_gpu * 2;
        model_hidden_dim_pergpu = model_hidden_dim / gpu_num;
        kqv_n = model_hidden_dim_pergpu + 2 * model_hidden_dim_pergpu / model_gqa;
        gpu_mem = size_t(64) * 1024 * 1024 * 1024;
        page_mem_size = frame_page_size * model_kv_heads_gpu * model_head_dim * 2 * 2;
        max_page_num = size_t(gpu_mem / model_layer / page_mem_size);
        k_temp_size = 2* size_t(max_batch_size) * model_hidden_dim * 16 + size_t(max_batch_size) * vocab_size * 2;
        offload_agg_alloc_size = size_t(model_layer) * max_offload_page_num * frame_page_size * 2 * model_kv_heads_gpu * model_head_dim * 2; // 2 for k and v, second 2 for sizeof(half)
        offload_agg_alloc_size_per_layer = size_t(max_offload_page_num) * frame_page_size * 2 * model_kv_heads_gpu * model_head_dim * 2;
        load_agg_alloc_size = size_t(model_layer) * max_load_page_num * frame_page_size * 2 * model_kv_heads_gpu * model_head_dim * 2;
        load_agg_alloc_size_per_layer = size_t(max_load_page_num) * frame_page_size * 2 * model_kv_heads_gpu * model_head_dim * 2;
        if (!rope_3_1){
            smooth_a = 0;
            smooth_b = 0;
        } else{
            smooth_a = original_max_position_embeddings / (2 * M_PI * high_freq_factor - 2 * M_PI * low_freq_factor);
            smooth_b = -1.0f / (high_freq_factor / low_freq_factor - 1.0f);
            std::cout << "[in calculate config]" << std::endl;
            std::cout << "SMOOTH_A: " << smooth_a << std::endl;
            std::cout << "SMOOTH_B: " << smooth_b << std::endl;
        }
    }
    void setConfig(int gpu_num_, int model_layer_, int run_layer_, int allocate_kv_data_batch_, 
                    int model_head_dim_, int model_gqa_, int model_kv_heads_, int model_qo_heads_,
                    int model_ff_dim_, int model_hidden_dim_, int frame_page_size_, int max_batch_size_,
                    float rms_norm_eps_, float rope_theta_, int vocab_size_, float factor_, 
                    float low_freq_factor_, float high_freq_factor_, int original_max_position_embeddings_,
                    float smooth_a_, float smooth_b_, bool rope_3_1_, int num_local_experts_, int num_experts_per_tok_,
                    int offload_page_num_, int load_page_num_) {
        gpu_num = gpu_num_;
        model_layer = model_layer_;
        run_layer = run_layer_;
        allocate_kv_data_batch = allocate_kv_data_batch_;
        model_head_dim = model_head_dim_;
        model_gqa = model_gqa_;
        model_kv_heads = model_kv_heads_;
        model_qo_heads = model_qo_heads_;
        model_ff_dim = model_ff_dim_;
        model_hidden_dim = model_hidden_dim_;
        frame_page_size = frame_page_size_;
        max_batch_size = max_batch_size_;
        rms_norm_eps = rms_norm_eps_;
        vocab_size = vocab_size_;
        rope_theta = rope_theta_;
        factor = factor_;
        low_freq_factor = low_freq_factor_;
        high_freq_factor = high_freq_factor_;
        original_max_position_embeddings = original_max_position_embeddings_;
        smooth_a = smooth_a_;
        smooth_b = smooth_b_;
        rope_3_1 = rope_3_1_;
        num_local_experts = num_local_experts_;
        num_experts_per_tok = num_experts_per_tok_;
        max_offload_page_num = offload_page_num_;
        max_load_page_num = load_page_num_;
        calculateConfig();
    }
};

extern vortexModelConfig ModelConfig;
// extern bool enable_logging;
void updateVortexModelConfig(const std::string& filename);