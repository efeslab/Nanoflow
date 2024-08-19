#pragma once
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "helper.h"
#include <nvtx3/nvToolsExt.h>

#define ENABLE_NETWORK
// #define ENABLE_MPI

constexpr int GPU_NUM = 8;
constexpr int MODEL_LAYER = 80;
constexpr int RUN_LAYER = 80;
constexpr int ALLOCATE_KV_DATA_BATCH = 1500;


constexpr int MODEL_HEAD_DIM = 128;
constexpr int MODEL_GQA = 8;
// FIXME: the KV and QO heads likely means heads per GPU
constexpr int MODEL_KV_HEADS = 1;
constexpr int MODEL_QO_HEADS = 8;

constexpr int MODEL_FF_DIM = 28 * 1024;
constexpr int MODEL_FF_DIM_GPU = MODEL_FF_DIM / GPU_NUM;

constexpr int UG_N = MODEL_FF_DIM_GPU * 2;

constexpr int MODEL_HIDDEN_DIM = 8192;
constexpr int MODEL_HIDDEN_DIM_PERGPU = MODEL_HIDDEN_DIM / GPU_NUM;
// TODO: when supporting GQA and updating the above MODEL_{KV,QO}_HEADS, update the below static_assert
static_assert(MODEL_HEAD_DIM * MODEL_QO_HEADS * GPU_NUM == MODEL_HIDDEN_DIM);
constexpr int KQV_N = MODEL_HIDDEN_DIM_PERGPU + 2*MODEL_HIDDEN_DIM_PERGPU / MODEL_GQA;

constexpr int FRAME_PAGE_SIZE = 16;

constexpr int MAX_BATCH_SIZE = 2048;
constexpr int PAGE_SIZE = 16;
constexpr size_t GPU_MEM = size_t(MODEL_LAYER) * 1024 * 1024 * 1024;
constexpr size_t PAGE_MEM_SIZE =  PAGE_SIZE * MODEL_KV_HEADS * MODEL_HEAD_DIM * 2 * 2;
constexpr size_t MAX_PAGE_NUM = size_t(GPU_MEM / MODEL_LAYER/ PAGE_MEM_SIZE);
