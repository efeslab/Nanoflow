#pragma once
#include <cuda.h>
#include "cutlass/cutlass.h"
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include "config.h"


__global__ void moveKVcacheKernel(int finished_req_num, int32_t * finished_index,
                                         int32_t* kv_indptr, int32_t* kv_indices, half* output, half* kv_data, bool host_to_gpu = true);