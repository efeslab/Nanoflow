// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

namespace ck_tile {
/*
 * construct this structure with behavior as:
 *
 *   // create stream config with default stream(NULL), and not timing the kernel
 *   stream_config s = stream_config{};
 *
 *   // create stream config with _some_stream_id_, and not timing the kernel
 *   stream_config s = stream_config{_some_stream_id_};
 *
 *   // create stream config with _some_stream_id_, and benchmark with warmup/repeat as default
 *   stream_config s = stream_config{_some_stream_id_, true};
 *
 *   // create stream config with _some_stream_id_, and benchmark using cpu timer
 *   stream_config s = stream_config{_some_stream_id_, true, 0, 3, 10, false};
 **/

struct stream_config
{
    hipStream_t stream_id_ = nullptr;
    bool time_kernel_      = false;
    int log_level_         = 0;
    int cold_niters_       = 3;
    int nrepeat_           = 10;
    bool is_gpu_timer_     = true; // keep compatible
};
} // namespace ck_tile
