// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <ostream>

#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v4_direct_load.hpp"

namespace ck {

enum struct PipelineVersion
{
    v1,
    v2,
    // v3 is only used in the Stream-K implementation.
    v4,
    weight_only,
};

template <PipelineVersion PipelineVer,
          index_t NumPrefetch     = 1,
          LoopScheduler LoopSched = LoopScheduler::Default,
          bool AEnableLds         = true,
          bool BEnableLds         = true>
constexpr auto GridwiseGemmPipeline_Selector()
{
    if constexpr(PipelineVer == PipelineVersion::v1)
    {
        if constexpr(LoopSched == LoopScheduler::Default)
        {
            return GridwiseGemmPipeline_v1<NumPrefetch, AEnableLds, BEnableLds>{};
        }
        else if constexpr(LoopSched == LoopScheduler::Interwave)
        {
            return GridwiseGemmPipelineInterwave_v1<NumPrefetch>{};
        }
    }
    else if constexpr(PipelineVer == PipelineVersion::v2)
    {
        return GridwiseGemmPipeline_v2{};
    }
    else if constexpr(PipelineVer == PipelineVersion::v4)
    {
        return GridwiseGemmPipeline_v4<NumPrefetch>{};
    }
    else if constexpr(PipelineVer == PipelineVersion::weight_only)
    {
        return GridwiseGemmPipeline_v1_WeightOnly<NumPrefetch, AEnableLds, BEnableLds>{};
    }
    else
    {
        std::cerr << "GridwiseGemmPipeline configuration is not available" << std::endl;
    }
}

} // namespace ck

inline std::ostream& operator<<(std::ostream& os, const ck::PipelineVersion& p)
{
    switch(p)
    {
    case ck::PipelineVersion::v1: os << "PipelineVersion::v1"; break;
    case ck::PipelineVersion::v2: os << "PipelineVersion::v2"; break;
    case ck::PipelineVersion::v4: os << "PipelineVersion::v4"; break;
    case ck::PipelineVersion::weight_only: os << "PipelineVersion::weight_only"; break;
    default: os << "";
    }
    return os;
}
