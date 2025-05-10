// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v2.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using F16 = ck::half_t;
using F32 = float;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmInstance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV2<
        ALayout,   BLayout,  CLayout,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 4,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>,
        1, 8, 4, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
