// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_activ_multi_ab_common.hpp"

using DataType    = ck::half_t;
using AccDataType = float;
using InDataType  = DataType;
using WeiDataType = DataType;
using OutDataType = DataType;
using ADataTypes  = ck::Tuple<DataType, DataType>;
using BDataTypes  = ck::Tuple<DataType, DataType>;

using InElementOp  = ck::tensor_operation::element_wise::ScaleAdd;
using WeiElementOp = ck::tensor_operation::element_wise::ScaleAdd;

using DeviceGroupedConvNDActivInstance = DeviceGroupedConvNDMultiABFwdInstance<DataType,
                                                                               AccDataType,
                                                                               ADataTypes,
                                                                               BDataTypes,
                                                                               InElementOp,
                                                                               WeiElementOp>;

#include "../run_convnd_activ_example.inc"

int main(int argc, char* argv[]) { return !run_convnd_example(argc, argv); }
