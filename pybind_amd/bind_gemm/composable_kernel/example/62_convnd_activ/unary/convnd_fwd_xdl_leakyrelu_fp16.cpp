// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_activ_unary_common.hpp"

using OutElementOp = ck::tensor_operation::element_wise::LeakyRelu;

using DeviceGroupedConvNDActivInstance = DeviceGroupedConvNDFwdInstance<OutElementOp>;
#include "../run_convnd_activ_example.inc"

int main(int argc, char* argv[]) { return !run_convnd_example(argc, argv); }
