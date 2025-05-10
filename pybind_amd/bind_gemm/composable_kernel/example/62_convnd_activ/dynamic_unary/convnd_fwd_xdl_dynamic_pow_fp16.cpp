// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_activ_dynamic_unary_common.hpp"

#include "../run_convnd_activ_dynamic_example.inc"

int main(int argc, char* argv[])
{

    ck::tensor_operation::element_wise::Power out_element_op(4.f, 1.f, 2.f);
    return !run_convnd_example(argc, argv, out_element_op);
}
