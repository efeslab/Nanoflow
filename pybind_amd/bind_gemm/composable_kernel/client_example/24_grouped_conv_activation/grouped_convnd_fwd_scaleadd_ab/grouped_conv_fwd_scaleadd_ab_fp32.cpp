// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"

using InDataType  = ck::Tuple<float, float>;
using WeiDataType = ck::Tuple<float, float>;
using OutDataType = float;

#include "grouped_conv_fwd_scaleadd_ab.inc"

int main() { return execute_conv_fwd_scaleadd_ab(); }
