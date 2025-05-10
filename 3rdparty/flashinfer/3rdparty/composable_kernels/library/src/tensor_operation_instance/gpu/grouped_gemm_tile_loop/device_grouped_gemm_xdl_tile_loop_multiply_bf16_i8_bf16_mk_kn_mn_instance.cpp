// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using I8   = int8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough         = ck::tensor_operation::element_wise::PassThrough;
using Multiply            = ck::tensor_operation::element_wise::Multiply;
using MultiplyAddFastGelu = ck::tensor_operation::element_wise::MultiplyAddFastGelu;
using MultiplyFastGelu    = ck::tensor_operation::element_wise::MultiplyFastGelu;
using MultiplyAdd         = ck::tensor_operation::element_wise::MultiplyAdd;

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <typename DsLayout, typename DsDataType, typename CDEElementwiseOp>
using device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_irregular_tile_instances = std::tuple<
// clang-format off
        //###########################################|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|      DsData| EData|           A|           B|                C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //###########################################| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|        Type|  Type| Elementwise| Elementwise|      Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //###########################################|       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|        Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //###########################################|       |       |            |       |      |      |        |         |            |      |            |            |                 |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
#if 1
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 8>,              8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,              8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,   128,    64,    32,   8,   2,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 8>,              8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,   128,    64,    32,   8,   8,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,              8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,    64,   128,    32,   8,   2,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 8>,              8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 8>,              8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,   128,    64,    32,   8,   2,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 4>,              8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,   128,    64,    32,   8,   8,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 4>,              8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,    64,   128,    32,   8,   2,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 16, 1, 8>,              8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,    64,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1,           1,           1,               S<1, 16, 1, 8>,              8>
#endif
#if 0
	//comp
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,   128,   128,    64,   8,   4,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<16, 16, 1>,    S<0, 2, 1>,     S<0, 2, 1>,             1,               8,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,   128,   128,    64,   8,   4,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<16, 16, 1>,    S<0, 2, 1>,     S<0, 2, 1>,             1,               8,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,   128,   256,    32,   8,   4,  32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,               8,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,   128,   128,    64,   8,   4,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<16, 16, 1>,    S<0, 2, 1>,     S<0, 2, 1>,             1,               8,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8>,

	//latency
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,    64,    16,   16,   256,   8,   4,  16,   16,    1,    1,     S<32, 2, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<64, 1, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 4>,               4>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,    16,   32,   256,   8,   4,  16,   16,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<64, 2, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 8>,               4>,

	//mem
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,    64,    16,   16,   256,   8,   4,  16,   16,    1,    1,     S<32, 2, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<64, 1, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 4>,               4>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,    16,   32,   256,   8,   4,  16,   16,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<64, 2, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 8>,               4>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,    16,   64,   128,   8,   4,  16,   16,    1,    2,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<32, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 8>,               4>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,    32,   64,   128,   8,   4,  32,   32,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<32, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 8>,               8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,    16,  128,    64,   8,   4,  16,   16,    1,    4,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<16, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 8>,               4>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   128,    32,  128,    64,   8,   4,  32,   32,    1,    2,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<16, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 8>,               8>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,    16,  256,    64,   8,   4,  16,   16,    1,    4,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<16, 16, 1>,    S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 16>,              4>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<    Row,    Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp, GemmMNKPadding,        1,   256,    32,  256,    64,   8,   4,  32,   32,    1,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<16, 16, 1>,    S<0, 2, 1>,     S<0, 2, 1>,             1,              16,              4,          0,          1,           1,                   S<1, 16, 1, 16>,              8>
#endif
    // clang-format on
    >;

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          ck::Tuple<Row>,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          ck::Tuple<BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_irregular_tile_instances<
            ck::Tuple<Row>,
            ck::Tuple<BF16>,
            Multiply>{});
}

void add_device_grouped_gemm_xdl_tile_loop_multiply_bias_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          ck::Tuple<Row, Row>,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          ck::Tuple<BF16, BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAdd>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_irregular_tile_instances<
            ck::Tuple<Row, Row>,
            ck::Tuple<BF16, BF16>,
            MultiplyAdd>{});
}

void add_device_grouped_gemm_xdl_tile_loop_multiply_bias_fastgelu_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          ck::Tuple<Row, Row>,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          ck::Tuple<BF16, BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAddFastGelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_irregular_tile_instances<
            ck::Tuple<Row, Row>,
            ck::Tuple<BF16, BF16>,
            MultiplyAddFastGelu>{});
}

void add_device_grouped_gemm_xdl_tile_loop_multiply_fastgelu_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          ck::Tuple<Row>,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          ck::Tuple<BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyFastGelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_irregular_tile_instances<
            ck::Tuple<Row>,
            ck::Tuple<BF16>,
            MultiplyFastGelu>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
