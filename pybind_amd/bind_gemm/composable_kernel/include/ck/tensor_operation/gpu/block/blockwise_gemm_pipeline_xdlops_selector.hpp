// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v4.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v5.hpp"

namespace ck {

enum struct BlockGemmPipelineVersion
{
    v1, // Naive
    v2, // Mem
    v3, // Comp
    v4, // Comp, double lds buffer
    v5, // Comp, double global prefetch register buffer
};

template <BlockGemmPipelineVersion BlkGemmPipelineVer,
          BlockGemmPipelineScheduler BlkGemmPipeSche,
          index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeDataType,
          typename AccDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
constexpr auto BlockGemmPipeline_Selector()
{
    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
    {
        return BlockwiseGemmXdlops_pipeline_v1<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
    {
        return BlockwiseGemmXdlops_pipeline_v2<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
    {
        return BlockwiseGemmXdlops_pipeline_v3<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
    {
        return BlockwiseGemmXdlops_pipeline_v4<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v5)
    {
        return BlockwiseGemmXdlops_pipeline_v5<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else
    {
        std::cerr << "BlockGemmPipeline configuration is not available" << std::endl;
    }
}

} // namespace ck
