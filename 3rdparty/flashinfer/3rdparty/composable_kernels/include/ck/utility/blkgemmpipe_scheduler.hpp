// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

enum struct BlockGemmPipelineScheduler
{
    Intrawave,
    Interwave,
};

enum struct TailNumber
{
    // Single / Double buffer pipeline
    Odd,
    Even,

    // Long prefetch pipeline, up to 8
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,

    // Unroll stages > Prefetch stages, number of loop is multiple of unroll stages
    Empty,
    // Unroll stages <= Prefetch stages, number of loop is multiple of unroll stages add
    // prefetchstages
    Full,
};
template <index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t ABufferLoadWidth,
          index_t BBufferLoadWidth,
          index_t ALDSWriteWidth,
          index_t BLDSWriteWidth,
          index_t ALDSReadWidth,
          index_t BLDSReadWidth,
          index_t MRepeat,
          index_t NRepeat,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t KPerXDL>
struct BlockwiseGemmXdlops_pipeline_hotloop_inst
{
    static constexpr index_t WaveSize = 64;
    static constexpr index_t WaveNumM = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t WaveNumN = NPerBlock / (NRepeat * NPerXDL);

    static constexpr index_t A_LDS_Read_Width = ALDSReadWidth;
    static constexpr index_t B_LDS_Read_Width = BLDSReadWidth;

    static constexpr index_t A_Buffer_Load_Inst_Num =
        MPerBlock * KPerBlock / (BlockSize * ABufferLoadWidth);
    static constexpr index_t B_Buffer_Load_Inst_Num =
        NPerBlock * KPerBlock / (BlockSize * BBufferLoadWidth);

    static constexpr index_t A_LDS_Write_Inst_Num =
        MPerBlock * KPerBlock / (BlockSize * ALDSWriteWidth);
    static constexpr index_t B_LDS_Write_Inst_Num =
        NPerBlock * KPerBlock / (BlockSize * BLDSWriteWidth);

    static constexpr index_t A_LDS_Read_Inst_Num =
        WaveNumN * MPerBlock * KPerBlock / (BlockSize * ALDSReadWidth);
    static constexpr index_t B_LDS_Read_Inst_Num =
        WaveNumM * MPerBlock * KPerBlock / (BlockSize * BLDSReadWidth);

    static constexpr index_t C_MFMA_Inst_Num =
        MPerBlock * NPerBlock * KPerBlock / (BlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

    static constexpr auto Print()
    {
        printf(" Blk/Wave Size: %d, %d, M/N/K PerBlk: %d, %d, %d, M/N/K PerXdl: %d, %d, %d\n",
               BlockSize,
               WaveSize,
               MPerBlock,
               NPerBlock,
               KPerBlock,
               MPerXDL,
               NPerXDL,
               KPerXDL);

        printf(" A/B buffer load inst: %d, %d\n A/B LDS write inst: %d, %d\n A/B LDS read inst: "
               "%d, %d\n C MFMA inst: %d\n",
               A_Buffer_Load_Inst_Num,
               B_Buffer_Load_Inst_Num,
               A_LDS_Write_Inst_Num,
               B_LDS_Write_Inst_Num,
               A_LDS_Read_Inst_Num,
               B_LDS_Read_Inst_Num,
               C_MFMA_Inst_Num);
    }
};

} // namespace ck
