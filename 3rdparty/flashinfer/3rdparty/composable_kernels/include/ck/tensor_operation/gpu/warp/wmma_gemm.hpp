// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_wmma.hpp"

namespace ck {

enum struct WmmaInstr
{
    wmma_f32_16x16x16_f16 = 0,
    wmma_f32_16x16x16_bf16,
    wmma_f16_16x16x16_f16,
    wmma_bf16_16x16x16_bf16,
    wmma_i32_16x16x16_iu8,
    wmma_i32_16x16x16_iu4
};

/*
 *  WMMA Wave Tile Always MxNxK = 16x16x16
 *  WAVE32
        -----------------------------------
        |RC0| | | | | | | | | | | | | | | |	   SubGroup 0
        |RC1| | | | | | | | | | | | | | | |
        |RC2| | | | | | | | | | | | | | | |
        |RC3|T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|
        |RC4|0|0|0|0|0|0|0|0|0|1|1|1|1|1|1|
        |RC5|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5|
        |RC6| | | | | | | | | | | | | | | |
        |RC7| | | | | | | | | | | | | | | |
        -----------------------------------
        |   | | | | | | | | | | | | | | | |	   SubGroup 1
        |   | | | | | | | | | | | | | | | |
        | T |T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|
        | 1 |1|1|1|2|2|2|2|2|2|2|2|2|2|3|3|
        | 6 |7|8|9|0|1|2|3|4|5|6|7|8|9|0|1|
        |   | | | | | | | | | | | | | | | |
        |   | | | | | | | | | | | | | | | |
        |   | | | | | | | | | | | | | | | |
        -----------------------------------


 *  WAVE64
        -----------------------------------
        |RC0|T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|	   SubGroup 0
        |RC1|0|0|0|0|0|0|0|0|0|1|1|1|1|1|1|
        |RC2|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5|
        |RC3|T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|
        -----------------------------------
        | T |T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|    SubGroup 1
        | 1 |1|1|1|2|2|2|2|2|2|2|2|2|2|3|3|
        | 6 |7|8|9|0|1|2|3|4|5|6|7|8|9|0|1|
        |   | | | | | | | | | | | | | | | |
        -----------------------------------
        | T |T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|	   SubGroup 2
        | 3 |3|3|3|3|3|3|3|4|4|4|4|4|4|4|4|
        | 2 |3|4|5|6|7|8|9|0|1|2|3|4|5|6|7|
        |   | | | | | | | | | | | | | | | |
        -----------------------------------
        | T |T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|    SubGroup 3
        | 4 |4|5|5|5|5|5|5|5|5|5|5|6|6|6|6|
        | 8 |9|0|1|2|3|4|5|6|7|8|9|0|1|2|3|
        |   | | | | | | | | | | | | | | | |
        -----------------------------------

*   RC = Register for storing accumalted result
*	T  = Thread ID
*/

template <WmmaInstr Instr, index_t WaveSize, typename = void>
struct wmma_type
{
};

// A-swizzled
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_f16,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    // * Data Pixel
    static constexpr index_t m_per_wmma      = 16;
    static constexpr index_t n_per_wmma      = 16;
    static constexpr index_t k_per_wmma      = 16;
    static constexpr index_t src_a_data_size = 2;
    static constexpr index_t src_b_data_size = 2;
    static constexpr index_t acc_data_size   = 4;
    static constexpr index_t acc_pack_number = 1;
    // * Thread mapping inside wave, num_thread_per_subgroups always alone N direction
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // * Fixed on gfx11, Will be wave mode dependent for future architectures
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    // * num_acc_vgprs_per_wave alone M direction
    // * num_subgroups alone M direction
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16x16_f16_w32<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_f32_16x16x16_f16_w64<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_bf16,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16x16_bf16_w32<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_f32_16x16x16_bf16_w64<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f16_16x16x16_f16,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 2;
    static constexpr index_t acc_pack_number          = 2;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f16_16x16x16_f16_w32<MPerWmma, NPerWmma, false>::Run(a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_f16_16x16x16_f16_w64<MPerWmma, NPerWmma, false>::Run(a, b, reg_c);
        }
    }
};
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_bf16_16x16x16_bf16,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 2;
    static constexpr index_t acc_pack_number          = 2;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t Opsel,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_bf16_16x16x16_bf16_w32<MPerWmma, NPerWmma, false>::Run(a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_bf16_16x16x16_bf16_w64<MPerWmma, NPerWmma, false>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_i32_16x16x16_iu8,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma,
              index_t NPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool neg_a = false,
              bool neg_b = false,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_i32_16x16x16_iu8_w32<MPerWmma, NPerWmma, neg_a, neg_b, clamp>::Run(
                a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_i32_16x16x16_iu8_w64<MPerWmma, NPerWmma, neg_a, neg_b, clamp>::Run(
                a, b, reg_c);
        }
    }
};

template <typename src_type_a,
          typename src_type_b,
          typename dst_type,
          index_t MPerWmma,
          index_t NPerWmma>
struct WmmaSelector
{
    template <typename src_type_a_,
              typename src_type_b_,
              typename dst_type_,
              index_t MPerWmma_,
              index_t NPerWmma_>
    static constexpr auto GetWmma();

    template <>
    static constexpr auto GetWmma<half_t, half_t, float, 16, 16>()
    {
        return WmmaInstr::wmma_f32_16x16x16_f16;
    }

    template <>
    static constexpr auto GetWmma<bhalf_t, bhalf_t, float, 16, 16>()
    {
        return WmmaInstr::wmma_f32_16x16x16_bf16;
    }

    template <>
    static constexpr auto GetWmma<half_t, half_t, half_t, 16, 16>()
    {
        return WmmaInstr::wmma_f16_16x16x16_f16;
    }

    template <>
    static constexpr auto GetWmma<bhalf_t, bhalf_t, bhalf_t, 16, 16>()
    {
        return WmmaInstr::wmma_bf16_16x16x16_bf16;
    }

    template <>
    static constexpr auto GetWmma<int8_t, int8_t, int, 16, 16>()
    {
        return WmmaInstr::wmma_i32_16x16x16_iu8;
    }
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    static constexpr auto GetWmma<int4_t, int4_t, int, 16, 16>()
    {
        return WmmaInstr::wmma_i32_16x16x16_iu4;
    }
#endif
    // get_warp_size do not return the correct wavesize, hardcode to 32 as workaround
    static constexpr auto selected_wmma =
        wmma_type<GetWmma<src_type_a, src_type_b, dst_type, MPerWmma, NPerWmma>(), Number<32>{}>{};

    __host__ __device__ constexpr WmmaSelector()
    {
        static_assert(selected_wmma.m_per_wmma == 16, "WRONG! WMMA_M must equal to 16");

        static_assert(selected_wmma.m_per_wmma == 16, "WRONG! WMMA_M must equal to 16");

        static_assert(selected_wmma.k_per_wmma == 16, "WRONG! WMMA_M must equal to 16");

        static_assert(selected_wmma.wave_size * selected_wmma.num_acc_vgprs_per_wave *
                              selected_wmma.acc_data_size * selected_wmma.acc_pack_number ==
                          selected_wmma.m_per_wmma * selected_wmma.n_per_wmma * 4,
                      "WRONG! Invalid Number of Accumulator Register");
    }
};

template <typename src_type_a,
          typename src_type_b,
          typename dst_type,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t KPack,
          bool TransposeC      = false,
          bool AssemblyBackend = false>
struct WmmaGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using CIndex   = MultiIndex<2>;
    using CIndex3D = MultiIndex<3>;

    __host__ __device__ constexpr WmmaGemm()
    {
        static_assert(NPerWmma == 16 && MPerWmma == 16,
                      "Only support GemmNPerWmma == 16 and GemmMPerWmma == 16 for wmma");

        static_assert(KPack % wmma_instr.k_per_wmma == 0, "KPack should be multiple of k_per_wmma");
    }

    // WMMA output supporting C = A * B
    // Vector Write
    // MPerWMMA_NPerWMMA -> MSubGroup_..._NPerWMMA_MAccVgprPerWave
    template <typename CDesc_MBlockxRepeat_MWave_MPerWMMA_NBlockxRepeat_NWave_NPerWMMA>
    __host__ __device__ static constexpr auto
    MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
        const CDesc_MBlockxRepeat_MWave_MPerWMMA_NBlockxRepeat_NWave_NPerWMMA&
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma)
    {
        const auto MBlockxRepeat =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I0);
        const auto NBlockxRepeat =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I3);
        const auto MWave =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I1);
        const auto NWave =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I4);

        return transform_tensor_descriptor(
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma,
            make_tuple(
                make_pass_through_transform(MBlockxRepeat),
                make_pass_through_transform(MWave),
                make_unmerge_transform(make_tuple(Number<wmma_instr.num_subgroups>{},
                                                  Number<wmma_instr.num_acc_vgprs_per_wave>{})),
                make_pass_through_transform(NBlockxRepeat),
                make_pass_through_transform(NWave),
                make_pass_through_transform(Number<wmma_instr.num_thread_per_subgroups>{})),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2, 6>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}));
    }

    // Transposed WMMA Output C' = B' * A'
    template <typename CDesc_MBlockxRepeat_MWave_MPerWMMA_NBlockxRepeat_NWave_NPerWMMA>
    __host__ __device__ static constexpr auto
    MakeCDesc_MBlockxRepeat_MWave_MThreadPerSubGroup_NBlockxRepeat_NWave_NSubGroup_NAccVgprs(
        const CDesc_MBlockxRepeat_MWave_MPerWMMA_NBlockxRepeat_NWave_NPerWMMA&
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma)
    {
        const auto MBlockxRepeat =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I0);
        const auto NBlockxRepeat =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I3);
        const auto MWave =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I1);
        const auto NWave =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I4);

        return transform_tensor_descriptor(
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma,
            make_tuple(
                make_pass_through_transform(MBlockxRepeat),
                make_pass_through_transform(MWave),
                make_pass_through_transform(Number<wmma_instr.num_thread_per_subgroups>{}),
                make_pass_through_transform(NBlockxRepeat),
                make_pass_through_transform(NWave),
                make_unmerge_transform(make_tuple(Number<wmma_instr.num_subgroups>{},
                                                  Number<wmma_instr.num_acc_vgprs_per_wave>{}))),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6>{}));
    }

    __device__ static constexpr index_t GetRegSizePerWmma()
    {
        return wmma_instr.num_acc_vgprs_per_wave * wmma_instr.acc_pack_number;
    }

    __device__ static constexpr index_t GetWaveSize() { return wmma_instr.wave_size; }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(
            (is_same<src_type_a, half_t>::value && is_same<src_type_b, half_t>::value &&
             is_same<dst_type, float>::value) ||
                (is_same<src_type_a, bhalf_t>::value && is_same<src_type_b, bhalf_t>::value &&
                 is_same<dst_type, float>::value) ||
                (is_same<src_type_a, half_t>::value && is_same<src_type_b, half_t>::value &&
                 is_same<dst_type, half_t>::value) ||
                (is_same<src_type_a, bhalf_t>::value && is_same<src_type_b, bhalf_t>::value &&
                 is_same<dst_type, bhalf_t>::value) ||
                (is_same<src_type_a, int8_t>::value && is_same<src_type_b, int8_t>::value &&
                 is_same<dst_type, int32_t>::value)
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                || (is_same<src_type_a, int4_t>::value && is_same<src_type_b, int4_t>::value &&
                    is_same<dst_type, int32_t>::value)
#endif
                ,
            "base type couple must be (half, float), (bhalf, float), (half, half), (bhalf, bhalf), "
            "(int8, int32) or (int4, int32)!");
        static_for<0, KPack / wmma_instr.k_per_wmma, 1>{}([&](auto k) {
            if constexpr(!TransposeC)
            {
                wmma_instr.template run<MPerWmma, NPerWmma>(p_a_wave[k], p_b_wave[k], p_c_thread);
            }
            else
            {
                wmma_instr.template run<MPerWmma, NPerWmma>(p_b_wave[k], p_a_wave[k], p_c_thread);
            }
        });
    }

    __device__ static auto GetLaneId() { return get_thread_local_1d_id() % wmma_instr.wave_size; }

    __device__ static auto GetSubGroupId()
    {
        return (GetLaneId() / wmma_instr.num_thread_per_subgroups) % wmma_instr.num_subgroups;
    }

    __device__ static auto GetLaneIdUnderSubGroup()
    {
        return GetLaneId() % wmma_instr.num_thread_per_subgroups;
    }
    __device__ static auto GetSwizzledLaneIdLow()
    {
        return ((GetLaneIdUnderSubGroup() & 1) << 3) | (GetLaneIdUnderSubGroup() >> 1);
    }

    __host__ __device__ static auto CalculateAThreadOriginDataIndex()
    {
        return TransposeC ? GetLaneIdUnderSubGroup() : GetSwizzledLaneIdLow();
    }

    __host__ __device__ static auto CalculateBThreadOriginDataIndex()
    {
        return TransposeC ? GetSwizzledLaneIdLow() : GetLaneIdUnderSubGroup();
    }

    __device__ static CIndex GetBeginOfThreadBlk()
    {
        index_t n_offset = GetLaneIdUnderSubGroup();
        index_t m_offset = GetSubGroupId() * wmma_instr.num_acc_vgprs_per_wave;

        return TransposeC ? CIndex{n_offset, m_offset} : CIndex{m_offset, n_offset};
    }

    __device__ static CIndex3D GetBeginOfThreadBlk3D()
    {
        index_t n_offset = GetLaneIdUnderSubGroup();
        index_t m_offset = GetSubGroupId();

        return TransposeC ? CIndex3D{n_offset, m_offset, I0} : CIndex3D{m_offset, n_offset, I0};
    }

    static constexpr auto wmma =
        WmmaSelector<src_type_a, src_type_b, dst_type, MPerWmma, NPerWmma>{};
    static constexpr auto wmma_instr = wmma.selected_wmma;

    __host__ __device__ static constexpr auto
    GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths()
    {
        return make_tuple(I1,
                          I1,
                          Number<wmma_instr.num_acc_vgprs_per_wave>{},
                          Number<wmma_instr.acc_pack_number>{});
    }
};

} // namespace ck
