// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
// Define the common macro for gfx94x models
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define __gfx94__
#endif

// fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x1f32;

template <>
struct intrin_mfma_f32_32x32x1f32<64, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
        reg_c.template AsType<float32_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<1>{}], 1, 1, 0);
    }
};

template <>
struct intrin_mfma_f32_32x32x1f32<32, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x2f32;

template <>
struct intrin_mfma_f32_32x32x2f32<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x2f32(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x4f32;

template <>
struct intrin_mfma_f32_16x16x4f32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x4f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x1f32;

template <>
struct intrin_mfma_f32_16x16x1f32<16, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x1f32(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 2, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_4x4x1f32;

template <>
struct intrin_mfma_f32_4x4x1f32<4, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
    }
};

template <>
struct intrin_mfma_f32_4x4x1f32<8, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
        reg_c.template AsType<float4_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<1>{}], 4, 1, 0);
    }
};

// fp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x4f16;

template <>
struct intrin_mfma_f32_32x32x4f16<64, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
        reg_c.template AsType<float32_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<1>{}], 1, 1, 0);
    }
};

template <>
struct intrin_mfma_f32_32x32x4f16<32, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x8f16;

template <>
struct intrin_mfma_f32_32x32x8f16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x8f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x16f16;

template <>
struct intrin_mfma_f32_16x16x16f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x16f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x4f16;

template <>
struct intrin_mfma_f32_16x16x4f16<16, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x4f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 2, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_4x4x4f16;

template <>
struct intrin_mfma_f32_4x4x4f16<4, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
    }
};

template <>
struct intrin_mfma_f32_4x4x4f16<8, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
        reg_c.template AsType<float4_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<1>{}], 4, 1, 0);
    }
};

// bfp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x8bf16_1k;

template <>
struct intrin_mfma_f32_32x32x8bf16_1k<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_a, const bhalf4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x16bf16_1k;

template <>
struct intrin_mfma_f32_16x16x16bf16_1k<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_a, const bhalf4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x4bf16;

template <>
struct intrin_mfma_f32_32x32x4bf16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bhalf2_t& reg_a, const bhalf2_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4bf16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x8bf16;

template <>
struct intrin_mfma_f32_16x16x8bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf2_t& reg_a, const bhalf2_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x8bf16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_32x32x8i8;

template <>
struct intrin_mfma_i32_32x32x8i8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const int8x4_t& reg_a, const int8x4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<int32x16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_32x32x8i8(bit_cast<int32_t>(reg_a),
                                                bit_cast<int32_t>(reg_b),
                                                reg_c.template AsType<int32x16_t>()[Number<0>{}],
                                                0,
                                                0,
                                                0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_16x16x16i8;

template <>
struct intrin_mfma_i32_16x16x16i8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const int8x4_t& reg_a, const int8x4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_16x16x16i8(bit_cast<int32_t>(reg_a),
                                                 bit_cast<int32_t>(reg_b),
                                                 reg_c.template AsType<int32x4_t>()[Number<0>{}],
                                                 0,
                                                 0,
                                                 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_32x32x16i8;

template <>
struct intrin_mfma_i32_32x32x16i8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<int32x16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_32x32x16_i8(bit_cast<int64_t>(reg_a),
                                                  bit_cast<int64_t>(reg_b),
                                                  reg_c.template AsType<int32x16_t>()[Number<0>{}],
                                                  0,
                                                  0,
                                                  0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_16x16x32i8;

template <>
struct intrin_mfma_i32_16x16x32i8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_16x16x32i8(bit_cast<int64_t>(reg_a),
                                                 bit_cast<int64_t>(reg_b),
                                                 reg_c.template AsType<int32x4_t>()[Number<0>{}],
                                                 0,
                                                 0,
                                                 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f64_16x16x4f64;

template <>
struct intrin_mfma_f64_16x16x4f64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const double& reg_a, const double& reg_b, FloatC& reg_c)
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        reg_c.template AsType<double4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f64_16x16x4f64(
            reg_a, reg_b, reg_c.template AsType<double4_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16f8f8;

template <>
struct intrin_mfma_f32_32x32x16f8f8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                bit_cast<long>(reg_a),
                bit_cast<long>(reg_b),
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
#else
        vector_type<f8_t, 8> reg_a_v(reg_a);
        vector_type<f8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<f8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<f8_t>()[Number<k>{}]);

            intrin_mfma_f32_32x32x2f32<32, 32>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32f8f8;

template <>
struct intrin_mfma_f32_16x16x32f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            bit_cast<long>(reg_a),
            bit_cast<long>(reg_b),
            reg_c.template AsType<float4_t>()[Number<0>{}],
            0,
            0,
            0);
#else
        vector_type<f8_t, 8> reg_a_v(reg_a);
        vector_type<f8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<f8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<f8_t>()[Number<k>{}]);

            intrin_mfma_f32_16x16x4f32<16, 16>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16bf8bf8;

template <>
struct intrin_mfma_f32_32x32x16bf8bf8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                bit_cast<long>(reg_a),
                bit_cast<long>(reg_b),
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
#else
        vector_type<bf8_t, 8> reg_a_v(reg_a);
        vector_type<bf8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<bf8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<bf8_t>()[Number<k>{}]);

            intrin_mfma_f32_32x32x2f32<32, 32>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32bf8bf8;

template <>
struct intrin_mfma_f32_16x16x32bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(
            bit_cast<long>(reg_a),
            bit_cast<long>(reg_b),
            reg_c.template AsType<float4_t>()[Number<0>{}],
            0,
            0,
            0);
#else
        vector_type<bf8_t, 8> reg_a_v(reg_a);
        vector_type<bf8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<bf8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<bf8_t>()[Number<k>{}]);

            intrin_mfma_f32_16x16x4f32<16, 16>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16f8bf8;

template <>
struct intrin_mfma_f32_32x32x16f8bf8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                bit_cast<long>(reg_a),
                bit_cast<long>(reg_b),
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
#else
        vector_type<f8_t, 8> reg_a_v(reg_a);
        vector_type<bf8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<f8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<bf8_t>()[Number<k>{}]);

            intrin_mfma_f32_32x32x2f32<32, 32>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32f8bf8;

template <>
struct intrin_mfma_f32_16x16x32f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(
            bit_cast<long>(reg_a),
            bit_cast<long>(reg_b),
            reg_c.template AsType<float4_t>()[Number<0>{}],
            0,
            0,
            0);
#else
        vector_type<f8_t, 8> reg_a_v(reg_a);
        vector_type<bf8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<f8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<bf8_t>()[Number<k>{}]);

            intrin_mfma_f32_16x16x4f32<16, 16>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16bf8f8;

template <>
struct intrin_mfma_f32_32x32x16bf8f8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                bit_cast<long>(reg_a),
                bit_cast<long>(reg_b),
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
#else
        vector_type<bf8_t, 8> reg_a_v(reg_a);
        vector_type<f8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<bf8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<f8_t>()[Number<k>{}]);

            intrin_mfma_f32_32x32x2f32<32, 32>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32bf8f8;

template <>
struct intrin_mfma_f32_16x16x32bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(
            bit_cast<long>(reg_a),
            bit_cast<long>(reg_b),
            reg_c.template AsType<float4_t>()[Number<0>{}],
            0,
            0,
            0);
#else
        vector_type<bf8_t, 8> reg_a_v(reg_a);
        vector_type<f8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<bf8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<f8_t>()[Number<k>{}]);

            intrin_mfma_f32_16x16x4f32<16, 16>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

} // namespace ck
