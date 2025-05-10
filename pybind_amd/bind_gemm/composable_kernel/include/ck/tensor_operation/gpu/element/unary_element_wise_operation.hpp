// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/amd_inline_asm.hpp"
#include <cassert>

namespace ck {

// Fast int4x4 to half8_t data type conversion based on paper
// [Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production]
// (https://arxiv.org/abs/2211.10017) and implementation:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__host__ __device__ inline half4_t pki4_to_half4(int q)
{
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;

    // Extract the two int4 at low bit and create two fp16 number.
    int lo = amd_assembly_and_or_b32(q, LO, EX);
    // Extract the two int4 at hight bit and create two fp16 number.
    int hi = amd_assembly_and_or_b32(q, HI, EX);

    const int SUB = 0xE408E408; // half2 {-1032, -1032}
    const int MUL = 0x2c002c00; // half2 {1 / 16, 1 / 16}
    const int ADD = 0xd480d480; // half2 {-72, -72}

    vector_type<half_t, 4> res;

    // for two fp16 from lowbit, subtract 1032 to get correct fp16 value
    res.template AsType<half2_t>()(Number<0>{}) =
        amd_assembly_pk_add_f16(bit_cast<half2_t>(lo), bit_cast<half2_t>(SUB));

    // for two fp16 from highbit, divide 16 and subtract 72 to get correct fp16 value
    res.template AsType<half2_t>()(Number<1>{}) = amd_assembly_pk_fma_f16(
        bit_cast<half2_t>(hi), bit_cast<half2_t>(MUL), bit_cast<half2_t>(ADD));

    return res.template AsType<half4_t>()[Number<0>{}];
}

__host__ __device__ inline half4_t pki4_to_half4_scale(int q, const ck::half2_t& scale)
{
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;

    // Extract the two int4 at low bit and create two fp16 number.
    int lo = amd_assembly_and_or_b32(q, LO, EX);
    // Extract the two int4 at hight bit and create two fp16 number.
    int hi = amd_assembly_and_or_b32(q, HI, EX);

    const int SUB = 0xE408E408; // half2 {-1032, -1032}
    const int MUL = 0x2c002c00; // half2 {1 / 16, 1 / 16}
    const int ADD = 0xd480d480; // half2 {-72, -72}

    vector_type<half_t, 4> res;

    res.template AsType<half2_t>()(Number<0>{}) =
        amd_assembly_pk_add_f16(bit_cast<half2_t>(lo), bit_cast<half2_t>(SUB));

    res.template AsType<half2_t>()(Number<1>{}) = amd_assembly_pk_fma_f16(
        bit_cast<half2_t>(hi), bit_cast<half2_t>(MUL), bit_cast<half2_t>(ADD));

    asm volatile("v_pk_mul_f16 %0, %1, %2"
                 : "=v"(res.template AsType<half2_t>()(Number<0>{}))
                 : "v"(res.template AsType<half2_t>()(Number<0>{})), "v"(scale));

    asm volatile("v_pk_mul_f16 %0, %1, %2"
                 : "=v"(res.template AsType<half2_t>()(Number<1>{}))
                 : "v"(res.template AsType<half2_t>()(Number<1>{})), "v"(scale));

    return res.template AsType<half4_t>()[Number<0>{}];
}

__host__ __device__ inline half2_t pki4_to_half2(pk_i4_t q)
{
#if 1
    uint8_t x_u8 = ck::bit_cast<uint8_t>(q);
    uint32_t i4s = ((x_u8 & 0x0f) << 16) | ((x_u8 & 0xf0) >> 4);

    const int EX  = 0x64006400;
    const int SUB = 0xE408E408; //-8

    int lo = i4s | EX;

    return amd_assembly_pk_add_f16(bit_cast<half2_t>(lo), bit_cast<half2_t>(SUB));
#else
    uint8_t x_u8 = ck::bit_cast<uint8_t>(q);

    vector_type<half_t, 2> res;

    half_t x_h = (x_u8 & 0x0f) - 8;
    half_t x_l = ((x_u8 & 0xf0) >> 4) - 8;

    res.template AsType<half_t>()(Number<0>{}) = x_l;
    res.template AsType<half_t>()(Number<1>{}) = x_h;

    return res.template AsType<half2_t>()[Number<0>{}];
#endif
}

__host__ __device__ inline bhalf4_t pki4_to_bhalf4(int q)
{
    uint32_t i8s = (q & 0xf) | ((q & 0xf0) << 4) | ((q & 0xf00) << 8) | ((q & 0xf000) << 12);

    static constexpr uint32_t fp32_base = 0x4B000000;

    float fp32_intermediates[4];

    uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);

    fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
    fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);
    fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7652);
    fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

    fp32_intermediates[0] -= 8388616.f;
    fp32_intermediates[1] -= 8388616.f;
    fp32_intermediates[2] -= 8388616.f;
    fp32_intermediates[3] -= 8388616.f;

    vector_type<bhalf_t, 4> res;
    res.template AsType<bhalf2_t>()(Number<0>{}) = bit_cast<bhalf2_t>(
        __byte_perm(fp32_intermediates_casted[1], fp32_intermediates_casted[0], 0x7632));
    res.template AsType<bhalf2_t>()(Number<1>{}) = bit_cast<bhalf2_t>(
        __byte_perm(fp32_intermediates_casted[3], fp32_intermediates_casted[2], 0x7632));

    return res.template AsType<bhalf4_t>()[Number<0>{}];
}

__host__ __device__ inline bhalf2_t pki4_to_bhalf2(pk_i4_t q)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(q);

    float x_h = ((x_u8 & 0x0f) >> 0) - 8.f;
    float x_l = ((x_u8 & 0xf0) >> 4) - 8.f;

    vector_type<bhalf_t, 2> res;

    res.template AsType<bhalf_t>()(Number<0>{}) = type_convert<bhalf_t>(x_l);
    res.template AsType<bhalf_t>()(Number<1>{}) = type_convert<bhalf_t>(x_h);

    return res.template AsType<bhalf2_t>()[Number<0>{}];
}

namespace tensor_operation {
namespace element_wise {

struct PassThroughPack8
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    __host__ __device__ constexpr void operator()(ck::half8_t& y, const ck::pk_i4x4_t& x) const
    {
#if 1
        vector_type<half_t, 8> result;

        result.template AsType<half4_t>()(Number<0>{}) = pki4_to_half4(bit_cast<int>(x));
        result.template AsType<half4_t>()(Number<1>{}) = pki4_to_half4(bit_cast<int>(x) >> 8);

        y = result.template AsType<half8_t>()[Number<0>{}];
#else
        vector_type<half_t, 8> dst;
        vector_type<pk_i4_t, 4> src{x};

        dst.template AsType<half2_t>()(Number<0>{}) =
            pki4_to_half2(src.template AsType<pk_i4_t>()[Number<0>{}]);
        dst.template AsType<half2_t>()(Number<1>{}) =
            pki4_to_half2(src.template AsType<pk_i4_t>()[Number<1>{}]);
        dst.template AsType<half2_t>()(Number<2>{}) =
            pki4_to_half2(src.template AsType<pk_i4_t>()[Number<2>{}]);
        dst.template AsType<half2_t>()(Number<3>{}) =
            pki4_to_half2(src.template AsType<pk_i4_t>()[Number<3>{}]);

        y = dst.template AsType<half8_t>()[Number<0>{}];
#endif
    }

    __host__ __device__ constexpr void operator()(ck::bhalf8_t& y, const ck::pk_i4x4_t& x) const
    {
#if 1
        vector_type<bhalf_t, 8> result;

        result.template AsType<bhalf4_t>()(Number<0>{}) = pki4_to_bhalf4(bit_cast<int>(x));
        result.template AsType<bhalf4_t>()(Number<1>{}) = pki4_to_bhalf4(bit_cast<int>(x) >> 16);

        y = result.template AsType<bhalf8_t>()[Number<0>{}];
#else
        vector_type<bhalf_t, 8> dst;
        vector_type<pk_i4_t, 4> src{x};

        dst.template AsType<bhalf2_t>()(Number<0>{}) =
            pki4_to_bhalf2(src.template AsType<pk_i4_t>()[Number<0>{}]);
        dst.template AsType<bhalf2_t>()(Number<1>{}) =
            pki4_to_bhalf2(src.template AsType<pk_i4_t>()[Number<1>{}]);
        dst.template AsType<bhalf2_t>()(Number<2>{}) =
            pki4_to_bhalf2(src.template AsType<pk_i4_t>()[Number<2>{}]);
        dst.template AsType<bhalf2_t>()(Number<3>{}) =
            pki4_to_bhalf2(src.template AsType<pk_i4_t>()[Number<3>{}]);

        y = dst.template AsType<bhalf8_t>()[Number<0>{}];
#endif
    }
    constexpr const static bool is_pack8_invocable = true;
};

struct DequantPack8
{
    template <typename Y, typename X, typename Z>
    __host__ __device__ void operator()(Y& y, const X& x, const Z& z) const;

    __host__ __device__ constexpr void
    operator()(ck::half8_t& y, const ck::pk_i4x4_t& x, const ck::half2_t& z) const
    {
#if 1
        vector_type<half_t, 8> result;

        result.template AsType<half4_t>()(Number<0>{}) = pki4_to_half4_scale(bit_cast<int>(x), z);
        result.template AsType<half4_t>()(Number<1>{}) =
            pki4_to_half4_scale(bit_cast<int>(x) >> 8, z);

        y = result.template AsType<half8_t>()[Number<0>{}];
#else
        vector_type<half_t, 8> dst;
        vector_type<pk_i4_t, 4> src{x};

        dst.template AsType<half2_t>()(Number<0>{}) =
            pki4_to_half2(src.template AsType<pk_i4_t>()[Number<0>{}]);
        dst.template AsType<half2_t>()(Number<1>{}) =
            pki4_to_half2(src.template AsType<pk_i4_t>()[Number<1>{}]);
        dst.template AsType<half2_t>()(Number<2>{}) =
            pki4_to_half2(src.template AsType<pk_i4_t>()[Number<2>{}]);
        dst.template AsType<half2_t>()(Number<3>{}) =
            pki4_to_half2(src.template AsType<pk_i4_t>()[Number<3>{}]);

        y          = dst.template AsType<half8_t>()[Number<0>{}];
#endif
    }

    constexpr const static bool is_pack8_invocable = true;
};

struct PassThroughPack2
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    __host__ __device__ constexpr void operator()(half2_t& y, const f8x2_t& x) const
    {
        auto t = type_convert<float2_t>(x);
        y      = type_convert<half2_t>(t);
    }

    __host__ __device__ constexpr void operator()(ck::half2_t& y, const ck::pk_i4_t& x) const
    {
#if 1
        uint8_t x_u8 = ck::bit_cast<uint8_t>(x);
        uint8_t x_l  = (x_u8 & 0x0f) >> 0;
        uint8_t x_h  = (x_u8 & 0xf0) >> 4;

        auto l_f16 = ck::type_convert<ck::half_t>(x_l);
        auto h_f16 = ck::type_convert<ck::half_t>(x_h);

        y = {l_f16, h_f16};
#else
        uint32_t t = ck::bit_cast<uint8_t>(x);
        y          = ck::bit_cast<half2_t>(t);
#endif
    }

    constexpr const static bool is_pack2_invocable = true;
};

struct PassThrough
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<pk_i4_t, pk_i4_t>(pk_i4_t& y, const pk_i4_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<double, double>(double& y, const double& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<float, double>(float& y, const double& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<double, float>(double& y, const float& x) const
    {
        y = type_convert<double>(x);
    }

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<half_t, float>(half_t& y, const float& x) const
    {
        y = type_convert<half_t>(x);
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<int32_t, int32_t>(int32_t& y, const int32_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, float>(bhalf_t& y, const float& x) const
    {
        y = type_convert<bhalf_t>(x);
    }

    template <>
    __host__ __device__ void operator()<float, bhalf_t>(float& y, const bhalf_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, half_t>(bhalf_t& y, const half_t& x) const
    {
        y = type_convert<bhalf_t>(x);
    }

    template <>
    __host__ __device__ void operator()<float, half_t>(float& y, const half_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<int8_t, int8_t>(int8_t& y, const int8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<half_t, int8_t>(half_t& y, const int8_t& x) const
    {
        y = type_convert<half_t>(x);
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, int8_t>(bhalf_t& y, const int8_t& x) const
    {
        y = type_convert<bhalf_t>(x);
    }

    template <>
    __host__ __device__ void operator()<uint8_t, uint8_t>(uint8_t& y, const uint8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<int8_t, int32_t>(int8_t& y, const int32_t& x) const
    {
        y = type_convert<int8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<int32_t, int8_t>(int32_t& y, const int8_t& x) const
    {
        y = type_convert<int32_t>(x);
    }

    template <>
    __host__ __device__ void operator()<int8_t, float>(int8_t& y, const float& x) const
    {
        y = type_convert<int8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<float, int8_t>(float& y, const int8_t& x) const
    {
        y = type_convert<float>(x);
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    __host__ __device__ void operator()<int4_t, int4_t>(int4_t& y, const int4_t& x) const
    {
        y = x;
    }
    template <>
    __host__ __device__ void operator()<int4_t, int>(int4_t& y, const int& x) const
    {
        y = type_convert<int4_t>(x);
    }
#endif

    template <>
    __host__ __device__ void operator()<f8_t, f8_t>(f8_t& y, const f8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<float, f8_t>(float& y, const f8_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<f8_t, float>(f8_t& y, const float& x) const
    {
        y = type_convert<f8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<half_t, f8_t>(half_t& y, const f8_t& x) const
    {
        y = type_convert<half_t>(x);
    }

    template <>
    __host__ __device__ void operator()<f8_t, half_t>(f8_t& y, const half_t& x) const
    {
        y = type_convert<f8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<bf8_t, bf8_t>(bf8_t& y, const bf8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<float, bf8_t>(float& y, const bf8_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<bf8_t, float>(bf8_t& y, const float& x) const
    {
        y = type_convert<bf8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<half_t, bf8_t>(half_t& y, const bf8_t& x) const
    {
        y = type_convert<half_t>(x);
    }

    template <>
    __host__ __device__ void operator()<bf8_t, half_t>(bf8_t& y, const half_t& x) const
    {
        y = type_convert<bf8_t>(x);
    }
};

struct UnaryConvert
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
    }
};

struct ConvertBF16RTN
{
    // convert to bf16 using round to nearest (rtn)
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(is_same<Y, bhalf_t>::value, "Data type is not supported by this operation!");

        // check X datatype
        static_assert(is_same<X, float>::value || is_same<X, half_t>::value,
                      "Data type is not supported by this operation!");

        y = bf16_convert_rtn<Y>(x);
    }
};

struct ConvertF8SR
{
    // convert to fp8 using stochastic rounding (SR)
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(is_same<Y, f8_t>::value || is_same<Y, bf8_t>::value,
                      "Data type is not supported by this operation!");

        // check X datatype
        static_assert(is_same<X, float>::value || is_same<X, half_t>::value,
                      "Data type is not supported by this operation!");

        y = f8_convert_sr<Y>(x);
    }
};

struct ConvertF8RNE
{
    // convert to fp8 using rounding to nearest even
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(is_same<Y, f8_t>::value || is_same<Y, bf8_t>::value,
                      "Data type is not supported by this operation!");

        // check X datatype
        static_assert(is_same<X, float>::value || is_same<X, half_t>::value,
                      "Data type is not supported by this operation!");

        y = f8_convert_rne<Y>(x);
    }
};

struct Scale
{
    __host__ __device__ Scale(float scale = 1.f) : scale_(scale) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(type_convert<float>(x) * scale_);
    }

    template <>
    __host__ __device__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        y = type_convert<half_t>(scale_) * x;
    };

    template <>
    __host__ __device__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        const float x_tmp = type_convert<float>(x);
        const float y_tmp = scale_ * x_tmp;
        y                 = type_convert<bhalf_t>(y_tmp);
    };

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = scale_ * x;
    };

    template <>
    __host__ __device__ void operator()<double, double>(double& y, const double& x) const
    {
        y = scale_ * x;
    };

    template <>
    __host__ __device__ void operator()<int8_t, int8_t>(int8_t& y, const int8_t& x) const
    {
        y = type_convert<int8_t>(scale_ * type_convert<float>(x));
    };

    float scale_;
};

struct ScaleAndResetNaNToMinusInfinity
{
    __host__ __device__ ScaleAndResetNaNToMinusInfinity(float scale) : scale_(scale) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = math::isnan(x) ? -NumericLimits<float>::Infinity() : scale_ * x;
    };

    float scale_;
};

struct UnaryDivide
{
    __host__ __device__ UnaryDivide(const int32_t divider = 1) : divider_(divider) {}

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value || is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");

        y = x / type_convert<T>(divider_);
    };

    template <>
    __host__ __device__ void operator()<half_t>(half_t& y, const half_t& x) const
    {
        float x_         = type_convert<float>(x);
        float divider_f_ = type_convert<float>(divider_);

        y = type_convert<half_t>(x_ / divider_f_);
    };

    template <>
    __host__ __device__ void operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        float x_         = type_convert<float>(x);
        float divider_f_ = type_convert<float>(divider_);

        y = type_convert<bhalf_t>(x_ / divider_f_);
    };

    template <>
    __host__ __device__ void operator()<f8_t>(f8_t& y, const f8_t& x) const
    {
        float x_         = type_convert<float>(x);
        float divider_f_ = type_convert<float>(divider_);

        y = type_convert<f8_t>(x_ / divider_f_);
    };

    int32_t divider_ = 1;
};

struct UnarySquare
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same_v<T, float> || is_same_v<T, half_t> || is_same_v<T, double> ||
                          is_same_v<T, int32_t> || is_same_v<T, int8_t>
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                          || is_same_v<T, int4_t>
#endif
                      ,
                      "Data type is not supported by this operation!");
        y = x * x;
    };
};

struct UnaryAbs
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {

        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");

        y = math::abs(x);
    };

    template <>
    __host__ __device__ void operator()(f8_t& y, const f8_t& x) const
    {
        y = ck::type_convert<f8_t>(ck::math::abs(ck::type_convert<float>(x)));
    };
};

struct UnarySqrt
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        y = math::sqrt(x);
    };
};

struct Relu
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        y = x > 0 ? x : 0;
    }

    template <>
    __host__ __device__ void operator()(bhalf_t& y, const bhalf_t& x) const
    {
        float x_f32 = type_convert<float>(x);
        float y_f32 = x_f32 > 0 ? x_f32 : 0;
        y           = type_convert<bhalf_t>(y_f32);
    }
};

// Fast GeLU
// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// host code use higher accuracy "exp" and "div"
// gpu code use lower accuracy "_ocml_exp_f32" and "rcp" function
struct FastGelu
{
    template <typename Y, typename X>
    __host__ void operator()(Y& y, const X& x) const;

    template <typename Y, typename X>
    __device__ void operator()(Y& y, const X& x) const;
#ifndef CK_CODE_GEN_RTC
    template <>
    __host__ void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = -2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = exp(u);
        y               = x / (1.f + emu);
    }
#endif

    // device code, use lower precision "__ocml_exp_f32" and "rcp"
    template <>
    __device__ void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = __ocml_exp_f32(u);

        y = x * math::rcp(1.f + emu);
    }

    template <>
    __host__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<half_t>(y_f);
    }

    template <>
    __device__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<half_t>(y_f);
    }

    template <>
    __host__ void operator()<half_t, float>(half_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<half_t>(y_f);
    }

    template <>
    __device__ void operator()<half_t, float>(half_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<half_t>(y_f);
    }

    template <>
    __host__ void operator()<bhalf_t, float>(bhalf_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<bhalf_t>(y_f);
    }

    template <>
    __device__ void operator()<bhalf_t, float>(bhalf_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<bhalf_t>(y_f);
    }

    template <>
    __device__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<bhalf_t>(y_f);
    }

    template <>
    __host__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<bhalf_t>(y_f);
    }
};

// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+erf(x/sqrt(2)))
struct Gelu
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = 0.5f * x * (1.f + erf(float(0.70710678118f * x)));
    }

    template <>
    __host__ __device__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        y = half_t(0.5) * x * (half_t(1) + half_t(erf(float(0.70710678118f * x))));
    }
};

struct Sigmoid
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");
        constexpr T one = type_convert<T>(1);
        y               = one / (one + math::exp(-x));
    };
};

struct Silu
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same_v<T, float> || is_same_v<T, double> || is_same_v<T, half_t> ||
                          is_same_v<T, int8_t> || is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");
        constexpr T one = type_convert<T>(1);
        y               = x * (one / (one + math::exp(-x)));
    };
};

struct TanH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::tanh(x);
    };
};

struct ACos
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::acos(x);
    };
};

struct Neg
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::neg(x);
    };
};

struct ATan
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::atan(x);
    };
};

struct Sin
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::sin(x);
    };
};

struct ASinH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::asinh(x);
    };
};

struct Cos
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = cos(x);
    };
};

struct ACosH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::acosh(x);
    };
};

struct Tan
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::tan(x);
    };
};

struct ATanH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::atanh(x);
    };
};

struct SinH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::sinh(x);
    };
};

struct Ceil
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::ceil(x);
    };
};

struct Exp
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::exp(x);
    };
};

struct CosH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::cosh(x);
    };
};

struct Floor
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::floor(x);
    };
};

struct Log
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::log(x);
    };
};

struct ASin
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::asin(x);
    };
};

struct Rcp
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = math::rcp(x);
    };
};

struct Swish
{
    Swish(float beta = 1.0f) : beta_(beta) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        static_assert(is_same<X, float>::value || is_same<X, double>::value ||
                          is_same<X, ck::half_t>::value || is_same<X, int8_t>::value,
                      "Data type is not supported by this operation!");

        static_assert(is_same<Y, float>::value || is_same<Y, double>::value ||
                          is_same<Y, ck::half_t>::value || is_same<Y, int8_t>::value,
                      "Data type is not supported by this operation!");

        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<Y>(x / (1.f + math::exp(bx)));
    };

    const float beta_;
};

struct SoftRelu
{
    SoftRelu(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha  = type_convert<T>(alpha_);
        constexpr T one = type_convert<T>(1);
        y               = math::log(one + math::exp(x * casted_alpha)) / casted_alpha;
    }
    const float alpha_;
};

struct Power
{
    Power(float alpha = 0.f, float beta = 1.f, float gamma = 2.f)
        : alpha_(alpha), beta_(beta), gamma_(gamma){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha     = type_convert<T>(alpha_);
        T casted_beta      = type_convert<T>(beta_);
        T casted_gamma     = type_convert<T>(gamma_);
        T shifted_scaled_x = casted_alpha + casted_beta * x;
        y                  = math::pow(shifted_scaled_x, casted_gamma);
    }
    const float alpha_;
    const float beta_;
    const float gamma_;
};

struct ClippedRelu
{
    ClippedRelu(float alpha = 0.f, float beta = 1.f) : alpha_(alpha), beta_(beta){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        T casted_beta  = type_convert<T>(beta_);
        y              = math::min(casted_beta, math::max(casted_alpha, x));
    }
    const float alpha_;
    const float beta_;
};

struct LeakyRelu
{
    LeakyRelu(float alpha = 0.01f) : alpha_(alpha){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        y              = x >= 0 ? x : x * casted_alpha;
    }
    const float alpha_;
};

struct Elu
{
    Elu(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        y              = x > 0 ? x : casted_alpha * math::expm1(x);
    }
    const float alpha_;
};

struct Logistic
{
    Logistic(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha  = type_convert<T>(alpha_);
        constexpr T one = type_convert<T>(1);
        y               = casted_alpha / (one + ck::math::exp(-x) * casted_alpha);
    }
    const float alpha_;
};

struct ConvInvscale
{
    __host__ __device__ ConvInvscale(float scale_in  = 1.f,
                                     float scale_wei = 1.f,
                                     float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    __host__ __device__ void operator()(E& e, const C& c) const;

    template <>
    __host__ __device__ void operator()<f8_t, float>(f8_t& e, const float& c) const
    {
        e = type_convert<f8_t>(c / scale_in_ / scale_wei_ / scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

struct ConvScale
{
    __host__ __device__ ConvScale(float scale_in  = 1.f,
                                  float scale_wei = 1.f,
                                  float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    __host__ __device__ void operator()(E& e, const C& c) const;

    template <>
    __host__ __device__ void operator()<f8_t, float>(f8_t& e, const float& c) const
    {
        e = type_convert<f8_t>(c * scale_in_ * scale_wei_ * scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

struct ConvScaleRelu
{
    __host__ __device__ ConvScaleRelu(float scale_in  = 1.f,
                                      float scale_wei = 1.f,
                                      float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    __host__ __device__ void operator()(E& e, const C& c) const;

    template <>
    __host__ __device__ void operator()<f8_t, float>(f8_t& e, const float& c) const
    {
        float x;
        Relu{}.template operator()<float>(x, c * scale_in_ * scale_wei_);
        e = type_convert<f8_t>(x * scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

// support fastconvert of int8 to fp16

template <typename InputDataType, typename OutputDataType, index_t RegPackNumber>
struct FastNumericArrayConverter
{
};

template <>
struct FastNumericArrayConverter<uint8_t, half_t, 4>
{
    using InputArray  = vector_type<uint8_t, 4>;
    using OutputArray = vector_type<half_t, 4>;

    __device__ static OutputArray convert(InputArray const& Input)
    {
        OutputArray Output;

        uint32_t* half_2       = reinterpret_cast<uint32_t*>(&Output);
        uint32_t const uint8_4 = reinterpret_cast<uint32_t const&>(Input);

        static constexpr uint32_t byte_selector_01 = 0x05010500;
        static constexpr uint32_t byte_selector_23 = 0x05030502;
        static constexpr uint32_t fp16_adder       = 0x64646464;
        half_2[0] = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_01);
        half_2[1] = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_23);

        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
        asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                     : "=v"(half_2[0])
                     : "v"(half_2[0]), "s"(I8s_TO_F16s_MAGIC_NUM));
        asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                     : "=v"(half_2[1])
                     : "v"(half_2[1]), "s"(I8s_TO_F16s_MAGIC_NUM));

        return Output;
    }

    __device__ OutputArray operator()(InputArray const& Input) { return convert(Input); }
};

template <index_t N>
struct FastNumericArrayConverter<uint8_t, half_t, N>
{
    static constexpr int VEC_WIDTH = 4;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

    using InputArray  = vector_type<uint8_t, N>;
    using OutputArray = vector_type<half_t, N>;

    __device__ static OutputArray convert(InputArray const& Input)
    {
        FastNumericArrayConverter<uint8_t, ck::half_t, 4> converter;

        OutputArray Output;

        using Vec_InputArray  = vector_type<uint8_t, 4>;
        using Vec_OutputArray = vector_type<half_t, 4>;

        Vec_OutputArray* half_4_ptr       = reinterpret_cast<Vec_OutputArray*>(&Output);
        Vec_InputArray const* uint8_4_ptr = reinterpret_cast<Vec_InputArray const*>(&Input);

        static_for<0, N / VEC_WIDTH, 1>{}(
            [&](auto i) { half_4_ptr[i] = converter(uint8_4_ptr[i]); });

        return Output;
    }

    __device__ OutputArray operator()(InputArray const& Input) { return convert(Input); }
};

struct DynamicUnaryOp
{
    __host__ __device__ DynamicUnaryOp() = delete;

    __host__ __device__ DynamicUnaryOp(const Swish& swish)
        : unary_op_type_(UnaryOpType::Swish), swish_{swish.beta_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const Swish&& swish)
        : unary_op_type_(UnaryOpType::Swish), swish_{swish.beta_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const Sigmoid&) : unary_op_type_(UnaryOpType::Sigmoid) {}

    __host__ __device__ DynamicUnaryOp(const Sigmoid&&) : unary_op_type_(UnaryOpType::Sigmoid) {}

    __host__ __device__ DynamicUnaryOp(const PassThrough&)
        : unary_op_type_(UnaryOpType::PassThrough)
    {
    }

    __host__ __device__ DynamicUnaryOp(const PassThrough&&)
        : unary_op_type_(UnaryOpType::PassThrough)
    {
    }

    __host__ __device__ DynamicUnaryOp(const Logistic& logistic)
        : unary_op_type_(UnaryOpType::Logistic), logistic_{logistic.alpha_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const Logistic&& logistic)
        : unary_op_type_(UnaryOpType::Logistic), logistic_{logistic.alpha_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const TanH&) : unary_op_type_(UnaryOpType::TanH) {}

    __host__ __device__ DynamicUnaryOp(const TanH&&) : unary_op_type_(UnaryOpType::TanH) {}

    __host__ __device__ DynamicUnaryOp(const Relu&) : unary_op_type_(UnaryOpType::Relu) {}

    __host__ __device__ DynamicUnaryOp(const Relu&&) : unary_op_type_(UnaryOpType::Relu) {}

    __host__ __device__ DynamicUnaryOp(const SoftRelu& softrelu)
        : unary_op_type_(UnaryOpType::SoftRelu), soft_relu_{softrelu.alpha_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const SoftRelu&& softrelu)
        : unary_op_type_(UnaryOpType::SoftRelu), soft_relu_{softrelu.alpha_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const UnaryAbs&) : unary_op_type_(UnaryOpType::UnaryAbs) {}

    __host__ __device__ DynamicUnaryOp(const UnaryAbs&&) : unary_op_type_(UnaryOpType::UnaryAbs) {}

    __host__ __device__ DynamicUnaryOp(const Power& pow)
        : unary_op_type_(UnaryOpType::Power), power_(pow.alpha_, pow.beta_, pow.gamma_)
    {
    }

    __host__ __device__ DynamicUnaryOp(const Power&& pow)
        : unary_op_type_(UnaryOpType::Power), power_(pow.alpha_, pow.beta_, pow.gamma_)
    {
    }

    __host__ __device__ DynamicUnaryOp(const ClippedRelu& clippedrelu)
        : unary_op_type_(UnaryOpType::ClippedRelu),
          clipped_relu_{clippedrelu.alpha_, clippedrelu.beta_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const ClippedRelu&& clippedrelu)
        : unary_op_type_(UnaryOpType::ClippedRelu),
          clipped_relu_{clippedrelu.alpha_, clippedrelu.beta_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const LeakyRelu& leakyrelu)
        : unary_op_type_(UnaryOpType::LeakyRelu), leaky_relu_{leakyrelu.alpha_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const LeakyRelu&& leakyrelu)
        : unary_op_type_(UnaryOpType::LeakyRelu), leaky_relu_{leakyrelu.alpha_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const Elu& elu)
        : unary_op_type_(UnaryOpType::Elu), elu_{elu.alpha_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const Elu&& elu)
        : unary_op_type_(UnaryOpType::Elu), elu_{elu.alpha_}
    {
    }

    __host__ __device__ DynamicUnaryOp(const DynamicUnaryOp& dynamic_op) = default;

    __host__ __device__ ~DynamicUnaryOp() {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        switch(unary_op_type_)
        {
        case(UnaryOpType::Swish): swish_(y, x); break;
        case(UnaryOpType::Sigmoid): sigmoid_(y, x); break;
        case(UnaryOpType::PassThrough): pass_through_(y, x); break;
        case(UnaryOpType::Logistic): logistic_(y, x); break;
        case(UnaryOpType::TanH): tanh_(y, x); break;
        case(UnaryOpType::Relu): relu_(y, x); break;
        case(UnaryOpType::SoftRelu): soft_relu_(y, x); break;
        case(UnaryOpType::UnaryAbs): unary_abs_(y, x); break;
        case(UnaryOpType::Power): power_(y, x); break;
        case(UnaryOpType::ClippedRelu): clipped_relu_(y, x); break;
        case(UnaryOpType::LeakyRelu): leaky_relu_(y, x); break;
        case(UnaryOpType::Elu): elu_(y, x); break;
        default: break;
        }
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        float y_float;
        float x_float = type_convert<float>(x);
        this->operator()(y_float, x_float);
        y = type_convert<bhalf_t>(y_float);
    }

    private:
    enum class UnaryOpType
    {
        Swish,
        Sigmoid,
        PassThrough,
        Logistic,
        TanH,
        Relu,
        SoftRelu,
        UnaryAbs,
        Power,
        ClippedRelu,
        LeakyRelu,
        Elu
    };

    public:
    UnaryOpType unary_op_type_;

    Swish swish_;
    Sigmoid sigmoid_;
    PassThrough pass_through_;
    Logistic logistic_;
    TanH tanh_;
    Relu relu_;
    SoftRelu soft_relu_;
    UnaryAbs unary_abs_;
    Power power_;
    ClippedRelu clipped_relu_;
    LeakyRelu leaky_relu_;
    Elu elu_;
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
