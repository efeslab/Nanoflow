// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/f8_utils.hpp"
#include "ck/utility/random_gen.hpp"

namespace ck {
// Define the common macro for gfx94x models
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define __gfx94__
#endif

// Convert X to Y, both X and Y are non-const data types.
template <typename Y,
          typename X,
          std::enable_if_t<!(std::is_const_v<Y> || std::is_const_v<X>), bool> = false>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    return static_cast<Y>(x);
}

// Convert X to Y, either X or Y is a const data type.
template <typename Y,
          typename X,
          std::enable_if_t<std::is_const_v<Y> || std::is_const_v<X>, bool> = false>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    using NonConstY = std::remove_const_t<Y>;
    using NonConstX = std::remove_const_t<X>;
    return static_cast<Y>(type_convert<NonConstY, NonConstX>(x));
}

// convert bfp16 to fp32
template <>
inline __host__ __device__ constexpr float type_convert<float, bhalf_t>(bhalf_t x)
{
    union
    {
        uint32_t int32;
        float fp32;
    } u = {uint32_t(x) << 16};

    return u.fp32;
}

// convert fp32 to bfp16
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, float>(float x)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    return uint16_t(u.int32 >> 16);
}

// convert bfp16 to fp16 via fp32
template <>
inline __host__ __device__ constexpr half_t type_convert<half_t, bhalf_t>(bhalf_t x)
{
    float x_fp32 = type_convert<float>(x);

    return static_cast<half_t>(x_fp32);
}

// convert fp16 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, half_t>(half_t x)
{
    float x_fp32 = static_cast<float>(x);

    return type_convert<bhalf_t>(x_fp32);
}

// convert bfp16 to int8 via fp32
template <>
inline __host__ __device__ constexpr int8_t type_convert<int8_t, bhalf_t>(bhalf_t x)
{
    float x_fp32 = type_convert<float>(x);

    return static_cast<int8_t>(x_fp32);
}

// convert int8 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, int8_t>(int8_t x)
{
    float x_fp32 = static_cast<float>(x);

    return type_convert<bhalf_t>(x_fp32);
}

// Convert X to Y
template <typename Y, typename X>
__host__ __device__ constexpr Y type_convert_sp(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    return static_cast<Y>(x);
}

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, float>(float x)
{
    union
    {
        float fp32;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr float type_convert_sp<float, int>(int x)
{
    union
    {
        int int32;
        float fp32;
    } u = {x};

    return u.fp32;
}

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, half_t>(half_t x)
{
    union
    {
        half_t fp16;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr half_t type_convert_sp<half_t, int>(int x)
{
    union
    {
        int int32;
        half_t fp16;
    } u = {x};

    return u.fp16;
}

// Declare a template function for fp8 conversion using SR
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_sr(X x);

// convert fp32 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_t f8_convert_sr<f8_t, float>(float x)
{
    constexpr int seed = 1254739;
    uint32_t rng       = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_fp8 = 240.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_fp8, -max_fp8);
    ival       = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
    val.i32val = ival;
    return val.i8val[0]; // little endian
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    return utils::
        cast_to_f8<float, f8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(x,
                                                                                               rng);
#endif
}

// convert fp16 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_t f8_convert_sr<f8_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_sr<f8_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 1254739;
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::
        cast_to_f8<half_t, f8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp32 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_t f8_convert_sr<bf8_t, float>(float x)
{
    constexpr int seed = 1254739;
    uint32_t rng       = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_bf8 = 57344.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_bf8, -max_bf8);
    ival       = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
    val.i32val = ival;
    return val.i8val[0]; // little endian
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    return utils::
        cast_to_f8<float, bf8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp16 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_t f8_convert_sr<bf8_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_sr<bf8_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 1254739;
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::
        cast_to_f8<half_t, bf8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_rne(X x);

// convert fp32 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_t f8_convert_rne<f8_t, float>(float x)
{
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_fp8 = 240.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_fp8, -max_fp8);
    ival       = __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false); // false -> WORD0
    val.i32val = ival;
    return val.i8val[0];
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::
        cast_to_f8<float, f8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(x,
                                                                                               rng);
#endif
}

// convert fp16 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_t f8_convert_rne<f8_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_rne<f8_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::
        cast_to_f8<half_t, f8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp32 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_t f8_convert_rne<bf8_t, float>(float x)
{
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_bf8 = 57344.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_bf8, -max_bf8);
    ival       = __builtin_amdgcn_cvt_pk_bf8_f32(val.fval, val.fval, ival, false); // false -> WORD0
    val.i32val = ival;
    return val.i8val[0];
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::
        cast_to_f8<float, bf8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp16 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_t f8_convert_rne<bf8_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_rne<bf8_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::
        cast_to_f8<half_t, bf8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp32 to fp8
template <>
inline __host__ __device__ f8_t type_convert<f8_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_t>(x);
#else
    return f8_convert_rne<f8_t>(x);
#endif
}

// convert fp8 to fp32
template <>
inline __host__ __device__ float type_convert<float, f8_t>(f8_t x)
{
#if defined(__gfx94__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    fval            = __builtin_amdgcn_cvt_f32_fp8(i32val, 0);
    // asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<f8_t, float, negative_zero_nan>(x);
#endif
}

template <>
inline __host__ __device__ float2_t type_convert<float2_t, f8x2_t>(f8x2_t x)
{
#if defined(__gfx94__)
    const auto i16val = bit_cast<uint16_t>(x);
    return __builtin_amdgcn_cvt_pk_f32_fp8(i16val, 0);
#else
    constexpr bool negative_zero_nan = true;
    const auto f8x2_v                = vector_type<f8_t, 2>(x);
    vector_type<float, 2> f32x2_v;
    f32x2_v.template AsType<float>()(Number<0>{}) =
        utils::cast_from_f8<f8_t, float, negative_zero_nan>(
            f8x2_v.template AsType<f8_t>()[Number<0>{}]);
    f32x2_v.template AsType<float>()(Number<1>{}) =
        utils::cast_from_f8<f8_t, float, negative_zero_nan>(
            f8x2_v.template AsType<f8_t>()[Number<1>{}]);
    return f32x2_v.template AsType<float2_t>()[Number<0>{}];
#endif
}

template <>
inline __host__ __device__ half2_t type_convert<half2_t, float2_t>(float2_t x)
{

    const vector_type<float, 2> f32x2_v(x);
    const auto y = __builtin_amdgcn_cvt_pkrtz(f32x2_v.template AsType<float>()[Number<0>{}],
                                              f32x2_v.template AsType<float>()[Number<1>{}]);
    return bit_cast<half2_t>(y);
}

// convert fp16 to fp8
template <>
inline __host__ __device__ f8_t type_convert<f8_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_t>(x);
#else
    return f8_convert_rne<f8_t>(x);
#endif
}

// convert fp8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, f8_t>(f8_t x)
{
#if defined(__gfx94__)
    // use native conversion to float and convert to fp16
    return type_convert<half_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<f8_t, half_t, negative_zero_nan>(x);
#endif
}

// convert fp32 to bf8
template <>
inline __host__ __device__ bf8_t type_convert<bf8_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_t>(x);
#else
    return f8_convert_rne<bf8_t>(x);
#endif
}

// convert bf8 to fp32
template <>
inline __host__ __device__ float type_convert<float, bf8_t>(bf8_t x)
{
#if defined(__gfx94__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    fval            = __builtin_amdgcn_cvt_f32_bf8(i32val, 0);
    // asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<bf8_t, float, negative_zero_nan>(x);
#endif
}

// convert fp16 to bf8
template <>
inline __host__ __device__ bf8_t type_convert<bf8_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_t>(x);
#else
    return f8_convert_rne<bf8_t>(x);
#endif
}

// convert bf8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, bf8_t>(bf8_t x)
{
#if defined(__gfx94__)
    // use native conversion to float and convert to fp16
    return type_convert<half_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<bf8_t, half_t, negative_zero_nan>(x);
#endif
}

// Declare a template function for bf16 conversion using RTN
template <typename Y, typename X>
__host__ __device__ constexpr Y bf16_convert_rtn(X x);

// Convert fp32 to bf16 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, float>(float x)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    // When the exponent bits are not all 1s, then the value is zero, normal,
    // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
    // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
    // This causes the bfloat16's mantissa to be incremented by 1 if the 16
    // least significant bits of the float mantissa are greater than 0x8000,
    // or if they are equal to 0x8000 and the least significant bit of the
    // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
    // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
    // has the value 0x7f, then incrementing it causes it to become 0x00 and
    // the exponent is incremented by one, which is the next higher FP value
    // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
    // with an exponent of 0x00 and a mantissa of 0x7f, it may be rounded up
    // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
    // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
    // incrementing it causes it to become an exponent of 0xFF and a mantissa
    // of 0x00, which is Inf, the next higher value to the unrounded value.
    bool flag0 = ~u.int32 & 0x7f800000;

    // When all of the exponent bits are 1, the value is Inf or NaN.
    // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
    // mantissa bit. Quiet NaN is indicated by the most significant mantissa
    // bit being 1. Signaling NaN is indicated by the most significant
    // mantissa bit being 0 but some other bit(s) being 1. If any of the
    // lower 16 bits of the mantissa are 1, we set the least significant bit
    // of the bfloat16 mantissa, in order to preserve signaling NaN in case
    // the bfloat16's mantissa bits are all 0.
    bool flag1 = !flag0 && (u.int32 & 0xffff);

    u.int32 += flag0 ? 0x7fff + ((u.int32 >> 16) & 1) : 0; // Round to nearest, round to even
    u.int32 |= flag1 ? 0x10000 : 0x0;                      // Preserve signaling NaN

    return uint16_t(u.int32 >> 16);
}

// convert fp16 to bfp16 via fp32 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, half_t>(half_t x)
{
    float x_fp32 = static_cast<float>(x);

    return bf16_convert_rtn<bhalf_t>(x_fp32);
}
} // namespace ck
