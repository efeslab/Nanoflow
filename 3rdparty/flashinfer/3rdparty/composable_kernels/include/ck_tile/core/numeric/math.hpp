// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include <type_traits>
#include <stdint.h>
#include <cmath>

namespace ck_tile {

template <typename Scale, Scale lhs>
struct scales_c
{
    template <typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Right& rhs) const -> decltype(lhs * rhs)
    {
        return lhs * rhs;
    }
};

template <typename Scale>
struct scales
{
    static_assert(std::is_copy_constructible_v<Scale>);

    CK_TILE_HOST_DEVICE constexpr explicit scales(Scale lhs) : lhs_(lhs) {}

    template <typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Right& rhs) const
        -> decltype(std::declval<const Scale&>() * rhs)
    {
        return lhs_ * rhs;
    }

    private:
    Scale lhs_;
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
template <typename Scale>
__host__ __device__ scales(Scale)->scales<Scale>;

template <typename Left = void, typename Right = Left>
struct plus
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs + rhs)
    {
        return lhs + rhs;
    }
};

template <>
struct plus<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs + rhs)
    {
        return lhs + rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ plus()->plus<void, void>;

template <typename Left = void, typename Right = Left>
struct minus
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs - rhs)
    {
        return lhs - rhs;
    }
};

template <>
struct minus<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs - rhs)
    {
        return lhs - rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ minus()->minus<void, void>;

template <typename Left = void, typename Right = Left>
struct multiplies
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs * rhs)
    {
        return lhs * rhs;
    }
};

template <>
struct multiplies<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs * rhs)
    {
        return lhs * rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ multiplies()->multiplies<void, void>;

template <typename T>
struct maximize
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const { return a >= b ? a : b; }
};

template <typename T>
struct minimize
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const { return a <= b ? a : b; }
};

template <typename T>
struct integer_divide_ceiler
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const
    {
        static_assert(std::is_same<T, index_t>{} || std::is_same<T, int>{}, "wrong type");
        return (a + b - number<1>{}) / b;
    }
};

template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto integer_divide_floor(X x, Y y)
{
    return x / y;
}

template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto integer_divide_ceil(X x, Y y)
{
    return (x + y - number<1>{}) / y;
}

template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto integer_least_multiple(X x, Y y)
{
    return y * integer_divide_ceil(x, y);
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T max(T x)
{
    return x;
}

template <typename T>
CK_TILE_HOST constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <typename T>
CK_TILE_DEVICE constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <>
CK_TILE_DEVICE constexpr float max(float x, float y)
{
    return __builtin_fmaxf(x, y); // can resultin v_max3_f32
}

template <>
CK_TILE_DEVICE constexpr double max(double x, double y)
{
    return __builtin_fmax(x, y); // maybe still v_max3_f32
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t max(number<X>, index_t y)
{
    return X > y ? X : y;
}

template <index_t Y>
CK_TILE_HOST_DEVICE constexpr index_t max(index_t x, number<Y>)
{
    return x > Y ? x : Y;
}

template <typename X, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto max(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");
    return max(x, max(ys...));
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T min(T x)
{
    return x;
}

template <typename T>
CK_TILE_HOST constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <typename T>
CK_TILE_DEVICE constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <>
CK_TILE_DEVICE constexpr float min(float x, float y)
{
    return __builtin_fminf(x, y);
}

template <>
CK_TILE_DEVICE constexpr double min(double x, double y)
{
    return __builtin_fmin(x, y);
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t min(number<X>, index_t y)
{
    return X < y ? X : y;
}

template <index_t Y>
CK_TILE_HOST_DEVICE constexpr index_t min(index_t x, number<Y>)
{
    return x < Y ? x : Y;
}

template <typename X, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto min(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");
    return min(x, min(ys...));
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T clamp(const T& x, const T& lowerbound, const T& upperbound)
{
    return min(max(x, lowerbound), upperbound);
}

CK_TILE_HOST int clz(uint32_t x) { return __builtin_clz(x); }
CK_TILE_DEVICE int clz(uint32_t x) { return __clz(x); }

// greatest common divisor, aka highest common factor
CK_TILE_HOST_DEVICE constexpr index_t gcd(index_t x, index_t y)
{
    if(x < 0)
    {
        return gcd(-x, y);
    }
    else if(y < 0)
    {
        return gcd(x, -y);
    }
    else if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x % y, y);
    }
    else
    {
        return gcd(x, y % x);
    }
}

template <index_t X, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto gcd(number<X>, number<Y>)
{
    constexpr auto r = gcd(X, Y);

    return number<r>{};
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto gcd(X x, Ys... ys)
{
    return gcd(x, gcd(ys...));
}

// least common multiple
template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto lcm(X x, Y y)
{
    return (x * y) / gcd(x, y);
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto lcm(X x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <typename Left = void, typename Right = Left>
struct equal
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs == rhs)
    {
        return lhs == rhs;
    }
};

template <>
struct equal<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs == rhs)
    {
        return lhs == rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ equal()->equal<void, void>;

template <>
struct equal<float, float>
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(float lhs, float rhs) const
    {
        return bit_cast<uint32_t>(lhs) == bit_cast<uint32_t>(rhs);
    }
};

template <>
struct equal<double, double>
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(double lhs, double rhs) const
    {
        return bit_cast<uint64_t>(lhs) == bit_cast<uint64_t>(rhs);
    }
};

template <typename Left = void, typename Right = Left>
struct less
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs < rhs)
    {
        return lhs < rhs;
    }
};

template <>
struct less<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs < rhs)
    {
        return lhs < rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ less()->less<void, void>;

template <typename Left = void, typename Right = Left>
struct less_equal
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs <= rhs)
    {
        return lhs <= rhs;
    }
};

template <>
struct less_equal<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs, const Right& rhs) const
        -> decltype(lhs <= rhs)
    {
        return lhs <= rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ less_equal()->less_equal<void, void>;

template <>
struct less_equal<float, float>
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(float lhs, float rhs) const
    {
        return lhs < rhs || bit_cast<uint32_t>(lhs) == bit_cast<uint32_t>(rhs);
    }
};

template <>
struct less_equal<double, double>
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(double lhs, double rhs) const
    {
        return lhs < rhs || bit_cast<uint64_t>(lhs) == bit_cast<uint64_t>(rhs);
    }
};

CK_TILE_HOST_DEVICE constexpr int32_t next_power_of_two(int32_t x)
{
    // TODO: x need to be 2 ~ 0x7fffffff. 0, 1, or larger than 0x7fffffff will compile fail
    return 1 << (32 - clz(x - 1));
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr auto next_power_of_two()
{
    constexpr index_t y = next_power_of_two(X);
    return number<y>{};
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr auto next_power_of_two(number<X>)
{
    constexpr index_t y = next_power_of_two(X);
    return number<y>{};
}

CK_TILE_HOST_DEVICE constexpr int32_t integer_log2_floor(int32_t x)
{
    // TODO: x need to be 1 ~ 0x7fffffff
    // __builtin_clz will produce unexpected result if x is 0;
    return 31 - __builtin_clz(x);
}

CK_TILE_HOST_DEVICE constexpr bool is_power_of_two_integer(int32_t x)
{
    // TODO: x need to be 1 ~ 0x7fffffff
    return x == (1 << integer_log2_floor(x));
}

#ifndef C_LOG2E
#define C_LOG2E 1.44269504088896340736 // log2(e)
#endif

template <typename T>
struct log2e;

template <>
struct log2e<double>
{
    static constexpr double value = C_LOG2E;
};

template <>
struct log2e<float>
{
    static constexpr float value = C_LOG2E;
};

template <typename T = double>
constexpr T log2e_v = log2e<T>::value;

// math
CK_TILE_HOST_DEVICE
float abs(const float& x)
{
    union
    {
        float f32;
        uint32_t u32;
    } y;
    y.f32 = x;
    y.u32 = y.u32 & 0x7fffffff;
    return y.f32;
}

CK_TILE_HOST_DEVICE
bool isnan(const float& x)
{
    uint32_t xx = bit_cast<uint32_t>(x);
    return (xx & 0x7fffffff) > 0x7F800000;
}

CK_TILE_HOST float sqrt(float x) { return std::sqrt(x); };

CK_TILE_HOST double sqrt(double x) { return std::sqrt(x); };

CK_TILE_DEVICE
float sqrt(float x) { return __builtin_amdgcn_sqrtf(x); };

CK_TILE_DEVICE
double sqrt(double x) { return __builtin_amdgcn_sqrt(x); };

CK_TILE_DEVICE
float exp(float x) { return __expf(x); };

CK_TILE_HOST
float exp(float x) { return std::expf(x); }

CK_TILE_DEVICE
float exp2(float x) { return exp2f(x); };

CK_TILE_HOST
float exp2(float x) { return std::exp2f(x); };

CK_TILE_DEVICE
float log(float x) { return __logf(x); };

CK_TILE_HOST
float log(float x) { return std::logf(x); };

CK_TILE_DEVICE uint32_t sad(uint32_t x, uint32_t y, uint32_t acc)
{
    // TODO: this is hacky, we use u16
    return __builtin_amdgcn_sad_u16(x, y, acc);
}

CK_TILE_HOST uint32_t sad(uint32_t x, uint32_t y, uint32_t acc)
{
    return (x > y ? (x - y) : (y - x)) + acc;
}

} // namespace ck_tile
