// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/amd_ck_fp8.hpp"
#include "ck/utility/statically_indexed_array.hpp"
#ifdef CK_CODE_GEN_RTC
using int8_t   = signed char;
using uint8_t  = unsigned char;
using int16_t  = signed short;
using uint16_t = unsigned short;
using float_t  = float;
#endif
namespace ck {

#ifdef CK_CODE_GEN_RTC
using byte = unsigned char;
#else
using std::byte;
#endif

using bhalf_t = ushort;
using half_t  = _Float16;
using int4_t  = _BitInt(4);

// custom data type - pack int4 data
struct pk_i4_t
{
    using type = int8_t;
    type data;
    __host__ __device__ constexpr pk_i4_t() : data{type{}} {}
    __host__ __device__ constexpr pk_i4_t(type init) : data{init} {}
};

inline constexpr auto next_pow2(uint32_t x)
{
    // Precondition: x > 1.
    return x > 1u ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}

// native types: double, float, _Float16, ushort, int32_t, int8_t, uint8_t, f8_fnuz_t, bf8_fnuz_t,
// native types: bool
template <typename T>
inline constexpr bool is_native_type()
{
    return is_same<T, double>::value || is_same<T, float>::value || is_same<T, half_t>::value ||
           is_same<T, bhalf_t>::value || is_same<T, int32_t>::value || is_same<T, int8_t>::value ||
           is_same<T, uint8_t>::value || is_same<T, f8_fnuz_t>::value ||
           is_same<T, bf8_fnuz_t>::value || is_same<T, bool>::value;
}

// vector_type
template <typename T, index_t N, typename Enable = void>
struct vector_type;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<T __attribute__((ext_vector_type(V))), N>;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<vector_type<T, V>, N>;

// vector_type_maker
// This is the right way to handle "vector of vectors": making a bigger vector instead
template <typename T, index_t N>
struct vector_type_maker
{
    using type = vector_type<T, N>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<T __attribute__((ext_vector_type(N1))), N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<vector_type<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
__host__ __device__ constexpr auto make_vector_type(Number<N>)
{
    return typename vector_type_maker<T, N>::type{};
}

// scalar_type
template <typename TV>
struct scalar_type;

// is_scalar_type
template <typename TV>
struct is_scalar_type
{
    static constexpr bool value = (scalar_type<remove_cvref_t<TV>>::vector_size == 1);
};

// has_same_scalar_type
template <typename X, typename Y>
using has_same_scalar_type = is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                     typename scalar_type<remove_cvref_t<Y>>::type>;

template <typename T, index_t N>
struct scalar_type<T __attribute__((ext_vector_type(N)))>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct scalar_type<vector_type<T, N>>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

//
template <>
struct scalar_type<double>
{
    using type                           = double;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<float>
{
    using type                           = float;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<half_t>
{
    using type                           = half_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bhalf_t>
{
    using type                           = bhalf_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int32_t>
{
    using type                           = int32_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int8_t>
{
    using type                           = int8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<uint8_t>
{
    using type                           = uint8_t;
    static constexpr index_t vector_size = 1;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct scalar_type<int4_t>
{
    using type                           = int4_t;
    static constexpr index_t vector_size = 1;
};
#endif

template <>
struct scalar_type<pk_i4_t>
{
    using type                           = pk_i4_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f8_fnuz_t>
{
    using type                           = f8_fnuz_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_fnuz_t>
{
    using type                           = bf8_fnuz_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f8_ocp_t>
{
    using type                           = f8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_ocp_t>
{
    using type                           = bf8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bool>
{
    using type                           = bool;
    static constexpr index_t vector_size = 1;
};

template <typename T>
struct vector_type<T, 1, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    using type = d1_t;

    union
    {
        T d1_;
        StaticallyIndexedArray<T, 1> d1x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }
};

__device__ int static err = 0;
template <typename T>
struct vector_type<T, 2, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));

    using type = d2_t;

    union
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 3, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));

    using type = d3_t;

    union
    {
        d3_t d3_;
        StaticallyIndexedArray<d1_t, 3> d1x3_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
        StaticallyIndexedArray<d3_t, 1> d3x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 4, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));

    using type = d4_t;

    union
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 5, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d5_t __attribute__((ext_vector_type(5)));

    using type = d5_t;

    union
    {
        d5_t d5_;
        StaticallyIndexedArray<d1_t, 5> d1x5_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
        StaticallyIndexedArray<d5_t, 1> d5x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value || is_same<X, d5_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value || is_same<X, d5_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 7, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d7_t __attribute__((ext_vector_type(7)));

    using type = d7_t;

    union
    {
        d7_t d7_;
        StaticallyIndexedArray<d1_t, 7> d1x7_;
        StaticallyIndexedArray<d2_t, 3> d2x3_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
        StaticallyIndexedArray<d7_t, 1> d7x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d7_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x7_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d7_t>::value)
        {
            return data_.d7x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d7_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x7_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d7_t>::value)
        {
            return data_.d7x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));

    using type = d8_t;

    union
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 13, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d13_t __attribute__((ext_vector_type(13)));

    using type = d13_t;

    union
    {
        d13_t d13_;
        StaticallyIndexedArray<d1_t, 13> d1x13_;
        StaticallyIndexedArray<d4_t, 3> d4x3_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
        StaticallyIndexedArray<d13_t, 1> d13x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d13_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x13_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d13_t>::value)
        {
            return data_.d13x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d13_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x13_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d13_t>::value)
        {
            return data_.d13x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));

    using type = d16_t;

    union
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 32, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));

    using type = d32_t;

    union
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 64, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));

    using type = d64_t;

    union
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 128, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));

    using type = d128_t;

    union
    {
        d128_t d128_;
        StaticallyIndexedArray<d1_t, 128> d1x128_;
        StaticallyIndexedArray<d2_t, 64> d2x64_;
        StaticallyIndexedArray<d4_t, 32> d4x32_;
        StaticallyIndexedArray<d8_t, 16> d8x16_;
        StaticallyIndexedArray<d16_t, 8> d16x8_;
        StaticallyIndexedArray<d32_t, 4> d32x4_;
        StaticallyIndexedArray<d64_t, 2> d64x2_;
        StaticallyIndexedArray<d128_t, 1> d128x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 256, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));
    typedef T d256_t __attribute__((ext_vector_type(256)));

    using type = d256_t;

    union
    {
        d256_t d256_;
        StaticallyIndexedArray<d1_t, 256> d1x256_;
        StaticallyIndexedArray<d2_t, 128> d2x128_;
        StaticallyIndexedArray<d4_t, 64> d4x64_;
        StaticallyIndexedArray<d8_t, 32> d8x32_;
        StaticallyIndexedArray<d16_t, 16> d16x16_;
        StaticallyIndexedArray<d32_t, 8> d32x8_;
        StaticallyIndexedArray<d64_t, 4> d64x4_;
        StaticallyIndexedArray<d128_t, 2> d128x2_;
        StaticallyIndexedArray<d256_t, 1> d256x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T, index_t N, typename Enable = void>
struct non_native_vector_base;

template <typename T>
struct nnvb_data_t_selector
{
    using type = unsigned _BitInt(8 * sizeof(T));
};

template <>
struct nnvb_data_t_selector<f8_ocp_t>
{
    using type = f8_ocp_t::data_type;
};
template <>
struct nnvb_data_t_selector<bf8_ocp_t>
{
    using type = bf8_ocp_t::data_type;
};

template <>
struct nnvb_data_t_selector<pk_i4_t>
{
    using type = pk_i4_t::type;
};

template <typename T, index_t N>
struct non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>>
{
    using data_t = typename nnvb_data_t_selector<T>::type; // select data_t based on the size of T
    static_assert(sizeof(T) == sizeof(data_t), "non_native_vector_base storage size mismatch");
    using data_v = data_t __attribute__((ext_vector_type(N)));
    using type   = non_native_vector_base<T, N>;

    union alignas(next_pow2(N * sizeof(T)))
    {
        data_v dN; // storage vector;
        StaticallyIndexedArray<data_t, N> dxN;
        StaticallyIndexedArray<T, N> dTxN;
        StaticallyIndexedArray<data_v, 1> dNx1;
    } data_;

    __host__ __device__ constexpr non_native_vector_base(data_t a) : data_{data_v(a)} {}
    __host__ __device__ constexpr non_native_vector_base(T f)
        : non_native_vector_base(bit_cast<data_t>(f))
    {
    }
    __host__ __device__ constexpr non_native_vector_base() : non_native_vector_base(T{}){};
    __host__ __device__ constexpr non_native_vector_base(data_v v) : data_{v} {}

    __host__ __device__ constexpr operator data_v() const { return data_.dN; }
    __host__ __device__ constexpr operator data_t() const
    {
        if constexpr(N == 1)
        {
            return data_.dxN[Number<0>{}];
        }
        else
        {
            return data_.dxN; // XXX this should cause an error
        }
    }
    __host__ __device__ constexpr operator T() const
    {
        if constexpr(N == 1)
        {
            return data_.dTxN[Number<0>{}];
        }
        else
        {
            return data_.dTxN; // XXX this should cause an error
        }
    }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }
};

template <typename T, index_t N>
struct scalar_type<non_native_vector_base<T, N>>;

template <index_t N>
struct scalar_type<non_native_vector_base<f8_ocp_t, N>>
{
    using type = typename non_native_vector_base<f8_ocp_t, N>::data_t;

    static constexpr index_t vector_size = N;
};

template <index_t N>
struct scalar_type<non_native_vector_base<bf8_ocp_t, N>>
{
    using type = typename non_native_vector_base<bf8_ocp_t, N>::data_t;

    static constexpr index_t vector_size = N;
};

template <index_t N>
struct scalar_type<non_native_vector_base<pk_i4_t, N>>
{
    using type = typename non_native_vector_base<pk_i4_t, N>::data_t;

    static constexpr index_t vector_size = N;
};

// non-native vector_type implementation
template <typename T>
struct vector_type<T, 1, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using type     = d1_nnv_t;

    union alignas(next_pow2(1 * sizeof(T)))
    {
        d1_t d1_;
        StaticallyIndexedArray<d1_t, 1> d1x1_;
        d1_nnv_t d1_nnv_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{d1_t{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 2, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;

    using type = d2_t;

    union alignas(next_pow2(2 * sizeof(T)))
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 4, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;
    using d4_t     = non_native_vector_base<T, 4>;

    using type = d4_t;

    union alignas(next_pow2(4 * sizeof(T)))
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;
    using d4_t     = non_native_vector_base<T, 4>;
    using d8_t     = non_native_vector_base<T, 8>;

    using type = d8_t;

    union alignas(next_pow2(8 * sizeof(T)))
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;
    using d4_t     = non_native_vector_base<T, 4>;
    using d8_t     = non_native_vector_base<T, 8>;
    using d16_t    = non_native_vector_base<T, 16>;

    using type = d16_t;

    union alignas(next_pow2(16 * sizeof(T)))
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 32, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;
    using d32_t = non_native_vector_base<T, 32>;

    using type = d32_t;

    union alignas(next_pow2(32 * sizeof(T)))
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 64, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;
    using d32_t = non_native_vector_base<T, 32>;
    using d64_t = non_native_vector_base<T, 64>;

    using type = d64_t;

    union alignas(next_pow2(64 * sizeof(T)))
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }
};

using int64_t = long;

// fp64
using double2_t = typename vector_type<double, 2>::type;
using double4_t = typename vector_type<double, 4>::type;

// fp32
using float2_t  = typename vector_type<float, 2>::type;
using float4_t  = typename vector_type<float, 4>::type;
using float8_t  = typename vector_type<float, 8>::type;
using float16_t = typename vector_type<float, 16>::type;
using float32_t = typename vector_type<float, 32>::type;
using float64_t = typename vector_type<float, 64>::type;

// fp16
using half2_t  = typename vector_type<half_t, 2>::type;
using half4_t  = typename vector_type<half_t, 4>::type;
using half8_t  = typename vector_type<half_t, 8>::type;
using half16_t = typename vector_type<half_t, 16>::type;
using half32_t = typename vector_type<half_t, 32>::type;
using half64_t = typename vector_type<half_t, 64>::type;

// bfp16
using bhalf2_t  = typename vector_type<bhalf_t, 2>::type;
using bhalf4_t  = typename vector_type<bhalf_t, 4>::type;
using bhalf8_t  = typename vector_type<bhalf_t, 8>::type;
using bhalf16_t = typename vector_type<bhalf_t, 16>::type;
using bhalf32_t = typename vector_type<bhalf_t, 32>::type;
using bhalf64_t = typename vector_type<bhalf_t, 64>::type;

// i32
using int32x2_t  = typename vector_type<int32_t, 2>::type;
using int32x4_t  = typename vector_type<int32_t, 4>::type;
using int32x8_t  = typename vector_type<int32_t, 8>::type;
using int32x16_t = typename vector_type<int32_t, 16>::type;
using int32x32_t = typename vector_type<int32_t, 32>::type;
using int32x64_t = typename vector_type<int32_t, 64>::type;

// i8
using int8x2_t  = typename vector_type<int8_t, 2>::type;
using int8x4_t  = typename vector_type<int8_t, 4>::type;
using int8x8_t  = typename vector_type<int8_t, 8>::type;
using int8x16_t = typename vector_type<int8_t, 16>::type;
using int8x32_t = typename vector_type<int8_t, 32>::type;
using int8x64_t = typename vector_type<int8_t, 64>::type;

// f8
using f8x2_fnuz_t  = typename vector_type<f8_fnuz_t, 2>::type;
using f8x4_fnuz_t  = typename vector_type<f8_fnuz_t, 4>::type;
using f8x8_fnuz_t  = typename vector_type<f8_fnuz_t, 8>::type;
using f8x16_fnuz_t = typename vector_type<f8_fnuz_t, 16>::type;
using f8x32_fnuz_t = typename vector_type<f8_fnuz_t, 32>::type;
using f8x64_fnuz_t = typename vector_type<f8_fnuz_t, 64>::type;

// bf8
using bf8x2_fnuz_t  = typename vector_type<bf8_fnuz_t, 2>::type;
using bf8x4_fnuz_t  = typename vector_type<bf8_fnuz_t, 4>::type;
using bf8x8_fnuz_t  = typename vector_type<bf8_fnuz_t, 8>::type;
using bf8x16_fnuz_t = typename vector_type<bf8_fnuz_t, 16>::type;
using bf8x32_fnuz_t = typename vector_type<bf8_fnuz_t, 32>::type;
using bf8x64_fnuz_t = typename vector_type<bf8_fnuz_t, 64>::type;

// f8
using f8x2_ocp_t  = typename vector_type<f8_ocp_t, 2>::type;
using f8x4_ocp_t  = typename vector_type<f8_ocp_t, 4>::type;
using f8x8_ocp_t  = typename vector_type<f8_ocp_t, 8>::type;
using f8x16_ocp_t = typename vector_type<f8_ocp_t, 16>::type;
using f8x32_ocp_t = typename vector_type<f8_ocp_t, 32>::type;
using f8x64_ocp_t = typename vector_type<f8_ocp_t, 64>::type;

// bf8
using bf8x2_ocp_t  = typename vector_type<bf8_ocp_t, 2>::type;
using bf8x4_ocp_t  = typename vector_type<bf8_ocp_t, 4>::type;
using bf8x8_ocp_t  = typename vector_type<bf8_ocp_t, 8>::type;
using bf8x16_ocp_t = typename vector_type<bf8_ocp_t, 16>::type;
using bf8x32_ocp_t = typename vector_type<bf8_ocp_t, 32>::type;
using bf8x64_ocp_t = typename vector_type<bf8_ocp_t, 64>::type;

#if CK_FP8_TYPE_OCP
// f8
using f8x2_t  = f8x2_ocp_t;
using f8x4_t  = f8x4_ocp_t;
using f8x8_t  = f8x8_ocp_t;
using f8x16_t = f8x16_ocp_t;
using f8x32_t = f8x32_ocp_t;
using f8x64_t = f8x64_ocp_t;

// bf8
using bf8x2_t  = bf8x2_ocp_t;
using bf8x4_t  = bf8x4_ocp_t;
using bf8x8_t  = bf8x8_ocp_t;
using bf8x16_t = bf8x16_ocp_t;
using bf8x32_t = bf8x32_ocp_t;
using bf8x64_t = bf8x64_ocp_t;
#elif CK_FP8_TYPE_FNUZ
// f8
using f8x2_t  = f8x2_fnuz_t;
using f8x4_t  = f8x4_fnuz_t;
using f8x8_t  = f8x8_fnuz_t;
using f8x16_t = f8x16_fnuz_t;
using f8x32_t = f8x32_fnuz_t;
using f8x64_t = f8x64_fnuz_t;

// bf8
using bf8x2_t  = bf8x2_fnuz_t;
using bf8x4_t  = bf8x4_fnuz_t;
using bf8x8_t  = bf8x8_fnuz_t;
using bf8x16_t = bf8x16_fnuz_t;
using bf8x32_t = bf8x32_fnuz_t;
using bf8x64_t = bf8x64_fnuz_t;
#endif

// u8
using uint8x2_t  = typename vector_type<uint8_t, 2>::type;
using uint8x4_t  = typename vector_type<uint8_t, 4>::type;
using uint8x8_t  = typename vector_type<uint8_t, 8>::type;
using uint8x16_t = typename vector_type<uint8_t, 16>::type;
using uint8x32_t = typename vector_type<uint8_t, 32>::type;
using uint8x64_t = typename vector_type<uint8_t, 64>::type;

// pack int4
using pk_i4x2_t = typename vector_type<pk_i4_t, 2>::type;
using pk_i4x4_t = typename vector_type<pk_i4_t, 4>::type;
using pk_i4x8_t = typename vector_type<pk_i4_t, 8>::type;

#ifdef CK_CODE_GEN_RTC
template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<int32_t>
{
    __host__ __device__ static constexpr int32_t Lowest() noexcept { return -2147483647 - 1; }

    __host__ __device__ static constexpr int32_t Min() noexcept { return -2147483647 - 1; }

    __host__ __device__ static constexpr int32_t Max() noexcept { return 2147483647; }

    __host__ __device__ static constexpr int32_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr int32_t QuietNaN() { return 0; }
};
template <>
struct NumericLimits<int16_t>
{
    __host__ __device__ static constexpr int16_t Lowest() noexcept { return -32768; }

    __host__ __device__ static constexpr int16_t Min() noexcept { return -32768; }

    __host__ __device__ static constexpr int16_t Max() noexcept { return 32767; }

    __host__ __device__ static constexpr int16_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr int16_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<int8_t>
{
    __host__ __device__ static constexpr int8_t Lowest() noexcept { return -128; }

    __host__ __device__ static constexpr int8_t Min() noexcept { return -128; }

    __host__ __device__ static constexpr int8_t Max() noexcept { return 127; }

    __host__ __device__ static constexpr int8_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr int8_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<uint32_t>
{
    __host__ __device__ static constexpr uint32_t Lowest() noexcept { return 0; }

    __host__ __device__ static constexpr uint32_t Min() noexcept { return 0; }

    __host__ __device__ static constexpr uint32_t Max() noexcept { return 4294967295U; }

    __host__ __device__ static constexpr uint32_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr uint32_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<uint16_t>
{
    __host__ __device__ static constexpr uint16_t Lowest() noexcept { return 0; }

    __host__ __device__ static constexpr uint16_t Min() noexcept { return 0; }

    __host__ __device__ static constexpr uint16_t Max() noexcept { return 65535U; }

    __host__ __device__ static constexpr uint16_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr uint16_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<float>
{
    static constexpr unsigned int binary_min    = 0x00800000;
    static constexpr unsigned int binary_max    = 0x7F7FFFFF;
    static constexpr unsigned int binary_lowest = 0xFF7FFFFF;
    static constexpr unsigned int binary_qnan   = 0xFFC00001;
    static constexpr unsigned int binary_inf    = 0x7F8000000;

    __host__ __device__ static constexpr float Min() { return bit_cast<float>(binary_min); }

    __host__ __device__ static constexpr float Max() { return bit_cast<float>(binary_max); }

    __host__ __device__ static constexpr float Lowest() { return bit_cast<float>(binary_lowest); }

    __host__ __device__ static constexpr float QuietNaN() { return bit_cast<float>(binary_qnan); }

    __host__ __device__ static constexpr float Infinity() { return bit_cast<float>(binary_inf); }
};

template <>
struct NumericLimits<half_t>
{
    static constexpr unsigned short binary_min    = 0x0400;
    static constexpr unsigned short binary_max    = 0x7BFF;
    static constexpr unsigned short binary_lowest = 0xFBFF;
    static constexpr unsigned short binary_qnan   = 0x7FFF;

    __host__ __device__ static constexpr half_t Min() { return bit_cast<half_t>(binary_min); }

    __host__ __device__ static constexpr half_t Max() { return bit_cast<half_t>(binary_max); }

    __host__ __device__ static constexpr half_t Lowest() { return bit_cast<half_t>(binary_lowest); }

    __host__ __device__ static constexpr half_t QuietNaN() { return bit_cast<half_t>(binary_qnan); }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct NumericLimits<int4_t>
{
    __host__ __device__ static constexpr int4_t Min() { return int4_t(-8); }

    __host__ __device__ static constexpr int4_t Max() { return int4_t(7); }

    __host__ __device__ static constexpr int4_t Lowest() { return int4_t(-8); }
};
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4

template <>
struct NumericLimits<f8_fnuz_t>
{
    // negative zero nan mode with exp bias = 8
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 7
    // static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    // static constexpr uint8_t binary_max    = 0x77; // 0b01110111
    // static constexpr uint8_t binary_lowest = 0xF7; // 0b11110111
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=0

    __host__ __device__ static constexpr f8_fnuz_t Min() { return f8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr f8_fnuz_t Max() { return f8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr f8_fnuz_t Lowest() { return f8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr f8_fnuz_t QuietNaN() { return f8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<bf8_fnuz_t>
{
    // negative zero nan mode with exp bias = 16
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 15
    // static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    // static constexpr uint8_t binary_max    = 0x7B; // 0b01111011
    // static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=

    __host__ __device__ static constexpr bf8_fnuz_t Min() { return bf8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr bf8_fnuz_t Max() { return bf8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr bf8_fnuz_t Lowest() { return bf8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr bf8_fnuz_t QuietNaN() { return bf8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<f8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000 = 2^-6
    static constexpr uint8_t binary_max    = 0x7E; // 0b01111110 = 448
    static constexpr uint8_t binary_lowest = 0xFE; // 0b11111110 = -448
    static constexpr uint8_t binary_qnan   = 0x7F; // 0b01111111

    __host__ __device__ static constexpr f8_ocp_t Min() { return bit_cast<f8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr f8_ocp_t Max() { return bit_cast<f8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr f8_ocp_t Lowest()
    {
        return bit_cast<f8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr f8_ocp_t QuietNaN()
    {
        return bit_cast<f8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<bf8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100 = 2^-14
    static constexpr uint8_t binary_max    = 0x7B; // 0b01111011 = 57344
    static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011 = -57344
    static constexpr uint8_t binary_qnan   = 0x7D; // 0b01111101

    __host__ __device__ static constexpr bf8_ocp_t Min() { return bit_cast<bf8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr bf8_ocp_t Max() { return bit_cast<bf8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr bf8_ocp_t Lowest()
    {
        return bit_cast<bf8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr bf8_ocp_t QuietNaN()
    {
        return bit_cast<bf8_ocp_t>(binary_qnan);
    }
};
#else
template <typename T>
struct NumericLimits
{
    __host__ __device__ static constexpr T Min() { return std::numeric_limits<T>::min(); }
    __host__ __device__ static constexpr T Max() { return std::numeric_limits<T>::max(); }
    __host__ __device__ static constexpr T Lowest() { return std::numeric_limits<T>::lowest(); }
    __host__ __device__ static constexpr T QuietNaN()
    {
        return std::numeric_limits<T>::quiet_NaN();
    }
    __host__ __device__ static constexpr T Infinity() { return std::numeric_limits<T>::infinity(); }
};

template <>
struct NumericLimits<half_t>
{
    static constexpr unsigned short binary_min    = 0x0400;
    static constexpr unsigned short binary_max    = 0x7BFF;
    static constexpr unsigned short binary_lowest = 0xFBFF;
    static constexpr unsigned short binary_qnan   = 0x7FFF;

    __host__ __device__ static constexpr half_t Min() { return bit_cast<half_t>(binary_min); }

    __host__ __device__ static constexpr half_t Max() { return bit_cast<half_t>(binary_max); }

    __host__ __device__ static constexpr half_t Lowest() { return bit_cast<half_t>(binary_lowest); }

    __host__ __device__ static constexpr half_t QuietNaN() { return bit_cast<half_t>(binary_qnan); }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct NumericLimits<int4_t>
{
    __host__ __device__ static constexpr int4_t Min() { return int4_t(-8); }

    __host__ __device__ static constexpr int4_t Max() { return int4_t(7); }

    __host__ __device__ static constexpr int4_t Lowest() { return int4_t(-8); }
};
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4

template <>
struct NumericLimits<f8_fnuz_t>
{
    // negative zero nan mode with exp bias = 8
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 7
    // static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    // static constexpr uint8_t binary_max    = 0x77; // 0b01110111
    // static constexpr uint8_t binary_lowest = 0xF7; // 0b11110111
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=0

    __host__ __device__ static constexpr f8_fnuz_t Min() { return f8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr f8_fnuz_t Max() { return f8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr f8_fnuz_t Lowest() { return f8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr f8_fnuz_t QuietNaN() { return f8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<bf8_fnuz_t>
{
    // negative zero nan mode with exp bias = 16
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 15
    // static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    // static constexpr uint8_t binary_max    = 0x7B; // 0b01111011
    // static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=

    __host__ __device__ static constexpr bf8_fnuz_t Min() { return bf8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr bf8_fnuz_t Max() { return bf8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr bf8_fnuz_t Lowest() { return bf8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr bf8_fnuz_t QuietNaN() { return bf8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<f8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000 = 2^-6
    static constexpr uint8_t binary_max    = 0x7E; // 0b01111110 = 448
    static constexpr uint8_t binary_lowest = 0xFE; // 0b11111110 = -448
    static constexpr uint8_t binary_qnan   = 0x7F; // 0b01111111

    __host__ __device__ static constexpr f8_ocp_t Min() { return bit_cast<f8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr f8_ocp_t Max() { return bit_cast<f8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr f8_ocp_t Lowest()
    {
        return bit_cast<f8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr f8_ocp_t QuietNaN()
    {
        return bit_cast<f8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<bf8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100 = 2^-14
    static constexpr uint8_t binary_max    = 0x7B; // 0b01111011 = 57344
    static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011 = -57344
    static constexpr uint8_t binary_qnan   = 0x7D; // 0b01111101

    __host__ __device__ static constexpr bf8_ocp_t Min() { return bit_cast<bf8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr bf8_ocp_t Max() { return bit_cast<bf8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr bf8_ocp_t Lowest()
    {
        return bit_cast<bf8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr bf8_ocp_t QuietNaN()
    {
        return bit_cast<bf8_ocp_t>(binary_qnan);
    }
};
#endif

template <typename T>
struct NumericUtils
{
};

template <>
struct NumericUtils<float>
{
    static constexpr int exp            = 8;
    static constexpr int mant           = 23;
    static constexpr int bias           = 127;
    static constexpr uint32_t nan_mask  = 0x7F800000;
    static constexpr uint32_t head_mask = 0xFF800000;
    static constexpr uint32_t mant_mask = 0x7FFFFF;
    static constexpr uint32_t exp_mask  = 0xFF;
    static constexpr uint32_t Inf       = 0x7F800000;
    static constexpr uint32_t NegInf    = 0xFF800000;
    static constexpr uint32_t NaN       = 0x7F800001;
    static constexpr uint32_t Neg0      = 0x80000000;
    using bitwise_type                  = uint32_t;
};

template <>
struct NumericUtils<half_t>
{
    static constexpr int exp            = 5;
    static constexpr int mant           = 10;
    static constexpr int bias           = 15;
    static constexpr uint16_t nan_mask  = 0x7C00;
    static constexpr uint16_t head_mask = 0xFC00;
    static constexpr uint16_t mant_mask = 0x3FF;
    static constexpr uint16_t exp_mask  = 0x1F;
    static constexpr uint32_t Inf       = 0x7C00;
    static constexpr uint32_t NegInf    = 0xFC00;
    static constexpr uint32_t NaN       = 0x7C01;
    static constexpr uint32_t Neg0      = 0x8000;
    using bitwise_type                  = uint16_t;
};

template <>
struct NumericUtils<f8_fnuz_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    static constexpr int bias = 8; // negative zero nan mode
    // static constexpr int bias = 7; // ieee mode
};

template <>
struct NumericUtils<bf8_fnuz_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    static constexpr int bias = 16; // negative zero nan mode
    // static constexpr int bias = 15; // ieee mode
};
template <>
struct NumericUtils<f8_ocp_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    static constexpr int bias = 7;
};

template <>
struct NumericUtils<bf8_ocp_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    static constexpr int bias = 15;
};

template <>
struct NumericUtils<bhalf_t>
{
    static constexpr int exp  = 8;
    static constexpr int mant = 7;
    static constexpr int bias = 128; // negative zero nan mode
    // static constexpr int bias = 127; // ieee mode
};
} // namespace ck
