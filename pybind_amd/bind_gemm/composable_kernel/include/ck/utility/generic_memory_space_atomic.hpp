// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "data_type.hpp"

namespace ck {

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_add explicit for
// each datatype.
template <typename X>
__device__ X atomic_add(X* p_dst, const X& x);

template <>
__device__ int32_t atomic_add<int32_t>(int32_t* p_dst, const int32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ uint32_t atomic_add<uint32_t>(uint32_t* p_dst, const uint32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float atomic_add<float>(float* p_dst, const float& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ double atomic_add<double>(double* p_dst, const double& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float2_t atomic_add<float2_t>(float2_t* p_dst, const float2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<float, 2> vx{x};
    vector_type<float, 2> vy{0};

    vy.template AsType<float>()(I0) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);

    return vy.template AsType<float2_t>()[I0];
}

template <>
__device__ double2_t atomic_add<double2_t>(double2_t* p_dst, const double2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<double, 2> vx{x};
    vector_type<double, 2> vy{0};

    vy.template AsType<double>()(I0) =
        atomicAdd(c_style_pointer_cast<double*>(p_dst), vx.template AsType<double>()[I0]);
    vy.template AsType<double>()(I1) =
        atomicAdd(c_style_pointer_cast<double*>(p_dst) + 1, vx.template AsType<double>()[I1]);

    return vy.template AsType<double2_t>()[I0];
}

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_max explicit for
// each datatype.

template <typename X>
__device__ X atomic_max(X* p_dst, const X& x);

template <>
__device__ int32_t atomic_max<int32_t>(int32_t* p_dst, const int32_t& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ uint32_t atomic_max<uint32_t>(uint32_t* p_dst, const uint32_t& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ float atomic_max<float>(float* p_dst, const float& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ double atomic_max<double>(double* p_dst, const double& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ float2_t atomic_max<float2_t>(float2_t* p_dst, const float2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<float, 2> vx{x};
    vector_type<float, 2> vy{0};

    vy.template AsType<float>()(I0) =
        atomicMax(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicMax(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);

    return vy.template AsType<float2_t>()[I0];
}

} // namespace ck
