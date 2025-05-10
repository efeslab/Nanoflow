// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/quantization_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

// Need to ensure compiler will fail if there is no matching candidate, instead of compiler
// siliently do implicit type conversion
//
// Example:
//
// struct ExampleElementwiseOp
// {
//     template<typename Y, typename X>
//     __host__ __device__ constexpr void
//     operator()(Y&, const X) const;
//
//     template<>
//     __host__ __device__ constexpr void
//     operator()<half_t, half_t>(half_t& y, const half_t& x) const
//     {
//     }
// };

struct AddReluAdd
{
    template <typename Y, typename X0, typename X1, typename X2>
    __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&, const X2&) const;

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& y, const half_t& x0, const half_t& x1, const half_t& x2) const
    {
        half_t a = x0 + x1;
        half_t b = a > 0 ? a : 0;
        y        = b + x2;
    }

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& y,
                                                                              const float& x0,
                                                                              const float& x1,
                                                                              const float& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, float, half_t, half_t>(
        half_t& y, const float& x0, const half_t& x1, const half_t& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }

    template <>
    __host__ __device__ constexpr void operator()<bhalf_t, float, bhalf_t, bhalf_t>(
        bhalf_t& y, const float& x0, const bhalf_t& x1, const bhalf_t& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }

    template <>
    __host__ __device__ constexpr void operator()<int8_t, int8_t, int8_t, int8_t>(
        int8_t& y, const int8_t& x0, const int8_t& x1, const int8_t& x2) const
    {
        int32_t a = x0 + x1;
        int32_t b = a > 0 ? a : 0;
        int32_t c = b + x2;
        y         = c;
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    __host__ __device__ constexpr void operator()<int4_t, int8_t, int4_t, int4_t>(
        int4_t& y, const int8_t& x0, const int4_t& x1, const int4_t& x2) const
    {
        int32_t a = x0 + x1;
        int32_t b = a > 0 ? a : 0;
        int32_t c = b + x2;
        y         = c;
    }
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
};

struct AddHardswishAdd
{
    template <typename Y, typename X0, typename X1, typename X2>
    __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&, const X2&) const;

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& y,
                                                                              const float& x0,
                                                                              const float& x1,
                                                                              const float& x2) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& y, const half_t& x0, const half_t& x1, const half_t& x2) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
    }
};

// C = A * B
// E = C + D0 + D1
struct AddAdd
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ void operator()(E& e, const C& c, const D0& d0, const D1& d1) const
    {
        // Only support floating so far
        static_assert(is_same<E, half_t>::value || is_same<E, float>::value ||
                          is_same<E, double>::value,
                      "Data type is not supported by this operation!");

        static_assert(is_same<C, half_t>::value || is_same<C, float>::value ||
                          is_same<C, double>::value,
                      "Data type is not supported by this operation!");

        static_assert(is_same<D0, half_t>::value || is_same<D0, float>::value ||
                          is_same<D0, double>::value,
                      "Data type is not supported by this operation!");

        static_assert(is_same<D1, half_t>::value || is_same<D1, float>::value ||
                          is_same<D1, double>::value,
                      "Data type is not supported by this operation!");

        const C y = c + type_convert<C>(d0) + type_convert<C>(d1);
        e         = type_convert<E>(y);
    }
};

// C = A * B
// E = (C + D0) x D1
struct AddMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ void operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ void operator()<half_t, half_t, half_t, half_t>(half_t& e,
                                                                        const half_t& c,
                                                                        const half_t& d0,
                                                                        const half_t& d1) const
    {
        const half_t y = (c + d0) * d1;
        e              = y;
    }
    template <>
    __host__ __device__ void operator()<half_t, float, half_t, half_t>(half_t& e,
                                                                       const float& c,
                                                                       const half_t& d0,
                                                                       const half_t& d1) const
    {
        const half_t y = (type_convert<half_t>(c) + d0) * d1;
        e              = y;
    }
    template <>
    __host__ __device__ void operator()<float, float, half_t, half_t>(float& e,
                                                                      const float& c,
                                                                      const half_t& d0,
                                                                      const half_t& d1) const
    {
        const float y = (c + d0) * d1;
        e             = y;
    }
};

// C = A * B
// E = C x D0 + D1
struct MultiplyAdd
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ void operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ void operator()<half_t, half_t, half_t, half_t>(half_t& e,
                                                                        const half_t& c,
                                                                        const half_t& d0,
                                                                        const half_t& d1) const
    {
        const half_t y = (c * d0) + d1;
        e              = y;
    }
    template <>
    __host__ __device__ void operator()<half_t, float, half_t, half_t>(half_t& e,
                                                                       const float& c,
                                                                       const half_t& d0,
                                                                       const half_t& d1) const
    {
        const half_t y = type_convert<half_t>(c) * d0 + d1;
        e              = y;
    }
    template <>
    __host__ __device__ void operator()<bhalf_t, float, bhalf_t, bhalf_t>(bhalf_t& e,
                                                                          const float& c,
                                                                          const bhalf_t& d0,
                                                                          const bhalf_t& d1) const
    {
        const bhalf_t y = type_convert<bhalf_t>(c) * d0 + d1;
        e               = y;
    }
    template <>
    __host__ __device__ void operator()<float, float, half_t, half_t>(float& e,
                                                                      const float& c,
                                                                      const half_t& d0,
                                                                      const half_t& d1) const
    {
        const float y = c * d0 + d1;
        e             = y;
    }
    template <>
    __host__ __device__ void operator()<half_t, float, float, float>(half_t& e,
                                                                     const float& c,
                                                                     const float& d0,
                                                                     const float& d1) const
    {
        const float y = c * d0 + d1;
        e             = y;
    }
};

struct MultiplyAddFastGelu
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, float, ck::bhalf_t, ck::bhalf_t>(
        ck::bhalf_t& e, const float& c, const ck::bhalf_t& d0, const ck::bhalf_t& d1) const
    {
        const float x0_f = c * ck::type_convert<float>(d0) + ck::type_convert<float>(d1);

        float x1_f = 0;

        FastGelu{}.template operator()<float, float>(x1_f, x0_f);

        e = ck::type_convert<ck::bhalf_t>(x1_f);
    }
};

// E = FastGelu(C + D0 + D1)
struct AddAddFastGelu
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& e,
                                                                              const float& c,
                                                                              const float& d0,
                                                                              const float& d1) const
    {
        const float x = c + d0 + d1;

        FastGelu{}.template operator()<float, float>(e, x);
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& e, const half_t& c, const half_t& d0, const half_t& d1) const
    {
        const half_t x = c + d0 + d1;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<half_t, half_t>(e, x);
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, float, half_t, half_t>(
        half_t& e, const float& c, const half_t& d0, const half_t& d1) const
    {
        const float x0_f = c + d0 + d1;

        float x1_f = 0;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(x1_f,
                                                                                         x0_f);

        e = type_convert<half_t>(x1_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<bhalf_t, float, bhalf_t, bhalf_t>(
        bhalf_t& e, const float& c, const bhalf_t& d0, const bhalf_t& d1) const
    {
        const float x0_f = c + type_convert<float>(d0) + type_convert<float>(d1);

        float x1_f = 0;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(x1_f,
                                                                                         x0_f);

        e = type_convert<bhalf_t>(x1_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<int8_t, int32_t, int8_t, int8_t>(
        int8_t& e, const int32_t& c, const int8_t& d0, const int8_t& d1) const
    {
        const float x0_f =
            type_convert<float>(c) + type_convert<float>(d0) + type_convert<float>(d1);

        float x1_f = 0;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(x1_f,
                                                                                         x0_f);

        e = type_convert<int8_t>(x1_f);
    }
};

// E = Relu(alpha1 * C + alpha2 * D0 + D1)
struct ScaleAddScaleAddRelu
{

    ScaleAddScaleAddRelu(const float alpha1 = 1.f, const float alpha2 = 1.f)
        : alpha1_(alpha1), alpha2_(alpha2)
    {
    }

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& e,
                                                                              const float& c,
                                                                              const float& d0,
                                                                              const float& d1) const
    {
        const float x = c * alpha1_ + alpha2_ * d0 + d1;
        Relu{}.template operator()<float>(e, x);
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& e, const half_t& c, const half_t& d0, const half_t& d1) const
    {
        const float x = type_convert<float>(c) * alpha1_ + alpha2_ * type_convert<float>(d0) +
                        type_convert<float>(d1);

        float result = 0;
        Relu{}.template operator()<float>(result, x);

        e = type_convert<half_t>(result);
    }

    template <>
    __host__ __device__ constexpr void operator()<bhalf_t, bhalf_t, bhalf_t, bhalf_t>(
        bhalf_t& e, const bhalf_t& c, const bhalf_t& d0, const bhalf_t& d1) const
    {
        const float x = type_convert<float>(c) * alpha1_ + alpha2_ * type_convert<float>(d0) +
                        type_convert<float>(d1);

        float result = 0;
        Relu{}.template operator()<float>(result, x);

        e = type_convert<bhalf_t>(result);
    }

    template <>
    __host__ __device__ constexpr void operator()<int8_t, int8_t, float, float>(
        int8_t& e, const int8_t& c, const float& d0, const float& d1) const
    {
        const float x = type_convert<float>(c) * alpha1_ + alpha2_ * d0 + d1;

        float result = 0;
        Relu{}.template operator()<float>(result, x);

        e = type_convert<int8_t>(result);
    }

    const float alpha1_;
    const float alpha2_;
};

struct Normalize
{
    // FIXME: is double absolutely necessary?
    Normalize(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T1, typename T2, typename T3>
    __host__ __device__ constexpr void operator()(T1& y,
                                                  const T1& x,
                                                  const T2& mean,
                                                  const T2& mean_square,
                                                  const T3& gamma,
                                                  const T3& beta) const;

    template <>
    __host__ __device__ constexpr void operator()<half_t, float, half_t>(half_t& y,
                                                                         const half_t& x,
                                                                         const float& mean,
                                                                         const float& mean_square,
                                                                         const half_t& gamma,
                                                                         const half_t& beta) const
    {
        using ck::math::sqrt;

        float variance = mean_square - (mean * mean);

        float tmp_x     = type_convert<float>(x);
        float tmp_gamma = type_convert<float>(gamma);
        float tmp_beta  = type_convert<float>(beta);

        float tmp_y =
            ((tmp_x - mean) / sqrt(variance + type_convert<float>(epsilon_))) * tmp_gamma +
            tmp_beta;

        y = type_convert<half_t>(tmp_y);
    };

    template <>
    __host__ __device__ constexpr void operator()<float, float, float>(float& y,
                                                                       const float& x,
                                                                       const float& mean,
                                                                       const float& mean_square,
                                                                       const float& gamma,
                                                                       const float& beta) const
    {
        using ck::math::sqrt;

        float variance = mean_square - (mean * mean);
        y = ((x - mean) / sqrt(variance + type_convert<float>(epsilon_))) * gamma + beta;
    };

    template <>
    __host__ __device__ constexpr void operator()<double, double, double>(double& y,
                                                                          const double& x,
                                                                          const double& mean,
                                                                          const double& mean_square,
                                                                          const double& gamma,
                                                                          const double& beta) const
    {
        using ck::math::sqrt;

        double variance = mean_square - (mean * mean);
        y               = ((x - mean) / sqrt(variance + epsilon_)) * gamma + beta;
    };

    // FIXME: is double absolutely necessary?
    double epsilon_;
};

// used by BatchNorm inference
// y = gamma * (x-mean) / sqrt(epsilon+variance) + beta
// The data type of mean and variance is used as AccDataType
struct NormalizeInInfer
{
    NormalizeInInfer(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T1, typename T2, typename T3, typename T4>
    __host__ __device__ constexpr void operator()(T1& y,
                                                  const T1& x,
                                                  const T2& mean,
                                                  const T2& variance,
                                                  const T3& gamma,
                                                  const T4& beta) const
    {
        static_assert(std::is_same<T2, float>::value || std::is_same<T2, double>::value,
                      "Data type is not supported by this operation!");

        using ck::type_convert;
        using ck::math::sqrt;

        T2 tmp_x, tmp_y;

        tmp_x = type_convert<T2>(x);

        tmp_y = ((tmp_x - mean) / sqrt(variance + type_convert<T2>(epsilon_))) *
                    type_convert<T2>(gamma) +
                type_convert<T2>(beta);
        y = type_convert<T1>(tmp_y);
    };

    double epsilon_;
};

template <typename Y, typename X>
struct UnaryTypeConvert;

template <>
struct UnaryTypeConvert<float, ck::bhalf_t>
{
    __host__ __device__ void operator()(float& y, ck::bhalf_t& x) const
    {
        y = ck::type_convert<float, ck::bhalf_t>(x);
    }
};

template <>
struct UnaryTypeConvert<ck::bhalf_t, float>
{
    __host__ __device__ void operator()(ck::bhalf_t& y, float& x) const
    {
        y = ck::type_convert<ck::bhalf_t, float>(x);
    }
};

struct ConvInvscale
{
    /// @brief Op to multiply convolution results by inverted scale factors
    /// @param e Output after scaling
    /// @param c Convolution result
    /// @param d0 Input scale factor
    /// @param d1 Weights scale factor
    /// @param d2 Output scale factor
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ void
    operator()(E& e, const C& c, const D0& d0, const D1& d1, const D2& d2) const;

    template <>
    __host__ __device__ void operator()<f8_t, float, float, float, float>(
        f8_t& e, const float& c, const float& d0, const float& d1, const float& d2) const
    {
        e = type_convert<f8_t>(c / d0 / d1 / d2);
    };
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
