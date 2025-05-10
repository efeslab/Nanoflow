// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename T>
struct FillUniformDistribution
{
    float a_{-5.f};
    float b_{5.f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::uniform_real_distribution<float> dis(a_, b_);
        std::generate(first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(dis(gen)); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillUniformDistribution&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillNormalDistribution
{
    float mean_{0.f};
    float variance_{1.f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::normal_distribution<float> dis(mean_, std::sqrt(variance_));
        std::generate(first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(dis(gen)); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillNormalDistribution&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

// Normally FillUniformDistributionIntegerValue should use std::uniform_int_distribution as below.
// However this produces segfaults in std::mt19937 which look like inifite loop.
//      template <typename T>
//      struct FillUniformDistributionIntegerValue
//      {
//          int a_{-5};
//          int b_{5};
//
//          template <typename ForwardIter>
//          void operator()(ForwardIter first, ForwardIter last) const
//          {
//              std::mt19937 gen(11939);
//              std::uniform_int_distribution<int> dis(a_, b_);
//              std::generate(
//                  first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(dis(gen)); });
//          }
//      };

// Workaround for uniform_int_distribution not working as expected. See note above.<
template <typename T>
struct FillUniformDistributionIntegerValue
{
    float a_{-5.f};
    float b_{5.f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::uniform_real_distribution<float> dis(a_, b_);
        std::generate(
            first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(std::round(dis(gen))); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillUniformDistributionIntegerValue&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillNormalDistributionIntegerValue
{
    float mean_{0.f};
    float variance_{1.f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::normal_distribution<float> dis(mean_, std::sqrt(variance_));
        std::generate(
            first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(std::round(dis(gen))); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillNormalDistributionIntegerValue&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillMonotonicSeq
{
    T init_value_{0};
    T step_{1};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::generate(first, last, [=, n = init_value_]() mutable {
            auto tmp = n;
            n += step_;
            return tmp;
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillMonotonicSeq&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillConstant
{
    T value_{0};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::fill(first, last, value_);
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const -> std::void_t<
        decltype(std::declval<const FillConstant&>()(std::begin(std::forward<ForwardRange>(range)),
                                                     std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T, bool UseCos = true, bool UseAbs = false>
struct FillTrigValue
{
    template <typename T_, bool UseCos_ = true, bool UseAbs_ = false>
    struct LinearTrigGen
    {
        int i{0};
        auto operator()()
        {
            float v = 0;
            if constexpr(UseCos_)
            {
                v = cos(i);
            }
            else
            {
                v = sin(i);
            }
            if constexpr(UseAbs_)
                v = abs(v);
            i++;
            return ck_tile::type_convert<T_>(v);
        }
    };
    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        LinearTrigGen<T, UseCos, UseAbs> gen;
        std::generate(first, last, gen);
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const -> std::void_t<
        decltype(std::declval<const FillTrigValue&>()(std::begin(std::forward<ForwardRange>(range)),
                                                      std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

} // namespace ck_tile
