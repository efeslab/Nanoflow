// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

namespace detail {
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector
{
    using value_t = std::false_type;
    using type    = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...>
{
    using value_t = std::true_type;
    using type    = Op<Args...>;
};
} // namespace detail

struct nonesuch
{
    ~nonesuch()               = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;

template <typename T>
using is_pack2_invocable_t = decltype(std::declval<T&>().is_pack2_invocable);

template <typename T>
using is_pack4_invocable_t = decltype(std::declval<T&>().is_pack4_invocable);

template <typename T>
using is_pack8_invocable_t = decltype(std::declval<T&>().is_pack8_invocable);

} // namespace ck
