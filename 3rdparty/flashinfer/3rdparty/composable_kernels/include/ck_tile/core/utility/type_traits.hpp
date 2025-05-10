// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include <type_traits>
#include <stdint.h>

namespace ck_tile {

// remove_cvref_t
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

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

namespace impl {

template <typename T>
using has_is_static = decltype(T::is_static());

template <typename T>
struct is_static_impl
{
    static constexpr bool value = []() {
        if constexpr(is_detected<has_is_static, T>{})
            return T::is_static();
        else
            return std::is_arithmetic<T>::value;
    }();
};
} // namespace impl

template <typename T>
using is_static = impl::is_static_impl<remove_cvref_t<T>>;

template <typename T>
inline constexpr bool is_static_v = is_static<T>::value;

// TODO: deprecate this
template <typename T>
using is_known_at_compile_time = is_static<T>;
// TODO: if evaluating a rvalue, e.g. a const integer
// , this helper will also return false, which is not good(?)
//       do we need something like is_constexpr()?

// FIXME: do we need this anymore?
template <
    typename PY,
    typename PX,
    typename std::enable_if<std::is_pointer_v<PY> && std::is_pointer_v<PX>, bool>::type = false>
CK_TILE_HOST_DEVICE PY c_style_pointer_cast(PX p_x)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"
    return (PY)p_x; // NOLINT(old-style-cast, cast-align)
#pragma clang diagnostic pop
}

} // namespace ck_tile
