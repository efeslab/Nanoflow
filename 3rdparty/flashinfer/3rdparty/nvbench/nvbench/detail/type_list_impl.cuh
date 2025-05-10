#pragma once

#include <cstdint>
#include <tuple>

namespace nvbench
{

template <typename... Ts>
struct type_list
{};

template <typename T>
struct wrapped_type
{
  using type = T;
};

namespace tl::detail
{

template <typename... Ts>
auto size(nvbench::type_list<Ts...>) -> std::integral_constant<std::size_t, sizeof...(Ts)>;

template <std::size_t Idx, typename... Ts>
auto get(nvbench::type_list<Ts...>) -> std::tuple_element_t<Idx, std::tuple<Ts...>>;

template <typename... Ts, typename... Us>
auto concat(nvbench::type_list<Ts...>, nvbench::type_list<Us...>)
  -> nvbench::type_list<Ts..., Us...>;

//------------------------------------------------------------------------------
template <typename T, typename TLs>
struct prepend_each;

template <typename T>
struct prepend_each<T, nvbench::type_list<>>
{
  using type = nvbench::type_list<>;
};

template <typename T, typename TL, typename... TLTail>
struct prepend_each<T, nvbench::type_list<TL, TLTail...>>
{
  using cur  = decltype(detail::concat(nvbench::type_list<T>{}, TL{}));
  using next = typename detail::prepend_each<T, nvbench::type_list<TLTail...>>::type;
  using type = decltype(detail::concat(nvbench::type_list<cur>{}, next{}));
};

//------------------------------------------------------------------------------
template <typename TLs>
struct cartesian_product;

template <>
struct cartesian_product<nvbench::type_list<>>
{ // If no input type_lists are provided, there's just one output --
  // a null type_list:
  using type = nvbench::type_list<nvbench::type_list<>>;
};

template <typename... TLTail>
struct cartesian_product<nvbench::type_list<nvbench::type_list<>, TLTail...>>
{ // This is a recursion base case -- in practice empty type_lists should
  // not be passed into cartesian_product.
  using type = nvbench::type_list<>;
};

template <typename T, typename... Ts>
struct cartesian_product<nvbench::type_list<nvbench::type_list<T, Ts...>>>
{
  using cur  = nvbench::type_list<nvbench::type_list<T>>;
  using next = std::conditional_t<
    sizeof...(Ts) != 0,
    typename detail::cartesian_product<nvbench::type_list<nvbench::type_list<Ts...>>>::type,
    nvbench::type_list<>>;
  using type = decltype(detail::concat(cur{}, next{}));
};

template <typename T, typename... Tail, typename TL, typename... TLTail>
struct cartesian_product<nvbench::type_list<nvbench::type_list<T, Tail...>, TL, TLTail...>>
{
  using tail_prod = typename detail::cartesian_product<nvbench::type_list<TL, TLTail...>>::type;
  using cur       = typename detail::prepend_each<T, tail_prod>::type;
  using next      = typename detail::cartesian_product<
    nvbench::type_list<nvbench::type_list<Tail...>, TL, TLTail...>>::type;
  using type = decltype(detail::concat(cur{}, next{}));
};

//------------------------------------------------------------------------------
template <typename TypeList, typename Functor, std::size_t... Is>
void foreach (std::index_sequence<Is...>, Functor && f)
{
  // Garmonbozia...
  ((f(wrapped_type<decltype(detail::get<Is>(TypeList{}))>{})), ...);
}

template <typename TypeList, typename Functor>
void foreach (Functor &&f)
{
  constexpr std::size_t list_size = decltype(detail::size(TypeList{}))::value;
  using indices                   = std::make_index_sequence<list_size>;

  detail::foreach<TypeList>(indices{}, std::forward<Functor>(f));
}

} // namespace tl::detail
} // namespace nvbench
