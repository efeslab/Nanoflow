// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
// FIXME: amd_buffer_coherence_enum is only meaningful for buffer addressing. Need to split
// buffer_view definition for different memory address space (Global/GenericLds/Vgpr)
template <address_space_enum BufferAddressSpace,
          typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default>
struct buffer_view;

// Address Space: generic
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct buffer_view<address_space_enum::generic,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   amd_buffer_coherence_enum::coherence_default>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::generic;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto
    get(index_t i, bool is_valid_element, bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp;

            __builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));

            return tmp;
#else
            return *c_style_pointer_cast<const X*>(&p_data_[i]);
#endif
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return X{numeric<remove_cvref_t<T>>::zero()};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, is_valid_element);
            this->template set<X>(i, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp = x;

            __builtin_memcpy(&(p_data_[i]), &tmp, sizeof(X));
#else
            *c_style_pointer_cast<X*>(&p_data_[i]) = x;
#endif
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("buffer_view{");

        // AddressSpace
        printf("AddressSpace: generic, ");

        // p_data_
        printf("p_data_: %p, ", static_cast<void*>(const_cast<remove_cvref_t<T>*>(p_data_)));

        // buffer_size_
        printf("buffer_size_: ");
        print(buffer_size_);
        printf(", ");

        // invalid_element_value_
        printf("invalid_element_value_: ");
        print(invalid_element_value_);

        printf("}");
    }
};

// Address Space: Global
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          amd_buffer_coherence_enum Coherence>
struct buffer_view<address_space_enum::global,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   Coherence>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::global;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto
    get(index_t i, bool is_valid_element, bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_TILE_USE_AMD_BUFFER_LOAD
        bool constexpr use_amd_buffer_addressing = true;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return amd_buffer_load_invalid_element_return_zero<remove_cvref_t<T>,
                                                                   t_per_x,
                                                                   Coherence,
                                                                   oob_conditional_check>(
                    p_data_, i, is_valid_element, buffer_size_);
            }
            else
            {
                return amd_buffer_load_invalid_element_return_customized_value<
                    remove_cvref_t<T>,
                    t_per_x,
                    Coherence,
                    oob_conditional_check>(
                    p_data_, i, is_valid_element, buffer_size_, invalid_element_value_);
            }
        }
        else
        {
            if(is_valid_element)
            {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp;

                __builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));

                return tmp;
#else
                return *c_style_pointer_cast<const X*>(&p_data_[i]);
#endif
            }
            else
            {
                if constexpr(InvalidElementUseNumericalZeroValue)
                {
                    return X{numeric<remove_cvref_t<T>>::zero()};
                }
                else
                {
                    return X{invalid_element_value_};
                }
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto
    get_raw(remove_cvref_t<X>& dst, index_t i, bool is_valid_element) const
    {
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        amd_buffer_load_raw<remove_cvref_t<T>, t_per_x, Coherence, oob_conditional_check>(
            dst, p_data_, i, buffer_size_, is_valid_element);
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto
    async_get(remove_cvref_t<T>* smem, index_t i, bool /*is_valid_element*/) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        amd_async_buffer_load_with_oob<remove_cvref_t<T>, t_per_x, Coherence>(
            smem, p_data_, i, buffer_size_);
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, is_valid_element, x);
        }
        else if constexpr(Op == memory_operation_enum::atomic_add)
        {
            this->template atomic_add<X>(i, is_valid_element, x);
        }
        else if constexpr(Op == memory_operation_enum::atomic_max)
        {
            this->template atomic_max<X>(i, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, is_valid_element);
            this->template set<X>(i, is_valid_element, x + tmp);
            // tmp += x;
            // this->template set<X>(i, is_valid_element, tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_TILE_USE_AMD_BUFFER_STORE
        bool constexpr use_amd_buffer_addressing = true;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_store<remove_cvref_t<T>, t_per_x, Coherence>(
                x, p_data_, i, is_valid_element, buffer_size_);
        }
        else
        {
            if(is_valid_element)
            {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp = x;

                __builtin_memcpy(&(p_data_[i]), &tmp, sizeof(X));
#else
                *c_style_pointer_cast<X*>(&p_data_[i]) = x;
#endif
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set_raw(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;
        amd_buffer_store_raw<remove_cvref_t<T>, t_per_x, Coherence, oob_conditional_check>(
            x, p_data_, i, is_valid_element, buffer_size_);
    }

    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void atomic_add(index_t i, bool is_valid_element, const X& x)
    {
        using scalar_t = typename vector_traits<remove_cvref_t<T>>::scalar_type;

        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(get_address_space() == address_space_enum::global, "only support global mem");

#if CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            std::is_same_v<remove_cvref_t<scalar_t>, int32_t> ||
            std::is_same_v<remove_cvref_t<scalar_t>, float> ||
            (std::is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0);
#elif CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && (!CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT)
        bool constexpr use_amd_buffer_addressing =
            std::is_same_v<remove_cvref_t<scalar_t>, int32_t>;
#elif(!CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER) && CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            std::is_same_v<remove_cvref_t<scalar_t>, float> ||
            (std::is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0);
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_atomic_add<remove_cvref_t<T>, t_per_x>(
                x, p_data_, i, is_valid_element, buffer_size_);
        }
        else
        {
            if(is_valid_element)
            {
                atomic_add<X>(c_style_pointer_cast<X*>(&p_data_[i]), x);
            }
        }
    }

    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void atomic_max(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(get_address_space() == address_space_enum::global, "only support global mem");

#if CK_TILE_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64
        using scalar_t = typename vector_traits<remove_cvref_t<T>>::scalar_type;
        bool constexpr use_amd_buffer_addressing = std::is_same_v<remove_cvref_t<scalar_t>, double>;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_atomic_max<remove_cvref_t<T>, t_per_x>(
                x, p_data_, i, is_valid_element, buffer_size_);
        }
        else if(is_valid_element)
        {
            atomic_max<X>(c_style_pointer_cast<X*>(&p_data_[i]), x);
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("buffer_view{");

        // AddressSpace
        printf("AddressSpace: Global, ");

        // p_data_
        printf("p_data_: %p, ", static_cast<void*>(const_cast<remove_cvref_t<T>*>(p_data_)));

        // buffer_size_
        printf("buffer_size_: ");
        print(buffer_size_);
        printf(", ");

        // invalid_element_value_
        printf("invalid_element_value_: ");
        print(invalid_element_value_);

        printf("}");
    }
};

// Address Space: LDS
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct buffer_view<address_space_enum::lds,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   amd_buffer_coherence_enum::coherence_default>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::lds;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto
    get(index_t i, bool is_valid_element, bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp;

            __builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));

            return tmp;
#else
            using buf_t = ext_vector_t<typename vector_traits<remove_cvref_t<T>>::scalar_type,
                                       scalar_per_t_vector * scalar_per_x_vector>;
            // using buf_t = ushort __attribute__((ext_vector_type(8)));
            auto rtn = *c_style_pointer_cast<const buf_t*>(&p_data_[i]);
            return bit_cast<X>(rtn);
#endif
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return X{numeric<remove_cvref_t<T>>::zero()};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, is_valid_element);
            this->template set<X>(i, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_TILE_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
        bool constexpr workaround_int8_ds_write_issue = true;
#else
        bool constexpr workaround_int8_ds_write_issue = false;
#endif

        if constexpr(std::is_same<typename vector_traits<remove_cvref_t<T>>::scalar_type,
                                  int8_t>::value &&
                     workaround_int8_ds_write_issue)
        {
            if(is_valid_element)
            {
                // HACK: compiler would lower IR "store<i8, 16> address_space(3)" into inefficient
                // ISA, so I try to let compiler emit IR "store<i32, 4>" which would be lower to
                // ds_write_b128
                // TODO: remove this after compiler fix
                static_assert((std::is_same<remove_cvref_t<T>, int8_t>::value &&
                               std::is_same<remove_cvref_t<X>, int8_t>::value) ||
                                  (std::is_same<remove_cvref_t<T>, int8_t>::value &&
                                   std::is_same<remove_cvref_t<X>, int8x2_t>::value) ||
                                  (std::is_same<remove_cvref_t<T>, int8_t>::value &&
                                   std::is_same<remove_cvref_t<X>, int8x4_t>::value) ||
                                  (std::is_same<remove_cvref_t<T>, int8_t>::value &&
                                   std::is_same<remove_cvref_t<X>, int8x8_t>::value) ||
                                  (std::is_same<remove_cvref_t<T>, int8_t>::value &&
                                   std::is_same<remove_cvref_t<X>, int8x16_t>::value) ||
                                  (std::is_same<remove_cvref_t<T>, int8x4_t>::value &&
                                   std::is_same<remove_cvref_t<X>, int8x4_t>::value) ||
                                  (std::is_same<remove_cvref_t<T>, int8x8_t>::value &&
                                   std::is_same<remove_cvref_t<X>, int8x8_t>::value) ||
                                  (std::is_same<remove_cvref_t<T>, int8x16_t>::value &&
                                   std::is_same<remove_cvref_t<X>, int8x16_t>::value),
                              "wrong! not implemented for this combination, please add "
                              "implementation");

                if constexpr(std::is_same<remove_cvref_t<T>, int8_t>::value &&
                             std::is_same<remove_cvref_t<X>, int8_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int8_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int8_t*>(&x);
                }
                else if constexpr(std::is_same<remove_cvref_t<T>, int8_t>::value &&
                                  std::is_same<remove_cvref_t<X>, int8x2_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int16_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int16_t*>(&x);
                }
                else if constexpr(std::is_same<remove_cvref_t<T>, int8_t>::value &&
                                  std::is_same<remove_cvref_t<X>, int8x4_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32_t*>(&x);
                }
                else if constexpr(std::is_same<remove_cvref_t<T>, int8_t>::value &&
                                  std::is_same<remove_cvref_t<X>, int8x8_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x2_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x2_t*>(&x);
                }
                else if constexpr(std::is_same<remove_cvref_t<T>, int8_t>::value &&
                                  std::is_same<remove_cvref_t<X>, int8x16_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x4_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x4_t*>(&x);
                }
                else if constexpr(std::is_same<remove_cvref_t<T>, int8x4_t>::value &&
                                  std::is_same<remove_cvref_t<X>, int8x4_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32_t*>(&x);
                }
                else if constexpr(std::is_same<remove_cvref_t<T>, int8x8_t>::value &&
                                  std::is_same<remove_cvref_t<X>, int8x8_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x2_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x2_t*>(&x);
                }
                else if constexpr(std::is_same<remove_cvref_t<T>, int8x16_t>::value &&
                                  std::is_same<remove_cvref_t<X>, int8x16_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x4_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x4_t*>(&x);
                }
            }
        }
        else
        {
            if(is_valid_element)
            {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp = x;

                __builtin_memcpy(&(p_data_[i]), &tmp, sizeof(X));
#else
                using buf_t = ext_vector_t<typename vector_traits<remove_cvref_t<T>>::scalar_type,
                                           scalar_per_t_vector * scalar_per_x_vector>;

                *c_style_pointer_cast<buf_t*>(&p_data_[i]) = reinterpret_cast<const buf_t&>(x);
#endif
            }
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("buffer_view{");

        // AddressSpace
        printf("AddressSpace: Lds, ");

        // p_data_
        printf("p_data_: %p, ", static_cast<void*>(const_cast<remove_cvref_t<T>*>(p_data_)));

        // buffer_size_
        printf("buffer_size_: ");
        print(buffer_size_);
        printf(", ");

        // invalid_element_value_
        printf("invalid_element_value_: ");
        print(invalid_element_value_);

        printf("}");
    }
};

// Address Space: Vgpr
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct buffer_view<address_space_enum::vgpr,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   amd_buffer_coherence_enum::coherence_default>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::vgpr;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto
    get(index_t i, bool is_valid_element, bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp;

            __builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));

            return tmp;
#else
            return *c_style_pointer_cast<const X*>(&p_data_[i]);
#endif
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return X{numeric<remove_cvref_t<T>>::zero()};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, is_valid_element);
            this->template set<X>(i, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp = x;

            __builtin_memcpy(&(p_data_[i]), &tmp, sizeof(X));
#else
            *c_style_pointer_cast<X*>(&p_data_[i]) = x;
#endif
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("buffer_view{");

        // AddressSpace
        printf("AddressSpace: Vgpr, ");

        // p_data_
        printf("p_data_: %p, ", static_cast<void*>(const_cast<remove_cvref_t<T>*>(p_data_)));

        // buffer_size_
        printf("buffer_size_: ");
        print(buffer_size_);
        printf(", ");

        // invalid_element_value_
        printf("invalid_element_value_: ");
        print(invalid_element_value_);

        printf("}");
    }
};

template <address_space_enum BufferAddressSpace,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename T,
          typename BufferSizeType>
CK_TILE_HOST_DEVICE constexpr auto make_buffer_view(T* p, BufferSizeType buffer_size)
{
    return buffer_view<BufferAddressSpace, T, BufferSizeType, true, Coherence>{p, buffer_size};
}

template <address_space_enum BufferAddressSpace,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename T,
          typename BufferSizeType,
          typename X,
          typename std::enable_if<std::is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value,
                                  bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto
make_buffer_view(T* p, BufferSizeType buffer_size, X invalid_element_value)
{
    return buffer_view<BufferAddressSpace, T, BufferSizeType, false, Coherence>{
        p, buffer_size, invalid_element_value};
}

} // namespace ck_tile
