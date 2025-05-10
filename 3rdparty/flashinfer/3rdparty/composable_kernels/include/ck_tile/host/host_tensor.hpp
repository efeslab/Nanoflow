// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/ranges.hpp"

namespace ck_tile {

template <typename Range>
CK_TILE_HOST std::ostream& LogRange(std::ostream& os,
                                    Range&& range,
                                    std::string delim,
                                    int precision = std::cout.precision(),
                                    int width     = 0)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << std::setw(width) << std::setprecision(precision) << v;
    }
    return os;
}

template <typename T, typename Range>
CK_TILE_HOST std::ostream& LogRangeAsType(std::ostream& os,
                                          Range&& range,
                                          std::string delim,
                                          int precision = std::cout.precision(),
                                          int width     = 0)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << std::setw(width) << std::setprecision(precision) << static_cast<T>(v);
    }
    return os;
}

template <typename F, typename T, std::size_t... Is>
CK_TILE_HOST auto call_f_unpack_args_impl(F f, T args, std::index_sequence<Is...>)
{
    return f(std::get<Is>(args)...);
}

template <typename F, typename T>
CK_TILE_HOST auto call_f_unpack_args(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return call_f_unpack_args_impl(f, args, std::make_index_sequence<N>{});
}

template <typename F, typename T, std::size_t... Is>
CK_TILE_HOST auto construct_f_unpack_args_impl(T args, std::index_sequence<Is...>)
{
    return F(std::get<Is>(args)...);
}

template <typename F, typename T>
CK_TILE_HOST auto construct_f_unpack_args(F, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return construct_f_unpack_args_impl<F>(args, std::make_index_sequence<N>{});
}

struct HostTensorDescriptor
{
    HostTensorDescriptor() = default;

    void CalculateStrides()
    {
        mStrides.clear();
        mStrides.resize(mLens.size(), 0);
        if(mStrides.empty())
            return;

        mStrides.back() = 1;
        std::partial_sum(mLens.rbegin(),
                         mLens.rend() - 1,
                         mStrides.rbegin() + 1,
                         std::multiplies<std::size_t>());
    }

    template <typename X, typename = std::enable_if_t<std::is_convertible_v<X, std::size_t>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens) : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    template <typename Lengths,
              typename = std::enable_if_t<
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Lengths>, std::size_t>>>
    HostTensorDescriptor(const Lengths& lens) : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    template <typename X,
              typename Y,
              typename = std::enable_if_t<std::is_convertible_v<X, std::size_t> &&
                                          std::is_convertible_v<Y, std::size_t>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens,
                         const std::initializer_list<Y>& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    template <typename Lengths,
              typename Strides,
              typename = std::enable_if_t<
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Lengths>, std::size_t> &&
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Strides>, std::size_t>>>
    HostTensorDescriptor(const Lengths& lens, const Strides& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    std::size_t get_num_of_dimension() const { return mLens.size(); }
    std::size_t get_element_size() const
    {
        assert(mLens.size() == mStrides.size());
        return std::accumulate(
            mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
    }
    std::size_t get_element_space_size() const
    {
        std::size_t space = 1;
        for(std::size_t i = 0; i < mLens.size(); ++i)
        {
            if(mLens[i] == 0)
                continue;

            space += (mLens[i] - 1) * mStrides[i];
        }
        return space;
    }

    const std::vector<std::size_t>& get_lengths() const { return mLens; }
    const std::vector<std::size_t>& GetStrides() const { return mStrides; }

    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        assert(sizeof...(Is) == this->get_num_of_dimension());
        std::initializer_list<std::size_t> iss{static_cast<std::size_t>(is)...};
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    std::size_t GetOffsetFromMultiIndex(std::vector<std::size_t> iss) const
    {
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    friend std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc);

    private:
    std::vector<std::size_t> mLens;
    std::vector<std::size_t> mStrides;
};

template <typename New2Old>
CK_TILE_HOST HostTensorDescriptor transpose_host_tensor_descriptor_given_new2old(
    const HostTensorDescriptor& a, const New2Old& new2old)
{
    std::vector<std::size_t> new_lengths(a.get_num_of_dimension());
    std::vector<std::size_t> new_strides(a.get_num_of_dimension());

    for(std::size_t i = 0; i < a.get_num_of_dimension(); i++)
    {
        new_lengths[i] = a.get_lengths()[new2old[i]];
        new_strides[i] = a.GetStrides()[new2old[i]];
    }

    return HostTensorDescriptor(new_lengths, new_strides);
}

struct joinable_thread : std::thread
{
    template <typename... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {
    }

    joinable_thread(joinable_thread&&) = default;
    joinable_thread& operator=(joinable_thread&&) = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

template <typename F, typename... Xs>
struct ParallelTensorFunctor
{
    F mF;
    static constexpr std::size_t NDIM = sizeof...(Xs);
    std::array<std::size_t, NDIM> mLens;
    std::array<std::size_t, NDIM> mStrides;
    std::size_t mN1d;

    ParallelTensorFunctor(F f, Xs... xs) : mF(f), mLens({static_cast<std::size_t>(xs)...})
    {
        mStrides.back() = 1;
        std::partial_sum(mLens.rbegin(),
                         mLens.rend() - 1,
                         mStrides.rbegin() + 1,
                         std::multiplies<std::size_t>());
        mN1d = mStrides[0] * mLens[0];
    }

    std::array<std::size_t, NDIM> GetNdIndices(std::size_t i) const
    {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0; idim < NDIM; ++idim)
        {
            indices[idim] = i / mStrides[idim];
            i -= indices[idim] * mStrides[idim];
        }

        return indices;
    }

    void operator()(std::size_t num_thread = 1) const
    {
        std::size_t work_per_thread = (mN1d + num_thread - 1) / num_thread;

        std::vector<joinable_thread> threads(num_thread);

        for(std::size_t it = 0; it < num_thread; ++it)
        {
            std::size_t iw_begin = it * work_per_thread;
            std::size_t iw_end   = std::min((it + 1) * work_per_thread, mN1d);

            auto f = [this, iw_begin, iw_end] {
                for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                {
                    call_f_unpack_args(this->mF, this->GetNdIndices(iw));
                }
            };
            threads[it] = joinable_thread(f);
        }
    }
};

template <typename F, typename... Xs>
CK_TILE_HOST auto make_ParallelTensorFunctor(F f, Xs... xs)
{
    return ParallelTensorFunctor<F, Xs...>(f, xs...);
}

template <typename T>
struct HostTensor
{
    using Descriptor = HostTensorDescriptor;
    using Data       = std::vector<T>;

    template <typename X>
    HostTensor(std::initializer_list<X> lens) : mDesc(lens), mData(mDesc.get_element_space_size())
    {
    }

    template <typename X, typename Y>
    HostTensor(std::initializer_list<X> lens, std::initializer_list<Y> strides)
        : mDesc(lens, strides), mData(mDesc.get_element_space_size())
    {
    }

    template <typename Lengths>
    HostTensor(const Lengths& lens) : mDesc(lens), mData(mDesc.get_element_space_size())
    {
    }

    template <typename Lengths, typename Strides>
    HostTensor(const Lengths& lens, const Strides& strides)
        : mDesc(lens, strides), mData(get_element_space_size())
    {
    }

    HostTensor(const Descriptor& desc) : mDesc(desc), mData(mDesc.get_element_space_size()) {}

    template <typename OutT>
    HostTensor<OutT> CopyAsType() const
    {
        HostTensor<OutT> ret(mDesc);
        std::transform(mData.cbegin(), mData.cend(), ret.mData.begin(), [](auto value) {
            return ck_tile::type_convert<OutT>(value);
        });
        return ret;
    }

    HostTensor()                  = delete;
    HostTensor(const HostTensor&) = default;
    HostTensor(HostTensor&&)      = default;

    ~HostTensor() = default;

    HostTensor& operator=(const HostTensor&) = default;
    HostTensor& operator=(HostTensor&&) = default;

    template <typename FromT>
    explicit HostTensor(const HostTensor<FromT>& other) : HostTensor(other.template CopyAsType<T>())
    {
    }

    decltype(auto) get_lengths() const { return mDesc.get_lengths(); }

    decltype(auto) GetStrides() const { return mDesc.GetStrides(); }

    std::size_t get_num_of_dimension() const { return mDesc.get_num_of_dimension(); }

    std::size_t get_element_size() const { return mDesc.get_element_size(); }

    std::size_t get_element_space_size() const { return mDesc.get_element_space_size(); }

    std::size_t get_element_space_size_in_bytes() const
    {
        return sizeof(T) * get_element_space_size();
    }

    // void SetZero() { ck_tile::ranges::fill<T>(mData, 0); }
    void SetZero() { std::fill(mData.begin(), mData.end(), 0); }

    template <typename F>
    void ForEach_impl(F&& f, std::vector<size_t>& idx, size_t rank)
    {
        if(rank == mDesc.get_num_of_dimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.get_lengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(F&& f)
    {
        std::vector<size_t> idx(mDesc.get_num_of_dimension(), 0);
        ForEach_impl(std::forward<F>(f), idx, size_t(0));
    }

    template <typename F>
    void ForEach_impl(const F&& f, std::vector<size_t>& idx, size_t rank) const
    {
        if(rank == mDesc.get_num_of_dimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.get_lengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<const F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(const F&& f) const
    {
        std::vector<size_t> idx(mDesc.get_num_of_dimension(), 0);
        ForEach_impl(std::forward<const F>(f), idx, size_t(0));
    }

    template <typename G>
    void GenerateTensorValue(G g, std::size_t num_thread = 1)
    {
        switch(mDesc.get_num_of_dimension())
        {
        case 1: {
            auto f = [&](auto i) { (*this)(i) = g(i); };
            make_ParallelTensorFunctor(f, mDesc.get_lengths()[0])(num_thread);
            break;
        }
        case 2: {
            auto f = [&](auto i0, auto i1) { (*this)(i0, i1) = g(i0, i1); };
            make_ParallelTensorFunctor(f, mDesc.get_lengths()[0], mDesc.get_lengths()[1])(
                num_thread);
            break;
        }
        case 3: {
            auto f = [&](auto i0, auto i1, auto i2) { (*this)(i0, i1, i2) = g(i0, i1, i2); };
            make_ParallelTensorFunctor(f,
                                       mDesc.get_lengths()[0],
                                       mDesc.get_lengths()[1],
                                       mDesc.get_lengths()[2])(num_thread);
            break;
        }
        case 4: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3) {
                (*this)(i0, i1, i2, i3) = g(i0, i1, i2, i3);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.get_lengths()[0],
                                       mDesc.get_lengths()[1],
                                       mDesc.get_lengths()[2],
                                       mDesc.get_lengths()[3])(num_thread);
            break;
        }
        case 5: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4) {
                (*this)(i0, i1, i2, i3, i4) = g(i0, i1, i2, i3, i4);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.get_lengths()[0],
                                       mDesc.get_lengths()[1],
                                       mDesc.get_lengths()[2],
                                       mDesc.get_lengths()[3],
                                       mDesc.get_lengths()[4])(num_thread);
            break;
        }
        case 6: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4, auto i5) {
                (*this)(i0, i1, i2, i3, i4, i5) = g(i0, i1, i2, i3, i4, i5);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.get_lengths()[0],
                                       mDesc.get_lengths()[1],
                                       mDesc.get_lengths()[2],
                                       mDesc.get_lengths()[3],
                                       mDesc.get_lengths()[4],
                                       mDesc.get_lengths()[5])(num_thread);
            break;
        }
        default: throw std::runtime_error("unspported dimension");
        }
    }

    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        return mDesc.GetOffsetFromMultiIndex(is...);
    }

    template <typename... Is>
    T& operator()(Is... is)
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...)];
    }

    template <typename... Is>
    const T& operator()(Is... is) const
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...)];
    }

    T& operator()(std::vector<std::size_t> idx)
    {
        return mData[mDesc.GetOffsetFromMultiIndex(idx)];
    }

    const T& operator()(std::vector<std::size_t> idx) const
    {
        return mData[mDesc.GetOffsetFromMultiIndex(idx)];
    }

    typename Data::iterator begin() { return mData.begin(); }

    typename Data::iterator end() { return mData.end(); }

    typename Data::pointer data() { return mData.data(); }

    typename Data::const_iterator begin() const { return mData.begin(); }

    typename Data::const_iterator end() const { return mData.end(); }

    typename Data::const_pointer data() const { return mData.data(); }

    typename Data::size_type size() const { return mData.size(); }

    template <typename U = T>
    auto AsSpan() const
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::add_const_t<std::remove_reference_t<U>>;
        return ck_tile::span<Element>{reinterpret_cast<Element*>(data()),
                                      size() * FromSize / ToSize};
    }

    template <typename U = T>
    auto AsSpan()
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::remove_reference_t<U>;
        return ck_tile::span<Element>{reinterpret_cast<Element*>(data()),
                                      size() * FromSize / ToSize};
    }

    Descriptor mDesc;
    Data mData;
};
} // namespace ck_tile
