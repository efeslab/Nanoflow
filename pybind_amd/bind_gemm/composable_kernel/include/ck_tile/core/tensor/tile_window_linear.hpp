// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/utility.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

#define WINDOW_DISPATCH_ISSUE()                                     \
    if constexpr(i_access < 0)                                      \
    {                                                               \
        static_for<0, NumAccess, 1>{}([&](auto ia) { issue(ia); }); \
    }                                                               \
    else                                                            \
    {                                                               \
        static_assert(i_access < NumAccess);                        \
        issue(number<i_access>{});                                  \
    }

//
// This version of tile window will pre-cache offset/flags based on need
//
// LinearBottomDims_, e.g seq<0, 1> for 2d tensor, the last one is linear dim
// so last dim can use immediate offset to indexing, can save register
// TODO: if using this struct, better use load_raw()/store_raw(), can control
//       the the immediate offset on the fly
// space-filing-curve is non-snaked here!
//
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename LinearBottomDims_>
struct tile_window_linear
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<StaticTileDistribution_>;

    using WindowAdaptor    = typename TileDstr::PsYs2XsAdaptor;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    using DataType         = remove_cvref_t<typename BottomTensorView::DataType>;
    using LinearBottomDims = remove_cvref_t<LinearBottomDims_>;

    static_assert(LinearBottomDims::size() == BottomTensorView::get_num_of_dimension());

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::get_num_of_top_dimension();
    static constexpr index_t NDimBottomTensor     = BottomTensorDesc::get_num_of_dimension();

    static constexpr index_t NDimP = TileDstr::get_num_of_dimension_p();
    static constexpr index_t NDimY = TileDstr::get_num_of_dimension_y();

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    // TODO: check WindowLengths and StaticTileDistribution are consistent

    static_assert(ck_tile::is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileDstr::is_static(), "wrong!");

    static_assert(NDimBottomTensor == WindowAdaptor::get_num_of_bottom_dimension(),
                  "wrong! inconsistent # of diemsnions");

    using AdaptorTopIndex   = array<index_t, NDimWindowAdaptorTop>;
    using BottomTensorIndex = array<index_t, NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    struct traits
    {
        private:
        // return vector dimension among [y0, y1, ...]
        CK_TILE_DEVICE static constexpr auto get_window_adaptor_ys_safe_vector_length_strides()
        {
            // bottom tensor top dimension vector lengths and strides
            const auto [bottom_tensor_top_dim_vector_lengths,
                        bottom_tensor_top_dim_vector_strides] =
                BottomTensorDesc::get_top_dimension_safe_vector_length_strides();

            // window vector lengths/strides
            const auto window_adaptor_bottom_dim_vector_lengths =
                bottom_tensor_top_dim_vector_lengths;
            const auto window_adaptor_bottom_dim_vector_strides =
                bottom_tensor_top_dim_vector_strides;

            // window adaptor [p0, p1, ..., y0, y1, ...]
            array<index_t, WindowAdaptor::get_num_of_hidden_dimension()>
                window_adaptor_vector_lengths{-1};
            array<index_t, WindowAdaptor::get_num_of_hidden_dimension()>
                window_adaptor_vector_strides{-1};

            constexpr auto window_adaptor_bottom_dims =
                WindowAdaptor::get_bottom_dimension_hidden_ids();

            set_container_subset(window_adaptor_vector_lengths,
                                 window_adaptor_bottom_dims,
                                 window_adaptor_bottom_dim_vector_lengths);
            set_container_subset(window_adaptor_vector_strides,
                                 window_adaptor_bottom_dims,
                                 window_adaptor_bottom_dim_vector_strides);

            const auto [window_adaptor_ps_ys_vector_lengths, window_adaptor_ps_ys_vector_strides] =
                WindowAdaptor{}.get_top_dimension_safe_vector_length_strides(
                    window_adaptor_vector_lengths, window_adaptor_vector_strides);

            // [y0, y1, ...]
            constexpr auto y_dims =
                typename arithmetic_sequence_gen<TileDstr::get_num_of_dimension_p(),
                                                 NDimWindowAdaptorTop,
                                                 1>::type{};

            return make_tuple(get_container_subset(window_adaptor_ps_ys_vector_lengths, y_dims),
                              get_container_subset(window_adaptor_ps_ys_vector_strides, y_dims));
        }

        static constexpr auto get_vector_dim_y_scalar_per_vector()
        {
            const auto [ys_vector_lengths, ys_vector_strides] =
                get_window_adaptor_ys_safe_vector_length_strides();

            index_t VectorDimY_      = 0;
            index_t ScalarPerVector_ = 1;

            for(index_t i = 0; i < NDimY; ++i)
            {
                if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector_)
                {
                    ScalarPerVector_ = ys_vector_lengths[i];
                    VectorDimY_      = i;
                }
            }

            return make_tuple(VectorDimY_, ScalarPerVector_);
        }

        public:
        static constexpr index_t VectorDimY = get_vector_dim_y_scalar_per_vector().template at<0>();
        static constexpr index_t ScalarPerVector =
            get_vector_dim_y_scalar_per_vector().template at<1>();

        using vector_t = thread_buffer<DataType, ScalarPerVector>;

        private:
        static constexpr auto scalars_per_access_ = [] {
            constexpr auto scalars_per_access_arr = generate_array(
                [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, number<NDimY>{});

            /// TODO: add non-automatic storage argument support to macro TO_SEQUENCE()
            constexpr auto NDimY_ = NDimY;

            return TO_SEQUENCE(scalars_per_access_arr, NDimY_);
        }();

        static constexpr auto get_space_filling_curve()
        {
            constexpr auto thread_tensor_lengths_ys =
                to_sequence(TileDstr{}.get_ys_to_d_descriptor().get_lengths());

            // FIXME: need logic to judge dim access order
            using DimAccessOrder = typename arithmetic_sequence_gen<0, NDimY, 1>::type;

            return space_filling_curve<decltype(thread_tensor_lengths_ys),
                                       DimAccessOrder,
                                       decltype(scalars_per_access_),
                                       false /*!!! no snaked curve! */>{};
        }

        public:
        using SFC_Ys = decltype(get_space_filling_curve());

        static constexpr index_t NumAccess = SFC_Ys::get_num_of_access();

        static_assert(0 < NumAccess, "Wrong! NumAccess should be larger than 0");

        private:
        static constexpr auto get_num_non_linear_access()
        {
            constexpr auto sfc_access_lens = SFC_Ys::access_lengths;
            using ys_to_rhs_major =
                typename decltype(TileDstr{}.get_static_tile_distribution_encoding())::Ys2RHsMajor;

            constexpr auto non_linear = [&]() {
                index_t cnt = 1;
                static_for<0, NDimY, 1>{}([&](auto i_dim_y) {
                    constexpr auto rhs_major    = ys_to_rhs_major{}[i_dim_y];
                    constexpr auto target_h_dim = number<rhs_major - 1>{}; // no r dim here!
                    if constexpr(LinearBottomDims{}[target_h_dim] == 0)
                    {
                        cnt *= sfc_access_lens[i_dim_y];
                    }
                });
                return cnt;
            }();

            return non_linear;
        }

        // example:
        // non_linear_access_map: sequence<0, 0, 0, 0, 1, 1, 1, 1> for 8 access, totally 2 register
        // used
        //  -> histogram : sequence<4, 4>
        //  -> prefixsum : seqneuce<0, 4, 8>
        // non_linear_access_map: sequence<0, 1, 2, 3, 4, 5, 6, 7> for 8 access, totally 8 register
        // used, will pre-cache 8
        //  -> histogram : sequence<1, 1, 1, 1, 1, 1, 1, 1>
        //  -> prefixsum : seqneuce<0, 1, 2, 3, 4, 5, 6, 7, 8>
        // non_linear_access_map: sequence<0, 0, 1, 1, 2, 2, 3, 3> for 8 access, totally 4 register
        // used, will pre-cache 4
        //  -> histogram : sequence<2, 2, 2, 2>
        //  -> prefixsum : seqneuce<0, 2, 4, 6, 8>
        static constexpr auto get_non_linear_access_map()
        {
            constexpr auto sfc_access_lens = SFC_Ys::access_lengths;
            using ys_to_rhs_major =
                typename decltype(TileDstr{}.get_static_tile_distribution_encoding())::Ys2RHsMajor;
            constexpr auto non_linear_map = [&]() {
                array<index_t, NumAccess> m_{0};
                index_t cumulative_len_            = 1;
                index_t cumulative_non_linear_len_ = 1;
                static_for<0, NDimY, 1>{}([&](auto i_y) {
                    constexpr auto i_dim_y       = number<NDimY - i_y - 1>{}; // from right to left
                    constexpr auto rhs_major     = ys_to_rhs_major{}[i_dim_y];
                    constexpr auto target_h_dim  = number<rhs_major - 1>{}; // no r dim here!
                    constexpr auto is_linear_dim = LinearBottomDims{}[target_h_dim];

                    array<index_t, NumAccess> current_m_{0};
                    constexpr auto current_len_ = sfc_access_lens[i_dim_y];

                    // copy cumulative length as current pattern
                    for(auto i_ = 0; i_ < cumulative_len_; i_++)
                    {
                        current_m_(i_) = m_[i_];
                    }
                    for(auto j_ = 0; j_ < current_len_; j_++)
                    {
                        auto j_offset_ = is_linear_dim ? 0 : j_ * cumulative_non_linear_len_;
                        for(auto i_ = 0; i_ < cumulative_len_; i_++)
                        {
                            m_(j_ * cumulative_len_ + i_) = current_m_[i_] + j_offset_;
                        }
                    }
                    cumulative_len_ *= current_len_;
                    if(!is_linear_dim)
                        cumulative_non_linear_len_ *= current_len_;
                });
                return m_;
            }();

            return TO_SEQUENCE(non_linear_map, NumAccess);
        }

        static constexpr auto get_non_linear_access_histogram()
        {
            constexpr auto m_ = get_non_linear_access_map();
            // m_.foo();

            constexpr auto r_ =
                typename arithmetic_sequence_gen<0, get_num_non_linear_access() + 1, 1>::type{};

            constexpr auto h_ = histogram_sorted_sequence(m_, r_);

            return h_;
        }

        static constexpr auto get_non_linear_access_histogram_prefix_sum()
        {
            constexpr auto h_            = get_non_linear_access_histogram();
            constexpr auto h_prefix_sum_ = prefix_sum_sequence(h_);
            return h_prefix_sum_;
        }

        public:
        static constexpr index_t NumAccess_NonLinear = get_num_non_linear_access();
        using AccessMap_NonLinear       = decltype(get_non_linear_access_map()); // sequence
        using AccessHistogram_NonLinear = decltype(get_non_linear_access_histogram());
        using AccessPrefixSum_NonLinear = decltype(get_non_linear_access_histogram_prefix_sum());
    };

    static constexpr index_t NumAccess           = traits::NumAccess;
    static constexpr index_t NumAccess_NonLinear = traits::NumAccess_NonLinear;
    using AccessMap_NonLinear                    = typename traits::AccessMap_NonLinear;
    using AccessHistogram_NonLinear              = typename traits::AccessHistogram_NonLinear;
    using AccessPrefixSum_NonLinear              = typename traits::AccessPrefixSum_NonLinear;

    CK_TILE_DEVICE constexpr tile_window_linear() = default;

    CK_TILE_DEVICE constexpr tile_window_linear(const BottomTensorView& bottom_tensor_view,
                                                const WindowLengths& window_lengths,
                                                const BottomTensorIndex& window_origin,
                                                const TileDstr& tile_distribution)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin},
          tile_dstr_{tile_distribution},
          cached_coords_{},
          cached_flags_{}
    {
        auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(make_tuple(get_warp_id(), get_lane_id()),
                             generate_tuple([&](auto) { return number<0>{}; }, number<NDimY>{})));

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();

        auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // future load/store() calls (might allocate more registers)
        using SFC_Ys = typename traits::SFC_Ys;

        static_for<0, NumAccess, 1>{}([&](auto i_access) {
            constexpr auto non_linear_id = number<AccessMap_NonLinear{}[i_access]>{};
            constexpr auto need_save_non_linear_coord =
                bool_constant<AccessPrefixSum_NonLinear{}[non_linear_id] == i_access>{};

            if constexpr(need_save_non_linear_coord)
            {
                cached_coords_(non_linear_id) = bottom_tensor_thread_coord_tmp;
            }

            // TODO: need pad_tensor_view to check which dim need use flag to check
            //      cached flag is independent from non-linear-coord
            //      but need be updated in move_tile, with proper dims
            cached_flags_(i_access) = coordinate_has_valid_offset_assuming_top_index_is_valid(
                bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_coord_tmp);

            if constexpr(i_access != (NumAccess - 1))
            {
                constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(i_access); // tuple of number
                constexpr auto idx_diff_ps_ys = container_concat(
                    generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                    idx_diff_ys);

                move_window_adaptor_and_bottom_tensor_thread_coordinate(
                    window_adaptor_thread_coord_tmp,
                    bottom_tensor_thread_coord_tmp,
                    idx_diff_ps_ys);
            }
        });
    }

    CK_TILE_DEVICE static constexpr index_t get_num_of_dimension() { return NDimBottomTensor; }

    CK_TILE_DEVICE static constexpr bool has_static_tile_distribution()
    {
        return TileDstr::is_static();
    }

    CK_TILE_DEVICE constexpr auto get_window_lengths() const { return window_lengths_; }

    CK_TILE_DEVICE constexpr auto get_tile_distribution() const { return tile_dstr_; }

    CK_TILE_DEVICE constexpr auto get_bottom_tensor_view() const { return bottom_tensor_view_; }

    CK_TILE_DEVICE constexpr auto get_window_origin() const { return window_origin_; }

    CK_TILE_DEVICE constexpr void
    set_bottom_tensor_view_data_ptr(typename BottomTensorView::DataType* data)
    {
        bottom_tensor_view_.buf_.p_data_ = data;
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    template <typename ATopIndex>
    CK_TILE_DEVICE void move_window_adaptor_and_bottom_tensor_thread_coordinate(
        WindowAdaptorCoord& window_adaptor_thread_coord,
        BottomTensorCoord& bottom_tensor_thread_coord,
        const ATopIndex& idx_diff_adaptor_top) const
    {
        array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                       window_adaptor_thread_coord,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                               bottom_tensor_thread_coord,
                               idx_diff_adaptor_bottom);
    }

    template <index_t i_access>
    CK_TILE_DEVICE static constexpr auto get_bottom_linear_coordinate(number<i_access>)
    {
        using SFC_Ys          = typename traits::SFC_Ys;
        constexpr auto idx_ys = SFC_Ys::get_index(number<i_access>{});
        using ys_to_rhs_major =
            typename decltype(TileDstr{}.get_static_tile_distribution_encoding())::Ys2RHsMajor;

        constexpr auto modified_idx_ys = generate_tuple(
            [&](auto i_dim_y) {
                constexpr auto rhs_major    = ys_to_rhs_major{}[i_dim_y];
                constexpr auto target_h_dim = number<rhs_major - 1>{}; // no r dim here!
                if constexpr(LinearBottomDims{}[target_h_dim] == 0)
                {
                    return number<0>{};
                }
                else
                {
                    return number<idx_ys[i_dim_y]>{};
                }
            },
            number<NDimY>{});

        constexpr auto adaptor_ = TileDstr{}.get_ps_ys_to_xs_adaptor();
        constexpr auto idx_ =
            container_concat(make_tuple(number<0>{}, number<0>{}), modified_idx_ys);

        return adaptor_.calculate_bottom_index(idx_);
    }

    template <index_t i_access>
    CK_TILE_DEVICE static constexpr index_t get_bottom_linear_offset(number<i_access>)
    {
        constexpr auto linear_coord = get_bottom_linear_coordinate(number<i_access>{});
        constexpr auto is_pure_linear_tensor =
            reduce_on_sequence(LinearBottomDims{}, multiplies{}, number<1>{});
        if constexpr(is_pure_linear_tensor)
        {
            // this case usually is a LDS window, everything is known at compile tile.
            // we directly use BottomTensorView transform to compute the offset, in case padding
            auto bottom_tensor_coord =
                make_tensor_coordinate(BottomTensorView{}.get_tensor_descriptor(), linear_coord);
            return bottom_tensor_coord.get_offset();
        }
        else
        {
            // this case usually is a global window, where last dim can be linear
            // we hack here, that use the original TileDstr to compute the linear offset
            // ... hoping that there is no extra padding between other dims, which make sense
            // since that would introduce runtime length (so can't use linear offset)
            constexpr index_t linear_offset = [&]() {
                constexpr auto x_idx_ = linear_coord;
                constexpr auto x_len_ = TileDstr{}.get_lengths();
                static_assert(x_idx_.size() == x_len_.size());
                constexpr index_t x_dims_ = x_idx_.size();
                index_t cu_stride_        = 1;
                index_t cu_offset_        = 0;
                static_for<0, x_dims_, 1>{}([&](auto i_) {
                    auto r_i_ = number<x_dims_ - i_ - 1>{};
                    cu_offset_ += x_idx_[r_i_] * cu_stride_;
                    cu_stride_ *= x_len_[r_i_];
                });
                return cu_offset_;
            }();
            return linear_offset;
        }
    }

    CK_TILE_DEVICE constexpr auto get_num_of_access() const { return traits::NumAccess; }

    template <index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load(number<i_access> = {}, bool_constant<oob_conditional_check> = {}) const
    {
        using vector_t = typename traits::vector_t;
        using SFC_Ys   = typename traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        auto dst_tensor = make_static_distributed_tensor<DataType>(tile_dstr);

        auto issue = [&](auto i_access_) {
            constexpr auto IAccess = number<i_access_>{};

            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            constexpr auto linear_offset = get_bottom_linear_offset(IAccess);

            // read from bottom tensor
            const vector_t vec_value =
                get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    linear_offset,
                    bottom_tensor_flag,
                    bool_constant<oob_conditional_check>{});
#if 1
            // data index [y0, y1, ...]
            constexpr auto idx_diff_ys = SFC_Ys::get_index(IAccess);
            // write into distributed tensor
            static_for<0, traits::ScalarPerVector, 1>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == traits::VectorDimY ? (idx_diff_ys[jj] + j) : idx_diff_ys[jj];
                    },
                    number<NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                dst_tensor.get_thread_buffer().template at<d>() =
                    vec_value.template get_as<DataType>()[j];
            });
#else
            constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
            static_assert(d % traits::ScalarPerVector == 0);

            dst_tensor.get_thread_buffer().template get_as<vector_t>()(
                number<d / traits::ScalarPerVector>{}) = bit_cast<vector_t>(vec_value);
#endif
        };

        WINDOW_DISPATCH_ISSUE();

        return dst_tensor;
    }

    template <typename DstTile, index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load(DstTile& dst_tensor,
                             number<i_access>                     = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        using vector_t = typename traits::vector_t;
        using SFC_Ys   = typename traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // auto dst_tensor = make_static_distributed_tensor<DataType>(tile_dstr);

        auto issue = [&](auto i_access_) {
            constexpr auto IAccess = number<i_access_>{};

            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            constexpr auto linear_offset = get_bottom_linear_offset(IAccess);

            // read from bottom tensor
            const vector_t vec_value =
                get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    linear_offset,
                    bottom_tensor_flag,
                    bool_constant<oob_conditional_check>{});
#if 1
            // data index [y0, y1, ...]
            constexpr auto idx_diff_ys = SFC_Ys::get_index(IAccess);
            // write into distributed tensor
            static_for<0, traits::ScalarPerVector, 1>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == traits::VectorDimY ? (idx_diff_ys[jj] + j) : idx_diff_ys[jj];
                    },
                    number<NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                dst_tensor.get_thread_buffer().template at<d>() =
                    vec_value.template get_as<DataType>()[j];
            });
#else
            constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
            static_assert(d % traits::ScalarPerVector == 0);

            dst_tensor.get_thread_buffer().template get_as<vector_t>()(
                number<d / traits::ScalarPerVector>{}) = bit_cast<vector_t>(vec_value);
#endif
        };

        WINDOW_DISPATCH_ISSUE();

        return dst_tensor;
    }

    template <typename DstTile,
              index_t i_access           = -1,
              bool oob_conditional_check = true,
              bool pre_nop               = false>
    CK_TILE_DEVICE void load_raw(DstTile& dst_tensor,
                                 number<i_access> = {}, // negative means loop over all num_access
                                 bool_constant<oob_conditional_check> = {},
                                 bool_constant<pre_nop>               = {}) const
    {
        using vector_t = typename traits::vector_t;
        using SFC_Ys   = typename traits::SFC_Ys;
        static constexpr index_t YElementSize =
            TileDstr{}.get_ys_to_d_descriptor().get_element_space_size();
        static_assert(YElementSize % traits::ScalarPerVector == 0);
        using vectorized_tbuf = array<vector_t, YElementSize / traits::ScalarPerVector>;

        constexpr auto tile_dstr = TileDstr{};

        auto& dst_vec_tbuf = reinterpret_cast<vectorized_tbuf&>(dst_tensor.get_thread_buffer());

        auto issue = [&](auto i_access_) {
            constexpr auto IAccess  = number<i_access_>{};
            constexpr auto pre_nop_ = [&]() {
                if constexpr(pre_nop && i_access_ == 0 &&
                             BottomTensorView::buffer_view::get_address_space() ==
                                 address_space_enum::global)
                    return bool_constant<true>{};
                else
                    return bool_constant<false>{};
            }();

            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);
            constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
            static_assert(d % traits::ScalarPerVector == 0);

            get_bottom_tensor_view().template get_vectorized_elements_raw<vector_t>(
                dst_vec_tbuf.template at<d / traits::ScalarPerVector>(),
                bottom_tensor_thread_coord,
                linear_offset /**/,
                bottom_tensor_flag,
                bool_constant<oob_conditional_check>{},
                pre_nop_);
#if CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE || \
    CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE
            asm volatile(""); // this is starting from rocm-6.2, but same sympton, reuse this flag
#endif
        };

        WINDOW_DISPATCH_ISSUE();
    }

    // TODO: currently async load only implemented in inline asm
    template <typename LdsTileWindow_,
              index_t i_access           = -1,
              bool oob_conditional_check = true,
              bool pre_nop               = false>
    CK_TILE_DEVICE auto async_load_raw(LdsTileWindow_&& lds_tile,
                                       number<i_access>                     = {},
                                       bool_constant<oob_conditional_check> = {},
                                       bool_constant<pre_nop>               = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        // currently we only support everything is non linear dim
        // actually it's not performant if we have linear dim(e.g. fast changing)
        static_assert(NumAccess_NonLinear == NumAccess);
        static_assert(BottomTensorView::buffer_view::get_address_space() ==
                      address_space_enum::global);

        // issues * warps * lanes
        static_assert(LdsTileWindow::get_num_of_dimension() == 3); // TODO: hard coded

        const index_t size_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{})) *
            sizeof(LdsDataType);

        const index_t size_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t size_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();
        m0_set_with_memory(m0_init_value); // This should be wave independent

        using vector_t = typename traits::vector_t;

        LdsDataType* smem = lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_;

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess  = number<i_access_>{};
            constexpr auto pre_nop_ = [&]() {
                if constexpr(pre_nop && i_access_ == 0)
                    return bool_constant<true>{};
                else
                    return bool_constant<false>{};
            }();

            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            auto bottom_tensor_flag         = cached_flags_[IAccess]; // get this flag anyway

            // read from bottom tensor
            get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                smem, bottom_tensor_thread_coord, 0, bottom_tensor_flag, pre_nop_);

            // move thread coordinate
            if constexpr(i_access_ != (NumAccess - 1))
            {
                m0_inc_with_memory(size_per_issue);
            }
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <typename LdsTileWindow_, index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto async_load(LdsTileWindow_&& lds_tile,
                                   number<i_access>                     = {},
                                   bool_constant<oob_conditional_check> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        // currently we only support everything is non linear dim
        // actually it's not performant if we have linear dim(e.g. fast changing)
        static_assert(NumAccess_NonLinear == NumAccess);
        static_assert(BottomTensorView::buffer_view::get_address_space() ==
                      address_space_enum::global);

        // issues * warps * lanes
        static_assert(LdsTileWindow::get_num_of_dimension() == 3); // TODO: hard coded

        // TODO: LDS offset is not good for intrinsic based implementation(compiler can't figure out
        // dependency) hence avoid use offset based solution. size_per_buf should be zero (how to
        // check?)
        constexpr index_t size_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{}));

        constexpr index_t size_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) -
            size_per_buf;

        constexpr index_t size_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) -
            size_per_buf;

        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();

        using vector_t = typename traits::vector_t;

        // TODO: we force CK_TILE_LDS_ADDR
        CK_TILE_LDS_ADDR LdsDataType* smem =
            lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_ + m0_init_value;

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            // read from bottom tensor
            get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                smem,
                bottom_tensor_thread_coord,
                0,
                bottom_tensor_flag,
                bool_constant<oob_conditional_check>{});

            // move thread coordinate
            if constexpr(i_access_ != (NumAccess - 1))
            {
                smem += size_per_issue; // Note we manually increase the per-issue offset
            }
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void store(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                              number<i_access>                     = {},
                              bool_constant<oob_conditional_check> = {}) const
    {

        using vector_t = typename traits::vector_t;
        using SFC_Ys   = typename traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];
            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);

            // read from distributed tensor
            vector_t vec_value;

            static_for<0, traits::ScalarPerVector, 1>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == traits::VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                    },
                    number<NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                vec_value.template get_as<DataType>()(j) =
                    dstr_tensor.get_thread_buffer().template at<d>();
            });

            // write into bottom tensor
            get_bottom_tensor_view().template set_vectorized_elements<vector_t>(
                bottom_tensor_thread_coord,
                linear_offset,
                bottom_tensor_flag,
                vec_value,
                bool_constant<oob_conditional_check>{});
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <index_t i_access = -1>
    CK_TILE_DEVICE void store_raw(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                                  number<i_access> = {}) const
    {
        using vector_t = typename traits::vector_t;
        using SFC_Ys   = typename traits::SFC_Ys;

        constexpr auto tile_dstr                    = TileDstr{};
        static constexpr bool oob_conditional_check = true;

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);

            // read from distributed tensor
            vector_t vec_value;
            static_for<0, traits::ScalarPerVector, 1>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == traits::VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                    },
                    number<NDimY>{});
                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);
                vec_value.template get_as<DataType>()(j) =
                    dstr_tensor.get_thread_buffer().template at<d>();
            });

            // write into bottom tensor
            get_bottom_tensor_view()
                .template set_vectorized_elements_raw<vector_t, oob_conditional_check>(
                    bottom_tensor_thread_coord, linear_offset, bottom_tensor_flag, vec_value);
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void update(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                               number<i_access>                     = {},
                               bool_constant<oob_conditional_check> = {}) const
    {

        using vector_t = typename traits::vector_t;
        using SFC_Ys   = typename traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);

            // read from distributed tensor
            vector_t vec_value;

            static_for<0, traits::ScalarPerVector, 1>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == traits::VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                    },
                    number<NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                vec_value.template get_as<DataType>()(j) =
                    dstr_tensor.get_thread_buffer().template at<d>();
            });

            // write into bottom tensor
            get_bottom_tensor_view().template update_vectorized_elements<vector_t>(
                bottom_tensor_thread_coord,
                linear_offset,
                bottom_tensor_flag,
                vec_value,
                bool_constant<oob_conditional_check>{});
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <index_t i_access = -1, bool oob_conditional_check = true, bool pre_nop = false>
    CK_TILE_DEVICE void update_raw(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                                   number<i_access>                     = {},
                                   bool_constant<oob_conditional_check> = {},
                                   bool_constant<pre_nop>               = {}) const
    {

        using vector_t = typename traits::vector_t;
        using SFC_Ys   = typename traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);

            // read from distributed tensor
            vector_t vec_value;

            static_for<0, traits::ScalarPerVector, 1>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == traits::VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                    },
                    number<NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                vec_value.template get_as<DataType>()(j) =
                    dstr_tensor.get_thread_buffer().template at<d>();
            });

            // write into bottom tensor
            get_bottom_tensor_view().template update_vectorized_elements_raw<vector_t>(
                bottom_tensor_thread_coord,
                linear_offset,
                bottom_tensor_flag,
                vec_value,
                bool_constant<oob_conditional_check>{},
                bool_constant<pre_nop>{});
        };

        WINDOW_DISPATCH_ISSUE();
    }

    // move thread's botom tensor coordiante
    // [x0', x1', ... ] ==> [offset]
    // also move window-origin
    CK_TILE_DEVICE void move(const BottomTensorIndex& step)
    {
        window_origin_ += step;

        static_for<0, NumAccess, 1>{}([&](auto i_access) {
            constexpr auto IAccess       = number<i_access>{};
            constexpr auto non_linear_id = number<AccessMap_NonLinear{}[i_access]>{};
            constexpr auto need_update_non_linear_coord =
                bool_constant<AccessPrefixSum_NonLinear{}[non_linear_id] == i_access>{};

            if constexpr(need_update_non_linear_coord)
            {
                move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                                       cached_coords_(non_linear_id),
                                       step);
            }

            // move the current coord with linear_coords
            auto tmp_coords             = cached_coords_[non_linear_id];
            constexpr auto linear_coord = get_bottom_linear_coordinate(IAccess);
            move_tensor_coordinate(
                bottom_tensor_view_.get_tensor_descriptor(), tmp_coords, linear_coord);

            cached_flags_(IAccess) = coordinate_has_valid_offset_assuming_top_index_is_valid(
                bottom_tensor_view_.get_tensor_descriptor(), tmp_coords);
        });
    }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& new_window_origin)
    {
        window_origin_ = new_window_origin;

        auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            TileDstr{}.get_ps_ys_to_xs_adaptor(),
            container_concat(make_tuple(get_warp_id(), get_lane_id()),
                             generate_tuple([&](auto) { return number<0>{}; }, number<NDimY>{})));

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin_ + window_adaptor_thread_coord_tmp.get_bottom_index();

        auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // future load/store() calls (might allocate more registers)
        using SFC_Ys = typename traits::SFC_Ys;

        static_for<0, NumAccess, 1>{}([&](auto i_access) {
            constexpr auto non_linear_id = number<AccessMap_NonLinear{}[i_access]>{};
            constexpr auto need_save_non_linear_coord =
                bool_constant<AccessPrefixSum_NonLinear{}[non_linear_id] == i_access>{};

            if constexpr(need_save_non_linear_coord)
            {
                cached_coords_(non_linear_id) = bottom_tensor_thread_coord_tmp;
            }

            if constexpr(i_access != (NumAccess - 1))
            {
                constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(i_access); // tuple of number
                constexpr auto idx_diff_ps_ys = container_concat(
                    generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                    idx_diff_ys);

                move_window_adaptor_and_bottom_tensor_thread_coordinate(
                    window_adaptor_thread_coord_tmp,
                    bottom_tensor_thread_coord_tmp,
                    idx_diff_ps_ys);
            }
        });
    }

    CK_TILE_HOST_DEVICE void init_raw() { bottom_tensor_view_.init_raw(); }

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    //
    WindowLengths window_lengths_;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;

    // Tile tensor distribution, which contains:
    //   1. adaptor for window: [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register: [y0, y1, ...] ==> [d]
    TileDstr tile_dstr_;

    // this contains:
    array<BottomTensorCoord, traits::NumAccess_NonLinear> cached_coords_;
    array<bool, traits::NumAccess> cached_flags_;
};

#undef WINDOW_DISPATCH_ISSUE

namespace impl {
template <address_space_enum, index_t len_>
struct default_linear_bottom_dims_impl
{
    using type = typename uniform_sequence_gen<len_, 0>::type;
};

template <index_t len_>
struct default_linear_bottom_dims_impl<address_space_enum::global, len_>
{
    // global default to seq<0,0,....1>
    using type = typename sequence_merge<typename uniform_sequence_gen<len_ - 1, 0>::type,
                                         sequence<1>>::type;
};

template <index_t len_>
struct default_linear_bottom_dims_impl<address_space_enum::lds, len_>
{
    // lds default to seq<1,1.....1>
    using type = typename uniform_sequence_gen<len_, 1>::type;
};
} // namespace impl

template <typename TensorView_>
using default_linear_bottom_dims =
    typename impl::default_linear_bottom_dims_impl<TensorView_::buffer_view::get_address_space(),
                                                   TensorView_::get_num_of_dimension()>::type;

// if using this API, will create a tile_window_linear
// this structure can have the chance to use immediate value, save register
// need pass in LinearBottomDims_ properly to control which dim is linear
// so to generate a constexpr offset as linear_offset for this dim
// (and finally pass to the immediate offset of buffer/lds instruction)
//
// Note: there is no internal check for which dim is OK to use linear offset
// user must make sure by themselves
//
// e.g.
// 2d global matrix, set LinearBottomDims_=seq<0, 1>, the last dim will generate
// immediate offset if each thread has multiple issue along last dim
//
// 2d LDS buffer, set LinearBottomDims_=seq<1, 1>, then only one vgpr used as offset
// everything else is just using immediate offset.
//
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename LinearBottomDims_ = default_linear_bottom_dims<TensorView_>>
CK_TILE_DEVICE constexpr auto
make_tile_window_linear(const TensorView_& tensor_view,
                        const WindowLengths_& window_lengths,
                        const multi_index<TensorView_::get_num_of_dimension()>& origin,
                        const StaticTileDistribution_& tile_distribution,
                        LinearBottomDims_ = {})
{
    static_assert(LinearBottomDims_::size() == TensorView_::get_num_of_dimension());
    return tile_window_linear<remove_cvref_t<TensorView_>,
                              remove_cvref_t<WindowLengths_>,
                              remove_cvref_t<StaticTileDistribution_>,
                              remove_cvref_t<LinearBottomDims_>>{
        tensor_view, window_lengths, origin, tile_distribution};
}

template <
    typename TileWindow_,
    typename StaticTileDistribution_,
    typename LinearBottomDims_ = default_linear_bottom_dims<typename TileWindow_::BottomTensorView>>
CK_TILE_DEVICE constexpr auto
make_tile_window_linear(const TileWindow_& tile_window,
                        const StaticTileDistribution_& tile_distribution,
                        LinearBottomDims_ = {})
{
    return make_tile_window_linear(tile_window.get_bottom_tensor_view(),
                                   tile_window.get_window_lengths(),
                                   tile_window.get_window_origin(),
                                   tile_distribution,
                                   LinearBottomDims_{});
}

// this version must not be called under a constexpr context
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename LinearBottomDims_ = default_linear_bottom_dims<TensorView_>>
CK_TILE_DEVICE auto
make_tile_window_linear_raw(const TensorView_& tensor_view,
                            const WindowLengths_& window_lengths,
                            const multi_index<TensorView_::get_num_of_dimension()>& origin,
                            const StaticTileDistribution_& tile_distribution,
                            LinearBottomDims_ = {})
{
    static_assert(LinearBottomDims_::size() == TensorView_::get_num_of_dimension());
    auto w = tile_window_linear<remove_cvref_t<TensorView_>,
                                remove_cvref_t<WindowLengths_>,
                                remove_cvref_t<StaticTileDistribution_>,
                                remove_cvref_t<LinearBottomDims_>>{
        tensor_view, window_lengths, origin, tile_distribution};
    w.init_raw();
    return w;
}

template <
    typename TileWindow_,
    typename StaticTileDistribution_,
    typename LinearBottomDims_ = default_linear_bottom_dims<typename TileWindow_::BottomTensorView>>
CK_TILE_DEVICE constexpr auto
make_tile_window_linear_raw(const TileWindow_& tile_window,
                            const StaticTileDistribution_& tile_distribution,
                            LinearBottomDims_ = {})
{
    return make_tile_window_linear_raw(tile_window.get_bottom_tensor_view(),
                                       tile_window.get_window_lengths(),
                                       tile_window.get_window_origin(),
                                       tile_distribution,
                                       LinearBottomDims_{});
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename LinearBottomDims_>
CK_TILE_DEVICE void move_tile_window(
    tile_window_linear<TensorView_, WindowLengths_, StaticTileDistribution_, LinearBottomDims_>&
        window,
    const typename tile_window_linear<TensorView_,
                                      WindowLengths_,
                                      StaticTileDistribution_,
                                      LinearBottomDims_>::BottomTensorIndex& step)
{
    window.move(step);
}

} // namespace ck_tile
