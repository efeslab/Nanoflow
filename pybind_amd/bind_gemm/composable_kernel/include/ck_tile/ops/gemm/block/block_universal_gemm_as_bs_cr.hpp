// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

// A is block window on shared memory
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmASmemBSmemCRegV1DefaultPolicy>
struct BlockUniversalGemmAsBsCr
{
    private:
    // TODO: This should be in Policy - UniversalGemmPolicyBase ?
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem        = remove_cvref_t<PipelineProblem_>;
        using Policy         = remove_cvref_t<GemmPolicy_>;
        using ADataType      = remove_cvref_t<typename Problem::ADataType>;
        using BDataType      = remove_cvref_t<typename Problem::BDataType>;
        using CDataType      = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr auto Scheduler     = Problem::Scheduler;

        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WarpGemm = remove_cvref_t<decltype(config.template at<0>())>;

        static constexpr index_t MWarp = config.template at<1>();
        static constexpr index_t NWarp = config.template at<2>();

        using I0 = number<0>;
        using I1 = number<1>;

        static_assert(MWarp == BlockGemmShape::BlockWarps::at(I0{}),
                      "Error! WarpGemm's MWarp is not consisten with BlockGemmShape!");
        static_assert(NWarp == BlockGemmShape::BlockWarps::at(I1{}),
                      "Error! WarpGemm's NWarp is not consisten with BlockGemmShape!");
        static_assert(WarpGemm::kM == BlockGemmShape::WarpTile::at(I0{}),
                      "Error! WarpGemm's M is not consisten with BlockGemmShape!");
        static_assert(WarpGemm::kN == BlockGemmShape::WarpTile::at(I1{}),
                      "Error! WarpGemm's N is not consisten with BlockGemmShape!");

        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        static_assert(MIterPerWarp * MWarp * WarpGemm::kM == MPerBlock,
                      "Error! Warps should cover all Block tile!");
        static_assert(NIterPerWarp * NWarp * WarpGemm::kN == NPerBlock,
                      "Error! Warps should cover all Block tile!");

        static constexpr index_t MPerBlockPerIter = MWarp * WarpGemm::kM;
        static constexpr index_t NPerBlockPerIter = NWarp * WarpGemm::kN;
        static constexpr index_t KPerBlockPerIter = WarpGemm::kK;

        using AWarpTileDistr = remove_cvref_t<decltype(make_static_tile_distribution(
            typename WarpGemm::AWarpDstrEncoding{}))>;
        using BWarpTileDistr = remove_cvref_t<decltype(make_static_tile_distribution(
            typename WarpGemm::BWarpDstrEncoding{}))>;

        using AWarpTile =
            remove_cvref_t<decltype(make_static_distributed_tensor<ADataType>(AWarpTileDistr{}))>;
        using BWarpTile =
            remove_cvref_t<decltype(make_static_distributed_tensor<BDataType>(BWarpTileDistr{}))>;

        // TODO: Should we have two policies? Interwave & Intrawave ??
        static constexpr index_t InterWaveSchedulingMacClusters = 1;

        static constexpr index_t KPack      = WarpGemm::kKPerThread;
        static constexpr index_t KPerThread = KIterPerWarp * KPack;
        static constexpr index_t KRepeat    = KPerThread / KPack;
    };

    public:
    using Traits = GemmTraits_<Problem_, Policy_>;

    using ADataType = remove_cvref_t<typename Traits::ADataType>;
    using BDataType = remove_cvref_t<typename Traits::BDataType>;
    using CDataType = remove_cvref_t<typename Traits::CDataType>;

    using WarpGemm = remove_cvref_t<typename Traits::WarpGemm>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr index_t MWarp = Traits::MWarp;
    static constexpr index_t NWarp = Traits::NWarp;

    static constexpr auto Scheduler = Traits::Scheduler;

    using I0 = number<0>;
    using I1 = number<1>;

    private:
    template <GemmPipelineScheduler Scheduler, typename GemmTraits>
    struct BlockGemmImpl
    {
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Default, GemmTraits>
    {
        // C += A * B
        template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ASmemBlockWindow& a_block_window,
                                       const BSmemBlockWindow& b_block_window)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");
            static_assert(std::is_same_v<ADataType, typename ASmemBlockWindow::DataType> &&
                              std::is_same_v<BDataType, typename BSmemBlockWindow::DataType>,
                          "The ADataType and BDataType as defined in "
                          "traits should be the same as correspoinding block window data type!");

            static_assert(
                GemmTraits::MPerBlock == ASmemBlockWindow{}.get_window_lengths()[I0{}] &&
                    GemmTraits::NPerBlock == BSmemBlockWindow{}.get_window_lengths()[I0{}] &&
                    GemmTraits::KPerBlock == ASmemBlockWindow{}.get_window_lengths()[I1{}],
                "MPerBlock, NPerBlock, KPerBlock defined in "
                " BlockGemmShape are different from A/B block smem windows apropriate dims!");

            const index_t iMWarp = get_warp_id() / NWarp;
            const index_t iNWarp = get_warp_id() - (iMWarp * NWarp);

            // TODO: refactor warp_window tile type to class member as it should be
            // compile-time known information.
            auto a_warp_window_tmp = make_tile_window(
                a_block_window.get_bottom_tensor_view(),
                make_tuple(number<WarpGemm::kM>{}, number<WarpGemm::kK>{}),
                a_block_window.get_window_origin() + multi_index<2>{iMWarp * WarpGemm::kM, 0},
                make_static_tile_distribution(typename WarpGemm::AWarpDstrEncoding{}));

            using AWarpWindow = remove_cvref_t<decltype(a_warp_window_tmp)>;

            static_assert(GemmTraits::AWarpTile::get_num_of_dimension() ==
                              AWarpWindow::get_num_of_dimension(),
                          "AWarpWindow number of dimensions must be equal to "
                          "AWarpTile number of dimensions!");
            static_assert(GemmTraits::AWarpTile::get_lengths() ==
                              AWarpWindow{}.get_window_lengths(),
                          "AWarpWindow lengths must be equal to AWarpTile lengths!");

            statically_indexed_array<
                statically_indexed_array<AWarpWindow, GemmTraits::KIterPerWarp>,
                MIterPerWarp>
                a_warp_windows;

            // construct B-warp-window
            auto b_warp_window_tmp = make_tile_window(
                b_block_window.get_bottom_tensor_view(),
                make_tuple(number<WarpGemm::kN>{}, number<WarpGemm::kK>{}),
                b_block_window.get_window_origin() + multi_index<2>{iNWarp * WarpGemm::kN, 0},
                make_static_tile_distribution(typename WarpGemm::BWarpDstrEncoding{}));

            using BWarpWindow = remove_cvref_t<decltype(b_warp_window_tmp)>;

            static_assert(GemmTraits::BWarpTile::get_num_of_dimension() ==
                              BWarpWindow::get_num_of_dimension(),
                          "BWarpWindow number of dimensions must be equal to "
                          "BWarpTile number of dimensions!");
            static_assert(GemmTraits::BWarpTile::get_lengths() ==
                              BWarpWindow{}.get_window_lengths(),
                          "BWarpWindow lengths must be equal to BWarpTile lengths!");

            statically_indexed_array<
                statically_indexed_array<BWarpWindow, GemmTraits::KIterPerWarp>,
                NIterPerWarp>
                b_warp_windows;

            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, GemmTraits::KIterPerWarp, 1>{}([&](auto kIter) {
                    a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                    // TODO: I don't have to move 0,0 window!
                    move_tile_window(a_warp_windows(mIter)(kIter),
                                     {mIter * GemmTraits::MPerBlockPerIter,
                                      kIter * GemmTraits::KPerBlockPerIter});
                });
            });

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, GemmTraits::KIterPerWarp, 1>{}([&](auto kIter) {
                    b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                    move_tile_window(b_warp_windows(nIter)(kIter),
                                     {nIter * GemmTraits::NPerBlockPerIter,
                                      kIter * GemmTraits::KPerBlockPerIter});
                });
            });

            using CWarpDstr   = typename WarpGemm::CWarpDstr;
            using CWarpTensor = typename WarpGemm::CWarpTensor;

            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            // hot loop:
            static_for<0, GemmTraits::KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    const auto a_warp_tile = load_tile(a_warp_windows(mIter)(kIter));

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        const auto b_warp_tile = load_tile(b_warp_windows(nIter)(kIter));

                        // read C warp tensor from C block tensor-
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WarpGemm{}(c_warp_tensor, a_warp_tile, b_warp_tile);

                        // write C warp tensor into C block tensor
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        }
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Intrawave, GemmTraits>
    {
        statically_indexed_array<
            statically_indexed_array<typename GemmTraits::AWarpTile, KIterPerWarp>,
            MIterPerWarp>
            a_warp_tiles_;

        statically_indexed_array<
            statically_indexed_array<typename GemmTraits::BWarpTile, KIterPerWarp>,
            NIterPerWarp>
            b_warp_tiles_;

        template <typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window)
        {
            static_assert(
                GemmTraits::MPerBlock == ASmemBlockWindow{}.get_window_lengths()[I0{}] &&
                    GemmTraits::NPerBlock == BSmemBlockWindow{}.get_window_lengths()[I0{}] &&
                    GemmTraits::KPerBlock == ASmemBlockWindow{}.get_window_lengths()[I1{}],
                "MPerBlock, NPerBlock, KPerBlock defined in "
                " BlockGemmShape are different from A/B block smem windows apropriate dims!");

            static_assert(std::is_same_v<ADataType, typename ASmemBlockWindow::DataType> &&
                              std::is_same_v<BDataType, typename BSmemBlockWindow::DataType>,
                          "The ADataType and BDataType as defined in "
                          "traits should be the same as correspoinding block window data type!");

            const index_t iMWarp = get_warp_id() / NWarp;
            const index_t iNWarp = get_warp_id() - (iMWarp * NWarp);

            // TODO: refactor warp_window tile type to class member as it should be
            // compile-time known information.
            auto a_warp_window_tmp = make_tile_window(
                a_block_window.get_bottom_tensor_view(),
                make_tuple(number<WarpGemm::kM>{}, number<WarpGemm::kK>{}),
                a_block_window.get_window_origin() + multi_index<2>{iMWarp * WarpGemm::kM, 0},
                make_static_tile_distribution(typename WarpGemm::AWarpDstrEncoding{}));

            using AWarpWindow = remove_cvref_t<decltype(a_warp_window_tmp)>;

            static_assert(GemmTraits::AWarpTile::get_num_of_dimension() ==
                              AWarpWindow::get_num_of_dimension(),
                          "AWarpWindow number of dimensions must be equal to "
                          "AWarpTile number of dimensions!");
            static_assert(GemmTraits::AWarpTile::get_lengths() ==
                              AWarpWindow{}.get_window_lengths(),
                          "AWarpWindow lengths must be equal to AWarpTile lengths!");

            statically_indexed_array<statically_indexed_array<AWarpWindow, KIterPerWarp>,
                                     MIterPerWarp>
                a_warp_windows;

            // construct B-warp-window
            auto b_warp_window_tmp = make_tile_window(
                b_block_window.get_bottom_tensor_view(),
                make_tuple(number<WarpGemm::kN>{}, number<WarpGemm::kK>{}),
                b_block_window.get_window_origin() + multi_index<2>{iNWarp * WarpGemm::kN, 0},
                make_static_tile_distribution(typename WarpGemm::BWarpDstrEncoding{}));

            using BWarpWindow = remove_cvref_t<decltype(b_warp_window_tmp)>;

            static_assert(GemmTraits::BWarpTile::get_num_of_dimension() ==
                              BWarpWindow::get_num_of_dimension(),
                          "BWarpWindow number of dimensions must be equal to "
                          "BWarpTile number of dimensions!");
            static_assert(GemmTraits::BWarpTile::get_lengths() ==
                              BWarpWindow{}.get_window_lengths(),
                          "BWarpWindow lengths must be equal to BWarpTile lengths!");

            statically_indexed_array<statically_indexed_array<BWarpWindow, KIterPerWarp>,
                                     NIterPerWarp>
                b_warp_windows;

            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                    // TODO: I don't have to move 0,0 window!
                    move_tile_window(a_warp_windows(mIter)(kIter),
                                     {mIter * GemmTraits::MPerBlockPerIter,
                                      kIter * GemmTraits::KPerBlockPerIter});
                });
            });

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                    move_tile_window(b_warp_windows(nIter)(kIter),
                                     {nIter * GemmTraits::NPerBlockPerIter,
                                      kIter * GemmTraits::KPerBlockPerIter});
                });
            });

            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block window
                    load_tile(a_warp_tiles_(mIter)(kIter), a_warp_windows(mIter)(kIter));
                });
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    load_tile(b_warp_tiles_(nIter)(kIter), b_warp_windows(nIter)(kIter));
                });
            });
        }

        // C += A * B
        template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       [[maybe_unused]] const ASmemBlockWindow& a_block_window,
                                       [[maybe_unused]] const BSmemBlockWindow& b_block_window)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            using CWarpDstr   = typename WarpGemm::CWarpDstr;
            using CWarpTensor = typename WarpGemm::CWarpTensor;

            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            // hot loop:
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor-
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WarpGemm{}(c_warp_tensor,
                                   a_warp_tiles_[mIter][kIter],
                                   b_warp_tiles_[nIter][kIter]);

                        // write C warp tensor into C block tensor
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        }
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Interwave, GemmTraits>
    {
        static constexpr index_t KPerThread     = GemmTraits::KPerThread;
        static constexpr index_t NumMacClusters = GemmTraits::InterWaveSchedulingMacClusters;
        static constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, GemmTraits::KPack);
        // TODO: do we really need this?? Are there any cases when this would be >=1 ??
        // Would we need InterWaveSchedulingMacClusters > 1 ???
        static constexpr index_t KRepeat        = KPerThread / KPerInnerLoop;
        static constexpr index_t KInnerLoopIter = KPerInnerLoop / GemmTraits::KPack;

        statically_indexed_array<
            statically_indexed_array<typename GemmTraits::AWarpTile, KInnerLoopIter>,
            MIterPerWarp>
            a_warp_tiles_;

        statically_indexed_array<
            statically_indexed_array<typename GemmTraits::BWarpTile, KInnerLoopIter>,
            NIterPerWarp>
            b_warp_tiles_;

        template <index_t KIdx, typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window)
        {
            static_assert(
                GemmTraits::MPerBlock == ASmemBlockWindow{}.get_window_lengths()[I0{}] &&
                    GemmTraits::NPerBlock == BSmemBlockWindow{}.get_window_lengths()[I0{}] &&
                    GemmTraits::KPerBlock == ASmemBlockWindow{}.get_window_lengths()[I1{}],
                "MPerBlock, NPerBlock, KPerBlock defined in "
                " BlockGemmShape are different from A/B block smem windows apropriate dims!");

            static_assert(std::is_same_v<ADataType, typename ASmemBlockWindow::DataType> &&
                              std::is_same_v<BDataType, typename BSmemBlockWindow::DataType>,
                          "The ADataType and BDataType as defined in "
                          "traits should be the same as correspoinding block window data type!");

            const index_t iMWarp = get_warp_id() / NWarp;
            const index_t iNWarp = get_warp_id() - (iMWarp * NWarp);

            // TODO: refactor warp_window tile type to class member as it should be
            // compile-time known information.
            auto a_warp_window_tmp = make_tile_window(
                a_block_window.get_bottom_tensor_view(),
                make_tuple(number<WarpGemm::kM>{}, number<WarpGemm::kK>{}),
                a_block_window.get_window_origin() +
                    multi_index<2>{iMWarp * WarpGemm::kM, KIdx * KPerInnerLoop},
                make_static_tile_distribution(typename WarpGemm::AWarpDstrEncoding{}));

            using AWarpWindow = remove_cvref_t<decltype(a_warp_window_tmp)>;

            static_assert(GemmTraits::AWarpTile::get_num_of_dimension() ==
                              AWarpWindow::get_num_of_dimension(),
                          "AWarpWindow number of dimensions must be equal to "
                          "AWarpTile number of dimensions!");
            static_assert(GemmTraits::AWarpTile::get_lengths() ==
                              AWarpWindow{}.get_window_lengths(),
                          "AWarpWindow lengths must be equal to AWarpTile lengths!");

            statically_indexed_array<statically_indexed_array<AWarpWindow, KInnerLoopIter>,
                                     MIterPerWarp>
                a_warp_windows;

            // construct B-warp-window
            auto b_warp_window_tmp = make_tile_window(
                b_block_window.get_bottom_tensor_view(),
                make_tuple(number<WarpGemm::kN>{}, number<WarpGemm::kK>{}),
                b_block_window.get_window_origin() +
                    multi_index<2>{iNWarp * WarpGemm::kN, KIdx * KPerInnerLoop},
                make_static_tile_distribution(typename WarpGemm::BWarpDstrEncoding{}));

            using BWarpWindow = remove_cvref_t<decltype(b_warp_window_tmp)>;

            static_assert(GemmTraits::BWarpTile::get_num_of_dimension() ==
                              BWarpWindow::get_num_of_dimension(),
                          "BWarpWindow number of dimensions must be equal to "
                          "BWarpTile number of dimensions!");
            static_assert(GemmTraits::BWarpTile::get_lengths() ==
                              BWarpWindow{}.get_window_lengths(),
                          "BWarpWindow lengths must be equal to BWarpTile lengths!");

            statically_indexed_array<statically_indexed_array<BWarpWindow, KInnerLoopIter>,
                                     NIterPerWarp>
                b_warp_windows;

            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, KInnerLoopIter, 1>{}([&](auto kIter) {
                    a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                    move_tile_window(a_warp_windows(mIter)(kIter),
                                     {mIter * GemmTraits::MPerBlockPerIter,
                                      kIter * GemmTraits::KPerBlockPerIter});
                });
            });

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, KInnerLoopIter, 1>{}([&](auto kIter) {
                    b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                    move_tile_window(b_warp_windows(nIter)(kIter),
                                     {nIter * GemmTraits::NPerBlockPerIter,
                                      kIter * GemmTraits::KPerBlockPerIter});
                });
            });

            // TODO check if a_warp_tiles has same desc as a_warp_window
            static_for<0, KInnerLoopIter, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block window
                    load_tile(a_warp_tiles_(mIter)(kIter), a_warp_windows(mIter)(kIter));
                });
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    load_tile(b_warp_tiles_(nIter)(kIter), b_warp_windows(nIter)(kIter));
                });
            });
        }

        // C += A * B
        template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ASmemBlockWindow& a_block_window,
                                       const BSmemBlockWindow& b_block_window)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            using CWarpDstr   = typename WarpGemm::CWarpDstr;
            using CWarpTensor = typename WarpGemm::CWarpTensor;

            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            // hot loop:
            static_for<0, KRepeat, 1>{}([&](auto kIter) {
                LocalPrefetch<kIter.value>(a_block_window, b_block_window);
                __builtin_amdgcn_sched_barrier(0);
                // NOTE: Synchronize threads in a workgroup at the start of each MAC
                // cluster, but except the first, as we can shorten non-MAC cluster a bit
                // and there's no observable negative impact. The desired effect is waves in
                // a workgroup executing MAC in sync. This avoids some out-of-sync waves
                // hijacking MAC resource from other workgroups and reducing the chance of
                // latency hiding by waiting for the rest of the workgroup at the eventual
                // sync point.
                if constexpr(kIter.value != 0 || KRepeat == 1)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                }

                static_for<0, KInnerLoopIter, 1>{}([&](auto kInnerIter) {
                    static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                            // read C warp tensor from C block tensor-
                            CWarpTensor c_warp_tensor;

                            c_warp_tensor.get_thread_buffer() =
                                c_block_tensor.get_y_sliced_thread_data(
                                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                            // The block_sync_lds() here performs double duty:
                            // A) safeguard against data hazard because barrier from
                            // blockwise_gemm is moved here B) reduce VMEM FIFO congestion
                            // by applying small delays to different wavefronts It is
                            // performed near the end of MAC cluster to minimize lgkmcnt
                            // penalty
                            if constexpr(kIter.value == KRepeat - 1 &&
                                         kInnerIter.value == KInnerLoopIter - 1 &&
                                         mIter.value == MIterPerWarp - 1 &&
                                         nIter.value == NIterPerWarp - 1)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                block_sync_lds();
                                __builtin_amdgcn_sched_barrier(0);
                            }
                            // warp GEMM
                            WarpGemm{}(c_warp_tensor,
                                       a_warp_tiles_[mIter][kInnerIter],
                                       b_warp_tiles_[nIter][kInnerIter]);

                            // write C warp tensor into C block tensor
                            c_block_tensor.set_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                c_warp_tensor.get_thread_buffer());

                            if constexpr(kInnerIter.value == 0 && mIter.value == 0 &&
                                         nIter.value == 0)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                __builtin_amdgcn_s_setprio(1);
                                __builtin_amdgcn_sched_barrier(0);
                            }
                        });
                    });
                });

                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    };

    public:
    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);

        return c_block_tensor;
    }

    template <typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                      const BSmemBlockWindow& b_block_window)
    {
        block_gemm_impl_.LocalPrefetch(a_block_window, b_block_window);
    }

    // C += A * B
    template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window)
    {
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_window);
    }

    // C = A * B
    template <typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE auto operator()(const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window)
    {
        auto c_block_tensor = MakeCBlockTile();
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_window);
        return c_block_tensor;
    }

    private:
    BlockGemmImpl<Scheduler, Traits> block_gemm_impl_{};
};

} // namespace ck_tile
