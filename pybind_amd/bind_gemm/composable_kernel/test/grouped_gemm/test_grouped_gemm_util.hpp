// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/number.hpp"
#include "profiler/profile_grouped_gemm_impl.hpp"

namespace ck {
namespace test {

template <typename Range>
std::string serialize_range(const Range& range)
{
    std::stringstream ss;
    for(auto& r : range)
    {
        ss << r << ", ";
    }
    std::string str = ss.str();
    return std::string(str.begin(), str.end() - 2);
}

template <typename Tuple>
class TestGroupedGemm : public testing::Test
{
    protected:
    using ALayout   = std::tuple_element_t<0, Tuple>;
    using BLayout   = std::tuple_element_t<1, Tuple>;
    using ELayout   = std::tuple_element_t<2, Tuple>;
    using ADataType = std::tuple_element_t<3, Tuple>;
    using BDataType = std::tuple_element_t<4, Tuple>;
    using EDataType = std::tuple_element_t<5, Tuple>;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1; // integer value initialization
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false; // measure kernel performance
    static constexpr int n_warmup_    = 0;
    static constexpr int n_iter_      = 1;
    std::vector<int> k_batches_;

    void SetUp() override { k_batches_ = {1, 2, 3, 5, 8}; }

    private:
    template <typename Layout>
    void SetStrides(std::vector<int>& strides,
                    const std::vector<int>& rows,
                    const std::vector<int>& cols) const
    {
        if(std::is_same_v<Layout, Row>)
        {
            for(const auto c : cols)
            {
                strides.emplace_back(c);
            }
        }
        else if(std::is_same_v<Layout, Col>)
        {
            for(const auto r : rows)
            {
                strides.emplace_back(r);
            }
        }
    }

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             const std::vector<int>& StrideAs = {},
             const std::vector<int>& StrideBs = {},
             const std::vector<int>& StrideCs = {})
    {
        std::vector<int> stride_as = StrideAs;
        std::vector<int> stride_bs = StrideBs;
        std::vector<int> stride_cs = StrideCs;

        if(stride_as.empty())
        {
            SetStrides<ALayout>(stride_as, Ms, Ks);
        }
        if(stride_bs.empty())
        {
            SetStrides<BLayout>(stride_bs, Ks, Ns);
        }
        if(stride_cs.empty())
        {
            SetStrides<ELayout>(stride_cs, Ms, Ns);
        }

        RunSingle(Ms, Ns, Ks, stride_as, stride_bs, stride_cs, k_batches_);
    }

    void RunSingle(const std::vector<int>& Ms,
                   const std::vector<int>& Ns,
                   const std::vector<int>& Ks,
                   const std::vector<int>& StrideAs,
                   const std::vector<int>& StrideBs,
                   const std::vector<int>& StrideCs,
                   const std::vector<int>& kbatches)
    {
        bool pass = ck::profiler::profile_grouped_gemm_impl<ADataType,
                                                            BDataType,
                                                            EDataType,
                                                            float,
                                                            ALayout,
                                                            BLayout,
                                                            ELayout>(verify_,
                                                                     init_method_,
                                                                     log_,
                                                                     bench_,
                                                                     Ms,
                                                                     Ns,
                                                                     Ks,
                                                                     StrideAs,
                                                                     StrideBs,
                                                                     StrideCs,
                                                                     kbatches,
                                                                     n_warmup_,
                                                                     n_iter_);
        EXPECT_TRUE(pass);
    }
};

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          tensor_operation::device::GemmSpecialization GemmSpec,
          ck::index_t KPerBlock,
          ck::index_t K1,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferSrcScalarPerVector,
          index_t CDEBlockTransferScalarPerVector_NPerBlock>
struct DeviceGroupedGemmSplitkInstanceWrapper
{
    using F16         = half_t;
    using F32         = float;
    using Row         = ck::tensor_layout::gemm::RowMajor;
    using Col         = ck::tensor_layout::gemm::ColumnMajor;
    using PassThrough = tensor_operation::element_wise::PassThrough;

    using EmptyTuple = ck::Tuple<>;

    template <ck::index_t... Is>
    using S = ck::Sequence<Is...>;

    template <ck::index_t N>
    using I = ck::Number<N>;

    using ABlockTransferThreadClusterArrageOrder =
        std::conditional_t<std::is_same_v<ALayout, Row>, S<0, 2, 1, 3>, S<0, 1, 3, 2>>;
    using ABlockTransferSrcAccessOrder =
        std::conditional_t<std::is_same_v<ALayout, Row>, S<0, 2, 1, 3>, S<0, 1, 3, 2>>;
    using ABlockTransferSrcVectorDim = std::conditional_t<std::is_same_v<ALayout, Row>, I<3>, I<2>>;
    using ABlockTransferDstScalarPerVector_K1 =
        std::conditional_t<std::is_same_v<ALayout, Row>, I<8>, I<2>>;
    using ABlockLdsAddExtraM = std::conditional_t<std::is_same_v<ALayout, Row>, I<1>, I<0>>;

    using BBlockTransferThreadClusterArrageOrder =
        std::conditional_t<std::is_same_v<BLayout, Row>, S<0, 1, 3, 2>, S<0, 2, 1, 3>>;
    using BBlockTransferSrcAccessOrder =
        std::conditional_t<std::is_same_v<BLayout, Row>, S<0, 1, 3, 2>, S<0, 2, 1, 3>>;
    using BBlockTransferSrcVectorDim = std::conditional_t<std::is_same_v<BLayout, Row>, I<2>, I<3>>;
    using BBlockTransferDstScalarPerVector_K1 =
        std::conditional_t<std::is_same_v<ALayout, Row>, I<2>, I<8>>;
    using BBlockLdsAddExtraM = std::conditional_t<std::is_same_v<ALayout, Row>, I<0>, I<1>>;

    using DeviceGroupedGemmSplitKInstance =
        tensor_operation::device::DeviceGroupedGemmXdlSplitKCShuffle<
            ALayout,
            BLayout,
            EmptyTuple,
            ELayout,
            F16,
            F16,
            F32,
            F16,
            EmptyTuple,
            F16,
            PassThrough,
            PassThrough,
            PassThrough,
            GemmSpec,
            1,
            128,
            128,
            128,
            KPerBlock,
            K1,
            K1,
            32,
            32,
            4,
            2,
            S<1, 4, 16, 1>,
            ABlockTransferThreadClusterArrageOrder,
            ABlockTransferSrcAccessOrder,
            ABlockTransferSrcVectorDim::value,
            ABlockTransferSrcScalarPerVector,
            ABlockTransferDstScalarPerVector_K1::value,
            ABlockLdsAddExtraM::value,
            S<1, 4, 16, 1>,
            BBlockTransferThreadClusterArrageOrder,
            BBlockTransferSrcAccessOrder,
            BBlockTransferSrcVectorDim::value,
            BBlockTransferSrcScalarPerVector,
            BBlockTransferDstScalarPerVector_K1::value,
            BBlockLdsAddExtraM::value,
            1,
            1,
            S<1, 16, 1, 8>,
            CDEBlockTransferScalarPerVector_NPerBlock>;

    bool IsSupported(const std::vector<int>& Ms,
                     const std::vector<int>& Ns,
                     const std::vector<int>& Ks,
                     const std::vector<int>& StrideAs,
                     const std::vector<int>& StrideBs,
                     const std::vector<int>& StrideCs,
                     int kbatch = 1) const
    {
        std::size_t n_groups = Ms.size();
        EXPECT_TRUE(Ns.size() == n_groups && Ks.size() == n_groups && StrideAs.size() == n_groups &&
                    StrideBs.size() == n_groups && StrideCs.size() == n_groups)
            << "The number of groups is not consistent!";

        std::vector<tensor_operation::device::GemmDesc> gemm_descs;

        for(std::size_t i = 0; i < n_groups; ++i)
        {
            gemm_descs.push_back(tensor_operation::device::GemmDesc{
                Ms[i], Ns[i], Ks[i], StrideAs[i], StrideBs[i], StrideCs[i], {}});
        }

        std::vector<const void*> p_As(n_groups, nullptr);
        std::vector<const void*> p_Bs(n_groups, nullptr);
        std::vector<void*> p_Cs(n_groups, nullptr);
        auto p_Ds = std::vector<std::array<const void*, 0>>{};

        auto ggemm_instance = DeviceGroupedGemmSplitKInstance{};
        auto argument       = ggemm_instance.MakeArgument(
            p_As, p_Bs, p_Ds, p_Cs, gemm_descs, PassThrough{}, PassThrough{}, PassThrough{});
        if(kbatch > 1)
        {
            ggemm_instance.SetKBatchSize(&argument, kbatch);
        }

        return ggemm_instance.IsSupportedArgument(argument);
    }

    float Run(const std::vector<int>& Ms,
              const std::vector<int>& Ns,
              const std::vector<int>& Ks,
              const std::vector<int>& StrideAs,
              const std::vector<int>& StrideBs,
              const std::vector<int>& StrideCs,
              int kbatch = 1) const
    {
        std::size_t n_groups = Ms.size();
        EXPECT_TRUE(Ns.size() == n_groups && Ks.size() == n_groups && StrideAs.size() == n_groups &&
                    StrideBs.size() == n_groups && StrideCs.size() == n_groups)
            << "The number of groups is not consistent!";

        std::vector<tensor_operation::device::GemmDesc> gemm_descs;

        for(std::size_t i = 0; i < n_groups; ++i)
        {
            gemm_descs.push_back(tensor_operation::device::GemmDesc{
                Ms[i], Ns[i], Ks[i], StrideAs[i], StrideBs[i], StrideCs[i], {}});
        }

        std::vector<const void*> p_As(n_groups, nullptr);
        std::vector<const void*> p_Bs(n_groups, nullptr);
        std::vector<void*> p_Cs(n_groups, nullptr);
        auto p_Ds = std::vector<std::array<const void*, 0>>{};

        auto ggemm_instance = DeviceGroupedGemmSplitKInstance{};
        auto argument       = ggemm_instance.MakeArgument(
            p_As, p_Bs, p_Ds, p_Cs, gemm_descs, PassThrough{}, PassThrough{}, PassThrough{});
        if(kbatch > 1)
        {
            ggemm_instance.SetKBatchSize(&argument, kbatch);
        }

        EXPECT_TRUE(ggemm_instance.IsSupportedArgument(argument));
        auto invoker = ggemm_instance.MakeInvoker();
        DeviceMem dev_gemm_kargs(ggemm_instance.GetDeviceKernelArgSize(&argument));
        ggemm_instance.SetDeviceKernelArgs(&argument, dev_gemm_kargs.GetDeviceBuffer());
        return invoker.Run(argument, StreamConfig{nullptr, false});
    }
};

} // namespace test
} // namespace ck
