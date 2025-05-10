// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <iostream>
#include <vector>
#include <sstream>

#include "device_grouped_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

///
/// @brief      Structure representing single GEMM problem arguments.
///
///             The pointer to the vector of those structures is passed to the GroupedGEMM entry
///             point kernel.
///
/// @tparam     NumDTensor  The number of D input tensors.
///
template <index_t NumDTensor = 0>
struct GroupedGemmMultipleDKernelArguments
{
    __host__ __device__
    GroupedGemmMultipleDKernelArguments(const void* p_a_grid_,
                                        const void* p_b_grid_,
                                        std::array<const void*, NumDTensor> p_ds_grid_,
                                        void* p_e_grid_,
                                        index_t M_,
                                        index_t N_,
                                        index_t K_,
                                        index_t StrideA_,
                                        index_t StrideB_,
                                        std::array<index_t, NumDTensor> StrideDs_,
                                        index_t StrideE_)
        : p_a_grid{p_a_grid_},
          p_b_grid{p_b_grid_},
          p_ds_grid{p_ds_grid_},
          p_e_grid{p_e_grid_},
          M{M_},
          N{N_},
          K{K_},
          StrideA{StrideA_},
          StrideB{StrideB_},
          StrideDs{StrideDs_},
          StrideE{StrideE_}
    {
    }

    const void* p_a_grid;
    const void* p_b_grid;
    std::array<const void*, NumDTensor> p_ds_grid;
    void* p_e_grid;
    index_t M;
    index_t N;
    index_t K;
    index_t StrideA;
    index_t StrideB;
    std::array<index_t, NumDTensor> StrideDs;
    index_t StrideE;

    void Print() const
    {
        std::stringstream str;
        for(auto sd : StrideDs)
            str << sd << ",";

        std::cout << "arg {"
                  << "M:" << M << ", "
                  << "N:" << N << ", "
                  << "K:" << K << ", "
                  << "SA:" << StrideA << ", "
                  << "SB:" << StrideB << ", "
                  << "SE:" << StrideE << ", "
                  << "SDs: {" << str.str() << "}"
                  << "}" << std::endl;
    }
};

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGroupedGemmMultipleDSplitK : public DeviceGroupedGemm<ALayout,
                                                                   BLayout,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   ADataType,
                                                                   BDataType,
                                                                   DsDataType,
                                                                   EDataType,
                                                                   AElementwiseOperation,
                                                                   BElementwiseOperation,
                                                                   CDEElementwiseOperation>
{
    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the k batch size.
    ///
    /// @param      p_arg   Pointer to the Argument we're going to change.
    /// @param[in]  kbatch  The kbatch value.
    ///
    virtual void SetKBatchSize(BaseArgument* p_arg, index_t kbatch) const = 0;

    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the device kernel arguments pointer.
    ///
    /// @param      p_arg              The pointer to the Argument we're going to update.
    /// @param[in]  p_dev_kernel_args  The pointer to the device memory which contains kernel
    ///                                arguments.
    ///
    virtual void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const = 0;

    //----------------------------------------------------------------------------------------------
    /// @brief      Gets the device kernel argument size.
    ///
    /// @param[in]  p_arg  The pointer to the Device op Argument.
    ///
    /// @return     The device kernel argument size.
    ///
    virtual size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
