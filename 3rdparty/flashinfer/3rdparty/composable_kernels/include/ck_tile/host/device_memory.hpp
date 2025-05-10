// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdexcept>
#include "ck_tile/host/hip_check_error.hpp"

namespace ck_tile {
template <typename T>
__global__ void set_buffer_value(T* p, T x, uint64_t buffer_element_size)
{
    for(uint64_t i = threadIdx.x; i < buffer_element_size; i += blockDim.x)
    {
        p[i] = x;
    }
}

/**
 * @brief Container for storing data in GPU device memory
 *
 */
struct DeviceMem
{
    DeviceMem() : mpDeviceBuf(nullptr), mMemSize(0) {}
    DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
    {
        if(mMemSize != 0)
        {
            HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
        }
        else
        {
            mpDeviceBuf = nullptr;
        }
    }
    void Realloc(std::size_t mem_size)
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(hipFree(mpDeviceBuf));
        }
        mMemSize = mem_size;
        if(mMemSize != 0)
        {
            HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
        }
        else
        {
            mpDeviceBuf = nullptr;
        }
    }
    void* GetDeviceBuffer() const { return mpDeviceBuf; }
    std::size_t GetBufferSize() const { return mMemSize; }
    void ToDevice(const void* p) const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(
                hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
        }
        // else
        // {
        //     throw std::runtime_error("ToDevice with an empty pointer");
        // }
    }
    void ToDevice(const void* p, const std::size_t cpySize) const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(
                hipMemcpy(mpDeviceBuf, const_cast<void*>(p), cpySize, hipMemcpyHostToDevice));
        }
    }
    void FromDevice(void* p) const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
        }
        // else
        // {
        //     throw std::runtime_error("FromDevice with an empty pointer");
        // }
    }
    void FromDevice(void* p, const std::size_t cpySize) const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(hipMemcpy(p, mpDeviceBuf, cpySize, hipMemcpyDeviceToHost));
        }
    }
    void SetZero() const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(hipMemset(mpDeviceBuf, 0, mMemSize));
        }
    }
    template <typename T>
    void SetValue(T x) const
    {
        if(mpDeviceBuf)
        {
            if(mMemSize % sizeof(T) != 0)
            {
                throw std::runtime_error("wrong! not entire DeviceMem will be set");
            }

            // TODO: call a gpu kernel to set the value (?)
            set_buffer_value<T><<<1, 1024>>>(static_cast<T*>(mpDeviceBuf), x, mMemSize / sizeof(T));
        }
    }
    ~DeviceMem()
    {
        if(mpDeviceBuf)
        {
            try
            {
                HIP_CHECK_ERROR(hipFree(mpDeviceBuf));
            }
            catch(std::runtime_error& re)
            {
                std::cerr << re.what() << std::endl;
            }
        }
    }

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

} // namespace ck_tile
