// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

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
    DeviceMem(std::size_t mem_size);
    void Realloc(std::size_t mem_size);
    void* GetDeviceBuffer() const;
    std::size_t GetBufferSize() const;
    void ToDevice(const void* p) const;
    void ToDevice(const void* p, const std::size_t cpySize) const;
    void FromDevice(void* p) const;
    void FromDevice(void* p, const std::size_t cpySize) const;
    void SetZero() const;
    template <typename T>
    void SetValue(T x) const;
    ~DeviceMem();

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

template <typename T>
void DeviceMem::SetValue(T x) const
{
    if(mMemSize % sizeof(T) != 0)
    {
        throw std::runtime_error("wrong! not entire DeviceMem will be set");
    }

    set_buffer_value<T><<<1, 1024>>>(static_cast<T*>(mpDeviceBuf), x, mMemSize / sizeof(T));
}
