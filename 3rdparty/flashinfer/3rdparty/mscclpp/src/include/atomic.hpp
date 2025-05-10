// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ATOMIC_HPP_
#define MSCCLPP_ATOMIC_HPP_

#if defined(USE_CUDA)
#define MSCCLPP_DEVICE_CUDA
#include <mscclpp/atomic_device.hpp>
#undef MSCCLPP_DEVICE_CUDA
#else  // !defined(USE_CUDA)
#define MSCCLPP_DEVICE_HIP
#include <mscclpp/atomic_device.hpp>
#undef MSCCLPP_DEVICE_HIP
#endif  // !defined(USE_CUDA)

#endif  // MSCCLPP_ATOMIC_HPP_
