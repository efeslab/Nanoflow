// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/gpu_utils.hpp>

namespace mscclpp {

AvoidCudaGraphCaptureGuard::AvoidCudaGraphCaptureGuard() : mode_(cudaStreamCaptureModeRelaxed) {
  MSCCLPP_CUDATHROW(cudaThreadExchangeStreamCaptureMode(&mode_));
}

AvoidCudaGraphCaptureGuard::~AvoidCudaGraphCaptureGuard() { (void)cudaThreadExchangeStreamCaptureMode(&mode_); }

CudaStreamWithFlags::CudaStreamWithFlags(unsigned int flags) {
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream_, flags));
}
CudaStreamWithFlags::~CudaStreamWithFlags() { (void)cudaStreamDestroy(stream_); }

}  // namespace mscclpp
