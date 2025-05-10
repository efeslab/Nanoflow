// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NPKIT_H_
#define NPKIT_H_

#include <mscclpp/gpu_utils.hpp>
#include <string>
#include <vector>

#include "npkit_event.h"
#include "npkit_struct.h"

class NpKit {
 public:
  static const uint64_t kNumGpuEventBuffers = 512;

  static const uint64_t kNumCpuEventBuffers = 32;

  static void Init(int rank);

  static void Dump(const std::string& dump_dir);

  static void Shutdown();

  static NpKitEventCollectContext* GetGpuEventCollectContexts();

#ifdef __CUDACC__
  static inline __device__ void CollectGpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp,
                                                NpKitEventCollectContext* ctx) {
    uint64_t event_buffer_head = ctx->event_buffer_head;
    if (event_buffer_head < kMaxNumGpuEventsPerBuffer) {
      NpKitEvent& event = ctx->event_buffer[event_buffer_head];
      event.fields.type = type;
      event.fields.size = size;
      event.fields.rsvd = rsvd;
      event.fields.timestamp = timestamp;
      ctx->event_buffer_head++;
    }
  }
#endif  // __CUDACC__

  static void CollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp, int channel_id);

  static uint64_t GetCpuTimestamp();

 private:
  // 64K * 512 * 16B = 512MB per GPU
  static const uint64_t kMaxNumGpuEventsPerBuffer = 1ULL << 16;

  // 64K * 2 (send/recv) * (512/32) = 2M, 2M * 32 * 16B = 1GB per CPU
  static const uint64_t kMaxNumCpuEventsPerBuffer = 1ULL << 21;

  static std::vector<mscclpp::UniqueCudaPtr<NpKitEvent>> gpu_event_buffers_;
  static std::vector<std::unique_ptr<NpKitEvent[]>> cpu_event_buffers_;

  static mscclpp::UniqueCudaPtr<NpKitEventCollectContext> gpu_collect_contexts_;
  static std::unique_ptr<NpKitEventCollectContext[]> cpu_collect_contexts_;

  static uint64_t cpu_base_system_timestamp_;
  static uint64_t cpu_base_steady_timestamp_;

  static uint64_t rank_;
};

#endif