// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "npkit.h"

#include <unistd.h>

#include <chrono>
#include <fstream>
#include <mscclpp/gpu.hpp>

uint64_t NpKit::rank_ = 0;

std::vector<mscclpp::UniqueCudaPtr<NpKitEvent>> NpKit::gpu_event_buffers_;
std::vector<std::unique_ptr<NpKitEvent[]>> NpKit::cpu_event_buffers_;

mscclpp::UniqueCudaPtr<NpKitEventCollectContext> NpKit::gpu_collect_contexts_;
std::unique_ptr<NpKitEventCollectContext[]> NpKit::cpu_collect_contexts_;
uint64_t NpKit::cpu_base_system_timestamp_ = 0;
uint64_t NpKit::cpu_base_steady_timestamp_ = 0;

void NpKit::Init(int rank) {
  uint64_t i = 0;
  NpKitEventCollectContext ctx;
  ctx.event_buffer_head = 0;
  rank_ = rank;

  // Init event data structures
  gpu_collect_contexts_ = mscclpp::allocUniqueCuda<NpKitEventCollectContext>(kNumGpuEventBuffers);
  for (i = 0; i < kNumGpuEventBuffers; i++) {
    gpu_event_buffers_.emplace_back(mscclpp::allocUniqueCuda<NpKitEvent>(kMaxNumGpuEventsPerBuffer));
    ctx.event_buffer = gpu_event_buffers_[i].get();
    mscclpp::memcpyCuda(gpu_collect_contexts_.get() + i, &ctx, 1);
  }

  cpu_collect_contexts_ = std::make_unique<NpKitEventCollectContext[]>(kNumCpuEventBuffers);
  for (i = 0; i < kNumCpuEventBuffers; i++) {
    cpu_event_buffers_.emplace_back(std::make_unique<NpKitEvent[]>(kMaxNumCpuEventsPerBuffer));
    ctx.event_buffer = cpu_event_buffers_[i].get();
    cpu_collect_contexts_[i] = ctx;
  }

  // Init timestamp
  cpu_base_system_timestamp_ = std::chrono::system_clock::now().time_since_epoch().count();
  cpu_base_steady_timestamp_ = std::chrono::steady_clock::now().time_since_epoch().count();
}

void NpKit::Dump(const std::string& dump_dir) {
  uint64_t i = 0;
  std::string dump_file_path;

  // Dump CPU events
  for (i = 0; i < kNumCpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/cpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_channel_";
    dump_file_path += std::to_string(i);
    auto cpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    cpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[i].get()),
                         cpu_collect_contexts_[i].event_buffer_head * sizeof(NpKitEvent));
    cpu_trace_file.close();
  }

  // Dump CPU clock info
  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_num_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_num_str = std::to_string(std::chrono::steady_clock::duration::period::num);
  auto clock_period_num_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_num_file.write(clock_period_num_str.c_str(), clock_period_num_str.length());
  clock_period_num_file.close();

  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_den_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_den_str = std::to_string(std::chrono::steady_clock::duration::period::den);
  auto clock_period_den_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_den_file.write(clock_period_den_str.c_str(), clock_period_den_str.length());
  clock_period_den_file.close();

  // Dump GPU events, reuse CPU struct
  for (i = 0; i < kNumGpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/gpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_buf_";
    dump_file_path += std::to_string(i);
    mscclpp::memcpyCuda(cpu_event_buffers_[0].get(), gpu_event_buffers_[i].get(), kMaxNumGpuEventsPerBuffer);
    mscclpp::memcpyCuda(cpu_collect_contexts_.get(), gpu_collect_contexts_.get() + i, 1);
    auto gpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    gpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[0].get()),
                         cpu_collect_contexts_[0].event_buffer_head * sizeof(NpKitEvent));
    gpu_trace_file.close();
  }

  // Dump GPU clockRate
  dump_file_path = dump_dir;
  dump_file_path += "/gpu_clock_rate_rank_";
  dump_file_path += std::to_string(rank_);
  cudaDeviceProp dev_prop;
  int dev;
  MSCCLPP_CUDATHROW(cudaGetDevice(&dev));
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&dev_prop, dev));
  std::string clock_rate_str = std::to_string(dev_prop.clockRate);
  auto gpu_clock_rate_file = std::fstream(dump_file_path, std::ios::out);
  gpu_clock_rate_file.write(clock_rate_str.c_str(), clock_rate_str.length());
  gpu_clock_rate_file.close();
}

void NpKit::Shutdown() {
  // Free CPU event data structures
  cpu_event_buffers_.clear();
  cpu_collect_contexts_.reset();

  // Free GPU event data structures
  gpu_event_buffers_.clear();
  gpu_collect_contexts_.reset();
}

NpKitEventCollectContext* NpKit::GetGpuEventCollectContexts() { return gpu_collect_contexts_.get(); }

void NpKit::CollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp, int channel_id) {
  uint64_t event_buffer_head = cpu_collect_contexts_[channel_id].event_buffer_head;
  if (event_buffer_head < kMaxNumCpuEventsPerBuffer) {
    NpKitEvent& event = cpu_collect_contexts_[channel_id].event_buffer[event_buffer_head];
    event.fields.type = type;
    event.fields.size = size;
    event.fields.rsvd = rsvd;
    event.fields.timestamp = timestamp;
    cpu_collect_contexts_[channel_id].event_buffer_head++;
  }
}

uint64_t NpKit::GetCpuTimestamp() {
  uint64_t cpu_curr_steady_timestamp_ = std::chrono::steady_clock::now().time_since_epoch().count();
  return cpu_base_steady_timestamp_ + (cpu_curr_steady_timestamp_ - cpu_base_steady_timestamp_);
}
