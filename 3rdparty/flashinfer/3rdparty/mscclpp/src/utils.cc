// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/utils.hpp>
#include <sstream>
#include <string>

// Throw upon SIGALRM.
static void sigalrmTimeoutHandler(int) {
  signal(SIGALRM, SIG_IGN);
  throw mscclpp::Error("Timer timed out", mscclpp::ErrorCode::Timeout);
}

namespace mscclpp {

Timer::Timer(int timeout) { set(timeout); }

Timer::~Timer() {
  if (timeout_ > 0) {
    alarm(0);
    signal(SIGALRM, SIG_DFL);
  }
}

int64_t Timer::elapsed() const {
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
}

void Timer::set(int timeout) {
  timeout_ = timeout;
  if (timeout > 0) {
    signal(SIGALRM, sigalrmTimeoutHandler);
    alarm(timeout);
  }
  start_ = std::chrono::steady_clock::now();
}

void Timer::reset() { set(timeout_); }

void Timer::print(const std::string& name) {
  auto us = elapsed();
  std::stringstream ss;
  ss << name << ": " << us << " us\n";
  std::cout << ss.str();
}

ScopedTimer::ScopedTimer(const std::string& name) : name_(name) {}

ScopedTimer::~ScopedTimer() { print(name_); }

std::string getHostName(int maxlen, const char delim) {
  std::string hostname(maxlen + 1, '\0');
  if (gethostname(const_cast<char*>(hostname.data()), maxlen) != 0) {
    throw Error("gethostname failed", ErrorCode::SystemError);
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
  hostname[i] = '\0';
  return hostname.substr(0, i);
}

bool isNvlsSupported() {
#if (CUDART_VERSION >= 12010)
  CUdevice dev;
  int isNvlsSupported;
  MSCCLPP_CUTHROW(cuCtxGetDevice(&dev));
  MSCCLPP_CUTHROW(cuDeviceGetAttribute(&isNvlsSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
  return isNvlsSupported == 1;
#endif
  return false;
}

}  // namespace mscclpp
