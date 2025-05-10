// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_UTILS_HPP_
#define MSCCLPP_UTILS_HPP_

#include <chrono>
#include <string>

namespace mscclpp {

struct Timer {
  std::chrono::steady_clock::time_point start_;
  int timeout_;

  Timer(int timeout = -1);

  ~Timer();

  /// Returns the elapsed time in microseconds.
  int64_t elapsed() const;

  void set(int timeout);

  void reset();

  void print(const std::string& name);
};

struct ScopedTimer : public Timer {
  const std::string name_;

  ScopedTimer(const std::string& name);

  ~ScopedTimer();
};

std::string getHostName(int maxlen, const char delim);

bool isNvlsSupported();

}  // namespace mscclpp

#endif  // MSCCLPP_UTILS_HPP_
