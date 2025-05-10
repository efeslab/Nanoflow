/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_UTILS_INTERNAL_HPP_
#define MSCCLPP_UTILS_INTERNAL_HPP_

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mscclpp/utils.hpp>

namespace mscclpp {

// PCI Bus ID <-> int64 conversion functions
std::string int64ToBusId(int64_t id);
int64_t busIdToInt64(const std::string busId);

uint64_t getHash(const char* string, int n);
uint64_t getHostHash();
uint64_t getPidHash();
void getRandomData(void* buffer, size_t bytes);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

template <class T>
inline void hashCombine(std::size_t& hash, const T& v) {
  std::hash<T> hasher;
  hash ^= hasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
}

struct PairHash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    std::size_t hash = 0;
    hashCombine(hash, x.first);
    hashCombine(hash, x.second);
    return hash;
  }
};

}  // namespace mscclpp

#endif
