#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="nvbench-ci"

CMAKE_OPTIONS=""

function version_lt() {
  local lhs="${1//v/}"
  local rhs="${2//v/}"
  # If the versions are equal, return false
  [ "$lhs" = "$rhs" ] && return 1
  # If the left-hand side is less than the right-hand side, return true
  [  "$lhs" = `echo -e "$lhs\n$rhs" | sort -V | head -n1` ]
}

# If CUDA_COMPILER is nvcc and the version < 11.3, disable CUPTI
if [[ "$CUDA_COMPILER" == *"nvcc"* ]]; then
  CUDA_VERSION=$(nvcc --version | grep release | sed -r 's/.*release ([0-9.]+).*/\1/')
  if version_lt "$CUDA_VERSION" "11.3"; then
    CMAKE_OPTIONS+=" -DNVBench_ENABLE_CUPTI=OFF "
  fi
fi

configure_and_build_preset "NVBench" "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
