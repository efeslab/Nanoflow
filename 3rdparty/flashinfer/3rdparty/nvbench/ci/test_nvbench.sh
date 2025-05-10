#!/bin/bash

source "$(dirname "$0")/build_common.sh"

# Run NVBench tests with high parallelism. If any need to be
# serialized, define the `RUN_SERIAL` CMake property on the
# test.
export CTEST_PARALLEL_LEVEL=${PARALLEL_LEVEL}

print_environment_details

./build_nvbench.sh "$@"

PRESET="nvbench-ci"

test_preset "NVBench" ${PRESET}

print_time_summary
