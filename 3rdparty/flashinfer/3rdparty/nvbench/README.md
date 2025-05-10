# Overview

This project is a work-in-progress. Everything is subject to change.

NVBench is a C++17 library designed to simplify CUDA kernel benchmarking. It
features:

* [Parameter sweeps](docs/benchmarks.md#parameter-axes): a powerful and
  flexible "axis" system explores a kernel's configuration space. Parameters may
  be dynamic numbers/strings or [static types](docs/benchmarks.md#type-axes).
* [Runtime customization](docs/cli_help.md): A rich command-line interface
  allows [redefinition of parameter axes](docs/cli_help_axis.md), CUDA device
  selection, locking GPU clocks (Volta+), changing output formats, and more.
* [Throughput calculations](docs/benchmarks.md#throughput-measurements): Compute
  and report:
  * Item throughput (elements/second)
  * Global memory bandwidth usage (bytes/second and per-device %-of-peak-bw)
* Multiple output formats: Currently supports markdown (default) and CSV output.
* [Manual timer mode](docs/benchmarks.md#explicit-timer-mode-nvbenchexec_tagtimer):
  (optional) Explicitly start/stop timing in a benchmark implementation.
* Multiple measurement types:
  * Cold Measurements:
    * Each sample runs the benchmark once with a clean device L2 cache.
    * GPU and CPU times are reported.
  * Batch Measurements:
    * Executes the benchmark multiple times back-to-back and records total time.
    * Reports the average execution time (total time / number of executions).

# Supported Compilers and Tools

- CMake > 2.23.1
- CUDA Toolkit + nvcc: 11.1 -> 12.4
- g++: 7 -> 12
- clang++: 9 -> 18
- cl.exe: 2019 -> 2022 (19.29, 29.39)
- Headers are tested with C++17 -> C++20.

# Getting Started

## Minimal Benchmark

A basic kernel benchmark can be created with just a few lines of CUDA C++:

```cpp
void my_benchmark(nvbench::state& state) {
  state.exec([](nvbench::launch& launch) {
    my_kernel<<<num_blocks, 256, 0, launch.get_stream()>>>();
  });
}
NVBENCH_BENCH(my_benchmark);
```

See [Benchmarks](docs/benchmarks.md) for information on customizing benchmarks
and implementing parameter sweeps.

## Command Line Interface

Each benchmark executable produced by NVBench provides a rich set of
command-line options for configuring benchmark execution at runtime. See the
[CLI overview](docs/cli_help.md)
and [CLI axis specification](docs/cli_help_axis.md) for more information.

## Examples

This repository provides a number of [examples](examples/) that demonstrate
various NVBench features and usecases:

- [Runtime and compile-time parameter sweeps](examples/axes.cu)
- [Enums and compile-time-constant-integral parameter axes](examples/enums.cu)
- [Reporting item/sec and byte/sec throughput statistics](examples/throughput.cu)
- [Skipping benchmark configurations](examples/skip.cu)
- [Benchmarking on a specific stream](examples/stream.cu)
- [Benchmarks that sync CUDA devices: `nvbench::exec_tag::sync`](examples/exec_tag_sync.cu)
- [Manual timing: `nvbench::exec_tag::timer`](examples/exec_tag_timer.cu)

### Building Examples

To build the examples:
```
mkdir -p build
cd build
cmake -DNVBench_ENABLE_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=70 .. && make
```
Be sure to set `CMAKE_CUDA_ARCHITECTURE` based on the GPU you are running on.

Examples are built by default into `build/bin` and are prefixed with `nvbench.example`.

<details>
  <summary>Example output from `nvbench.example.throughput`</summary>

```
# Devices

## [0] `Quadro GV100`
* SM Version: 700 (PTX Version: 700)
* Number of SMs: 80
* SM Default Clock Rate: 1627 MHz
* Global Memory: 32163 MiB Free / 32508 MiB Total
* Global Memory Bus Peak: 870 GiB/sec (4096-bit DDR @850MHz)
* Max Shared Memory: 96 KiB/SM, 48 KiB/Block
* L2 Cache Size: 6144 KiB
* Maximum Active Blocks: 32/SM
* Maximum Active Threads: 2048/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

Run:  throughput_bench [Device=0]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.26% > 0.50%)
Pass: Cold: 0.262392ms GPU, 0.267860ms CPU, 7.19s total GPU, 27393x
Pass: Batch: 0.261963ms GPU, 7.18s total GPU, 27394x

# Benchmark Results

## throughput_bench

### [0] Quadro GV100

| NumElements |  DataSize  | Samples |  CPU Time  | Noise |  GPU Time  | Noise | Elem/s  | GlobalMem BW  | BWPeak | Batch GPU  | Batch  |
|-------------|------------|---------|------------|-------|------------|-------|---------|---------------|--------|------------|--------|
|    16777216 | 64.000 MiB |  27393x | 267.860 us | 1.25% | 262.392 us | 1.26% | 63.940G | 476.387 GiB/s | 58.77% | 261.963 us | 27394x |
```

</details>


## Demo Project

To get started using NVBench with your own kernels, consider trying out
the [NVBench Demo Project](https://github.com/allisonvacanti/nvbench_demo).

`nvbench_demo` provides a simple CMake project that uses NVBench to build an
example benchmark. It's a great way to experiment with the library without a lot
of investment.

# Contributing

Contributions are welcome!

For current issues, see the [issue board](https://github.com/NVIDIA/nvbench/issues). Issues labeled with [![](https://img.shields.io/github/labels/NVIDIA/nvbench/good%20first%20issue)](https://github.com/NVIDIA/nvbench/labels/good%20first%20issue) are good for first time contributors.

## Tests

To build `nvbench` tests:
```
mkdir -p build
cd build
cmake -DNVBench_ENABLE_TESTING=ON .. && make
```

Tests are built by default into `build/bin` and prefixed with `nvbench.test`.

To run all tests:
```
make test
```
or
```
ctest
```
# License

NVBench is released under the Apache 2.0 License with LLVM exceptions.
See [LICENSE](./LICENSE).

# Scope and Related Projects

NVBench will measure the CPU and CUDA GPU execution time of a ***single
host-side critical region*** per benchmark. It is intended for regression
testing and parameter tuning of individual kernels. For in-depth analysis of
end-to-end performance of multiple applications, the NVIDIA Nsight tools are
more appropriate.

NVBench is focused on evaluating the performance of CUDA kernels and is not
optimized for CPU microbenchmarks. This may change in the future, but for now,
consider using Google Benchmark for high resolution CPU benchmarks.
