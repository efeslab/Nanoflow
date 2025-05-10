# Minimal Benchmark

A basic kernel benchmark can be created with just a few lines of CUDA C++:

```cpp
void my_benchmark(nvbench::state& state) {
  state.exec([](nvbench::launch& launch) { 
    my_kernel<<<num_blocks, 256, 0, launch.get_stream()>>>();
  });
}
NVBENCH_BENCH(my_benchmark);
```

There are three main components in the definition of a benchmark:

- A `KernelGenerator` callable (`my_benchmark` above)
- A `KernelLauncher` callable (the lambda passed to `nvbench::exec`), and
- A `BenchmarkDeclaration` using `NVBENCH_BENCH` or similar macros.

The `KernelGenerator` is called with an `nvbench::state` object that provides
configuration information, as shown in later sections. The generator is
responsible for configuring and instantiating a `KernelLauncher`, which is
(unsurprisingly) responsible for launching a kernel. The launcher should contain
only the minimum amount of code necessary to start the CUDA kernel,
since `nvbench::exec` will execute it repeatedly to gather timing information.
An `nvbench::launch` object is provided to the launcher to specify kernel
execution details, such as the CUDA stream to use. `NVBENCH_BENCH` registers
the benchmark with NVBench and initializes various attributes, including its
name and parameter axes.

# Benchmark Name

By default, a benchmark is named by converting the first argument
of `NVBENCH_BENCH` into a string.

This can be changed to something more descriptive if desired.
The `NVBENCH_BENCH` macro produces a customization object that allows such
attributes to be modified.

```cpp
NVBENCH_BENCH(my_benchmark).set_name("my_kernel<<<num_blocks, 256>>>");
```

# CUDA Streams

NVBench records GPU execution times on a specific CUDA stream. By default, a new
stream is created and passed to the `KernelLauncher` via the
`nvbench::launch::get_stream()` method, as shown in
[Minimal Benchmark](#minimal-benchmark). All benchmarked kernels and other
stream-ordered work must be launched on this stream for NVBench to capture it.

In some instances, it may be inconvenient or impossible to specify an explicit
CUDA stream for the benchmarked operation to use. For example, a library may
manage and use its own streams, or an opaque API may always launch work on the
default stream. In these situations, users may provide NVBench with an explicit
stream via `nvbench::state::set_cuda_stream` and `nvbench::make_cuda_stream_view`.
It is assumed that all work of interest executes on or synchronizes with this
stream.

```cpp
void my_benchmark(nvbench::state& state) {
  cudaStream_t default_stream = 0;
  state.set_cuda_stream(nvbench::make_cuda_stream_view(default_stream));
  state.exec([](nvbench::launch&) {
    my_func(); // a host API invoking GPU kernels on the default stream
    my_kernel<<<num_blocks, 256>>>(); // or a kernel launched with the default stream
  });
}
NVBENCH_BENCH(my_benchmark);
```

A full example can be found in [examples/stream.cu](../examples/stream.cu).

# Parameter Axes

Some kernels will be used with a variety of options, input data types/sizes, and
other factors that impact performance. NVBench explores these different
scenarios by sweeping through a set of user-defined parameter axes.

A parameter axis defines a set of interesting values for a single kernel
parameter — for example, the size of the input, or the type of values being
processed. These parameter axes are used to customize a `KernelGenerator` with
static and runtime configurations. There are four supported types of parameters:
int64, float64, string, and type.

More examples can found in [examples/axes.cu](../examples/axes.cu).

## Int64 Axes

A common example of a parameter axis is to vary the number of input values a
kernel should process during a benchmark measurement. An `int64_axis` is ideal
for this:

```cpp
void benchmark(nvbench::state& state)
{
  const auto num_inputs = state.get_int64("NumInputs");
  thrust::device_vector<int> data = generate_input(num_inputs);

  state.exec([&data](nvbench::launch& launch) { 
    my_kernel<<<blocks, threads, 0, launch.get_stream()>>>(data.begin(), data.end());
  });
}
NVBENCH_BENCH(benchmark).add_int64_axis("NumInputs", {16, 64, 256, 1024, 4096});
```

NVBench will run the `benchmark` kernel generator once for each specified value
in the "NumInputs" axis. The `state` object provides the current parameter value
to `benchmark`.

### Int64 Power-Of-Two Axes

Using powers-of-two is quite common for these sorts of axes. `int64_axis` has a
unique power-of-two mode that simplifies how such axes are defined and helps
provide more readable output. A power-of-two int64 axis is defined using the
integer exponents, but the benchmark will be run with the computed 2^N value.

```cpp
// Equivalent to above, {16, 64, 256, 1024, 4096} = {2^4, 2^6, 2^8, 2^10, 2^12}
NVBENCH_BENCH(benchmark).add_int64_power_of_two_axis("NumInputs",
                                                     {4, 6, 8, 10, 12});
// Or, as shown in a later section:
NVBENCH_BENCH(benchmark).add_int64_power_of_two_axis("NumInputs",
                                                     nvbench::range(4, 12, 2));
```

## Float64 Axes

For floating point numbers, a `float64_axis` is available:

```cpp
void benchmark(nvbench::state& state)
{
  const auto quality = state.get_float64("Quality");

  state.exec([&quality](nvbench::launch& launch)
  { 
    my_kernel<<<blocks, threads, 0, launch.get_stream()>>>(quality);
  });
}
NVBENCH_BENCH(benchmark).add_float64_axis("Quality", {0.05, 0.1, 0.25, 0.5, 0.75, 1.});
```

## String Axes

For non-numeric data, an axis of arbitrary strings provides additional
flexibility:

```cpp
void benchmark(nvbench::state& state)
{
  const auto rng_dist = state.get_string("RNG Distribution");
  thrust::device_vector<int> data = generate_input(rng_dist);

  state.exec([&data](nvbench::launch& launch)
  { 
    my_kernel<<<blocks, threads, 0, launch.get_stream()>>>(data.begin(), data.end());
  });
}
NVBENCH_BENCH(benchmark).add_string_axis("RNG Distribution", {"Uniform", "Gaussian"});
```

A common use for string axes is to encode enum values, as shown in
[examples/enums.cu](../examples/enums.cu).

## Type Axes

Another common situation involves benchmarking a templated kernel with multiple
compile-time configurations. NVBench strives to make such benchmarks as easy to
write as possible through the use of type axes.

A `type_axis` is a list of types (`T1`, `T2`, `Ts`...) wrapped in
a `nvbench::type_list<T1, T2, Ts...>`. The kernel generator becomes a template
function and will be instantiated using types defined by the axis. The current
configuration's type is passed into the kernel generator using
a `nvbench::type_list<T>`.

```cpp
template <typename T>
void my_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  thrust::device_vector<T> data = generate_input<T>();

  state.exec([&data](nvbench::launch& launch)
  { 
    my_kernel<<<blocks, threads, 0, launch.get_stream()>>>(data.begin(), data.end());
  });
}
using my_types = nvbench::type_list<int, float, double>;
NVBENCH_BENCH_TYPES(my_benchmark, NVBENCH_TYPE_AXES(my_types))
  .set_type_axes_names({"ValueType"});
```

The `NVBENCH_TYPE_AXES` macro is unfortunately necessary to prevent commas in
the `type_list<...>` from breaking macro parsing.

Type axes can be used to encode compile-time enum and integral constants using
the `nvbench::enum_type_list` helper. See
[examples/enums.cu](../examples/enums.cu) for detail.

## `nvbench::range`

Since parameter sweeps often explore a range of evenly-spaced numeric values, a
strided range can be generated using the `nvbench::range(start, end, stride=1)`
helper.

```cpp
assert(nvbench::range(2, 5) == {2, 3, 4, 5});
assert(nvbench::range(2.0, 5.0) == {2.0, 3.0, 4.0, 5.0});
assert(nvbench::range(2, 12, 2) == {2, 4, 6, 8, 10, 12});
assert(nvbench::range(2, 12, 5) == {2, 7, 12});
assert(nvbench::range(2, 12, 6) == {2, 8});
assert(nvbench::range(0.0, 10.0, 2.5) == { 0.0, 2.5, 5.0, 7.5, 10.0});
```

Note that start and end are inclusive. This utility can be used to define axis
values for all numeric axes.

## Multiple Parameter Axes

If more than one axis is defined, the complete cartesian product of all axes
will be benchmarked. For example, consider a benchmark with two type axes, one
int64 axis, and one float64 axis:

```cpp
// InputTypes: {char, int, unsigned int}
// OutputTypes: {float, double}
// NumInputs: {2^10, 2^20, 2^30}
// Quality: {0.5, 1.0}

using input_types = nvbench::type_list<char, int, unsigned int>;
using output_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(benchmark, NVBENCH_TYPE_AXES(input_types, output_types))
  .set_type_axes_names({"InputType", "OutputType"})
  .add_int64_power_of_two_axis("NumInputs", nvbench::range(10, 30, 10))
  .add_float64_axis("Quality", {0.5, 1.0});
```

This would generate a total of 36 configurations and instantiate the benchmark 6
times. Keep the rapid growth of these combinations in mind when choosing the
number of values in an axis. See the section about combinatorial explosion for
more examples and information.

# Throughput Measurements

In additional to raw timing information, NVBench can track a kernel's
throughput, reporting the amount of data processed as:

- Number of items per second
- Number of bytes per second
- Percentage of device's peak memory bandwidth utilized

To enable throughput measurements, the kernel generator can specify the number
of items and/or bytes handled in a single kernel execution using
the `nvbench::state` API.

```cpp
state.add_element_count(size);
state.add_global_memory_reads<InputType>(size);
state.add_global_memory_writes<OutputType>(size);
```

In general::
- Add only the input element count (no outputs).
- Add all reads and writes to global memory.

More examples can found in [examples/throughput.cu](../examples/throughput.cu).


# Skip Uninteresting / Invalid Benchmarks

Sometimes particular combinations of parameters aren't useful or interesting —
or for type axes, some configurations may not even compile.

The `nvbench::state` object provides a `skip("Reason")` method that can be used
to avoid running these benchmarks. To skip uncompilable type axis
configurations, create an overload for the kernel generator that selects for the
invalid type combination:

```cpp
template <typename T, typename U>
void my_benchmark(nvbench::state& state, nvbench::type_list<T, U>)
{
  // Skip benchmarks at runtime:
  if (should_skip_this_config)
  {
    state.skip("Reason for skip.");
    return;
  }

  /* ... */
};

// Skip benchmarks at compile time -- for example, always skip when T == U
// (Note that the `type_list` argument defines the same type twice).
template <typename SameType>
void my_benchmark(nvbench::state& state, 
                  nvbench::type_list<SameType, SameType>)
{
  state.skip("T must not be the same type as U.");
}
using Ts = nvbench::type_list<...>;
using Us = nvbench::type_list<...>;
NVBENCH_BENCH_TYPES(my_benchmark, NVBENCH_TYPE_AXES(Ts, Us));
```

More examples can found in [examples/skip.cu](../examples/skip.cu).

# Execution Tags For Special Cases

By default, NVBench assumes that the entire execution time of the
`KernelLauncher` should be measured, and that no syncs are performed
(e.g. `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventSynchronize`,
etc. are not called).

Execution tags may be passed to `state.exec` when these assumptions are not
true:

- `nvbench::exec_tag::sync` tells NVBench that the kernel launcher will
  synchronize internally.
- `nvbench::exec_tag::timer` requests a timer object that can be used to
  restrict the timed region.

Multiple execution tags may be combined using `operator|`, e.g.

```cpp
state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
           [](nvbench::launch &launch, auto& timer) { /*...*/ });
```

The following sections provide more details on these features.

## Benchmarks that sync: `nvbench::exec_tag::sync`

If a `KernelLauncher` synchronizes the CUDA device internally without passing
this tag, **the benchmark will deadlock at runtime**. Passing the `sync` tag
will fix this issue. Note that this disables batch measurements.

```cpp
void sync_example(nvbench::state& state)
{
  // Pass the `sync` exec tag to tell NVBench that this benchmark will sync:
  state.exec(nvbench::exec_tag::sync, [](nvbench::launch& launch) {
    /* Benchmark that implicitly syncs here. */
  });
}
NVBENCH_BENCH(sync_example);
```

See [examples/exec_tag_sync.cu](../examples/exec_tag_sync.cu) for a complete
example.

## Explicit timer mode: `nvbench::exec_tag::timer`

For some kernels, the working data may need to be reset between launches. This
is particularly common for kernels that modify their input in-place.

Resetting the input data to prepare for a new trial shouldn't be included in the
benchmark's execution time. NVBench provides a manual timer mode that allows the
kernel launcher to specify the critical section to be measured and exclude any
per-trial reset operations.

To enable the manual timer mode, pass the tag object `nvbench::exec_tag::timer`
to `state.exec`, and declare the kernel launcher with an
additional `auto& timer` argument.

Note that using manual timer mode disables batch measurements.

```cpp
void timer_example(nvbench::state& state)
{
  // Pass the `timer` exec tag to request a timer:
  state.exec(nvbench::exec_tag::timer, 
    // Lambda now accepts a timer:
    [](nvbench::launch& launch, auto& timer)
    {
      /* Reset code here, excluded from timing */

      /* Timed region is explicitly marked.
       * The timer handles any synchronization, flushes, etc when/if
       * needed for the current measurement.
       */
      timer.start();
      /* Launch kernel on `launch.get_stream()` here */
      timer.stop();
    });
}
NVBENCH_BENCH(timer_example);
```

See [examples/exec_tag_timer.cu](../examples/exec_tag_timer.cu) for a complete
example.

# Beware: Combinatorial Explosion Is Lurking

Be very careful of how quickly the configuration space can grow. The following
example generates 960 total runtime benchmark configurations, and will compile
192 different static parametrizations of the kernel generator. This is likely
excessive, especially for routine regression testing.

```cpp
using value_types = nvbench::type_list<nvbench::uint8_t,
                                       nvbench::int32_t,
                                       nvbench::float32_t,
                                       nvbench::float64_t>;
using op_types = nvbench::type_list<thrust::plus<>, 
                                    thrust::multiplies<>,
                                    thrust::maximum<>>;

NVBENCH_BENCH_TYPES(my_benchmark,
                    NVBENCH_TYPE_AXES(value_types,
                                      value_types,
                                      value_types,
                                      op_types>))
  .set_type_axes_names({"T", "U", "V", "Op"})
  .add_int64_power_of_two_axis("NumInputs", nvbench::range(10, 30, 5));
```

```
960 total configs
= 4 [T=(U8, I32, F32, F64)] 
* 4 [U=(U8, I32, F32, F64)]
* 4 [V=(U8, I32, F32, F64)]
* 3 [Op=(plus, multiplies, max)]
* 5 [NumInputs=(2^10, 2^15, 2^20, 2^25, 2^30)]
```

For large configuration spaces like this, pruning some of the less useful
combinations (e.g. `sizeof(init_type) < sizeof(output)`) using the techniques
described in the "Skip Uninteresting / Invalid Benchmarks" section can help
immensely with keeping compile / run times manageable.

Splitting a single large configuration space into multiple, more focused
benchmarks with reduced dimensionality will likely be worth the effort as well.
