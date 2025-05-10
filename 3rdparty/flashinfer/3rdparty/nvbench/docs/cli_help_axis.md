# Axis Specification

The `--axis <axis spec>` option redefines the values in a benchmark's axis. It
applies to the benchmark created by the most recent `--benchmark` argument, or
all benchmarks if it precedes all `--benchmark` arguments (if any).

Valid axis specification follow the form:

* `<axis_name>=<value>`
* `<axis_name>=[<value1>,<value2>,...]`
* `<axis_name>=[<start>:<stop>]`
* `<axis_name>=[<start>:<stop>:<stride>]`
* `<axis_name>[<flags>]=<value>`
* `<axis_name>[<flags>]=[<value1>,<value2>,...]`
* `<axis_name>[<flags>]=[<start>:<stop>]`
* `<axis_name>[<flags>]=[<start>:<stop>:<stride>]`

Whitespace is ignored if the argument is quoted.

The axis type is taken from the benchmark definition. Some axes have additional
restrictions:

* Numeric axes:
  * A single value, explicit list of values, or strided range may be specified.
  * For `int64` axes, the `power_of_two` flag is specified by adding `[pow2]`
    after the axis name.
  * Values may differ from those defined in the benchmark.
* String axes:
  * A single value or explicit list of values may be specified.
  * Values may differ from those defined in the benchmark.
* Type axes:
  * A single value or explicit list of values may be specified.
  * Values **MUST** be a subset of the types defined in the benchmark.
  * Values **MUST** match the input strings provided by `--list` (e.g. `I32`
    for `int`).
  * Provide a `nvbench::type_strings<T>` specialization to modify a custom
    type's input string.

# Examples

## Single Value

| Axis Type | Example                 | Example Result   |
|-----------|-------------------------|------------------|
| Int64     | `-a InputSize=12345`    | 12345            |
| Int64Pow2 | `-a InputSize[pow2]=8`  | 256              |
| Float64   | `-a Quality=0.5`        | 0.5              |
| String    | `-a RNG=Uniform`        | "Uniform"        |
| Type      | `-a ValueType=I32`      | `int32_t`        |

## Explicit List

| Axis Type | Example                         | Example Result                 |
|-----------|---------------------------------|--------------------------------|
| Int64     | `-a InputSize=[1,2,3,4,5]`      | 1, 2, 3, 4, 5                  |
| Int64Pow2 | `-a InputSize[pow2]=[4,6,8,10]` | 16, 64, 256, 1024              |
| Float64   | `-a Quality=[0.5,0.75,1.0]`     | 0.5, 0.75, 1.0                 |
| String    | `-a RNG=[Uniform,Gaussian]`     | "Uniform", "Gaussian"          |
| Type      | `-a ValueType=[U8,I32,F64]`     | `uint8_t`, `int32_t`, `double` |

## Strided Range

| Axis Type | Example                         | Example Result               |
|-----------|---------------------------------|------------------------------|
| Int64     | `-a InputSize=[2:10:2]`         | 2, 4, 6, 8, 10               |
| Int64Pow2 | `-a InputSize[pow2]=[2:10:2]`   | 4, 16, 64, 256, 1024         |
| Float64   | `-a Quality=[.5:1:.1]`          | 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 |
| String    | [Not supported]                 |                              |
| Type      | [Not supported]                 |                              |
