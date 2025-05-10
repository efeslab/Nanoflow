# Queries

* `--list`, `-l`
  * List all devices and benchmarks without running them.

* `--help`, `-h`
  * Print usage information and exit.

* `--help-axes`, `--help-axis`
  * Print axis specification documentation and exit.

* `--version`
  * Print information about the version of NVBench used to build the executable.

# Device Modification

* `--persistence-mode <state>`, `--pm <state>`
  * Sets persistence mode for one or more GPU devices.
  * Applies to the devices described by the most recent `--devices` option,
    or all devices if `--devices` is not specified.
  * This option requires root / admin permissions.
  * This option is only supported on Linux.
  * This call must precede all other device modification options, if any.
  * Note that persistence mode is deprecated and will be removed at some point
    in favor of the new persistence daemon. See the following link for more
    details: https://docs.nvidia.com/deploy/driver-persistence/index.html
  * Valid values for `state` are:
    * `0`: Disable persistence mode.
    * `1`: Enable persistence mode.

* `--lock-gpu-clocks <rate>`, `--lgc <rate>`
  * Lock GPU clocks for one or more devices to a particular rate.
  * Applies to the devices described by the most recent `--devices` option,
    or all devices if `--devices` is not specified.
  * This option requires root / admin permissions.
  * This option is only supported in Volta+ (sm_70+) devices.
  * Valid values for `rate` are:
    * `reset`, `unlock`, `none`: Unlock the GPU clocks.
    * `base`, `tdp`: Lock clocks to base frequency (best for stable results).
    * `max`, `maximum`: Lock clocks to max frequency (best for fastest results).

# Output

* `--csv <filename/stream>`
  * Write CSV output to a file, or "stdout" / "stderr".

* `--json <filename/stream>`
  * Write JSON output to a file, or "stdout" / "stderr".

* `--markdown <filename/stream>`, `--md <filename/stream>`
  * Write markdown output to a file, or "stdout" / "stderr".
  * Markdown is written to "stdout" by default.

* `--quiet`, `-q`
  * Suppress output.

* `--color`
  * Use color in output (markdown + stdout only).

# Benchmark / Axis Specification

* `--benchmark <benchmark name/index>`, `-b <benchmark name/index>`
  * Execute a specific benchmark.
  * Argument is a benchmark name or index, taken from `--list`.
  * If not specified, all benchmarks will run.
  * `--benchmark` may be specified multiple times to run several benchmarks.
  * The same benchmark may be specified multiple times with different
    configurations.

* `--axis <axis specification>`, `-a <axis specification>`
  * Override an axis specification.
  * See `--help-axis`
    for [details on axis specifications](./cli_help_axis.md).
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

# Benchmark Properties

* `--devices <device ids>`, `--device <device ids>`, `-d <device ids>`
  * Limit execution to one or more devices.
  * `<device ids>` is a single id, a comma separated list, or the string "all".
  * Device ids can be obtained from `--list`.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--min-samples <count>`
  * Gather at least `<count>` samples per measurement.
  * Default is 10 samples.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--stopping-criterion <criterion>`
  * After `--min-samples` is satisfied, use `<criterion>` to detect if enough 
    samples were collected.
  * Only applies to Cold measurements.
  * Default is stdrel (`--stopping-criterion stdrel`)

* `--min-time <seconds>`
  * Accumulate at least `<seconds>` of execution time per measurement.
  * Only applies to `stdrel` stopping criterion.
  * Default is 0.5 seconds.
  * If both GPU and CPU times are gathered, this applies to GPU time only.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--max-noise <value>`
  * Gather samples until the error in the measurement drops below `<value>`.
  * Noise is specified as the percent relative standard deviation.
  * Default is 0.5% (`--max-noise 0.5`)
  * Only applies to `stdrel` stopping criterion.
  * Only applies to Cold measurements.
  * If both GPU and CPU times are gathered, this applies to GPU noise only.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--skip-time <seconds>`
  * Skip a measurement when a warmup run executes in less than `<seconds>`.
  * Default is -1 seconds (disabled).
  * Intended for testing / debugging only.
  * Very fast kernels (<5us) often require an extremely large number of samples
    to converge `max-noise`. This option allows them to be skipped to save time
    during testing.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--timeout <seconds>`
  * Measurements will timeout after `<seconds>` have elapsed.
  * Default is 15 seconds.
  * `<seconds>` is walltime, not accumulated sample time.
  * If a measurement times out, the default markdown log will print a warning to
    report any outstanding termination criteria (min samples, min time, max
    noise).
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--run-once`
  * Only run the benchmark once, skipping any warmup runs and batched
    measurements.
  * Intended for use with external profiling tools.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--disable-blocking-kernel`
  * Don't use the `blocking_kernel`.
  * Intended for use with external profiling tools.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--profile`
  * Implies `--run-once` and `--disable-blocking-kernel`.
  * Intended for use with external profiling tools.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.
