# Composable Kernel

The Composable Kernel (CK) library provides a programming model for writing performance-critical
kernels for machine learning workloads across multiple architectures (GPUs, CPUs, etc.). The CK library
uses general purpose kernel languages, such as HIP C++.

CK uses two concepts to achieve performance portability and code maintainability:

* A tile-based programming model
* Algorithm complexity reduction for complex machine learning (ML) operators. This uses an innovative
   technique called *Tensor Coordinate Transformation*.

![ALT](/docs/data/ck_component.png "CK Components")

The current CK library is structured into four layers:

* Templated Tile Operators
* Templated Kernel and Invoker
* Instantiated Kernel and Invoker
* Client API

![ALT](/docs/data/ck_layer.png "CK Layers")

## General information

To build our documentation locally, use the following code:

``` bash
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

You can find a list of our developers and contributors on our [Contributors](/CONTRIBUTORS.md) page.

```note
If you use CK, cite us as follows:

* [Realizing Tensor Operators Using Coordinate Transformations and Tile Based Programming](???):
  This paper will be available on arXiv soon.
* [CITATION.cff](/CITATION.cff)
```

CK is released under the **[MIT license](/LICENSE)**.

## Building CK

We recommend building CK inside Docker containers, which include all necessary packages. Pre-built
Docker images are available on [DockerHub](https://hub.docker.com/r/rocm/composable_kernel/tags).

1. To build a new Docker image, use the Dockerfile provided with the source code:

    ```bash
    DOCKER_BUILDKIT=1 docker build -t ck:latest -f Dockerfile .
    ```

2. Launch the Docker container:

    ```bash
    docker run                                     \
    -it                                            \
    --privileged                                   \
    --group-add sudo                               \
    -w /root/workspace                             \
    -v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace  \
    ck:latest                                      \
    /bin/bash
    ```

3. Clone CK source code from the GitHub repository and start the build:

    ```bash
    git clone https://github.com/ROCm/composable_kernel.git && \
    cd composable_kernel && \
    mkdir build && \
    cd build
    ```

    You must set the `GPU_TARGETS` macro to specify the GPU target architecture(s) you want
    to run CK on. You can specify single or multiple architectures. If you specify multiple architectures,
    use a semicolon between each; for example, `gfx908;gfx90a;gfx940`.

    ```bash
    cmake                                                                                             \
    -D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
    -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
    -D CMAKE_BUILD_TYPE=Release                                                                       \
    -D GPU_TARGETS="gfx908;gfx90a"                                                                    \
    ..
    ```

    If you don't set `GPU_TARGETS` on the cmake command line, CK is built for all GPU targets
    supported by the current compiler (this may take a long time).

4. Build the entire CK library:

    ```bash
    make -j
    ```

5. Install CK:

    ```bash
    make -j install
    ```

## Optional post-install steps

* Build examples and tests:

    ```bash
    make -j examples tests
    ```

* Build and run all examples and tests:

    ```bash
    make -j check
    ```

    You can find instructions for running each individual example in [example](/example).

* Build ckProfiler:

    ```bash
    make -j ckProfiler
    ```

    You can find instructions for running ckProfiler in [profiler](/profiler).

Note the `-j` option for building with multiple threads in parallel. This speeds up the build significantly.
Depending on the number of CPU cores and the amount of RAM on your system, you may want to
limit the number of threads. For example, if you have a 128-core CPU and 64 Gb of RAM.

By default, `-j` launches one thread per CPU core, which can cause the build to run out of memory and
crash. In such cases, you can reduce the number of threads to 32 by using `-j32`.

Additional cmake flags can be used to significantly speed-up the build:

* `INSTANCES_ONLY` (default is OFF) must be set to ON in order to build only the instances and library
  while skipping all tests, examples, and profiler. This is useful in cases when you plan to use CK as a
  dependency and don't plan to run any examples or tests.

* `DTYPES` (default is not set) can be set to any subset of "fp64;fp32;fp16;fp8;bf16;int8" to build
  instances of select data types only. The main default data types are fp32 and fp16; you can safely skip
  other data types.

* `DL_KERNELS` (default is OFF) must be set to ON in order to build instances, such as `gemm_dl` or
  `batched_gemm_multi_d_dl`. These instances are useful on architectures like the NAVI2x, as most
  other platforms have faster instances, such as `xdl` or `wmma`, available.

## Using sccache for building

The default CK Docker images come with a pre-installed version of sccache, which supports clang
being used as hip-compiler (" -x hip"). Using sccache can help reduce the time to re-build code from
hours to 1-2 minutes. In order to invoke sccache, you need to run:

```bash
 sccache --start-server
```

then add the following flags to the cmake command line:

```bash
 -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache
```

You may need to clean up the build folder and repeat the cmake and make steps in order to take
advantage of the sccache during subsequent builds.

## Using CK as pre-built kernel library

You can find instructions for using CK as a pre-built kernel library in [client_example](/client_example).

## Contributing to CK

When you contribute to CK, make sure you run `clang-format` on all changed files. We highly
recommend using git hooks that are managed by the `pre-commit` framework. To install hooks, run:

```bash
sudo script/install_precommit.sh
```

With this approach, `pre-commit` adds the appropriate hooks to your local repository and
automatically runs `clang-format` (and possibly additional checks) before any commit is created.

If you need to uninstall hooks from the repository, you can do so by running the following command:

```bash
script/uninstall_precommit.sh
```

If you need to temporarily disable pre-commit hooks, you can add the `--no-verify` option to the
`git commit` command.
