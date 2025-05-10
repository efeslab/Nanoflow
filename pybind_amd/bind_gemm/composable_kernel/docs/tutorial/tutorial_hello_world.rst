.. meta::
  :description: Composable Kernel documentation and API reference library
  :keywords: composable kernel, CK, ROCm, API, documentation

.. _hello-world:

********************************************************************
Hello World Tutorial
********************************************************************

This tutorial is for engineers dealing with artificial intelligence and machine learning who
would like to optimize pipelines and improve performance using the Composable
Kernel (CK) library. This tutorial provides an introduction to the CK library. You will build the library and run some examples using a "Hello World" example. 

Description
===========

Modern AI technology solves more and more problems in a variety of fields, but crafting fast and
efficient workflows is still challenging. CK can make the AI workflow fast
and efficient. CK is a collection of optimized AI operator kernels with tools to create
new kernels. The library has components required for modern neural network architectures
including matrix multiplication, convolution, contraction, reduction, attention modules, a variety of activation functions, and fused operators.

CK library acceleration features are based on:

* Layered structure
* Tile-based computation model
* Tensor coordinate transformation
* Hardware acceleration use
* Support of low precision data types including fp16, bf16, int8 and int4

If you need more technical details and benchmarking results read the following 
`blog post <https://community.amd.com/t5/instinct-accelerators/amd-composable-kernel-library-efficient-fused-kernels-for-ai/ba-p/553224>`_.

To download the library visit the `composable_kernel repository <https://github.com/ROCm/composable_kernel>`_.

Hardware targets
================

CK library fully supports `gfx908` and `gfx90a` GPU architectures, while only some operators are
supported for `gfx1030` devices. Check your hardware to determine the target GPU architecture.

==========     =========
GPU Target     AMD GPU
==========     =========
gfx908 	       Radeon Instinct MI100
gfx90a 	       Radeon Instinct MI210, MI250, MI250X
gfx1030        Radeon PRO V620, W6800, W6800X, W6800X Duo, W6900X, RX 6800, RX 6800 XT, RX 6900 XT, RX 6900 XTX, RX 6950 XT
==========     =========

There are also `cloud options <https://aws.amazon.com/ec2/instance-types/g4/>`_ you can find if
you don't have an AMD GPU at hand.

Build the library
=================

This tutorial is based on the use of docker images as explained in :ref:`docker-hub`. Download a docker image suitable for your OS and ROCm release, run or start the docker container, and then resume the tutorial from this point. 

.. note::

   You can also `install ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ on your system, clone the `Composable Kernel repository <https://github.com/ROCm/composable_kernel.git>`_ on GitHub, and use that to build and run the examples using the commands described below.

Both the docker container and GitHub repository include the Composable Kernel library. Navigate to the library::

    cd composable_kernel/

Create and change to a ``build`` directory::

    mkdir build && cd build

The previous section discussed supported GPU architecture. Once you decide which hardware targets are needed, run CMake using the ``GPU_TARGETS`` flag::

    cmake  \
    -D CMAKE_PREFIX_PATH=/opt/rocm  \
    -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc  \
    -D CMAKE_CXX_FLAGS="-O3"  \
    -D CMAKE_BUILD_TYPE=Release  \
    -D BUILD_DEV=OFF  \
    -D GPU_TARGETS="gfx908;gfx90a;gfx1030" ..

If everything goes well the CMake command will return::

    -- Configuring done
    -- Generating done
    -- Build files have been written to: "/root/workspace/composable_kernel/build"

Finally, you can build examples and tests::

    make -j examples tests

When complete you should see::

    Scanning dependencies of target tests
    [100%] Built target tests

Run examples and tests
======================

Examples are listed as test cases as well, so you can run all examples and tests with::

    ctest

You can check the list of all tests by running::

    ctest -N

You can also run examples separately as shown in the following example execution::

    ./bin/example_gemm_xdl_fp16 1 1 1

The arguments ``1 1 1`` mean that you want to run this example in the mode: verify results with CPU, initialize matrices with integers, and benchmark the kernel execution. You can play around with these parameters and see how output and execution results change.

If you have a device based on `gfx908` or `gfx90a` architecture, and if the example runs as expected, you should see something like::

    a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
    b_k_n: dim 2, lengths {4096, 4096}, strides {4096, 1}
    c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
    Perf: 1.08153 ms, 119.136 TFlops, 89.1972 GB/s, DeviceGemm_Xdl_CShuffle<Default, 256, 256, 128, 32, 8, 2, 32, 32, 4, 2, 8, 4, 1, 2> LoopScheduler: Interwave, PipelineVersion: v1

However, running it on a `gfx1030` device should result in the following::

    a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
    b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
    c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
    DeviceGemmXdl<256, 256, 128, 4, 8, 32, 32, 4, 2> NumPrefetch: 1, LoopScheduler: Default, PipelineVersion: v1 does not support this problem

Don't worry, some operators are supported on `gfx1030` architecture, so you can run a
separate example like::

    ./bin/example_gemm_dl_fp16 1 1 1

and it should return something like::

    a_m_k: dim 2, lengths {3840, 4096}, strides {1, 4096}
    b_k_n: dim 2, lengths {4096, 4096}, strides {4096, 1}
    c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
    arg.a_grid_desc_k0_m0_m1_k1_{2048, 3840, 2}
    arg.b_grid_desc_k0_n0_n1_k1_{2048, 4096, 2}
    arg.c_grid_desc_m_n_{ 3840, 4096}
    launch_and_time_kernel: grid_dim {960, 1, 1}, block_dim {256, 1, 1}
    Warm up 1 time
    Start running 10 times...
    Perf: 3.65695 ms, 35.234 TFlops, 26.3797 GB/s, DeviceGemmDl<256, 128, 128, 16, 2, 4, 4, 1>

.. note::

    A new CMake flag ``DL_KERNELS`` has been added to the latest versions of CK. If you do not see the above results when running ``example_gemm_dl_fp16``, you might need to add ``-D DL_KERNELS=ON`` to your CMake command to build the operators supported on the `gfx1030` architecture.

You can also run a separate test::

    ctest -R test_gemm_fp16

If everything goes well you should see something like::

    Start 121: test_gemm_fp16
    1/1 Test #121: test_gemm_fp16 ...................   Passed   51.81 sec

    100% tests passed, 0 tests failed out of 1

Summary
=======

In this tutorial you took the first look at the Composable Kernel library, built it on your system and ran some examples and tests. In the next tutorial you will run kernels with different configurations to find out the best one for your hardware and task.

P.S.: If you are running on a cloud instance, don't forget to switch off the cloud instance. 
