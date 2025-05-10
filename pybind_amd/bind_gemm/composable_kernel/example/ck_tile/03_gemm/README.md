# GEMM Matrix Multiplication

This folder contains example for GEMM using ck_tile tile-programming implementation. Currently, it only supports the basic feature of the CK Tile GEMM, but creates the placeholders for the future support on different GEMM pipeline and different GEMM modules. In the near future, we will gradually migrate all the GEMM features from old CK to CK Tile.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
# The basic pipeline method on the gemm calculation
make tile_example_gemm_basic -j
# The memory bound pipeline on the gemm calculation
make tile_example_gemm_universal -j
```
This will result in an executable `build/bin/tile_example_gemm_basic` & `build/bin/tile_example_gemm_universal`

## example
```
args:
          -b    batch size (default:1)
          -m    m dimension (default:1024)
          -n    n dimension (default:2048)
          -k    k dimension (default:64)
   -a_layout    Tensor A data layout (default: R)
   -b_layout    Tensor B data layout (default: R)
   -c_layout    Tensor C data layout (default: R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
          -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:2)
          -e    Absolute error tolerance (default:1e-5)
       -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
     -warmup    number of iterations before benchmark the kernel (default:10)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
```
