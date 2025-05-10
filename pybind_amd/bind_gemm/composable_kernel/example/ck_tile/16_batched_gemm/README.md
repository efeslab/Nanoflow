# Batched GEMM

This folder contains example for batched GEMM using ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
make tile_example_batched_gemm -j
```
This will result in an executable `build/bin/tile_example_batched_gemm`

## example
```
args:
              -m     m dimension (default:256)
              -n     n dimension (default:128)
              -k     k dimension (default:128)
       -a_layout     A tensor data layout (default:R) (R for Row, C for Col)
       -b_layout     B tensor data layout (default:R) (R for Row, C for Col)
       -c_layout     C tensor data layout (default:R) (R for Row, C for Col)
       -stride_a     Tensor A stride (default:128)
       -stride_b     Tensor B stride (default:128)
       -stride_c     Tensor C stride (default:128)
 -batch_stride_a     Batch A stride (default:32768)
 -batch_stride_b     Batch B stride (default:16384)
 -batch_stride_c     Batch C stride (default:32768)
    -batch_count     Batch count (default:16)
              -v     0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:2)
              -e     Absolute error tolerance (default:1e-5)
           -prec     data type. fp16/bf16/fp8/bf8 (default:fp16)
         -warmup     number of iterations before benchmark the kernel (default:10)
         -repeat     number of iterations to benchmark the kernel (default:100)
          -timer     gpu:gpu timer, cpu:cpu timer (default:gpu)
```