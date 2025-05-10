# Grouped CShuffle GEMM

This folder contains example for Grouped GEMM using ck_tile tile-programming implementation. Currently, it only supports the basic feature of the CK Tile GEMM, but creates the placeholders for the future support on different GEMM pipeline and different GEMM modules. In the near future, we will gradually migrate all the GEMM features from old CK to CK Tile.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
# The basic pipeline method on the gemm calculation
make tile_example_grouped_gemm -j
```
This will result in an executable `build/bin/tile_example_grouped_gemm`

## example
```
args:
   -a_layout    Tensor A layout (default:R)
   -b_layout    Tensor B layout (default:R)
   -c_layout    Tensor C layout (default:R)
          -v    0. No validation, 1. Validation on CPU
     -warmup    number of iterations before benchmark the kernel (default:10)
     -repeat    number of iterations to benchmark the kernel (default:100)
```
