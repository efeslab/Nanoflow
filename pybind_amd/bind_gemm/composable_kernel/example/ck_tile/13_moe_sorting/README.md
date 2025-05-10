# moe-sorting

This folder contains example for moe-sorting kernel using ck_tile tile-programming implementation. This kernel is often used in Moe model, before launching the fused-moe-gemm block. The input&weight is a `token*topk` 2d matrix. The op rearange the input weight ids into different experts and feed into fuse moe gemm kernel.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_moe_sorting -j
```
This will result in an executable `build/bin/tile_example_moe_sorting`

## example
```
args:
          -v    weather do CPU validation or not (default:1)
       -pr_i    index data type. (currently only fp32 supported now) (default:int32)
       -pr_w    output weight data type(currently only fp32 supported now) (default:fp32)
          -t    number of input tokens (default:32)
          -e    number of experts (default:8)
          -k    topk (default:2)
       -st_i    row stride of input, -1 means same as experts (default:-1)
       -seed    seed to be used, -1 means random every time (default:-1)
      -kname    when set to 1 it will print kernel name (default:0)

```
