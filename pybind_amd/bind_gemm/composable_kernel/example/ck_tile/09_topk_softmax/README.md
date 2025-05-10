# topk-softmax

This folder contains example for topk-softmax kernel using ck_tile tile-programming implementation. This kernel is often used in Moe model, before launching the fused-moe-gemm block. The input is a `token*expert` 2d matrix. The op will do a softmax per row(`expert`), then find the `topk` value for each row. Output is a `token*topk`  weight(usually fp32) and index(int32) 2d tensor.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_topk_softmax -j
```
This will result in an executable `build/bin/tile_example_topk_softmax`

## example
```
args:
          -v    weather do CPU validation or not (default:1)
       -pr_i    input data type. fp16/fp32 (representing 8/16/32 bit data) (default:fp16)
       -pr_w    output weight data type(currently only fp32 supported now) (default:fp32)
          -t    number of input tokens (default:32)
          -e    number of experts (default:8)
          -k    topk (default:2)
       -st_i    row stride of input, -1 means same as experts (default:-1)
       -st_o    row stride of output/indices, -1 means same as topk (default:-1)
       -seed    seed to be used, -1 means random every time (default:-1)
      -kname    when set to 1 it will print kernel name (default:0)

```
