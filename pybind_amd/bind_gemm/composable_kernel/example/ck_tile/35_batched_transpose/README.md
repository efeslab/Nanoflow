# Batched Transpose
This folder contains example for batched Transpose using ck_tile tile-programming implementation. Currently, it supports the batched transpose with NCHW to NHWC or NHWC to NCHW. So in this way from NCHW you could transpose to either NHWC or NWCH(two transposes). Now the transpose read with single data point. We would soon put it in vectorized transpose.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
# Make the transpose executable
make tile_example_batched_transpose -j
```
This will result in an executable `build/bin/tile_example_batched_transpose`

## example
```
args:
          -N    input batch size (default:2)
          -C    input channel size. (default:16)
          -H    input height size. (default:1)
          -W    input width size. (default:16)
          -v    whether do CPU validation or not (default: 1)
  -layout_in    input tensor data layout - NCHW by default
 -layout_out    output tensor data layout - NHWC by default
       -seed    seed to be used, -1 means random every time (default:-1)
     -k_name    t to 1 will print kernel name (default:0)
```