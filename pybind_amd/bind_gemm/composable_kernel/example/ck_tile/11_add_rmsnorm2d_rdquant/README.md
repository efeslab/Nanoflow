# Add + Rmsnorm2D + rowwise dynamic quantization forward

This folder contains example for add + Rmsnorm2D + rowwise dynamic quantization forward using ck_tile tile-programming implementation. Rdquant is short for rowwise dynamic quantization here.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_add_rmsnorm2d_rdquant_fwd -j
```
This will result in an executable `build/bin/tile_add_rmsnorm2d_rdquant_fwd`

## cmdline
```
args:
          -m    m dimension (default:3328)
          -n    m dimension (default:4096)
          -e    epsilon (default:1e-5)
          -v    cpu validation or not (default:1)
       -prec    precision (default:fp16)
```
