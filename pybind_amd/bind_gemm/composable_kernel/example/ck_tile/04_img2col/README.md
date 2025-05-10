# Image to Column

This folder contains example for Image to Column using ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
make tile_example_img2col -j
```
This will result in an executable `build/bin/tile_example_img2col`
