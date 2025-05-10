#!/bin/sh

EXE=./build/bin/tile_example_batched_transpose

for pr in "fp32" "fp16" "int8" ; do
$EXE -pr=$pr -N=1 -C=32 -H=1 -W=32 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -N=2 -C=12 -H=1 -W=32 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -N=3 -C=1334 -H=1 -W=37 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -N=4 -C=27 -H=1 -W=32 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -N=5 -C=1234 -H=1 -W=12 -layout_in='NCHW' -layout_out='NHWC'
done
