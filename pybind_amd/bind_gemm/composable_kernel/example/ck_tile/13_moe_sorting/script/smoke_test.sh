# #!/bin/sh

EXE=./build/bin/tile_example_moe_sorting

$EXE -t=80 -e=17 -moe_buf_size=16
$EXE -t=111 -e=117 -moe_buf_size=4
$EXE -t=1000 -e=55 -moe_buf_size=1024
$EXE -t=99 -e=120  -moe_buf_size=10244
$EXE -t=175 -e=64 -k=8
$EXE -t=65 -e=8 -k=2
$EXE -t=1 -e=25
$EXE -t=31 -e=19 -k=15
$EXE -t=81 -e=37 -k=7
$EXE -t=23 -e=1 -k=1
$EXE -t=127 -e=99 -k=19
$EXE -t=71 -e=11 -k=11
$EXE -t=1 -e=1 -k=1
$EXE -t=99 -e=2 -k=1
$EXE -t=333 -e=99 -k=13
$EXE -t=128 -e=32 -k=5 -moe_buf_size=262144
