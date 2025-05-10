#!/bin/sh
# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_fwd -type f | head -n 1)"
VALID=0

for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do
for hdim in 64 128 256 ; do

nhead=$((2048 / $hdim))     # follow fav2 setup
$EXE -prec=$prec -b=32 -h=$nhead -d=$hdim -s=512   -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=16 -h=$nhead -d=$hdim -s=1024  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=8  -h=$nhead -d=$hdim -s=2048  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=4  -h=$nhead -d=$hdim -s=4096  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=2  -h=$nhead -d=$hdim -s=8192  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=1  -h=$nhead -d=$hdim -s=16384 -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3

done
done
done

for perm in 0 1 ; do

$EXE -prec=fp8 -squant=1 -b=32 -h=16 -d=128 -s=512   -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=1 -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=16 -h=16 -d=128 -s=1024  -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=1 -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=8  -h=16 -d=128 -s=2048  -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=1 -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=4  -h=16 -d=128 -s=4096  -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=1 -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=2  -h=16 -d=128 -s=8192  -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=1 -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=1  -h=16 -d=128 -s=16384 -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=1 -v=$VALID ; sleep 3

done