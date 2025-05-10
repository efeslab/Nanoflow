#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/tile_example_fmha_fwd
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'
# mode=0
# export HIP_VISIBLE_DEVICES=4

for prec in "fp16" "bf16" ; do
for mode in 1 0 ; do
for perm in 0 1 ; do
for vlayout in "r" "c" ; do
for hdim in 32 64 128 256 ; do
for lse in 0 1 ; do
for bias in "n" "e" "a"; do

# $EXE -prec=$prec -mode=$mode -b=1 -h=1 -d=$hdim -s=1024 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=2 -h=2 -h_k=1 -d=16, -d_v=$hdim -s=55 -s_k=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=1 -h=3 -d=$hdim -s=100 -s_k=51 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=2 -h=1 -d=16 -d_v=$hdim -s=99 -s_k=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=1 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim -s=1024 -s_k=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=2 -h=1 -d=$hdim -d_v=24 -s=3 -s_k=99 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=3 -h=2 -h_k=1 -d=$hdim -s=200 -s_k=520 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=t:128,30 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=2 -h=1 -d=$hdim -s=99 -s_k=32 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=b:4,35 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim -s=33 -s_k=0 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim -s=1 -s_k=10 -s_kpad=32 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS

done
done
done
done
done
done
done

for perm in 0 1 ; do
for bias in "n" "e" "a" ; do
for b in 1 2 ; do
for hdim in 64 128 256 ; do
$EXE -prec=fp8 -init=3 -b=$b -h=1 -d=128 -s=128 -bias=$bias -iperm=$perm -operm=$perm -vlayout=c -squant=1 -kname=$KNAME $COMMON_ARGS
done
done
done
done
