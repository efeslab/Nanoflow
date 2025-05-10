#!/bin/sh
# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_bwd -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1'
set -x
for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do
for hdim in 32 64 128 256 ; do
for mode in 0 1 ; do
for bias in "n" "a" ; do
for dbias in 0 ; do
for p_drop in 0.0 0.2 ; do
for deterministic in 0 ; do

$EXE -prec=$prec -b=1 -h=4 -h_k=2 -d=$hdim -s=259 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -deterministic=$deterministic -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=2 -h=2 -d=$hdim -s=516 -s_k=253 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -deterministic=$deterministic -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=1 -h=4 -h_k=1 -d=$hdim -s=500 -s_k=251 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -mask=1 -deterministic=$deterministic -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=1 -h=2 -d=$hdim -s=900 -s_k=258 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -mask=2 -v=1 -deterministic=$deterministic -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=2 -h=1 -d=$hdim -s=987 -s_k=219 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -mask=t:128,30 -deterministic=$deterministic -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=2 -h=3 -h_k=1 -d=$hdim -s=244 -s_k=499 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -mask=b:4,35 -deterministic=$deterministic -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS

done
done
done
done
done
done
done
done
set +x
