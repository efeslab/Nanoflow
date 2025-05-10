#!/bin/bash
# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_fwd -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'
# mode=0
# export HIP_VISIBLE_DEVICES=4

TEST_SPLITKV=0
TEST_APPENDKV=0
# options:
#    -s: run splitkv tests
#    -a: run appendkv tests
while getopts ":sa" opt; do
    case "${opt}" in
        s)
            TEST_SPLITKV=1
            ;;
        a)
            TEST_APPENDKV=1
            ;;
        *)
            ;;
    esac
done

run_fp16_bf16_tests() {
    local NUM_SPLITS="1"
    local PAGE_BLOCK_SIZE="0"
    local CACHE_BATCH_IDX="0"

    if [ $TEST_SPLITKV -eq 1 ] ; then
        NUM_SPLITS="$NUM_SPLITS 2 3"
        PAGE_BLOCK_SIZE="$PAGE_BLOCK_SIZE 128"
        CACHE_BATCH_IDX="$CACHE_BATCH_IDX 1"
    fi

    for prec in "fp16" "bf16" ; do
    for mode in 1 0 ; do
    for perm in 0 1 ; do
    for vlayout in "r" "c" ; do
    for hdim in 32 64 128 256 ; do
    for lse in 0 1 ; do
    for bias in "n" "e" "a" ; do
    for p_drop in 0.0 0.2 ; do
    for num_splits in $NUM_SPLITS ; do
    for page_block_size in $PAGE_BLOCK_SIZE ; do
    for cache_batch_idx in $CACHE_BATCH_IDX ; do

    # $EXE -prec=$prec -mode=$mode -b=1 -h=1 -d=$hdim -s=1024 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=2 -h=2 -h_k=1 -d=16, -d_v=$hdim -s=55 -s_k=256 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=1 -h=3 -d=$hdim -s=100 -s_k=51 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=2 -h=1 -d=16 -d_v=$hdim -s=99 -s_k=256 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=1 -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim -s=1024 -s_k=256 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=2 -h=1 -d=$hdim -d_v=24 -s=3 -s_k=99 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=3 -h=2 -h_k=1 -d=$hdim -s=200 -s_k=520 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=t:128,30 -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=2 -h=1 -d=$hdim -s=99 -s_k=32 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=b:4,35 -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim -s=33 -s_k=0 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim -s=1 -s_k=10 -s_kpad=32 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS  

    done ; done ; done ; done ; done
    done ; done ; done ; done ; done
    done ;
}

run_fp8_tests() {
    for perm in 0 1 ; do
    for bias in "n" "e" "a" ; do
    for b in 1 2 ; do
    for hdim in 64 128 256 ; do
    
    $EXE -prec=fp8 -init=3 -b=$b -h=1 -d=128 -s=128 -bias=$bias -iperm=$perm -operm=$perm -vlayout=c -squant=1 -kname=$KNAME $COMMON_ARGS  

    done ; done ; done ; done
}

run_fp16_appendkv_tests() {
    for s in $(seq 63 1 65) ; do
    for s_k in 65 129 ; do
    for s_knew in 0 64 $s_k ; do
    for hdim in 32 64 128 256 ; do
    for ri in 0 1 ; do
    for rdim in 0 16 32 $hdim ; do
    for page_block_size in 0 128 ; do
    for cache_batch_idx in 0 1 ; do

    $EXE -prec=fp16 -b=3 -h=3 -d=$hdim -s=$s -s_k=$s_k -s_knew=$s_knew -rotary_dim=$rdim -rotary_interleaved=$ri -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -iperm=1 -operm=1 -kname=1 $COMMON_ARGS

    done ; done ; done ; done ; done 
    done ; done ; done
}

set -x

run_fp16_bf16_tests
run_fp8_tests

if [ $TEST_APPENDKV -eq 1 ] ; then
    run_fp16_appendkv_tests
fi

set +x
