#!/bin/bash
EXE="$(find . -name tile_example_gemm_universal -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=2 -warmup=0 -repeat=1'

run_fp16_tests() {
    for batch in 1 2; do
        for m in 128 1024; do
            for n in 128 2048; do
                for k in 32 64; do

                    $EXE -b=$batch -m=$m -n=$n -k=$k -stride_a=0 -stride_b=0 -stride_c=0 -e=1e-5 -prec=fp16 $COMMON_ARGS
                    if [ $? -eq 0 ]; then
                        echo "Success: Test with batch=$batch, m=$m, n=$n, k=$k executed successfully."
                    else
                        echo "Error: Test with batch=$batch, m=$m, n=$n, k=$k failed to execute properly."
                        # Optionally, exit or break if you need to halt further execution
                        # exit 1
                    fi

                done
            done
        done
    done
}

set -x

run_fp16_tests

set +x
