# Instructions for ```example_gemm_add_add_fastgelu_xdl_fp16```

## Run ```example_gemm_add_add_fastgelu_xdl_fp16```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
#arg4 to 11: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD0, StrideD1, StrideE"
./bin/example_gemm_add_add_fastgelu_xdl_fp16 1 1 1
```
