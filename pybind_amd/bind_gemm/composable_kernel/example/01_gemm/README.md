# Instructions for ```example_gemm_xdl```

## Run ```example_gemm_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
./bin/example_gemm_xdl 0 1 5
```

# Instructions for ```example_gemm_xdl_fp16_streamk_v3```

## Run ```example_gemm_xdl_fp16_streamk_v3```
```bash
arg1: verification (0=no, 1=yes)
arg2: initialization (0=no init, 1=integer value, 2=decimal value)
arg3: time kernel (0=no, 1=yes)
arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC
arg10: stream-k select (-1: default config, 0: all DP, 1: 1-tile SK, 2: 2-tile SK)
arg11: Grid_size(-1 for max occupancy)
bin/example_gemm_xdl_fp16_streamk_v3 1 2 1 3840 4096 4096 4096 4096 4096 1 -1
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {4096, 1}
c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
problem {M:3840, N:4096, K:4096, SA:4096, SB:4096, SC:4096, MP:4032, NP:4096, KRead:4096, KP:4096, AK0:512, BK0:2048, MBlock: 18, NBlock: 16, Stream-K Selection:1, Grid size:-1}
Perf: 0.292022 ms, 441.23 TFlops, 330.348 GB/s, DeviceGemmXdlUniversal<MNPadding, RRR> BlkSize: 256, BlkTile: 224x256x64, WaveTile: 16x16, WaveMap: 7x8, VmemReadVec: 8x8, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3, BlkGemmPipelinePrefetchStages: 2
```
