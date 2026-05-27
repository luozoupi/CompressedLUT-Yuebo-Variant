# Yuebo CompressedLUT CPU/CUDA Variant

This branch keeps the upstream CompressedLUT command-line tool intact and adds software benchmarking hooks around the compression artifact. The purpose is to evaluate whether the lossless LUT compression scheme is useful on server CPUs and NVIDIA GPUs, not only in the FPGA/HLS setting targeted by the original paper.

## What Changed

- `compressedlut.h` and `compressedlut.cpp` now expose `CompressedTable`, `CompressedLevel`, `compress_table_artifact`, and CPU `decode` helpers.
- The original CLI `main` is guarded by `COMPRESSEDLUT_NO_MAIN`, so benchmark binaries can link the compressor without duplicating the executable entry point.
- `benchmarks/bench_cpu.cpp` measures CPU plain LUT, CompressedLUT decode, and libm evaluation.
- `benchmarks/bench_cuda.cu` measures CUDA plain LUT, CompressedLUT decode, and CUDA math evaluation.
- `benchmarks/profile_cuda.sh` and `benchmarks/profile_all_cuda.sh` run focused Nsight Systems and Nsight Compute profiles.
- `benchmarks/v2/bench_cuda_v2.cu` adds a GPU-oriented runtime layout that pre-expands the top-level bias table while preserving the compact storage artifact.
- `benchmarks/pivots/` adds follow-up experiments for large LUTs, many physical LUTs, and LLM-style grouped codebook traffic.

Generated benchmark binaries and result directories are ignored by git. Re-run the commands below to regenerate local results.

## Build

```bash
make all bench_cpu bench_cuda bench_cuda_v2
make bench_pivots
```

The CUDA targets default to `-arch=sm_120`, matching the Blackwell GPUs on the benchmark server.

## Full Benchmark

Run the full CPU and CUDA benchmark suite on GPU 7:

```bash
/home/luo00466/miniconda3/envs/py310/bin/python benchmarks/run_benchmarks.py --device 7 --out bench_results/latest
```

Run the CUDA v2 comparison:

```bash
/home/luo00466/miniconda3/envs/py310/bin/python benchmarks/v2/run_v2_benchmarks.py --device 7 --out bench_results/v2/latest
```

Run the pivot experiments:

```bash
/home/luo00466/miniconda3/envs/py310/bin/python benchmarks/pivots/run_pivot_benchmarks.py --device 7 --out bench_results/pivots/latest
```

The runners write CSV files and compact Markdown summaries under `bench_results/`. Those files are treated as generated artifacts and are not committed.

## Benchmark Scope

The current corpus uses seven 4096-entry LUTs:

- `example_txt`
- `exp_x_minus_1`
- `log1p_x`
- `sigmoid_8x`
- `sin_half_pi_x`
- `sqrt_x`
- `tanh_3x`

Each CUDA benchmark row uses 16,777,216 lookups, five repeats, sequential and random address patterns, and a 256-thread block size. CPU runs sweep thread counts from 1 to 128.

## Plain LUT Baseline

The CUDA plain-LUT baseline is intentionally simple and strong for small tables:

- `uint32_t` lookup table in device global memory;
- `uint32_t` address stream;
- one thread per lookup with a grid-stride loop;
- `__ldg(table + address)` read-only loads;
- one global output write per lookup;
- compiled with `nvcc -O3 -std=c++17 -arch=sm_120`.

It does not use cuBLAS, CUTLASS, CUDA graphs, texture memory, shared-memory staging, TMA, or cooperative groups. For 4096-entry 12/13-bit LUTs, the plain table is only about 16 KiB, so it is cache resident and is a very difficult speed baseline to beat.

## Current Results

On the measured Blackwell server GPU, the original CUDA CompressedLUT implementation is much faster than evaluating nonlinear functions directly, but slower than the tiny cache-resident plain LUT:

- CPU CompressedLUT: about 1.7B to 2.2B lookups/s.
- CUDA CompressedLUT: about 75B to 85B lookups/s.
- CUDA plain LUT: about 290B to 311B lookups/s.
- CUDA CompressedLUT vs CUDA math: about 3.3x to 4.0x faster on average.
- CUDA CompressedLUT vs CUDA plain LUT: about 0.26x as fast.

Compression is still strong. Across the benchmark corpus, compressed storage ratios range from about 0.119 to 0.223 of the original bit footprint.

## CUDA V2 Result

V2 spends more runtime memory to reduce dependent decode work:

- pre-expanded top-level bias table;
- `uint16_t` runtime arrays for the current 12/13-bit corpus;
- storage artifact remains the compact multi-level CompressedLUT form.

Measured outcome:

- average v2 speedup over original CUDA CompressedLUT: 3.27x;
- speedup range: 3.07x to 3.53x;
- v2 vs CUDA plain LUT: about 0.86x on average;
- v2 vs CUDA math: about 11.8x faster on average;
- runtime footprint is larger than the compact storage artifact, about 1.27x to 1.68x the compressed bit count.

This is a useful tradeoff point: the GPU runtime form is close to plain-LUT speed while still retaining much of the storage compression advantage.

## Profiling

Profile one LUT and address pattern:

```bash
OUT_DIR=bench_results/profile DEVICE=7 DATASET=example_txt PATTERN=random \
  VARIANTS="plain_lut compresslut cuda_math_f64" benchmarks/profile_cuda.sh
```

Sweep several LUTs:

```bash
OUT_DIR=bench_results/profile_all DEVICE=7 DATASETS="example_txt exp_x_minus_1 sigmoid_8x" \
  PATTERNS="sequential random" VARIANTS="plain_lut compresslut" benchmarks/profile_all_cuda.sh
```

Nsight Compute observations from the current kernels:

- Original CUDA CompressedLUT uses about 48 registers per thread, with achieved occupancy around 76% in the sampled profile.
- Plain LUT uses about 20 registers per thread, with achieved occupancy around 80%.
- V2 drops to about 36 registers per thread and achieved occupancy around 84%.
- Original CompressedLUT is not primarily DRAM-bound for these small tables; the bottleneck is dependent decode work, register pressure, and L1/TEX pressure.
- V2 removes much of the recursive decode dependency and moves the bottleneck closer to cache bandwidth.

## Interpretation

For tiny LUTs that fit comfortably in GPU cache, the plain LUT baseline is near ideal. The current evidence does not support making "faster than plain LUT for 4K tables" the central research goal.

The stronger direction is compression-aware LUT systems for memory-capacity and memory-bandwidth constrained workloads:

- many LUTs or much larger LUTs where plain tables stop being cache resident;
- LUT-quantized inference and LLM-scale deployment, where memory traffic and model footprint dominate;
- autotuned storage/runtime layouts that preserve compact storage but specialize runtime decode for the target GPU;
- FPGA/ASIC settings where the original paper's throughput-per-resource argument maps directly to hardware cost.

The CUDA v2 result supports this pivot: it nearly recovers plain-LUT throughput on small tables while preserving a compressed storage story, making larger memory-bound experiments the next important benchmark target.

## Pivot Experiments

The pivot suite turns that recommendation into runnable tests:

- `large_lut`: grows a single smooth function LUT and checks when plain-LUT cache residency stops dominating.
- `many_lut`: creates many physical LUT instances to stress aggregate cache footprint and memory locality.
- `llm_lut`: simulates grouped codebook lookup plus activation multiply, with both per-group and shared-codebook plain baselines.

Use [benchmarks/pivots/README.md](../benchmarks/pivots/README.md) for the command line and result files. The current GPU 7 result snapshot is in [docs/PIVOT_RESULTS.md](PIVOT_RESULTS.md).
