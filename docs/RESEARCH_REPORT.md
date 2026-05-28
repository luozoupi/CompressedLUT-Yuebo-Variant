# CompressedLUT CPU/CUDA Research Report

Date: 2026-05-28

This report summarizes the CUDA/CPU porting work, benchmark progress, measured results, profiling insights, and recommended research direction for extending CompressedLUT beyond its original FPGA-oriented setting.

## Executive Summary

The original CompressedLUT algorithm still delivers strong lossless compression on nonlinear function tables: the 4096-entry benchmark corpus compressed to roughly 12% to 22% of the original packed bit footprint. On GPUs, however, the first direct CUDA translation did not beat a tiny plain LUT because the plain 4096-entry table is only about 16 KiB and remains cache resident.

The important progress was the CUDA v2 runtime layout. V2 keeps the compact multi-level CompressedLUT artifact as the storage format, but expands the top-level bias table and uses a GPU-oriented runtime layout. That improved CUDA CompressedLUT throughput by 3.27x over the first CUDA version and brought it to about 86% of tiny plain-LUT speed, while retaining a compressed-storage story.

The best future direction is not "beat a 4K plain LUT in cache." The best current direction is large or cache-unfriendly LUT workloads, especially random access. In the pivot benchmark, 65K-entry random LUTs showed v2 beating plain LUT by 1.70x to 1.93x. LLM-style grouped LUT traffic remains a secondary direction focused on footprint and memory hierarchy pressure; it is close to optimized shared plain codebooks but is not yet a speed win. The current many-independent-LUT v2 layout is not promising without redesign because descriptor and pointer indirection dominate.

## Work Completed

Implementation changes:

- Exposed an in-memory `CompressedTable` and `CompressedLevel` representation from the original compressor.
- Added CPU decode helpers that match the generated HLS/RTL recurrence.
- Preserved the original CLI and Verilog/HLS generation flow by guarding `main` with `COMPRESSEDLUT_NO_MAIN`.
- Added CPU and CUDA benchmark binaries for plain LUT, CompressedLUT decode, and math-function baselines.
- Added Nsight Systems and Nsight Compute profiling scripts.
- Added CUDA v2 under `benchmarks/v2/`, with a top-level-bias-expanded runtime layout.
- Added pivot benchmark tracks under `benchmarks/pivots/`:
  - `large_lut`: large single LUTs;
  - `many_lut`: many physical LUT instances;
  - `llm_lut`: grouped codebook lookup plus activation multiply.

Key documentation:

- [Yuebo variant overview](YUEBO_VARIANT.md)
- [Pivot result snapshot](PIVOT_RESULTS.md)
- [Benchmark usage](../benchmarks/README.md)
- [CUDA v2 experiment](../benchmarks/v2/README.md)
- [Pivot benchmarks](../benchmarks/pivots/README.md)

## Experimental Setup

Hardware and compiler:

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU 7.
- CUDA target: `-arch=sm_120`.
- CUDA build: `nvcc -O3 -std=c++17`.
- CPU build: `g++ -O3 -std=c++17 -pthread`.
- Python environment: `/home/luo00466/miniconda3/envs/py310`.

Primary corpus:

- `example_txt`
- `exp_x_minus_1`
- `log1p_x`
- `sigmoid_8x`
- `sin_half_pi_x`
- `sqrt_x`
- `tanh_3x`

Most baseline rows used 4096-entry LUTs, 16,777,216 lookups per CUDA row, five repeats, and both sequential and random address streams.

## Compression Results

| dataset | original bits | compressed bits | ratio | levels |
| --- | ---: | ---: | ---: | ---: |
| `example_txt` | 49,152 | 10,568 | 0.215 | 4 |
| `exp_x_minus_1` | 49,152 | 5,970 | 0.121 | 5 |
| `log1p_x` | 49,152 | 5,848 | 0.119 | 5 |
| `sigmoid_8x` | 49,152 | 8,982 | 0.183 | 5 |
| `sin_half_pi_x` | 53,248 | 8,716 | 0.164 | 4 |
| `sqrt_x` | 49,152 | 10,568 | 0.215 | 4 |
| `tanh_3x` | 49,152 | 10,970 | 0.223 | 5 |

Takeaway: compression is strong and consistent. The average ratio across this corpus is about 0.177, roughly a 5.6x reduction versus packed table bits.

## Baseline CPU/CUDA Results

Observed throughput ranges:

| backend / variant | throughput range | interpretation |
| --- | ---: | --- |
| CPU CompressedLUT | about 1.7B to 2.2B lookups/s | Fast, but slower than CPU plain LUT. |
| CUDA CompressedLUT v1 | about 75B to 85B lookups/s | Much faster than CPU, but only about 0.26x tiny CUDA plain LUT. |
| CUDA plain LUT | about 290B to 311B lookups/s | Extremely strong because the 4K table is cache resident. |
| CUDA math functions | function dependent | CUDA CompressedLUT v1 is about 3.3x to 4.0x faster on average. |

Plain CUDA LUT baseline:

- `uint32_t` table in global memory;
- `uint32_t` address stream;
- one thread per lookup;
- grid-stride loop;
- read-only `__ldg` table loads;
- one output write per lookup.

Takeaway: the baseline is simple but strong. For a 16 KiB table, the GPU sees a mostly cache-resident lookup problem. CompressedLUT cannot rely on memory savings to win until the table footprint or access pattern stresses cache/memory.

## CUDA V2 Results

V2 design:

- Keep the full multi-level CompressedLUT artifact as the compact storage format.
- Use a top-level-only GPU runtime layout for the current corpus.
- Pre-expand the top-level bias table to remove recursive bias decoding from the kernel.
- Store runtime arrays as `uint16_t` for 12/13-bit output tables.

Measured result:

| metric | result |
| --- | ---: |
| Average speedup over CUDA v1 | 3.273x |
| Speedup range over CUDA v1 | 3.069x to 3.530x |
| Average v2/plain tiny-LUT speed | about 0.86x |
| Average v2/CUDA-math speed | about 11.8x |
| Runtime bits vs compact storage bits | about 1.27x to 1.68x larger |

Takeaway: v2 confirms that the direct CUDA translation was not the right GPU runtime form. Spending some runtime memory to remove dependency chains is worthwhile. V2 is close enough to tiny plain LUT speed that further work should target workloads where compressed footprint matters.

## Profiling Insights

Nsight Compute observations from sampled kernels:

| kernel | registers/thread | achieved occupancy | main pressure |
| --- | ---: | ---: | --- |
| Plain LUT | about 20 | about 80% | memory/cache load path, but table is tiny |
| CUDA CompressedLUT v1 | about 48 | about 76% | dependent decode, register pressure, L1/TEX pressure |
| CUDA CompressedLUT v2 | about 36 | about 84% | less dependency pressure; closer to cache-bandwidth behavior |

Takeaway: v1 is not primarily DRAM-bound on tiny tables. It loses to plain LUT because each lookup does more dependent work and uses more registers. V2 improves by reducing recursive decode and increasing occupancy.

## Pivot Benchmark Results

The pivot suite tests where compressed LUTs may matter more than the 4K cache-resident case.

### Pivot Scorecard

| direction | speed result | footprint result | readout |
| --- | ---: | ---: | --- |
| Large LUT | 1.096x average v2/plain in quick sweep | runtime bytes 0.679x plain bytes | Best current pivot. |
| Many physical LUTs | 0.460x average v2/plain | runtime bytes 0.749x plain bytes | Not good with current descriptor-heavy layout. |
| LLM LUT vs per-group plain | 0.934x average v2/plain | runtime bytes 0.0293x per-group plain bytes | Good footprint story, not consistent speed win. |
| LLM LUT vs shared plain | 0.744x average v2/plain | runtime bytes 0.749x shared plain bytes | Shared plain remains the speed baseline. |

### Large LUT Details

| case | pattern | plain Mops/s | v2 Mops/s | v2/plain |
| --- | --- | ---: | ---: | ---: |
| `sqrt_f16` | random | 88,682 | 171,336 | 1.93x |
| `sigmoid_f16` | random | 96,518 | 164,148 | 1.70x |
| `sqrt_f16` | sequential | 253,035 | 190,097 | 0.75x |
| `sigmoid_f16` | sequential | 266,678 | 190,650 | 0.72x |

Interpretation: compression helps when random access makes a larger plain LUT less cache friendly. Sequential plain LUT remains very hard to beat.

### Many-LUT Targeted Check

For 1024 physical 4096-entry LUTs:

| pattern | plain Mops/s | v2 Mops/s | v2/plain |
| --- | ---: | ---: | ---: |
| sequential | 107,107 | 17,152 | 0.16x |
| random | 92,174 | 13,855 | 0.15x |

Interpretation: many independent compressed runtime objects are not promising in the current design. The per-lookup descriptor load and pointer chasing overpower the memory savings.

### LLM-Style Targeted Check

For 4096 groups, 16 shared codebooks, and 4096 entries per codebook:

| pattern | plain per-group Mops/s | plain shared Mops/s | v2 shared Mops/s | v2/per-group | v2/shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| sequential | 86,760 | 90,754 | 82,552 | 0.95x | 0.91x |
| random | 75,481 | 77,147 | 73,122 | 0.97x | 0.95x |

Interpretation: this is close enough to keep as a footprint direction, but the correct speed baseline is shared plain codebooks, not per-group duplicated plain codebooks.

## Research Interpretation

The original paper's strongest claim is resource efficiency: reduced table footprint while preserving exact values. That maps naturally to FPGA/ASIC logic and memory resources. On a modern GPU, the same advantage appears only when memory hierarchy pressure is real. For tiny tables, the GPU turns plain lookup into a cache-resident operation, and extra decode work is a liability.

The CUDA results suggest a two-format model:

- Storage format: keep CompressedLUT compact, multi-level, and lossless.
- Runtime format: specialize for the target hardware, even if it expands selected metadata or collapses levels.

This is the main lesson from v2. The storage artifact can remain compact, while the runtime representation should be autotuned for GPU latency, register pressure, cache behavior, and access pattern.

## Recommended Future Direction

Priority 1: large random-access LUTs.

- Expand `large_lut` to 2^18, 2^20, and 2^22 entries where compression time is acceptable.
- Add more functions and less smooth generated tables to understand when compression ratio degrades.
- Use Nsight Compute on the large-LUT cases where v2 beats plain to confirm whether the win comes from DRAM traffic, cache hit rate, or memory-level parallelism.
- Add a table-size sweep plot: table bytes vs throughput vs compression ratio.

Priority 2: runtime-layout autotuning.

- Generate specialized kernels per table shape instead of metadata-driven generic kernels.
- Pack descriptors into structure-of-arrays form to avoid per-lookup pointer-heavy loads.
- Place small descriptors in constant memory.
- Test 16-bit, 32-bit, and bit-packed runtime arrays separately; current v2 uses simple `uint16_t` arrays, which is fast but not always compact.
- Add variants that pre-expand `idx`, `rsh`, or bias selectively based on profile results.

Priority 3: LLM/codebook direction as a memory-footprint project.

- Treat shared plain codebooks as the primary speed baseline.
- Measure end-to-end kernels where LUT traffic competes with activation/weight traffic, not isolated lookup only.
- Explore whether compressed storage helps model loading, KV/cache-adjacent tables, or rarely reused per-layer codebooks.
- Consider a hybrid design: compressed storage in global memory, decompressed/shared runtime tiles in shared memory or persistent cache.

Priority 4: many-LUT redesign only if locality can be imposed.

- Current many-independent-v2 is a poor direction.
- Revisit only with grouped/sorted table IDs, codebook sharing, descriptor flattening, or batched kernels where each block works on one table.
- Avoid per-lookup random descriptor and pointer chasing.

Priority 5: hardware/resource track.

- Preserve the FPGA/ASIC story because CompressedLUT's original advantage is strongest where memory blocks, LUTs, and routing resources are first-class costs.
- Add an estimator that reports compressed bits, runtime bits, descriptor bytes, and equivalent SRAM/BRAM footprint for each table.
- Use the GPU work to identify runtime layouts, not as the only target platform.

## Immediate Next Experiments

1. Run a larger `large_lut` sweep with `f_in=18,20` for `sqrt`, `sigmoid`, `exp`, and `tanh`.
2. Profile `sqrt_f16 random` and `sigmoid_f16 random` with NCU to confirm the source of the v2 win.
3. Implement a v3 specialized large-LUT kernel with compile-time level parameters and no generic descriptor path.
4. Add a block-local many-LUT benchmark where each CTA processes one table ID to test whether locality rescues the many-LUT direction.
5. Build an end-to-end LLM-style microkernel that includes code lookup, activation multiply, accumulation, and realistic reuse patterns.

## Limitations

- The current benchmark corpus is synthetic and focused on smooth nonlinear functions.
- The LLM-style benchmark is a microbenchmark, not an integrated transformer inference kernel.
- Compression build time is measured but not optimized.
- Results are from one Blackwell server GPU and should be repeated on other architectures.
- Generated CSV results live under `bench_results/` and are not committed; committed docs preserve the key numbers.

## Bottom Line

The research should pivot away from trying to beat a tiny cache-resident plain LUT. The promising path is a compression-aware LUT system with separate storage and runtime formats. The strongest immediate direction is large, random-access, memory-hierarchy-sensitive LUT evaluation. The LLM/codebook direction is worth keeping for footprint and end-to-end memory pressure, but it must compete against shared plain codebooks and should not be framed as an isolated lookup-speed win yet.

## Reference

Khataei, Alireza, and Kia Bazargan. "CompressedLUT: An Open Source Tool for Lossless Compression of Lookup Tables for Function Evaluation and Beyond." FPGA 2024. DOI: https://doi.org/10.1145/3626202.3637575
