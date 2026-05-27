# Pivot Benchmark Results

These are the current pivot results from the Blackwell server GPU run on GPU 7. Raw CSV files were generated under `bench_results/pivots/`, which is intentionally ignored by git.

Commands used:

```bash
/home/luo00466/miniconda3/envs/py310/bin/python benchmarks/pivots/run_pivot_benchmarks.py --quick --device 7 --out bench_results/pivots/quick
benchmarks/pivots/many_lut/bench_many_lut --device 7 --table-counts 1024 --lookups 16777216 --repeats 3 --out bench_results/pivots/targeted_many1024.csv
benchmarks/pivots/llm_lut/bench_llm_lut --device 7 --group-counts 4096 --lookups 16777216 --repeats 3 --out bench_results/pivots/targeted_llm4096.csv
```

## Quick Scorecard

| direction | avg speed ratio | runtime bytes / plain bytes | storage bits / plain 32-bit bits | readout |
| --- | --- | --- | --- | --- |
| Large LUT | 1.096x v2/plain | 0.679 | 0.051 | Best current pivot. Random 65K-entry LUTs beat plain LUT because plain loses cache locality. |
| Many physical LUTs | 0.460x v2/plain | 0.749 | 0.100 | Not good in this runtime layout. Descriptor and pointer chasing dominate. |
| LLM LUT vs per-group plain | 0.934x v2/plain | 0.0293 | n/a | Useful memory-footprint story against naive per-group tables, but not consistently faster. |
| LLM LUT vs shared plain | 0.744x v2/plain | 0.749 | n/a | Shared plain codebooks remain the stronger speed baseline. |

## Large LUT Details

| case | pattern | plain Mops/s | v2 Mops/s | v2/plain |
| --- | --- | --- | --- | --- |
| `sqrt_f16` | random | 88,682 | 171,336 | 1.93x |
| `sigmoid_f16` | random | 96,518 | 164,148 | 1.70x |
| `sqrt_f16` | sequential | 253,035 | 190,097 | 0.75x |
| `sigmoid_f16` | sequential | 266,678 | 190,650 | 0.72x |

This is the strongest pivot signal: compression helps random large-LUT access once the plain table is less cache-resident. Sequential plain LUT remains very strong.

## Many-LUT Targeted Check

For 1024 physical 4096-entry LUTs and 16,777,216 lookups:

| pattern | plain Mops/s | v2 Mops/s | v2/plain |
| --- | --- | --- | --- |
| sequential | 107,107 | 17,152 | 0.16x |
| random | 92,174 | 13,855 | 0.15x |

This direction should not continue as "many independent compressed runtime objects" unless the runtime layout is redesigned to remove descriptor/pointer indirection and improve table locality.

## LLM-Style Targeted Check

For 4096 groups, 16 shared codebooks, 4096 entries per codebook, and 16,777,216 fused lookup-multiply operations:

| pattern | plain per-group Mops/s | plain shared Mops/s | v2 shared Mops/s | v2/per-group | v2/shared |
| --- | --- | --- | --- | --- | --- |
| sequential | 86,760 | 90,754 | 82,552 | 0.95x | 0.91x |
| random | 75,481 | 77,147 | 73,122 | 0.97x | 0.95x |

This is close enough to keep as a memory-footprint direction, but the baseline should be an optimized shared-codebook implementation. The research angle should be model footprint and memory hierarchy pressure, not raw lookup speed.

## Recommendation

Continue with `large_lut` first, especially random-access large tables and table sets that exceed cache residency. Keep `llm_lut` as a secondary track focused on model-footprint and bandwidth-limited inference. Do not invest further in the current many-physical-LUT v2 layout without a layout redesign.
