# LLM-Style LUT Pivot

This benchmark simulates grouped codebook lookup traffic of the kind that appears in LUT-quantized inference experiments. Each lookup chooses a group, chooses a code, loads a LUT/codebook value, and multiplies it by an activation.

Variants:

- `plain_per_group`: every logical group owns a physical plain table;
- `plain_shared`: groups map to a smaller set of shared plain tables;
- `compresslut_v2_shared`: groups map to shared CompressedLUT v2 runtime tables.

The important comparison is two-sided. Beating `plain_per_group` shows a memory-footprint opportunity. Matching or beating `plain_shared` would mean the compressed runtime is also competitive against an optimized codebook-sharing implementation.

Run directly:

```bash
make bench_pivot_llm
benchmarks/pivots/llm_lut/bench_llm_lut --quick --device 7 --out /tmp/llm_lut.csv
```
