# Large LUT Pivot

This benchmark tests the case where a single LUT grows beyond the original 4096-entry corpus. It compares:

- `plain_lut`: direct `uint32_t` global-memory table lookup;
- `compresslut_v2`: the GPU-oriented CompressedLUT v2 runtime layout.

The default quick run uses 4096-entry and 65536-entry tables for `sqrt` and `sigmoid`. The full run adds 262144-entry tables.

Run directly:

```bash
make bench_pivot_large
benchmarks/pivots/large_lut/bench_large_lut --quick --device 7 --out /tmp/large_lut.csv
```
