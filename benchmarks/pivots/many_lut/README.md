# Many LUT Pivot

This benchmark tests aggregate LUT footprint. It creates many physical LUT instances, then uses random and sequential table selection.

Variants:

- `plain_lut`: flat `table_count * entries` plain table storage;
- `compresslut_v2`: one physical CompressedLUT v2 runtime table per physical LUT instance.

This is the closest GPU analogue to a memory-capacity or cache-pressure argument: the plain baseline gets worse only when the aggregate table set stops behaving like a tiny resident cache object.

Run directly:

```bash
make bench_pivot_many
benchmarks/pivots/many_lut/bench_many_lut --quick --device 7 --out /tmp/many_lut.csv
```
