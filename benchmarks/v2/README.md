# CUDA V2 Experiment

This subfolder contains an isolated second CUDA implementation for CompressLUT runtime evaluation.

V2 keeps the full multi-level CompressLUT artifact as the storage format, but uses a GPU-runtime layout that:

- pre-expands the top-level bias table, removing recursive bias decoding from the CUDA kernel;
- stores runtime compressed arrays as `uint16_t` for the current 12/13-bit benchmark corpus;
- reports both `storage_compressed_bits` and `runtime_bits`, since the runtime form intentionally spends more cached memory to reduce dependent decode work.

Run the full v2 comparison:

```bash
/home/luo00466/miniconda3/envs/py310/bin/python benchmarks/v2/run_v2_benchmarks.py --device 7 --out bench_results/v2/latest
```

Quick smoke:

```bash
/home/luo00466/miniconda3/envs/py310/bin/python benchmarks/v2/run_v2_benchmarks.py --quick --device 7 --out /tmp/clut_v2_quick
```

Outputs:

- `cuda_v2.csv`: raw v2 benchmark rows
- `comparison_vs_baseline.csv`: direct comparison with `bench_results/latest/all_results.csv`
- `summary.md`: compact interpretation table
