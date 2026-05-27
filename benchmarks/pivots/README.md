# Pivot Benchmarks

This directory contains exploratory benchmark tracks for the directions suggested after the first CPU/CUDA study. The goal is to test where CompressedLUT has a stronger story than "beat a tiny cache-resident plain LUT."

## Directions

- `large_lut/`: scale a single smooth LUT from 4K entries upward and compare plain LUT against the CUDA v2 runtime layout.
- `many_lut/`: evaluate many physical LUTs under random and sequential table selection to stress aggregate cache footprint.
- `llm_lut/`: simulate LLM-style grouped codebook lookup traffic with a per-group plain baseline, a shared plain-codebook baseline, and shared CompressedLUT v2 codebooks.

## Run

Quick run on GPU 7:

```bash
/home/luo00466/miniconda3/envs/py310/bin/python benchmarks/pivots/run_pivot_benchmarks.py --quick --device 7 --out bench_results/pivots/quick
```

Fuller run:

```bash
/home/luo00466/miniconda3/envs/py310/bin/python benchmarks/pivots/run_pivot_benchmarks.py --device 7 --out bench_results/pivots/latest
```

Outputs:

- `large_lut.csv`
- `many_lut.csv`
- `llm_lut.csv`
- `all_pivots.csv`
- `summary.md`

The generated result directory is ignored by git.

The current GPU 7 result snapshot and recommendation are summarized in [../../docs/PIVOT_RESULTS.md](../../docs/PIVOT_RESULTS.md).
