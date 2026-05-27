#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-example_txt exp_x_minus_1 sigmoid_8x sqrt_x sin_half_pi_x log1p_x tanh_3x}"
PATTERNS="${PATTERNS:-random}"
VARIANTS="${VARIANTS:-plain_lut compresslut}"
OUT_ROOT="${OUT_ROOT:-bench_results/profile}"

for dataset in ${DATASETS}; do
  for pattern in ${PATTERNS}; do
    OUT_DIR="${OUT_ROOT}/${dataset}_${pattern}" \
    DATASET="${dataset}" \
    PATTERN="${pattern}" \
    VARIANTS="${VARIANTS}" \
      benchmarks/profile_cuda.sh
  done
done
