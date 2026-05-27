#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-bench_results/profile}"
DEVICE="${DEVICE:-7}"
DATASET="${DATASET:-example_txt}"
PATTERN="${PATTERN:-random}"
LOOKUPS="${LOOKUPS:-16777216}"
REPEATS="${REPEATS:-1}"
VARIANTS="${VARIANTS:-plain_lut compresslut}"
NCU_SET="${NCU_SET:-basic}"

mkdir -p "${OUT_DIR}"
make bench_cuda

for variant in ${VARIANTS}; do
  case "${variant}" in
    plain_lut) kernel_regex="plain_lut_kernel" ;;
    compresslut) kernel_regex="compresslut_kernel" ;;
    cuda_math_f64) kernel_regex="math_kernel" ;;
    *) echo "unknown variant: ${variant}" >&2; exit 1 ;;
  esac

  base="${OUT_DIR}/${DATASET}_${PATTERN}_${variant}"
  common=(
    benchmarks/bench_cuda
    --repo-root .
    --device "${DEVICE}"
    --dataset "${DATASET}"
    --pattern "${PATTERN}"
    --variant "${variant}"
    --lookups "${LOOKUPS}"
    --repeats "${REPEATS}"
    --out "${base}.csv"
  )

  nsys profile \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --cuda-memory-usage true \
    --output "${base}_nsys" \
    "${common[@]}"

  PYTHONNOUSERSITE=1 ncu \
    --force-overwrite \
    --target-processes all \
    --kernel-name-base function \
    --kernel-name "regex:${kernel_regex}" \
    --set "${NCU_SET}" \
    --export "${base}_ncu" \
    "${common[@]}"
done
