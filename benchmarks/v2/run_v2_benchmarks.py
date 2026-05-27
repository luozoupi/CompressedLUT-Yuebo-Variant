#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


def run(cmd, cwd, env):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def markdown_table(df):
    if df.empty:
        return ""
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_summary(v2, baseline, out_dir):
    path = out_dir / "summary.md"
    ok = v2[v2["status"] == "ok"].copy()
    with path.open("w", encoding="utf-8") as f:
        f.write("# CompressedLUT CUDA V2 Benchmark Summary\n\n")
        f.write("V2 uses a GPU-runtime layout with top-level bias pre-expanded and 16-bit compressed arrays.\n\n")

        storage = (
            ok[[
                "dataset",
                "initial_bits",
                "storage_compressed_bits",
                "runtime_bits",
                "runtime_bytes",
                "storage_ratio",
                "runtime_ratio",
                "levels",
                "selected_levels",
            ]]
            .drop_duplicates("dataset")
            .sort_values("dataset")
        )
        f.write("## Storage vs Runtime Footprint\n\n")
        f.write(markdown_table(storage))
        f.write("\n\n")

        if baseline is not None and not baseline.empty:
            base = baseline[
                (baseline["backend"] == "cuda")
                & (baseline["variant"] == "compresslut")
                & (baseline["status"] == "ok")
            ][["dataset", "address_pattern", "entries", "mops", "ns_per_lookup", "compressed_bits"]].rename(
                columns={
                    "mops": "baseline_mops",
                    "ns_per_lookup": "baseline_ns",
                    "compressed_bits": "baseline_runtime_bits",
                }
            )
            comp = ok.merge(base, on=["dataset", "address_pattern", "entries"], how="left")
            comp["speedup_vs_baseline"] = comp["mops"] / comp["baseline_mops"]
            comp["runtime_bits_vs_baseline"] = comp["runtime_bits"] / comp["baseline_runtime_bits"]
            comp = comp[[
                "dataset",
                "address_pattern",
                "mops",
                "baseline_mops",
                "speedup_vs_baseline",
                "ns_per_lookup",
                "baseline_ns",
                "storage_ratio",
                "runtime_ratio",
                "runtime_bits_vs_baseline",
            ]].sort_values(["dataset", "address_pattern"])
            comp.to_csv(out_dir / "comparison_vs_baseline.csv", index=False)
            f.write("## V2 vs Existing CUDA CompressLUT\n\n")
            f.write(markdown_table(comp))
            f.write("\n\n")
            f.write(
                f"Average speedup: {comp['speedup_vs_baseline'].mean():.3f}x; "
                f"min: {comp['speedup_vs_baseline'].min():.3f}x; "
                f"max: {comp['speedup_vs_baseline'].max():.3f}x.\n"
            )
        else:
            f.write("No baseline CSV found for direct comparison.\n")


def main():
    parser = argparse.ArgumentParser(description="Run CUDA v2 CompressLUT benchmark and compare with baseline.")
    parser.add_argument("--out", default="bench_results/v2/latest")
    parser.add_argument("--baseline", default="bench_results/latest/all_results.csv")
    parser.add_argument("--device", default="7")
    parser.add_argument("--lookups", default=None)
    parser.add_argument("--repeats", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if not args.skip_build:
        run(["make", "bench_cuda_v2"], repo_root, env)

    cmd = [
        str(repo_root / "benchmarks" / "v2" / "bench_cuda_v2"),
        "--repo-root",
        str(repo_root),
        "--device",
        args.device,
        "--out",
        str(out_dir / "cuda_v2.csv"),
    ]
    if args.quick:
        cmd.append("--quick")
    else:
        if args.lookups is not None:
            cmd.extend(["--lookups", args.lookups])
        if args.repeats is not None:
            cmd.extend(["--repeats", args.repeats])
    if args.dataset is not None:
        cmd.extend(["--dataset", args.dataset])
    if args.pattern is not None:
        cmd.extend(["--pattern", args.pattern])

    run(cmd, repo_root, env)

    v2 = pd.read_csv(out_dir / "cuda_v2.csv")
    baseline_path = (repo_root / args.baseline).resolve() if not Path(args.baseline).is_absolute() else Path(args.baseline)
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None
    write_summary(v2, baseline, out_dir)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
