#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def run(cmd, cwd, env):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def read_csv(path):
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def markdown_table(df):
    if df.empty:
        return ""
    columns = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
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


def write_summary(df, out_dir):
    summary_path = out_dir / "summary.md"
    ok = df[df["status"] == "ok"].copy()
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# CompressedLUT Benchmark Summary\n\n")
        if df.empty:
            f.write("No benchmark rows were produced.\n")
            return

        compression = (
            df[["dataset", "initial_bits", "compressed_bits", "compression_ratio", "levels", "entries", "output_bits", "compression_ms"]]
            .drop_duplicates("dataset")
            .sort_values("dataset")
        )
        f.write("## Compression\n\n")
        f.write(markdown_table(compression))
        f.write("\n\n")

        if ok.empty:
            f.write("No successful throughput rows were produced.\n")
            return

        best = (
            ok.sort_values("mops", ascending=False)
            .groupby(["dataset", "backend", "variant", "address_pattern"], as_index=False)
            .first()
            [["dataset", "backend", "variant", "address_pattern", "threads", "mops", "ns_per_lookup", "median_ms"]]
            .sort_values(["dataset", "backend", "variant", "address_pattern"])
        )
        f.write("## Best Throughput Per Variant\n\n")
        f.write(markdown_table(best))
        f.write("\n\n")

        failures = df[df["status"] != "ok"]
        if not failures.empty:
            f.write("## Non-OK Rows\n\n")
            f.write(markdown_table(failures[["dataset", "backend", "variant", "address_pattern", "status"]].drop_duplicates()))
            f.write("\n")


def write_plot(df, out_dir):
    ok = df[(df["status"] == "ok") & (df["address_pattern"] == "sequential")].copy()
    if ok.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = (
        ok.sort_values("mops", ascending=False)
        .groupby(["dataset", "backend", "variant"], as_index=False)
        .first()
    )
    best["label"] = best["backend"] + ":" + best["variant"]

    datasets = list(best["dataset"].drop_duplicates())
    labels = list(best["label"].drop_duplicates())
    width = 0.8 / max(1, len(labels))
    x = list(range(len(datasets)))

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.4), 4.8))
    for i, label in enumerate(labels):
        values = []
        for dataset in datasets:
            row = best[(best["dataset"] == dataset) & (best["label"] == label)]
            values.append(float(row["mops"].iloc[0]) if not row.empty else 0.0)
        offsets = [v - 0.4 + width * (i + 0.5) for v in x]
        ax.bar(offsets, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Mlookups/s")
    ax.set_title("Sequential Lookup Throughput")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_sequential.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build and run CPU/CUDA CompressLUT benchmarks.")
    parser.add_argument("--out", default="bench_results/latest", help="Output directory for CSVs and report.")
    parser.add_argument("--device", default="7", help="CUDA device ordinal for the CUDA benchmark.")
    parser.add_argument("--lookups", default=None, help="Lookups per benchmark row.")
    parser.add_argument("--repeats", default=None, help="Timed repeats per benchmark row.")
    parser.add_argument("--threads", default="1,2,4,8,16,32,64,128", help="CPU thread counts.")
    parser.add_argument("--quick", action="store_true", help="Use smaller tables, fewer lookups, and fewer repeats.")
    parser.add_argument("--cpu-only", action="store_true", help="Run only the CPU benchmark.")
    parser.add_argument("--cuda-only", action="store_true", help="Run only the CUDA benchmark.")
    parser.add_argument("--skip-build", action="store_true", help="Do not invoke make before running.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(out_dir / "mplconfig")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = env["MPLCONFIGDIR"]

    if not args.skip_build:
        if not args.cuda_only:
            run(["make", "bench_cpu"], repo_root, env)
        if not args.cpu_only:
            run(["make", "bench_cuda"], repo_root, env)

    common = ["--repo-root", str(repo_root)]
    if args.quick:
        common.append("--quick")
    else:
        if args.lookups is not None:
            common.extend(["--lookups", args.lookups])
        if args.repeats is not None:
            common.extend(["--repeats", args.repeats])

    csvs = []
    if not args.cuda_only:
        cpu_csv = out_dir / "cpu.csv"
        cmd = [str(repo_root / "benchmarks" / "bench_cpu"), *common, "--threads", args.threads, "--out", str(cpu_csv)]
        run(cmd, repo_root, env)
        csvs.append(cpu_csv)

    if not args.cpu_only:
        cuda_csv = out_dir / "cuda.csv"
        cmd = [str(repo_root / "benchmarks" / "bench_cuda"), *common, "--device", args.device, "--out", str(cuda_csv)]
        run(cmd, repo_root, env)
        csvs.append(cuda_csv)

    import pandas as pd

    frames = [read_csv(path) for path in csvs]
    frames = [frame for frame in frames if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(out_dir / "all_results.csv", index=False)
    write_summary(combined, out_dir)
    write_plot(combined, out_dir)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
