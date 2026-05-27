#!/usr/bin/env python3
import argparse
import csv
import subprocess
from pathlib import Path


def run(cmd, cwd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def index_rows(rows):
    return {(r["direction"], r["case_name"], r["address_pattern"], r["variant"]): r for r in rows}


def markdown_table(headers, rows):
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def summarize_large_or_many(direction, rows):
    idx = index_rows(rows)
    summary = []
    speedups = []
    runtime_fracs = []
    storage_fracs = []
    keys = sorted({(r["case_name"], r["address_pattern"]) for r in rows if r["direction"] == direction})
    for case_name, pattern in keys:
        plain = idx.get((direction, case_name, pattern, "plain_lut"))
        v2 = idx.get((direction, case_name, pattern, "compresslut_v2"))
        if not plain or not v2:
            continue
        plain_mops = fnum(plain, "mops")
        v2_mops = fnum(v2, "mops")
        speedup = v2_mops / plain_mops if plain_mops else 0.0
        runtime_frac = fnum(v2, "runtime_bytes") / fnum(plain, "plain_bytes") if fnum(plain, "plain_bytes") else 0.0
        storage_frac = fnum(v2, "compressed_storage_bits") / (8.0 * fnum(plain, "plain_bytes")) if fnum(plain, "plain_bytes") else 0.0
        speedups.append(speedup)
        runtime_fracs.append(runtime_frac)
        storage_fracs.append(storage_frac)
        summary.append([
            case_name,
            pattern,
            f"{plain_mops:.1f}",
            f"{v2_mops:.1f}",
            f"{speedup:.3f}x",
            f"{runtime_frac:.3f}",
            f"{storage_frac:.3f}",
        ])
    avg = {
        "speedup": sum(speedups) / len(speedups) if speedups else 0.0,
        "runtime_frac": sum(runtime_fracs) / len(runtime_fracs) if runtime_fracs else 0.0,
        "storage_frac": sum(storage_fracs) / len(storage_fracs) if storage_fracs else 0.0,
    }
    return avg, summary


def summarize_llm(rows):
    idx = index_rows(rows)
    summary = []
    speed_pg = []
    speed_shared = []
    memory_pg = []
    memory_shared = []
    keys = sorted({(r["case_name"], r["address_pattern"]) for r in rows if r["direction"] == "llm_lut"})
    for case_name, pattern in keys:
        per_group = idx.get(("llm_lut", case_name, pattern, "plain_per_group"))
        shared = idx.get(("llm_lut", case_name, pattern, "plain_shared"))
        v2 = idx.get(("llm_lut", case_name, pattern, "compresslut_v2_shared"))
        if not per_group or not shared or not v2:
            continue
        v2_mops = fnum(v2, "mops")
        pg_mops = fnum(per_group, "mops")
        shared_mops = fnum(shared, "mops")
        s_pg = v2_mops / pg_mops if pg_mops else 0.0
        s_shared = v2_mops / shared_mops if shared_mops else 0.0
        m_pg = fnum(v2, "runtime_bytes") / fnum(per_group, "plain_bytes") if fnum(per_group, "plain_bytes") else 0.0
        m_shared = fnum(v2, "runtime_bytes") / fnum(shared, "plain_bytes") if fnum(shared, "plain_bytes") else 0.0
        speed_pg.append(s_pg)
        speed_shared.append(s_shared)
        memory_pg.append(m_pg)
        memory_shared.append(m_shared)
        summary.append([
            case_name,
            pattern,
            f"{pg_mops:.1f}",
            f"{shared_mops:.1f}",
            f"{v2_mops:.1f}",
            f"{s_pg:.3f}x",
            f"{s_shared:.3f}x",
            f"{m_pg:.4f}",
            f"{m_shared:.3f}",
        ])
    avg = {
        "speedup_per_group": sum(speed_pg) / len(speed_pg) if speed_pg else 0.0,
        "speedup_shared": sum(speed_shared) / len(speed_shared) if speed_shared else 0.0,
        "runtime_frac_per_group": sum(memory_pg) / len(memory_pg) if memory_pg else 0.0,
        "runtime_frac_shared": sum(memory_shared) / len(memory_shared) if memory_shared else 0.0,
    }
    return avg, summary


def write_summary(path, rows, quick):
    large_avg, large_rows = summarize_large_or_many("large_lut", rows)
    many_avg, many_rows = summarize_large_or_many("many_lut", rows)
    llm_avg, llm_rows = summarize_llm(rows)

    with path.open("w", encoding="utf-8") as f:
        f.write("# CompressedLUT Pivot Benchmark Summary\n\n")
        f.write("Mode: quick\n\n" if quick else "Mode: full\n\n")

        f.write("## Scorecard\n\n")
        score_rows = [
            ["large_lut", f"{large_avg['speedup']:.3f}x", f"{large_avg['runtime_frac']:.3f}", f"{large_avg['storage_frac']:.3f}", "Tests whether compression matters as a single LUT grows."],
            ["many_lut", f"{many_avg['speedup']:.3f}x", f"{many_avg['runtime_frac']:.3f}", f"{many_avg['storage_frac']:.3f}", "Tests aggregate cache and memory pressure from many LUTs."],
            ["llm_lut vs per-group plain", f"{llm_avg['speedup_per_group']:.3f}x", f"{llm_avg['runtime_frac_per_group']:.4f}", "n/a", "Tests LLM-like groups without ideal codebook sharing."],
            ["llm_lut vs shared plain", f"{llm_avg['speedup_shared']:.3f}x", f"{llm_avg['runtime_frac_shared']:.3f}", "n/a", "Tests against an optimized shared-codebook baseline."],
        ]
        f.write(markdown_table(["direction", "avg_speed_ratio", "runtime_bytes/plain_bytes", "storage_bits/plain_bits", "interpretation"], score_rows))
        f.write("\n\n")

        f.write("## Large LUT\n\n")
        f.write(markdown_table(["case", "pattern", "plain_mops", "v2_mops", "v2/plain", "runtime_frac", "storage_frac"], large_rows))
        f.write("\n\n")

        f.write("## Many LUTs\n\n")
        f.write(markdown_table(["case", "pattern", "plain_mops", "v2_mops", "v2/plain", "runtime_frac", "storage_frac"], many_rows))
        f.write("\n\n")

        f.write("## LLM-Style LUT Traffic\n\n")
        f.write(markdown_table([
            "case",
            "pattern",
            "plain_per_group_mops",
            "plain_shared_mops",
            "v2_shared_mops",
            "v2/per_group",
            "v2/shared",
            "runtime/per_group",
            "runtime/shared",
        ], llm_rows))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--out", default="bench_results/pivots/latest")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    out_dir = (repo / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run(["make", "bench_pivots"], repo)

    commands = [
        ["benchmarks/pivots/large_lut/bench_large_lut", "--device", args.device, "--out", out_dir / "large_lut.csv"],
        ["benchmarks/pivots/many_lut/bench_many_lut", "--device", args.device, "--out", out_dir / "many_lut.csv"],
        ["benchmarks/pivots/llm_lut/bench_llm_lut", "--device", args.device, "--out", out_dir / "llm_lut.csv"],
    ]
    if args.quick:
        for cmd in commands:
            cmd.append("--quick")

    for cmd in commands:
        run([str(x) for x in cmd], repo)

    rows = []
    for name in ["large_lut.csv", "many_lut.csv", "llm_lut.csv"]:
        rows.extend(read_csv(out_dir / name))
    write_csv(out_dir / "all_pivots.csv", rows)
    write_summary(out_dir / "summary.md", rows, args.quick)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
