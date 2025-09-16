#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(csv_path: Path, key: str, dedupe: str):
    df = pd.read_csv(csv_path)
    if key not in df.columns:
        raise ValueError(f"Key '{key}' not in {csv_path}. Available: {list(df.columns)}")

    cols = [c for c in ["file", "kernel", "median_time_ms", "throughput_gbps", "returncode"] if c in df.columns]
    df = df[cols].copy()

    for col in ["median_time_ms", "throughput_gbps", "returncode"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if dedupe == "first":
        df = df.drop_duplicates(subset=[key], keep="first")
    elif dedupe == "min":
        df = df.sort_values(by=["median_time_ms"], ascending=True).drop_duplicates(subset=[key], keep="first")
    elif dedupe == "max":
        df = df.sort_values(by=["median_time_ms"], ascending=False).drop_duplicates(subset=[key], keep="first")
    elif dedupe == "mean":
        agg = {"median_time_ms": "mean"}
        if "throughput_gbps" in df.columns: agg["throughput_gbps"] = "mean"
        if "returncode" in df.columns: agg["returncode"] = "max"
        df = df.groupby(key, as_index=False).agg(agg)
    else:
        raise ValueError(f"Unknown dedupe: {dedupe}")

    return df


def scatter_time(df, label_a, label_b, out_png, log_scale=True, band_pct=0.10):
    x = df["median_time_ms_a"].to_numpy()
    y = df["median_time_ms_b"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]

    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.7)

    if x.size:
        xmin = max(min(x.min(), y.min()), 1e-9)
        xmax = max(x.max(), y.max()) * 1.05
    else:
        xmin, xmax = 1e-6, 1.0

    if log_scale:
        plt.xscale("log"); plt.yscale("log")
        xs = np.logspace(np.log10(xmin), np.log10(xmax), 256)
    else:
        xs = np.linspace(0, xmax, 256)
        xmin = 0.0

    plt.plot(xs, xs, linewidth=1)  # parity
    if band_pct and band_pct > 0:
        plt.plot(xs, xs * (1.0 - band_pct), linestyle="--", linewidth=1)
        plt.plot(xs, xs * (1.0 + band_pct), linestyle="--", linewidth=1)

    plt.xlim(xmin, xmax); plt.ylim(xmin, xmax)
    title = f"Kernel median time: {label_a} vs {label_b}"
    if np.isfinite(r):
        title += f"  (r={r:.3f})"
    plt.title(title)
    plt.xlabel(f"{label_a} median time (ms)")
    plt.ylabel(f"{label_b} median time (ms)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def bar_top_deltas(df, key, out_png, topn, label_a, label_b):
    d = df.dropna(subset=["pct_slower"]).copy()
    d["abs_pct"] = d["pct_slower"].abs()
    d = d.sort_values("abs_pct", ascending=False).head(topn)

    labels = d[key].astype(str).tolist()
    values = d["pct_slower"].tolist()

    plt.figure(figsize=(12, max(6, 0.4 * len(labels))))
    plt.barh(labels, values)
    plt.axvline(0)
    plt.xlabel(f"% slower vs {label_a}  (positive = {label_b} slower)")
    plt.title(f"Top {len(labels)} timing differences by %")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def bar_time_compare(df, key, label_a, label_b, out_png, mode="top-delta", topn=30):
    """
    Side-by-side bars for A vs B median time.
    mode:
      - 'top-delta': largest |% diff|
      - 'top-a': largest times on A
      - 'top-b': largest times on B
      - 'all': all rows (use topn to cap if desired; <=0 means all)
    """
    d = df.copy()
    if mode == "top-delta":
        d = d.dropna(subset=["pct_slower"]).copy()
        d["abs_pct"] = d["pct_slower"].abs()
        d = d.sort_values("abs_pct", ascending=False).head(topn)
    elif mode == "top-a":
        d = d.sort_values("median_time_ms_a", ascending=False).head(topn)
    elif mode == "top-b":
        d = d.sort_values("median_time_ms_b", ascending=False).head(topn)
    elif mode == "all":
        d = d.sort_values("median_time_ms_a", ascending=False)
        if topn and topn > 0:
            d = d.head(topn)
    else:
        raise ValueError(f"Unknown bar mode: {mode}")

    labels = d[key].astype(str).tolist()
    a = d["median_time_ms_a"].to_numpy()
    b = d["median_time_ms_b"].to_numpy()

    idx = np.arange(len(labels))
    width = 0.45

    plt.figure(figsize=(12, max(6, 0.35 * len(labels))))
    plt.barh(idx - width/2, a, height=width, label=label_a)
    plt.barh(idx + width/2, b, height=width, label=label_b)
    plt.yticks(idx, labels)
    plt.xlabel("Median time (ms) — lower is better")
    subtitle = {
        "top-delta": "Top differences by |%|",
        "top-a": f"Largest on {label_a}",
        "top-b": f"Largest on {label_b}",
        "all": "All (capped by --bar-topn if set)"
    }[mode]
    plt.title(f"Median time comparison: {label_a} vs {label_b} — {subtitle}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compare two Triton perf CSVs from different machines and plot timing differences.")
    ap.add_argument("csv_a", help="CSV from Machine A")
    ap.add_argument("csv_b", help="CSV from Machine B")
    ap.add_argument("-k", "--key", choices=["kernel", "file"], default="kernel",
                    help="Column to align on (default: kernel)")
    ap.add_argument("-o", "--outdir", default="compare_out", help="Output directory (default: compare_out)")
    ap.add_argument("--label-a", default="Machine A", help="Label for CSV A (default: Machine A)")
    ap.add_argument("--label-b", default="Machine B", help="Label for CSV B (default: Machine B)")
    ap.add_argument("--dedupe", choices=["first", "min", "max", "mean"], default="min",
                    help="How to handle duplicates per CSV (default: min)")
    ap.add_argument("--topn", type=int, default=30, help="How many bars in the top-difference chart (default: 30)")
    ap.add_argument("--also-throughput", action="store_true",
                    help="Also compare throughput (adds scatter + top deltas plots)")
    # NEW: side-by-side bar chart options
    ap.add_argument("--bar-time-compare", action="store_true",
                    help="Generate side-by-side bar chart for median time (A vs B).")
    ap.add_argument("--bar-set", choices=["top-delta", "top-a", "top-b", "all"], default="top-delta",
                    help="Which rows to show in the time bar chart (default: top-delta).")
    ap.add_argument("--bar-topn", type=int, default=30,
                    help="How many rows in the time bar chart (default: 30). For --bar-set all, <=0 means all.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_a = load_results(Path(args.csv_a), key=args.key, dedupe=args.dedupe)
    df_b = load_results(Path(args.csv_b), key=args.key, dedupe=args.dedupe)

    only_a = df_a[~df_a[args.key].isin(df_b[args.key])]
    only_b = df_b[~df_b[args.key].isin(df_a[args.key])]

    if not only_a.empty:
        only_a.to_csv(outdir / "only_in_A.csv", index=False)
    if not only_b.empty:
        only_b.to_csv(outdir / "only_in_B.csv", index=False)

    df = pd.merge(df_a, df_b, on=args.key, how="inner", suffixes=("_a", "_b"))
    df = df.dropna(subset=["median_time_ms_a", "median_time_ms_b"]).copy()

    df["delta_ms"] = df["median_time_ms_b"] - df["median_time_ms_a"]
    df["pct_slower"] = (df["delta_ms"] / df["median_time_ms_a"]) * 100.0

    if "throughput_gbps_a" in df.columns and "throughput_gbps_b" in df.columns:
        df["delta_throughput_gbps"] = df["throughput_gbps_b"] - df["throughput_gbps_a"]
        df["pct_throughput"] = (df["delta_throughput_gbps"] / df["throughput_gbps_a"]) * 100.0

    keep_cols = [args.key,
                 "median_time_ms_a", "median_time_ms_b", "delta_ms", "pct_slower"]
    if "throughput_gbps_a" in df.columns:
        keep_cols += ["throughput_gbps_a", "throughput_gbps_b",
                      "delta_throughput_gbps", "pct_throughput"]
    comp_csv = outdir / "comparison.csv"
    df[keep_cols].sort_values("pct_slower", ascending=False).to_csv(comp_csv, index=False)

    valid = df["pct_slower"].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid.empty:
        print(f"Rows compared: {len(df)}")
        print(f"Only in A: {len(only_a)} | Only in B: {len(only_b)}")
        print("Timing % diff (B vs A): "
              f"mean={valid.mean():.2f}%, median={valid.median():.2f}%, "
              f"min={valid.min():.2f}%, max={valid.max():.2f}%")

    # Scatter (timing)
    scatter_time(df, args.label_a, args.label_b, outdir / "scatter_time.png")

    # Top timing deltas bar (uses your labels in axis)
    bar_top_deltas(df, args.key, outdir / "top_time_deltas.png", args.topn, args.label_a, args.label_b)

    # NEW: Side-by-side bar chart for times
    if args.bar_time_compare:
        bar_time_compare(
            df, args.key, args.label_a, args.label_b,
            out_png=outdir / "bar_time_compare.png",
            mode=args.bar_set, topn=args.bar_topn
        )

    # Optional throughput comparison plots (labels fixed here)
    if args.also_throughput and "throughput_gbps_a" in df.columns and "throughput_gbps_b" in df.columns:
        # Scatter throughput
        plt.figure(figsize=(8, 8))
        plt.scatter(df["throughput_gbps_a"], df["throughput_gbps_b"], alpha=0.7)
        lim = [0, max(df["throughput_gbps_a"].max(), df["throughput_gbps_b"].max()) * 1.05]
        plt.plot(lim, lim)
        plt.xlim(lim); plt.ylim(lim)
        plt.xlabel(f"{args.label_a} throughput (GB/s)")
        plt.ylabel(f"{args.label_b} throughput (GB/s)")
        plt.title(f"Throughput: {args.label_a} vs {args.label_b}")
        plt.tight_layout()
        plt.savefig(outdir / "scatter_throughput.png", dpi=200)
        plt.close()

        # Bar of top throughput deltas (positive = B higher)
        d = df.dropna(subset=["pct_throughput"]).copy()
        d["abs_pct"] = d["pct_throughput"].abs()
        d = d.sort_values("abs_pct", ascending=False).head(args.topn)
        labels = d[args.key].astype(str).tolist()
        vals = d["pct_throughput"].tolist()
        plt.figure(figsize=(12, max(6, 0.4 * len(labels))))
        plt.barh(labels, vals)
        plt.axvline(0)
        plt.xlabel(f"% throughput change (positive = {args.label_b} higher)")
        plt.title(f"Top {len(labels)} throughput differences by %")
        plt.tight_layout()
        plt.savefig(outdir / "top_throughput_deltas.png", dpi=200)
        plt.close()

    print(f"Wrote: {comp_csv}")
    print(f"Wrote: {outdir/'scatter_time.png'}")
    print(f"Wrote: {outdir/'top_time_deltas.png'}")
    if args.bar_time_compare:
        print(f"Wrote: {outdir/'bar_time_compare.png'}")
    if (outdir / "only_in_A.csv").exists():
        print(f"Wrote: {outdir/'only_in_A.csv'}")
    if (outdir / "only_in_B.csv").exists():
        print(f"Wrote: {outdir/'only_in_B.csv'}")


if __name__ == "__main__":
    main()
