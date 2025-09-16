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

    plt.plot(xs, xs, linewidth=1)  # parity line
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


def bar_percent(values_df, key, value_col, out_png, topn, title, xlabel):
    """
    Generic horizontal bar plot for a percentage column.
    Sorts by absolute percent and shows top N (default 30).
    """
    d = values_df.dropna(subset=[value_col]).copy()
    d["abs_pct"] = d[value_col].abs()
    d = d.sort_values("abs_pct", ascending=False).head(topn)

    labels = d[key].astype(str).tolist()
    vals = d[value_col].tolist()

    plt.figure(figsize=(12, max(6, 0.4 * len(labels))))
    plt.barh(labels, vals)
    plt.axvline(0)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def bar_time_compare_raw(df, key, label_a, label_b, out_png, mode="top-delta", topn=30):
    """
    Optional: side-by-side raw median-time bars (A vs B).
    """
    d = df.copy()
    # “top-delta” sorted by |% faster time|
    if mode == "top-delta":
        d = d.dropna(subset=["pct_faster_time"]).copy()
        d["abs_pct"] = d["pct_faster_time"].abs()
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
    plt.title(f"Median time comparison (raw): {label_a} vs {label_b}")
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
    ap.add_argument("--topn", type=int, default=30,
                    help="How many bars in the % charts (default: 30)")
    ap.add_argument("--also-throughput", action="store_true",
                    help="Also compare throughput (adds scatter + %faster bar)")
    # Optional raw side-by-side time bars
    ap.add_argument("--bar-time-raw", action="store_true",
                    help="Also produce side-by-side raw median time bars (A vs B).")
    ap.add_argument("--bar-set", choices=["top-delta", "top-a", "top-b", "all"], default="top-delta",
                    help="Rows for the raw time bar chart (default: top-delta).")
    ap.add_argument("--bar-topn", type=int, default=30,
                    help="Rows in the raw time bar chart (default: 30).")
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

    # --- Metrics ---
    # Time: % faster (B vs A). Positive => B is faster (lower time).
    df["pct_faster_time"] = np.where(
        df["median_time_ms_a"] > 0,
        (df["median_time_ms_a"] - df["median_time_ms_b"]) / df["median_time_ms_a"] * 100.0,
        np.nan,
    )
    df["delta_ms"] = df["median_time_ms_b"] - df["median_time_ms_a"]  # keep for reference

    # Throughput: % faster (B vs A). Positive => B is higher throughput.
    if "throughput_gbps_a" in df.columns and "throughput_gbps_b" in df.columns:
        df["pct_faster_throughput"] = np.where(
            df["throughput_gbps_a"] > 0,
            (df["throughput_gbps_b"] - df["throughput_gbps_a"]) / df["throughput_gbps_a"] * 100.0,
            np.nan,
        )

    # --- Output CSV (with % faster) ---
    keep_cols = [args.key,
                 "median_time_ms_a", "median_time_ms_b", "delta_ms", "pct_faster_time"]
    if "throughput_gbps_a" in df.columns:
        keep_cols += ["throughput_gbps_a", "throughput_gbps_b", "pct_faster_throughput"]
    comp_csv = outdir / "comparison.csv"
    df[keep_cols].sort_values("pct_faster_time", ascending=True).to_csv(comp_csv, index=False)

    # --- Summary ---
    vf = df["pct_faster_time"].replace([np.inf, -np.inf], np.nan).dropna()
    if not vf.empty:
        print(f"Rows compared: {len(df)}")
        print(f"Only in A: {len(only_a)} | Only in B: {len(only_b)}")
        print("% faster (time) B vs A: "
              f"mean={vf.mean():.2f}%, median={vf.median():.2f}%, "
              f"min={vf.min():.2f}%, max={vf.max():.2f}%")

    # --- Plots ---
    # 1) Scatter (time)
    scatter_time(df, args.label_a, args.label_b, outdir / "scatter_time.png")

    # 2) Time % faster bar (TOP N by |%|)
    bar_percent(
        df, args.key, "pct_faster_time",
        out_png=outdir / "top_time_percent_faster.png",
        topn=args.topn,
        title=f"Top {args.topn} kernels: % faster (time) — {args.label_b} vs {args.label_a}",
        xlabel=f"% faster (time) — positive = {args.label_b} faster",
    )

    # 3) Optional raw side-by-side time bars
    if args.bar_time_raw:
        bar_time_compare_raw(
            df, args.key, args.label_a, args.label_b,
            out_png=outdir / "bar_time_compare_raw.png",
            mode=args.bar_set, topn=args.bar_topn
        )

    # 4) Throughput plots (if available)
    if args.also_throughput and "throughput_gbps_a" in df.columns and "throughput_gbps_b" in df.columns:
        # scatter
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

        # % faster (throughput)
        bar_percent(
            df, args.key, "pct_faster_throughput",
            out_png=outdir / "top_throughput_percent_faster.png",
            topn=args.topn,
            title=f"Top {args.topn} kernels: % faster (throughput) — {args.label_b} vs {args.label_a}",
            xlabel=f"% faster (throughput) — positive = {args.label_b} faster",
        )

    print(f"Wrote: {comp_csv}")
    print(f"Wrote: {outdir/'scatter_time.png'}")
    print(f"Wrote: {outdir/'top_time_percent_faster.png'}")
    if args.bar_time_raw:
        print(f"Wrote: {outdir/'bar_time_compare_raw.png'}")
    if (outdir / 'scatter_throughput.png').exists():
        print(f"Wrote: {outdir/'scatter_throughput.png'}")
    if (outdir / 'top_throughput_percent_faster.png').exists():
        print(f"Wrote: {outdir/'top_throughput_percent_faster.png'}")
    if (outdir / "only_in_A.csv").exists():
        print(f"Wrote: {outdir/'only_in_A.csv'}")
    if (outdir / "only_in_B.csv").exists():
        print(f"Wrote: {outdir/'only_in_B.csv'}")


if __name__ == "__main__":
    main()
