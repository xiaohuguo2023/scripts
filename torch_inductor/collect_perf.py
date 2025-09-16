#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt

# ---------- Parsers ----------

TIME_RE = re.compile(r"Median\s*time:\s*([0-9]*\.?[0-9]+)\s*([unµm]?s)", re.IGNORECASE)
THROUGHPUT_RE = re.compile(r"Throughput:\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?B)/s", re.IGNORECASE)
KERNEL_RE = re.compile(r"(?:^|\s)(?:K|k)?ernel:\s*([^\n\r]+)", re.IGNORECASE)
GRID_RE = re.compile(r"Grid:\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", re.IGNORECASE)

TIME_UNIT_TO_MS = {"s": 1000.0, "ms": 1.0, "us": 0.001, "µs": 0.001, "ns": 0.000001}
THROUGHPUT_TO_GB = {"B": 1/(1024**3), "KB": 1/(1024**2), "MB": 1/1024, "GB": 1.0, "TB": 1024.0, "PB": 1024.0*1024.0}

def to_ms(v: float, u: str) -> float:
    return v * TIME_UNIT_TO_MS.get(u.lower(), 1.0)

def to_gbps(v: float, u: str) -> float:
    return v * THROUGHPUT_TO_GB.get(u.upper(), 1.0)

def parse_output(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "kernel": None, "grid_x": None, "grid_y": None, "grid_z": None,
        "median_time_ms": None, "throughput_gbps": None,
    }
    if (m := KERNEL_RE.search(text)): out["kernel"] = m.group(1).strip()
    if (g := GRID_RE.search(text)):
        out["grid_x"], out["grid_y"], out["grid_z"] = map(int, g.groups())
    if (t := TIME_RE.search(text)):
        out["median_time_ms"] = to_ms(float(t.group(1)), t.group(2))
    if (th := THROUGHPUT_RE.search(text)):
        out["throughput_gbps"] = to_gbps(float(th.group(1)), th.group(2))
    return out

# ---------- Runner ----------

def run_script(pyfile: Path, timeout: float = 120.0) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(pyfile)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", f"TIMEOUT: {e}"
    except Exception as e:
        return 1, "", f"ERROR: {e}"

# ---------- Plotting ----------

def plot_bars(names: List[str], values: List[float], title: str, xlabel: str, outfile: Path, asc: bool):
    idx = sorted(range(len(values)), key=lambda i: (values[i] if values[i] is not None else float("inf")),
                 reverse=not asc)
    names_s = [names[i] for i in idx]
    values_s = [values[i] for i in idx]
    plt.figure(figsize=(12, max(6, 0.25 * len(names_s))))
    plt.barh(names_s, values_s)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

# ---------- Utility ----------

def clean_log(s: str) -> str:
    # keep on one line: remove \r, replace \n with literal "\n"
    return (s or "").replace("\r", "").replace("\n", "\\n")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Run benchmark scripts, collect outputs, write CSV, and plot results.")
    ap.add_argument("directory", help="Directory containing the Python benchmark scripts.")
    ap.add_argument("--glob", default="*.py", help="Glob for scripts (default: *.py). Example: triton_*.py")
    ap.add_argument("--csv", default="triton_perf_summary.csv", help="Output CSV filename.")
    ap.add_argument("--time-plot", default="median_time_ms.png", help="Output PNG for median time plot.")
    ap.add_argument("--throughput-plot", default="throughput_gbps.png", help="Output PNG for throughput plot.")
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-script timeout seconds (default: 120).")
    ap.add_argument("--name-field", choices=["file", "kernel"], default="file",
                    help="Bar labels by 'file' or parsed 'kernel' (default: file).")
    ap.add_argument("--logs", choices=["clean", "raw", "none"], default="clean",
                    help="How to include stdout/stderr in CSV: 'clean' one-line, 'raw' multiline, or 'none'.")
    args = ap.parse_args()

    directory = Path(args.directory).expanduser().resolve()
    if not directory.is_dir():
        print(f"ERROR: {directory} is not a directory.", file=sys.stderr)
        sys.exit(2)

    scripts = sorted(directory.glob(args.glob))
    if not scripts:
        print(f"No scripts found matching {args.glob} in {directory}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for py in scripts:
        rc, out, err = run_script(py, timeout=args.timeout)
        parsed = parse_output(out)
        row = {
            "file": py.name,
            "kernel": parsed["kernel"],
            "grid_x": parsed["grid_x"],
            "grid_y": parsed["grid_y"],
            "grid_z": parsed["grid_z"],
            "median_time_ms": parsed["median_time_ms"],
            "throughput_gbps": parsed["throughput_gbps"],
            "returncode": rc,
        }
        if args.logs == "raw":
            row["stdout"] = (out or "")
            row["stderr"] = (err or "")
        elif args.logs == "clean":
            row["stdout"] = clean_log(out)
            row["stderr"] = clean_log(err)
        # if 'none', omit the columns entirely (we'll set fieldnames accordingly)
        rows.append(row)
        status = "OK" if rc == 0 else f"RC={rc}"
        print(f"[{status}] {py.name} | kernel={row.get('kernel')} | t={row.get('median_time_ms')} ms | th={row.get('throughput_gbps')} GB/s")

    # CSV fields
    base_fields = ["file", "kernel", "grid_x", "grid_y", "grid_z", "median_time_ms", "throughput_gbps", "returncode"]
    if args.logs in ("raw", "clean"):
        fieldnames = base_fields + ["stdout", "stderr"]
    else:
        fieldnames = base_fields

    csv_path = Path(args.csv)
    # Use lineterminator="\n" to avoid CRLF (^M)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            # ensure missing keys exist when logs are omitted
            for k in fieldnames:
                r.setdefault(k, "")
            writer.writerow(r)
    print(f"\nWrote CSV: {csv_path}")

    # Plots (only rows with numeric values)
    def label_for(r):
        return r["file"] if args.name_field == "file" or not r.get("kernel") else r["kernel"]

    labels_time, times = [], []
    labels_thru, thrus = [], []
    for r in rows:
        if isinstance(r.get("median_time_ms"), (int, float)):
            labels_time.append(label_for(r)); times.append(r["median_time_ms"])
        if isinstance(r.get("throughput_gbps"), (int, float)):
            labels_thru.append(label_for(r)); thrus.append(r["throughput_gbps"])

    if times:
        plot_bars(labels_time, times, "Median Kernel Time (lower is better)", "Median time (ms)", Path(args.time_plot), asc=True)
        print(f"Wrote time plot: {args.time_plot}")
    else:
        print("No valid median_time_ms values parsed; skipping time plot.")

    if thrus:
        plot_bars(labels_thru, thrus, "Throughput (higher is better)", "Throughput (GB/s)", Path(args.throughput_plot), asc=False)
        print(f"Wrote throughput plot: {args.throughput_plot}")
    else:
        print("No valid throughput_gbps values parsed; skipping throughput plot.")

if __name__ == "__main__":
    main()
