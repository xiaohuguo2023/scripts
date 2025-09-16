#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import matplotlib.pyplot as plt

# ---------------- Parsing helpers ----------------

# One-line format:
#   <time><unit>   <size><unitB>   <bandwidth><unitB>/s
# Examples:
#   0.088ms    0.503GB    5748.18GB/s
#   80us       512MB      6.3GB/s
MINLINE_RE = re.compile(
    r"^\s*([0-9]*\.?[0-9]+)\s*([unµm]?s)"
    r"\s+([0-9]*\.?[0-9]+)\s*([KMGTP]?B)"
    r"\s+([0-9]*\.?[0-9]+)\s*([KMGTP]?B)/s\s*$",
    re.IGNORECASE,
)

# Legacy verbose format fallbacks (still supported)
TIME_RE = re.compile(r"Median\s*time:\s*([0-9]*\.?[0-9]+)\s*([unµm]?s)", re.IGNORECASE)
THROUGHPUT_RE = re.compile(r"Throughput:\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?B)/s", re.IGNORECASE)
KERNEL_RE = re.compile(r"(?:^|\s)(?:K|k)?ernel:\s*([^\n\r]+)", re.IGNORECASE)
GRID_RE = re.compile(r"Grid:\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", re.IGNORECASE)

TIME_UNIT_TO_MS = {"s": 1000.0, "ms": 1.0, "us": 0.001, "µs": 0.001, "ns": 0.000001}

def unit_maps(bytes_base: int):
    """Return (BYTES->GB map, THROUGHPUT->GB map) for either 1024 or 1000 base."""
    if bytes_base == 1000:
        # decimal GB (SI)
        f = lambda p: 1.0 / (1000.0 ** p)
    else:
        # binary GiB-style scaling to 'GB' field (common in HPC)
        f = lambda p: 1.0 / (1024.0 ** p)
    BYTES_TO_GB = {
        "B": f(3),
        "KB": f(2),
        "MB": f(1),
        "GB": 1.0,
        "TB": f(-1),
        "PB": f(-2),
    }
    # For throughput we also want GB/s on the same scale
    THROUGHPUT_TO_GB = dict(BYTES_TO_GB)
    return BYTES_TO_GB, THROUGHPUT_TO_GB

def to_ms(value: float, unit: str) -> float:
    return float(value) * TIME_UNIT_TO_MS.get(unit.lower(), 1.0)

def to_gb(value: float, unit: str, BYTES_TO_GB: Dict[str, float]) -> float:
    return float(value) * BYTES_TO_GB.get(unit.upper(), 1.0)

def to_gbps(value: float, unit: str, THROUGHPUT_TO_GB: Dict[str, float]) -> float:
    return float(value) * THROUGHPUT_TO_GB.get(unit.upper(), 1.0)

def parse_minline(text: str, BYTES_TO_GB: Dict[str, float], THROUGHPUT_TO_GB: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """
    Parse the one-line format: time, size, bandwidth.
    Returns dict with time_ms, size_gb, bandwidth_gbps if match; otherwise None.
    """
    # Look for a match on any line (script might print extra lines)
    for line in text.splitlines():
        m = MINLINE_RE.match(line.strip())
        if m:
            t_val, t_unit, sz_val, sz_unit, bw_val, bw_unit = m.groups()
            return {
                "median_time_ms": to_ms(float(t_val), t_unit),
                "size_gb": to_gb(float(sz_val), sz_unit, BYTES_TO_GB),
                "throughput_gbps": to_gbps(float(bw_val), bw_unit, THROUGHPUT_TO_GB),
            }
    return None

def parse_verbose(text: str, BYTES_TO_GB: Dict[str, float], THROUGHPUT_TO_GB: Dict[str, float]) -> Dict[str, Any]:
    """
    Parse the older verbose output as a fallback.
    """
    out: Dict[str, Any] = {
        "kernel": None,
        "grid_x": None, "grid_y": None, "grid_z": None,
        "median_time_ms": None, "throughput_gbps": None,
        "size_gb": None,  # not usually present in verbose format
    }
    if (m := KERNEL_RE.search(text)): out["kernel"] = m.group(1).strip()
    if (g := GRID_RE.search(text)):
        out["grid_x"], out["grid_y"], out["grid_z"] = map(int, g.groups())
    if (t := TIME_RE.search(text)):
        out["median_time_ms"] = to_ms(float(t.group(1)), t.group(2))
    if (th := THROUGHPUT_RE.search(text)):
        out["throughput_gbps"] = to_gbps(float(th.group(1)), th.group(2), THROUGHPUT_TO_GB)
    return out

def parse_output(text: str, fmt: str, bytes_base: int) -> Dict[str, Any]:
    """
    fmt: "auto" | "minline" | "verbose"
    """
    BYTES_TO_GB, THROUGHPUT_TO_GB = unit_maps(bytes_base)
    if fmt in ("auto", "minline"):
        m = parse_minline(text, BYTES_TO_GB, THROUGHPUT_TO_GB)
        if m:
            return m
        if fmt == "minline":
            return {}  # force no fallback
    # fallback to verbose
    return parse_verbose(text, BYTES_TO_GB, THROUGHPUT_TO_GB)

# ---------------- Runner & plotting ----------------

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

def plot_bars(names: List[str], values: List[float], title: str, xlabel: str, outfile: Path, ascending: bool):
    idx = sorted(range(len(values)),
                 key=lambda i: (values[i] if values[i] is not None else float("inf")),
                 reverse=not ascending)
    names_s = [names[i] for i in idx]
    values_s = [values[i] for i in idx]

    plt.figure(figsize=(12, max(6, 0.25 * len(names_s))))
    plt.barh(names_s, values_s)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def clean_log(s: str) -> str:
    return (s or "").replace("\r", "").replace("\n", "\\n")

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Run benchmark scripts, parse one-line 'time size bandwidth' output (or legacy verbose), save CSV, plot time & bandwidth."
    )
    ap.add_argument("directory", help="Directory containing the Python benchmark scripts.")
    ap.add_argument("--glob", default="*.py", help="Glob pattern for scripts (default: *.py). Example: triton_*.py")
    ap.add_argument("--csv", default="triton_perf_summary.csv", help="Output CSV filename.")
    ap.add_argument("--time-plot", default="time_ms.png", help="Output PNG for time plot.")
    ap.add_argument("--throughput-plot", default="throughput_gbps.png", help="Output PNG for throughput plot.")
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-script timeout in seconds (default: 120).")
    ap.add_argument("--name-field", choices=["file", "kernel"], default="file",
                    help="Bar labels by 'file' or parsed 'kernel' (default: file).")
    ap.add_argument("--logs", choices=["clean", "raw", "none"], default="clean",
                    help="How to include stdout/stderr in CSV: 'clean' one-line, 'raw' multiline, or 'none'.")
    ap.add_argument("--format", choices=["auto", "minline", "verbose"], default="auto",
                    help="Parsing mode. 'minline' = only the one-line format; 'verbose' = legacy only; 'auto' tries both (default).")
    ap.add_argument("--bytes-base", type=int, choices=[1024, 1000], default=1024,
                    help="Unit base for GB conversions (1024=binary/GiB-style, 1000=decimal SI). Default 1024.")
    args = ap.parse_args()

    directory = Path(args.directory).expanduser().resolve()
    scripts = sorted(directory.glob(args.glob))
    if not directory.is_dir() or not scripts:
        print(f"ERROR: No scripts found matching {args.glob} in {directory}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for py in scripts:
        rc, out, err = run_script(py, timeout=args.timeout)
        parsed = parse_output(out, fmt=args.format, bytes_base=args.bytes_base)

        row: Dict[str, Any] = {
            "file": py.name,
            # Optional legacy fields (may be None)
            "kernel": parsed.get("kernel"),
            "grid_x": parsed.get("grid_x"),
            "grid_y": parsed.get("grid_y"),
            "grid_z": parsed.get("grid_z"),
            # Unified metrics
            "time_ms": parsed.get("median_time_ms"),
            "size_gb": parsed.get("size_gb"),
            "throughput_gbps": parsed.get("throughput_gbps"),
            "returncode": rc,
        }
        if args.logs == "raw":
            row["stdout"] = (out or "")
            row["stderr"] = (err or "")
        elif args.logs == "clean":
            row["stdout"] = clean_log(out)
            row["stderr"] = clean_log(err)

        rows.append(row)
        print(f"[{'OK' if rc == 0 else f'RC={rc}'}] {py.name} | t={row['time_ms']} ms | size={row['size_gb']} GB | bw={row['throughput_gbps']} GB/s")

    # CSV writing (LF line endings)
    csv_path = Path(args.csv)
    base_fields = ["file", "kernel", "grid_x", "grid_y", "grid_z", "time_ms", "size_gb", "throughput_gbps", "returncode"]
    if args.logs in ("raw", "clean"):
        fieldnames = base_fields + ["stdout", "stderr"]
    else:
        fieldnames = base_fields

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            for k in fieldnames:
                r.setdefault(k, "")
            writer.writerow(r)

    print(f"Wrote CSV: {csv_path}")

    # ---- Plots ----
    def label_for(r: Dict[str, Any]) -> str:
        if args.name_field == "kernel" and r.get("kernel"):
            return str(r["kernel"])
        return str(r["file"])

    # Time bars (ascending)
    labels_time, vals_time = [], []
    for r in rows:
        v = r.get("time_ms")
        if isinstance(v, (int, float)):
            labels_time.append(label_for(r))
            vals_time.append(v)
    if vals_time:
        plot_bars(labels_time, vals_time,
                  title="Kernel Time (lower is better)",
                  xlabel="Time (ms)",
                  outfile=Path(args.time_plot),
                  ascending=True)
        print(f"Wrote time plot: {args.time_plot}")
    else:
        print("No valid time_ms values parsed; skipping time plot.")

    # Throughput bars (descending)
    labels_bw, vals_bw = [], []
    for r in rows:
        v = r.get("throughput_gbps")
        if isinstance(v, (int, float)):
            labels_bw.append(label_for(r))
            vals_bw.append(v)
    if vals_bw:
        plot_bars(labels_bw, vals_bw,
                  title="Bandwidth (higher is better)",
                  xlabel="Throughput (GB/s)",
                  outfile=Path(args.throughput_plot),
                  ascending=False)
        print(f"Wrote throughput plot: {args.throughput_plot}")
    else:
        print("No valid throughput_gbps values parsed; skipping throughput plot.")

if __name__ == "__main__":
    main()
