#!/usr/bin/env bash
set -euo pipefail

# ----- Defaults matching your setup -----
TRITON_DIR="/home/openai/triton"
BENCH_CWD="$(pwd)"              # directory where tritonblas_matmul.py lives
PYTHON_BIN="${PYTHON:-python3}" # python used for parsing/averaging

# Fixed benchmark command (exactly as you provided)
BENCH_CMD=(python tritonblas_matmul.py --input-yaml ../datasets/test.yaml --output-csv test.csv)
CSV_NAME="test.csv"

# Metric selection / thresholds
METRIC="tritonblas_gflops"  # or "us"
MIN=""                      # set this for higher-is-better metrics (e.g., GFLOPs)
MAX=""                      # set this for lower-is-better metrics  (e.g., us)
REPEATS=1                   # run benchmark N times and average across runs
KEEP=0                      # keep CSV/logs when debugging (1=yes)

usage() {
  cat <<'EOF'
Usage: git bisect run ./bisect_triton_perf_fixed.sh [options]

Options:
  --triton-dir PATH       Triton repo path (default: /home/openai/triton)
  --bench-cwd PATH        Directory containing tritonblas_matmul.py (default: current dir)
  --python BIN            Python executable used for parsing (default: python3)
  --metric NAME           Metric column in CSV (default: tritonblas_gflops) or "us"
  --min VAL               Threshold for higher-is-better metric (GOOD if avg >= VAL)
  --max VAL               Threshold for lower-is-better metric (GOOD if avg <= VAL)
  --repeats N             Run N times and average (default: 1)
  --keep                  Keep CSV/logs (default: off)

Exit codes for git bisect:
  0 = GOOD (meets threshold), 1 = BAD (regression), 125 = SKIP (build/run issue)
EOF
}

# ----- Parse CLI -----
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --triton-dir) TRITON_DIR="$2"; shift 2 ;;
    --bench-cwd) BENCH_CWD="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --metric) METRIC="$2"; shift 2 ;;
    --min) MIN="$2"; shift 2 ;;
    --max) MAX="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MIN" && -z "$MAX" ]]; then
  echo "Error: provide either --min (higher-is-better) or --max (lower-is-better)." >&2
  exit 2
fi

# ----- Helpers -----
build_triton() {
  cd "$TRITON_DIR"
  git submodule update --init --recursive || return 125
  # Build with the user's command (pip install -e .)
  if ! pip install -e . >/tmp/triton_build.log 2>&1 ; then
    echo "Build failed; see /tmp/triton_build.log" >&2
    return 125
  fi
  return 0
}

run_once() {
  local rc
  # Ensure fresh CSV each run
  ( cd "$BENCH_CWD" && rm -f "$CSV_NAME" ) || true

  # Run the benchmark (exact command you use)
  set +e
  ( cd "$BENCH_CWD" && "${BENCH_CMD[@]}" ) >"/tmp/bench_stdout.log" 2>"/tmp/bench_stderr.log"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "Benchmark failed (exit $rc)." >&2
    tail -n 200 /tmp/bench_stderr.log >&2 || true
    return 125
  fi

  # Check CSV exists
  if [[ ! -f "$BENCH_CWD/$CSV_NAME" ]]; then
    echo "CSV not produced: $BENCH_CWD/$CSV_NAME" >&2
    return 125
  fi

  # Parse the CSV -> list of metric values, average across rows
  local value
  value="$("$PYTHON_BIN" - "$BENCH_CWD/$CSV_NAME" "$METRIC" <<'PY'
import csv, sys, statistics
fn, metric = sys.argv[1], sys.argv[2]
vals = []
with open(fn, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        if metric not in row:
            print("MISSING_METRIC", file=sys.stderr); sys.exit(2)
        try:
            vals.append(float(row[metric]))
        except Exception:
            pass
if not vals:
    print("NO_VALUES", file=sys.stderr); sys.exit(2)
print(statistics.fmean(vals))
PY
)" || true

  if [[ -z "$value" || "$value" == "MISSING_METRIC" || "$value" == "NO_VALUES" ]]; then
    echo "Failed to parse metric '$METRIC' from CSV." >&2
    return 125
  fi

  # Optionally keep artifacts
  if [[ $KEEP -eq 0 ]]; then
    rm -f "$BENCH_CWD/$CSV_NAME"
  else
    echo "Kept CSV at $BENCH_CWD/$CSV_NAME (parsed $METRIC=$value)" >&2
  fi

  printf "%s\n" "$value"
  return 0
}

# ----- Build Triton at this commit -----
echo "=== Building Triton at $(cd "$TRITON_DIR" && git rev-parse --short HEAD) ===" >&2
build_triton || exit 125

# ----- Run benchmark REPEATS times -----
vals=()
for i in $(seq 1 "$REPEATS"); do
  v="$(run_once)" || rc=$?
  if [[ "${rc:-0}" -eq 125 ]]; then
    exit 125
  fi
  vals+=("$v")
done

AVG="$("$PYTHON_BIN" - "${vals[@]}" <<'PY'
import sys, statistics
vals = [float(x) for x in sys.argv[1:]]
print(statistics.fmean(vals))
PY
)"

echo "Metric ${METRIC} average over ${REPEATS} run(s): ${AVG}" >&2

# ----- Decide GOOD/BAD -----
if [[ -n "$MIN" ]]; then
  "$PYTHON_BIN" - "$AVG" "$MIN" <<'PY' >/dev/null
import sys
avg, thr = map(float, sys.argv[1:])
sys.exit(0 if avg >= thr else 1)
PY
  status=$?
else
  "$PYTHON_BIN" - "$AVG" "$MAX" <<'PY' >/dev/null
import sys
avg, thr = map(float, sys.argv[1:])
sys.exit(0 if avg <= thr else 1)
PY
  status=$?
fi

if [[ $status -eq 0 ]]; then
  echo "GOOD: meets threshold." >&2
else
  echo "BAD: regression vs threshold." >&2
fi
exit $status
