#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# ZipServ one-click benchmark + HTML report.
#
# Pipeline:
#   1. source Init.sh and test_env (so LInfer_HOME / LD_LIBRARY_PATH are set)
#   2. build  ../build/libL_API.so  if missing
#   3. build  ./test_mm             if missing  (via existing Makefile)
#   4. detect GPU via nvidia-smi
#   5. run every canonical workload (defined in generate_report.py)
#   6. render reports/report_<gpu>.html from dataflow_visualization.html
#
# Usage:
#   ./run_benchmark.sh                      # full pipeline, fresh CSV
#   ./run_benchmark.sh --keep-csv           # do not wipe existing CSV
#   ./run_benchmark.sh --no-build           # skip rebuild
#   ./run_benchmark.sh --report-only        # only re-render HTML
#   ./run_benchmark.sh --peak-bw 1792 \
#                      --peak-tc 419 \
#                      --gpu-name "NVIDIA RTX 5090"
# ----------------------------------------------------------------------------
# NB: do NOT enable `set -u` — the project's test_env / Init.sh reference
# environment vars (LD_LIBRARY_PATH, CPLUS_INCLUDE_PATH, CONDA_PREFIX) that
# are commonly unset on fresh shells.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}"
export CONDA_PREFIX="${CONDA_PREFIX:-/usr}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
BENCH_DIR="${SCRIPT_DIR}"
CSV_FILE="${BENCH_DIR}/bf16_triplebm_res.csv"
TEMPLATE_FILE="${BENCH_DIR}/dataflow_visualization.html"
REPORT_DIR="${BENCH_DIR}/reports"
PYTHON="${PYTHON:-python3}"

KEEP_CSV=0
DO_BUILD=1
REPORT_ONLY=0
GPU_NAME_OVERRIDE=""
PEAK_BW_OVERRIDE=""
PEAK_TC_OVERRIDE=""
EXTRA_NVCC_ENV=""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-csv)     KEEP_CSV=1; shift ;;
    --no-build)     DO_BUILD=0; shift ;;
    --report-only)  REPORT_ONLY=1; KEEP_CSV=1; DO_BUILD=0; shift ;;
    --gpu-name)     GPU_NAME_OVERRIDE="$2"; shift 2 ;;
    --peak-bw)      PEAK_BW_OVERRIDE="$2"; shift 2 ;;
    --peak-tc)      PEAK_TC_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,25p' "$0"; exit 0 ;;
    *)
      echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
echo "[env]  sourcing ${REPO_ROOT}/Init.sh"
# Init.sh uses `pwd` to set LInfer_HOME — must be sourced from REPO_ROOT.
pushd "${REPO_ROOT}" >/dev/null
# shellcheck disable=SC1091
source "${REPO_ROOT}/Init.sh"
popd >/dev/null
echo "[env]  LInfer_HOME = ${LInfer_HOME}"
echo "[env]  sourcing ${BENCH_DIR}/test_env"
# shellcheck disable=SC1091
source "${BENCH_DIR}/test_env"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [[ "${DO_BUILD}" -eq 1 ]]; then
  if [[ ! -f "${BUILD_DIR}/libL_API.so" ]]; then
    echo "[build] libL_API.so not found — building it"
    make -C "${BUILD_DIR}" -j"$(nproc)"
  else
    echo "[build] libL_API.so already present (use 'make -C build clean' to force rebuild)"
  fi

  if [[ ! -x "${BENCH_DIR}/test_mm" ]]; then
    echo "[build] test_mm not found — building it"
    make -C "${BENCH_DIR}" -j"$(nproc)"
  else
    echo "[build] test_mm already present"
  fi
fi

if [[ ! -x "${BENCH_DIR}/test_mm" ]]; then
  echo "[error] ${BENCH_DIR}/test_mm not built; cannot run benchmarks." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Detect GPU
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME_DETECTED="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
else
  GPU_NAME_DETECTED="Unknown GPU"
fi
GPU_NAME="${GPU_NAME_OVERRIDE:-${GPU_NAME_DETECTED}}"
echo "[gpu]  ${GPU_NAME}"

# ---------------------------------------------------------------------------
# Reset CSV
# ---------------------------------------------------------------------------
if [[ "${KEEP_CSV}" -eq 0 && -f "${CSV_FILE}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  echo "[csv]  archiving existing ${CSV_FILE} -> ${CSV_FILE}.${ts}.bak"
  mv "${CSV_FILE}" "${CSV_FILE}.${ts}.bak"
fi

# ---------------------------------------------------------------------------
# Run benchmarks (skip if --report-only)
# ---------------------------------------------------------------------------
if [[ "${REPORT_ONLY}" -eq 0 ]]; then
  echo "[run]  enumerating workloads..."
  mapfile -t WORKLOADS < <("${PYTHON}" "${BENCH_DIR}/generate_report.py" --emit-workloads)
  echo "[run]  ${#WORKLOADS[@]} workloads to execute"

  cd "${BENCH_DIR}"
  i=0
  total=${#WORKLOADS[@]}
  for wl in "${WORKLOADS[@]}"; do
    i=$((i+1))
    IFS='|' read -r KEY MODEL LAYER M K N SK <<<"${wl}"
    echo
    echo "[run  ${i}/${total}] ${KEY}  model=${MODEL}  layer=${LAYER}  M=${M} K=${K} N=${N} SplitK=${SK}"
    log_file="${BENCH_DIR}/reports/.runlog_${KEY}.txt"
    mkdir -p "$(dirname "${log_file}")"
    ./test_mm "${M}" "${K}" "${N}" "${SK}" \
              --model "${MODEL}" --layer "${LAYER}" \
              > "${log_file}" 2>&1 || {
      echo "[error] ./test_mm failed for ${KEY}; tail of log:" >&2
      tail -n 20 "${log_file}" >&2
      exit 1
    }
    # Echo the key timing lines for live feedback.
    grep -E '^(BF16_triple_bitmap|CuBLAS_TC|CuBLAS_non-TC)' "${log_file}" || true
  done
fi

# ---------------------------------------------------------------------------
# Render HTML report
# ---------------------------------------------------------------------------
mkdir -p "${REPORT_DIR}"
slug="$(echo "${GPU_NAME}" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+|_+$//g')"
[[ -z "${slug}" ]] && slug="unknown_gpu"
OUT_HTML="${REPORT_DIR}/report_${slug}.html"

REPORT_ARGS=(--csv "${CSV_FILE}" --template "${TEMPLATE_FILE}" --output "${OUT_HTML}")
REPORT_ARGS+=(--gpu-name "${GPU_NAME}")
[[ -n "${PEAK_BW_OVERRIDE}" ]] && REPORT_ARGS+=(--peak-bw "${PEAK_BW_OVERRIDE}")
[[ -n "${PEAK_TC_OVERRIDE}" ]] && REPORT_ARGS+=(--peak-tc "${PEAK_TC_OVERRIDE}")

echo
echo "[render] generate_report.py ${REPORT_ARGS[*]}"
"${PYTHON}" "${BENCH_DIR}/generate_report.py" "${REPORT_ARGS[@]}"

echo
echo "[done]  CSV    : ${CSV_FILE}"
echo "[done]  Report : ${OUT_HTML}"
