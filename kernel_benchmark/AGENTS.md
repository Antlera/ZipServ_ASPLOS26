# Agent Guide — ZipServ Kernel Benchmark + Report

This directory ships a one-click pipeline that:

1. builds `libL_API.so` and `test_mm`
2. runs every canonical GEMM workload (LLaMA / Mistral shapes)
3. emits a hardware-specific interactive HTML report at
   `reports/report_<gpu_slug>.html`

**As an agent on a new machine, do the steps in `## Quick Run` verbatim.**

---

## Prerequisites (verify, do NOT install silently)

| Requirement       | How to check                                            | Notes                              |
|-------------------|---------------------------------------------------------|------------------------------------|
| NVIDIA GPU + driver | `nvidia-smi`                                          | Compute Capability ≥ 8.0           |
| CUDA toolkit      | `nvcc --version` (expects `/usr/local/cuda` by default) | Override with `CUDA_PATH=...`      |
| g++ with C++14    | `g++ --version`                                         |                                    |
| python3           | `python3 --version`                                     | stdlib only, no pip deps           |

If any of the above is missing, **stop and report** to the user — do not auto-install.

---

## Quick Run

From the **repo root**:

```bash
cd kernel_benchmark
./run_benchmark.sh
```

Expected runtime: ~5–15 min depending on GPU (11 workloads × ~2100 iterations each).

Outputs:

- `kernel_benchmark/bf16_triplebm_res.csv` — raw timings (any prior CSV is auto-archived to `*.bak`)
- `kernel_benchmark/reports/report_<gpu_slug>.html` — interactive report
- `kernel_benchmark/reports/.runlog_<workload>.txt` — per-workload stdout (for debugging)

---

## Useful Flags

| Flag                          | Purpose                                              |
|-------------------------------|------------------------------------------------------|
| `--report-only`               | Re-render HTML from existing CSV, no benchmarking    |
| `--keep-csv`                  | Append to existing CSV instead of archiving it       |
| `--no-build`                  | Skip `make`, assume binaries are present             |
| `--gpu-name "<name>"`         | Override `nvidia-smi` detected name (also affects output filename) |
| `--peak-bw <GB/s>`            | Override theoretical DRAM bandwidth used in report   |
| `--peak-tc <TFLOPS>`          | Override peak BF16 Tensor-Core throughput            |

---

## What if the GPU is unknown?

Auto-detected GPU name is matched against the `GPU_PRESETS` table in
`generate_report.py`. If no preset matches, the script falls back to
`1000 GB/s / 200 TFLOPS` and prints those values. Two options:

1. **Quick fix (per-run)**: pass `--peak-bw` and `--peak-tc` from public specs.
2. **Permanent fix**: add a row to `GPU_PRESETS` in `generate_report.py`,
   tuple `(name_substring, bw_gbs, bf16_tc_tflops_dense)`. Substring match is
   case-insensitive, first hit wins — keep more specific entries above generic
   ones (e.g. `"A100 80GB"` before `"A100"`).

Reference values (dense BF16 Tensor-Core TFLOPS, no sparsity):
- A100 SXM 40 GB: 1555 GB/s, 312 TFLOPS
- A100 SXM 80 GB: 2039 GB/s, 312 TFLOPS
- H100 SXM: 3350 GB/s, 989 TFLOPS
- L40S: 864 GB/s, 362 TFLOPS
- RTX 4090: 1008 GB/s, 165 TFLOPS
- RTX 5090: 1792 GB/s, 419 TFLOPS

---

## Modifying the Workload Set

`WORKLOADS` in `generate_report.py` is the **single source of truth** for both
the benchmark driver and the HTML dropdown. To add/remove a shape, edit only
that list — `run_benchmark.sh` enumerates it via
`python3 generate_report.py --emit-workloads`.

Each entry must have `key, model, layer, M, K, N, SplitK, label`. The `key`
becomes the `<option value>` in the report's dropdown.

---

## Troubleshooting

| Symptom                                                | Cause / Fix                                                                                  |
|--------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `LInfer_HOME` is the wrong path                        | `Init.sh` uses `pwd`; `run_benchmark.sh` already `cd`s to repo root before sourcing — do not source `Init.sh` from elsewhere. |
| `LD_LIBRARY_PATH: unbound variable`                    | Your shell has `set -u` upstream; `run_benchmark.sh` pre-exports an empty default. If you source the scripts manually, do `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"` first. |
| `cannot find -lL_API` during `make`                    | `MY_PATH` not exported; either `source ../Init.sh && source test_env`, or just rerun `./run_benchmark.sh` (it sources both). |
| nvcc errors `unsupported gpu architecture 'compute_120'` | Your CUDA is older than 12.8. Edit `Makefile` and the build's `Makefile`, drop `120` from `SMS ?= 80 86 89 120`. |
| Some workloads OOM (e.g. `70b_n128`)                   | Need ≥ ~24 GB VRAM. Either skip those entries in `WORKLOADS` or run on a larger GPU. |
| `[warn] missing data for workload ...`                  | `test_mm` for that shape failed mid-run; check `reports/.runlog_<key>.txt`.                  |

---

## How the Report Is Built (one paragraph)

`run_benchmark.sh` archives the prior CSV, then loops the workloads from
`generate_report.py --emit-workloads`, invoking
`./test_mm M K N SplitK --model X --layer Y` for each. `test_mm` appends three
rows (`cuBLAS`, `cuBLAS_TC`, `CompGEMM`) per call to `bf16_triplebm_res.csv`.
`generate_report.py` then opens `dataflow_visualization.html` as a template and
patches four regions in place: the `const WORKLOADS = {...}` JS dict, the
`<select id="workload">` dropdown options, the GPU name in the subtitle, and
the `PEAK_DRAM` / `PEAK_TC` constants used in the bandwidth-utilization view.
The static didactic sections (data-flow diagram, encoding walkthrough,
pipeline timeline) are kept verbatim — they describe the algorithm, not a
specific run.
