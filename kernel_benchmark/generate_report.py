#!/usr/bin/env python3
"""
ZipServ benchmark report generator.

Two modes:
  1. --emit-workloads      : print the canonical workload table
                             (one workload per line, used by run_benchmark.sh)
  2. (default)             : read benchmark CSV + GPU info, render the
                             interactive dataflow_visualization HTML report
                             with the live numbers patched in.

Usage:
  python3 generate_report.py --emit-workloads
  python3 generate_report.py --csv bf16_triplebm_res.csv \
                             --template dataflow_visualization.html \
                             --output  reports/report_<gpu>.html \
                             [--gpu-name "NVIDIA RTX 5090"] \
                             [--peak-bw 1792] [--peak-tc 419]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical workload table — single source of truth for both
# the benchmark driver (run_benchmark.sh) and the HTML report.
#
# key:                   identifier used inside the HTML <select> dropdown
# model / layer:         passed to ./test_mm via --model / --layer
# M, K, N, SplitK:       positional args to ./test_mm
# label:                 human-readable label for the dropdown option
# ---------------------------------------------------------------------------
WORKLOADS = [
    {"key": "70b_gateup_n32",       "model": "LLaMA3.1-70B", "layer": "GateUp_proj",
     "M": 57344, "K": 8192,  "N": 32,  "SplitK": 1,
     "label": "LLaMA-70B GateUp_proj | M=57344 K=8192 N=32"},
    {"key": "123b_gateup_n32",      "model": "Mistral-123B", "layer": "GateUp_proj",
     "M": 45056, "K": 12288, "N": 32,  "SplitK": 1,
     "label": "Mistral-123B GateUp_proj | M=45056 K=12288 N=32"},
    {"key": "mistral24_gateup_n32", "model": "Mistral-24B",  "layer": "GateUp_proj",
     "M": 17920, "K": 5120,  "N": 32,  "SplitK": 1,
     "label": "Mistral-24B GateUp_proj | M=17920 K=5120 N=32"},
    {"key": "8b_gateup_n32",        "model": "LLaMA3.1-8B",  "layer": "GateUp_proj",
     "M": 28672, "K": 4096,  "N": 32,  "SplitK": 1,
     "label": "LLaMA-8B GateUp_proj | M=28672 K=4096 N=32"},
    {"key": "8b_qkv_n32",           "model": "LLaMA3.1-8B",  "layer": "QKV_proj",
     "M": 12288, "K": 4096,  "N": 32,  "SplitK": 1,
     "label": "LLaMA-8B QKV_proj | M=12288 K=4096 N=32"},
    {"key": "8b_o_n32",             "model": "LLaMA3.1-8B",  "layer": "O_proj",
     "M": 4096,  "K": 4096,  "N": 32,  "SplitK": 1,
     "label": "LLaMA-8B O_proj | M=4096 K=4096 N=32"},
    {"key": "8b_down_n32",          "model": "LLaMA3.1-8B",  "layer": "Down_proj",
     "M": 4096,  "K": 14336, "N": 32,  "SplitK": 1,
     "label": "LLaMA-8B Down_proj | M=4096 K=14336 N=32"},
    {"key": "70b_down_n32",         "model": "LLaMA3.1-70B", "layer": "Down_proj",
     "M": 8192,  "K": 28672, "N": 32,  "SplitK": 1,
     "label": "LLaMA-70B Down_proj | M=8192 K=28672 N=32"},
    {"key": "70b_n8",               "model": "LLaMA3.1-70B", "layer": "GateUp_N8",
     "M": 57344, "K": 8192,  "N": 8,   "SplitK": 1,
     "label": "LLaMA-70B GateUp | M=57344 K=8192 N=8"},
    {"key": "70b_n64",              "model": "LLaMA3.1-70B", "layer": "GateUp_N64",
     "M": 57344, "K": 8192,  "N": 64,  "SplitK": 1,
     "label": "LLaMA-70B GateUp | M=57344 K=8192 N=64"},
    {"key": "70b_n128",             "model": "LLaMA3.1-70B", "layer": "GateUp_N128",
     "M": 57344, "K": 8192,  "N": 128, "SplitK": 1,
     "label": "LLaMA-70B GateUp | M=57344 K=8192 N=128"},
]

# ---------------------------------------------------------------------------
# GPU presets: substring-matched against `nvidia-smi --query-gpu=name`.
# bw_gbs    : theoretical peak DRAM bandwidth (GB/s)
# bf16_tc   : peak Tensor Core BF16 throughput (TFLOPS, dense)
# Override at runtime with --peak-bw / --peak-tc.
# ---------------------------------------------------------------------------
GPU_PRESETS = [
    # (name_substring,                  bw_gbs, bf16_tc)
    ("RTX PRO 6000",                    1792,   419),   # Blackwell workstation
    ("RTX 6000 Ada",                    960,    181),
    ("RTX 5090",                        1792,   419),
    ("RTX 4090",                        1008,   165),
    ("RTX 4080",                        717,    97),
    ("RTX 3090 Ti",                     1008,   80),
    ("RTX 3090",                        936,    71),
    ("RTX A6000",                       768,    155),
    ("L40S",                            864,    362),
    ("L40",                             864,    181),
    ("L4",                              300,    121),
    ("A40",                             696,    150),
    ("A100 80GB",                       2039,   312),
    ("A100",                            1555,   312),
    ("H100",                            3350,   989),
    ("H200",                            4800,   989),
    ("H800",                            3350,   989),
    ("H20",                             4000,   148),
    ("V100",                            900,    125),
]


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------
def read_csv(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "model":  r["Model"],
                "layer":  r["Layer"],
                "M":      int(r["M"]),
                "K":      int(r["K"]),
                "N":      int(r["N"]),
                "SplitK": int(r["SplitK"]),
                "kernel": r["Kernel"],
                "ms":     float(r["Duration(ms)"]),
                "tflops": float(r["TFLOPS"]),
            })
    return rows


def find_row(rows: list[dict], wl: dict, kernel: str) -> dict | None:
    """Pick the LAST matching row (most recent run) for a given workload+kernel."""
    match = None
    for r in rows:
        if (r["M"] == wl["M"] and r["K"] == wl["K"]
                and r["N"] == wl["N"] and r["SplitK"] == wl["SplitK"]
                and r["kernel"] == kernel):
            match = r
    return match


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True,
        )
        first = out.strip().splitlines()[0].strip()
        return first or "Unknown GPU"
    except Exception:
        return "Unknown GPU"


def lookup_gpu_specs(name: str) -> tuple[float, float]:
    for sub, bw, tc in GPU_PRESETS:
        if sub.lower() in name.lower():
            return float(bw), float(tc)
    # Fallback (mid-range estimates)
    return 1000.0, 200.0


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def render_workloads_js(rows: list[dict]) -> tuple[str, str]:
    """Return (workloads_dict_js, dropdown_options_html)."""
    js_entries = []
    options = []
    for wl in WORKLOADS:
        tc = find_row(rows, wl, "cuBLAS_TC")
        zg = find_row(rows, wl, "CompGEMM")
        if tc is None or zg is None:
            print(f"  [warn] missing data for workload '{wl['key']}' "
                  f"(M={wl['M']} K={wl['K']} N={wl['N']} SplitK={wl['SplitK']})",
                  file=sys.stderr)
            continue
        speedup = tc["ms"] / zg["ms"]
        winner = (f"ZipGEMM wins {speedup:.2f}\u00d7" if speedup > 1.005
                  else f"cuBLAS wins {speedup:.2f}\u00d7" if speedup < 0.995
                  else f"tie {speedup:.2f}\u00d7")
        js_entries.append(
            f"  '{wl['key']}': {{M:{wl['M']}, K:{wl['K']}, N:{wl['N']}, "
            f"cuBLAS_ms:{tc['ms']:.6f}, zipg_ms:{zg['ms']:.6f}, "
            f"cuBLAS_tflops:{tc['tflops']:.1f}, zipg_tflops:{zg['tflops']:.1f}}}"
        )
        options.append(
            f'    <option value="{wl["key"]}">{wl["label"]} \u2014 {winner}</option>'
        )

    workloads_js = "const WORKLOADS = {\n" + ",\n".join(js_entries) + "\n};"
    dropdown_html = "\n".join(options)
    return workloads_js, dropdown_html


# Regex to swap the entire `const WORKLOADS = { ... };` block in the template.
# Uses non-greedy + DOTALL to span the multi-line dict literal.
_WORKLOADS_RE = re.compile(
    r"const\s+WORKLOADS\s*=\s*\{.*?\};",
    re.DOTALL,
)

# Regex to swap the dropdown options block.
_DROPDOWN_RE = re.compile(
    r'(<select id="workload">)(.*?)(</select>)',
    re.DOTALL,
)

# Regex to swap the GPU name shown in the subtitle (`<code>...</code>`).
# We anchor on the surrounding text to avoid matching unrelated <code> tags.
_GPU_NAME_RE = re.compile(
    r'(measured from <code>bf16_triplebm_res\.csv</code>)',
    re.DOTALL,
)
_SUBTITLE_GPU_RE = re.compile(
    r'(<code>)RTX PRO 6000 Blackwell Max-Q(</code>)',
)

# Regex to swap PEAK_DRAM and PEAK_TC constants in the inline JS.
_PEAK_DRAM_RE = re.compile(r'const\s+PEAK_DRAM\s*=\s*[\d.]+\s*;')
_PEAK_TC_RE   = re.compile(r'const\s+PEAK_TC\s*=\s*[\d.]+\s*;')


def render_html(template_path: str, csv_path: str, out_path: str,
                gpu_name: str, peak_bw: float, peak_tc: float) -> None:
    with open(template_path, "r") as f:
        html = f.read()

    rows = read_csv(csv_path)
    workloads_js, dropdown_html = render_workloads_js(rows)

    # NB: re.sub treats backslashes in the replacement string as group refs,
    # so we always wrap dynamic replacements in a lambda to disable that.

    # 1) WORKLOADS dict
    if not _WORKLOADS_RE.search(html):
        raise RuntimeError("template: could not find `const WORKLOADS = { ... };` block")
    html = _WORKLOADS_RE.sub(lambda _m: workloads_js, html, count=1)

    # 2) <select> dropdown options
    if not _DROPDOWN_RE.search(html):
        raise RuntimeError('template: could not find `<select id="workload">` block')
    html = _DROPDOWN_RE.sub(
        lambda m: f"{m.group(1)}\n{dropdown_html}\n  {m.group(3)}",
        html,
        count=1,
    )

    # 3) GPU name in subtitle
    html = _SUBTITLE_GPU_RE.sub(
        lambda m: f"{m.group(1)}{gpu_name}{m.group(2)}", html, count=1)

    # 4) Peak DRAM / Tensor-Core constants (used in the bandwidth-utilization view)
    html = _PEAK_DRAM_RE.sub(f"const PEAK_DRAM = {int(peak_bw)};", html)
    html = _PEAK_TC_RE.sub(f"const PEAK_TC = {int(peak_tc)};", html)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[ok] wrote {out_path}")
    print(f"     GPU      : {gpu_name}")
    print(f"     peak BW  : {peak_bw:.0f} GB/s")
    print(f"     peak BF16 TC : {peak_tc:.0f} TFLOPS")
    print(f"     workloads: {len([w for w in WORKLOADS if find_row(rows, w, 'cuBLAS_TC') and find_row(rows, w, 'CompGEMM')])}/{len(WORKLOADS)} populated")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_emit_workloads() -> None:
    """Print one workload per line in a shell-friendly format:
    KEY|MODEL|LAYER|M|K|N|SPLITK
    """
    for wl in WORKLOADS:
        print("|".join([wl["key"], wl["model"], wl["layer"],
                        str(wl["M"]), str(wl["K"]), str(wl["N"]), str(wl["SplitK"])]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emit-workloads", action="store_true",
                   help="print canonical workload list and exit")
    p.add_argument("--csv",      default="bf16_triplebm_res.csv")
    p.add_argument("--template", default="dataflow_visualization.html")
    p.add_argument("--output",   default=None,
                   help="output HTML path (default: reports/report_<gpu_slug>.html)")
    p.add_argument("--gpu-name", default=None,
                   help="override GPU name (default: detected via nvidia-smi)")
    p.add_argument("--peak-bw",  type=float, default=None,
                   help="peak DRAM bandwidth in GB/s (override preset)")
    p.add_argument("--peak-tc",  type=float, default=None,
                   help="peak BF16 Tensor-Core TFLOPS (override preset)")
    args = p.parse_args()

    if args.emit_workloads:
        cmd_emit_workloads()
        return 0

    gpu_name = args.gpu_name or detect_gpu_name()
    bw_default, tc_default = lookup_gpu_specs(gpu_name)
    peak_bw = args.peak_bw if args.peak_bw is not None else bw_default
    peak_tc = args.peak_tc if args.peak_tc is not None else tc_default

    if args.output is None:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", gpu_name).strip("_") or "unknown_gpu"
        args.output = f"reports/report_{slug}.html"

    if not os.path.isfile(args.template):
        print(f"[error] template not found: {args.template}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.csv):
        print(f"[error] csv not found: {args.csv}", file=sys.stderr)
        return 1

    render_html(args.template, args.csv, args.output,
                gpu_name=gpu_name, peak_bw=peak_bw, peak_tc=peak_tc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
