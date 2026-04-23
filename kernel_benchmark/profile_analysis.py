#!/usr/bin/env python3
"""
ZipGEMM Performance Analysis Tool
Computes theoretical bandwidth, roofline position, and instruction analysis
based on benchmark results and kernel source code understanding.
"""

import csv
import sys

# ============================================================
# Hardware Specs: RTX PRO 6000 Blackwell Max-Q
# ============================================================
# Memory: 96GB GDDR7, Memory Clock 14001 MHz
# 96GB with GDDR7 8GB modules → 12 modules × 32-bit = 384-bit bus
# GDDR7 effective rate: 28 Gbps/pin (2× memory clock for DDR)
# Theoretical BW: 28 × 384 / 8 = 1344 GB/s
# Alternative: if 512-bit bus (16×2GB modules): 28 × 512 / 8 = 1792 GB/s
# SM: 3090 MHz max boost, Blackwell architecture

# We'll estimate actual bandwidth from the cuBLAS non-TC results
# (which are purely memory-bound at small N)

GPU_NAME = "RTX PRO 6000 Blackwell Max-Q"
MEMORY_CLOCK_MHZ = 14001
SM_CLOCK_MHZ = 3090

# ============================================================
# Read benchmark data
# ============================================================
def read_csv(filename):
    results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'model': row['Model'],
                'layer': row['Layer'],
                'M': int(row['M']),
                'K': int(row['K']),
                'N': int(row['N']),
                'splitk': int(row['SplitK']),
                'kernel': row['Kernel'],
                'time_ms': float(row['Duration(ms)']),
                'tflops': float(row['TFLOPS']),
            })
    return results

def compute_analysis(results):
    # Group by (M, K, N, SplitK)
    groups = {}
    for r in results:
        key = (r['M'], r['K'], r['N'], r['splitk'], r['model'], r['layer'])
        if key not in groups:
            groups[key] = {}
        groups[key][r['kernel']] = r

    print("=" * 120)
    print(f"ZipGEMM Performance Deep Analysis on {GPU_NAME}")
    print("=" * 120)

    # ============================================================
    # Section 1: Bandwidth & Roofline Analysis
    # ============================================================
    print("\n" + "=" * 120)
    print("SECTION 1: DRAM Bandwidth & Roofline Analysis")
    print("=" * 120)
    print(f"\n{'Model':<16} {'Layer':<16} {'M':>6} {'K':>6} {'N':>4} | "
          f"{'cuBLAS_TC':>10} {'ZipGEMM':>10} {'Speedup':>8} | "
          f"{'BW_cuBLAS':>10} {'BW_ZipG':>10} {'CI_cuBLAS':>10} {'CI_ZipG':>10}")
    print("-" * 120)

    for key, kernels in sorted(groups.items()):
        M, K, N, splitk, model, layer = key
        if 'cuBLAS_TC' not in kernels or 'CompGEMM' not in kernels:
            continue

        tc = kernels['cuBLAS_TC']
        zg = kernels['CompGEMM']

        # FLOPs
        flops = 2.0 * M * N * K

        # cuBLAS DRAM bytes: Weight(M×K×2) + Activation(K×N×2) + Output(M×N×2)
        cublas_bytes = M * K * 2 + K * N * 2 + M * N * 2

        # ZipGEMM DRAM bytes (compressed weight):
        # Compression ratio ~1.42x, so compressed weight = M*K*2/1.42
        cr = 1.42
        compressed_weight_bytes = M * K * 2 / cr
        # Plus: bitmap overhead (3 × 64-bit per 8×8 tile) = 3×8 bytes per 64 elements = 24/64 = 0.375 bytes/element
        num_tiles_8x8 = (M // 8) * (K // 8)
        bitmap_bytes = num_tiles_8x8 * 3 * 8  # 3 bitmaps × 8 bytes each
        # Median/Global tile offsets (small relative to weight)
        num_median_tiles = (M // 16) * (K // 64)
        num_global_tiles = (M // 64) * (K // 64)
        offset_bytes = num_median_tiles * 2 * 4 + (num_global_tiles + 1) * 2 * 4
        # Sign+mantissa array: ~97% elements × 1 byte each
        hf_ratio = 0.97
        sign_mantissa_bytes = M * K * hf_ratio * 1
        # Full BF16 fallback: ~3% elements × 2 bytes each
        full_bf16_bytes = M * K * (1 - hf_ratio) * 2

        zipgemm_weight_bytes = sign_mantissa_bytes + full_bf16_bytes + bitmap_bytes + offset_bytes
        zipgemm_bytes = zipgemm_weight_bytes + K * N * 2 + M * N * 2

        # Effective bandwidth (GB/s)
        bw_cublas = cublas_bytes / (tc['time_ms'] * 1e-3) / 1e9
        bw_zipgemm = zipgemm_bytes / (zg['time_ms'] * 1e-3) / 1e9

        # Compute Intensity (FLOPs/byte)
        ci_cublas = flops / cublas_bytes
        ci_zipgemm = flops / zipgemm_bytes

        speedup = tc['time_ms'] / zg['time_ms']

        print(f"{model:<16} {layer:<16} {M:>6} {K:>6} {N:>4} | "
              f"{tc['time_ms']:>9.3f}ms {zg['time_ms']:>9.3f}ms {speedup:>7.2f}x | "
              f"{bw_cublas:>8.0f}GB/s {bw_zipgemm:>8.0f}GB/s "
              f"{ci_cublas:>9.1f} {ci_zipgemm:>9.1f}")

    # ============================================================
    # Section 2: Memory Traffic Reduction Analysis
    # ============================================================
    print("\n" + "=" * 120)
    print("SECTION 2: Memory Traffic Reduction Analysis (Weight Matrix Only)")
    print("=" * 120)
    print(f"\n{'M':>6} {'K':>6} {'N':>4} | {'Original':>12} {'Compressed':>12} {'Ratio':>7} | "
          f"{'Bitmap OH':>10} {'SM+Full':>10} {'Offsets':>10} | "
          f"{'Weight%':>8}")
    print("-" * 120)

    seen = set()
    for key, kernels in sorted(groups.items()):
        M, K, N, splitk, model, layer = key
        mk_key = (M, K, N)
        if mk_key in seen:
            continue
        seen.add(mk_key)

        # Original weight bytes
        orig = M * K * 2

        # Compressed components
        hf_ratio = 0.97
        num_tiles_8x8 = (M // 8) * (K // 8)
        bitmap_bytes = num_tiles_8x8 * 3 * 8
        sign_mantissa = int(M * K * hf_ratio) * 1
        full_bf16 = int(M * K * (1 - hf_ratio)) * 2
        num_median_tiles = (M // 16) * (K // 64)
        num_global_tiles = (M // 64) * (K // 64)
        offsets = num_median_tiles * 2 * 4 + (num_global_tiles + 1) * 2 * 4

        compressed = sign_mantissa + full_bf16 + bitmap_bytes + offsets
        cr = orig / compressed

        # Activation bytes
        act_bytes = K * N * 2
        out_bytes = M * N * 2
        total_cublas = orig + act_bytes + out_bytes
        weight_pct = orig / total_cublas * 100

        print(f"{M:>6} {K:>6} {N:>4} | {orig/1e6:>10.2f}MB {compressed/1e6:>10.2f}MB {cr:>6.2f}x | "
              f"{bitmap_bytes/1e6:>8.2f}MB {(sign_mantissa+full_bf16)/1e6:>8.2f}MB {offsets/1e3:>8.1f}KB | "
              f"{weight_pct:>6.1f}%")

    # ============================================================
    # Section 3: Decompression Instruction Cost Analysis
    # ============================================================
    print("\n" + "=" * 120)
    print("SECTION 3: Per-Element Decompression Instruction Cost (Theoretical)")
    print("=" * 120)
    print("""
Per-element decompression in ZipGEMM (from L_Kernel.cuh):

Each thread processes 2 elements (pos1, pos2) from one 8×8 tile.
For each element pair, the instruction sequence is:

  1. Bitmap Load:     3× smem load (bitmap1,2,3)  → 3 LDS instructions
  2. Indicator OR:    bitmap1 | bitmap2 | bitmap3  → 2 LOP3 instructions
  3. Code Extract:    3× shift+AND+shift+OR        → ~9 integer ALU ops (LOP3/SHF)
  4. Exponent Calc:   start_exp + code → shift      → 2 IADD + 1 SHL
  5. Mask Compute:    (1ULL << pos) - 1             → 1 SHL + 1 IADD
  6. PopCount:        __popcll(indicator & mask)     → 1 LOP3 + 1 POPC
  7. Branch:          is_high_freq check             → 1 SETP + 1 BRA
  8. Buffer Load:     smem load (sign_mantissa/full) → 1 LDS
  9. BF16 Assembly:   sign|exp|mantissa reconstruct  → ~3 LOP3 ops
  10. Shuffle:        __shfl_sync for offset update   → 1 SHFL

  Total per 2 elements: ~25-30 ALU ops + 4-5 smem loads + 1 shuffle
  Amortized per element: ~13-15 ALU ops

For comparison, cuBLAS loads weight via ldmatrix (1 instruction for 16×16 tile = 256 elements)
  → cuBLAS: ~0.5 instructions per element for weight loading
  → ZipGEMM: ~15 instructions per element for decompression

  This ~30x instruction overhead is the PRICE for ~30% bandwidth reduction.
  ZipGEMM wins when bandwidth savings > ALU cost (memory-bound regime).
""")

    # ============================================================
    # Section 4: Pipeline Overlap Analysis
    # ============================================================
    print("=" * 120)
    print("SECTION 4: Double-Buffered Pipeline Analysis")
    print("=" * 120)
    print("""
ZipGEMM's main loop (from L_Kernel.cuh BF16TripleBitmap_MM_Kernel_Fast):

  Each K-iteration processes a 64×64 weight tile in 4 slices (each 16×64):

  ┌─── Async cp.async: Load NEXT tile (bitmap+values+activation) ────────────┐
  │                                                                           │
  │   Slice 0:  [LoadNext(1)]  →  [MMA(0)]                                   │
  │   Slice 1:  [LoadNext(2)]  →  [MMA(1)]                                   │
  │   Slice 2:  [LoadNext(3)]  →  [MMA(2)]                                   │
  │   Slice 3:  cp_async_wait  →  __syncthreads  →  [MMA(3)]                 │
  │                                                                           │
  │   If next tile exists: [LoadNext(0) from new buffer]                      │
  └───────────────────────────────────────────────────────────────────────────┘

  Pipeline stages overlap:
  ┌──────────┬──────────┬──────────┐
  │ Global→  │ Decomp   │ Tensor   │
  │ Shared   │ (ALU)    │ Core MMA │
  │ (cp.async│          │          │
  └──────────┴──────────┴──────────┘
     ↑ async      ↑ INT pipe    ↑ FP pipe

  Key: cp.async bypasses L1 and uses the memory subsystem independently.
  Decompression uses INTEGER pipeline (LOP3, POPC, IADD, SHF).
  MMA uses TENSOR CORE pipeline.
  All three can execute concurrently on separate hardware units!

  This is WHY ZipGEMM can hide decompression cost:
  - Tensor Cores are idle during memory stalls in cuBLAS
  - ZipGEMM fills this idle time with useful decompression work
  - Net effect: less DRAM traffic, same (or slightly less) Tensor Core throughput
""")

    # ============================================================
    # Section 5: Why ZipGEMM wins/loses analysis
    # ============================================================
    print("=" * 120)
    print("SECTION 5: When & Why ZipGEMM Wins or Loses")
    print("=" * 120)

    print("\n--- Speedup vs Matrix Size (M×K) at N=32 ---")
    print(f"{'Model':<16} {'Layer':<16} {'M':>6} {'K':>6} {'M×K':>12} {'Speedup':>8} {'Winner':>10}")
    print("-" * 80)

    n32_results = []
    for key, kernels in sorted(groups.items()):
        M, K, N, splitk, model, layer = key
        if N != 32 or splitk != 1:
            continue
        if 'cuBLAS_TC' not in kernels or 'CompGEMM' not in kernels:
            continue
        tc = kernels['cuBLAS_TC']
        zg = kernels['CompGEMM']
        speedup = tc['time_ms'] / zg['time_ms']
        winner = "ZipGEMM" if speedup > 1.0 else "cuBLAS"
        mk = M * K
        n32_results.append((mk, M, K, speedup, model, layer, winner))

    for mk, M, K, speedup, model, layer, winner in sorted(n32_results):
        print(f"{model:<16} {layer:<16} {M:>6} {K:>6} {mk:>12,} {speedup:>7.2f}x {winner:>10}")

    print("""
Analysis:
  - ZipGEMM wins when M×K is LARGE (≥ ~92M elements for GateUp_proj shapes)
  - This is because larger matrices are more memory-bound at small N
  - For the same M×K, shapes with LARGE M (many rows) win vs large K (many cols)
    because ZipGEMM tiles along M (each thread block handles TILE_M=16 rows)
    → More M means more thread blocks → better GPU occupancy
  - Down_proj (small M, large K) loses badly because:
    1. Few thread blocks along M → low occupancy
    2. Long K iterations per block → more pressure on software pipeline
    3. cuBLAS can use different, optimized algorithms for tall-skinny shapes
""")

    # ============================================================
    # Section 6: N-scaling analysis
    # ============================================================
    print("=" * 120)
    print("SECTION 6: Batch Size (N) Scaling Analysis for 70B GateUp_proj (M=57344, K=8192)")
    print("=" * 120)

    print(f"\n{'N':>4} | {'cuBLAS_TC':>10} {'TFLOPS':>8} {'ZipGEMM':>10} {'TFLOPS':>8} | "
          f"{'Speedup':>8} {'CI_cuBLAS':>10} {'CI_ZipG':>10} {'Regime':>15}")
    print("-" * 100)

    for key, kernels in sorted(groups.items()):
        M, K, N, splitk, model, layer = key
        if M != 57344 or K != 8192 or splitk != 1:
            continue
        if 'cuBLAS_TC' not in kernels or 'CompGEMM' not in kernels:
            continue

        tc = kernels['cuBLAS_TC']
        zg = kernels['CompGEMM']
        flops = 2.0 * M * N * K
        cublas_bytes = M * K * 2 + K * N * 2 + M * N * 2
        cr = 1.42
        zipgemm_bytes = M * K * 2 / cr + K * N * 2 + M * N * 2
        ci_cublas = flops / cublas_bytes
        ci_zipgemm = flops / zipgemm_bytes
        speedup = tc['time_ms'] / zg['time_ms']

        # Determine regime based on Tensor Core utilization
        # Peak BF16 TC TFLOPS for this GPU is probably ~500+ TFLOPS
        peak_tflops_estimate = 500
        tc_util = tc['tflops'] / peak_tflops_estimate * 100
        if tc_util < 10:
            regime = "memory-bound"
        elif tc_util < 40:
            regime = "transitional"
        else:
            regime = "compute-bound"

        print(f"{N:>4} | {tc['time_ms']:>9.3f}ms {tc['tflops']:>7.1f} {zg['time_ms']:>9.3f}ms {zg['tflops']:>7.1f} | "
              f"{speedup:>7.2f}x {ci_cublas:>9.1f} {ci_zipgemm:>9.1f} {regime:>15}")

    print("""
Key Insight:
  ZipGEMM wins at N=32 and N=64 (transitional regime), loses at N=8,16 and N=128.

  Why N=8,16 LOSE despite being memory-bound:
  - Very small N → few thread blocks along N → GPU underutilized
  - Kernel launch overhead and decompression setup cost dominate
  - cuBLAS uses specialized tiny-N algorithms (e.g., batched GEMV)

  Why N=32,64 WIN:
  - Enough parallelism to saturate GPU, but still memory-bound
  - Bandwidth savings from compression directly translate to speedup
  - Decompression ALU cost hidden by pipeline overlap

  Why N=128 LOSES:
  - Enters compute-bound regime (cuBLAS reaches 137.8 TFLOPS)
  - Tensor Cores are well-utilized → bandwidth is no longer bottleneck
  - ZipGEMM's decompression ALU work now competes with useful compute
""")


if __name__ == '__main__':
    csv_file = 'bf16_triplebm_res.csv'
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    results = read_csv(csv_file)
    compute_analysis(results)
