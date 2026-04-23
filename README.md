# ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

**📄 ASPLOS'26 Paper** | Paper (To appear - contact authors: ruibo.fan@connect.hkust-gz.edu.cn) | [Code](https://github.com/HPMLL/ZipServ_ASPLOS26)

**Authors**: Ruibo Fan¹, Xiangrui Yu¹, Xinglin Pan¹, Zeyu Li¹, Weile Luo¹, Qiang Wang², Wei Wang³, Xiaowen Chu¹,³  
¹ HKUST(GZ), ² HIT(SZ), ³ HKUST

## Overview

ZipServ is the first **lossless compression framework** co-designed for high-performance LLM inference on GPUs. Through hardware-aware compression algorithms and fused kernel design, ZipServ achieves significant inference acceleration while maintaining bit-exact numerical precision, solving the performance bottleneck of traditional lossless compression methods on GPUs.

> **Note**: ZipServ is optimized for **GDDR memory-based GPUs** (e.g., RTX 4090, L40S, RTX 5090, RTX 6000 Pro) where memory bandwidth is the primary bottleneck. Performance gains may vary on HBM-based datacenter GPUs.

### Key Advantages

- **Lossless Compression**: Up to 30% model size reduction with bit-exact numerical precision
- **Significant Speedup**: Up to 2.21× kernel-level speedup over cuBLAS, 1.22× average end-to-end speedup over vLLM
- **Hardware Co-design**: Optimized for NVIDIA Tensor Core architectures
- **First Practical System**: First demonstration that lossless compression can simultaneously provide storage savings and inference acceleration

## Core Techniques

### 1. Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE)

Addressing the fundamental mismatch between traditional entropy coding (e.g., Huffman, ANS) and GPU SIMT execution model, ZipServ proposes TCA-TBE encoding:

- **Fixed-Length Format**: Exploits highly skewed distribution of BFloat16 weight exponents using triple bitmap encoding
- **Constant-Time Parallel Decoding**: Achieves branch-free parallel decoding through lightweight bitwise operations, eliminating control-flow divergence
- **Tensor Core Alignment**: Hierarchical tiling organization (8×8, 16×64, 64×64) seamlessly aligns with Tensor Core computation patterns

### 2. ZipGEMM Fused Kernel

Addressing the redundant memory access caused by traditional decoupled pipelines (decompress to global memory, then compute), ZipServ designs a fused decompression-GEMM kernel:

- **"Load-Compressed, Compute-Decompressed"**: Compressed weights are loaded directly into registers and decoded on-the-fly
- **Eliminate Intermediate Buffers**: Decompressed results feed directly into Tensor Cores without going through global memory
- **Maximize Compute Intensity**: Overlaps data movement with computation to fully utilize memory bandwidth

## Quick Start

### Prerequisites

- **CUDA Toolkit**: 11.8 or later
- **GPU**: NVIDIA Ampere architecture or newer (Compute Capability ≥ 8.0)
- **Compiler**: g++ with C++14 support

### Build

```bash
# Initialize environment
source Init.sh

# Build core library
cd build && make

# Build benchmarks
cd ../kernel_benchmark && source test_env && make
```

### Run Benchmarks

Single-shape sanity check:

```bash
# Matrix multiplication benchmark
cd kernel_benchmark
./test_mm M K N SplitK

# Example: ./test_mm 4096 4096 128 1
```

One-click benchmark + interactive HTML report (covers all canonical LLaMA /
Mistral GEMM shapes, auto-detects GPU, writes
`kernel_benchmark/reports/report_<gpu>.html`):

```bash
cd kernel_benchmark
./run_benchmark.sh
```

For full agent-followable instructions (prereq checks, flags, troubleshooting,
how to add new GPUs/workloads), see
[`kernel_benchmark/AGENTS.md`](kernel_benchmark/AGENTS.md).

## Citation

If you use ZipServ in your research, please cite our ASPLOS'26 paper:

```bibtex
@inproceedings{zipserv2026,
  title={ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression},
  author={Fan, Ruibo and Yu, Xiangrui and Pan, Xinglin and Li, Zeyu and Luo, Weile and Wang, Qiang and Wang, Wei and Chu, Xiaowen},
  booktitle={Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS'26)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**Note**: This is research prototype code. Please conduct thorough testing and validation before production use.

