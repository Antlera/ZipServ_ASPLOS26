// Phase 0 benchmark: does sharing a base weight matrix across multiple
// variants within a batch produce runtime (bandwidth) savings vs. storing
// each variant's weights independently?
//
// Two kernels:
//   A. gemv_naive_per_request:      W_variants[V][M][K] packed as-if merged.
//                                   Each block handles (row, request); weights
//                                   are read independently per request.
//   B. gemv_shared_base_segmented:  W_base[M][K] + W_deltas[V][M][K] stored
//                                   separately. Requests are pre-sorted by
//                                   variant. Each block handles (row, variant):
//                                   merges base+delta into smem once, then
//                                   iterates over all requests for that variant.
//
// Bandwidth model (weights only, dominates):
//   Naive:        R        * M * K * 2 bytes
//   Shared-base:  2 * V    * M * K * 2 bytes  (base read V times in worst case)
//                 (V+1)    * M * K * 2 bytes  (base perfectly L2-cached)
//   Shared-base wins when R > 2V (conservative) or R > V+1 (L2 friendly).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                            \
                cudaGetErrorString(_e), __FILE__, __LINE__);                   \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

// =============================================================================
// Kernel A: naive per-request GEMV
// Grid = (M, R), block = 32 threads (one warp).
// Each block: row = blockIdx.x, request = blockIdx.y.
// Reads the full weight row for that request's variant.
// =============================================================================
__global__ void gemv_naive_per_request(
    const __nv_bfloat16* __restrict__ W_variants,   // [V, M, K], row-major
    const __nv_bfloat16* __restrict__ X,            // [R, K]
    const int*           __restrict__ variant_id,   // [R]
    __nv_bfloat16*       __restrict__ Y,            // [R, M]
    int M, int K)
{
    int row  = blockIdx.x;
    int r    = blockIdx.y;
    int lane = threadIdx.x;
    int v    = variant_id[r];

    size_t MK = (size_t)M * (size_t)K;
    const __nv_bfloat16* w_row = W_variants + (size_t)v * MK + (size_t)row * K;
    const __nv_bfloat16* x_r   = X + (size_t)r * K;

    float acc = 0.0f;
    #pragma unroll 4
    for (int k = lane; k < K; k += 32) {
        acc += __bfloat162float(w_row[k]) * __bfloat162float(x_r[k]);
    }
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        Y[(size_t)r * M + row] = __float2bfloat16(acc);
    }
}

// =============================================================================
// Kernel B: shared-base segmented GEMV
// Grid  = (M, V), block = N_WARPS * 32 threads.
// Each block: row = blockIdx.x, variant = blockIdx.y.
// All threads cooperatively load merged (base + delta) row into smem once;
// then each warp takes a stride of requests within the variant's segment.
// =============================================================================
template <int N_WARPS>
__global__ void gemv_shared_base_segmented(
    const __nv_bfloat16* __restrict__ W_base,         // [M, K]
    const __nv_bfloat16* __restrict__ W_deltas,       // [V, M, K]
    const __nv_bfloat16* __restrict__ X_sorted,       // [R, K], sorted by variant
    const int*           __restrict__ variant_offsets,// [V+1], prefix sum
    __nv_bfloat16*       __restrict__ Y_sorted,       // [R, M]
    int M, int K)
{
    int row     = blockIdx.x;
    int v       = blockIdx.y;
    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;

    int r_beg = variant_offsets[v];
    int r_end = variant_offsets[v + 1];
    if (r_beg == r_end) return;

    size_t MK = (size_t)M * (size_t)K;
    const __nv_bfloat16* wb_row = W_base   + (size_t)row * K;
    const __nv_bfloat16* wd_row = W_deltas + (size_t)v * MK + (size_t)row * K;

    extern __shared__ __nv_bfloat16 smem_W[];  // size = K * sizeof(bf16)

    // Phase 1: cooperative load + merge base+delta into smem.
    // Merge in fp32 so numerical result matches the naive kernel's pre-merged input.
    #pragma unroll 4
    for (int k = tid; k < K; k += N_WARPS * 32) {
        float wm = __bfloat162float(wb_row[k]) + __bfloat162float(wd_row[k]);
        smem_W[k] = __float2bfloat16(wm);
    }
    __syncthreads();

    // Phase 2: each warp strides through this variant's request segment.
    for (int r = r_beg + warp_id; r < r_end; r += N_WARPS) {
        const __nv_bfloat16* x_r = X_sorted + (size_t)r * K;
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = lane; k < K; k += 32) {
            acc += __bfloat162float(smem_W[k]) * __bfloat162float(x_r[k]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            Y_sorted[(size_t)r * M + row] = __float2bfloat16(acc);
        }
    }
}

// =============================================================================
// Kernel C: shared-base segmented GEMV with SIMULATED compressed delta
// Delta is stored as 1 byte/element (int8) instead of 2 bytes (bf16).
// This simulates ~2x delta compression (zeltax-style) without actually
// implementing the zigzag/bitmap decode — we just need to measure the DRAM
// bandwidth effect.  The byte is interpreted as a small-scale value
// (delta_byte / 1024.0), which keeps the math numerically sensible but the
// output will NOT match the naive kernel — we skip the correctness check.
// =============================================================================
template <int N_WARPS>
__global__ void gemv_shared_base_compressed_sim(
    const __nv_bfloat16* __restrict__ W_base,         // [M, K]
    const int8_t*        __restrict__ W_deltas_i8,    // [V, M, K], 1 byte/element
    const __nv_bfloat16* __restrict__ X_sorted,       // [R, K]
    const int*           __restrict__ variant_offsets,// [V+1]
    __nv_bfloat16*       __restrict__ Y_sorted,       // [R, M]
    int M, int K)
{
    int row     = blockIdx.x;
    int v       = blockIdx.y;
    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;

    int r_beg = variant_offsets[v];
    int r_end = variant_offsets[v + 1];
    if (r_beg == r_end) return;

    size_t MK = (size_t)M * (size_t)K;
    const __nv_bfloat16* wb_row = W_base       + (size_t)row * K;
    const int8_t*        wd_row = W_deltas_i8  + (size_t)v * MK + (size_t)row * K;

    extern __shared__ __nv_bfloat16 smem_W[];

    const float delta_scale = 1.0f / 1024.0f;
    #pragma unroll 4
    for (int k = tid; k < K; k += N_WARPS * 32) {
        float b = __bfloat162float(wb_row[k]);
        float d = (float)((int8_t)wd_row[k]) * delta_scale;
        smem_W[k] = __float2bfloat16(b + d);
    }
    __syncthreads();

    for (int r = r_beg + warp_id; r < r_end; r += N_WARPS) {
        const __nv_bfloat16* x_r = X_sorted + (size_t)r * K;
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = lane; k < K; k += 32) {
            acc += __bfloat162float(smem_W[k]) * __bfloat162float(x_r[k]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            Y_sorted[(size_t)r * M + row] = __float2bfloat16(acc);
        }
    }
}

// =============================================================================
// Helper kernel: merge W_variants[v] = W_base + W_deltas[v] on device.
// Used once at setup time to build the "naive" reference input.
// =============================================================================
__global__ void merge_base_plus_delta_kernel(
    const __nv_bfloat16* __restrict__ W_base,      // [M, K]
    const __nv_bfloat16* __restrict__ W_deltas,    // [V, M, K]
    __nv_bfloat16*       __restrict__ W_variants,  // [V, M, K]
    size_t N_per_variant, int V)
{
    size_t total = (size_t)V * N_per_variant;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    size_t off_in_variant = tid % N_per_variant;
    float b = __bfloat162float(W_base[off_in_variant]);
    float d = __bfloat162float(W_deltas[tid]);
    W_variants[tid] = __float2bfloat16(b + d);
}

// =============================================================================
// Host utilities
// =============================================================================
static void fill_bf16_random(std::vector<uint16_t>& buf, std::mt19937& rng, float scale = 0.02f) {
    // Generate normal(0, scale) values, store as bf16 bits.
    std::normal_distribution<float> dist(0.0f, scale);
    for (auto& h : buf) {
        float v = dist(rng);
        uint32_t as_int;
        std::memcpy(&as_int, &v, 4);
        h = static_cast<uint16_t>((as_int + 0x7fff + ((as_int >> 16) & 1)) >> 16);
    }
}

struct KernelTime {
    float ms;
    float bandwidth_gbs;
};

static KernelTime run_naive(
    const __nv_bfloat16* W_variants_d,
    const __nv_bfloat16* X_d,
    const int* variant_id_d,
    __nv_bfloat16* Y_d,
    int M, int K, int R, int V, int warmup, int iters)
{
    dim3 grid(M, R);
    dim3 block(32);

    for (int i = 0; i < warmup; i++) {
        gemv_naive_per_request<<<grid, block>>>(W_variants_d, X_d, variant_id_d, Y_d, M, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) {
        gemv_naive_per_request<<<grid, block>>>(W_variants_d, X_d, variant_id_d, Y_d, M, K);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float elapsed;
    cudaEventElapsedTime(&elapsed, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    float ms_per_iter = elapsed / iters;
    // Weight bytes / iter = R * M * K * 2
    double bytes = (double)R * (double)M * (double)K * 2.0;
    double gbs   = bytes / (ms_per_iter * 1e-3) / 1e9;
    return {ms_per_iter, (float)gbs};
}

template <int N_WARPS>
static KernelTime run_shared_impl(
    const __nv_bfloat16* W_base_d,
    const __nv_bfloat16* W_deltas_d,
    const __nv_bfloat16* X_sorted_d,
    const int* variant_offsets_d,
    __nv_bfloat16* Y_sorted_d,
    int M, int K, int V, int R, int warmup, int iters)
{
    dim3 grid(M, V);
    dim3 block(N_WARPS * 32);
    int shmem_bytes = K * (int)sizeof(__nv_bfloat16);

    auto kptr = gemv_shared_base_segmented<N_WARPS>;
    CUDA_CHECK(cudaFuncSetAttribute(
        kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));

    for (int i = 0; i < warmup; i++) {
        kptr<<<grid, block, shmem_bytes>>>(
            W_base_d, W_deltas_d, X_sorted_d, variant_offsets_d, Y_sorted_d, M, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) {
        kptr<<<grid, block, shmem_bytes>>>(
            W_base_d, W_deltas_d, X_sorted_d, variant_offsets_d, Y_sorted_d, M, K);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float elapsed;
    cudaEventElapsedTime(&elapsed, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    float ms_per_iter = elapsed / iters;
    // Best-case weight traffic (base shared across variants via L2):
    // (V + 1) * M * K * 2 bytes. We report this bound so the number is comparable
    // across kernels (same numerator, so bandwidth reflects kernel efficiency).
    double bytes = ((double)V + 1.0) * (double)M * (double)K * 2.0;
    double gbs   = bytes / (ms_per_iter * 1e-3) / 1e9;
    return {ms_per_iter, (float)gbs};
}

static KernelTime run_shared(
    const __nv_bfloat16* W_base_d,
    const __nv_bfloat16* W_deltas_d,
    const __nv_bfloat16* X_sorted_d,
    const int* variant_offsets_d,
    __nv_bfloat16* Y_sorted_d,
    int M, int K, int V, int R, int R_per_variant, int warmup, int iters)
{
    // Pick N_WARPS = min(4, next-pow-2 of R_per_variant). 4 warps covers R_v up to 4.
    if (R_per_variant <= 1) {
        return run_shared_impl<1>(W_base_d, W_deltas_d, X_sorted_d, variant_offsets_d,
                                   Y_sorted_d, M, K, V, R, warmup, iters);
    } else if (R_per_variant <= 2) {
        return run_shared_impl<2>(W_base_d, W_deltas_d, X_sorted_d, variant_offsets_d,
                                   Y_sorted_d, M, K, V, R, warmup, iters);
    } else if (R_per_variant <= 4) {
        return run_shared_impl<4>(W_base_d, W_deltas_d, X_sorted_d, variant_offsets_d,
                                   Y_sorted_d, M, K, V, R, warmup, iters);
    } else {
        return run_shared_impl<8>(W_base_d, W_deltas_d, X_sorted_d, variant_offsets_d,
                                   Y_sorted_d, M, K, V, R, warmup, iters);
    }
}

template <int N_WARPS>
static KernelTime run_compressed_sim_impl(
    const __nv_bfloat16* W_base_d,
    const int8_t*        W_deltas_i8_d,
    const __nv_bfloat16* X_sorted_d,
    const int* variant_offsets_d,
    __nv_bfloat16* Y_sorted_d,
    int M, int K, int V, int R, int warmup, int iters)
{
    dim3 grid(M, V);
    dim3 block(N_WARPS * 32);
    int shmem_bytes = K * (int)sizeof(__nv_bfloat16);
    auto kptr = gemv_shared_base_compressed_sim<N_WARPS>;
    CUDA_CHECK(cudaFuncSetAttribute(
        kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));

    for (int i = 0; i < warmup; i++) {
        kptr<<<grid, block, shmem_bytes>>>(
            W_base_d, W_deltas_i8_d, X_sorted_d, variant_offsets_d, Y_sorted_d, M, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) {
        kptr<<<grid, block, shmem_bytes>>>(
            W_base_d, W_deltas_i8_d, X_sorted_d, variant_offsets_d, Y_sorted_d, M, K);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float elapsed;
    cudaEventElapsedTime(&elapsed, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    float ms_per_iter = elapsed / iters;
    // DRAM bytes: base (2 bytes/element) + V * delta (1 byte/element)
    double bytes = (double)M * (double)K * 2.0 + (double)V * (double)M * (double)K * 1.0;
    double gbs   = bytes / (ms_per_iter * 1e-3) / 1e9;
    return {ms_per_iter, (float)gbs};
}

static KernelTime run_compressed_sim(
    const __nv_bfloat16* W_base_d,
    const int8_t*        W_deltas_i8_d,
    const __nv_bfloat16* X_sorted_d,
    const int* variant_offsets_d,
    __nv_bfloat16* Y_sorted_d,
    int M, int K, int V, int R, int R_per_variant, int warmup, int iters)
{
    if (R_per_variant <= 1) {
        return run_compressed_sim_impl<1>(W_base_d, W_deltas_i8_d, X_sorted_d,
            variant_offsets_d, Y_sorted_d, M, K, V, R, warmup, iters);
    } else if (R_per_variant <= 2) {
        return run_compressed_sim_impl<2>(W_base_d, W_deltas_i8_d, X_sorted_d,
            variant_offsets_d, Y_sorted_d, M, K, V, R, warmup, iters);
    } else if (R_per_variant <= 4) {
        return run_compressed_sim_impl<4>(W_base_d, W_deltas_i8_d, X_sorted_d,
            variant_offsets_d, Y_sorted_d, M, K, V, R, warmup, iters);
    } else {
        return run_compressed_sim_impl<8>(W_base_d, W_deltas_i8_d, X_sorted_d,
            variant_offsets_d, Y_sorted_d, M, K, V, R, warmup, iters);
    }
}

// Compare two Y buffers (bf16) with an absolute-error tolerance.
static bool verify(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                   int R, int M, float atol = 3e-2f) {
    int mismatches = 0;
    float max_err = 0.0f;
    int first_r = -1, first_m = -1;
    for (int r = 0; r < R; r++) {
        for (int m = 0; m < M; m++) {
            uint32_t ai = (uint32_t)a[r * M + m] << 16;
            uint32_t bi = (uint32_t)b[r * M + m] << 16;
            float af, bf;
            std::memcpy(&af, &ai, 4);
            std::memcpy(&bf, &bi, 4);
            float err = std::fabs(af - bf);
            if (err > max_err) max_err = err;
            if (err > atol) {
                if (first_r < 0) { first_r = r; first_m = m; }
                mismatches++;
            }
        }
    }
    if (mismatches) {
        printf("  [verify] %d mismatches (first at r=%d m=%d), max_err=%f\n",
               mismatches, first_r, first_m, max_err);
    } else {
        printf("  [verify] OK (max_err=%f within tol=%f)\n", max_err, atol);
    }
    return mismatches == 0;
}

int main(int argc, char** argv) {
    // Defaults modeled after LLaMA-8B down_proj: y = W @ x where W is [hidden, ffn].
    int M = 4096;
    int K = 14336;
    int V = 8;
    int R_per_variant = 4;
    int warmup = 10;
    int iters  = 50;
    unsigned seed = 42;

    // CLI: ./test_multi_variant M K V R_per_variant [warmup iters seed]
    if (argc >= 5) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        V = std::atoi(argv[3]);
        R_per_variant = std::atoi(argv[4]);
    }
    if (argc >= 7) {
        warmup = std::atoi(argv[5]);
        iters  = std::atoi(argv[6]);
    }
    if (argc >= 8) {
        seed = (unsigned)std::atoi(argv[7]);
    }

    int R = V * R_per_variant;

    printf("\n=== Phase 0 Multi-Variant GEMV Benchmark ===\n");
    printf("M=%d K=%d V=%d R_per_variant=%d  (R=%d)\n", M, K, V, R_per_variant, R);
    printf("warmup=%d iters=%d seed=%u\n\n", warmup, iters, seed);

    size_t N_per_variant = (size_t)M * (size_t)K;
    size_t base_bytes   = N_per_variant * sizeof(__nv_bfloat16);
    size_t delta_bytes  = (size_t)V * base_bytes;
    size_t x_bytes      = (size_t)R * (size_t)K * sizeof(__nv_bfloat16);
    size_t y_bytes      = (size_t)R * (size_t)M * sizeof(__nv_bfloat16);

    double total_vram_gb =
        (base_bytes + delta_bytes * 2 /* deltas + merged variants */ +
         x_bytes * 2 /* sorted + unsorted */ + y_bytes * 2) / 1e9;
    printf("Approx VRAM footprint: %.2f GB\n", total_vram_gb);

    // ---- Host init ----
    std::mt19937 rng(seed);
    std::vector<uint16_t> h_W_base(N_per_variant);
    std::vector<uint16_t> h_W_deltas((size_t)V * N_per_variant);
    std::vector<uint16_t> h_X((size_t)R * K);
    fill_bf16_random(h_W_base,   rng, 0.02f);
    fill_bf16_random(h_W_deltas, rng, 0.002f); // deltas are ~10x smaller than base
    fill_bf16_random(h_X,        rng, 1.0f);

    // Assign each request to a variant (round-robin initially; could randomize)
    std::vector<int> h_variant_id(R);
    for (int r = 0; r < R; r++) h_variant_id[r] = r / R_per_variant;

    // Build sorted arrangement (sort requests by variant).
    // Since we used round-robin assignment, it's already sorted; but do it
    // explicitly so the code generalizes to random assignment later.
    std::vector<int> sort_perm(R);
    for (int r = 0; r < R; r++) sort_perm[r] = r;
    std::stable_sort(sort_perm.begin(), sort_perm.end(),
        [&](int a, int b) { return h_variant_id[a] < h_variant_id[b]; });

    std::vector<uint16_t> h_X_sorted((size_t)R * K);
    std::vector<int>      h_variant_id_sorted(R);
    for (int r = 0; r < R; r++) {
        int src = sort_perm[r];
        std::memcpy(&h_X_sorted[(size_t)r * K], &h_X[(size_t)src * K], K * sizeof(uint16_t));
        h_variant_id_sorted[r] = h_variant_id[src];
    }
    // Prefix sum → variant_offsets
    std::vector<int> h_variant_offsets(V + 1, 0);
    for (int r = 0; r < R; r++) h_variant_offsets[h_variant_id_sorted[r] + 1]++;
    for (int v = 0; v < V; v++) h_variant_offsets[v + 1] += h_variant_offsets[v];

    // ---- Device alloc ----
    __nv_bfloat16 *d_W_base = nullptr, *d_W_deltas = nullptr, *d_W_variants = nullptr;
    int8_t        *d_W_deltas_i8 = nullptr;
    __nv_bfloat16 *d_X = nullptr, *d_X_sorted = nullptr;
    __nv_bfloat16 *d_Y_naive = nullptr, *d_Y_shared = nullptr, *d_Y_compsim = nullptr;
    int *d_variant_id = nullptr, *d_variant_offsets = nullptr;

    size_t delta_i8_bytes = (size_t)V * N_per_variant * sizeof(int8_t);
    CUDA_CHECK(cudaMalloc(&d_W_base,         base_bytes));
    CUDA_CHECK(cudaMalloc(&d_W_deltas,       delta_bytes));
    CUDA_CHECK(cudaMalloc(&d_W_deltas_i8,    delta_i8_bytes));
    CUDA_CHECK(cudaMalloc(&d_W_variants,     delta_bytes));
    CUDA_CHECK(cudaMalloc(&d_X,              x_bytes));
    CUDA_CHECK(cudaMalloc(&d_X_sorted,       x_bytes));
    CUDA_CHECK(cudaMalloc(&d_Y_naive,        y_bytes));
    CUDA_CHECK(cudaMalloc(&d_Y_shared,       y_bytes));
    CUDA_CHECK(cudaMalloc(&d_Y_compsim,      y_bytes));
    CUDA_CHECK(cudaMalloc(&d_variant_id,     R * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_variant_offsets,(V + 1) * sizeof(int)));

    // Populate the i8 delta buffer with the low byte of each bf16 delta — it's
    // only used to emulate "1-byte-per-element" DRAM traffic, not correctness.
    {
        std::vector<int8_t> h_W_deltas_i8(delta_i8_bytes);
        const uint16_t* src = h_W_deltas.data();
        for (size_t i = 0; i < (size_t)V * N_per_variant; i++) {
            h_W_deltas_i8[i] = (int8_t)(src[i] & 0xFF);
        }
        CUDA_CHECK(cudaMemcpy(d_W_deltas_i8, h_W_deltas_i8.data(),
                              delta_i8_bytes, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpy(d_W_base,         h_W_base.data(),    base_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_deltas,       h_W_deltas.data(),  delta_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X,              h_X.data(),         x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X_sorted,       h_X_sorted.data(),  x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_variant_id,     h_variant_id.data(),R * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_variant_offsets,h_variant_offsets.data(),(V + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Build merged W_variants on device (one-time)
    {
        size_t total = (size_t)V * N_per_variant;
        int block_sz = 256;
        size_t grid_sz = (total + block_sz - 1) / block_sz;
        merge_base_plus_delta_kernel<<<grid_sz, block_sz>>>(
            d_W_base, d_W_deltas, d_W_variants, N_per_variant, V);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Run & time ----
    printf("--- Kernel A: naive per-request ---\n");
    KernelTime tA = run_naive(d_W_variants, d_X, d_variant_id, d_Y_naive, M, K, R, V, warmup, iters);
    double bytes_naive_raw = (double)R * (double)M * (double)K * 2.0;
    double bytes_shared_best = ((double)V + 1.0) * (double)M * (double)K * 2.0;
    printf("  time: %.3f ms/iter,  raw weight-bw (R*M*K*2 read each): %.1f GB/s\n",
           tA.ms, tA.bandwidth_gbs);

    printf("--- Kernel B: shared-base segmented (delta=bf16, 2 B/elem) ---\n");
    KernelTime tB = run_shared(d_W_base, d_W_deltas, d_X_sorted, d_variant_offsets,
                               d_Y_shared, M, K, V, R, R_per_variant, warmup, iters);
    printf("  time: %.3f ms/iter,  weight-bw ((V+1)*M*K*2): %.1f GB/s\n",
           tB.ms, tB.bandwidth_gbs);

    printf("--- Kernel C: shared-base compressed-sim (delta=int8, 1 B/elem) ---\n");
    KernelTime tC = run_compressed_sim(d_W_base, d_W_deltas_i8, d_X_sorted, d_variant_offsets,
                                       d_Y_compsim, M, K, V, R, R_per_variant, warmup, iters);
    printf("  time: %.3f ms/iter,  weight-bw (M*K*2 + V*M*K*1): %.1f GB/s\n",
           tC.ms, tC.bandwidth_gbs);

    double bytes_compsim = (double)M * (double)K * 2.0 + (double)V * (double)M * (double)K * 1.0;

    printf("\nDRAM byte budget (best-case L2 sharing, weight reads only):\n");
    printf("  naive       : %.2f GB   (V*M*K*2 unique variants)\n",
           (double)V * (double)M * (double)K * 2.0 / 1e9);
    printf("  shared (bf16 delta) : %.2f GB   ((V+1)*M*K*2)\n",  bytes_shared_best / 1e9);
    printf("  compressed-sim      : %.2f GB   (M*K*2 + V*M*K*1)\n", bytes_compsim / 1e9);
    printf("\nMeasured speedups (vs naive):\n");
    printf("  shared (bf16 delta)  : %.3fx\n", tA.ms / tB.ms);
    printf("  compressed-sim       : %.3fx\n", tA.ms / tC.ms);

    // ---- Correctness: compare Y outputs (account for sort permutation) ----
    std::vector<uint16_t> h_Y_naive(R * M), h_Y_shared_sorted(R * M);
    CUDA_CHECK(cudaMemcpy(h_Y_naive.data(),          d_Y_naive,  y_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y_shared_sorted.data(),  d_Y_shared, y_bytes, cudaMemcpyDeviceToHost));

    // Un-sort the shared output: shared kernel wrote to Y_sorted[r,:] where r is sorted index.
    std::vector<uint16_t> h_Y_shared(R * M);
    for (int r = 0; r < R; r++) {
        int dst = sort_perm[r]; // h_Y_shared_sorted[r] corresponds to original request dst
        std::memcpy(&h_Y_shared[(size_t)dst * M],
                    &h_Y_shared_sorted[(size_t)r * M], M * sizeof(uint16_t));
    }
    printf("\n--- Correctness check ---\n");
    verify(h_Y_naive, h_Y_shared, R, M);

    // ---- Cleanup ----
    cudaFree(d_W_base);
    cudaFree(d_W_deltas);
    cudaFree(d_W_deltas_i8);
    cudaFree(d_W_variants);
    cudaFree(d_X);
    cudaFree(d_X_sorted);
    cudaFree(d_Y_naive);
    cudaFree(d_Y_shared);
    cudaFree(d_Y_compsim);
    cudaFree(d_variant_id);
    cudaFree(d_variant_offsets);

    printf("\n");
    return 0;
}
