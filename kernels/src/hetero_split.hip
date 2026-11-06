// hipfire — device-side hetero expert split (Qwen4 Flash-Next, design v3).
//
// The primary (gfx1201) and the expert device (gfx1151) exchange one layer's
// cold routed slots through a pinned mailbox in the expert device's RAM
// (crates/hipfire-dispatch/src/hetero_mailbox.rs owns the layout). Every
// cross-device word uses system-scope atomic loads/stores only (no RMW: the
// USB4 link has no PCIe atomics), and every wait is bounded.
//
// Primary:  hetero_token_advance, hetero_publish_cold, hetero_await_cold,
//           hetero_install_cold; prefill: hetero_publish_rows,
//           hetero_release_rows, hetero_add_partial.
// Expert:   hetero_mask_hot, hetero_store_seq; prefill: hetero_f16_to_f32,
//           hetero_f32_to_bf16.

#include <hip/hip_runtime.h>

__device__ __forceinline__ unsigned hs_load(const unsigned* p) {
    return __hip_atomic_load(const_cast<unsigned*>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void hs_store(unsigned* p, unsigned v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Mirrors hetero_mailbox::seq_for: token * layers + layer + 1, wrapping; 0 -> 1.
__device__ __forceinline__ unsigned hs_seq(unsigned token, unsigned layer, unsigned layers) {
    unsigned v = token * layers + layer + 1u;
    return v == 0u ? 1u : v;
}

__device__ __forceinline__ bool hs_cold(int e, const unsigned char* bitmap, int n_experts) {
    return e >= 0 && e < n_experts && bitmap[e] != 0;
}

// One thread: advance the device token counter (start of each decode token).
extern "C" __global__ void hetero_token_advance(unsigned* counter) {
    if (blockIdx.x == 0 && threadIdx.x == 0) counter[0] += 1u;
}

// One block. Copy the layer's `rows` rows and route into its mailbox slot (and
// the row count), write the primary's masked ids (cold -> -1), then release the
// request sequence.
extern "C" __global__ void hetero_publish_cold(
    const unsigned* __restrict__ counter,
    unsigned layer,
    unsigned layers,
    const float* __restrict__ x_rot,
    const int* __restrict__ ids,
    const float* __restrict__ weights,
    const unsigned char* __restrict__ cold_bitmap,
    int n_experts,
    int rows,
    int top_k,
    int dim,
    float* mb_x,
    int* mb_ids,
    float* mb_w,
    int* masked_ids,
    unsigned* req_seq,
    unsigned* mb_rows) {
    const int slots = rows * top_k;
    if (threadIdx.x == 0) mb_rows[0] = (unsigned)rows;
    for (int i = threadIdx.x; i < rows * dim; i += blockDim.x) mb_x[i] = x_rot[i];
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        const int e = ids[s];
        mb_ids[s] = e;
        mb_w[s] = weights[s];
        masked_ids[s] = hs_cold(e, cold_bitmap, n_experts) ? -1 : e;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        hs_store(req_seq, hs_seq(counter[0], layer, layers));
    }
}

// One thread. Bounded wait for the layer's response; on timeout record the
// expected sequence in the slot's error word (the host fails the request).
extern "C" __global__ void hetero_await_cold(
    const unsigned* __restrict__ counter,
    unsigned layer,
    unsigned layers,
    const unsigned* resp_seq,
    unsigned* error,
    unsigned max_polls) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const unsigned want = hs_seq(counter[0], layer, layers);
    // Relaxed polling, one acquire on the match: a system-scope acquire per poll
    // invalidates the device caches every few hundred cycles, which stalls the
    // other streams' kernels for the whole wait.
    for (unsigned n = 0; n < max_polls; ++n) {
        if (__hip_atomic_load(const_cast<unsigned*>(resp_seq), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == want) {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            return;
        }
        __builtin_amdgcn_s_sleep(8);
    }
    hs_store(error, want);
}

// One block per routed slot. Copy the cold slots' rows from the mailbox into
// the expanded down buffer; hot slots keep the primary's own rows. Skipped
// entirely when the slot's wait timed out.
extern "C" __global__ void hetero_install_cold(
    const unsigned* error,
    const float* mb_cold_out,
    const int* __restrict__ ids,
    const unsigned char* __restrict__ cold_bitmap,
    int n_experts,
    int dim,
    float* __restrict__ down_expanded) {
    const int s = blockIdx.x;
    if (!hs_cold(ids[s], cold_bitmap, n_experts)) return;
    const float* src = mb_cold_out + (long long)s * dim;
    float* dst = down_expanded + (long long)s * dim;
    // The rows live in the expert device's memory, across the link: issue every
    // load (and the error word's) before the first store, so the block pays one
    // link round trip instead of one per loop iteration. The await's acquire
    // already ordered these reads after the response.
    constexpr int kVec = 8;
    const int n4 = dim / 4;
    const bool vec = (dim & 3) == 0 && n4 <= kVec * (int)blockDim.x;
    if (vec) {
        const float4* src4 = reinterpret_cast<const float4*>(src);
        float4 v[kVec];
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
            const int i = threadIdx.x + k * blockDim.x;
            if (i < n4) v[k] = src4[i];
        }
        if (__hip_atomic_load(const_cast<unsigned*>(error), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0u) return;
        float4* dst4 = reinterpret_cast<float4*>(dst);
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
            const int i = threadIdx.x + k * blockDim.x;
            if (i < n4) dst4[i] = v[k];
        }
        return;
    }
    if (hs_load(error) != 0u) return;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

// Prefill (grid-stride, any grid). Convert a chunk's normed rows to F16 in the
// primary's local staging (a copy engine moves them across the link), write
// the route straight into the prefill slot (small), and the primary's masked
// ids (cold -> -1). Requests are released per sub-chunk by
// hetero_release_rows after the staging copy, never by this kernel.
extern "C" __global__ void hetero_publish_rows(
    const float* __restrict__ x_norm,
    const int* __restrict__ ids,
    const float* __restrict__ weights,
    const unsigned char* __restrict__ cold_bitmap,
    int n_experts,
    int rows,
    int top_k,
    int dim,
    _Float16* __restrict__ x_f16,
    int* mb_ids,
    float* mb_w,
    int* masked_ids) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (long long i = tid; i < (long long)rows * dim; i += stride) x_f16[i] = (_Float16)x_norm[i];
    for (long long s = tid; s < (long long)rows * top_k; s += stride) {
        const int e = ids[s];
        mb_ids[s] = e;
        mb_w[s] = weights[s];
        masked_ids[s] = hs_cold(e, cold_bitmap, n_experts) ? -1 : e;
    }
}

// Prefill, one thread: record one sub-chunk's layer, first row, row count and
// row format on its meta line, then release its request word. Queued after the copy of
// the sub-chunk's rows (and after hetero_publish_rows) on the same stream.
extern "C" __global__ void hetero_release_rows(
    const unsigned* __restrict__ counter,
    unsigned layer,
    unsigned layers,
    int row_off,
    int rows,
    unsigned format,
    unsigned* meta,
    unsigned* req_seq) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    hs_store(meta, layer);
    hs_store(meta + 1, (unsigned)row_off);
    hs_store(meta + 2, (unsigned)rows);
    hs_store(meta + 3, format);
    __threadfence_system();
    hs_store(req_seq, hs_seq(counter[0], layer, layers));
}

// Prefill (grid-stride): x_batch += the expert device's BF16 cold partial (in
// the primary's local memory after the copy back), skipped entirely when any
// sub-chunk's wait timed out (the shared error word).
extern "C" __global__ void hetero_add_partial(
    const unsigned* error,
    const unsigned short* __restrict__ partial,
    int n,
    float* __restrict__ x_batch) {
    // One load of the error word per block (a system-scope load per thread
    // costs hundreds of microseconds over a whole chunk).
    __shared__ unsigned skip;
    if (threadIdx.x == 0) skip = hs_load(error);
    __syncthreads();
    if (skip != 0u) return;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        x_batch[i] += __uint_as_float((unsigned)partial[i] << 16);
    }
}

// Expert device (grid-stride): F16 rows from the prefill slot -> F32 scratch.
extern "C" __global__ void hetero_f16_to_f32(const _Float16* src, float* __restrict__ dst, int n) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = (float)src[i];
}

// Expert device (grid-stride): F32 partial -> BF16 in the prefill slot,
// round-to-nearest-even (exact for the BF16-exact partial; NaN stays NaN).
extern "C" __global__ void hetero_f32_to_bf16(const float* __restrict__ src, unsigned short* dst, int n) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const unsigned u = __float_as_uint(src[i]);
        unsigned short b;
        if ((u & 0x7fffffffu) > 0x7f800000u) {
            b = (unsigned short)((u >> 16) | 0x40u);
        } else {
            b = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        }
        dst[i] = b;
    }
}

// Expert device, one block: masked ids for the cold kernels (hot -> -1).
extern "C" __global__ void hetero_mask_hot(
    const int* mb_ids,
    const unsigned char* __restrict__ cold_bitmap,
    int n_experts,
    int slots,
    int* __restrict__ masked_ids) {
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        const int e = mb_ids[s];
        masked_ids[s] = hs_cold(e, cold_bitmap, n_experts) ? e : -1;
    }
}

// Expert device, one thread, last in a layer's graph: answer with the request's
// sequence after the cold rows are visible system-wide.
extern "C" __global__ void hetero_store_seq(const unsigned* req_seq, unsigned* resp_seq) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    __threadfence_system();
    hs_store(resp_seq, hs_load(req_seq));
}
