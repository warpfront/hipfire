// hipfire — device-side hetero expert split (Qwen4 Flash-Next, design v3).
//
// The primary (gfx1201) and the expert device (gfx1151) exchange one layer's
// cold routed slots through a pinned mailbox in the expert device's RAM
// (crates/hipfire-dispatch/src/hetero_mailbox.rs owns the layout). Every
// cross-device word uses system-scope atomic loads/stores only (no RMW: the
// USB4 link has no PCIe atomics), and every wait is bounded.
//
// Primary:  hetero_token_advance, hetero_publish_cold, hetero_await_cold,
//           hetero_install_cold.
// Expert:   hetero_mask_hot, hetero_store_seq.

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

// One block. Copy the layer's route into its mailbox slot, write the primary's
// masked ids (cold -> -1), then release the request sequence.
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
    unsigned* req_seq) {
    const int slots = rows * top_k;
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
    for (unsigned n = 0; n < max_polls; ++n) {
        if (hs_load(resp_seq) == want) return;
        __builtin_amdgcn_s_sleep(1);
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
    if (hs_load(error) != 0u) return;
    const int s = blockIdx.x;
    if (!hs_cold(ids[s], cold_bitmap, n_experts)) return;
    const float* src = mb_cold_out + (long long)s * dim;
    float* dst = down_expanded + (long long)s * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
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
