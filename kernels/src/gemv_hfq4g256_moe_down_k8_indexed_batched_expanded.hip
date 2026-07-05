#include <hip/hip_runtime.h>

// Atomic-free variant of `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched`.
//
// Same per-(token, krank) GEMV compute. Instead of `atomicAdd(scale * acc)`
// into a shared `x_residual[N × M]` row (K_TOP-way contention per output
// cell, observed ~387 GiB/s on R9700/gfx1201 vs the sister gate_up at
// 954 GiB/s with the same MQ4 layout but no atomic), this kernel writes
// each (token, krank) result to its own row in an expanded buffer
// `expert_outputs[N × K_TOP × M]`. A separate `moe_down_combine_k8_batched`
// kernel folds K_TOP outputs back into the residual stream with topk_weights
// applied — see kernels/src/moe_down_combine_k8_batched.hip.
//
// Shapes (vs the residual_scaled variant):
//   expert_ptrs    : [n_exp]                — unchanged
//   topk_indices   : [N × K_TOP]            — unchanged
//   rot_batch      : [N × K_TOP × K]        — unchanged
//   expert_outputs : [N × K_TOP × M]        — replaces x_residual; non-atomic
// Note: `topk_weights` is NOT consumed here; the combine kernel applies the
// scale during reduction over K_TOP. Dropping the weights load also saves
// a small amount of bandwidth.
//
// Grid: (M, K_TOP, N). Block = 32 threads. No atomic ops — each output
// address is written by exactly one workgroup, so the writes are pure
// stores.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
    int M, int K, int K_TOP
) {
    // R7c1_down_r4x: map one active block to four adjacent output rows.
    // The launch grid is still M blocks in x; blocks with row0 >= M exit.
    // This preserves the ABI while reusing the same x[base..base+7] values
    // across four rows and testing whether the R2/R3 row-reuse ladder has
    // more headroom before VGPR pressure overtakes the memory-traffic win.
    const int row0 = blockIdx.x * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;
    const bool have_row1 = row1 < M;
    const bool have_row2 = row2 < M;
    const bool have_row3 = row3 < M;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const char* row_ptr0 = A + (long long)row0 * groups_per_row * 136;
    const char* row_ptr1 = A + (long long)row1 * groups_per_row * 136;
    const char* row_ptr2 = A + (long long)row2 * groups_per_row * 136;
    const char* row_ptr3 = A + (long long)row3 * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    float ccc0 = 0.0f, ccc1 = 0.0f, ccc2 = 0.0f, ccc3 = 0.0f;
    float dcc0 = 0.0f, dcc1 = 0.0f, dcc2 = 0.0f, dcc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8(b) \
        const float x0 = x[(b) + 0]; \
        const float x1 = x[(b) + 1]; \
        const float x2 = x[(b) + 2]; \
        const float x3 = x[(b) + 3]; \
        const float x4 = x[(b) + 4]; \
        const float x5 = x[(b) + 5]; \
        const float x6 = x[(b) + 6]; \
        const float x7 = x[(b) + 7]; \
        const float xs = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))

    #define DOG_X8(gp, a) do { \
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        float zp = __builtin_bit_cast(float, *(const unsigned int*)((gp) + 4)); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        const float dot = (float)((pk) & 0xFu) * x0 \
             + (float)(((pk) >> 4) & 0xFu) * x1 \
             + (float)(((pk) >> 8) & 0xFu) * x2 \
             + (float)(((pk) >> 12) & 0xFu) * x3 \
             + (float)(((pk) >> 16) & 0xFu) * x4 \
             + (float)(((pk) >> 20) & 0xFu) * x5 \
             + (float)(((pk) >> 24) & 0xFu) * x6 \
             + (float)(((pk) >> 28) & 0xFu) * x7; \
        (a) += sc * dot + zp * xs; \
    } while (0)

    #define DOG_X8_STAGED(sc, zp, pk, b, a) do { \
        const float x0 = x[(b) + 0]; \
        const float x1 = x[(b) + 1]; \
        const float x2 = x[(b) + 2]; \
        const float x3 = x[(b) + 3]; \
        const float x4 = x[(b) + 4]; \
        const float x5 = x[(b) + 5]; \
        const float x6 = x[(b) + 6]; \
        const float x7 = x[(b) + 7]; \
        const float xs = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)); \
        const float dot = (float)(((pk)) & 0xFu) * x0 \
             + (float)((((pk)) >> 4) & 0xFu) * x1 \
             + (float)((((pk)) >> 8) & 0xFu) * x2 \
             + (float)((((pk)) >> 12) & 0xFu) * x3 \
             + (float)((((pk)) >> 16) & 0xFu) * x4 \
             + (float)((((pk)) >> 20) & 0xFu) * x5 \
             + (float)((((pk)) >> 24) & 0xFu) * x6 \
             + (float)((((pk)) >> 28) & 0xFu) * x7; \
        (a) += (sc) * dot + (zp) * xs; \
    } while (0)

    #define DOG_X8_TAIL2(row_ptr, g0, g1, base0, base1, a0, a1) do { \
        const char* gp_tail0 = (row_ptr) + (g0) * 136; \
        const char* gp_tail1 = (row_ptr) + (g1) * 136; \
        const float sc_tail0 = __builtin_bit_cast(float, *(const unsigned int*)(gp_tail0)); \
        const float zp_tail0 = __builtin_bit_cast(float, *(const unsigned int*)(gp_tail0 + 4)); \
        const unsigned int pk_tail0 = *(const unsigned int*)(gp_tail0 + 8 + boff); \
        const float sc_tail1 = __builtin_bit_cast(float, *(const unsigned int*)(gp_tail1)); \
        const float zp_tail1 = __builtin_bit_cast(float, *(const unsigned int*)(gp_tail1 + 4)); \
        const unsigned int pk_tail1 = *(const unsigned int*)(gp_tail1 + 8 + boff); \
        DOG_X8_STAGED(sc_tail0, zp_tail0, pk_tail0, base0, a0); \
        DOG_X8_STAGED(sc_tail1, zp_tail1, pk_tail1, base1, a1); \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        {
            LOAD_X8(base0);
            const char* gp0 = row_ptr0 + g * 136;
            DOG_X8(gp0, acc0);
            if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc0);
            if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc0);
            if (have_row3) DOG_X8(row_ptr3 + g * 136, dcc0);
        }
        {
            LOAD_X8(base0 + 256);
            const char* gp0 = row_ptr0 + (g + 1) * 136;
            DOG_X8(gp0, acc1);
            if (have_row1) DOG_X8(row_ptr1 + (g + 1) * 136, bcc1);
            if (have_row2) DOG_X8(row_ptr2 + (g + 1) * 136, ccc1);
            if (have_row3) DOG_X8(row_ptr3 + (g + 1) * 136, dcc1);
        }
        {
            LOAD_X8(base0 + 512);
            const char* gp0 = row_ptr0 + (g + 2) * 136;
            DOG_X8(gp0, acc2);
            if (have_row1) DOG_X8(row_ptr1 + (g + 2) * 136, bcc2);
            if (have_row2) DOG_X8(row_ptr2 + (g + 2) * 136, ccc2);
            if (have_row3) DOG_X8(row_ptr3 + (g + 2) * 136, dcc2);
        }
        {
            LOAD_X8(base0 + 768);
            const char* gp0 = row_ptr0 + (g + 3) * 136;
            DOG_X8(gp0, acc3);
            if (have_row1) DOG_X8(row_ptr1 + (g + 3) * 136, bcc3);
            if (have_row2) DOG_X8(row_ptr2 + (g + 3) * 136, ccc3);
            if (have_row3) DOG_X8(row_ptr3 + (g + 3) * 136, dcc3);
        }
    }

    if (tail >= 2) {
        const int g0 = quads << 2;
        const int g1 = g0 + 1;
        const int base0 = g0 * 256 + tid * 8;
        const int base1 = g1 * 256 + tid * 8;
        DOG_X8_TAIL2(row_ptr0, g0, g1, base0, base1, acc0, acc1);
        if (have_row1) DOG_X8_TAIL2(row_ptr1, g0, g1, base0, base1, bcc0, bcc1);
        if (have_row2) DOG_X8_TAIL2(row_ptr2, g0, g1, base0, base1, ccc0, ccc1);
        if (have_row3) DOG_X8_TAIL2(row_ptr3, g0, g1, base0, base1, dcc0, dcc1);
    } else if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc0);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc0);
        if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc0);
        if (have_row3) DOG_X8(row_ptr3 + g * 136, dcc0);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc2);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc2);
        if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc2);
        if (have_row3) DOG_X8(row_ptr3 + g * 136, dcc2);
    }
    #undef DOG_X8_TAIL2
    #undef DOG_X8_STAGED
    #undef DOG_X8
    #undef LOAD_X8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    float ccc = (ccc0 + ccc1) + (ccc2 + ccc3);
    float dcc = (dcc0 + dcc1) + (dcc2 + dcc3);
    acc += __shfl_down(acc, 16);
    bcc += __shfl_down(bcc, 16);
    acc += __shfl_down(acc, 8);
    bcc += __shfl_down(bcc, 8);
    acc += __shfl_down(acc, 4);
    bcc += __shfl_down(bcc, 4);
    acc += __shfl_down(acc, 2);
    bcc += __shfl_down(bcc, 2);
    acc += __shfl_down(acc, 1);
    bcc += __shfl_down(bcc, 1);
    ccc += __shfl_down(ccc, 16);
    dcc += __shfl_down(dcc, 16);
    ccc += __shfl_down(ccc, 8);
    dcc += __shfl_down(dcc, 8);
    ccc += __shfl_down(ccc, 4);
    dcc += __shfl_down(dcc, 4);
    ccc += __shfl_down(ccc, 2);
    dcc += __shfl_down(dcc, 2);
    ccc += __shfl_down(ccc, 1);
    dcc += __shfl_down(dcc, 1);

    if (tid == 0) {
        // Direct store to per-(token, krank) row — no atomic contention.
        // Layout: expert_outputs[bid][krank][row].
        const long long out_base =
            ((long long)bid * K_TOP + krank) * (long long)M;
        expert_outputs[out_base + row0] = acc;
        if (have_row1) expert_outputs[out_base + row1] = bcc;
        if (have_row2) expert_outputs[out_base + row2] = ccc;
        if (have_row3) expert_outputs[out_base + row3] = dcc;
    }
}
