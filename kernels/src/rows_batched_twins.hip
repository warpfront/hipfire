// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Rows-batched twins of singleton decode kernels (continuous batching).
//
// The host (crates/rdna-compute/src/rows_batched.rs `rows_twin_source`)
// assembles each module as
//
//   #define HIPFIRE_ROWS_<X> 1
//   <singleton source, entry renamed `<kernel>_row_body` and its
//    `extern "C" [__launch_bounds__] __global__` qualifiers rewritten to
//    `__device__ __forceinline__`; nothing else changed except the checked
//    edits listed per twin below>
//   <this file>
//
// Each wrapper runs one request row per blockIdx.z, reading that row's
// pointers from a host-validated row table, and calls the singleton body with
// exactly the arguments a per-row singleton launch would pass. Grid x/y and
// block shape equal the singleton launch's. Same arithmetic by construction.
// Row tables are 8-byte pointers only; layouts mirror rows_batched.rs.

#ifdef HIPFIRE_ROWS_GDN
// Checked edit: the singleton entry's `lane = blockIdx.z` becomes 0 (the
// singleton decode launch has grid.z == 1, so it always sees lane 0).
struct HfGdnRow {
    const float* q;
    const float* k;
    const float* v;
    const float* gate;
    const float* beta;
    signed char* s_q8;
    float* s_scales;
    float* output;
    __half* s_ef_residual;   // nullptr = no error feedback (as singleton)
};

// Row r uses requant frame frame_base + r: the frame the r-th of a sequence
// of per-row singleton launches would have reserved (one per launch).
extern "C" __launch_bounds__(HIPFIRE_GDN_BLOCK_SIZE, HIPFIRE_GDN_MIN_BLOCKS)
__global__ void HIPFIRE_ROWS_GDN_ENTRY(
    const HfGdnRow* __restrict__ rows,
    int n_tokens,
    int n_heads,
    int head_dim,
    unsigned int frame_base
) {
    const HfGdnRow r = rows[blockIdx.z];
    HIPFIRE_GDN_KERNEL(r.q, r.k, r.v, r.gate, r.beta, r.s_q8, r.s_scales, r.output,
                       n_tokens, n_heads, head_dim, frame_base + (unsigned int)blockIdx.z,
                       r.s_ef_residual);
}
#endif

#ifdef HIPFIRE_ROWS_CQN
struct HfConvRow {
    float* q_out;
    float* k_out;
    float* v_out;
    const float* input;
    float* state;
};

extern "C" __launch_bounds__(HIPFIRE_CQN_BLOCK)
__global__ void HIPFIRE_ROWS_CQN_ENTRY(
    const HfConvRow* __restrict__ rows,
    const float* __restrict__ weight,
    int k_dim,
    int v_dim,
    int n_heads,
    int head_dim,
    float q_scale,
    float eps
) {
    const HfConvRow r = rows[blockIdx.z];
    HIPFIRE_CQN_KERNEL(r.q_out, r.k_out, r.v_out, r.input, weight, r.state,
                       k_dim, v_dim, n_heads, head_dim, q_scale, eps);
}
#endif

#ifdef HIPFIRE_ROWS_GATED_NORM
struct HfGatedNormRow {
    const float* x;
    const float* z;
    float* x_rot;
};

extern "C" __launch_bounds__(64, 1)
__global__ void HIPFIRE_ROWS_GATED_NORM_ENTRY(
    const HfGatedNormRow* __restrict__ rows,
    const float* __restrict__ weight,
#ifdef HIPFIRE_GATED_NORM_MQ_ROTATE_AWQ
    const float* __restrict__ awq_scale,
#endif
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int head_dim,
    float eps
) {
    const HfGatedNormRow r = rows[blockIdx.z];
    HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL(r.x, r.z, weight,
#ifdef HIPFIRE_GATED_NORM_MQ_ROTATE_AWQ
                                        awq_scale,
#endif
                                        signs1, signs2, r.x_rot, n_heads, head_dim, eps);
}
#endif

#ifdef HIPFIRE_ROWS_FA_PREP
struct HfFaPrepRow {
    const float* q_interleaved;
    float* q;
    float* gate;
    float* k;               // read and written in place (as singleton)
    const int* pos;         // this row's position word
};

extern "C" __launch_bounds__(256, 1)
__global__ void HIPFIRE_ROWS_FA_PREP_ENTRY(
    const HfFaPrepRow* __restrict__ rows,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    float eps,
    float freq_base
) {
    const HfFaPrepRow r = rows[blockIdx.z];
    HIPFIRE_QWEN35_FA_PREP_KERNEL(r.q_interleaved, r.q, r.gate, r.k, q_weight, k_weight,
                                  r.pos, eps, freq_base);
}
#endif
