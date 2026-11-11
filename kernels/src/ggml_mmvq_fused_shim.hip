// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Derived from ggml (MIT, Copyright (c) 2023-2026 The ggml authors): this file
// #includes the pinned upstream headers under third_party/ggml/ (see
// third_party/ggml/PIN, LICENSE) and loops upstream's `vec_dot_*_q8_1` device
// functions; the kernel bodies below are transcriptions of upstream
// `mul_mat_vec_q` (mmvq.cu) and `quantize_q8_1` (quantize.cu), see "Provenance".

// ggml_mmvq_fused_shim: the S3 additions to the GGUF-native decode GEMV set for
// gfx1201 (RDNA4). Companion of ggml_mmvq_shim.hip (S1, not included, not
// edited); same numeric contract `ggml.a8q81.<type>`, same launch geometry,
// same kernarg layout, same reduction order. ncols_dst = 1.
//
// Exports (all wave32, NO atomics, NO dynamic LDS):
//
//   ggml_mmvq_q5_k_gfx1201       y[row] = dot(W[row,:], q8_1(x)), ggml Q5_K
//   ggml_mmvq_{iq4_xs,q4_k,q5_k,q6_k,q8_0}_res_gfx1201
//                                dst[row] = dst[row] + dot(W[row,:], q8_1(x))
//     kernargs (offsets in bytes), identical for all six:
//       vx    @0   const void*    weights, row-major, `K/qk` blocks per row
//                                 (block_q5_K 176 B; iq4_xs 136 B, q4_k 144 B,
//                                  q6_k 210 B, q8_0 34 B)
//       vy    @8   const block_q8_1*  activations from ggml_q8_1_quantize_gfx1201
//                                 (or the gather producer below), K/32 blocks
//       dst   @16  float*         one f32 per weight row; the `_res` exports READ it
//                                 (the residual stream) and write the sum
//       K     @24  int32          reduction length, multiple of qk (256; q8_0: 32)
//     launch: block (32,8,1) = (warp_size, nwarps), grid (M,1,1) (rows_per_block 1,
//     one row per workgroup), static LDS 7*32*4 = 896 B, no atomics.
//     Domain: K % qk == 0, M >= 1. M is the grid; it is not a kernarg.
//     Alignment of vx: q5_k reads 4-byte words inside its 176-B blocks
//     (`ql`/`qh` int loads) so vx needs 4-byte alignment; the other alignments are
//     those of ggml_mmvq_shim.hip. vy and dst: 4.
//     `_res` epilogue: after the (unchanged) reduction lane 0 of warp 0 stores
//     `dst[row] = dst[row] + tmp`, one f32 add of the previous value, no other
//     change (no atomics: each row is owned by exactly one workgroup, so the
//     read-modify-write has no concurrent writer).
//
//   ggml_q8_1_quantize_gather_gfx1201   single-row q8_1 producer with a block permutation
//     kernargs (offsets in bytes):
//       x       @0   const float*       activations, `ne0` floats
//       vy      @8   block_q8_1*        output blocks, ne0/32 of them
//       blk_map @16  const int32*       ne0/32 entries; output block j (32 elements) is
//                                       quantised from x[blk_map[j]*32 .. +32]
//       ne0     @24  int32              K, multiple of 32
//     launch: block (256,1,1), grid (ceil(ne0/256),1,1), no LDS.
//     Domain: ne0 % 32 == 0, ne0 > 0, every blk_map entry in [0, ne0/32).
//     With blk_map[j] = j the output is byte-for-byte that of
//     ggml_q8_1_quantize_gfx1201 (ne00 = s01 = ne0, one row). Every thread of a
//     32-lane group takes the same early-return (ne0 % 32 == 0), so the
//     `warp_reduce_*<32>` butterflies never see a partial warp.
//
// Reduction order (fixed, deterministic): each thread folds its blocks in
// ascending `kbx += blocks_per_iter` order; warps 1..7 write LDS, warp 0 adds
// them in l = 0..6 order, then a `__shfl_xor` butterfly (warp_reduce_sum<32>)
// and lane 0 stores. Upstream mmvq.cu:774-827 verbatim in structure.
//
// Parameters (gfx1201 = RDNA4, upstream MMVQ_PARAMETERS_RDNA4, ncols_dst = 1):
// nwarps = 8, rows_per_block = 1, small_k off (S0 census 3.6: nwarps 8 for
// Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS; the per-type vdr / vec_dot constants are
// upstream's, see the namespace comment).
//
// Provenance (pin 2e0d86bebc210947680e8f0d67daa79f3260c106, see third_party/ggml/PIN):
//  * mul_mat_vec_q body: mmvq.cu `mul_mat_vec_q<type, ncols_dst=1, ...>` with
//    has_fusion = false, ids = nullptr, one channel / one sample, stride_row_x
//    = K/qk. The structure and the reduction order are unchanged; the template
//    bool ACCUMULATE only selects the final store.
//  * quantize body: quantize.cu `quantize_q8_1` with ne1 = 1, ne2 = ne3 = 1; the
//    block index of the READ is remapped through blk_map (the write index is not).
//
// Build (see scripts/build-ggml-mmvq-fused-shim.sh): hipfire's pinned core flags
// (`--genco --offload-arch=gfx1201 -O3 --no-offload-compress -fuse-cuid=none
// -mcode-object-version=6`, rdna-compute/src/compiler.rs core_hipcc_flags)
// plus `-ffp-contract=fast-honor-pragmas -DGGML_USE_HIP -Ithird_party/ggml/include
// -Ithird_party/ggml/src -Ithird_party/ggml/src/ggml-cuda`. No fast-math. FP
// contraction policy: as ggml_mmvq_shim.hip (the clang HIP default).

#include "ggml-cuda/vecdotq.cuh"

static_assert(ggml_cuda_get_physical_warp_size() == 32, "gfx1201 is wave32");

namespace ggml_mmvq_shim {

// gfx1201 parameter set, upstream MMVQ_PARAMETERS_RDNA4 at ncols_dst = 1
// (mmvq.cu `calc_nwarps` / `calc_rows_per_block` at the pin; S0-CENSUS.md 3.6):
// nwarps 8 for Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, rows_per_block 1, small_k off.
// mmvq.cu itself is NOT included: its host launchers instantiate every
// upstream kernel template into the object. The per-type constants below are
// the upstream tables (`get_vdr_mmvq`, `get_vec_dot_q_cuda`) for the five
// exported types, pinned by the drift test over third_party/ggml/PIN.
constexpr int NWARPS = 8;
constexpr int WARP   = 32;

typedef float (*vec_dot_fn_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

template <ggml_type type> struct mmvq_type;
template <> struct mmvq_type<GGML_TYPE_IQ4_XS> {
    static constexpr int vdr = VDR_IQ4_XS_Q8_1_MMVQ;
    static constexpr vec_dot_fn_t vec_dot = vec_dot_iq4_xs_q8_1;
};
template <> struct mmvq_type<GGML_TYPE_Q4_K> {
    static constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    static constexpr vec_dot_fn_t vec_dot = vec_dot_q4_K_q8_1;
};
template <> struct mmvq_type<GGML_TYPE_Q5_K> {
    static constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ;
    static constexpr vec_dot_fn_t vec_dot = vec_dot_q5_K_q8_1;
};
template <> struct mmvq_type<GGML_TYPE_Q6_K> {
    static constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    static constexpr vec_dot_fn_t vec_dot = vec_dot_q6_K_q8_1;
};
template <> struct mmvq_type<GGML_TYPE_Q8_0> {
    static constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    static constexpr vec_dot_fn_t vec_dot = vec_dot_q8_0_q8_1;
};

template <ggml_type type, bool ACCUMULATE>
static __device__ __forceinline__ void mmvq_ncols1(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int ncols_x) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = mmvq_type<type>::vdr;
    constexpr vec_dot_fn_t vec_dot_q_cuda = mmvq_type<type>::vec_dot;

    const int tid            = WARP*threadIdx.y + threadIdx.x;
    const int row0           = blockIdx.x;
    const int blocks_per_row = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * NWARPS*WARP / qi;

    float tmp = 0.0f;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);               // y block index aligned with kbx
        const int kqs = vdr * (tid % (qi/vdr));         // quant index inside the block
        tmp += vec_dot_q_cuda(vx, &y[kby], row0*blocks_per_row + kbx, kqs);
    }

    __shared__ float tmp_shared[NWARPS-1][WARP];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y-1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int l = 0; l < NWARPS-1; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
    }
    tmp = warp_reduce_sum<WARP>(tmp);
    if (threadIdx.x == 0) {
        if (ACCUMULATE) {
            dst[row0] = dst[row0] + tmp;
        } else {
            dst[row0] = tmp;
        }
    }
}

} // namespace ggml_mmvq_shim

extern "C" {

// Evidence builds only: -DGGML_SHIM_ONLY=<0..6> compiles one export (0 gather producer,
// 1 q5_k, 2 iq4_xs_res, 3 q4_k_res, 4 q5_k_res, 5 q6_k_res, 6 q8_0_res) so per-export
// lift/emu verdicts can be taken. The shipped object defines none of it and contains
// all seven exports.
#if !defined(GGML_SHIM_ONLY) || GGML_SHIM_ONLY == 0
__global__ __launch_bounds__(256, 1) void ggml_q8_1_quantize_gather_gfx1201(
        const float * x, void * vy, const int * blk_map, const int ne0) {
    const int i0 = blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }
    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib  = i0 / QK8_1;   // output block index
    const int iqs = i0 % QK8_1;   // quant index

    const float xi = x[blk_map[ib]*QK8_1 + iqs];
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;
    if (iqs > 0) {
        return;
    }
    y[ib].ds = make_half2(d, sum);
}
#endif

#if !defined(GGML_SHIM_ONLY) || GGML_SHIM_ONLY == 1
__global__ __launch_bounds__(256, 1) void ggml_mmvq_q5_k_gfx1201(
        const void * vx, const void * vy, float * dst, const int K) {
    ggml_mmvq_shim::mmvq_ncols1<GGML_TYPE_Q5_K, false>(vx, (const block_q8_1 *) vy, dst, K);
}
#endif

#if !defined(GGML_SHIM_ONLY) || GGML_SHIM_ONLY == 2
__global__ __launch_bounds__(256, 1) void ggml_mmvq_iq4_xs_res_gfx1201(
        const void * vx, const void * vy, float * dst, const int K) {
    ggml_mmvq_shim::mmvq_ncols1<GGML_TYPE_IQ4_XS, true>(vx, (const block_q8_1 *) vy, dst, K);
}
#endif

#if !defined(GGML_SHIM_ONLY) || GGML_SHIM_ONLY == 3
__global__ __launch_bounds__(256, 1) void ggml_mmvq_q4_k_res_gfx1201(
        const void * vx, const void * vy, float * dst, const int K) {
    ggml_mmvq_shim::mmvq_ncols1<GGML_TYPE_Q4_K, true>(vx, (const block_q8_1 *) vy, dst, K);
}
#endif

#if !defined(GGML_SHIM_ONLY) || GGML_SHIM_ONLY == 4
__global__ __launch_bounds__(256, 1) void ggml_mmvq_q5_k_res_gfx1201(
        const void * vx, const void * vy, float * dst, const int K) {
    ggml_mmvq_shim::mmvq_ncols1<GGML_TYPE_Q5_K, true>(vx, (const block_q8_1 *) vy, dst, K);
}
#endif

#if !defined(GGML_SHIM_ONLY) || GGML_SHIM_ONLY == 5
__global__ __launch_bounds__(256, 1) void ggml_mmvq_q6_k_res_gfx1201(
        const void * vx, const void * vy, float * dst, const int K) {
    ggml_mmvq_shim::mmvq_ncols1<GGML_TYPE_Q6_K, true>(vx, (const block_q8_1 *) vy, dst, K);
}
#endif

#if !defined(GGML_SHIM_ONLY) || GGML_SHIM_ONLY == 6
__global__ __launch_bounds__(256, 1) void ggml_mmvq_q8_0_res_gfx1201(
        const void * vx, const void * vy, float * dst, const int K) {
    ggml_mmvq_shim::mmvq_ncols1<GGML_TYPE_Q8_0, true>(vx, (const block_q8_1 *) vy, dst, K);
}
#endif

} // extern "C"
