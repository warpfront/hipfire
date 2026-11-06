// SPDX-License-Identifier: Apache-2.0
// hipfire: top-10 indexed down projection with expanded output, K = 640, wave32 gfx12 (gfx1201).
// Byte-identical drop-in for gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_k640_r4 (shipped
// MQ4G128V2, 68 B/group) and gemv_mq4g128v2p_moe_down_top10_indexed_batched_expanded_k640_r4 (F1 drop-zero
// planar MQ4G128V2P). Same ABI, same zero-write for expert < 0, same K != 640 early return.
//
// Launch: grid [M/32, k_top, N], block 128 (4 waves, no LDS, no barriers). Each wave owns 8 adjacent output
// rows of one (token, rank) pick and one expert. Launcher admission: K == 640, M % 32 == 0, wave32.
//   shipped: expert tensor = M rows x (5 groups x [f16 scale, f16 zero, 64 nibble bytes]) = M*340 B
//   planar : expert tensor = [codes M*320 B][scales M*5 f16], zp = -8*scale (f32 exact)
//
// Arithmetic (identical to production per lane/element): w = fma(scale, n, zero) single rounding,
//   t = fma(w3,x3, fma(w2,x2, fma(w0,x0, w1*x1))); acc = acc + t (acc starts +0.0, groups 0..4 in order);
// then the lane-0 shfl_down tree 16,8,4,2,1. Two implementation tricks keep bits equal:
//  * the nibble bits are used directly as an f16 denormal (n * 2^-24) by v_fma_mix_f32 with the f32 scale
//    pre-multiplied by 2^24 (n1/n3 sit at bits 7:4 so they use scale*2^20): same real product, one rounding;
//  * the 16-lane tree is evaluated for 8 rows at once with permlanex16/DPP, same pairing and add order
//    (float add is commutative); row rho ends in lane 4*rho.
// gfx12 only (v_fma_mix_f32 with f16 denormal inputs, SALU float ops, s_wait_alu).
// gfx12 SGPR hazard (the reason for every s_wait_alu below): an SGPR that a VALU has read is not interlocked against a later SALU
// write: any following read (SALU or VALU) needs s_wait_alu depctr_sa_sdst(0) first (LLVM AMDGPUWaitSGPRHazards). The compiler pass
// cannot see the inline-asm VALU/SALU here, so it treats those SGPRs as untracked and emits NO wait for compiler-generated SALU
// writes (s_cvt_f32_f16 feeding the asm s_mul_f32, the permlane lane-select s_mov, the store-address SALU math) -> rare stale
// reads (wrong words, nondeterministic). Rule kept below: no compiler SALU write may feed a read after the first asm VALU read
// without an explicit wait; the store base is computed before any asm VALU and the permlane selector is made by asm + wait.
// HIPFIRE_COMPILER_FLAGS: -mllvm -amdgpu-sched-strategy=max-ilp

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef short dni_v2s __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ __amdgpu_buffer_rsrc_t dni_rsrc(const void* p, int nbytes) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)p, 0, nbytes, 0x31004000);
}
static __device__ __forceinline__ float dni_h2f(unsigned h) { return __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); }
static __device__ __forceinline__ float dni_smul(float a, float b) { float r; asm("s_mul_f32 %0, %1, %2" : "=s"(r) : "s"(a), "s"(b)); return r; }
static __device__ __forceinline__ float dni_mix_lo(unsigned q, float s, float z) {
    float r; asm("v_fma_mix_f32 %0, %1, %2, %3 op_sel_hi:[1,0,0]" : "=v"(r) : "v"(q), "s"(s), "s"(z)); return r;
}
static __device__ __forceinline__ float dni_mix_hi(unsigned q, float s, float z) {
    float r; asm("v_fma_mix_f32 %0, %1, %2, %3 op_sel:[1,0,0] op_sel_hi:[1,0,0]" : "=v"(r) : "v"(q), "s"(s), "s"(z)); return r;
}
static __device__ __forceinline__ float dni_ror8(float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x128, 0xf, 0xf, true)); }
static __device__ __forceinline__ float dni_qp2301(float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xf, 0xf, true)); }
static __device__ __forceinline__ float dni_qp1032(float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xf, 0xf, true)); }
static __device__ __forceinline__ float dni_dpp8_x4(float v) { return __int_as_float(__builtin_amdgcn_mov_dpp8(__float_as_int(v), 6852524)); }
static __device__ __forceinline__ unsigned dni_perm_sel() {
    unsigned s;
    asm volatile("s_mov_b32 %0, 0x76543210\n\ts_wait_alu depctr_sa_sdst(0)" : "=s"(s));
    return s;
}
static __device__ __forceinline__ float dni_xrow16(float o, unsigned sel) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(o), __float_as_int(o), sel, 0xfedcba98u, false, false));
}

// a[r] = per-lane partial of row r (r = 0..7). Returns, in lane 4*rho, the shfl_down(16,8,4,2,1) lane-0 tree of row rho.
static __device__ __forceinline__ float dni_butterfly8(const float (&a)[8], int tid) {
    const bool lo16 = tid < 16;
    const unsigned sel = dni_perm_sel();
    float v[4];
    const int pa[4] = {0, 2, 1, 3};
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float x = a[pa[p]], y = a[pa[p] + 4];
        float vv = lo16 ? x : y;
        const float o = lo16 ? y : x;
        vv += dni_xrow16(o, sel);
        v[p] = vv;
    }
    const bool lo8 = (tid & 15) < 8;
    float W[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const float t1 = lo8 ? v[2 * p] : v[2 * p + 1];
        const float t2 = lo8 ? v[2 * p + 1] : v[2 * p];
        W[p] = t1 + dni_ror8(t2);
    }
    const bool lo4 = (tid & 7) < 4;
    const float t1 = lo4 ? W[0] : W[1];
    const float t2 = lo4 ? W[1] : W[0];
    float z = t1 + dni_dpp8_x4(t2);
    z = z + dni_qp2301(z);
    z = z + dni_qp1032(z);
    return z;
}

template <bool PLANAR>
static __device__ __forceinline__ void dni_down_k640(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch, float* __restrict__ expert_outputs, int M, int K, int N, int n_exp, int k_top) {
    constexpr int G = 5, NC = 8 * G;  // 5 groups, 40 (row, group) chunks per wave
    const int tid = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int row0 = (blockIdx.x * 4 + wave) * 8;
    const int rank = blockIdx.y;
    const int token = blockIdx.z;
    if (row0 >= M || k_top <= 0 || rank >= k_top || token >= N || K != G * 128) return;
    const int flat = token * k_top + rank;
    // output base: SALU math done before any asm VALU runs (see the hazard note in the header), kept live in an SGPR pair
    __attribute__((address_space(1))) float* obase = (__attribute__((address_space(1))) float*)(expert_outputs + (long long)flat * M + row0);
    asm volatile("" : "+s"(obase));
    const int expert = topk_indices[flat];
    float res = 0.0f;
    if (expert >= 0 && expert < n_exp) {
        const unsigned long long aptr = expert_ptrs[expert];
        const __amdgpu_buffer_rsrc_t wr = dni_rsrc((const void*)aptr, PLANAR ? M * 330 : M * 340);
        const __amdgpu_buffer_rsrc_t xr = dni_rsrc((const void*)rot_batch, N * k_top * 2560);
        // per-chunk headers (uniform): gathered one chunk per lane, then moved to SGPRs
        unsigned H[NC];
        {
            unsigned hv[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                int ci = l * 32 + tid;
                if (ci > NC - 1) ci = NC - 1;
                if constexpr (PLANAR) hv[l] = (unsigned)__builtin_amdgcn_raw_buffer_load_b16(wr, ci * 2, M * 320 + row0 * 10, 0);
                else hv[l] = __builtin_amdgcn_raw_buffer_load_b32(wr, ci * 68, row0 * 340, 0);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) H[c] = (unsigned)__builtin_amdgcn_readlane((int)hv[c / 32], c % 32);
        }
        // payload: lane tid owns bytes 2*tid, 2*tid+1 of each 64 B group payload
        unsigned short P[NC];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int c = r * G + g;
                if constexpr (PLANAR) P[c] = __builtin_amdgcn_raw_buffer_load_b16(wr, tid * 2 + c * 64, row0 * 320, 0);
                else P[c] = __builtin_amdgcn_raw_buffer_load_b16(wr, tid * 2 + 4 + c * 68, row0 * 340, 0);
            }
        __builtin_amdgcn_sched_barrier(0);
        float4 xv[G];
#pragma unroll
        for (int g = 0; g < G; ++g)
            xv[g] = __builtin_bit_cast(float4, __builtin_amdgcn_raw_buffer_load_b128(xr, tid * 16, flat * 2560 + g * 512, 0));
        __builtin_amdgcn_sched_barrier(0);

        float acc[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[r] = 0.0f;
#pragma unroll
        for (int rp = 0; rp < 4; ++rp) {
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int c0 = (2 * rp) * G + g, c1 = (2 * rp + 1) * G + g;
                const dni_v2s pv = dni_v2s{(short)P[c0], (short)P[c1]};
                const unsigned pp = __builtin_bit_cast(unsigned, pv);
                const unsigned q0 = pp & 0x000F000Fu;
                const unsigned q1 = pp & 0x00F000F0u;
                const unsigned p8 = pp >> 8;
                const unsigned q2 = p8 & 0x000F000Fu;
                const unsigned q3 = p8 & 0x00F000F0u;
                float sA = dni_h2f(H[c0]), sB = dni_h2f(H[c1]);
                // gfx12: the compiler's SALU-float s_cvt_f32_f16 feeding the opaque inline-asm
                // s_mul_f32 below gets no wait (the asm consumer is invisible to the hazard
                // recognizer); without this, rare nondeterministic wrong words (K2W det5 receipt).
                asm volatile("s_wait_alu depctr_sa_sdst(0)" : "+s"(sA), "+s"(sB));
                float zA, zB;
                if constexpr (PLANAR) { zA = dni_smul(sA, -8.0f); zB = dni_smul(sB, -8.0f); }
                else { zA = dni_h2f(H[c0] >> 16); zB = dni_h2f(H[c1] >> 16); }
                float a24 = dni_smul(sA, 16777216.0f), a20 = dni_smul(sA, 1048576.0f);
                float b24 = dni_smul(sB, 16777216.0f), b20 = dni_smul(sB, 1048576.0f);
                // gfx12: SALU-written SGPRs read by inline-asm VALU need an explicit wait (not inferable through asm)
                asm volatile("s_wait_alu depctr_sa_sdst(0)" : "+s"(a24), "+s"(a20), "+s"(zA), "+s"(b24), "+s"(b20), "+s"(zB));
                const float wa0 = dni_mix_lo(q0, a24, zA), wb0 = dni_mix_hi(q0, b24, zB);
                const float wa1 = dni_mix_lo(q1, a20, zA), wb1 = dni_mix_hi(q1, b20, zB);
                const float wa2 = dni_mix_lo(q2, a24, zA), wb2 = dni_mix_hi(q2, b24, zB);
                const float wa3 = dni_mix_lo(q3, a20, zA), wb3 = dni_mix_hi(q3, b20, zB);
                const float4 x = xv[g];
                const float ta = __builtin_fmaf(wa3, x.w, __builtin_fmaf(wa2, x.z, __builtin_fmaf(wa0, x.x, wa1 * x.y)));
                const float tb = __builtin_fmaf(wb3, x.w, __builtin_fmaf(wb2, x.z, __builtin_fmaf(wb0, x.x, wb1 * x.y)));
                acc[2 * rp] += ta;
                acc[2 * rp + 1] += tb;
                asm volatile("" : "+v"(acc[2 * rp]), "+v"(acc[2 * rp + 1]));
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        res = dni_butterfly8(acc, tid);
    }
    if ((tid & 3) == 0) obase[tid >> 2] = res;
}

__launch_bounds__(128)
extern "C" __global__ void gemv_mq4g128v2_moe_down_top10_dni_k640(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch, float* __restrict__ expert_outputs, int M, int K, int N, int n_exp, int k_top) {
    dni_down_k640<false>(expert_ptrs, topk_indices, rot_batch, expert_outputs, M, K, N, n_exp, k_top);
}

__launch_bounds__(128)
extern "C" __global__ void gemv_mq4g128v2p_moe_down_top10_dni_k640(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch, float* __restrict__ expert_outputs, int M, int K, int N, int n_exp, int k_top) {
    dni_down_k640<true>(expert_ptrs, topk_indices, rot_batch, expert_outputs, M, K, N, n_exp, k_top);
}
