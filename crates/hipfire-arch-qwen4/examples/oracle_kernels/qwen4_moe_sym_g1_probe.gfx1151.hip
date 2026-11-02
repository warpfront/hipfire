// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// ORACLE-ONLY diagnostic kernels for `examples/qwen4_moe_sym_real_g1.rs` (real-layer K2 G1).
// NEVER production, never dispatched by the runtime, never registered in a bundle or manifest.
//
// gfx1151 twin of `qwen4_moe_sym_g1_probe.gfx1201.hip`: same kernels, byte-identical body below the arch guard
// (the sealed gfx1201 source and its recorded sha256 stay untouched).
// Build (code object v6, the identity-harness recipe; contraction off so every float op below is
// exactly the one written):
//   hipcc --genco --offload-arch=gfx1151 -O3 --no-offload-compress -fuse-cuid=none \
//       -mcode-object-version=6 -ffp-contract=off \
//       -o /tmp/qwen4_moe_sym_g1_probe_gfx1151.co \
//       crates/hipfire-arch-qwen4/examples/oracle_kernels/qwen4_moe_sym_g1_probe.gfx1151.hip
//
// 1. Epilogue intrinsic probes. The PM K2 epilogue (`gemm_uk/epilogue.rs::silu_dense`) uses three
//    device results Rust cannot reproduce: `v_exp_f32`, `v_ldexp_f32` and the IEEE f32 divide
//    expansion (`v_div_scale_f32` x2, `v_rcp_f32`, fma refinement, `v_div_fmas_f32`,
//    `v_div_fixup_f32`). These kernels execute exactly those instructions on a list of inputs so
//    `mq_i4_reference::grouped_prefill_epilogue_with` can run over recorded DEVICE results.
//    The divide probe also records hipcc's `__fdiv_rn` and the harness requires it to equal the
//    explicit sequence (a transcription self-check).
// 2. MQ4 F32 accumulator oracle. The shipped MQ4 module has no `_acc` entry and its bytes never
//    change, so the exact per-output F32 fold is produced here by an independent scalar kernel that
//    reads the SAME pointer-addressed raw QT44 / QT53 bytes: per K128 group an exact int32 dot of
//    signed weight nibble `(u ^ 8) - 8 == u - 8` (the kernel's XOR 0x88888888 rebias) and signed A4,
//    `scale = RN(d_act * f32(sc_f16))`, `sum = fma(scale, float(dot), sum)` from +0, ascending groups.
//    It is NOT the production kernel; the production MQ4 bytes are checked at BF16 only.
//
// Kernarg ABI of the two accumulator kernels = the PM 60-byte ABI (5 pointers, 5 u32):
//   ptrs@0 tiles@8 sorted@16 Xq@24 Y@32  M@40 K@44 x_row_div@48 m_total@52 x_src_rows@56.
// Fixed block sizes are baked in (no blockDim reads: the harness launches via a raw kernarg blob).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if !defined(__gfx1151__) && defined(__HIP_DEVICE_COMPILE__)
#error "qwen4_moe_sym_g1_probe.gfx1151.hip is gfx1151-only"
#endif

static __device__ __forceinline__ float bits_f(uint32_t b) { return __uint_as_float(b); }
static __device__ __forceinline__ uint32_t f_bits(float f) { return __float_as_uint(f); }

// ---- 1. epilogue intrinsic probes (block 256) ------------------------------------------------------------

extern "C" __global__ __launch_bounds__(256) void g1_probe_exp2(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i >= n) return;
    float r;
    asm volatile("v_exp_f32 %0, %1" : "=v"(r) : "v"(bits_f(in[i])));
    out[i] = f_bits(r);
}

extern "C" __global__ __launch_bounds__(256) void g1_probe_ldexp(const uint32_t* __restrict__ x, const int32_t* __restrict__ e,
                                                               uint32_t* __restrict__ out, int n) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i >= n) return;
    float r;
    asm volatile("v_ldexp_f32 %0, %1, %2" : "=v"(r) : "v"(bits_f(x[i])), "v"(e[i]));
    out[i] = f_bits(r);
}

// LowerFDIV32 (denormals enabled, so no mode switch): the sequence of `silu_dense_with`.
extern "C" __global__ __launch_bounds__(256) void g1_probe_div(const uint32_t* __restrict__ num, const uint32_t* __restrict__ den,
                                                             uint32_t* __restrict__ out_seq, uint32_t* __restrict__ out_fdiv_rn, int n) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i >= n) return;
    const float nu = bits_f(num[i]);
    const float de = bits_f(den[i]);
    bool f0, f1;
    // v_div_scale_f32 d, null, den, den, num   /   v_div_scale_f32 n, vcc, num, den, num
    const float ds = __builtin_amdgcn_div_scalef(nu, de, false, &f0); // NOTE: the `f` forms; plain div_scale/div_fmas/div_fixup are the f64 builtins
    const float ns = __builtin_amdgcn_div_scalef(nu, de, true, &f1);
    const float rcp = __builtin_amdgcn_rcpf(ds);
    const float neg_ds = -ds;
    const float fma0 = __builtin_fmaf(neg_ds, rcp, 1.0f);
    const float fma1 = __builtin_fmaf(fma0, rcp, rcp);
    const float mul = ns * fma1;
    const float fma2 = __builtin_fmaf(neg_ds, mul, ns);
    const float fma3 = __builtin_fmaf(fma2, fma1, mul);
    const float fma4 = __builtin_fmaf(neg_ds, fma3, ns);
    const float fmas = __builtin_amdgcn_div_fmasf(fma4, fma1, fma3, f1);
    out_seq[i] = f_bits(__builtin_amdgcn_div_fixupf(fmas, de, nu));
    out_fdiv_rn[i] = f_bits(__fdiv_rn(nu, de));
}

// ---- 2. MQ4 F32 accumulator oracle (block 64 = one thread per weight row) --------------------------------

static __device__ __forceinline__ float f16_bits_to_f32(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short)bits)); }

// Signed value of stored nibble u: the kernel XORs 0x8 into every nibble and reads it as two's complement.
static __device__ __forceinline__ int signed_nibble(uint32_t u) {
    int w = (int)((u ^ 8u) & 15u);
    return w >= 8 ? w - 16 : w;
}

static __device__ __forceinline__ int a4_elem(const unsigned char* qs, int j) {
    const uint32_t nib = ((uint32_t)qs[j >> 1] >> (4 * (j & 1))) & 15u;
    return (int)(nib ^ 8u) - 8; // two's complement nibble
}

// DOWN == 0: QT44 gate/up, row = (K/256) pairs of [sc,zp][sc,zp][64 B][64 B] (136 B);
// DOWN == 1: QT53 down, row = (K/128) groups of [sc,zp][64 B] (68 B).
template <int DOWN>
static __device__ __forceinline__ void mq4_acc_body(const unsigned long long* __restrict__ ptrs, const int* __restrict__ tiles,
                                                    const int* __restrict__ sorted, const unsigned char* __restrict__ xq,
                                                    float* __restrict__ y, int M, int K, int x_row_div, int m_total, int x_src_rows) {
    const int tile = (int)blockIdx.y;
    if (tile * 16 >= m_total) return;
    const int e = tiles[tile];
    if (e < 0) return; // sentinel tile: untouched
    const int row = (int)blockIdx.x * 64 + (int)threadIdx.x;
    if (row >= M) return;
    const unsigned char* w = (const unsigned char*)ptrs[e];
    const int groups = K >> 7;
    const size_t row_bytes = DOWN ? (size_t)groups * 68 : (size_t)(K >> 8) * 136;
    const unsigned char* wrow = w + (size_t)row * row_bytes;
    for (int lane = 0; lane < 16; ++lane) {
        const int p = tile * 16 + lane;
        const int s = sorted[p];
        if (s < 0) { y[(size_t)p * M + row] = 0.0f; continue; }
        const int xr = s / x_row_div;
        float sum = 0.0f;
        for (int g = 0; g < groups; ++g) {
            const unsigned char* a = xq + ((size_t)g * x_src_rows + xr) * 72;
            const float d = *(const float*)a;
            const unsigned char* qs = a + 8;
            const unsigned char *hdr, *nib;
            if (DOWN) { hdr = wrow + (size_t)g * 68; nib = hdr + 4; }
            else { hdr = wrow + (size_t)(g >> 1) * 136 + 4 * (g & 1); nib = wrow + (size_t)(g >> 1) * 136 + 8 + 64 * (g & 1); }
            const float sc = f16_bits_to_f32((uint32_t)hdr[0] | ((uint32_t)hdr[1] << 8));
            int dot = 0;
            for (int j = 0; j < 128; ++j) {
                const uint32_t u = ((uint32_t)nib[j >> 1] >> (4 * (j & 1))) & 15u;
                dot += signed_nibble(u) * a4_elem(qs, j);
            }
            const float scale = d * sc;
            sum = __builtin_fmaf(scale, (float)dot, sum);
        }
        // gate/up F32 layout `[P][M]`: gate rows 0..M/2, up rows M/2.. (the weight row index is the column).
        y[(size_t)p * M + row] = sum;
    }
}

extern "C" __global__ __launch_bounds__(64) void g1_mq4_acc_gate_up(const unsigned long long* ptrs, const int* tiles, const int* sorted,
                                                                  const unsigned char* xq, float* y, int M, int K, int x_row_div,
                                                                  int m_total, int x_src_rows) {
    mq4_acc_body<0>(ptrs, tiles, sorted, xq, y, M, K, x_row_div, m_total, x_src_rows);
}

extern "C" __global__ __launch_bounds__(64) void g1_mq4_acc_down(const unsigned long long* ptrs, const int* tiles, const int* sorted,
                                                               const unsigned char* xq, float* y, int M, int K, int x_row_div,
                                                               int m_total, int x_src_rows) {
    mq4_acc_body<1>(ptrs, tiles, sorted, xq, y, M, K, x_row_div, m_total, x_src_rows);
}
