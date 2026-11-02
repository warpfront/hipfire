// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Gate-0 byte oracle for exact gfx1201 verify chunks of 64..128 rows
// (PLAN-CHUNK64 §2 Gate 0, §5 kernel matrix, slice O).
//
// On the real 27B MQ4G256V2 weights (qkvza, qkv, gate_up, wo, down, lm_head)
// with seeded fp16 activation bytes (distinct rows, signed zeros, subnormals,
// alternating signs, exact-cancellation rows) and nonzero distinct residual
// initial bytes, every candidate's COMPLETE output buffers (valid rows, untouched
// padded rows, canary tail) are compared at 0-byte tolerance against the named
// singleton F16 WMMA kernel (`gemm_*_mq4g256v2_wmma_gfx12`), itself checked
// against per-row N=1 launches. Candidates:
//   generic   — the public generic dispatch (`gemm_*_hfq4g256_mq4v2`,
//               `gemm_mq4g256v2_batched_lmhead`): attribution only (the selected
//               symbol is recorded by the caller's kernel trace);
//   vt        — the existing exact VT kernels at 17..63 and VT4 explicitly at 64;
//   bt        — the existing F16 BT4@64 / BT8@128 kernels;
//   chunks    — the current DFlash C8 packing, VT 48+48+32 at row offsets;
//   k32       — the Gate-0 standalone 128-row spike of this example (§3: K32
//               slab, fp16 row stride 40, double buffered, BT4/BT8 x W4/W8) for
//               every N=1..128 (BT8) / 1..64 (BT4);
//   prod      — the production exact entry points (`Gpu::gemm_*_verify_exact`,
//               the head through `gemm_mq4g256v2_lmhead_verify_exact` against a
//               +0-initialized singleton reference) for every N=1..128, with the
//               selected tier symbol and its loaded state recorded. The family
//               is the process's `HIPFIRE_CB_VERIFY_PM` (1 PM bundle, 0 hipcc
//               `_k32`); `prodcross` runs the other family in-process over
//               N=64..128 and compares it byte for byte with both the singleton
//               and the selected family. Per-N output digests go to digests.txt.
// perm/neg/graph run on each `--cands` candidate (prod, k32): row permutations,
// negative controls (input bit, header byte, row offset, canary, padded row;
// out-of-domain N refusal for prod) and GraphState capture + 3 replays.
// `time` event-times VT 48+48+32 against the 128-row launches (`prodtime`:
// both production families, fp16 X cached, head via the residual entry).
//
// Build: cargo build -p hipfire-arch-qwen35 --features lab,deltanet --release \
//          --example cb_verify_gemm_oracle -j12
// Run:   cb_verify_gemm_oracle --model <path> --out <abs dir> [--ops a,b]
//          [--seeds 1,2,3] [--n-max 128] [--tests t,..] [--cands prod,k32]
//          [--reps 30] [--mode eager|graph|both]
// Sampled head: --model <real-27B-path> --out <absolute-dir>
//   --tests sampled-head --seeds 1,2,3 --mode eager|graph|both --reps 30
// Run HIPFIRE_CB_VERIFY_PM=1 and =0 in separate fresh processes. Partitions
// are internal and independent of --ops/--n-max. Timings with whole-head=false
// exclude rotation but include the public head's conversion/zero-init work;
// they are not pure kernel-only timings. Private CB refusal/owner fallback
// remains an integration-state-oracle obligation, explicitly printed below.

use hip_bridge::KernargBlob;
use hipfire_arch_qwen35::load_qwen35_bundle;
use hipfire_arch_qwen35::qwen35::LayerWeights;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SequenceHint, SpecLoadCfg};
use rdna_compute::{DType, Gpu, GpuTensor};

/// The existing kernel sources, byte-for-byte the text `rdna_compute::kernels`
/// embeds (that module is crate-private), so the same code objects load.
mod kernels {
    macro_rules! k {
        ($f:literal) => {
            include_str!(concat!("../../../kernels/src/", $f))
        };
    }
    pub const GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_SRC: &str = k!("gemm_qkvza_mq4g256v2_wmma.gfx12.hip");
    pub const GEMM_QKV_MQ4G256V2_WMMA_GFX12_SRC: &str = k!("gemm_qkv_mq4g256v2_wmma.gfx12.hip");
    pub const GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_SRC: &str =
        k!("gemm_gate_up_mq4g256v2_wmma.gfx12.hip");
    pub const GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_SRC: &str =
        k!("gemm_mq4g256v2_residual_wmma.gfx12.hip");
    pub const GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_BT_SRC: &str =
        k!("gemm_qkvza_mq4g256v2_wmma_gfx12_bt.hip");
    pub const GEMM_QKV_MQ4G256V2_WMMA_GFX1201_BT_SRC: &str =
        k!("gemm_qkv_mq4g256v2_wmma_gfx1201_bt.hip");
    pub const GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_BT_SRC: &str =
        k!("gemm_gate_up_mq4g256v2_wmma_gfx12_bt.hip");
    pub const GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_BT_SRC: &str =
        k!("gemm_mq4g256v2_residual_wmma_gfx12_bt.hip");
    pub const GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_VT_SRC: &str = concat!(
        k!("gemm_mq4g256v2_wmma_gfx12_vt_core.hip"),
        k!("gemm_qkvza_mq4g256v2_wmma_gfx12_vt.hip")
    );
    pub const GEMM_QKV_MQ4G256V2_WMMA_GFX12_VT_SRC: &str = concat!(
        k!("gemm_mq4g256v2_wmma_gfx12_vt_core.hip"),
        k!("gemm_qkv_mq4g256v2_wmma_gfx12_vt.hip")
    );
    pub const GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_VT_SRC: &str = concat!(
        k!("gemm_mq4g256v2_wmma_gfx12_vt_core.hip"),
        k!("gemm_gate_up_mq4g256v2_wmma_gfx12_vt.hip")
    );
    pub const GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_VT_SRC: &str = concat!(
        k!("gemm_mq4g256v2_wmma_gfx12_vt_core.hip"),
        k!("gemm_mq4g256v2_residual_wmma_gfx12_vt.hip")
    );
}
use std::ffi::c_void;
use std::fmt::Write as _;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const NMAX: usize = 128;
/// f32 canary words after the NMAX rows of every output buffer.
const CANARY: usize = 256;
/// Overwrite-epilogue initial/canary word (a quiet-NaN payload no kernel writes).
const SENTINEL: u32 = 0x7FBA_DBAD;

/// Standalone K32-slab variant (§3), one core + the four existing epilogues.
const K32_MODULE: &str = "cb_verify_oracle_k32";
const K32_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) uint4v_t;

// The one-tile kernels' dequantization, expression for expression.
__device__ __forceinline__ half8_t k32_dq8(unsigned int pk, _Float16 sc_h, _Float16 zp_h) {
    half8_t a_reg;
    a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h;
    a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h;
    a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h;
    a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h;
    a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h;
    a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h;
    a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h;
    a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h;
    return a_reg;
}

// Block = WAVES waves, wave w owns 16 weight rows; the whole batch panel
// (NR = 16*BT rows, one grid column for N <= NR) is staged per 32-K slab in
// LDS (row stride 40 fp16 = 32 + 8 pad), double buffered: 2*NR*40*2 bytes.
// Per output: one f32 chain from +0, one WMMA per 16-K tile in ascending K,
// the singleton's header selection (tile kt < 8 -> word 0), dequant and C map.
template <int BT, int WAVES, class Op>
__device__ __forceinline__ void mq4v2_k32_gemm(const Op& op, const _Float16* __restrict__ X,
                                               int total_m, int K, int N) {
    constexpr int NT  = WAVES * 32;
    constexpr int NR  = BT * 16;
    constexpr int XS  = 40;
    constexpr int NCH = NR * 4;               // 16-byte chunks per 32-K slab
    constexpr int CPT = (NCH + NT - 1) / NT;
    constexpr bool EVEN = (NCH % NT) == 0;

    __shared__ __attribute__((aligned(16))) _Float16 xs[2][NR][XS];

    const int tid  = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml   = lane & 15;
    const int kg   = lane >> 4;
    const int rs   = (blockIdx.x * WAVES + wave) * 16;
    const int bs   = blockIdx.y * NR;

    if (blockIdx.x * WAVES * 16 >= total_m || bs >= N) return;   // block-uniform
    const bool active = rs < total_m;                            // wave-uniform

    const int my_row   = rs + ml;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);
    const int gpr      = K / 256;
    const int S        = gpr * 8;             // 32-K slabs
    const char* rb     = op.row(safe_row, (long long)gpr * 136);

    const _Float16* xsrc[CPT];
    int xdst[CPT];
    bool xok[CPT];
    #pragma unroll
    for (int j = 0; j < CPT; j++) {
        const int c = tid + j * NT;
        const int r = c >> 2;
        const int cc = c & 3;
        const int t = bs + r;
        xsrc[j] = X + (long long)((t < N) ? t : 0) * K + cc * 8;
        xdst[j] = r * XS + cc * 8;
        xok[j] = EVEN || (c < NCH);
    }

    uint4v_t xr[CPT];
    unsigned int p0, p1, q0, q1, h, hn;

    #define K32_LDX(s) \
        _Pragma("unroll") \
        for (int j = 0; j < CPT; j++) if (xok[j]) xr[j] = *(const uint4v_t*)(xsrc[j] + (long long)(s) * 32)
    #define K32_STX(buf) \
        _Pragma("unroll") \
        for (int j = 0; j < CPT; j++) if (xok[j]) *(uint4v_t*)(&xs[buf][0][0] + xdst[j]) = xr[j]
    #define K32_LDW(s, a0, a1, hh) { \
        const char* gp_ = rb + ((s) >> 3) * 136; \
        hh = *(const unsigned int*)(gp_ + (((s) >> 2) & 1) * 4); \
        const char* pw_ = gp_ + 8 + ((s) & 7) * 16 + kg * 4; \
        a0 = *(const unsigned int*)(pw_); \
        a1 = *(const unsigned int*)(pw_ + 8); }

    K32_LDX(0);
    K32_LDW(0, p0, p1, h);
    K32_STX(0);
    __syncthreads();

    float8_t acc[BT];
    #pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};

    for (int s = 0; s < S; s++) {
        const int cur = s & 1;
        const bool more = (s + 1) < S;
        if (more) {
            K32_LDX(s + 1);
            K32_LDW(s + 1, q0, q1, hn);
        }
        if (active) {
            const _Float16 sc_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
            const _Float16 zp_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16)));
            const half8_t a0 = k32_dq8(p0, sc_h, zp_h);
            #pragma unroll
            for (int b = 0; b < BT; b++) {
                const half8_t bb = *(const half8_t*)(&xs[cur][b * 16 + ml][kg * 8]);
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, bb, acc[b]);
            }
            const half8_t a1 = k32_dq8(p1, sc_h, zp_h);
            #pragma unroll
            for (int b = 0; b < BT; b++) {
                const half8_t bb = *(const half8_t*)(&xs[cur][b * 16 + ml][16 + kg * 8]);
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, bb, acc[b]);
            }
        }
        if (more) {
            K32_STX(cur ^ 1);
            p0 = q0; p1 = q1; h = hn;
        }
        __syncthreads();
    }
    #undef K32_LDX
    #undef K32_STX
    #undef K32_LDW

    if (!active) return;
    #pragma unroll
    for (int b = 0; b < BT; b++) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int orow = rs + 8 * kg + j;
                if (orow < total_m) op.store(orow, oc, acc[b][j]);
            }
        }
    }
}

struct ResidualOp {
    const char* A; float* Y; int M;
    __device__ __forceinline__ const char* row(int sr, long long st) const { return A + (long long)sr * st; }
    __device__ __forceinline__ void store(int r, int c, float v) const { Y[(long long)c * M + r] += v; }
};
struct QkvzaOp {
    const char* A_qkv; const char* A_z; const char* A_beta; const char* A_alpha;
    float* Y_qkv; float* Y_z; float* Y_beta; float* Y_alpha;
    int qkv_m, z_m, beta_m, alpha_m;
    __device__ __forceinline__ const char* row(int sr, long long st) const {
        if (sr < qkv_m) return A_qkv + (long long)sr * st;
        if (sr < qkv_m + z_m) return A_z + (long long)(sr - qkv_m) * st;
        if (sr < qkv_m + z_m + beta_m) return A_beta + (long long)(sr - (qkv_m + z_m)) * st;
        return A_alpha + (long long)(sr - (qkv_m + z_m + beta_m)) * st;
    }
    __device__ __forceinline__ void store(int r, int c, float v) const {
        if (r < qkv_m) Y_qkv[(long long)c * qkv_m + r] = v;
        else if (r < qkv_m + z_m) Y_z[(long long)c * z_m + (r - qkv_m)] = v;
        else if (r < qkv_m + z_m + beta_m) Y_beta[(long long)c * beta_m + (r - (qkv_m + z_m))] = v;
        else Y_alpha[(long long)c * alpha_m + (r - (qkv_m + z_m + beta_m))] = v;
    }
};
struct QkvOp {
    const char* A_q; const char* A_k; const char* A_v;
    float* Y_q; float* Y_k; float* Y_v;
    int q_m, k_m, v_m;
    __device__ __forceinline__ const char* row(int sr, long long st) const {
        if (sr < q_m) return A_q + (long long)sr * st;
        if (sr < q_m + k_m) return A_k + (long long)(sr - q_m) * st;
        return A_v + (long long)(sr - q_m - k_m) * st;
    }
    __device__ __forceinline__ void store(int r, int c, float v) const {
        if (r < q_m) Y_q[(long long)c * q_m + r] = v;
        else if (r < q_m + k_m) Y_k[(long long)c * k_m + (r - q_m)] = v;
        else Y_v[(long long)c * v_m + (r - q_m - k_m)] = v;
    }
};
struct GateUpOp {
    const char* A_gate; const char* A_up; float* Y_gate; float* Y_up; int gate_m, up_m;
    __device__ __forceinline__ const char* row(int sr, long long st) const {
        return (sr < gate_m) ? (A_gate + (long long)sr * st) : (A_up + (long long)(sr - gate_m) * st);
    }
    __device__ __forceinline__ void store(int r, int c, float v) const {
        if (r < gate_m) Y_gate[(long long)c * gate_m + r] = v;
        else Y_up[(long long)c * up_m + (r - gate_m)] = v;
    }
};

#define GEN_K32(BT, W) \
extern "C" __launch_bounds__(W * 32) __global__ void cbo_k32_residual_bt##BT##w##W( \
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, int M, int K, int N) { \
    const ResidualOp op{A, Y, M}; mq4v2_k32_gemm<BT, W>(op, X, M, K, N); } \
extern "C" __launch_bounds__(W * 32) __global__ void cbo_k32_qkvza_bt##BT##w##W( \
    const char* __restrict__ A_qkv, const char* __restrict__ A_z, const char* __restrict__ A_beta, \
    const char* __restrict__ A_alpha, const _Float16* __restrict__ X, float* __restrict__ Y_qkv, \
    float* __restrict__ Y_z, float* __restrict__ Y_beta, float* __restrict__ Y_alpha, \
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int N) { \
    const QkvzaOp op{A_qkv, A_z, A_beta, A_alpha, Y_qkv, Y_z, Y_beta, Y_alpha, qkv_m, z_m, beta_m, alpha_m}; \
    mq4v2_k32_gemm<BT, W>(op, X, qkv_m + z_m + beta_m + alpha_m, K, N); } \
extern "C" __launch_bounds__(W * 32) __global__ void cbo_k32_qkv_bt##BT##w##W( \
    const char* __restrict__ A_q, const char* __restrict__ A_k, const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, float* __restrict__ Y_q, float* __restrict__ Y_k, float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, int K, int N) { \
    const QkvOp op{A_q, A_k, A_v, Y_q, Y_k, Y_v, q_m, k_m, v_m}; \
    mq4v2_k32_gemm<BT, W>(op, X, q_m + k_m + v_m, K, N); } \
extern "C" __launch_bounds__(W * 32) __global__ void cbo_k32_gate_up_bt##BT##w##W( \
    const char* __restrict__ A_gate, const char* __restrict__ A_up, const _Float16* __restrict__ X, \
    float* __restrict__ Y_gate, float* __restrict__ Y_up, int gate_m, int up_m, int K, int N) { \
    const GateUpOp op{A_gate, A_up, Y_gate, Y_up, gate_m, up_m}; \
    mq4v2_k32_gemm<BT, W>(op, X, gate_m + up_m, K, N); }

GEN_K32(4, 4)
GEN_K32(4, 8)
GEN_K32(8, 4)
GEN_K32(8, 8)
"#;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fam {
    Qkvza,
    Qkv,
    GateUp,
    Residual,
}

impl Fam {
    fn prefix(self) -> &'static str {
        match self {
            Fam::Qkvza => "gemm_qkvza_mq4g256v2_wmma_gfx12",
            Fam::Qkv => "gemm_qkv_mq4g256v2_wmma_gfx12",
            Fam::GateUp => "gemm_gate_up_mq4g256v2_wmma_gfx12",
            Fam::Residual => "gemm_mq4g256v2_residual_wmma_gfx12",
        }
    }
    fn short(self) -> &'static str {
        match self {
            Fam::Qkvza => "qkvza",
            Fam::Qkv => "qkv",
            Fam::GateUp => "gate_up",
            Fam::Residual => "residual",
        }
    }
    /// (module, source) of the named singleton one-tile kernel, as gemm.rs loads it.
    fn single(self) -> (&'static str, &'static str) {
        match self {
            Fam::Qkvza => (
                "gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2",
                kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_SRC,
            ),
            Fam::Qkv => (
                "gemm_qkv_hfq4g256_wmma_gfx12_mq4v2",
                kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_SRC,
            ),
            Fam::GateUp => (
                "gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2",
                kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_SRC,
            ),
            Fam::Residual => (
                "gemm_hfq4g256_residual_wmma_gfx12_mq4v2",
                kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_SRC,
            ),
        }
    }
    fn vt(self) -> (String, &'static str) {
        let src = match self {
            Fam::Qkvza => kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_VT_SRC,
            Fam::Qkv => kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_VT_SRC,
            Fam::GateUp => kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_VT_SRC,
            Fam::Residual => kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_VT_SRC,
        };
        (format!("{}_vt", self.prefix()), src)
    }
    /// (module, source, symbol) of the existing F16 BT kernel.
    fn bt(self, bv: usize) -> (String, &'static str, String) {
        match self {
            Fam::Qkvza => (
                format!("gemm_qkvza_hfq4g256_wmma_gfx12_bt{bv}_mq4v2"),
                kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_BT_SRC,
                format!("gemm_qkvza_mq4g256v2_wmma_gfx12_bt{bv}"),
            ),
            Fam::Qkv => (
                "gemm_qkv_mq4g256v2_wmma_gfx1201_bt".into(),
                kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX1201_BT_SRC,
                format!("gemm_qkv_mq4g256v2_wmma_gfx1201_bt{bv}"),
            ),
            Fam::GateUp => (
                format!("gemm_gate_up_hfq4g256_wmma_gfx12_bt{bv}_mq4v2"),
                kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_BT_SRC,
                format!("gemm_gate_up_mq4g256v2_wmma_gfx12_bt{bv}"),
            ),
            Fam::Residual => (
                format!("gemm_hfq4g256_residual_wmma_gfx12_bt{bv}_mq4v2"),
                kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_BT_SRC,
                format!("gemm_mq4g256v2_residual_wmma_gfx12_bt{bv}"),
            ),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Named one-tile singleton WMMA.
    Single,
    /// Existing verify-tile kernel `_vt{bt}w{w}`.
    Vt(usize, usize),
    /// Existing F16 BT kernel `_bt{bv}`.
    Bt(usize),
    /// This example's K32 variant `cbo_k32_*_bt{bt}w{w}`.
    K32(usize, usize),
    /// The production exact entry point (`Gpu::gemm_*_verify_exact`; the head
    /// case uses `gemm_mq4g256v2_lmhead_verify_exact`) on the F32 panel, with
    /// the fp16 activation cache invalidated first.
    Prod,
    /// The production entry point for kernel timing: fp16 cache kept, the head
    /// case through the residual entry (no memset).
    ProdKernel,
}

/// The two activation panels: raw fp16 bytes (named kernels) and their exact
/// F32 values (production entries, which convert back to the same fp16).
struct Panels {
    x16: GpuTensor,
    x32: GpuTensor,
}

/// Product wave rule (gemm_vt.rs VT_WIDE_MIN_ROW_TILES).
fn waves_for(total_m: usize) -> usize {
    if (total_m + 15) / 16 >= 1024 {
        8
    } else {
        4
    }
}

struct Case<'a> {
    name: &'static str,
    fam: Fam,
    w: Vec<&'a GpuTensor>,
    ms: Vec<usize>,
    k: usize,
    lmhead: bool,
}

impl Case<'_> {
    fn total_m(&self) -> usize {
        self.ms.iter().sum()
    }
}

/// Output buffers of one case: device Y, device init copy, host init bytes.
struct Outs {
    y: Vec<GpuTensor>,
    init_dev: Vec<GpuTensor>,
    init: Vec<Vec<u8>>,
}

fn ptr_at(t: &GpuTensor, byte_off: usize) -> *mut c_void {
    unsafe { (t.buf.as_ptr() as *mut u8).add(byte_off) as *mut c_void }
}

fn ensure(gpu: &mut Gpu, case: &Case, kind: Kind) -> R<String> {
    let f = case.fam;
    let sym = match kind {
        Kind::Single => {
            let (m, s) = f.single();
            gpu.ensure_kernel_public(m, s, f.prefix())?;
            f.prefix().to_string()
        }
        Kind::Vt(bt, w) => {
            let (m, s) = f.vt();
            let sym = format!("{}_vt{bt}w{w}", f.prefix());
            gpu.ensure_kernel_public(&m, s, &sym)?;
            sym
        }
        Kind::Bt(bv) => {
            let (m, s, sym) = f.bt(bv);
            gpu.ensure_kernel_public(&m, s, &sym)?;
            sym
        }
        Kind::K32(bt, w) => {
            let sym = format!("cbo_k32_{}_bt{bt}w{w}", f.short());
            gpu.ensure_kernel_public(K32_MODULE, K32_SRC, &sym)?;
            sym
        }
        Kind::Prod | Kind::ProdKernel => {
            return Err("production entries load their own kernels".into())
        }
    };
    Ok(sym)
}

/// Launch `kind` over batch rows [row0, row0 + n) of X/Y (row offsets applied
/// to the pointers, as a chunk launch would).
fn launch(
    gpu: &mut Gpu,
    case: &Case,
    kind: Kind,
    px: &Panels,
    y: &[GpuTensor],
    row0: usize,
    n: usize,
) -> R<()> {
    if matches!(kind, Kind::Prod | Kind::ProdKernel) {
        return prod(gpu, case, kind == Kind::Prod, &px.x32, y, row0, n);
    }
    let sym = ensure(gpu, case, kind)?;
    let tm = case.total_m();
    let mut b = KernargBlob::new();
    for w in &case.w {
        b.push_ptr(w.buf.as_ptr());
    }
    b.push_ptr(ptr_at(&px.x16, row0 * case.k * 2));
    for (t, &m) in y.iter().zip(&case.ms) {
        b.push_ptr(ptr_at(t, row0 * m * 4));
    }
    for &m in &case.ms {
        b.push_i32(m as i32);
    }
    b.push_i32(case.k as i32);
    b.push_i32(n as i32);
    let (grid, block) = match kind {
        Kind::Single => (
            [((tm + 15) / 16) as u32, ((n + 15) / 16) as u32, 1],
            [32, 1, 1],
        ),
        Kind::Bt(bv) => (
            [
                ((tm + 15) / 16) as u32,
                ((n + 16 * bv - 1) / (16 * bv)) as u32,
                1,
            ],
            [32, 1, 1],
        ),
        Kind::Vt(bt, w) | Kind::K32(bt, w) => (
            [
                ((tm + 16 * w - 1) / (16 * w)) as u32,
                ((n + 16 * bt - 1) / (16 * bt)) as u32,
                1,
            ],
            [(32 * w) as u32, 1, 1],
        ),
        Kind::Prod | Kind::ProdKernel => unreachable!(),
    };
    if gpu.graphs.capture_mode {
        // A captured launch reads its kernarg blob at replay: keep it alive in
        // the graph state's blob list (heap data of a pushed Vec is stable).
        gpu.graphs.capture_blobs.push(b.as_bytes().to_vec());
        let blob = gpu.graphs.capture_blobs.last_mut().unwrap().as_mut_slice() as *mut [u8];
        gpu.launch_kernel_blob(&sym, grid, block, 0, unsafe { &mut *blob })?;
    } else {
        gpu.launch_kernel_blob(&sym, grid, block, 0, b.as_mut_slice())?;
    }
    Ok(())
}

/// The production exact entry point over batch rows [row0, row0 + n) of the
/// F32 panel and Y. `full`: invalidate the fp16 activation cache first and
/// route the head case through the lm_head entry; otherwise (timing) keep the
/// cache and use the residual entry.
fn prod(
    gpu: &mut Gpu,
    case: &Case,
    full: bool,
    x32: &GpuTensor,
    y: &[GpuTensor],
    row0: usize,
    n: usize,
) -> R<()> {
    let (w, m, k) = (&case.w, &case.ms, case.k);
    // Views cover at most the NMAX-row panels: an out-of-domain `n` (refusal
    // control) still reaches the entry, which must reject it before launching.
    let nv = n.clamp(1, NMAX - row0);
    let x = x32.sub_offset(row0 * k, nv * k);
    let ys: Vec<GpuTensor> = y
        .iter()
        .zip(m)
        .map(|(t, &mm)| t.sub_offset(row0 * mm, nv * mm))
        .collect();
    if full {
        gpu.invalidate_fp16_cache();
    }
    match case.fam {
        Fam::Qkvza => gpu.gemm_qkvza_mq4g256v2_verify_exact(
            w[0], w[1], w[2], w[3], &x, &ys[0], &ys[1], &ys[2], &ys[3], m[0], m[1], m[2], m[3], k,
            n,
        )?,
        Fam::Qkv => gpu.gemm_qkv_mq4g256v2_verify_exact(
            w[0], w[1], w[2], &x, &ys[0], &ys[1], &ys[2], m[0], m[1], m[2], k, n,
        )?,
        Fam::GateUp => gpu.gemm_gate_up_mq4g256v2_verify_exact(
            w[0], w[1], &x, &ys[0], &ys[1], m[0], m[1], k, n,
        )?,
        Fam::Residual if case.lmhead && full => {
            gpu.gemm_mq4g256v2_lmhead_verify_exact(w[0], &x, &ys[0], m[0], k, n)?
        }
        Fam::Residual => gpu.gemm_mq4g256v2_residual_verify_exact(w[0], &x, &ys[0], m[0], k, n)?,
    }
    Ok(())
}

/// The symbol the production entry selects (the gemm_vt.rs tier rule), for
/// the receipt; its loaded state is checked separately.
fn prod_symbol(case: &Case, n: usize, pm: bool) -> (String, u32) {
    let (f, w) = (case.fam, waves_for(case.total_m()));
    if n <= 16 {
        return (f.prefix().to_string(), 32);
    }
    if n < 64 {
        return (
            format!("{}_vt{}w{w}", f.prefix(), (n + 15) / 16),
            32 * w as u32,
        );
    }
    let bt = if n <= 64 { 4 } else { 8 };
    let sym = if pm {
        format!("mq4_verify_{}_pm_gfx1201_bt{bt}w{w}", f.short())
    } else {
        format!("{}_vt{bt}w{w}_k32", f.prefix())
    };
    (sym, 32 * w as u32)
}

/// Whether `sym` is loaded in this process (the occupancy query resolves the
/// loaded function by name).
fn loaded(gpu: &Gpu, sym: &str, block: u32) -> bool {
    gpu.occupancy_max_active_blocks(sym, [block, 1, 1], 0)
        .is_ok()
}

/// Run `f` with `HIPFIRE_CB_VERIFY_PM` forced to `pm` (oracle-only flag swap
/// for the in-process HIP-vs-PM cross check), restoring the parsed flags.
fn with_pm<T>(gpu: &mut Gpu, pm: bool, f: impl FnOnce(&mut Gpu) -> R<T>) -> R<T> {
    let saved = gpu.flags.clone();
    let mut swapped = (*saved).clone();
    swapped.cb_verify_pm = pm;
    gpu.flags = std::sync::Arc::new(swapped);
    let r = f(gpu);
    gpu.flags = saved;
    r
}

/// The public generic dispatch over rows [0, n) (F32 activations).
fn generic(gpu: &mut Gpu, case: &Case, x32: &GpuTensor, y: &[GpuTensor], n: usize) -> R<()> {
    let x = x32.sub_offset(0, n * case.k);
    // The fp16/fp8 activation caches key on the source pointer only; this
    // panel is rewritten between seeds and N, so force reconversion.
    gpu.scratch.invalidate_x_caches_for(x.buf.as_ptr());
    let (w, m, k) = (&case.w, &case.ms, case.k);
    match case.fam {
        Fam::Qkvza => gpu.gemm_qkvza_hfq4g256_mq4v2(
            w[0], w[1], w[2], w[3], &x, &y[0], &y[1], &y[2], &y[3], m[0], m[1], m[2], m[3], k, n,
        )?,
        Fam::Qkv => gpu.gemm_qkv_hfq4g256_mq4v2(
            w[0], w[1], w[2], &x, &y[0], &y[1], &y[2], m[0], m[1], m[2], k, n,
        )?,
        Fam::GateUp => {
            gpu.gemm_gate_up_hfq4g256_mq4v2(w[0], w[1], &x, &y[0], &y[1], m[0], m[1], k, n)?
        }
        Fam::Residual if case.lmhead => {
            gpu.gemm_mq4g256v2_batched_lmhead(w[0], &x, &y[0], m[0], k, n)?
        }
        Fam::Residual => gpu.gemm_hfq4g256_residual_mq4v2(w[0], &x, &y[0], m[0], k, n)?,
    }
    Ok(())
}

fn hash3(a: u64, b: u64, c: u64) -> u64 {
    let mut h = a.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ c.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^ (h >> 33)
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Seeded fp16 activation bits [NMAX, k]: normals 2^-5..2^3, +-0, subnormals;
/// rows r%8==6 alternate sign by column, rows r%8==7 are the exact negation
/// of row r-1 (cancellation across rows of one panel).
fn gen_x(seed: u64, k: usize) -> Vec<u16> {
    let mut x = vec![0u16; NMAX * k];
    for r in 0..NMAX {
        for c in 0..k {
            let v = if r % 8 == 7 {
                x[(r - 1) * k + c] ^ 0x8000
            } else {
                let u = hash3(seed, r as u64, c as u64);
                let sign = ((u >> 63) as u16) << 15;
                let mut bits = match u % 100 {
                    0 | 1 => 0x0000,
                    2 => 0x8000,
                    3 => sign | ((u >> 8) as u16 & 0x03FF).max(1),
                    _ => sign | ((((u >> 16) % 9) as u16 + 10) << 10) | ((u >> 32) as u16 & 0x03FF),
                };
                if r % 8 == 6 && c % 2 == 1 {
                    bits ^= 0x8000;
                }
                bits
            };
            x[r * k + c] = v;
        }
    }
    x
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | ((e2 as u32) << 23) | ((m2 & 0x3FF) << 13)
        }
    } else if e == 31 {
        s | 0x7F80_0000 | (m << 13)
    } else {
        s | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Initial output bytes: residual = distinct nonzero f32 per (row, col)
/// (a sprinkling of -0.0); overwrite = SENTINEL. Canary tail = SENTINEL.
fn gen_init(seed: u64, out_idx: usize, m: usize, residual: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity((NMAX * m + CANARY) * 4);
    for i in 0..NMAX * m {
        let w = if residual {
            let u = hash3(seed ^ 0xA5A5, out_idx as u64, i as u64);
            if u % 97 == 0 {
                0x8000_0000u32
            } else {
                let f = (((u >> 40) as u32 | 1) as f32 / 16_777_216.0 - 0.5) * 4.0;
                f.to_bits()
            }
        } else {
            SENTINEL
        };
        v.extend_from_slice(&w.to_le_bytes());
    }
    for _ in 0..CANARY {
        v.extend_from_slice(&SENTINEL.to_le_bytes());
    }
    v
}

fn alloc_outs(gpu: &mut Gpu, case: &Case, seed: u64, zero_init: bool) -> R<Outs> {
    let resid = case.fam == Fam::Residual;
    let mut o = Outs {
        y: Vec::new(),
        init_dev: Vec::new(),
        init: Vec::new(),
    };
    for (i, &m) in case.ms.iter().enumerate() {
        let mut init = gen_init(seed, i, m, resid);
        if zero_init {
            init[..NMAX * m * 4].fill(0);
        }
        let n = NMAX * m + CANARY;
        let y = gpu.zeros(&[n], DType::F32)?;
        let d = gpu.zeros(&[n], DType::F32)?;
        gpu.hip.memcpy_htod(&d.buf, &init)?;
        o.y.push(y);
        o.init_dev.push(d);
        o.init.push(init);
    }
    Ok(o)
}

fn free_outs(gpu: &mut Gpu, o: Outs) -> R<()> {
    for t in o.y.into_iter().chain(o.init_dev) {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

fn reset(gpu: &mut Gpu, o: &Outs) -> R<()> {
    gpu.hip.device_synchronize()?;
    for (y, d) in o.y.iter().zip(&o.init_dev) {
        gpu.hip.memcpy_dtod(&y.buf, &d.buf, d.buf.size())?;
    }
    gpu.hip.device_synchronize()?;
    Ok(())
}

fn read(gpu: &mut Gpu, o: &Outs) -> R<Vec<Vec<u8>>> {
    gpu.hip.device_synchronize()?;
    let mut out = Vec::new();
    for (y, i) in o.y.iter().zip(&o.init) {
        let mut v = vec![0u8; i.len()];
        gpu.hip.memcpy_dtoh_at(&mut v, &y.buf, 0)?;
        out.push(v);
    }
    Ok(out)
}

/// Expected buffers when rows [0, n) are computed: reference rows + init rest.
fn expect(refb: &[Vec<u8>], init: &[Vec<u8>], ms: &[usize], n: usize) -> Vec<Vec<u8>> {
    refb.iter()
        .zip(init)
        .zip(ms)
        .map(|((r, i), &m)| {
            let cut = n * m * 4;
            let mut v = i.clone();
            v[..cut].copy_from_slice(&r[..cut]);
            v
        })
        .collect()
}

/// First mismatch as "out<o> row<r> col<c> word got/want (+count)", or None.
fn first_diff(got: &[Vec<u8>], want: &[Vec<u8>], ms: &[usize]) -> Option<String> {
    let mut total = 0usize;
    let mut first = None;
    for (o, ((g, w), &m)) in got.iter().zip(want).zip(ms).enumerate() {
        for (wi, (gc, wc)) in g.chunks_exact(4).zip(w.chunks_exact(4)).enumerate() {
            if gc != wc {
                total += 1;
                if first.is_none() {
                    let (r, c) = if wi < NMAX * m {
                        (format!("{}", wi / m), format!("{}", wi % m))
                    } else {
                        ("canary".into(), format!("{}", wi - NMAX * m))
                    };
                    first = Some(format!(
                        "out{o} batchrow {r} outrow {c}: got {:08x} want {:08x}",
                        u32::from_le_bytes(gc.try_into().unwrap()),
                        u32::from_le_bytes(wc.try_into().unwrap())
                    ));
                }
            }
        }
    }
    first.map(|f| format!("{f} (+{} differing words)", total - 1))
}

struct Report {
    text: String,
    fails: usize,
    checks: usize,
}

impl Report {
    fn line(&mut self, s: String) {
        println!("{s}");
        self.text.push_str(&s);
        self.text.push('\n');
    }
    /// A gated 0-byte comparison.
    fn cmp(&mut self, tag: &str, got: &[Vec<u8>], want: &[Vec<u8>], ms: &[usize]) -> bool {
        self.checks += 1;
        match first_diff(got, want, ms) {
            None => true,
            Some(d) => {
                self.fails += 1;
                self.line(format!("FAIL {tag}: {d}"));
                false
            }
        }
    }
}

struct Args {
    model: String,
    out: String,
    ops: Vec<String>,
    seeds: Vec<u64>,
    n_max: usize,
    tests: Vec<String>,
    reps: usize,
    mode: String,
    /// Candidates of the perm/neg/graph tests: `prod` (production entry) and/or `k32` (spike).
    cands: Vec<String>,
}

fn parse_args() -> R<Args> {
    let mut a = Args {
        model: String::new(),
        out: String::new(),
        ops: "qkvza,qkv,gate_up,wo,fa_wo,down,head"
            .split(',')
            .map(String::from)
            .collect(),
        seeds: vec![1, 2, 3],
        n_max: NMAX,
        tests: "refself,prod,prodcross,perm,neg,graph"
            .split(',')
            .map(String::from)
            .collect(),
        reps: 30,
        mode: "both".into(),
        cands: vec!["prod".into()],
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().ok_or_else(|| format!("{k}: missing value"))?;
        let list = |v: &str| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect::<Vec<_>>()
        };
        match k.as_str() {
            "--model" => a.model = v,
            "--out" => a.out = v,
            "--ops" => a.ops = list(&v),
            "--seeds" => a.seeds = v.split(',').map(|s| s.parse()).collect::<Result<_, _>>()?,
            "--n-max" => a.n_max = v.parse()?,
            "--tests" => a.tests = list(&v),
            "--reps" => a.reps = v.parse()?,
            "--mode" => a.mode = v,
            "--cands" => a.cands = list(&v),
            _ => return Err(format!("unknown argument {k}").into()),
        }
    }
    if a.model.is_empty() || !a.out.starts_with('/') {
        return Err("--model <path> and absolute --out <dir> are required".into());
    }
    if !(1..=NMAX).contains(&a.n_max) || !["eager", "graph", "both"].contains(&a.mode.as_str()) {
        return Err("--n-max 1..128, --mode eager|graph|both".into());
    }
    Ok(a)
}

/// Product is deliberately dispatched at each lane's original b, never at S.
fn sampled_head_call(
    gpu: &mut Gpu,
    case: &Case,
    head: &hipfire_runtime::llama::WeightTensor,
    hidden: &GpuTensor,
    rotated: &GpuTensor,
    y: &[GpuTensor],
    partition: &[usize],
    exact: bool,
    residual: bool,
    rotate: bool,
) -> R<()> {
    let total: usize = partition.iter().sum();
    let mut offset = 0;
    let counts = if exact {
        vec![total]
    } else {
        partition.to_vec()
    };
    for n in counts {
        let x = hidden.sub_offset(offset * case.k, n * case.k);
        let xr = rotated.sub_offset(offset * case.k, n * case.k);
        let out = y[0].sub_offset(offset * case.ms[0], n * case.ms[0]);
        if rotate {
            hipfire_runtime::llama::rotate_x_mq_batched_for(gpu, head, &x, &xr, case.k, n)?;
            gpu.invalidate_fp16_cache();
            gpu.scratch.invalidate_x_caches_for(xr.buf.as_ptr());
        }
        if residual {
            if exact {
                gpu.gemm_mq4g256v2_residual_verify_exact(
                    case.w[0], &xr, &out, case.ms[0], case.k, n,
                )?;
            } else {
                gpu.gemm_hfq4g256_residual_mq4v2(case.w[0], &xr, &out, case.ms[0], case.k, n)?;
            }
        } else if exact {
            gpu.gemm_mq4g256v2_lmhead_verify_exact(case.w[0], &xr, &out, case.ms[0], case.k, n)?;
        } else {
            gpu.gemm_mq4g256v2_batched_lmhead(case.w[0], &xr, &out, case.ms[0], case.k, n)?;
        }
        offset += n;
    }
    Ok(())
}

fn sampled_picks(bytes: &[u8], m: usize, n: usize) -> Vec<usize> {
    bytes[..n * m * 4]
        .chunks_exact(m * 4)
        .map(|row| {
            let mut best = f32::NEG_INFINITY;
            let mut pick = 0;
            for (i, word) in row.chunks_exact(4).enumerate() {
                let v = f32::from_le_bytes(word.try_into().unwrap());
                if v > best {
                    best = v;
                    pick = i;
                }
            }
            pick
        })
        .collect()
}

fn sampled_head(
    gpu: &mut Gpu,
    case: &Case,
    head: &hipfire_runtime::llama::WeightTensor,
    args: &Args,
    rep: &mut Report,
) -> R<()> {
    if head.gpu_dtype != DType::MQ4G256V2 || case.k == 0 || case.k % 256 != 0 {
        return Err("sampled-head requires the real model's MQ4G256V2 head with positive K divisible by 256".into());
    }
    // IU4 prefill (default on) only engages at >= 64 rows outside the IU4
    // verify scope (`Gpu::iu4_*_active`); the per-lane Product reference runs
    // at b <= 16 and the exact head API never consults it. So only the verify
    // scope itself (any batch) and an A8 route active at a reference b refuse.
    if !gpu.mq4_verify_singleton_chain()
        || gpu.flags.cb_verify_fp8
        || gpu.iu4_verify_scope
        || (2..=16).any(|b| gpu.a8_prefill_active(b, case.k))
    {
        return Err("sampled-head requires admitted normal gfx1201 F16/head/batch-tiles policy without FP8/A8/IU4 verify scope".into());
    }
    let mut partitions: Vec<Vec<usize>> = (2..=16).map(|b| vec![b; 2]).collect();
    partitions.push(vec![8; 8]);
    partitions.push(vec![16; 8]);
    for total in [16, 17, 32, 33, 48, 49, 63, 64, 65, 96, 128] {
        let mut left = total;
        let mut p = Vec::new();
        while left > 16 {
            let b = if left == 17 { 15 } else { 16 };
            p.push(b);
            left -= b;
        }
        p.push(left);
        partitions.push(p);
    }
    partitions.push(vec![2, 3, 5, 7, 8, 9, 14, 16]);
    let hidden = gpu.zeros(&[NMAX * case.k], DType::F32)?;
    let rotated = gpu.zeros(&[NMAX * case.k], DType::F32)?;
    // Backend policy only: the private CB owner predicate needs the state oracle.
    for fixture in ["arch", "fp16", "head", "tiles", "lds"] {
        let mut flags = (*gpu.flags).clone();
        let arch = if fixture == "arch" {
            "gfx1100"
        } else {
            "gfx1201"
        };
        match fixture {
            "fp16" => flags.fp16_disabled = true,
            "head" => flags.lm_head_wmma_disabled = true,
            "tiles" => flags.wmma_batch_tiles = false,
            "lds" => flags.hfq4g256_ldsstage_wmma = true,
            _ => {}
        }
        rep.checks += 1;
        let refused = !Gpu::mq4_verify_singleton_chain_for(arch, &flags);
        if !refused {
            rep.fails += 1;
        }
        rep.line(format!(
            "{} sampled-head backend-policy refusal={fixture}",
            if refused { "PASS" } else { "FAIL" }
        ));
    }
    let refusal = alloc_outs(gpu, case, 1, false)?;
    reset(gpu, &refusal)?;
    let result = prod(gpu, case, true, &hidden, &refusal.y, 0, 129);
    rep.checks += 1;
    if result.is_ok() {
        rep.fails += 1;
        rep.line("FAIL sampled-head exact S=129 not refused".into());
    } else {
        rep.line("PASS sampled-head exact S=129 refused".into());
    }
    rep.cmp(
        "sampled-head exact S=129 no writes",
        &read(gpu, &refusal)?,
        &refusal.init,
        &case.ms,
    );
    free_outs(gpu, refusal)?;
    rep.line("UNEXERCISED private CB fallback/owner admission for b1, FP8/IU4/A8 scope, recording/G0; requires integration state oracle".into());
    for &seed in &args.seeds {
        let bits = gen_x(seed, case.k);
        for permuted in [false, true] {
            let input: Vec<u8> = (0..NMAX)
                .flat_map(|r| {
                    let source = if permuted { (r * 37 + 11) % NMAX } else { r };
                    bits[source * case.k..(source + 1) * case.k]
                        .iter()
                        .flat_map(|&h| f16_to_f32(h).to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .collect();
            gpu.hip.memcpy_htod(&hidden.buf, &input)?;
            // b1 is intentionally run only through the unchanged generic path:
            // its scalar GEMV chain is outside the sampled packed contract.
            let scalar = alloc_outs(gpu, case, seed, true)?;
            reset(gpu, &scalar)?;
            sampled_head_call(
                gpu,
                case,
                head,
                &hidden,
                &rotated,
                &scalar.y,
                &[1],
                false,
                false,
                true,
            )?;
            let scalar_want = read(gpu, &scalar)?;
            reset(gpu, &scalar)?;
            sampled_head_call(
                gpu,
                case,
                head,
                &hidden,
                &rotated,
                &scalar.y,
                &[1],
                false,
                false,
                true,
            )?;
            let scalar_ok = rep.cmp(
                "sampled-head b1 old-Product rerun (not exact admission)",
                &read(gpu, &scalar)?,
                &scalar_want,
                &case.ms,
            );
            rep.line(format!(
                "{} sampled-head b1 old-Product only",
                if scalar_ok { "PASS" } else { "FAIL" }
            ));
            // The comparator itself must detect an output bit and a canary bit.
            for index in [0, scalar_want[0].len() - 1] {
                let mut changed = scalar_want.clone();
                changed[0][index] ^= 1;
                rep.checks += 1;
                if first_diff(&changed, &scalar_want, &case.ms).is_none() {
                    rep.fails += 1;
                    rep.line(format!(
                        "FAIL sampled-head negative comparator byte={index}"
                    ));
                } else {
                    rep.line(format!(
                        "PASS sampled-head negative comparator byte={index} detected"
                    ));
                }
            }
            free_outs(gpu, scalar)?;
            for partition in &partitions {
                let total: usize = partition.iter().sum();
                for residual in [false, true] {
                    let reference = alloc_outs(gpu, case, seed, !residual)?;
                    let candidate = alloc_outs(gpu, case, seed, !residual)?;
                    reset(gpu, &reference)?;
                    sampled_head_call(
                        gpu,
                        case,
                        head,
                        &hidden,
                        &rotated,
                        &reference.y,
                        partition,
                        false,
                        residual,
                        true,
                    )?;
                    let want = read(gpu, &reference)?;
                    reset(gpu, &candidate)?;
                    sampled_head_call(
                        gpu,
                        case,
                        head,
                        &hidden,
                        &rotated,
                        &candidate.y,
                        partition,
                        true,
                        residual,
                        true,
                    )?;
                    let got = read(gpu, &candidate)?;
                    let tag = format!("sampled-head seed={seed} perm={permuted} b={partition:?} S={total} residual={residual}");
                    let mut ok = rep.cmp(&tag, &got, &want, &case.ms);
                    // Independently verify untouched padding and canaries in both arms.
                    for (name, bytes, init) in [
                        ("Product", &want, &reference.init),
                        ("exact", &got, &candidate.init),
                    ] {
                        let cut = total * case.ms[0] * 4;
                        rep.checks += 1;
                        if bytes[0][cut..] != init[0][cut..] {
                            rep.fails += 1;
                            ok = false;
                            rep.line(format!("FAIL {tag} {name} padding/canary"));
                        }
                    }
                    rep.checks += 1;
                    if sampled_picks(&got[0], case.ms[0], total)
                        != sampled_picks(&want[0], case.ms[0], total)
                    {
                        rep.fails += 1;
                        ok = false;
                        rep.line(format!("FAIL {tag} argmax"));
                    }
                    if args.mode != "eager" {
                        let stream = gpu.hip.stream_create()?;
                        gpu.active_stream = Some(stream);
                        reset(gpu, &candidate)?;
                        gpu.invalidate_fp16_cache();
                        let dev = gpu.device_id;
                        gpu.graphs.begin_graph_capture(
                            &gpu.hip,
                            dev,
                            gpu.active_stream.as_ref().unwrap(),
                        )?;
                        let launch = sampled_head_call(
                            gpu,
                            case,
                            head,
                            &hidden,
                            &rotated,
                            &candidate.y,
                            partition,
                            true,
                            residual,
                            true,
                        );
                        let end = gpu.graphs.end_graph_capture(
                            &gpu.hip,
                            dev,
                            gpu.active_stream.as_ref().unwrap(),
                        );
                        launch?;
                        end?;
                        ok &= rep.cmp(
                            &format!("{tag} capture-not-executed"),
                            &read(gpu, &candidate)?,
                            &candidate.init,
                            &case.ms,
                        );
                        for replay in 0..3 {
                            reset(gpu, &candidate)?;
                            gpu.graphs.graph_launch(
                                &gpu.hip,
                                dev,
                                gpu.active_stream.as_ref().unwrap(),
                            )?;
                            ok &= rep.cmp(
                                &format!("{tag} replay={replay}"),
                                &read(gpu, &candidate)?,
                                &want,
                                &case.ms,
                            );
                        }
                        gpu.graphs.drop_captured_graph(&gpu.hip, dev);
                        let captured_stream = gpu.active_stream.take().unwrap();
                        gpu.hip.stream_destroy(captured_stream)?;
                    }
                    rep.line(format!(
                        "{} {tag} digest={:016x}",
                        if ok { "PASS" } else { "FAIL" },
                        fnv(&got[0])
                    ));
                    if !permuted && !residual && args.reps > 0 {
                        for exact in [false, true] {
                            for rotate in [false, true] {
                                let e0 = gpu.hip.event_create()?;
                                let e1 = gpu.hip.event_create()?;
                                let mut times = Vec::new();
                                for i in 0..args.reps + 3 {
                                    reset(gpu, &candidate)?;
                                    gpu.hip.event_record(&e0, None)?;
                                    sampled_head_call(
                                        gpu,
                                        case,
                                        head,
                                        &hidden,
                                        &rotated,
                                        &candidate.y,
                                        partition,
                                        exact,
                                        false,
                                        rotate,
                                    )?;
                                    gpu.hip.event_record(&e1, None)?;
                                    gpu.hip.event_synchronize(&e1)?;
                                    if i >= 3 {
                                        times.push(gpu.hip.event_elapsed_ms(&e0, &e1)? as f64);
                                    }
                                }
                                rep.line(format!(
                                    "time {tag} exact={exact} whole-head={rotate} median_ms={:.6}",
                                    median(&mut times)
                                ));
                                gpu.hip.event_destroy(e0)?;
                                gpu.hip.event_destroy(e1)?;
                            }
                        }
                    }
                    free_outs(gpu, reference)?;
                    free_outs(gpu, candidate)?;
                }
            }
        }
    }
    gpu.free_tensor(hidden)?;
    gpu.free_tensor(rotated)?;
    Ok(())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    match run() {
        Ok(0) => std::process::exit(0),
        Ok(_) => std::process::exit(1),
        Err(e) => {
            eprintln!("cb_verify_gemm_oracle: {e}");
            std::process::exit(2);
        }
    }
}

fn run() -> R<usize> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)?;
    let has = |t: &str| args.tests.iter().any(|x| x == t);
    let mut rep = Report {
        text: String::new(),
        fails: 0,
        checks: 0,
    };

    let mut gpu = Gpu::init()?;
    rep.line(format!("arch {} model {}", gpu.arch, args.model));
    rep.line(format!("flags {:?}", gpu.flags));
    rep.line(format!(
        "flags.iu4_prefill_enabled={}",
        gpu.flags.iu4_prefill_enabled()
    ));
    let mut env: Vec<String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("HIPFIRE_"))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    env.sort();
    rep.line(format!("env {env:?}"));
    rep.line(format!(
        "args ops={:?} seeds={:?} n_max={} tests={:?} reps={} mode={}",
        args.ops, args.seeds, args.n_max, args.tests, args.reps, args.mode
    ));

    let src = ModelSource::from_path(&args.model)?;
    let cask = CaskConfig::default();
    let mut lc = LoadCtx {
        path: &args.model,
        max_seq: 4096,
        sequence: Some(SequenceHint {
            model_ctx: 4096,
            automatic: true,
            card_cap: 0,
            reserve_bytes: 0,
        }),
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        vision_path: None,
        mtp_path: None,
        vision_mode: "off".into(),
        kv_mode_override: None,
        kv_k_override: None,
        kv_v_override: None,
        qwen_default_q8: true,
        kv_backend: KvBackend::default(),
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        pp_layer_counts: None,
        admitted_mesh: None,
        band_placement: None,
        spec: SpecLoadCfg::default(),
        gpu: &mut gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
        xdna: None,
        batch_request: Default::default(),
    };
    let b = load_qwen35_bundle(src, &mut lc).map_err(|e| e.to_string())?;
    let dn = b
        .weights
        .layers
        .iter()
        .find_map(|l| {
            if let LayerWeights::DeltaNet(d) = l {
                Some(d)
            } else {
                None
            }
        })
        .ok_or("no DeltaNet layer")?;
    let fa = b
        .weights
        .layers
        .iter()
        .find_map(|l| {
            if let LayerWeights::FullAttn(f) = l {
                Some(f)
            } else {
                None
            }
        })
        .ok_or("no FullAttn layer")?;
    let kd = b.config.dim;
    let all = vec![
        Case {
            name: "qkvza",
            fam: Fam::Qkvza,
            w: vec![&dn.wqkv.buf, &dn.wz.buf, &dn.w_beta.buf, &dn.w_alpha.buf],
            ms: vec![dn.wqkv.m, dn.wz.m, dn.w_beta.m, dn.w_alpha.m],
            k: kd,
            lmhead: false,
        },
        Case {
            name: "qkv",
            fam: Fam::Qkv,
            w: vec![&fa.wq.buf, &fa.wk.buf, &fa.wv.buf],
            ms: vec![fa.wq.m, fa.wk.m, fa.wv.m],
            k: kd,
            lmhead: false,
        },
        Case {
            name: "gate_up",
            fam: Fam::GateUp,
            w: vec![&dn.w_gate.buf, &dn.w_up.buf],
            ms: vec![dn.w_gate.m, dn.w_up.m],
            k: kd,
            lmhead: false,
        },
        Case {
            name: "wo",
            fam: Fam::Residual,
            w: vec![&dn.wo.buf],
            ms: vec![dn.wo.m],
            k: dn.wo.k,
            lmhead: false,
        },
        Case {
            name: "fa_wo",
            fam: Fam::Residual,
            w: vec![&fa.wo.buf],
            ms: vec![fa.wo.m],
            k: fa.wo.k,
            lmhead: false,
        },
        Case {
            name: "down",
            fam: Fam::Residual,
            w: vec![&dn.w_down.buf],
            ms: vec![dn.w_down.m],
            k: dn.w_down.k,
            lmhead: false,
        },
        Case {
            name: "head",
            fam: Fam::Residual,
            w: vec![&b.weights.output.buf],
            ms: vec![b.weights.output.m],
            k: kd,
            lmhead: true,
        },
    ];
    for c in &all {
        let dts: Vec<String> = c.w.iter().map(|t| format!("{:?}", t.dtype)).collect();
        rep.line(format!(
            "shape {} fam={:?} ms={:?} total_m={} k={} dtype={:?} wave_rule=W{}",
            c.name,
            c.fam,
            c.ms,
            c.total_m(),
            c.k,
            dts,
            waves_for(c.total_m())
        ));
        for w in &c.w {
            if w.buf.size() < 136 * (c.k / 256) {
                return Err(format!("{}: weight buffer too small", c.name).into());
            }
        }
        if c.k % 256 != 0 {
            return Err(format!("{}: K {} not a multiple of 256", c.name, c.k).into());
        }
    }
    if has("sampled-head") {
        sampled_head(&mut gpu, &all[6], &b.weights.output, &args, &mut rep)?;
        if args.tests.len() == 1 {
            rep.line(if rep.fails == 0 {
                "VERDICT PASS".into()
            } else {
                "VERDICT FAIL".into()
            });
            std::fs::write(format!("{}/report.txt", args.out), &rep.text)?;
            return Ok(rep.fails);
        }
    }
    let cases: Vec<&Case> = all
        .iter()
        .filter(|c| args.ops.iter().any(|o| o == c.name))
        .collect();
    let n_max = args.n_max;

    // Activation panels: the widest K among selected cases.
    let kmax = cases.iter().map(|c| c.k).max().unwrap_or(kd);
    let px = Panels {
        x16: gpu.zeros(&[NMAX * kmax / 2 + 64], DType::F32)?,
        x32: gpu.zeros(&[NMAX * kmax], DType::F32)?,
    };
    let load_x = |gpu: &mut Gpu, bits: &[u16]| -> R<()> {
        let bytes: Vec<u8> = bits.iter().flat_map(|h| h.to_le_bytes()).collect();
        gpu.hip.memcpy_htod(&px.x16.buf, &bytes)?;
        let f: Vec<u8> = bits
            .iter()
            .flat_map(|&h| f16_to_f32(h).to_le_bytes())
            .collect();
        gpu.hip.memcpy_htod(&px.x32.buf, &f)?;
        gpu.hip.device_synchronize()?;
        Ok(())
    };
    let pm_sel = gpu.mq4_verify_pm_selected();
    let family = |pm: bool| if pm { "PM" } else { "HIP" };
    rep.line(format!(
        "production family {} (HIPFIRE_CB_VERIFY_PM -> flags.cb_verify_pm={pm_sel}); wmma_batch_tiles={} mq4_verify_chunk_rows={}",
        family(pm_sel),
        gpu.flags.wmma_batch_tiles,
        gpu.mq4_verify_chunk_rows()
    ));
    // Candidates the perm/neg/graph tests run against.
    let cand_kinds: Vec<&str> = args.cands.iter().map(String::as_str).collect();
    let mut digests = String::new();
    // Other-family wide symbols the cross test has loaded (excluded from the
    // "not loaded" assertion of later cases sharing a family).
    let mut crossed: std::collections::HashSet<String> = std::collections::HashSet::new();

    for case in &cases {
        let (ms, k, tm) = (&case.ms, case.k, case.total_m());
        let wr = waves_for(tm);
        for (si, &seed) in args.seeds.iter().enumerate() {
            let xb = gen_x(seed, k);
            load_x(&mut gpu, &xb)?;
            let tag = |t: &str| format!("{} seed{} {}", case.name, seed, t);
            let outs = alloc_outs(&mut gpu, case, seed, false)?;
            let xh = fnv(&xb.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>());
            let yh: Vec<String> = outs
                .init
                .iter()
                .map(|i| format!("{:016x}", fnv(i)))
                .collect();
            // Reference: named singleton over all NMAX rows.
            reset(&mut gpu, &outs)?;
            launch(&mut gpu, case, Kind::Single, &px, &outs.y, 0, NMAX)?;
            let refb = read(&mut gpu, &outs)?;
            // Determinism control.
            reset(&mut gpu, &outs)?;
            launch(&mut gpu, case, Kind::Single, &px, &outs.y, 0, NMAX)?;
            rep.cmp(&tag("single rerun"), &read(&mut gpu, &outs)?, &refb, ms);
            let rh: Vec<String> = refb.iter().map(|r| format!("{:016x}", fnv(r))).collect();
            rep.line(format!(
                "{} x_fnv={xh:016x} y_init_fnv={yh:?} ref_fnv={rh:?}",
                tag("inputs")
            ));
            // The lm_head entry is zero + acc: its reference is the singleton
            // over +0-initialized rows (rows past N keep the nonzero init).
            let refz = if case.lmhead {
                let z = alloc_outs(&mut gpu, case, seed, true)?;
                reset(&mut gpu, &z)?;
                launch(&mut gpu, case, Kind::Single, &px, &z.y, 0, NMAX)?;
                let r = read(&mut gpu, &z)?;
                free_outs(&mut gpu, z)?;
                rep.line(format!(
                    "{} lm_head zero-init ref_fnv={:016x}",
                    tag("inputs"),
                    fnv(&r[0])
                ));
                Some(r)
            } else {
                None
            };
            let ref_for = |kind: Kind| -> &Vec<Vec<u8>> {
                match (&refz, kind) {
                    (Some(z), Kind::Prod) => z,
                    _ => &refb,
                }
            };
            let cand_kind = |c: &str, n: usize| -> Kind {
                match c {
                    "k32" => Kind::K32(if n <= 64 { 4 } else { 8 }, wr),
                    _ => Kind::Prod,
                }
            };

            if has("refself") {
                // Each row alone (N=1 tile, X/Y at the row offset) == the NMAX launch.
                reset(&mut gpu, &outs)?;
                for r in 0..NMAX {
                    launch(&mut gpu, case, Kind::Single, &px, &outs.y, r, 1)?;
                }
                let ok = rep.cmp(
                    &tag("single N=1 per-row vs N=128"),
                    &read(&mut gpu, &outs)?,
                    &refb,
                    ms,
                );
                rep.line(format!(
                    "{} {}",
                    tag("refself"),
                    if ok { "same" } else { "DIFF" }
                ));
            }

            if has("attrib") {
                // Generic dispatch vs singleton (lm_head generic zeroes Y: zero-init reference).
                let zo = if case.lmhead {
                    Some(alloc_outs(&mut gpu, case, seed, true)?)
                } else {
                    None
                };
                let (o, rb) = match &zo {
                    Some(z) => {
                        reset(&mut gpu, z)?;
                        launch(&mut gpu, case, Kind::Single, &px, &z.y, 0, NMAX)?;
                        let r = read(&mut gpu, z)?;
                        (z, r)
                    }
                    None => (&outs, refb.clone()),
                };
                let mut v = Vec::new();
                for n in [1usize, 16, 17, 48, 63, 64, 65, 96, 128] {
                    if n > n_max {
                        continue;
                    }
                    reset(&mut gpu, o)?;
                    generic(&mut gpu, case, &px.x32, &o.y, n)?;
                    let got = read(&mut gpu, o)?;
                    let want = expect(&rb, &o.init, ms, n);
                    let d = first_diff(&got, &want, ms);
                    v.push(format!(
                        "N={n}:{}",
                        match &d {
                            None => "same".to_string(),
                            Some(s) => format!("DIFF[{s}]"),
                        }
                    ));
                }
                rep.line(format!(
                    "{} generic-vs-single {}",
                    tag("attrib"),
                    v.join(" ")
                ));
                if let Some(z) = zo {
                    free_outs(&mut gpu, z)?;
                }
            }

            if has("vt") {
                let mut v = Vec::new();
                for n in [17usize, 31, 32, 33, 48, 63] {
                    let bt = (n + 15) / 16;
                    reset(&mut gpu, &outs)?;
                    launch(&mut gpu, case, Kind::Vt(bt, wr), &px, &outs.y, 0, n)?;
                    let ok = rep.cmp(
                        &tag(&format!("vt{bt}w{wr} N={n}")),
                        &read(&mut gpu, &outs)?,
                        &expect(&refb, &outs.init, ms, n),
                        ms,
                    );
                    v.push(format!(
                        "vt{bt}w{wr}@{n}:{}",
                        if ok { "same" } else { "DIFF" }
                    ));
                }
                // The boundary test: existing VT4 storage holds 64 rows.
                for w in [4usize, 8] {
                    for n in [64usize, 49] {
                        reset(&mut gpu, &outs)?;
                        launch(&mut gpu, case, Kind::Vt(4, w), &px, &outs.y, 0, n)?;
                        let ok = rep.cmp(
                            &tag(&format!("vt4w{w} N={n}")),
                            &read(&mut gpu, &outs)?,
                            &expect(&refb, &outs.init, ms, n),
                            ms,
                        );
                        v.push(format!("vt4w{w}@{n}:{}", if ok { "same" } else { "DIFF" }));
                    }
                }
                rep.line(format!("{} {}", tag("vt"), v.join(" ")));
            }

            if has("bt") {
                let mut v = Vec::new();
                for (bv, n) in [(4usize, 64usize), (8, 128), (8, 100)] {
                    if n > n_max {
                        continue;
                    }
                    reset(&mut gpu, &outs)?;
                    launch(&mut gpu, case, Kind::Bt(bv), &px, &outs.y, 0, n)?;
                    let ok = rep.cmp(
                        &tag(&format!("bt{bv} N={n}")),
                        &read(&mut gpu, &outs)?,
                        &expect(&refb, &outs.init, ms, n),
                        ms,
                    );
                    v.push(format!("bt{bv}@{n}:{}", if ok { "same" } else { "DIFF" }));
                }
                rep.line(format!("{} {}", tag("bt"), v.join(" ")));
            }

            if has("chunks") {
                // Current C8 packing: VT3 48 + VT3 48 + VT2 32 at row offsets.
                reset(&mut gpu, &outs)?;
                for (r0, n) in [(0usize, 48usize), (48, 48), (96, 32)] {
                    launch(
                        &mut gpu,
                        case,
                        Kind::Vt((n + 15) / 16, wr),
                        &px,
                        &outs.y,
                        r0,
                        n,
                    )?;
                }
                let ok = rep.cmp(&tag("vt 48+48+32"), &read(&mut gpu, &outs)?, &refb, ms);
                rep.line(format!(
                    "{} 48+48+32 {}",
                    tag("chunks"),
                    if ok { "same" } else { "DIFF" }
                ));
            }

            if has("k32") {
                for (bt, w) in [(8usize, 4usize), (8, 8), (4, 4), (4, 8)] {
                    let top = (16 * bt).min(n_max);
                    let mut bad = Vec::new();
                    for n in 1..=top {
                        reset(&mut gpu, &outs)?;
                        launch(&mut gpu, case, Kind::K32(bt, w), &px, &outs.y, 0, n)?;
                        if !rep.cmp(
                            &tag(&format!("k32 bt{bt}w{w} N={n}")),
                            &read(&mut gpu, &outs)?,
                            &expect(&refb, &outs.init, ms, n),
                            ms,
                        ) {
                            bad.push(n);
                        }
                    }
                    rep.line(format!(
                        "{} k32 bt{bt}w{w} N=1..{top}: {}",
                        tag("k32"),
                        if bad.is_empty() {
                            "all same".to_string()
                        } else {
                            format!("DIFF at {bad:?}")
                        }
                    ));
                }
            }

            // Production outputs of this seed, per N (for the cross check).
            let mut prod_fnv: Vec<Option<Vec<u64>>> = vec![None; NMAX + 1];
            if has("prod") {
                // Every N through the production entry, 0-byte vs the singleton
                // over the complete buffers (padded rows and canaries included).
                let rz = ref_for(Kind::Prod);
                let mut bad = Vec::new();
                let mut syms: Vec<String> = Vec::new();
                for n in 1..=n_max {
                    reset(&mut gpu, &outs)?;
                    launch(&mut gpu, case, Kind::Prod, &px, &outs.y, 0, n)?;
                    let got = read(&mut gpu, &outs)?;
                    let (sym, _) = prod_symbol(case, n, pm_sel);
                    if syms.last() != Some(&sym) {
                        syms.push(sym.clone());
                    }
                    if !rep.cmp(
                        &tag(&format!("prod[{}] {sym} N={n}", family(pm_sel))),
                        &got,
                        &expect(rz, &outs.init, ms, n),
                        ms,
                    ) {
                        bad.push(n);
                    }
                    let f: Vec<u64> = got.iter().map(|g| fnv(g)).collect();
                    let _ = writeln!(
                        digests,
                        "{} {seed} {n} {}",
                        case.name,
                        f.iter()
                            .map(|x| format!("{x:016x}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    prod_fnv[n] = Some(f);
                }
                rep.line(format!(
                    "{} prod[{}] N=1..{n_max}: {} | tiers {}",
                    tag("prod"),
                    family(pm_sel),
                    if bad.is_empty() {
                        "all same".to_string()
                    } else {
                        format!("DIFF at {bad:?}")
                    },
                    syms.join(" -> ")
                ));
                if si == 0 {
                    // Loaded identity: the selected family's wide symbols are
                    // loaded, the other family's are not (before any cross run).
                    let mut v = Vec::new();
                    for n in [64usize, 128] {
                        if n > n_max {
                            continue;
                        }
                        let (s_sel, blk) = prod_symbol(case, n, pm_sel);
                        let (s_oth, _) = prod_symbol(case, n, !pm_sel);
                        let (l_sel, l_oth) = (loaded(&gpu, &s_sel, blk), loaded(&gpu, &s_oth, blk));
                        rep.checks += 1;
                        if !l_sel || (l_oth && !crossed.contains(&s_oth)) {
                            rep.fails += 1;
                            rep.line(format!("FAIL {}: selected {s_sel} loaded={l_sel}, other {s_oth} loaded={l_oth}", tag("identity")));
                        }
                        v.push(format!(
                            "N={n}: {s_sel} loaded={l_sel}; {s_oth} loaded={l_oth}{}",
                            if crossed.contains(&s_oth) {
                                " (by earlier cross)"
                            } else {
                                ""
                            }
                        ));
                    }
                    rep.line(format!("{} {}", tag("identity"), v.join(" | ")));
                }
            }

            if has("prodcross") {
                // The other family in-process (flag swapped for these calls
                // only) over the wide range: 0-byte vs the singleton and vs the
                // selected family's bytes of the same N.
                let rz = ref_for(Kind::Prod);
                let mut bad = Vec::new();
                let mut vs_sel = 0usize;
                for n in 64..=n_max {
                    let got = with_pm(&mut gpu, !pm_sel, |g| {
                        reset(g, &outs)?;
                        launch(g, case, Kind::Prod, &px, &outs.y, 0, n)?;
                        read(g, &outs)
                    })?;
                    crossed.insert(prod_symbol(case, n, !pm_sel).0);
                    if !rep.cmp(
                        &tag(&format!("prodcross[{}] N={n}", family(!pm_sel))),
                        &got,
                        &expect(rz, &outs.init, ms, n),
                        ms,
                    ) {
                        bad.push(n);
                    }
                    if let Some(f) = &prod_fnv[n] {
                        rep.checks += 1;
                        let g: Vec<u64> = got.iter().map(|x| fnv(x)).collect();
                        if &g == f {
                            vs_sel += 1;
                        } else {
                            rep.fails += 1;
                            rep.line(format!(
                                "FAIL {}: {} vs {} bytes differ at N={n}",
                                tag("prodcross"),
                                family(!pm_sel),
                                family(pm_sel)
                            ));
                        }
                    }
                }
                rep.line(format!(
                    "{} prod[{}] N=64..{n_max} vs singleton: {}; == prod[{}] bytes at {vs_sel} N",
                    tag("prodcross"),
                    family(!pm_sel),
                    if bad.is_empty() {
                        "all same".to_string()
                    } else {
                        format!("DIFF at {bad:?}")
                    },
                    family(pm_sel)
                ));
            }

            for &cand in &cand_kinds {
                if has("perm") {
                    // Arbitrary-row permutation of the first n rows (X and residual init).
                    let mut v = Vec::new();
                    for n in [31usize, 32, 33, 48, 63, 64, 65, 96, 127, 128] {
                        if n > n_max {
                            continue;
                        }
                        let kind = cand_kind(cand, n);
                        let rk = ref_for(kind);
                        let mut pi: Vec<usize> = (0..n).collect();
                        for i in (1..n).rev() {
                            let j = (hash3(seed, n as u64, i as u64) % (i as u64 + 1)) as usize;
                            pi.swap(i, j);
                        }
                        let mut xp = xb.clone();
                        for (i, &p) in pi.iter().enumerate() {
                            xp[i * k..(i + 1) * k].copy_from_slice(&xb[p * k..(p + 1) * k]);
                        }
                        load_x(&mut gpu, &xp)?;
                        let mut init_p = outs.init.clone();
                        let mut want = outs.init.clone();
                        for (o, &m) in ms.iter().enumerate() {
                            for (i, &p) in pi.iter().enumerate() {
                                let (d, s) =
                                    (i * m * 4..(i + 1) * m * 4, p * m * 4..(p + 1) * m * 4);
                                init_p[o][d.clone()].copy_from_slice(&outs.init[o][s.clone()]);
                                want[o][d].copy_from_slice(&rk[o][s]);
                            }
                            gpu.hip.memcpy_htod(&outs.y[o].buf, &init_p[o])?;
                        }
                        launch(&mut gpu, case, kind, &px, &outs.y, 0, n)?;
                        let ok = rep.cmp(
                            &tag(&format!("perm {cand} N={n}")),
                            &read(&mut gpu, &outs)?,
                            &want,
                            ms,
                        );
                        v.push(format!("N={n}:{}", if ok { "same" } else { "DIFF" }));
                    }
                    load_x(&mut gpu, &xb)?;
                    rep.line(format!(
                        "{} {cand} permuted rows {}",
                        tag("perm"),
                        v.join(" ")
                    ));
                }

                if has("neg") {
                    // Negative controls: each must be DETECTED by the comparator.
                    let kn = cand_kind(cand, NMAX);
                    let rk = ref_for(kn);
                    let mut det = Vec::new();
                    let mut detect = |rep: &mut Report, what: &str, detected: bool| {
                        rep.checks += 1;
                        if !detected {
                            rep.fails += 1;
                            rep.line(format!(
                                "FAIL {}: {cand} negative control '{what}' NOT detected",
                                tag("neg")
                            ));
                        }
                        det.push(format!(
                            "{what}:{}",
                            if detected { "detected" } else { "MISSED" }
                        ));
                    };
                    // (a) one altered activation bit.
                    let mut xa = xb.clone();
                    xa[5 * k + 37] ^= 0x0400;
                    load_x(&mut gpu, &xa)?;
                    reset(&mut gpu, &outs)?;
                    launch(&mut gpu, case, kn, &px, &outs.y, 0, NMAX)?;
                    let got = read(&mut gpu, &outs)?;
                    detect(&mut rep, "input bit", first_diff(&got, rk, ms).is_some());
                    load_x(&mut gpu, &xb)?;
                    // (b) altered header (scale of the second half of group 0, weight row 3).
                    let wbytes = case.w[0].buf.size();
                    let wcopy = gpu.zeros(&[wbytes / 4 + 1], DType::F32)?;
                    gpu.hip.memcpy_dtod(&wcopy.buf, &case.w[0].buf, wbytes)?;
                    let row_bytes = (k / 256) * 136;
                    let mut hdr = [0u8; 4];
                    gpu.hip
                        .memcpy_dtoh_at(&mut hdr, &wcopy.buf, 3 * row_bytes + 4)?;
                    hdr[1] ^= 0x04;
                    gpu.hip
                        .memcpy_htod_offset(&wcopy.buf, 3 * row_bytes + 4, &hdr)?;
                    let mut alt = Case {
                        name: case.name,
                        fam: case.fam,
                        w: case.w.clone(),
                        ms: case.ms.clone(),
                        k,
                        lmhead: case.lmhead,
                    };
                    alt.w[0] = &wcopy;
                    reset(&mut gpu, &outs)?;
                    launch(&mut gpu, &alt, kn, &px, &outs.y, 0, NMAX)?;
                    let got_h = read(&mut gpu, &outs)?;
                    detect(
                        &mut rep,
                        "header byte",
                        first_diff(&got_h, rk, ms).is_some(),
                    );
                    gpu.free_tensor(wcopy)?;
                    // (c) row offset: output batch row r vs reference row r+1.
                    reset(&mut gpu, &outs)?;
                    launch(&mut gpu, case, kn, &px, &outs.y, 0, NMAX)?;
                    let good = read(&mut gpu, &outs)?;
                    let shifted: Vec<Vec<u8>> = rk
                        .iter()
                        .zip(ms)
                        .map(|(r, &m)| {
                            let mut v = r.clone();
                            v.copy_within(m * 4..NMAX * m * 4, 0);
                            v
                        })
                        .collect();
                    detect(
                        &mut rep,
                        "row offset",
                        first_diff(&good, &shifted, ms).is_some(),
                    );
                    // (d) canary write and (e) padded-row write after an N=77 run.
                    let k77 = cand_kind(cand, 77);
                    reset(&mut gpu, &outs)?;
                    launch(&mut gpu, case, k77, &px, &outs.y, 0, 77)?;
                    let mut g77 = read(&mut gpu, &outs)?;
                    let w77 = expect(ref_for(k77), &outs.init, ms, 77);
                    let clean = first_diff(&g77, &w77, ms).is_none();
                    let last = g77[0].len() - 4;
                    g77[0][last] ^= 1;
                    detect(
                        &mut rep,
                        "canary",
                        clean && first_diff(&g77, &w77, ms).is_some(),
                    );
                    g77[0][last] ^= 1;
                    g77[0][77 * ms[0] * 4] ^= 1;
                    detect(
                        &mut rep,
                        "padded row",
                        clean && first_diff(&g77, &w77, ms).is_some(),
                    );
                    if kn == Kind::Prod {
                        // (f) out-of-domain N fails closed before any write.
                        for n in [0usize, NMAX + 1] {
                            reset(&mut gpu, &outs)?;
                            let refused =
                                launch(&mut gpu, case, Kind::Prod, &px, &outs.y, 0, n).is_err();
                            let untouched =
                                first_diff(&read(&mut gpu, &outs)?, &outs.init, ms).is_none();
                            detect(&mut rep, &format!("refuse N={n}"), refused && untouched);
                        }
                    }
                    rep.line(format!("{} {cand} {}", tag("neg"), det.join(" ")));
                }

                if has("graph") && args.mode != "eager" {
                    // Prewarm eagerly (load + scratch), capture one call through
                    // the GraphState capture path (blob kernargs), replay 3x:
                    // bytes stable and == reference.
                    let mut v = Vec::new();
                    let ns: &[usize] = if cand == "prod" {
                        &[128, 77, 65, 64, 33, 1]
                    } else {
                        &[128, 77]
                    };
                    for &n in ns {
                        if n > n_max {
                            continue;
                        }
                        let kind = cand_kind(cand, n);
                        let want = expect(ref_for(kind), &outs.init, ms, n);
                        let stream = gpu.hip.stream_create()?;
                        gpu.active_stream = Some(stream);
                        reset(&mut gpu, &outs)?;
                        launch(&mut gpu, case, kind, &px, &outs.y, 0, n)?;
                        let eager_ok = rep.cmp(
                            &tag(&format!("graph prewarm {cand} N={n}")),
                            &read(&mut gpu, &outs)?,
                            &want,
                            ms,
                        );
                        reset(&mut gpu, &outs)?;
                        gpu.invalidate_fp16_cache();
                        let dev = gpu.device_id;
                        gpu.graphs.begin_graph_capture(
                            &gpu.hip,
                            dev,
                            gpu.active_stream.as_ref().unwrap(),
                        )?;
                        let lr = launch(&mut gpu, case, kind, &px, &outs.y, 0, n);
                        let er = gpu.graphs.end_graph_capture(
                            &gpu.hip,
                            dev,
                            gpu.active_stream.as_ref().unwrap(),
                        );
                        lr?;
                        er?;
                        // Capture must not have executed anything.
                        let not_run = first_diff(&read(&mut gpu, &outs)?, &outs.init, ms).is_none();
                        rep.checks += 1;
                        if !not_run {
                            rep.fails += 1;
                            rep.line(format!(
                                "FAIL {}: Y changed during capture N={n}",
                                tag("graph")
                            ));
                        }
                        let mut okall = eager_ok && not_run;
                        for i in 0..3 {
                            reset(&mut gpu, &outs)?;
                            gpu.graphs.graph_launch(
                                &gpu.hip,
                                dev,
                                gpu.active_stream.as_ref().unwrap(),
                            )?;
                            okall &= rep.cmp(
                                &tag(&format!("graph replay{i} {cand} N={n}")),
                                &read(&mut gpu, &outs)?,
                                &want,
                                ms,
                            );
                        }
                        gpu.hip.device_synchronize()?;
                        gpu.graphs.drop_captured_graph(&gpu.hip, dev);
                        let s = gpu.active_stream.take().unwrap();
                        gpu.hip.stream_destroy(s)?;
                        v.push(format!("N={n}:{}", if okall { "3x same" } else { "DIFF" }));
                    }
                    rep.line(format!("{} {cand} {}", tag("graph"), v.join(" ")));
                }
            }
            free_outs(&mut gpu, outs)?;
        }

        if has("time") {
            // Event timing on the null stream, one launch set per sample.
            let outs = alloc_outs(&mut gpu, case, 1, false)?;
            load_x(&mut gpu, &gen_x(1, k))?;
            gpu.invalidate_fp16_cache();
            let e0 = gpu.hip.event_create()?;
            let e1 = gpu.hip.event_create()?;
            let time_set = |gpu: &mut Gpu, set: &[(Kind, usize, usize)]| -> R<(f64, f64)> {
                let mut ts = Vec::with_capacity(args.reps);
                for rep_i in 0..args.reps + 3 {
                    gpu.hip.device_synchronize()?;
                    gpu.hip.event_record(&e0, None)?;
                    for &(kind, r0, n) in set {
                        launch(gpu, case, kind, &px, &outs.y, r0, n)?;
                    }
                    gpu.hip.event_record(&e1, None)?;
                    gpu.hip.event_synchronize(&e1)?;
                    if rep_i >= 3 {
                        ts.push(gpu.hip.event_elapsed_ms(&e0, &e1)? as f64);
                    }
                }
                let min = ts.iter().cloned().fold(f64::INFINITY, f64::min);
                Ok((median(&mut ts), min))
            };
            let mut line = Vec::new();
            let vt = [
                (Kind::Vt(3, wr), 0, 48),
                (Kind::Vt(3, wr), 48, 48),
                (Kind::Vt(2, wr), 96, 32),
            ];
            let (m, mn) = time_set(&mut gpu, &vt)?;
            line.push(format!("vt3+vt3+vt2 48+48+32: med {m:.4} min {mn:.4} ms"));
            if has("prodtime") {
                // Production entry, fp16 X cached (kernel cost; the head case
                // through the residual entry, no memset).
                let (s_sel, _) = prod_symbol(case, 128, pm_sel);
                let (m, mn) = time_set(&mut gpu, &[(Kind::ProdKernel, 0, 128)])?;
                line.push(format!(
                    "prod[{}] {s_sel} 128: med {m:.4} min {mn:.4} ms",
                    family(pm_sel)
                ));
                let (s_oth, _) = prod_symbol(case, 128, !pm_sel);
                let (m, mn) = with_pm(&mut gpu, !pm_sel, |g| {
                    time_set(g, &[(Kind::ProdKernel, 0, 128)])
                })?;
                line.push(format!(
                    "prod[{}] {s_oth} 128 (flag-swapped, same process): med {m:.4} min {mn:.4} ms",
                    family(!pm_sel)
                ));
            }
            if has("k32") || !has("prodtime") {
                // The Gate-0 comparison sets.
                let sets: [(&str, Vec<(Kind, usize, usize)>); 5] = [
                    ("k32 spike bt8w4 128", vec![(Kind::K32(8, 4), 0, 128)]),
                    ("k32 spike bt8w8 128", vec![(Kind::K32(8, 8), 0, 128)]),
                    ("bt8 128", vec![(Kind::Bt(8), 0, 128)]),
                    ("single 128", vec![(Kind::Single, 0, 128)]),
                    (
                        "vt4 64+64",
                        vec![(Kind::Vt(4, wr), 0, 64), (Kind::Vt(4, wr), 64, 64)],
                    ),
                ];
                for (name, set) in &sets {
                    let (m, mn) = time_set(&mut gpu, set)?;
                    line.push(format!("{name}: med {m:.4} min {mn:.4} ms"));
                }
            }
            rep.line(format!(
                "time {} total_m={tm} k={k} W={wr} reps={} | {}",
                case.name,
                args.reps,
                line.join(" | ")
            ));
            free_outs(&mut gpu, outs)?;
        }
    }

    // Occupancy / loaded state of every wide symbol of both families (the
    // query resolves loaded functions by name; "not loaded" = never loaded here).
    for case in &cases {
        for n in [64usize, 128] {
            for pm in [pm_sel, !pm_sel] {
                let (sym, block) = prod_symbol(case, n, pm);
                match gpu.occupancy_max_active_blocks(&sym, [block, 1, 1], 0) {
                    Ok(o) => rep.line(format!(
                        "occupancy {} {sym} block={block} max_active_blocks_per_cu={o}",
                        family(pm)
                    )),
                    Err(_) => rep.line(format!("occupancy {} {sym}: not loaded", family(pm))),
                }
            }
        }
    }

    std::fs::write(format!("{}/digests.txt", args.out), &digests)?;
    let mut summary = String::new();
    let _ = writeln!(summary, "checks {} fails {}", rep.checks, rep.fails);
    rep.line(summary.trim_end().to_string());
    rep.line(if rep.fails == 0 {
        "VERDICT PASS".into()
    } else {
        "VERDICT FAIL".into()
    });
    std::fs::write(format!("{}/report.txt", args.out), &rep.text)?;
    let Panels { x16, x32 } = px;
    gpu.free_tensor(x16)?;
    gpu.free_tensor(x32)?;
    Ok(rep.fails)
}
