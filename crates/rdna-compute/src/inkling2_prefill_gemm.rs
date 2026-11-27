// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched multi-row projection GEMM for brief F1 (chunked prefill), kernel #1.
//!
//! The visit's four QKV/R projections (`wq`, `wk`, `wv`, `wr`) stream a full
//! weight matrix per single-row GEMV in 11's loop. This kernel dots B stacked
//! F32 activation rows against one MQ4G256V2Lloyd weight matrix in a single
//! launch, writing **F16 out from day one** (S1 folded into F1 per the
//! auditor's ruling — never F32-out-then-rewrite). Downstream batched-visit
//! kernels consume the F16 rows directly; the widening back to F32 happens
//! once at the visit output, not per projection.
//!
//! Bit-exactness contract: each (row, b) block runs the SAME per-row math as
//! the single-row `gemv_mq4g256v2_lloyd` kernel (same LUT staging, same
//! 4-accumulator interleave, same group/tail order, same shuffle reduction),
//! only the x vector (`x + b*K`) and the y address space (`[B x M]` F16)
//! differ. So the F16 output bits must equal the single-row F32 output cast
//! to F16 — the parity gate compares in F16 space, where `<= 1e-5` holds with
//! room to spare (expected: exact). Comparing F16-out against the F32 seam
//! directly cannot meet 1e-5 (fp16 rounding); that residual is inherent to S1,
//! not kernel error, and is covered end to end by the auditor's agreement bar
//! (argmax 31274, KL <= 0.005).

use std::ffi::c_void;

use hip_bridge::HipResult;

/// Inline HIP source: one kernel, no shared edits.
pub const INKLING2_PREFILL_GEMM_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batched MQ4G256V2-Lloyd GEMM, F16 out. Block (row, b) dots weight row
// against x row b. Per-row math is instruction-identical to
// gemv_mq4g256v2_lloyd (LUT path): same LDS codebook staging, same
// float4 x-hoist, same DOG/TAIL_DOG FMA order, same (acc0+acc1)+(acc2+acc3)
// combine and shuffle reduction. Only the store narrows to F16.
__launch_bounds__(32, 16)
extern "C" __global__ void inkling2_prefill_gemm_mq4v2_lloyd_f16out(
    const char* __restrict__ A,
    const float* __restrict__ x, // [B x K] row-major F32
    __half* __restrict__ y,      // [B x M] row-major F16
    int M, int K, int B,
    const unsigned int LUT0, const unsigned int LUT1,
    const unsigned int LUT2, const unsigned int LUT3,
    const unsigned int LUT4, const unsigned int LUT5,
    const unsigned int LUT6, const unsigned int LUT7)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int b = blockIdx.y;
    if (b >= B) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const unsigned int row_weight_offset =
        (unsigned int)((long long)row * groups_per_row * 136);
    // Centered-codebook LUT: 16 f16 levels in 8 dwords (2 per dword, LE).
    // Same staging as the scalar kernel: lanes 0..7 expand one dword each
    // to two disjoint float slots, then barrier. f16->f32 is exact, so the
    // FMA chain that follows is bit-identical.
    __shared__ float LUT_SMEM[16];
    {
        unsigned int d;
        if (tid == 0) d = LUT0; else if (tid == 1) d = LUT1;
        else if (tid == 2) d = LUT2; else if (tid == 3) d = LUT3;
        else if (tid == 4) d = LUT4; else if (tid == 5) d = LUT5;
        else if (tid == 6) d = LUT6; else d = LUT7;
        if (tid < 8) {
            LUT_SMEM[tid * 2] = __half2float(__ushort_as_half((unsigned short)(d & 0xFFFFu)));
            LUT_SMEM[tid * 2 + 1] = __half2float(__ushort_as_half((unsigned short)(d >> 16)));
        }
    }
    __syncthreads();
#define MQ4V2_LUT(q) (LUT_SMEM[(q)])
    const float* xb = x + (long long)b * K;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        const unsigned int goff = row_weight_offset + (unsigned int)g * 136u;
        const unsigned int hA0 = *(const unsigned int*)(gp0);
        const unsigned int hB0 = *(const unsigned int*)(gp0 + 4);
        const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
        const unsigned int hA1 = *(const unsigned int*)(gp1);
        const unsigned int hB1 = *(const unsigned int*)(gp1 + 4);
        const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
        const unsigned int hA2 = *(const unsigned int*)(gp2);
        const unsigned int hB2 = *(const unsigned int*)(gp2 + 4);
        const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
        const unsigned int hA3 = *(const unsigned int*)(gp3);
        const unsigned int hB3 = *(const unsigned int*)(gp3 + 4);
        const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + tid * 8;

        // Plain float4 x-hoist, same values and FMA order as scalar loads:
        // bit-exact. Safe: K % 256 == 0 so every row starts 16B-aligned,
        // and base advances by 4 floats.
        const float4 xv0a = *reinterpret_cast<const float4*>(xb + base);
        const float4 xv0b = *reinterpret_cast<const float4*>(xb + base + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(xb + base + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(xb + base + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(xb + base + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(xb + base + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(xb + base + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(xb + base + 772);
        #define DOG_X_LUT(pk, sc, zp, a, x0, x1, x2, x3, x4, x5, x6, x7) \
            (a) += (sc * MQ4V2_LUT((pk) & 0xFu)        + zp) * (x0) \
                 + (sc * MQ4V2_LUT(((pk) >> 4) & 0xFu)  + zp) * (x1) \
                 + (sc * MQ4V2_LUT(((pk) >> 8) & 0xFu)  + zp) * (x2) \
                 + (sc * MQ4V2_LUT(((pk) >> 12) & 0xFu) + zp) * (x3) \
                 + (sc * MQ4V2_LUT(((pk) >> 16) & 0xFu) + zp) * (x4) \
                 + (sc * MQ4V2_LUT(((pk) >> 20) & 0xFu) + zp) * (x5) \
                 + (sc * MQ4V2_LUT(((pk) >> 24) & 0xFu) + zp) * (x6) \
                 + (sc * MQ4V2_LUT(((pk) >> 28) & 0xFu) + zp) * (x7)
        DOG_X_LUT(pk0, sc0, zp0, acc0,
                  xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X_LUT(pk1, sc1, zp1, acc1,
                  xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X_LUT(pk2, sc2, zp2, acc2,
                  xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X_LUT(pk3, sc3, zp3, acc3,
                  xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
        #undef DOG_X_LUT
    }

    // Tail groups accumulate into their own accumulator (acc[g % 4]),
    // same as the scalar kernel.
    #define TAIL_DOG_LUT(pk, sc, zp, b_, a) \
        (a) += (sc * MQ4V2_LUT((pk) & 0xFu)        + zp) * xb[(b_)]     \
             + (sc * MQ4V2_LUT(((pk) >> 4) & 0xFu)  + zp) * xb[(b_) + 1] \
             + (sc * MQ4V2_LUT(((pk) >> 8) & 0xFu)  + zp) * xb[(b_) + 2] \
             + (sc * MQ4V2_LUT(((pk) >> 12) & 0xFu) + zp) * xb[(b_) + 3] \
             + (sc * MQ4V2_LUT(((pk) >> 16) & 0xFu) + zp) * xb[(b_) + 4] \
             + (sc * MQ4V2_LUT(((pk) >> 20) & 0xFu) + zp) * xb[(b_) + 5] \
             + (sc * MQ4V2_LUT(((pk) >> 24) & 0xFu) + zp) * xb[(b_) + 6] \
             + (sc * MQ4V2_LUT(((pk) >> 28) & 0xFu) + zp) * xb[(b_) + 7]
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 136u;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG_LUT(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 136u;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG_LUT(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 136u;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG_LUT(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG_LUT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    // S1: narrow to F16 at the store. Downstream batched kernels read F16.
    if (tid == 0) y[(long long)b * M + row] = __float2half_rn(acc);
}
"#;

impl crate::dispatch::Gpu {
    /// Batched MQ4G256V2-Lloyd GEMM with F16 output (F1 kernel #1).
    ///
    /// `a` is the packed `[M, K]` weight buffer (`K % 256 == 0`,
    /// `groups_per_row * 136` bytes per row), `x` is `[B, K]` F32 stacked
    /// rows, `y` is `[B, M]` F16. `lut_f16` packs the 16 centered f16 codebook
    /// levels as 8 dwords (same ABI as `gemv_mq4g256v2_lloyd`). Fails closed
    /// on bad shapes so a misshapen block never runs.
    ///
    /// SYSTEM CONTRACT (2026-11-25, learned the hard way): `x` must carry
    /// the FWHT-rotated activations (`rotate_x_mq_batched_for` with the
    /// consuming weight first — same as `weight_gemv` rotates internally).
    /// The offline weights are FWHT-rotated; feeding raw `x` is silent
    /// fluent corruption (KL ~14 end to end), not a fault: no shape check
    /// can catch it, and kernel-vs-kernel parity gates pass either way
    /// because both arms share the same (un)rotated input.
    pub fn inkling2_prefill_gemm_mq4v2_lloyd_f16out(
        &mut self,
        a_raw: &crate::dispatch::GpuTensor,
        x: &crate::dispatch::GpuTensor,
        y: &crate::dispatch::GpuTensor,
        m: usize,
        k: usize,
        b: usize,
        lut_f16: [u32; 8],
    ) -> HipResult<()> {
        use crate::dispatch::DType;
        self.bind_thread()?;
        if k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("inkling2 prefill gemm: K divisible by 256 required (got {k})"),
            ));
        }
        if b == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "inkling2 prefill gemm: batch rows B must be > 0",
            ));
        }
        if x.dtype != DType::F32 || x.shape != vec![b, k] {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 prefill gemm: x must be F32 [B,K]=[{b},{k}], got {:?} {:?}",
                    x.dtype, x.shape
                ),
            ));
        }
        if y.dtype != DType::F16 || y.shape != vec![b, m] {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 prefill gemm: y must be F16 [B,M]=[{b},{m}], got {:?} {:?}",
                    y.dtype, y.shape
                ),
            ));
        }
        let expect_bytes = m
            .checked_mul(k / 256)
            .and_then(|g| g.checked_mul(136))
            .ok_or_else(|| hip_bridge::HipError::new(0, "inkling2 prefill gemm: size overflow"))?;
        if a_raw.buf.size() < expect_bytes {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 prefill gemm: weight buffer {} B < {expect_bytes} B for [{m},{k}]",
                    a_raw.buf.size()
                ),
            ));
        }
        let func_name = "inkling2_prefill_gemm_mq4v2_lloyd_f16out";
        self.ensure_kernel(
            "inkling2_prefill_gemm",
            INKLING2_PREFILL_GEMM_SRC,
            func_name,
        )?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut b_val = b as i32;
        let mut l = lut_f16;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut b_val as *mut _ as *mut c_void,
            &mut l[0] as *mut _ as *mut c_void,
            &mut l[1] as *mut _ as *mut c_void,
            &mut l[2] as *mut _ as *mut c_void,
            &mut l[3] as *mut _ as *mut c_void,
            &mut l[4] as *mut _ as *mut c_void,
            &mut l[5] as *mut _ as *mut c_void,
            &mut l[6] as *mut _ as *mut c_void,
            &mut l[7] as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            func_name,
            [m as u32, b as u32, 1],
            [32, 1, 1],
            16 * 4,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(m_val);
                blob.push_i32(k_val);
                blob.push_i32(b_val);
                for v in l {
                    blob.push_u32(v);
                }
                blob
            },
        )
    }
}

// ---------------------------------------------------------------------------
// F1 wiring primitives (widen-step): the batched GEMM above writes F16 so
// the downstream visit stays F32 via one exact widen per projection output,
// and the blocked driver resets the D1 device pos word once per visit.
// ---------------------------------------------------------------------------

/// Identity codebook for uniform MQ4G256V2 weights (qt44, the dense
/// artifact's dtype): levels 0.0..15.0, packed 2-per-dword LE — the same
/// packing the Lloyd path uses. Hardcoded f16 bits (`half` is a dev-only
/// dependency; the unit gate below asserts each word decodes to its index):
/// 0x0000,0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,
/// 0x4800,0x4880,0x4900,0x4980,0x4A00,0x4A80,0x4B00,0x4B80.
/// With this LUT the batched kernel above is operation-for-operation the
/// uniform `gemv_mq4g256v2` kernel (same header decode, same interleave,
/// same combine; `sc * f16(q)` is bitwise `sc * (float)q` since q < 16 is
/// exact in f16), so Lloyd and uniform weights share one kernel. Gated
/// bit-exact below.
pub fn inkling2_identity_lut() -> [u32; 8] {
    [
        0x3C00_0000,
        0x4200_4000,
        0x4500_4400,
        0x4700_4600,
        0x4880_4800,
        0x4980_4900,
        0x4A80_4A00,
        0x4B80_4B00,
    ]
}

/// Inline HIP source: exact F16 -> F32 widen + device pos-word reset.
pub const INKLING2_PREFILL_UTIL_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Elementwise widen: every f16 value is exact in f32, so this is bitwise
// the host `half -> f32` cast. One launch covers a whole [B, M] projection
// output; grid is sized by the caller.
__launch_bounds__(256, 4)
extern "C" __global__ void inkling2_widen_f16_f32(
    const __half* __restrict__ x,
    float* __restrict__ y,
    int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    y[i] = __half2float(x[i]);
}

// Absolute device pos-word write for the blocked driver: each visit sets
// the word to the block's first position, then `inkling2_gf_advance_pos`
// steps per row exactly like the single-row token loop, so every row's
// slots kernel reads the same word the seam would. 1 thread, 1 launch
// per visit, plus the per-row advances.
__launch_bounds__(32, 16)
extern "C" __global__ void inkling2_gf_set_pos(
    unsigned int* __restrict__ pos_dev,
    unsigned int pos)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *pos_dev = pos;
}
"#;

impl crate::dispatch::Gpu {
    /// Exact F16 -> F32 widen of `n` elements (`x` F16 `[n]`, `y` F32 `[n]`).
    /// Fails closed on dtype/shape mismatch.
    pub fn inkling2_widen_f16_f32(
        &mut self,
        x: &crate::dispatch::GpuTensor,
        y: &crate::dispatch::GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        use crate::dispatch::DType;
        self.bind_thread()?;
        if n == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "inkling2 widen: n must be > 0",
            ));
        }
        if x.dtype != DType::F16 || x.shape.iter().product::<usize>() != n {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 widen: x must be F16 [n={n}], got {:?} {:?}",
                    x.dtype, x.shape
                ),
            ));
        }
        if y.dtype != DType::F32 || y.shape.iter().product::<usize>() != n {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 widen: y must be F32 [n={n}], got {:?} {:?}",
                    y.dtype, y.shape
                ),
            ));
        }
        let func_name = "inkling2_widen_f16_f32";
        self.ensure_kernel(
            "inkling2_prefill_util",
            INKLING2_PREFILL_UTIL_SRC,
            func_name,
        )?;
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            func_name,
            [n.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(n_val);
                blob
            },
        )
    }

    /// Write `pos` to the D1 device pos word (blocked-driver visit reset).
    pub fn inkling2_gf_set_pos(
        &mut self,
        pos_dev: &crate::dispatch::GpuTensor,
        pos: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let func_name = "inkling2_gf_set_pos";
        self.ensure_kernel(
            "inkling2_prefill_util",
            INKLING2_PREFILL_UTIL_SRC,
            func_name,
        )?;
        let mut p_ptr = pos_dev.buf.as_ptr() as *mut u32;
        let mut pos_val = pos as u32;
        let mut params: Vec<*mut c_void> = vec![
            &mut p_ptr as *mut _ as *mut c_void,
            &mut pos_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(func_name, [1, 1, 1], [32, 1, 1], 0, &mut params, || {
            let mut blob = hip_bridge::KernargBlob::new();
            blob.push_ptr(p_ptr as *const c_void);
            blob.push_u32(pos_val);
            blob
        })
    }
}

#[cfg(test)]
mod gpu_tests {
    use crate::dispatch::DType;
    use crate::inkling2_attn_prefill::init_inkling2_gpu;

    fn rng_next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn rng_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let u = (rng_next(state) >> 11) as f32 / (1u64 << 53) as f32;
        lo + u * (hi - lo)
    }

    /// Synthetic MQ4G256V2 weight row: per 256-group, 8B header (two u32;
    /// low half-group scale/zp in the first, high half-group in the second;
    /// each u32 packs sc f16 low, zp f16 high) + 128B packed nibbles.
    fn synth_group(state: &mut u64, sc_lo: f32, sc_hi: f32) -> Vec<u8> {
        let sc_lo_b = half::f16::from_f32(sc_lo).to_bits();
        let sc_hi_b = half::f16::from_f32(sc_hi).to_bits();
        let zp_b = half::f16::from_f32(0.0).to_bits();
        let mut out = Vec::with_capacity(136);
        for sc_b in [sc_lo_b, sc_hi_b] {
            let h = (sc_b as u32) | ((zp_b as u32) << 16);
            out.extend_from_slice(&h.to_le_bytes());
        }
        for _ in 0..32 {
            // Two nibbles per byte via two rng draws; any 0..15 pattern valid.
            let mut w = 0u32;
            for n in 0..8 {
                w |= ((rng_next(state) & 0xF) as u32) << (4 * n);
            }
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Gate: batched F16-out GEMM == per-row single-row GEMV cast to F16,
    /// bitwise, at several (M, K, B) including every tail path
    /// (K=512 tail 2, K=1024 tail 0, K=2304 tail 1) and B=1/5/64.
    /// Bar: 0 differing F16 words (hence max|d| <= 1e-5 holds trivially).
    #[test]
    #[ignore = "requires gfx1100/gfx1151 and the HIP toolchain"]
    fn gpu_batched_matches_per_row_lloyd() {
        let Some(mut gpu) = init_inkling2_gpu(&["gfx1100", "gfx1151"]) else {
            return;
        };
        // Centered codebook levels (i - 7.5), same role as the artifact LUT.
        let levels: Vec<u16> = (0..16)
            .map(|i| half::f16::from_f32(i as f32 - 7.5).to_bits())
            .collect();
        let mut lut = [0u32; 8];
        for i in 0..8 {
            lut[i] = (levels[2 * i] as u32) | ((levels[2 * i + 1] as u32) << 16);
        }
        let mut s = 0xF1e0_7ac0_1190u64;
        for (m, k) in [(256usize, 512usize), (128, 1024), (128, 2304)] {
            for b in [1usize, 5, 64] {
                // Weight buffer: m rows x (k/256 groups) x 136 B.
                let mut wbytes = Vec::with_capacity(m * (k / 256) * 136);
                for _ in 0..m * (k / 256) {
                    wbytes.extend_from_slice(&synth_group(&mut s, 0.5, 2.0));
                }
                let a = gpu
                    .upload_raw(&wbytes, &[m * (k / 256) * 136])
                    .expect("weight upload");
                // Stacked inputs [B, K].
                let xh: Vec<f32> = (0..b * k).map(|_| rng_f32(&mut s, -2.0, 2.0)).collect();
                let x = gpu.upload_f32(&xh, &[b, k]).expect("x upload");
                let y = gpu.alloc_tensor(&[b, m], DType::F16).expect("y alloc");
                gpu.inkling2_prefill_gemm_mq4v2_lloyd_f16out(&a, &x, &y, m, k, b, lut)
                    .expect("batched gemm");
                let got_bits = gpu.download_f16_bits(&y).expect("y download");
                assert_eq!(got_bits.len(), b * m);
                // Reference: one single-row GEMV per batch row, cast to F16.
                let mut diff_words = 0usize;
                let mut max_abs_f32 = 0.0f32;
                for row in 0..b {
                    let xr = gpu
                        .upload_f32(&xh[row * k..(row + 1) * k], &[k])
                        .expect("row upload");
                    let yr = gpu.alloc_tensor(&[m], DType::F32).expect("row alloc");
                    gpu.gemv_mq4g256v2_lloyd(&a, &xr, &yr, m, k, lut)
                        .expect("scalar gemv");
                    let ref_f32 = gpu.download_f32(&yr).expect("row download");
                    for (j, &v) in ref_f32.iter().enumerate() {
                        let want = half::f16::from_f32(v).to_bits();
                        let got = got_bits[row * m + j];
                        if got != want {
                            diff_words += 1;
                        }
                        let d = (half::f16::from_bits(got).to_f32() - v).abs();
                        if d > max_abs_f32 {
                            max_abs_f32 = d;
                        }
                    }
                }
                eprintln!(
                    "m={m} k={k} b={b}: diff_f16_words={diff_words}/{} max|f16out-f32seam|={max_abs_f32:e}",
                    b * m
                );
                assert_eq!(
                    diff_words, 0,
                    "m={m} k={k} b={b}: batched F16 must equal per-row F32 cast to F16 bitwise"
                );
            }
        }
    }

    /// Gate: widen is bitwise the host f16 -> f32 cast (lossless by format).
    #[test]
    #[ignore = "requires gfx1100/gfx1151 and the HIP toolchain"]
    fn gpu_widen_matches_host_cast() {
        let Some(mut gpu) = init_inkling2_gpu(&["gfx1100", "gfx1151"]) else {
            return;
        };
        let mut s = 0x91d3_7e57u64;
        let n = 8192;
        let fh: Vec<u16> = (0..n).map(|_| (rng_next(&mut s) & 0xFFFF) as u16).collect();
        let x = gpu.upload_f16_bits(&fh, &[n]).expect("f16 upload");
        let y = gpu.alloc_tensor(&[n], DType::F32).expect("f32 alloc");
        gpu.inkling2_widen_f16_f32(&x, &y, n).expect("widen");
        let got = gpu.download_f32(&y).expect("download");
        assert_eq!(got.len(), n);
        for (i, &g) in got.iter().enumerate() {
            let want = half::f16::from_bits(fh[i]).to_f32();
            assert!(
                g.to_bits() == want.to_bits(),
                "widen differs at {i}: {g} vs {want}"
            );
        }
        eprintln!("widen: {n}/{n} words bitwise");
    }

    /// Gate: set_pos writes the device word exactly (blocked-visit reset).
    #[test]
    #[ignore = "requires gfx1100/gfx1151 and the HIP toolchain"]
    fn gpu_set_pos_roundtrip() {
        let Some(mut gpu) = init_inkling2_gpu(&["gfx1100", "gfx1151"]) else {
            return;
        };
        let w = gpu.alloc_tensor(&[1], DType::F32).expect("word alloc");
        for pos in [0usize, 1, 511, 1_000_000] {
            gpu.inkling2_gf_set_pos(&w, pos).expect("set_pos");
            let got = gpu.download_f32(&w).expect("download");
            assert_eq!(got[0].to_bits(), pos as u32, "pos word mismatch at {pos}");
        }
        eprintln!("set_pos: 4/4 roundtrips exact");
    }

    /// Gate: batched GEMM with the identity codebook == per-row UNIFORM
    /// `gemv_mq4g256v2` cast to F16, bitwise. This is the artifact's dense
    /// dtype (qt44 uniform; the dense loader attaches no Lloyd LUT), so the
    /// wiring serves real weights through this arm. Same (M, K, B) cells as
    /// the Lloyd gate, including every tail path and multi-block grids.
    #[test]
    #[ignore = "requires gfx1100/gfx1151 and the HIP toolchain"]
    fn gpu_batched_matches_per_row_uniform() {
        let Some(mut gpu) = init_inkling2_gpu(&["gfx1100", "gfx1151"]) else {
            return;
        };
        let lut = super::inkling2_identity_lut();
        // Identity check on the host: LUT words decode to 0.0..15.0.
        for q in 0..16u32 {
            let d = lut[(q / 2) as usize];
            let bits = if q % 2 == 0 {
                (d & 0xFFFF) as u16
            } else {
                (d >> 16) as u16
            };
            assert_eq!(
                half::f16::from_bits(bits).to_f32(),
                q as f32,
                "identity LUT word {q}"
            );
        }
        let mut s = 0x071f_0ac0de_u64;
        for (m, k) in [(256usize, 512usize), (128, 1024), (128, 2304)] {
            for b in [1usize, 5, 64] {
                let mut wbytes = Vec::with_capacity(m * (k / 256) * 136);
                for _ in 0..m * (k / 256) {
                    wbytes.extend_from_slice(&synth_group(&mut s, 0.5, 2.0));
                }
                let a = gpu
                    .upload_raw(&wbytes, &[m * (k / 256) * 136])
                    .expect("weight upload");
                let xh: Vec<f32> = (0..b * k).map(|_| rng_f32(&mut s, -2.0, 2.0)).collect();
                // Batched arm on the stacked rows.
                let x = gpu.upload_f32(&xh, &[b, k]).expect("x upload");
                let y = gpu.alloc_tensor(&[b, m], DType::F16).expect("y alloc");
                gpu.inkling2_prefill_gemm_mq4v2_lloyd_f16out(&a, &x, &y, m, k, b, lut)
                    .expect("batched gemm");
                let got_bits = gpu.download_f16_bits(&y).expect("y download");
                assert_eq!(got_bits.len(), b * m);
                // Reference: one UNIFORM gemv per batch row, cast to F16.
                let mut diff_words = 0usize;
                for row in 0..b {
                    let xr = gpu
                        .upload_f32(&xh[row * k..(row + 1) * k], &[k])
                        .expect("row upload");
                    let yr = gpu.alloc_tensor(&[m], DType::F32).expect("row alloc");
                    gpu.gemv_mq4g256v2(&a, &xr, &yr, m, k)
                        .expect("uniform gemv");
                    let ref_f32 = gpu.download_f32(&yr).expect("row download");
                    for (j, &v) in ref_f32.iter().enumerate() {
                        let want = half::f16::from_f32(v).to_bits();
                        if got_bits[row * m + j] != want {
                            diff_words += 1;
                        }
                    }
                }
                eprintln!(
                    "uniform m={m} k={k} b={b}: diff_f16_words={diff_words}/{}",
                    b * m
                );
                assert_eq!(
                    diff_words, 0,
                    "m={m} k={k} b={b}: identity-LUT batched GEMM must equal uniform GEMV bitwise"
                );
            }
        }
    }
}
