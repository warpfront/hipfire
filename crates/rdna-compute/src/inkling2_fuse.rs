// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 blocked-body glue fusion (gfx1201 decode and prefill rows).
//!
//! Every kernel here replaces a chain of row-wise glue launches of
//! `hipfire-arch-inkling2`'s blocked body with one launch whose per-element
//! arithmetic is the chain's, in the same order, so outputs are bitwise those
//! of the unfused chain:
//!
//! * [`INKLING2_FZ_MIX_NORM_ROT`]: AttnRes `mix()` (`ink_pf_mix_rows_f32`) or
//!   a plain input row, then up to [`FZ_TARGETS`] RMSNorms of that row (gamma:
//!   `rmsnorm_f32`; gamma-less: `inkling2_rmsnorm_nogamma_f32`), each feeding
//!   up to [`FZ_ROTS`] AWQ+FWHT256 producers (`rotate_x_mq_awq`). One block
//!   (256 threads) per (row, query row); the mix keeps the states in registers,
//!   so the bank is read once instead of three times and the per-state
//!   reductions share their barrier levels (each state's tree is unchanged).

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor, ReplayLaunchBindings};
use hip_bridge::{HipError, HipResult, KernargBlob};

/// Module of the fused glue kernels.
pub const INKLING2_FUSE_MODULE: &str = "inkling2_fuse";

/// Mix + norms + AWQ/FWHT producers ([`Gpu::inkling2_fz_mix_norm_rot`]).
pub const INKLING2_FZ_MIX_NORM_ROT: &str = "inkling2_fz_mix_norm_rot_f32";

/// Threads per block, and the lane stride of every row reduction.
pub const FZ_THREADS: usize = 256;
/// Elements per thread: rows up to `FZ_THREADS * FZ_ELEMS` (2304, Inkling-2 `n_embd`).
pub const FZ_ELEMS: usize = 9;
/// Mix states (`ink_pf_mix_rows_f32` bound).
pub const FZ_STATES: usize = 6;
/// Query rows (or plain inputs) per launch.
pub const FZ_QROWS: usize = 4;
/// Norm targets per query row.
pub const FZ_TARGETS: usize = 3;
/// AWQ+FWHT producers per norm target.
pub const FZ_ROTS: usize = 5;

/// HIP device function `fz_awq_fwht`, shared by the fused modules.
macro_rules! fz_awq_fwht_hip {
    () => {
        r#"
// One 256-element group of `rotate_x_mq_awq` given the lane's 8 consecutive
// inputs `x` (elements base..base+7): AWQ divide, signs1, 3 local + 5 wave
// butterflies, x 1/16, signs2. Same arithmetic as the standalone producer.
__device__ __forceinline__ void fz_awq_fwht(
    const float (&x)[8], const float* __restrict__ awq_scale, float* __restrict__ dst,
    const float* __restrict__ signs1, const float* __restrict__ signs2,
    int group, int tid)
{
    const int d0 = tid * 8;
    const int base = group * 256 + d0;
    float v0 = (x[0] / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (x[1] / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (x[2] / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (x[3] / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (x[4] / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (x[5] / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (x[6] / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (x[7] / awq_scale[base + 7]) * signs1[d0 + 7];
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)
    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)
    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY
    const float s = 0.0625f;
    dst[base    ] = v0 * s * signs2[d0    ];
    dst[base + 1] = v1 * s * signs2[d0 + 1];
    dst[base + 2] = v2 * s * signs2[d0 + 2];
    dst[base + 3] = v3 * s * signs2[d0 + 3];
    dst[base + 4] = v4 * s * signs2[d0 + 4];
    dst[base + 5] = v5 * s * signs2[d0 + 5];
    dst[base + 6] = v6 * s * signs2[d0 + 6];
    dst[base + 7] = v7 * s * signs2[d0 + 7];
}
"#
    };
}
pub(crate) use fz_awq_fwht_hip;

pub const INKLING2_FUSE_HIP_SRC: &str = concat!(
    r#"
#include <hip/hip_runtime.h>

#define FZ_T 256
#define FZ_J 9
#define FZ_S 6
#define FZ_Q 4
#define FZ_NT 3
#define FZ_NR 5

struct FzStates { const float* p[FZ_S]; };
// 8-byte fields only, so the host blob's natural alignment is the C layout.
struct FzTarget {
    const float* gamma;   // null: gamma-less norm
    float* out;           // normed row (nullable)
    long long out_ld;
    float eps; int pad;
    long long n_rot;
    const float* awq[FZ_NR];
    float* rot[FZ_NR];    // [rows, n] each
};
struct FzQRow {
    float* raw;           // mix output row (nullable)
    long long raw_ld;
    long long n_t;
    FzTarget t[FZ_NT];
};
struct FzQRows { FzQRow q[FZ_Q]; };

"#,
    fz_awq_fwht_hip!(),
    r#"

// `fz_awq_fwht` of group `group` of the row `src` (LDS).
__device__ __forceinline__ void fz_rot_group(
    const float* src, const float* __restrict__ awq_scale, float* __restrict__ dst,
    const float* __restrict__ signs1, const float* __restrict__ signs2,
    int group, int tid)
{
    const int base = group * 256 + tid * 8;
    float x[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) x[k] = src[base + k];
    fz_awq_fwht(x, awq_scale, dst, signs1, signs2, group, tid);
}

// Block (row b, query row q), 256 threads; thread t owns elements
// i = t + 256 j (the lane stride of every replaced reduction).
// n_states > 0: y = mix(st.p[0..n_states) row b, query row q), the singleton
//   `inkling2_rmsnorm_nogamma_f32` + `inkling2_mix_query_row_f32` arithmetic
//   (blockDim 256); n_states == 0: y = row b of st.p[q].
// Then: raw y (optional) and, for each target, the RMSNorm of y
// (gamma: (y * gamma) * rsqrtf(ss / n + eps); gamma-less: y * (1 / sqrtf(ss / n + eps)),
// ss the 256-lane tree of y*y) written to `out` and/or rotated per AWQ scale.
extern "C" __global__ __launch_bounds__(FZ_T)
void inkling2_fz_mix_norm_rot_f32(
    FzStates st, int n_states,
    const float* __restrict__ query,
    FzQRows qs,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int rows, int n, float inv_sqrt_n, float mix_eps)
{
    const int b = (int)blockIdx.x;
    const int q = (int)blockIdx.y;
    const int t = (int)threadIdx.x;
    if (b >= rows) return;
    __shared__ float sh[FZ_S][FZ_T];
    __shared__ float w[FZ_S];
    __shared__ float rowbuf[FZ_T * FZ_J];
    float y[FZ_J];
    if (n_states == 0) {
        const float* xr = st.p[q] + (size_t)b * n;
#pragma unroll
        for (int j = 0; j < FZ_J; ++j) {
            const int i = t + FZ_T * j;
            y[j] = i < n ? xr[i] : 0.0f;
        }
    } else {
        float x[FZ_S][FZ_J];
#pragma unroll
        for (int s = 0; s < FZ_S; ++s) {
            if (s < n_states) {
                const float* xr = st.p[s] + (size_t)b * n;
#pragma unroll
                for (int j = 0; j < FZ_J; ++j) {
                    const int i = t + FZ_T * j;
                    x[s][j] = i < n ? xr[i] : 0.0f;
                }
            }
        }
#pragma unroll
        for (int s = 0; s < FZ_S; ++s) {
            if (s < n_states) {
                float acc = 0.0f;
#pragma unroll
                for (int j = 0; j < FZ_J; ++j) {
                    if (t + FZ_T * j < n) {
                        const float v = x[s][j];
                        acc += v * v;
                    }
                }
                sh[s][t] = acc;
            }
        }
        __syncthreads();
        for (int s2 = FZ_T >> 1; s2 > 0; s2 >>= 1) {
            if (t < s2) {
#pragma unroll
                for (int s = 0; s < FZ_S; ++s) {
                    if (s < n_states) sh[s][t] += sh[s][t + s2];
                }
            }
            __syncthreads();
        }
        float inv[FZ_S];
#pragma unroll
        for (int s = 0; s < FZ_S; ++s) {
            inv[s] = 0.0f;
            if (s < n_states) inv[s] = 1.0f / sqrtf(sh[s][0] / (float)n + mix_eps);
        }
        __syncthreads();
        const float* qq = query + (size_t)q * n;
        float qv[FZ_J];
#pragma unroll
        for (int j = 0; j < FZ_J; ++j) {
            const int i = t + FZ_T * j;
            qv[j] = i < n ? qq[i] : 0.0f;
        }
#pragma unroll
        for (int s = 0; s < FZ_S; ++s) {
            if (s < n_states) {
                float acc = 0.0f;
#pragma unroll
                for (int j = 0; j < FZ_J; ++j) {
                    if (t + FZ_T * j < n) {
                        const float ns = x[s][j] * inv[s];
                        acc += ns * qv[j];
                    }
                }
                sh[s][t] = acc;
            }
        }
        __syncthreads();
        for (int s2 = FZ_T >> 1; s2 > 0; s2 >>= 1) {
            if (t < s2) {
#pragma unroll
                for (int s = 0; s < FZ_S; ++s) {
                    if (s < n_states) sh[s][t] += sh[s][t + s2];
                }
            }
            __syncthreads();
        }
        if (t == 0) {
            for (int s = 0; s < n_states; ++s) w[s] = sh[s][0] * inv_sqrt_n;
            float mx = w[0];
            for (int s = 1; s < n_states; ++s) {
                if (w[s] > mx) mx = w[s];
            }
            float sum = 0.0f;
            for (int s = 0; s < n_states; ++s) {
                w[s] = expf(w[s] - mx);
                sum += w[s];
            }
            for (int s = 0; s < n_states; ++s) w[s] /= sum;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < FZ_J; ++j) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < FZ_S; ++s) {
                if (s < n_states) {
                    const float ns = x[s][j] * inv[s];
                    acc += w[s] * ns;
                }
            }
            y[j] = acc;
        }
    }
    const FzQRow& oq = qs.q[q];
    if (oq.raw) {
        float* rr = oq.raw + (size_t)b * oq.raw_ld;
#pragma unroll
        for (int j = 0; j < FZ_J; ++j) {
            const int i = t + FZ_T * j;
            if (i < n) rr[i] = y[j];
        }
    }
    const int n_t = (int)oq.n_t;
    if (n_t == 0) return;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < FZ_J; ++j) {
        if (t + FZ_T * j < n) {
            const float v = y[j];
            acc += v * v;
        }
    }
    __syncthreads();
    sh[0][t] = acc;
    __syncthreads();
    for (int s2 = FZ_T >> 1; s2 > 0; s2 >>= 1) {
        if (t < s2) sh[0][t] += sh[0][t + s2];
        __syncthreads();
    }
    const float ss = sh[0][0];
    const int lane = t & 31;
    const int wave = t >> 5;
    const int groups = n / 256;
    for (int k = 0; k < n_t; ++k) {
        const FzTarget& tg = oq.t[k];
        float z[FZ_J];
        if (tg.gamma) {
            const float rms = rsqrtf(ss / (float)n + tg.eps);
#pragma unroll
            for (int j = 0; j < FZ_J; ++j) {
                const int i = t + FZ_T * j;
                z[j] = i < n ? y[j] * tg.gamma[i] * rms : 0.0f;
            }
        } else {
            const float inv = 1.0f / sqrtf(ss / (float)n + tg.eps);
#pragma unroll
            for (int j = 0; j < FZ_J; ++j) z[j] = y[j] * inv;
        }
        if (tg.out) {
            float* orow = tg.out + (size_t)b * tg.out_ld;
#pragma unroll
            for (int j = 0; j < FZ_J; ++j) {
                const int i = t + FZ_T * j;
                if (i < n) orow[i] = z[j];
            }
        }
        const int n_rot = (int)tg.n_rot;
        if (n_rot > 0) {
#pragma unroll
            for (int j = 0; j < FZ_J; ++j) {
                const int i = t + FZ_T * j;
                if (i < n) rowbuf[i] = z[j];
            }
            __syncthreads();
            for (int r = 0; r < n_rot; ++r) {
                float* dst = tg.rot[r] + (size_t)b * n;
                for (int g = wave; g < groups; g += FZ_T / 32) {
                    fz_rot_group(rowbuf, tg.awq[r], dst, signs1, signs2, g, lane);
                }
            }
            __syncthreads();
        }
    }
}
"#
);

/// One RMSNorm target of a [`FzRow`]: `gamma` (or gamma-less), the normed row
/// written to `out` (`[rows]` at row stride `ld` elements), and AWQ+FWHT256
/// producers `(awq_scale [n], x_rot [rows, n])`.
pub struct FzTarget<'a> {
    pub gamma: Option<&'a GpuTensor>,
    pub eps: f32,
    pub out: Option<(&'a GpuTensor, usize)>,
    pub rots: Vec<(&'a GpuTensor, &'a GpuTensor)>,
}

/// Outputs of one query row (or plain input): the raw row (`[rows]` at row
/// stride `ld`) and its norm targets.
#[derive(Default)]
pub struct FzRow<'a> {
    pub raw: Option<(&'a GpuTensor, usize)>,
    pub targets: Vec<FzTarget<'a>>,
}

/// [`Gpu::inkling2_fz_mix_norm_rot`] operands. `query: Some(q [qr, n])`:
/// mix over `states` (1..=6 `[rows, n]`, oldest first), `out.len() == qr`.
/// `None`: `out[q]` normalizes row-wise input `states[q]`.
pub struct Inkling2FzMixNormRot<'a> {
    pub states: &'a [&'a GpuTensor],
    pub query: Option<&'a GpuTensor>,
    pub out: &'a [FzRow<'a>],
    pub rows: usize,
    pub n: usize,
    pub mix_eps: f32,
}

impl Gpu {
    /// AttnRes mix (or plain rows) + RMSNorm targets + AWQ/FWHT producers in
    /// one launch, bitwise the unfused `ink_pf_mix_rows_f32` / `rmsnorm_f32` /
    /// `inkling2_rmsnorm_nogamma_f32` / `rotate_x_mq_awq` chain per row.
    pub fn inkling2_fz_mix_norm_rot(&mut self, a: &Inkling2FzMixNormRot<'_>) -> HipResult<()> {
        self.bind_thread()?;
        let bad = |m: String| Err(HipError::new(0, &format!("inkling2_fz_mix_norm_rot: {m}")));
        let (n, rows) = (a.n, a.rows);
        if n == 0 || n > FZ_THREADS * FZ_ELEMS || rows == 0 || a.out.is_empty() {
            return bad(format!("n {n} rows {rows} out {}", a.out.len()));
        }
        let qr = a.out.len();
        let f32_len = |t: &GpuTensor| t.buf.size() / 4;
        match a.query {
            Some(q) => {
                if !(1..=FZ_STATES).contains(&a.states.len()) || qr > FZ_QROWS {
                    return bad(format!("mix states {} / query rows {qr}", a.states.len()));
                }
                if f32_len(q) < qr * n {
                    return bad("query too small".into());
                }
            }
            None => {
                if a.states.len() != qr || qr > FZ_QROWS {
                    return bad(format!("inputs {} != rows out {qr}", a.states.len()));
                }
            }
        }
        if a.states.iter().any(|s| f32_len(s) < rows * n) {
            return bad("state/input too small".into());
        }
        let rows_ok = |t: &GpuTensor, ld: usize| ld >= n && f32_len(t) >= (rows - 1) * ld + n;
        let mut any_rot = false;
        for o in a.out {
            if o.targets.len() > FZ_TARGETS {
                return bad(format!("{} targets", o.targets.len()));
            }
            if o.raw.is_some_and(|(t, ld)| !rows_ok(t, ld)) {
                return bad("raw out too small".into());
            }
            for tg in &o.targets {
                if tg.rots.len() > FZ_ROTS
                    || tg.gamma.is_some_and(|g| f32_len(g) < n)
                    || tg.out.is_some_and(|(t, ld)| !rows_ok(t, ld))
                    || tg
                        .rots
                        .iter()
                        .any(|(s, x)| f32_len(s) < n || f32_len(x) < rows * n)
                {
                    return bad("target operand".into());
                }
                any_rot |= !tg.rots.is_empty();
            }
        }
        if any_rot && n % 256 != 0 {
            return bad(format!("rotation needs n % 256 == 0, n {n}"));
        }
        let (s1, s2) = if any_rot {
            self.ensure_mq_signs()?;
            let s1 = self.scratch.mq_signs1.as_ref().map(|t| t.buf.as_ptr());
            let s2 = self.scratch.mq_signs2.as_ref().map(|t| t.buf.as_ptr());
            match (s1, s2) {
                (Some(a), Some(b)) => (a as *const c_void, b as *const c_void),
                _ => return bad("no mq signs".into()),
            }
        } else {
            (std::ptr::null(), std::ptr::null())
        };
        self.ensure_kernel(
            INKLING2_FUSE_MODULE,
            INKLING2_FUSE_HIP_SRC,
            INKLING2_FZ_MIX_NORM_ROT,
        )?;
        let p = |t: &GpuTensor| t.buf.as_ptr() as *const c_void;
        let null = std::ptr::null::<c_void>();
        let mut b = KernargBlob::new();
        for i in 0..FZ_STATES {
            b.push_ptr(a.states.get(i).map_or(null, |t| p(t)));
        }
        b.push_i32(if a.query.is_some() {
            a.states.len() as i32
        } else {
            0
        });
        b.push_ptr(a.query.map_or(null, p));
        for qi in 0..FZ_QROWS {
            let o = a.out.get(qi);
            let raw = o.and_then(|o| o.raw);
            b.push_ptr(raw.map_or(null, |(t, _)| p(t)));
            b.push_u64(raw.map_or(0, |(_, ld)| ld as u64));
            b.push_u64(o.map_or(0, |o| o.targets.len() as u64));
            for ti in 0..FZ_TARGETS {
                let tg = o.and_then(|o| o.targets.get(ti));
                b.push_ptr(tg.and_then(|t| t.gamma).map_or(null, p));
                let out = tg.and_then(|t| t.out);
                b.push_ptr(out.map_or(null, |(t, _)| p(t)));
                b.push_u64(out.map_or(0, |(_, ld)| ld as u64));
                b.push_f32(tg.map_or(0.0, |t| t.eps));
                b.push_i32(0);
                b.push_u64(tg.map_or(0, |t| t.rots.len() as u64));
                for r in 0..FZ_ROTS {
                    b.push_ptr(tg.and_then(|t| t.rots.get(r)).map_or(null, |(s, _)| p(s)));
                }
                for r in 0..FZ_ROTS {
                    b.push_ptr(tg.and_then(|t| t.rots.get(r)).map_or(null, |(_, x)| p(x)));
                }
            }
        }
        b.push_ptr(s1);
        b.push_ptr(s2);
        b.push_i32(rows as i32);
        b.push_i32(n as i32);
        b.push_f32(1.0 / (n as f32).sqrt());
        b.push_f32(a.mix_eps);
        b.pad_to(16);
        let rows_u = u32::try_from(rows).map_err(|_| HipError::new(0, "rows exceed u32"))?;
        self.launch_blob_recorded(
            INKLING2_FZ_MIX_NORM_ROT,
            [rows_u, qr as u32, 1],
            [FZ_THREADS as u32, 1, 1],
            0,
            b.as_mut_slice(),
            ReplayLaunchBindings::NONE,
        )?;
        for o in a.out {
            for tg in &o.targets {
                for (_, x) in &tg.rots {
                    self.scratch.invalidate_x_caches_for(x.buf.as_ptr());
                }
            }
        }
        Ok(())
    }
}
