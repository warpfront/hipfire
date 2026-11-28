// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 blocked (multi-row) causal prefill attention, FA2 (gfx1201 only).
//!
//! The prefill producer of the split-KV decode kernel in
//! [`crate::inkling2_attention`]: fragment-major LDS planes, swapped-operand
//! WMMA and hi/lo f16 Q and P, but one workgroup owns a 64-row query tile of
//! one KV head (twelve waves: three Q heads x four 16-row subtiles) and walks
//! the causal key range once, so there is no partition merge. Shape
//! H24/KV8/D128 (GQA3).
//!
//! Schedule (AMD-KERNELS §4 exact items): query tiles dispatch longest-first;
//! 32-key K/V tiles are double-buffered in LDS with the next-but-one tile
//! prefetched into registers (one barrier per tile); V is transposed into its
//! fragment layout from 4-key register blocks with 8-byte stores; the
//! relative table is a 128-column LDS ring that takes 32 new columns per
//! tile; interior 16-key subtiles (every key visible to every row and inside
//! the extent) skip the mask and extent tests, subtiles wholly past the extent
//! skip the relative dot, and the `O` rescale is skipped when no row's max
//! moved; `1/l` is applied once at the end. The online-softmax update still
//! runs per 16-key subtile with the same expressions, on a 16-key grid set by
//! the tile's first key `max(0, pos0 + q0 - window + 1)`. That grid is the
//! former 32-row kernel's whenever no 64-row tile straddles the SWA clamp at
//! key 0 (with the 1024-key window: always when `pos0 % 64 == 0`, so every
//! 512-row prefill block), and
//! there the output is bitwise the former kernel's; otherwise only the
//! rounding order differs (within the f64-reference budget).
//!
//! Score contract (identical to the decode route on the same operands):
//! `dot(Q_h,K_j)/sqrt(128) + (sum_r table[r,d] * R[h,r] - sink[h])` with
//! `d = pos - j`; the relative term is zero for `d >= n_extent` (upstream
//! zero-term, not a clamp) and is computed in-tile from a staged table
//! window, so no `[rows, heads, keys]` bias slab exists. The sink is beta's
//! synthetic zero-score, zero-value key: the online state starts at
//! `m = 0, l = 1, O = 0`.
//!
//! Key addressing: logical key `j` of the block starting at `pos0` reads the
//! block's own F16 staging planes when `j >= pos0` and the visit's KV cache
//! slot `j % ring_rows` otherwise. The caller writes the block into the cache
//! only after this kernel, so a block never clobbers a pre-block key an early
//! query still reads (SWA ring). Query row `i` sees `[max(0, pos - window + 1),
//! pos]` (`window = 0`: full causal).
//!
//! Scratch: none. Output `out [rows, 24, 128]` F32.

pub const INKLING2_ATTN_PREFILL_MODULE: &str = "inkling2_attention_prefill";

pub const INKLING2_ATTN_PREFILL_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

#define IH 24
#define IKV 8
#define ID 128
#define IGQA 3
#define NQT 4
#define QROWS (16 * NQT)
#define NT (32 * IGQA * NQT)
#define KT 32
#define NREL 16
#define TBR 128   // relative-table ring columns: >= QROWS + 2 * KT - 1, a power of two
#define TBS 20    // ring column stride in floats: 16 channels + 4 pad (conflict-free b128 reads)
#define VTH 128   // threads staging V (4 keys x 8 channels each)
#define KPER (512 / (NT - VTH))
#define NPF (KPER > 4 ? KPER : 4)
#define TPER ((NREL * KT + NT - 1) / NT)

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(4))) half4_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef float __attribute__((ext_vector_type(4))) float4_t;

struct KvPlanes {
    const unsigned short* p[8];
};

__device__ __forceinline__ int ip_kfrag(int sub, int dc, int lane)
{
    return (dc * 2 + sub) * 32 + lane;
}
__device__ __forceinline__ int ip_vfrag(int sub, int dc, int lane)
{
    return (sub * 8 + dc) * 32 + lane;
}

__device__ __forceinline__ half8_t ip_load(const unsigned short* sb, const unsigned short* cb,
    int j, int jhi, int c, int pos0, int ring_rows)
{
    half8_t x = {0, 0, 0, 0, 0, 0, 0, 0};
    if (j <= jhi) {
        if (j >= pos0)
            x = *(const half8_t*)(sb + (size_t)(j - pos0) * ID + (size_t)c * 8);
        else
            x = *(const half8_t*)(cb + (size_t)(j % ring_rows) * ID + (size_t)c * 8);
    }
    return x;
}

// Relative-term dot for distance d (caller guarantees 0 <= d < n_extent).
__device__ __forceinline__ float ip_rel(const float* TB, int d, const float* R)
{
    const float4_t* t = (const float4_t*)(TB + (d & (TBR - 1)) * TBS);
    float v = 0.0f;
#pragma unroll
    for (int q4 = 0; q4 < 4; ++q4) {
        const float4_t w = t[q4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v += w[i] * R[q4 * 4 + i];
    }
    return v;
}

// Grid (ceil(rows / 64), 8), block 384: wave w owns Q head 3*kv_h + w%3 over
// query subtile w/3 (16 rows).
extern "C" __global__ __launch_bounds__(NT, 1) void inkling2_attn_fa2_prefill_gfx1201(
    const float* __restrict__ q,
    const float* __restrict__ rel,
    const float* __restrict__ sinks,
    const float* __restrict__ table,
    KvPlanes kc,
    KvPlanes vc,
    KvPlanes ks,
    KvPlanes vs,
    float* __restrict__ out,
    int n_rows,
    int pos0,
    int window,
    int ring_rows,
    int n_extent,
    float inv_sqrt_d)
{
    const int kv_h = (int)blockIdx.y;
    // Longest-first: the last query tile has the longest causal key range.
    const int q0 = ((int)gridDim.x - 1 - (int)blockIdx.x) * QROWS;
    if (kv_h >= IKV || q0 >= n_rows)
        return;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const int hsub = wave % IGQA;
    const int qt = wave / IGQA;
    const int h = IGQA * kv_h + hsub;
    const int qi = q0 + qt * 16 + ml;
    const bool qlive = qi < n_rows;
    const int qpos = pos0 + qi;
    const int qlo = window > 0 ? max(0, qpos - window + 1) : 0;

    __shared__ half8_t KV8[2][1024];          // per buffer: K frags [0,512), V frags [512,1024)
    __shared__ float TB[TBR * TBS];           // relative-table ring: column d at (d & 127) * 20

    const unsigned short* kcb = kc.p[kv_h];
    const unsigned short* vcb = vc.p[kv_h];
    const unsigned short* ksb = ks.p[kv_h];
    const unsigned short* vsb = vs.p[kv_h];

    half8_t qhi[8];
    half8_t qlw[8];
#pragma unroll
    for (int dc = 0; dc < 8; ++dc) {
        half8_t f = {0, 0, 0, 0, 0, 0, 0, 0};
        half8_t g = {0, 0, 0, 0, 0, 0, 0, 0};
        if (qlive) {
            const float* qr = q + ((size_t)qi * IH + h) * ID + dc * 16 + kg * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                f[j] = (_Float16)qr[j];
                g[j] = (_Float16)(qr[j] - (float)f[j]);
            }
        }
        qhi[dc] = f;
        qlw[dc] = g;
    }
    float R[NREL];
    float sink = 0.0f;
#pragma unroll
    for (int r = 0; r < NREL; ++r)
        R[r] = qlive ? rel[((size_t)qi * IH + h) * NREL + r] : 0.0f;
    if (qlive)
        sink = sinks[(size_t)qi * IH + h];

    const int q_last = min(q0 + QROWS, n_rows) - 1;
    const int p_first = pos0 + q0;
    const int jlo = window > 0 ? max(0, p_first - window + 1) : 0;
    const int jhi = pos0 + q_last;
    const int ntiles = (jhi - jlo) / KT + 1;

    // Wave-uniform live query range (subtile skips and the unmasked fast path).
    const int wq_first = q0 + qt * 16;
    const int wq_last = min(wq_first + 15, n_rows - 1);
    const bool wave_live = wq_first < n_rows;
    const bool wave_full = wq_first + 15 < n_rows;
    const int w_key_hi = pos0 + wq_last;
    const int w_key_lo = window > 0 ? max(0, pos0 + wq_first - window + 1) : 0;
    const int w_pos_lo = pos0 + wq_first;
    const int w_qlo_hi = window > 0 ? max(0, pos0 + wq_last - window + 1) : 0;

    // Prefetch registers: threads [0, VTH) stage V as 4 keys x 8 channels
    // (transposed into the fragment layout with 8-byte stores), the rest
    // stage K rows; every thread carries TPER relative-table entries.
    half8_t pf[NPF];
    float tpf[TPER];
    auto load_tile = [&](int kt) {
        if (tid < VTH) {
            const int k0 = (tid >> 4) * 4;
            const int c = tid & 15;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                pf[i] = ip_load(vsb, vcb, kt + k0 + i, jhi, c, pos0, ring_rows);
        } else {
#pragma unroll
            for (int i = 0; i < KPER; ++i) {
                const int idx = (tid - VTH) + i * (NT - VTH);
                pf[i] = ip_load(ksb, kcb, kt + (idx >> 4), jhi, idx & 15, pos0, ring_rows);
            }
        }
        // New relative-table columns of this tile: d in [p_first - kt - KT + 1, +KT).
        const int dnew = p_first - kt - (KT - 1);
#pragma unroll
        for (int i = 0; i < TPER; ++i) {
            const int idx = tid + i * NT;
            float x = 0.0f;
            if (idx < NREL * KT) {
                const int d = dnew + (idx & (KT - 1));
                if (d >= 0 && d < n_extent)
                    x = table[(size_t)(idx / KT) * n_extent + d];
            }
            tpf[i] = x;
        }
    };
    auto store_tile = [&](int kt, int b) {
        half8_t* KB = KV8[b];
        if (tid < VTH) {
            const int k0 = (tid >> 4) * 4;
            const int c = tid & 15;
            const int sub = k0 >> 4;
            const int vkg = (k0 >> 3) & 1;
            _Float16* V16 = (_Float16*)(KB + 512);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d = c * 8 + i;
                const half4_t w = {pf[0][i], pf[1][i], pf[2][i], pf[3][i]};
                *(half4_t*)(V16 + ip_vfrag(sub, d >> 4, 16 * vkg + (d & 15)) * 8 + (k0 & 7)) = w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < KPER; ++i) {
                const int idx = (tid - VTH) + i * (NT - VTH);
                const int k = idx >> 4;
                const int c = idx & 15;
                KB[ip_kfrag(k >> 4, c >> 1, 16 * (c & 1) + (k & 15))] = pf[i];
            }
        }
        const int dnew = p_first - kt - (KT - 1);
#pragma unroll
        for (int i = 0; i < TPER; ++i) {
            const int idx = tid + i * NT;
            if (idx < NREL * KT) {
                const int d = dnew + (idx & (KT - 1));
                TB[(d & (TBR - 1)) * TBS + idx / KT] = tpf[i];
            }
        }
    };

    // Prologue: tile 0 into buffer 0 with its whole table window, tile 1 into registers.
    load_tile(jlo);
    store_tile(jlo, 0);
    {
        const int d0 = p_first - jlo - (KT - 1);
        for (int idx = tid; idx < NREL * (QROWS - 1); idx += NT) {
            const int r = idx / (QROWS - 1);
            const int d = d0 + KT + (idx - r * (QROWS - 1));
            TB[(d & (TBR - 1)) * TBS + r] = (d >= 0 && d < n_extent) ? table[(size_t)r * n_extent + d] : 0.0f;
        }
    }
    if (ntiles > 1)
        load_tile(jlo + KT);
    __syncthreads();

    float8_t Ofr[8];
#pragma unroll
    for (int dc = 0; dc < 8; ++dc)
        Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m_old = 0.0f;
    float l_val = 1.0f;

#pragma unroll 1
    for (int t = 0; t < ntiles; ++t) {
        const int kt = jlo + t * KT;
        if (t + 1 < ntiles)
            store_tile(kt + KT, (t + 1) & 1);
        if (t + 2 < ntiles)
            load_tile(kt + 2 * KT);
        const half8_t* KB = KV8[t & 1];

        if (wave_live) {
#pragma unroll 1
            for (int sub = 0; sub < 2; ++sub) {
                const int ks0 = kt + sub * 16;
                if (ks0 > w_key_hi)
                    break;
                if (ks0 + 15 < w_key_lo)
                    continue;
                float8_t sacc = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int dc = 0; dc < 8; ++dc) {
                    const half8_t kf = KB[ip_kfrag(sub, dc, lane)];
                    sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf, qhi[dc], sacc);
                    sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf, qlw[dc], sacc);
                }
                float mx = -INFINITY;
                const int kb = ks0 + 8 * kg;
                if (wave_full && ks0 + 15 <= w_pos_lo && ks0 >= w_qlo_hi
                    && w_key_hi - ks0 < n_extent) {
                    // Interior subtile: every key is visible to every row and
                    // inside the table extent; no mask, no extent test.
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float v = ip_rel(TB, qpos - kb - j, R);
                        const float x = sacc[j] * inv_sqrt_d + (v - sink);
                        sacc[j] = x;
                        mx = fmaxf(mx, x);
                    }
                } else if (w_pos_lo - (ks0 + 15) >= n_extent) {
                    // Every visible key is past the extent: relative term zero.
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int key = kb + j;
                        float x = -INFINITY;
                        if (qlive && key >= qlo && key <= qpos) {
                            const float v = 0.0f;
                            x = sacc[j] * inv_sqrt_d + (v - sink);
                        }
                        sacc[j] = x;
                        mx = fmaxf(mx, x);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int key = kb + j;
                        float x = -INFINITY;
                        if (qlive && key >= qlo && key <= qpos) {
                            const int d = qpos - key;
                            float v = 0.0f;
                            if (d < n_extent)
                                v = ip_rel(TB, d, R);
                            x = sacc[j] * inv_sqrt_d + (v - sink);
                        }
                        sacc[j] = x;
                        mx = fmaxf(mx, x);
                    }
                }
                mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
                const float m_new = fmaxf(m_old, mx);
                const float alpha = expf(m_old - m_new);
                m_old = m_new;
                half8_t phi, plo;
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float e = expf(sacc[j] - m_new);
                    phi[j] = (_Float16)e;
                    plo[j] = (_Float16)(e - (float)phi[j]);
                    sum += e;
                }
                sum += __shfl_xor(sum, 16, 32);
                l_val = l_val * alpha + sum;
                if (alpha != 1.0f) {
#pragma unroll
                    for (int dc = 0; dc < 8; ++dc)
                        Ofr[dc] *= alpha;
                }
#pragma unroll
                for (int dc = 0; dc < 8; ++dc) {
                    const half8_t vf = KB[512 + ip_vfrag(sub, dc, lane)];
                    Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf, phi, Ofr[dc]);
                    Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf, plo, Ofr[dc]);
                }
            }
        }
        __syncthreads();
    }

    if (qlive) {
        float* orow = out + ((size_t)qi * IH + h) * ID;
#pragma unroll
        for (int dc = 0; dc < 8; ++dc)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                orow[dc * 16 + kg * 8 + j] = Ofr[dc][j] / l_val;
    }
}
"#;

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::inkling2_attention::{
    INKLING2_ATTN_HEADS, INKLING2_ATTN_HEAD_DIM, INKLING2_ATTN_KV_HEADS,
};
use hip_bridge::{HipError, HipResult};

/// Relative-bias channels the prefill kernel is compiled for.
pub const INKLING2_ATTN_PREFILL_REL: usize = 16;
/// Query rows per workgroup.
pub const INKLING2_ATTN_PREFILL_QROWS: usize = 64;
/// Threads per workgroup (twelve wave32s).
const INKLING2_ATTN_PREFILL_THREADS: u32 = 384;

#[repr(C)]
struct KvPlanes([*const c_void; 8]);

/// Operands of [`Gpu::inkling2_attn_prefill_fa2`].
pub struct Inkling2PrefillAttn<'a> {
    /// F32 `[rows, 24, 128]` clipped (and tau-scaled) queries.
    pub q: &'a GpuTensor,
    /// F32 `[rows, 24, 16]` clipped relative queries.
    pub rel: &'a GpuTensor,
    /// F32 `[rows, 24]` content sinks.
    pub sinks: &'a GpuTensor,
    /// F32 `[16, n_extent]` relative table.
    pub table: &'a GpuTensor,
    /// The visit's eight F16 `[ring_rows, 128]` K and V cache head planes.
    pub k_cache: &'a [GpuTensor],
    pub v_cache: &'a [GpuTensor],
    /// The block's F16 keys and values, `[8, rows, 128]` (head-major planes).
    pub k_stage: &'a GpuTensor,
    pub v_stage: &'a GpuTensor,
    /// F32 `[rows, 24, 128]`.
    pub out: &'a GpuTensor,
    pub rows: usize,
    /// Logical position of block row 0.
    pub pos0: usize,
    /// Sliding window (`0` = full causal).
    pub window: usize,
    /// Physical cache rows (`slot = key % ring_rows`).
    pub ring_rows: usize,
    pub n_extent: usize,
}

impl Gpu {
    /// Blocked causal FA2 prefill attention for `rows` consecutive Inkling-2
    /// query rows starting at logical position `pos0`. Fails closed off gfx1201.
    pub fn inkling2_attn_prefill_fa2(&mut self, a: &Inkling2PrefillAttn<'_>) -> HipResult<()> {
        self.bind_thread()?;
        let (h, kvh, d, nrel) = (
            INKLING2_ATTN_HEADS,
            INKLING2_ATTN_KV_HEADS,
            INKLING2_ATTN_HEAD_DIM,
            INKLING2_ATTN_PREFILL_REL,
        );
        let bad = |m: String| Err(HipError::new(0, &format!("inkling2_attn_prefill_fa2: {m}")));
        if !self.arch_caps.is_gfx1201() {
            return bad(format!("gfx1201 only, not {}", self.arch));
        }
        if a.k_cache.len() != kvh || a.v_cache.len() != kvh {
            return bad("need 8 K and 8 V head planes".into());
        }
        if a.rows == 0 || a.ring_rows == 0 || a.n_extent == 0 {
            return bad(format!(
                "rows {}, ring_rows {}, n_extent {}",
                a.rows, a.ring_rows, a.n_extent
            ));
        }
        let last = a.pos0 + a.rows - 1;
        if i32::try_from(last).is_err()
            || i32::try_from(a.ring_rows).is_err()
            || i32::try_from(a.window).is_err()
            || i32::try_from(a.n_extent).is_err()
        {
            return bad("extent exceeds i32".into());
        }
        // Pre-block keys come from the cache: a full-causal block must fit the
        // cache rows (slot == key) and a sliding window must fit the ring.
        if (a.window == 0 && last >= a.ring_rows) || a.window > a.ring_rows {
            return bad(format!(
                "window {} / last key {last} does not fit ring rows {}",
                a.window, a.ring_rows
            ));
        }
        let f32_len = |t: &GpuTensor| t.buf.size() / 4;
        let plane = a.rows * d * 2;
        if f32_len(a.q) < a.rows * h * d
            || f32_len(a.out) < a.rows * h * d
            || f32_len(a.rel) < a.rows * h * nrel
            || f32_len(a.sinks) < a.rows * h
            || f32_len(a.table) < nrel * a.n_extent
            || a.k_stage.buf.size() < kvh * plane
            || a.v_stage.buf.size() < kvh * plane
            || a.k_cache
                .iter()
                .chain(a.v_cache.iter())
                .any(|t| t.buf.size() < a.ring_rows * d * 2)
        {
            return bad("operand too small".into());
        }
        self.ensure_kernel(
            INKLING2_ATTN_PREFILL_MODULE,
            INKLING2_ATTN_PREFILL_HIP_SRC,
            "inkling2_attn_fa2_prefill_gfx1201",
        )?;
        let planes = |ts: &[GpuTensor]| {
            KvPlanes(std::array::from_fn(|i| ts[i].buf.as_ptr() as *const c_void))
        };
        let stage = |t: &GpuTensor| {
            let base = t.buf.as_ptr() as *const u8;
            KvPlanes(std::array::from_fn(|i| {
                base.wrapping_add(i * plane) as *const c_void
            }))
        };
        let kc = planes(a.k_cache);
        let vc = planes(a.v_cache);
        let ks = stage(a.k_stage);
        let vs = stage(a.v_stage);
        let q_ptr = a.q.buf.as_ptr();
        let rel_ptr = a.rel.buf.as_ptr();
        let sink_ptr = a.sinks.buf.as_ptr();
        let table_ptr = a.table.buf.as_ptr();
        let out_ptr = a.out.buf.as_ptr();
        let rows = a.rows as i32;
        let pos0 = a.pos0 as i32;
        let window = a.window as i32;
        let ring = a.ring_rows as i32;
        let ext = a.n_extent as i32;
        let isd = 1.0f32 / (d as f32).sqrt();
        let mut params: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &rel_ptr as *const _ as *mut c_void,
            &sink_ptr as *const _ as *mut c_void,
            &table_ptr as *const _ as *mut c_void,
            &kc as *const _ as *mut c_void,
            &vc as *const _ as *mut c_void,
            &ks as *const _ as *mut c_void,
            &vs as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &rows as *const _ as *mut c_void,
            &pos0 as *const _ as *mut c_void,
            &window as *const _ as *mut c_void,
            &ring as *const _ as *mut c_void,
            &ext as *const _ as *mut c_void,
            &isd as *const _ as *mut c_void,
        ];
        let grid_x = a.rows.div_ceil(INKLING2_ATTN_PREFILL_QROWS) as u32;
        self.launch_maybe_blob(
            "inkling2_attn_fa2_prefill_gfx1201",
            [grid_x, kvh as u32, 1],
            [INKLING2_ATTN_PREFILL_THREADS, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(rel_ptr);
                b.push_ptr(sink_ptr);
                b.push_ptr(table_ptr);
                for p in
                    kc.0.iter()
                        .chain(vc.0.iter())
                        .chain(ks.0.iter())
                        .chain(vs.0.iter())
                {
                    b.push_ptr(*p);
                }
                b.push_ptr(out_ptr);
                b.push_i32(rows);
                b.push_i32(pos0);
                b.push_i32(window);
                b.push_i32(ring);
                b.push_i32(ext);
                b.push_f32(isd);
                b
            },
        )
    }
}
