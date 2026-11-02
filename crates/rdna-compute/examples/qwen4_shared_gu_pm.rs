// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 fused shared-expert gate/up PeaceMaker byte ORACLE (gfx1151;
//! developer example, needs a GPU: build only on other hosts).
//!
//! Question: is the one-launch certified module `qwen4_shared_gu_pm_gfx1151`
//! (`Gpu::qwen4_shared_gu_pm_launch`, entries `_w8` and `_w4`) byte for byte the
//! chain the Flash-Next prefill runs today?  Arm A is that chain on the real
//! `rdna-compute` launchers over the same operands:
//!   1. the F32 -> F16 conversion of X (`qwen4_convert_f16_x`, the conversion
//!      the grouped F16 WMMA projections issue; its bytes are copied to an own
//!      F16 tensor both arms read, so the F16 scratch the next conversion
//!      overwrites is never an input),
//!   2. the gate GEMM and 3. the up GEMM (`gemm_bf16_xf16_f16_wmma`: BF16
//!      weights through their F16 shadows, `gemm_wmma_lds_128_128_32_64_k64_p` /
//!      the tile `lds_tile_for` selects), F32 `[N, I]`,
//!   4. `shared_expert_activation_bf16_f32` (hidden F32 `[N, I]`, the selector
//!      sigmoid in place),
//!   5. the F32 -> F16 conversion of the hidden (`qwen4_convert_f16_x`: what the
//!      shared down GEMM converts its F32 input with).
//! Arm B is one `qwen4_shared_gu_pm_launch` (per entry) over the same X16 and
//! weights into guarded F16 hidden / F32 selector buffers (4 KiB guards each
//! side, 0xA5 poison, the selector restored from the same pre-sigmoid source
//! before every run).  Compared over the WHOLE buffers: the F16 hidden bytes and
//! the post-sigmoid selector bytes (guards included), after the first run and
//! again after the timed loops.
//!
//! Cases: (a) N = 8192, K = 2560, I = 640 (the Flash-Next shape), random
//! operands; (b) the K4 `--small` stress set (N = 520 ragged, K = 256, I = 192:
//! +-0, BF16 ties, F16 subnormal/overflow, Inf/NaN in X, W and the selector);
//! (c) N in {1, 7, 129, 513} at the Flash-Next shape.  Then, per case, 3 warm +
//! 20 interleaved timed reps: median microseconds of the chain's four launches
//! (gate, up, activation, hidden conversion; between two events) against each
//! entry's one launch.  The product route runs `_w8`.
//!
//! One JSON object per case on stdout and a final `OVERALL PASS|FAIL` line;
//! exit 0 only when every byte of every case and entry matches (2 usage,
//! 3 not gfx1151 / HIP error).
//!
//! Usage: `qwen4_shared_gu_pm [--cases a,b,c] [--seed N] [--reps N]`

use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const GUARD: usize = 4096;
const POISON: u8 = 0xA5;
const WARM: usize = 3;

// ----------------------------------------------------------------- generators

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn f16_to_f32(h: u16) -> f32 {
    half_to_f32(h)
}
fn half_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        u32::from(h >> 15),
        u32::from((h >> 10) & 31),
        u32::from(h & 1023),
    );
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let k = 31 - m.leading_zeros();
            (s << 31) | ((k + 103) << 23) | ((m << (23 - k)) & 0x7f_ffff)
        }
    } else if e == 31 {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
/// The BF16 bits of a finite-or-special F32 that is exactly representable (the operands below are).
fn bf16_exact(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// Default operand: finite F16-range value, sign-varying, ~1/64 each +0 / -0 / subnormal-sized, exponent field 9..15.
fn gen_half(idx: u64, seed: u64) -> u16 {
    let h = mix64(idx.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ seed);
    let (cat, sign, mant, ex) = (
        (h & 63) as u32,
        ((h >> 6) & 1) as u32,
        ((h >> 7) & 0x3ff) as u32,
        9 + ((h >> 17) % 7) as u32,
    );
    match cat {
        0 => 0x0000,
        1 => 0x8000,
        2 => ((sign << 15) | (1 + mant % 1023)) as u16,
        _ => ((sign << 15) | (ex << 10) | mant) as u16,
    }
}

/// Stress X: exponent field 1..30, +-0, subnormals, rare +-Inf and NaN (random payload / sign).
fn gen_x_stress(idx: u64, seed: u64) -> u16 {
    let h = mix64(idx.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ seed);
    let (cat, sign, mant, special) = (
        (h & 63) as u32,
        ((h >> 6) & 1) as u32,
        ((h >> 7) & 0x3ff) as u32,
        ((h >> 17) & 2047) as u32,
    );
    if special == 0 {
        return ((sign << 15) | 0x7c00) as u16;
    }
    if special == 1 {
        return ((sign << 15) | 0x7c00 | (1 + mant % 1023)) as u16;
    }
    match cat {
        0 => 0x0000,
        1 => 0x8000,
        2 => ((sign << 15) | (1 + mant % 1023)) as u16,
        _ => ((sign << 15) | ((1 + ((h >> 28) % 30) as u32) << 10) | mant) as u16,
    }
}

/// Stress W row classes (see the K4 probe): sparse single +-{1,.5,2} rows, dense moderate, rare Inf/NaN rows, all-zero rows and the
/// pinned NaN / Inf channels (`which`: 0 gate, 1 up), as F16 bits.
fn gen_w_stress(row: u64, k: u64, kk: u64, seed: u64, which: u32) -> u16 {
    let rh = mix64(row.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed);
    let rc = (rh & 7) as u32;
    let h = mix64((row * kk + k).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ seed ^ 0xABCD);
    let (sign, mant, cat) = (
        ((h >> 6) & 1) as u32,
        ((h >> 7) & 0x3ff) as u32,
        (h & 63) as u32,
    );
    {
        let pk = (rh >> 20) % kk;
        let payload = 1 + ((rh >> 44) % 1023) as u32;
        let nsign = ((rh >> 43) & 1) as u32;
        let r32 = (row % 32) as u32;
        if (r32 == 5 || (r32 == 9 && which == 0) || (r32 == 13 && which == 1)) && k == pk {
            return ((nsign << 15) | 0x7c00 | payload) as u16;
        }
        if r32 == 21 && which == 0 {
            if k == pk {
                return 0x7c00;
            }
            if k == (pk + 1) % kk {
                return 0xfc00;
            }
        }
    }
    if rc <= 3 {
        if k == (rh >> 8) % kk {
            const MAG: [u16; 3] = [0x3c00, 0x3800, 0x4000];
            return ((((rh >> 50) & 1) as u16) << 15) | MAG[((rh >> 40) % 3) as usize];
        }
        return if h & 1 == 1 { 0x8000 } else { 0 };
    }
    if rc == 7 {
        return if h & 1 == 1 { 0x8000 } else { 0 };
    }
    if rc == 6 {
        let special = ((h >> 17) & 511) as u32;
        if special == 0 {
            return ((sign << 15) | 0x7c00) as u16;
        }
        if special == 1 {
            return ((sign << 15) | 0x7c00 | (1 + mant % 1023)) as u16;
        }
    }
    match cat {
        0 => 0,
        1 => 0x8000,
        2 => ((sign << 15) | (1 + mant % 1023)) as u16,
        _ => ((sign << 15) | ((8 + ((h >> 17) % 9) as u32) << 10) | mant) as u16,
    }
}

/// Pre-sigmoid selector scalars (F32 bits): +-0, BF16 ties, F32 subnormals, (stress) +-Inf / NaN payloads / huge, else finite.
fn gen_selector(idx: u64, seed: u64, stress: bool) -> u32 {
    let h = mix64(idx.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ seed);
    let (cat, sign, r16) = (
        (h & 15) as u32,
        ((h >> 4) & 1) as u32,
        ((h >> 8) & 0xffff) as u32,
    );
    match cat {
        0 => return 0,
        1 => return 0x8000_0000,
        2 => {
            let mut hi = r16;
            if (hi >> 7) & 0xff == 0xff {
                hi &= 0xbf7f;
            }
            return (hi << 16) | 0x8000;
        }
        3 => return (sign << 31) | (1 + ((h >> 8) & 0x7f_fffe) as u32),
        _ => {}
    }
    if stress {
        match cat {
            4 => return (sign << 31) | 0x7f80_0000,
            5 => return (sign << 31) | 0x7f80_0000 | (1 + ((h >> 8) & 0x7f_fffe) as u32),
            6 => return (sign << 31) | 0x7b00_0000 | ((h >> 8) & 0x7f_ffff) as u32,
            _ => {}
        }
    }
    (((h >> 8) & 0xffff) as f32 * (40.0 / 65536.0) - 20.0).to_bits()
}

// -------------------------------------------------------------------- device

/// F16 / F32 storage with `GUARD` bytes of poison either side.
struct Guarded {
    store: GpuTensor,
    elem: usize,
    payload_elems: usize,
}
impl Guarded {
    fn new(gpu: &mut Gpu, dtype: DType, elems: usize) -> Res<Self> {
        let elem = dtype.size();
        let total = elems + 2 * GUARD / elem;
        let store = gpu.alloc_tensor(&[total], dtype)?;
        let mut g = Self {
            store,
            elem,
            payload_elems: elems,
        };
        g.poison(gpu)?;
        Ok(g)
    }
    fn poison(&mut self, gpu: &Gpu) -> Res<()> {
        gpu.hip
            .memset(&self.store.buf, i32::from(POISON), self.store.byte_size())?;
        Ok(())
    }
    fn view(&self) -> GpuTensor {
        self.store.sub_offset(GUARD / self.elem, self.payload_elems)
    }
    fn fetch(&self, gpu: &Gpu) -> Res<Vec<u8>> {
        Ok(gpu.download_raw_bytes(&self.store)?)
    }
    fn restore(&self, gpu: &Gpu, bytes: &[u8]) -> Res<()> {
        let v = self.view();
        gpu.hip.memcpy_htod(&v.buf, bytes)?;
        Ok(())
    }
}

fn upload_dtype(gpu: &Gpu, bytes: &[u8], dtype: DType, elems: usize) -> Res<GpuTensor> {
    let mut t = gpu.upload_raw(bytes, &[elems])?;
    t.dtype = dtype;
    t.shape = vec![elems];
    Ok(t)
}

fn median(mut v: Vec<f32>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        f64::from(v[n / 2])
    } else {
        0.5 * (f64::from(v[n / 2 - 1]) + f64::from(v[n / 2]))
    }
}

fn first_diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n, first)
}

/// Where a first-diff byte offset of a guarded buffer lands.
fn locate(first: Option<usize>, payload_bytes: usize, elem: usize, cols: usize) -> String {
    let Some(f) = first else { return "none".into() };
    if f < GUARD {
        format!("guard_lo+{f}")
    } else if f >= GUARD + payload_bytes {
        format!("guard_hi+{}", f - GUARD - payload_bytes)
    } else {
        let w = (f - GUARD) / elem;
        if cols > 0 {
            format!("out[n={},c={}]", w / cols, w % cols)
        } else {
            format!("out[n={w}]")
        }
    }
}

struct Case {
    name: &'static str,
    n: usize,
    k: usize,
    i: usize,
    stress: bool,
}

/// One case: returns `(json, all_bytes_equal)`.
fn run_case(
    gpu: &mut Gpu,
    c: &Case,
    seed: u64,
    reps: usize,
    keep: &mut Vec<GpuTensor>,
) -> Res<(Value, bool)> {
    let (n, k, i) = (c.n, c.k, c.i);
    assert!(rdna_compute::gemm::qwen4_shared_gu_pm_shape(i, k, n));
    // ---- operands: BF16 weights `[I, K]` (their F16 shadows are what the GEMMs read), F32 X `[N, K]`, selector.
    let (mut wg, mut wu, mut x) = (
        Vec::with_capacity(i * k * 2),
        Vec::with_capacity(i * k * 2),
        Vec::with_capacity(n * k),
    );
    for r in 0..i as u64 {
        for kk in 0..k as u64 {
            let (g, u) = if c.stress {
                (
                    gen_w_stress(r, kk, k as u64, seed ^ 0x57A11, 0),
                    gen_w_stress(r, kk, k as u64, seed ^ 0x0BEEF, 1),
                )
            } else {
                (
                    gen_half(r * k as u64 + kk, seed ^ 0x57A11),
                    gen_half(r * k as u64 + kk, seed ^ 0x0BEEF),
                )
            };
            wg.extend(bf16_exact(f16_to_f32(g)).to_le_bytes());
            wu.extend(bf16_exact(f16_to_f32(u)).to_le_bytes());
        }
    }
    for e in 0..(n * k) as u64 {
        let h = if c.stress {
            gen_x_stress(e, seed ^ 0xC0FFEE)
        } else {
            gen_half(e, seed ^ 0xC0FFEE)
        };
        // Half the values keep a non-F16 mantissa (the conversion rounds), the rest are exact F16.
        let f = f16_to_f32(h);
        let f = if mix64(e ^ seed) & 1 == 1 && f.is_finite() {
            f32::from_bits(f.to_bits() ^ ((mix64(e) as u32) & 0x1fff))
        } else {
            f
        };
        x.push(f);
    }
    let sel_src: Vec<u8> = (0..n as u64)
        .flat_map(|t| gen_selector(t, seed ^ 0x5E1, c.stress).to_le_bytes())
        .collect();
    let w_gate = upload_dtype(gpu, &wg, DType::BF16, i * k)?;
    let w_up = upload_dtype(gpu, &wu, DType::BF16, i * k)?;
    let x32 = gpu.upload_f32(&x, &[n * k])?;
    // ---- X16: the chain's own conversion, copied out of the shared scratch.
    let x16_view = gpu.qwen4_convert_f16_x(&x32, n * k)?;
    let x16_bytes = gpu.download_raw_bytes(&x16_view)?;
    let x16 = upload_dtype(gpu, &x16_bytes, DType::F16, n * k)?;
    // ---- buffers
    let gate_out = gpu.zeros(&[n * i], DType::F32)?;
    let up_out = gpu.zeros(&[n * i], DType::F32)?;
    let hid32 = gpu.zeros(&[n * i], DType::F32)?;
    let sel_a = Guarded::new(gpu, DType::F32, n)?;
    let hid_b: Vec<Guarded> = (0..2)
        .map(|_| Guarded::new(gpu, DType::F16, n * i))
        .collect::<Res<_>>()?;
    let sel_b: Vec<Guarded> = (0..2)
        .map(|_| Guarded::new(gpu, DType::F32, n))
        .collect::<Res<_>>()?;
    let entries = [8u8, 4u8];

    let chain = |gpu: &mut Gpu| -> Res<GpuTensor> {
        gpu.gemm_bf16_xf16_f16_wmma(&w_gate, &x16, &gate_out, i, k, n)?;
        gpu.gemm_bf16_xf16_f16_wmma(&w_up, &x16, &up_out, i, k, n)?;
        gpu.shared_expert_activation_bf16_f32(&gate_out, &up_out, &hid32, &sel_a.view(), n)?;
        Ok(gpu.qwen4_convert_f16_x(&hid32, n * i)?)
    };
    let fused = |gpu: &mut Gpu, e: usize| -> Res<()> {
        gpu.qwen4_shared_gu_pm_launch(
            entries[e],
            &w_gate,
            &w_up,
            &x16,
            &hid_b[e].view(),
            &sel_b[e].view(),
            i,
            k,
            n,
        )?;
        Ok(())
    };

    // ---- first run of every arm
    sel_a.restore(gpu, &sel_src)?;
    let h16 = chain(gpu)?;
    gpu.hip.device_synchronize()?;
    let ref_hid = gpu.download_raw_bytes(&h16)?;
    let ref_sel = sel_a.fetch(gpu)?;
    let gate_ref = gpu.download_f32(&gate_out)?;
    let up_ref = gpu.download_f32(&up_out)?;
    let mut results = Vec::new();
    let mut all_equal = true;
    for e in 0..2 {
        sel_b[e].restore(gpu, &sel_src)?;
        fused(gpu, e)?;
        gpu.hip.device_synchronize()?;
        let (got_hid, got_sel) = (hid_b[e].fetch(gpu)?, sel_b[e].fetch(gpu)?);
        let hid_payload = &got_hid[GUARD..GUARD + n * i * 2];
        let (hd, hf) = first_diff(hid_payload, &ref_hid);
        // Selector: payload against A's, guards against the poison.
        let sel_payload = &got_sel[GUARD..GUARD + n * 4];
        let ref_sel_payload = &ref_sel[GUARD..GUARD + n * 4];
        let (sd, sf) = first_diff(sel_payload, ref_sel_payload);
        let guards = |b: &[u8], payload: usize| {
            b[..GUARD]
                .iter()
                .chain(&b[GUARD + payload..])
                .all(|&x| x == POISON)
        };
        let guards_ok = guards(&got_hid, n * i * 2) && guards(&got_sel, n * 4);
        let a_guards_ok = guards(&ref_sel, n * 4);
        let poison_left = hid_payload
            .chunks_exact(2)
            .filter(|c| c[0] == POISON && c[1] == POISON)
            .count();
        let equal = hd == 0 && sd == 0 && guards_ok && a_guards_ok;
        all_equal &= equal;
        results.push(json!({
            "entry": format!("w{}", entries[e]),
            "hidden_f16_diff_bytes": hd, "hidden_first_diff_at": locate(hf.map(|f| f + GUARD), n * i * 2, 2, i),
            "selector_diff_bytes": sd, "selector_first_diff_at": locate(sf.map(|f| f + GUARD), n * 4, 4, 0),
            "guards_intact": guards_ok, "a_selector_guards_intact": a_guards_ok, "hidden_poison_halfwords": poison_left,
            "bytes_equal": equal,
        }));
    }
    // ---- coverage of what A exercised
    let (mut g_nonfinite, mut g_ties, mut u_nonfinite, mut u_ties) =
        (0usize, 0usize, 0usize, 0usize);
    for (g, u) in gate_ref.iter().zip(&up_ref) {
        let (gb, ub) = (g.to_bits(), u.to_bits());
        if (gb >> 23) & 0xff == 0xff {
            g_nonfinite += 1
        } else if gb & 0xffff == 0x8000 {
            g_ties += 1
        }
        if (ub >> 23) & 0xff == 0xff {
            u_nonfinite += 1
        } else if ub & 0xffff == 0x8000 {
            u_ties += 1
        }
    }
    let (mut h_nan, mut h_inf, mut h_sub, mut h_negzero) = (0usize, 0usize, 0usize, 0usize);
    for c in ref_hid.chunks_exact(2) {
        let b = u16::from_le_bytes([c[0], c[1]]);
        let (ex, m) = ((b >> 10) & 31, b & 1023);
        if ex == 31 && m != 0 {
            h_nan += 1
        } else if ex == 31 {
            h_inf += 1
        } else if ex == 0 && m != 0 {
            h_sub += 1
        } else if b == 0x8000 {
            h_negzero += 1
        }
    }
    let sel_nan = ref_sel[GUARD..GUARD + n * 4]
        .chunks_exact(4)
        .filter(|c| {
            let b = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            (b >> 23) & 0xff == 0xff && b & 0x7f_ffff != 0
        })
        .count();

    // ---- timing: WARM warm launches per arm, then `reps` timed, interleaved A, B(w8), B(w4), A, ...
    let ev = (gpu.hip.event_create()?, gpu.hip.event_create()?);
    let mut time = |gpu: &mut Gpu, which: usize| -> Res<f32> {
        if which == 0 {
            sel_a.restore(gpu, &sel_src)?
        } else {
            sel_b[which - 1].restore(gpu, &sel_src)?
        }
        gpu.hip.event_record(&ev.0, None)?;
        if which == 0 {
            chain(gpu)?;
        } else {
            fused(gpu, which - 1)?;
        }
        gpu.hip.event_record(&ev.1, None)?;
        gpu.hip.event_synchronize(&ev.1)?;
        Ok(gpu.hip.event_elapsed_ms(&ev.0, &ev.1)?)
    };
    for _ in 0..WARM {
        for which in 0..3 {
            time(gpu, which)?;
        }
    }
    let mut ms: [Vec<f32>; 3] = Default::default();
    for _ in 0..reps {
        for which in 0..3 {
            ms[which].push(time(gpu, which)?);
        }
    }
    let med: Vec<f64> = ms.iter().map(|v| median(v.clone()) * 1000.0).collect();
    // ---- post-timing re-check (the last launch of every arm followed a selector restore)
    let ref2_sel = sel_a.fetch(gpu)?;
    let mut post_equal = ref2_sel == ref_sel;
    let mut post = Vec::new();
    for e in 0..2 {
        let (gh, gs) = (hid_b[e].fetch(gpu)?, sel_b[e].fetch(gpu)?);
        let ok = gh[GUARD..GUARD + n * i * 2] == ref_hid[..]
            && gs[GUARD..GUARD + n * 4] == ref_sel[GUARD..GUARD + n * 4]
            && gh[..GUARD].iter().all(|&b| b == POISON)
            && gh[GUARD + n * i * 2..].iter().all(|&b| b == POISON);
        post_equal &= ok;
        post.push(ok);
    }
    all_equal &= post_equal;

    // ---- downstream: the shared down GEMM that stores BF16 rows (`..._bf16st`, the HC row fold's) reads the F32
    // hidden through its own F16 conversion (arm A) or the fused kernel's F16 hidden directly (arm B): same bytes.
    const MD: usize = 256;
    let mut wd = Vec::with_capacity(MD * i * 2);
    for e in 0..(MD * i) as u64 {
        wd.extend(bf16_exact(f16_to_f32(gen_half(e, seed ^ 0xD0D0))).to_le_bytes());
    }
    let w_down = upload_dtype(gpu, &wd, DType::BF16, MD * i)?;
    let down_applies = gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd_applies(&w_down, MD, i, n);
    let mut down_equal = Value::Null;
    if down_applies {
        let (out_a, out_b) = (
            gpu.zeros(&[n * MD], DType::BF16)?,
            gpu.zeros(&[n * MD], DType::BF16)?,
        );
        gpu.gemm_bf16_xf32_f16_wmma_qwen4_bf16st(&w_down, &hid32, MD, i, n, &out_a)?;
        let hid_view = hid_b[0].view();
        gpu.gemm_bf16_xf32_f16_wmma_qwen4_bf16st(&w_down, &hid_view, MD, i, n, &out_b)?;
        gpu.hip.device_synchronize()?;
        let (ba, bb) = (
            gpu.download_raw_bytes(&out_a)?,
            gpu.download_raw_bytes(&out_b)?,
        );
        let equal = ba == bb;
        all_equal &= equal;
        down_equal = json!(equal);
        gpu.free_tensor(out_a)?;
        gpu.free_tensor(out_b)?;
    }

    // The BF16 -> F16 shadows are cached by weight pointer for the process: the weights stay alive
    // (never freed, never recycled by the pool), so no later case can alias a stale shadow.
    keep.push(w_gate);
    keep.push(w_up);
    keep.push(w_down);
    for t in [x32, x16, gate_out, up_out, hid32] {
        gpu.free_tensor(t)?;
    }
    Ok((
        json!({
            "case": c.name, "N": n, "K": k, "I": i, "stress": c.stress, "reps": reps,
            "entries": results,
            "post_timing_equal": {"w8": post[0], "w4": post[1], "a_selector": ref2_sel == ref_sel},
            "downstream_bf16st": {"applies": down_applies, "bytes_equal": down_equal},
            "coverage": {
                "gate_nonfinite": g_nonfinite, "gate_bf16_ties": g_ties, "up_nonfinite": u_nonfinite, "up_bf16_ties": u_ties,
                "hidden_f16_nan": h_nan, "hidden_f16_inf": h_inf, "hidden_f16_subnormal": h_sub, "hidden_f16_negzero": h_negzero,
                "selector_out_nan": sel_nan,
            },
            "chain_median_us": med[0], "w8_median_us": med[1], "w4_median_us": med[2],
            "ratio_w8_over_chain": med[1] / med[0], "ratio_w4_over_chain": med[2] / med[0],
            "bytes_equal": all_equal,
        }),
        all_equal,
    ))
}

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cases = arg(&args, "--cases", "a,b,c");
    let seed: u64 = match arg(&args, "--seed", "1").parse() {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "--seed: {e}\nusage: qwen4_shared_gu_pm [--cases a,b,c] [--seed N] [--reps N]"
            );
            std::process::exit(2);
        }
    };
    let reps: usize = arg(&args, "--reps", "20").parse().unwrap_or(20);
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("{}", json!({"event": "error", "error": e.to_string()}));
            println!("OVERALL FAIL");
            std::process::exit(3);
        }
    };
    if gpu.arch != "gfx1151" {
        println!(
            "{}",
            json!({"event": "skip", "reason": format!("device is {}, the fused shared gate/up module is gfx1151-only", gpu.arch)})
        );
        println!("OVERALL FAIL");
        std::process::exit(3);
    }
    // What the Qwen4 forward sets (the F16 WMMA route reads it through `route_rows`).
    gpu.qwen4_scope = true;
    let mut list: Vec<Case> = Vec::new();
    for tag in cases.split(',').filter(|t| !t.is_empty()) {
        match tag {
            "a" => list.push(Case {
                name: "a_flash_next",
                n: 8192,
                k: 2560,
                i: 640,
                stress: false,
            }),
            "b" => list.push(Case {
                name: "b_small_stress",
                n: 520,
                k: 256,
                i: 192,
                stress: true,
            }),
            "c" => {
                for (name, n) in [("c_n1", 1), ("c_n7", 7), ("c_n129", 129), ("c_n513", 513)] {
                    list.push(Case {
                        name,
                        n,
                        k: 2560,
                        i: 640,
                        stress: false,
                    });
                }
            }
            other => {
                eprintln!("unknown case {other}: expected a, b or c");
                std::process::exit(2);
            }
        }
    }
    println!(
        "{}",
        json!({"event": "start", "arch": gpu.arch, "seed": seed, "reps": reps, "shared_gu_pm_waves_default": rdna_compute::gemm::QWEN4_SHARED_GU_PM_WAVES})
    );
    let mut ok = true;
    let mut keep: Vec<GpuTensor> = Vec::new();
    for c in &list {
        match run_case(&mut gpu, c, seed, reps, &mut keep) {
            Ok((j, equal)) => {
                ok &= equal;
                println!("{j}");
            }
            Err(e) => {
                ok = false;
                println!(
                    "{}",
                    json!({"case": c.name, "error": e.to_string(), "bytes_equal": false})
                );
            }
        }
    }
    println!("OVERALL {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
