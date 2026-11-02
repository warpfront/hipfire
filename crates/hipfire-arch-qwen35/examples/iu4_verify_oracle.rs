// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Measurement (IU4-VERIFY-CHECK): row-count independence + speed of the
// gfx1201 IU4 (W4A4) MMQ prefill GEMMs on the real 27B verify shapes.
// Modelled on the FP8-VERIFY `fp8_verify_oracle`.
//
// Byte oracle (0-byte tolerance, complete output buffers incl. untouched rows
// past N and a canary tail). Two IU4 routes, each with its own reference:
//   iu4  — standalone `quantize_int4_mmq_ds128` (`ensure_int4_mmq_x`, token
//          layout) + `gemm_mq4g256v2_mmq_{set,add}_prequant_iu4` per
//          projection (the `_b1` ISA bundle when eligible);
//   slab — the gfx1201 A4 producer the DFlash IU4 verify uses for that
//          projection (slab layout) + its IU4 consumer (`_b1s`):
//            qkvza/qkv/gate_up — AWQ RMSNorm/FWHT
//              (`fused_rmsnorm_rotate_mq_i4_gfx12_batched`);
//            dn_o — gated RMSNorm (`gated_norm_rotate_mq_i4_gfx12_batched`);
//            fa_o — sigmoid gate (`sigmoid_mul_rotate_x_mq_awq_i4_gfx12_batched`);
//            down — SwiGLU of separate gate/up
//              (`fused_silu_mul_rotate_mq_i4_gfx12_batched`);
//            gate_up_silu — AWQ RMSNorm/FWHT + the F1-lite fused gate/up +
//              SwiGLU GEMM (`gemm_gate_up_silu_mq4g256v2_iu4_prepared`, h out);
//            ffn_a4 — the whole verify FFN: RMSNorm producer, F1-lite h, the
//              h-in producer (`fused_silu_hin_rotate_mq_i4_batched`) and the
//              down ADD (residual out).
// Reference = the route launched as 8 x 16-row chunks at row offsets.
// Candidates: 128 one-row launches (N=1), one launch over [0,N) for
// N = 32, 64, 128, 64+64 and 48+48+32 at row offsets, rerun determinism.
//
// Also: IU4-vs-F16 numeric distance per shape (diagnostic), and event timing
// at N = 16, 64, 128 of F16 (`*_verify_exact`: one-tile at 16, K32 PM/HIP at
// 64..128; with and without the f16 conversion) vs IU4 (quant + GEMM, GEMM
// only, slab producer + GEMM).
//
// Build: cargo build -p hipfire-arch-qwen35 --release --example iu4_verify_oracle -j12
// Run:   iu4_verify_oracle --model <path> --out <abs dir> [--ops a,b] [--seeds 1,2]
//          [--reps 30] [--tests oracle,slab,dist,time]

use hipfire_arch_qwen35::load_qwen35_bundle;
use hipfire_arch_qwen35::qwen35::LayerWeights;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SequenceHint, SpecLoadCfg};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const NMAX: usize = 128;
const CANARY: usize = 256;
const SENTINEL: u32 = 0x7FBA_DBAD;
const STREAM_GBS: f64 = 640.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fam {
    Qkvza,
    Qkv,
    GateUp,
    Residual,
    /// F1-lite gate/up + SwiGLU: `w = [gate, up]`, one h output (SET).
    GateUpSilu,
    /// Whole A4 FFN: `w = [gate, up, down]`, one residual output (ADD).
    FfnA4,
}

/// The gfx1201 A4 producer of a case's slab route.
#[derive(Clone, Copy)]
enum Slab<'a> {
    /// AWQ RMSNorm/FWHT: (norm weight, AWQ scale, eps).
    Norm(&'a GpuTensor, &'a GpuTensor, f32),
    /// Gated RMSNorm/FWHT (x and z both read from X): (norm weight, AWQ, eps, heads, head_dim).
    GatedNorm(&'a GpuTensor, Option<&'a GpuTensor>, f32, usize, usize),
    /// Sigmoid gate (attn and gate both read from X): AWQ scale.
    Sigmoid(&'a GpuTensor),
    /// SwiGLU of separate gate/up (both read from X): AWQ scale.
    Silu(Option<&'a GpuTensor>),
    /// FFN chain: RMSNorm (norm, gate AWQ, eps), F1-lite h into the h
    /// scratch, h-in producer (down AWQ), down ADD; last field = hidden dim.
    Ffn(&'a GpuTensor, &'a GpuTensor, f32, &'a GpuTensor, usize),
}

struct Case<'a> {
    name: &'static str,
    fam: Fam,
    w: Vec<&'a GpuTensor>,
    ms: Vec<usize>,
    k: usize,
    layers: usize,
    /// F16 verify_exact comparison exists for this case.
    f16: bool,
    /// The A4 producer for the slab route.
    slab: Option<Slab<'a>>,
    /// The token-layout `iu4` route exists (plain SET/ADD consumers).
    iu4: bool,
}

impl Case<'_> {
    fn total_m(&self) -> usize {
        self.ms.iter().sum()
    }
    fn weight_bytes(&self) -> usize {
        self.total_m() * (self.k / 256) * 136
    }
    fn add(&self) -> bool {
        matches!(self.fam, Fam::Residual | Fam::FfnA4)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    Iu4,
    Slab,
    F16Exact,
    F16ExactKept,
}

fn views(
    case: &Case,
    x: &GpuTensor,
    y: &[GpuTensor],
    row0: usize,
    n: usize,
) -> (GpuTensor, Vec<GpuTensor>) {
    let xv = x.sub_offset(row0 * case.k, n * case.k);
    let ys = y
        .iter()
        .zip(&case.ms)
        .map(|(t, &m)| t.sub_offset(row0 * m, n * m))
        .collect();
    (xv, ys)
}

/// The IU4 GEMMs of one case from a quantized activation pointer.
fn iu4_gemms(gpu: &mut Gpu, case: &Case, xq: *mut c_void, ys: &[GpuTensor], n: usize) -> R<()> {
    for ((w, &m), y) in case.w.iter().zip(&case.ms).zip(ys) {
        if case.add() {
            gpu.gemm_mq4g256v2_mmq_add_prequant_iu4(w, xq, y, m, case.k, n)?;
        } else {
            gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(w, xq, y, m, case.k, n)?;
        }
    }
    Ok(())
}

/// A4 producer over rows `xv` (N rows of K); returns the prepared handle.
fn slab_prepare(
    gpu: &mut Gpu,
    case: &Case,
    xv: &GpuTensor,
    n: usize,
) -> R<rdna_compute::Int4MmqPrepared> {
    let k = case.k;
    let res = gpu.reserve_int4_mmq(k, n)?;
    Ok(match case.slab.ok_or("no slab producer for this case")? {
        Slab::Norm(norm, awq, eps) | Slab::Ffn(norm, awq, eps, _, _) => {
            gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(xv, norm, Some(awq), None, res, k, eps, n)?
        }
        Slab::GatedNorm(norm, awq, eps, heads, hd) => gpu.gated_norm_rotate_mq_i4_gfx12_batched(
            xv,
            rdna_compute::norm::GdnScanOut::F32,
            xv,
            norm,
            awq,
            None,
            res,
            heads,
            hd,
            eps,
            k,
            n,
        )?,
        Slab::Sigmoid(awq) => gpu.sigmoid_mul_rotate_x_mq_awq_i4_gfx12_batched(
            xv,
            rdna_compute::gemv::SigmoidGate::Rows(xv),
            awq,
            None,
            res,
            k,
            n,
        )?,
        Slab::Silu(awq) => {
            gpu.fused_silu_mul_rotate_mq_i4_gfx12_batched(xv, xv, awq, None, res, k, n)?
        }
    })
}

/// Slab producer; returns the prepared pointer and whether it is slab layout.
fn slab_produce(gpu: &mut Gpu, case: &Case, xv: &GpuTensor, n: usize) -> R<(*mut c_void, bool)> {
    let prep = slab_prepare(gpu, case, xv, n)?;
    let xq = gpu.int4_mmq_prepared_ptr(&prep, case.k, n)?;
    let slab = gpu.scratch.int4_mmq_slab_at(xq);
    Ok((xq, slab))
}

/// The slab route's consumers from a prepared handle (`h`: the FFN chain's
/// h scratch view, rows `[row0, row0 + n)`).
fn slab_consume(
    gpu: &mut Gpu,
    case: &Case,
    prep: &rdna_compute::Int4MmqPrepared,
    ys: &[GpuTensor],
    h: Option<&GpuTensor>,
    n: usize,
) -> R<()> {
    let (w, m, k) = (&case.w, &case.ms, case.k);
    match case.fam {
        Fam::GateUpSilu => {
            if !gpu.gemm_gate_up_silu_mq4g256v2_iu4_prepared(
                w[0], w[1], prep, &ys[0], m[0], m[0], k, n,
            )? {
                return Err(format!(
                    "{}: the F1-lite gate/up+SwiGLU GEMM was not admitted",
                    case.name
                )
                .into());
            }
        }
        Fam::FfnA4 => {
            let Some(Slab::Ffn(_, _, _, awq_down, hd)) = case.slab else {
                return Err("ffn_a4 without its chain".into());
            };
            let h = h.ok_or("ffn_a4 without the h scratch")?;
            if !gpu.gemm_gate_up_silu_mq4g256v2_iu4_prepared(w[0], w[1], prep, h, hd, hd, k, n)? {
                return Err("ffn_a4: the F1-lite gate/up+SwiGLU GEMM was not admitted".into());
            }
            let res = gpu.reserve_int4_mmq(hd, n)?;
            let dprep = gpu.fused_silu_hin_rotate_mq_i4_batched(h, awq_down, res, hd, n)?;
            gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(w[2], &dprep, &ys[0], m[0], hd, n)?;
        }
        _ => {
            let xq = gpu.int4_mmq_prepared_ptr(prep, k, n)?;
            iu4_gemms(gpu, case, xq, ys, n)?;
        }
    }
    Ok(())
}

fn launch(
    gpu: &mut Gpu,
    case: &Case,
    route: Route,
    x: &GpuTensor,
    y: &[GpuTensor],
    h: Option<&GpuTensor>,
    row0: usize,
    n: usize,
) -> R<()> {
    let (xv, ys) = views(case, x, y, row0, n);
    let (w, m, k) = (&case.w, &case.ms, case.k);
    match route {
        Route::Iu4 => {
            let xq = gpu.ensure_int4_mmq_x(&xv, n, k)?;
            iu4_gemms(gpu, case, xq, &ys, n)?;
        }
        Route::Slab => {
            let prep = slab_prepare(gpu, case, &xv, n)?;
            let hv = h.map(|t| t.sub_offset(row0 * m_hidden(case), n * m_hidden(case)));
            slab_consume(gpu, case, &prep, &ys, hv.as_ref(), n)?;
        }
        Route::F16Exact | Route::F16ExactKept => {
            if route == Route::F16Exact {
                gpu.invalidate_fp16_cache();
            }
            match case.fam {
                Fam::Qkvza => gpu.gemm_qkvza_mq4g256v2_verify_exact(
                    w[0], w[1], w[2], w[3], &xv, &ys[0], &ys[1], &ys[2], &ys[3], m[0], m[1], m[2],
                    m[3], k, n,
                )?,
                Fam::Qkv => gpu.gemm_qkv_mq4g256v2_verify_exact(
                    w[0], w[1], w[2], &xv, &ys[0], &ys[1], &ys[2], m[0], m[1], m[2], k, n,
                )?,
                Fam::GateUp => gpu.gemm_gate_up_mq4g256v2_verify_exact(
                    w[0], w[1], &xv, &ys[0], &ys[1], m[0], m[1], k, n,
                )?,
                Fam::Residual => {
                    gpu.gemm_mq4g256v2_residual_verify_exact(w[0], &xv, &ys[0], m[0], k, n)?
                }
                Fam::GateUpSilu | Fam::FfnA4 => {
                    return Err(format!("{}: no F16 twin", case.name).into())
                }
            }
        }
    }
    Ok(())
}

/// Row width of the FFN chain's h scratch (0 for every other case).
fn m_hidden(case: &Case) -> usize {
    match case.slab {
        Some(Slab::Ffn(_, _, _, _, hd)) => hd,
        _ => 0,
    }
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

/// Seeded F32 activations [NMAX, k]: per-row magnitudes 2^-6..2^6, outlier
/// columns, ±0, an exact-negation row, an all-zero row, a near-subnormal row.
fn gen_x(seed: u64, k: usize) -> Vec<f32> {
    let mut x = vec![0f32; NMAX * k];
    for r in 0..NMAX {
        let row_exp = ((hash3(seed, r as u64, 0xE) % 13) as i32) - 6;
        for c in 0..k {
            let v = match r % 16 {
                7 => -x[(r - 1) * k + c],
                11 => 0.0,
                13 => {
                    let u = hash3(seed, r as u64, c as u64);
                    ((u >> 40) as f32 / 16_777_216.0 - 0.5) * 1e-38
                }
                _ => {
                    let u = hash3(seed, r as u64, c as u64);
                    let mut v = ((u >> 40) as f32 / 16_777_216.0 - 0.5) * 2.0;
                    if u % 97 == 0 {
                        v = 0.0;
                    } else if u % 89 == 0 {
                        v = -0.0;
                    }
                    if c % 509 == (r * 31) % 509 {
                        v *= 64.0;
                    }
                    v * (row_exp as f32).exp2()
                }
            };
            x[r * k + c] = v;
        }
    }
    x
}

fn gen_init(seed: u64, out_idx: usize, m: usize, residual: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity((NMAX * m + CANARY) * 4);
    for i in 0..NMAX * m {
        let w = if residual {
            let u = hash3(seed ^ 0xA5A5, out_idx as u64, i as u64);
            if u % 97 == 0 {
                0x8000_0000u32
            } else {
                ((((u >> 40) as u32 | 1) as f32 / 16_777_216.0 - 0.5) * 4.0).to_bits()
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

struct Outs {
    y: Vec<GpuTensor>,
    init_dev: Vec<GpuTensor>,
    init: Vec<Vec<u8>>,
    /// The FFN chain's h scratch (`NMAX` rows of the hidden dim).
    h: Option<GpuTensor>,
}

fn alloc_outs(gpu: &mut Gpu, case: &Case, seed: u64) -> R<Outs> {
    let mut o = Outs {
        y: Vec::new(),
        init_dev: Vec::new(),
        init: Vec::new(),
        h: None,
    };
    for (i, &m) in case.ms.iter().enumerate() {
        let init = gen_init(seed, i, m, case.add());
        let n = NMAX * m + CANARY;
        let y = gpu.zeros(&[n], DType::F32)?;
        let d = gpu.zeros(&[n], DType::F32)?;
        gpu.hip.memcpy_htod(&d.buf, &init)?;
        o.y.push(y);
        o.init_dev.push(d);
        o.init.push(init);
    }
    let hd = m_hidden(case);
    if hd > 0 {
        o.h = Some(gpu.zeros(&[NMAX * hd], DType::F32)?);
    }
    Ok(o)
}

fn free_outs(gpu: &mut Gpu, o: Outs) -> R<()> {
    for t in o.y.into_iter().chain(o.init_dev).chain(o.h) {
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Relative / absolute L2 of `got` vs `want` over rows [0, NMAX) (init subtracted for residual).
fn distance(
    got: &[Vec<u8>],
    want: &[Vec<u8>],
    init: &[Vec<u8>],
    ms: &[usize],
    residual: bool,
) -> (f64, f64, usize) {
    let (mut num, mut den, mut nonfinite) = (0f64, 0f64, 0usize);
    for ((g, w), (i, &m)) in got.iter().zip(want).zip(init.iter().zip(ms)) {
        for idx in 0..NMAX * m {
            let f =
                |b: &[u8]| f32::from_le_bytes(b[idx * 4..idx * 4 + 4].try_into().unwrap()) as f64;
            let base = if residual { f(i) } else { 0.0 };
            let (a, b) = (f(g) - base, f(w) - base);
            if !a.is_finite() || !b.is_finite() {
                nonfinite += 1;
                continue;
            }
            num += (a - b) * (a - b);
            den += b * b;
        }
    }
    let rel_l2 = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    (rel_l2, num.sqrt(), nonfinite)
}

fn main() {
    match run() {
        Ok(0) => std::process::exit(0),
        Ok(_) => std::process::exit(1),
        Err(e) => {
            eprintln!("iu4_verify_oracle: {e}");
            std::process::exit(2);
        }
    }
}

/// Byte oracle of one route against its own 8 x 16-row reference.
fn oracle(
    gpu: &mut Gpu,
    rep: &mut Report,
    case: &Case,
    route: Route,
    x: &GpuTensor,
    outs: &Outs,
    tag: &str,
) -> R<Vec<Vec<u8>>> {
    let ms = &case.ms;
    let (y, h) = (&outs.y, outs.h.as_ref());
    reset(gpu, outs)?;
    for c in 0..NMAX / 16 {
        launch(gpu, case, route, x, y, h, c * 16, 16)?;
    }
    let refb = read(gpu, outs)?;
    let mut v = Vec::new();
    reset(gpu, outs)?;
    for c in 0..NMAX / 16 {
        launch(gpu, case, route, x, y, h, c * 16, 16)?;
    }
    let ok = rep.cmp(&format!("{tag} ref16 rerun"), &read(gpu, outs)?, &refb, ms);
    v.push(format!("rerun:{}", if ok { "same" } else { "DIFF" }));
    reset(gpu, outs)?;
    for r in 0..NMAX {
        launch(gpu, case, route, x, y, h, r, 1)?;
    }
    let ok = rep.cmp(&format!("{tag} N=1 x128"), &read(gpu, outs)?, &refb, ms);
    v.push(format!("N=1x128:{}", if ok { "same" } else { "DIFF" }));
    for n in [32usize, 64, 128] {
        reset(gpu, outs)?;
        launch(gpu, case, route, x, y, h, 0, n)?;
        let ok = rep.cmp(
            &format!("{tag} N={n}"),
            &read(gpu, outs)?,
            &expect(&refb, &outs.init, ms, n),
            ms,
        );
        v.push(format!("N={n}:{}", if ok { "same" } else { "DIFF" }));
    }
    for (name, set) in [
        ("64+64", vec![(0usize, 64usize), (64, 64)]),
        ("48+48+32", vec![(0, 48), (48, 48), (96, 32)]),
    ] {
        reset(gpu, outs)?;
        for (r0, n) in set {
            launch(gpu, case, route, x, y, h, r0, n)?;
        }
        let ok = rep.cmp(&format!("{tag} {name}"), &read(gpu, outs)?, &refb, ms);
        v.push(format!("{name}:{}", if ok { "same" } else { "DIFF" }));
    }
    let rh: Vec<String> = refb.iter().map(|r| format!("{:016x}", fnv(r))).collect();
    rep.line(format!("{tag} ref16_fnv={rh:?} | {}", v.join(" ")));
    Ok(refb)
}

fn run() -> R<usize> {
    let mut model = String::new();
    let mut out = String::new();
    let mut ops: Vec<String> = "qkvza,qkvzaf,dn_o,gate_up,down,qkv,fa_o,head,gate_up_silu,ffn_a4"
        .split(',')
        .map(String::from)
        .collect();
    let mut seeds: Vec<u64> = vec![1, 2];
    let mut reps = 30usize;
    let mut tests: Vec<String> = "oracle,slab,dist,time"
        .split(',')
        .map(String::from)
        .collect();
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
            "--model" => model = v,
            "--out" => out = v,
            "--ops" => ops = list(&v),
            "--seeds" => seeds = v.split(',').map(|s| s.parse()).collect::<Result<_, _>>()?,
            "--reps" => reps = v.parse()?,
            "--tests" => tests = list(&v),
            _ => return Err(format!("unknown argument {k}").into()),
        }
    }
    if model.is_empty() || !out.starts_with('/') {
        return Err("--model <path> and absolute --out <dir> are required".into());
    }
    std::fs::create_dir_all(&out)?;
    let has = |t: &str| tests.iter().any(|x| x == t);
    let mut rep = Report {
        text: String::new(),
        fails: 0,
        checks: 0,
    };

    let mut gpu = Gpu::init()?;
    rep.line(format!("arch {} model {model}", gpu.arch));
    let mut env: Vec<String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("HIPFIRE_"))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    env.sort();
    rep.line(format!("env {env:?}"));

    let src = ModelSource::from_path(&model)?;
    let cask = CaskConfig::default();
    let mut lc = LoadCtx {
        path: &model,
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
    rep.line(format!(
        "mq4v2_symmetric={} iu4_prefill={} g12_iu4_isa={} a4_slab_active={} cb_verify_pm={}",
        gpu.mq4v2_symmetric,
        gpu.flags.iu4_prefill_enabled(),
        gpu.flags.g12_iu4_isa,
        gpu.a4_slab_active(),
        gpu.flags.cb_verify_pm
    ));
    let n_dn = b
        .weights
        .layers
        .iter()
        .filter(|l| matches!(l, LayerWeights::DeltaNet(_)))
        .count();
    let n_fa = b
        .weights
        .layers
        .iter()
        .filter(|l| matches!(l, LayerWeights::FullAttn(_)))
        .count();
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
    let eps = b.config.norm_eps;
    let row_bytes = kd / 256 * 136;
    // gfx1201 A4 loader fold: Z buffer = Z rows + one 128-row tile holding beta (row 0) / alpha (row 64).
    let z_rows = dn.wz.buf.buf.size() / row_bytes;
    rep.line(format!(
        "layers dn={n_dn} fa={n_fa} dim={kd} eps={eps} wz.m={} wz buffer rows={z_rows} (fold={}) awq: wqkv={} wq={} w_gate={}",
        dn.wz.m,
        z_rows == dn.wz.m + 128,
        dn.wqkv.awq_scale.is_some(),
        fa.wq.awq_scale.is_some(),
        dn.w_gate.awq_scale.is_some()
    ));
    let slab = |norm: &'static str| -> Option<Slab> {
        match norm {
            "dn_attn" => dn
                .wqkv
                .awq_scale
                .as_ref()
                .map(|a| Slab::Norm(&dn.attn_norm, a, eps)),
            "fa_attn" => fa
                .wq
                .awq_scale
                .as_ref()
                .map(|a| Slab::Norm(&fa.attn_norm, a, eps)),
            "dn_ffn" => dn
                .w_gate
                .awq_scale
                .as_ref()
                .map(|a| Slab::Norm(&dn.ffn_norm, a, eps)),
            "dn_o" => Some(Slab::GatedNorm(
                &dn.norm_weight,
                dn.wo.awq_scale.as_ref(),
                eps,
                b.config.linear_num_value_heads,
                b.config.linear_value_head_dim,
            )),
            "fa_o" => fa.wo.awq_scale.as_ref().map(Slab::Sigmoid),
            "down" => Some(Slab::Silu(dn.w_down.awq_scale.as_ref())),
            "ffn" => match (dn.w_gate.awq_scale.as_ref(), dn.w_down.awq_scale.as_ref()) {
                (Some(g), Some(d)) => Some(Slab::Ffn(&dn.ffn_norm, g, eps, d, dn.w_down.k)),
                _ => None,
            },
            _ => None,
        }
    };
    let head_mq4 = b.weights.output.gpu_dtype == DType::MQ4G256V2;
    let mut all = vec![
        Case {
            name: "qkvza",
            fam: Fam::Qkvza,
            w: vec![&dn.wqkv.buf, &dn.wz.buf, &dn.w_beta.buf, &dn.w_alpha.buf],
            ms: vec![dn.wqkv.m, dn.wz.m, dn.w_beta.m, dn.w_alpha.m],
            k: kd,
            layers: n_dn,
            f16: true,
            slab: slab("dn_attn"),
            iu4: true,
        },
        Case {
            name: "dn_o",
            fam: Fam::Residual,
            w: vec![&dn.wo.buf],
            ms: vec![dn.wo.m],
            k: dn.wo.k,
            layers: n_dn,
            f16: true,
            slab: slab("dn_o"),
            iu4: true,
        },
        Case {
            name: "gate_up",
            fam: Fam::GateUp,
            w: vec![&dn.w_gate.buf, &dn.w_up.buf],
            ms: vec![dn.w_gate.m, dn.w_up.m],
            k: kd,
            layers: n_dn + n_fa,
            f16: true,
            slab: slab("dn_ffn"),
            iu4: true,
        },
        Case {
            name: "down",
            fam: Fam::Residual,
            w: vec![&dn.w_down.buf],
            ms: vec![dn.w_down.m],
            k: dn.w_down.k,
            layers: n_dn + n_fa,
            f16: true,
            slab: slab("down"),
            iu4: true,
        },
        Case {
            name: "qkv",
            fam: Fam::Qkv,
            w: vec![&fa.wq.buf, &fa.wk.buf, &fa.wv.buf],
            ms: vec![fa.wq.m, fa.wk.m, fa.wv.m],
            k: kd,
            layers: n_fa,
            f16: true,
            slab: slab("fa_attn"),
            iu4: true,
        },
        Case {
            name: "fa_o",
            fam: Fam::Residual,
            w: vec![&fa.wo.buf],
            ms: vec![fa.wo.m],
            k: fa.wo.k,
            layers: n_fa,
            f16: true,
            slab: slab("fa_o"),
            iu4: true,
        },
        // The F1-lite gate/up+SwiGLU GEMM (h out) and the whole verify FFN.
        Case {
            name: "gate_up_silu",
            fam: Fam::GateUpSilu,
            w: vec![&dn.w_gate.buf, &dn.w_up.buf],
            ms: vec![dn.w_gate.m],
            k: kd,
            layers: n_dn + n_fa,
            f16: false,
            slab: slab("dn_ffn"),
            iu4: false,
        },
        Case {
            name: "ffn_a4",
            fam: Fam::FfnA4,
            w: vec![&dn.w_gate.buf, &dn.w_up.buf, &dn.w_down.buf],
            ms: vec![dn.w_down.m],
            k: kd,
            layers: n_dn + n_fa,
            f16: false,
            slab: slab("ffn"),
            iu4: false,
        },
    ];
    if z_rows == dn.wz.m + 128 {
        // IU4-only form of qkvza on the loader's Z fold: 2 SET launches.
        all.push(Case {
            name: "qkvzaf",
            fam: Fam::GateUp,
            w: vec![&dn.wqkv.buf, &dn.wz.buf],
            ms: vec![dn.wqkv.m, z_rows],
            k: kd,
            layers: n_dn,
            f16: false,
            slab: slab("dn_attn"),
            iu4: true,
        });
    }
    if head_mq4 {
        all.push(Case {
            name: "head",
            fam: Fam::Residual,
            w: vec![&b.weights.output.buf],
            ms: vec![b.weights.output.m],
            k: kd,
            layers: 1,
            f16: true,
            slab: None,
            iu4: true,
        });
    } else {
        rep.line(format!(
            "head dtype {:?}: not MQ4G256V2, skipped",
            b.weights.output.gpu_dtype
        ));
    }
    for c in &all {
        rep.line(format!(
            "shape {} fam={:?} ms={:?} total_m={} k={} layers={} weight_MB={:.1} b1_eligible(m%4)={} slab_route={}",
            c.name,
            c.fam,
            c.ms,
            c.total_m(),
            c.k,
            c.layers,
            c.weight_bytes() as f64 / 1e6,
            c.ms.iter().all(|m| m % 4 == 0),
            c.slab.is_some()
        ));
        if c.k % 256 != 0 {
            return Err(format!("{}: K {} not a multiple of 256", c.name, c.k).into());
        }
    }
    let cases: Vec<&Case> = all
        .iter()
        .filter(|c| ops.iter().any(|o| o == c.name))
        .collect();
    let kmax = cases.iter().map(|c| c.k).max().unwrap_or(kd);
    let x = gpu.zeros(&[NMAX * kmax], DType::F32)?;
    let load_x = |gpu: &mut Gpu, v: &[f32]| -> R<()> {
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        gpu.hip.memcpy_htod(&x.buf, &bytes)?;
        gpu.hip.device_synchronize()?;
        Ok(())
    };
    let mut timing_rows: Vec<String> = Vec::new();

    for case in &cases {
        let ms = &case.ms;
        for &seed in &seeds {
            let xv = gen_x(seed, case.k);
            load_x(&mut gpu, &xv)?;
            let tag = |t: &str| format!("{} seed{} {}", case.name, seed, t);
            let outs = alloc_outs(&mut gpu, case, seed)?;
            let xh = fnv(&xv.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());
            rep.line(format!("{} x_fnv={xh:016x}", tag("inputs")));
            let mut iu4_ref = None;
            if has("oracle") && case.iu4 {
                iu4_ref = Some(oracle(
                    &mut gpu,
                    &mut rep,
                    case,
                    Route::Iu4,
                    &x,
                    &outs,
                    &tag("iu4"),
                )?);
            }
            if has("slab") && case.slab.is_some() {
                let probe = slab_produce(&mut gpu, case, &x.sub_offset(0, 16 * case.k), 16)?;
                rep.line(format!(
                    "{} slab producer wrote slab layout: {}",
                    tag("slab"),
                    probe.1
                ));
                oracle(
                    &mut gpu,
                    &mut rep,
                    case,
                    Route::Slab,
                    &x,
                    &outs,
                    &tag("slab"),
                )?;
            }
            if has("dist") && case.f16 {
                let iu4b = match iu4_ref {
                    Some(r) => r,
                    None => {
                        reset(&mut gpu, &outs)?;
                        launch(&mut gpu, case, Route::Iu4, &x, &outs.y, None, 0, NMAX)?;
                        read(&mut gpu, &outs)?
                    }
                };
                reset(&mut gpu, &outs)?;
                launch(&mut gpu, case, Route::F16Exact, &x, &outs.y, None, 0, NMAX)?;
                let f16b = read(&mut gpu, &outs)?;
                let (rel, abs, nf) = distance(&iu4b, &f16b, &outs.init, ms, case.add());
                rep.line(format!(
                    "{} iu4-vs-f16 rel_l2={rel:.3e} abs_l2={abs:.3e} nonfinite={nf}",
                    tag("dist")
                ));
            }
            free_outs(&mut gpu, outs)?;
        }

        if has("time") {
            let outs = alloc_outs(&mut gpu, case, 1)?;
            load_x(&mut gpu, &gen_x(1, case.k))?;
            let e0 = gpu.hip.event_create()?;
            let e1 = gpu.hip.event_create()?;
            let time = |gpu: &mut Gpu, f: &mut dyn FnMut(&mut Gpu) -> R<()>| -> R<f64> {
                let mut ts = Vec::with_capacity(reps);
                for i in 0..reps + 3 {
                    gpu.hip.device_synchronize()?;
                    gpu.hip.event_record(&e0, None)?;
                    f(gpu)?;
                    gpu.hip.event_record(&e1, None)?;
                    gpu.hip.event_synchronize(&e1)?;
                    if i >= 3 {
                        ts.push(gpu.hip.event_elapsed_ms(&e0, &e1)? as f64);
                    }
                }
                Ok(median(&mut ts))
            };
            let tm = case.total_m();
            let floor_ms = case.weight_bytes() as f64 / (STREAM_GBS * 1e9) * 1e3;
            for n in [16usize, 64, 128] {
                let tf = |ms: f64| 2.0 * tm as f64 * n as f64 * case.k as f64 / (ms * 1e-3) / 1e12;
                let (f16cvt, f16) = if case.f16 {
                    (
                        time(&mut gpu, &mut |g| {
                            launch(g, case, Route::F16Exact, &x, &outs.y, None, 0, n)
                        })?,
                        time(&mut gpu, &mut |g| {
                            launch(g, case, Route::F16ExactKept, &x, &outs.y, None, 0, n)
                        })?,
                    )
                } else {
                    (f64::NAN, f64::NAN)
                };
                let (xv, ys) = views(case, &x, &outs.y, 0, n);
                let (iu4q, iu4g) = if case.iu4 {
                    let q = time(&mut gpu, &mut |g| {
                        launch(g, case, Route::Iu4, &x, &outs.y, None, 0, n)
                    })?;
                    let xq = gpu.ensure_int4_mmq_x(&xv, n, case.k)?;
                    (q, time(&mut gpu, &mut |g| iu4_gemms(g, case, xq, &ys, n))?)
                } else {
                    (f64::NAN, f64::NAN)
                };
                let (slabp, slabg) = if case.slab.is_some() {
                    let h = outs.h.as_ref();
                    let p = time(&mut gpu, &mut |g| {
                        launch(g, case, Route::Slab, &x, &outs.y, h, 0, n)
                    })?;
                    let sp = slab_prepare(&mut gpu, case, &xv, n)?;
                    let gm = time(&mut gpu, &mut |g| slab_consume(g, case, &sp, &ys, h, n))?;
                    (p, gm)
                } else {
                    (f64::NAN, f64::NAN)
                };
                rep.line(format!(
                    "time {} N={n} | f16 cvt+gemm {f16cvt:.4} | f16 gemm {f16:.4} ms ({:.1} TF, floor/t {:.2}) | iu4 quant+gemm {iu4q:.4} | iu4 gemm {iu4g:.4} ms ({:.1} TF, floor/t {:.2}) | slab prod+gemm {slabp:.4} | slab gemm {slabg:.4} | floor {floor_ms:.4} ms | x{} layers",
                    case.name,
                    tf(f16),
                    floor_ms / f16,
                    tf(iu4g),
                    floor_ms / iu4g,
                    case.layers
                ));
                timing_rows.push(format!(
                    "{},{},{},{},{f16cvt},{f16},{iu4q},{iu4g},{slabp},{slabg},{floor_ms}",
                    case.name, n, case.layers, tm
                ));
            }
            free_outs(&mut gpu, outs)?;
        }
    }
    if !timing_rows.is_empty() {
        let mut csv = String::from("shape,n,layers,total_m,f16_cvt_gemm_ms,f16_gemm_ms,iu4_quant_gemm_ms,iu4_gemm_ms,slab_prod_gemm_ms,slab_gemm_ms,floor_ms\n");
        for r in &timing_rows {
            csv.push_str(r);
            csv.push('\n');
        }
        std::fs::write(format!("{out}/timing.csv"), csv)?;
    }
    rep.line(format!("checks {} fails {}", rep.checks, rep.fails));
    rep.line(if rep.fails == 0 {
        "VERDICT PASS".into()
    } else {
        "VERDICT FAIL".into()
    });
    std::fs::write(format!("{out}/report.txt"), &rep.text)?;
    gpu.free_tensor(x)?;
    Ok(rep.fails)
}
