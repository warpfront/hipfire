// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// PeaceMaker FP8 verify GEMM family (gfx1201, `mq4_fp8_verify_pm_gfx1201`):
// byte oracle for row-count independence, graph capture, pack equivalence,
// numeric sanity and timing on the real 27B MQ4G256V2 verify shapes.
//
// Shapes (real model weights): qkvza (layer-0 DeltaNet wqkv/wz/w_beta/w_alpha,
// K 5120), dn_o (DeltaNet wo, residual, K 6144), gate_up (K 5120), down
// (residual, K 17408), qkv (first full-attention layer wq/wk/wv), fa_o (FA wo,
// residual, K 6144) and lm_head (`output`, `set` op, K 5120). Layer counts per
// verify step: 48, 48, 64, 64, 16, 16, 1.
//
// Activations are a seeded F32 panel of 128 rows: per-row magnitudes
// 2^-6..2^6, outlier columns, +-0, an all-zero row, an exact-negation row and
// a near-subnormal row (so the per-row E4M3 scale differs across rows of one
// launch). Residual outputs start from distinct nonzero values, overwrite
// outputs from a sentinel, and every output buffer ends in a 256-word canary
// tail: every comparison covers the COMPLETE buffers (valid rows, untouched
// rows past N, canary) at 0-byte tolerance.
//
// Scope: the production entries (`gemm_*_verify_exact`) only route to FP8
// while `gpu.fp8_verify_scope` is true. `Gpu::fp8_verify_enter` requires the
// `HIPFIRE_CB_VERIFY_FP8` flag; this harness sets the public field directly
// (and restores it) so the oracle runs with the flag off and the F16 baseline
// (`scope` off) stays available in the same process. The arch crate's
// enter/exit pairing is therefore NOT exercised here.
//
// Tests (`--tests`, default all):
//   rows  reference = the panel packed once (`fp8_verify_pack_x`, copied to an
//         owned buffer) and launched as 16-row chunks with tile (1,1) at row
//         offsets (the singleton). Candidates: the production entry (pack +
//         GEMM inside the scope) for every N = 1..128 on the panel prefix; the
//         direct `_verify_fp8` entries at selected N; every explicit tile
//         (1,1),(2,4),(2,8),(4,4),(4,8),(8,4),(8,8) pre-packed at
//         N = 1,16,17,33,65,100,128; production chunkings 48+48+32, 1+127,
//         63+65, 8x16 at row offsets; 128 one-row launches (production and
//         tile (8,8) pre-packed).
//   perm  random row permutations of the panel (activation rows and residual
//         initial values) give the correspondingly permuted reference rows.
//   graph the production launches (pack + GEMM) captured in a HIP graph
//         (single launches N = 128,100,65,33,16,1 and the 48+48+32 chunking in
//         one graph) and replayed twice equal the eager reference bytes.
//   pack  lean pack bytes + scales equal `prepare_mq4v2_fp8_x(x, n, k, 1)` for
//         N = 1, 16, 128 on every distinct K (not seed dependent beyond the
//         panel).
//   dist  (diagnostic, not gated in the VERDICT) FP8 vs the F16 `verify_exact`
//         relative L2 per shape; FP8 vs a CPU f64 reference of the contract
//         arithmetic on 64 sampled weight rows per matrix and all 128 tokens
//         (dequant (q-8)*sc, activations decoded from the packed bytes, scales
//         read back, exact products): per-token relative L2, `FAIL dist` when
//         above 1e-3. The near-subnormal row class (row % 16 == 13) is reported
//         separately and excluded from that gate (f32 scale denormals).
//   time  per shape at N = 16 and 128: fp8 gemm-only (prepacked, production
//         tile plan), fp8 pack+gemm (production entry), f16 gemm
//         (`verify_exact`, fp16 X cache kept; the head through the residual
//         entry) and f16 cvt+gemm (head through `lmhead_verify_exact`);
//         median of event timing; floor = weight bytes / 640 GB/s;
//         TFLOPS = 2*M*N*K/t; per-step sums over layer counts (six shapes,
//         and with lm_head). `--tile-sweep` times every explicit tile at
//         N = 16, 32, 64, 128.
//
// Outputs in `--out`: report.txt (one line per check + `VERDICT PASS/FAIL`,
// gated on rows/perm/graph/pack), timing.csv, identity.txt, digests.txt.
//
// Build: cargo build -p hipfire-arch-qwen35 --features lab,deltanet --release \
//          --example pm_fp8_verify_oracle -j12
// Run:   pm_fp8_verify_oracle --out <abs dir> [--model <path>]
//          [--ops qkvza,dn_o,gate_up,down,qkv,fa_o,lm_head] [--seeds 1,2]
//          [--tests rows,perm,graph,pack,dist,time] [--reps 30] [--warm 3]
//          [--tile-sweep]

use hipfire_arch_qwen35::load_qwen35_bundle;
use hipfire_arch_qwen35::qwen35::LayerWeights;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SequenceHint, SpecLoadCfg};
use rdna_compute::{
    fp8_verify_tile, DType, Fp8VerifyOp, Fp8VerifyTile, Fp8VerifyX, Gpu, GpuTensor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt::Write as _;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const NMAX: usize = 128;
const CANARY: usize = 256;
const SENTINEL: u32 = 0x7FBA_DBAD;
/// Weight-stream reference bandwidth (BATCH-OVERHEAD §2).
const STREAM_GBS: f64 = 640.0;
const DEFAULT_MODEL: &str =
    "/home/kaden/qcal/release-0.4.0/fp8-tp-ep-geo/models/qwen3.8-27b.mq4-xts";
const TILES: [(u8, u8); 7] = [(1, 1), (2, 4), (2, 8), (4, 4), (4, 8), (8, 4), (8, 8)];
/// CPU-reference sample: weight rows per matrix.
const CPU_ROWS: usize = 64;
const CPU_TOL: f64 = 1e-3;

fn tile(bt: u8, waves: u8) -> Fp8VerifyTile {
    Fp8VerifyTile { bt, waves }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fam {
    Qkvza,
    Qkv,
    GateUp,
    Residual,
    Head,
}

impl Fam {
    fn op(self) -> Fp8VerifyOp {
        match self {
            Fam::Qkvza => Fp8VerifyOp::Qkvza,
            Fam::Qkv => Fp8VerifyOp::Qkv,
            Fam::GateUp => Fp8VerifyOp::GateUp,
            Fam::Residual => Fp8VerifyOp::Residual,
            Fam::Head => Fp8VerifyOp::Set,
        }
    }
}

struct Case<'a> {
    name: &'static str,
    fam: Fam,
    w: Vec<&'a GpuTensor>,
    ms: Vec<usize>,
    k: usize,
    layers: usize,
    /// One of the six per-step shapes (lm_head runs once per step on top).
    in_step: bool,
}

impl Case<'_> {
    fn total_m(&self) -> usize {
        self.ms.iter().sum()
    }
    fn weight_bytes(&self) -> usize {
        self.total_m() * (self.k / 256) * 136
    }
    fn resid(&self) -> bool {
        self.fam == Fam::Residual
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Pre-packed owned panel, explicit tile (None = production plan).
    Pre(Option<Fp8VerifyTile>),
    /// Production entry (`verify_exact`) inside the FP8 scope (pack + GEMM).
    Prod,
    /// Direct `_verify_fp8` twin (pack + GEMM), scope off.
    Direct,
    /// F16 `verify_exact`, scope off, fp16 cache invalidated first.
    F16,
    /// F16 `verify_exact`, scope off, fp16 cache kept (head: residual entry).
    F16Kept,
}

/// Owned packed activation panel (stable across other packs on the Gpu).
struct Panel {
    x8: GpuTensor,
    s: GpuTensor,
    k: usize,
}

impl Panel {
    fn view(&self, row0: usize, n: usize) -> Fp8VerifyX {
        // SAFETY: only pointer arithmetic inside the owned allocations.
        let x8 = unsafe { (self.x8.buf.as_ptr() as *mut u8).add(row0 * self.k) } as *mut c_void;
        let s = unsafe { (self.s.buf.as_ptr() as *mut u8).add(row0 * 4) } as *mut c_void;
        Fp8VerifyX {
            x8,
            scales: s,
            n,
            k: self.k,
        }
    }
}

fn call_exact(
    gpu: &mut Gpu,
    c: &Case,
    x: &GpuTensor,
    y: &[GpuTensor],
    n: usize,
    kept: bool,
) -> Result<(), rdna_compute::HipError> {
    let (w, m, k) = (&c.w, &c.ms, c.k);
    match c.fam {
        Fam::Qkvza => gpu.gemm_qkvza_mq4g256v2_verify_exact(
            w[0], w[1], w[2], w[3], x, &y[0], &y[1], &y[2], &y[3], m[0], m[1], m[2], m[3], k, n,
        ),
        Fam::Qkv => gpu.gemm_qkv_mq4g256v2_verify_exact(
            w[0], w[1], w[2], x, &y[0], &y[1], &y[2], m[0], m[1], m[2], k, n,
        ),
        Fam::GateUp => {
            gpu.gemm_gate_up_mq4g256v2_verify_exact(w[0], w[1], x, &y[0], &y[1], m[0], m[1], k, n)
        }
        Fam::Residual => gpu.gemm_mq4g256v2_residual_verify_exact(w[0], x, &y[0], m[0], k, n),
        Fam::Head if kept => gpu.gemm_mq4g256v2_residual_verify_exact(w[0], x, &y[0], m[0], k, n),
        Fam::Head => gpu.gemm_mq4g256v2_lmhead_verify_exact(w[0], x, &y[0], m[0], k, n),
    }
}

fn call_fp8(
    gpu: &mut Gpu,
    c: &Case,
    x: &GpuTensor,
    y: &[GpuTensor],
    n: usize,
) -> Result<(), rdna_compute::HipError> {
    let (w, m, k) = (&c.w, &c.ms, c.k);
    match c.fam {
        Fam::Qkvza => gpu.gemm_qkvza_mq4g256v2_verify_fp8(
            w[0], w[1], w[2], w[3], x, &y[0], &y[1], &y[2], &y[3], m[0], m[1], m[2], m[3], k, n,
        ),
        Fam::Qkv => gpu.gemm_qkv_mq4g256v2_verify_fp8(
            w[0], w[1], w[2], x, &y[0], &y[1], &y[2], m[0], m[1], m[2], k, n,
        ),
        Fam::GateUp => {
            gpu.gemm_gate_up_mq4g256v2_verify_fp8(w[0], w[1], x, &y[0], &y[1], m[0], m[1], k, n)
        }
        Fam::Residual => gpu.gemm_mq4g256v2_residual_verify_fp8(w[0], x, &y[0], m[0], k, n),
        Fam::Head => gpu.gemm_mq4g256v2_lmhead_verify_fp8(w[0], x, &y[0], m[0], k, n),
    }
}

/// One launch of `kind` over panel rows [row0, row0+n) of the F32 panel `x`
/// (and of the owned packed `panel` for `Kind::Pre`), outputs at the same row
/// offsets of `y`.
#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &mut Gpu,
    case: &Case,
    kind: Kind,
    x: &GpuTensor,
    panel: &Panel,
    y: &[GpuTensor],
    row0: usize,
    n: usize,
) -> R<()> {
    let k = case.k;
    let ys: Vec<GpuTensor> = y
        .iter()
        .zip(&case.ms)
        .map(|(t, &m)| t.sub_offset(row0 * m, n * m))
        .collect();
    let xv = x.sub_offset(row0 * k, n * k);
    let prev = gpu.fp8_verify_scope;
    let r = match kind {
        Kind::Pre(t) => {
            let xq = panel.view(row0, n);
            let yr: Vec<&GpuTensor> = ys.iter().collect();
            gpu.gemm_mq4g256v2_verify_fp8_prepacked(
                case.fam.op(),
                t,
                &case.w,
                &xq,
                &yr,
                &case.ms,
                k,
                n,
            )
        }
        Kind::Prod => {
            gpu.fp8_verify_scope = true;
            call_exact(gpu, case, &xv, &ys, n, false)
        }
        Kind::Direct => {
            gpu.fp8_verify_scope = false;
            call_fp8(gpu, case, &xv, &ys, n)
        }
        Kind::F16 => {
            gpu.fp8_verify_scope = false;
            gpu.invalidate_fp16_cache();
            call_exact(gpu, case, &xv, &ys, n, false)
        }
        Kind::F16Kept => {
            gpu.fp8_verify_scope = false;
            call_exact(gpu, case, &xv, &ys, n, true)
        }
    };
    gpu.fp8_verify_scope = prev;
    r?;
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

/// Word-wise 64-bit digest (fast over the 100+ MB lm_head buffers; not FNV).
fn digest(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut it = bytes.chunks_exact(8);
    for c in &mut it {
        h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
    }
    for &b in it.remainder() {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Seeded F32 activations [NMAX, k] (see the header).
fn gen_x(seed: u64, k: usize) -> Vec<f32> {
    let mut x = vec![0f32; NMAX * k];
    for r in 0..NMAX {
        let row_exp = ((hash3(seed, r as u64, 0xE) % 13) as i32) - 6;
        for c in 0..k {
            let v = match r % 16 {
                // Exact negation of the previous row.
                7 => -x[(r - 1) * k + c],
                // All-zero row (unit scale branch).
                11 => 0.0,
                // Tiny row: values near the f32 subnormal range.
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
                    // Outlier columns (activation-like heavy tail).
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

/// Distinct nonzero residual initial values (incl. -0) / sentinel, + canary.
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
}

fn alloc_outs(gpu: &mut Gpu, case: &Case, seed: u64) -> R<Outs> {
    let mut o = Outs {
        y: Vec::new(),
        init_dev: Vec::new(),
        init: Vec::new(),
    };
    for (i, &m) in case.ms.iter().enumerate() {
        let init = gen_init(seed, i, m, case.resid());
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

/// Expected buffers when rows [0, n) hold `refb` and everything else (rows
/// past n, canary) is untouched.
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

fn matches_prefix(
    got: &[Vec<u8>],
    refb: &[Vec<u8>],
    init: &[Vec<u8>],
    ms: &[usize],
    n: usize,
) -> bool {
    got.iter()
        .zip(refb)
        .zip(init)
        .zip(ms)
        .all(|(((g, r), i), &m)| {
            let cut = n * m * 4;
            g[..cut] == r[..cut] && g[cut..] == i[cut..]
        })
}

fn first_diff(got: &[Vec<u8>], want: &[Vec<u8>], ms: &[usize]) -> Option<String> {
    let mut total = 0usize;
    let mut first = None;
    for (o, ((g, w), &m)) in got.iter().zip(want).zip(ms).enumerate() {
        if g == w {
            continue;
        }
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

#[derive(Default)]
struct Report {
    text: String,
    /// group -> (checks, fails)
    groups: BTreeMap<&'static str, (usize, usize)>,
}

impl Report {
    fn line(&mut self, s: String) {
        println!("{s}");
        self.text.push_str(&s);
        self.text.push('\n');
    }
    fn tally(&mut self, group: &'static str, ok: bool, detail: String) -> bool {
        let e = self.groups.entry(group).or_default();
        e.0 += 1;
        if !ok {
            e.1 += 1;
        }
        let s = format!("{} {detail}", if ok { "PASS" } else { "FAIL" });
        self.line(s);
        ok
    }
    /// Complete-buffer compare against the reference rows [0, n) + untouched rest.
    #[allow(clippy::too_many_arguments)]
    fn cmp_prefix(
        &mut self,
        group: &'static str,
        tag: &str,
        got: &[Vec<u8>],
        refb: &[Vec<u8>],
        init: &[Vec<u8>],
        ms: &[usize],
        n: usize,
    ) -> bool {
        if matches_prefix(got, refb, init, ms, n) {
            return self.tally(group, true, tag.to_string());
        }
        let want = expect(refb, init, ms, n);
        let d = first_diff(got, &want, ms).unwrap_or_else(|| "mismatch".into());
        self.tally(group, false, format!("{tag}: {d}"))
    }
    fn cmp_want(
        &mut self,
        group: &'static str,
        tag: &str,
        got: &[Vec<u8>],
        want: &[Vec<u8>],
        ms: &[usize],
    ) -> bool {
        match first_diff(got, want, ms) {
            None => self.tally(group, true, tag.to_string()),
            Some(d) => self.tally(group, false, format!("{tag}: {d}")),
        }
    }
    fn gated_fails(&self) -> usize {
        ["rows", "perm", "graph", "pack"]
            .iter()
            .map(|g| self.groups.get(g).map_or(0, |e| e.1))
            .sum()
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Numeric distance of `got` from `want` over rows [0, NMAX) of every output
/// (finite pairs; for residual outputs the init is subtracted from both).
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

fn f16_to_f32(h: u16) -> f32 {
    let s = if h >> 15 == 1 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    s * match e {
        0 => m * 2f32.powi(-24),
        31 => {
            if m == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// OCP E4M3 (fn): bias 7, no infinities, S.1111.111 = NaN.
fn e4m3(b: u8) -> f64 {
    let s = if b >> 7 == 1 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f64;
    if e == 0xF && m == 7.0 {
        return f64::NAN;
    }
    s * if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

/// Copy `bytes` from a raw device address into `scratch` and read them back.
fn snap(gpu: &mut Gpu, scratch: &GpuTensor, src: *mut c_void, bytes: usize) -> R<Vec<u8>> {
    gpu.hip.device_synchronize()?;
    // SAFETY: `src` is a live pack buffer of this Gpu holding at least `bytes`;
    // `scratch` is a distinct allocation of at least that size.
    unsafe {
        gpu.hip
            .memcpy_dtod_raw(scratch.buf.as_ptr(), src as *const c_void, bytes)?
    };
    gpu.hip.device_synchronize()?;
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh_at(&mut v, &scratch.buf, 0)?;
    Ok(v)
}

fn load_x(gpu: &mut Gpu, x: &GpuTensor, v: &[f32]) -> R<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&x.buf, &bytes)?;
    gpu.hip.device_synchronize()?;
    Ok(())
}

/// Pack the whole 128-row panel through the lean pack once and copy it into
/// the owned `panel` buffers.
fn fill_panel(gpu: &mut Gpu, panel: &Panel, x: &GpuTensor, k: usize) -> R<()> {
    let xv = x.sub_offset(0, NMAX * k);
    let px = gpu.fp8_verify_pack_x(&xv, NMAX, k)?;
    gpu.hip.device_synchronize()?;
    // SAFETY: pack buffers are live until the next pack; destinations are the
    // owned panel allocations sized for NMAX*kmax bytes / NMAX words.
    unsafe {
        gpu.hip
            .memcpy_dtod_raw(panel.x8.buf.as_ptr(), px.x8 as *const c_void, NMAX * k)?;
        gpu.hip
            .memcpy_dtod_raw(panel.s.buf.as_ptr(), px.scales as *const c_void, NMAX * 4)?;
    }
    gpu.hip.device_synchronize()?;
    Ok(())
}

/// CPU f64 reference of the contract arithmetic on sampled weight rows vs the
/// GPU singleton reference buffers. Returns (max token rel-L2 outside the tiny
/// row class, its token, max token rel-L2 over the tiny class, rows sampled).
fn cpu_reference(
    gpu: &mut Gpu,
    case: &Case,
    x8: &[u8],
    scales: &[f32],
    refb: &[Vec<u8>],
    init: &[Vec<u8>],
) -> R<(f64, usize, f64, usize)> {
    let k = case.k;
    let lut: Vec<f64> = (0..256).map(|b| e4m3(b as u8)).collect();
    let xdec: Vec<f64> = x8[..NMAX * k].iter().map(|&b| lut[b as usize]).collect();
    let row_bytes = (k / 256) * 136;
    let mut num = vec![0f64; NMAX];
    let mut den = vec![0f64; NMAX];
    let mut rows_total = 0usize;
    for (oi, &m) in case.ms.iter().enumerate() {
        let mut rows: Vec<usize> = [0usize, 1, 2, 3, m.saturating_sub(1), m.saturating_sub(2)]
            .iter()
            .map(|&r| r.min(m - 1))
            .collect();
        for i in 0..CPU_ROWS - rows.len().min(CPU_ROWS) {
            rows.push((hash3(0xC0FFEE, oi as u64, i as u64) % m as u64) as usize);
        }
        rows.sort_unstable();
        rows.dedup();
        for &r in &rows {
            rows_total += 1;
            let mut rb = vec![0u8; row_bytes];
            gpu.hip
                .memcpy_dtoh_at(&mut rb, &case.w[oi].buf, r * row_bytes)?;
            let mut t_acc = vec![0f64; NMAX];
            for g in 0..k / 256 {
                let base = g * 136;
                let sc = [
                    f16_to_f32(u16::from_le_bytes([rb[base], rb[base + 1]])) as f64,
                    f16_to_f32(u16::from_le_bytes([rb[base + 4], rb[base + 5]])) as f64,
                ];
                let codes = &rb[base + 8..base + 136];
                for (h, &sch) in sc.iter().enumerate() {
                    let mut wq = [0f64; 128];
                    for (j, w) in wq.iter_mut().enumerate() {
                        let kk = 128 * h + j;
                        let byte = codes[kk / 2];
                        let q = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
                        *w = q as f64 - 8.0;
                    }
                    let c0 = 256 * g + 128 * h;
                    for t in 0..NMAX {
                        let xr = &xdec[t * k + c0..t * k + c0 + 128];
                        let p: f64 = xr.iter().zip(&wq).map(|(a, b)| a * b).sum();
                        t_acc[t] += sch * p;
                    }
                }
            }
            for t in 0..NMAX {
                let idx = (t * m + r) * 4;
                let got = f32::from_le_bytes(refb[oi][idx..idx + 4].try_into().unwrap()) as f64;
                let yold = if case.resid() {
                    f32::from_le_bytes(init[oi][idx..idx + 4].try_into().unwrap()) as f64
                } else {
                    0.0
                };
                let want = yold + scales[t] as f64 * t_acc[t];
                let d = got - want;
                num[t] += if d.is_nan() { f64::INFINITY } else { d * d };
                den[t] += want * want;
            }
        }
    }
    let rel = |t: usize| {
        if den[t] > 0.0 {
            (num[t] / den[t]).sqrt()
        } else if num[t] == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    };
    let (mut worst, mut worst_t, mut tiny) = (0f64, 0usize, 0f64);
    for t in 0..NMAX {
        let r = rel(t);
        if t % 16 == 13 {
            tiny = tiny.max(r);
        } else if !(r <= worst) {
            worst = r;
            worst_t = t;
        }
    }
    Ok((worst, worst_t, tiny, rows_total))
}

struct Args {
    model: String,
    out: String,
    ops: Vec<String>,
    seeds: Vec<u64>,
    tests: Vec<String>,
    reps: usize,
    warm: usize,
    tile_sweep: bool,
}

fn parse_args() -> R<Args> {
    let list = |v: &str| {
        v.split(',')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect::<Vec<_>>()
    };
    let mut a = Args {
        model: DEFAULT_MODEL.into(),
        out: String::new(),
        ops: list("qkvza,dn_o,gate_up,down,qkv,fa_o,lm_head"),
        seeds: vec![1, 2],
        tests: list("rows,perm,graph,pack,dist,time"),
        reps: 30,
        warm: 3,
        tile_sweep: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        if k == "--tile-sweep" {
            a.tile_sweep = true;
            continue;
        }
        let v = it.next().ok_or_else(|| format!("{k}: missing value"))?;
        match k.as_str() {
            "--model" => a.model = v,
            "--out" => a.out = v,
            "--ops" => a.ops = list(&v),
            "--seeds" => a.seeds = v.split(',').map(|s| s.parse()).collect::<Result<_, _>>()?,
            "--tests" => a.tests = list(&v),
            "--reps" => a.reps = v.parse()?,
            "--warm" => a.warm = v.parse()?,
            _ => return Err(format!("unknown argument {k}").into()),
        }
    }
    if !a.out.starts_with('/') {
        return Err("absolute --out <dir> is required".into());
    }
    for t in &a.tests {
        if !["rows", "perm", "graph", "pack", "dist", "time"].contains(&t.as_str()) {
            return Err(format!("unknown test {t}").into());
        }
    }
    Ok(a)
}

fn main() {
    match run() {
        Ok(0) => std::process::exit(0),
        Ok(_) => std::process::exit(1),
        Err(e) => {
            eprintln!("pm_fp8_verify_oracle: {e}");
            std::process::exit(2);
        }
    }
}

fn cmd_out(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn run() -> R<usize> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)?;
    let out = args.out.clone();
    let has = |t: &str| args.tests.iter().any(|x| x == t);
    let mut rep = Report::default();

    let mut gpu = Gpu::init()?;
    rep.line(format!("arch {} model {}", gpu.arch, args.model));
    if gpu.arch != "gfx1201" {
        return Err(format!("gfx1201 required, got {}", gpu.arch).into());
    }
    let mut env: Vec<String> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("HIPFIRE_")
                || k.starts_with("HIP_")
                || k.starts_with("ROCR_")
                || k.starts_with("HSA_")
        })
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    env.sort();
    rep.line(format!("env {env:?}"));
    rep.line(format!(
        "flags cb_verify_fp8={} cb_verify_pm={} {:?}",
        gpu.flags.cb_verify_fp8, gpu.flags.cb_verify_pm, gpu.flags
    ));
    rep.line(format!(
        "args ops={:?} seeds={:?} tests={:?} reps={} warm={} tile_sweep={}",
        args.ops, args.seeds, args.tests, args.reps, args.warm, args.tile_sweep
    ));

    // identity.txt: binary sha256, git describe, env.
    {
        let exe = std::env::current_exe()?;
        let sha = hex(&Sha256::digest(std::fs::read(&exe)?));
        let git = cmd_out(
            "git",
            &[
                "-C",
                env!("CARGO_MANIFEST_DIR"),
                "describe",
                "--always",
                "--dirty",
                "--tags",
            ],
        );
        let mut id = String::new();
        let _ = writeln!(id, "binary {}", exe.display());
        let _ = writeln!(id, "binary_sha256 {sha}");
        let _ = writeln!(id, "git_describe {git}");
        let _ = writeln!(id, "arch {}", gpu.arch);
        let _ = writeln!(id, "model {}", args.model);
        let _ = writeln!(id, "argv {:?}", std::env::args().collect::<Vec<_>>());
        for e in &env {
            let _ = writeln!(id, "env {e}");
        }
        std::fs::write(format!("{out}/identity.txt"), id)?;
    }

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
    rep.line(format!("mq4v2_symmetric={}", gpu.mq4v2_symmetric));
    if !gpu.mq4v2_symmetric {
        return Err(
            "model is not MQ4G256V2 symmetric (FP8 verify requires symmetric weights)".into(),
        );
    }
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
    let all = vec![
        Case {
            name: "qkvza",
            fam: Fam::Qkvza,
            w: vec![&dn.wqkv.buf, &dn.wz.buf, &dn.w_beta.buf, &dn.w_alpha.buf],
            ms: vec![dn.wqkv.m, dn.wz.m, dn.w_beta.m, dn.w_alpha.m],
            k: kd,
            layers: n_dn,
            in_step: true,
        },
        Case {
            name: "dn_o",
            fam: Fam::Residual,
            w: vec![&dn.wo.buf],
            ms: vec![dn.wo.m],
            k: dn.wo.k,
            layers: n_dn,
            in_step: true,
        },
        Case {
            name: "gate_up",
            fam: Fam::GateUp,
            w: vec![&dn.w_gate.buf, &dn.w_up.buf],
            ms: vec![dn.w_gate.m, dn.w_up.m],
            k: kd,
            layers: n_dn + n_fa,
            in_step: true,
        },
        Case {
            name: "down",
            fam: Fam::Residual,
            w: vec![&dn.w_down.buf],
            ms: vec![dn.w_down.m],
            k: dn.w_down.k,
            layers: n_dn + n_fa,
            in_step: true,
        },
        Case {
            name: "qkv",
            fam: Fam::Qkv,
            w: vec![&fa.wq.buf, &fa.wk.buf, &fa.wv.buf],
            ms: vec![fa.wq.m, fa.wk.m, fa.wv.m],
            k: kd,
            layers: n_fa,
            in_step: true,
        },
        Case {
            name: "fa_o",
            fam: Fam::Residual,
            w: vec![&fa.wo.buf],
            ms: vec![fa.wo.m],
            k: fa.wo.k,
            layers: n_fa,
            in_step: true,
        },
        Case {
            name: "lm_head",
            fam: Fam::Head,
            w: vec![&b.weights.output.buf],
            ms: vec![b.weights.output.m],
            k: kd,
            layers: 1,
            in_step: false,
        },
    ];
    for c in &all {
        let dts: Vec<String> = c.w.iter().map(|t| format!("{:?}", t.dtype)).collect();
        rep.line(format!(
            "shape {} fam={:?} ms={:?} total_m={} k={} layers={} weight_MB={:.1} dtype={dts:?}",
            c.name,
            c.fam,
            c.ms,
            c.total_m(),
            c.k,
            c.layers,
            c.weight_bytes() as f64 / 1e6
        ));
        if c.k % 256 != 0 {
            return Err(format!("{}: K {} not a multiple of 256", c.name, c.k).into());
        }
        for (w, &m) in c.w.iter().zip(&c.ms) {
            if w.buf.size() < m * (c.k / 256) * 136 {
                return Err(format!("{}: weight buffer smaller than m*K/256*136", c.name).into());
            }
        }
    }
    for o in &args.ops {
        if !all.iter().any(|c| c.name == o) {
            return Err(format!("unknown op {o}").into());
        }
    }
    let cases: Vec<&Case> = all
        .iter()
        .filter(|c| args.ops.iter().any(|o| o == c.name))
        .collect();
    let kmax = cases.iter().map(|c| c.k).max().unwrap_or(kd);
    let x = gpu.zeros(&[NMAX * kmax], DType::F32)?;
    let mk_panel = |gpu: &mut Gpu, k: usize| -> R<Panel> {
        Ok(Panel {
            x8: gpu.zeros(&[(NMAX * kmax) / 4 + 1], DType::F32)?,
            s: gpu.zeros(&[NMAX], DType::F32)?,
            k,
        })
    };
    let mut panel = mk_panel(&mut gpu, kmax)?;
    let mut digests = String::from("# shape seed N per-output word-digests (production entry, scope on); ref16 = singleton reference\n");
    let mut timing: Vec<String> = Vec::new();
    // (variant, n) -> [(ms, layers, in_step)] for the per-step sums.
    let mut steps: BTreeMap<(&'static str, usize), Vec<(f64, usize, bool)>> = BTreeMap::new();
    let receipt0 = gpu.fp8_verify_receipt();

    // ---- pack: lean pack bytes+scales vs the existing prepare_mq4v2_fp8_x(scale_mode 1).
    if has("pack") {
        let tmp = mk_panel(&mut gpu, kmax)?;
        let mut ks: Vec<usize> = cases.iter().map(|c| c.k).collect();
        ks.sort_unstable();
        ks.dedup();
        for &k in &ks {
            for &seed in &args.seeds {
                load_x(&mut gpu, &x, &gen_x(seed, k))?;
                for n in [1usize, 16, 128] {
                    let xv = x.sub_offset(0, n * k);
                    let lean = gpu.fp8_verify_pack_x(&xv, n, k)?;
                    let lx = snap(&mut gpu, &panel.x8, lean.x8, n * k)?;
                    let ls = snap(&mut gpu, &panel.s, lean.scales, n * 4)?;
                    let old = gpu.prepare_mq4v2_fp8_x(&xv, n, k, 1)?;
                    let ox = snap(&mut gpu, &tmp.x8, old.x_fp8, n * k)?;
                    let os = snap(&mut gpu, &tmp.s, old.row_scales, n * 4)?;
                    let okx = lx == ox;
                    let oks = ls == os;
                    let detail = |what: &str, a: &[u8], b: &[u8]| -> String {
                        match a.iter().zip(b).position(|(p, q)| p != q) {
                            Some(i) => format!(
                                "{what} first diff byte {i}: lean {:02x} existing {:02x}",
                                a[i], b[i]
                            ),
                            None => String::new(),
                        }
                    };
                    let tag = format!("pack k={k} seed{seed} N={n} x8+scales vs prepare_mq4v2_fp8_x(scale_mode 1)");
                    if okx && oks {
                        rep.tally("pack", true, tag);
                    } else {
                        let d =
                            format!("{} {}", detail("x8", &lx, &ox), detail("scales", &ls, &os));
                        rep.tally("pack", false, format!("{tag}: {d}"));
                    }
                }
            }
        }
        gpu.free_tensor(tmp.x8)?;
        gpu.free_tensor(tmp.s)?;
    }

    for case in &cases {
        let (ms, k) = (&case.ms, case.k);
        panel.k = k;
        let need_ref = has("rows") || has("perm") || has("graph") || has("dist");
        for &seed in &args.seeds {
            if !need_ref {
                break;
            }
            let xv = gen_x(seed, k);
            load_x(&mut gpu, &x, &xv)?;
            fill_panel(&mut gpu, &panel, &x, k)?;
            let tag = |g: &str, t: &str| format!("{} seed{} {} {}", case.name, seed, g, t);
            let outs = alloc_outs(&mut gpu, case, seed)?;

            // Singleton reference: tile (1,1), 16-row chunks at row offsets.
            reset(&mut gpu, &outs)?;
            for c in 0..NMAX / 16 {
                launch(
                    &mut gpu,
                    case,
                    Kind::Pre(Some(tile(1, 1))),
                    &x,
                    &panel,
                    &outs.y,
                    c * 16,
                    16,
                )?;
            }
            let refb = read(&mut gpu, &outs)?;
            reset(&mut gpu, &outs)?;
            for c in 0..NMAX / 16 {
                launch(
                    &mut gpu,
                    case,
                    Kind::Pre(Some(tile(1, 1))),
                    &x,
                    &panel,
                    &outs.y,
                    c * 16,
                    16,
                )?;
            }
            let got = read(&mut gpu, &outs)?;
            rep.cmp_prefix(
                "rows",
                &tag("rows", "ref16 rerun"),
                &got,
                &refb,
                &outs.init,
                ms,
                NMAX,
            );
            // Reference must have written every row (no sentinel left in rows < NMAX).
            let untouched = refb.iter().zip(ms).any(|(r, &m)| {
                r[..NMAX * m * 4]
                    .chunks_exact(4)
                    .any(|w| w == SENTINEL.to_le_bytes())
                    && !case.resid()
            });
            rep.tally(
                "rows",
                !untouched,
                tag("rows", "ref16 wrote every output word"),
            );
            let rh: Vec<String> = refb.iter().map(|r| format!("{:016x}", digest(r))).collect();
            let xh = digest(&xv.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());
            rep.line(format!(
                "{} x_digest={xh:016x} ref16_digest={rh:?}",
                tag("inputs", "")
            ));
            let _ = writeln!(digests, "{} {seed} ref16 {}", case.name, rh.join(","));

            if has("rows") {
                // Production entry, every N = 1..128 on the panel prefix.
                for n in 1..=NMAX {
                    reset(&mut gpu, &outs)?;
                    launch(&mut gpu, case, Kind::Prod, &x, &panel, &outs.y, 0, n)?;
                    let got = read(&mut gpu, &outs)?;
                    let d: Vec<String> =
                        got.iter().map(|g| format!("{:016x}", digest(g))).collect();
                    let _ = writeln!(digests, "{} {seed} {n} {}", case.name, d.join(","));
                    rep.cmp_prefix(
                        "rows",
                        &tag("rows", &format!("prod N={n}")),
                        &got,
                        &refb,
                        &outs.init,
                        ms,
                        n,
                    );
                }
                // Direct `_verify_fp8` entries.
                for n in [1usize, 16, 17, 33, 64, 65, 100, 128] {
                    reset(&mut gpu, &outs)?;
                    launch(&mut gpu, case, Kind::Direct, &x, &panel, &outs.y, 0, n)?;
                    rep.cmp_prefix(
                        "rows",
                        &tag("rows", &format!("direct N={n}")),
                        &read(&mut gpu, &outs)?,
                        &refb,
                        &outs.init,
                        ms,
                        n,
                    );
                }
                // Explicit tiles, pre-packed.
                for &(bt, w) in &TILES {
                    for n in [1usize, 16, 17, 33, 65, 100, 128] {
                        reset(&mut gpu, &outs)?;
                        launch(
                            &mut gpu,
                            case,
                            Kind::Pre(Some(tile(bt, w))),
                            &x,
                            &panel,
                            &outs.y,
                            0,
                            n,
                        )?;
                        rep.cmp_prefix(
                            "rows",
                            &tag("rows", &format!("tile ({bt},{w}) N={n}")),
                            &read(&mut gpu, &outs)?,
                            &refb,
                            &outs.init,
                            ms,
                            n,
                        );
                    }
                }
                // Chunkings at row offsets (production entry per chunk).
                let chunkings: [(&str, Vec<(usize, usize)>); 4] = [
                    ("48+48+32", vec![(0, 48), (48, 48), (96, 32)]),
                    ("1+127", vec![(0, 1), (1, 127)]),
                    ("63+65", vec![(0, 63), (63, 65)]),
                    ("8x16", (0..8).map(|c| (c * 16, 16)).collect()),
                ];
                for (name, ch) in &chunkings {
                    reset(&mut gpu, &outs)?;
                    for &(r0, n) in ch {
                        launch(&mut gpu, case, Kind::Prod, &x, &panel, &outs.y, r0, n)?;
                    }
                    rep.cmp_prefix(
                        "rows",
                        &tag("rows", &format!("chunks prod {name}")),
                        &read(&mut gpu, &outs)?,
                        &refb,
                        &outs.init,
                        ms,
                        NMAX,
                    );
                }
                // One-row launches.
                for (name, kind) in [
                    ("prod", Kind::Prod),
                    ("tile(8,8) prepacked", Kind::Pre(Some(tile(8, 8)))),
                ] {
                    reset(&mut gpu, &outs)?;
                    for r in 0..NMAX {
                        launch(&mut gpu, case, kind, &x, &panel, &outs.y, r, 1)?;
                    }
                    rep.cmp_prefix(
                        "rows",
                        &tag("rows", &format!("one-row x128 {name}")),
                        &read(&mut gpu, &outs)?,
                        &refb,
                        &outs.init,
                        ms,
                        NMAX,
                    );
                }
            }

            if has("perm") {
                for n in [16usize, 33, 64, 100, 128] {
                    let mut pi: Vec<usize> = (0..n).collect();
                    for i in (1..n).rev() {
                        let j = (hash3(seed, n as u64, i as u64) % (i as u64 + 1)) as usize;
                        pi.swap(i, j);
                    }
                    let mut xp = xv.clone();
                    for (i, &p) in pi.iter().enumerate() {
                        xp[i * k..(i + 1) * k].copy_from_slice(&xv[p * k..(p + 1) * k]);
                    }
                    load_x(&mut gpu, &x, &xp)?;
                    let mut init_p = outs.init.clone();
                    let mut want = outs.init.clone();
                    for (o, &m) in ms.iter().enumerate() {
                        for (i, &p) in pi.iter().enumerate() {
                            let (d, s) = (i * m * 4..(i + 1) * m * 4, p * m * 4..(p + 1) * m * 4);
                            init_p[o][d.clone()].copy_from_slice(&outs.init[o][s.clone()]);
                            want[o][d].copy_from_slice(&refb[o][s]);
                        }
                        gpu.hip.memcpy_htod(&outs.y[o].buf, &init_p[o])?;
                    }
                    launch(&mut gpu, case, Kind::Prod, &x, &panel, &outs.y, 0, n)?;
                    let got = read(&mut gpu, &outs)?;
                    rep.cmp_want(
                        "perm",
                        &tag("perm", &format!("prod N={n} permuted rows")),
                        &got,
                        &want,
                        ms,
                    );
                }
                load_x(&mut gpu, &x, &xv)?;
            }

            if has("graph") {
                let sets: [(&str, Vec<(usize, usize)>); 7] = [
                    ("N=128", vec![(0, 128)]),
                    ("N=100", vec![(0, 100)]),
                    ("N=65", vec![(0, 65)]),
                    ("N=33", vec![(0, 33)]),
                    ("N=16", vec![(0, 16)]),
                    ("N=1", vec![(0, 1)]),
                    ("48+48+32", vec![(0, 48), (48, 48), (96, 32)]),
                ];
                for (name, ch) in &sets {
                    let n_total: usize = ch.iter().map(|c| c.1).sum();
                    let stream = gpu.hip.stream_create()?;
                    gpu.active_stream = Some(stream);
                    // Eager prewarm: kernel load + scratch growth, and the eager bytes.
                    reset(&mut gpu, &outs)?;
                    for &(r0, n) in ch {
                        launch(&mut gpu, case, Kind::Prod, &x, &panel, &outs.y, r0, n)?;
                    }
                    let eager = read(&mut gpu, &outs)?;
                    let eager_ok = rep.cmp_prefix(
                        "graph",
                        &tag("graph", &format!("{name} eager (prewarm)")),
                        &eager,
                        &refb,
                        &outs.init,
                        ms,
                        n_total,
                    );
                    reset(&mut gpu, &outs)?;
                    let dev = gpu.device_id;
                    gpu.graphs.begin_graph_capture(
                        &gpu.hip,
                        dev,
                        gpu.active_stream.as_ref().unwrap(),
                    )?;
                    let mut lr: R<()> = Ok(());
                    for &(r0, n) in ch {
                        lr = launch(&mut gpu, case, Kind::Prod, &x, &panel, &outs.y, r0, n);
                        if lr.is_err() {
                            break;
                        }
                    }
                    let er = gpu.graphs.end_graph_capture(
                        &gpu.hip,
                        dev,
                        gpu.active_stream.as_ref().unwrap(),
                    );
                    lr?;
                    er?;
                    // Capture must not have executed anything.
                    let not_run = first_diff(&read(&mut gpu, &outs)?, &outs.init, ms).is_none();
                    rep.tally(
                        "graph",
                        not_run,
                        tag("graph", &format!("{name} capture executed nothing")),
                    );
                    let mut all_ok = eager_ok && not_run;
                    for i in 0..2 {
                        reset(&mut gpu, &outs)?;
                        gpu.graphs.graph_launch(
                            &gpu.hip,
                            dev,
                            gpu.active_stream.as_ref().unwrap(),
                        )?;
                        let got = read(&mut gpu, &outs)?;
                        all_ok &= rep.cmp_prefix(
                            "graph",
                            &tag("graph", &format!("{name} replay{i} == eager bytes")),
                            &got,
                            &eager,
                            &outs.init,
                            ms,
                            n_total,
                        );
                    }
                    gpu.hip.device_synchronize()?;
                    gpu.graphs.drop_captured_graph(&gpu.hip, dev);
                    let s = gpu.active_stream.take().unwrap();
                    gpu.hip.stream_destroy(s)?;
                    rep.line(format!(
                        "{} {name} {}",
                        tag("graph", "summary"),
                        if all_ok {
                            "eager+2 replays same"
                        } else {
                            "DIFF"
                        }
                    ));
                }
            }

            if has("dist") {
                // F16 verify_exact (scope off) over all rows vs the FP8 singleton.
                reset(&mut gpu, &outs)?;
                launch(&mut gpu, case, Kind::F16, &x, &panel, &outs.y, 0, NMAX)?;
                let f16b = read(&mut gpu, &outs)?;
                let (rel, abs, nf) = distance(&refb, &f16b, &outs.init, ms, case.resid());
                rep.line(format!(
                    "{} fp8-vs-f16 rel_l2={rel:.3e} abs_l2={abs:.3e} nonfinite={nf}",
                    tag("dist", "")
                ));
                // CPU f64 reference of the contract arithmetic.
                let mut x8 = vec![0u8; NMAX * k];
                gpu.hip.memcpy_dtoh_at(&mut x8, &panel.x8.buf, 0)?;
                let mut sb = vec![0u8; NMAX * 4];
                gpu.hip.memcpy_dtoh_at(&mut sb, &panel.s.buf, 0)?;
                let scales: Vec<f32> = sb
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let (worst, wt, tiny, nrows) =
                    cpu_reference(&mut gpu, case, &x8, &scales, &refb, &outs.init)?;
                let ok = worst <= CPU_TOL;
                rep.tally(
                    "dist",
                    ok,
                    format!(
                        "{} cpu-f64 contract reference: max token rel_l2={worst:.3e} (token {wt}) tiny-row-class max={tiny:.3e} sampled_weight_rows={nrows} tol={CPU_TOL:e}",
                        tag("dist", "")
                    ),
                );
            }
            free_outs(&mut gpu, outs)?;
        }

        if has("time") {
            let seed = 1u64;
            load_x(&mut gpu, &x, &gen_x(seed, k))?;
            fill_panel(&mut gpu, &panel, &x, k)?;
            let outs = alloc_outs(&mut gpu, case, seed)?;
            let e0 = gpu.hip.event_create()?;
            let e1 = gpu.hip.event_create()?;
            let (reps, warm) = (args.reps, args.warm);
            let time = |gpu: &mut Gpu, f: &mut dyn FnMut(&mut Gpu) -> R<()>| -> R<f64> {
                let mut ts = Vec::with_capacity(reps);
                for i in 0..reps + warm {
                    gpu.hip.device_synchronize()?;
                    gpu.hip.event_record(&e0, None)?;
                    f(gpu)?;
                    gpu.hip.event_record(&e1, None)?;
                    gpu.hip.event_synchronize(&e1)?;
                    if i >= warm {
                        ts.push(gpu.hip.event_elapsed_ms(&e0, &e1)? as f64);
                    }
                }
                Ok(median(&mut ts))
            };
            let tm = case.total_m();
            let floor_ms = case.weight_bytes() as f64 / (STREAM_GBS * 1e9) * 1e3;
            let tf = |ms: f64, n: usize| 2.0 * tm as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12;
            let mut row = |kind: &str, n: usize, tl: &str, ms: f64| {
                timing.push(format!(
                    "{kind},{},{n},{tl},{},{tm},{ms},{},{floor_ms},{}",
                    case.name,
                    case.layers,
                    tf(ms, n),
                    floor_ms / ms
                ));
            };
            for n in [16usize, 128] {
                let f16cvt = time(&mut gpu, &mut |g| {
                    launch(g, case, Kind::F16, &x, &panel, &outs.y, 0, n)
                })?;
                gpu.invalidate_fp16_cache();
                let f16 = time(&mut gpu, &mut |g| {
                    launch(g, case, Kind::F16Kept, &x, &panel, &outs.y, 0, n)
                })?;
                let fp8 = time(&mut gpu, &mut |g| {
                    launch(g, case, Kind::Prod, &x, &panel, &outs.y, 0, n)
                })?;
                let fp8g = time(&mut gpu, &mut |g| {
                    launch(g, case, Kind::Pre(None), &x, &panel, &outs.y, 0, n)
                })?;
                let pt = fp8_verify_tile(n, tm);
                rep.line(format!(
                    "time {} N={n} | f16 cvt+gemm {f16cvt:.4} ms ({:.1} TF, floor/t {:.2}) | f16 gemm {f16:.4} ms ({:.1} TF, {:.2}) | fp8 pack+gemm {fp8:.4} ms ({:.1} TF, {:.2}) | fp8 gemm {fp8g:.4} ms ({:.1} TF, {:.2}) tile ({},{}) | floor {floor_ms:.4} ms | x{} layers",
                    case.name,
                    tf(f16cvt, n), floor_ms / f16cvt,
                    tf(f16, n), floor_ms / f16,
                    tf(fp8, n), floor_ms / fp8,
                    tf(fp8g, n), floor_ms / fp8g,
                    pt.bt, pt.waves,
                    case.layers
                ));
                for (kind, v) in [
                    ("f16_cvt_gemm", f16cvt),
                    ("f16_gemm", f16),
                    ("fp8_pack_gemm", fp8),
                    ("fp8_gemm", fp8g),
                ] {
                    let tl = if kind == "fp8_gemm" {
                        format!("({}/{})", pt.bt, pt.waves)
                    } else {
                        String::new()
                    };
                    row(kind, n, &tl, v);
                    steps
                        .entry((kind, n))
                        .or_default()
                        .push((v, case.layers, case.in_step));
                }
            }
            if args.tile_sweep {
                for n in [16usize, 32, 64, 128] {
                    let pt = fp8_verify_tile(n, tm);
                    let mut parts = Vec::new();
                    for &(bt, w) in &TILES {
                        let ms = time(&mut gpu, &mut |g| {
                            launch(
                                g,
                                case,
                                Kind::Pre(Some(tile(bt, w))),
                                &x,
                                &panel,
                                &outs.y,
                                0,
                                n,
                            )
                        })?;
                        parts.push(format!("({bt},{w}) {ms:.4} ms {:.1} TF", tf(ms, n)));
                        row("tile", n, &format!("({bt}/{w})"), ms);
                    }
                    rep.line(format!(
                        "tile-sweep {} N={n} production ({},{}) | {}",
                        case.name,
                        pt.bt,
                        pt.waves,
                        parts.join(" | ")
                    ));
                }
            }
            gpu.hip.event_destroy(e0)?;
            gpu.hip.event_destroy(e1)?;
            free_outs(&mut gpu, outs)?;
        }
    }

    // Per-step sums (layer counts per verify step).
    for ((kind, n), v) in &steps {
        let six: Vec<&(f64, usize, bool)> = v.iter().filter(|e| e.2).collect();
        if six.len() == 6 {
            let s6: f64 = six.iter().map(|e| e.0 * e.1 as f64).sum();
            let head: f64 = v.iter().filter(|e| !e.2).map(|e| e.0 * e.1 as f64).sum();
            let has_head = v.iter().any(|e| !e.2);
            rep.line(format!(
                "step {kind} N={n}: six shapes {s6:.3} ms{}",
                if has_head {
                    format!(", with lm_head {:.3} ms", s6 + head)
                } else {
                    String::new()
                }
            ));
            timing.push(format!("step6_{kind},STEP,{n},,1,0,{s6},,,"));
            if has_head {
                timing.push(format!("step7_{kind},STEP,{n},,1,0,{},,,", s6 + head));
            }
        }
    }
    if !timing.is_empty() {
        let mut csv =
            String::from("kind,shape,n,tile,layers,total_m,ms,tflops,floor_ms,floor_over_ms\n");
        for r in &timing {
            csv.push_str(r);
            csv.push('\n');
        }
        std::fs::write(format!("{out}/timing.csv"), csv)?;
    }
    let rc = gpu.fp8_verify_receipt();
    rep.line(format!(
        "fp8 receipt (informational; the harness sets the scope field directly, no enter/exit): scopes {} calls {} rows {}",
        rc.scopes - receipt0.scopes,
        rc.calls - receipt0.calls,
        rc.rows - receipt0.rows
    ));
    std::fs::write(format!("{out}/digests.txt"), &digests)?;
    let gated = rep.gated_fails();
    let summary: Vec<String> = rep
        .groups
        .iter()
        .map(|(g, (c, f))| format!("{g}: {c} checks {f} fails"))
        .collect();
    rep.line(format!("checks {}", summary.join(" | ")));
    rep.line(format!(
        "gated fails {gated} (rows/perm/graph/pack); dist fails {} (diagnostic)",
        rep.groups.get("dist").map_or(0, |e| e.1)
    ));
    rep.line(if gated == 0 {
        "VERDICT PASS".into()
    } else {
        "VERDICT FAIL".into()
    });
    std::fs::write(format!("{out}/report.txt"), &rep.text)?;
    gpu.free_tensor(x)?;
    gpu.free_tensor(panel.x8)?;
    gpu.free_tensor(panel.s)?;
    Ok(gated)
}
