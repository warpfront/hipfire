// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-equality probe of the Flash-Next GDN in-projection group routes
//! (`qkv`, `z`, `a`, `b` of one shared rotated input, K = 2560) against the
//! singleton launches, on real shapes, exactly the way
//! `hipfire-dispatch::pipeline::layer_ops::project_weights_inner` issues them
//! in few-row verify / lane steps (`2..=few_row_limit` rows; `1` row too):
//!
//! * rotation: `Gpu::rotate_x_mq_batched` (what `rotate_input` runs when
//!   `route_rows(rows) > 1`; `rotate_x_mq` at one row) against `rotate_x_mq` on
//!   each raw row alone (the AR decode rotation);
//! * the members of one rotated input are split by weight format like the
//!   dispatch: MQ6G256V2 members, when two or more, share
//!   `gemm_mq6g256v2_f32_rows_x4` launches (`gemv_mq6g256v2_x4` at one row),
//!   four per launch, empty slots `m = 0`; a lone MQ6 member takes
//!   `gemm_mq6g256v2_f32_rows` (one row: the singleton itself, skipped);
//!   MQ4G256V2 members share `gemv_mq4g256v2_rows_x4` launches (a lone member
//!   takes the same launch with empty slots, one row included);
//! * every grouped output element must be bitwise the singleton
//!   (`gemv_mq6g256v2` / `gemv_mq4g256v2` per activation row) in two readings:
//!   `proj-same-x` (singleton projections of the very rotated rows the grouped
//!   launch read: isolates the projection kernel) and `e2e-ar` (singleton
//!   rotation of the raw row, then the singleton projection: the AR bytes).
//!
//! `Gpu::qwen4_scope` and `Gpu::lane_rows_scope` are set true for the whole
//! run, as the lane step sets them (`prefill_invariant_scope` stays off; every
//! other scope stays at its `Gpu::init` default). Outputs are NaN-prefilled
//! with 64-word NaN guards on both sides; a guard touched, an unwritten word
//! or one different bit is a mismatch. `scratch.prerotated` (the
//! `hyper_read_projected_rotate` fusion cache) is never armed here, so the
//! fused-rotation skip is not exercised.
//!
//! Weights: synthetic, valid per format (finite f16 `[s0 z0 s1 z1]` header,
//! MQ6: 192 payload bytes of 6-bit codes = 200 B / 256-group; MQ4: 128 payload
//! bytes = 136 B / group), patterns `random zero_scale codes_max codes_min
//! codes_alt scale_edge`; or the real tensors of one layer of an HFQ artifact
//! (`--model PATH --layer N`, wpattern `real`, shapes and formats read from the
//! file). Activations: `k` distinct F32 rows per pattern (`rand scaled outlier
//! dup pairs zero_odd altsign tiny big onehot ramp subnormal cancel mixed`),
//! row `r` depends on `(seed, pattern, r)` only.
//!
//! Routes (`--route`, `;`-separated, default `all`): a preset name (`--list`)
//! or a spec `member=fmt,...` with members `qkv z a b` and fmt `mq6`/`mq4`,
//! e.g. `qkv=mq6,z=mq4,a=mq4,b=mq4`. Member order is the group order.
//!
//! Usage (gfx1201; nothing else is required of the host):
//!   cargo run --release -p hipfire-arch-qwen4 --features lab \
//!     --example qwen4_gdn_in_probe -- [--route R] [--rows 1-8] [--xpat LIST|all]
//!     [--wpat LIST|all] [--seed N] [--k N] [--dims qkv=N,z=N,a=N,b=N]
//!     [--model PATH --layer N] [--x-off LIST] [--y-off LIST] [--repeat N]
//!     [--stop-first] [--verbose] [--require-arch gfx1201] [--list]
//! Single-route diagnosis:
//!   ... -- --route mixed --rows 8 --xpat rand --wpat random --stop-first
//! Exit 0: every comparison bit-identical; 1: any mismatch; 2: usage / setup.

use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{Gpu, GpuTensor};
use std::collections::BTreeMap;
use std::path::Path;

type Res<T> = Result<T, String>;

const GUARD: usize = 64;
const NAN_BITS: u32 = 0x7fc0_0000;
const MEMBERS: [&str; 4] = ["qkv", "z", "a", "b"];
const HFQ_TAILS: [&str; 4] = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"];
const DEFAULT_K: usize = 2560;
/// The pinned Flash-Next GDN geometry (`linear_num_key_heads` 16 x 128,
/// `linear_num_value_heads` 48 x 128): qkv = 2 * key width + value width.
const KEY_WIDTH: usize = 16 * 128;
const VALUE_WIDTH: usize = 48 * 128;
const VALUE_HEADS: usize = 48;
const DEFAULT_SEED: u64 = 0x6764_6e69_6e00_0001;

const XPATS: [&str; 14] = [
    "rand",
    "scaled",
    "outlier",
    "dup",
    "pairs",
    "zero_odd",
    "altsign",
    "tiny",
    "big",
    "onehot",
    "ramp",
    "subnormal",
    "cancel",
    "mixed",
];
/// The per-row patterns `mixed` cycles through (row `r` takes `[r % 8]`).
const MIXED: [&str; 8] = [
    "rand",
    "scaled",
    "outlier",
    "altsign",
    "tiny",
    "big",
    "onehot",
    "subnormal",
];
const WPATS: [&str; 6] = [
    "random",
    "zero_scale",
    "codes_max",
    "codes_min",
    "codes_alt",
    "scale_edge",
];

const ENV_REPORT: [&str; 6] = [
    "HIPFIRE_QWEN4_MQ4_ROWS_PM",
    "HIPFIRE_QWEN4_WIDE_TWIN_MQ4",
    "HIPFIRE_QWEN4_WIDE_TWIN_MQ6",
    "HIPFIRE_FN_LANE_PM_ROWS",
    "HIPFIRE_QWEN4_MQ6_X4_GFX1201",
    "HIPFIRE_PREFILL_PM_INVARIANT",
];

// ------------------------------------------------------------------ host math

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// `seed` mixed with the bytes of `s` (FNV-1a, then one splitmix round).
fn mix(seed: u64, s: &str) -> u64 {
    let mut h = seed ^ 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    Rng(h).next()
}

/// Round-to-nearest-even f16 bits of a finite value in the normal f16 range.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

// ------------------------------------------------------------------- formats

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Fmt {
    Mq6,
    Mq4,
}

impl Fmt {
    /// Bytes of one 256-element group (8 header + payload).
    fn stride(self) -> usize {
        match self {
            Fmt::Mq6 => 200,
            Fmt::Mq4 => 136,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Fmt::Mq6 => "mq6",
            Fmt::Mq4 => "mq4",
        }
    }
    fn parse(s: &str) -> Res<Fmt> {
        match s {
            "mq6" => Ok(Fmt::Mq6),
            "mq4" => Ok(Fmt::Mq4),
            other => Err(format!("unknown format {other:?} (mq6 | mq4)")),
        }
    }
    /// The serialized HFQ quant byte (`hipfire-runtime` `loader_api`).
    fn from_qt(qt: u8) -> Option<Fmt> {
        match qt {
            44 => Some(Fmt::Mq4),
            47 => Some(Fmt::Mq6),
            _ => None,
        }
    }
}

/// `m` rows x `k / 256` groups of `fmt`: `[s0 z0 s1 z1]` F16 LE headers, then
/// the payload bytes (any byte pattern is a valid set of codes).
fn pack_weights(fmt: Fmt, wpat: &str, m: usize, k: usize, seed: u64) -> Vec<u8> {
    let payload = fmt.stride() - 8;
    let mut rng = Rng(mix(seed, wpat) ^ 0x6d71_0000_0000_0001 ^ fmt as u64);
    let mut out = Vec::with_capacity(m * (k / 256) * fmt.stride());
    for g in 0..m * (k / 256) {
        for _half in 0..2 {
            let mut scale = {
                let mag = 2f32.powi(-9) + rng.unit() * 0.03;
                f16_bits(if rng.next() & 1 == 0 { mag } else { -mag })
            };
            let mut zero = {
                let mut v = (rng.unit() - 0.5) * 4.0;
                if v.abs() < 2f32.powi(-12) {
                    v = 0.125;
                }
                f16_bits(v)
            };
            match wpat {
                // An exact (signed) zero scale: the group is its zero point.
                "zero_scale" => scale = if rng.next() & 1 == 0 { 0x0000 } else { 0x8000 },
                // Largest and smallest normal-ish scales / zero points.
                "scale_edge" => {
                    let big = rng.next() & 1 == 0;
                    scale = f16_bits(if big { 0.125 } else { 2f32.powi(-14) });
                    zero = f16_bits(if big { -2.0 } else { 2f32.powi(-12) });
                }
                _ => {}
            }
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&zero.to_le_bytes());
        }
        for i in 0..payload {
            out.push(match wpat {
                "codes_max" => 0xff,
                "codes_min" => 0x00,
                "codes_alt" => {
                    if (i + g) % 2 == 0 {
                        0xaa
                    } else {
                        0x55
                    }
                }
                _ => (rng.next() & 0xff) as u8,
            });
        }
    }
    out
}

// --------------------------------------------------------------- activations

/// One raw F32 activation row of `pat`; a function of `(seed, pat, r, k)`.
fn x_row(pat: &str, seed: u64, r: usize, k: usize) -> Vec<f32> {
    let (pat, src) = match pat {
        "mixed" => (MIXED[r % 8], r),
        "dup" => ("rand", 0),
        "pairs" => ("rand", r & !1),
        "zero_odd" if r % 2 == 1 => return vec![0.0; k],
        "zero_odd" => ("rand", r),
        other => (other, r),
    };
    let mut rng = Rng(mix(seed, pat) ^ (src as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let base = |rng: &mut Rng| (rng.unit() - 0.5) * 4.0;
    let mut v: Vec<f32> = match pat {
        "scaled" => {
            let s = 2f32.powi((src % 9) as i32 - 4);
            (0..k).map(|_| base(&mut rng) * s).collect()
        }
        "outlier" => {
            let mut v: Vec<f32> = (0..k).map(|_| base(&mut rng) * 0.5).collect();
            for j in 0..3 {
                let at = (src * 37 + j * 811) % k;
                v[at] = if j % 2 == 0 { 64.0 } else { -64.0 };
            }
            v
        }
        "altsign" | "cancel" => {
            let jitter = if pat == "cancel" { 0.015625 } else { 0.0 };
            (0..k)
                .map(|j| {
                    let m = 0.75 + jitter * (rng.unit() - 0.5);
                    if j % 2 == 0 {
                        m
                    } else {
                        -m
                    }
                })
                .collect()
        }
        "tiny" => (0..k).map(|_| base(&mut rng) * 2f32.powi(-60)).collect(),
        "big" => (0..k).map(|_| base(&mut rng) * 2f32.powi(12)).collect(),
        "onehot" => {
            let mut v = vec![0.0; k];
            v[(src * 257 + 1) % k] = 1.0 + src as f32;
            v
        }
        "ramp" => (0..k)
            .map(|j| (j as f32 - (k / 2) as f32) / k as f32 * (src + 1) as f32)
            .collect(),
        "subnormal" => {
            // Exact subnormal scales by bit pattern (`powi(-135)` would overflow its reciprocal to 0):
            // mantissa bit i is 2^(i-149), so 1<<22 = 2^-127 and 1<<14 = 2^-135.
            let scale = f32::from_bits(if src % 2 == 0 { 1 << 22 } else { 1 << 14 });
            (0..k)
                .map(|j| match j % 3 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => base(&mut rng) * scale,
                })
                .collect()
        }
        // "rand" (and the dup / pairs / zero_odd sources).
        _ => (0..k).map(|_| base(&mut rng)).collect(),
    };
    v.truncate(k);
    v
}

fn x_rows(pat: &str, seed: u64, rows: usize, k: usize) -> Vec<f32> {
    (0..rows).flat_map(|r| x_row(pat, seed, r, k)).collect()
}

// -------------------------------------------------------------------- routes

#[derive(Clone, Debug, PartialEq)]
struct Route {
    label: String,
    /// `(index into MEMBERS, format)` in group order.
    members: Vec<(usize, Fmt)>,
}

fn member_index(name: &str) -> Res<usize> {
    MEMBERS
        .iter()
        .position(|m| *m == name)
        .ok_or_else(|| format!("unknown member {name:?} (qkv z a b)"))
}

fn parse_spec(spec: &str) -> Res<Route> {
    let mut members: Vec<(usize, Fmt)> = Vec::new();
    for item in spec.split(',').filter(|s| !s.is_empty()) {
        let (name, fmt) = item
            .split_once('=')
            .ok_or_else(|| format!("route item {item:?} is not member=fmt"))?;
        let idx = member_index(name.trim())?;
        if members.iter().any(|(i, _)| *i == idx) {
            return Err(format!("route {spec:?}: member {name} twice"));
        }
        members.push((idx, Fmt::parse(fmt.trim())?));
    }
    if members.is_empty() {
        return Err("empty route".into());
    }
    Ok(Route {
        label: spec.to_string(),
        members,
    })
}

/// `(name, spec)`; every preset is a `member=fmt` spec.
const PRESETS: [(&str, &str); 19] = [
    ("all6", "qkv=mq6,z=mq6,a=mq6,b=mq6"),
    ("all4", "qkv=mq4,z=mq4,a=mq4,b=mq4"),
    ("mixed", "qkv=mq6,z=mq4,a=mq4,b=mq4"),
    ("mixed_inv", "qkv=mq4,z=mq6,a=mq6,b=mq6"),
    ("mixed2", "qkv=mq6,z=mq6,a=mq4,b=mq4"),
    ("mq6_zba", "z=mq6,a=mq6,b=mq6"),
    ("mq6_qkvz", "qkv=mq6,z=mq6"),
    ("mq6_ab", "a=mq6,b=mq6"),
    ("mq4_zba", "z=mq4,a=mq4,b=mq4"),
    ("mq4_qkvz", "qkv=mq4,z=mq4"),
    ("mq4_ab", "a=mq4,b=mq4"),
    ("mq6_single_qkv", "qkv=mq6"),
    ("mq6_single_z", "z=mq6"),
    ("mq6_single_a", "a=mq6"),
    ("mq6_single_b", "b=mq6"),
    ("mq4_single_qkv", "qkv=mq4"),
    ("mq4_single_z", "z=mq4"),
    ("mq4_single_a", "a=mq4"),
    ("mq4_single_b", "b=mq4"),
];

fn resolve_route(item: &str) -> Res<Route> {
    if item.contains('=') {
        return parse_spec(item);
    }
    let (name, spec) = PRESETS
        .iter()
        .find(|(n, _)| *n == item)
        .ok_or_else(|| format!("unknown route preset {item:?} (--list)"))?;
    let mut r = parse_spec(spec)?;
    r.label = (*name).to_string();
    Ok(r)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Mq6X4,
    Mq6Single,
    Mq4X4,
}

impl Kind {
    fn symbol(self, rows: usize) -> &'static str {
        match (self, rows) {
            (Kind::Mq6X4, 1) => "gemv_mq6g256v2_x4",
            (Kind::Mq6X4, _) => "gemm_mq6g256v2_f32_rows_x4",
            (Kind::Mq6Single, _) => "gemm_mq6g256v2_f32_rows",
            (Kind::Mq4X4, _) => "gemv_mq4g256v2_rows_x4",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Launch {
    kind: Kind,
    /// Positions in `Route::members`.
    members: Vec<usize>,
}

/// The launches `project_weights_inner` issues for the group's members (all of
/// one K and basis): two or more MQ6 members share x4 launches four at a
/// time, a lone one runs `gemm_mq6g256v2_f32_rows`; MQ4 members share x4 rows
/// launches four at a time, a lone one too.
fn plan(route: &Route) -> Vec<Launch> {
    let pick = |f: Fmt| -> Vec<usize> {
        route
            .members
            .iter()
            .enumerate()
            .filter(|(_, (_, g))| *g == f)
            .map(|(p, _)| p)
            .collect()
    };
    let (mq6, mq4) = (pick(Fmt::Mq6), pick(Fmt::Mq4));
    let mut out = Vec::new();
    if mq6.len() >= 2 {
        for c in mq6.chunks(4) {
            out.push(Launch {
                kind: Kind::Mq6X4,
                members: c.to_vec(),
            });
        }
    } else if mq6.len() == 1 {
        out.push(Launch {
            kind: Kind::Mq6Single,
            members: mq6,
        });
    }
    for c in mq4.chunks(4) {
        out.push(Launch {
            kind: Kind::Mq4X4,
            members: c.to_vec(),
        });
    }
    out
}

// ------------------------------------------------------------- compare / log

#[derive(Debug, PartialEq)]
struct Diff {
    count: usize,
    first: usize,
    /// Mismatching words by activation-row parity (`index / m`) and by
    /// column parity (`index % m`).
    odd_rows: usize,
    even_rows: usize,
    odd_cols: usize,
    even_cols: usize,
}

fn diff_words(got: &[u32], want: &[u32], m: usize) -> Option<Diff> {
    assert_eq!(got.len(), want.len());
    assert!(m > 0);
    let mut d = Diff {
        count: 0,
        first: 0,
        odd_rows: 0,
        even_rows: 0,
        odd_cols: 0,
        even_cols: 0,
    };
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g != w {
            if d.count == 0 {
                d.first = i;
            }
            d.count += 1;
            if (i / m) % 2 == 1 {
                d.odd_rows += 1;
            } else {
                d.even_rows += 1;
            }
            if (i % m) % 2 == 1 {
                d.odd_cols += 1;
            } else {
                d.even_cols += 1;
            }
        }
    }
    (d.count > 0).then_some(d)
}

/// True when every word outside `start..start + len` is the NaN guard.
fn guards_clean(bits: &[u32], start: usize, len: usize) -> bool {
    bits[..start]
        .iter()
        .chain(&bits[start + len..])
        .all(|&b| b == NAN_BITS)
}

fn describe(d: &Diff, got: &[u32], want: &[u32], m: usize) -> String {
    let (g, w) = (got[d.first], want[d.first]);
    format!(
        "first row={} col={} got=0x{g:08x} ({:e}) want=0x{w:08x} ({:e}) mismatching_words={} \
         rows(odd/even)={}/{} cols(odd/even)={}/{}",
        d.first / m,
        d.first % m,
        f32::from_bits(g),
        f32::from_bits(w),
        d.count,
        d.odd_rows,
        d.even_rows,
        d.odd_cols,
        d.even_cols
    )
}

#[derive(Default)]
struct Report {
    cases: usize,
    launches: usize,
    skipped: usize,
    bad_words: usize,
    /// Route label -> (cases, mismatching comparisons).
    by_route: BTreeMap<String, (usize, usize)>,
    by_stage: BTreeMap<&'static str, usize>,
    verbose: bool,
    stop_first: bool,
}

impl Report {
    fn fail(&mut self, route: &str, stage: &'static str, words: usize) {
        self.by_route.entry(route.to_string()).or_default().1 += 1;
        *self.by_stage.entry(stage).or_default() += 1;
        self.bad_words += words;
    }
    fn stop(&self) -> bool {
        self.stop_first && self.bad_words > 0
    }
}

// ------------------------------------------------------------------- device

fn he<T, E: std::fmt::Debug>(r: Result<T, E>) -> Res<T> {
    r.map_err(|e| format!("{e:?}"))
}

fn nan_tensor(gpu: &mut Gpu, n: usize) -> Res<GpuTensor> {
    he(gpu.upload_f32(&vec![f32::from_bits(NAN_BITS); n], &[n]))
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Res<Vec<u32>> {
    he(gpu.hip.device_synchronize())?;
    Ok(he(gpu.download_f32(t))?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

/// The singleton projection of every activation row of `x` (`rows` rows of
/// `k`) into one `[rows, m]` result.
fn singleton_ref(
    gpu: &mut Gpu,
    fmt: Fmt,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
) -> Res<Vec<u32>> {
    let y = nan_tensor(gpu, rows * m)?;
    for r in 0..rows {
        let (xr, yr) = (x.sub_offset(r * k, k), y.sub_offset(r * m, m));
        match fmt {
            Fmt::Mq6 => he(gpu.gemv_mq6g256v2(w, &xr, &yr, m, k))?,
            Fmt::Mq4 => he(gpu.gemv_mq4g256v2(w, &xr, &yr, m, k))?,
        }
    }
    let bits = fetch(gpu, &y)?;
    he(gpu.free_tensor(y))?;
    Ok(bits)
}

struct Case<'a> {
    route: &'a Route,
    wlabel: &'a str,
    xpat: &'a str,
    rows: usize,
    k: usize,
    x_off: usize,
    y_off: usize,
    repeat: usize,
}

impl Case<'_> {
    fn tag(&self) -> String {
        format!(
            "route={} wsrc={} xpat={} k={} rows={} x_off={} y_off={}",
            self.route.label, self.wlabel, self.xpat, self.k, self.rows, self.x_off, self.y_off
        )
    }
}

/// One full case: rotation check, singleton references, every launch of the
/// route's plan. `x` is the raw F32 activation (at least `rows * k` words);
/// `w` / `ms` are the route members' weights and output widths.
fn run_case(
    gpu: &mut Gpu,
    rep: &mut Report,
    c: &Case,
    w: &[&GpuTensor],
    ms: &[usize],
    x: &GpuTensor,
) -> Res<()> {
    let (rows, k) = (c.rows, c.k);
    let label = c.route.label.as_str();
    rep.cases += 1;
    rep.by_route.entry(label.to_string()).or_default().0 += 1;

    // -- rotation: batched (the dispatch route) vs every raw row alone.
    let rot_n = GUARD + c.x_off + rows * k + GUARD;
    let (rot_b, rot_s) = (nan_tensor(gpu, rot_n)?, nan_tensor(gpu, rot_n)?);
    let (vb, vs) = (
        rot_b.sub_offset(GUARD + c.x_off, rows * k),
        rot_s.sub_offset(GUARD + c.x_off, rows * k),
    );
    let xr = x.sub_offset(0, rows * k);
    if gpu.route_rows(rows) > 1 {
        he(gpu.rotate_x_mq_batched(&xr, &vb, k, rows))?;
    } else {
        he(gpu.rotate_x_mq(&xr, &vb, k))?;
    }
    for r in 0..rows {
        he(gpu.rotate_x_mq(&x.sub_offset(r * k, k), &vs.sub_offset(r * k, k), k))?;
    }
    let (bb, bs) = (fetch(gpu, &rot_b)?, fetch(gpu, &rot_s)?);
    let lo = GUARD + c.x_off;
    for (name, bits) in [("batched", &bb), ("singleton", &bs)] {
        if !guards_clean(bits, lo, rows * k) {
            println!(
                "MISMATCH {} stage=rotation-guard rotation={name}: a guard word was written",
                c.tag()
            );
            rep.fail(label, "rotation-guard", 1);
        }
    }
    if let Some(d) = diff_words(&bb[lo..lo + rows * k], &bs[lo..lo + rows * k], k) {
        println!(
            "MISMATCH {} stage=rotation batched-vs-singleton (width k): {}",
            c.tag(),
            describe(&d, &bb[lo..], &bs[lo..], k)
        );
        rep.fail(label, "rotation", d.count);
    }

    // -- references per member (both readings).
    let mut ref_same = Vec::new();
    let mut ref_ar = Vec::new();
    for (j, (_, fmt)) in c.route.members.iter().enumerate() {
        ref_same.push(singleton_ref(gpu, *fmt, w[j], &vb, ms[j], k, rows)?);
        ref_ar.push(singleton_ref(gpu, *fmt, w[j], &vs, ms[j], k, rows)?);
    }

    // -- the grouped launches.
    for launch in plan(c.route) {
        if launch.kind == Kind::Mq6Single && rows == 1 {
            // `project_rotated` at one row is `gemv_mq6g256v2` itself.
            rep.skipped += 1;
            continue;
        }
        let n = launch.members.len();
        for _ in 0..c.repeat {
            rep.launches += 1;
            let mut outs = Vec::with_capacity(n);
            for &j in &launch.members {
                outs.push(nan_tensor(gpu, GUARD + c.y_off + rows * ms[j] + GUARD)?);
            }
            let views: Vec<GpuTensor> = launch
                .members
                .iter()
                .zip(&outs)
                .map(|(&j, y)| y.sub_offset(GUARD + c.y_off, rows * ms[j]))
                .collect();
            let parts: [(&GpuTensor, &GpuTensor, usize); 4] = std::array::from_fn(|slot| {
                let p = slot.min(n - 1);
                let j = launch.members[p];
                (w[j], &views[p], if slot < n { ms[j] } else { 0 })
            });
            let prior = gpu.wide_verify_rows;
            if rows > 8 {
                gpu.wide_verify_rows = rows;
            }
            let result =
                match launch.kind {
                    Kind::Mq6X4 if rows == 1 => he(gpu.gemv_mq6g256v2_x4(parts, &vb, k)),
                    Kind::Mq6X4 => he(gpu.gemm_mq6g256v2_f32_rows_x4(parts, &vb, k, rows)),
                    Kind::Mq6Single => he(gpu
                        .gemm_mq6g256v2_f32_rows(parts[0].0, &vb, parts[0].1, parts[0].2, k, rows)),
                    Kind::Mq4X4 => he(gpu.gemv_mq4g256v2_rows_x4(parts, &vb, k, rows)),
                };
            gpu.wide_verify_rows = prior;
            let symbol = gpu.last_launched_kernel().unwrap_or("<none>").to_string();
            let desc = format!(
                "launch={} slots=[{}] symbol={symbol}",
                launch.kind.symbol(rows),
                launch
                    .members
                    .iter()
                    .map(|&j| {
                        let (mi, f) = c.route.members[j];
                        format!("{}:{} m={}", MEMBERS[mi], f.name(), ms[j])
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            if let Err(e) = result {
                println!("ERROR {} {desc}: {e}", c.tag());
                rep.fail(label, "launch-error", 1);
                for y in outs {
                    he(gpu.free_tensor(y))?;
                }
                continue;
            }
            for (p, &j) in launch.members.iter().enumerate() {
                let m = ms[j];
                let name = MEMBERS[c.route.members[j].0];
                let bits = fetch(gpu, &outs[p])?;
                let at = GUARD + c.y_off;
                if !guards_clean(&bits, at, rows * m) {
                    println!(
                        "MISMATCH {} {desc} member={name} stage=guard: a guard word was written",
                        c.tag()
                    );
                    rep.fail(label, "guard", 1);
                }
                let got = &bits[at..at + rows * m];
                let same = diff_words(got, &ref_same[j], m);
                if let Some(d) = &same {
                    println!(
                        "MISMATCH {} {desc} member={name} stage=proj-same-x: {}",
                        c.tag(),
                        describe(d, got, &ref_same[j], m)
                    );
                    rep.fail(label, "proj-same-x", d.count);
                }
                if ref_ar[j] != ref_same[j] || same.is_some() {
                    if let Some(d) = diff_words(got, &ref_ar[j], m) {
                        println!(
                            "MISMATCH {} {desc} member={name} stage=e2e-ar: {}",
                            c.tag(),
                            describe(&d, got, &ref_ar[j], m)
                        );
                        rep.fail(label, "e2e-ar", d.count);
                    }
                }
                if rep.verbose && same.is_none() {
                    println!("ok {} {desc} member={name}", c.tag());
                }
            }
            for y in outs {
                he(gpu.free_tensor(y))?;
            }
        }
    }
    he(gpu.free_tensor(rot_b))?;
    he(gpu.free_tensor(rot_s))?;
    Ok(())
}

// ------------------------------------------------------------- real weights

struct Real {
    name: String,
    qt: u8,
    fmt: Option<Fmt>,
    m: usize,
    k: usize,
    bytes: Vec<u8>,
}

/// The four GDN in-projection tensors of `layer` of an HFQ artifact.
fn load_real(path: &Path, layer: usize) -> Res<Vec<Real>> {
    let hfq = HfqFile::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for tail in HFQ_TAILS {
        let candidates = [
            format!("model.language_model.layers.{layer}.linear_attn.{tail}.weight"),
            format!("model.layers.{layer}.linear_attn.{tail}.weight"),
        ];
        let name = candidates
            .iter()
            .find(|n| hfq.find_tensor_info(n).is_some())
            .ok_or_else(|| {
                format!(
                    "layer {layer}: none of {candidates:?} in {}",
                    path.display()
                )
            })?;
        let (info, data) = hfq
            .tensor_data(name)
            .ok_or_else(|| format!("{name}: no resident data (mmap dropped or truncated)"))?;
        if info.shape.len() < 2 {
            return Err(format!("{name}: shape {:?} is not [m, k]", info.shape));
        }
        let m = info.shape[0] as usize;
        let k: usize = info.shape[1..].iter().map(|&d| d as usize).product();
        let fmt = Fmt::from_qt(info.quant_type);
        if let Some(f) = fmt {
            if k == 0 || k % 256 != 0 || data.len() != m * (k / 256) * f.stride() {
                return Err(format!(
                    "{name}: {} bytes is not {m} x {k} of {} ({} B / group)",
                    data.len(),
                    f.name(),
                    f.stride()
                ));
            }
        }
        out.push(Real {
            name: name.clone(),
            qt: info.quant_type,
            fmt,
            m,
            k,
            bytes: data.to_vec(),
        });
    }
    Ok(out)
}

// ----------------------------------------------------------------------- cli

fn parse_list(spec: &str) -> Res<Vec<usize>> {
    let mut v = Vec::new();
    for tok in spec.split(',').filter(|t| !t.is_empty()) {
        let num = |s: &str| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| format!("--list item {tok:?}: {e}"))
        };
        match tok.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (num(a)?, num(b)?);
                if a > b {
                    return Err(format!("range {tok:?} is empty"));
                }
                v.extend(a..=b);
            }
            None => v.push(num(tok)?),
        }
    }
    if v.is_empty() {
        return Err(format!("empty list {spec:?}"));
    }
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

fn parse_names(spec: &str, all: &[&str], what: &str) -> Res<Vec<String>> {
    if spec == "all" {
        return Ok(all.iter().map(|s| (*s).to_string()).collect());
    }
    spec.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            if all.contains(&s) {
                Ok(s.to_string())
            } else {
                Err(format!("unknown {what} {s:?} (one of {all:?} or all)"))
            }
        })
        .collect()
}

fn parse_dims(spec: &str, dims: &mut [usize; 4]) -> Res<()> {
    for item in spec.split(',').filter(|s| !s.is_empty()) {
        let (name, n) = item
            .split_once('=')
            .ok_or_else(|| format!("--dims item {item:?} is not member=N"))?;
        dims[member_index(name)?] = n
            .parse::<usize>()
            .map_err(|e| format!("--dims {item:?}: {e}"))?;
    }
    Ok(())
}

const VALUE_FLAGS: [&str; 13] = [
    "--route",
    "--rows",
    "--xpat",
    "--wpat",
    "--seed",
    "--k",
    "--dims",
    "--model",
    "--layer",
    "--x-off",
    "--y-off",
    "--repeat",
    "--require-arch",
];
const BOOL_FLAGS: [&str; 3] = ["--stop-first", "--verbose", "--list"];

struct Args(Vec<String>);

impl Args {
    fn validate(&self) -> Res<()> {
        let mut i = 0;
        while i < self.0.len() {
            let a = self.0[i].as_str();
            if VALUE_FLAGS.contains(&a) {
                if i + 1 >= self.0.len() {
                    return Err(format!("{a} needs a value"));
                }
                i += 2;
            } else if BOOL_FLAGS.contains(&a) {
                i += 1;
            } else {
                return Err(format!("unknown argument {a:?}"));
            }
        }
        Ok(())
    }
    fn value(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .position(|a| a == key)
            .and_then(|i| self.0.get(i + 1))
            .map(String::as_str)
    }
    fn flag(&self, key: &str) -> bool {
        self.0.iter().any(|a| a == key)
    }
}

fn run() -> Res<bool> {
    let args = Args(std::env::args().skip(1).collect());
    args.validate()?;
    if args.flag("--list") {
        for (name, spec) in PRESETS {
            println!("route {name:<16} {spec}");
        }
        println!("xpat  {XPATS:?}");
        println!("wpat  {WPATS:?} (+ real with --model)");
        return Ok(true);
    }
    let seed = match args.value("--seed") {
        Some(s) => s.parse::<u64>().map_err(|e| format!("--seed: {e}"))?,
        None => DEFAULT_SEED,
    };
    let rows_list = parse_list(args.value("--rows").unwrap_or("1-8"))?;
    if rows_list.first() == Some(&0) || rows_list.iter().any(|&r| r > 32) {
        return Err("--rows must be within 1..=32".into());
    }
    let max_rows = *rows_list.last().unwrap();
    let x_offs = parse_list(args.value("--x-off").unwrap_or("0"))?;
    let y_offs = parse_list(args.value("--y-off").unwrap_or("0"))?;
    let repeat = match args.value("--repeat") {
        Some(s) => s
            .parse::<usize>()
            .map_err(|e| format!("--repeat: {e}"))?
            .max(1),
        None => 1,
    };
    let xpats = parse_names(args.value("--xpat").unwrap_or("all"), &XPATS, "xpat")?;

    // Real tensors (shapes, formats, bytes) of one layer, if asked.
    let real = match (args.value("--model"), args.value("--layer")) {
        (Some(p), Some(l)) => {
            let layer = l.parse::<usize>().map_err(|e| format!("--layer: {e}"))?;
            let r = load_real(Path::new(p), layer)?;
            for (i, t) in r.iter().enumerate() {
                println!(
                    "real {} {}: qt={} fmt={} m={} k={} bytes={}",
                    MEMBERS[i],
                    t.name,
                    t.qt,
                    t.fmt.map_or("unsupported", Fmt::name),
                    t.m,
                    t.k,
                    t.bytes.len()
                );
            }
            Some(r)
        }
        (None, None) => None,
        _ => return Err("--model and --layer go together".into()),
    };

    // Dimensions: source shapes beat the pinned defaults; --k / --dims beat both.
    let mut k = DEFAULT_K;
    let mut dims = [
        2 * KEY_WIDTH + VALUE_WIDTH,
        VALUE_WIDTH,
        VALUE_HEADS,
        VALUE_HEADS,
    ];
    if let Some(r) = &real {
        k = r[0].k;
        for (i, t) in r.iter().enumerate() {
            if t.k != k {
                return Err(format!(
                    "real member {} has k={} but qkv has k={k}",
                    MEMBERS[i], t.k
                ));
            }
            dims[i] = t.m;
        }
    }
    if let Some(s) = args.value("--k") {
        k = s.parse::<usize>().map_err(|e| format!("--k: {e}"))?;
    }
    if let Some(s) = args.value("--dims") {
        parse_dims(s, &mut dims)?;
    }
    if k == 0 || k % 256 != 0 {
        return Err(format!("k={k} must be a positive multiple of 256"));
    }
    if let Some(r) = &real {
        let moved = k != r[0].k || (0..4).any(|i| dims[i] != r[i].m);
        if moved {
            return Err("--k / --dims disagree with the real tensor shapes".into());
        }
    }

    let wpat_default = if real.is_some() { "real" } else { "all" };
    let mut wall: Vec<&str> = WPATS.to_vec();
    if real.is_some() {
        wall.push("real");
    }
    let wpats = parse_names(args.value("--wpat").unwrap_or(wpat_default), &wall, "wpat")?;
    if wpats.iter().any(|w| w == "real") && real.is_none() {
        return Err("wpat real needs --model PATH --layer N".into());
    }

    // Routes: presets and specs; with real weights and no --route, the layer's own group.
    let route_arg = args.value("--route");
    let mut routes: Vec<Route> = Vec::new();
    match route_arg {
        Some("all") | None if real.is_none() || route_arg == Some("all") => {
            for (name, _) in PRESETS {
                routes.push(resolve_route(name)?);
            }
        }
        Some(s) => {
            for item in s.split(';').filter(|i| !i.is_empty()) {
                routes.push(resolve_route(item.trim())?);
            }
        }
        None => {
            let r = real.as_ref().unwrap();
            let spec = r
                .iter()
                .enumerate()
                .filter_map(|(i, t)| t.fmt.map(|f| format!("{}={}", MEMBERS[i], f.name())))
                .collect::<Vec<_>>()
                .join(",");
            if spec.is_empty() {
                return Err("no MQ4G256V2 / MQ6G256V2 member in the real layer".into());
            }
            let mut route = parse_spec(&spec)?;
            route.label = format!("model[{spec}]");
            routes.push(route);
        }
    }

    let mut gpu = Gpu::init().map_err(|e| format!("Gpu::init: {e:?}"))?;
    if let Some(want) = args.value("--require-arch") {
        if gpu.arch != want {
            eprintln!(
                "qwen4_gdn_in_probe: arch {} is not the required {want}",
                gpu.arch
            );
            std::process::exit(2);
        }
    }
    // The lane step's scopes (`gpu_forward_lanes.rs`).
    gpu.qwen4_scope = true;
    gpu.lane_rows_scope = true;
    println!(
        "arch {} qwen4_scope={} lane_rows_scope={} prefill_invariant_scope={} few_row_limit={} \
         lane_rows_pm_enabled={} mq4v2_symmetric={} seed={seed} k={k} dims={dims:?} rows={rows_list:?} \
         x_off={x_offs:?} y_off={y_offs:?} repeat={repeat}",
        gpu.arch,
        gpu.qwen4_scope,
        gpu.lane_rows_scope,
        gpu.prefill_invariant_scope,
        gpu.few_row_limit(),
        gpu.lane_rows_pm_enabled(),
        gpu.mq4v2_symmetric
    );
    for var in ENV_REPORT {
        println!(
            "env {var}={}",
            std::env::var(var).map_or("<unset>".to_string(), |v| format!("{v:?}"))
        );
    }
    for r in &routes {
        let launches: Vec<String> = plan(r)
            .iter()
            .map(|l| {
                format!(
                    "{:?}[{}]",
                    l.kind,
                    l.members
                        .iter()
                        .map(|&j| MEMBERS[r.members[j].0])
                        .collect::<Vec<_>>()
                        .join("+")
                )
            })
            .collect();
        println!("route {} => {}", r.label, launches.join(" ; "));
    }

    let mut rep = Report {
        verbose: args.flag("--verbose"),
        stop_first: args.flag("--stop-first"),
        ..Report::default()
    };
    let mut bank: BTreeMap<(usize, Fmt, String), GpuTensor> = BTreeMap::new();

    'sweep: for wpat in &wpats {
        for route in &routes {
            // Weights of this (wpat, route): synthetic, or the real tensor when its format matches.
            let mut usable = true;
            for &(mi, fmt) in &route.members {
                let key = (mi, fmt, wpat.clone());
                if bank.contains_key(&key) {
                    continue;
                }
                let bytes = if wpat == "real" {
                    let t = &real.as_ref().unwrap()[mi];
                    if t.fmt != Some(fmt) {
                        println!(
                            "skip route={} wsrc=real: member {} is qt={} ({}) in the file, route wants {}",
                            route.label,
                            MEMBERS[mi],
                            t.qt,
                            t.fmt.map_or("unsupported", Fmt::name),
                            fmt.name()
                        );
                        usable = false;
                        break;
                    }
                    t.bytes.clone()
                } else {
                    pack_weights(fmt, wpat, dims[mi], k, mix(seed, MEMBERS[mi]))
                };
                bank.insert(key, he(gpu.upload_raw(&bytes, &[bytes.len()]))?);
            }
            if !usable {
                continue;
            }
            let wlabel = if wpat == "real" {
                "real".to_string()
            } else {
                format!("synthetic:{wpat}")
            };
            let w: Vec<&GpuTensor> = route
                .members
                .iter()
                .map(|&(mi, f)| &bank[&(mi, f, wpat.clone())])
                .collect();
            let ms: Vec<usize> = route.members.iter().map(|&(mi, _)| dims[mi]).collect();

            for xpat in &xpats {
                let host = x_rows(xpat, seed, max_rows, k);
                for &x_off in &x_offs {
                    let mut buf = vec![f32::from_bits(NAN_BITS); GUARD + x_off];
                    buf.extend_from_slice(&host);
                    buf.extend(std::iter::repeat(f32::from_bits(NAN_BITS)).take(GUARD));
                    let xt = he(gpu.upload_f32(&buf, &[buf.len()]))?;
                    let xv = xt.sub_offset(GUARD + x_off, max_rows * k);
                    for &rows in &rows_list {
                        for &y_off in &y_offs {
                            let case = Case {
                                route,
                                wlabel: &wlabel,
                                xpat,
                                rows,
                                k,
                                x_off,
                                y_off,
                                repeat,
                            };
                            run_case(&mut gpu, &mut rep, &case, &w, &ms, &xv)?;
                            if rep.stop() {
                                break 'sweep;
                            }
                        }
                    }
                    he(gpu.free_tensor(xt))?;
                }
            }
        }
    }

    println!(
        "SUMMARY cases={} launches={} skipped_single_row_mq6={} mismatching_words={}",
        rep.cases, rep.launches, rep.skipped, rep.bad_words
    );
    for (route, (cases, bad)) in &rep.by_route {
        println!("  route {route}: cases={cases} mismatching_comparisons={bad}");
    }
    for (stage, n) in &rep.by_stage {
        println!("  stage {stage}: mismatching_comparisons={n}");
    }
    let pass = rep.bad_words == 0 && rep.by_stage.is_empty();
    println!("RESULT {}", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen4_gdn_in_probe: {e}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(2f32.powi(-14)), 0x0400);
    }

    #[test]
    fn lists_parse() {
        assert_eq!(parse_list("1-3,8").unwrap(), vec![1, 2, 3, 8]);
        assert_eq!(parse_list("8,8,2").unwrap(), vec![2, 8]);
        assert!(parse_list("3-1").is_err());
        assert!(parse_list("").is_err());
    }

    #[test]
    fn every_preset_resolves_with_unique_members() {
        for (name, _) in PRESETS {
            let r = resolve_route(name).unwrap();
            assert_eq!(r.label, name);
            assert!(!plan(&r).is_empty());
        }
        assert!(parse_spec("qkv=mq6,qkv=mq4").is_err());
        assert!(parse_spec("zz=mq6").is_err());
        assert!(parse_spec("qkv=mq5").is_err());
    }

    #[test]
    fn plan_follows_the_dispatch_split() {
        let l = |k: Kind, m: &[usize]| Launch {
            kind: k,
            members: m.to_vec(),
        };
        let p = |n: &str| plan(&resolve_route(n).unwrap());
        assert_eq!(p("all6"), vec![l(Kind::Mq6X4, &[0, 1, 2, 3])]);
        assert_eq!(p("all4"), vec![l(Kind::Mq4X4, &[0, 1, 2, 3])]);
        // The mixed group: a lone MQ6 member plus three MQ4 members (slot 3 empty).
        assert_eq!(
            p("mixed"),
            vec![l(Kind::Mq6Single, &[0]), l(Kind::Mq4X4, &[1, 2, 3])]
        );
        assert_eq!(
            p("mixed2"),
            vec![l(Kind::Mq6X4, &[0, 1]), l(Kind::Mq4X4, &[2, 3])]
        );
        assert_eq!(p("mq6_single_z"), vec![l(Kind::Mq6Single, &[0])]);
        assert_eq!(p("mq4_single_a"), vec![l(Kind::Mq4X4, &[0])]);
    }

    #[test]
    fn symbols_name_the_one_row_variant() {
        assert_eq!(Kind::Mq6X4.symbol(1), "gemv_mq6g256v2_x4");
        assert_eq!(Kind::Mq6X4.symbol(2), "gemm_mq6g256v2_f32_rows_x4");
        assert_eq!(Kind::Mq6Single.symbol(5), "gemm_mq6g256v2_f32_rows");
    }

    #[test]
    fn packed_weights_are_valid_per_format() {
        for fmt in [Fmt::Mq6, Fmt::Mq4] {
            for wpat in WPATS {
                let (m, k) = (5, 512);
                let a = pack_weights(fmt, wpat, m, k, 7);
                assert_eq!(a.len(), m * (k / 256) * fmt.stride(), "{fmt:?} {wpat}");
                assert_eq!(a, pack_weights(fmt, wpat, m, k, 7), "deterministic");
                assert_ne!(
                    a,
                    pack_weights(fmt, wpat, m, k, 8),
                    "{wpat} depends on the seed"
                );
                for g in a.chunks_exact(fmt.stride()) {
                    for h in g[..8].chunks_exact(2) {
                        let bits = u16::from_le_bytes([h[0], h[1]]);
                        let exp = (bits >> 10) & 31;
                        // Finite: never the all-ones exponent; never a subnormal other than +-0.
                        assert!(exp != 31, "{fmt:?} {wpat}: non-finite header 0x{bits:04x}");
                        assert!(
                            exp != 0 || bits & 0x3ff == 0,
                            "{fmt:?} {wpat}: subnormal header"
                        );
                    }
                }
                match wpat {
                    "codes_max" => assert!(a
                        .chunks_exact(fmt.stride())
                        .all(|g| g[8..].iter().all(|&b| b == 0xff))),
                    "codes_min" => assert!(a
                        .chunks_exact(fmt.stride())
                        .all(|g| g[8..].iter().all(|&b| b == 0))),
                    "zero_scale" => assert!(a
                        .chunks_exact(fmt.stride())
                        .all(|g| g[0..2] == [0, 0] || g[0..2] == [0, 0x80])),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn activation_rows_are_a_function_of_the_row_only() {
        let k = 512;
        for pat in XPATS {
            let all = x_rows(pat, 3, 8, k);
            assert_eq!(all.len(), 8 * k);
            let prefix = x_rows(pat, 3, 3, k);
            assert_eq!(
                &all[..3 * k],
                &prefix[..],
                "{pat}: row r must not depend on the row count"
            );
            assert!(all.iter().all(|v| v.is_finite()), "{pat}: finite");
            if !["onehot", "ramp", "altsign"].contains(&pat) {
                assert_ne!(all, x_rows(pat, 4, 8, k), "{pat}: seeded");
            }
        }
    }

    #[test]
    fn activation_patterns_have_their_shape() {
        let k = 256;
        let dup = x_rows("dup", 1, 4, k);
        assert!(dup.chunks(k).all(|r| r == &dup[..k]));
        let pairs = x_rows("pairs", 1, 4, k);
        assert_eq!(pairs[..k], pairs[k..2 * k]);
        assert_eq!(pairs[2 * k..3 * k], pairs[3 * k..]);
        assert_ne!(pairs[..k], pairs[2 * k..3 * k]);
        let zo = x_rows("zero_odd", 1, 4, k);
        assert!(zo[k..2 * k].iter().all(|v| v.to_bits() == 0));
        assert!(zo[..k].iter().any(|v| *v != 0.0));
        let rand = x_rows("rand", 1, 8, k);
        for a in 0..8 {
            for b in a + 1..8 {
                assert_ne!(
                    rand[a * k..(a + 1) * k],
                    rand[b * k..(b + 1) * k],
                    "rows {a} {b} distinct"
                );
            }
        }
        let sub = x_rows("subnormal", 1, 2, k);
        assert!(sub.iter().any(|v| *v != 0.0 && v.abs() < f32::MIN_POSITIVE));
    }

    #[test]
    fn diff_reports_first_word_and_parities() {
        let want = vec![1u32, 2, 3, 4, 5, 6];
        assert_eq!(diff_words(&want, &want, 3), None);
        let mut got = want.clone();
        got[4] = 9; // row 1, col 1
        got[5] = 9; // row 1, col 2
        let d = diff_words(&got, &want, 3).unwrap();
        assert_eq!(
            d,
            Diff {
                count: 2,
                first: 4,
                odd_rows: 2,
                even_rows: 0,
                odd_cols: 1,
                even_cols: 1
            }
        );
        let text = describe(&d, &got, &want, 3);
        assert!(text.contains("row=1 col=1"), "{text}");
        assert!(text.contains("got=0x00000009"), "{text}");
        assert!(text.contains("want=0x00000005"), "{text}");
    }

    #[test]
    fn guards_are_checked_on_both_sides() {
        let mut v = vec![NAN_BITS; 8];
        v[3] = 1;
        v[4] = 2;
        assert!(guards_clean(&v, 3, 2));
        v[0] = 0;
        assert!(!guards_clean(&v, 3, 2));
        v[0] = NAN_BITS;
        v[7] = 0;
        assert!(!guards_clean(&v, 3, 2));
    }

    #[test]
    fn default_dims_are_the_flash_next_gdn_group() {
        assert_eq!(2 * KEY_WIDTH + VALUE_WIDTH, 10240);
        assert_eq!(VALUE_WIDTH, 6144);
        assert_eq!(Fmt::from_qt(44), Some(Fmt::Mq4));
        assert_eq!(Fmt::from_qt(47), Some(Fmt::Mq6));
        assert_eq!(Fmt::from_qt(54), None);
    }
}
