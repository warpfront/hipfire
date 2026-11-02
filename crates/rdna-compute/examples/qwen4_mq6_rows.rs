// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 MQ6G256V2 prefill GEMM row-invariance ORACLE (gfx1151; developer
//! example).
//!
//! Question: is an output row of the Flash-Next MQ6 prefill projection the
//! same bytes whatever number of other rows share its launch?  Every shape is
//! run at the canonical R = 8192 rows (the reference) and then, for every R of
//! `--rows`, on the FIRST R input rows (offset 0) and with the same R rows
//! placed at row offsets o inside an R + o row batch whose other rows are
//! different random data.  Each output row's bytes must equal the reference's
//! same row.  Synthetic weights (random 6-bit codes, random finite F16
//! scale/zero headers in the real 200 B group layout), random F32 inputs; no
//! model.
//!
//! The call sequence is `project_weights_inner` / `project_output_into_hyper_write`
//! of `hipfire-dispatch::pipeline::layer_ops` re-issued on `rdna-compute`
//! launchers under `Gpu::qwen4_scope = true` (what the Qwen4 forward sets), with
//! the routing predicates fed `gpu.route_rows(R)` exactly as the invariant
//! profile feeds them (the profile hook is idempotent where it lives inside a
//! launcher):
//!   * `gemm_mq6g256v2_xf16_applies(k, route_rows(R))` true: ONE shared F16
//!     rotation `rotate_x_mq_batched_f16` (kernel `mq_rotate_x_f16`), then the
//!     three-region a/b/z fold `gemm_mq6g256v2_xf16_regions` when
//!     `qwen4_mq6_x4_regions()` and three F32 members, else one
//!     `gemm_mq6g256v2_xf16` per projection (F32 overwrite; a BF16 output
//!     tensor takes the `_bf16out` entry).  The entry that actually runs
//!     (hipcc `gemm_mq6g256v2_wmma_gfx11_bt8_x4*`, the U3 tile twins
//!     `gemm_mq6g256v2_wmma_gfx11_u3_b*_r*_p*{,_bf16out,_regions,_hcw}`, or the
//!     PM builder `qwen4_mq6_x4_pm_gfx1151_w8{,_bf16out,_regions}`) is chosen
//!     inside the launcher by (m, k, n) and is reported per run from
//!     `Gpu::last_launched_kernel`.
//!   * otherwise (legacy small R): `rotate_x_mq_batched` (F32) then
//!     `gemv_mq6g256v2_x4` / `gemm_mq6g256v2_f32_rows_x4` (R <= 8, two or more
//!     projections), `gemv_mq6g256v2` / `gemm_mq6g256v2_f32_rows` (R = 1 / 2..=8)
//!     or `gemm_mq6g256v2` (memset + residual WMMA).  A BF16-output projection
//!     has no such route (the dispatch never reaches it with a BF16 tensor):
//!     reported as `skip`.
//!   * HC write (`gdn_out_hcw_*`): `gemm_mq6g256v2_hcw_applies` then
//!     `rotate_x_mq_batched_f16` + `gemm_mq6g256v2_hcw` (streams updated in
//!     place from per-row random streams and gates); `skip` where it does not
//!     apply (the unfused route is the plain `gdn_out` shape + `hyper_write`).
//!
//! CLI: [--scope on|off|both] (default both: `Gpu::prefill_invariant_scope`;
//!       the reference is recomputed per scope and the two are compared)
//!      [--rows LIST] (`N`, `A-B`, `A-B:STEP`, comma separated; default
//!       1..=64, every multiple of 16 to 1024, 255,256,257,511,512,513,1023,
//!       1025,2047,2048,2049,4095,4096,4097,8191)
//!      [--offsets LIST] (default `1,17,256,t`; `t` = 4096 - R, R < 4096 only)
//!      [--shapes NAME,..] [--ref-rows 8192] [--seed N] [--print all|fail]
//! Shapes: gdn_qkv_bf16 gdn_qkv_f32 gdn_z gdn_ab gdn_zba gdn_in_group gdn_out
//! gdn_out_hcw_bf16 gdn_out_hcw_f32 qsa_q qsa_kv qsa_idx.
//! Output: one JSON object per run on stdout (`event` start | ref |
//! ref_cross_scope | run | summary); exit 1 when any run differs or errors.

use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::HashMap;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const GROUP_BYTES: usize = 200;
const FILL_BASE: u64 = 1 << 40;
const DOMAIN_X: u64 = 0x5858_0000_0000_0001;
const DOMAIN_S: u64 = 0x5353_0000_0000_0002;
const DOMAIN_G: u64 = 0x4747_0000_0000_0003;

// ---------------------------------------------------------------- host math

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Finite normal F16 bits with a random sign and biased exponent in `lo..=hi`.
fn half(r: &mut Rng, lo: u64, hi: u64) -> u16 {
    let sign = (r.below(2) as u16) << 15;
    let e = (lo + r.below(hi - lo + 1)) as u16;
    sign | (e << 10) | r.below(1024) as u16
}

/// `m` rows x `k/256` MQ6G256V2 groups: `[s0 z0 s1 z1]` F16 LE headers
/// (scale 2^-9..2^-5, zero 8..64 scales away, an exact zero scale now and
/// then), 192 payload bytes of random 6-bit codes.
fn mq6_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed ^ 0x6d71_3600_0000_0001);
    let mut w = vec![0u8; m * (k / 256) * GROUP_BYTES];
    for g in w.chunks_exact_mut(GROUP_BYTES) {
        for h in 0..2 {
            let mut sc = half(&mut r, 6, 10);
            let zp = half(
                &mut r,
                ((sc >> 10) & 31) as u64 + 3,
                ((sc >> 10) & 31) as u64 + 6,
            );
            if r.below(32) == 0 {
                sc &= 0x8000;
            }
            g[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for c in g[8..].chunks_mut(8) {
            let v = r.next().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }
    w
}

/// Rows `first..first + n` (each a function of `(seed, domain, row index)`
/// only) of `width` F32 values: per-row magnitude 0.25..2 and a sparse outlier.
fn gen_rows(seed: u64, first: u64, n: usize, width: usize, domain: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(n * width);
    for j in 0..n as u64 {
        let idx = first + j;
        let mut r = Rng(seed ^ domain ^ idx.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let scale = [0.25f32, 0.5, 1.0, 2.0][(idx % 4) as usize];
        for i in 0..width {
            let x = (r.unit() - 0.5) * 4.0 * scale;
            v.push(if i % 97 == 13 { x * 8.0 } else { x });
        }
    }
    v
}

fn bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    ((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) as u16
}

fn to_bytes(v: &[f32], bf16: bool) -> Vec<u8> {
    if bf16 {
        v.iter().flat_map(|x| bf16_bits(*x).to_le_bytes()).collect()
    } else {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

fn bytes_of(v: &[f32]) -> &[u8] {
    // SAFETY: f32 slice read as bytes of the same extent.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn rows_default() -> Vec<usize> {
    let mut v: Vec<usize> = (1..=64).collect();
    v.extend((16..=1024).step_by(16));
    v.extend([
        255, 256, 257, 511, 512, 513, 1023, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191,
    ]);
    v.sort_unstable();
    v.dedup();
    v
}

fn rows_list(spec: &str) -> Res<Vec<usize>> {
    if spec.is_empty() {
        return Ok(rows_default());
    }
    let mut v = Vec::new();
    for tok in spec.split(',').filter(|t| !t.is_empty()) {
        let (range, step) = match tok.split_once(':') {
            Some((r, s)) => (r, s.parse::<usize>()?),
            None => (tok, 1),
        };
        if step == 0 {
            return Err(format!("--rows {tok}: step 0").into());
        }
        match range.split_once('-') {
            Some((a, b)) => v.extend((a.parse::<usize>()?..=b.parse::<usize>()?).step_by(step)),
            None => v.push(range.parse()?),
        }
    }
    v.sort_unstable();
    v.dedup();
    if v.first() == Some(&0) {
        return Err("--rows: 0 rows".into());
    }
    Ok(v)
}

fn rows_offsets(spec: &str, r: usize) -> Res<Vec<usize>> {
    let mut v = Vec::new();
    for tok in spec.split(',').filter(|s| !s.is_empty()) {
        if tok == "t" {
            if r < 4096 {
                v.push(4096 - r);
            }
        } else {
            v.push(tok.parse()?);
        }
    }
    v.retain(|&o| o > 0);
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

/// Byte comparison by rows: `None` when identical, else the JSON of the first
/// differing row / element / bytes and the number of differing rows.
fn rows_diff(got: &[u8], want: &[u8], rb: usize, elem: usize) -> Option<Value> {
    if got == want {
        return None;
    }
    let n = got.len().min(want.len()) / rb;
    let mut count = 0usize;
    let mut first: Option<(usize, usize)> = None;
    for r in 0..n {
        let (a, b) = (&got[r * rb..(r + 1) * rb], &want[r * rb..(r + 1) * rb]);
        if a != b {
            count += 1;
            if first.is_none() {
                first = Some((r, a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0)));
            }
        }
    }
    let (row, byte) = first.unwrap_or((0, 0));
    let col = byte / elem;
    let at = |v: &[u8]| hex(&v[row * rb + col * elem..row * rb + (col + 1) * elem]);
    Some(json!({
        "row": row, "col": col, "byte": byte,
        "got": at(got), "want": at(want), "rows_differing": count,
    }))
}

// ---------------------------------------------------------------- shapes

/// One projection group of a Flash-Next MQ6 site.
struct Shape {
    name: &'static str,
    /// Output rows (M) of each projection of the group (one shared input).
    ms: &'static [usize],
    k: usize,
    /// Projection 0 writes a BF16 tensor (the chunked GDN qkv).
    bf16_first: bool,
    /// `region_members` of `project_weights_inner` (the a/b/z fold).
    members: Option<&'static [bool]>,
    /// `Some(bf16_state)`: the HC-write epilogue GEMM (`gemm_mq6g256v2_hcw`).
    hcw: Option<bool>,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "gdn_qkv_bf16",
        ms: &[10240],
        k: 2560,
        bf16_first: true,
        members: None,
        hcw: None,
    },
    Shape {
        name: "gdn_qkv_f32",
        ms: &[10240],
        k: 2560,
        bf16_first: false,
        members: None,
        hcw: None,
    },
    Shape {
        name: "gdn_z",
        ms: &[6144],
        k: 2560,
        bf16_first: false,
        members: None,
        hcw: None,
    },
    Shape {
        name: "gdn_ab",
        ms: &[48],
        k: 2560,
        bf16_first: false,
        members: None,
        hcw: None,
    },
    Shape {
        name: "gdn_zba",
        ms: &[6144, 48, 48],
        k: 2560,
        bf16_first: false,
        members: Some(&[true, true, true]),
        hcw: None,
    },
    Shape {
        name: "gdn_in_group",
        ms: &[10240, 6144, 48, 48],
        k: 2560,
        bf16_first: true,
        members: Some(&[false, true, true, true]),
        hcw: None,
    },
    Shape {
        name: "gdn_out",
        ms: &[2560],
        k: 6144,
        bf16_first: false,
        members: None,
        hcw: None,
    },
    Shape {
        name: "gdn_out_hcw_bf16",
        ms: &[2560],
        k: 6144,
        bf16_first: false,
        members: None,
        hcw: Some(true),
    },
    Shape {
        name: "gdn_out_hcw_f32",
        ms: &[2560],
        k: 6144,
        bf16_first: false,
        members: None,
        hcw: Some(false),
    },
    Shape {
        name: "qsa_q",
        ms: &[12288],
        k: 2560,
        bf16_first: false,
        members: None,
        hcw: None,
    },
    Shape {
        name: "qsa_kv",
        ms: &[512],
        k: 2560,
        bf16_first: false,
        members: None,
        hcw: None,
    },
    Shape {
        name: "qsa_idx",
        ms: &[640],
        k: 2560,
        bf16_first: false,
        members: None,
        hcw: None,
    },
];

/// Device state of one shape, sized for the largest batch.
struct State {
    w: Vec<GpuTensor>,
    /// (tensor, row bytes, element bytes) per output; the HC-write shape's one
    /// output is its in-place streams.
    outs: Vec<(GpuTensor, usize, usize)>,
    x: GpuTensor,
    rot: GpuTensor,
    gates: Option<GpuTensor>,
}

impl State {
    fn new(gpu: &mut Gpu, sd: &Shape, seed: u64, si: usize, tmax: usize) -> Res<State> {
        let mut w = Vec::new();
        for (i, &m) in sd.ms.iter().enumerate() {
            let bytes = mq6_weights(m, sd.k, seed ^ ((si as u64) << 20) ^ ((i as u64) << 8));
            w.push(gpu.upload_raw(&bytes, &[bytes.len()])?);
        }
        let mut outs = Vec::new();
        match sd.hcw {
            Some(bf16_state) => {
                let es = if bf16_state { 2 } else { 4 };
                let rb = 4 * sd.ms[0] * es;
                outs.push((gpu.zeros(&[(tmax * rb).div_ceil(4)], DType::F32)?, rb, es));
            }
            None => {
                for (i, &m) in sd.ms.iter().enumerate() {
                    let bf = i == 0 && sd.bf16_first;
                    let t = gpu.zeros(&[tmax * m], if bf { DType::BF16 } else { DType::F32 })?;
                    outs.push((t, m * if bf { 2 } else { 4 }, if bf { 2 } else { 4 }));
                }
            }
        }
        let gates = if sd.hcw.is_some() {
            Some(gpu.zeros(&[tmax * 4], DType::F32)?)
        } else {
            None
        };
        Ok(State {
            w,
            outs,
            x: gpu.zeros(&[tmax * sd.k], DType::F32)?,
            rot: gpu.zeros(&[tmax * sd.k], DType::F32)?,
            gates,
        })
    }
}

/// Host pools of one shape: base rows (the tested rows) and filler rows
/// (other data placed before them at an offset).
struct Pools<'a> {
    x_base: &'a [f32],
    x_fill: &'a [f32],
    s_base: &'a [u8],
    s_fill: &'a [u8],
    g_base: &'a [f32],
    g_fill: &'a [f32],
}

struct Run {
    /// Output bytes of the `t0` tested rows, one buffer per output.
    outs: Vec<Vec<u8>>,
    kernels: Vec<String>,
    route: &'static str,
    skip: Option<&'static str>,
}

fn note(gpu: &Gpu, kernels: &mut Vec<String>) {
    if let Some(n) = gpu.last_launched_kernel() {
        if kernels.last().map(String::as_str) != Some(n) {
            kernels.push(n.to_string());
        }
    }
}

/// `project_weights_inner` for a group of MQ6G256V2 projections of one F32
/// input (`st.x`, `rows` rows).
fn project(
    gpu: &mut Gpu,
    sd: &Shape,
    st: &State,
    rows: usize,
    kernels: &mut Vec<String>,
) -> Res<(&'static str, Option<&'static str>)> {
    let k = sd.k;
    let n = sd.ms.len();
    let projs: Vec<(&GpuTensor, &GpuTensor, usize)> = (0..n)
        .map(|i| (&st.w[i], &st.outs[i].0, sd.ms[i]))
        .collect();
    if gpu.gemm_mq6g256v2_xf16_applies(k, gpu.route_rows(rows)) {
        let x16 = gpu.rotate_x_mq_batched_f16(&st.x, k, rows)?;
        note(gpu, kernels);
        let mut done = vec![false; n];
        let mut route = "xf16";
        if let (Some(members), true) = (sd.members, gpu.qwen4_mq6_x4_regions()) {
            let idx: Vec<usize> = (0..n)
                .filter(|&j| members[j] && !(j == 0 && sd.bf16_first))
                .collect();
            if idx.len() == 3 {
                let regions = [projs[idx[0]], projs[idx[1]], projs[idx[2]]];
                if gpu.gemm_mq6g256v2_xf16_regions(&regions, &x16, k, rows)? {
                    note(gpu, kernels);
                    for &j in &idx {
                        done[j] = true;
                    }
                    route = "xf16_regions";
                }
            }
        }
        for j in 0..n {
            if !done[j] {
                let (w, y, m) = projs[j];
                gpu.gemm_mq6g256v2_xf16(w, &x16, y, m, k, rows)?;
                note(gpu, kernels);
            }
        }
        return Ok((route, None));
    }
    if sd.bf16_first {
        return Ok(("legacy", Some("bf16_out_has_no_legacy_route")));
    }
    if rows > 1 {
        gpu.rotate_x_mq_batched(&st.x, &st.rot, k, rows)?;
    } else {
        gpu.rotate_x_mq(&st.x, &st.rot, k)?;
    }
    note(gpu, kernels);
    if rows <= 8 && n >= 2 {
        let all: Vec<usize> = (0..n).collect();
        for chunk in all.chunks(4) {
            let part = |i: usize| {
                let (w, y, m) = projs[chunk[i.min(chunk.len() - 1)]];
                (w, y, if i < chunk.len() { m } else { 0 })
            };
            let parts = [part(0), part(1), part(2), part(3)];
            if rows == 1 {
                gpu.gemv_mq6g256v2_x4(parts, &st.rot, k)?;
            } else {
                gpu.gemm_mq6g256v2_f32_rows_x4(parts, &st.rot, k, rows)?;
            }
            note(gpu, kernels);
        }
    } else {
        for &(w, y, m) in &projs {
            match rows {
                1 => gpu.gemv_mq6g256v2(w, &st.rot, y, m, k)?,
                2..=8 => gpu.gemm_mq6g256v2_f32_rows(w, &st.rot, y, m, k, rows)?,
                _ => gpu.gemm_mq6g256v2(w, &st.rot, y, m, k, rows)?,
            }
            note(gpu, kernels);
        }
    }
    Ok(("legacy", None))
}

/// One launch over a batch of `o + t0` rows: `o` filler rows (other data)
/// followed by the first `t0` pool rows, at the current scope.
fn exec(gpu: &mut Gpu, sd: &Shape, st: &State, pl: &Pools<'_>, o: usize, t0: usize) -> Res<Run> {
    let rows = o + t0;
    let k = sd.k;
    if o > 0 {
        gpu.hip
            .memcpy_htod_offset(&st.x.buf, 0, bytes_of(&pl.x_fill[..o * k]))?;
    }
    gpu.hip
        .memcpy_htod_offset(&st.x.buf, o * k * 4, bytes_of(&pl.x_base[..t0 * k]))?;
    let mut kernels = Vec::new();
    let (route, skip): (&'static str, Option<&'static str>) = if let Some(bf16_state) = sd.hcw {
        let (out, rb, _) = &st.outs[0];
        let gates = st.gates.as_ref().ok_or("hcw gates missing")?;
        if o > 0 {
            gpu.hip
                .memcpy_htod_offset(&out.buf, 0, &pl.s_fill[..o * rb])?;
            gpu.hip
                .memcpy_htod_offset(&gates.buf, 0, bytes_of(&pl.g_fill[..o * 4]))?;
        }
        gpu.hip
            .memcpy_htod_offset(&out.buf, o * rb, &pl.s_base[..t0 * rb])?;
        gpu.hip
            .memcpy_htod_offset(&gates.buf, o * 16, bytes_of(&pl.g_base[..t0 * 4]))?;
        let m = sd.ms[0];
        if gpu.gemm_mq6g256v2_hcw_applies(m, k, gpu.route_rows(rows)) {
            let x16 = gpu.rotate_x_mq_batched_f16(&st.x, k, rows)?;
            note(gpu, &mut kernels);
            gpu.gemm_mq6g256v2_hcw(
                &st.w[0], &x16, true, None, m, k, rows, out, gates, bf16_state,
            )?;
            note(gpu, &mut kernels);
            ("hcw", None)
        } else {
            ("legacy", Some("hcw_not_applicable"))
        }
    } else {
        for (buf, rb, _) in &st.outs {
            gpu.hip.memset(&buf.buf, 0xA5, rows * rb)?;
        }
        project(gpu, sd, st, rows, &mut kernels)?
    };
    gpu.hip.device_synchronize()?;
    let mut outs = Vec::new();
    if skip.is_none() {
        for (buf, rb, _) in &st.outs {
            let mut v = vec![0u8; t0 * rb];
            gpu.hip.memcpy_dtoh_at(&mut v, &buf.buf, o * rb)?;
            outs.push(v);
        }
    }
    Ok(Run {
        outs,
        kernels,
        route,
        skip,
    })
}

/// First differing output of `run` against `want` (the reference's rows).
fn compare(st: &State, run: &Run, want: &Run, t0: usize) -> Option<Value> {
    for (j, (_, rb, elem)) in st.outs.iter().enumerate() {
        if let Some(mut d) = rows_diff(&run.outs[j], &want.outs[j][..t0 * rb], *rb, *elem) {
            d["out"] = json!(j);
            return Some(d);
        }
    }
    None
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let scopes: Vec<bool> = match arg(&args, "--scope", "both").as_str() {
        "on" => vec![true],
        "off" => vec![false],
        "both" => vec![false, true],
        other => return Err(format!("--scope {other}: expected on, off or both").into()),
    };
    let seed: u64 = arg(&args, "--seed", "1").parse()?;
    let ref_rows: usize = arg(&args, "--ref-rows", "8192").parse()?;
    let rows = rows_list(&arg(&args, "--rows", ""))?;
    let offset_spec = arg(&args, "--offsets", "1,17,256,t");
    let print_all = arg(&args, "--print", "all") != "fail";
    let filter = arg(&args, "--shapes", "");
    let wanted: Vec<&str> = filter.split(',').filter(|s| !s.is_empty()).collect();
    for w in &wanted {
        if !SHAPES.iter().any(|s| s.name == *w) {
            return Err(format!("--shapes {w}: unknown").into());
        }
    }
    let rmax = *rows.last().ok_or("--rows is empty")?;
    if ref_rows == 0 || rmax > ref_rows {
        return Err(format!("--rows max {rmax} exceeds --ref-rows {ref_rows}").into());
    }
    let (mut tmax, mut omax) = (ref_rows, 1usize);
    for &r in &rows {
        for o in rows_offsets(&offset_spec, r)? {
            tmax = tmax.max(r + o);
            omax = omax.max(o);
        }
    }

    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1151" {
        return Err(format!(
            "the MQ6 prefill route sweep is gfx1151-only, device is {}",
            gpu.arch
        )
        .into());
    }
    // What `Qwen4GpuForward::forward_chunk_scoped` sets for the forward.
    gpu.qwen4_scope = true;
    println!(
        "{}",
        json!({
            "event": "start", "arch": gpu.arch, "seed": seed, "ref_rows": ref_rows,
            "rows_count": rows.len(), "offsets": offset_spec,
            "scopes": scopes.iter().map(|s| if *s { "on" } else { "off" }).collect::<Vec<_>>(),
            "qwen4_scope": gpu.qwen4_scope,
            "profile_env_flag": rdna_compute::prefill_invariant::prefill_pm_invariant_enabled(),
            "mq6_x4_regions": gpu.qwen4_mq6_x4_regions(),
            "mq6_x4_tile": gpu.qwen4_mq6_x4_tile().map(|t| t.to_vec()),
        })
    );

    let mut xpools: HashMap<usize, (Vec<f32>, Vec<f32>)> = HashMap::new();
    let (mut runs, mut fails, mut skips) = (0usize, 0usize, 0usize);
    for (si, sd) in SHAPES.iter().enumerate() {
        if !wanted.is_empty() && !wanted.contains(&sd.name) {
            continue;
        }
        let st = State::new(&mut gpu, sd, seed, si, tmax)?;
        xpools.entry(sd.k).or_insert_with(|| {
            (
                gen_rows(seed, 0, ref_rows, sd.k, DOMAIN_X),
                gen_rows(seed, FILL_BASE, omax, sd.k, DOMAIN_X),
            )
        });
        let (xb, xf) = &xpools[&sd.k];
        let (mut s_base, mut s_fill, mut g_base, mut g_fill) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        if let Some(bf16_state) = sd.hcw {
            let width = 4 * sd.ms[0];
            s_base = to_bytes(&gen_rows(seed, 0, ref_rows, width, DOMAIN_S), bf16_state);
            s_fill = to_bytes(
                &gen_rows(seed, FILL_BASE, omax, width, DOMAIN_S),
                bf16_state,
            );
            g_base = gen_rows(seed, 0, ref_rows, 4, DOMAIN_G);
            g_fill = gen_rows(seed, FILL_BASE, omax, 4, DOMAIN_G);
        }
        let pl = Pools {
            x_base: xb,
            x_fill: xf,
            s_base: &s_base,
            s_fill: &s_fill,
            g_base: &g_base,
            g_fill: &g_fill,
        };

        let mut prev: Option<Run> = None;
        for &scope in &scopes {
            gpu.prefill_invariant_scope = scope;
            let sname = if scope { "on" } else { "off" };
            let rf = match exec(&mut gpu, sd, &st, &pl, 0, ref_rows) {
                Ok(r) => r,
                Err(e) => {
                    fails += 1;
                    println!(
                        "{}",
                        json!({"event": "ref", "shape": sd.name, "scope": sname, "verdict": "error", "error": e.to_string()})
                    );
                    continue;
                }
            };
            println!(
                "{}",
                json!({
                    "event": "ref", "shape": sd.name, "scope": sname, "rows": ref_rows, "route": rf.route,
                    "verdict": if rf.skip.is_some() { "skip" } else { "ok" }, "skip": rf.skip,
                    "kernels": rf.kernels,
                })
            );
            if rf.skip.is_some() {
                continue;
            }
            if let Some(p) = &prev {
                let d = compare(&st, &rf, p, ref_rows);
                runs += 1;
                fails += usize::from(d.is_some());
                println!(
                    "{}",
                    json!({
                        "event": "ref_cross_scope", "shape": sd.name, "rows": ref_rows,
                        "verdict": if d.is_some() { "differs" } else { "identical" }, "first_diff": d,
                    })
                );
            }
            for &r in &rows {
                let mut offs = vec![0usize];
                offs.extend(rows_offsets(&offset_spec, r)?);
                for o in offs {
                    if o == 0 && r == ref_rows {
                        continue;
                    }
                    runs += 1;
                    let outcome = exec(&mut gpu, sd, &st, &pl, o, r).map_err(|e| e.to_string());
                    let (verdict, diff, route, skip, kernels, err): (
                        &str,
                        Option<Value>,
                        &str,
                        Option<&str>,
                        Vec<String>,
                        Option<String>,
                    ) = match outcome {
                        Ok(run) if run.skip.is_some() => {
                            skips += 1;
                            ("skip", None, run.route, run.skip, run.kernels, None)
                        }
                        Ok(run) => {
                            let d = compare(&st, &run, &rf, r);
                            (
                                if d.is_some() { "differs" } else { "identical" },
                                d,
                                run.route,
                                None,
                                run.kernels,
                                None,
                            )
                        }
                        Err(e) => ("error", None, "?", None, Vec::new(), Some(e)),
                    };
                    if verdict == "differs" || verdict == "error" {
                        fails += 1;
                    }
                    if print_all || verdict == "differs" || verdict == "error" {
                        println!(
                            "{}",
                            json!({
                                "event": "run", "shape": sd.name, "k": sd.k, "ms": sd.ms, "scope": sname,
                                "rows": r, "offset": o, "batch": r + o, "route_rows": gpu.route_rows(r + o),
                                "route": route, "verdict": verdict, "skip": skip, "error": err,
                                "first_diff": diff, "kernels": kernels,
                            })
                        );
                    }
                }
            }
            prev = Some(rf);
        }
        for (t, _, _) in st.outs {
            gpu.free_tensor(t)?;
        }
        gpu.free_tensor(st.x)?;
        gpu.free_tensor(st.rot)?;
        if let Some(g) = st.gates {
            gpu.free_tensor(g)?;
        }
    }
    println!(
        "{}",
        json!({"event": "summary", "runs": runs, "differing_or_error": fails, "skipped": skips})
    );
    if fails > 0 {
        std::process::exit(1);
    }
    Ok(())
}
