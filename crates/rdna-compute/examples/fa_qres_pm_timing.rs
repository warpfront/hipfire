// SPDX-License-Identifier: Apache-2.0
//! LAB-only GPU event timing of the A3B native-fp8 Q-resident FA2 prefill attention:
//! the embedded PeaceMaker twin (`fa_qresident_a3b_pm_gfx1201`) against the hipcc kernel,
//! both launched through `Gpu::attention_fp8_e4m3_fa2_gqa_qresident_v2_q8*_a3b_routed_gfx1201`
//! with an explicit route (`HIPFIRE_FA_QRESIDENT_PM` is ignored).
//!
//! ```text
//! CPU only (no HIP init):
//!   fa_qres_pm_timing --plan            shapes, launch counts, device-memory census
//! GPU (gfx1201, timing class):
//!   fa_qres_pm_timing                   all shapes
//!   fa_qres_pm_timing --shape SUBSTR    shapes whose name contains SUBSTR
//! ```
//!
//! Shapes (the two pp16000 prefill chunks of A3B Qwen3.6): `n8192_s0` (8192 rows, context
//! 8192, start 0, direct), `n7808_s8192` (7808 rows, context 16000, positions 8192.., direct)
//! and `n7808_s8192_range` (the same rows through the range twin: FULL planes of
//! `rows + 512`, `q_row_offset` 512). Per shape: both arms run once on identical inputs and
//! the outputs must be BYTE-IDENTICAL (raw u8) before any timing (fail closed: exit 1),
//! then 5 warmups and 30 timed launches per arm, interleaved ABAB (hipcc then PM per
//! iteration), each launch timed by its own HIP event pair. JSON on stdout per shape:
//! median / min / mean us per launch of both arms and the ratio pm/hip (of the medians and
//! of the means).
use hip_bridge::HipResult;
use rdna_compute::attention::A3bQresRoute;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::error::Error;

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const H: usize = 16;
const KV: usize = 2;
const D: usize = 256;
const ROW: usize = H * D;
/// Bytes of one native-fp8 cache position: 2 heads x 256 codes + 2 f16 scales.
const POS_BYTES: usize = KV * (D + 2);
const SCALE_OFF: usize = KV * D;
const Q_BYTES_ROW: usize = H * D + H * 4;
const WARMUPS: usize = 5;
const ITERS: usize = 30;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    rows: usize,
    start: usize,
    ctx: usize,
    /// Range twin: `(q_total_rows, q_row_offset)`.
    range: Option<(usize, usize)>,
}
const SHAPES: [Shape; 3] = [
    Shape {
        name: "n8192_s0",
        rows: 8192,
        start: 0,
        ctx: 8192,
        range: None,
    },
    Shape {
        name: "n7808_s8192",
        rows: 7808,
        start: 8192,
        ctx: 16000,
        range: None,
    },
    Shape {
        name: "n7808_s8192_range",
        rows: 7808,
        start: 8192,
        ctx: 16000,
        range: Some((7808 + 512, 512)),
    },
];
impl Shape {
    fn q_total(&self) -> usize {
        self.range.map_or(self.rows, |(total, _)| total)
    }
    fn q_bytes(&self) -> usize {
        self.q_total() * Q_BYTES_ROW
    }
    fn cache_bytes(&self) -> usize {
        self.ctx * POS_BYTES
    }
    fn out_bytes(&self) -> usize {
        self.rows * ROW * 4
    }
    /// Device bytes: Q planes, K, V, positions, two outputs.
    fn device_bytes(&self) -> usize {
        self.q_bytes() + 2 * self.cache_bytes() + self.rows * 4 + 2 * self.out_bytes()
    }
    fn grid(&self) -> [u32; 3] {
        [
            (self.rows.min(512) * 8).div_ceil(384) as u32,
            2,
            self.rows.div_ceil(512) as u32,
        ]
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A finite E4M3FN code (never `0x7f` / `0xff`, the NaN encodings).
    fn code(&mut self) -> u8 {
        loop {
            let b = (self.next() >> 24) as u8;
            if b & 0x7f != 0x7f {
                return b;
            }
        }
    }
}

/// Deterministic finite fixtures: Q two planes (codes, then F32 scales), K/V native rows.
fn fixtures(s: &Shape) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<i32>) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ ((s.rows as u64) << 20) ^ s.start as u64);
    let total = s.q_total();
    let mut q = Vec::with_capacity(s.q_bytes());
    for _ in 0..total * H * D {
        q.push(rng.code());
    }
    for i in 0..total * H {
        let scale = 0.01f32 + (i % 13) as f32 * 0.001;
        q.extend_from_slice(&scale.to_le_bytes());
    }
    let scales = [0x1400u16, 0x1800, 0x1c00, 0x2000];
    let cache = |rng: &mut Rng| {
        let mut c = vec![0u8; s.cache_bytes()];
        for p in 0..s.ctx {
            let row = &mut c[p * POS_BYTES..(p + 1) * POS_BYTES];
            for b in row[..SCALE_OFF].iter_mut() {
                *b = rng.code();
            }
            for g in 0..KV {
                let sc = scales[(p * 3 + g) % scales.len()].to_le_bytes();
                row[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2].copy_from_slice(&sc);
            }
        }
        c
    };
    let k = cache(&mut rng);
    let v = cache(&mut rng);
    let positions: Vec<i32> = (0..s.rows).map(|r| (s.start + r) as i32).collect();
    (q, k, v, positions)
}

struct Buffers {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    pos: GpuTensor,
    /// One output per arm (arm 0 is hipcc).
    outs: Vec<GpuTensor>,
}

/// A timed arm: its name and kernel route.
#[derive(Clone)]
struct Arm {
    name: String,
    route: A3bQresRoute,
}

fn upload(gpu: &Gpu, t: &GpuTensor, data: &[u8]) -> Result<()> {
    gpu.hip.memcpy_htod_offset(&t.buf, 0, data)?;
    Ok(())
}
fn download(gpu: &Gpu, t: &GpuTensor, len: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; len];
    gpu.hip.memcpy_dtoh_at(&mut v, &t.buf, 0)?;
    Ok(v)
}

fn launch(
    gpu: &mut Gpu,
    s: &Shape,
    route: A3bQresRoute,
    b: &Buffers,
    out: &GpuTensor,
) -> HipResult<()> {
    match s.range {
        None => gpu.attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_routed_gfx1201(
            route, &b.q, &b.k, &b.v, out, &b.pos, H, KV, D, s.ctx, s.rows,
        ),
        Some((total, off)) => gpu
            .attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_routed_gfx1201(
                route, &b.q, &b.k, &b.v, out, &b.pos, H, KV, D, s.ctx, s.rows, total, off,
            ),
    }
}

fn stats(us: &mut Vec<f64>) -> Value {
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = us.len();
    let median = if n % 2 == 1 {
        us[n / 2]
    } else {
        (us[n / 2 - 1] + us[n / 2]) / 2.0
    };
    let mean = us.iter().sum::<f64>() / n as f64;
    json!({"median_us": median, "min_us": us[0], "mean_us": mean, "max_us": us[n - 1],
        "samples": n})
}

fn time_shape(gpu: &mut Gpu, s: &Shape, arms: &[Arm]) -> Result<bool> {
    let arms: Vec<&Arm> = arms.iter().collect();
    let (q, k, v, positions) = fixtures(s);
    let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
    let mut outs = Vec::new();
    for _ in &arms {
        outs.push(gpu.alloc_tensor(&[s.rows * ROW], DType::F32)?);
    }
    let b = Buffers {
        q: gpu.alloc_tensor(&[s.q_bytes()], DType::Raw)?,
        k: gpu.alloc_tensor(&[s.cache_bytes()], DType::Raw)?,
        v: gpu.alloc_tensor(&[s.cache_bytes()], DType::Raw)?,
        pos: gpu.alloc_tensor(&[s.rows], DType::F32)?,
        outs,
    };
    upload(gpu, &b.q, &q)?;
    upload(gpu, &b.k, &k)?;
    upload(gpu, &b.v, &v)?;
    upload(gpu, &b.pos, &pos_bytes)?;
    // Distinct poisons: an unwritten output cannot match.
    for (i, o) in b.outs.iter().enumerate() {
        gpu.hip
            .memset(&o.buf, if i == 0 { 0xA5 } else { 0x5A }, s.out_bytes())?;
    }
    gpu.hip.device_synchronize()?;
    // Byte identity of every arm against hipcc first, fail closed.
    for (a, o) in arms.iter().zip(&b.outs) {
        launch(gpu, s, a.route, &b, o)?;
    }
    gpu.hip.device_synchronize()?;
    let hip = download(gpu, &b.outs[0], s.out_bytes())?;
    let mut all_ok = hip.iter().any(|&x| x != 0xA5);
    let mut ident = Vec::new();
    for (a, o) in arms.iter().zip(&b.outs).skip(1) {
        let pm = download(gpu, o, s.out_bytes())?;
        let mismatch_words = hip
            .chunks_exact(4)
            .zip(pm.chunks_exact(4))
            .filter(|(x, y)| x != y)
            .count();
        let first = hip
            .chunks_exact(4)
            .zip(pm.chunks_exact(4))
            .position(|(x, y)| x != y)
            .map(|i| {
                json!({"row": i / ROW, "head": (i % ROW) / D, "dim": i % D,
                    "hip_bits": format!("0x{:08x}", u32::from_le_bytes(hip[i * 4..i * 4 + 4].try_into().unwrap())),
                    "pm_bits": format!("0x{:08x}", u32::from_le_bytes(pm[i * 4..i * 4 + 4].try_into().unwrap()))})
            });
        let identical = mismatch_words == 0 && hip.len() == pm.len();
        let written = pm.iter().any(|&x| x != 0x5A);
        all_ok &= identical && written;
        ident.push(
            json!({"arm": a.name, "byte_identical": identical, "outputs_written": written,
            "mismatch_words": mismatch_words, "first_mismatch": first}),
        );
    }
    if !all_ok {
        println!(
            "{}",
            json!({"kind": "timing_shape", "shape": s.name, "rows": s.rows, "context": s.ctx,
                "identity": ident, "timed": false, "pass": false})
        );
        return Ok(false);
    }
    drop(hip);
    let (ea, eb) = (gpu.hip.event_create()?, gpu.hip.event_create()?);
    let time_one = |gpu: &mut Gpu, i: usize| -> Result<f64> {
        gpu.hip.event_record(&ea, None)?;
        launch(gpu, s, arms[i].route, &b, &b.outs[i])?;
        gpu.hip.event_record(&eb, None)?;
        gpu.hip.event_synchronize(&eb)?;
        Ok(gpu.hip.event_elapsed_ms(&ea, &eb)? as f64 * 1000.0)
    };
    for _ in 0..WARMUPS {
        for i in 0..arms.len() {
            time_one(gpu, i)?;
        }
    }
    let mut us: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for _ in 0..ITERS {
        for i in 0..arms.len() {
            us[i].push(time_one(gpu, i)?);
        }
    }
    gpu.hip.event_destroy(ea)?;
    gpu.hip.event_destroy(eb)?;
    let stat: Vec<Value> = us.iter_mut().map(stats).collect();
    let ratio =
        |i: usize, key: &str| stat[i][key].as_f64().unwrap() / stat[0][key].as_f64().unwrap();
    let mut arms_json = serde_json::Map::new();
    for (i, a) in arms.iter().enumerate() {
        let mut e = stat[i].clone();
        if i > 0 {
            e["ratio_over_hip_median"] = json!(ratio(i, "median_us"));
            e["ratio_over_hip_mean"] = json!(ratio(i, "mean_us"));
            e["ratio_over_hip_min"] = json!(ratio(i, "min_us"));
        }
        arms_json.insert(a.name.clone(), e);
    }
    let mut out = json!({"kind": "timing_shape", "shape": s.name, "rows": s.rows, "start": s.start,
            "context": s.ctx, "range": s.range.map(|(t, o)| json!({"q_total_rows": t, "q_row_offset": o})),
            "grid": s.grid(), "block": [768, 1, 1], "lds": 49_936,
            "byte_identical": true, "warmups_per_arm": WARMUPS, "timed_per_arm": ITERS,
            "order": "interleaved ABAB (A = hipcc, B = PM)",
            "arms": arms_json, "pass": true});
    // The two-arm form of the original harness stays readable on its own.
    if let Some(pm) = arms.iter().position(|a| a.name == "pm") {
        out["hipcc"] = stat[0].clone();
        out["pm"] = stat[pm].clone();
        out["ratio_pm_over_hip_median"] = json!(ratio(pm, "median_us"));
        out["ratio_pm_over_hip_mean"] = json!(ratio(pm, "mean_us"));
        out["ratio_pm_over_hip_min"] = json!(ratio(pm, "min_us"));
    }
    println!("{out}");
    Ok(true)
}

fn run() -> Result<bool> {
    let mut plan = false;
    let mut filter: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => plan = true,
            "--shape" => filter = Some(it.next().ok_or("--shape needs a substring")?),
            _ => {
                return Err(format!(
                    "unknown argument {a}; usage: fa_qres_pm_timing [--plan] [--shape SUBSTR]"
                )
                .into())
            }
        }
    }
    // Arm 0 is hipcc.
    let mut arms = vec![Arm {
        name: "hipcc".into(),
        route: A3bQresRoute::Hipcc,
    }];
    arms.push(Arm {
        name: "pm".into(),
        route: A3bQresRoute::Pm,
    });
    let shapes: Vec<Shape> = SHAPES
        .iter()
        .copied()
        .filter(|s| {
            filter
                .as_ref()
                .map_or(true, |f| s.name.contains(f.as_str()))
        })
        .collect();
    if shapes.is_empty() {
        return Err("--shape matched no shape".into());
    }
    if plan {
        let mut peak = 0usize;
        for s in &shapes {
            peak = peak.max(s.device_bytes());
            println!(
                "{}",
                json!({"timing_shape": s.name, "rows": s.rows, "start": s.start, "context": s.ctx,
                    "range": s.range.map(|(t, o)| json!({"q_total_rows": t, "q_row_offset": o})),
                    "grid": s.grid(), "block": [768, 1, 1], "lds": 49_936,
                    "device_bytes": s.device_bytes(),
                    "launches_per_arm": 1 + WARMUPS + ITERS})
            );
        }
        // Buffers are freed per shape; the pool may retain them, so budget the sum.
        let sum: usize = shapes.iter().map(Shape::device_bytes).sum();
        println!(
            "{}",
            json!({"plan": true, "shapes": shapes.len(), "peak_shape_bytes": peak,
                "sum_bytes": sum,
                "suggested_mem_gb": ((sum as f64 / (1u64 << 30) as f64) + 1.0).ceil() as usize,
                "warmups_per_arm": WARMUPS, "timed_per_arm": ITERS,
                "pm_module": rdna_compute::fa_qresident_pm::MODULE,
                "pm_symbols": [rdna_compute::fa_qresident_pm::SYMBOL_DIRECT,
                    rdna_compute::fa_qresident_pm::SYMBOL_RANGE],
                "gpu_launched": false})
        );
        return Ok(true);
    }
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("requires gfx1201, got {}", gpu.arch).into());
    }
    println!(
        "{}",
        json!({"kind": "provenance", "arch": gpu.arch,
            "pm_module": rdna_compute::fa_qresident_pm::MODULE,
            "pm_image_bytes": rdna_compute::fa_qresident_pm::IMAGE.len()})
    );
    let mut ok = true;
    for s in &shapes {
        ok &= time_shape(&mut gpu, s, &arms)?;
        if !ok {
            break;
        }
    }
    println!("QRES_PM_TIMING shapes={} pass={ok}", shapes.len());
    Ok(ok)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("fa_qres_pm_timing: {e}");
            std::process::exit(1);
        }
    }
}
