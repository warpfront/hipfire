// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G1 producer byte oracle of the canonical GDN ordinary-prefill plan (CHUNK-CANONICAL-DESIGN
//! §1.2/§1.3/§2.2, S1 slice), gfx1201.
//!
//! ## What is compared
//!
//! One reference projection of `REF_ROWS` (8704 = 17 * 512) activation rows runs through the
//! BULK route (no ordinary-prefill intent; every row of a >= 64-row call). Then for every
//! `n` of the G1 list and several start offsets `off` (different positions inside the 64/128/256-row
//! physical producer tiles) the rows `[off, off + n)` are projected again as an INDEPENDENT call of
//! exactly `n` rows under [`Gpu::ordinary_prefill_enter`]. A row's projected bytes (QKV, Z, beta,
//! alpha) must be identical to the reference row. `--scope off` is the fail-before control: without
//! the intent the n < 64 calls take the unprepared F16 route and MUST mismatch.
//!
//! Arms (all over the same random symmetric MQ4G256V2 weights):
//!
//! * `unprep`: F32 RMSNorm+FWHT, then the standalone-quantizer wrapper
//!   `gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2` (IU4 / A8 / FP8 per the process flags).
//! * `prep`: the gfx1201 fused producer `fused_rmsnorm_rotate_mq_i4_gfx12_batched` feeding the
//!   prepared consumers (`gemm_qkvza_mq4g256v2_wmma_iu4_prepared`); IU4 configurations only.
//! * `zba` (27B geometry, IU4 only): the prepared Z|beta|alpha fold route
//!   `gemm_qkvza_mq4g256v2_wmma_iu4_prepared_zba`; additionally compared against `prep` bytes at
//!   the same call.
//!
//! The process flags select the precision configuration: default (IU4), `HIPFIRE_IU4_PREFILL=0`
//! (native FP8), `HIPFIRE_A8_PREFILL=1` (A8).
//!
//! `aligned_sha256` lines hash the bytes of the aligned bulk calls (n % 512 == 0, offset 0, no
//! intent) so a base build and the candidate build can be compared byte for byte.
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab,deltanet --example gdn_canonical_producer_oracle -- --scope on
//! ```
//! Exit 0 = every compared row identical, 1 = mismatch (summary printed), 2 = unsupported setup.

use half::f16;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const REF_ROWS: usize = 8704;
const GROUP_BYTES: usize = 136;
const EPS: f32 = 1.0e-6;
const SENTINEL: i32 = 0xA5;

const N_LIST: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 95, 96, 97, 127, 128, 129, 191,
    192, 223, 224, 255, 256, 257, 287, 288, 383, 384, 415, 416, 511, 512, 513, 537, 8191, 8192,
    8193,
];

#[derive(Clone, Copy)]
struct Geo {
    name: &'static str,
    k: usize,
    qkv: usize,
    z: usize,
    beta: usize,
    alpha: usize,
}

const GEOS: [Geo; 2] = [
    // Ornith / A3B family LA geometry.
    Geo { name: "ornith", k: 2048, qkv: 8192, z: 4096, beta: 32, alpha: 32 },
    // Qwen3.6/3.8 27B dense LA geometry (the fused qkvzagdn dims).
    Geo { name: "q27b", k: 5120, qkv: 10240, z: 6144, beta: 48, alpha: 48 },
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
}

/// Random valid SYMMETRIC MQ4G256V2 rows (`zp == -8 * sc` per half, both exact F16).
fn synth_sym_mq4g256v2(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(m * (k / 256) * GROUP_BYTES);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            let scale = 0.004 + (rng.next() & 0x3f) as f32 * 2.0e-4;
            let sc = f16::from_f32(scale);
            let zp = f16::from_f32(-8.0 * sc.to_f32());
            out.extend_from_slice(&sc.to_bits().to_le_bytes());
            out.extend_from_slice(&zp.to_bits().to_le_bytes());
        }
        for _ in 0..32 {
            out.extend_from_slice(&(rng.next() as u32).to_le_bytes());
        }
    }
    out
}

fn synth_x(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = 0.25 * (1 + r % 5) as f32;
        for _ in 0..k {
            out.push(((rng.next() & 0xff_ffff) as f32 / 8_388_608.0 - 1.0) * scale);
        }
    }
    out
}

fn synth_vec(len: usize, seed: u64, lo: f32, span: f32) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..len)
        .map(|_| lo + (rng.next() & 0xff_ffff) as f32 / 16_777_216.0 * span)
        .collect()
}

struct Res {
    geo: Geo,
    w: Vec<GpuTensor>,
    /// Z fold (`z + 128` rows: Z, then beta at rows z.., alpha at rows z+64..) for the zba arm.
    z_fold: Option<GpuTensor>,
    norm: GpuTensor,
    x: GpuTensor,
    x_rot: GpuTensor,
    ys: Vec<GpuTensor>,
    y_z_fold: Option<GpuTensor>,
}

fn upload(gpu: &mut Gpu, geo: Geo, seed: u64) -> Res {
    let ms = [geo.qkv, geo.z, geo.beta, geo.alpha];
    let host: Vec<Vec<u8>> = ms
        .iter()
        .enumerate()
        .map(|(i, m)| synth_sym_mq4g256v2(*m, geo.k, seed + 0x100 * (i as u64 + 1)))
        .collect();
    let w = host
        .iter()
        .map(|b| gpu.upload_raw(b, &[b.len()]).expect("upload weight"))
        .collect();
    let z_fold = (geo.beta == 48 && geo.alpha == 48 && geo.z == 6144).then(|| {
        let row = geo.k / 256 * GROUP_BYTES;
        let mut f = host[1].clone();
        f.extend_from_slice(&host[2]);
        f.extend(std::iter::repeat(0u8).take((64 - 48) * row));
        f.extend_from_slice(&host[3]);
        f.extend(std::iter::repeat(0u8).take((64 - 48) * row));
        assert_eq!(f.len(), (geo.z + 128) * row);
        gpu.upload_raw(&f, &[f.len()]).expect("upload z fold")
    });
    let norm = gpu
        .upload_f32(&synth_vec(geo.k, seed + 0x200, 0.5, 1.0), &[geo.k])
        .expect("norm");
    let x = gpu
        .upload_f32(&synth_x(REF_ROWS, geo.k, seed + 0x300), &[REF_ROWS, geo.k])
        .expect("x");
    let x_rot = gpu
        .alloc_tensor(&[REF_ROWS, geo.k], DType::F32)
        .expect("x_rot");
    let ys = ms
        .iter()
        .map(|m| gpu.alloc_tensor(&[REF_ROWS, *m], DType::F32).expect("y"))
        .collect();
    let y_z_fold = z_fold.as_ref().map(|_| {
        gpu.alloc_tensor(&[REF_ROWS, geo.z + 128], DType::F32)
            .expect("y_z_fold")
    });
    Res { geo, w, z_fold, norm, x, x_rot, ys, y_z_fold }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Unprep,
    Prep,
    Zba,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Unprep => "unprep",
            Arm::Prep => "prep",
            Arm::Zba => "zba",
        }
    }
}

type R = rdna_compute::HipResult<()>;

/// Project `n` rows starting at row `off` of `r.x`; outputs land in views of `r.ys` row 0..n.
fn run(gpu: &mut Gpu, r: &Res, arm: Arm, off: usize, n: usize) -> R {
    let g = r.geo;
    let x = r.x.sub_offset(off * g.k, n * g.k);
    let xr = r.x_rot.sub_offset(0, n * g.k);
    let y: Vec<GpuTensor> = [g.qkv, g.z, g.beta, g.alpha]
        .iter()
        .zip(&r.ys)
        .map(|(m, t)| t.sub_offset(0, n * m))
        .collect();
    for t in &y {
        gpu.hip.memset(&t.buf, SENTINEL, t.byte_size())?;
    }
    match arm {
        Arm::Unprep => {
            gpu.fused_rmsnorm_rotate_mq_batched(&x, &r.norm, &xr, g.k, EPS, n)?;
            gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2(
                &r.w[0], &r.w[1], &r.w[2], &r.w[3], &xr, &y[0], &y[1], &y[2], &y[3], g.qkv, g.z,
                g.beta, g.alpha, g.k, n,
            )
        }
        Arm::Prep => {
            let res = gpu.reserve_int4_mmq(g.k, n)?;
            let prep =
                gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(&x, &r.norm, None, None, res, g.k, EPS, n)?;
            gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
                &r.w[0], &r.w[1], &r.w[2], &r.w[3], &xr, &prep, &y[0], &y[1], &y[2], &y[3], g.qkv,
                g.z, g.beta, g.alpha, g.k, n,
            )
        }
        Arm::Zba => {
            let fold = r.z_fold.as_ref().expect("zba arm needs the 27B fold");
            let yf = r.y_z_fold.as_ref().unwrap().sub_offset(0, n * (g.z + 128));
            let res = gpu.reserve_int4_mmq(g.k, n)?;
            let prep =
                gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(&x, &r.norm, None, None, res, g.k, EPS, n)?;
            gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared_zba(
                &r.w[0], fold, &r.w[2], &r.w[3], &xr, &prep, &y[0], &yf, &y[1], &y[2], &y[3], g.qkv,
                g.z, g.beta, g.alpha, g.k, n,
            )
        }
    }
}

fn download(gpu: &Gpu, r: &Res, n: usize) -> Vec<Vec<u8>> {
    let g = r.geo;
    [g.qkv, g.z, g.beta, g.alpha]
        .iter()
        .zip(&r.ys)
        .map(|(m, t)| {
            let v = t.sub_offset(0, n * m);
            gpu.download_raw_bytes(&v).expect("download")
        })
        .collect()
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn offsets_for(n: usize) -> Vec<usize> {
    let all: &[usize] = if n <= 600 {
        &[0, 1, 63, 64, 127, 130, 509, 1000, 4097]
    } else if n < 4000 {
        &[0, 65]
    } else {
        &[0]
    };
    all.iter().copied().filter(|o| o + n <= REF_ROWS).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scope_on = !args.windows(2).any(|w| w[0] == "--scope" && w[1] == "off");
    let geos: Vec<Geo> = match args.windows(2).find(|w| w[0] == "--geo").map(|w| w[1].as_str()) {
        Some(name) => GEOS.iter().copied().filter(|g| g.name == name).collect(),
        None => GEOS.to_vec(),
    };
    let quick = args.iter().any(|a| a == "--quick");
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("gdn_canonical_producer_oracle: arch {} is not gfx1201", gpu.arch);
        std::process::exit(2);
    }
    gpu.mq4v2_symmetric = true;
    let config = json!({
        "iu4": gpu.flags.iu4_prefill_enabled(),
        "a8": gpu.flags.a8_prefill_enabled(),
        "fp8_qkvza": gpu.flags.gfx12_mq4v2_fp8_qkvza,
        "fp8_v2": gpu.flags.gfx12_mq4v2_fp8_v2,
        "scope": scope_on,
    });
    let mut total = 0usize;
    let mut bad = 0usize;
    let mut first_bad: Option<Value> = None;
    let mut cross_bad = 0usize;
    for geo in geos {
        let r = upload(&mut gpu, geo, 0x6d7a + geo.k as u64);
        let k = geo.k;
        // Static predicate check: the intent makes every extent take the bulk admission.
        let bulk = (
            gpu.a8_prefill_active(512, k),
            gpu.iu4_producer_quant_fused_active(512, k),
            gpu.iu4_silu_quant_fused_active(512, k),
            gpu.fp8_stream_active(512, k),
            gpu.mq4g256v2_shared_iu4_supported(k, 512),
        );
        // Unscoped n=25 admission (the pre-change behavior the control exercises).
        let unscoped_n25 = (
            gpu.a8_prefill_active(25, k),
            gpu.iu4_producer_quant_fused_active(25, k),
            gpu.iu4_silu_quant_fused_active(25, k),
            gpu.fp8_stream_active(25, k),
            gpu.mq4g256v2_shared_iu4_supported(k, 25),
        );
        {
            let prev = gpu.ordinary_prefill_enter();
            for n in [1usize, 25, 63, 64, 537] {
                let got = (
                    gpu.a8_prefill_active(n, k),
                    gpu.iu4_producer_quant_fused_active(n, k),
                    gpu.iu4_silu_quant_fused_active(n, k),
                    gpu.fp8_stream_active(n, k),
                    gpu.mq4g256v2_shared_iu4_supported(k, n),
                );
                assert_eq!(got, bulk, "{} predicates at n={n} differ from the bulk admission", geo.name);
            }
            gpu.ordinary_prefill_exit(prev);
        }
        println!(
            "{}",
            json!({"predicates": geo.name, "order": "a8,iu4_producer_quant_fused,iu4_silu_quant_fused,fp8_stream,shared_iu4",
                "bulk_n512": format!("{bulk:?}"), "unscoped_n25": format!("{unscoped_n25:?}"),
                "scoped_n1_25_63_64_537_equal_bulk": true})
        );
        #[cfg(feature = "deltanet")]
        {
            use rdna_compute::prefill_invariant::{GdnLayerDesc, GdnProjDesc};
            let d = |m| GdnProjDesc { dtype: DType::MQ4G256V2, m };
            let desc = GdnLayerDesc {
                dim: k,
                key_heads: 16,
                value_heads: geo.beta,
                head_dim: 128,
                qkv: d(geo.qkv),
                z: d(geo.z),
                beta: d(geo.beta),
                alpha: d(geo.alpha),
                out_dtype: DType::MQ4G256V2,
                out_residual: true,
                z_folded: false,
                moe: geo.name == "ornith",
                state: rdna_compute::gdn_stream::GdnStateQuant::Q8Ef,
                scan64_admitted: true,
            };
            let plan = gpu.resolve_gdn_prefill_plan(&desc).expect("plan");
            println!("{}", json!({"plan": geo.name, "recipe_id": format!("{:#x}", plan.recipe_id),
                "algorithm": format!("{:?}", plan.algorithm), "proj": format!("{:?}", plan.projection_recipe),
                "scan_out": format!("{:?}", plan.scan_output_format), "tile": plan.producer_tile_rows}));
        }
        let arms: Vec<Arm> = {
            let mut a = vec![Arm::Unprep];
            if gpu.flags.iu4_prefill_enabled() {
                a.push(Arm::Prep);
                if r.z_fold.is_some() {
                    a.push(Arm::Zba);
                }
            }
            a
        };
        for arm in arms {
            // Reference: the bulk route, no intent.
            run(&mut gpu, &r, arm, 0, REF_ROWS).expect("reference run");
            gpu.hip.device_synchronize().unwrap();
            let refo = download(&gpu, &r, REF_ROWS);
            let ms = [geo.qkv, geo.z, geo.beta, geo.alpha];
            let mut arm_total = 0usize;
            let mut arm_bad = 0usize;
            let mut arm_first: Option<Value> = None;
            for &n in N_LIST {
                if quick && !(n <= 70 || n == 128 || n == 512 || n == 537 || n == 8192) {
                    continue;
                }
                for off in offsets_for(n) {
                    let prev = if scope_on { Some(gpu.ordinary_prefill_enter()) } else { None };
                    let res = run(&mut gpu, &r, arm, off, n);
                    if let Some(p) = prev {
                        gpu.ordinary_prefill_exit(p);
                    }
                    res.unwrap_or_else(|e| panic!("{} {} n={n} off={off}: {e:?}", geo.name, arm.name()));
                    gpu.hip.device_synchronize().unwrap();
                    let got = download(&gpu, &r, n);
                    let mut ok = true;
                    for (f, m) in ms.iter().enumerate() {
                        let want = &refo[f][off * m * 4..(off + n) * m * 4];
                        if let Some(b) = first_diff(&got[f], want) {
                            ok = false;
                            if arm_first.is_none() {
                                let field = ["qkv", "z", "beta", "alpha"][f];
                                arm_first = Some(json!({"n": n, "off": off, "field": field,
                                    "row": b / 4 / m, "col": b / 4 % m}));
                            }
                        }
                    }
                    arm_total += 1;
                    if !ok {
                        arm_bad += 1;
                    }
                }
            }
            // The raw-tail materializer (carry rows of an incomplete scan tile) must produce the
            // reference rows' QKV/beta/alpha bytes as an independent `rows`-row call.
            if arm == Arm::Prep && scope_on {
                let (mut raw_calls, mut raw_bad) = (0usize, 0usize);
                for &rows in &[1usize, 2, 3, 15, 16, 17, 25, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129] {
                    for &off in &[0usize, 5, 64, 1000, 8191 - rows] {
                        let x = r.x.sub_offset(off * geo.k, rows * geo.k);
                        let y: Vec<GpuTensor> = [geo.qkv, geo.beta, geo.alpha]
                            .iter()
                            .zip([&r.ys[0], &r.ys[2], &r.ys[3]])
                            .map(|(m, t)| t.sub_offset(0, rows * m))
                            .collect();
                        for t in &y {
                            gpu.hip.memset(&t.buf, SENTINEL, t.byte_size()).unwrap();
                        }
                        let prev = gpu.ordinary_prefill_enter();
                        let did = gpu
                            .gdn_raw_rows_g12_iu4(
                                &x, &r.norm, None, EPS, &r.w[0], &r.w[2], &r.w[3], &y[0], &y[1], &y[2],
                                [geo.qkv, geo.beta, geo.alpha], geo.k, rows,
                            )
                            .unwrap();
                        gpu.ordinary_prefill_exit(prev);
                        assert!(did, "raw-rows materializer declined at rows={rows}");
                        gpu.hip.device_synchronize().unwrap();
                        let mut ok = true;
                        for (t, (f, m)) in y.iter().zip([(0usize, geo.qkv), (2, geo.beta), (3, geo.alpha)]) {
                            let got = gpu.download_raw_bytes(t).unwrap();
                            ok &= got == refo[f][off * m * 4..(off + rows) * m * 4];
                        }
                        raw_calls += 1;
                        raw_bad += usize::from(!ok);
                    }
                }
                println!(
                    "{}",
                    json!({"raw_rows_materializer": geo.name, "calls": raw_calls, "mismatching_calls": raw_bad})
                );
                total += raw_calls;
                bad += raw_bad;
            }
            total += arm_total;
            bad += arm_bad;
            if first_bad.is_none() {
                if let Some(f) = &arm_first {
                    first_bad = Some(json!({"geo": geo.name, "arm": arm.name(), "first": f}));
                }
            }
            // Aligned bulk hash (no intent): bytes of the n % 512 == 0 offset-0 calls.
            let mut aligned = Sha256::new();
            for n in [512usize, 1024, 2048, 4096, 8192] {
                let res = run(&mut gpu, &r, arm, 0, n);
                res.expect("aligned run");
                gpu.hip.device_synchronize().unwrap();
                for g in download(&gpu, &r, n) {
                    aligned.update(&g);
                }
            }
            println!(
                "{}",
                json!({"geo": geo.name, "arm": arm.name(), "config": config, "calls": arm_total,
                    "mismatching_calls": arm_bad, "first_mismatch": arm_first,
                    "aligned_sha256": format!("{:x}", aligned.finalize())})
            );
        }
        // zba vs prep cross check at a few extents, same intent.
        if r.z_fold.is_some() && gpu.flags.iu4_prefill_enabled() {
            for &n in &[1usize, 25, 63, 64, 127, 128, 129, 537] {
                let prev = gpu.ordinary_prefill_enter();
                run(&mut gpu, &r, Arm::Prep, 0, n).unwrap();
                gpu.hip.device_synchronize().unwrap();
                let a = download(&gpu, &r, n);
                run(&mut gpu, &r, Arm::Zba, 0, n).unwrap();
                gpu.hip.device_synchronize().unwrap();
                let b = download(&gpu, &r, n);
                gpu.ordinary_prefill_exit(prev);
                if a != b {
                    cross_bad += 1;
                    eprintln!("zba != prep at n={n}");
                }
            }
        }
        // The fused qkvzagdn producer (+ fixup) versus the byte-identical materializing
        // counterpart (separate prepared SETs + plain `gdn_chunk_prep`) on the 27B dims: every
        // q/k/v/g/beta/Z byte and the conv ring, at every extent under the intent. The raw QKV
        // rows the fused kernel stores must equal the SET's rows (that is what makes the SET
        // output the exact raw carry of an incomplete tile).
        #[cfg(feature = "deltanet")]
        if geo.name == "q27b" && r.z_fold.is_some() && gpu.flags.iu4_prefill_enabled() {
            let nl: Vec<usize> = N_LIST
                .iter()
                .copied()
                .filter(|n| !quick || *n <= 70 || *n == 128 || *n == 512 || *n == 537 || *n == 8192)
                .collect();
            let (mut fused_calls, mut fused_bad, mut declined) = (0usize, 0usize, 0usize);
            let mut first_fused: Option<Value> = None;
            for n in nl {
                match fused_vs_plain(&mut gpu, &r, n) {
                    Ok(None) => declined += 1,
                    Ok(Some(diff)) => {
                        fused_calls += 1;
                        if let Some(d) = diff {
                            fused_bad += 1;
                            if first_fused.is_none() {
                                first_fused = Some(json!({"n": n, "diff": d}));
                            }
                        }
                    }
                    Err(e) => panic!("fused_vs_plain n={n}: {e:?}"),
                }
            }
            println!(
                "{}",
                json!({"fused_vs_plain": geo.name, "calls": fused_calls, "declined": declined,
                    "mismatching_calls": fused_bad, "first_mismatch": first_fused})
            );
            total += fused_calls;
            bad += fused_bad;
        }
    }
    println!(
        "{}",
        json!({"summary": true, "oracle": "gdn_canonical_producer", "config": config,
            "compared_calls": total, "mismatching_calls": bad, "zba_vs_prep_mismatches": cross_bad,
            "first_mismatch": first_bad, "all_identical": bad == 0 && cross_bad == 0})
    );
    if bad != 0 || cross_bad != 0 {
        std::process::exit(1);
    }
}

/// One extent of the fused-versus-plain GDN preamble comparison on the 27B dims. `Ok(None)`: the
/// fused launcher declined (not admitted). `Ok(Some(None))`: identical; `Ok(Some(Some(what)))`:
/// the first differing buffer.
#[cfg(feature = "deltanet")]
fn fused_vs_plain(gpu: &mut Gpu, r: &Res, n: usize) -> rdna_compute::HipResult<Option<Option<String>>> {
    use rdna_compute::F2GdnTargets;
    let g = r.geo;
    let (hk, hv, hd) = (16usize, 48usize, 128usize);
    let qkv_w = 10240usize;
    let conv_w = gpu.upload_f32(&synth_vec(qkv_w * 4, 0xc0de1, -0.5, 1.0), &[qkv_w * 4])?;
    let conv_init = synth_vec(qkv_w * 3, 0xc0de2, -0.5, 1.0);
    let dt_bias = gpu.upload_f32(&synth_vec(hv, 0xc0de3, -0.2, 0.4), &[hv])?;
    let a_log = gpu.upload_f32(&synth_vec(hv, 0xc0de4, -1.0, 1.0), &[hv])?;
    let q_scale = 1.0 / (hd as f32).sqrt();
    let x = r.x.sub_offset(0, n * g.k);
    let xr = r.x_rot.sub_offset(0, n * g.k);
    let fold = r.z_fold.as_ref().unwrap();
    let mk = |gpu: &mut Gpu| -> rdna_compute::HipResult<Vec<GpuTensor>> {
        Ok(vec![
            gpu.alloc_tensor(&[n, g.qkv], DType::F32)?,
            gpu.alloc_tensor(&[n, g.z], DType::F32)?,
            gpu.alloc_tensor(&[n, hv], DType::F32)?,
            gpu.alloc_tensor(&[n, hv], DType::F32)?,
            gpu.alloc_tensor(&[n, hk * hd], DType::F16)?,
            gpu.alloc_tensor(&[n, hk * hd], DType::F16)?,
            gpu.alloc_tensor(&[n, hv * hd], DType::F16)?,
            gpu.upload_f32(&conv_init, &[qkv_w * 3])?,
        ])
    };
    let prev = gpu.ordinary_prefill_enter();
    let out = (|| -> rdna_compute::HipResult<Option<Option<String>>> {
        // Plain: separate prepared SETs then the plain prep.
        let p = mk(gpu)?;
        for t in &p[..4] {
            gpu.hip.memset(&t.buf, SENTINEL, t.byte_size())?;
        }
        let res = gpu.reserve_int4_mmq(g.k, n)?;
        let prep = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(&x, &r.norm, None, None, res, g.k, EPS, n)?;
        gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
            &r.w[0], &r.w[1], &r.w[2], &r.w[3], &xr, &prep, &p[0], &p[1], &p[2], &p[3], g.qkv, g.z,
            g.beta, g.alpha, g.k, n,
        )?;
        let raw_plain = gpu.download_raw_bytes(&p[0])?;
        gpu.gdn_chunk_prep(
            &p[0], &conv_w, &p[7], &p[3], &p[2], &dt_bias, &a_log, &p[4], &p[5], &p[6], n, q_scale,
            1.0e-6, hk, hv,
        )?;
        // Fused: one launch + fixup.
        let f = mk(gpu)?;
        for t in &f[..4] {
            gpu.hip.memset(&t.buf, SENTINEL, t.byte_size())?;
        }
        let res = gpu.reserve_int4_mmq(g.k, n)?;
        let prep = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(&x, &r.norm, None, None, res, g.k, EPS, n)?;
        let targets = F2GdnTargets {
            conv_weight: &conv_w,
            conv_state: &f[7],
            q: &f[4],
            k: &f[5],
            v: &f[6],
            q_scale,
            eps: 1.0e-6,
        };
        let admitted = gpu.gemm_qkvza_mq4g256v2_iu4_gdn_prepared(
            &r.w[0], fold, &prep, &f[0], &f[1], &f[2], &f[3], [g.qkv, g.z, g.beta, g.alpha], g.k, n,
            &targets,
        )?;
        if !admitted {
            for t in p.into_iter().chain(f) {
                gpu.free_tensor(t)?;
            }
            return Ok(None);
        }
        let raw_fused = gpu.download_raw_bytes(&f[0])?;
        gpu.gdn_chunk_prep_fixup(
            &f[0], &conv_w, &f[7], &f[3], &f[2], &dt_bias, &a_log, &f[4], &f[5], &f[6], n, q_scale,
            1.0e-6, hk, hv,
        )?;
        gpu.hip.device_synchronize()?;
        let names = ["qkv_raw", "z", "beta", "g", "q", "k", "v", "conv_state"];
        let mut verdict: Option<String> = None;
        for i in 1..8 {
            let a = gpu.download_raw_bytes(&p[i])?;
            let b = gpu.download_raw_bytes(&f[i])?;
            if a != b && verdict.is_none() {
                verdict = Some(format!("{} differs at byte {:?}", names[i], first_diff(&a, &b)));
            }
        }
        // Raw rows the fused kernel stored must be the SET rows: compare only rows it wrote
        // (everything not sentinel-filled).
        let row = qkv_w * 4;
        for t in 0..n {
            let fr = &raw_fused[t * row..(t + 1) * row];
            let pr = &raw_plain[t * row..(t + 1) * row];
            if fr.iter().any(|b| *b != SENTINEL as u8) && fr != pr && verdict.is_none() {
                verdict = Some(format!("raw qkv row {t} stored by the fused kernel differs from the SET"));
            }
        }
        for t in p.into_iter().chain(f) {
            gpu.free_tensor(t)?;
        }
        Ok(Some(verdict))
    })();
    gpu.ordinary_prefill_exit(prev);
    out
}
