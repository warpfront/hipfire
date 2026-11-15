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
//! ## Fused `qkvzagdn` raw-export contract (PLAN-prefill-tail §3.1, §9 G1)
//!
//! `fused_vs_plain` runs, on the 27B dims, the fused A4 producer
//! (`gemm_qkvza_mq4g256v2_iu4_gdn_prepared`, `_b1` token / `_b1s` slab) against the separate
//! prepared SETs + plain `gdn_chunk_prep`, for every extent of the §9 G1 list (`n = 1..65`,
//! `127..129`, `134`, `192`, `511..513`, `537`, `5909`, `8191..8193`, and `64+r, 128+r, 512+r, 576+r,
//! 1024+r` for every `r = 1..63`) at start rows `off in {0, 64}` of the activation plane. The set of raw
//! QKV rows the producer MUST define is computed from the frozen formula
//! `export_raw(t) = valid(t) && (t128 < 3 || t128 >= 125 || t + 3 >= (n & !63))`
//! (call-local token `t`, `t128 = t % 128`), never by sentinel detection: every required row's
//! F32 bytes must equal the independent SET row; every other raw row is poisoned with NaN bytes
//! (`0xFF`) and must not be read. All raw alpha/beta rows are compared before any prep. The
//! consumer chain then mirrors the stream executor (`fixup(T = rc)` on the complete prefix, raw
//! `gdn_chunk_prep` on the tail `[rc, n)`; `fixup(T = n)` when `rc == 0`) and must equal the whole
//! plain prep on Z, sigmoid beta, gates, q/k/v (F16 operands) and the conv ring. The first mismatching
//! plane/row is reported; a declined launch or a wrong activation layout is a failure, not a skip.
//!
//! The activation layout is process-global (`HIPFIRE_A4_SLAB`, default slab): `--layout slab|token`
//! asserts it (exit 2 otherwise), `--layout both` re-executes itself once per layout (the
//! token child with `HIPFIRE_A4_SLAB=0`; `--json PATH` becomes `PATH.slab.json` / `PATH.token.json`).
//! `--fused-only` skips every other arm and site (exit 2 when no fused coverage ran);
//! `--json PATH` writes the receipt (architecture, both bundle hashes, geometry, seeds, per-geometry
//! fused coverage and the first mismatches).
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab,deltanet --example gdn_canonical_producer_oracle -- --scope on
//! gdn_canonical_producer_oracle --geo q27b --fused-only --layout both --json OUT/producer.json
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
    Geo {
        name: "ornith",
        k: 2048,
        qkv: 8192,
        z: 4096,
        beta: 32,
        alpha: 32,
    },
    // Qwen3.6/3.8 27B dense LA geometry (the fused qkvzagdn dims).
    Geo {
        name: "q27b",
        k: 5120,
        qkv: 10240,
        z: 6144,
        beta: 48,
        alpha: 48,
    },
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
    /// The production layer-input AWQ scale with reciprocal planes (`[K, AWQ_RCP_ROWS]`): the slab
    /// producer twin (`_v2_slab_fdiv`) is selected only with an AWQ tensor, so the `_b1s` fused
    /// launch is reachable only through it. `None` in the token layout and for non-27B geometries.
    awq: Option<GpuTensor>,
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
    let awq = (gpu.a4_slab_active() && geo.name == "q27b").then(|| {
        let a: Vec<f32> = synth_vec(geo.k, seed + 0x400, 0.9, 0.2);
        let bytes: Vec<u8> = rdna_compute::gemv::awq_rcp_planes(&a)
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        gpu.upload_raw(&bytes, &[geo.k, rdna_compute::gemv::AWQ_RCP_ROWS])
            .expect("upload awq rcp planes")
    });
    Res {
        geo,
        w,
        z_fold,
        norm,
        x,
        x_rot,
        ys,
        y_z_fold,
        awq,
    }
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
            if gpu.arch == "gfx1201" {
                gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2(
                    &r.w[0], &r.w[1], &r.w[2], &r.w[3], &xr, &y[0], &y[1], &y[2], &y[3], g.qkv,
                    g.z, g.beta, g.alpha, g.k, n,
                )
            } else {
                gpu.gemm_qkvza_mq4g256v2_wmma(
                    &r.w[0], &r.w[1], &r.w[2], &r.w[3], &xr, &y[0], &y[1], &y[2], &y[3], g.qkv,
                    g.z, g.beta, g.alpha, g.k, n,
                )
            }
        }
        Arm::Prep => {
            let res = gpu.reserve_int4_mmq(g.k, n)?;
            let prep = if gpu.arch == "gfx1201" {
                gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
                    &x, &r.norm, None, None, res, g.k, EPS, n,
                )?
            } else {
                gpu.fused_rmsnorm_rotate_mq_i4_batched(
                    &x,
                    &r.norm,
                    None,
                    Some(&xr),
                    res,
                    g.k,
                    EPS,
                    n,
                )?
            };
            gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
                &r.w[0], &r.w[1], &r.w[2], &r.w[3], &xr, &prep, &y[0], &y[1], &y[2], &y[3], g.qkv,
                g.z, g.beta, g.alpha, g.k, n,
            )
        }
        Arm::Zba => {
            let fold = r.z_fold.as_ref().expect("zba arm needs the 27B fold");
            let yf = r.y_z_fold.as_ref().unwrap().sub_offset(0, n * (g.z + 128));
            let res = gpu.reserve_int4_mmq(g.k, n)?;
            let prep = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
                &x, &r.norm, None, None, res, g.k, EPS, n,
            )?;
            gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared_zba(
                &r.w[0], fold, &r.w[2], &r.w[3], &xr, &prep, &y[0], &yf, &y[1], &y[2], &y[3],
                g.qkv, g.z, g.beta, g.alpha, g.k, n,
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
    if let Some(i) = args
        .windows(2)
        .position(|w| w[0] == "--layout" && w[1] == "both")
    {
        // The activation layout is fixed per process (`HIPFIRE_A4_SLAB`, parsed once): one child
        // per layout, same arguments.
        let exe = std::env::current_exe().expect("current_exe");
        let mut worst = 0i32;
        for (layout, env_slab) in [("slab", "1"), ("token", "0")] {
            let mut child: Vec<String> = args.clone();
            child[i + 1] = layout.to_string();
            if let Some(j) = child.iter().position(|a| a == "--json") {
                if let Some(p) = child.get_mut(j + 1) {
                    *p = format!("{p}.{layout}.json");
                }
            }
            eprintln!(
                "gdn_canonical_producer_oracle: --layout both -> {layout} (HIPFIRE_A4_SLAB={env_slab})"
            );
            let st = std::process::Command::new(&exe)
                .args(&child)
                .env("HIPFIRE_A4_SLAB", env_slab)
                .status()
                .expect("spawn layout child");
            if !st.success() {
                worst = worst.max(st.code().unwrap_or(1));
            }
        }
        std::process::exit(worst);
    }
    let scope_on = !args.windows(2).any(|w| w[0] == "--scope" && w[1] == "off");
    let geos: Vec<Geo> = match args
        .windows(2)
        .find(|w| w[0] == "--geo")
        .map(|w| w[1].as_str())
    {
        Some(name) => GEOS.iter().copied().filter(|g| g.name == name).collect(),
        None => GEOS.to_vec(),
    };
    let quick = args.iter().any(|a| a == "--quick");
    let fused_only = args.iter().any(|a| a == "--fused-only");
    let json_path: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--json")
        .map(|w| w[1].clone());
    let want_layout: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--layout")
        .map(|w| w[1].clone());
    if let Some(l) = want_layout.as_deref() {
        if !matches!(l, "slab" | "token") {
            eprintln!(
                "gdn_canonical_producer_oracle: --layout must be slab, token or both (got {l})"
            );
            std::process::exit(2);
        }
    }
    #[allow(unused_mut, unused_variables)]
    let mut fused_ran = false;
    #[allow(unused_mut, unused_variables)]
    let mut fused_receipts: Vec<Value> = Vec::new();
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" && !gpu.arch.starts_with("gfx11") {
        eprintln!(
            "gdn_canonical_producer_oracle: unsupported arch {}",
            gpu.arch
        );
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
                assert_eq!(
                    got, bulk,
                    "{} predicates at n={n} differ from the bulk admission",
                    geo.name
                );
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
            let d = |m| GdnProjDesc {
                dtype: DType::MQ4G256V2,
                m,
            };
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
            println!(
                "{}",
                json!({"plan": geo.name, "recipe_id": format!("{:#x}", plan.recipe_id),
                "algorithm": format!("{:?}", plan.algorithm), "proj": format!("{:?}", plan.projection_recipe),
                "scan_out": format!("{:?}", plan.scan_output_format), "tile": plan.producer_tile_rows})
            );
        }
        let arms: Vec<Arm> = if fused_only {
            Vec::new()
        } else {
            let mut a = vec![Arm::Unprep];
            if gpu.flags.iu4_prefill_enabled() {
                a.push(Arm::Prep);
                if r.z_fold.is_some() && gpu.arch == "gfx1201" {
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
                    let prev = if scope_on {
                        Some(gpu.ordinary_prefill_enter())
                    } else {
                        None
                    };
                    let res = run(&mut gpu, &r, arm, off, n);
                    if let Some(p) = prev {
                        gpu.ordinary_prefill_exit(p);
                    }
                    res.unwrap_or_else(|e| {
                        panic!("{} {} n={n} off={off}: {e:?}", geo.name, arm.name())
                    });
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
            if arm == Arm::Prep && scope_on && gpu.iu4_producer_quant_fused_active(512, geo.k) {
                let (mut raw_calls, mut raw_bad) = (0usize, 0usize);
                for &rows in &[
                    1usize, 2, 3, 15, 16, 17, 25, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129,
                ] {
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
                                &x,
                                &r.norm,
                                None,
                                EPS,
                                &r.w[0],
                                &r.w[2],
                                &r.w[3],
                                &y[0],
                                &y[1],
                                &y[2],
                                [geo.qkv, geo.beta, geo.alpha],
                                geo.k,
                                rows,
                            )
                            .unwrap();
                        gpu.ordinary_prefill_exit(prev);
                        assert!(did, "raw-rows materializer declined at rows={rows}");
                        gpu.hip.device_synchronize().unwrap();
                        let mut ok = true;
                        for (t, (f, m)) in
                            y.iter()
                                .zip([(0usize, geo.qkv), (2, geo.beta), (3, geo.alpha)])
                        {
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
        if !fused_only
            && gpu.arch == "gfx1201"
            && r.z_fold.is_some()
            && gpu.flags.iu4_prefill_enabled()
        {
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
        // The fused qkvzagdn producer (+ completion passes) versus the byte-identical
        // materializing counterpart (separate prepared SETs + plain `gdn_chunk_prep`) on the 27B
        // dims, at every extent of the G1 list and start rows 0 and 64 (see module docs): the QKV
        // rows the producer must define are computed from the frozen §3.1 formula, every required
        // row must equal the SET row, all raw alpha/beta rows must match, and the prefix-fixup +
        // raw-tail-prep chain must equal the whole plain prep on every plane.
        #[cfg(feature = "deltanet")]
        if geo.name == "q27b"
            && gpu.arch == "gfx1201"
            && r.z_fold.is_some()
            && gpu.flags.iu4_prefill_enabled()
        {
            let slab_now = gpu.a4_slab_active();
            let layout_now = if slab_now { "slab" } else { "token" };
            if let Some(want) = want_layout.as_deref() {
                if want != layout_now {
                    eprintln!(
                        "gdn_canonical_producer_oracle: --layout {want} requested but this process runs the {layout_now} layout (HIPFIRE_A4_SLAB, G12_IU4_B1_CONTROL)"
                    );
                    std::process::exit(2);
                }
            }
            let nl = fused_n_list(quick);
            let (mut expected, mut fused_calls, mut fused_bad, mut declined, mut layout_bad) =
                (0usize, 0usize, 0usize, 0usize, 0usize);
            let mut required_rows_total = 0usize;
            let mut first_fused: Option<Value> = None;
            let mut first_declined: Option<Value> = None;
            let mut failures: Vec<Value> = Vec::new();
            let mut failing_extents: Vec<Value> = Vec::new();
            let mut extents_run = std::collections::BTreeSet::new();
            for &n in &nl {
                for &off in &[0usize, 64] {
                    if off + n > REF_ROWS {
                        continue;
                    }
                    expected += 1;
                    let c = fused_vs_plain(&mut gpu, &r, n, off)
                        .unwrap_or_else(|e| panic!("fused_vs_plain n={n} off={off}: {e:?}"));
                    if !c.admitted {
                        declined += 1;
                        if first_declined.is_none() {
                            first_declined = Some(json!({"n": n, "off": off}));
                        }
                        continue;
                    }
                    fused_calls += 1;
                    extents_run.insert(n);
                    required_rows_total += c.required_rows;
                    if c.slab != slab_now {
                        layout_bad += 1;
                    }
                    if let Some(d) = c.mismatch {
                        fused_bad += 1;
                        let f = json!({"n": n, "off": off, "layout": layout_now, "diff": d});
                        if first_fused.is_none() {
                            first_fused = Some(f.clone());
                        }
                        failing_extents.push(json!([n, off]));
                        if failures.len() < 64 {
                            failures.push(f);
                        }
                    }
                }
            }
            // Every r = 1..63 of every base must have run when the full list is requested.
            let all63 = !quick
                && [64usize, 128, 512, 576, 1024]
                    .iter()
                    .all(|b| (1..=63).all(|r| extents_run.contains(&(b + r))));
            let coverage_ok = declined == 0 && fused_calls == expected && (quick || all63);
            let rec = json!({"fused_vs_plain": geo.name, "layout": layout_now,
                "expected_calls": expected, "calls": fused_calls, "declined": declined,
                "layout_mismatch_calls": layout_bad, "required_qkv_rows_checked": required_rows_total,
                "mismatching_calls": fused_bad, "first_mismatch": first_fused,
                "first_declined": first_declined, "all_63_remainders_of_64_128_512_576_1024": all63,
                "offsets": [0, 64], "quick": quick, "coverage_ok": coverage_ok, "failures": failures,
                "failing_extents_n_off": failing_extents});
            println!("{rec}");
            fused_receipts.push(rec);
            fused_ran = true;
            total += fused_calls;
            bad += fused_bad + declined + layout_bad + usize::from(!coverage_ok);
        }
    }
    // The other ordinary-prefill consumers the same gates reached: full-attention QKV, FFN
    // gate/up, residual (ADD into a live residual) and the shared-expert down.
    if !fused_only && !args.iter().any(|a| a == "--no-sites") {
        let (c, b, first) = other_sites(&mut gpu, scope_on, quick);
        total += c;
        bad += b;
        if first_bad.is_none() {
            first_bad = first;
        }
    }
    let summary = json!({"summary": true, "oracle": "gdn_canonical_producer", "config": config,
        "compared_calls": total, "mismatching_calls": bad, "zba_vs_prep_mismatches": cross_bad,
        "first_mismatch": first_bad, "all_identical": bad == 0 && cross_bad == 0});
    println!("{summary}");
    if let Some(p) = &json_path {
        let receipt = json!({
            "oracle": "gdn_canonical_producer_oracle",
            "arch": gpu.arch,
            "fused_only": fused_only,
            "layout_requested": want_layout,
            "layout_actual": if gpu.a4_slab_active() { "slab" } else { "token" },
            "geometry": {"hk": 16, "hv": 48, "head_dim": 128, "qkv": 10240, "z": 6144, "beta": 48, "alpha": 48,
                "k": 5120, "quant": "MQ4G256V2 symmetric", "activations": "IU4 A4 (block_i4_128)",
                "raw_format": "F32 [n][10240]", "operand_format": "F16 q/k/v"},
            "seeds": {"weights_x": format!("{:#x}", 0x6d7au64 + 5120), "conv": ["0xc0de1", "0xc0de2", "0xc0de3", "0xc0de4"]},
            "bundles": bundle_info(),
            "export_raw": "valid(t) && (t%128 < 3 || t%128 >= 125 || t + 3 >= (n & !63))",
            "fused": fused_receipts,
            "summary": summary,
        });
        std::fs::write(p, serde_json::to_string_pretty(&receipt).unwrap())
            .unwrap_or_else(|e| panic!("write {p}: {e}"));
    }
    if fused_only && !fused_ran {
        eprintln!(
            "gdn_canonical_producer_oracle: --fused-only but no fused coverage ran (needs gfx1201, IU4, geo q27b, fold admitted)"
        );
        std::process::exit(2);
    }
    if bad != 0 || cross_bad != 0 {
        std::process::exit(1);
    }
}

/// Embedded (or developer-overridden) `_b1` / `_b1s` bundle identity: path/source, length, SHA-256.
fn bundle_info() -> Value {
    let one = |var: &str, embedded: &[u8]| -> Value {
        let (source, path, bytes): (&str, Option<String>, Vec<u8>) =
            match hipfire_config::developer_var(var) {
                Ok(p) => (
                    "override",
                    Some(p.clone()),
                    std::fs::read(&p).unwrap_or_default(),
                ),
                Err(_) => ("embedded", None, embedded.to_vec()),
            };
        json!({"env": var, "source": source, "path": path, "len": bytes.len(),
            "sha256": format!("{:x}", Sha256::digest(&bytes))})
    };
    json!({
        "b1_token": one("HIPFIRE_G12_IU4_B1_BUNDLE", rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1),
        "b1s_slab": one("HIPFIRE_G12_IU4_B1S_BUNDLE", rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S),
        "control_b1_in_use": hipfire_config::developer_bool("HIPFIRE_G12_IU4_B1_CONTROL", false),
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Site {
    FaQkv,
    GateUp,
    ResidualAdd,
    SharedDown,
}

/// `(name, k, output rows)` per site (A3B/27B-like shapes).
fn site_shape(s: Site) -> (&'static str, usize, Vec<usize>) {
    match s {
        Site::FaQkv => ("fa_qkv", 2048, vec![8192, 512, 512]),
        Site::GateUp => ("gate_up", 2048, vec![4096, 4096]),
        Site::ResidualAdd => ("residual_add", 4096, vec![2048]),
        Site::SharedDown => ("shared_down", 512, vec![2048]),
    }
}

fn run_site(
    gpu: &mut Gpu,
    s: Site,
    w: &[GpuTensor],
    x: &GpuTensor,
    ys: &[GpuTensor],
    y0: &GpuTensor,
    off: usize,
    n: usize,
) -> R {
    let (_, k, ms) = site_shape(s);
    let xv = x.sub_offset(off * k, n * k);
    let y: Vec<GpuTensor> = ms
        .iter()
        .zip(ys)
        .map(|(m, t)| t.sub_offset(0, n * m))
        .collect();
    match s {
        Site::ResidualAdd => {
            let r0 = y0.sub_offset(off * ms[0], n * ms[0]);
            gpu.hip.memcpy_dtod(&y[0].buf, &r0.buf, r0.byte_size())?;
        }
        _ => {
            for t in &y {
                gpu.hip.memset(&t.buf, SENTINEL, t.byte_size())?;
            }
        }
    }
    let g11 = gpu.arch.starts_with("gfx11");
    match (s, g11) {
        (Site::FaQkv, false) => gpu.gemm_qkv_hfq4g256_wmma_gfx12_mq4v2(
            &w[0], &w[1], &w[2], &xv, &y[0], &y[1], &y[2], ms[0], ms[1], ms[2], k, n,
        ),
        (Site::FaQkv, true) => gpu.gemm_qkv_mq4g256v2_wmma(
            &w[0], &w[1], &w[2], &xv, &y[0], &y[1], &y[2], ms[0], ms[1], ms[2], k, n,
        ),
        (Site::GateUp, false) => gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2(
            &w[0], &w[1], &xv, &y[0], &y[1], ms[0], ms[1], k, n,
        ),
        (Site::GateUp, true) => {
            gpu.gemm_gate_up_mq4g256v2_wmma(&w[0], &w[1], &xv, &y[0], &y[1], ms[0], ms[1], k, n)
        }
        (Site::ResidualAdd, false) => {
            gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2(&w[0], &xv, &y[0], ms[0], k, n)
        }
        (Site::ResidualAdd, true) => {
            gpu.gemm_mq4g256v2_residual_wmma(&w[0], &xv, &y[0], ms[0], k, n)
        }
        (Site::SharedDown, _) => gpu.gemm_mq4g256v2(&w[0], &xv, &y[0], ms[0], k, n),
    }
}

fn other_sites(gpu: &mut Gpu, scope_on: bool, quick: bool) -> (usize, usize, Option<Value>) {
    let (mut total, mut bad) = (0usize, 0usize);
    let mut first: Option<Value> = None;
    for s in [
        Site::FaQkv,
        Site::GateUp,
        Site::ResidualAdd,
        Site::SharedDown,
    ] {
        let (name, k, ms) = site_shape(s);
        let w: Vec<GpuTensor> = ms
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let b = synth_sym_mq4g256v2(*m, k, 0x51de + 0x100 * i as u64 + k as u64);
                gpu.upload_raw(&b, &[b.len()]).unwrap()
            })
            .collect();
        let x = gpu
            .upload_f32(&synth_x(REF_ROWS, k, 0x7a11 + k as u64), &[REF_ROWS, k])
            .unwrap();
        let y0 = gpu
            .upload_f32(&synth_x(REF_ROWS, ms[0], 0x7a12), &[REF_ROWS, ms[0]])
            .unwrap();
        let ys: Vec<GpuTensor> = ms
            .iter()
            .map(|m| gpu.alloc_tensor(&[REF_ROWS, *m], DType::F32).unwrap())
            .collect();
        let dl = |gpu: &Gpu, n: usize| -> Vec<Vec<u8>> {
            ms.iter()
                .zip(&ys)
                .map(|(m, t)| gpu.download_raw_bytes(&t.sub_offset(0, n * m)).unwrap())
                .collect()
        };
        run_site(gpu, s, &w, &x, &ys, &y0, 0, REF_ROWS).expect("site reference");
        gpu.hip.device_synchronize().unwrap();
        let refo = dl(gpu, REF_ROWS);
        let (mut calls, mut mism) = (0usize, 0usize);
        let mut site_first: Option<Value> = None;
        let mut failing_n: Vec<usize> = Vec::new();
        for &n in N_LIST {
            if quick && !(n <= 70 || n == 128 || n == 512 || n == 537 || n == 8192) {
                continue;
            }
            for off in offsets_for(n) {
                let prev = scope_on.then(|| gpu.ordinary_prefill_enter());
                let res = run_site(gpu, s, &w, &x, &ys, &y0, off, n);
                if let Some(p) = prev {
                    gpu.ordinary_prefill_exit(p);
                }
                res.unwrap_or_else(|e| panic!("{name} n={n} off={off}: {e:?}"));
                gpu.hip.device_synchronize().unwrap();
                let got = dl(gpu, n);
                let mut ok = true;
                for (f, m) in ms.iter().enumerate() {
                    if let Some(b) = first_diff(&got[f], &refo[f][off * m * 4..(off + n) * m * 4]) {
                        ok = false;
                        if site_first.is_none() {
                            site_first = Some(json!({"site": name, "n": n, "off": off, "field": f,
                                "row": b / 4 / m, "col": b / 4 % m}));
                        }
                    }
                }
                calls += 1;
                mism += usize::from(!ok);
                if !ok && off == 0 {
                    failing_n.push(n);
                }
            }
        }
        println!(
            "{}",
            json!({"site": name, "calls": calls, "mismatching_calls": mism, "first_mismatch": site_first,
                "failing_n_at_off0": failing_n, "flags": {"fp8_qkv": gpu.flags.gfx12_mq4v2_fp8_qkv,
                "fp8_gateup": gpu.flags.gfx12_mq4v2_fp8_gateup, "fp8_resid": gpu.flags.gfx12_mq4v2_fp8_resid}})
        );
        total += calls;
        bad += mism;
        if first.is_none() {
            first = site_first;
        }
    }
    (total, bad, first)
}

/// Extents of the fused-versus-plain comparison (PLAN-prefill-tail §9 G1): `1..=65`, the fixed
/// list, and `base + r` for every `r = 1..=63` of the five bases. `quick` keeps a subset.
#[cfg(feature = "deltanet")]
fn fused_n_list(quick: bool) -> Vec<usize> {
    let mut v: std::collections::BTreeSet<usize> = (1..=65).collect();
    v.extend([
        127, 128, 129, 134, 192, 511, 512, 513, 537, 5909, 8191, 8192, 8193,
    ]);
    for b in [64usize, 128, 512, 576, 1024] {
        v.extend(b..=b + 63);
    }
    v.extend([85, 149]);
    v.into_iter()
        .filter(|n| {
            !quick
                || *n <= 70
                || [85, 128, 134, 149, 512, 537, 5909, 8192].contains(n)
                || (*n > 64 && n % 64 == 21)
        })
        .collect()
}

/// PLAN-prefill-tail §3.1: does the producer have to define raw QKV row `t` (call-local) of an
/// `n`-row call? The old predicate with its `t + 3 >= n` threshold replaced by `t + 3 >= (n & !63)`.
#[cfg(feature = "deltanet")]
fn export_raw(t: usize, n: usize) -> bool {
    let t128 = t % 128;
    let rc = n & !63;
    t < n && (t128 < 3 || t128 >= 125 || t + 3 >= rc)
}

#[cfg(feature = "deltanet")]
struct FusedCase {
    /// The fused launcher launched (not declined).
    admitted: bool,
    /// The activations were slab planes (`_b1s`) rather than token records (`_b1`).
    slab: bool,
    /// Raw QKV rows the formula demanded (and that were compared).
    required_rows: usize,
    /// First differing plane/row, `None` when everything required is identical.
    mismatch: Option<String>,
}

/// One extent of the fused-versus-plain GDN preamble comparison on the 27B dims over the `n`
/// activation rows starting at row `off`. See the module docs for what is compared.
#[cfg(feature = "deltanet")]
fn fused_vs_plain(
    gpu: &mut Gpu,
    r: &Res,
    n: usize,
    off: usize,
) -> rdna_compute::HipResult<FusedCase> {
    use rdna_compute::F2GdnTargets;
    let g = r.geo;
    let (hk, hv, hd) = (16usize, 48usize, 128usize);
    let qkv_w = 10240usize;
    let conv_w = gpu.upload_f32(&synth_vec(qkv_w * 4, 0xc0de1, -0.5, 1.0), &[qkv_w * 4])?;
    let conv_init = synth_vec(qkv_w * 3, 0xc0de2, -0.5, 1.0);
    let dt_bias = gpu.upload_f32(&synth_vec(hv, 0xc0de3, -0.2, 0.4), &[hv])?;
    let a_log = gpu.upload_f32(&synth_vec(hv, 0xc0de4, -1.0, 1.0), &[hv])?;
    let q_scale = 1.0 / (hd as f32).sqrt();
    let x = r.x.sub_offset(off * g.k, n * g.k);
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
    let (rc, rt) = (n & !63, n - (n & !63));
    let prev = gpu.ordinary_prefill_enter();
    let out = (|| -> rdna_compute::HipResult<FusedCase> {
        // Plain: separate prepared SETs, then the whole plain prep.
        let p = mk(gpu)?;
        for t in &p[..4] {
            gpu.hip.memset(&t.buf, SENTINEL, t.byte_size())?;
        }
        let res = gpu.reserve_int4_mmq(g.k, n)?;
        let prep = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
            &x,
            &r.norm,
            r.awq.as_ref(),
            None,
            res,
            g.k,
            EPS,
            n,
        )?;
        gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
            &r.w[0], &r.w[1], &r.w[2], &r.w[3], &xr, &prep, &p[0], &p[1], &p[2], &p[3], g.qkv, g.z,
            g.beta, g.alpha, g.k, n,
        )?;
        let raw_plain = gpu.download_raw_bytes(&p[0])?;
        let beta_plain = gpu.download_raw_bytes(&p[2])?;
        let alpha_plain = gpu.download_raw_bytes(&p[3])?;
        gpu.gdn_chunk_prep(
            &p[0], &conv_w, &p[7], &p[3], &p[2], &dt_bias, &a_log, &p[4], &p[5], &p[6], n, q_scale,
            1.0e-6, hk, hv,
        )?;
        // Fused: one launch. Unspecified QKV rows are NaN bytes: a read of one is visible.
        let f = mk(gpu)?;
        gpu.hip.memset(&f[0].buf, 0xFF, f[0].byte_size())?;
        for t in &f[1..4] {
            gpu.hip.memset(&t.buf, SENTINEL, t.byte_size())?;
        }
        let res = gpu.reserve_int4_mmq(g.k, n)?;
        let prep = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
            &x,
            &r.norm,
            r.awq.as_ref(),
            None,
            res,
            g.k,
            EPS,
            n,
        )?;
        let slab = {
            let ptr = gpu.int4_mmq_prepared_ptr(&prep, g.k, n)?;
            gpu.scratch.int4_mmq_slab_at(ptr)
        };
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
            &r.w[0],
            fold,
            &prep,
            &f[0],
            &f[1],
            &f[2],
            &f[3],
            [g.qkv, g.z, g.beta, g.alpha],
            g.k,
            n,
            &targets,
        )?;
        if !admitted {
            for t in p.into_iter().chain(f) {
                gpu.free_tensor(t)?;
            }
            return Ok(FusedCase {
                admitted: false,
                slab,
                required_rows: 0,
                mismatch: None,
            });
        }
        let raw_fused = gpu.download_raw_bytes(&f[0])?;
        let beta_fused = gpu.download_raw_bytes(&f[2])?;
        let alpha_fused = gpu.download_raw_bytes(&f[3])?;
        let mut verdict: Option<String> = None;
        let note = |v: &mut Option<String>, s: String| {
            if v.is_none() {
                *v = Some(s);
            }
        };
        // 1. Every raw QKV row the formula requires equals the independent SET row. Rows outside
        //    the formula are not compared (and are NaN-poisoned, so a read would show below).
        let row = qkv_w * 4;
        let mut required_rows = 0usize;
        for t in 0..n {
            if !export_raw(t, n) {
                continue;
            }
            required_rows += 1;
            let (fr, pr) = (
                &raw_fused[t * row..(t + 1) * row],
                &raw_plain[t * row..(t + 1) * row],
            );
            if let Some(b) = first_diff(fr, pr) {
                note(
                    &mut verdict,
                    format!(
                        "qkv_raw row {t} col {} (required export, t%128={}, n={n} rc={rc} rt={rt}) differs from the independent SET",
                        b / 4,
                        t % 128
                    ),
                );
                break;
            }
        }
        // 2. Every raw alpha/beta row (before any prep).
        for (name, a, b) in [
            ("beta_raw", &beta_fused, &beta_plain),
            ("alpha_raw", &alpha_fused, &alpha_plain),
        ] {
            if let Some(d) = first_diff(a, b) {
                note(
                    &mut verdict,
                    format!(
                        "{name} row {} col {} differs from the independent SET",
                        d / 4 / hv,
                        d / 4 % hv
                    ),
                );
            }
        }
        // 3. Consumer chain exactly as the stream executor runs it: completion pass over the
        //    complete prefix, raw prep of the tail (`rc == 0`: the whole call, every row exported).
        let fix_t = if rc == 0 { n } else { rc };
        gpu.gdn_chunk_prep_fixup(
            &f[0], &conv_w, &f[7], &f[3], &f[2], &dt_bias, &a_log, &f[4], &f[5], &f[6], fix_t,
            q_scale, 1.0e-6, hk, hv,
        )?;
        if rc > 0 && rt > 0 {
            let (x0, a0, b0) = (
                f[0].sub_offset(rc * qkv_w, rt * qkv_w),
                f[3].sub_offset(rc * hv, rt * hv),
                f[2].sub_offset(rc * hv, rt * hv),
            );
            let (q0, k0, v0) = (
                f[4].sub_offset(rc * hk * hd, rt * hk * hd),
                f[5].sub_offset(rc * hk * hd, rt * hk * hd),
                f[6].sub_offset(rc * hv * hd, rt * hv * hd),
            );
            gpu.gdn_chunk_prep(
                &x0, &conv_w, &f[7], &a0, &b0, &dt_bias, &a_log, &q0, &k0, &v0, rt, q_scale,
                1.0e-6, hk, hv,
            )?;
        }
        gpu.hip.device_synchronize()?;
        // Plane order: z, sigmoid beta, gate, q, k, v (F16 operands), conv ring.
        let planes: [(&str, usize, usize); 7] = [
            ("z", 1, g.z * 4),
            ("beta_sigmoid", 2, hv * 4),
            ("gate", 3, hv * 4),
            ("q", 4, hk * hd * 2),
            ("k", 5, hk * hd * 2),
            ("v", 6, hv * hd * 2),
            ("conv_state", 7, 12),
        ];
        for (name, i, row_bytes) in planes {
            let a = gpu.download_raw_bytes(&p[i])?;
            let b = gpu.download_raw_bytes(&f[i])?;
            if let Some(d) = first_diff(&a, &b) {
                let what = if name == "conv_state" {
                    "channel"
                } else {
                    "row"
                };
                note(
                    &mut verdict,
                    format!(
                        "{name} {what} {} (byte {d}) differs after fixup(T={fix_t})+tail prep(rt={}) vs whole plain prep",
                        d / row_bytes,
                        if rc > 0 { rt } else { 0 }
                    ),
                );
            }
        }
        for t in p.into_iter().chain(f) {
            gpu.free_tensor(t)?;
        }
        Ok(FusedCase {
            admitted: true,
            slab,
            required_rows,
            mismatch: verdict,
        })
    })();
    gpu.ordinary_prefill_exit(prev);
    out
}
