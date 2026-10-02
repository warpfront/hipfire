// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU↔CPU GEMV parity for the host-mapped offload path, one row per quant
//! format.
//!
//! `hipfire_cpu::gemv` is a transcription of the *decode*, not of the GPU
//! kernel's accumulation order (the offload path's contract is coherence, see
//! `docs/perf-checkpoints/2026-09-26-llamacpp-offload-scaling-baseline.md`), so
//! this asserts a tolerance rather than bit-identity. What it does prove per
//! format: the byte layout, the group header/codebook, the code unpacking and
//! the activation-rotation convention all match the real launcher — a wrong
//! nibble order, a missing codebook offset, or a mirrored FWHT lands orders of
//! magnitude outside `TOL`, not a rounding difference.
//!
//! Two sources of weights:
//!
//! * real projection tensors out of the pulled fixtures ([`REAL`]), and
//! * the fixture's own AWQ sidecar attached to the real tensors that have one
//!   (the launcher applies that per-channel divide inside the rotation), and
//! * synthetic buffers for the formats no fixture on disk carries ([`SYNTH`]).
//!   A decode check does not care whether the bytes came from a quantizer, so
//!   this keeps the matrix complete instead of "whatever the local model
//!   directory happens to contain" — but the real tensors are the stronger
//!   evidence and run first.
//!
//! `#[ignore]`d: needs an RDNA GPU with a working HIP toolchain; the real-tensor
//! rows additionally need `hipfire pull qwen3.5:2b`, `:2b-mq3`, `:2b-mq6`,
//! `:2b-hf6`. Run explicitly (both arms):
//!
//!   cargo test -p hipfire-arch-qwen35 --release --test gpu_gemv_parity -- --ignored --nocapture

use std::collections::BTreeMap;
use std::path::PathBuf;

use hipfire_cpu::gemv::gemv as cpu_gemv;
use hipfire_cpu::quant::{divide_by_awq_scale, rotate_x, CpuQuant};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::{GemvFamily, GemvParams, WeightRef};
use hipfire_dispatch::types::GemvVariant;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_backend::load_awq_scale_for;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Relative tolerance against `max|reference|`.
///
/// Measured worst case over the whole matrix on gfx1201 (2026-09-27): `6.65e-7`
/// (`Mq3G256Lloyd`, real 2B tensor); several formats came back *bit-exact*, and
/// the element formats with a single accumulation chain agree exactly. `1e-4` is
/// ~150x that noise and still 3+ orders below any layout/sign failure (a wrong
/// nibble order or a mirrored FWHT changes the result by O(1) relative).
const TOL: f32 = 1e-4;

/// Largest tensor (elements) put through the CPU side, so the test stays fast.
/// The 9B's `down_proj` is `[4096, 12288]` = 50M elements, which the CPU side
/// handles in well under a second.
const MAX_ELEMS: usize = 64 << 20;

/// Tensors per (fixture, quant type), largest first — big `k` is what exercises
/// many groups and a real row stride.
const PER_FORMAT: usize = 2;

/// (fixture file, quant_type, DType, CpuQuant) — measured quant types, not
/// assumed: the registry's `-mq3` tags ship the Lloyd-Max tier (qt 20).
const REAL: &[(&str, u8, DType, CpuQuant)] = &[
    ("qwen3.5-2b.mq4", 13, DType::MQ4G256, CpuQuant::Mq4G256),
    (
        "qwen3.5-2b.mq3",
        20,
        DType::MQ3G256Lloyd,
        CpuQuant::Mq3G256Lloyd,
    ),
    ("qwen3.5-2b.mq6", 15, DType::MQ6G256, CpuQuant::Mq6G256),
    ("qwen3.5-2b.hf6", 8, DType::HFQ6G256, CpuQuant::Hfq6G256),
    // Same format, larger model: the 9B's 4096-wide rows exercise 16 groups.
    ("qwen3.5-9b.mq4", 13, DType::MQ4G256, CpuQuant::Mq4G256),
];

/// Formats with no fixture on disk: (quant_type, DType, CpuQuant).
const SYNTH: &[(u8, DType, CpuQuant)] = &[
    (13, DType::MQ4G256, CpuQuant::Mq4G256),
    (44, DType::MQ4G256V2, CpuQuant::Mq4G256V2),
    (45, DType::MQ4CG256, CpuQuant::Mq4CG256),
    (15, DType::MQ6G256, CpuQuant::Mq6G256),
    (47, DType::MQ6G256V2, CpuQuant::Mq6G256V2),
    (31, DType::MQ5G256, CpuQuant::Mq5G256),
    (48, DType::MQ5G256V2, CpuQuant::Mq5G256V2),
    (17, DType::MQ3G256, CpuQuant::Mq3G256),
    (49, DType::MQ3G256V2, CpuQuant::Mq3G256V2),
    (20, DType::MQ3G256Lloyd, CpuQuant::Mq3G256Lloyd),
    (18, DType::MQ2G256, CpuQuant::Mq2G256),
    (50, DType::MQ2G256V2, CpuQuant::Mq2G256V2),
    (19, DType::MQ2G256Lloyd, CpuQuant::Mq2G256Lloyd),
    (51, DType::MQ2G256LloydU, CpuQuant::Mq2G256LloydU),
    (30, DType::MQ4G256Lloyd, CpuQuant::Mq4G256Lloyd),
    (8, DType::HFQ6G256, CpuQuant::Hfq6G256),
    (6, DType::HFQ4G256, CpuQuant::Hfq4G256),
    (7, DType::HFQ4G128, CpuQuant::Hfq4G128),
    (11, DType::HFQ3G256, CpuQuant::Hfq3G256),
    (12, DType::HFQ3G128, CpuQuant::Hfq3G128),
    (9, DType::HFQ2G256, CpuQuant::Hfq2G256),
    (10, DType::HFQ2G128, CpuQuant::Hfq2G128),
    (40, DType::TQ2G128, CpuQuant::Tq2G128),
    (41, DType::BQ1G128, CpuQuant::Bq1G128),
    (3, DType::Q8_0, CpuQuant::Q8F16),
];

/// A real-tensor row for a fixture outside `~/.hipfire/models`, covering formats
/// no local artifact carries. `HIPFIRE_PARITY_EXTRA_MODEL=<path>` opts in;
/// `HIPFIRE_PARITY_EXTRA_QT` (default 49) picks the quant type to compare. This
/// is the oracle of record for a format with no canonical `dequantize_to_f32`
/// arm — e.g. qt 49, which is what `qwen3.8-27b.mq3-xt` ships.
const EXTRA: &[(u8, DType, CpuQuant)] = &[
    (49, DType::MQ3G256V2, CpuQuant::Mq3G256V2),
    (45, DType::MQ4CG256, CpuQuant::Mq4CG256),
];

fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HIPFIRE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| PathBuf::from("/").into());
    PathBuf::from(home).join(".hipfire").join("models")
}

/// Deterministic activation, magnitudes around 1e-3 like a normalized hidden
/// state. Both paths get the identical bytes.
fn activation(k: usize, salt: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let v = ((i + salt) as u64 * 2654435761) % 8192;
            (v as f32 - 4096.0) * 0.000_244_140_625
        })
        .collect()
}

/// Synthetic `[m, k]` weights: a sane header/codebook per group plus a
/// byte-diverse payload, so no value is `Inf`/`NaN` and every code path in the
/// decode is hit.
fn synth_weights(q: CpuQuant, m: usize, k: usize) -> Vec<u8> {
    let (ge, gb) = (q.group_elems(), q.group_bytes());
    let mut out = Vec::with_capacity(m * (k / ge) * gb);
    for row in 0..m {
        for g in 0..k / ge {
            let salt = (row * (k / ge) + g) * 7;
            let mut bytes: Vec<u8> = (0..gb).map(|i| ((i + salt) * 37 + 11) as u8).collect();
            let f32x2 = |a: f32, b: f32| {
                let mut v = [0u8; 8];
                v[..4].copy_from_slice(&a.to_le_bytes());
                v[4..].copy_from_slice(&b.to_le_bytes());
                v
            };
            let f16s = |vals: &[u16]| {
                let mut v = Vec::with_capacity(vals.len() * 2);
                for x in vals {
                    v.extend_from_slice(&x.to_le_bytes());
                }
                v
            };
            match q {
                CpuQuant::Mq4G256 | CpuQuant::Hfq4G256 => {
                    bytes[..8].copy_from_slice(&f32x2(0.03125, -0.5))
                }
                CpuQuant::Mq6G256 | CpuQuant::Hfq6G256 => {
                    bytes[..8].copy_from_slice(&f32x2(0.0078125, -0.125))
                }
                CpuQuant::Mq3G256 => bytes[..8].copy_from_slice(&f32x2(0.015625, 0.25)),
                CpuQuant::Mq4G256V2
                | CpuQuant::Mq3G256V2
                | CpuQuant::Mq5G256V2
                | CpuQuant::Mq6G256V2
                | CpuQuant::Mq2G256V2 => bytes[..8].copy_from_slice(&f16s(&[
                    0x2c00, 0xb400, 0x3800, 0x3a00, // 0.0625, -0.25, 0.5, 0.75
                ])),
                CpuQuant::Mq4CG256 => bytes[..4].copy_from_slice(&f16s(&[0x2c00, 0xb400])),
                CpuQuant::Hfq3G256
                | CpuQuant::Hfq2G256
                | CpuQuant::Hfq4G128
                | CpuQuant::Hfq3G128
                | CpuQuant::Hfq2G128
                | CpuQuant::Mq2G256
                | CpuQuant::Mq5G256 => bytes[..8].copy_from_slice(&f32x2(0.03125, -0.5)),
                CpuQuant::Tq2G128 | CpuQuant::Bq1G128 => {
                    bytes[..2].copy_from_slice(&0x3800u16.to_le_bytes())
                }
                CpuQuant::Mq2G256Lloyd | CpuQuant::Mq2G256LloydU => {
                    bytes[..8].copy_from_slice(&f16s(&[0xbc00, 0xb400, 0x3400, 0x3c00]))
                }
                CpuQuant::Mq3G256Lloyd => bytes[..16].copy_from_slice(&f16s(&[
                    0xbc00, 0xb800, 0xb400, 0xb000, 0x3000, 0x3400, 0x3800, 0x3c00,
                ])),
                CpuQuant::Mq4G256Lloyd => bytes[..32].copy_from_slice(&f16s(&[
                    0xbc00, 0xb800, 0xb400, 0xb000, 0x3000, 0x3400, 0x3800, 0x3c00, 0xbc00, 0xb800,
                    0xb400, 0xb000, 0x3000, 0x3400, 0x3800, 0x3c00,
                ])),
                CpuQuant::Q8F16 => bytes[..2].copy_from_slice(&0x3800u16.to_le_bytes()),
                // The element formats are covered bit-exactly by the fixture
                // tables in `hipfire-cpu` and by the cross-check over real norms.
                CpuQuant::F16 | CpuQuant::F32 | CpuQuant::Bf16 => unreachable!("not in SYNTH"),
            }
            out.extend_from_slice(&bytes);
        }
    }
    out
}

struct Parity {
    max_abs: f32,
    rel: f32,
}

/// Compare the production launcher against the CPU transcription on identical
/// bytes and activation.
#[allow(clippy::too_many_arguments)]
fn compare(
    gpu: &mut Gpu,
    gemv: &GemvFamily,
    label: &str,
    q: CpuQuant,
    dtype: DType,
    bytes: &[u8],
    m: usize,
    k: usize,
    awq: Option<&GpuTensor>,
    worst: &mut BTreeMap<&'static str, Parity>,
) -> bool {
    let w = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload w");
    let x_host = activation(k, m);
    let x_dev = gpu.upload_f32(&x_host, &[k]).expect("upload x");
    let y_dev = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");
    let ctx = DispatchCtx::new(gpu);
    let wr = WeightRef {
        buf: &w,
        dtype,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: awq,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    match gemv.run_auto(&ctx, gpu, &wr, &x_dev, &y_dev) {
        Ok(()) => {}
        Err(hipfire_dispatch::types::DispatchError::MissingImpl { key }) => {
            // The *production launcher* has no dense GEMV for this dtype on this
            // arch (e.g. `GemvHfq3G256` on gfx12): the format cannot appear as a
            // `Step::Gemv` weight here, so there is nothing to compare against.
            // Its decode is still pinned by `hipfire-cpu`'s expectation table
            // where a canonical arm exists. Reported, never silently skipped.
            eprintln!("{label:58} SKIP — production launcher has no kernel ({key:?}) on this arch");
            gpu.free_tensor(w).ok();
            gpu.free_tensor(x_dev).ok();
            gpu.free_tensor(y_dev).ok();
            return false;
        }
        Err(e) => panic!("{label}: launcher failed: {e:?}"),
    }
    gpu.hip.device_synchronize().expect("sync");
    let mut gpu_y = vec![0.0f32; m];
    let gpu_y_bytes =
        unsafe { std::slice::from_raw_parts_mut(gpu_y.as_mut_ptr() as *mut u8, m * 4) };
    gpu.hip.memcpy_dtoh(gpu_y_bytes, &y_dev.buf).expect("dtoh");

    // Mirror `cpu_exec::prepare_activation`: the AWQ divide happens *inside* the
    // rotation (the quantizer pre-scaled the weights by `s`), so it must precede
    // the FWHT — and must not be applied at all to a pre-rotated input.
    let mut x_cpu = x_host;
    if q.is_fwht_g256() {
        if let Some(scale) = awq {
            let mut sc = vec![0.0f32; k];
            let sc_bytes =
                unsafe { std::slice::from_raw_parts_mut(sc.as_mut_ptr() as *mut u8, k * 4) };
            gpu.hip
                .memcpy_dtoh(sc_bytes, &scale.buf)
                .expect("dtoh scale");
            divide_by_awq_scale(&mut x_cpu, &sc);
        }
        rotate_x(&mut x_cpu);
    }
    let mut cpu_y = vec![0.0f32; m];
    cpu_gemv(q, bytes, m, k, &x_cpu, &mut cpu_y);

    let max_abs = gpu_y
        .iter()
        .zip(&cpu_y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = cpu_y.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    let rel = max_abs / scale;
    eprintln!("{label:58} m={m:<6} k={k:<6} max_abs={max_abs:.3e} rel={rel:.3e}",);
    assert!(
        rel <= TOL,
        "{label} (qt {q:?}): relative error {rel:.3e} exceeds {TOL:.0e} \
         (max_abs {max_abs:.3e}, scale {scale:.3e})"
    );
    let entry = worst.entry(q_format_name(q)).or_insert(Parity {
        max_abs: 0.0,
        rel: 0.0,
    });
    entry.max_abs = entry.max_abs.max(max_abs);
    entry.rel = entry.rel.max(rel);

    gpu.free_tensor(w).ok();
    gpu.free_tensor(x_dev).ok();
    gpu.free_tensor(y_dev).ok();
    true
}

/// The `Prerotated` arm: the launcher's per-row kernels take an activation that
/// was rotated by an earlier step, and the CPU path must then **not** rotate it
/// again. A double rotation is a silent `R^2` error — the output still looks like
/// activations (it is a norm-preserving transform of a real activation), so only
/// this comparison catches it.
///
/// The rotation here is `hipfire_cpu::rotate_x`, verified against the GPU
/// rotation kernel by construction of the same sign tables and by
/// `hipfire_cpu`'s Walsh-Hadamard oracle; both arms get byte-identical input.
/// The `Prerotated` half of a format's rotation contract: with an activation
/// rotated by `rotate_x`, the per-row kernel must produce the same result the
/// CPU gets by dotting the *decoded codes* against that already-rotated input —
/// i.e. neither side may rotate again. `compare` covers the `Raw` half (both
/// sides rotate exactly once); this covers the "already rotated" half, which the
/// same wrong flag breaks in the same silent `R^2` way.
///
/// Returns `false` for a format that does not rotate at all (nothing to check)
/// or whose launcher has no kernel on this arch.
fn check_prerotated(
    gpu: &mut Gpu,
    gemv: &GemvFamily,
    label: &str,
    q: CpuQuant,
    dtype: DType,
    bytes: &[u8],
    m: usize,
    k: usize,
) -> bool {
    if !q.is_fwht_g256() {
        return false;
    }
    let w = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload w");
    let mut x_rot_host = activation(k, m);
    rotate_x(&mut x_rot_host);
    let x_dev = gpu.upload_f32(&x_rot_host, &[k]).expect("upload x_rot");
    let y_dev = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");
    let ctx = DispatchCtx::new(gpu);
    let wr = WeightRef {
        buf: &w,
        dtype,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let launched = gemv.run(
        &ctx,
        gpu,
        &GemvParams {
            w: &wr,
            x: &x_dev,
            y: &y_dev,
            variant: GemvVariant::Prerotated,
            residual: None,
            gate: None,
            up: None,
        },
    );
    if let Err(hipfire_dispatch::types::DispatchError::MissingImpl { key }) = &launched {
        eprintln!("{label:58} SKIP prerotated — no kernel ({key:?}) on this arch");
        gpu.free_tensor(w).ok();
        gpu.free_tensor(x_dev).ok();
        gpu.free_tensor(y_dev).ok();
        return false;
    }
    launched.unwrap_or_else(|e| panic!("{label}: prerotated launcher failed: {e:?}"));
    gpu.hip.device_synchronize().expect("sync");
    let mut gpu_y = vec![0.0f32; m];
    let gpu_y_bytes =
        unsafe { std::slice::from_raw_parts_mut(gpu_y.as_mut_ptr() as *mut u8, m * 4) };
    gpu.hip.memcpy_dtoh(gpu_y_bytes, &y_dev.buf).expect("dtoh");

    // CPU: the same rotated activation, used exactly as handed over.
    let mut cpu_y = vec![0.0f32; m];
    cpu_gemv(q, bytes, m, k, &x_rot_host, &mut cpu_y);
    let max_abs = gpu_y
        .iter()
        .zip(&cpu_y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = cpu_y.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    let rel = max_abs / scale;
    eprintln!("{label:58} prerotated m={m:<6} k={k:<6} max_abs={max_abs:.3e} rel={rel:.3e}");
    assert!(
        rel <= TOL,
        "{label}: Prerotated parity {rel:.3e} exceeds {TOL:.0e} — the CPU path must not \
         rotate an already-rotated activation"
    );
    gpu.free_tensor(w).ok();
    gpu.free_tensor(x_dev).ok();
    gpu.free_tensor(y_dev).ok();
    true
}

#[test]
#[ignore]
fn gpu_cpu_gemv_parity_prerotated_input() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    gpu.ensure_mq_signs().expect("mq signs");
    let dir = models_dir();
    let gemv = GemvFamily::new();
    let mut checked = 0usize;

    // Real tensors from the fixtures …
    for (file, qt, dtype, q) in REAL {
        if !q.is_fwht_g256() {
            continue;
        }
        let path = dir.join(file);
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            continue;
        }
        let hfq = HfqFile::open(&path).expect("open fixture");
        let Some((name, m, k)) = hfq
            .tensors()
            .iter()
            .filter(|i| {
                i.quant_type == *qt && i.shape.len() == 2 && !i.name.contains("embed_tokens") && {
                    let k = i.shape[1] as usize;
                    k % 256 == 0 && (i.shape[0] as usize) * k <= MAX_ELEMS
                }
            })
            .max_by_key(|i| (i.shape[0] as usize) * (i.shape[1] as usize))
            .map(|i| (i.name.clone(), i.shape[0] as usize, i.shape[1] as usize))
        else {
            continue;
        };
        let (_, bytes) = hfq.tensor_data_vec(&name).expect("tensor bytes");
        if check_prerotated(
            &mut gpu,
            &gemv,
            &format!("{file} {name}"),
            *q,
            *dtype,
            &bytes,
            m,
            k,
        ) {
            checked += 1;
        }
    }
    // … and one synthetic buffer per rotating format, so a format with no local
    // artifact still gets both halves of the contract.
    for (qt, dtype, q) in SYNTH {
        let (m, k) = (64usize, 1024usize);
        let bytes = synth_weights(*q, m, k);
        if check_prerotated(
            &mut gpu,
            &gemv,
            &format!("synthetic qt={qt} {q:?}"),
            *q,
            *dtype,
            &bytes,
            m,
            k,
        ) {
            checked += 1;
        }
    }
    eprintln!("prerotated parity: {checked} checks");
    assert!(checked > 0, "no fixture was exercised");
}

#[test]
#[ignore]
fn gpu_cpu_gemv_parity_per_format() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    gpu.ensure_mq_signs().expect("mq signs");
    let dir = models_dir();
    let gemv = GemvFamily::new();
    let mut worst: BTreeMap<&'static str, Parity> = BTreeMap::new();
    let mut synthetic_formats = 0usize;
    let mut ineligible: Vec<CpuQuant> = Vec::new();

    for (file, qt, dtype, q) in REAL {
        let path = dir.join(file);
        if !path.exists() {
            eprintln!(
                "skip: {} not present (hipfire pull the matching tag)",
                path.display()
            );
            continue;
        }
        let hfq = HfqFile::open(&path).expect("open fixture");
        // Largest `[m, k]` tensors of this format, so `k` spans many groups and
        // `m` spans many rows.
        let mut candidates: Vec<(usize, usize, String)> = hfq
            .tensors()
            .iter()
            .filter(|i| {
                i.quant_type == *qt && i.shape.len() == 2 && !i.name.contains("embed_tokens") && {
                    let (m, k) = (i.shape[0] as usize, i.shape[1] as usize);
                    k % 256 == 0 && m * k <= MAX_ELEMS
                }
            })
            .map(|i| (i.shape[0] as usize, i.shape[1] as usize, i.name.clone()))
            .collect();
        candidates.sort_by_key(|(m, k, _)| std::cmp::Reverse(m * k));
        let mut done = 0usize;
        for (m, k, name) in candidates {
            if done >= PER_FORMAT {
                break;
            }
            done += 1;
            let (_, bytes) = hfq
                .tensor_data_vec(&name)
                .unwrap_or_else(|| panic!("{file}: no bytes for {name}"));
            let label = format!(
                "{file} qt={qt} {}",
                name.rsplit('.').nth(1).unwrap_or(&name)
            );
            compare(
                &mut gpu, &gemv, &label, *q, *dtype, &bytes, m, k, None, &mut worst,
            );
            // AWQ arm (`RotateMqAwq`): the fixture's own sidecar, attached exactly
            // when the loader would attach it. This is the arm that catches a
            // dropped per-channel divide — the failure mode is `(W·s)·x`, which no
            // tolerance on an unsclaed comparison can see.
            if dtype.supports_awq_sidecar() && done == 1 {
                if let Some(scale) = load_awq_scale_for(&hfq, &gpu, &name, k) {
                    let label = format!("{label} +awq");
                    compare(
                        &mut gpu,
                        &gemv,
                        &label,
                        *q,
                        *dtype,
                        &bytes,
                        m,
                        k,
                        Some(&scale),
                        &mut worst,
                    );
                }
            }
        }
        if done == 0 {
            eprintln!("note: {file} carries no 2-D qt {qt} tensor under {MAX_ELEMS} elements");
        }
    }

    for (qt, dtype, q) in SYNTH {
        let (m, k) = (64usize, 1024usize);
        let bytes = synth_weights(*q, m, k);
        let label = format!("synthetic qt={qt} {q:?}");
        if compare(
            &mut gpu, &gemv, &label, *q, *dtype, &bytes, m, k, None, &mut worst,
        ) {
            synthetic_formats += 1;
        } else {
            ineligible.push(*q);
        }
    }

    // Opt-in real-tensor rows for formats no local artifact carries.
    if let Some(path) = std::env::var_os("HIPFIRE_PARITY_EXTRA_MODEL") {
        let hfq = HfqFile::open(std::path::Path::new(&path)).expect("open extra fixture");
        let want: u8 = std::env::var("HIPFIRE_PARITY_EXTRA_QT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(49);
        for (qt, dtype, q) in EXTRA {
            if *qt != want {
                continue;
            }
            let mut candidates: Vec<(usize, usize, String)> = hfq
                .tensors()
                .iter()
                .filter(|i| {
                    i.quant_type == *qt
                        && i.shape.len() == 2
                        && !i.name.contains("embed_tokens")
                        && {
                            let (m, k) = (i.shape[0] as usize, i.shape[1] as usize);
                            k % 256 == 0 && m * k <= MAX_ELEMS
                        }
                })
                .map(|i| (i.shape[0] as usize, i.shape[1] as usize, i.name.clone()))
                .collect();
            candidates.sort_by_key(|(m, k, _)| std::cmp::Reverse(m * k));
            for (m, k, name) in candidates.into_iter().take(PER_FORMAT) {
                let (_, bytes) = hfq.tensor_data_vec(&name).expect("extra tensor bytes");
                let label = format!(
                    "{} qt={qt} {}",
                    path.to_string_lossy().rsplit('/').next().unwrap_or("extra"),
                    name.rsplit('.').nth(1).unwrap_or(&name)
                );
                compare(
                    &mut gpu, &gemv, &label, *q, *dtype, &bytes, m, k, None, &mut worst,
                );
            }
        }
    }

    eprintln!("\nper-format worst case (max_abs, relative):");
    for (fmt, p) in &worst {
        eprintln!("  {fmt:<14} {:.3e}  {:.3e}", p.max_abs, p.rel);
    }
    assert_eq!(
        synthetic_formats + ineligible.len(),
        SYNTH.len(),
        "every synthetic format must either compare or be reported arch-ineligible"
    );
    if !ineligible.is_empty() {
        eprintln!(
            "arch-ineligible on {} (no production dense GEMV kernel; canonical-table oracle only): {ineligible:?}",
            gpu.arch
        );
    }
    // A format with no canonical decoder has no other oracle: state which ones
    // were only checked synthetically, so the record is not read as if every row
    // came from a real tensor.
    eprintln!(
        "\nnote: qt 45/47/48/49/50/9/10/31 have no canonical `dequantize_to_f32` arm; \
         their rows are the production launcher vs the transcription (synthetic buffers, \
         or a real tensor via HIPFIRE_PARITY_EXTRA_MODEL)."
    );
}

fn q_format_name(q: CpuQuant) -> &'static str {
    match q {
        CpuQuant::Mq4G256 => "Mq4G256",
        CpuQuant::Mq4G256V2 => "Mq4G256V2",
        CpuQuant::Mq6G256 => "Mq6G256",
        CpuQuant::Mq3G256 => "Mq3G256",
        CpuQuant::Mq3G256Lloyd => "Mq3G256Lloyd",
        CpuQuant::Mq3G256V2 => "Mq3G256V2",
        CpuQuant::Mq2G256 => "Mq2G256",
        CpuQuant::Mq2G256V2 => "Mq2G256V2",
        CpuQuant::Mq2G256Lloyd => "Mq2G256Lloyd",
        CpuQuant::Mq2G256LloydU => "Mq2G256LloydU",
        CpuQuant::Mq4G256Lloyd => "Mq4G256Lloyd",
        CpuQuant::Mq4CG256 => "Mq4CG256",
        CpuQuant::Mq5G256 => "Mq5G256",
        CpuQuant::Mq5G256V2 => "Mq5G256V2",
        CpuQuant::Mq6G256V2 => "Mq6G256V2",
        CpuQuant::Hfq6G256 => "Hfq6G256",
        CpuQuant::Hfq4G256 => "Hfq4G256",
        CpuQuant::Hfq4G128 => "Hfq4G128",
        CpuQuant::Hfq3G256 => "Hfq3G256",
        CpuQuant::Hfq3G128 => "Hfq3G128",
        CpuQuant::Hfq2G256 => "Hfq2G256",
        CpuQuant::Hfq2G128 => "Hfq2G128",
        CpuQuant::Tq2G128 => "Tq2G128",
        CpuQuant::Bq1G128 => "Bq1G128",
        CpuQuant::F16 => "F16",
        CpuQuant::F32 => "F32",
        CpuQuant::Bf16 => "Bf16",
        CpuQuant::Q8F16 => "Q8F16",
    }
}
