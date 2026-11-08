// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness gate for `HIPFIRE_QWEN4_MOE_EPILOGUE_FUSE`: the one launch
//! `moe_shared_down_combine_rows_r<R>` ([`rdna_compute::tensor_ops::moe_shared_down_combine`])
//! against the production chain it replaces on the same operands,
//!
//!   `Gpu::moe_down_combine_top10_batched`  (expanded rows -> target)
//!   `Gpu::gemv_bf16_xf32_x4_rows` / `Gpu::gemv_bf16_xf32` (shared-down F32 rows; what
//!                                          `project_weight` runs for a BF16 weight at 2..=8 / 1 rows)
//!   `bf16_scaled_add_batched`              (shared fold with the selector)
//!
//! with raw f32 bits of the whole `[rows, 2560]` target compared. Real shapes: hidden 2560,
//! shared intermediate 640, 10 routes, 256 experts, rows 1..=8 (the 8-lane C8 step is rows 8).
//! The shared-down weight is the real BF16 `shared_expert.down_proj.weight` of the Flash-Next
//! model (layers 0, 17, 47) when `HIPFIRE_FN_MODEL` (default the 0.4.3 `mgfn` copy) is readable,
//! else seeded; the expanded rows, shared hidden, route, selector and combine seed are seeded.
//! Cases: BF16-exact and raw-F32 inputs, tied / duplicated expert ids, `-1` / out-of-range
//! (unrouted) ids, all-unrouted rows, all-equal ids, NaN / inf / denormal / signed-zero
//! values, a non-zero combine seed.
//!
//! A NaN that meets another NaN in an add / fma takes whichever operand the compiler placed
//! first; the two compiled kernels may differ in that choice, so a word that is NaN in both
//! outputs is counted as a payload-only difference (reported, never a failure). Every other
//! word must be bit-identical.
//!
//! `HIPFIRE_QWEN4_FFN_CLEAR_ABSORB_ROWS` (`moe_epilogue_fuse_zinit_matches_zero_fill_hw`): the
//! `moe_shared_down_combine_rows_r<R>_zinit` entries on a DIRTY target (NaN / garbage poison,
//! plus a sentinel tail past the `[rows, 2560]` extent) against a zero-filled target + the plain
//! `r<R>` entry, every word of the target compared bit for bit (NaN payloads included), rows
//! 1..=8, the scenarios above plus EP owner masks (foreign experts as `-1` ids with NaN expert
//! rows / weights at their slots). A negative control runs the plain entry on the garbage target
//! and must differ from the zero-fill reference at every row count (the comparison sees a seed
//! that is not zero). `moe_epilogue_fuse_zinit_bench_hw` times `zero_f32 + r<R>` against `r<R>_zinit`.
//!
//! `#[ignore]`d: needs an exact gfx1201 device. Sets the flag itself.
//!
//!   cargo test -p rdna-compute --release --test moe_epilogue_fuse_hw -- --ignored --nocapture

use rdna_compute::tensor_ops::{
    bf16_scaled_add_batched, moe_shared_down_combine, Bf16ScaledAddBatched, MoeSharedDownCombine,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::io::{Read, Seek, SeekFrom};

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() & 0xFF_FFFF) as f32 / (0x100_0000 as f32)
    }
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

fn bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if (u & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}
fn bf16_round(v: f32) -> f32 {
    f32::from_bits((bf16_bits(v) as u32) << 16)
}
fn is_nan_bits(u: u32) -> bool {
    (u & 0x7F80_0000) == 0x7F80_0000 && (u & 0x7F_FFFF) != 0
}

fn upload_bytes(gpu: &mut Gpu, dtype: DType, numel: usize, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[numel], dtype).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    upload_bytes(gpu, DType::F32, data.len(), bytes)
}
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    upload_bytes(gpu, DType::F32, data.len(), bytes)
}
fn upload_bf16(gpu: &mut Gpu, data: &[u16]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    upload_bytes(gpu, DType::BF16, data.len(), bytes)
}
fn download_u32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

/// The raw BF16 bytes of tensor `name` of an HFQM v1 file (no decoding), or `None`.
fn read_hfqm_tensor(path: &str, name: &str) -> Option<Vec<u8>> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut header = [0u8; 32];
    f.read_exact(&mut header).ok()?;
    if &header[..4] != b"HFQM" || u32::from_le_bytes(header[4..8].try_into().ok()?) != 1 {
        return None;
    }
    let metadata = u64::from_le_bytes(header[16..24].try_into().ok()?);
    let data = u64::from_le_bytes(header[24..32].try_into().ok()?);
    if data < metadata || data - metadata >= 32_000_000 {
        return None;
    }
    f.seek(SeekFrom::Start(metadata)).ok()?;
    let mut bytes = vec![0u8; (data - metadata) as usize];
    f.read_exact(&mut bytes).ok()?;
    // Skip the leading JSON object (brace matching, string aware).
    let (mut depth, mut i, mut in_str, mut esc) = (0usize, 0usize, false, false);
    loop {
        let ch = *bytes.get(i)?;
        if in_str {
            if esc {
                esc = false;
            } else if ch == b'\\' {
                esc = true;
            } else if ch == b'"' {
                in_str = false;
            }
        } else {
            match ch {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    let word = |pos: &mut usize| -> Option<usize> {
        let v = u32::from_le_bytes(bytes.get(*pos..*pos + 4)?.try_into().ok()?) as usize;
        *pos += 4;
        Some(v)
    };
    let mut pos = i;
    let count = word(&mut pos)?;
    let mut offset = data;
    for _ in 0..count {
        let n = u16::from_le_bytes(bytes.get(pos..pos + 2)?.try_into().ok()?) as usize;
        pos += 2;
        let tname = std::str::from_utf8(bytes.get(pos..pos + n)?)
            .ok()?
            .to_owned();
        pos += n;
        let _qt = *bytes.get(pos)?;
        let ndim = *bytes.get(pos + 1)? as usize;
        pos += 2;
        for _ in 0..ndim {
            word(&mut pos)?;
        }
        word(&mut pos)?;
        let size = u64::from_le_bytes(bytes.get(pos..pos + 8)?.try_into().ok()?);
        pos += 8;
        if tname == name {
            f.seek(SeekFrom::Start(offset)).ok()?;
            let mut raw = vec![0u8; size as usize];
            f.read_exact(&mut raw).ok()?;
            return Some(raw);
        }
        offset += size;
    }
    None
}

const M: usize = 2560;
const K: usize = 640;
const N_EXP: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scenario {
    /// BF16-exact rows, hidden, weights, selector; distinct ids.
    Exact,
    /// Raw F32 rows / hidden / weights / selector (every rounding point is exercised).
    Raw,
    /// BF16-exact rows; tied / duplicated ids, -1 and out-of-range ids, all-unrouted rows,
    /// all-equal ids.
    IdEdges,
    /// NaN / inf / denormal / signed zero sprinkled through every F32 input.
    Special,
    /// A non-zero BF16-exact combine seed (the target is not always the zero fill).
    Seeded,
    /// EP2 owner mask, rank 0 / rank 1: the experts of the other parity are foreign, masked to
    /// `-1` ids with NaN expert rows / weights at their slots (a dead rank's row is valid memory,
    /// never used).
    Foreign0,
    Foreign1,
    /// Every expert foreign (a rank that owns none of a row's routes).
    ForeignAll,
}

fn special(rng: &mut Lcg, v: f32) -> f32 {
    match rng.next_u32() % 64 {
        0 => f32::INFINITY,
        1 => f32::NEG_INFINITY,
        2 => f32::NAN,
        3 => 1e-39,
        4 => 3.4e38,
        5 => 0.0,
        6 => -0.0,
        7 => f32::from_bits(0x7FC0_0001),
        _ => v,
    }
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn moe_epilogue_fuse_matches_unfused_chain_hw() {
    // The flag is read once per process: set it before the first launcher call.
    std::env::set_var("HIPFIRE_QWEN4_MOE_EPILOGUE_FUSE", "1");
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return;
    }
    let model = std::env::var("HIPFIRE_FN_MODEL").unwrap_or_else(|_| {
        "/home/kaden/qcal/release-0.4.3/mgfn/model/qwen3.8-flash-next-base.mq4-xts".to_owned()
    });
    // Weights: the real shared-down matrices of three layers, then a seeded one.
    let mut weights: Vec<(String, Vec<u16>)> = Vec::new();
    for layer in [0usize, 17, 47] {
        let name =
            format!("model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight");
        if let Some(raw) = read_hfqm_tensor(&model, &name) {
            assert_eq!(raw.len(), M * K * 2, "{name}: not [2560, 640] BF16");
            let w: Vec<u16> = raw
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            weights.push((format!("real layer {layer}"), w));
        }
    }
    eprintln!(
        "moe_epilogue_fuse: {} real shared-down matrices from {model}",
        weights.len()
    );
    {
        let mut rng = Lcg(0xD0_0001);
        weights.push((
            "seeded".to_owned(),
            (0..M * K).map(|_| bf16_bits(rng.signed() * 0.04)).collect(),
        ));
    }
    let mut cases = 0usize;
    let mut mismatches = 0usize;
    let mut payload_only = 0usize;
    let scenarios = [
        Scenario::Exact,
        Scenario::Raw,
        Scenario::IdEdges,
        Scenario::Special,
        Scenario::Seeded,
    ];
    for (wi, (wname, w)) in weights.iter().enumerate() {
        let w_t = upload_bf16(&mut gpu, w);
        for (si, scenario) in scenarios.into_iter().enumerate() {
            // The real matrices run the headline scenarios; the seeded one runs them all.
            if wi + 1 != weights.len() && !matches!(scenario, Scenario::Exact | Scenario::Raw) {
                continue;
            }
            for rows in 1..=8usize {
                for seed in 0..2u64 {
                    let mut rng = Lcg(0x4D0E_0000
                        + (wi as u64) * 100_000
                        + (si as u64) * 1000
                        + (rows as u64) * 10
                        + seed);
                    let exact = !matches!(scenario, Scenario::Raw | Scenario::Special);
                    let gen = |rng: &mut Lcg, scale: f32| -> f32 {
                        let v = rng.signed() * scale;
                        let v = if scenario == Scenario::Special {
                            special(rng, v)
                        } else {
                            v
                        };
                        if exact {
                            bf16_round(v)
                        } else {
                            v
                        }
                    };
                    let eo: Vec<f32> = (0..rows * 10 * M).map(|_| gen(&mut rng, 0.06)).collect();
                    // The shared hidden: silu(gate) * up, a few units wide.
                    let x: Vec<f32> = (0..rows * K).map(|_| gen(&mut rng, 1.5)).collect();
                    let seed_target: Vec<f32> = (0..rows * M)
                        .map(|_| {
                            if scenario == Scenario::Seeded {
                                bf16_round(rng.signed() * 0.5)
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let scalar: Vec<f32> = (0..rows)
                        .map(|_| {
                            let v = rng.unit();
                            if exact {
                                bf16_round(v)
                            } else {
                                v
                            }
                        })
                        .collect();
                    let mut ids = vec![0i32; rows * 10];
                    let mut wts = vec![0f32; rows * 10];
                    for r in 0..rows {
                        let mut pool: Vec<i32> = (0..N_EXP as i32).collect();
                        for k in 0..10 {
                            let pick = rng.next_u32() as usize % pool.len();
                            ids[r * 10 + k] = pool.swap_remove(pick);
                            let v = rng.unit() * 0.3;
                            wts[r * 10 + k] = if exact { bf16_round(v) } else { v };
                        }
                        if scenario == Scenario::IdEdges {
                            if r % 2 == 0 {
                                ids[r * 10 + 3] = ids[r * 10 + 7];
                                ids[r * 10 + 1] = ids[r * 10 + 2];
                            }
                            if r % 3 == 0 {
                                ids[r * 10 + 5] = -1;
                                ids[r * 10 + 9] = N_EXP as i32 + 5;
                            }
                            if r % 5 == 1 {
                                for k in 0..10 {
                                    ids[r * 10 + k] = -1;
                                }
                            }
                            if r % 4 == 2 {
                                for k in 0..10 {
                                    ids[r * 10 + k] = 7;
                                }
                            }
                        }
                    }
                    let eo_t = upload_f32(&mut gpu, &eo);
                    let x_t = upload_f32(&mut gpu, &x);
                    let scalar_t = upload_f32(&mut gpu, &scalar);
                    let ids_t = upload_i32(&mut gpu, &ids);
                    let wts_t = upload_f32(&mut gpu, &wts);

                    // Reference: the production chain.
                    let ref_target = upload_f32(&mut gpu, &seed_target);
                    let rows_out = upload_f32(&mut gpu, &vec![0.0; rows * M]);
                    gpu.moe_down_combine_top10_batched(
                        &eo_t,
                        &ids_t,
                        &wts_t,
                        &ref_target,
                        M,
                        rows,
                        N_EXP,
                    )
                    .expect("combine");
                    if rows == 1 {
                        gpu.gemv_bf16_xf32(&w_t, &x_t, &rows_out, M, K)
                            .expect("gemv");
                    } else {
                        let part = (&w_t, &rows_out, M);
                        let none = (&w_t, &rows_out, 0);
                        gpu.gemv_bf16_xf32_x4_rows([part, none, none, none], &x_t, K, rows)
                            .expect("rows gemv");
                    }
                    bf16_scaled_add_batched(
                        &mut gpu,
                        &Bf16ScaledAddBatched {
                            residual: &ref_target,
                            value: &rows_out,
                            scalar: &scalar_t,
                            rows,
                            elements: M,
                        },
                    )
                    .expect("scaled add");
                    gpu.hip.device_synchronize().expect("sync");
                    let want = download_u32(&gpu, &ref_target, rows * M);

                    // Candidate: the one launch.
                    let got_target = upload_f32(&mut gpu, &seed_target);
                    let p = MoeSharedDownCombine {
                        weight: &w_t,
                        x: &x_t,
                        target: &got_target,
                        initial_zero: false,
                        expert_outputs: &eo_t,
                        topk_indices: &ids_t,
                        topk_weights: &wts_t,
                        scalar: &scalar_t,
                        m: M,
                        k: K,
                        rows,
                        n_experts: N_EXP,
                    };
                    assert!(p.applies(&gpu), "operands rejected: rows {rows}");
                    moe_shared_down_combine(&mut gpu, &p).expect("fused launch");
                    gpu.hip.device_synchronize().expect("sync");
                    let got = download_u32(&gpu, &got_target, rows * M);

                    let mut bad = 0usize;
                    let mut first = None;
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        if g != w {
                            if is_nan_bits(*g) && is_nan_bits(*w) {
                                payload_only += 1;
                            } else {
                                bad += 1;
                                first.get_or_insert((i, *g, *w));
                            }
                        }
                    }
                    if bad != 0 {
                        eprintln!(
                            "MISMATCH {wname} {scenario:?} rows {rows} seed {seed}: {bad} words, first {first:?} (index, got, want)"
                        );
                        mismatches += bad;
                    }
                    cases += 1;
                }
            }
        }
    }
    eprintln!(
        "moe_epilogue_fuse: {cases} cases, mismatching words {mismatches}, NaN payload-only differences {payload_only}"
    );
    assert_eq!(
        mismatches, 0,
        "the fused shared-down GEMV + combine is not bit-identical to combine + GEMV + scaled_add"
    );
}

/// The case inputs of one `(scenario, rows, seed)`: `(expanded rows, hidden, scalar, ids, weights)`.
fn zinit_case(
    scenario: Scenario,
    rows: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<f32>) {
    let mut rng = Lcg(0x21_17_00_00 + (rows as u64) * 100 + seed);
    let exact = !matches!(scenario, Scenario::Raw | Scenario::Special);
    let gen = |rng: &mut Lcg, scale: f32| -> f32 {
        let v = rng.signed() * scale;
        let v = if scenario == Scenario::Special { special(rng, v) } else { v };
        if exact {
            bf16_round(v)
        } else {
            v
        }
    };
    let mut eo: Vec<f32> = (0..rows * 10 * M).map(|_| gen(&mut rng, 0.06)).collect();
    let x: Vec<f32> = (0..rows * K).map(|_| gen(&mut rng, 1.5)).collect();
    let scalar: Vec<f32> = (0..rows)
        .map(|_| {
            let v = rng.unit();
            if exact {
                bf16_round(v)
            } else {
                v
            }
        })
        .collect();
    let mut ids = vec![0i32; rows * 10];
    let mut wts = vec![0f32; rows * 10];
    for r in 0..rows {
        let mut pool: Vec<i32> = (0..N_EXP as i32).collect();
        for k in 0..10 {
            let pick = rng.next_u32() as usize % pool.len();
            ids[r * 10 + k] = pool.swap_remove(pick);
            let v = rng.unit() * 0.3;
            wts[r * 10 + k] = if exact { bf16_round(v) } else { v };
        }
        if scenario == Scenario::IdEdges {
            if r % 2 == 0 {
                ids[r * 10 + 3] = ids[r * 10 + 7];
                ids[r * 10 + 1] = ids[r * 10 + 2];
            }
            if r % 3 == 0 {
                ids[r * 10 + 5] = -1;
                ids[r * 10 + 9] = N_EXP as i32 + 5;
            }
            if r % 5 == 1 {
                for k in 0..10 {
                    ids[r * 10 + k] = -1;
                }
            }
            if r % 4 == 2 {
                for k in 0..10 {
                    ids[r * 10 + k] = 7;
                }
            }
        }
        let owner = match scenario {
            Scenario::Foreign0 => Some(0),
            Scenario::Foreign1 => Some(1),
            _ => None,
        };
        for k in 0..10 {
            let slot = r * 10 + k;
            let foreign = scenario == Scenario::ForeignAll
                || owner.is_some_and(|rank| (ids[slot] as usize) % 2 != rank);
            if foreign {
                ids[slot] = -1;
                wts[slot] = f32::NAN;
                for c in 0..M {
                    eo[slot * M + c] = if c % 3 == 0 { f32::NAN } else { f32::INFINITY };
                }
            }
        }
    }
    (eo, x, scalar, ids, wts)
}

/// `[rows * M]` target words followed by a sentinel tail of `TAIL` words, the head filled by `head`.
const TAIL: usize = 1024;
const SENTINEL: u32 = 0xA5A5_A5A5;
fn tailed_target(gpu: &mut Gpu, rows: usize, mut head: impl FnMut(usize) -> u32) -> GpuTensor {
    let words: Vec<u32> = (0..rows * M + TAIL)
        .map(|i| if i < rows * M { head(i) } else { SENTINEL })
        .collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    upload_bytes(gpu, DType::F32, words.len(), bytes)
}

fn real_or_seeded_weights() -> Vec<(String, Vec<u16>)> {
    let model = std::env::var("HIPFIRE_FN_MODEL").unwrap_or_else(|_| {
        "/home/kaden/qcal/release-0.4.3/mgfn/model/qwen3.8-flash-next-base.mq4-xts".to_owned()
    });
    let mut weights: Vec<(String, Vec<u16>)> = Vec::new();
    for layer in [0usize, 17, 47] {
        let name =
            format!("model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight");
        if let Some(raw) = read_hfqm_tensor(&model, &name) {
            assert_eq!(raw.len(), M * K * 2, "{name}: not [2560, 640] BF16");
            weights.push((
                format!("real layer {layer}"),
                raw.chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect(),
            ));
        }
    }
    let mut rng = Lcg(0xD0_0001);
    weights.push((
        "seeded".to_owned(),
        (0..M * K).map(|_| bf16_bits(rng.signed() * 0.04)).collect(),
    ));
    weights
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn moe_epilogue_fuse_zinit_matches_zero_fill_hw() {
    std::env::set_var("HIPFIRE_QWEN4_MOE_EPILOGUE_FUSE", "1");
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return;
    }
    let weights = real_or_seeded_weights();
    eprintln!("zinit oracle: {} weight matrices ({} real)", weights.len(), weights.len() - 1);
    let scenarios = [
        Scenario::Exact,
        Scenario::Raw,
        Scenario::IdEdges,
        Scenario::Special,
        Scenario::Foreign0,
        Scenario::Foreign1,
        Scenario::ForeignAll,
    ];
    // Dirty fills of the candidate's target head: the poison the absorbed fill must never read.
    type Fill = (&'static str, fn(usize) -> u32);
    let fills: [Fill; 4] = [
        ("nan", |_| 0x7FC0_0000),
        ("neg_snan", |_| 0xFFA0_0001),
        ("garbage", |i| (i as u32).wrapping_mul(2654435761) ^ 0x3F80_0000),
        ("inf_mix", |i| if i % 2 == 0 { 0x7F80_0000 } else { 0xFF80_0000 }),
    ];
    let (mut cells, mut words, mut mismatches, mut tail_written) = (0usize, 0usize, 0usize, 0usize);
    let mut control_cells = 0usize;
    let mut control_rows_seen = [false; 8];
    for (wi, (wname, w)) in weights.iter().enumerate() {
        let w_t = upload_bf16(&mut gpu, w);
        for (si, scenario) in scenarios.into_iter().enumerate() {
            // The real matrices run the headline scenarios; the seeded one runs them all.
            if wi + 1 != weights.len() && !matches!(scenario, Scenario::Exact | Scenario::Raw) {
                continue;
            }
            for rows in 1..=8usize {
                for seed in 0..2u64 {
                    let (eo, x, scalar, ids, wts) = zinit_case(scenario, rows, seed + 7 * si as u64);
                    let eo_t = upload_f32(&mut gpu, &eo);
                    let x_t = upload_f32(&mut gpu, &x);
                    let scalar_t = upload_f32(&mut gpu, &scalar);
                    let ids_t = upload_i32(&mut gpu, &ids);
                    let wts_t = upload_f32(&mut gpu, &wts);
                    let run = |gpu: &mut Gpu, target: &GpuTensor, initial_zero: bool| {
                        let p = MoeSharedDownCombine {
                            weight: &w_t,
                            x: &x_t,
                            target,
                            initial_zero,
                            expert_outputs: &eo_t,
                            topk_indices: &ids_t,
                            topk_weights: &wts_t,
                            scalar: &scalar_t,
                            m: M,
                            k: K,
                            rows,
                            n_experts: N_EXP,
                        };
                        assert!(p.applies(gpu), "operands rejected: rows {rows}");
                        moe_shared_down_combine(gpu, &p).expect("fused launch");
                        gpu.hip.device_synchronize().expect("sync");
                    };
                    // Reference: today's route, `zero_f32` + the plain entry.
                    let ref_t = tailed_target(&mut gpu, rows, |_| 0);
                    run(&mut gpu, &ref_t, false);
                    let want = download_u32(&gpu, &ref_t, rows * M + TAIL);
                    for (fname, fill) in fills {
                        let got_t = tailed_target(&mut gpu, rows, fill);
                        run(&mut gpu, &got_t, true);
                        let got = download_u32(&gpu, &got_t, rows * M + TAIL);
                        let bad = got[..rows * M]
                            .iter()
                            .zip(&want[..rows * M])
                            .filter(|(g, w)| g != w)
                            .count();
                        let tail_bad = got[rows * M..].iter().filter(|&&v| v != SENTINEL).count();
                        if bad != 0 || tail_bad != 0 {
                            eprintln!(
                                "MISMATCH {wname} {scenario:?} rows {rows} seed {seed} fill {fname}: {bad} words, tail {tail_bad}"
                            );
                        }
                        mismatches += bad;
                        tail_written += tail_bad;
                        words += rows * M;
                        cells += 1;
                    }
                    // Negative control: the plain entry on the garbage target must NOT match the
                    // zero-fill reference (the seed is read), or the oracle could not see a seed.
                    let ctl_t = tailed_target(&mut gpu, rows, fills[2].1);
                    run(&mut gpu, &ctl_t, false);
                    let ctl = download_u32(&gpu, &ctl_t, rows * M);
                    let differs = ctl.iter().zip(&want[..rows * M]).any(|(g, w)| g != w);
                    if matches!(scenario, Scenario::Exact | Scenario::Raw) {
                        assert!(differs, "negative control did not differ: {wname} {scenario:?} rows {rows}");
                        control_rows_seen[rows - 1] = true;
                        control_cells += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "zinit oracle: {cells} cells, {words} words compared, mismatching words {mismatches}, sentinel tail words written {tail_written}; negative control (plain entry on garbage target differs from the zero fill): {control_cells} cells PASS, rows covered {control_rows_seen:?}"
    );
    assert!(control_rows_seen.iter().all(|&b| b), "negative control missed a row count");
    assert_eq!(mismatches, 0, "_zinit on a dirty target is not bit-identical to zero fill + plain entry");
    assert_eq!(tail_written, 0, "_zinit wrote past the [rows, 2560] target");
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// `zero_f32 + r<R>` against `r<R>_zinit`, per row count, events around each sequence (median
/// of 300, 8 rotated weight sets so the shared-down weight is not L2-resident across iterations),
/// and a 48-sequence layer loop per variant.
#[test]
#[ignore = "needs an exact gfx1201 GPU; timing"]
fn moe_epilogue_fuse_zinit_bench_hw() {
    std::env::set_var("HIPFIRE_QWEN4_MOE_EPILOGUE_FUSE", "1");
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return;
    }
    const SETS: usize = 8;
    const WARM: usize = 30;
    const ITERS: usize = 300;
    let mut rng = Lcg(0xBE_0001);
    let sets: Vec<GpuTensor> = (0..SETS)
        .map(|_| {
            let w: Vec<u16> = (0..M * K).map(|_| bf16_bits(rng.signed() * 0.04)).collect();
            upload_bf16(&mut gpu, &w)
        })
        .collect();
    let e0 = gpu.hip.event_create().expect("event");
    let e1 = gpu.hip.event_create().expect("event");
    println!("rows | zero_f32+r<R> us p10/p50/p90 | r<R>_zinit us p10/p50/p90 | saved p50 | loop48 per-seq us before / after");
    for rows in 1..=8usize {
        let (eo, x, scalar, ids, wts) = zinit_case(Scenario::Exact, rows, 3);
        let eo_t = upload_f32(&mut gpu, &eo);
        let x_t = upload_f32(&mut gpu, &x);
        let scalar_t = upload_f32(&mut gpu, &scalar);
        let ids_t = upload_i32(&mut gpu, &ids);
        let wts_t = upload_f32(&mut gpu, &wts);
        let target = gpu.alloc_tensor(&[rows * M], DType::F32).expect("target");
        let launch = |gpu: &mut Gpu, set: usize, initial_zero: bool| {
            let p = MoeSharedDownCombine {
                weight: &sets[set],
                x: &x_t,
                target: &target,
                initial_zero,
                expert_outputs: &eo_t,
                topk_indices: &ids_t,
                topk_weights: &wts_t,
                scalar: &scalar_t,
                m: M,
                k: K,
                rows,
                n_experts: N_EXP,
            };
            moe_shared_down_combine(gpu, &p).expect("fused launch");
        };
        let (mut before, mut after) = (Vec::new(), Vec::new());
        for it in 0..WARM + ITERS {
            let (sb, sa) = (it % SETS, (it + 3) % SETS);
            gpu.hip.event_record(&e0, gpu.active_stream.as_ref()).unwrap();
            gpu.zero_f32(&target).expect("zero_f32");
            launch(&mut gpu, sb, false);
            gpu.hip.event_record(&e1, gpu.active_stream.as_ref()).unwrap();
            gpu.hip.event_synchronize(&e1).unwrap();
            let b = gpu.hip.event_elapsed_ms(&e0, &e1).unwrap() as f64 * 1000.0;
            gpu.hip.event_record(&e0, gpu.active_stream.as_ref()).unwrap();
            launch(&mut gpu, sa, true);
            gpu.hip.event_record(&e1, gpu.active_stream.as_ref()).unwrap();
            gpu.hip.event_synchronize(&e1).unwrap();
            let a = gpu.hip.event_elapsed_ms(&e0, &e1).unwrap() as f64 * 1000.0;
            if it >= WARM {
                before.push(b);
                after.push(a);
            }
        }
        let mut loops = [Vec::new(), Vec::new()];
        for rep in 0..25 {
            for (variant, out) in loops.iter_mut().enumerate() {
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref()).unwrap();
                for layer in 0..48usize {
                    if variant == 0 {
                        gpu.zero_f32(&target).expect("zero_f32");
                    }
                    launch(&mut gpu, (layer + rep) % SETS, variant == 1);
                }
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref()).unwrap();
                gpu.hip.event_synchronize(&e1).unwrap();
                if rep >= 5 {
                    out.push(gpu.hip.event_elapsed_ms(&e0, &e1).unwrap() as f64 * 1000.0 / 48.0);
                }
            }
        }
        let (b50, a50) = (pct(&mut before, 0.5), pct(&mut after, 0.5));
        println!(
            "{rows} | {:.2}/{b50:.2}/{:.2} | {:.2}/{a50:.2}/{:.2} | {:.2} | {:.2} / {:.2}",
            pct(&mut before, 0.1),
            pct(&mut before, 0.9),
            pct(&mut after, 0.1),
            pct(&mut after, 0.9),
            b50 - a50,
            pct(&mut loops[0], 0.5),
            pct(&mut loops[1], 0.5),
        );
    }
}
