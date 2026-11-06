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
