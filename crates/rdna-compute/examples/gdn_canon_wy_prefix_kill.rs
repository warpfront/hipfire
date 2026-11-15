// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S0 kill 2, C16 family (CHUNK-CANONICAL-DESIGN §3.4, §6 S0): **scan prefix
//! stability of the Qwen4 WY chunk kernel** (`gated_delta_chunk_gate_wmma` via
//! `gated_delta_step_gate_wmma`, the product call) on gfx1100 / gfx1151.
//!
//! For every r = 1..15 and head ∈ {0, 16, 512} completed rows before the
//! microtile, with a **nonzero incoming state** (Q8Ef: nonzero Q8 codes/scales
//! plus nonzero F16 EF residual; and an F32 state), two calls over identical
//! inputs:
//!
//! * `A` = `head + 16` real rows (the 16-row microtile is complete);
//! * `B(pad)` = `head + r` rows, every operand row past the valid rows (BF16
//!   projection, gate, beta, z) and the whole output tail filled with zero /
//!   NaN / huge bytes.
//!
//! Output rows `0..head+r` of `B(pad)` must be byte-identical to the same rows
//! of `A`; `B(nan)`, `B(huge)` byte-identical to `B(zero)`; padded output rows
//! untouched. Controls: a sign flip of row 0's value head must change the
//! compared rows, and repeating `B(zero)` must reproduce its bytes. The
//! partial end state is reported, not gated.
//!
//! The WY route is admitted below 512 rows by entering the prefill-invariant
//! scope (`Gpu::prefill_invariant_scope`, route_rows = canonical 8192), exactly
//! how Qwen4 canonical prefill reaches it; the actual extents stay the real row
//! counts.
//!
//! Build: cargo build --release -p rdna-compute --features lab,deltanet --example gdn_canon_wy_prefix_kill
//! Run:   HIPFIRE_GRAPH=0 gdn_canon_wy_prefix_kill --json <receipt.json> [--heads 0,16,512]
//! Exit: 0 prefix-stable | 1 prefix identity fails | 3 environment / route not taken.

#[path = "gdn_canon_common/mod.rs"]
mod canon;

use canon::util::*;
use half::bf16;
use rdna_compute::tensor_ops::*;
use rdna_compute::{DType, Gpu};
use serde_json::{json, Value};

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const DIM: usize = 128;
const QKV_WIDTH: usize = 2 * KEY_HEADS * DIM + VALUE_HEADS * DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * DIM;
const MICRO: usize = 16;
const STATE_ELEMS: usize = VALUE_HEADS * DIM * DIM;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pad {
    Zero,
    Nan,
    Huge,
}

impl Pad {
    fn byte(self) -> u8 {
        match self {
            Pad::Zero => 0x00,
            Pad::Nan => 0xff,
            Pad::Huge => 0x77,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Pad::Zero => "zero",
            Pad::Nan => "nan",
            Pad::Huge => "huge",
        }
    }
    fn f32(self) -> f32 {
        let b = self.byte() as u32;
        f32::from_bits(b | b << 8 | b << 16 | b << 24)
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

fn gauss(rng: &mut Rng) -> f32 {
    let (a, b) = (rng.unit().max(1e-7), rng.unit());
    (-2.0 * a.ln()).sqrt() * (2.0 * std::f32::consts::PI * b).cos()
}

struct Fixture {
    full_rows: usize,
    /// BF16-exact `[rows][QKV_WIDTH]`
    projection: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    z: Vec<f32>,
    norm: Vec<f32>,
    slot: Vec<u8>,
    ef: Vec<u8>,
    dense: Vec<f32>,
}

impl Fixture {
    fn new(full_rows: usize) -> Self {
        let mut rng = Rng::new(SEED_BASE | 0x201);
        let projection: Vec<f32> = (0..full_rows * QKV_WIDTH)
            .map(|_| bf16::from_f32(gauss(&mut rng)).to_f32())
            .collect();
        let gate = lcg_vec(SEED_BASE | 0x202, full_rows * VALUE_HEADS, -0.5, -0.001);
        let beta = lcg_vec(SEED_BASE | 0x203, full_rows * VALUE_HEADS, 0.05, 0.99);
        let mut zr = Rng::new(SEED_BASE | 0x204);
        let z: Vec<f32> = (0..full_rows * VALUE_WIDTH)
            .map(|_| gauss(&mut zr))
            .collect();
        let norm: Vec<f32> = lcg_vec(SEED_BASE | 0x205, DIM, 0.8, 1.2)
            .iter()
            .map(|x| bf16::from_f32(*x).to_f32())
            .collect();
        // Q8Ef slot: [codes: VH*128*128 i8][scales: VH*128 f32]; EF: F16 same index as codes.
        let codes: Vec<u8> = lcg_vec(SEED_BASE | 0x206, STATE_ELEMS, -64.0, 64.0)
            .iter()
            .map(|x| (*x as i8) as u8)
            .collect();
        let scales = lcg_vec(SEED_BASE | 0x207, VALUE_HEADS * DIM, 0.005, 0.02);
        let mut slot = codes;
        slot.extend(f32_bytes(&scales));
        let ef: Vec<u8> = lcg_vec(SEED_BASE | 0x208, STATE_ELEMS, -0.004, 0.004)
            .iter()
            .flat_map(|x| canon::q8ef::f32_to_f16_bits(*x).to_le_bytes())
            .collect();
        let dense = lcg_vec(SEED_BASE | 0x209, STATE_ELEMS, -0.3, 0.3);
        Self {
            full_rows,
            projection,
            gate,
            beta,
            z,
            norm,
            slot,
            ef,
            dense,
        }
    }
}

struct CallResult {
    out_valid: Vec<u8>,
    out_pad_untouched: bool,
    state: Vec<u8>,
    residual: Vec<u8>,
}

/// One WY call over the first `rows` rows of `fx` (optionally with row 0's first
/// value head sign-flipped); operand rows and the output past `rows` hold `pad`.
fn run_wy(
    gpu: &mut Gpu,
    fx: &Fixture,
    rows: usize,
    pad: Pad,
    format: GdnStateFormat,
    flip_row0: bool,
) -> Result<CallResult, String> {
    let cap = fx.full_rows;
    let e = |what: &str, err: &dyn std::fmt::Display| format!("{what}: {err}");
    let mut proj = fx.projection[..rows * QKV_WIDTH].to_vec();
    if flip_row0 {
        for c in 0..DIM {
            proj[2 * KEY_HEADS * DIM + c] = -proj[2 * KEY_HEADS * DIM + c];
        }
    }
    let mut proj_bytes = bf16_bytes(&proj);
    proj_bytes.resize(cap * QKV_WIDTH * 2, pad.byte());
    let mut projection = gpu
        .upload_raw(&proj_bytes, &[cap * QKV_WIDTH])
        .map_err(|x| e("projection", &x))?;
    projection.dtype = DType::BF16;
    let mut nbytes = bf16_bytes(&fx.norm);
    nbytes.truncate(DIM * 2);
    let mut norm = gpu.upload_raw(&nbytes, &[DIM]).map_err(|x| e("norm", &x))?;
    norm.dtype = DType::BF16;
    let padded = |v: &[f32], per_row: usize| -> Vec<f32> {
        let mut o = v[..rows * per_row].to_vec();
        o.resize(cap * per_row, pad.f32());
        o
    };
    let gate = gpu
        .upload_f32(&padded(&fx.gate, VALUE_HEADS), &[cap * VALUE_HEADS])
        .map_err(|x| e("gate", &x))?;
    let beta = gpu
        .upload_f32(&padded(&fx.beta, VALUE_HEADS), &[cap * VALUE_HEADS])
        .map_err(|x| e("beta", &x))?;
    let z = gpu
        .upload_f32(&padded(&fx.z, VALUE_WIDTH), &[cap * VALUE_WIDTH])
        .map_err(|x| e("z", &x))?;
    let output = gpu
        .upload_f32(&vec![pad.f32(); cap * VALUE_WIDTH], &[cap * VALUE_WIDTH])
        .map_err(|x| e("output", &x))?;
    let (state, residual) = match format {
        GdnStateFormat::Q8Ef => {
            let units = format.state_units(VALUE_HEADS, DIM, DIM);
            let s = gpu
                .zeros(&[units], DType::Raw)
                .map_err(|x| e("state", &x))?;
            gpu.memcpy_htod_auto(&s.buf, &fx.slot)
                .map_err(|x| e("state upload", &x))?;
            let r = gpu
                .zeros(&[format.residual_units(VALUE_HEADS, DIM, DIM)], DType::F16)
                .map_err(|x| e("residual", &x))?;
            gpu.memcpy_htod_auto(&r.buf, &fx.ef)
                .map_err(|x| e("residual upload", &x))?;
            (s, Some(r))
        }
        GdnStateFormat::F32 => (
            gpu.upload_f32(&fx.dense, &[STATE_ELEMS])
                .map_err(|x| e("state", &x))?,
            None,
        ),
        GdnStateFormat::Q8 => return Err("plain Q8 not exercised".into()),
    };
    let step = GatedDeltaStepBatched {
        projection: &projection,
        gate: &gate,
        beta: &beta,
        state: &state,
        format,
        residual: residual.as_ref(),
        output: &output,
        row_states: None,
        rows,
        qkv_width: QKV_WIDTH,
        key_heads: KEY_HEADS,
        value_heads: VALUE_HEADS,
        key_dim: DIM,
        value_dim: DIM,
        position: 0,
    };
    if !gated_delta_chunk_route(gpu, &step) {
        return Err("ROUTE: the chunked WY route is not taken on this device/config".into());
    }
    let gate_args = GatedDeltaGateBatched {
        recurrent_output: &output,
        z: &z,
        norm: &norm,
        output: &output,
        rows,
        value_heads: VALUE_HEADS,
        value_dim: DIM,
    };
    gated_delta_step_gate_wmma(gpu, &step, &gate_args)
        .map_err(|x| e("gated_delta_step_gate_wmma", &x))?;
    gpu.hip.device_synchronize().map_err(|x| e("sync", &x))?;
    let out = gpu
        .download_raw_bytes(&output)
        .map_err(|x| e("download", &x))?;
    let st = gpu
        .download_raw_bytes(&state)
        .map_err(|x| e("download state", &x))?;
    let rs = match &residual {
        Some(r) => gpu
            .download_raw_bytes(r)
            .map_err(|x| e("download residual", &x))?,
        None => vec![],
    };
    let ob = VALUE_WIDTH * 4;
    let res = CallResult {
        out_valid: out[..rows * ob].to_vec(),
        out_pad_untouched: out[rows * ob..]
            .chunks_exact(4)
            .all(|c| c == pad.f32().to_le_bytes()),
        state: st,
        residual: rs,
    };
    for t in [projection, norm, gate, beta, z, output, state] {
        gpu.free_tensor(t).map_err(|x| e("free", &x))?;
    }
    if let Some(r) = residual {
        gpu.free_tensor(r).map_err(|x| e("free", &x))?;
    }
    Ok(res)
}

fn ord_f32(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(4)
        .map(|c| {
            let x = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if x & 0x8000_0000 != 0 {
                -((x & 0x7fff_ffff) as i64)
            } else {
                x as i64
            }
        })
        .collect()
}

fn dec_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn row_diag(got: &[u8], want: &[u8]) -> Value {
    let (g, w) = (ord_f32(got), ord_f32(want));
    let rows = got.len() / (VALUE_WIDTH * 4);
    let mut rows_differing = Vec::new();
    let (mut elems, mut max_ulp) = (0usize, 0i64);
    for (ri, (gr, wr)) in g.chunks(VALUE_WIDTH).zip(w.chunks(VALUE_WIDTH)).enumerate() {
        let d = gr.iter().zip(wr).filter(|(a, b)| a != b).count();
        if d > 0 {
            rows_differing.push(ri);
            elems += d;
            max_ulp = max_ulp.max(
                gr.iter()
                    .zip(wr)
                    .map(|(a, b)| (a - b).abs())
                    .max()
                    .unwrap_or(0),
            );
        }
    }
    json!({
        "rows_differing": rows_differing.len(),
        "differing_rows_first8": rows_differing.iter().take(8).collect::<Vec<_>>(),
        "only_last_valid_row_differs": rows_differing.len() == 1 && rows_differing[0] + 1 == rows,
        "elements_differing": elems,
        "max_ulp_distance": max_ulp,
    })
}

fn run() -> R<i32> {
    let args = Args::from_env();
    let json_path = args.value("--json");
    let heads = args
        .csv_usize("--heads")?
        .unwrap_or_else(|| vec![0, 16, 512]);
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gdn_canon_wy_prefix_kill: GPU init failed: {e}");
            return Ok(EXIT_ENV);
        }
    };
    if !gpu.arch_caps.has_wmma_w32() {
        eprintln!(
            "gdn_canon_wy_prefix_kill: arch {} has no gfx11 wmma_w32 route",
            gpu.arch
        );
        return Ok(EXIT_ENV);
    }
    // Reach the WY route below the 512-row floor the way canonical Qwen4 prefill does.
    gpu.prefill_invariant_scope = true;

    let max_full = heads.iter().max().copied().unwrap_or(0) + MICRO;
    let fx_all = Fixture::new(max_full);
    let mut fmt_json: Vec<Value> = Vec::new();
    let mut failing: Vec<Value> = Vec::new();
    let mut controls_ok = true;
    let mut cells_total = 0usize;
    for (fmt_name, format) in [("q8ef", GdnStateFormat::Q8Ef), ("f32", GdnStateFormat::F32)] {
        let mut head_json: Vec<Value> = Vec::new();
        for &head in &heads {
            let full = head + MICRO;
            // Same data prefix for every head: truncate the fixture's capacity to this head.
            let fx = Fixture {
                full_rows: full,
                projection: fx_all.projection[..full * QKV_WIDTH].to_vec(),
                gate: fx_all.gate[..full * VALUE_HEADS].to_vec(),
                beta: fx_all.beta[..full * VALUE_HEADS].to_vec(),
                z: fx_all.z[..full * VALUE_WIDTH].to_vec(),
                norm: fx_all.norm.clone(),
                slot: fx_all.slot.clone(),
                ef: fx_all.ef.clone(),
                dense: fx_all.dense.clone(),
            };
            let call = |gpu: &mut Gpu, rows, pad, flip| run_wy(gpu, &fx, rows, pad, format, flip);
            let a = match call(&mut gpu, full, Pad::Zero, false) {
                Ok(a) => a,
                Err(e) if e.starts_with("ROUTE") => {
                    eprintln!("gdn_canon_wy_prefix_kill: {e}");
                    return Ok(EXIT_ENV);
                }
                Err(e) => return Err(e.into()),
            };
            let ctl = call(&mut gpu, full, Pad::Zero, true)?;
            let control_detected = ctl.out_valid != a.out_valid;
            controls_ok &= control_detected;
            // Non-vacuity: the complete-tile output must be finite and nonzero.
            let a_nonfinite = dec_f32(&a.out_valid)
                .iter()
                .filter(|v| !v.is_finite())
                .count();
            let a_nonzero = a.out_valid.iter().any(|b| *b != 0);
            controls_ok &= a_nonfinite == 0 && a_nonzero;

            let mut per_r: Vec<Value> = Vec::new();
            let mut zero_outs: Vec<Vec<u8>> = Vec::new();
            for r in 1..MICRO {
                let rows = head + r;
                let mut results = Vec::new();
                for pad in [Pad::Zero, Pad::Nan, Pad::Huge] {
                    results.push((pad, call(&mut gpu, rows, pad, false)?));
                }
                let again = call(&mut gpu, rows, Pad::Zero, false)?;
                let a_prefix = &a.out_valid[..rows * VALUE_WIDTH * 4];
                let a_dec = dec_f32(a_prefix);
                let mut pads_json = Vec::new();
                let mut cell_pass = true;
                for (pad, b) in &results {
                    let pd = first_diff_bytes(&b.out_valid, a_prefix);
                    let (maxabs, nonfinite) = maxabs_diff(&dec_f32(&b.out_valid), &a_dec);
                    let same_as_zero = b.out_valid == results[0].1.out_valid;
                    let ok = pd.is_none() && same_as_zero && b.out_pad_untouched;
                    cell_pass &= ok;
                    pads_json.push(json!({
                        "pad": pad.name(),
                        "prefix_identical_to_complete_tile": pd.is_none(),
                        "first_diff_row": pd.map(|i| i / (VALUE_WIDTH * 4)),
                        "maxabs": maxabs,
                        "nonfinite_mismatch": nonfinite,
                        "identical_to_zero_padding": same_as_zero,
                        "padded_out_rows_untouched": b.out_pad_untouched,
                    }));
                }
                let diag = row_diag(&results[0].1.out_valid, a_prefix);
                zero_outs.push(results[0].1.out_valid.clone());
                let st = &results[0].1;
                let state_info = json!({
                    "state_bytes_differing_vs_complete_tile_end": count_diff(&st.state, &a.state),
                    "residual_bytes_differing_vs_complete_tile_end": count_diff(&st.residual, &a.residual),
                });
                cells_total += 1;
                if !cell_pass {
                    let worst = pads_json
                        .iter()
                        .map(|p| p["maxabs"].as_f64().unwrap_or(0.0))
                        .fold(0.0f64, f64::max);
                    failing.push(json!({"state": fmt_name, "head": head, "r": r, "worst_maxabs": worst, "pads": pads_json.clone()}));
                }
                per_r.push(json!({
                    "r": r, "pass": cell_pass, "pads": pads_json,
                    "diff_vs_complete_tile": diag,
                    "repeat_identical": again.out_valid == results[0].1.out_valid,
                    "partial_end_state_info_only": state_info,
                }));
            }
            for i in 0..per_r.len() - 1 {
                let r = i + 1;
                per_r[i]["adjacent_extension_prefix_identical"] =
                    json!(zero_outs[i][..] == zero_outs[i + 1][..(head + r) * VALUE_WIDTH * 4]);
            }
            let n_pass = per_r
                .iter()
                .filter(|c| c["pass"].as_bool() == Some(true))
                .count();
            head_json.push(json!({
                "completed_rows_before_tile": head,
                "complete_tile_rows": full,
                "sign_flip_control_detected": control_detected,
                "complete_tile_nonfinite": a_nonfinite,
                "complete_tile_nonzero": a_nonzero,
                "r_pass": n_pass,
                "r_total": MICRO - 1,
                "per_r": per_r,
            }));
        }
        fmt_json.push(json!({"incoming_state": fmt_name, "heads": head_json}));
    }

    let all_pass = failing.is_empty() && controls_ok;
    let verdict = if all_pass { "pass" } else { "fail" };
    let receipt = envelope(
        "gdn_canon_wy_prefix_kill",
        &gpu.arch,
        verdict,
        json!({
            "kill": 2,
            "family": format!("C16 WY chunk kernel (gated_delta_step_gate_wmma) on {}", gpu.arch),
            "route_admission": "Gpu::prefill_invariant_scope = true (route_rows = canonical 8192); actual extents are the real row counts",
            "geometry": {"key_heads": KEY_HEADS, "value_heads": VALUE_HEADS, "dim": DIM, "micro_rows": MICRO},
            "output": "gated RMSNorm output, F32 store, [rows][6144]",
            "cells_total": cells_total,
            "cells_failing": failing.len(),
            "sign_flip_controls_all_detected": controls_ok,
            "failing_cells": failing,
            "states": fmt_json,
        }),
    );
    write_receipt(&json_path, &receipt)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "oracle": "gdn_canon_wy_prefix_kill",
            "arch": gpu.arch,
            "verdict": verdict,
            "cells": cells_total,
            "failing": failing.len(),
            "controls_ok": controls_ok,
        }))?
    );
    Ok(if all_pass { EXIT_PASS } else { EXIT_KILL_FAILS })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("gdn_canon_wy_prefix_kill: {e}");
            std::process::exit(EXIT_ENV);
        }
    }
}
