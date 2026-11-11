// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S0 kill 2 (CHUNK-CANONICAL-DESIGN §3.4, §6 S0): **scan prefix stability** of
//! the gfx1201 C64 chunk scan (`gdn_chunk_prep` → KKT → `gdn_chunk_scan_*`, the
//! production route via the existing `rdna-compute` bindings).
//!
//! For every r = 1..63 and head ∈ {0, 64, 512} completed rows before the
//! microtile, with **nonzero incoming Q8 state + EF + conv history**:
//!
//! * `A` = a call over `head + 64` real rows (the microtile is complete);
//! * `B(pad)` = a call over `head + r` rows only, with every scratch region past
//!   the valid rows filled with zero / NaN / huge bytes.
//!
//! The published output rows `0..head+r` of `B(pad)` must be byte-identical to
//! the same rows of `A` (earlier rows must not depend on future rows or inactive
//! lanes), and `B(nan)`, `B(huge)` must be byte-identical to `B(zero)`. Padded
//! out rows must not be written. A sign-flip control on row 0 of `A` must change
//! the compared rows, so the comparisons cannot be blind. The partial end
//! *state* is allowed to differ (that is why the carry replays); it is reported,
//! not gated. If prefix identity fails, the bounded-replay design cannot make
//! consumed later-layer activations exact and promotion must stop (design §8).
//!
//! Only the C64 gfx1201 family is exercisable on hiptrx; the C16 WY family needs
//! gfx1100/gfx1151 hardware and is reported as `not_run`.
//!
//! Build: cargo build --release -p rdna-compute --features lab,deltanet --example gdn_canon_scan_prefix_kill
//! Run:   HIPFIRE_GRAPH=0 gdn_canon_scan_prefix_kill --json <receipt.json> [--hv 48|32] [--heads 0,64,512]
//! Exit: 0 prefix-stable | 1 prefix identity fails | 3 environment.

#[path = "gdn_canon_common/mod.rs"]
mod canon;

use canon::gpu::*;
use canon::stream::*;
use canon::util::*;
use rdna_compute::Gpu;
use serde_json::{json, Value};

/// Monotone integer ordering of a BF16 / F32 bit pattern (for ULP distances).
fn ord_bits(bytes: &[u8], fmt: rdna_compute::norm::GdnScanOut) -> Vec<i64> {
    match fmt {
        rdna_compute::norm::GdnScanOut::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let x = u16::from_le_bytes([c[0], c[1]]);
                if x & 0x8000 != 0 {
                    -((x & 0x7fff) as i64)
                } else {
                    x as i64
                }
            })
            .collect(),
        rdna_compute::norm::GdnScanOut::F32 => bytes
            .chunks_exact(4)
            .map(|c| {
                let x = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if x & 0x8000_0000 != 0 {
                    -((x & 0x7fff_ffff) as i64)
                } else {
                    x as i64
                }
            })
            .collect(),
    }
}

/// Which rows / elements differ between `got` and `want` (same length), with ULP distances.
fn row_diag(
    got: &[u8],
    want: &[u8],
    row_bytes: usize,
    fmt: rdna_compute::norm::GdnScanOut,
) -> Value {
    let (g, w) = (ord_bits(got, fmt), ord_bits(want, fmt));
    let rows = got.len() / row_bytes;
    let per_row = g.len() / rows;
    let mut rows_differing: Vec<usize> = Vec::new();
    let (mut elems, mut max_ulp) = (0usize, 0i64);
    for (ri, (gr, wr)) in g.chunks(per_row).zip(w.chunks(per_row)).enumerate() {
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
        "elements_per_row": per_row,
    })
}

fn run() -> R<i32> {
    let args = Args::from_env();
    let json_path = args.value("--json");
    let hv: usize = args
        .value("--hv")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(48);
    let heads = args
        .csv_usize("--heads")?
        .unwrap_or_else(|| vec![0, 64, 512]);
    if !matches!(hv, 48 | 32) {
        return Err("--hv must be 48 or 32".into());
    }
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gdn_canon_scan_prefix_kill: GPU init failed: {e}");
            return Ok(EXIT_ENV);
        }
    };
    if gpu.arch != "gfx1201" {
        eprintln!(
            "gdn_canon_scan_prefix_kill: needs gfx1201 (got {})",
            gpu.arch
        );
        return Ok(EXIT_ENV);
    }
    if hipfire_config::developer_var("HIPFIRE_GDN_SCAN_OUT_EMU").is_ok() {
        return Err("unset HIPFIRE_GDN_SCAN_OUT_EMU".into());
    }
    let out_fmt = scan_out_fmt(&gpu);
    let consts = GpuConsts::new(&mut gpu, hv)?;
    let st = DevState::new(&mut gpu, hv)?;
    let init = gen_gpu_initial(hv, 0);
    let g = gpu_geometry(hv);
    let ob = hv
        * HD
        * if out_fmt == rdna_compute::norm::GdnScanOut::Bf16 {
            2
        } else {
            4
        };

    let mut head_json: Vec<Value> = Vec::new();
    let mut failing: Vec<Value> = Vec::new();
    let mut control_ok_all = true;
    let mut total_cells = 0usize;
    for &head in &heads {
        let full_rows = head + 64;
        let data = gen_gpu_rows(hv, full_rows);

        st.upload_init(&gpu, &init)?;
        let a = run_call(&mut gpu, &consts, &st, &data, out_fmt, Pad::Zero)?;
        let a_state = st.read(&gpu, full_rows as u64)?;

        // Sign-flip control on row 0: the compared rows must change.
        let mut flipped = data.clone();
        for c in 0..HD {
            flipped.qkv[2 * HK * HD + c] = -flipped.qkv[2 * HK * HD + c]; // row 0, first value head
        }
        st.upload_init(&gpu, &init)?;
        let ctl = run_call(&mut gpu, &consts, &st, &flipped, out_fmt, Pad::Zero)?;
        let control_detected = ctl.out_valid != a.out_valid;
        control_ok_all &= control_detected;

        let mut per_r: Vec<Value> = Vec::new();
        let mut zero_pad_outs: Vec<Vec<u8>> = Vec::new();
        for r in 1..=63usize {
            let rows = head + r;
            let b_in = data.slice(&g, 0, rows);
            let mut results: Vec<(Pad, CallOut, SealedState)> = Vec::new();
            for pad in [Pad::Zero, Pad::Nan, Pad::Huge] {
                st.upload_init(&gpu, &init)?;
                let b = run_call(&mut gpu, &consts, &st, &b_in, out_fmt, pad)?;
                let s = st.read(&gpu, rows as u64)?;
                results.push((pad, b, s));
            }
            let a_prefix = &a.out_valid[..rows * ob];
            let a_dec = decode_out(a_prefix, out_fmt);
            let mut pads_json = Vec::new();
            let mut cell_pass = true;
            for (pad, b, _s) in &results {
                let prefix_diff = first_diff_bytes(&b.out_valid, a_prefix);
                let b_dec = decode_out(&b.out_valid, out_fmt);
                let (maxabs, nonfinite) = maxabs_diff(&b_dec, &a_dec);
                let first_row = prefix_diff.map(|i| i / ob);
                let identical_to_zero_pad = b.out_valid == results[0].1.out_valid;
                let ok = prefix_diff.is_none() && identical_to_zero_pad && b.out_pad_untouched;
                cell_pass &= ok;
                pads_json.push(json!({
                    "pad": pad.name(),
                    "prefix_identical_to_complete_tile": prefix_diff.is_none(),
                    "first_diff_row": first_row,
                    "first_diff_row_in_partial_tile": first_row.map(|r0| r0 as i64 - head as i64),
                    "maxabs": maxabs,
                    "nonfinite_mismatch": nonfinite,
                    "identical_to_zero_padding": identical_to_zero_pad,
                    "padded_out_rows_untouched": b.out_pad_untouched,
                    "route": {"kkt_batched": b.route.kkt_batched, "mseg": b.route.mseg, "segments": b.route.segments},
                }));
            }
            // Informational: the partial end state vs the complete tile's state is NOT gated.
            let partial_state = &results[0].2;
            let state_info = json!({
                "q8_codes_diff_vs_complete_tile_end": count_diff(&partial_state.codes, &a_state.codes),
                "ef_diff_vs_complete_tile_end": count_diff(&partial_state.ef, &a_state.ef),
            });
            total_cells += 1;
            if !cell_pass {
                let worst = pads_json
                    .iter()
                    .map(|p| p["maxabs"].as_f64().unwrap_or(0.0))
                    .fold(0.0f64, f64::max);
                failing.push(
                    json!({"head": head, "r": r, "worst_maxabs": worst, "pads": pads_json.clone()}),
                );
            }
            let diag = row_diag(&results[0].1.out_valid, a_prefix, ob, out_fmt);
            // Determinism control: the same call again must reproduce the same bytes.
            st.upload_init(&gpu, &init)?;
            let again = run_call(&mut gpu, &consts, &st, &b_in, out_fmt, Pad::Zero)?;
            let repeat_identical = again.out_valid == results[0].1.out_valid;
            zero_pad_outs.push(results[0].1.out_valid.clone());
            let b0 = &results[0].1;
            let prep_planes = json!({
                "q_prefix_identical": b0.q_valid[..] == a.q_valid[..rows * HK * HD * 2],
                "k_prefix_identical": b0.k_valid[..] == a.k_valid[..rows * HK * HD * 2],
                "v_prefix_identical": b0.v_valid[..] == a.v_valid[..rows * hv * HD * 2],
            });
            per_r.push(json!({"r": r, "pass": cell_pass, "pads": pads_json, "partial_end_state_info_only": state_info, "diff_vs_complete_tile": diag, "prep_planes_vs_complete_tile": prep_planes, "repeat_identical": repeat_identical}));
        }
        // Extension stability (diagnostic): does call(head+r) agree with the first head+r rows of
        // call(head+r+1)? Separates "last row of a call is special" from "tile length matters".
        for i in 0..per_r.len() - 1 {
            let r = i + 1;
            let ext_ok = zero_pad_outs[i][..] == zero_pad_outs[i + 1][..(head + r) * ob];
            per_r[i]["adjacent_extension_prefix_identical"] = json!(ext_ok);
        }
        let n_pass = per_r
            .iter()
            .filter(|c| c["pass"].as_bool() == Some(true))
            .count();
        head_json.push(json!({
            "completed_rows_before_tile": head,
            "complete_tile_rows": full_rows,
            "complete_tile_route": {"kkt_batched": a.route.kkt_batched, "mseg": a.route.mseg, "segments": a.route.segments},
            "sign_flip_control_detected": control_detected,
            "r_pass": n_pass,
            "r_total": 63,
            "per_r": per_r,
        }));
    }
    st.free(&mut gpu)?;
    consts.free(&mut gpu)?;

    let all_pass = failing.is_empty() && control_ok_all;
    let verdict = if all_pass { "pass" } else { "fail" };
    let receipt = envelope(
        "gdn_canon_scan_prefix_kill",
        &gpu.arch,
        verdict,
        json!({
            "kill": 2,
            "family": "C64 gfx1201 chunk scan (production prep -> KKT -> scan route)",
            "c16_wy_family": "not_run: needs gfx1100/gfx1151 hardware",
            "hv": hv,
            "scan_out": format!("{out_fmt:?}"),
            "incoming_state": "nonzero Q8 codes + scales + F16 EF + conv history (pinned lcg seeds 9..12)",
            "cells_total": total_cells,
            "cells_failing": failing.len(),
            "sign_flip_controls_all_detected": control_ok_all,
            "failing_cells": failing,
            "heads": head_json,
        }),
    );
    write_receipt(&json_path, &receipt)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "oracle": "gdn_canon_scan_prefix_kill",
            "verdict": verdict,
            "cells": total_cells,
            "failing": receipt["result"]["cells_failing"],
            "controls_ok": control_ok_all,
        }))?
    );
    Ok(if all_pass { EXIT_PASS } else { EXIT_KILL_FAILS })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("gdn_canon_scan_prefix_kill: {e}");
            std::process::exit(EXIT_ENV);
        }
    }
}
