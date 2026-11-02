// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare one tensor's decoded f32 values between two sources (a dir and an
//! `.hfq` built from it). This is the dequant oracle: for a compressed `.hfq`
//! format the per-weight relative error must sit at that format's bit-width
//! level (a few percent for MQ4, well under a percent for Q8), not at O(1) —
//! an O(1) error means a broken dequant arm (wrong FWHT inverse, scale/zero
//! mix-up), which is otherwise invisible because the encoder still runs.
//!
//! Usage: cargo run --release -p hipfire-arch-qwen35-embed \
//!          --example compare_tensor --features lab -- <a> <b> <tensor> [vision]

use hipfire_arch_qwen35_embed::names;
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::model_source::ModelSource as _;
use hipfire_runtime::weight_backend::dequantize_weight_to_f32;

fn decode(src: &ModelSource, name: &str, vision: bool) -> Result<Vec<f32>, String> {
    let t = names::read(src, name, vision)?;
    let n = t.numel();
    if t.quant_type != 0xFF {
        return Ok(dequantize_weight_to_f32(t.quant_type, &t.bytes, n));
    }
    Ok(match t.dtype.as_str() {
        "F32" => t.bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" => t.bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => return Err(format!("unsupported dtype {other}")),
    })
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let (pa, pb, name) = (
        a.get(1)
            .ok_or("usage: compare_tensor <a> <b> <tensor> [vision]")?,
        a.get(2)
            .ok_or("usage: compare_tensor <a> <b> <tensor> [vision]")?,
        a.get(3)
            .ok_or("usage: compare_tensor <a> <b> <tensor> [vision]")?,
    );
    let vision = a.get(4).map(|v| v == "vision").unwrap_or(false);

    let sa = ModelSource::from_path(pa)?;
    let sb = ModelSource::from_path(pb)?;
    let ta = names::read(&sa, name, vision)?;
    let tb = names::read(&sb, name, vision)?;
    println!(
        "a: {} qt={} shape={:?}",
        sa.describe(),
        ta.quant_type,
        ta.shape
    );
    println!(
        "b: {} qt={} shape={:?}",
        sb.describe(),
        tb.quant_type,
        tb.shape
    );
    drop(ta);
    drop(tb);
    let va = decode(&sa, name, vision)?;
    let vb = decode(&sb, name, vision)?;
    if va.len() != vb.len() {
        return Err(format!("length mismatch {} vs {}", va.len(), vb.len()));
    }
    let (mut num, mut den, mut mx, mut nz) = (0f64, 0f64, 0f32, 0usize);
    for (x, y) in va.iter().zip(vb.iter()) {
        let d = (x - y).abs();
        num += (d as f64) * (d as f64);
        den += (*x as f64) * (*x as f64);
        mx = mx.max(d);
        if *x != 0.0 {
            nz += 1;
        }
    }
    let rel = (num / den.max(1e-30)).sqrt();
    println!(
        "elements={} nonzero={} rel_l2={rel:.4e} max_abs={mx:.4e}",
        va.len(),
        nz
    );
    // A compressed tier must stay at its bit-width error budget; a broken
    // dequant arm lands at O(1).
    if rel > 0.25 {
        return Err(format!(
            "rel_l2 {rel:.3e} is O(1) — dequant arm looks broken"
        ));
    }
    println!("dequant OK");
    Ok(())
}
