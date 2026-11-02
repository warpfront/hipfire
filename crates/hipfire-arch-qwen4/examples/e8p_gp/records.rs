// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real solved-E8P layer records (`vq-e8p` arm, BF16-source solved, all 512 experts of a layer) -> device blobs.
//!
//! Source format (`vq_books.py::Book.payload`, `halves == 1`), one record per expert, concatenated in expert order:
//!   8-byte header (`b"e8p\0\0\0\0\0"`) | idx plane `m*k/8` bytes | signs plane `m*k/8` bytes (bits 0..6 = signs of lanes
//!   0..6, bit 7 = the E8P `-1/4` shift) | FP16 scales `m*k/128` (little endian).
//! Device blob (the PM E8P kernel ABI, planar): `m*k/8` u16 words `idx | signs << 8` (the K-E0 wire word of
//!   `e8p_i4_reference::pack_word`) then the same FP16 scales. gate_up `[1280, 2560]` = 870 400 B, down `[2560, 640]` =
//!   435 200 B per expert.
//!
//! Everything is pinned and hashed on the CPU before any device is touched: the manifest, both record files (whole-file
//! SHA-256 against the CLI pin AND the manifest), the manifest's per-expert offsets/sizes, and every record
//! (raw SHA-256) and repacked blob (SHA-256). The repack is verified by an independent inverse through the K-E0
//! `unpack_word_fields` and the writer guard is evaluated on every scale.

use std::path::Path;

use hipfire_runtime::e8p_i4_reference::{self as e8p, writer_guard, Reject};
use rayon::prelude::*;
use serde_json::{json, Value};

use super::common::*;

pub const HEADER: [u8; 8] = *b"e8p\0\0\0\0\0";
pub const EXPERTS: usize = 512;

pub struct RecordPins<'a> {
    pub layer: usize,
    pub manifest: &'a Path,
    pub manifest_sha256: &'a str,
    pub gate_up: &'a Path,
    pub gate_up_sha256: &'a str,
    pub down: &'a Path,
    pub down_sha256: &'a str,
}

/// One layer's E8P experts, ready to upload.
pub struct E8pLayer {
    pub layer: usize,
    /// `[expert][gate_up, down]` device blobs.
    pub blobs: Vec<[Vec<u8>; 2]>,
    /// Receipt block: manifest, files, census, guard, solver status (all computed before the GPU).
    pub inputs: Value,
    /// Per-expert record/blob hashes (written to `e8p_records.json`).
    pub per_expert: Value,
}

struct ExpertOut {
    blob: Vec<u8>,
    raw_sha: String,
    blob_sha: String,
    guard_below: u64,
    guard_nonfinite: u64,
    shift_words: u64,
    zero_scales: u64,
    d_min: u16,
    d_max: u16,
    idx_hist: Vec<u64>,
}

struct FileOut {
    experts: Vec<ExpertOut>,
    file_sha: String,
    file_bytes: u64,
}

fn load_file(
    proj: Proj,
    path: &Path,
    pinned: &str,
    entry: &Value,
    manifest_sha: &str,
) -> Result<FileOut> {
    let (m, k) = (proj.m(), proj.k());
    let want = proj.e8p_record_bytes();
    // manifest cross-checks
    let shape: Vec<u64> = entry["expert_shape"]
        .as_array()
        .ok_or("manifest: expert_shape")?
        .iter()
        .filter_map(|x| x.as_u64())
        .collect();
    if shape != [m as u64, k as u64] {
        return Err(format!(
            "{}: manifest expert_shape {shape:?} != [{m}, {k}]",
            proj.name()
        ));
    }
    let sizes: Vec<u64> = entry["sizes"]
        .as_array()
        .ok_or("manifest: sizes")?
        .iter()
        .filter_map(|x| x.as_u64())
        .collect();
    let offsets: Vec<u64> = entry["offsets"]
        .as_array()
        .ok_or("manifest: offsets")?
        .iter()
        .filter_map(|x| x.as_u64())
        .collect();
    if sizes.len() != EXPERTS
        || offsets.len() != EXPERTS
        || sizes.iter().any(|&s| s != want as u64)
        || offsets
            .iter()
            .enumerate()
            .any(|(i, &o)| o != (i * want) as u64)
    {
        return Err(format!(
            "{}: manifest offsets/sizes are not {EXPERTS} x {want} B back to back",
            proj.name()
        ));
    }
    let src: Vec<&str> = entry["src"]
        .as_array()
        .ok_or("manifest: src")?
        .iter()
        .filter_map(|x| x.as_str())
        .collect();
    if src.len() != EXPERTS || src.iter().any(|s| *s != "vq") {
        return Err(format!(
            "{}: manifest src is not all \"vq\" ({} entries, {} non-vq)",
            proj.name(),
            src.len(),
            src.iter().filter(|s| **s != "vq").count()
        ));
    }
    if entry["sha256"].as_str() != Some(pinned) {
        return Err(format!(
            "{}: manifest file sha256 {:?} != pinned {pinned}",
            proj.name(),
            entry["sha256"]
        ));
    }
    let total = (EXPERTS * want) as u64;
    if entry["bytes"].as_u64() != Some(total) {
        return Err(format!(
            "{}: manifest bytes {:?} != {total}",
            proj.name(),
            entry["bytes"]
        ));
    }
    let _ = manifest_sha;

    let raw = read_file(path)?;
    if raw.len() as u64 != total {
        return Err(format!(
            "{}: {} bytes, expected {total}",
            path.display(),
            raw.len()
        ));
    }
    let file_sha = sha(&raw);
    if file_sha != pinned {
        return Err(format!(
            "{}: sha256 {file_sha} != pinned {pinned}",
            path.display()
        ));
    }

    let nb = m * k / 8;
    let ns = m * k / 128;
    let experts: Vec<ExpertOut> = (0..EXPERTS)
        .into_par_iter()
        .map(|e| -> Result<ExpertOut> {
            let rec = &raw[e * want..(e + 1) * want];
            if rec[..8] != HEADER {
                return Err(format!("{} expert {e}: record header {:?} != {:?}", proj.name(), &rec[..8], HEADER));
            }
            let idx = &rec[8..8 + nb];
            let signs = &rec[8 + nb..8 + 2 * nb];
            let dbytes = &rec[8 + 2 * nb..];
            if dbytes.len() != ns * 2 {
                return Err(format!("{} expert {e}: scale plane {} B != {}", proj.name(), dbytes.len(), ns * 2));
            }
            let mut blob = Vec::with_capacity(proj.e8p_blob_bytes());
            for i in 0..nb {
                let w = u16::from(idx[i]) | (u16::from(signs[i]) << 8);
                blob.extend_from_slice(&w.to_le_bytes());
            }
            blob.extend_from_slice(dbytes);
            if blob.len() != proj.e8p_blob_bytes() {
                return Err(format!("{} expert {e}: blob {} B != {}", proj.name(), blob.len(), proj.e8p_blob_bytes()));
            }
            // Independent inverse through the K-E0 wire-word fields; every word must also be a legal pack_word input.
            let (mut shift_words, mut idx_hist) = (0u64, vec![0u64; 256]);
            for i in 0..nb {
                let w = u16::from_le_bytes([blob[2 * i], blob[2 * i + 1]]);
                let (ix, s7, sh) = e8p::unpack_word_fields(w);
                if ix != idx[i] || (s7 | (u8::from(sh) << 7)) != signs[i] || e8p::pack_word(ix, s7, sh) != w {
                    return Err(format!("{} expert {e} word {i}: repack inverse mismatch (idx {} signs {:#04x} -> word {w:#06x})", proj.name(), idx[i], signs[i]));
                }
                shift_words += u64::from(sh);
                idx_hist[ix as usize] += 1;
            }
            let (mut below, mut nonfinite, mut zero) = (0u64, 0u64, 0u64);
            let (mut d_min, mut d_max) = (u16::MAX, 0u16);
            for c in dbytes.chunks_exact(2) {
                let d = u16::from_le_bytes([c[0], c[1]]);
                match writer_guard(d) {
                    Ok(()) => {}
                    Err(Reject::NonFinite { .. }) => nonfinite += 1,
                    Err(Reject::BelowMin { .. }) => below += 1,
                }
                zero += u64::from(d & 0x7fff == 0);
                if d & 0x7c00 != 0x7c00 {
                    d_min = d_min.min(d);
                    d_max = d_max.max(d);
                }
            }
            Ok(ExpertOut { raw_sha: sha(rec), blob_sha: sha(&blob), blob, guard_below: below, guard_nonfinite: nonfinite, shift_words, zero_scales: zero, d_min, d_max, idx_hist })
        })
        .collect::<Result<_>>()?;
    Ok(FileOut {
        experts,
        file_sha,
        file_bytes: raw.len() as u64,
    })
}

/// Load, pin, hash and repack one layer's real E8P records. Pure CPU.
pub fn load_layer(p: &RecordPins) -> Result<E8pLayer> {
    let (mj, mraw) = read_json(p.manifest)?;
    let msha = sha(&mraw);
    if msha != p.manifest_sha256 {
        return Err(format!(
            "{}: sha256 {msha} != pinned {}",
            p.manifest.display(),
            p.manifest_sha256
        ));
    }
    if mj["arm"].as_str() != Some("vq-e8p") || mj["book"].as_str() != Some("e8p") {
        return Err(format!(
            "manifest arm/book {:?}/{:?} != vq-e8p/e8p",
            mj["arm"], mj["book"]
        ));
    }
    if mj["books_sha256"].as_str() != Some(e8p::E8P_BOOKS_SHA256) {
        return Err(format!(
            "manifest books_sha256 {:?} != K-E0 E8P_BOOKS_SHA256 {}",
            mj["books_sha256"],
            e8p::E8P_BOOKS_SHA256
        ));
    }
    if mj["guard"]
        .as_array()
        .map(|a| !a.is_empty())
        .unwrap_or(true)
    {
        return Err(format!(
            "manifest guard is not an empty list: {:?}",
            mj["guard"]
        ));
    }
    if mj["refits"].as_u64() != Some(0) {
        return Err(format!("manifest refits {:?} != 0", mj["refits"]));
    }
    let mut files = serde_json::Map::new();
    let mut outs: Vec<FileOut> = Vec::new();
    let mut census = serde_json::Map::new();
    for (proj, path, pin) in [
        (Proj::GateUp, p.gate_up, p.gate_up_sha256),
        (Proj::Down, p.down, p.down_sha256),
    ] {
        let key = format!("L{}.{}", p.layer, proj.record_suffix());
        let entry = &mj["files"][&key];
        if entry.is_null() {
            return Err(format!("manifest has no `{key}`"));
        }
        let fo = load_file(proj, path, pin, entry, &msha)?;
        files.insert(proj.name().into(), json!({"manifest_key": key, "path": path.display().to_string(), "bytes": fo.file_bytes, "sha256": fo.file_sha, "pinned_sha256": pin,
            "manifest_sha256": entry["sha256"], "manifest_source_path": entry["path"], "expert_record_bytes": proj.e8p_record_bytes(), "device_blob_bytes": proj.e8p_blob_bytes(),
            "experts": EXPERTS, "src_all_vq": true, "offsets_back_to_back_per_manifest": true}));
        let mut hist = vec![0u64; 256];
        for e in &fo.experts {
            for (h, x) in hist.iter_mut().zip(&e.idx_hist) {
                *h += x;
            }
        }
        let total_words: u64 = hist.iter().sum();
        census.insert(
            proj.name().into(),
            json!({
                "words": total_words, "shift_words": fo.experts.iter().map(|e| e.shift_words).sum::<u64>(),
                "idx_rows_used": hist.iter().filter(|&&c| c > 0).count(), "idx_row_min_count": hist.iter().min(), "idx_row_max_count": hist.iter().max(),
                "scales": fo.experts.len() * proj.m() * proj.groups(),
                "writer_guard_below_min": fo.experts.iter().map(|e| e.guard_below).sum::<u64>(),
                "writer_guard_nonfinite": fo.experts.iter().map(|e| e.guard_nonfinite).sum::<u64>(),
                "zero_scale_groups": fo.experts.iter().map(|e| e.zero_scales).sum::<u64>(),
                "finite_scale_bits_min": format!("{:#06x}", fo.experts.iter().map(|e| e.d_min).min().unwrap_or(0)),
                "finite_scale_bits_max": format!("{:#06x}", fo.experts.iter().map(|e| e.d_max).max().unwrap_or(0)),
            }),
        );
        outs.push(fo);
    }
    let guard_total: u64 = census
        .values()
        .map(|c| {
            c["writer_guard_below_min"].as_u64().unwrap_or(0)
                + c["writer_guard_nonfinite"].as_u64().unwrap_or(0)
        })
        .sum();
    let layer_key = p.layer.to_string();
    let stats = mj["layers"][&layer_key]["stats"].clone();
    let down = outs.pop().unwrap();
    let gate = outs.pop().unwrap();
    let mut per_expert = Vec::with_capacity(EXPERTS);
    let mut blobs = Vec::with_capacity(EXPERTS);
    for (e, (g, d)) in gate.experts.into_iter().zip(down.experts).enumerate() {
        per_expert.push(json!({"expert": e,
            "gate_up": {"record_sha256": g.raw_sha, "blob_sha256": g.blob_sha, "guard_rejects": g.guard_below + g.guard_nonfinite},
            "down": {"record_sha256": d.raw_sha, "blob_sha256": d.blob_sha, "guard_rejects": d.guard_below + d.guard_nonfinite}}));
        blobs.push([g.blob, d.blob]);
    }
    let inputs = json!({
        "layer": p.layer, "kind": "real solved E8P records (vq-e8p arm), all 512 experts; NOT re-encoded by this harness",
        "record_source": {"source_model": mj["source"], "source_note": "BF16-source solved 512-expert records (manifest `source` = pinned BF16 checkpoint path)", "pilot_sha256": mj["pilot_sha256"],
            "books_sha256": mj["books_sha256"], "code_sha256": mj["code_sha256"], "training_windows": mj["training_windows"].as_array().map(|a| a.len()), "license": mj["license"],
            "kappa": mj["kappa"], "solver_status": stats},
        "manifest": {"path": p.manifest.display().to_string(), "bytes": mraw.len(), "sha256": msha, "pinned_sha256": p.manifest_sha256, "guard": mj["guard"], "refits": mj["refits"]},
        "files": Value::Object(files), "census": Value::Object(census), "writer_guard_rejects_total": guard_total,
        "cpu_validation": {"header": "every record starts with b\"e8p\\0\\0\\0\\0\\0\"", "repack": "word = idx | signs << 8; inverse through e8p_i4_reference::unpack_word_fields/pack_word over every word equals the record planes",
            "all_hashes_before_gpu": true, "record_hashes_file": "e8p_records.json"},
    });
    Ok(E8pLayer {
        layer: p.layer,
        blobs,
        inputs,
        per_expert: Value::Array(per_expert),
    })
}
