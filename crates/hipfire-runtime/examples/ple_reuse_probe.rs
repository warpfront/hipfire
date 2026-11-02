// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only PLE sidecar reuse probe. Reports the actual existing binding
//! verdict (`HfqFile::attach_ple_rows`, an accepted/rejected result, never a
//! crash) and raw-byte identity of sidecars; never changes binding policy.
//!
//! cargo run --release -p hipfire-runtime --example ple_reuse_probe -- \
//!     --base BASE.mq3-xts --sidecar S.ple-q4 [--compare-sidecar OTHER.ple-q4]
//!
//! Row-semantic equality (same decoded rows, different order) is NOT byte
//! identity: `raw_data_identical` compares only raw bytes, and the data region
//! of a rank-ordered table is never equal to a row-ordered one by construction.
//! Remote (HF) identity is not claimed; only local paths/bytes are hashed.

use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::ple_sidecar::{
    merkle_sha256, ple_fingerprint_from_metadata, ple_shard_name, PleRowsFile, PLE_SHARD_PREFIX,
    PLE_SHARD_SUFFIX,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// One streaming pass: whole-file sha256 and sha256 of each `[start, start+len)`
/// range. Fixed 8 MiB buffer; nothing file-sized is allocated.
fn hash_file(path: &Path, ranges: &[(u64, u64)]) -> Result<(u64, String, Vec<String>), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf = vec![0u8; 8 << 20];
    let (mut whole, mut parts) = (Sha256::new(), vec![Sha256::new(); ranges.len()]);
    let mut pos = 0u64;
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        whole.update(&buf[..n]);
        for (h, &(start, len)) in parts.iter_mut().zip(ranges) {
            let lo = start.max(pos);
            let hi = (start + len).min(pos + n as u64);
            if lo < hi {
                h.update(&buf[(lo - pos) as usize..(hi - pos) as usize]);
            }
        }
        pos += n as u64;
    }
    Ok((
        pos,
        hex(&whole.finalize()),
        parts.into_iter().map(|h| hex(&h.finalize())).collect(),
    ))
}

fn canon(p: &Path) -> String {
    std::fs::canonicalize(p)
        .unwrap_or_else(|_| p.to_path_buf())
        .display()
        .to_string()
}

/// Header summary plus streamed raw hashes of one sidecar.
fn describe(path: &Path, base_len: u64, base_index: &str) -> Result<Value, String> {
    let rows = PleRowsFile::open(path)?;
    let (h, t) = (rows.header(), rows.table());
    let (size, whole, parts) = hash_file(
        path,
        &[(h.data_offset, h.data_bytes), (h.rank_offset, h.rank_bytes)],
    )?;
    Ok(json!({
        "path": canon(path),
        "file_bytes": size,
        "whole_file_sha256": whole,
        "raw_data_region_sha256": parts[0],
        "raw_rank_region_sha256": if h.rank_bytes > 0 { json!(parts[1]) } else { Value::Null },
        "header": {
            "version": h.version, "order": format!("{:?}", h.order),
            "encoding": h.row_encoding.dtype_name(),
            "source_encoding": t.source_encoding.dtype_name(),
            "padded_rows": h.padded_rows, "rows_per_shard": h.rows_per_shard,
            "row_width": h.row_width, "encoded_row_bytes": h.encoded_row_bytes,
            "ple_fingerprint": h.ple_fingerprint, "generator": h.generator,
            "rank_offset": h.rank_offset, "rank_bytes": h.rank_bytes,
            "data_offset": h.data_offset, "data_bytes": h.data_bytes,
            "base_file_bytes": t.base_file_bytes, "base_index_sha256": t.base_index_sha256,
            "source_rows_sha256": t.source_rows_sha256, "header_data_sha256": t.data_sha256,
            "segment_bytes": t.segment_bytes,
        },
        "base_header_binding": {
            "base_file_bytes_match": t.base_file_bytes == base_len,
            "base_index_sha256_match": t.base_index_sha256 == base_index,
        },
    }))
}

fn arg(args: &[String], flag: &str) -> Option<PathBuf> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
}

/// ple_convert's algorithm: per-shard sha256 of the raw shard bytes, then
/// sha256(concat(digests)). Bounded 8 MiB pread buffer.
fn shard_merkle(path: &Path, shards: &[(u64, u64)]) -> Result<String, String> {
    let f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf = vec![0u8; 8 << 20];
    let mut digests: Vec<[u8; 32]> = Vec::with_capacity(shards.len());
    for &(off, len) in shards {
        let mut h = Sha256::new();
        let mut done = 0u64;
        while done < len {
            let n = (buf.len() as u64).min(len - done) as usize;
            f.read_exact_at(&mut buf[..n], off + done)
                .map_err(|e| format!("read {} @{}: {e}", path.display(), off + done))?;
            h.update(&buf[..n]);
            done += n as u64;
        }
        digests.push(h.finalize().into());
    }
    Ok(hex(&merkle_sha256(&digests)))
}

/// Hash the base's actual embedded PLE shard tensors (canonical shard_0..n order,
/// as ple_convert's `find_shards`) and compare with the sidecar's recorded
/// `source_rows_sha256`, geometry and PLE fingerprint (attach does not check
/// the fingerprint). A base without embedded shards is reported, not failed.
fn verify_embedded(base: &HfqFile, rows: &PleRowsFile) -> Result<Value, String> {
    let (h, t) = (rows.header(), rows.table());
    let mut found: Vec<(usize, &HfqTensorInfo)> = Vec::new();
    for info in base.tensor_infos() {
        let Some(num) = info
            .name
            .strip_prefix(PLE_SHARD_PREFIX)
            .and_then(|r| r.strip_suffix(PLE_SHARD_SUFFIX))
        else {
            continue;
        };
        let s: usize = num
            .parse()
            .map_err(|_| format!("bad shard name {}", info.name))?;
        if ple_shard_name(s) != info.name {
            return Err(format!("non-canonical shard name {}", info.name));
        }
        found.push((s, info));
    }
    found.sort_by_key(|&(s, _)| s);
    if found.iter().enumerate().any(|(i, &(s, _))| i != s) {
        return Err("embedded PLE shards are not dense shard_0..n".into());
    }
    let fingerprint = ple_fingerprint_from_metadata(&base.metadata_json);
    let fp_match = fingerprint.as_ref().ok().map(|f| *f == h.ple_fingerprint);
    if found.is_empty() {
        return Ok(json!({
            "path": canon(base.path()), "embedded_shards": 0, "marker": base.ple_rows_marker(),
            "base_fingerprint": fingerprint.ok(), "fingerprint_match": fp_match,
            "note": "no embedded shards; a marker's q8_rows_sha256 is a header claim only",
        }));
    }
    let (rps, width) = (h.rows_per_shard, h.row_width);
    let want = rps * width / 32 * 34;
    let shape_ok = found.iter().all(|(_, i)| {
        i.quant_type == 3 && i.shape == [rps as u32, width as u32] && i.data_size as u64 == want
    });
    let ranges: Vec<(u64, u64)> = found
        .iter()
        .map(|(_, i)| (i.data_offset as u64, i.data_size as u64))
        .collect();
    let actual = shard_merkle(base.path(), &ranges)?;
    Ok(json!({
        "path": canon(base.path()), "embedded_shards": found.len(),
        "algorithm": "sha256(concat(sha256(raw q8 shard bytes), shard 0..n))",
        "all_shards_q8_0_expected_shape_and_bytes": shape_ok,
        "padded_rows_match": found.len() as u64 * rps == h.padded_rows,
        "actual_embedded_rows_sha256": actual,
        "sidecar_source_rows_sha256": t.source_rows_sha256,
        "source_rows_sha256_match": actual == t.source_rows_sha256,
        "base_fingerprint": fingerprint.ok(), "sidecar_fingerprint": h.ple_fingerprint,
        "fingerprint_match": fp_match,
    }))
}

fn run() -> Result<Value, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: ple_reuse_probe --base PATH --sidecar PATH [--compare-sidecar PATH] \
                 [--verify-embedded-rows] [--embedded-reference PATH]";
    let base_path = arg(&args, "--base").ok_or(usage)?;
    let sidecar = arg(&args, "--sidecar").ok_or(usage)?;
    let compare = arg(&args, "--compare-sidecar");

    let mut base = HfqFile::open(&base_path).map_err(|e| format!("open base: {e}"))?;
    let base_len = base.file_len();
    let base_index = hex(&base
        .index_sha256()
        .map_err(|e| format!("base index sha256: {e}"))?);

    let sc = describe(&sidecar, base_len, &base_index)?;
    // Actual existing API; a rejection is the recorded result, not a failure.
    let attach = match base.attach_ple_rows(PleRowsFile::open(&sidecar)?) {
        Ok(()) => json!({"accepted": true, "error": Value::Null}),
        Err(e) => json!({"accepted": false, "error": e}),
    };
    base.drop_mmap();

    // Actual embedded Q8 shard bytes of --base (or --embedded-reference), hashed with
    // ple_convert's algorithm and compared with the sidecar's recorded source_rows_sha256.
    let reference = arg(&args, "--embedded-reference");
    let verify = if args.iter().any(|a| a == "--verify-embedded-rows") || reference.is_some() {
        let rows = PleRowsFile::open(&sidecar)?;
        let mut v = match &reference {
            Some(p) => {
                let mut r = HfqFile::open(p).map_err(|e| format!("open reference: {e}"))?;
                let v = verify_embedded(&r, &rows)?;
                r.drop_mmap();
                v
            }
            None => verify_embedded(&base, &rows)?,
        };
        let (h, t) = (rows.header(), rows.table());
        if h.rank_bytes == 0 && h.row_encoding == t.source_encoding {
            // Q8 row-order sidecar: its own data region, per shard, same algorithm.
            let len = h.rows_per_shard * h.encoded_row_bytes;
            let ranges: Vec<_> = (0..h.padded_rows / h.rows_per_shard)
                .map(|s| (h.data_offset + s * len, len))
                .collect();
            let m = shard_merkle(&sidecar, &ranges)?;
            v["sidecar_data_shard_merkle"] = json!(m);
            v["sidecar_data_shard_merkle_matches_header_claim"] = json!(m == t.source_rows_sha256);
            v["sidecar_data_shard_merkle_matches_embedded"] =
                json!(v["actual_embedded_rows_sha256"] == json!(m));
        }
        Some(v)
    } else {
        None
    };

    let cmp = match &compare {
        Some(p) => {
            let c = describe(p, base_len, &base_index)?;
            let same = |k: &str| sc[k] == c[k];
            let hd = |k: &str| sc["header"][k] == c["header"][k];
            Some(json!({
                "whole_file_bytes_identical": same("file_bytes") && same("whole_file_sha256"),
                // Raw bytes only; semantic row equality is not byte equality.
                "raw_data_identical": hd("data_bytes") && same("raw_data_region_sha256"),
                "same_order": hd("order"), "same_encoding": hd("encoding"),
                "same_source_rows_sha256": hd("source_rows_sha256"),
                "same_base_index_sha256": hd("base_index_sha256"),
                "same_base_file_bytes": hd("base_file_bytes"),
                "other": c,
            }))
        }
        None => None,
    };
    Ok(json!({
        "command": std::env::args().collect::<Vec<_>>().join(" "),
        "base": {"path": canon(&base_path), "file_bytes": base_len, "index_sha256": base_index},
        "sidecar": sc,
        "attach_ple_rows": attach,
        "verify_embedded_rows": verify,
        "compare": cmp,
        "note": "local bytes only; no remote/HF identity claimed",
    }))
}

fn main() {
    match run() {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
        Err(e) => {
            eprintln!("ple_reuse_probe: {e}");
            std::process::exit(1);
        }
    }
}
