// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Live cross-check of `hipfire_cpu`'s transcription of the canonical weight
//! decoder, over the **real** tensors of the on-disk fixtures.
//!
//! `hipfire-cpu` is a dependency leaf (its tests run in the GPU-free gate with
//! no GPU stack linked) so it cannot call
//! `hipfire_runtime::weight_backend::dequantize_to_f32` — it reimplements the
//! byte layouts instead. This test is what holds the two copies together: for
//! every tensor of every fixture whose `quant_type` maps to a
//! [`CpuQuant`], `hipfire_cpu::quant::dequant_group` must reproduce the
//! canonical decoder **bit for bit**. A sign or normalization drift in the FWHT
//! un-rotation is the "token soup" attractor failure mode, so equality here is
//! exact, not approximate.
//!
//! Not part of `scripts/no-gpu-ci.sh`'s curated list (it needs model files), and
//! it decodes real tensors, so run it optimized:
//!
//! ```text
//! cargo test --release -p hipfire-runtime --test cpu_quant_cross_check -- --nocapture
//! ```
//!
//! Fixtures are looked up in `$HIPFIRE_MODELS_DIR` (default `~/.hipfire/models`)
//! and a missing file is reported and skipped, like the `skip: no GPU` shape the
//! rest of the suite uses. A file that *is* present but yields no checkable
//! tensor is a failure: that would mean the check itself stopped exercising the
//! decode path.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Rows of each tensor checked. The group stride is fully exercised by one row
/// (all `k / group_elems` groups); the row stride by taking more than one, which
/// is why this is 4 and not 1. Bounded so the check stays fast on the 9B's
/// `[4096, 12288]` and `[12288, 4096]` tensors.
const CHECK_ROWS: usize = 4;

use hipfire_cpu::quant::{dequant_group, CpuQuant};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_backend::dequantize_weight_to_f32;

/// Registry fixtures, smallest first. The 2B variants are the per-format
/// fixtures (`mq3` = qt 17, `mq6` = qt 15, `hf6` = qt 8, `mq4` = qt 13).
const FIXTURES: [&str; 5] = [
    "qwen3.5-2b.mq4",
    "qwen3.5-2b.mq3",
    "qwen3.5-2b.mq6",
    "qwen3.5-2b.hf6",
    "qwen3.5-9b.mq4",
];

fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HIPFIRE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| PathBuf::from("/").into());
    PathBuf::from(home).join(".hipfire").join("models")
}

#[test]
fn dequant_group_matches_the_canonical_decoder_on_real_tensors() {
    let dir = models_dir();
    let mut checked = 0usize;
    let mut files_seen = 0usize;
    let mut formats: BTreeSet<u8> = BTreeSet::new();

    for name in FIXTURES {
        let path = dir.join(name);
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            continue;
        }
        files_seen += 1;
        let hfq = HfqFile::open(&path).expect("open fixture");
        for info in hfq.tensors() {
            let Some(q) = CpuQuant::from_quant_type(info.quant_type) else {
                continue;
            };
            let elems: usize = info.shape.iter().map(|d| *d as usize).product();
            if elems == 0 || elems % q.group_elems() != 0 {
                continue;
            }
            let (_, data) = hfq
                .tensor_data_vec(&info.name)
                .unwrap_or_else(|| panic!("{}: no bytes for {}", name, info.name));
            assert_eq!(
                data.len(),
                info.data_size,
                "{name}: {} data_size",
                info.name
            );

            // A 2-D tensor's byte count must be exactly `m` rows of the group
            // layout — the property `hipfire_cpu::gemv` derives its row pointer
            // from, checked here against the file rather than against itself.
            let m = if info.shape.len() == 2 {
                let rows = info.shape[0] as usize;
                assert_eq!(
                    data.len(),
                    rows * hipfire_cpu::gemv::row_bytes(q, elems / rows),
                    "{name}: {} byte count vs m * row_bytes",
                    info.name
                );
                rows
            } else {
                1
            };
            let k = elems / m;
            let (ge, gb) = (q.group_elems(), q.group_bytes());
            let rb = (k / ge) * gb;

            let rows = m.min(CHECK_ROWS);
            let mut got = vec![0.0f32; k];
            for row in 0..rows {
                let row_bytes = &data[row * rb..(row + 1) * rb];
                // Canonical decoder over the row …
                let want = dequantize_weight_to_f32(info.quant_type, row_bytes, k);
                assert_eq!(want.len(), k, "{name}: {} decode length", info.name);

                // … versus the transcription, group by group.
                for g in 0..k / ge {
                    dequant_group(q, &row_bytes[g * gb..], &mut got[g * ge..(g + 1) * ge]);
                }

                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{name}: {} (qt {}) row {row} element {i}: {a} (0x{:08x}) != {b} (0x{:08x})",
                        info.name,
                        info.quant_type,
                        a.to_bits(),
                        b.to_bits()
                    );
                }
            }
            checked += 1;
            formats.insert(info.quant_type);
        }
    }

    eprintln!(
        "cpu_quant_cross_check: {checked} tensors over {files_seen} fixtures, quant types {formats:?}"
    );
    if files_seen == 0 {
        eprintln!("skip: no fixture found in {}", dir.display());
        return;
    }
    assert!(
        checked > 0,
        "no tensor in {files_seen} fixture(s) mapped to a CpuQuant — the cross-check \
         stopped exercising the decode path"
    );
    // Every fixture on disk is MQ4/MQ3/MQ6/HF6/F16/Q8, so a run against real
    // files must cover at least the rotated 4-bit path.
    assert!(
        formats.contains(&13) || formats.contains(&17),
        "expected at least one FWHT-rotated format among the checked tensors, got {formats:?}"
    );
}
