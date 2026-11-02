// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Prompt scatter byte oracle, one arm per fresh process. Both arms compare
//! every destination byte (including untouched poison) and every source byte
//! against the same independent CPU mapping. The destination is the draft's
//! target_hidden state; no projection arithmetic is changed by this lever.
//! Run the ignored test twice with HIPFIRE_DFLASH_PREFILL_SCATTER_FAST=0/1
//! (pitched scatter) or HIPFIRE_DFLASH_PREFILL_DEVICE=0/1 (resident prefill),
//! comparing HIPFIRE_DFLASH_SCATTER_ORACLE_OUT artifacts. Piece capture also
//! exercises the production host-required policy and append helper: any
//! temporary host bytes are checked independently, the long-reach backfill
//! shadow (the only remaining host reader) is included in the cross-arm
//! artifact, and the resident arm must capture exactly when that reader
//! exists. Production has no persistent host shadow after prefill. No model
//! load is required.

use hipfire_arch_qwen35::dflash_spec::dflash_prefill_host_required;
use hipfire_arch_qwen35::speculative::{
    append_prefill_hidden_host, scatter_hidden_prefill_to_interleaved, HiddenStateRingBuffer,
};
use hipfire_runtime::dflash::DraftCtxMode;
use rdna_compute::{DType, Gpu};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "requires HIP GPU, admitted HIPFIRE_DEVICES and HIPFIRE_DFLASH_SCATTER_ORACLE_OUT"]
fn dflash_prefill_scatter_oracle_arm() {
    let out = std::env::var("HIPFIRE_DFLASH_SCATTER_ORACLE_OUT").expect("oracle output path");
    let mut gpu = Gpu::init().expect("GPU init");
    use hipfire_config::devices::{self, DeviceRoot, IdentitySource, ObservedDevice};
    assert_eq!(gpu.hip.device_count().unwrap(), 1);
    let source = DeviceRoot::system().identity_source();
    let observed = vec![ObservedDevice {
        logical: 0,
        arch: gpu.hip.get_arch(0).unwrap(),
        pci_bus_id: gpu.hip.device_pci_bus_id(0).unwrap(),
        uuid: (source == IdentitySource::Hip).then(|| gpu.hip.device_uuid(0).unwrap()),
    }];
    let active = devices::active_devices().expect("admitted device selection");
    active.verify_visible(&observed).expect("device verify");
    let device = &active.devices[0];
    eprintln!(
        "[devices] verified logical 0: HIP {} {} = index {} {} rocr={}",
        observed[0].arch,
        observed[0].pci_bus_id,
        device.index,
        device.lock_identity(),
        device.rocr_selector()
    );
    let mut records = Vec::new();
    // (ring, hidden, extracts, head, block, kept, destination offset, modulus)
    let cases = [
        (7, 17, 1, 0, 0, 0, 0, usize::MAX),
        (7, 17, 1, 1, 1, 1, 0, usize::MAX),
        (7, 17, 3, 1, 2, 2, 6, 7),
        (7, 17, 5, 3, 7, 7, 5, 11),
        (7, 17, 5, 2, 17, 17, 3, 7),
        (7, 17, 5, 6, 7, 4, 3, usize::MAX),
        (7, 17, 5, 6, 17, 5, 3, usize::MAX),
        (2048, 5120, 5, 383, 383, 383, 0, 2048),
        (2048, 5120, 5, 0, 2048, 2048, 0, 2048),
        (2048, 5120, 5, 1, 2049, 2049, 0, 2048),
        (2048, 5120, 5, 2032, 8176, 8176, 0, 2048),
        (2048, 5120, 5, 33, 2055, 2055, 8176, 2048),
        (2048, 5120, 5, 7, 513, 513, 383, usize::MAX),
    ];
    for (ring, hidden, ne, head, block, kept, offset, modulus) in cases {
        let mut rb = HiddenStateRingBuffer::new_for_layers(
            &mut gpu,
            &(0..ne).collect::<Vec<_>>(),
            hidden,
            ring,
            1,
        )
        .unwrap();
        rb.head = head;
        rb.written = block;
        let row_bytes = hidden * 4;
        let sources: Vec<Vec<u8>> = (0..ne)
            .map(|ext| {
                (0..ring * hidden)
                    .flat_map(|i| {
                        // Includes NaN payloads, signed zeros and all exponent ranges.
                        ((i as u32).wrapping_mul(0x9e3779b9)
                            ^ (ext as u32).wrapping_mul(0x85ebca6b))
                        .to_le_bytes()
                    })
                    .collect()
            })
            .collect();
        for (tensor, bytes) in rb.layer_bufs.iter().zip(&sources) {
            gpu.hip.memcpy_htod(&tensor.buf, bytes).unwrap();
        }
        let dst_rows = if modulus == usize::MAX {
            offset + kept + 2
        } else {
            modulus
        };
        let dst = gpu
            .alloc_tensor(&[dst_rows * ne * hidden], DType::F32)
            .unwrap();
        let mut expected = vec![0xa5; dst_rows * ne * row_bytes];
        gpu.hip.memcpy_htod(&dst.buf, &expected).unwrap();
        // Independent reference: start with the logical block's first row,
        // discard rows overwritten by the ring, then map absolute destinations.
        for r in 0..kept {
            if block - r > ring {
                continue;
            }
            let slot = (head + ring - ((block - r) % ring)) % ring;
            let dst_row = if modulus == usize::MAX {
                offset + r
            } else {
                (offset + r) % modulus
            };
            for ext in 0..ne {
                let d = (dst_row * ne + ext) * row_bytes;
                expected[d..d + row_bytes]
                    .copy_from_slice(&sources[ext][slot * row_bytes..(slot + 1) * row_bytes]);
            }
        }
        scatter_hidden_prefill_to_interleaved(&gpu, &rb, &dst, offset, block, kept, modulus)
            .unwrap();
        let mut actual = vec![0; expected.len()];
        gpu.hip.memcpy_dtoh(&mut actual, &dst.buf).unwrap();
        assert!(
            actual == expected,
            "draft target_hidden differs: ring={ring} head={head} block={block} offset={offset}"
        );
        assert_eq!(
            (rb.head, rb.written),
            (head, block),
            "ring metadata mutated"
        );
        for (tensor, bytes) in rb.layer_bufs.iter().zip(&sources) {
            let mut after = vec![0; bytes.len()];
            gpu.hip.memcpy_dtoh(&mut after, &tensor.buf).unwrap();
            assert!(after == *bytes, "source ring mutated");
        }
        let digest = format!("{:x}", Sha256::digest(&actual));
        println!("DFLASH_SCATTER_ROW ring={ring} hidden={hidden} extracts={ne} block={block} kept={kept} head={head} offset={offset} identical=true sha256={digest}");
        records.push(serde_json::json!({"ring":ring,"hidden":hidden,"extracts":ne,"head":head,"block":block,"kept":kept,"offset":offset,"modulus":modulus,"bytes":actual.len(),"sha256":digest}));
        gpu.free_tensor(dst).unwrap();
        rb.free_gpu(&mut gpu);
    }
    records.extend(prefill_capture_cases(&mut gpu));
    let count = records.len();
    std::fs::write(out, serde_json::to_vec_pretty(&serde_json::json!({
        "device": {"arch":observed[0].arch,"pci":observed[0].pci_bus_id,"uuid":device.lock_identity()},
        "cases":records,
    })).unwrap()).unwrap();
    println!("DFLASH_SCATTER_VERDICT identical=true cases={count}");
}

fn hidden_word(row: usize, ext: usize, col: usize, hidden: usize) -> [u8; 4] {
    ((row * hidden + col) as u32)
        .wrapping_mul(0x9e3779b9)
        .wrapping_add((ext as u32).wrapping_mul(0x85ebca6b))
        .to_le_bytes()
}

fn prefill_capture_cases(gpu: &mut Gpu) -> Vec<serde_json::Value> {
    // (case, ring, hidden, extracts, initial head, prompt rows, draft mode,
    // backfill reads the host shadow)
    let cases = [
        (
            "empty-window",
            7,
            17,
            5,
            0,
            0,
            DraftCtxMode::Windowed { w: 7, w_full: 7 },
            false,
        ),
        (
            "wrapped-window",
            7,
            17,
            3,
            2,
            23,
            DraftCtxMode::Windowed { w: 7, w_full: 7 },
            false,
        ),
        (
            "window-8176",
            2048,
            5120,
            5,
            37,
            8176,
            DraftCtxMode::Windowed {
                w: 2048,
                w_full: 2048,
            },
            false,
        ),
        (
            "long-reach",
            7,
            17,
            5,
            5,
            23,
            DraftCtxMode::Windowed { w: 7, w_full: 14 },
            true,
        ),
        (
            "long-reach-in-window",
            7,
            17,
            5,
            5,
            7,
            DraftCtxMode::Windowed { w: 7, w_full: 14 },
            false,
        ),
        ("legacy", 31, 17, 5, 5, 23, DraftCtxMode::Legacy, false),
    ];
    let resident = hipfire_config::developer_bool("HIPFIRE_DFLASH_PREFILL_DEVICE", true);
    let mut records = Vec::new();
    for (case, ring, hidden, ne, initial_head, total, mode, required) in cases {
        let capture = dflash_prefill_host_required(mode, total);
        assert_eq!(
            capture,
            required || !resident,
            "host capture policy: {case}"
        );
        let modulus = if matches!(mode, DraftCtxMode::Legacy) {
            usize::MAX
        } else {
            ring
        };
        let mut rb = HiddenStateRingBuffer::new_for_layers(
            gpu,
            &(0..ne).collect::<Vec<_>>(),
            hidden,
            ring,
            1,
        )
        .unwrap();
        rb.head = initial_head;
        let row_bytes = hidden * 4;
        let mut expected_ring = vec![vec![0xa5; ring * row_bytes]; ne];
        for (tensor, bytes) in rb.layer_bufs.iter().zip(&expected_ring) {
            gpu.hip.memcpy_htod(&tensor.buf, bytes).unwrap();
        }
        let source = gpu
            .alloc_tensor(&[ring.min(total).max(1) * hidden], DType::F32)
            .unwrap();
        let mut host = Vec::new();
        // No full-prompt CPU allocation on the resident arm. The control's
        // temporary bytes are verified before the production backfill reader.
        let mut expected_host = Vec::new();
        if capture {
            expected_host.reserve(total * ne * row_bytes);
        }
        let mut off = 0;
        while off < total {
            let n = ring.min(total - off);
            for ext in 0..ne {
                let mut bytes = Vec::with_capacity(n * row_bytes);
                for row in off..off + n {
                    for col in 0..hidden {
                        bytes.extend_from_slice(&hidden_word(row, ext, col, hidden));
                    }
                    let slot = (initial_head + row) % ring;
                    let src = (row - off) * row_bytes;
                    expected_ring[ext][slot * row_bytes..(slot + 1) * row_bytes]
                        .copy_from_slice(&bytes[src..src + row_bytes]);
                }
                gpu.hip.memcpy_htod(&source.buf, &bytes).unwrap();
                rb.write_chunk_rows(gpu, ext, &source, n).unwrap();
            }
            rb.finish_prefill_chunk(gpu, n).unwrap();
            append_prefill_hidden_host(gpu, &rb, n, capture.then_some(&mut host)).unwrap();
            if capture {
                for row in off..off + n {
                    for ext in 0..ne {
                        for col in 0..hidden {
                            expected_host.extend_from_slice(&hidden_word(row, ext, col, hidden));
                        }
                    }
                }
            }
            off += n;
        }
        assert_eq!(
            (rb.head, rb.written),
            ((initial_head + total) % ring, total)
        );
        let actual_host =
            unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4) };
        assert!(
            actual_host == expected_host,
            "temporary host bytes differ: {case}"
        );
        assert!(
            capture || host.is_empty(),
            "resident prefill created a host shadow"
        );
        let required_host_sha = required.then(|| format!("{:x}", Sha256::digest(actual_host)));
        println!("DFLASH_CAPTURE_ROW case={case} host_capture={capture} host_bytes={} host_identical=true",
            actual_host.len());
        let dst_rows = if modulus == usize::MAX {
            total + 2
        } else {
            ring
        };
        let target_hidden = gpu
            .alloc_tensor(&[dst_rows * ne * hidden], DType::F32)
            .unwrap();
        let mut expected = vec![0xa5; dst_rows * ne * row_bytes];
        gpu.hip.memcpy_htod(&target_hidden.buf, &expected).unwrap();
        for row in total.saturating_sub(ring)..total {
            let dst_row = if modulus == usize::MAX {
                row
            } else {
                row % ring
            };
            for ext in 0..ne {
                for col in 0..hidden {
                    let at = ((dst_row * ne + ext) * hidden + col) * 4;
                    expected[at..at + 4].copy_from_slice(&hidden_word(row, ext, col, hidden));
                }
            }
        }
        scatter_hidden_prefill_to_interleaved(gpu, &rb, &target_hidden, 0, total, total, modulus)
            .unwrap();
        let mut actual = vec![0; expected.len()];
        gpu.hip
            .memcpy_dtoh(&mut actual, &target_hidden.buf)
            .unwrap();
        assert!(
            actual == expected,
            "resident draft target_hidden differs: {case}"
        );
        let mut source_hash = Sha256::new();
        for (tensor, bytes) in rb.layer_bufs.iter().zip(&expected_ring) {
            let mut after = vec![0; bytes.len()];
            gpu.hip.memcpy_dtoh(&mut after, &tensor.buf).unwrap();
            assert!(after == *bytes, "piece capture mutated source ring: {case}");
            source_hash.update(after);
        }
        records.push(serde_json::json!({
            "case":case, "ring":ring, "hidden":hidden, "extracts":ne, "rows":total,
            "head":rb.head, "written":rb.written,
            "target_hidden_bytes":actual.len(),
            "target_hidden_sha256":format!("{:x}", Sha256::digest(&actual)),
            "source_ring_sha256":format!("{:x}", source_hash.finalize()),
            "required_temporary_host_sha256":required_host_sha,
        }));
        gpu.free_tensor(target_hidden).unwrap();
        gpu.free_tensor(source).unwrap();
        rb.free_gpu(gpu);
    }
    records
}
