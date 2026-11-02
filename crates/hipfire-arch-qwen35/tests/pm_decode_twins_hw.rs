// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! PM decode twins G0: capture real H2 inputs of the W1 projection launches
//! from one actual Qwen3.6-27B decode step, for the byte oracle
//! `crates/rdna-compute/examples/pm_decode_twins.rs`.
//!
//! Method (no launcher or model edits):
//! 1. load the trunk, prefill the Redline synthetic context
//!    (`10 + i % 1000`), run one warm decode step at `ctx`;
//! 2. snapshot recurrent + KV state, record the decode step at `ctx + 1`
//!    (token 101) through the replay recorder (manual capture);
//! 3. restore the snapshot, re-prepare the step inputs and relaunch the full
//!    recorded HIP prefix: logits, recurrent and KV must be byte-identical to
//!    the recorded step, proving the restore covers everything the tape reads;
//! 4. for every selected W1 launch, restore, relaunch the prefix up to (not
//!    including) that launch, download each pointer argument's whole backing
//!    range from the exact recorded kernarg VAs ("pre"), launch the recorded
//!    blob once and download again ("post").
//!
//! Output: `<out>/<symbol>/<occurrence>/{meta.json,arg<offset>.pre.bin,arg<offset>.post.bin}`.
//!
//! Run under a card lock, GPUs restricted to the locked device:
//!
//! ```bash
//! HIPFIRE_REPLAY_BACKEND=shadow HIPFIRE_REPLAY_MANUAL_CAPTURE=1 HIPFIRE_GRAPH=0 \
//! HIPFIRE_AR_GRAPH=0 HIPFIRE_CASK_OFF=1 \
//! PMDT_MODEL=/abs/model.mq4-xts PMDT_OUT=/abs/oracle-inputs \
//! cargo test --release -p hipfire-arch-qwen35 --test pm_decode_twins_hw -- --ignored --nocapture
//! ```

use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use rdna_compute::{Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const TOKEN: u32 = 101;

/// Pointer argument roles of the selected W1 symbols (offsets are the frozen
/// clang24 kernarg layout from inventory.json).
#[derive(Clone, Copy)]
enum Arg {
    /// MQ4v2 weight matrix: rows read from the i32 at `m`, `K/256` 136-byte groups per row.
    Weight { ptr: usize, m: usize },
    /// Activation: K f32.
    X { ptr: usize },
    /// Output vector: rows read from the i32 at `m`.
    Y { ptr: usize, m: usize },
}

struct Symbol {
    name: &'static str,
    k: usize,
    args: &'static [Arg],
}

const SYMBOLS: &[Symbol] = &[
    Symbol {
        name: "fused_qkvza_mq4g256v2",
        k: 88,
        args: &[
            Arg::Weight { ptr: 0, m: 72 },
            Arg::Weight { ptr: 8, m: 76 },
            Arg::Weight { ptr: 16, m: 80 },
            Arg::Weight { ptr: 24, m: 84 },
            Arg::X { ptr: 32 },
            Arg::Y { ptr: 40, m: 72 },
            Arg::Y { ptr: 48, m: 76 },
            Arg::Y { ptr: 56, m: 80 },
            Arg::Y { ptr: 64, m: 84 },
        ],
    },
    Symbol {
        name: "fused_qkv_mq4g256v2",
        k: 68,
        args: &[
            Arg::Weight { ptr: 0, m: 56 },
            Arg::Weight { ptr: 8, m: 60 },
            Arg::Weight { ptr: 16, m: 64 },
            Arg::X { ptr: 24 },
            Arg::Y { ptr: 32, m: 56 },
            Arg::Y { ptr: 40, m: 60 },
            Arg::Y { ptr: 48, m: 64 },
        ],
    },
    Symbol {
        name: "fused_gate_up_mq4g256v2",
        k: 48,
        args: &[
            Arg::Weight { ptr: 0, m: 40 },
            Arg::Weight { ptr: 8, m: 44 },
            Arg::X { ptr: 16 },
            Arg::Y { ptr: 24, m: 40 },
            Arg::Y { ptr: 32, m: 44 },
        ],
    },
    Symbol {
        name: "gemv_mq4g256v2_residual",
        k: 28,
        args: &[
            Arg::Weight { ptr: 0, m: 24 },
            Arg::X { ptr: 8 },
            Arg::Y { ptr: 16, m: 24 },
        ],
    },
    Symbol {
        name: "gemv_mq4g256v2_multirow_r2",
        k: 28,
        args: &[
            Arg::Weight { ptr: 0, m: 24 },
            Arg::X { ptr: 8 },
            Arg::Y { ptr: 16, m: 24 },
        ],
    },
];

fn env_path(key: &str, default: &str) -> PathBuf {
    let p = PathBuf::from(std::env::var(key).unwrap_or_else(|_| default.to_owned()));
    assert!(p.is_absolute(), "{key} must be absolute: {}", p.display());
    p
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn rd_i32(kernarg: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(kernarg[off..off + 4].try_into().unwrap())
}

fn rd_u64(kernarg: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(kernarg[off..off + 8].try_into().unwrap())
}

fn mapped_len(gpu: &Gpu, t: &GpuTensor) -> usize {
    gpu.vmm_mapped_bytes(t)
        .map_or(t.buf.size(), |m| m.min(t.buf.size()))
}

/// Whole-surface snapshot of the state a decode step reads and writes.
struct State {
    tensors: Vec<Vec<u8>>,
}

fn state_tensors(slot: &ModelSlot) -> Vec<&GpuTensor> {
    let kv = &slot.kv_cache;
    let dn = &slot.dn_state;
    kv.k_gpu
        .iter()
        .chain(kv.v_gpu.iter())
        .chain(kv.k_scales.iter())
        .chain(kv.v_scales.iter())
        .chain(dn.s_matrices.iter())
        .chain(dn.s_scales.iter())
        .chain(dn.conv_states.iter())
        .chain(dn.s_ef_residual.iter())
        .collect()
}

fn snapshot(gpu: &Gpu, slot: &ModelSlot) -> State {
    gpu.hip.device_synchronize().expect("sync");
    let tensors = state_tensors(slot)
        .into_iter()
        .map(|t| {
            let mut b = vec![0u8; mapped_len(gpu, t)];
            gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("dtoh state");
            b
        })
        .collect();
    State { tensors }
}

fn restore(gpu: &Gpu, slot: &ModelSlot, s: &State) {
    for (t, b) in state_tensors(slot).into_iter().zip(&s.tensors) {
        gpu.hip.memcpy_htod(&t.buf, b).expect("htod state");
    }
    gpu.hip.device_synchronize().expect("sync");
}

fn logits(gpu: &Gpu, slot: &ModelSlot) -> Vec<u8> {
    gpu.hip.device_synchronize().expect("sync");
    let t = &slot.scratch.logits;
    let mut b = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("dtoh logits");
    b
}

fn arg_bytes(sym: &Symbol, kernarg: &[u8], arg: Arg) -> (usize, usize) {
    let k = rd_i32(kernarg, sym.k);
    assert!(k > 0 && k % 256 == 0, "{}: K={k}", sym.name);
    let k = k as usize;
    let rows = |m: usize| {
        let v = rd_i32(kernarg, m);
        assert!(v >= 0, "{}: m@{m}={v}", sym.name);
        v as usize
    };
    match arg {
        Arg::Weight { ptr, m } => (ptr, rows(m) * (k / 256) * 136),
        Arg::X { ptr } => (ptr, k * 4),
        Arg::Y { ptr, m } => (ptr, rows(m) * 4),
    }
}

fn download_args(gpu: &Gpu, sym: &Symbol, kernarg: &[u8]) -> Vec<(usize, Vec<u8>)> {
    gpu.hip.device_synchronize().expect("sync");
    sym.args
        .iter()
        .map(|&arg| {
            let (off, len) = arg_bytes(sym, kernarg, arg);
            let mut b = vec![0u8; len];
            if len > 0 {
                let va = rd_u64(kernarg, off) as *mut std::ffi::c_void;
                assert!(
                    !va.is_null(),
                    "{}: null pointer @{off} with {len} bytes",
                    sym.name
                );
                // SAFETY: non-owning view of the live model/scratch allocation
                // the recorded launch addresses; `len` is the kernel's own
                // declared extent of that argument.
                let view = unsafe { hip_bridge::DeviceBuffer::from_raw(va, len) };
                gpu.hip.memcpy_dtoh(&mut b, &view).expect("dtoh arg");
                std::mem::forget(view);
            }
            (off, b)
        })
        .collect()
}

#[test]
#[ignore]
fn capture_w1_decode_inputs() {
    for (key, want) in [
        ("HIPFIRE_REPLAY_BACKEND", "shadow"),
        ("HIPFIRE_REPLAY_MANUAL_CAPTURE", "1"),
        ("HIPFIRE_GRAPH", "0"),
        ("HIPFIRE_AR_GRAPH", "0"),
    ] {
        assert_eq!(
            std::env::var(key).as_deref(),
            Ok(want),
            "{key} must be {want}"
        );
    }
    let model = env_path(
        "PMDT_MODEL",
        "/home/kaden/qcal/release-0.4.0/fp8-tp-ep-geo/models/qwen3.8-27b.mq4-xts",
    );
    let out = env_path(
        "PMDT_OUT",
        "/home/kaden/qcal/release-0.4.2/pm-decode-twins/oracle-inputs",
    );
    let ctx: usize = std::env::var("PMDT_CTX")
        .ok()
        .map_or(512, |v| v.parse().unwrap());

    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx1201", "capture is gfx1201-only");
    let mut cfg = ModelSlotConfig::default();
    cfg.max_seq = ctx + 64;
    let mut slot = ModelSlot::load(&mut gpu, &model, "pmdt-capture", cfg).expect("load trunk");

    // Prime: Redline synthetic context, then one warm decode step so lazy
    // allocations and first-use kernel loads stay out of the recorded step.
    let prompt: Vec<u32> = (0..ctx as u32).map(|i| 10 + (i % 1000)).collect();
    {
        let ModelSlot {
            weights,
            config,
            kv_cache,
            dn_state,
            scratch,
            ..
        } = &mut slot;
        qwen35::forward_prefill_batch(
            &mut gpu, weights, config, &prompt, 0, kv_cache, dn_state, scratch, None, None, None,
            None,
        )
        .expect("prefill");
        qwen35::forward_scratch(
            &mut gpu, weights, config, TOKEN, ctx, kv_cache, dn_state, scratch,
        )
        .expect("warm decode");
    }
    let pos = ctx + 1;
    let before = snapshot(&gpu, &slot);
    let frame = rdna_compute::norm::gdn_requant_frame_checkpoint();

    {
        let ModelSlot {
            weights,
            config,
            kv_cache,
            dn_state,
            scratch,
            ..
        } = &mut slot;
        qwen35::prepare_scratch_inputs(&mut gpu, weights, config, TOKEN, pos, scratch)
            .expect("prepare inputs");
        gpu.hip.device_synchronize().expect("sync");
        gpu.replay.begin_capture().expect("begin capture");
        qwen35::forward_scratch(
            &mut gpu, weights, config, TOKEN, pos, kv_cache, dn_state, scratch,
        )
        .expect("recorded decode");
        gpu.hip.device_synchronize().expect("sync");
    }
    let summary = gpu.replay.finish_capture().expect("finish capture");
    let launches: Vec<(String, [u32; 3], [u32; 3], u32, Vec<u8>)> = gpu
        .replay
        .recorded_launches()
        .iter()
        .map(|l| {
            (
                l.kernel.clone(),
                l.grid,
                l.block,
                l.shared_mem,
                l.kernarg.clone(),
            )
        })
        .collect();
    let recorded_logits = logits(&gpu, &slot);
    let recorded_state = snapshot(&gpu, &slot);
    eprintln!(
        "recorded {} launches, {} unique kernels, sequence {:016x}",
        summary.launch_count, summary.unique_kernel_count, summary.sequence_hash
    );
    assert_eq!(summary.launch_count, launches.len());

    let prefix = |gpu: &mut Gpu, slot: &mut ModelSlot, count: usize| {
        restore(gpu, slot, &before);
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame);
        let ModelSlot {
            weights,
            config,
            scratch,
            ..
        } = slot;
        qwen35::prepare_scratch_inputs(gpu, weights, config, TOKEN, pos, scratch)
            .expect("prepare inputs");
        gpu.replay_recorded_hip_prefix(count)
            .expect("prefix replay");
        gpu.hip.device_synchronize().expect("sync");
    };

    // Restore completeness: full prefix must reproduce the recorded step.
    prefix(&mut gpu, &mut slot, launches.len());
    let replay_logits = logits(&gpu, &slot);
    let replay_state = snapshot(&gpu, &slot);
    assert!(
        replay_logits == recorded_logits,
        "full-prefix replay logits differ from recorded step"
    );
    assert!(
        replay_state.tensors == recorded_state.tensors,
        "full-prefix replay recurrent/KV differ from recorded step"
    );
    eprintln!("restore check: full-prefix replay byte-identical (logits, KV, recurrent)");

    for sym in SYMBOLS {
        let hits: Vec<usize> = launches
            .iter()
            .enumerate()
            .filter(|(_, l)| l.0 == sym.name)
            .map(|(i, _)| i)
            .collect();
        assert!(
            !hits.is_empty(),
            "{} not launched in the decode step",
            sym.name
        );
        let mut picks = vec![(0usize, hits[0])];
        if hits.len() > 1 {
            picks.push((hits.len() - 1, hits[hits.len() - 1]));
        }
        for (ordinal, index) in picks {
            let (kernel, grid, block, shm, kernarg) = launches[index].clone();
            prefix(&mut gpu, &mut slot, index);
            let pre = download_args(&gpu, sym, &kernarg);
            let mut blob = kernarg.clone();
            gpu.launch_kernel_blob(&kernel, grid, block, shm, &mut blob)
                .expect("callsite launch");
            let post = download_args(&gpu, sym, &kernarg);
            let dir = out.join(sym.name).join(format!("occ{ordinal}"));
            std::fs::create_dir_all(&dir).expect("mkdir");
            let mut files = serde_json::Map::new();
            for ((off, a), (_, b)) in pre.iter().zip(&post) {
                for (tag, bytes) in [("pre", a), ("post", b)] {
                    let name = format!("arg{off}.{tag}.bin");
                    std::fs::write(dir.join(&name), bytes).expect("write arg");
                    files.insert(
                        name,
                        serde_json::json!({"bytes": bytes.len(), "sha256": sha256_hex(bytes)}),
                    );
                }
            }
            let meta = serde_json::json!({
                "schema": "pmdt-capture-v1",
                "symbol": sym.name,
                "occurrence_ordinal": ordinal,
                "occurrences_in_step": hits.len(),
                "launch_index": index,
                "launches_in_step": launches.len(),
                "grid": grid,
                "block": block,
                "dynamic_lds": shm,
                "kernarg_hex": kernarg.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                "model": model.display().to_string(),
                "context_tokens": ctx,
                "position": pos,
                "token": TOKEN,
                "prompt": "redline synthetic 10 + i % 1000",
                "kv_mode": "q8 (ModelSlot default)",
                "arch": gpu.arch,
                "restore_check": "full-prefix replay logits/KV/recurrent byte-identical",
                "files": files,
            });
            std::fs::write(
                dir.join("meta.json"),
                serde_json::to_vec_pretty(&meta).unwrap(),
            )
            .expect("write meta");
            eprintln!(
                "captured {} occ{ordinal} launch {index} -> {}",
                sym.name,
                dir.display()
            );
        }
    }
}
