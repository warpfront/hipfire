// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Session-cache restores on Flash-Next are byte-identical to a cold prefill.
//!
//! `#[ignore]`d: it needs a real HIP GPU and the canonical
//! `qwen3.8-flash-next-gptq3.mq4` artifact named by
//! `HIPFIRE_SESSION_CACHE_MODEL`.
//!
//! Prompt A (three chunks and a tail) is prefilled cold, capturing a chain of
//! delta snapshots at its three chunk boundaries. Prompt B shares only A's
//! first chunk and is also prefilled cold, so its two-chunk snapshot is a
//! delta over A's first (asserted through the stored bytes). A is then
//! restored through its three-link chain and B through the shared root.
//!
//! Each restored prefill must leave the live state byte-equal to its cold
//! prefill: a SHA-256 of every state part (metadata, every fixed part and the
//! valid rows of every row stream) is compared, so a wrong byte in rows that
//! attention never selects still fails. The AR leg also compares the final
//! logits and 16 greedy ids; the MTP leg repeats the chain and branch on the
//! native MTP route (target plus draft head and its policy) and compares the
//! seed token.

use hipfire_arch_qwen4::bundle::{session_snapshot_bytes, Qwen4Bundle};
use hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter;
use hipfire_arch_qwen4::{admit_hfqm_artifact, Qwen4KvBackend, Qwen4StateFormat};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::serve_contract::CacheDomain;
use hipfire_runtime::session_cache::{SessionCache, SessionRoute, SessionState, SnapshotParts};
use hipfire_runtime::spec::MtpDrafter;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const MODEL_ENV: &str = "HIPFIRE_SESSION_CACHE_MODEL";
const MAX_SEQ: usize = 32768;
const DECODE: usize = 16;

fn prompt(seed: u64, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (1000 + (state >> 33) % 99_000) as u32
        })
        .collect()
}

/// SHA-256 of every part of the live state at `position` on `route`: the
/// metadata, each fixed part, then the valid rows of each row stream.
fn state_digests(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    route: SessionRoute,
    position: usize,
) -> Vec<[u8; 32]> {
    let SnapshotParts { meta, layout } =
        SessionState::snapshot_parts(bundle, route, position).expect("state parts");
    gpu.hip.device_synchronize().expect("sync");
    let mut digests = vec![Sha256::digest(&meta).into()];
    let ranges = layout
        .fixed
        .iter()
        .map(|part| (part.buf, part.offset, part.bytes))
        .chain(
            layout
                .rows
                .iter()
                .map(|stream| (stream.buf, 0, stream.rows * stream.row_bytes)),
        );
    for (buf, offset, bytes) in ranges {
        let mut host = vec![0u8; bytes];
        gpu.hip
            .memcpy_dtoh_at(&mut host, buf, offset)
            .expect("download state part");
        digests.push(Sha256::digest(&host).into());
    }
    digests
}

/// Fail with the index of the first state part a restore got wrong.
fn assert_same_state(name: &str, cold: &[[u8; 32]], warm: &[[u8; 32]]) {
    assert_eq!(cold.len(), warm.len(), "{name}: state part count");
    if let Some(part) = (0..cold.len()).find(|&i| cold[i] != warm[i]) {
        panic!(
            "{name}: state part {part} of {} differs from cold (0 = metadata)",
            cold.len()
        );
    }
}

/// Prefill `tokens` reusing `reused` cached tokens and commit. Returns the
/// state digests, the final logits bytes and `DECODE` greedy ids.
fn run(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    tokens: &[u32],
    reused: usize,
) -> (Vec<[u8; 32]>, Vec<u8>, Vec<u32>) {
    bundle
        .prefill_final(gpu, tokens, reused, logits)
        .expect("prefill");
    let digests = state_digests(bundle, gpu, SessionRoute::Ar, tokens.len());
    bundle.session_commit();
    let mut bytes = vec![0u8; logits.byte_size()];
    gpu.hip.device_synchronize().expect("sync");
    gpu.hip
        .memcpy_dtoh(&mut bytes, &logits.buf)
        .expect("download logits");
    let ids = (0..DECODE)
        .map(|_| {
            bundle
                .forward_token_or_argmax(gpu, None, logits)
                .expect("decode")
        })
        .collect();
    (digests, bytes, ids)
}

/// Native MTP prefill of `tokens` reusing `reused` cached tokens, then
/// commit. Returns the state digests (target and draft head) and the seed.
fn run_mtp(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    drafter: &mut Qwen4MtpDrafter,
    tokens: &[u32],
    reused: usize,
) -> (Vec<[u8; 32]>, u32) {
    let seed = drafter
        .mtp_prefill(
            gpu,
            bundle,
            tokens,
            &tokens[reused..],
            reused,
            reused > 0,
            &|| false,
        )
        .expect("MTP prefill");
    let digests = state_digests(bundle, gpu, SessionRoute::Mtp, tokens.len());
    bundle.session_commit();
    (digests, seed)
}

fn stored_bytes(bundle: &Qwen4Bundle) -> u64 {
    bundle
        .session_cache()
        .expect("session cache")
        .stored_bytes()
}

/// Four snapshots (A's three links and B's one) of one chunk each: a full
/// copy of B's two chunks would exceed four chunk-sized snapshots.
fn assert_four_deltas(route: &str, stored: u64, one: u64) {
    println!("{route}: {stored} bytes stored, {one} per one-chunk snapshot");
    assert!(
        stored > 3 * one && stored <= 4 * one,
        "{route}: expected four one-chunk snapshots ({stored} bytes, {one} each)"
    );
}

/// A loaded Flash-Next bundle with the session cache attached.
struct Loaded {
    bundle: Qwen4Bundle,
    gpu: Gpu,
    tokenizer: Tokenizer,
    state_format: Qwen4StateFormat,
    vocab: usize,
    backend: Qwen4KvBackend,
}

/// Load the model named by `HIPFIRE_SESSION_CACHE_MODEL` and attach the
/// forward and an unbounded session cache, with a disk tier in `disk` when
/// given. `turn_snapshots` sets message-end snapshots explicitly, whatever
/// `HIPFIRE_QWEN4_TURN_SNAPSHOTS` says (the `<|im_end|>` token is still the
/// caller's `set_turn_end_token`).
fn load(turn_snapshots: bool, disk: Option<&Path>) -> Loaded {
    let model = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name qwen3.8-flash-next-gptq3.mq4"));
    let mut hfq = HfqFile::open(&model).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let receipt = admit_hfqm_artifact(&hfq).expect("admission");
    let mut gpu = Gpu::init().expect("GPU");
    let domain = CacheDomain::for_model(&hfq, &tokenizer, None, "qwen4", gpu.device_id);
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().expect("mesh");
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &receipt.manifest.weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            source
                .tensor_range(&entry.name)
                .map_err(|e| e.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .expect("weights");
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        "",
        &gpu,
        &receipt.config,
    )
    .expect("state format");
    let vocab = receipt.config.vocab_size;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        transaction,
        &receipt.placements,
        &mut gpu,
        MAX_SEQ,
        receipt.ple,
        state_format,
        backend,
    )
    .expect("assemble");
    bundle.set_turn_snapshots(turn_snapshots);
    bundle.attach_forward(&mut gpu, MAX_SEQ).expect("forward");
    let cache = SessionCache::new(domain, u64::MAX >> 1);
    bundle.attach_session_cache(match disk {
        Some(dir) => cache.with_disk(dir, 1 << 40),
        None => cache,
    });
    Loaded {
        bundle,
        gpu,
        tokenizer,
        state_format,
        vocab,
        backend,
    }
}

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_SESSION_CACHE_MODEL"]
fn restored_prefill_matches_cold_on_flash_next() {
    let Loaded {
        mut bundle,
        mut gpu,
        state_format,
        vocab,
        backend,
        ..
    } = load(false, None);
    let chunk = bundle.spec_chunk_rows().expect("chunk rows");
    let logits = gpu.zeros(&[vocab], DType::F32).expect("logits");
    let a = prompt(0xa, 3 * chunk + 300);
    let mut b = a[..chunk].to_vec();
    b.extend(prompt(0xb, chunk + 200));
    println!(
        "chunk={chunk} state={state_format:?} backend={}",
        backend.name()
    );

    let one = |mtp| {
        session_snapshot_bytes(&bundle.config, state_format, mtp, chunk).expect("snapshot bytes")
    };
    let (one_ar, one_mtp) = (one(false), one(true));

    let cold_a = run(&mut bundle, &mut gpu, &logits, &a, 0);
    let cold_b = run(&mut bundle, &mut gpu, &logits, &b, 0);
    assert_four_deltas("AR", stored_bytes(&bundle), one_ar);
    for (name, tokens, cold, links) in [("A", &a, &cold_a, 3), ("B", &b, &cold_b, 2)] {
        let reused = bundle.session_plan(tokens, SessionRoute::Ar);
        assert_eq!(
            reused,
            links * chunk,
            "plan must offer {name}'s {links}-chunk snapshot"
        );
        let (digests, warm_logits, warm_ids) = run(&mut bundle, &mut gpu, &logits, tokens, reused);
        println!(
            "AR {name} cold ids {:?}\nAR {name} warm ids {warm_ids:?}",
            cold.2
        );
        assert_same_state(&format!("AR {name}"), &cold.0, &digests);
        assert!(
            warm_logits == cold.1,
            "AR {name}: restored final logits differ from cold"
        );
        assert_eq!(warm_ids, cold.2, "AR {name}");
    }

    bundle.attach_mtp(&mut gpu, MAX_SEQ).expect("MTP head");
    let mut drafter = Qwen4MtpDrafter::new(3, MAX_SEQ, None);
    let before = stored_bytes(&bundle);
    let cold_a = run_mtp(&mut bundle, &mut gpu, &mut drafter, &a, 0);
    let cold_b = run_mtp(&mut bundle, &mut gpu, &mut drafter, &b, 0);
    assert_four_deltas("MTP", stored_bytes(&bundle) - before, one_mtp);
    for (name, tokens, cold, links) in [("A", &a, &cold_a, 3), ("B", &b, &cold_b, 2)] {
        let reused = bundle.session_plan(tokens, SessionRoute::Mtp);
        assert_eq!(
            reused,
            links * chunk,
            "MTP plan must offer {name}'s {links}-chunk snapshot"
        );
        let (digests, seed) = run_mtp(&mut bundle, &mut gpu, &mut drafter, tokens, reused);
        println!("MTP {name} seed cold {} warm {seed}", cold.1);
        assert_same_state(&format!("MTP {name}"), &cold.0, &digests);
        assert_eq!(seed, cold.1, "MTP {name} seed");
    }

    gpu.free_tensor(logits).expect("free logits");
    bundle.free_gpu(&mut gpu).expect("free bundle");
}

const DISK_WRITE_PHASE: &str = "HIPFIRE_SESSION_CACHE_DISK_WRITE_PHASE";

fn snap_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter(|e| {
                    e.as_ref()
                        .is_ok_and(|e| e.path().extension() == Some("snap".as_ref()))
                })
                .count()
        })
        .unwrap_or(0)
}

/// Snapshots demoted to the disk tier at unload restore in a new process (a
/// daemon restart of the same build) byte-equal to a cold prefill. The write
/// phase runs in a child process: one process cannot load the model twice,
/// because unloading does not return the weight memory to the system.
#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_SESSION_CACHE_MODEL"]
fn disk_snapshots_restore_after_restart_on_flash_next() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen4-session-disk");
    let Loaded {
        mut bundle,
        mut gpu,
        vocab,
        ..
    } = if std::env::var_os(DISK_WRITE_PHASE).is_some() {
        // Child: prefill A, then unload, which demotes its three snapshots.
        let mut loaded = load(false, Some(&dir));
        let chunk = loaded.bundle.spec_chunk_rows().expect("chunk rows");
        let a = prompt(0xa, 3 * chunk + 300);
        let logits = loaded
            .gpu
            .zeros(&[loaded.vocab], DType::F32)
            .expect("logits");
        run(&mut loaded.bundle, &mut loaded.gpu, &logits, &a, 0);
        loaded.gpu.free_tensor(logits).expect("free logits");
        loaded
            .bundle
            .free_gpu(&mut loaded.gpu)
            .expect("free bundle");
        return;
    } else {
        let _ = std::fs::remove_dir_all(&dir);
        let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--ignored",
                "--exact",
                "disk_snapshots_restore_after_restart_on_flash_next",
            ])
            .args(["--nocapture", "--test-threads", "1"])
            .env(DISK_WRITE_PHASE, "1")
            .status()
            .expect("spawn write phase");
        assert!(status.success(), "write phase failed: {status}");
        assert_eq!(
            snap_files(&dir),
            3,
            "unload must demote A's three snapshots"
        );
        load(false, Some(&dir))
    };
    let chunk = bundle.spec_chunk_rows().expect("chunk rows");
    let a = prompt(0xa, 3 * chunk + 300);
    let logits = gpu.zeros(&[vocab], DType::F32).expect("logits");
    let reused = bundle.session_plan(&a, SessionRoute::Ar);
    assert_eq!(reused, 3 * chunk, "plan must offer A's disk snapshot");
    let (digests, warm_logits, warm_ids) = run(&mut bundle, &mut gpu, &logits, &a, reused);
    assert_eq!(snap_files(&dir), 0, "restore must promote all three links");
    let cold = run(&mut bundle, &mut gpu, &logits, &a, 0);
    assert_same_state("disk A", &cold.0, &digests);
    assert!(
        warm_logits == cold.1,
        "disk A: restored final logits differ from cold"
    );
    assert_eq!(warm_ids, cold.2, "disk A");
    gpu.free_tensor(logits).expect("free logits");
    bundle.free_gpu(&mut gpu).expect("free bundle");
    let _ = std::fs::remove_dir_all(&dir);
}

/// P1b: message-end snapshots (`HIPFIRE_QWEN4_TURN_SNAPSHOTS`, forced on through
/// `set_turn_snapshots`). A prompt with two message ends below one chunk is
/// prefilled cold and committed; a second prompt extending the first message
/// ends plans exactly the deepest of them. Message-end snapshots are
/// cold-schedule split at the message ends, so the restored prefill is
/// compared with (a) a second restore of the same snapshot and (b) a cold
/// prefill of the second prompt with the same flag on, which splits at the
/// same message ends by construction. It is NOT compared with the chunk-only
/// cold schedule.
#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_SESSION_CACHE_MODEL"]
fn message_end_snapshots_restore_on_flash_next() {
    let Loaded {
        mut bundle,
        mut gpu,
        tokenizer,
        vocab,
        ..
    } = load(true, None);
    let im_end = tokenizer
        .special_token_id("<|im_end|>")
        .expect("tokenizer has <|im_end|>");
    bundle.set_turn_end_token(Some(im_end));
    assert!(bundle.turn_snapshots_active());
    let chunk = bundle.spec_chunk_rows().expect("chunk rows");
    assert!(chunk >= 128, "test needs message ends below one chunk");
    let logits = gpu.zeros(&[vocab], DType::F32).expect("logits");

    // Two messages (message ends at u + 1 and 2u + 2) and a 40-token tail, all
    // below one chunk.
    let u = chunk / 4;
    let mut first = prompt(0x1, u);
    first.push(im_end);
    first.extend(prompt(0x2, u));
    first.push(im_end);
    let (end_a, end_b) = (u + 1, 2 * u + 2);
    assert_eq!(first.len(), end_b);
    let mut s1 = first.clone();
    s1.extend(prompt(0x3, 40));
    assert!(s1.len() < chunk);
    // Extends the message-end prefix with different content.
    let mut s2 = first.clone();
    s2.extend(prompt(0x4, 70));

    assert_eq!(bundle.session_plan(&s1, SessionRoute::Ar), 0);
    run(&mut bundle, &mut gpu, &logits, &s1, 0);
    assert!(stored_bytes(&bundle) > 0, "message ends were not published");
    // The first prompt itself reuses its deepest message end, and one that
    // only passes the first message reuses that one.
    assert_eq!(bundle.session_plan(&s1, SessionRoute::Ar), end_b);
    let mut one_message = s1[..end_a].to_vec();
    one_message.extend(prompt(0x5, 30));
    assert_eq!(bundle.session_plan(&one_message, SessionRoute::Ar), end_a);

    // The second prompt plans exactly the deepest shared message end.
    let reused = bundle.session_plan(&s2, SessionRoute::Ar);
    assert_eq!(reused, end_b, "plan must offer the message-end snapshot");

    // Flag off scopes to chunk multiples only: nothing is offered.
    bundle.set_turn_snapshots(false);
    assert_eq!(bundle.session_plan(&s2, SessionRoute::Ar), 0);
    bundle.set_turn_snapshots(true);
    assert_eq!(bundle.session_plan(&s2, SessionRoute::Ar), end_b);

    let warm = run(&mut bundle, &mut gpu, &logits, &s2, reused);
    let again = run(&mut bundle, &mut gpu, &logits, &s2, reused);
    assert_same_state("restore twice", &warm.0, &again.0);
    assert!(warm.1 == again.1, "same snapshot restored twice: logits");
    assert_eq!(warm.2, again.2, "same snapshot restored twice: ids");

    // Cold with the same split (flag on): message ends at u + 1 and 2u + 2.
    let cold = run(&mut bundle, &mut gpu, &logits, &s2, 0);
    println!(
        "AR message-end cold ids {:?}\nAR message-end warm ids {:?}",
        cold.2, warm.2
    );
    assert_same_state("AR message-end", &cold.0, &warm.0);
    assert!(
        warm.1 == cold.1,
        "message-end restored final logits differ from the same-split cold prefill"
    );
    assert_eq!(warm.2, cold.2, "AR message-end ids");

    gpu.free_tensor(logits).expect("free logits");
    bundle.free_gpu(&mut gpu).expect("free bundle");
}
