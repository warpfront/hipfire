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
//!
//! `retired_turn_commits_live_on_flash_next` (`--features lab` only) checks
//! that a native MTP turn the AR floor retired still ends with its head at
//! the target and commits live for the conversation's next turn.
//!
//! `terminal_stop_inside_window_repairs_live_on_flash_next` (`--features lab`
//! only) checks that a terminal stop inside a committed MTP window restores
//! the retained pre-window state, replays only the consumed prefix and
//! commits live.
//!
//! Each test loads Flash-Next (tens of GiB of pinned host memory), so run them
//! one per process thread and only inside a fabric hold that declares that
//! memory and holds the Flash-Next host lock, e.g. on an R9700:
//!
//! ```text
//! fabric-run.sh --class correct --card B --mem-gb 30 --hostmem-gb 50 --hostmem-lock \
//!   --max-secs 3600 -- session_cache_hw --ignored --test-threads=1 --exact <test>
//! ```
//!
//! (Halo: `--card halo --mem-gb 120 --hostmem-lock`.) [`load`] refuses to run
//! outside such a hold, naming the processes that hold the host lock, and
//! without `--test-threads=1`.

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
        .and_then(|outcome| outcome.into_first_token())
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

/// The Flash-Next host lock fabric takes with `--hostmem-lock` (shared for
/// correct, exclusive for timing holds).
const HOSTMEM_LOCK: &str = "hostmem-flashnext.lock";

/// `/proc/<pid>/stat` fields after the command: 1 = ppid, 19 = starttime.
fn proc_stat(pid: u32, index: usize) -> Option<u64> {
    let raw = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    raw.get(raw.rfind(')')? + 2..)?.split_whitespace().nth(index)?.parse().ok()
}

/// Granted flocks on files named [`HOSTMEM_LOCK`] that `pids` hold, read from
/// `/proc/locks` through each process's open files: `(pid, mode)`.
fn hostmem_lock_holders(pids: impl Iterator<Item = u32>) -> Vec<(u32, String)> {
    use std::os::unix::fs::MetadataExt;
    let locks = std::fs::read_to_string("/proc/locks").unwrap_or_default();
    let mut held = Vec::new();
    for pid in pids {
        let Ok(fds) = std::fs::read_dir(format!("/proc/{pid}/fd")) else { continue };
        for fd in fds.flatten() {
            let Ok(target) = std::fs::read_link(fd.path()) else { continue };
            if target.file_name().and_then(|n| n.to_str()) != Some(HOSTMEM_LOCK) {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&target) else { continue };
            let inode = format!(":{}", meta.ino());
            for line in locks.lines() {
                let f: Vec<&str> = line.split_whitespace().collect();
                if f.len() >= 6 && !f.contains(&"->") && f[1] == "FLOCK"
                    && f[4] == pid.to_string() && f[5].ends_with(&inode)
                    && !held.iter().any(|(p, _): &(u32, String)| *p == pid)
                {
                    held.push((pid, f[3].to_string()));
                }
            }
        }
    }
    held
}

/// Why a Flash-Next load must not run here, or `None` inside its own hold:
/// the harness runs one test at a time (two loads in one process exhaust
/// host memory), `chain` (this process, then its ancestors) contains the live
/// fabric wrapper `FABRIC_JOB` names (`pid:starttime:cards`), and a process
/// on the chain up to that wrapper holds [`HOSTMEM_LOCK`].
fn hold_refusal(
    threads_one: bool,
    marker: &str,
    chain: &[u32],
    start_of: impl Fn(u32) -> Option<u64>,
    holds_lock: impl Fn(&[u32]) -> bool,
) -> Option<String> {
    if !threads_one {
        return Some("each test loads Flash-Next: run with --test-threads=1".into());
    }
    let mut parts = marker.splitn(3, ':');
    let wrapper = parts.next().and_then(|p| p.parse::<u32>().ok());
    let start = parts.next().and_then(|s| s.parse::<u64>().ok());
    let Some((wrapper, start)) = wrapper.zip(start) else {
        return Some("no fabric hold (FABRIC_JOB unset)".into());
    };
    let Some(depth) = chain.iter().position(|&pid| pid == wrapper) else {
        return Some(format!("fabric wrapper {wrapper} is not an ancestor of this process"));
    };
    if start_of(wrapper) != Some(start) {
        return Some(format!("fabric wrapper {wrapper} is gone (stale FABRIC_JOB)"));
    }
    if !holds_lock(&chain[..=depth]) {
        return Some(format!("fabric job {marker} does not hold {HOSTMEM_LOCK} (no --hostmem-lock)"));
    }
    None
}

/// Panic before a Flash-Next load outside its own fabric hold
/// ([`hold_refusal`]), naming the processes that hold the host lock.
fn require_fabric_hold() {
    let args: Vec<String> = std::env::args().collect();
    let threads_one = std::env::var("RUST_TEST_THREADS").is_ok_and(|v| v == "1")
        || args.iter().any(|a| a == "--test-threads=1")
        || args.windows(2).any(|w| w[0] == "--test-threads" && w[1] == "1");
    let mut chain = vec![std::process::id()];
    while let Some(ppid) = proc_stat(*chain.last().unwrap(), 1).filter(|&p| p > 1) {
        chain.push(ppid as u32);
    }
    let marker = std::env::var("FABRIC_JOB").unwrap_or_default();
    let refusal = hold_refusal(threads_one, &marker, &chain, |pid| proc_stat(pid, 19), |pids| {
        !hostmem_lock_holders(pids.iter().copied()).is_empty()
    });
    if let Some(why) = refusal {
        let all = std::fs::read_dir("/proc").into_iter().flatten().flatten()
            .filter_map(|e| e.file_name().to_str()?.parse::<u32>().ok());
        panic!(
            "refusing to load Flash-Next: {why}. {HOSTMEM_LOCK} holders (pid, mode): {:?}. Run \
             inside `fabric-run.sh ... --hostmem-gb <true peak> --hostmem-lock -- session_cache_hw \
             --ignored --test-threads=1 ...` (see the module docs)",
            hostmem_lock_holders(all)
        );
    }
}

#[test]
fn fabric_hold_guard_refuses_loads_outside_a_hold() {
    let chain = [30, 20, 10];
    let start = |pid| (pid == 20).then_some(77);
    let refusal = |threads, marker: &str, holds: bool| {
        hold_refusal(threads, marker, &chain, start, |pids: &[u32]| holds && pids.contains(&20))
    };
    assert!(refusal(false, "20:77:B", true).unwrap().contains("--test-threads=1"));
    assert!(refusal(true, "", true).unwrap().contains("FABRIC_JOB unset"));
    assert!(refusal(true, "40:77:B", true).unwrap().contains("not an ancestor"));
    assert!(refusal(true, "20:78:B", true).unwrap().contains("stale"));
    assert!(refusal(true, "20:77:B", false).unwrap().contains("--hostmem-lock"));
    assert_eq!(refusal(true, "20:77:B", true), None);
}

/// The holder scan finds a real shared flock on a file named like the host
/// lock, held by a child `flock(1)` process, and nothing for this process.
#[test]
fn hostmem_lock_holders_reads_proc_locks() {
    let dir = std::env::temp_dir().join(format!("session-cache-hw-lock-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lock = dir.join(HOSTMEM_LOCK);
    std::fs::write(&lock, b"").unwrap();
    // `flock` holds the lock while its `cat` child waits on our stdin pipe,
    // which closing below ends.
    let Ok(mut child) = std::process::Command::new("flock")
        .arg("-s")
        .arg(&lock)
        .arg("cat")
        .stdin(std::process::Stdio::piped())
        .spawn()
    else {
        eprintln!("skip: flock(1) unavailable");
        return;
    };
    let pid = child.id();
    let mut held = Vec::new();
    for _ in 0..100 {
        held = hostmem_lock_holders(std::iter::once(pid));
        if !held.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(hostmem_lock_holders(std::iter::once(std::process::id())).is_empty());
    drop(child.stdin.take());
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(held, vec![(pid, "READ".to_string())]);
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
/// forward and a session cache of `budget` bytes, with a disk tier in `disk`
/// when given. `turn_snapshots` sets message-end snapshots explicitly,
/// whatever `HIPFIRE_QWEN4_TURN_SNAPSHOTS` says (the `<|im_end|>` token is
/// still the caller's `set_turn_end_token`).
fn load(turn_snapshots: bool, budget: u64, disk: Option<&Path>) -> Loaded {
    require_fabric_hold();
    let model = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name qwen3.8-flash-next-gptq3.mq4"));
    let mut hfq = HfqFile::open(&model).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let receipt = admit_hfqm_artifact(&hfq).expect("admission");
    let mut gpu = Gpu::init().expect("GPU");
    let domain = CacheDomain::for_model(&hfq, &tokenizer, None, "qwen4", gpu.device_id);
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
        &gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .expect("scratch plan");
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
        &receipt.manifest.identity.recipe,
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
    bundle
        .attach_forward(&mut gpu, MAX_SEQ, scratch_mode)
        .expect("forward");
    let cache = SessionCache::new(domain, budget);
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
    } = load(false, u64::MAX >> 1, None);
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

/// The resident snapshot alone, as for a 192K prompt past an 8 GiB pool: with
/// a budget no pool snapshot fits, a prompt sharing three chunks with the
/// previous one restores them over the rows the live state still holds
/// (rows above them belong to another prompt) and must match its cold
/// prefill byte for byte, on AR and on native MTP.
#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_SESSION_CACHE_MODEL"]
fn resident_restore_matches_cold_on_flash_next() {
    let Loaded {
        mut bundle,
        mut gpu,
        vocab,
        ..
    } = load(false, 1, None);
    let chunk = bundle.spec_chunk_rows().expect("chunk rows");
    let logits = gpu.zeros(&[vocab], DType::F32).expect("logits");
    let a = prompt(0xa, 3 * chunk + 300);
    let mut a2 = a[..3 * chunk + 100].to_vec();
    a2.extend(prompt(0xc, 150));

    let cold = run(&mut bundle, &mut gpu, &logits, &a2, 0);
    run(&mut bundle, &mut gpu, &logits, &a, 0);
    assert_eq!(stored_bytes(&bundle), 0, "no pool snapshot fits the budget");
    let reused = bundle.session_plan(&a2, SessionRoute::Ar);
    assert_eq!(reused, 3 * chunk, "plan must offer the resident snapshot");
    let (digests, warm_logits, warm_ids) = run(&mut bundle, &mut gpu, &logits, &a2, reused);
    assert_same_state("AR resident", &cold.0, &digests);
    assert!(
        warm_logits == cold.1,
        "AR resident: restored final logits differ from cold"
    );
    assert_eq!(warm_ids, cold.2, "AR resident");
    // A reset outside the cache rewrites the live rows' meaning: dropped.
    bundle.reset(&mut gpu).expect("reset");
    assert_eq!(bundle.session_plan(&a2, SessionRoute::Ar), 0);

    bundle.attach_mtp(&mut gpu, MAX_SEQ).expect("MTP head");
    let mut drafter = Qwen4MtpDrafter::new(3, MAX_SEQ, None);
    let cold = run_mtp(&mut bundle, &mut gpu, &mut drafter, &a2, 0);
    run_mtp(&mut bundle, &mut gpu, &mut drafter, &a, 0);
    let reused = bundle.session_plan(&a2, SessionRoute::Mtp);
    assert_eq!(
        reused,
        3 * chunk,
        "MTP plan must offer the resident snapshot"
    );
    let (digests, seed) = run_mtp(&mut bundle, &mut gpu, &mut drafter, &a2, reused);
    assert_same_state("MTP resident", &cold.0, &digests);
    assert_eq!(seed, cold.1, "MTP resident seed");

    gpu.free_tensor(logits).expect("free logits");
    bundle.free_gpu(&mut gpu).expect("free bundle");
}

/// Lever 1: the tail snapshot. A cold prompt A ends with a user message, its
/// final `<|im_end|>` and an assistant header; with the turn end token set
/// (message-end snapshots off) its cold prefill also splits and snapshots at
/// the index BEFORE that final `<|im_end|>`, at least the test's 256-row minimum gap
/// past the previous chunk boundary. Prompt B extends A's last user message
/// (A up to that index, 200 new content tokens, then A's template suffix from
/// the `<|im_end|>` on, so B's last `<|im_end|>` is at `tail + 200`) and must
/// plan exactly A's index. B's own tail split is skipped: 200 rows past the
/// restored tail is under that 256-row gap (and a turn restored from a tail
/// never splits again). The tail schedule is
/// session-exact, not cold-exact, so the restored B is compared with a cold B
/// split manually at the same index (chunk boundary, then the index, then the
/// rest as a live continuation), never with the unsplit cold schedule. A
/// restore after the tail adds nothing to the pool. Runs with the tail
/// snapshot on (`HIPFIRE_QWEN4_TAIL_SNAPSHOT` unset) and the thresholds
/// lowered to a 256-row gap and no minimum prompt (`set_tail_snapshot_limits`).
#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_SESSION_CACHE_MODEL"]
fn tail_snapshot_restore_matches_split_cold_on_flash_next() {
    let Loaded {
        mut bundle,
        mut gpu,
        tokenizer,
        vocab,
        ..
    } = load(false, u64::MAX >> 1, None);
    let im_end = tokenizer
        .special_token_id("<|im_end|>")
        .expect("tokenizer has <|im_end|>");
    bundle.set_turn_end_token(Some(im_end));
    // The production thresholds (1024-row gap, 16384-token prompt) would skip
    // a split this short; keep the test prompts small.
    bundle.set_tail_snapshot_limits(256, 0);
    assert!(
        !bundle.turn_snapshots_active(),
        "message-end snapshots must stay off: only the tail splits"
    );
    let chunk = bundle.spec_chunk_rows().expect("chunk rows");
    assert!(chunk >= 1024, "test needs B to end before the second chunk");
    let logits = gpu.zeros(&[vocab], DType::F32).expect("logits");

    // The tail index sits 300 rows past the chunk boundary (test minimum
    // gap 256) and A ends in the same chunk. Neither prompt has an earlier
    // `<|im_end|>`.
    let tail = chunk + 300;
    let mut a = prompt(0xa7, tail);
    a.push(im_end);
    a.extend(prompt(0xa8, 40));
    assert_eq!(a.iter().rposition(|&t| t == im_end), Some(tail));
    assert_eq!(a.iter().filter(|&&t| t == im_end).count(), 1);
    // B continues A's last user message with new content and keeps A's
    // template suffix, so its last `<|im_end|>` is 200 rows past the tail; it
    // stays below the second chunk boundary.
    let mut b = a[..tail].to_vec();
    b.extend(prompt(0xb7, 200));
    b.extend_from_slice(&a[tail..]);
    assert_eq!(b.iter().rposition(|&t| t == im_end), Some(tail + 200));
    assert_eq!(b.iter().filter(|&&t| t == im_end).count(), 1);
    assert!(b.len() < 2 * chunk);
    // The gap from the restored tail (200) is under the test minimum
    // (256): B's own tail split is skipped, so B has no boundary after the
    // restore.
    assert_eq!(SessionState::lineage_boundary(&bundle, &b, tail, b.len()), None);

    assert_eq!(bundle.session_plan(&a, SessionRoute::Ar), 0);
    run(&mut bundle, &mut gpu, &logits, &a, 0);
    let stored = stored_bytes(&bundle);
    assert!(stored > 0, "cold A published no snapshot");

    // B plans exactly A's tail index, past A's chunk snapshot.
    let reused = bundle.session_plan(&b, SessionRoute::Ar);
    assert_eq!(reused, tail, "plan must offer A's tail snapshot");
    let warm = run(&mut bundle, &mut gpu, &logits, &b, reused);
    assert_eq!(
        stored_bytes(&bundle),
        stored,
        "a restore after the tail must not publish to the pool"
    );

    // Cold B split manually at A's tail index: the cold prefix up to it, kept
    // live, then the rest continuing in place (no capture, no tail).
    bundle
        .prefill_final(&mut gpu, &b[..tail], 0, &logits)
        .expect("split prefix prefill");
    bundle.session_commit_live(&b[..tail]);
    assert_eq!(
        bundle.session_plan(&b, SessionRoute::Ar),
        tail,
        "the split reference continues the live prefix"
    );
    let cold = run(&mut bundle, &mut gpu, &logits, &b, tail);
    println!(
        "AR tail split-cold ids {:?}\nAR tail warm ids {:?}",
        cold.2, warm.2
    );
    assert_same_state("AR tail", &cold.0, &warm.0);
    assert!(
        warm.1 == cold.1,
        "tail-restored final logits differ from the same-split cold prefill"
    );
    assert_eq!(warm.2, cold.2, "AR tail ids");

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
        let mut loaded = load(false, u64::MAX >> 1, Some(&dir));
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
        load(false, u64::MAX >> 1, Some(&dir))
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
    } = load(true, u64::MAX >> 1, None);
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

/// A native MTP turn the AR floor retired still commits live. Retired AR
/// windows (`lab_ar_window(.., false)`, the floor's head-free route) advance
/// the target alone and owe their head rows; the terminal flush of the
/// pending token (`mtp_forced_advance`) appends them teacher-forced first, so
/// head and target end together, `session_commit_live` keeps the state and
/// the conversation's next turn prefills only its suffix in place. Before
/// the fix the flush advanced the target alone and the head stayed behind,
/// so the commit dropped the live state and the next turn restored a chunk
/// snapshot instead. One owed row takes the per-row Append, more take the
/// batched decode fill (where it runs). The caught-up state (target and
/// head) must equal the same tokens teacher-forced with the head synced
/// throughout, and both must continue the next turn to the same seed.
///
/// Built only with `--features lab` (`lab_ar_window`):
/// `cargo test -p hipfire-arch-qwen4 --release --features lab --test
/// session_cache_hw -- --ignored retired_turn_commits_live_on_flash_next`.
#[cfg(feature = "lab")]
#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_SESSION_CACHE_MODEL; build with --features lab"]
fn retired_turn_commits_live_on_flash_next() {
    use hipfire_runtime::spec::SpecRequestConfig;
    let Loaded {
        mut bundle,
        mut gpu,
        ..
    } = load(false, u64::MAX >> 1, None);
    bundle.attach_mtp(&mut gpu, MAX_SEQ).expect("MTP head");
    let mut drafter = Qwen4MtpDrafter::new(3, MAX_SEQ, None);
    let never = || false;
    let a = prompt(0x7e7, 1500);
    let mut mismatches = Vec::new();
    for owed in [1usize, 2, 37] {
        // The retired turn: cold prefill, `owed` head-free AR windows, then
        // the terminal flush of the pending token.
        MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
        let seed = drafter
            .mtp_prefill(&mut gpu, &mut bundle, &a, &a, 0, false, &never)
            .and_then(|outcome| outcome.into_first_token())
            .expect("MTP prefill");
        let mut emitted = vec![seed];
        for i in 0..owed {
            let window = drafter
                .lab_ar_window(&mut gpu, &mut bundle, a.len() + i, emitted[i], false)
                .expect("retired AR window");
            emitted.extend(window.committed);
            assert_eq!(
                SessionState::live_position(&bundle, SessionRoute::Mtp),
                None,
                "owed {owed}: the head lags while it owes rows"
            );
        }
        assert_eq!(
            emitted.len(),
            owed + 1,
            "owed {owed}: one token per AR window"
        );
        let flushed = MtpDrafter::mtp_forced_advance(
            &mut drafter,
            &mut gpu,
            &mut bundle,
            &emitted[owed..],
            a.len() + owed,
            &never,
        )
        .expect("terminal flush");
        assert!(
            flushed,
            "owed {owed}: the drafter advances the retired flush itself"
        );
        let consumed: Vec<u32> = a.iter().chain(&emitted).copied().collect();
        assert_eq!(
            SessionState::live_position(&bundle, SessionRoute::Mtp),
            Some(consumed.len()),
            "owed {owed}: head and target end at the consumed history"
        );
        let retired = state_digests(&mut bundle, &mut gpu, SessionRoute::Mtp, consumed.len());
        let next: Vec<u32> = consumed.iter().copied().chain(prompt(0x2e7, 64)).collect();
        bundle.session_commit_live(&consumed);
        assert_eq!(
            bundle.session_plan(&next, SessionRoute::Mtp),
            consumed.len(),
            "owed {owed}: the next turn continues live"
        );
        let live_seed = drafter
            .mtp_prefill(
                &mut gpu,
                &mut bundle,
                &next,
                &next[consumed.len()..],
                consumed.len(),
                true,
                &never,
            )
            .and_then(|outcome| outcome.into_first_token())
            .expect("live prefill after the retired turn");
        bundle.session_commit();

        // Reference: the same tokens teacher-forced with the head synced.
        MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
        let reference_seed = drafter
            .mtp_prefill(&mut gpu, &mut bundle, &a, &a, 0, false, &never)
            .and_then(|outcome| outcome.into_first_token())
            .expect("reference MTP prefill");
        assert_eq!(reference_seed, seed, "owed {owed}: cold seed");
        assert!(
            MtpDrafter::mtp_forced_advance(
                &mut drafter,
                &mut gpu,
                &mut bundle,
                &emitted,
                a.len(),
                &never
            )
            .expect("teacher-forced reference"),
            "owed {owed}: the reference advance runs on the drafter"
        );
        let reference = state_digests(&mut bundle, &mut gpu, SessionRoute::Mtp, consumed.len());
        bundle.session_commit_live(&consumed);
        assert_eq!(
            bundle.session_plan(&next, SessionRoute::Mtp),
            consumed.len(),
            "owed {owed}: the reference continues live"
        );
        let reference_live_seed = drafter
            .mtp_prefill(
                &mut gpu,
                &mut bundle,
                &next,
                &next[consumed.len()..],
                consumed.len(),
                true,
                &never,
            )
            .and_then(|outcome| outcome.into_first_token())
            .expect("live prefill after the reference turn");
        bundle.session_commit();

        let parts: Vec<usize> = (0..reference.len().max(retired.len()))
            .filter(|&i| reference.get(i) != retired.get(i))
            .collect();
        println!(
            "owed {owed}: consumed {} live seeds retired {live_seed} reference {reference_live_seed}; \
             state parts differing from the reference: {parts:?} of {}",
            consumed.len(),
            reference.len()
        );
        if !parts.is_empty() || live_seed != reference_live_seed {
            mismatches.push((owed, parts, live_seed, reference_live_seed));
        }
    }
    assert!(
        mismatches.is_empty(),
        "caught-up retired turns differ from the teacher-forced reference \
         (owed, differing state parts (0 = metadata), live seed, reference live seed): {mismatches:?}"
    );
    bundle.free_gpu(&mut gpu).expect("free bundle");
}

/// A terminal stop inside a committed window (a stop sequence the emitter
/// observes after the window committed more rows than it consumed) is
/// repaired in place and commits live. The window is an n-gram takeover over
/// the target's own greedy continuation, so it accepts every candidate and
/// commits four tokens; the emitter consumes only the first `n` of them.
/// `Speculator::repair_terminal_prefix` restores the pre-window target and head
/// tickets the window retained and replays the seed plus the first `n - 1`
/// consumed tokens; the pending token is then flushed as the caller does. The
/// conversation must continue live (no re-prefill), and its next turn must
/// seed exactly as the same tokens teacher-forced do.
///
/// `stop_sequences` is set, so `HIPFIRE_MTP_INCREMENTAL=1` runs the same test
/// on the interleaved takeover route (which snapshots only then); the default
/// runs the batched takeover. Built only with `--features lab`
/// (`lab_ar_window` gives the greedy stream):
/// `cargo test -p hipfire-arch-qwen4 --release --features lab --test
/// session_cache_hw -- --ignored terminal_stop_inside_window_repairs_live_on_flash_next`.
#[cfg(feature = "lab")]
#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_SESSION_CACHE_MODEL; build with --features lab"]
fn terminal_stop_inside_window_repairs_live_on_flash_next() {
    use hipfire_runtime::spec::SpecRequestConfig;
    let Loaded {
        mut bundle,
        mut gpu,
        ..
    } = load(false, u64::MAX >> 1, None);
    bundle.attach_mtp(&mut gpu, MAX_SEQ).expect("MTP head");
    let mut drafter = Qwen4MtpDrafter::new(3, MAX_SEQ, None);
    let never = || false;
    let cfg = SpecRequestConfig {
        stop_sequences: true,
        ..SpecRequestConfig::default()
    };
    let a = prompt(0x7e7, 1500);

    // The target's greedy continuation: the seed and four more tokens, from
    // head-free AR windows.
    MtpDrafter::configure_request(&mut drafter, cfg);
    let seed = drafter
        .mtp_prefill(&mut gpu, &mut bundle, &a, &a, 0, false, &never)
        .and_then(|outcome| outcome.into_first_token())
        .expect("MTP prefill");
    let mut stream = vec![seed];
    for i in 0..4 {
        let window = drafter
            .lab_ar_window(&mut gpu, &mut bundle, a.len() + i, stream[i], false)
            .expect("AR window");
        stream.extend(window.committed);
    }
    assert_eq!(stream.len(), 5);

    let mut mismatches = Vec::new();
    for n in 1..=3usize {
        MtpDrafter::configure_request(&mut drafter, cfg);
        let cold_seed = drafter
            .mtp_prefill(&mut gpu, &mut bundle, &a, &a, 0, false, &never)
            .and_then(|outcome| outcome.into_first_token())
            .expect("MTP prefill");
        assert_eq!(cold_seed, seed, "n {n}: cold seed");
        let window = drafter
            .mtp_takeover_step(
                &mut gpu,
                &mut bundle,
                a.len(),
                seed,
                &[seed],
                &stream[1..4],
                u32::MAX,
            )
            .expect("takeover window");
        assert_eq!(
            window.committed,
            stream[1..5].to_vec(),
            "n {n}: the window accepts the whole greedy continuation"
        );
        let consumed = &window.committed[..n];
        let repaired = MtpDrafter::mtp_repair_terminal_prefix(
            &mut drafter,
            &mut gpu,
            &mut bundle,
            a.len(),
            seed,
            consumed,
        )
        .expect("terminal repair");
        assert!(repaired, "n {n}: the retained window repairs the stop");
        assert!(
            MtpDrafter::mtp_forced_advance(
                &mut drafter,
                &mut gpu,
                &mut bundle,
                &consumed[n - 1..],
                a.len() + n,
                &never
            )
            .expect("terminal flush"),
            "n {n}: the drafter advances the flush itself"
        );
        let history: Vec<u32> = a
            .iter()
            .chain(&[seed])
            .chain(consumed)
            .copied()
            .collect();
        assert_eq!(
            SessionState::live_position(&bundle, SessionRoute::Mtp),
            Some(history.len()),
            "n {n}: head and target end at the consumed history"
        );
        let repaired_state = state_digests(&mut bundle, &mut gpu, SessionRoute::Mtp, history.len());
        let next: Vec<u32> = history.iter().copied().chain(prompt(0x2e7, 64)).collect();
        bundle.session_commit_live(&history);
        assert_eq!(
            bundle.session_plan(&next, SessionRoute::Mtp),
            history.len(),
            "n {n}: the next turn continues live"
        );
        let live_seed = drafter
            .mtp_prefill(
                &mut gpu,
                &mut bundle,
                &next,
                &next[history.len()..],
                history.len(),
                true,
                &never,
            )
            .and_then(|outcome| outcome.into_first_token())
            .expect("live prefill after the repaired turn");
        bundle.session_commit();

        // Reference: the same history teacher-forced.
        MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
        drafter
            .mtp_prefill(&mut gpu, &mut bundle, &a, &a, 0, false, &never)
            .expect("reference MTP prefill");
        let forced: Vec<u32> = std::iter::once(seed).chain(consumed.iter().copied()).collect();
        assert!(
            MtpDrafter::mtp_forced_advance(
                &mut drafter,
                &mut gpu,
                &mut bundle,
                &forced,
                a.len(),
                &never
            )
            .expect("teacher-forced reference"),
            "n {n}: the reference advance runs on the drafter"
        );
        let reference = state_digests(&mut bundle, &mut gpu, SessionRoute::Mtp, history.len());
        bundle.session_commit_live(&history);
        let reference_seed = drafter
            .mtp_prefill(
                &mut gpu,
                &mut bundle,
                &next,
                &next[history.len()..],
                history.len(),
                true,
                &never,
            )
            .and_then(|outcome| outcome.into_first_token())
            .expect("live prefill after the reference turn");
        bundle.session_commit();

        let parts: Vec<usize> = (0..reference.len().max(repaired_state.len()))
            .filter(|&i| reference.get(i) != repaired_state.get(i))
            .collect();
        println!(
            "n {n}: history {} live seeds repaired {live_seed} reference {reference_seed}; \
             state parts differing from the teacher-forced reference: {parts:?} of {}",
            history.len(),
            reference.len()
        );
        if live_seed != reference_seed {
            mismatches.push((n, parts, live_seed, reference_seed));
        }
    }
    assert!(
        mismatches.is_empty(),
        "a repaired terminal differs from the teacher-forced reference \
         (n, differing state parts (0 = metadata), live seed, reference live seed): {mismatches:?}"
    );
    bundle.free_gpu(&mut gpu).expect("free bundle");
}
