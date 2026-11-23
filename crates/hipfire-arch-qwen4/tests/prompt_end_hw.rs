// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! #1198: an exact retry of a Flash-Next native MTP prompt restores its prompt
//! end and prefills nothing, byte-equal to the cold prefill.
//!
//! `#[ignore]`d: it needs one real HIP GPU and a Flash-Next artifact named by
//! `HIPFIRE_PROMPT_END_MODEL` (e.g. `qwen3.8-flash-next-gptq3.mq4`). It loads
//! through the product loader (`hipfire_loader::load_qwen4_rank`, native MTP,
//! the serve's session cache, expert hot set and host-mapped experts), so it
//! runs on a 32 GB card, unlike `session_cache_hw`, which places every weight
//! on the device. Run it inside a fabric hold that declares the host memory and
//! holds the Flash-Next host lock, one test thread:
//!
//! ```text
//! HIPFIRE_PROMPT_END_MODEL=$M fabric-run.sh --class correct --card A --mem-gb 30 \
//!   --hostmem-gb 48 --hostmem-lock --max-secs 1800 -- prompt_end_hw --ignored --test-threads=1
//! ```
//!
//! A cold native MTP prefill of A (13,585 tokens, the #1198 repro's length)
//! keeps its prompt end; decode then moves the live state on (a forced advance
//! of the seed and 40 more tokens, as accepted windows would). The exact retry
//! of A must plan the whole prompt and prefill nothing, yet leave the state
//! (target, head with its boundary hidden, draft policy: a SHA-256 of the
//! metadata, every fixed part and the valid rows of every row stream), the
//! last row's logits (spec logits row 0, what a sampled first draw reads) and
//! the seed byte-equal to the cold prefill; the same continuation (which pairs
//! its first head row with the restored hidden) must end in the same state. A
//! prompt differing in its last token plans below its end.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter;
use hipfire_arch_qwen4::Qwen4KvBackend;
use hipfire_loader::{load_qwen4_rank, Qwen4RankShared, Qwen4SessionSeed};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
use hipfire_runtime::session_cache::{SessionRoute, SessionState, SnapshotParts};
use hipfire_runtime::spec::MtpDrafter;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use sha2::{Digest, Sha256};

const MODEL_ENV: &str = "HIPFIRE_PROMPT_END_MODEL";
const MAX_SEQ: usize = 32768;
const MTP_K: usize = 3;

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

/// Load the model named by `HIPFIRE_PROMPT_END_MODEL` as `hipfire serve`
/// loads it, with native MTP and the serve's session cache.
fn load(model: &str) -> (Qwen4Bundle, Gpu) {
    let tokenizer = {
        let hfq = HfqFile::open(std::path::Path::new(model)).expect("open model");
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer")
    };
    hipfire_loader::prepare_host_memory_for(std::path::Path::new(model));
    let mut gpu = Gpu::init().expect("GPU");
    let mut source = ModelSource::from_path(model).expect("model source");
    hipfire_loader::admission::resolve_qwen4_ple_rows(&mut source, model, &gpu.arch)
        .expect("PLE rows");
    let backend = Qwen4KvBackend::automatic(&gpu);
    let vmm_granularity = match backend {
        Qwen4KvBackend::Vmm => gpu.vmm_recommended_granularity().expect("VMM granularity"),
        Qwen4KvBackend::Legacy => 1,
    };
    let shared = Qwen4RankShared {
        batch: Default::default(),
        path: model.to_string(),
        max_seq: MAX_SEQ,
        sequence: None,
        vision_reserve_bytes: 0,
        backend,
        vmm_granularity,
        kv_mode: hipfire_runtime::config::get().kv_mode.clone(),
        state_quant: None,
        spec: SpecLoadCfg::default(),
        native_mtp: true,
        max_k: MTP_K,
    };
    let session = Qwen4SessionSeed {
        tokenizer: &tokenizer,
        chat_template: None,
    };
    let loaded =
        load_qwen4_rank(&mut gpu, None, &shared, None, Some(session), source).expect("load");
    assert!(
        loaded.native_mtp,
        "the product load declined the native MTP head"
    );
    assert!(
        loaded.bundle.session_cache().is_some(),
        "the load attached no session cache"
    );
    (loaded.bundle, gpu)
}

/// SHA-256 of every part of the live state at `position` on the MTP route:
/// the metadata, each fixed part, then the valid rows of each row stream.
fn state_digests(bundle: &mut Qwen4Bundle, gpu: &mut Gpu, position: usize) -> Vec<[u8; 32]> {
    let SnapshotParts { meta, layout } =
        SessionState::snapshot_parts(bundle, SessionRoute::Mtp, position).expect("state parts");
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

/// Fail with the index of the first state part that differs.
fn assert_same_state(name: &str, cold: &[[u8; 32]], warm: &[[u8; 32]]) {
    assert_eq!(cold.len(), warm.len(), "{name}: state part count");
    if let Some(part) = (0..cold.len()).find(|&i| cold[i] != warm[i]) {
        panic!(
            "{name}: state part {part} of {} differs from cold (0 = metadata)",
            cold.len()
        );
    }
}

/// Native MTP prefill of `tokens` reusing `reused` cached tokens, then
/// commit. Returns the state digests and the seed.
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
    let digests = state_digests(bundle, gpu, tokens.len());
    bundle.session_commit();
    (digests, seed)
}

/// Bits of spec logits row 0.
fn row0(bundle: &Qwen4Bundle, gpu: &Gpu) -> Vec<u32> {
    let mut host = Vec::new();
    bundle
        .spec_row_logits(gpu, 0, &mut host)
        .expect("logits row");
    assert_eq!(host.len(), bundle.config.vocab_size);
    host.iter().map(|value| value.to_bits()).collect()
}

/// Force the seed and `tail` after `at` (as accepted windows would) and
/// return the state digests at the end.
fn continue_after(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    drafter: &mut Qwen4MtpDrafter,
    at: usize,
    seed: u32,
    tail: &[u32],
) -> Vec<[u8; 32]> {
    let forced: Vec<u32> = std::iter::once(seed).chain(tail.iter().copied()).collect();
    assert!(
        MtpDrafter::mtp_forced_advance(drafter, gpu, bundle, &forced, at, &|| false)
            .expect("continuation"),
        "the drafter advances the continuation itself"
    );
    state_digests(bundle, gpu, at + forced.len())
}

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_PROMPT_END_MODEL (Flash-Next), inside a fabric hold"]
fn prompt_end_retry_matches_cold_on_flash_next() {
    let model = std::env::var(MODEL_ENV)
        .unwrap_or_else(|_| panic!("{MODEL_ENV} must name a Flash-Next artifact"));
    let (mut bundle, mut gpu) = load(&model);
    let chunk = bundle.spec_chunk_rows().expect("chunk rows");
    let mut drafter = Qwen4MtpDrafter::new(MTP_K, MAX_SEQ, None);
    let a = prompt(0xe, 13_585);
    let tail = prompt(0xd, 40);
    println!(
        "chunk={chunk} prompt={} state={:?}",
        a.len(),
        bundle.state_format()
    );

    let (cold, cold_seed) = run_mtp(&mut bundle, &mut gpu, &mut drafter, &a, 0);
    let cold_logits = row0(&bundle, &gpu);
    let cold_after = continue_after(
        &mut bundle,
        &mut gpu,
        &mut drafter,
        a.len(),
        cold_seed,
        &tail,
    );

    assert_eq!(
        bundle.session_plan(&a, SessionRoute::Mtp),
        a.len(),
        "the exact retry plans the whole prompt"
    );
    let (warm, warm_seed) = run_mtp(&mut bundle, &mut gpu, &mut drafter, &a, a.len());
    println!("MTP prompt end seed cold {cold_seed} warm {warm_seed}");
    assert_same_state("MTP prompt end", &cold, &warm);
    assert_eq!(warm_seed, cold_seed, "MTP prompt end seed");
    assert!(
        row0(&bundle, &gpu) == cold_logits,
        "MTP prompt end: restored last-row logits differ from cold"
    );
    let warm_after = continue_after(
        &mut bundle,
        &mut gpu,
        &mut drafter,
        a.len(),
        warm_seed,
        &tail,
    );
    assert_same_state("MTP prompt end continuation", &cold_after, &warm_after);

    let mut edited = a[..a.len() - 1].to_vec();
    edited.push(a[a.len() - 1] ^ 1);
    assert!(
        bundle.session_plan(&edited, SessionRoute::Mtp) < edited.len(),
        "only the exact prompt reuses its end"
    );
    bundle.free_gpu(&mut gpu).expect("free bundle");
}
