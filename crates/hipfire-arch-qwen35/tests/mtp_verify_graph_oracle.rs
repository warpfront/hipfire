// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Eager-vs-replay oracle for the compressed-serial MTP verify-graph route.
//!
//! ONE `#[ignore]`d test with two modes selected by env:
//!
//! - Orchestrator (default): spawns two sequential fresh-process workers via
//!   `current_exe` — eager (`HIPFIRE_VERIFY_GRAPH=0`) then graph
//!   (`HIPFIRE_VERIFY_GRAPH=1`, master `HIPFIRE_GRAPH` inherited) — and
//!   compares every window. Sequential so only one 35B session is resident.
//!   Controls travel as per-`Command` env only; no process-global mutation.
//! - Worker (`HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_ARM=eager|graph`): loads the
//!   graded fixture with an EXPLICIT fixed offload placement (first 16 expert
//!   layers host-mapped, trunk own-weights resident — never
//!   `ModelSlot::load`, which is fully resident and OOMs), runs native k=1/2/3
//!   windows plus teacher-forced takeover windows, a reset/second generation,
//!   and an unload/reload spot-check, writing `digests.log` plus raw `.bin`
//!   sidecars per window into its own unique scratch dir.
//!
//! Byte-exact contract (no tolerance, no NaN skipping, no report-only state):
//! committed ids, accept, advance, and the FULL raw bytes of every family —
//! verify logits/hidden, `prev_hidden`, DN S codes, DN scales, DN conv, DN EF
//! residual, retained trunk KV prefix, MTP KV prefix, GDN tape prefix.
//! Formats are honored, never aliased: F32 families download as F32 and must
//! be finite; F16 EF decodes as F16 and must be finite; Q8 KV families compare
//! as raw bytes with each 34-byte block's F16 scale checked finite (int8 codes
//! are naturally finite); DN S compares as raw int8 codes with its separate
//! F32 scales checked finite. EF must be ACTIVE (non-empty) under the default
//! Q8 state build on both arms.
//!
//! Takeover windows stay inside the validated 2..4-row envelope (never the old
//! 8+ spill): full / partial / zero accept derived from a deterministic k=0
//! reference plus one wrong token, followed by native continuation windows
//! proving retained state after rollback.
//!
//! ```bash
//! HIPFIRE_MTP_BYTE_IDENTITY_MODEL=~/.hipfire/models/qwen3.6-35b-a3b.mq4p \
//! HIPFIRE_MTP_BYTE_IDENTITY_HEAD=~/.hipfire/models/qwen3.6-35b-a3b.mtp \
//! cargo test --release -p hipfire-arch-qwen35 \
//!   --test mtp_verify_graph_oracle -- --ignored --nocapture
//! ```

#![allow(clippy::all)]

use hipfire_arch_qwen35::mtp_head::{self, MtpKvMode};
use hipfire_arch_qwen35::mtp_spec::{
    prefill_trunk_and_mtp_cache, spec_step_mtp_compressed_serial_with_k,
    spec_step_mtp_compressed_serial_with_takeover_candidates, MtpPromptRoute, MtpSpecState,
};
use hipfire_arch_qwen35::qwen35::{
    config_from_hfq, host_mapped_expert_accounting, load_weights, DeltaNetState, HfqSource,
    LayerType, Layout, Qwen35Scratch, StateQuant,
};
use hipfire_arch_qwen35::speculative::{KvMode, ModelSlot, ModelSlotConfig};
use hipfire_dispatch::cpu_exec_counters;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::offload::{ExpertResidency, Placement};
use rdna_compute::{Gpu, GpuTensor};
use std::path::{Path, PathBuf};

const CTX: usize = 4096;
const MAX_N: usize = 3;
const VERIFY_CAP: usize = 4;
const NATIVE_WINDOWS_PER_K: usize = 5;
const REF_STEPS: usize = 24;
const HOST_EXPERT_LAYERS: usize = 16;
const REPEAT_WINDOW: usize = 128;
const ARM_ENV: &str = "HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_ARM";
const OUT_ENV: &str = "HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_OUT";
const OUT_ROOT_ENV: &str = "HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_OUT_ROOT";
const FAMS: [&str; 10] = [
    "logits",
    "hidden",
    "prev",
    "dn_s",
    "dn_scales",
    "dn_conv",
    "dn_ef",
    "kv",
    "mtpkv",
    "tape",
];
const PROMPT: &str = "Write a Rust function `fn parse_kv(line: &str) -> Option<(String, String)>` that splits \
a `key=value` line on the first '=', trims both sides, and rejects an empty key. Add three unit tests.";

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn f16_to_f32(b: u16) -> f32 {
    let s = ((b >> 15) & 1) as u32;
    let mut e = ((b >> 10) & 0x1f) as i32;
    let mut m = (b & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // Subnormal: normalize.
            e = 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (s << 31) | (((e + 112) as u32) << 23) | (m << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | (((e + 112) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn assert_all_finite_f32(tag: &str, v: &[f32]) {
    assert!(!v.is_empty(), "{tag}: empty float array");
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    assert!(bad == 0, "{tag}: {bad}/{} non-finite f32", v.len());
}

fn assert_all_finite_f16(tag: &str, bytes: &[u8]) {
    assert!(!bytes.is_empty(), "{tag}: empty f16 bytes");
    assert!(bytes.len() % 2 == 0, "{tag}: ragged f16 bytes");
    let mut bad = 0usize;
    for c in bytes.chunks_exact(2) {
        if !f16_to_f32(u16::from_le_bytes([c[0], c[1]])).is_finite() {
            bad += 1;
        }
    }
    assert!(bad == 0, "{tag}: {bad} non-finite f16 lanes");
}

/// Q8_0 block layout: 2-byte f16 scale + 32 int8 codes. Codes are naturally
/// finite; every scale must decode finite. Not a float array — never read as
/// f32 lanes.
fn assert_q8_blocks(tag: &str, bytes: &[u8]) {
    assert!(!bytes.is_empty(), "{tag}: empty q8 bytes");
    assert!(
        bytes.len() % 34 == 0,
        "{tag}: ragged q8 len {}",
        bytes.len()
    );
    for (i, blk) in bytes.chunks_exact(34).enumerate() {
        let s = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        assert!(s.is_finite(), "{tag} block {i}: non-finite scale");
    }
}

fn raw_prefix(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u8> {
    assert!(
        n <= t.buf.size(),
        "prefix {n} exceeds allocation {}",
        t.buf.size()
    );
    let mut out = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("dtoh prefix");
    out
}

fn raw_all(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    raw_prefix(gpu, t, t.buf.size())
}

fn model_paths() -> Option<(PathBuf, PathBuf)> {
    let model: PathBuf = std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_MODEL")
        .ok()?
        .into();
    let head: PathBuf = std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_HEAD")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model.with_extension("mtp"));
    Some((model, head))
}

fn unique_out_dir(root: Option<PathBuf>, arm: &str) -> PathBuf {
    let base = root.unwrap_or_else(|| std::env::temp_dir().join("vg-oracle"));
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    base.join(format!("{}-{}-{}", ms, std::process::id(), arm))
}

/// Test-only offloaded session assembly. Mirrors `ModelSlot::load` field by
/// field, except weights go through the production placement loader with an
/// EXPLICIT fixed placement: first 16 expert layers host-mapped, every layer's
/// own weights resident. No production API edits; no `ModelSlot::load` (fully
/// resident — OOMs on this fixture).
fn load_offloaded_session(
    gpu: &mut Gpu,
    model: &Path,
    name: &str,
    lines: &mut Vec<String>,
) -> ModelSlot {
    let mut hfq = HfqFile::open(model).expect("open trunk");
    let config = config_from_hfq(&hfq).expect("trunk config");
    assert!(config.num_experts > 0, "oracle fixture must be MoE");
    let n = config.n_layers;
    assert!(n > HOST_EXPERT_LAYERS, "fixture has only {n} layers");

    let mut placement = Placement::all_device(n);
    for r in placement
        .expert_residency
        .iter_mut()
        .take(HOST_EXPERT_LAYERS)
    {
        *r = ExpertResidency::HostMapped;
    }
    assert_eq!(placement.host_expert_layers(), HOST_EXPERT_LAYERS);
    assert_eq!(
        placement.host_layers(),
        0,
        "trunk own-weights must stay resident"
    );
    // Host-weight count drives the model-level CPU-offload flag exactly as the
    // production loader sets it (`Architecture::load_weights_with_placement` /
    // the slots loader): a host-mapped weight that the CPU may multiply is the
    // host-sync point that must keep hipGraph capture (and the narrow verify
    // graph) off. `placement.host_layers()` is 0 here by construction, so this
    // is `HOST_EXPERT_LAYERS`; without `memory.offload_exec=cpu` the flag stays
    // false and every existing arm is byte-for-byte unchanged.
    let host_weight_layers = placement.host_layers() + placement.host_expert_layers();
    let layout = Layout::single(n).with_placement(placement);

    let weights = {
        let mut src = HfqSource::new(&mut hfq, &config);
        load_weights(&mut src, std::slice::from_mut(gpu), &layout).expect("load weights")
    };
    let (host_tensors, host_bytes) = host_mapped_expert_accounting(&weights.layers);
    assert!(host_tensors > 0, "fixed placement spilled 0 expert tensors");
    gpu.set_host_cpu_weights(hipfire_dispatch::cpu_offload_active(host_weight_layers));
    // Route into digests.log (not just stderr) — the orchestrator reads these.
    let host_line = format!(
        "HOST tensors={host_tensors} bytes={host_bytes} host_expert_layers={HOST_EXPERT_LAYERS} \
host_cpu_weights={} cpu_offload_active={}",
        u8::from(gpu.owns_host_cpu_weights()),
        u8::from(hipfire_dispatch::cpu_offload_active(host_weight_layers)),
    );
    eprintln!("[vg-oracle] {host_line}");
    lines.push(host_line);
    let place_line = "PLACEMENT explicit-fixed-first16 not-auto".to_string();
    eprintln!("[vg-oracle] {place_line}");
    lines.push(place_line);

    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let kv_cache =
        KvCache::new_gpu_q8_filtered(gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, CTX)
            .expect("q8 kv");
    let dn_state =
        DeltaNetState::new_with_quant(gpu, &config, StateQuant::Q8).expect("q8 dn state");
    // Default Q8 build runs with F16 error-feedback; the orchestrator enforces
    // non-empty EF agreement across arms (both-active normally, both-empty
    // only under an explicit HIPFIRE_DN_STATE_EF=0 opt-out).
    let scratch = Qwen35Scratch::new(gpu, &config, REPEAT_WINDOW).expect("scratch");
    ModelSlot {
        name: name.to_string(),
        hfq,
        config,
        weights,
        kv_cache,
        dn_state,
        scratch,
        slot_config: ModelSlotConfig {
            max_seq: CTX,
            kv_mode: KvMode::Q8,
            repeat_window: REPEAT_WINDOW,
            state_quant: StateQuant::Q8,
        },
        dspark_extract_layers: Vec::new(),
        vision_config: None,
        vision_weights: None,
    }
}

fn free_session(
    gpu: &mut Gpu,
    slot: ModelSlot,
    state: MtpSpecState,
    head: mtp_head::Qwen35MtpHead,
) {
    gpu.invalidate_graph_state();
    state.free_gpu(&mut *gpu);
    gpu.invalidate_weight_caches();
    head.free_gpu(&mut *gpu);
    slot.kv_cache.free_gpu(&mut *gpu).expect("free kv");
    slot.dn_state.free_gpu(&mut *gpu);
    slot.scratch.free_gpu(&mut *gpu).expect("free scratch");
    slot.weights.free_gpu(&mut *gpu);
    gpu.drain_pool();
}

fn build_prompt(tokenizer: &hipfire_runtime::tokenizer::Tokenizer) -> Vec<u32> {
    [
        "<|im_start|>",
        "user",
        "\n",
        PROMPT,
        "<|im_end|>",
        "\n",
        "<|im_start|>",
        "assistant",
        "\n",
        "<think>\n\n</think>\n\n",
    ]
    .iter()
    .flat_map(|s| tokenizer.encode(s))
    .collect()
}

fn prefill_and_seed(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    head: &mtp_head::Qwen35MtpHead,
    state: &mut MtpSpecState,
    prompt: &[u32],
) -> u32 {
    slot.reset_state(gpu).expect("reset trunk");
    state.reset(gpu).expect("reset head");
    prefill_trunk_and_mtp_cache(gpu, slot, head, state, prompt, 0, MtpPromptRoute::ArRoute)
        .expect("prefill");
    let logits = gpu.download_f32(&slot.scratch.logits).expect("seed logits");
    assert_all_finite_f32("prefill-logits", &logits);
    logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i as u32, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// Raw bytes of every compared family in `FAMS` order: verify logits/hidden
/// (live `n_verify` rows), full `prev_hidden`, full DN S codes / scales / conv
/// / EF, retained trunk KV prefix (`end_pos` rows), MTP KV prefix, GDN tape
/// prefix (`n_verify` rows). Formats checked here; equality checked by caller.
fn collect_fams(
    gpu: &Gpu,
    slot: &ModelSlot,
    state: &MtpSpecState,
    n_verify: usize,
    end_pos: usize,
) -> Vec<Vec<u8>> {
    let dim = slot.config.dim;
    let vocab = slot.config.vocab_size;
    let logits = raw_prefix(gpu, &state.verify_logits, n_verify * vocab * 4);
    let hidden = raw_prefix(gpu, &state.verify_hidden, n_verify * dim * 4);
    let prev = raw_all(gpu, &state.prev_hidden);
    assert_all_finite_f32("verify-logits", &{
        assert!(logits.len() % 4 == 0);
        logits
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>()
    });
    assert_all_finite_f32("verify-hidden", &{
        hidden
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>()
    });
    assert_all_finite_f32(
        "prev-hidden",
        &gpu.download_f32(&state.prev_hidden).expect("prev"),
    );

    let mut dn_s = Vec::new();
    let mut dn_scales = Vec::new();
    let mut dn_conv = Vec::new();
    let mut dn_ef = Vec::new();
    for t in &slot.dn_state.s_matrices {
        dn_s.extend_from_slice(&raw_all(gpu, t));
    }
    for t in &slot.dn_state.s_scales {
        dn_scales.extend_from_slice(&raw_all(gpu, t));
    }
    for t in &slot.dn_state.conv_states {
        dn_conv.extend_from_slice(&raw_all(gpu, t));
    }
    for t in &slot.dn_state.s_ef_residual {
        dn_ef.extend_from_slice(&raw_all(gpu, t));
    }
    // DN S is raw int8 codes (naturally finite) + separate per-row F32 scales
    // (checked below), NOT blocked Q8_0: never read as f32 lanes or 34B blocks.
    assert!(!dn_s.is_empty(), "dn-s empty");
    assert_all_finite_f32("dn-scales", &gpu_download_all(gpu, &slot.dn_state.s_scales));
    assert_all_finite_f32(
        "dn-conv",
        &gpu_download_all(gpu, &slot.dn_state.conv_states),
    );
    assert_all_finite_f16("dn-ef", &dn_ef);

    let kv = trunk_kv_prefix(gpu, slot, end_pos);
    assert_q8_blocks("trunk-kv", &kv);
    assert!(kv.iter().any(|&b| b != 0), "trunk kv prefix all zero");
    let mtpkv = mtp_kv_prefix(gpu, state, end_pos);
    assert_q8_blocks("mtp-kv", &mtpkv);
    let tape = tape_prefix(gpu, state, n_verify);
    assert_all_finite_f32("tape", &{
        tape.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>()
    });
    vec![
        logits, hidden, prev, dn_s, dn_scales, dn_conv, dn_ef, kv, mtpkv, tape,
    ]
}

fn gpu_download_all(gpu: &Gpu, ts: &[GpuTensor]) -> Vec<f32> {
    let mut out = Vec::new();
    for t in ts {
        out.extend_from_slice(&gpu.download_f32(t).expect("download f32 fam"));
    }
    out
}

fn trunk_kv_prefix(gpu: &Gpu, slot: &ModelSlot, positions: usize) -> Vec<u8> {
    let kv = &slot.kv_cache;
    let row = kv.n_kv_heads * (kv.head_dim / 32) * 34;
    let mut out = Vec::new();
    for layer in 0..kv.k_gpu.len() {
        for buf in [&kv.k_gpu[layer], &kv.v_gpu[layer]] {
            if buf.buf.size() < row {
                continue;
            }
            out.extend_from_slice(&raw_prefix(gpu, buf, positions * row));
        }
    }
    out
}

fn mtp_kv_prefix(gpu: &Gpu, state: &MtpSpecState, positions: usize) -> Vec<u8> {
    let kv = &state.mtp_kv;
    let row = kv.n_head_kv * (kv.head_dim / 32) * 34;
    let mut out = Vec::new();
    for buf in kv.inner.k_gpu.iter().chain(kv.inner.v_gpu.iter()) {
        if buf.buf.size() < row {
            continue;
        }
        out.extend_from_slice(&raw_prefix(gpu, buf, positions * row));
    }
    out
}

fn tape_prefix(gpu: &Gpu, state: &MtpSpecState, n_verify: usize) -> Vec<u8> {
    let tape = &state.trunk_gdn_tape;
    let mut out = Vec::new();
    for t in &tape.qkv_bufs {
        out.extend_from_slice(&raw_prefix(gpu, t, n_verify * tape.qkv_dim * 4));
    }
    for t in tape.alpha_bufs.iter().chain(tape.beta_bufs.iter()) {
        out.extend_from_slice(&raw_prefix(gpu, t, n_verify * tape.n_v_heads * 4));
    }
    out
}
struct Worker {
    out_dir: PathBuf,
    lines: Vec<String>,
}

impl Worker {
    fn emit(&mut self, s: String) {
        eprintln!("[vg-oracle] {s}");
        self.lines.push(s);
    }

    #[allow(clippy::too_many_arguments)]
    fn snap(
        &mut self,
        gpu: &Gpu,
        slot: &ModelSlot,
        state: &MtpSpecState,
        tag: &str,
        k: usize,
        pos: usize,
        seed: u32,
        committed: &[u32],
        accept: usize,
        advance: usize,
        drafts: usize,
        end_pos: usize,
    ) -> Vec<Vec<u8>> {
        let n_verify = drafts + 1;
        let fams = collect_fams(gpu, slot, state, n_verify, end_pos);
        let mut h = 0xcbf29ce484222325u64;
        for f in &fams {
            h ^= fnv1a(f);
            h = h.wrapping_mul(0x100000001b3);
        }
        let toks = committed
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");
        self.emit(format!(
            "WIN tag={tag} k={k} pos={pos} seed={seed} committed={toks} \
accept={accept} advance={advance} drafts={drafts} n={n_verify} end={end_pos} h={h:016x}"
        ));
        for (name, bytes) in FAMS.iter().zip(fams.iter()) {
            std::fs::write(self.out_dir.join(format!("{tag}-{name}.bin")), bytes)
                .expect("write sidecar");
        }
        fams
    }

    fn replay(&mut self, state: &MtpSpecState, tag: &str) {
        self.emit(format!(
            "REPLAY tag={tag} mode={} captures={} replays={}",
            state.mtp_verify_graph_mode(),
            state.mtp_verify_graph_captures(),
            state.mtp_verify_graph_replays(),
        ));
    }
}

fn run_worker(arm: &str, out_dir: PathBuf) {
    let Some((model, head_path)) = model_paths() else {
        eprintln!("skipping: HIPFIRE_MTP_BYTE_IDENTITY_MODEL not set");
        return;
    };
    assert!(
        model.is_file(),
        "trunk fixture missing: {}",
        model.display()
    );
    assert!(
        head_path.is_file(),
        "mtp sidecar missing: {}",
        head_path.display()
    );
    std::fs::create_dir_all(&out_dir).expect("create oracle out dir");

    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!(
        "[vg-oracle] arm={arm} gpu={} model={}",
        gpu.arch,
        model.display()
    );
    assert_eq!(
        gpu.arch.as_str(),
        "gfx1201",
        "oracle requires single-GPU gfx1201"
    );

    let mut w = Worker {
        out_dir,
        lines: Vec::new(),
    };
    let mut slot = load_offloaded_session(&mut gpu, &model, "vg-oracle", &mut w.lines);
    let head = mtp_head::load_mtp_head(&head_path, &mut gpu, CTX).expect("load head");
    let tokenizer = slot.load_tokenizer().expect("tokenizer");
    let prompt = build_prompt(&tokenizer);
    let n = prompt.len();
    let eos = slot.config.eos_token;
    let vocab = slot.config.vocab_size;
    let mut state = MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(
        &mut gpu,
        &slot,
        &head,
        MAX_N,
        VERIFY_CAP,
        MtpKvMode::Q8,
    )
    .expect("state");
    if let Some(cvs) = head.weights.compressed_vocab_size {
        state
            .mtp_scratch
            .ensure_compressed_logits(&mut gpu, cvs)
            .expect("head compressed logits");
        state
            .ensure_compressed_lm_logits(&mut gpu, cvs)
            .expect("batched compressed logits");
    }
    w.emit(format!("PROMPT n={n} eos={eos}"));

    // Phase A: native windows at n_verify 2/3/4, same prefilled prompt/seed.
    let seed0 = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
    let mut first_k3: Option<(Vec<u32>, Vec<Vec<u8>>)> = None;
    let mut seen: std::collections::BTreeSet<(usize, usize, u32)> =
        std::collections::BTreeSet::new();
    for k in 1..=MAX_N {
        let mut seed = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
        assert_eq!(seed, seed0, "k={k}: prefill seed changed after reset");
        let mut pos = n;
        let rb = state.mtp_verify_graph_replays();
        for win in 0..NATIVE_WINDOWS_PER_K {
            let r = spec_step_mtp_compressed_serial_with_k(
                &mut gpu, &mut slot, &head, &mut state, pos, seed, eos, k,
            )
            .expect("native window");
            assert!(!r.hit_eos, "phase A hit EOS");
            assert_eq!(r.drafts_generated, k);
            let end = pos + r.advance;
            let tag = format!("A-k{k}-{win}");
            let fams = w.snap(
                &gpu,
                &slot,
                &state,
                &tag,
                k,
                pos,
                seed,
                &r.committed,
                r.accept_count,
                r.advance,
                r.drafts_generated,
                end,
            );
            w.replay(&state, &tag);
            assert!(
                seen.insert((k, pos, seed)),
                "k={k}: repeated (pos, seed) input"
            );
            if k == MAX_N && win == 0 {
                first_k3 = Some((r.committed.clone(), fams));
            }
            seed = *r.committed.last().expect("empty commit");
            pos = end;
        }
        if arm == "graph" {
            assert!(
                state.mtp_verify_graph_captures() > 0,
                "graph: no capture at k={k}"
            );
            assert!(
                state.mtp_verify_graph_replays() - rb >= 2,
                "graph: <2 replays at k={k}"
            );
        }
    }
    let (caps_a, reps_a) = (
        state.mtp_verify_graph_captures(),
        state.mtp_verify_graph_replays(),
    );
    if arm == "graph" {
        assert!(caps_a >= 3, "graph: only {caps_a} captures");
        assert!(reps_a >= 6, "graph: only {reps_a} replays");
    } else {
        assert_eq!(caps_a, 0, "eager: {caps_a} captures despite optout");
        assert_eq!(reps_a, 0, "eager: {reps_a} replays despite optout");
    }

    // Phase B: deterministic k=0 reference, then full/partial/zero takeovers
    // inside the 2..4-row envelope, each followed by native continuation.
    let mut toks = vec![prefill_and_seed(
        &mut gpu, &mut slot, &head, &mut state, &prompt,
    )];
    assert_eq!(toks[0], seed0);
    for i in 0..REF_STEPS {
        let r = spec_step_mtp_compressed_serial_with_k(
            &mut gpu,
            &mut slot,
            &head,
            &mut state,
            n + i,
            toks[i],
            eos,
            0,
        )
        .expect("ref step");
        assert!(!r.hit_eos, "reference hit EOS");
        toks.push(r.committed[0]);
    }
    let seed_b = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
    assert_eq!(seed_b, seed0, "re-prefill must reproduce the seed");
    let wrong = |t: u32| {
        let c = if t == 0 { 1 } else { t - 1 };
        if c == eos {
            (c + 1) % vocab as u32
        } else {
            c
        }
    };
    // (label, n_match, append_wrong): rows = matches + wrong? + bonus...
    // drafts = candidates.len(), n_verify = drafts + 1.
    let plan: [(&str, usize, bool); 3] =
        [("full", 2, false), ("partial", 2, true), ("zero", 0, true)];
    let mut idx = 0usize;
    for (label, nmatch, add_wrong) in plan {
        let p = n + idx;
        let mut cands: Vec<u32> = if nmatch == 0 {
            vec![wrong(toks[idx + 1])]
        } else {
            let mut c = toks[idx + 1..=idx + nmatch].to_vec();
            if add_wrong {
                c.push(wrong(toks[idx + nmatch + 1]));
            }
            c
        };
        assert!(
            (2..=4).contains(&(cands.len() + 1)),
            "{label}: outside 2..4-row envelope"
        );
        let r = spec_step_mtp_compressed_serial_with_takeover_candidates(
            &mut gpu, &mut slot, &head, &mut state, p, toks[idx], eos, &cands,
        )
        .expect("takeover window");
        assert_eq!(r.accept_count, nmatch, "{label}: accept mismatch");
        assert_eq!(
            &r.committed[..],
            &toks[idx + 1..=idx + r.advance],
            "{label}: committed drift"
        );
        let end = p + r.advance;
        let tag = format!("B-{label}");
        w.snap(
            &gpu,
            &slot,
            &state,
            &tag,
            cands.len().saturating_sub(1).min(MAX_N).max(1),
            p,
            toks[idx],
            &r.committed,
            r.accept_count,
            r.advance,
            r.drafts_generated,
            end,
        );
        w.replay(&state, &tag);
        idx += r.advance;
        // Native continuation on the retained post-rollback state.
        let c = spec_step_mtp_compressed_serial_with_k(
            &mut gpu, &mut slot, &head, &mut state, end, toks[idx], eos, 2,
        )
        .expect("post-takeover continuation");
        assert!(!c.hit_eos, "{label}: continuation hit EOS");
        assert_eq!(
            &c.committed[..],
            &toks[idx + 1..=idx + c.advance],
            "{label}: continuation drift"
        );
        let cend = end + c.advance;
        let ctag = format!("B-{label}-cont");
        w.snap(
            &gpu,
            &slot,
            &state,
            &ctag,
            2,
            end,
            toks[idx],
            &c.committed,
            c.accept_count,
            c.advance,
            c.drafts_generated,
            cend,
        );
        w.replay(&state, &ctag);
        idx += c.advance;
    }

    // Phase C: reset + second generation from the same prompt/seed.
    let seed_c = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
    assert_eq!(seed_c, seed0, "gen2 reset must reproduce the seed");
    let (mut pos_c, mut seed_c_mut) = (n, seed_c);
    for win in 0..4 {
        let r = spec_step_mtp_compressed_serial_with_k(
            &mut gpu, &mut slot, &head, &mut state, pos_c, seed_c_mut, eos, MAX_N,
        )
        .expect("gen2 window");
        assert!(!r.hit_eos, "gen2 hit EOS");
        let end = pos_c + r.advance;
        let tag = format!("C-3-{win}");
        w.snap(
            &gpu,
            &slot,
            &state,
            &tag,
            MAX_N,
            pos_c,
            seed_c_mut,
            &r.committed,
            r.accept_count,
            r.advance,
            r.drafts_generated,
            end,
        );
        w.replay(&state, &tag);
        seed_c_mut = *r.committed.last().unwrap();
        pos_c = end;
    }

    // Phase D: unload + reload, first k=3 window must match Phase A bytes.
    let (want_toks, want_fams) = first_k3.expect("phase A recorded no k=3 window");
    free_session(&mut gpu, slot, state, head);
    let mut slot2 = load_offloaded_session(&mut gpu, &model, "vg-oracle-reload", &mut w.lines);
    let head2 = mtp_head::load_mtp_head(&head_path, &mut gpu, CTX).expect("reload head");
    let mut state2 = MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(
        &mut gpu,
        &slot2,
        &head2,
        MAX_N,
        VERIFY_CAP,
        MtpKvMode::Q8,
    )
    .expect("reload state");
    if let Some(cvs) = head2.weights.compressed_vocab_size {
        state2
            .mtp_scratch
            .ensure_compressed_logits(&mut gpu, cvs)
            .expect("reload head compressed logits");
        state2
            .ensure_compressed_lm_logits(&mut gpu, cvs)
            .expect("reload batched compressed logits");
    }
    let seed_r = prefill_and_seed(&mut gpu, &mut slot2, &head2, &mut state2, &prompt);
    assert_eq!(seed_r, seed0, "reload prefill must reproduce the seed");
    let r = spec_step_mtp_compressed_serial_with_k(
        &mut gpu,
        &mut slot2,
        &head2,
        &mut state2,
        n,
        seed_r,
        eos,
        MAX_N,
    )
    .expect("reload window");
    assert_eq!(
        r.committed, want_toks,
        "reload committed diverged from Phase A"
    );
    let got = collect_fams(&gpu, &slot2, &state2, r.drafts_generated + 1, n + r.advance);
    assert_eq!(got.len(), want_fams.len(), "reload family count diverged");
    for (i, (g, want)) in got.iter().zip(want_fams.iter()).enumerate() {
        assert_eq!(g, want, "reload family {} diverged from Phase A", FAMS[i]);
    }
    w.emit(format!(
        "RELOAD committed={} match=1",
        r.committed
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    ));
    free_session(&mut gpu, slot2, state2, head2);

    std::fs::write(w.out_dir.join("digests.log"), w.lines.join("\n") + "\n")
        .expect("write digests");
    eprintln!(
        "[vg-oracle] arm={arm} DONE lines={} out={}",
        w.lines.len(),
        w.out_dir.display()
    );
}

fn win_field<'a>(line: &'a str, key: &str) -> &'a str {
    line.split_whitespace()
        .find(|p| p.starts_with(&format!("{key}=")))
        .unwrap_or_else(|| panic!("WIN line missing {key}: {line}"))
        .split_once('=')
        .map(|(_, v)| v)
        .unwrap()
}

fn read_sidecar(dir: &Path, tag: &str, fam: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{tag}-{fam}.bin")))
        .unwrap_or_else(|_| panic!("missing {tag}-{fam}.bin"))
}

fn run_orchestrator() {
    let Some(_) = model_paths() else {
        eprintln!("skipping: HIPFIRE_MTP_BYTE_IDENTITY_MODEL not set");
        return;
    };
    let exe = std::env::current_exe().expect("current_exe");
    let root = std::env::var(OUT_ROOT_ENV).map(PathBuf::from).ok();
    let mut dirs = Vec::new();
    for (arm, vg) in [("eager", "0"), ("graph", "1")] {
        let dir = unique_out_dir(root.clone(), arm);
        eprintln!("[vg-oracle-cmp] spawning {arm} arm (HIPFIRE_VERIFY_GRAPH={vg}) ...");
        let st = std::process::Command::new(&exe)
            .arg("--exact")
            .arg("mtp_verify_graph_eager_vs_replay")
            .arg("--ignored")
            .arg("--nocapture")
            .env(ARM_ENV, arm)
            .env(OUT_ENV, &dir)
            .env("HIPFIRE_VERIFY_GRAPH", vg)
            .status()
            .expect("spawn worker");
        assert!(st.success(), "{arm} worker failed with {st}");
        dirs.push((arm, dir));
    }
    let (eager_dir, graph_dir) = (&dirs[0].1, &dirs[1].1);
    let eager = std::fs::read_to_string(eager_dir.join("digests.log")).expect("read eager digests");
    let graph = std::fs::read_to_string(graph_dir.join("digests.log")).expect("read graph digests");
    let ew: Vec<_> = eager.lines().filter(|l| l.starts_with("WIN ")).collect();
    let gw: Vec<_> = graph.lines().filter(|l| l.starts_with("WIN ")).collect();
    assert_eq!(
        ew.len(),
        gw.len(),
        "window count diverged eager={} graph={}",
        ew.len(),
        gw.len()
    );
    assert!(!ew.is_empty(), "no WIN lines recorded");
    for (e, g) in ew.iter().zip(&gw) {
        for key in [
            "tag",
            "k",
            "pos",
            "seed",
            "committed",
            "accept",
            "advance",
            "drafts",
            "n",
            "end",
        ] {
            assert_eq!(
                win_field(e, key),
                win_field(g, key),
                "eager/graph {key}: {e} vs {g}"
            );
        }
        let tag = win_field(e, "tag");
        for fam in FAMS {
            let eager_bytes = read_sidecar(eager_dir, tag, fam);
            let graph_bytes = read_sidecar(graph_dir, tag, fam);
            if eager_bytes != graph_bytes {
                let offset = eager_bytes
                    .iter()
                    .zip(&graph_bytes)
                    .position(|(a, b)| a != b)
                    .unwrap_or(eager_bytes.len().min(graph_bytes.len()));
                panic!(
                    "{tag} family {fam}: eager/graph differ at byte {offset}, lengths {} vs {}",
                    eager_bytes.len(),
                    graph_bytes.len()
                );
            }
        }
    }

    // EF contract: ACTIVE (non-empty) on both arms under the default Q8 build.
    let ef_len = |dir: &Path, tag: &str| read_sidecar(dir, tag, "dn_ef").len();
    let first_tag = win_field(ew[0], "tag");
    let (eager_ef, graph_ef) = (ef_len(eager_dir, first_tag), ef_len(graph_dir, first_tag));
    assert_eq!(eager_ef, graph_ef, "dn_ef length diverged across arms");
    assert!(
        eager_ef > 0,
        "dn_ef empty on both arms (HIPFIRE_DN_STATE_EF=0 set?)"
    );

    for (name, text, is_graph) in [
        ("eager", eager.as_str(), false),
        ("graph", graph.as_str(), true),
    ] {
        let host = text
            .lines()
            .find(|l| l.starts_with("HOST "))
            .expect("HOST line");
        assert!(host.contains("host_expert_layers=16"), "{name}: {host}");
        assert!(
            text.lines()
                .any(|l| l.starts_with("PLACEMENT explicit-fixed-first16")),
            "{name}: placement line missing"
        );
        assert_eq!(
            text.lines().filter(|l| l.starts_with("RELOAD ")).count(),
            1,
            "{name}: reload check missing"
        );
        let reps: Vec<&str> = text.lines().filter(|l| l.starts_with("REPLAY ")).collect();
        assert!(!reps.is_empty(), "{name}: no REPLAY lines");
        let last = reps.last().unwrap();
        let get = |k: &str| -> u64 {
            last.split_whitespace()
                .find(|p| p.starts_with(&format!("{k}=")))
                .and_then(|p| p.split_once('=').unwrap().1.parse().ok())
                .unwrap_or(u64::MAX)
        };
        let (caps, reps_n) = (get("captures"), get("replays"));
        if is_graph {
            assert!(caps >= 3, "graph: only {caps} captures");
            assert!(reps_n >= 6, "graph: only {reps_n} replays");
        } else {
            assert_eq!(caps, 0, "eager: {caps} captures despite optout");
            assert_eq!(reps_n, 0, "eager: {reps_n} replays despite optout");
        }
    }
    eprintln!(
        "[vg-oracle-cmp] PASS windows={} eager={} graph={} (tokens exact, all 10 families byte-exact, replay proven, reload match)",
        ew.len(),
        eager_dir.display(),
        graph_dir.display()
    );
}

#[test]
#[ignore = "requires real HIP GPU (gfx1201) + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (qwen3.6-35b-a3b.mq4p) + MTP sidecar"]
fn mtp_verify_graph_eager_vs_replay() {
    if let Ok(arm) = std::env::var(ARM_ENV) {
        let out = std::env::var(OUT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                unique_out_dir(std::env::var(OUT_ROOT_ENV).map(PathBuf::from).ok(), &arm)
            });
        run_worker(&arm, out);
    } else {
        run_orchestrator();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU-offload MTP state/rollback oracle
// ═══════════════════════════════════════════════════════════════════════════
//
// ONE extra `#[ignore]`d test (`mtp_cpu_offload_rollback_oracle`) with three
// roles: orchestrator (default) plus two FRESH-subprocess workers. The CPU
// offload is armed per-`Command` (`HIPFIRE_OFFLOAD_EXEC=cpu`), so it is a
// property of the child process only and only one 35B fixture is resident at a
// time — never two targets, never a mutated global env.
//
// Why this is not the eager-vs-graph oracle above (which runs both arms with the
// graph off, i.e. the same code twice):
//
// * `ref` (independent process, same fixture + same explicit fixed 16-layer
//   host placement) records the SERIAL CPU reference: a greedy AR run of
//   `CPU_REF_DEPTH` one-row `k=0` steps — each one a CPU-spliced forward —
//   snapshotting the state at depths `1..=CPU_REF_WINDOW` (`REF-d<depth>`),
//   plus, per rejection shape, the state after that window's WHOLE batch ran
//   serially (`BAD-<label>`): the un-rolled-back model. The tokens travel to
//   the `verify` worker as data (`ref_toks.txt`), so every expected prefix
//   comes from that process's own AR control, never from the tested output.
// * `verify` (second fresh CPU-offloaded process) runs the accepted-prefix
//   takeover windows for K=1 and K=3 with full / partial / zero accept,
//   snapshotting the retained state (`V-<label>`) and then a subsequent native
//   `k=1` continuation (`VC-<label>`); each trial also runs the same-route
//   truncated window over just the accepted prefix plus the same native
//   continuation from it (`VT-`/`VTC-<label>`); a reset + second request
//   repeats the partial trial and must reproduce it byte for byte.
// * The orchestrator judges: committed ids / accept / advance exact against the
//   reference tokens (per-label contract hard-coded in the judging process);
//   every retained family within `CPU_REL_TOL` on dequantized lanes — the trunk
//   KV rows `n..end`, the DN state, DN conv, `prev_hidden`, the selected verify
//   logits/hidden row against the serial CPU reference; the WHOLE retained state
//   (recurrent state, trunk KV, head KV, GDN tape) byte-exactly against the
//   same-route truncated window (`{prefix}T-`, the same route over the
//   accepted-prefix rows while a one-row step writes neither head KV nor tape),
//   and the continuation byte-exactly against that window's own continuation
//   (`{prefix}TC-`); drift against the serial reference one step further out is
//   logged as `CONTSERIAL …` diagnostic evidence, never asserted; rejection
//   trials must match the ACCEPTED prefix and NOT the un-rolled-back `BAD-`
//   state (≥ `CPU_BAD_MIN_RATIO`× the good delta); and the CPU evidence must be
//   real — the splice counters
//   (`hipfire_dispatch::cpu_exec_counters`) must show the splice ran on the
//   one-row reference steps AND on the 2..4-row windows, no host-mapped step may
//   leak to the GPU, `graph_captures == 0` (the CPU gate keeps the narrow verify
//   eager, so this can never degenerate into an eager-vs-graph diff), and the
//   wide prompt prefill must take zero CPU steps.
//
// Contract: llama.cpp-level, not bit-identity. Both arms run the SAME
// per-(token, rank) CPU expert FFN (the `CpuSplice` stage ->
// `hipfire_dispatch::cpu_exec::moe_cpu_experts`), but the surrounding GPU kernels
// run at width 1 vs width `k+1`, so a Q8 lane can flip by one code on a row's
// max element (1/127 ≈ 7.9e-3). Two consequences are load-bearing:
//
// * The DN state is judged as the REPRESENTED `S = code × per-row scale`, never
//   as raw codes: a requant is free to trade a code against its row scale, so a
//   raw-code delta is not a state delta (raw codes moved by up to 239 of 127
//   while the represented state agreed to 6.7e-3). The EF residual is measured
//   in those same recurrent-state units, not against its own tiny residual
//   magnitude. The DN row scales stay a separate family, as a plain-stream
//   sensitivity check on the quantizer.
// * The serial reference is a DIFFERENT route, so it only BOUNDS the direct
//   window (`CPU_REL_TOL`); continuation drift against it is logged as
//   `CONTSERIAL …` evidence and never asserted, because that quantity sits one
//   decode step further out and has no validated bound. Exactness is pinned by
//   the same-route windows instead: `{prefix}T-` must reproduce the retained
//   state byte for byte and `{prefix}TC-` the continuation after it, so no
//   tolerance can hide a skipped rollback.
//
// Every per-family delta is emitted to `digests.log`, and the judging log is
// written even when an assertion fires (see `JudgeLog`), because the panic
// message alone loses every other family's evidence. A wrong rollback — a
// retained rejected row in the recurrent state — is O(1) relative and is what
// the negative control pins.
//
// ```bash
// HIPFIRE_MTP_BYTE_IDENTITY_MODEL=~/.hipfire/models/qwen3.6-35b-a3b.mq4p \
// HIPFIRE_MTP_BYTE_IDENTITY_HEAD=~/.hipfire/models/qwen3.6-35b-a3b.mtp \
// HIPFIRE_MTP_CPU_ORACLE_REQUIRE=1 \
// HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_OUT_ROOT=/tmp/cpu-oracle \
// cargo test --release -p hipfire-arch-qwen35 \
//   --test mtp_verify_graph_oracle mtp_cpu_offload_rollback_oracle -- --ignored --nocapture
// ```

const CPU_ARM_ENV: &str = "HIPFIRE_MTP_CPU_ORACLE_ARM";
const CPU_OUT_ENV: &str = "HIPFIRE_MTP_CPU_ORACLE_OUT";
const CPU_REQUIRE_ENV: &str = "HIPFIRE_MTP_CPU_ORACLE_REQUIRE";
const CPU_TOKS_ENV: &str = "HIPFIRE_MTP_CPU_ORACLE_REF_TOKS";
/// `memory.offload_exec`'s compat env key, set per-`Command` on the workers.
const CPU_OFFLOAD_ENV: &str = "HIPFIRE_OFFLOAD_EXEC";

/// Serial AR steps the reference process runs (tokens `0..=CPU_REF_DEPTH`).
const CPU_REF_DEPTH: usize = 8;
/// Highest reference depth snapshotted. Covers every trial's retained depth
/// (`advance` ≤ 4) plus its `k=1` continuation (≤ 2 more).
const CPU_REF_WINDOW: usize = 6;
/// Direct-window tolerance against the serial CPU reference (see the module
/// note above; llama.cpp-level, same convention as `gpu_moe_cpu_parity`). The
/// measured worst case over the trial table is 1.65e-2 (trunk KV, 4-row
/// window).
const CPU_REL_TOL: f32 = 2e-2;
/// The un-rolled-back state must differ from the retained state by at least this
/// factor of the good delta, so the tolerance above could not hide a skip.
const CPU_BAD_MIN_RATIO: f32 = 4.0;

/// Per-label contract, restated independently in the judging process so a
/// missing or mislabelled trial can never pass: (label, K, accept, advance).
const CPU_TRIAL_EXPECT: [(&str, usize, usize, usize); 5] = [
    ("k1-full", 1, 1, 2),
    ("k1-zero", 1, 0, 1),
    ("k3-full", 3, 3, 4),
    ("k3-partial", 3, 2, 3),
    ("k3-zero", 3, 0, 1),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum FamKind {
    F32,
    F16,
    Q8,
}

/// Storage kind of a family the judge compares as a plain lane stream. `dn_s`
/// is deliberately absent: an int8 code is not a state, so it is only ever
/// judged coupled with its per-row scales (`dn_coupled_delta`); `dn_ef` is
/// likewise normalized by that coupled state, never by its own residual.
fn plain_kind(fam: &str) -> FamKind {
    match fam {
        "logits" | "hidden" | "prev" | "dn_scales" | "dn_conv" | "tape" => FamKind::F32,
        "kv" | "mtpkv" => FamKind::Q8,
        other => panic!("no plain lane kind for family {other}"),
    }
}

/// Whole-state families (not row-windowed): the only ones the un-rolled-back
/// negative control can compare without a row-count mismatch.
const CPU_BAD_FAMS: [&str; 5] = ["prev", "dn_s", "dn_scales", "dn_conv", "dn_ef"];

/// Families that are only like-for-like against another *batched* takeover
/// window, never against the one-row serial reference:
///
/// * `mtpkv` — the takeover path pre-fills the head KV for the retained rows
///   with decode's `(tok_p, h_{p-1})` pairing, while the serial decode path
///   defers that row to the next cycle's draft step;
/// * `tape` — the GDN capture predicate requires `n >= MIN_BATCH` (2), so a
///   one-row serial step never writes it at all.
///
/// Both are pinned by the same-route windows (`{prefix}T-<label>` for the
/// retained state, `{prefix}TC-<label>` for the continuation after it): same
/// route, same retained rows, same tokens, one row shorter than the tested
/// batch.
const CPU_SAME_ROUTE_FAMS: [&str; 2] = ["mtpkv", "tape"];

/// Retained-state families pinned BYTE-EXACTLY against the same-route truncated
/// window `{prefix}T-<label>`: the recurrent state (`prev`, DN S codes and their
/// per-row scales, DN conv, DN EF), the retained trunk KV rows, plus the two
/// `CPU_SAME_ROUTE_FAMS`.
///
/// The truncated window is the same takeover route over exactly the
/// accepted-prefix rows — it is always full-accept — so it is the state the
/// tested window must have kept. Comparing it against the one-row serial
/// reference can only *bound* the serial-vs-batch drift, and a tolerance is
/// exactly what could hide a skipped rollback; this byte comparison cannot.
const CPU_SAME_ROUTE_STATE_FAMS: [&str; 8] = [
    "prev",
    "dn_s",
    "dn_scales",
    "dn_conv",
    "dn_ef",
    "kv",
    "mtpkv",
    "tape",
];

fn fam_idx(name: &str) -> usize {
    FAMS.iter()
        .position(|f| *f == name)
        .unwrap_or_else(|| panic!("unknown family {name}"))
}

fn cpu_csv(v: &[u32]) -> String {
    v.iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn cpu_parse_ids(s: &str, what: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(|x| {
            x.trim()
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("{what}: bad id {x:?} in {s:?}"))
        })
        .collect()
}

fn f32_le(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0, "ragged f32 bytes: {}", bytes.len());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// First `id` that is provably different from the token the model predicts at
/// that position: the previous id (skipping 0 and EOS). Shared rule so the
/// reference and verify workers build the same rejected candidates.
fn cpu_wrong_token(vocab: usize, eos: u32, t: u32) -> u32 {
    let c = if t == 0 { 1 } else { t - 1 };
    if c == eos {
        (c + 1) % vocab as u32
    } else {
        c
    }
}

/// The trial table: a pure function of the reference AR tokens, so both workers
/// derive the same windows and the judged expectations cannot come from the
/// tested output. `bad` is the un-rolled-back model of a rejected window: the
/// accepted prefix's rows plus the window's rejected tail, consumed serially.
struct CpuTrial {
    label: &'static str,
    k: usize,
    cands: Vec<u32>,
    accept: usize,
    advance: usize,
    bad: Option<Vec<u32>>,
}

fn cpu_trials(toks: &[u32], vocab: usize, eos: u32) -> Vec<CpuTrial> {
    assert!(
        toks.len() > CPU_REF_WINDOW,
        "reference token trail has only {} tokens (need > {CPU_REF_WINDOW})",
        toks.len()
    );
    let w = |i: usize| cpu_wrong_token(vocab, eos, toks[i]);
    let trials = vec![
        CpuTrial {
            label: "k1-full",
            k: 1,
            cands: vec![toks[1]],
            accept: 1,
            advance: 2,
            bad: None,
        },
        CpuTrial {
            label: "k1-zero",
            k: 1,
            cands: vec![w(1)],
            accept: 0,
            advance: 1,
            bad: Some(vec![toks[0], w(1)]),
        },
        CpuTrial {
            label: "k3-full",
            k: 3,
            cands: vec![toks[1], toks[2], toks[3]],
            accept: 3,
            advance: 4,
            bad: None,
        },
        CpuTrial {
            label: "k3-partial",
            k: 3,
            cands: vec![toks[1], toks[2], w(3)],
            accept: 2,
            advance: 3,
            bad: Some(vec![toks[0], toks[1], toks[2], w(3)]),
        },
        CpuTrial {
            label: "k3-zero",
            k: 3,
            cands: vec![w(1), w(2), w(3)],
            accept: 0,
            advance: 1,
            bad: Some(vec![toks[0], w(1), w(2), w(3)]),
        },
    ];
    for t in &trials {
        assert!(
            t.k <= VERIFY_CAP,
            "{}: K={} exceeds verify capacity {VERIFY_CAP}",
            t.label,
            t.k
        );
        assert_eq!(
            t.k,
            t.cands.len(),
            "{}: k must be the candidate count",
            t.label
        );
        assert_eq!(
            t.advance,
            t.accept + 1,
            "{}: advance must be accept+1",
            t.label
        );
        for (i, &c) in t.cands.iter().enumerate() {
            assert!(
                (c as usize) < vocab,
                "{}: candidate {c} out of vocab",
                t.label
            );
            if i < t.accept {
                assert_eq!(
                    c,
                    toks[i + 1],
                    "{}: accepted candidate {i} is not the AR token",
                    t.label
                );
            } else {
                assert_ne!(
                    c,
                    toks[i + 1],
                    "{}: rejected candidate {i} equals the AR token",
                    t.label
                );
            }
        }
        match &t.bad {
            Some(bad) => {
                assert_eq!(
                    bad.len(),
                    t.cands.len() + 1,
                    "{}: the un-rolled-back model must be the window's whole batch",
                    t.label
                );
                assert_eq!(
                    &bad[..t.advance],
                    &toks[..t.advance],
                    "{}: the un-rolled-back model must agree with the accepted prefix",
                    t.label
                );
                assert_eq!(
                    &bad[t.advance..],
                    &t.cands[t.accept..],
                    "{}: the un-rolled-back model's tail must be the rejected candidates",
                    t.label
                );
            }
            None => assert_eq!(
                t.accept, t.k,
                "{}: only a full-accept window has no rejected tail",
                t.label
            ),
        }
    }
    trials
}

// ── snapshot / worker-side collection ──────────────────────────────────────

fn raw_at(gpu: &Gpu, t: &GpuTensor, offset: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    gpu.hip
        .memcpy_dtoh_at(&mut out, &t.buf, offset)
        .expect("dtoh at");
    out
}

fn trunk_kv_window(gpu: &Gpu, slot: &ModelSlot, start: usize, end: usize) -> Vec<u8> {
    let kv = &slot.kv_cache;
    let row = kv.n_kv_heads * (kv.head_dim / 32) * 34;
    let mut out = Vec::new();
    for layer in 0..kv.k_gpu.len() {
        for buf in [&kv.k_gpu[layer], &kv.v_gpu[layer]] {
            if buf.buf.size() < row {
                continue;
            }
            assert!(
                buf.buf.size() >= end * row,
                "trunk kv buffer holds {} bytes, window needs {}",
                buf.buf.size(),
                end * row
            );
            out.extend_from_slice(&raw_at(gpu, buf, start * row, (end - start) * row));
        }
    }
    out
}

fn mtp_kv_window(gpu: &Gpu, state: &MtpSpecState, start: usize, end: usize) -> Vec<u8> {
    let kv = &state.mtp_kv;
    let row = kv.n_head_kv * (kv.head_dim / 32) * 34;
    let mut out = Vec::new();
    for buf in kv.inner.k_gpu.iter().chain(kv.inner.v_gpu.iter()) {
        if buf.buf.size() < row {
            continue;
        }
        assert!(
            buf.buf.size() >= end * row,
            "mtp kv buffer holds {} bytes, window needs {}",
            buf.buf.size(),
            end * row
        );
        out.extend_from_slice(&raw_at(gpu, buf, start * row, (end - start) * row));
    }
    out
}

fn tape_row(gpu: &Gpu, state: &MtpSpecState, row_sel: usize) -> Vec<u8> {
    let tape = &state.trunk_gdn_tape;
    let mut out = Vec::new();
    for t in &tape.qkv_bufs {
        out.extend_from_slice(&raw_at(
            gpu,
            t,
            row_sel * tape.qkv_dim * 4,
            tape.qkv_dim * 4,
        ));
    }
    for t in tape.alpha_bufs.iter().chain(tape.beta_bufs.iter()) {
        out.extend_from_slice(&raw_at(
            gpu,
            t,
            row_sel * tape.n_v_heads * 4,
            tape.n_v_heads * 4,
        ));
    }
    out
}

/// Row-window snapshot for the cross-process comparison: the retained window
/// rows only (`kv`/`mtpkv` rows `win_start..win_end`) and the single selected
/// row for the row-indexed families (`logits`/`hidden`/`tape`). Writes
/// `{tag}-{fam}.bin` for every family in `FAMS` except the ones in `omit`, and
/// checks each written family's format on exactly the bytes written. Returns the
/// total bytes written.
#[allow(clippy::too_many_arguments)]
fn cpu_snap(
    gpu: &Gpu,
    slot: &ModelSlot,
    state: &MtpSpecState,
    out_dir: &Path,
    tag: &str,
    row_sel: usize,
    win_start: usize,
    win_end: usize,
    omit: &[&str],
) -> usize {
    let dim = slot.config.dim;
    let vocab = slot.config.vocab_size;
    assert!(
        win_start < win_end,
        "{tag}: empty row window {win_start}..{win_end}"
    );
    assert!(
        row_sel < state.verify_capacity + 1,
        "{tag}: row {row_sel} outside the verify envelope"
    );

    let logits = raw_at(gpu, &state.verify_logits, row_sel * vocab * 4, vocab * 4);
    let hidden = raw_at(gpu, &state.verify_hidden, row_sel * dim * 4, dim * 4);
    let prev = raw_all(gpu, &state.prev_hidden);

    let mut dn_s = Vec::new();
    let mut dn_scales = Vec::new();
    let mut dn_conv = Vec::new();
    let mut dn_ef = Vec::new();
    for t in &slot.dn_state.s_matrices {
        dn_s.extend_from_slice(&raw_all(gpu, t));
    }
    for t in &slot.dn_state.s_scales {
        dn_scales.extend_from_slice(&raw_all(gpu, t));
    }
    for t in &slot.dn_state.conv_states {
        dn_conv.extend_from_slice(&raw_all(gpu, t));
    }
    for t in &slot.dn_state.s_ef_residual {
        dn_ef.extend_from_slice(&raw_all(gpu, t));
    }

    let kv = trunk_kv_window(gpu, slot, win_start, win_end);
    let mtpkv = mtp_kv_window(gpu, state, win_start, win_end);
    let tape = tape_row(gpu, state, row_sel);

    // Formats are checked on exactly the bytes written; an omitted family is one
    // this arm never writes (see `CPU_SAME_ROUTE_FAMS`), so a stale buffer must
    // not be validated as if it were the captured quantity.
    let mut total = 0usize;
    let mut write = |name: &str, bytes: Vec<u8>| {
        if omit.contains(&name) {
            return;
        }
        match name {
            "logits" => assert_all_finite_f32("snap-logits", &f32_le(&bytes)),
            "hidden" => assert_all_finite_f32("snap-hidden", &f32_le(&bytes)),
            "prev" => assert_all_finite_f32("snap-prev", &f32_le(&bytes)),
            "dn_s" => assert!(!bytes.is_empty(), "{tag}: dn-s empty"),
            "dn_scales" => assert_all_finite_f32("snap-dn-scales", &f32_le(&bytes)),
            "dn_conv" => assert_all_finite_f32("snap-dn-conv", &f32_le(&bytes)),
            "dn_ef" => assert_all_finite_f16("snap-dn-ef", &bytes),
            "kv" => assert_q8_blocks("snap-kv", &bytes),
            "mtpkv" => assert_q8_blocks("snap-mtpkv", &bytes),
            "tape" => assert_all_finite_f32("snap-tape", &f32_le(&bytes)),
            other => panic!("unknown family {other}"),
        }
        assert!(!bytes.is_empty(), "{tag}-{name}: empty family");
        total += bytes.len();
        std::fs::write(out_dir.join(format!("{tag}-{name}.bin")), &bytes).expect("write sidecar");
    };
    write("logits", logits);
    write("hidden", hidden);
    write("prev", prev);
    write("dn_s", dn_s);
    write("dn_scales", dn_scales);
    write("dn_conv", dn_conv);
    write("dn_ef", dn_ef);
    write("kv", kv);
    write("mtpkv", mtpkv);
    write("tape", tape);
    total
}

fn cpu_require_fixture() -> (PathBuf, PathBuf) {
    let (model, head) = model_paths().unwrap_or_else(|| {
        panic!(
            "cpu oracle worker requires HIPFIRE_MTP_BYTE_IDENTITY_MODEL \
             (+ optional HIPFIRE_MTP_BYTE_IDENTITY_HEAD)"
        )
    });
    assert!(
        model.is_file(),
        "trunk fixture missing: {}",
        model.display()
    );
    assert!(head.is_file(), "mtp sidecar missing: {}", head.display());
    (model, head)
}

// ── reference worker (independent CPU AR control) ──────────────────────────

fn run_cpu_ref_worker(out_dir: PathBuf) {
    let (model, head_path) = cpu_require_fixture();
    std::fs::create_dir_all(&out_dir).expect("create ref out dir");
    assert!(
        hipfire_dispatch::moe_cpu_experts_enabled() && hipfire_dispatch::cpu_exec_enabled(),
        "ref worker must run with memory.offload_exec=cpu (per-Command HIPFIRE_OFFLOAD_EXEC) \
         and the expert splice unset"
    );

    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(
        gpu.arch.as_str(),
        "gfx1201",
        "cpu oracle requires single-GPU gfx1201"
    );
    let mut lines = Vec::new();
    let mut slot = load_offloaded_session(&mut gpu, &model, "cpu-oracle-ref", &mut lines);
    let head = mtp_head::load_mtp_head(&head_path, &mut gpu, CTX).expect("load head");
    let tokenizer = slot.load_tokenizer().expect("tokenizer");
    let prompt = build_prompt(&tokenizer);
    let n = prompt.len();
    let eos = slot.config.eos_token;
    let vocab = slot.config.vocab_size;
    let mut state = MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(
        &mut gpu,
        &slot,
        &head,
        MAX_N,
        VERIFY_CAP,
        MtpKvMode::Q8,
    )
    .expect("state");
    if let Some(cvs) = head.weights.compressed_vocab_size {
        state
            .mtp_scratch
            .ensure_compressed_logits(&mut gpu, cvs)
            .expect("head compressed logits");
        state
            .ensure_compressed_lm_logits(&mut gpu, cvs)
            .expect("batched compressed logits");
    }
    assert!(
        gpu.owns_host_cpu_weights(),
        "loader did not set the host CPU-weight gate"
    );
    lines.push(format!(
        "CPU-ORACLE role=ref exec=cpu cpu_exec={} moe_cpu_experts={} host_cpu_weights={}",
        u8::from(hipfire_dispatch::cpu_exec_enabled()),
        u8::from(hipfire_dispatch::moe_cpu_experts_enabled()),
        u8::from(gpu.owns_host_cpu_weights()),
    ));
    lines.push(format!("MODEL vocab={vocab} eos={eos} n={n} roles=ref"));

    // Wide prompt prefill must stay on the GPU route: zero CPU splice steps.
    let before = cpu_exec_counters();
    let seed0 = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
    let after_prefill = cpu_exec_counters();
    let prefill_steps = after_prefill.0 - before.0;
    assert_eq!(
        prefill_steps, 0,
        "wide prompt prefill took {prefill_steps} CPU steps; the splice must stay off above {VERIFY_CAP} rows"
    );

    // Serial CPU AR reference: one-row k=0 steps, each a CPU-spliced forward.
    let mut toks = vec![seed0];
    let mut m_per_forward: Option<usize> = None;
    for d in 1..=CPU_REF_DEPTH {
        let c0 = cpu_exec_counters();
        let r = spec_step_mtp_compressed_serial_with_k(
            &mut gpu,
            &mut slot,
            &head,
            &mut state,
            n + d - 1,
            toks[d - 1],
            eos,
            0,
        )
        .expect("serial reference step");
        let c1 = cpu_exec_counters();
        let steps = c1.0 - c0.0;
        assert!(!r.hit_eos, "reference hit EOS at depth {d}");
        assert_eq!(r.advance, 1, "reference step must commit exactly one token");
        match m_per_forward {
            None => {
                assert!(
                    steps > 0,
                    "serial reference step d={d} took 0 CPU steps: the CPU expert splice is not armed \
                     (host placement or memory.offload_exec)"
                );
                m_per_forward = Some(steps);
            }
            Some(m) => assert_eq!(
                steps, m,
                "reference step d={d}: {steps} CPU steps, expected {m}"
            ),
        }
        toks.push(r.committed[0]);
        if d <= CPU_REF_WINDOW {
            // The one-row serial path writes neither a head-KV row nor a GDN tape
            // row: omit both here and pin them against the same-route batched
            // window in the judging process.
            cpu_snap(
                &gpu,
                &slot,
                &state,
                &out_dir,
                &format!("REF-d{d}"),
                0,
                n,
                n + d,
                &CPU_SAME_ROUTE_FAMS,
            );
            lines.push(format!(
                "SNAP kind=ref tag=REF-d{d} depth={d} rows=1 win={n}..{}",
                n + d
            ));
        }
    }
    let m = m_per_forward.expect("no reference step ran");
    lines.push(format!(
        "REF toks={} depths=1..{CPU_REF_WINDOW}",
        cpu_csv(&toks)
    ));

    // Per-rejection-shape un-rolled-back models: the window's whole batch
    // consumed serially from a fresh prefill.
    for t in cpu_trials(&toks, vocab, eos) {
        let Some(bad) = &t.bad else { continue };
        let seed = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
        assert_eq!(
            seed, toks[0],
            "{}: bad-model prefill seed diverged",
            t.label
        );
        for (i, &tok) in bad.iter().enumerate() {
            let r = spec_step_mtp_compressed_serial_with_k(
                &mut gpu,
                &mut slot,
                &head,
                &mut state,
                n + i,
                tok,
                eos,
                0,
            )
            .expect("bad-model step");
            assert!(!r.hit_eos, "{}: bad-model step hit EOS", t.label);
            assert_eq!(r.advance, 1);
        }
        let tag = format!("BAD-{}", t.label);
        cpu_snap(
            &gpu,
            &slot,
            &state,
            &out_dir,
            &tag,
            0,
            n,
            n + bad.len(),
            &CPU_SAME_ROUTE_FAMS,
        );
        lines.push(format!(
            "SNAP kind=bad tag={tag} label={} depth={} rows=1 win={n}..{}",
            t.label,
            bad.len(),
            n + bad.len()
        ));
    }

    // GEMV-step coverage for this worker's narrow (≤4-row) forwards: every window
    // splices, so no host-mapped GEMV step may stay on the GPU. This counter sees
    // only the step seam — no MoE grouped GPU read occurs here because narrow
    // windows splice; narrow-window MoE coverage itself comes from the per-shape
    // splice trace.
    let final_counters = cpu_exec_counters();
    assert_eq!(
        final_counters.1, 0,
        "{} host-mapped GEMV steps stayed on the GPU",
        final_counters.1
    );
    lines.push(format!(
        "COUNTERS role=ref on_cpu={} leak={} per_forward={m} prefill_steps={prefill_steps} \
graph_captures={} graph_replays={} graph_mode={}",
        final_counters.0,
        final_counters.1,
        state.mtp_verify_graph_captures(),
        state.mtp_verify_graph_replays(),
        state.mtp_verify_graph_mode(),
    ));
    free_session(&mut gpu, slot, state, head);
    std::fs::write(out_dir.join("digests.log"), lines.join("\n") + "\n")
        .expect("write ref digests");
    eprintln!("[cpu-oracle] role=ref DONE out={}", out_dir.display());
}

// ── verify worker (CPU-spliced narrow windows + rollback) ──────────────────

/// One window's measured outcome plus the manifest line fields.
struct CpuTrialOutcome {
    steps: usize,
    forwards: usize,
    leak: usize,
    caps: u64,
    reps: u64,
    committed: Vec<u32>,
    accept: usize,
    advance: usize,
    cont_advance: usize,
    cont_first: u32,
    end: usize,
}

struct CpuCtx<'a> {
    gpu: &'a mut Gpu,
    slot: &'a mut ModelSlot,
    head: &'a mtp_head::Qwen35MtpHead,
    state: &'a mut MtpSpecState,
    out_dir: PathBuf,
    prompt: Vec<u32>,
    toks: Vec<u32>,
    vocab: usize,
    eos: u32,
    n: usize,
    m_per_forward: usize,
}

impl CpuCtx<'_> {
    /// Fresh reset + prefill; returns `(seed, cpu steps)` so every window's
    /// starting point is the same state and the wide prefill's CPU-step count is
    /// observable.
    fn prefill(&mut self) -> (u32, usize) {
        let before = cpu_exec_counters();
        let seed = prefill_and_seed(self.gpu, self.slot, self.head, self.state, &self.prompt);
        let after = cpu_exec_counters();
        (seed, after.0 - before.0)
    }

    fn snap(&self, tag: &str, row_sel: usize, win_start: usize, win_end: usize) -> usize {
        cpu_snap(
            self.gpu,
            self.slot,
            self.state,
            &self.out_dir,
            tag,
            row_sel,
            win_start,
            win_end,
            &[],
        )
    }

    /// Fresh prefill + takeover window + native `k=1` continuation, with the
    /// CPU-splice evidence for the narrow window. Snapshots `{prefix}-<label>`
    /// and `{prefix}C-<label>`.
    fn trial(&mut self, t: &CpuTrial, prefix: &str) -> CpuTrialOutcome {
        let (seed, prefill_steps) = self.prefill();
        assert_eq!(
            seed, self.toks[0],
            "{}: prefill seed diverged from the reference process",
            t.label
        );
        assert_eq!(
            prefill_steps, 0,
            "{}: wide prompt prefill took {prefill_steps} CPU steps",
            t.label
        );

        let c0 = cpu_exec_counters();
        let r = spec_step_mtp_compressed_serial_with_takeover_candidates(
            self.gpu,
            self.slot,
            self.head,
            self.state,
            self.n,
            self.toks[0],
            self.eos,
            &t.cands,
        )
        .expect("takeover window");
        let c1 = cpu_exec_counters();
        let steps = c1.0 - c0.0;
        assert!(
            steps > 0,
            "{}: takeover window took 0 CPU steps — the CPU splice did not run",
            t.label
        );
        assert_eq!(
            steps % self.m_per_forward,
            0,
            "{}: {steps} CPU steps is not a multiple of {}/forward",
            t.label,
            self.m_per_forward
        );
        assert!(!r.hit_eos, "{}: hit EOS", t.label);
        assert_eq!(
            r.accept_count, t.accept,
            "{}: accept_count drifted",
            t.label
        );
        assert_eq!(r.advance, t.advance, "{}: advance drifted", t.label);
        assert_eq!(
            r.drafts_generated, t.k,
            "{}: drafts_generated drifted",
            t.label
        );
        assert_eq!(
            r.committed.as_slice(),
            &self.toks[1..=t.advance],
            "{}: committed ids diverged from the reference tokens",
            t.label
        );

        let end = self.n + r.advance;
        let tag = format!("{prefix}-{}", t.label);
        self.snap(&tag, r.advance - 1, self.n, end);

        // Continuation on the retained post-rollback state: the first committed
        // token is the reference's next greedy token either way (accepted
        // candidate or bonus), which is exactly what a wrong rollback breaks.
        let cc0 = cpu_exec_counters();
        let c = spec_step_mtp_compressed_serial_with_k(
            self.gpu,
            self.slot,
            self.head,
            self.state,
            end,
            *r.committed.last().unwrap(),
            self.eos,
            1,
        )
        .expect("post-window continuation");
        let cc1 = cpu_exec_counters();
        let csteps = cc1.0 - cc0.0;
        assert!(
            csteps > 0 && csteps % self.m_per_forward == 0,
            "{}: continuation took {csteps} CPU steps, not a positive multiple of {}/forward",
            t.label,
            self.m_per_forward
        );
        assert!(!c.hit_eos, "{}: continuation hit EOS", t.label);
        assert_eq!(
            c.drafts_generated, 1,
            "{}: continuation must draft exactly one",
            t.label
        );
        assert!(
            (1..=2).contains(&c.advance),
            "{}: continuation advance {}",
            t.label,
            c.advance
        );
        assert_eq!(
            c.committed[0],
            self.toks[t.advance + 1],
            "{}: continuation token diverged from the reference AR token",
            t.label
        );
        let cend = end + c.advance;
        let ctag = format!("{prefix}C-{}", t.label);
        self.snap(&ctag, c.advance - 1, self.n, cend);

        // Same-route truncated reference: the same takeover route over exactly
        // the accepted-prefix rows, one row shorter than the tested batch, so it
        // is precisely the state the tested window must have kept — the whole
        // retained state is compared against it byte for byte (see
        // `CPU_SAME_ROUTE_STATE_FAMS`), not just the head KV. (The takeover fill
        // writes rows `[cur_pos, cur_pos + advance)` with decode's
        // `(tok_p, h_{p-1})` pairing, while the serial `k=0` path defers that row
        // to the next cycle's draft step, so the serial reference can never pin
        // the head KV — nor the GDN tape, which a one-row step does not write.)
        let tr_cands = if t.accept >= 1 {
            self.toks[1..=t.accept].to_vec()
        } else {
            vec![(self.toks[1] + 1) % self.vocab as u32]
        };
        let (tr_seed, tr_prefill_steps) = self.prefill();
        assert_eq!(
            tr_seed, self.toks[0],
            "{}: truncated-reference prefill seed diverged",
            t.label
        );
        assert_eq!(
            tr_prefill_steps, 0,
            "{}: truncated-reference prefill took {tr_prefill_steps} CPU steps",
            t.label
        );
        let tr = spec_step_mtp_compressed_serial_with_takeover_candidates(
            self.gpu,
            self.slot,
            self.head,
            self.state,
            self.n,
            self.toks[0],
            self.eos,
            &tr_cands,
        )
        .expect("truncated reference window");
        assert_eq!(
            tr.accept_count, t.accept,
            "{}: truncated reference accepted {} of {}",
            t.label, tr.accept_count, t.accept
        );
        assert_eq!(
            tr.advance, t.advance,
            "{}: truncated reference advance {} != {}",
            t.label, tr.advance, t.advance
        );
        assert_eq!(
            tr.committed.as_slice(),
            &self.toks[1..=tr.advance],
            "{}: truncated reference committed ids diverged from the reference AR tokens",
            t.label
        );
        let tr_tag = format!("{prefix}T-{}", t.label);
        self.snap(&tr_tag, tr.advance - 1, self.n, self.n + tr.advance);

        // The SAME native `k=1` continuation, from the truncated window's
        // retained state: same route, the same committed token and (per the
        // byte-identity check) the same retained state, so it must reproduce this
        // window's continuation exactly. This — not the one-row serial reference
        // one decode step further out — is the continuation oracle.
        let tc = spec_step_mtp_compressed_serial_with_k(
            self.gpu,
            self.slot,
            self.head,
            self.state,
            self.n + tr.advance,
            *tr.committed.last().unwrap(),
            self.eos,
            1,
        )
        .expect("truncated-reference continuation");
        assert!(
            !tc.hit_eos,
            "{}: truncated-reference continuation hit EOS",
            t.label
        );
        assert_eq!(
            tc.drafts_generated, 1,
            "{}: truncated-reference continuation must draft exactly one",
            t.label
        );
        assert_eq!(
            tc.committed, c.committed,
            "{}: truncated-reference continuation committed ids diverged",
            t.label
        );
        assert_eq!(
            tc.advance, c.advance,
            "{}: truncated-reference continuation advance {} != {}",
            t.label, tc.advance, c.advance
        );
        let tc_tag = format!("{prefix}TC-{}", t.label);
        self.snap(
            &tc_tag,
            tc.advance - 1,
            self.n,
            self.n + tr.advance + tc.advance,
        );

        CpuTrialOutcome {
            steps,
            forwards: steps / self.m_per_forward,
            leak: c1.1,
            caps: self.state.mtp_verify_graph_captures(),
            reps: self.state.mtp_verify_graph_replays(),
            committed: r.committed.clone(),
            accept: r.accept_count,
            advance: r.advance,
            cont_advance: c.advance,
            cont_first: c.committed[0],
            end,
        }
    }
}

fn run_cpu_verify_worker(out_dir: PathBuf) {
    let (model, head_path) = cpu_require_fixture();
    let toks_path: PathBuf = std::env::var(CPU_TOKS_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("{CPU_TOKS_ENV} must name the reference token trail"));
    let toks: Vec<u32> = std::fs::read_to_string(&toks_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", toks_path.display()))
        .split_whitespace()
        .map(|x| x.parse::<u32>().expect("reference token"))
        .collect();
    std::fs::create_dir_all(&out_dir).expect("create verify out dir");
    assert!(
        hipfire_dispatch::moe_cpu_experts_enabled() && hipfire_dispatch::cpu_exec_enabled(),
        "verify worker must run with memory.offload_exec=cpu (per-Command HIPFIRE_OFFLOAD_EXEC)"
    );

    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(
        gpu.arch.as_str(),
        "gfx1201",
        "cpu oracle requires single-GPU gfx1201"
    );
    let mut lines = Vec::new();
    let mut slot = load_offloaded_session(&mut gpu, &model, "cpu-oracle-verify", &mut lines);
    let head = mtp_head::load_mtp_head(&head_path, &mut gpu, CTX).expect("load head");
    let tokenizer = slot.load_tokenizer().expect("tokenizer");
    let prompt = build_prompt(&tokenizer);
    let n = prompt.len();
    let eos = slot.config.eos_token;
    let vocab = slot.config.vocab_size;
    let mut state = MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(
        &mut gpu,
        &slot,
        &head,
        MAX_N,
        VERIFY_CAP,
        MtpKvMode::Q8,
    )
    .expect("state");
    if let Some(cvs) = head.weights.compressed_vocab_size {
        state
            .mtp_scratch
            .ensure_compressed_logits(&mut gpu, cvs)
            .expect("head compressed logits");
        state
            .ensure_compressed_lm_logits(&mut gpu, cvs)
            .expect("batched compressed logits");
    }
    assert!(
        gpu.owns_host_cpu_weights(),
        "loader did not set the host CPU-weight gate"
    );
    lines.push(format!(
        "CPU-ORACLE role=verify exec=cpu cpu_exec={} moe_cpu_experts={} host_cpu_weights={}",
        u8::from(hipfire_dispatch::cpu_exec_enabled()),
        u8::from(hipfire_dispatch::moe_cpu_experts_enabled()),
        u8::from(gpu.owns_host_cpu_weights()),
    ));
    lines.push(format!("MODEL vocab={vocab} eos={eos} n={n} roles=verify"));
    lines.push(format!(
        "REFTOKS path={} len={} toks={}",
        toks_path.display(),
        toks.len(),
        cpu_csv(&toks)
    ));

    let trials = cpu_trials(&toks, vocab, eos);
    let mut ctx = CpuCtx {
        gpu: &mut gpu,
        slot: &mut slot,
        head: &head,
        state: &mut state,
        out_dir: out_dir.clone(),
        prompt,
        toks: toks.clone(),
        vocab,
        eos,
        n,
        m_per_forward: 0,
    };

    // Calibration: the prefill route and the per-forward CPU splice count, both
    // measured against the reference tokens (an independent process's trail).
    let (seed, cal_prefill_steps) = ctx.prefill();
    assert_eq!(
        seed, toks[0],
        "prefill seed diverged from the reference process"
    );
    let c0 = cpu_exec_counters();
    let cal = spec_step_mtp_compressed_serial_with_k(
        ctx.gpu, ctx.slot, ctx.head, ctx.state, n, toks[0], eos, 0,
    )
    .expect("calibration step");
    let c1 = cpu_exec_counters();
    let m = c1.0 - c0.0;
    assert!(
        m > 0,
        "calibration one-row step took 0 CPU steps: the CPU expert splice is not armed \
         (host placement / memory.offload_exec=cpu / HIPFIRE_MOE_CPU_EXPERTS)"
    );
    assert_eq!(cal.advance, 1, "calibration step must commit one token");
    assert_eq!(
        cal.committed[0], toks[1],
        "calibration token diverged from the reference trail"
    );
    ctx.m_per_forward = m;
    lines.push(format!(
        "CALIB prefill_steps={cal_prefill_steps} per_forward={m} leak={} caps={} reps={} seed={seed} next={}",
        c1.1,
        ctx.state.mtp_verify_graph_captures(),
        ctx.state.mtp_verify_graph_replays(),
        cal.committed[0],
    ));

    for t in &trials {
        let o = ctx.trial(t, "V");
        lines.push(format!(
            "TRIAL tag={} k={} n_verify={} cands={} accept={} advance={} committed={} ref_committed={} end={} \
cont_advance={} cont_committed={} cont_ref={} steps={} forwards={} leak={} caps={} reps={}",
            t.label,
            t.k,
            t.cands.len() + 1,
            cpu_csv(&t.cands),
            o.accept,
            o.advance,
            cpu_csv(&o.committed),
            cpu_csv(&toks[1..=t.advance]),
            o.end,
            o.cont_advance,
            o.cont_first,
            toks[t.advance + 1],
            o.steps,
            o.forwards,
            o.leak,
            o.caps,
            o.reps,
        ));
    }

    // Reset + second request: the partial-accept trial must reproduce its own
    // bytes in the same process after a full state reset.
    let repeat = trials
        .iter()
        .find(|t| t.label == "k3-partial")
        .expect("k3-partial trial is part of the contract");
    let o2 = ctx.trial(repeat, "V2");
    lines.push(format!(
        "RESET tag={} k={} n_verify={} cands={} accept={} advance={} committed={} ref_committed={} end={} \
cont_advance={} cont_committed={} cont_ref={} steps={} forwards={} leak={} caps={} reps={}",
        repeat.label,
        repeat.k,
        repeat.cands.len() + 1,
        cpu_csv(&repeat.cands),
        o2.accept,
        o2.advance,
        cpu_csv(&o2.committed),
        cpu_csv(&toks[1..=repeat.advance]),
        o2.end,
        o2.cont_advance,
        o2.cont_first,
        toks[repeat.advance + 1],
        o2.steps,
        o2.forwards,
        o2.leak,
        o2.caps,
        o2.reps,
    ));
    for fam in FAMS {
        let first = read_sidecar(&out_dir, "V-k3-partial", fam);
        let again = read_sidecar(&out_dir, "V2-k3-partial", fam);
        assert_eq!(
            first,
            again,
            "reset/second request diverged in family {fam} ({} vs {} bytes)",
            first.len(),
            again.len()
        );
        let first_c = read_sidecar(&out_dir, "VC-k3-partial", fam);
        let again_c = read_sidecar(&out_dir, "V2C-k3-partial", fam);
        assert_eq!(
            first_c, again_c,
            "reset/second continuation diverged in family {fam}"
        );
    }
    lines.push("RESETCHECK tag=k3-partial fams=10 byte_identical=1".to_string());

    // End the borrows held by `ctx` before touching the session again.
    drop(ctx);

    let final_counters = cpu_exec_counters();
    assert_eq!(
        final_counters.1, 0,
        "{} host-mapped steps leaked to the GPU",
        final_counters.1
    );
    assert_eq!(
        state.mtp_verify_graph_captures(),
        0,
        "the CPU gate must keep the narrow verify eager (captures={})",
        state.mtp_verify_graph_captures()
    );
    lines.push(format!(
        "COUNTERS role=verify on_cpu={} leak={} per_forward={m} graph_captures={} graph_replays={} graph_mode={}",
        final_counters.0,
        final_counters.1,
        state.mtp_verify_graph_captures(),
        state.mtp_verify_graph_replays(),
        state.mtp_verify_graph_mode(),
    ));
    free_session(&mut gpu, slot, state, head);
    std::fs::write(out_dir.join("digests.log"), lines.join("\n") + "\n")
        .expect("write verify digests");
    eprintln!("[cpu-oracle] role=verify DONE out={}", out_dir.display());
}

// ── orchestrator (judging) ─────────────────────────────────────────────────

/// Dequantized lanes of one family: Q8 blocks become `scale * code`, F16
/// decodes as F16. `dn_s`/`dn_ef` never reach here — see `plain_kind`.
fn dequant_lanes(kind: FamKind, bytes: &[u8]) -> Vec<f32> {
    match kind {
        FamKind::F32 => f32_le(bytes),
        FamKind::F16 => {
            assert!(bytes.len() % 2 == 0, "ragged f16 family");
            bytes
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        FamKind::Q8 => {
            assert!(bytes.len() % 34 == 0, "ragged q8 family: {}", bytes.len());
            let mut out = Vec::with_capacity(bytes.len() / 34 * 32);
            for blk in bytes.chunks_exact(34) {
                let s = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &c in &blk[2..] {
                    out.push(s * (c as i8 as f32));
                }
            }
            out
        }
    }
}

/// Represented DN Q8 state: the production `StateQuant::Q8` layout stores
/// `n_heads * s_dim * s_dim` int8 codes with `n_heads * s_dim` per-row F32
/// scales — one scale per 128-code row (`s_dim == 128`; `prefill.rs` pins
/// `48 * 128 * 128` codes against `48 * 128` scales and `weights.rs` sizes the
/// scale buffer as `batch * n_heads * s_dim`). The code stream alone is NOT the
/// state: a requant is free to trade a code against its row scale, which is
/// exactly why the old raw-code comparison failed on a numerically identical
/// state (codes moved by up to 239 while the represented state held to 6.7e-3).
fn dn_state_lanes(codes: &[u8], scales: &[u8]) -> Vec<f32> {
    let scales = f32_le(scales);
    assert!(!scales.is_empty(), "empty DN Q8 scale stream");
    assert_eq!(
        codes.len() % scales.len(),
        0,
        "DN Q8 layout drift: {} codes over {} row scales",
        codes.len(),
        scales.len()
    );
    let per_row = codes.len() / scales.len();
    assert_eq!(
        per_row, 128,
        "DN Q8 layout drift: {per_row} codes/scale (expected a 128-wide row)"
    );
    let mut out = Vec::with_capacity(codes.len());
    for (row, &s) in codes.chunks_exact(per_row).zip(scales.iter()) {
        for &c in row {
            out.push(c as i8 as f32 * s);
        }
    }
    out
}

/// One family's delta between the reference and a candidate state.
#[derive(Clone, Copy)]
struct FamDelta {
    max_abs: f32,
    scale: f32,
    rel: f32,
    n_diff: usize,
    n: usize,
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().chain(b.iter()).fold(0f32, |m, v| m.max(v.abs()))
}

/// Lane delta with an explicit denominator: `rel = max|x - y| / scale`, the
/// denominator floored at 1e-4 so a zero-signal family is judged on an absolute
/// floor instead of dividing by ~0.
fn lane_delta(a: &[f32], b: &[f32], scale: f32) -> FamDelta {
    assert_eq!(
        a.len(),
        b.len(),
        "family lane count diverged: {} vs {}",
        a.len(),
        b.len()
    );
    assert!(!a.is_empty(), "empty family");
    let mut max_abs = 0f32;
    let mut n_diff = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > 0.0 {
            n_diff += 1;
        }
        max_abs = max_abs.max(d);
    }
    let scale = scale.max(1e-4);
    FamDelta {
        max_abs,
        scale,
        rel: max_abs / scale,
        n_diff,
        n: a.len(),
    }
}

/// Delta on a pair's own magnitude (the plain-family convention).
fn fam_delta(kind: FamKind, want: &[u8], got: &[u8]) -> FamDelta {
    let a = dequant_lanes(kind, want);
    let b = dequant_lanes(kind, got);
    lane_delta(&a, &b, max_abs(&a, &b))
}

/// The DN Q8 state as one pair of judgments: the coupled representation
/// `S = code * per-row scale` (the quantity the kernel reads), and the EF
/// residual in those same recurrent-state units — `max|ΔEF| / max|S|` over both
/// arms. Normalizing EF by its own (tiny) residual magnitude, as a plain lane
/// delta does, reports an O(1) "error" for a one-code state correction that says
/// nothing at all about the state.
fn dn_coupled_delta(
    want_dir: &Path,
    want_tag: &str,
    got_dir: &Path,
    got_tag: &str,
) -> (FamDelta, FamDelta) {
    let want = dn_state_lanes(
        &read_sidecar(want_dir, want_tag, "dn_s"),
        &read_sidecar(want_dir, want_tag, "dn_scales"),
    );
    let got = dn_state_lanes(
        &read_sidecar(got_dir, got_tag, "dn_s"),
        &read_sidecar(got_dir, got_tag, "dn_scales"),
    );
    let state = lane_delta(&want, &got, max_abs(&want, &got));
    let want_ef = dequant_lanes(FamKind::F16, &read_sidecar(want_dir, want_tag, "dn_ef"));
    let got_ef = dequant_lanes(FamKind::F16, &read_sidecar(got_dir, got_tag, "dn_ef"));
    let ef = lane_delta(&want_ef, &got_ef, state.scale);
    (state, ef)
}

fn cpu_log_fam(line_tag: &str, fam: &str, d: &FamDelta, ok: bool) -> String {
    format!(
        "STATE tag={line_tag} fam={fam} rel={:.3e} max_abs={:.3e} scale={:.3e} diff={}/{} ok={}",
        d.rel,
        d.max_abs,
        d.scale,
        d.n_diff,
        d.n,
        u8::from(ok)
    )
}

struct CpuTrialLine {
    tag: String,
    k: usize,
    n_verify: usize,
    cands: Vec<u32>,
    accept: usize,
    advance: usize,
    committed: Vec<u32>,
    ref_committed: Vec<u32>,
    cont_advance: usize,
    cont_first: u32,
    cont_ref: u32,
    steps: usize,
    forwards: usize,
    leak: usize,
    caps: u64,
    reps: u64,
}

fn cpu_parse_trial(line: &str) -> CpuTrialLine {
    let id = |key: &str| win_field(line, key).to_string();
    CpuTrialLine {
        tag: id("tag"),
        k: id("k").parse().expect("trial k"),
        n_verify: id("n_verify").parse().expect("trial n_verify"),
        cands: cpu_parse_ids(&id("cands"), "cands"),
        accept: id("accept").parse().expect("trial accept"),
        advance: id("advance").parse().expect("trial advance"),
        committed: cpu_parse_ids(&id("committed"), "committed"),
        ref_committed: cpu_parse_ids(&id("ref_committed"), "ref_committed"),
        cont_advance: id("cont_advance").parse().expect("trial cont_advance"),
        cont_first: id("cont_committed").parse().expect("trial cont_committed"),
        cont_ref: id("cont_ref").parse().expect("trial cont_ref"),
        steps: id("steps").parse().expect("trial steps"),
        forwards: id("forwards").parse().expect("trial forwards"),
        leak: id("leak").parse().expect("trial leak"),
        caps: id("caps").parse().expect("trial caps"),
        reps: id("reps").parse().expect("trial reps"),
    }
}

/// Judging evidence log. Written from `Drop`, so an assertion failure still
/// leaves a complete `judge.log` next to the snapshots: the panic message alone
/// would lose every other family's delta, which is the whole point of recording
/// them. `finish` marks a run that reached the end of judging.
struct JudgeLog {
    path: PathBuf,
    lines: Vec<String>,
    finished: bool,
}

impl JudgeLog {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            lines: vec![
                "LEGEND fam=dn_s rel is on the represented S = int8 code x per-row F32 scale \
                 (128 codes/scale); fam=dn_ef rel is |dEF| / max|S|; SAMEROUTE* lines compare raw \
                 bytes against the same-route truncated window"
                    .to_string(),
            ],
            finished: false,
        }
    }

    fn push(&mut self, line: String) {
        self.lines.push(line);
    }

    fn finish(&mut self) {
        self.finished = true;
    }
}

impl Drop for JudgeLog {
    fn drop(&mut self) {
        self.lines.push(if self.finished {
            "VERDICT pass=1".to_string()
        } else {
            "VERDICT fail=1 note=judging-aborted — the last STATE/BADCHECK line is the offending one"
                .to_string()
        });
        if let Err(e) = std::fs::write(&self.path, self.lines.join("\n") + "\n") {
            eprintln!(
                "[cpu-oracle-cmp] could not write {}: {e}",
                self.path.display()
            );
        }
    }
}

/// Judge one window: token contract against the reference trail, every retained
/// family against the serial CPU reference at the matching depth, the retained
/// state byte-exactly against the same-route truncated window, and — for a
/// rejected tail — the un-rolled-back negative control.
#[allow(clippy::too_many_arguments)]
fn cpu_judge_trial(
    ref_dir: &Path,
    ver_dir: &Path,
    log: &mut JudgeLog,
    trial: &CpuTrialLine,
    snap_prefix: &str,
    toks: &[u32],
    m_per_forward: usize,
    vocab: usize,
) {
    let label = trial.tag.as_str();
    let expect = CPU_TRIAL_EXPECT
        .iter()
        .find(|e| e.0 == label)
        .unwrap_or_else(|| panic!("unexpected trial tag {label}"));
    let (_, k, accept, advance) = *expect;
    assert_eq!(trial.k, k, "{label}: K drifted");
    assert_eq!(trial.accept, accept, "{label}: accept drifted");
    assert_eq!(trial.advance, advance, "{label}: advance drifted");
    assert_eq!(
        trial.advance,
        trial.committed.len(),
        "{label}: committed length != advance"
    );
    assert_eq!(trial.cands.len(), k, "{label}: candidate count != K");
    assert!(
        trial.advance + trial.cont_advance <= CPU_REF_WINDOW,
        "{label}: reference depth {} exceeds the snapshotted window {CPU_REF_WINDOW}",
        trial.advance + trial.cont_advance
    );
    // Candidates are the reference's own tokens exactly where the window claims
    // to accept, and provably different where it claims to reject — so the
    // expectation cannot have been derived from the tested output.
    for (i, &c) in trial.cands.iter().enumerate() {
        assert!(
            (c as usize) < vocab,
            "{label}: candidate {c} outside the vocabulary"
        );
        if i < accept {
            assert_eq!(
                c,
                toks[i + 1],
                "{label}: accepted candidate {i} is not the reference token"
            );
        } else {
            assert_ne!(
                c,
                toks[i + 1],
                "{label}: rejected candidate {i} equals the reference token"
            );
        }
    }
    // CPU evidence: the splice must have run for this *multi-row* narrow window,
    // at the same per-forward layer count as the one-row serial step.
    assert_eq!(trial.n_verify, k + 1, "{label}: verify width drifted");
    assert!(
        (2..=4).contains(&trial.n_verify),
        "{label}: verify width {} outside the 2..4-row splice envelope",
        trial.n_verify
    );
    assert!(
        trial.steps > 0,
        "{label}: no CPU splice steps recorded for the window"
    );
    assert_eq!(
        trial.steps % m_per_forward,
        0,
        "{label}: {} CPU steps is not a multiple of {m_per_forward}/forward",
        trial.steps
    );
    assert_eq!(
        trial.forwards,
        trial.steps / m_per_forward,
        "{label}: forward count drifted"
    );
    assert!(
        trial.forwards >= 1,
        "{label}: no CPU-spliced forward in the window"
    );
    log.push(format!(
        "CPUROWS tag={label} n_verify={} steps={} forwards={} per_forward={m_per_forward} leak={} caps={} reps={}",
        trial.n_verify, trial.steps, trial.forwards, trial.leak, trial.caps, trial.reps
    ));
    assert_eq!(
        trial.committed.as_slice(),
        &toks[1..=advance],
        "{label}: committed ids diverged from the reference prefix"
    );
    assert_eq!(
        trial.ref_committed.as_slice(),
        &toks[1..=advance],
        "{label}: worker's reference-committed field diverged"
    );
    assert_eq!(
        trial.cont_first,
        toks[advance + 1],
        "{label}: continuation token diverged"
    );
    assert_eq!(
        trial.cont_ref,
        toks[advance + 1],
        "{label}: continuation expectation diverged"
    );
    assert_eq!(
        trial.leak, 0,
        "{label}: {} host-mapped steps leaked to the GPU",
        trial.leak
    );
    assert_eq!(
        trial.caps, 0,
        "{label}: graph captured during a CPU-spliced window"
    );
    assert_eq!(
        trial.reps, 0,
        "{label}: graph replayed during a CPU-spliced window"
    );

    let good_tag = format!("REF-d{advance}");
    let got_tag = format!("{snap_prefix}-{label}");
    // Coupled DN Q8 state (`dn_s` x per-row scales) and its EF residual in the
    // same units, computed once and shared by the `dn_s`/`dn_ef` arms below.
    let (dn_state, dn_ef) = dn_coupled_delta(ref_dir, &good_tag, ver_dir, &got_tag);
    let mut good_rels = vec![0f32; FAMS.len()];
    for fam in FAMS.iter() {
        // The head KV and the GDN tape are pinned byte-exactly against the
        // same-route truncated window right below, never against the one-row
        // serial reference (a one-row step writes neither).
        if CPU_SAME_ROUTE_FAMS.contains(fam) {
            continue;
        }
        let d = match *fam {
            "dn_s" => dn_state,
            "dn_ef" => dn_ef,
            _ => fam_delta(
                plain_kind(fam),
                &read_sidecar(ref_dir, &good_tag, fam),
                &read_sidecar(ver_dir, &got_tag, fam),
            ),
        };
        let ok = d.rel <= CPU_REL_TOL;
        log.push(cpu_log_fam(&label, fam, &d, ok));
        assert!(
            ok,
            "{label} family {fam}: rel={:.3e} > tol {CPU_REL_TOL:e} (max_abs={:.3e}, scale={:.3e}, {}/{} lanes differ)",
            d.rel, d.max_abs, d.scale, d.n_diff, d.n
        );
        good_rels[fam_idx(fam)] = d.rel;
    }

    // Same-route exactness: the retained state IS the accepted prefix's state on
    // the same route, so it must be byte-identical to the same-route truncated
    // window `{prefix}T-<label>` — same takeover route, same accepted-prefix
    // rows, only a shorter candidate list. The serial reference above is a
    // different route (one-row steps) and can only bound the serial-vs-batch
    // drift, which is precisely what a tolerance could hide a skipped rollback
    // behind; this check cannot. The truncated window is always full-accept, so
    // for a rejected tail it independently proves the rollback landed on the
    // prefix state and not on the un-rolled-back batch.
    for fam in CPU_SAME_ROUTE_STATE_FAMS {
        let want = read_sidecar(ver_dir, &format!("{snap_prefix}T-{label}"), fam);
        let got = read_sidecar(ver_dir, &got_tag, fam);
        let same = want == got;
        log.push(format!(
            "SAMEROUTE tag={label} fam={fam} bytes={} byte_identical={}",
            got.len(),
            u8::from(same)
        ));
        assert!(
            same,
            "{label} family {fam}: retained state differs from the same-route truncated window \
             ({} vs {} bytes) — the accepted prefix was not the state that was kept",
            want.len(),
            got.len()
        );
    }

    // Continuation drift against the one-row SERIAL reference: DIAGNOSTIC ONLY.
    // That reference is a different route one decode step further out, so it
    // carries this window's serial-vs-batch drift plus that step's own (measured
    // up to 2.45e-2 on this fixture) — evidence, not a validated acceptance
    // bound. The continuation oracle proper is the same-route `{prefix}TC-`
    // window compared byte-exactly just below.
    let cdepth = advance + trial.cont_advance;
    let ctag = format!("REF-d{cdepth}");
    let cgot = format!("{snap_prefix}C-{label}");
    let (dn_state_c, dn_ef_c) = dn_coupled_delta(ref_dir, &ctag, ver_dir, &cgot);
    for fam in FAMS.iter() {
        if CPU_SAME_ROUTE_FAMS.contains(fam) {
            // A one-row serial step writes neither a head-KV row nor a GDN tape
            // row, so there is nothing to compare on these two: both are pinned
            // byte-exactly by the same-route windows instead.
            log.push(format!(
                "SKIP tag={label}-cont fam={fam} reason=serial-one-row-reference-writes-nothing"
            ));
            continue;
        }
        let d = match *fam {
            "dn_s" => dn_state_c,
            "dn_ef" => dn_ef_c,
            _ => fam_delta(
                plain_kind(fam),
                &read_sidecar(ref_dir, &ctag, fam),
                &read_sidecar(ver_dir, &cgot, fam),
            ),
        };
        log.push(format!(
            "CONTSERIAL tag={label}-cont fam={fam} rel={:.3e} max_abs={:.3e} scale={:.3e} within_2pct={}",
            d.rel,
            d.max_abs,
            d.scale,
            u8::from(d.rel <= CPU_REL_TOL)
        ));
    }

    // Continuation same-route exactness: the truncated window's native `k=1`
    // continuation (`{prefix}TC-<label>`) runs from the byte-identical retained
    // state with the same committed token, so it must reproduce this window's
    // continuation (`{prefix}C-<label>`) exactly — all ten families, including
    // the head KV and the GDN tape that a two-row step writes. This is the
    // continuation acceptance criterion (strictly stronger than a tolerance
    // across routes).
    for fam in FAMS {
        let want = read_sidecar(ver_dir, &format!("{snap_prefix}TC-{label}"), fam);
        let got = read_sidecar(ver_dir, &cgot, fam);
        let same = want == got;
        log.push(format!(
            "SAMEROUTE tag={label}-cont fam={fam} bytes={} byte_identical={}",
            got.len(),
            u8::from(same)
        ));
        assert!(
            same,
            "{label}-cont family {fam}: continuation differs from the same-route truncated \
             continuation ({} vs {} bytes)",
            want.len(),
            got.len()
        );
    }

    // Negative control: a rejected tail's retained state must match the ACCEPTED
    // prefix and must not match the un-rolled-back (whole-batch) state. Uses the
    // same coupled representation as the trial comparison above.
    if accept < k {
        let bad_tag = format!("BAD-{label}");
        let (dn_state_b, dn_ef_b) = dn_coupled_delta(ref_dir, &bad_tag, ver_dir, &got_tag);
        let mut agg_good = 0f32;
        let mut agg_bad = 0f32;
        for fam in CPU_BAD_FAMS {
            let d = match fam {
                "dn_s" => dn_state_b,
                "dn_ef" => dn_ef_b,
                _ => fam_delta(
                    plain_kind(fam),
                    &read_sidecar(ref_dir, &bad_tag, fam),
                    &read_sidecar(ver_dir, &got_tag, fam),
                ),
            };
            let good_rel = good_rels[fam_idx(fam)];
            agg_bad = agg_bad.max(d.rel);
            agg_good = agg_good.max(good_rel);
            log.push(format!(
                "BAD tag={label} fam={fam} rel_good={good_rel:.3e} rel_bad={:.3e} max_abs_bad={:.3e}",
                d.rel, d.max_abs
            ));
        }
        assert!(
            agg_bad > CPU_BAD_MIN_RATIO * CPU_REL_TOL,
            "{label}: un-rolled-back state differs by only {agg_bad:.3e} — the rollback check is not discriminating"
        );
        assert!(
            agg_bad >= CPU_BAD_MIN_RATIO * agg_good,
            "{label}: un-rolled-back delta {agg_bad:.3e} is not ≫ the retained-state delta {agg_good:.3e}"
        );
        log.push(format!(
            "BADCHECK tag={label} agg_good={agg_good:.3e} agg_bad={agg_bad:.3e} ok=1"
        ));
    }
}

fn cpu_spawn_worker(
    exe: &Path,
    arm: &str,
    out_dir: &Path,
    model: &Path,
    head: &Path,
    toks_file: Option<&Path>,
) -> String {
    eprintln!("[cpu-oracle-cmp] spawning {arm} worker (HIPFIRE_OFFLOAD_EXEC=cpu) ...");
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--exact")
        .arg("mtp_cpu_offload_rollback_oracle")
        .arg("--ignored")
        .arg("--nocapture")
        .env(CPU_ARM_ENV, arm)
        .env(CPU_OUT_ENV, out_dir)
        .env(CPU_REQUIRE_ENV, "1")
        .env(CPU_OFFLOAD_ENV, "cpu")
        .env("HIPFIRE_MTP_BYTE_IDENTITY_MODEL", model)
        .env("HIPFIRE_MTP_BYTE_IDENTITY_HEAD", head);
    if let Some(p) = toks_file {
        cmd.env(CPU_TOKS_ENV, p);
    }
    let status = cmd.status().expect("spawn cpu worker");
    assert!(status.success(), "{arm} cpu worker failed with {status}");
    std::fs::read_to_string(out_dir.join("digests.log")).expect("read worker digests")
}

fn run_cpu_orchestrator() {
    let require = std::env::var(CPU_REQUIRE_ENV).as_deref() == Ok("1");
    let Some((model, head)) = model_paths() else {
        if require {
            panic!(
                "{CPU_REQUIRE_ENV}=1 but HIPFIRE_MTP_BYTE_IDENTITY_MODEL is unset — the \
                 CPU-offload MTP oracle was explicitly requested"
            );
        }
        eprintln!("skipping cpu oracle: HIPFIRE_MTP_BYTE_IDENTITY_MODEL not set");
        return;
    };
    assert!(
        model.is_file(),
        "explicitly requested fixture missing: {}",
        model.display()
    );
    assert!(
        head.is_file(),
        "explicitly requested MTP sidecar missing: {}",
        head.display()
    );
    let exe = std::env::current_exe().expect("current_exe");
    let root = std::env::var(OUT_ROOT_ENV).map(PathBuf::from).ok();

    // Sequential fresh processes: exactly one 35B session is ever resident.
    let ref_dir = unique_out_dir(root.clone(), "cpu-ref");
    let ref_log = cpu_spawn_worker(&exe, "ref", &ref_dir, &model, &head, None);
    let ref_line = ref_log
        .lines()
        .find(|l| l.starts_with("REF toks="))
        .expect("reference worker recorded no token trail");
    let toks = cpu_parse_ids(win_field(ref_line, "toks"), "reference toks");
    assert!(
        toks.len() > CPU_REF_WINDOW,
        "reference trail has only {} tokens (need > {CPU_REF_WINDOW})",
        toks.len()
    );
    let verify_dir = unique_out_dir(root, "cpu-verify");
    let toks_file = verify_dir.join("ref_toks.txt");
    std::fs::create_dir_all(&verify_dir).expect("create verify dir");
    std::fs::write(
        &toks_file,
        toks.iter().map(|t| format!("{t}\n")).collect::<String>(),
    )
    .expect("write reference tokens");
    let ver_log = cpu_spawn_worker(&exe, "verify", &verify_dir, &model, &head, Some(&toks_file));

    // Evidence for the whole judging pass; written even if an assertion fires.
    let mut judge = JudgeLog::new(verify_dir.join("judge.log"));
    let mut per_forward: Option<usize> = None;
    for (role, log) in [("ref", &ref_log), ("verify", &ver_log)] {
        let oracle = log
            .lines()
            .find(|l| l.starts_with("CPU-ORACLE "))
            .unwrap_or_else(|| panic!("{role}: no CPU-ORACLE line"));
        assert_eq!(win_field(oracle, "exec"), "cpu", "{role}: {oracle}");
        assert_eq!(win_field(oracle, "cpu_exec"), "1", "{role}: {oracle}");
        assert_eq!(
            win_field(oracle, "moe_cpu_experts"),
            "1",
            "{role}: {oracle}"
        );
        assert_eq!(
            win_field(oracle, "host_cpu_weights"),
            "1",
            "{role}: {oracle}"
        );
        let host = log
            .lines()
            .find(|l| l.starts_with("HOST "))
            .unwrap_or_else(|| panic!("{role}: no HOST line"));
        assert!(host.contains("host_expert_layers=16"), "{role}: {host}");
        assert!(host.contains("host_cpu_weights=1"), "{role}: {host}");
        assert!(
            log.lines()
                .any(|l| l.starts_with("PLACEMENT explicit-fixed-first16")),
            "{role}: explicit fixed placement line missing"
        );
        let calib = log
            .lines()
            .find(|l| l.starts_with("CALIB ") || l.starts_with("COUNTERS role=ref"))
            .unwrap_or_else(|| panic!("{role}: no CALIB/COUNTERS line"));
        let m: usize = win_field(calib, "per_forward")
            .parse()
            .expect("per_forward");
        assert!(
            m > 0,
            "{role}: per-forward CPU splice count is 0 — the CPU expert path never ran"
        );
        let prefill_steps: usize = win_field(calib, "prefill_steps")
            .parse()
            .expect("prefill_steps");
        assert_eq!(
            prefill_steps, 0,
            "{role}: wide prompt prefill took {prefill_steps} CPU steps (must stay on the GPU route)"
        );
        match per_forward {
            None => per_forward = Some(m),
            Some(m0) => assert_eq!(
                m, m0,
                "the two processes armed different host MoE layer sets"
            ),
        }
        let counters = log
            .lines()
            .find(|l| l.starts_with(&format!("COUNTERS role={role} ")))
            .unwrap_or_else(|| panic!("{role}: no final COUNTERS line"));
        assert_eq!(win_field(counters, "leak"), "0", "{role}: {counters}");
        assert_eq!(
            win_field(counters, "graph_captures"),
            "0",
            "{role}: capture ran under CPU offload"
        );
        assert_eq!(
            win_field(counters, "graph_replays"),
            "0",
            "{role}: replay ran under CPU offload"
        );
        assert!(
            win_field(counters, "on_cpu")
                .parse::<usize>()
                .expect("on_cpu")
                > 0
        );
        judge.push(format!("ROLE {role} ok=1 per_forward={m} prefill_steps=0"));
    }
    let m = per_forward.expect("no worker reported a per-forward CPU splice count");

    let trials: Vec<CpuTrialLine> = ver_log
        .lines()
        .filter(|l| l.starts_with("TRIAL "))
        .map(cpu_parse_trial)
        .collect();
    assert_eq!(
        trials.len(),
        CPU_TRIAL_EXPECT.len(),
        "verify worker recorded {} trials, contract has {}",
        trials.len(),
        CPU_TRIAL_EXPECT.len()
    );
    for (label, ..) in CPU_TRIAL_EXPECT {
        assert!(
            trials.iter().any(|t| t.tag == label),
            "trial {label} missing — the CPU rollback contract was not fully exercised"
        );
    }
    let vocab = win_field(
        ver_log
            .lines()
            .find(|l| l.starts_with("MODEL "))
            .expect("verify MODEL line"),
        "vocab",
    )
    .parse()
    .expect("vocab");
    for t in &trials {
        cpu_judge_trial(&ref_dir, &verify_dir, &mut judge, t, "V", &toks, m, vocab);
    }

    // Reset + second request: byte-identical sidecars and the same verdict.
    let reset_line = ver_log
        .lines()
        .find(|l| l.starts_with("RESET "))
        .expect("verify worker recorded no reset/second-request trial");
    let reset = cpu_parse_trial(reset_line);
    cpu_judge_trial(
        &ref_dir,
        &verify_dir,
        &mut judge,
        &reset,
        "V2",
        &toks,
        m,
        vocab,
    );
    assert!(
        ver_log
            .lines()
            .any(|l| l.starts_with("RESETCHECK ") && l.contains("byte_identical=1")),
        "reset/second request byte-identity check missing"
    );

    // Every reference depth the judging needed must exist (no silent pass on a
    // missing snapshot). The same-route families are deliberately absent from the
    // serial reference's sidecars.
    for d in 1..=CPU_REF_WINDOW {
        for fam in FAMS {
            if CPU_SAME_ROUTE_FAMS.contains(&fam) {
                continue;
            }
            read_sidecar(&ref_dir, &format!("REF-d{d}"), fam);
        }
    }
    for t in &trials {
        if t.accept < t.k {
            for fam in CPU_BAD_FAMS {
                read_sidecar(&ref_dir, &format!("BAD-{}", t.tag), fam);
            }
        }
    }

    judge.finish();
    drop(judge);
    eprintln!(
        "[cpu-oracle-cmp] PASS trials={} per_forward={m} ref={} verify={} judge={}",
        trials.len() + 1,
        ref_dir.display(),
        verify_dir.display(),
        verify_dir.join("judge.log").display()
    );
}

#[test]
#[ignore = "requires real HIP GPU (gfx1201) + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (qwen3.6-35b-a3b.mq4p) + MTP sidecar + HIPFIRE_OFFLOAD_EXEC=cpu worker env"]
fn mtp_cpu_offload_rollback_oracle() {
    let Some(arm) = std::env::var(CPU_ARM_ENV).ok() else {
        run_cpu_orchestrator();
        return;
    };
    assert!(
        arm == "ref" || arm == "verify",
        "{CPU_ARM_ENV} must be `ref` or `verify`, got {arm:?}"
    );
    let out = std::env::var(CPU_OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            unique_out_dir(std::env::var(OUT_ROOT_ENV).map(PathBuf::from).ok(), &arm)
        });
    if arm == "ref" {
        run_cpu_ref_worker(out);
    } else {
        run_cpu_verify_worker(out);
    }
}
