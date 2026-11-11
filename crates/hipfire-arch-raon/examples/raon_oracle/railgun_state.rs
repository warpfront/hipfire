// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `railgun-state`: the PRODUCT Raon railgun state oracle (raon-redline
//! PLAN.md section 5 state invariants and section 8.7).
//!
//! One arm runs in ONE FRESH PROCESS from the same prefill preimage, then `--steps`
//! sequential greedy decode steps on real, changing input ids (the argmax of step `i`
//! is the input of step `i+1`; this is not a fixed-state replay). The railgun arm is
//! driven only through the product route: it sets `HIPFIRE_RAON_RAILGUN=1` and
//! `HIPFIRE_RAON_RAILGUN_CORPUS=<dir>` in the process environment before the bundle
//! is loaded and calls `RaonBundle::decode_step`; the oracle never authors or lowers
//! a program. Each step is classified from the owner's counters
//! (`RaonBundle::railgun_stats` deltas): executed by railgun, declined and served by
//! the HIP graph, owner idle, or owner absent. A railgun arm that executed zero steps
//! is a FAIL, and every fallback to the graph is visible per step and in the summary.
//!
//! ```text
//! raon_oracle railgun-state --model M --arm graph|railgun [--corpus DIR] --ctx 127,128,129 --steps 256
//!                           --out ARM.json [--dump DIR] [--prompt-file F] [--boundaries]
//!                           [--digest-every 1] [--raw-at 0,100|all] [--raw-prefix 1] [--poison 0]
//!                           [--max-declined 0] [--mirror-release-at STEP] [--g4 TABLE.json]
//!                           [--weights-digest 1] [--model-sha256 HEX|skip]
//! raon_oracle railgun-state --compare GRAPH.json RAILGUN.json [--out F] [--flip-byte STEP:TENSOR:OFFSET]
//! raon_oracle railgun-state --compare-dir GRAPH_DUMP RAILGUN_DUMP [--out F] [--flip-byte STEP:TENSOR:OFFSET]
//! ```
//!
//! Recorded after the prefill (step 0) and after EVERY decode step (step `i`):
//!
//! * the full logits bytes (`logits.bin`, one vector per step) and their digest128;
//! * the persistent slab (RDA matrices, q/k/v conv histories, device `seq_len`,
//!   alignment padding: `RaonState::download_persist`) as digest128 per region every
//!   step, raw bytes at step 0, the last step and `--raw-at`;
//! * the live MLA latent rows `[0, p+i)` of every layer and the live expanded K/V
//!   mirror rows `[0, mirror_valid_rows)` (digest128 every `--digest-every` steps and
//!   at both ends; raw at the raw steps unless `--raw-prefix 0`) plus the newly
//!   written row `p+i-1` of every latent layer and mirror plane, raw, every step;
//! * the mHC completion streams (`stream_a`, `stream_b` row 0, `final_mean`,
//!   `final_norm`), raw every step;
//! * host `n_tokens`, device `seq_len`, the mirror validity watermark, and the
//!   physical capacity / mirror rows (informational);
//! * the model weights as digest128 before the first and after the last case (they
//!   must be bit-identical). The digest covers the immutable weight bytes only
//!   (`weights_digest_scope`): the loader-owned MoE pointer tables hold device virtual
//!   addresses, which differ between processes, so they are excluded and checked
//!   separately: each must hold exactly the addresses of the allocations it names
//!   (`pointer_tables_*.consistent`) and must not change during an arm (raw digest,
//!   before vs after within one process; never compared across arms).
//!
//! Only live, defined bytes are recorded: latent rows beyond `n_tokens` and mirror
//! rows beyond the watermark are never read. `--poison 1` fills the backed latent tail
//! `[n_tokens, capacity_rows)` with `0xA5` after the prefill in every case (both arms),
//! so a program that reads an unbacked/unwritten row diverges instead of passing on
//! zeros. The mirror tail is ignored, not poisoned (its backing is chunk-mapped).
//!
//! `--ctx` takes a list; every ctx is one case of the same process (reset, mirror
//! released so every case starts with the same mirror history, batched prefill,
//! `--steps` decode steps). `--boundaries` adds the 127/128/129 samples. Growth
//! crossings (latent capacity, mirror rows) are visible per step as `capacity_rows` /
//! `mirror_rows` / `mirror_valid_rows`; pick ctx values around a reported
//! `capacity_rows` to sample a map-growth edge. `--mirror-release-at STEP` releases
//! the expanded mirror before decode step STEP (both arms), forcing a binding change
//! the graph arm re-records and the railgun owner must rebuild for.
//!
//! `--compare` compares two arm dumps byte for byte (graph vs railgun): logits every
//! step, persist slab, live latent/mirror, streams, positions. The first divergence is
//! reported with case ctx, step, tensor and byte offset; the exit status is non-zero
//! on any mismatch, on an arm whose own verdict failed, and on a requested-railgun arm
//! that executed no step. `--flip-byte` is the negative control: the byte is flipped in
//! the loaded copy of the SECOND dump, the comparator must report exactly that byte,
//! and the verdict is OK only if it does (and the unflipped comparison is clean).
//!
//! Every use of the owner API is in `mod owner` below.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use hip_bridge::DeviceBuffer;
use hipfire_arch_raon::bundle::{DecodeMode, PrefillMode, RaonBundle};
use hipfire_arch_raon::state::{conv_plane_bytes, mirror_plane_row_bytes, RaonState};
use hipfire_arch_raon::weights::{GatedFfnWeights, RaonFfn, RaonMixer, RaonWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};

use super::logits::load_bundle;

const SCHEMA: &str = "raon-railgun-state/1";
const DEFAULT_PROMPT: &str = "benchmarks/prompts/raon/state.txt";
const POISON_BYTE: u8 = 0xA5;

pub const USAGE: &str = "usage:
  raon_oracle railgun-state --model M --arm graph|railgun [--corpus DIR] [--ctx 127,128,129 | --boundaries] [--steps 256]
                            --out ARM.json [--dump DIR] [--prompt-file F] [--digest-every 1] [--raw-at 0,100|all]
                            [--raw-prefix 1] [--poison 0] [--max-declined 0] [--mirror-release-at STEP]
                            [--g4 TABLE.json] [--weights-digest 1] [--model-sha256 HEX|skip]
  raon_oracle railgun-state --compare GRAPH.json RAILGUN.json [--out F] [--flip-byte STEP:TENSOR:OFFSET]
  raon_oracle railgun-state --compare-dir GRAPH_DUMP RAILGUN_DUMP [--out F] [--flip-byte STEP:TENSOR:OFFSET]
(--corpus is required by, and only used for, --arm railgun)";

pub enum Exit {
    Usage(String),
    Fail(String),
}

// ───────────────────────── the railgun owner API (one spot) ─────────────────────────

/// Every use of the railgun owner API by this oracle. A signature drift of
/// `RaonBundle::railgun_stats` / `RaonRailgunStats` is a one-spot fix here.
mod owner {
    use hipfire_arch_raon::bundle::RaonBundle;
    use hipfire_arch_raon::railgun_decode::RaonRailgunStats;
    use serde_json::{json, Value};

    /// Process environment of the product route (read once by `RaonBundle::new`).
    pub const ENV_ON: &str = "HIPFIRE_RAON_RAILGUN";
    pub const ENV_CORPUS: &str = "HIPFIRE_RAON_RAILGUN_CORPUS";
    /// Optional G4 cache table of the owner.
    pub const ENV_G4: &str = "HIPFIRE_RAILGUN_CACHE_TABLE";

    /// Copy of [`RaonRailgunStats`] (every field named, none elided).
    #[derive(Clone, Debug)]
    pub struct Snapshot {
        pub attempts: u64,
        pub completed: u64,
        pub declined: u64,
        pub builds: u64,
        pub rebuilds: u64,
        pub quarantined: bool,
        pub last_refusal: Option<String>,
        /// Declined steps by refusal kind (`RaonRailgunRefusal::kind`), as the owner counts them.
        pub declined_by_kind: std::collections::BTreeMap<String, u64>,
        pub nodes: usize,
        pub ib_dwords: u32,
        pub object_digests: Vec<(String, String)>,
        pub last_build_ms: f64,
        pub last_rebuild_cause: Option<String>,
    }

    impl Snapshot {
        pub fn json(&self) -> Value {
            json!({
                "attempts": self.attempts,
                "completed": self.completed,
                "declined": self.declined,
                "builds": self.builds,
                "rebuilds": self.rebuilds,
                "quarantined": self.quarantined,
                "last_refusal": self.last_refusal,
                "declined_by_kind": self.declined_by_kind,
                "nodes": self.nodes,
                "ib_dwords": self.ib_dwords,
                "last_build_ms": self.last_build_ms,
                "last_rebuild_cause": self.last_rebuild_cause,
                "object_digests": self.object_digests.iter()
                    .map(|(symbol, sha256)| json!({"symbol": symbol, "sha256": sha256}))
                    .collect::<Vec<_>>(),
            })
        }
    }

    fn from_stats(s: RaonRailgunStats) -> Snapshot {
        let RaonRailgunStats {
            attempts,
            completed,
            declined,
            builds,
            rebuilds,
            quarantined,
            last_refusal,
            declined_by_kind,
            nodes,
            ib_dwords,
            object_digests,
            last_build_ms,
            last_rebuild_cause,
        } = s;
        Snapshot {
            attempts,
            completed,
            declined,
            builds,
            rebuilds,
            quarantined,
            last_refusal,
            declined_by_kind,
            nodes,
            ib_dwords,
            object_digests,
            last_build_ms,
            last_rebuild_cause,
        }
    }

    /// The owner's counters; `None` when the owner is off.
    pub fn snapshot(bundle: &RaonBundle) -> Option<Snapshot> {
        bundle.railgun_stats().map(from_stats)
    }
}

// ───────────────────────── small helpers ─────────────────────────

fn hip_err<T>(r: hip_bridge::HipResult<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e:?}"))
}

fn io_err<T>(r: std::io::Result<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e}"))
}

fn md5_hex(bytes: &[u8]) -> String {
    format!("{:x}", md5::compute(bytes))
}

fn ids_md5(ids: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(ids.len() * 4);
    for &t in ids {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    md5_hex(&bytes)
}

/// First maximum (lowest index on ties): the greedy next token.
fn first_argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Two independent 64-bit digests of `bytes` (FNV-1a over u64 words and a
/// multiply-rotate mix); a difference in either is a difference in the bytes.
fn digest128(bytes: &[u8]) -> (u64, u64) {
    let mut a = 0xcbf2_9ce4_8422_2325u64;
    let mut b = 0x9e37_79b9_7f4a_7c15u64;
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        a = (a ^ w).wrapping_mul(0x0000_0100_0000_01b3);
        b = (b.rotate_left(23) ^ w).wrapping_mul(0xff51_afd7_ed55_8ccd);
    }
    for &r in chunks.remainder() {
        a = (a ^ r as u64).wrapping_mul(0x0000_0100_0000_01b3);
        b = (b.rotate_left(23) ^ r as u64).wrapping_mul(0xff51_afd7_ed55_8ccd);
    }
    (a, b ^ (bytes.len() as u64))
}

const SLICE: usize = 16 << 20;

/// digest128 of every 16 MiB slice of `bytes`, hashed across threads, in order.
fn hash_slices(bytes: &[u8]) -> Vec<(u64, u64)> {
    let slices: Vec<&[u8]> = bytes.chunks(SLICE).collect();
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let mut out = vec![(0u64, 0u64); slices.len()];
    let per = slices.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (o, i) in out.chunks_mut(per).zip(slices.chunks(per)) {
            scope.spawn(move || {
                for (d, s) in o.iter_mut().zip(i) {
                    *d = digest128(s);
                }
            });
        }
    });
    out
}

fn fold_digests(ds: &[(u64, u64)], len: usize) -> (u64, u64) {
    let mut acc = Vec::with_capacity(ds.len() * 16 + 8);
    for &(a, b) in ds {
        acc.extend_from_slice(&a.to_le_bytes());
        acc.extend_from_slice(&b.to_le_bytes());
    }
    acc.extend_from_slice(&(len as u64).to_le_bytes());
    digest128(&acc)
}

/// The one digest every recorded buffer uses: `digest128` up to 16 MiB, else the
/// `digest128` of the per-slice digests (identical on both arms by construction).
fn digest(bytes: &[u8]) -> (u64, u64) {
    if bytes.len() <= SLICE {
        digest128(bytes)
    } else {
        fold_digests(&hash_slices(bytes), bytes.len())
    }
}

fn hx(d: (u64, u64)) -> String {
    format!("{:016x}{:016x}", d.0, d.1)
}

fn read_dev(
    gpu: &Gpu,
    buf: &DeviceBuffer,
    off: usize,
    len: usize,
    what: &str,
) -> Result<Vec<u8>, String> {
    if off.checked_add(len).is_none_or(|e| e > buf.size()) {
        return Err(format!(
            "{what}: range [{off}, {off}+{len}) exceeds device extent {}",
            buf.size()
        ));
    }
    let mut host = vec![0u8; len];
    if len > 0 {
        let v = buf.byte_view(off, len);
        hip_err(gpu.hip.memcpy_dtoh(&mut host, &v), what)?;
    }
    Ok(host)
}

fn env_snapshot() -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in std::env::vars() {
        if k.starts_with("HIPFIRE_")
            || k.starts_with("HIP_")
            || k.starts_with("ROCR_")
            || k.starts_with("GPU_")
            || k.starts_with("HSA_")
        {
            m.insert(k, json!(v));
        }
    }
    Value::Object(m)
}

fn uptime() -> String {
    std::fs::read_to_string("/proc/uptime")
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn binary_md5() -> String {
    match std::env::current_exe().and_then(std::fs::read) {
        Ok(b) => md5_hex(&b),
        Err(e) => format!("unavailable: {e}"),
    }
}

fn controls() -> Value {
    json!({
        "uptime_s": uptime(),
        "binary_md5": binary_md5(),
        "host": std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim(),
        "gpu_max_hw_queues": std::env::var("GPU_MAX_HW_QUEUES").ok(),
        "env": env_snapshot(),
    })
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn first_field(text: &str) -> Option<String> {
    text.split_whitespace().next().map(str::to_string)
}

/// Model identity: size, mtime, SHA256 (`--model-sha256`, else the `<model>.sha256`
/// sidecar, else computed once with `sha256sum` and written to the sidecar when the
/// directory allows) and the `<model>.md5` sidecar when present (as `bench` pins it).
fn model_identity(model: &str, sha_opt: Option<&str>) -> Result<Value, String> {
    let meta = std::fs::metadata(model).map_err(|e| format!("stat {model}: {e}"))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    let sidecar = format!("{model}.sha256");
    let (sha, source): (Option<String>, &str) = match sha_opt {
        Some("skip") => (None, "skipped"),
        Some(h) if is_hex(h, 64) => (Some(h.to_ascii_lowercase()), "cli"),
        Some(h) => return Err(format!("--model-sha256 `{h}`: 64 hex digits or `skip`")),
        None => match std::fs::read_to_string(&sidecar)
            .ok()
            .and_then(|t| first_field(&t))
        {
            Some(h) if is_hex(&h, 64) => (Some(h.to_ascii_lowercase()), "sidecar"),
            Some(_) => return Err(format!("{sidecar}: first field is not a 64-hex sha256")),
            None => {
                let out = std::process::Command::new("sha256sum")
                    .arg(model)
                    .output()
                    .map_err(|e| format!("sha256sum {model}: {e}"))?;
                if !out.status.success() {
                    return Err(format!(
                        "sha256sum {model}: {}",
                        String::from_utf8_lossy(&out.stderr).trim()
                    ));
                }
                let h = first_field(&String::from_utf8_lossy(&out.stdout))
                    .filter(|h| is_hex(h, 64))
                    .ok_or("sha256sum: no 64-hex digest in its output")?
                    .to_ascii_lowercase();
                let name = Path::new(model)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let _ = std::fs::write(&sidecar, format!("{h}  {name}\n"));
                (Some(h), "computed")
            }
        },
    };
    let md5 = std::fs::read_to_string(format!("{model}.md5"))
        .ok()
        .and_then(|t| first_field(&t))
        .filter(|h| is_hex(h, 32));
    Ok(json!({
        "path": model,
        "size_bytes": meta.len(),
        "mtime_unix": mtime,
        "sha256": sha,
        "sha256_source": source,
        "md5_sidecar": md5,
    }))
}

// ───────────────────────── weights digest ─────────────────────────

fn push_g<'a>(v: &mut Vec<(String, &'a DeviceBuffer)>, name: String, t: &'a GpuTensor) {
    v.push((name, &t.buf));
}

fn push_w<'a>(v: &mut Vec<(String, &'a DeviceBuffer)>, name: String, t: &'a WeightTensor) {
    if let Some(a) = &t.awq_scale {
        v.push((format!("{name}.awq_scale"), &a.buf));
    }
    v.push((name, &t.buf.buf));
}

fn push_gated<'a>(v: &mut Vec<(String, &'a DeviceBuffer)>, p: &str, f: &'a GatedFfnWeights) {
    push_w(v, format!("{p}.gate"), &f.gate);
    push_w(v, format!("{p}.up"), &f.up);
    push_w(v, format!("{p}.down"), &f.down);
    for (i, d) in f.decode.iter().enumerate() {
        if let Some(d) = d {
            push_w(v, format!("{p}.decode[{i}]"), d);
        }
    }
}

/// Every device weight buffer reachable from `w`: projections (with AWQ sidecars),
/// embedding, norms, conv taps, mHC selectors/params, router, expert tables.
fn weight_buffers(w: &RaonWeights) -> Vec<(String, &DeviceBuffer)> {
    let mut v = Vec::new();
    if let Some(t) = &w.embed {
        push_g(&mut v, "embed".into(), t);
    }
    if let Some(t) = &w.final_norm {
        push_g(&mut v, "final_norm".into(), t);
    }
    if let Some(t) = &w.lm_head {
        push_w(&mut v, "lm_head".into(), t);
    }
    for (l, layer) in w.layers.iter().enumerate() {
        let p = format!("layer[{l}]");
        push_g(&mut v, format!("{p}.input_norm"), &layer.input_norm);
        push_g(&mut v, format!("{p}.post_attn_norm"), &layer.post_attn_norm);
        for (n, s) in [("attn_mhc", &layer.attn_mhc), ("ffn_mhc", &layer.ffn_mhc)] {
            push_w(&mut v, format!("{p}.{n}.selector"), &s.selector);
            push_g(&mut v, format!("{p}.{n}.params"), &s.params);
        }
        match &layer.mixer {
            RaonMixer::Rda(r) => {
                for (n, t) in [
                    ("q_proj", &r.q_proj),
                    ("k_proj", &r.k_proj),
                    ("v_proj", &r.v_proj),
                    ("g_proj", &r.g_proj),
                    ("o_proj", &r.o_proj),
                    ("f_a_proj", &r.f_a_proj),
                    ("f_b_proj", &r.f_b_proj),
                    ("b_proj", &r.b_proj),
                    ("rda_w_proj", &r.rda_w_proj),
                ] {
                    push_w(&mut v, format!("{p}.rda.{n}"), t);
                }
                for (n, t) in [
                    ("q_conv", &r.q_conv),
                    ("k_conv", &r.k_conv),
                    ("v_conv", &r.v_conv),
                    ("a_log", &r.a_log),
                    ("dt_bias", &r.dt_bias),
                    ("o_norm", &r.o_norm),
                ] {
                    push_g(&mut v, format!("{p}.rda.{n}"), t);
                }
            }
            RaonMixer::Mla(m) => {
                for (n, t) in [
                    ("q_a_proj", &m.q_a_proj),
                    ("q_b_proj", &m.q_b_proj),
                    ("kv_a_proj", &m.kv_a_proj),
                    ("kv_b_proj", &m.kv_b_proj),
                    ("g_proj", &m.g_proj),
                    ("o_proj", &m.o_proj),
                ] {
                    push_w(&mut v, format!("{p}.mla.{n}"), t);
                }
                push_g(&mut v, format!("{p}.mla.q_a_norm"), &m.q_a_norm);
                push_g(&mut v, format!("{p}.mla.kv_a_norm"), &m.kv_a_norm);
            }
        }
        match &layer.ffn {
            RaonFfn::Dense(f) => push_gated(&mut v, &format!("{p}.dense"), f),
            RaonFfn::Moe(m) => {
                push_w(&mut v, format!("{p}.moe.router"), &m.router);
                push_g(&mut v, format!("{p}.moe.e_bias"), &m.e_bias);
                push_gated(&mut v, &format!("{p}.moe.shared"), &m.shared);
                for (e, ex) in m.experts.iter().enumerate() {
                    push_w(&mut v, format!("{p}.moe.expert[{e}].gate_up"), &ex.gate_up);
                    push_w(&mut v, format!("{p}.moe.expert[{e}].down"), &ex.down);
                }
                // The loader-owned pointer tables (`gate_up_ptrs`, `down_ptrs`, `down_awq_ptrs`) are NOT
                // weights: they hold device virtual addresses, which differ between processes. They are
                // checked separately by `pointer_tables`.
                if let Some(t) = &m.gate_up_awq {
                    push_g(&mut v, format!("{p}.moe.gate_up_awq"), t);
                }
                for (e, t) in m.expert_down_awq.iter().enumerate() {
                    push_g(&mut v, format!("{p}.moe.expert[{e}].down_awq"), t);
                }
            }
        }
    }
    v
}

/// The loader-owned device pointer tables of the MoE layers, each with the addresses it must hold: one
/// `u64` per expert, the base of that expert's `gate_up` / `down` allocation (`weights.rs` `ptr_table`,
/// `seal_experts`) or of its down-AWQ tensor. `(name, table buffer, expected entries)`.
fn pointer_tables(w: &RaonWeights) -> Vec<(String, &DeviceBuffer, Vec<u64>)> {
    let mut v = Vec::new();
    for (l, layer) in w.layers.iter().enumerate() {
        let RaonFfn::Moe(m) = &layer.ffn else {
            continue;
        };
        let p = format!("layer[{l}].moe");
        v.push((
            format!("{p}.gate_up_ptrs"),
            &m.expert_gate_up_ptrs.buf,
            m.experts
                .iter()
                .map(|x| x.gate_up.buf.buf.as_ptr() as u64)
                .collect(),
        ));
        v.push((
            format!("{p}.down_ptrs"),
            &m.expert_down_ptrs.buf,
            m.experts
                .iter()
                .map(|x| x.down.buf.buf.as_ptr() as u64)
                .collect(),
        ));
        if let Some(t) = &m.expert_down_awq_ptrs {
            v.push((
                format!("{p}.down_awq_ptrs"),
                &t.buf,
                m.expert_down_awq
                    .iter()
                    .map(|x| x.buf.as_ptr() as u64)
                    .collect(),
            ));
        }
    }
    v
}

/// Scope of `weights_digest_before/after`: every immutable weight byte, and nothing that holds device
/// addresses. Dumps with another (or no) scope are not comparable.
const WEIGHTS_DIGEST_SCOPE: &str = "immutable-bytes-v2";

/// The pointer tables as read back: each must hold exactly the addresses of the allocations it names
/// (`consistent`), and its raw bytes must not change during an arm (`digest128`, compared before vs after
/// WITHIN a process only: the addresses themselves are process-specific and never compared across arms).
fn pointer_tables_summary(gpu: &Gpu, w: &RaonWeights) -> Result<Value, String> {
    let tables = pointer_tables(w);
    let mut acc: Vec<u8> = Vec::new();
    let mut entries = 0usize;
    let mut mismatches: Vec<Value> = Vec::new();
    let mut bad = 0usize;
    for (name, b, want) in &tables {
        let size = b.size();
        let mut host = vec![0u8; size];
        let v = b.byte_view(0, size);
        hip_err(
            gpu.hip.memcpy_dtoh(&mut host[..], &v),
            &format!("pointer table read-back {name}"),
        )?;
        let got: Vec<u64> = host
            .chunks_exact(8)
            .take(want.len())
            .map(|c| u64::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        if got.len() != want.len() {
            bad += 1;
            mismatches.push(json!({"table": name, "detail": format!("{} entries read, {} expected", got.len(), want.len())}));
        }
        for (i, (g, e)) in got.iter().zip(want).enumerate() {
            if g != e {
                bad += 1;
                if mismatches.len() < 8 {
                    mismatches.push(json!({"table": name, "index": i, "got": format!("{g:#x}"), "want": format!("{e:#x}")}));
                }
            }
        }
        entries += want.len();
        acc.extend_from_slice(&(size as u64).to_le_bytes());
        acc.extend_from_slice(&host);
    }
    Ok(json!({
        "tables": tables.len(),
        "entries": entries,
        "consistent": bad == 0,
        "mismatched_entries": bad,
        "mismatches": mismatches,
        "digest128": hx(digest128(&acc)),
    }))
}

/// One digest over every weight buffer (read back in 256 MiB chunks, hashed in
/// 16 MiB slices across threads). Taken before the first and after the last case.
/// `pointer_tables` is the separate consistency record of the loader-owned address tables.
fn weights_digest(gpu: &Gpu, w: &RaonWeights) -> Result<Value, String> {
    const CHUNK: usize = 256 << 20;
    let bufs = weight_buffers(w);
    let mut host: Vec<u8> = Vec::new();
    let mut total = 0usize;
    let mut acc: Vec<u8> = Vec::new();
    for (name, b) in &bufs {
        let size = b.size();
        let mut slice_ds: Vec<(u64, u64)> = Vec::new();
        let mut off = 0usize;
        while off < size {
            let n = CHUNK.min(size - off);
            host.resize(n, 0);
            let v = b.byte_view(off, n);
            hip_err(
                gpu.hip.memcpy_dtoh(&mut host[..n], &v),
                &format!("weights read-back {name}"),
            )?;
            slice_ds.extend(hash_slices(&host[..n]));
            off += n;
        }
        let td = fold_digests(&slice_ds, size);
        acc.extend_from_slice(&(size as u64).to_le_bytes());
        acc.extend_from_slice(&td.0.to_le_bytes());
        acc.extend_from_slice(&td.1.to_le_bytes());
        total += size;
    }
    Ok(json!({
        "scope": WEIGHTS_DIGEST_SCOPE,
        "tensors": bufs.len(),
        "bytes": total,
        "digest128": hx(digest128(&acc)),
        "pointer_tables": pointer_tables_summary(gpu, w)?,
    }))
}

// ───────────────────────── options ─────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Graph,
    Railgun,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Graph => "graph",
            Arm::Railgun => "railgun",
        }
    }
}

struct ArmOpts {
    model: String,
    arm: Arm,
    corpus: Option<String>,
    ctxs: Vec<usize>,
    steps: usize,
    prompt_file: String,
    out: PathBuf,
    dump: PathBuf,
    digest_every: usize,
    raw_all: bool,
    raw_steps: BTreeSet<usize>,
    raw_prefix: bool,
    poison: bool,
    max_declined: u64,
    mirror_release_at: Option<usize>,
    g4: Option<String>,
    weights_digest: bool,
    model_sha256: Option<String>,
    argv: Vec<String>,
}

struct Args {
    kv: BTreeMap<String, String>,
    compare: Option<(String, String)>,
    compare_dir: Option<(String, String)>,
    boundaries: bool,
}

const VALUE_FLAGS: [&str; 18] = [
    "model",
    "arm",
    "corpus",
    "ctx",
    "steps",
    "prompt-file",
    "out",
    "dump",
    "digest-every",
    "raw-at",
    "raw-prefix",
    "poison",
    "max-declined",
    "mirror-release-at",
    "g4",
    "weights-digest",
    "model-sha256",
    "flip-byte",
];

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        kv: BTreeMap::new(),
        compare: None,
        compare_dir: None,
        boundaries: false,
    };
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i]
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument `{}`", argv[i]))?;
        match key {
            "boundaries" => {
                a.boundaries = true;
                i += 1;
            }
            "compare" | "compare-dir" => {
                let (x, y) = match (argv.get(i + 1), argv.get(i + 2)) {
                    (Some(x), Some(y)) => (x.clone(), y.clone()),
                    _ => return Err(format!("`--{key}` needs two paths")),
                };
                let slot = if key == "compare" {
                    &mut a.compare
                } else {
                    &mut a.compare_dir
                };
                if slot.replace((x, y)).is_some() {
                    return Err(format!("`--{key}` given twice"));
                }
                i += 3;
            }
            k if VALUE_FLAGS.contains(&k) => {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| format!("`--{k}` needs a value"))?;
                if a.kv.insert(k.to_string(), v.clone()).is_some() {
                    return Err(format!("`--{k}` given twice"));
                }
                i += 2;
            }
            other => return Err(format!("unknown flag `--{other}`")),
        }
    }
    Ok(a)
}

fn opt_usize(kv: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize, String> {
    match kv.get(key) {
        Some(s) => s.trim().parse().map_err(|e| format!("--{key} {s:?}: {e}")),
        None => Ok(default),
    }
}

fn opt_bool(kv: &BTreeMap<String, String>, key: &str, default: bool) -> Result<bool, String> {
    match kv.get(key).map(String::as_str) {
        None => Ok(default),
        Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => Err(format!("--{key} {other:?}: 0 or 1")),
    }
}

fn parse_list(s: &str, what: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|x| {
            x.trim()
                .parse::<usize>()
                .map_err(|e| format!("{what} {x:?}: {e}"))
        })
        .collect()
}

fn parse_arm_opts(a: &Args, argv: &[String]) -> Result<ArmOpts, String> {
    let kv = &a.kv;
    let model = kv.get("model").ok_or("missing `--model`")?.clone();
    let arm = match kv.get("arm").map(String::as_str) {
        Some("graph") => Arm::Graph,
        Some("railgun") => Arm::Railgun,
        Some(o) => return Err(format!("--arm `{o}`: graph|railgun")),
        None => return Err("missing `--arm graph|railgun`".into()),
    };
    let corpus = kv.get("corpus").cloned();
    if arm == Arm::Railgun && corpus.is_none() {
        return Err("--arm railgun needs `--corpus DIR`".into());
    }
    if arm == Arm::Graph && corpus.is_some() {
        return Err("--corpus applies to --arm railgun only".into());
    }
    let mut ctxs = match kv.get("ctx") {
        Some(s) => parse_list(s, "--ctx")?,
        None if a.boundaries => Vec::new(),
        None => vec![512],
    };
    if a.boundaries {
        for c in [127, 128, 129] {
            if !ctxs.contains(&c) {
                ctxs.push(c);
            }
        }
    }
    let mut seen = BTreeSet::new();
    ctxs.retain(|c| seen.insert(*c));
    if ctxs.is_empty() || ctxs.contains(&0) {
        return Err("--ctx: positive prompt lengths".into());
    }
    let steps = opt_usize(kv, "steps", 256)?;
    if steps == 0 {
        return Err("--steps must be >= 1".into());
    }
    let out = PathBuf::from(kv.get("out").ok_or("missing `--out ARM.json`")?);
    let dump = match kv.get("dump") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(format!("{}.dump", out.display())),
    };
    let (mut raw_all, mut raw_steps) = (false, BTreeSet::new());
    if let Some(s) = kv.get("raw-at") {
        if s == "all" {
            raw_all = true;
        } else {
            raw_steps.extend(parse_list(s, "--raw-at")?);
        }
    }
    let max_declined = match kv.get("max-declined") {
        Some(s) => s
            .trim()
            .parse::<u64>()
            .map_err(|e| format!("--max-declined {s:?}: {e}"))?,
        None => 0,
    };
    let mirror_release_at = match kv.get("mirror-release-at") {
        Some(s) => Some(
            s.trim()
                .parse::<usize>()
                .map_err(|e| format!("--mirror-release-at {s:?}: {e}"))?,
        ),
        None => None,
    };
    Ok(ArmOpts {
        model,
        arm,
        corpus,
        ctxs,
        steps,
        prompt_file: kv
            .get("prompt-file")
            .cloned()
            .unwrap_or_else(|| DEFAULT_PROMPT.to_string()),
        out,
        dump,
        digest_every: opt_usize(kv, "digest-every", 1)?,
        raw_all,
        raw_steps,
        raw_prefix: opt_bool(kv, "raw-prefix", true)?,
        poison: opt_bool(kv, "poison", false)?,
        max_declined,
        mirror_release_at,
        g4: kv.get("g4").cloned(),
        weights_digest: opt_bool(kv, "weights-digest", true)?,
        model_sha256: kv.get("model-sha256").cloned(),
        argv: argv.to_vec(),
    })
}

// ───────────────────────── step classification ─────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    /// No owner counters at all (owner off).
    OwnerOff,
    /// The owner completed the step.
    Railgun,
    /// The owner declined (reason); the HIP graph served the step.
    DeclinedGraph(String),
    /// Owner on, but it neither completed nor declined the step.
    OwnerIdle,
    /// Counters moved in a way no single step can produce.
    Anomaly(String),
}

impl Kind {
    fn name(&self) -> &'static str {
        match self {
            Kind::OwnerOff => "owner_off",
            Kind::Railgun => "railgun",
            Kind::DeclinedGraph(_) => "declined_graph",
            Kind::OwnerIdle => "owner_idle",
            Kind::Anomaly(_) => "anomaly",
        }
    }
}

fn classify(before: &Option<owner::Snapshot>, after: &Option<owner::Snapshot>) -> Kind {
    match (before, after) {
        (None, None) => Kind::OwnerOff,
        (Some(b), Some(a)) => {
            let (Some(dc), Some(dd), Some(da)) = (
                a.completed.checked_sub(b.completed),
                a.declined.checked_sub(b.declined),
                a.attempts.checked_sub(b.attempts),
            ) else {
                return Kind::Anomaly("owner counters went backwards".into());
            };
            match (dc, dd) {
                (1, 0) => Kind::Railgun,
                (0, 1) => Kind::DeclinedGraph(
                    a.last_refusal
                        .clone()
                        .unwrap_or_else(|| "<no last_refusal>".to_string()),
                ),
                (0, 0) if da == 0 => Kind::OwnerIdle,
                (0, 0) => Kind::Anomaly(format!("{da} attempts with no outcome")),
                _ => Kind::Anomaly(format!("completed +{dc}, declined +{dd} in one step")),
            }
        }
        _ => Kind::Anomaly("owner counters appeared or disappeared mid-run".into()),
    }
}

#[derive(Default, Clone)]
struct Counts {
    steps: u64,
    owner_off: u64,
    railgun: u64,
    declined_graph: u64,
    owner_idle: u64,
    anomaly: u64,
    /// Railgun-completed steps during which the HIP graph also captured/replayed.
    railgun_steps_with_graph_activity: u64,
    reasons: BTreeMap<String, u64>,
}

impl Counts {
    fn add(&mut self, other: &Counts) {
        self.steps += other.steps;
        self.owner_off += other.owner_off;
        self.railgun += other.railgun;
        self.declined_graph += other.declined_graph;
        self.owner_idle += other.owner_idle;
        self.anomaly += other.anomaly;
        self.railgun_steps_with_graph_activity += other.railgun_steps_with_graph_activity;
        for (k, v) in &other.reasons {
            *self.reasons.entry(k.clone()).or_default() += v;
        }
    }

    fn executing_backend(&self) -> &'static str {
        if self.steps > 0 && self.railgun == self.steps {
            "railgun"
        } else if self.railgun == 0 {
            "graph"
        } else {
            "mixed"
        }
    }

    fn json(&self) -> Value {
        json!({
            "steps": self.steps,
            "steps_railgun": self.railgun,
            "steps_declined_graph": self.declined_graph,
            "steps_owner_idle": self.owner_idle,
            "steps_owner_off": self.owner_off,
            "steps_anomaly": self.anomaly,
            "railgun_steps_with_hip_graph_activity": self.railgun_steps_with_graph_activity,
            "executing_backend": self.executing_backend(),
            "declined_reasons": self.reasons,
        })
    }
}

// ───────────────────────── state capture ─────────────────────────

/// `[name, offset, len]` tiling of the persistent slab `[0, total)` exactly: RDA
/// matrices, q/k/v conv planes, device `seq_len`, and the alignment gaps (`pad@off`).
fn persist_regions(st: &RaonState) -> Result<Vec<(String, usize, usize)>, String> {
    let d = st.dims();
    let l = *st.persist_layout();
    let planes = conv_plane_bytes(d)?;
    let mut r: Vec<(String, usize, usize)> = Vec::new();
    for i in 0..d.n_rda {
        r.push((
            format!("rda[{i}]"),
            l.rda_off + i * l.rda_stride,
            st.rda[i].buf.size(),
        ));
        let base = l.conv_off + i * l.conv_stride;
        r.push((format!("conv_q[{i}]"), base, st.conv_q[i].buf.size()));
        r.push((
            format!("conv_k[{i}]"),
            base + planes[0],
            st.conv_k[i].buf.size(),
        ));
        r.push((
            format!("conv_v[{i}]"),
            base + planes[0] + planes[1],
            st.conv_v[i].buf.size(),
        ));
    }
    r.push(("seq_len".to_string(), l.seq_off, st.seq_len.buf.size()));
    r.sort_by_key(|x| x.1);
    let total = st.persist_part().1;
    let mut out = Vec::with_capacity(r.len() * 2);
    let mut pos = 0usize;
    for (n, o, len) in r {
        if o < pos {
            return Err(format!(
                "persist region {n} at {o} overlaps the previous end {pos}"
            ));
        }
        if o > pos {
            out.push((format!("pad@{pos}"), pos, o - pos));
        }
        out.push((n, o, len));
        pos = o + len;
    }
    if pos > total {
        return Err(format!("persist regions end at {pos} > slab {total}"));
    }
    if pos < total {
        out.push((format!("pad@{pos}"), pos, total - pos));
    }
    Ok(out)
}

/// Files of one case: logits (one vector per step) and raw records.
struct CaseFiles {
    logits: BufWriter<File>,
    raw: BufWriter<File>,
    raw_off: u64,
    raw_index: Vec<Value>,
}

impl CaseFiles {
    fn create(dir: &Path) -> Result<Self, String> {
        io_err(std::fs::create_dir_all(dir), &dir.display().to_string())?;
        let mk = |n: &str| -> Result<BufWriter<File>, String> {
            let p = dir.join(n);
            Ok(BufWriter::with_capacity(
                8 << 20,
                io_err(File::create(&p), &p.display().to_string())?,
            ))
        };
        Ok(Self {
            logits: mk("logits.bin")?,
            raw: mk("raw.bin")?,
            raw_off: 0,
            raw_index: Vec::new(),
        })
    }

    fn put_logits(&mut self, bytes: &[u8]) -> Result<(), String> {
        io_err(self.logits.write_all(bytes), "write logits.bin")
    }

    fn put_raw(&mut self, step: usize, name: &str, bytes: &[u8]) -> Result<(), String> {
        io_err(self.raw.write_all(bytes), "write raw.bin")?;
        self.raw_index.push(json!({
            "step": step, "name": name, "off": self.raw_off, "len": bytes.len(),
            "digest128": hx(digest(bytes)),
        }));
        self.raw_off += bytes.len() as u64;
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<Value>, String> {
        io_err(self.logits.flush(), "flush logits.bin")?;
        io_err(self.raw.flush(), "flush raw.bin")?;
        Ok(self.raw_index)
    }
}

struct Captured {
    rec: Value,
    logits: Vec<f32>,
    n_tokens: usize,
    device_seq_len: i32,
}

struct CapturePlan {
    /// Full-prefix digests of latent/mirror at this step.
    prefix_digest: bool,
    /// Raw persist slab at this step.
    raw_persist: bool,
    /// Raw full live latent/mirror prefixes at this step.
    raw_prefix: bool,
}

/// Capture everything the oracle compares at one step (0 = after the prefill).
fn capture_step(
    gpu: &Gpu,
    b: &RaonBundle,
    regions: &[(String, usize, usize)],
    step: usize,
    plan: &CapturePlan,
    files: &mut CaseFiles,
) -> Result<Captured, String> {
    let logits = b.download_logits(gpu)?;
    let lbytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    files.put_logits(&lbytes)?;
    hip_err(gpu.hip.device_synchronize(), "synchronize before capture")?;

    let st = &b.state;
    let d = st.dims();
    let n = st.n_tokens();

    // Persistent slab: RDA, conv, device seq_len.
    let persist = st.download_persist(gpu)?;
    let total = st.persist_part().1;
    if persist.len() != total {
        return Err(format!(
            "download_persist returned {} B, slab is {total} B",
            persist.len()
        ));
    }
    let seq_off = st.persist_layout().seq_off;
    let dev = i32::from_le_bytes(persist[seq_off..seq_off + 4].try_into().unwrap());
    let region_digests: Vec<String> = regions
        .iter()
        .map(|(_, o, l)| hx(digest(&persist[*o..*o + *l])))
        .collect();
    if plan.raw_persist {
        files.put_raw(step, "persist", &persist)?;
    }
    drop(persist);

    // mHC completion streams (decode row 0) and the head chain.
    let sb = d.n_streams * d.hidden * 4;
    let hb = d.hidden * 4;
    let mut stream_digests = serde_json::Map::new();
    for (name, buf, len) in [
        ("stream_a", &b.scratch.stream_a.buf, sb),
        ("stream_b", &b.scratch.stream_b.buf, sb),
        ("final_mean", &b.scratch.final_mean.buf, hb),
        ("final_norm", &b.scratch.final_norm.buf, hb),
    ] {
        let bytes = read_dev(gpu, buf, 0, len, name)?;
        stream_digests.insert(name.to_string(), json!(hx(digest(&bytes))));
        files.put_raw(step, name, &bytes)?;
    }

    // Live MLA latent rows [0, n) and the newly written row n-1.
    let rb = d.latent_row_bytes();
    let (mut lat_d, mut mk_d, mut mv_d) = (Vec::new(), Vec::new(), Vec::new());
    for (l, t) in st.mla_cache.iter().enumerate() {
        if plan.prefix_digest || plan.raw_prefix {
            let bytes = read_dev(gpu, &t.buf, 0, n * rb, &format!("latent[{l}]"))?;
            if plan.prefix_digest {
                lat_d.push(hx(digest(&bytes)));
            }
            if plan.raw_prefix {
                files.put_raw(step, &format!("latent[{l}]"), &bytes)?;
            }
        }
        if step > 0 && n > 0 {
            let row = read_dev(gpu, &t.buf, (n - 1) * rb, rb, &format!("latent_row[{l}]"))?;
            files.put_raw(step, &format!("latent_row[{l}]"), &row)?;
        }
    }

    // Live expanded K/V mirror rows [0, min(valid, rows)).
    let mvalid = st.mirror_valid_rows();
    let mut mirror_rows = 0usize;
    if let Some(m) = st.expanded_mirror() {
        mirror_rows = m.rows;
        let [krow, vrow] = mirror_plane_row_bytes(d);
        let rows = mvalid.min(m.rows);
        for (l, (k, v)) in m.knope.iter().zip(&m.vmir).enumerate() {
            for (tag, plane, row_bytes, dig) in [
                ("mirror_k", k, krow, &mut mk_d),
                ("mirror_v", v, vrow, &mut mv_d),
            ] {
                if plan.prefix_digest || plan.raw_prefix {
                    let bytes =
                        read_dev(gpu, &plane.buf, 0, rows * row_bytes, &format!("{tag}[{l}]"))?;
                    if plan.prefix_digest {
                        dig.push(hx(digest(&bytes)));
                    }
                    if plan.raw_prefix {
                        files.put_raw(step, &format!("{tag}[{l}]"), &bytes)?;
                    }
                }
                if step > 0 && n > 0 && rows >= n {
                    let row = read_dev(
                        gpu,
                        &plane.buf,
                        (n - 1) * row_bytes,
                        row_bytes,
                        &format!("{tag}_row[{l}]"),
                    )?;
                    files.put_raw(step, &format!("{tag}_row[{l}]"), &row)?;
                }
            }
        }
    }

    let prefix = if plan.prefix_digest {
        json!({"latent": lat_d, "mirror_k": mk_d, "mirror_v": mv_d})
    } else {
        Value::Null
    };
    let rec = json!({
        "n_tokens": n,
        "device_seq_len": dev,
        "mirror_valid_rows": mvalid,
        "mirror_rows": mirror_rows,
        "capacity_rows": st.capacity_rows(),
        "next_token": first_argmax(&logits),
        "logits_digest": hx(digest(&lbytes)),
        "persist_region_digests": region_digests,
        "stream_digests": Value::Object(stream_digests),
        "prefix_digests": prefix,
    });
    Ok(Captured {
        rec,
        logits,
        n_tokens: n,
        device_seq_len: dev,
    })
}

/// Poison the backed latent tail `[n_tokens, capacity_rows)` of every cache.
fn poison_latent_tail(gpu: &Gpu, st: &RaonState) -> Result<usize, String> {
    let n = st.n_tokens();
    let cap = st.capacity_rows();
    if cap <= n {
        return Ok(0);
    }
    let rb = st.dims().latent_row_bytes();
    let host = vec![POISON_BYTE; (cap - n) * rb];
    for (l, t) in st.mla_cache.iter().enumerate() {
        let v = t.buf.byte_view(n * rb, host.len());
        hip_err(
            gpu.hip.memcpy_htod(&v, &host),
            &format!("poison latent[{l}] tail"),
        )?;
    }
    Ok(cap - n)
}

// ───────────────────────── one case ─────────────────────────

struct CaseOutcome {
    summary: Value,
    counts: Counts,
    last_owner: Option<owner::Snapshot>,
    failures: Vec<String>,
}

fn case_dir_name(ctx: usize) -> String {
    format!("ctx{ctx}")
}

fn run_case(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    o: &ArmOpts,
    ctx: usize,
    prompt: &[u32],
    dump: &Path,
) -> Result<CaseOutcome, String> {
    let dir = dump.join(case_dir_name(ctx));
    let mut files = CaseFiles::create(&dir)?;
    let mut failures: Vec<String> = Vec::new();

    // Same preimage for every case and both arms: cold slot, no expanded mirror
    // history (a mirror that already grew would skip the growth the other arm does).
    bundle.reset(gpu)?;
    bundle.state.release_expanded_mirror(gpu)?;
    bundle.set_prefill_mode(PrefillMode::Batched);
    bundle.set_decode_mode(DecodeMode::Graph);
    let (gc0, gr0) = bundle.decode_graph_stats();
    let owner_before = owner::snapshot(bundle);
    let t_prefill = Instant::now();
    bundle.prefill(gpu, prompt)?;
    hip_err(gpu.hip.device_synchronize(), "synchronize after prefill")?;
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    if bundle.state.n_tokens() != ctx {
        return Err(format!(
            "prefill advanced state to {} tokens, expected {ctx}",
            bundle.state.n_tokens()
        ));
    }
    let poisoned_rows = if o.poison {
        poison_latent_tail(gpu, &bundle.state)?
    } else {
        0
    };
    let regions = persist_regions(&bundle.state)?;

    let raw_step = |s: usize| s == 0 || s == o.steps || o.raw_all || o.raw_steps.contains(&s);
    let plan_for = |s: usize| CapturePlan {
        prefix_digest: s == 0 || s == o.steps || (o.digest_every > 0 && s % o.digest_every == 0),
        raw_persist: raw_step(s),
        raw_prefix: raw_step(s) && o.raw_prefix,
    };

    let mut steps_json: Vec<Value> = Vec::with_capacity(o.steps + 1);
    let mut counts = Counts::default();
    let mut positions_ok = true;
    let mut decode_inputs: Vec<u32> = Vec::with_capacity(o.steps);
    let mut error: Option<String> = None;

    let c0 = capture_step(gpu, bundle, &regions, 0, &plan_for(0), &mut files)?;
    let pre_capacity = c0.rec["capacity_rows"].clone();
    let pre_mirror = c0.rec["mirror_valid_rows"].clone();
    if c0.n_tokens != ctx || c0.device_seq_len as usize != ctx {
        positions_ok = false;
        failures.push(format!(
            "preimage positions: host {} / device {} != {ctx}",
            c0.n_tokens, c0.device_seq_len
        ));
    }
    let mut rec0 = c0.rec;
    rec0["step"] = json!(0);
    rec0["input_token"] = Value::Null;
    steps_json.push(rec0);
    let mut logits = c0.logits;

    let mut owner_prev = owner::snapshot(bundle);
    let mut last_owner = owner_prev.clone();
    let t_decode = Instant::now();
    for s in 1..=o.steps {
        if o.mirror_release_at == Some(s - 1) {
            hip_err(
                gpu.hip.device_synchronize(),
                "synchronize before mirror release",
            )?;
            bundle.state.release_expanded_mirror(gpu)?;
        }
        let tok = first_argmax(&logits);
        decode_inputs.push(tok);
        let (gc_b, gr_b) = bundle.decode_graph_stats();
        if let Err(e) = bundle.decode_step(gpu, tok) {
            error = Some(format!("decode step {s} (input id {tok}): {e}"));
            break;
        }
        let owner_now = owner::snapshot(bundle);
        let (gc_a, gr_a) = bundle.decode_graph_stats();
        let kind = classify(&owner_prev, &owner_now);
        let graph_activity = gc_a != gc_b || gr_a != gr_b;
        counts.steps += 1;
        match &kind {
            Kind::OwnerOff => counts.owner_off += 1,
            Kind::Railgun => {
                counts.railgun += 1;
                if graph_activity {
                    counts.railgun_steps_with_graph_activity += 1;
                }
            }
            Kind::DeclinedGraph(reason) => {
                counts.declined_graph += 1;
                *counts.reasons.entry(reason.clone()).or_default() += 1;
            }
            Kind::OwnerIdle => counts.owner_idle += 1,
            Kind::Anomaly(_) => counts.anomaly += 1,
        }
        let cap = capture_step(gpu, bundle, &regions, s, &plan_for(s), &mut files)?;
        let want = ctx + s;
        if cap.n_tokens != want || cap.device_seq_len as i64 != want as i64 {
            positions_ok = false;
            failures.push(format!(
                "step {s}: host n_tokens {} / device seq_len {} != {want}",
                cap.n_tokens, cap.device_seq_len
            ));
        }
        let mut rec = cap.rec;
        rec["step"] = json!(s);
        rec["input_token"] = json!(tok);
        rec["exec"] = json!(kind.name());
        rec["refusal"] = match &kind {
            Kind::DeclinedGraph(r) | Kind::Anomaly(r) => json!(r),
            _ => Value::Null,
        };
        rec["owner"] = match &owner_now {
            Some(a) => json!({
                "attempts": a.attempts, "completed": a.completed, "declined": a.declined,
                "builds": a.builds, "rebuilds": a.rebuilds,
            }),
            None => Value::Null,
        };
        rec["graph_captures_replays"] = json!([gc_a, gr_a]);
        steps_json.push(rec);
        logits = cap.logits;
        last_owner = owner_now.clone();
        owner_prev = owner_now;
    }
    hip_err(gpu.hip.device_synchronize(), "synchronize after decode")?;
    let decode_secs = t_decode.elapsed().as_secs_f64();
    let (gc1, gr1) = bundle.decode_graph_stats();
    let raw_index = files.finish()?;

    if let Some(e) = &error {
        failures.push(e.clone());
    }
    if !positions_ok {
        failures.push("host/device positions diverged from p+i".into());
    }
    let graph_served = gr1 - gr0 > 0;
    match o.arm {
        Arm::Graph => {
            if counts.owner_off != counts.steps {
                failures.push(format!(
                    "graph arm: {} of {} steps carried owner counters (owner must be off)",
                    counts.steps - counts.owner_off,
                    counts.steps
                ));
            }
            if error.is_none() && o.steps >= 3 && !graph_served {
                failures.push("graph arm: no step was served by a graph replay".into());
            }
        }
        Arm::Railgun => {
            if error.is_none() && counts.railgun == 0 {
                failures.push(format!(
                    "requested railgun, executed 0 of {} steps",
                    counts.steps
                ));
            }
            if counts.owner_off > 0 {
                failures.push(format!(
                    "requested railgun but the owner exposed no counters on {} steps (owner off)",
                    counts.owner_off
                ));
            }
            if counts.owner_idle > 0 {
                failures.push(format!(
                    "{} steps were neither completed nor declined by the owner (hidden fallback)",
                    counts.owner_idle
                ));
            }
            if counts.anomaly > 0 {
                failures.push(format!(
                    "{} steps with anomalous owner counters",
                    counts.anomaly
                ));
            }
            if counts.declined_graph > o.max_declined {
                failures.push(format!(
                    "{} steps declined to the graph (> --max-declined {})",
                    counts.declined_graph, o.max_declined
                ));
            }
            if last_owner.as_ref().is_some_and(|a| a.quarantined) {
                failures.push("owner quarantined".into());
            }
        }
    }

    let pass = failures.is_empty();
    let rb = bundle.state.dims().latent_row_bytes();
    let summary = json!({
        "ctx": ctx,
        "steps": o.steps,
        "steps_recorded": steps_json.len() - 1,
        "dir": case_dir_name(ctx),
        "pass": pass,
        "failures": failures,
        "error": error,
        "prompt_tokens": prompt.len(),
        "prompt_ids_md5": ids_md5(prompt),
        "prompt_ids_head": &prompt[..prompt.len().min(16)],
        "decode_input_tokens": decode_inputs,
        "requested_backend": o.arm.name(),
        "executing": counts.json(),
        "owner_before_prefill": owner_before.as_ref().map(|s| s.json()),
        "owner_final": last_owner.as_ref().map(|s| s.json()),
        "graph_captures_replays": [gc1 - gc0, gr1 - gr0],
        "preimage": {
            "capacity_rows": pre_capacity,
            "mirror_valid_rows": pre_mirror,
            "poisoned_latent_tail_rows": poisoned_rows,
        },
        "positions_ok": positions_ok,
        "prefill_secs_informational": prefill_secs,
        "decode_secs_informational": decode_secs,
        "digest_every": o.digest_every,
        "raw_prefix": o.raw_prefix,
        "mirror_release_at": o.mirror_release_at,
        "logit_bytes_each": logits.len() * 4,
    });
    // case.json: the summary plus everything compare needs.
    let mut full = summary.clone();
    full["schema"] = json!(SCHEMA);
    full["latent_row_bytes"] = json!(rb);
    full["persist_regions"] = json!(regions
        .iter()
        .map(|(n, off, len)| json!([n, off, len]))
        .collect::<Vec<_>>());
    full["persist_total_bytes"] = json!(bundle.state.persist_part().1);
    full["step_records"] = Value::Array(steps_json);
    full["raw_index"] = Value::Array(raw_index);
    let text = serde_json::to_string_pretty(&full).map_err(|e| e.to_string())?;
    io_err(
        std::fs::write(dir.join("case.json"), text),
        &dir.join("case.json").display().to_string(),
    )?;
    Ok(CaseOutcome {
        summary,
        counts,
        last_owner,
        failures,
    })
}

// ───────────────────────── one arm ─────────────────────────

fn load_prompt(o: &ArmOpts) -> Result<(String, Vec<u32>), String> {
    let text = std::fs::read_to_string(&o.prompt_file)
        .map_err(|e| format!("read prompt {}: {e}", o.prompt_file))?;
    let hfq = HfqFile::open(Path::new(&o.model)).map_err(|e| format!("open {}: {e}", o.model))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer from HFQ metadata: {e:?}"))?;
    let base = tokenizer.encode(&text);
    if base.is_empty() {
        return Err(format!("{} tokenized to zero ids", o.prompt_file));
    }
    Ok((md5_hex(text.as_bytes()), base))
}

fn run_arm(o: &ArmOpts) -> Result<bool, String> {
    // The product route reads its switches ONCE in `RaonBundle::new`: the process
    // environment must be final before anything loads. Nothing above reads config.
    match o.arm {
        Arm::Graph => std::env::set_var(owner::ENV_ON, "0"),
        Arm::Railgun => {
            std::env::set_var(owner::ENV_ON, "1");
            std::env::set_var(owner::ENV_CORPUS, o.corpus.as_deref().unwrap_or_default());
        }
    }
    if let Some(g4) = &o.g4 {
        std::env::set_var(owner::ENV_G4, g4);
    }
    if std::env::var("HIPFIRE_GRAPH").ok().as_deref() == Some("0") {
        return Err(
            "HIPFIRE_GRAPH=0 forces the eager route, which neither arm of this oracle is".into(),
        );
    }
    if let Some(c) = &o.corpus {
        if !Path::new(c).is_dir() {
            return Err(format!("--corpus {c}: not a directory"));
        }
    }

    let started = Instant::now();
    let model_id = model_identity(&o.model, o.model_sha256.as_deref())?;
    let (prompt_md5, base) = load_prompt(o)?;
    let max_ctx = *o.ctxs.iter().max().unwrap();
    let max_seq = (max_ctx + o.steps + 64).max(512);
    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let src = ModelSource::from_path(&o.model)?;
    let mut bundle = load_bundle(&mut gpu, src, &o.model, max_seq)?;
    let owner_at_load = owner::snapshot(&bundle);

    let result = run_cases(&mut gpu, &mut bundle, o, &base);
    bundle.free_all(&mut gpu);
    let (cases, counts, last_owner, weights_full, case_failures) = result?;
    let digest_of = |v: &Option<Value>| -> Option<String> {
        v.as_ref()
            .map(|v| v["digest128"].as_str().unwrap_or_default().to_string())
    };
    let weights = (digest_of(&weights_full.0), digest_of(&weights_full.1));
    let pointers = (
        weights_full.0.as_ref().map(|v| v["pointer_tables"].clone()),
        weights_full.1.as_ref().map(|v| v["pointer_tables"].clone()),
    );

    let mut failures: Vec<String> = Vec::new();
    for f in case_failures {
        failures.push(f);
    }
    if let (Some(b), Some(a)) = (&weights.0, &weights.1) {
        if b != a {
            failures.push(format!("weights mutated: digest {b} before, {a} after"));
        }
    }
    for (when, p) in [("before", &pointers.0), ("after", &pointers.1)] {
        if let Some(p) = p {
            if p["consistent"] != json!(true) {
                failures.push(format!(
                    "pointer tables {when} the run do not hold the addresses of the allocations they name: {}",
                    p["mismatches"]
                ));
            }
        }
    }
    if let (Some(b), Some(a)) = (&pointers.0, &pointers.1) {
        if b["digest128"] != a["digest128"] {
            failures.push("pointer tables changed during the run".into());
        }
    }
    if model_id["sha256"].is_null() {
        failures.push("model sha256 not pinned (--model-sha256 skip)".into());
    }
    let pass = failures.is_empty();
    let weights_unchanged = match (&weights.0, &weights.1) {
        (Some(b), Some(a)) => json!(b == a),
        _ => Value::Null,
    };
    let mut doc = json!({
        "schema": SCHEMA,
        "oracle": "railgun-state",
        "arm": o.arm.name(),
        "requested_backend": o.arm.name(),
        "executing_backend": counts.executing_backend(),
        "pass": pass,
        "failures": failures,
        "model": model_id,
        "prompt": {
            "file": o.prompt_file,
            "file_text_md5": prompt_md5,
            "base_token_ids": base,
            "base_ids_md5_u32le": ids_md5(&base),
            "construction": "base ids repeated cyclically to ctx (as bench/probe)",
        },
        "ctxs": o.ctxs,
        "steps": o.steps,
        "options": {
            "digest_every": o.digest_every, "raw_all": o.raw_all, "raw_steps": o.raw_steps,
            "raw_prefix": o.raw_prefix, "poison": o.poison, "max_declined": o.max_declined,
            "mirror_release_at": o.mirror_release_at, "corpus": o.corpus, "g4": o.g4,
            "weights_digest": o.weights_digest, "max_seq": max_seq,
        },
        "argv": o.argv,
        "executing": counts.json(),
        "owner_at_load": owner_at_load.as_ref().map(|s| s.json()),
        "owner_final": last_owner.as_ref().map(|s| s.json()),
        "weights_digest_scope": o.weights_digest.then_some(WEIGHTS_DIGEST_SCOPE),
        "weights_digest_before": weights.0,
        "weights_digest_after": weights.1,
        "weights_unchanged": weights_unchanged,
        "pointer_tables_before": pointers.0,
        "pointer_tables_after": pointers.1,
        "cases": cases,
        "secs_total_informational": started.elapsed().as_secs_f64(),
        "controls": controls(),
    });
    // Resolve dump_dir after the case files exist (canonicalize needs the directory).
    doc["dump_dir"] = json!(std::fs::canonicalize(&o.dump)
        .map_err(|e| format!("{}: {e}", o.dump.display()))?
        .display()
        .to_string());
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    io_err(
        std::fs::write(o.dump.join("arm.json"), &text),
        &o.dump.join("arm.json").display().to_string(),
    )?;
    if let Some(parent) = o.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        io_err(
            std::fs::create_dir_all(parent),
            &parent.display().to_string(),
        )?;
    }
    io_err(std::fs::write(&o.out, &text), &o.out.display().to_string())?;

    println!(
        "raon_oracle railgun-state {}: {}",
        o.arm.name(),
        if pass { "OK" } else { "FAILED" }
    );
    println!(
        "  requested={} executing={} steps={} railgun={} declined_graph={} owner_idle={} owner_off={} anomaly={}",
        o.arm.name(),
        counts.executing_backend(),
        counts.steps,
        counts.railgun,
        counts.declined_graph,
        counts.owner_idle,
        counts.owner_off,
        counts.anomaly
    );
    for (reason, n) in &counts.reasons {
        println!("  declined x{n}: {reason}");
    }
    for f in doc["failures"].as_array().into_iter().flatten() {
        println!("  FAIL {}", f.as_str().unwrap_or("?"));
    }
    println!("  dump {}", o.dump.display());
    Ok(pass)
}

type CasesResult = (
    Vec<Value>,
    Counts,
    Option<owner::Snapshot>,
    (Option<Value>, Option<Value>),
    Vec<String>,
);

fn run_cases(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    o: &ArmOpts,
    base: &[u32],
) -> Result<CasesResult, String> {
    let vocab = bundle.config.vocab;
    if let Some(&bad) = base.iter().find(|&&t| t as usize >= vocab) {
        return Err(format!("prompt id {bad} >= vocab {vocab}"));
    }
    io_err(
        std::fs::create_dir_all(&o.dump),
        &o.dump.display().to_string(),
    )?;
    let before = if o.weights_digest {
        Some(weights_digest(gpu, &bundle.weights)?)
    } else {
        None
    };
    let mut cases = Vec::new();
    let mut counts = Counts::default();
    let mut last_owner = None;
    let mut failures = Vec::new();
    for &ctx in &o.ctxs {
        let prompt: Vec<u32> = (0..ctx).map(|i| base[i % base.len()]).collect();
        let c = run_case(gpu, bundle, o, ctx, &prompt, &o.dump)?;
        counts.add(&c.counts);
        last_owner = c.last_owner.clone().or(last_owner);
        for f in &c.failures {
            failures.push(format!("ctx {ctx}: {f}"));
        }
        cases.push(c.summary);
    }
    let after = if o.weights_digest {
        Some(weights_digest(gpu, &bundle.weights)?)
    } else {
        None
    };
    Ok((cases, counts, last_owner, (before, after), failures))
}

// ───────────────────────── compare ─────────────────────────

#[derive(Clone)]
struct Flip {
    step: usize,
    tensor: String,
    offset: usize,
}

fn parse_flip(s: &str) -> Result<Flip, String> {
    let (step, rest) = s.split_once(':').ok_or("--flip-byte STEP:TENSOR:OFFSET")?;
    let (tensor, offset) = rest
        .rsplit_once(':')
        .ok_or("--flip-byte STEP:TENSOR:OFFSET")?;
    Ok(Flip {
        step: step
            .trim()
            .parse()
            .map_err(|e| format!("--flip-byte step {step:?}: {e}"))?,
        tensor: tensor.to_string(),
        offset: offset
            .trim()
            .parse()
            .map_err(|e| format!("--flip-byte offset {offset:?}: {e}"))?,
    })
}

#[derive(Clone)]
struct Div {
    ctx: usize,
    step: i64,
    tensor: String,
    offset: Option<usize>,
    region: Option<String>,
    detail: String,
}

impl Div {
    fn json(&self) -> Value {
        json!({
            "ctx": self.ctx, "step": self.step, "tensor": self.tensor,
            "offset": self.offset, "region": self.region, "detail": self.detail,
        })
    }

    fn line(&self) -> String {
        format!(
            "ctx {} step {} tensor {}{}{}: {}",
            self.ctx,
            self.step,
            self.tensor,
            self.offset
                .map(|o| format!(" offset {o}"))
                .unwrap_or_default(),
            self.region
                .as_ref()
                .map(|r| format!(" ({r})"))
                .unwrap_or_default(),
            self.detail
        )
    }
}

/// Resolve `<arm json or dump dir>` to `(arm doc, dump dir)`.
fn load_arm(p: &str, is_dir: bool) -> Result<(Value, PathBuf), String> {
    let path = Path::new(p);
    let json_path = if is_dir || path.is_dir() {
        path.join("arm.json")
    } else {
        path.to_path_buf()
    };
    let text =
        std::fs::read_to_string(&json_path).map_err(|e| format!("{}: {e}", json_path.display()))?;
    let doc: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", json_path.display()))?;
    if doc["schema"] != json!(SCHEMA) {
        return Err(format!(
            "{}: schema {} != {SCHEMA}",
            json_path.display(),
            doc["schema"]
        ));
    }
    let candidates = [
        doc["dump_dir"].as_str().map(PathBuf::from),
        Some(PathBuf::from(format!("{}.dump", json_path.display()))),
        json_path.parent().map(Path::to_path_buf),
    ];
    for c in candidates.into_iter().flatten() {
        if c.join(case_dir_name(doc["ctxs"][0].as_u64().unwrap_or(0) as usize))
            .join("case.json")
            .exists()
        {
            return Ok((doc, c));
        }
    }
    Err(format!(
        "{}: cannot find the dump directory (dump_dir {}, nor <json>.dump)",
        json_path.display(),
        doc["dump_dir"]
    ))
}

fn read_exact_at(f: &mut File, off: u64, len: usize, what: &str) -> Result<Vec<u8>, String> {
    let mut v = vec![0u8; len];
    io_err(f.seek(SeekFrom::Start(off)), what)?;
    io_err(f.read_exact(&mut v), what)?;
    Ok(v)
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a == b {
        return None;
    }
    Some(
        a.iter()
            .zip(b)
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len())),
    )
}

fn diff_count(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

struct CaseReport {
    steps_compared: usize,
    logit_vectors: usize,
    raw_records: usize,
    bytes_compared: u64,
    divs: Vec<Div>,
    info_diffs: Vec<String>,
}

struct FlipHit {
    ctx: usize,
    step: usize,
    tensor: String,
    offset: usize,
}

fn region_of(regions: &Value, off: usize) -> Option<String> {
    for r in regions.as_array()? {
        let (o, l) = (r[1].as_u64()? as usize, r[2].as_u64()? as usize);
        if off >= o && off < o + l {
            return Some(format!("{} +{}", r[0].as_str()?, off - o));
        }
    }
    None
}

/// Digest-record keys compared exactly per step.
const META_KEYS: [&str; 5] = [
    "input_token",
    "next_token",
    "n_tokens",
    "device_seq_len",
    "mirror_valid_rows",
];

#[allow(clippy::too_many_arguments)]
fn compare_case(
    ctx: usize,
    da: &Path,
    ca: &Value,
    db: &Path,
    cb: &Value,
    flip: Option<&Flip>,
    hits: &mut Vec<FlipHit>,
) -> Result<CaseReport, String> {
    let mut rep = CaseReport {
        steps_compared: 0,
        logit_vectors: 0,
        raw_records: 0,
        bytes_compared: 0,
        divs: Vec::new(),
        info_diffs: Vec::new(),
    };
    let div = |rep: &mut CaseReport, step: i64, tensor: &str, detail: String| {
        rep.divs.push(Div {
            ctx,
            step,
            tensor: tensor.to_string(),
            offset: None,
            region: None,
            detail,
        })
    };
    let (sa, sb) = (
        ca["step_records"].as_array().cloned().unwrap_or_default(),
        cb["step_records"].as_array().cloned().unwrap_or_default(),
    );
    if sa.len() != sb.len() {
        div(
            &mut rep,
            -1,
            "step_records",
            format!("{} vs {} records", sa.len(), sb.len()),
        );
    }
    if ca["persist_regions"] != cb["persist_regions"] {
        div(
            &mut rep,
            -1,
            "persist_regions",
            "slab layouts differ".into(),
        );
    }
    let (each_a, each_b) = (
        ca["logit_bytes_each"].as_u64().unwrap_or(0) as usize,
        cb["logit_bytes_each"].as_u64().unwrap_or(0) as usize,
    );
    if each_a == 0 || each_a != each_b {
        div(
            &mut rep,
            -1,
            "logits",
            format!("logit vector bytes {each_a} vs {each_b}"),
        );
    }
    let mut la = File::open(da.join(case_dir_name(ctx)).join("logits.bin"))
        .map_err(|e| format!("ctx {ctx} A logits.bin: {e}"))?;
    let mut lb = File::open(db.join(case_dir_name(ctx)).join("logits.bin"))
        .map_err(|e| format!("ctx {ctx} B logits.bin: {e}"))?;

    // Raw record maps: (step, name) -> (off, len, recorded digest).
    let index = |c: &Value| -> Vec<(usize, String, u64, usize, String)> {
        c["raw_index"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|e| {
                        (
                            e["step"].as_u64().unwrap_or(0) as usize,
                            e["name"].as_str().unwrap_or("").to_string(),
                            e["off"].as_u64().unwrap_or(0),
                            e["len"].as_u64().unwrap_or(0) as usize,
                            e["digest128"].as_str().unwrap_or("").to_string(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let (ia, ib) = (index(ca), index(cb));
    let mut ra = File::open(da.join(case_dir_name(ctx)).join("raw.bin"))
        .map_err(|e| format!("ctx {ctx} A raw.bin: {e}"))?;
    let mut rb = File::open(db.join(case_dir_name(ctx)).join("raw.bin"))
        .map_err(|e| format!("ctx {ctx} B raw.bin: {e}"))?;
    let mut b_seen: BTreeSet<usize> = BTreeSet::new();

    let n_steps = sa.len().min(sb.len());
    for step in 0..n_steps {
        let (xa, xb) = (&sa[step], &sb[step]);
        let st = step as i64;
        rep.steps_compared += 1;

        // Positions: the invariant of each arm on its own, then equality.
        for (arm, x) in [("A", xa), ("B", xb)] {
            let want = (ctx + step) as u64;
            if x["n_tokens"].as_u64() != Some(want) || x["device_seq_len"].as_u64() != Some(want) {
                div(
                    &mut rep,
                    st,
                    "positions",
                    format!(
                        "arm {arm}: host n_tokens {} / device seq_len {} != p+i = {want}",
                        x["n_tokens"], x["device_seq_len"]
                    ),
                );
            }
        }
        for k in META_KEYS {
            if xa[k] != xb[k] {
                div(&mut rep, st, k, format!("{} vs {}", xa[k], xb[k]));
            }
        }
        for k in ["mirror_rows", "capacity_rows"] {
            if xa[k] != xb[k] {
                rep.info_diffs
                    .push(format!("step {step} {k}: {} vs {}", xa[k], xb[k]));
            }
        }

        // Logits, byte for byte (the stored digests are an integrity check of the file).
        if each_a > 0 && each_a == each_b {
            let off = (step * each_a) as u64;
            let ba = read_exact_at(&mut la, off, each_a, "A logits.bin")?;
            let mut bb = read_exact_at(&mut lb, off, each_a, "B logits.bin")?;
            for (arm, bytes, x) in [("A", &ba, xa), ("B", &bb, xb)] {
                if x["logits_digest"].as_str() != Some(hx(digest(bytes)).as_str()) {
                    div(
                        &mut rep,
                        st,
                        "dump_integrity:logits",
                        format!("arm {arm} logits.bin does not match its recorded digest"),
                    );
                }
            }
            if let Some(f) = flip {
                if f.step == step && f.tensor == "logits" {
                    if f.offset >= bb.len() {
                        return Err(format!(
                            "flip offset {} beyond logits ({} B)",
                            f.offset,
                            bb.len()
                        ));
                    }
                    bb[f.offset] ^= 0x01;
                    hits.push(FlipHit {
                        ctx,
                        step,
                        tensor: "logits".into(),
                        offset: f.offset,
                    });
                }
            }
            rep.logit_vectors += 1;
            rep.bytes_compared += each_a as u64;
            if let Some(at) = first_diff(&ba, &bb) {
                let elem = at / 4;
                rep.divs.push(Div {
                    ctx,
                    step: st,
                    tensor: "logits".into(),
                    offset: Some(at),
                    region: Some(format!("f32 element {elem}")),
                    detail: format!("{} of {} bytes differ", diff_count(&ba, &bb), ba.len()),
                });
            }
        }

        // Digest records.
        let (pa, pb) = (
            xa["persist_region_digests"]
                .as_array()
                .cloned()
                .unwrap_or_default(),
            xb["persist_region_digests"]
                .as_array()
                .cloned()
                .unwrap_or_default(),
        );
        if pa.len() != pb.len() {
            div(
                &mut rep,
                st,
                "persist",
                format!("{} vs {} region digests", pa.len(), pb.len()),
            );
        }
        for (i, (p, q)) in pa.iter().zip(&pb).enumerate() {
            if p != q {
                let name = ca["persist_regions"][i][0]
                    .as_str()
                    .unwrap_or("?")
                    .to_string();
                rep.divs.push(Div {
                    ctx,
                    step: st,
                    tensor: "persist".into(),
                    offset: ca["persist_regions"][i][1].as_u64().map(|o| o as usize),
                    region: Some(name),
                    detail: "region digest differs (offset = region start)".into(),
                });
            }
        }
        if xa["stream_digests"] != xb["stream_digests"] {
            div(
                &mut rep,
                st,
                "mhc_streams",
                format!("{} vs {}", xa["stream_digests"], xb["stream_digests"]),
            );
        }
        if !xa["prefix_digests"].is_null() && !xb["prefix_digests"].is_null() {
            for k in ["latent", "mirror_k", "mirror_v"] {
                let (pa, pb) = (
                    xa["prefix_digests"][k]
                        .as_array()
                        .cloned()
                        .unwrap_or_default(),
                    xb["prefix_digests"][k]
                        .as_array()
                        .cloned()
                        .unwrap_or_default(),
                );
                if pa.len() != pb.len() {
                    div(
                        &mut rep,
                        st,
                        k,
                        format!("{} vs {} layer digests", pa.len(), pb.len()),
                    );
                }
                for (l, (p, q)) in pa.iter().zip(&pb).enumerate() {
                    if p != q {
                        div(
                            &mut rep,
                            st,
                            &format!("{k}[{l}]"),
                            "live prefix digest differs".into(),
                        );
                    }
                }
            }
        }

        // Raw records of this step.
        for (s, name, off, len, dig) in ia.iter().filter(|e| e.0 == step) {
            let Some(j) = ib.iter().position(|e| e.0 == *s && e.1 == *name) else {
                div(&mut rep, st, name, "raw record present in A only".into());
                continue;
            };
            b_seen.insert(j);
            let (_, _, off_b, len_b, dig_b) = &ib[j];
            let ba = read_exact_at(&mut ra, *off, *len, "A raw.bin")?;
            let mut bb = read_exact_at(&mut rb, *off_b, *len_b, "B raw.bin")?;
            for (arm, bytes, rec) in [("A", &ba, dig), ("B", &bb, dig_b)] {
                if hx(digest(bytes)) != *rec {
                    div(
                        &mut rep,
                        st,
                        &format!("dump_integrity:{name}"),
                        format!("arm {arm} raw.bin record does not match its recorded digest"),
                    );
                }
            }
            if let Some(f) = flip {
                if f.step == *s && f.tensor == *name {
                    if f.offset >= bb.len() {
                        return Err(format!(
                            "flip offset {} beyond {name} ({} B)",
                            f.offset,
                            bb.len()
                        ));
                    }
                    bb[f.offset] ^= 0x01;
                    hits.push(FlipHit {
                        ctx,
                        step,
                        tensor: name.clone(),
                        offset: f.offset,
                    });
                }
            }
            rep.raw_records += 1;
            rep.bytes_compared += ba.len() as u64;
            if let Some(at) = first_diff(&ba, &bb) {
                rep.divs.push(Div {
                    ctx,
                    step: st,
                    tensor: name.clone(),
                    offset: Some(at),
                    region: if name == "persist" {
                        region_of(&ca["persist_regions"], at)
                    } else {
                        None
                    },
                    detail: format!(
                        "{} of {} bytes differ{}",
                        diff_count(&ba, &bb),
                        ba.len(),
                        if ba.len() != bb.len() {
                            format!(" (lengths {} vs {})", ba.len(), bb.len())
                        } else {
                            String::new()
                        }
                    ),
                });
            }
        }
    }
    for (j, e) in ib.iter().enumerate() {
        if !b_seen.contains(&j) && !ia.iter().any(|x| x.0 == e.0 && x.1 == e.1) {
            div(
                &mut rep,
                e.0 as i64,
                &e.1,
                "raw record present in B only".into(),
            );
        }
    }
    Ok(rep)
}

struct ArmsCompare {
    json: Value,
    pass: bool,
    hits: Vec<FlipHit>,
    divs: Vec<Div>,
    lines: Vec<String>,
}

fn compare_arms(
    pa: &str,
    pb: &str,
    is_dir: bool,
    flip: Option<&Flip>,
) -> Result<ArmsCompare, String> {
    let (a, da) = load_arm(pa, is_dir)?;
    let (b, db) = load_arm(pb, is_dir)?;
    let mut pre: Vec<String> = Vec::new();

    // Each arm's own verdict, and requested vs executing backend.
    let mut arms = Vec::new();
    for (tag, doc) in [("A", &a), ("B", &b)] {
        let requested = doc["requested_backend"].as_str().unwrap_or("?");
        let executed = doc["executing"]["steps_railgun"].as_u64().unwrap_or(0);
        if doc["pass"] != json!(true) {
            pre.push(format!(
                "arm {tag} ({}) failed its own verdict: {}",
                doc["arm"], doc["failures"]
            ));
        }
        if requested == "railgun" && executed == 0 {
            pre.push(format!(
                "arm {tag}: requested railgun but executed {executed} steps"
            ));
        }
        arms.push(json!({
            "tag": tag, "arm": doc["arm"], "requested_backend": requested,
            "executing_backend": doc["executing_backend"],
            "steps_railgun": executed,
            "steps_declined_graph": doc["executing"]["steps_declined_graph"],
            "declined_reasons": doc["executing"]["declined_reasons"],
            "owner_final": doc["owner_final"], "model_sha256": doc["model"]["sha256"],
            "binary_md5": doc["controls"]["binary_md5"],
        }));
    }
    if a["arm"] == b["arm"] {
        pre.push(format!(
            "both dumps are arm {}: expected graph vs railgun",
            a["arm"]
        ));
    }
    // Same preimage.
    for (k, what) in [
        ("model", "model sha256"),
        ("prompt", "prompt ids"),
        ("ctxs", "ctx list"),
        ("steps", "steps"),
    ] {
        let (x, y) = match k {
            "model" => (a["model"]["sha256"].clone(), b["model"]["sha256"].clone()),
            "prompt" => (
                a["prompt"]["base_ids_md5_u32le"].clone(),
                b["prompt"]["base_ids_md5_u32le"].clone(),
            ),
            _ => (a[k].clone(), b[k].clone()),
        };
        if x != y || x.is_null() {
            pre.push(format!("{what} differ or are not pinned: {x} vs {y}"));
        }
    }
    if a["weights_digest_scope"] != b["weights_digest_scope"] {
        pre.push(format!(
            "weights digest scope differs: {} vs {} (a dump taken before the pointer tables were excluded, or without a weights digest, is not comparable)",
            a["weights_digest_scope"], b["weights_digest_scope"]
        ));
    }
    for (tag, doc) in [("A", &a), ("B", &b)] {
        for when in ["pointer_tables_before", "pointer_tables_after"] {
            let p = &doc[when];
            if !p.is_null() && p["consistent"] != json!(true) {
                pre.push(format!(
                    "arm {tag}: {when}: the loader pointer tables do not hold the addresses of the allocations they name"
                ));
            }
        }
        if doc["pointer_tables_before"]["digest128"] != doc["pointer_tables_after"]["digest128"] {
            pre.push(format!(
                "arm {tag}: the loader pointer tables changed during the run"
            ));
        }
    }
    // The pointer tables' raw bytes are process addresses: never compared across arms.
    if a["weights_digest_before"] != b["weights_digest_before"]
        || a["weights_digest_after"] != b["weights_digest_after"]
    {
        pre.push(format!(
            "weights digests differ across arms: {} / {} vs {} / {}",
            a["weights_digest_before"],
            a["weights_digest_after"],
            b["weights_digest_before"],
            b["weights_digest_after"]
        ));
    }
    for (tag, doc) in [("A", &a), ("B", &b)] {
        if doc["weights_unchanged"] == json!(false) {
            pre.push(format!("arm {tag}: weights mutated during the run"));
        }
    }

    let ctxs: Vec<usize> = a["ctxs"]
        .as_array()
        .map(|v| {
            v.iter()
                .filter_map(|x| x.as_u64().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_default();
    let mut cases_json = Vec::new();
    let mut all_divs: Vec<Div> = Vec::new();
    let mut hits: Vec<FlipHit> = Vec::new();
    let mut lines = Vec::new();
    if a["ctxs"] == b["ctxs"] {
        for &ctx in &ctxs {
            let read_case = |d: &Path| -> Result<Value, String> {
                let p = d.join(case_dir_name(ctx)).join("case.json");
                let t = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                serde_json::from_str(&t).map_err(|e| format!("{}: {e}", p.display()))
            };
            let (ca, cb) = (read_case(&da)?, read_case(&db)?);
            let rep = compare_case(ctx, &da, &ca, &db, &cb, flip, &mut hits)?;
            cases_json.push(json!({
                "ctx": ctx,
                "steps_compared": rep.steps_compared,
                "logit_vectors_compared": rep.logit_vectors,
                "raw_records_compared": rep.raw_records,
                "bytes_compared": rep.bytes_compared,
                "divergences": rep.divs.len(),
                "first_divergence": rep.divs.first().map(Div::json),
                "info_diffs": rep.info_diffs.iter().take(16).collect::<Vec<_>>(),
                "info_diff_count": rep.info_diffs.len(),
                "pass": rep.divs.is_empty(),
            }));
            all_divs.extend(rep.divs);
        }
    }
    for p in &pre {
        lines.push(format!("PRECONDITION {p}"));
    }
    for d in all_divs.iter().take(8) {
        lines.push(format!("DIVERGENCE {}", d.line()));
    }
    let pass = pre.is_empty() && all_divs.is_empty();
    let json = json!({
        "arms": arms,
        "preconditions_failed": pre,
        "cases": cases_json,
        "divergence_count": all_divs.len(),
        "first_divergence": all_divs.first().map(Div::json),
        "divergences": all_divs.iter().take(64).map(Div::json).collect::<Vec<_>>(),
        "pass": pass,
    });
    Ok(ArmsCompare {
        json,
        pass,
        hits,
        divs: all_divs,
        lines,
    })
}

fn run_compare(a: &Args) -> Result<bool, String> {
    let (is_dir, (pa, pb)) = match (&a.compare, &a.compare_dir) {
        (Some(p), None) => (false, p.clone()),
        (None, Some(p)) => (true, p.clone()),
        _ => return Err("use exactly one of --compare / --compare-dir".into()),
    };
    let flip = a.kv.get("flip-byte").map(|s| parse_flip(s)).transpose()?;
    let base = compare_arms(&pa, &pb, is_dir, None)?;
    let mut doc = json!({
        "schema": SCHEMA,
        "oracle": "railgun-state-compare",
        "a": pa, "b": pb,
        "compare": base.json.clone(),
    });
    let mut ok = base.pass;
    let mut lines = base.lines.clone();
    if let Some(f) = &flip {
        let flipped = compare_arms(&pa, &pb, is_dir, Some(f))?;
        let detected: Vec<bool> = flipped
            .hits
            .iter()
            .map(|h| {
                flipped.divs.iter().any(|d| {
                    d.ctx == h.ctx
                        && d.step == h.step as i64
                        && d.tensor == h.tensor
                        && d.offset == Some(h.offset)
                })
            })
            .collect();
        let applied = !flipped.hits.is_empty();
        let all_detected = applied && detected.iter().all(|&x| x);
        doc["negative_control"] = json!({
            "flip": {"step": f.step, "tensor": f.tensor, "offset": f.offset},
            "baseline_clean": base.pass,
            "applied": flipped.hits.iter().map(|h| json!({
                "ctx": h.ctx, "step": h.step, "tensor": h.tensor, "offset": h.offset})).collect::<Vec<_>>(),
            "detected": detected,
            "compare_with_flip": flipped.json,
        });
        // The control passes only when the comparison without the flip is clean and the
        // flipped byte (and nothing but it) is reported where it was flipped.
        ok = base.pass && all_detected;
        lines = vec![format!(
            "negative control {}:{}:{} -> {}",
            f.step,
            f.tensor,
            f.offset,
            if !base.pass {
                "baseline comparison is not clean (control meaningless)"
            } else if !applied {
                "flip target not found in any case (control did not run)"
            } else if all_detected {
                "flipped byte detected at its exact location"
            } else {
                "flipped byte NOT detected"
            }
        )];
        lines.extend(flipped.lines.iter().take(4).cloned());
    }
    doc["ok"] = json!(ok);
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    if let Some(out) = a.kv.get("out") {
        io_err(std::fs::write(out, &text), out)?;
    }
    println!(
        "raon_oracle railgun-state compare{}: {}",
        if flip.is_some() {
            " (negative control)"
        } else {
            ""
        },
        if ok { "OK" } else { "FAILED" }
    );
    for l in &lines {
        println!("  {l}");
    }
    if flip.is_none() {
        for c in base.json["cases"].as_array().into_iter().flatten() {
            println!(
                "  ctx {}: {} steps, {} logit vectors, {} raw records, {} bytes compared, {} divergences",
                c["ctx"], c["steps_compared"], c["logit_vectors_compared"],
                c["raw_records_compared"], c["bytes_compared"], c["divergences"]
            );
        }
    }
    Ok(ok)
}

// ───────────────────────── entry ─────────────────────────

/// `raon_oracle railgun-state <args>` (`argv` excludes the subcommand). Prints its
/// own verdict and writes its own JSON; `Ok(false)` is a verdict failure (exit 1).
pub fn main(argv: &[String]) -> Result<bool, Exit> {
    let args = parse_args(argv).map_err(Exit::Usage)?;
    if args.compare.is_some() || args.compare_dir.is_some() {
        return run_compare(&args).map_err(Exit::Fail);
    }
    if args.kv.contains_key("flip-byte") {
        return Err(Exit::Usage(
            "--flip-byte applies to --compare/--compare-dir".into(),
        ));
    }
    let opts = parse_arm_opts(&args, argv).map_err(Exit::Usage)?;
    run_arm(&opts).map_err(Exit::Fail)
}

// ───────────────────────── comparator tests (CPU only) ─────────────────────────

/// The comparator (`run_compare` / `compare_arms` / `compare_case`) on synthetic dumps.
/// No `Gpu`, no model: dumps are written with the arm run's own `CaseFiles` writer
/// (logits.bin / raw.bin / raw_index with digests), `digest`/`hx`, `Counts::json` and
/// `owner::Snapshot::json`, and the case.json / arm.json documents carry the same keys
/// `run_case` / `run_arm` emit. Dumps live under `<workspace>/target/railgun-state-tests`.
#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 4;
    const LROW: usize = 8;
    const MROW: usize = 6;
    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    /// `[name, offset, len]` tiling of a 64-byte persist slab (seq_len at 40).
    const REGIONS: [(&str, usize, usize); 6] = [
        ("rda[0]", 0, 16),
        ("conv_q[0]", 16, 8),
        ("conv_k[0]", 24, 8),
        ("conv_v[0]", 32, 8),
        ("seq_len", 40, 4),
        ("pad@44", 44, 20),
    ];
    const PERSIST: usize = 64;
    const SEQ_OFF: usize = 40;

    #[derive(Clone)]
    struct StepModel {
        input_token: Option<u32>,
        next_token: u32,
        n_tokens: u64,
        device_seq_len: i64,
        mirror_valid_rows: u64,
        logits: Vec<u8>,
        raw: Vec<(String, Vec<u8>)>,
    }

    #[derive(Clone)]
    struct CaseModel {
        ctx: usize,
        steps: Vec<StepModel>,
    }

    #[derive(Clone)]
    struct DumpModel {
        arm: Arm,
        sha: Option<String>,
        prompt_md5: String,
        ctxs: Vec<usize>,
        steps: usize,
        cases: Vec<CaseModel>,
        /// Steps the owner executed in total (railgun arm only).
        railgun_steps: u64,
        /// The arm's own verdict as written to arm.json.
        arm_pass: bool,
        weights: String,
        scope: Option<String>,
        /// Raw pointer-table digest (a process-specific value: differs between arms on purpose).
        ptr_digest: String,
        ptr_consistent: bool,
    }

    /// Deterministic bytes, identical for both arms.
    fn fill(tag: &str, ctx: usize, step: usize, len: usize) -> Vec<u8> {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in tag
            .bytes()
            .chain(ctx.to_le_bytes())
            .chain(step.to_le_bytes())
        {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        (0..len)
            .map(|i| {
                h = h
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407 + i as u64);
                (h >> 33) as u8
            })
            .collect()
    }

    fn build_model(arm: Arm, ctxs: &[usize], steps: usize) -> DumpModel {
        let total = (ctxs.len() * steps) as u64;
        let mut cases = Vec::new();
        for &ctx in ctxs {
            let mut recs: Vec<StepModel> = Vec::new();
            for s in 0..=steps {
                let logits: Vec<f32> = (0..VOCAB)
                    .map(|i| ((ctx * 7 + s * 3 + i * 5) % 11) as f32 * 0.25 - 1.0)
                    .collect();
                let lbytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
                let mut persist = fill("persist", ctx, s, PERSIST);
                persist[SEQ_OFF..SEQ_OFF + 4].copy_from_slice(&((ctx + s) as i32).to_le_bytes());
                let mut raw: Vec<(String, Vec<u8>)> = vec![
                    ("persist".into(), persist),
                    ("stream_a".into(), fill("stream_a", ctx, s, 32)),
                    ("stream_b".into(), fill("stream_b", ctx, s, 32)),
                    ("final_mean".into(), fill("final_mean", ctx, s, 16)),
                    ("final_norm".into(), fill("final_norm", ctx, s, 16)),
                    (
                        "latent[0]".into(),
                        fill("latent[0]", ctx, s, (ctx + s) * LROW),
                    ),
                    (
                        "mirror_k[0]".into(),
                        fill("mirror_k[0]", ctx, s, (ctx + s) * MROW),
                    ),
                    (
                        "mirror_v[0]".into(),
                        fill("mirror_v[0]", ctx, s, (ctx + s) * MROW),
                    ),
                ];
                if s > 0 {
                    raw.push(("latent_row[0]".into(), fill("latent_row[0]", ctx, s, LROW)));
                    raw.push((
                        "mirror_k_row[0]".into(),
                        fill("mirror_k_row[0]", ctx, s, MROW),
                    ));
                    raw.push((
                        "mirror_v_row[0]".into(),
                        fill("mirror_v_row[0]", ctx, s, MROW),
                    ));
                }
                let next_token = first_argmax(&logits);
                recs.push(StepModel {
                    input_token: recs.last().map(|p: &StepModel| p.next_token),
                    next_token,
                    n_tokens: (ctx + s) as u64,
                    device_seq_len: (ctx + s) as i64,
                    mirror_valid_rows: (ctx + s) as u64,
                    logits: lbytes,
                    raw,
                });
            }
            cases.push(CaseModel { ctx, steps: recs });
        }
        DumpModel {
            arm,
            sha: Some(SHA.to_string()),
            prompt_md5: "p".repeat(32),
            ctxs: ctxs.to_vec(),
            steps,
            cases,
            railgun_steps: if arm == Arm::Railgun { total } else { 0 },
            arm_pass: true,
            weights: "w".repeat(32),
            scope: Some(WEIGHTS_DIGEST_SCOPE.to_string()),
            ptr_digest: if arm == Arm::Railgun {
                "r".repeat(32)
            } else {
                "g".repeat(32)
            },
            ptr_consistent: true,
        }
    }

    fn raw_of<'a>(s: &'a StepModel, name: &str) -> &'a [u8] {
        &s.raw.iter().find(|(n, _)| n == name).unwrap().1
    }

    fn step_mut<'a>(m: &'a mut DumpModel, ctx: usize, step: usize) -> &'a mut StepModel {
        &mut m.cases.iter_mut().find(|c| c.ctx == ctx).unwrap().steps[step]
    }

    /// Flip one bit of raw record `name` at `(ctx, step)`.
    fn mutate_raw(m: &mut DumpModel, ctx: usize, step: usize, name: &str, off: usize) {
        let s = step_mut(m, ctx, step);
        s.raw.iter_mut().find(|(n, _)| n == name).unwrap().1[off] ^= 0x40;
    }

    fn mutate_logits(m: &mut DumpModel, ctx: usize, step: usize, off: usize) {
        step_mut(m, ctx, step).logits[off] ^= 0x10;
    }

    fn test_dir(name: &str) -> PathBuf {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/railgun-state-tests")
            .join(name);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Write `m` as a dump directory (case dirs + arm.json) with the arm run's writers.
    fn write_dump(dir: &Path, m: &DumpModel) {
        std::fs::create_dir_all(dir).unwrap();
        let regions: Vec<Value> = REGIONS.iter().map(|(n, o, l)| json!([n, o, l])).collect();
        let dig = |bytes: &[u8]| json!(hx(digest(bytes)));
        let mut cases_summary = Vec::new();
        for case in &m.cases {
            let cdir = dir.join(case_dir_name(case.ctx));
            let mut files = CaseFiles::create(&cdir).unwrap();
            let mut recs: Vec<Value> = Vec::new();
            for (s, st) in case.steps.iter().enumerate() {
                files.put_logits(&st.logits).unwrap();
                for (name, bytes) in &st.raw {
                    files.put_raw(s, name, bytes).unwrap();
                }
                let persist = raw_of(st, "persist");
                let region_digests: Vec<String> = REGIONS
                    .iter()
                    .map(|(_, o, l)| hx(digest(&persist[*o..*o + *l])))
                    .collect();
                recs.push(json!({
                    "step": s,
                    "input_token": st.input_token,
                    "next_token": st.next_token,
                    "n_tokens": st.n_tokens,
                    "device_seq_len": st.device_seq_len,
                    "mirror_valid_rows": st.mirror_valid_rows,
                    "mirror_rows": 64,
                    "capacity_rows": 256,
                    "logits_digest": dig(&st.logits),
                    "persist_region_digests": region_digests,
                    "stream_digests": {
                        "stream_a": dig(raw_of(st, "stream_a")),
                        "stream_b": dig(raw_of(st, "stream_b")),
                        "final_mean": dig(raw_of(st, "final_mean")),
                        "final_norm": dig(raw_of(st, "final_norm")),
                    },
                    "prefix_digests": {
                        "latent": [dig(raw_of(st, "latent[0]"))],
                        "mirror_k": [dig(raw_of(st, "mirror_k[0]"))],
                        "mirror_v": [dig(raw_of(st, "mirror_v[0]"))],
                    },
                }));
            }
            let raw_index = files.finish().unwrap();
            let full = json!({
                "schema": SCHEMA,
                "ctx": case.ctx,
                "steps": m.steps,
                "steps_recorded": recs.len() - 1,
                "dir": case_dir_name(case.ctx),
                "pass": true,
                "logit_bytes_each": VOCAB * 4,
                "latent_row_bytes": LROW,
                "persist_regions": regions,
                "persist_total_bytes": PERSIST,
                "step_records": recs,
                "raw_index": raw_index,
            });
            cases_summary
                .push(json!({"ctx": case.ctx, "dir": case_dir_name(case.ctx), "pass": true}));
            std::fs::write(
                cdir.join("case.json"),
                serde_json::to_string_pretty(&full).unwrap(),
            )
            .unwrap();
        }
        let total = (m.ctxs.len() * m.steps) as u64;
        let counts = match m.arm {
            Arm::Graph => Counts {
                steps: total,
                owner_off: total,
                ..Counts::default()
            },
            Arm::Railgun => Counts {
                steps: total,
                railgun: m.railgun_steps,
                declined_graph: total - m.railgun_steps,
                ..Counts::default()
            },
        };
        let owner_final = match m.arm {
            Arm::Graph => Value::Null,
            Arm::Railgun => owner::Snapshot {
                attempts: total,
                completed: m.railgun_steps,
                declined: total - m.railgun_steps,
                builds: 1,
                rebuilds: 0,
                quarantined: false,
                last_refusal: None,
                declined_by_kind: BTreeMap::new(),
                nodes: 0,
                ib_dwords: 0,
                object_digests: Vec::new(),
                last_build_ms: 0.0,
                last_rebuild_cause: None,
            }
            .json(),
        };
        let doc = json!({
            "schema": SCHEMA,
            "oracle": "railgun-state",
            "arm": m.arm.name(),
            "requested_backend": m.arm.name(),
            "executing_backend": counts.executing_backend(),
            "pass": m.arm_pass,
            "failures": Vec::<String>::new(),
            "model": {"sha256": m.sha},
            "prompt": {"base_ids_md5_u32le": m.prompt_md5},
            "ctxs": m.ctxs,
            "steps": m.steps,
            "executing": counts.json(),
            "owner_final": owner_final,
            "weights_digest_scope": m.scope,
            "weights_digest_before": m.weights,
            "weights_digest_after": m.weights,
            "weights_unchanged": true,
            "pointer_tables_before": {"tables": 2, "entries": 8, "consistent": m.ptr_consistent, "digest128": m.ptr_digest},
            "pointer_tables_after": {"tables": 2, "entries": 8, "consistent": m.ptr_consistent, "digest128": m.ptr_digest},
            "cases": cases_summary,
            "controls": {"binary_md5": "x"},
            "dump_dir": std::fs::canonicalize(dir).unwrap().display().to_string(),
        });
        std::fs::write(
            dir.join("arm.json"),
            serde_json::to_string_pretty(&doc).unwrap(),
        )
        .unwrap();
    }

    fn pair(name: &str, a: &DumpModel, b: &DumpModel) -> (PathBuf, PathBuf) {
        let root = test_dir(name);
        let (da, db) = (root.join("a"), root.join("b"));
        write_dump(&da, a);
        write_dump(&db, b);
        (da, db)
    }

    fn args(v: &[String]) -> Args {
        parse_args(v).unwrap()
    }

    /// `--compare-dir A B <extra>` through the real `run_compare`.
    fn run(a: &Path, b: &Path, extra: &[&str]) -> Result<bool, String> {
        let mut v = vec![
            "--compare-dir".to_string(),
            a.display().to_string(),
            b.display().to_string(),
        ];
        v.extend(extra.iter().map(|s| s.to_string()));
        run_compare(&args(&v))
    }

    fn cmp(a: &Path, b: &Path, flip: Option<&Flip>) -> ArmsCompare {
        compare_arms(a.to_str().unwrap(), b.to_str().unwrap(), true, flip).unwrap()
    }

    fn has_div(c: &ArmsCompare, ctx: usize, step: i64, tensor: &str, off: Option<usize>) -> bool {
        c.divs
            .iter()
            .any(|d| d.ctx == ctx && d.step == step && d.tensor == tensor && d.offset == off)
    }

    fn has_tensor(c: &ArmsCompare, ctx: usize, step: i64, tensor: &str) -> bool {
        c.divs
            .iter()
            .any(|d| d.ctx == ctx && d.step == step && d.tensor == tensor)
    }

    fn has_pre(c: &ArmsCompare, needle: &str) -> bool {
        c.json["preconditions_failed"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p.as_str().unwrap().contains(needle))
    }

    fn graph_railgun(ctxs: &[usize], steps: usize) -> (DumpModel, DumpModel) {
        (
            build_model(Arm::Graph, ctxs, steps),
            build_model(Arm::Railgun, ctxs, steps),
        )
    }

    // (1) identical graph / railgun dumps with the owner having completed steps.
    #[test]
    fn identical_dumps_compare_clean() {
        let (g, r) = graph_railgun(&[3, 4], 3);
        let (da, db) = pair("identical", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(c.pass, "{:?}", c.lines);
        assert!(c.divs.is_empty());
        assert_eq!(c.json["arms"][1]["steps_railgun"], json!(6));
        for case in c.json["cases"].as_array().unwrap() {
            assert_eq!(case["steps_compared"], json!(4));
            assert_eq!(case["logit_vectors_compared"], json!(4));
            assert!(case["raw_records_compared"].as_u64().unwrap() > 4 * 8);
            assert!(case["bytes_compared"].as_u64().unwrap() > 0);
        }
        assert_eq!(run(&da, &db, &[]), Ok(true));
        // --compare ARM.json ARM.json (the dump_dir key resolves the case files) + --out.
        let out = da.join("..").join("compare.json");
        let ok = run_compare(&args(&[
            "--compare".to_string(),
            da.join("arm.json").display().to_string(),
            db.join("arm.json").display().to_string(),
            "--out".to_string(),
            out.display().to_string(),
        ]));
        assert_eq!(ok, Ok(true));
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(doc["ok"], json!(true));
        assert_eq!(doc["compare"]["divergence_count"], json!(0));
    }

    // (2) one differing logits byte: ctx, step, tensor and exact byte offset reported.
    #[test]
    fn one_logits_byte_is_reported_exactly() {
        let (g, mut r) = graph_railgun(&[3, 4], 3);
        mutate_logits(&mut r, 4, 2, 9);
        let (da, db) = pair("logits_byte", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(!c.pass);
        assert_eq!(c.divs.len(), 1, "{:?}", c.lines);
        let d = &c.divs[0];
        assert_eq!(
            (d.ctx, d.step, d.tensor.as_str(), d.offset),
            (4, 2, "logits", Some(9))
        );
        assert_eq!(d.region.as_deref(), Some("f32 element 2"));
        assert!(d.line().contains("ctx 4 step 2 tensor logits offset 9"));
        let first = &c.json["first_divergence"];
        assert_eq!(first["ctx"], json!(4));
        assert_eq!(first["step"], json!(2));
        assert_eq!(first["tensor"], json!("logits"));
        assert_eq!(first["offset"], json!(9));
        assert!(c
            .lines
            .iter()
            .any(|l| l.starts_with("DIVERGENCE ctx 4 step 2 tensor logits offset 9")));
        assert_eq!(run(&da, &db, &[]), Ok(false));
    }

    // (3a) one persist-slab byte.
    #[test]
    fn persist_slab_byte_fails_and_names_region() {
        let (g, mut r) = graph_railgun(&[3, 4], 3);
        mutate_raw(&mut r, 3, 1, "persist", 20);
        let (da, db) = pair("persist_byte", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(!c.pass);
        let raw = c
            .divs
            .iter()
            .find(|d| d.tensor == "persist" && d.offset == Some(20))
            .unwrap_or_else(|| panic!("no raw persist div: {:?}", c.lines));
        assert_eq!((raw.ctx, raw.step), (3, 1));
        assert_eq!(raw.region.as_deref(), Some("conv_q[0] +4"));
        // The per-step region digest also names the slab region.
        let dig = c
            .divs
            .iter()
            .find(|d| d.tensor == "persist" && d.offset == Some(16))
            .unwrap();
        assert_eq!(dig.region.as_deref(), Some("conv_q[0]"));
        assert_eq!(c.divs.len(), 2);
        assert_eq!(run(&da, &db, &[]), Ok(false));
    }

    // (3b) differing latent / mirror raw bytes (live prefix and newly written row).
    #[test]
    fn latent_and_mirror_bytes_fail_and_name_tensor() {
        for (name, tensor_in_div, off) in [
            ("latent[0]", "latent[0]", 5usize),
            ("latent_row[0]", "latent_row[0]", 3),
            ("mirror_k[0]", "mirror_k[0]", 7),
            ("mirror_v[0]", "mirror_v[0]", 2),
            ("mirror_k_row[0]", "mirror_k_row[0]", 1),
            ("mirror_v_row[0]", "mirror_v_row[0]", 4),
        ] {
            let (g, mut r) = graph_railgun(&[3, 4], 3);
            mutate_raw(&mut r, 4, 2, name, off);
            let (da, db) = pair(
                &format!("latent_mirror_{}", name.replace(['[', ']'], "")),
                &g,
                &r,
            );
            let c = cmp(&da, &db, None);
            assert!(!c.pass, "{name}");
            assert!(
                has_div(&c, 4, 2, tensor_in_div, Some(off)),
                "{name}: {:?}",
                c.lines
            );
            assert!(c.divs.iter().all(|d| d.ctx == 4 && d.step == 2), "{name}");
            assert_eq!(run(&da, &db, &[]), Ok(false), "{name}");
        }
        // The live-prefix digest of a full latent/mirror layer is a divergence on its own.
        let (g, mut r) = graph_railgun(&[3], 2);
        mutate_raw(&mut r, 3, 1, "mirror_v[0]", 0);
        let (da, db) = pair("latent_mirror_digest", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(has_tensor(&c, 3, 1, "mirror_v[0]"));
        assert!(c
            .divs
            .iter()
            .any(|d| d.tensor == "mirror_v[0]" && d.offset.is_none()));
    }

    // (3c) differing position fields.
    #[test]
    fn position_fields_fail_and_name_tensor() {
        let (g, mut r) = graph_railgun(&[3, 4], 3);
        step_mut(&mut r, 3, 2).device_seq_len = 9;
        let (da, db) = pair("position_seq", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(!c.pass);
        assert!(has_tensor(&c, 3, 2, "positions"), "{:?}", c.lines);
        assert!(has_tensor(&c, 3, 2, "device_seq_len"));
        let p = c.divs.iter().find(|d| d.tensor == "positions").unwrap();
        assert!(p.detail.contains("arm B"));
        assert_eq!(run(&da, &db, &[]), Ok(false));

        let (g, mut r) = graph_railgun(&[3, 4], 3);
        step_mut(&mut r, 4, 1).n_tokens = 99;
        let (da, db) = pair("position_ntokens", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(has_tensor(&c, 4, 1, "positions"));
        assert!(has_tensor(&c, 4, 1, "n_tokens"));
        assert_eq!(run(&da, &db, &[]), Ok(false));

        // Both arms agree with each other but not with p+i: positions still fails.
        let (mut g, mut r) = graph_railgun(&[3], 2);
        step_mut(&mut g, 3, 1).device_seq_len = 8;
        step_mut(&mut r, 3, 1).device_seq_len = 8;
        let (da, db) = pair("position_both_wrong", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(has_tensor(&c, 3, 1, "positions"));
        assert!(!has_tensor(&c, 3, 1, "device_seq_len"));
        assert_eq!(run(&da, &db, &[]), Ok(false));

        let (g, mut r) = graph_railgun(&[3], 2);
        step_mut(&mut r, 3, 2).mirror_valid_rows = 1;
        let (da, db) = pair("position_mirror_valid", &g, &r);
        assert!(has_tensor(&cmp(&da, &db, None), 3, 2, "mirror_valid_rows"));
    }

    // (4) requested railgun, owner executed zero steps, every byte equal: still fails.
    #[test]
    fn railgun_arm_with_zero_executed_steps_fails() {
        let (g, mut r) = graph_railgun(&[3, 4], 3);
        r.railgun_steps = 0;
        let (da, db) = pair("zero_executed", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(c.divs.is_empty(), "bytes are equal: {:?}", c.lines);
        assert!(!c.pass);
        assert!(has_pre(&c, "requested railgun but executed 0 steps"));
        assert_eq!(c.json["arms"][1]["steps_railgun"], json!(0));
        assert_eq!(run(&da, &db, &[]), Ok(false));
        // ... and it stays a failure under the negative control (baseline not clean).
        assert_eq!(run(&da, &db, &["--flip-byte", "1:logits:0"]), Ok(false));
    }

    // (5) `--flip-byte`: the control detects exactly the flipped byte.
    #[test]
    fn flip_byte_control_detects_exactly_that_byte() {
        let (g, r) = graph_railgun(&[3, 4], 3);
        let (da, db) = pair("flip", &g, &r);
        let f = parse_flip("1:logits:0").unwrap();
        let c = cmp(&da, &db, Some(&f));
        assert_eq!(c.hits.len(), 2, "one hit per ctx");
        for h in &c.hits {
            assert_eq!((h.step, h.tensor.as_str(), h.offset), (1, "logits", 0));
            assert!(has_div(&c, h.ctx, 1, "logits", Some(0)));
        }
        assert_eq!(
            c.divs.len(),
            2,
            "nothing but the flipped bytes: {:?}",
            c.lines
        );
        let out = da.join("..").join("flip.json");
        assert_eq!(
            run(
                &da,
                &db,
                &["--flip-byte", "1:logits:0", "--out", out.to_str().unwrap()]
            ),
            Ok(true)
        );
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(doc["ok"], json!(true));
        assert_eq!(doc["negative_control"]["baseline_clean"], json!(true));
        assert_eq!(doc["negative_control"]["detected"], json!([true, true]));
        // Raw tensors flip too (persist slab / latent row / mirror plane).
        for spec in [
            "2:persist:20",
            "3:latent_row[0]:7",
            "2:mirror_k[0]:0",
            "1:final_norm:15",
            "0:stream_a:31",
        ] {
            assert_eq!(run(&da, &db, &["--flip-byte", spec]), Ok(true), "{spec}");
        }
        // The unflipped comparison is untouched by a flip run (the loaded copy only).
        assert_eq!(run(&da, &db, &[]), Ok(true));
    }

    // (5b) controls that must FAIL. A flip that is applied is always detectable (the byte
    // is XORed in the loaded copy before the diff), so an "applied but undetected" dump
    // set is not constructible; the constructible refusals are checked instead.
    #[test]
    fn flip_byte_control_fails_when_not_meaningful() {
        let (g, r) = graph_railgun(&[3], 3);
        let (da, db) = pair("flip_refused", &g, &r);
        // Target not found in any case: the control did not run.
        let f = parse_flip("1:no_such_tensor:0").unwrap();
        let c = cmp(&da, &db, Some(&f));
        assert!(c.hits.is_empty());
        assert_eq!(
            run(&da, &db, &["--flip-byte", "1:no_such_tensor:0"]),
            Ok(false)
        );
        // Step outside the recorded range.
        assert_eq!(run(&da, &db, &["--flip-byte", "99:logits:0"]), Ok(false));
        // A step that has no such raw record (latent_row is recorded from step 1).
        assert_eq!(
            run(&da, &db, &["--flip-byte", "0:latent_row[0]:0"]),
            Ok(false)
        );
        // Offset beyond the tensor is an error, not a silent pass.
        assert!(run(&da, &db, &["--flip-byte", "1:logits:16"]).is_err());
        // Malformed spec.
        assert!(run(&da, &db, &["--flip-byte", "logits"]).is_err());
        assert!(run(&da, &db, &["--flip-byte", "x:logits:0"]).is_err());
        // Baseline not clean: the control is meaningless and fails.
        let (g, mut r) = graph_railgun(&[3], 3);
        mutate_logits(&mut r, 3, 2, 4);
        let (da, db) = pair("flip_dirty_baseline", &g, &r);
        assert_eq!(run(&da, &db, &["--flip-byte", "1:logits:0"]), Ok(false));
    }

    // (6) mismatched model sha / ctx list / steps / prompt / same arm are refused.
    #[test]
    fn mismatched_preimages_are_refused() {
        // model sha
        let (g, mut r) = graph_railgun(&[3, 4], 3);
        r.sha = Some("f".repeat(64));
        let (da, db) = pair("mismatch_sha", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(!c.pass);
        assert!(has_pre(&c, "model sha256 differ"));
        assert_eq!(run(&da, &db, &[]), Ok(false));
        // model sha not pinned
        let (mut g, mut r) = graph_railgun(&[3], 2);
        g.sha = None;
        r.sha = None;
        let (da, db) = pair("mismatch_sha_null", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(has_pre(&c, "not pinned"));
        assert_eq!(run(&da, &db, &[]), Ok(false));
        // ctx lists
        let g = build_model(Arm::Graph, &[3, 4], 3);
        let r = build_model(Arm::Railgun, &[3, 5], 3);
        let (da, db) = pair("mismatch_ctxs", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(!c.pass);
        assert!(has_pre(&c, "ctx list differ"));
        assert!(
            c.json["cases"].as_array().unwrap().is_empty(),
            "no case is compared"
        );
        assert_eq!(run(&da, &db, &[]), Ok(false));
        // steps
        let g = build_model(Arm::Graph, &[3, 4], 3);
        let r = build_model(Arm::Railgun, &[3, 4], 2);
        let (da, db) = pair("mismatch_steps", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(!c.pass);
        assert!(has_pre(&c, "steps differ"));
        assert!(has_tensor(&c, 3, -1, "step_records"), "{:?}", c.lines);
        assert_eq!(run(&da, &db, &[]), Ok(false));
        // prompt ids
        let (g, mut r) = graph_railgun(&[3], 2);
        r.prompt_md5 = "q".repeat(32);
        let (da, db) = pair("mismatch_prompt", &g, &r);
        assert!(has_pre(&cmp(&da, &db, None), "prompt ids differ"));
        // weights digests
        let (g, mut r) = graph_railgun(&[3], 2);
        r.weights = "v".repeat(32);
        let (da, db) = pair("mismatch_weights", &g, &r);
        assert!(has_pre(&cmp(&da, &db, None), "weights digests differ"));
        // The pointer tables hold process addresses: arms with different raw table digests (the fixture
        // gives the two arms different ones) still compare clean when the immutable weights agree.
        let (g, r) = graph_railgun(&[3], 2);
        assert_ne!(g.ptr_digest, r.ptr_digest);
        let (da, db) = pair("pointer_digests_differ_by_design", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(c.pass, "{:?}", c.lines);
        // ... but a table that does not hold the addresses it names is refused,
        let (g, mut r) = graph_railgun(&[3], 2);
        r.ptr_consistent = false;
        let (da, db) = pair("pointer_table_inconsistent", &g, &r);
        assert!(has_pre(&cmp(&da, &db, None), "pointer tables do not hold"));
        // ... and so is a dump whose weights digest has another scope (taken with the pointer tables in it).
        let (g, mut r) = graph_railgun(&[3], 2);
        r.scope = None;
        let (da, db) = pair("scope_missing", &g, &r);
        assert!(has_pre(
            &cmp(&da, &db, None),
            "weights digest scope differs"
        ));
        // two graph arms / two railgun arms are not a graph-vs-railgun comparison
        let (g, _) = graph_railgun(&[3], 2);
        let (da, db) = pair("mismatch_same_arm", &g, &g);
        assert!(has_pre(&cmp(&da, &db, None), "expected graph vs railgun"));
        assert_eq!(run(&da, &db, &[]), Ok(false));
        // an arm whose own verdict failed
        let (g, mut r) = graph_railgun(&[3], 2);
        r.arm_pass = false;
        let (da, db) = pair("mismatch_arm_verdict", &g, &r);
        let c = cmp(&da, &db, None);
        assert!(c.divs.is_empty());
        assert!(has_pre(&c, "failed its own verdict"));
        assert_eq!(run(&da, &db, &[]), Ok(false));
    }

    // A dump whose logits.bin was edited after the arm wrote its digests is flagged as an
    // integrity failure, not as a clean comparison of two equally-corrupt files.
    #[test]
    fn corrupted_dump_files_are_flagged() {
        let (g, r) = graph_railgun(&[3], 2);
        let (da, db) = pair("corrupt", &g, &r);
        let p = db.join("ctx3").join("logits.bin");
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[VOCAB * 4 + 1] ^= 0x01;
        std::fs::write(&p, bytes).unwrap();
        let c = cmp(&da, &db, None);
        assert!(!c.pass);
        assert!(has_tensor(&c, 3, 1, "dump_integrity:logits"));
        assert!(has_div(&c, 3, 1, "logits", Some(1)));
        assert_eq!(run(&da, &db, &[]), Ok(false));
    }
}
