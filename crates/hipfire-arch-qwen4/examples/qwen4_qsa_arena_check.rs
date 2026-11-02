// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! BF-1 QSA arena byte oracle on a real Flash-Next model and committed text
//! prompt, through production prefill, rollback and prefix restore.
//!
//! One fresh process per arm (off / BF-1). Each process runs, on one
//! real prompt (the prompt file framed exactly as the serve request frames it:
//! one user message, `enable_thinking=false`, the model chat template;
//! `QSA_ORACLE_FRAME=raw` for a plain encode; no repetition; 8192-row chunks):
//!
//! * R0 cold, tapped: reset, then every chunk through `forward_chunk_final`.
//!   For every QSA layer of every chunk and decode step the tap hashes the
//!   selection bytes (`selected_scratch`, all rows) and the attention output
//!   bytes (`qsa_output`, all rows) with SHA-256 and MD5. After every chunk the
//!   full persistent F32 state (every `SnapshotParts` fixed part and the valid
//!   rows of every row stream, unselected rows included) is hashed. Rolling
//!   chunk-boundary hashes are checked by a full re-read at the end. Then
//!   `DECODE` greedy tokens, tapped. A snapshot at the first chunk is restored
//!   after decode; its watermark must clamp, and continuing must match cold.
//! * R1 cold, untapped production `prefill_final` (captures session snapshots
//!   at every chunk boundary) + the same greedy decode, untapped. Final
//!   state, final logits and ids must equal R0 (the tap is transparent).
//! * R2 branch: a different real-text prefix is prefilled over the live state
//!   (and its arenas), so a restore that leaves anything stale is exposed.
//! * R3a restore only: `prefill_final(prompt[..reused], reused)` restores the
//!   deepest snapshot; the restored state must equal R0's boundary state, and
//!   a derived F16 arena must not claim rows beyond the restored position.
//! * R3 restore + tail + decode, tapped: the restored run's tail chunk and
//!   decode steps (selection bytes, attention bytes, final state, logits, ids)
//!   must equal R0 byte for byte.
//! * Derived consistency (BF-1): after every tap the watermark must not
//!   exceed the operation end. At the last prefill chunk, every valid F16 row
//!   is compared bitwise with a fresh whole-prefix existing GPU conversion.
//!   Decode and non-gathered routes do not convert or advance the arena.
//!   Prefix restore invalidates it; the next gathered attend rebuilds it.
//! * Pooled BF16 mirror (BF-2, opt-in `HIPFIRE_QWEN4_QSA_POOLED_BF16=1`): a
//!   derived BF16 copy of the canonical F32 pooled index keys. The arm must
//!   reproduce the off arm's selection, attention, state, logits and ids
//!   through prefill, rollback, branch and restore. After every phase each
//!   layer's mirror prefix `[0, valid)` is downloaded and compared with the
//!   high half of the canonical F32 pooled rows, whose low half must be zero
//!   (the exactness the mirror relies on). Prefill phases must leave the
//!   watermark at every pooled block, rollback must clamp it to the restored
//!   pooled length, no phase may leave it beyond the pooled length, and
//!   decode never rewinds it. Without the flag no layer may hold a mirror.
//!
//! The output protocol (stdout, one line each, parsed fail-closed):
//!
//! ```text
//! ARENA_DEVICE {json}     arch, PCI bus id, device, flags, model, prompt
//! ARENA_ROUTE {json}      state formats, index dtype, arena presence, env
//! QSA_ARENA_HASH <label> sha256=<64 hex> md5=<32 hex>
//!                         labels: tokens selection attention state logits ids canonical
//! ARENA_FAIL <message>    zero or more
//! ARENA_RESULT PASS | ARENA_RESULT FAIL <reason>
//! ```
//!
//! Exit code 0 only on PASS, 1 on a failed internal comparison, 2 on a setup
//! error. Every canonical/per-layer hash must agree across the two arms;
//! the per-arm JSONL has every record behind it.
//!
//! usage: qwen4_qsa_arena_check MODEL PROMPT_TEXT OUT.jsonl
//!   QSA_ORACLE_EXPECT_TOKENS=N   fail unless the framed prompt is exactly N
//!                                tokens (served counts: 16343 / 57544)
//!   QSA_ORACLE_EXPECT_PCI=BUS    fail unless the device PCI bus id matches
//!   QSA_ORACLE_DECODE=8          greedy continuation steps
//!   QSA_ORACLE_BRANCH_FILE=PATH  alternate real text for R2 (default: a slice
//!                                of the prompt from a third of the way in)
//!   QSA_ORACLE_FRAME=serve|raw   prompt framing (default serve)
//!   QSA_ORACLE_ALLOW_ARCH=1      permit a device other than gfx1151
//!   QSA_ORACLE_COMPARE=1        run fresh off/on children and compare every
//!                                canonical hash before emitting PASS
//!   QSA_ORACLE_KV_BACKEND=auto|legacy|vmm  exercise either KV backend
//!   QSA_ORACLE_POOLED=1          with QSA_ORACLE_COMPARE: also run the pooled
//!                                BF16 mirror arms (without / with the F16
//!                                arena) and require canonical equality

use hip_bridge::DeviceBuffer;
use hipfire_arch_qwen4::admit_hfqm_artifact_on;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::gpu_forward::Qwen4QsaTap;
use hipfire_dispatch::pipeline::IndexedAttentionOp;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::prompt_frame::JinjaChatFrame;
use hipfire_runtime::serve_contract::CacheDomain;
use hipfire_runtime::session_cache::{SessionCache, SessionRoute, SessionState, SnapshotParts};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::tensor_ops::QsaKvFormat;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, LazyLock, Mutex};

type Result<T, E = String> = std::result::Result<T, E>;

/// The prefill chunk (rows) every arm must run at.
const CHUNK: usize = 8192;

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

// ---------------------------------------------------------------- hashing

#[derive(Clone)]
struct Md5 {
    state: [u32; 4],
    block: [u8; 64],
    filled: usize,
    length: u64,
}

static MD5_K: LazyLock<[u32; 64]> = LazyLock::new(|| {
    let mut k = [0u32; 64];
    for (i, slot) in k.iter_mut().enumerate() {
        *slot = (((i + 1) as f64).sin().abs() * 4294967296.0).floor() as u32;
    }
    k
});

const MD5_SHIFT: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

impl Md5 {
    fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            block: [0; 64],
            filled: 0,
            length: 0,
        }
    }

    fn compress(state: &mut [u32; 4], block: &[u8; 64]) {
        let k = &*MD5_K;
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            *word = u32::from_le_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        let [mut a, mut b, mut c, mut d] = *state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(MD5_SHIFT[i]));
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
    }

    fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        if self.filled > 0 {
            let take = (64 - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled == 64 {
                let block = self.block;
                Self::compress(&mut self.state, &block);
                self.filled = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            Self::compress(&mut self.state, &block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.block[..data.len()].copy_from_slice(data);
            self.filled = data.len();
        }
    }

    fn finalize(mut self) -> [u8; 16] {
        let bits = self.length.wrapping_mul(8);
        let pad = if self.filled < 56 {
            56 - self.filled
        } else {
            120 - self.filled
        };
        let mut tail = vec![0u8; pad + 8];
        tail[0] = 0x80;
        tail[pad..].copy_from_slice(&bits.to_le_bytes());
        // `update` would count the padding into `length`; the bits are fixed.
        let saved = self.length;
        self.update(&tail);
        self.length = saved;
        let mut out = [0u8; 16];
        for (i, word) in self.state.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        out
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 and MD5 of the same byte stream.
#[derive(Clone)]
struct Dual {
    sha: Sha256,
    md5: Md5,
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Hx {
    sha: String,
    md5: String,
}

impl Hx {
    fn json(&self) -> Value {
        json!({"sha256": self.sha, "md5": self.md5})
    }
}

impl Dual {
    fn new() -> Self {
        Self {
            sha: Sha256::new(),
            md5: Md5::new(),
        }
    }
    fn update(&mut self, bytes: &[u8]) {
        self.sha.update(bytes);
        self.md5.update(bytes);
    }
    fn finish(self) -> Hx {
        let sha: [u8; 32] = self.sha.finalize().into();
        Hx {
            sha: hex(&sha),
            md5: hex(&self.md5.finalize()),
        }
    }
    fn peek(&self) -> Hx {
        self.clone().finish()
    }
}

fn hash_bytes(bytes: &[u8]) -> Hx {
    let mut d = Dual::new();
    d.update(bytes);
    d.finish()
}

fn hash_f32(values: &[f32]) -> Hx {
    let mut d = Dual::new();
    for v in values {
        d.update(&v.to_le_bytes());
    }
    d.finish()
}

/// One digest over named digests, in order.
fn combine(items: &[(String, Hx)]) -> Hx {
    let mut d = Dual::new();
    for (key, h) in items {
        d.update(format!("{key} {} {}\n", h.sha, h.md5).as_bytes());
    }
    d.finish()
}

fn self_test_hashes() -> Result<()> {
    let empty = hash_bytes(b"");
    let abc = hash_bytes(b"abc");
    let long = hash_bytes(&[b'a'; 1000]);
    let mut split = Dual::new();
    split.update(&[b'a'; 333]);
    split.update(&[b'a'; 667]);
    if empty.md5 != "d41d8cd98f00b204e9800998ecf8427e"
        || abc.md5 != "900150983cd24fb0d6963f7d28e17f72"
        || split.finish() != long
        || empty.sha != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    {
        return Err("md5/sha256 self test failed".into());
    }
    Ok(())
}

// ------------------------------------------------------------ device reads

fn read_bytes(gpu: &Gpu, tensor: &GpuTensor, offset: usize, len: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; len];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &tensor.buf.byte_view(offset, len))
        .map_err(err)?;
    Ok(bytes)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
        .0
}

// -------------------------------------------------------------------- tap

#[derive(Clone)]
struct TapRec {
    position: usize,
    rows: usize,
    slot: usize,
    sel: Hx,
    att: Hx,
    sel_live: usize,
    att_nonzero: usize,
    arena_valid: Option<usize>,
}

impl TapRec {
    fn key(&self) -> (usize, usize, usize) {
        (self.position, self.rows, self.slot)
    }
    fn name(&self) -> String {
        format!("p{}r{}s{}", self.position, self.rows, self.slot)
    }
}

#[derive(Clone)]
struct Geo {
    index_dim: usize,
    compress: usize,
    capacity: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    index_dtype: DType,
    kv_format: QsaKvFormat,
    slots: usize,
    arena: bool,
}

struct Probe {
    records: Vec<TapRec>,
    checks: Vec<Value>,
    geo: Option<Geo>,
    failures: Vec<String>,
    want_arena: bool,
    /// Derived-arena check on the prefill op that ends at this position.
    derived_end: Option<usize>,
    phase: String,
}

impl Probe {
    fn fail(&mut self, message: String) {
        if self.failures.len() < 64 && !self.failures.contains(&message) {
            self.failures.push(message);
        }
    }

    fn tap(&mut self, gpu: &mut Gpu, slot: usize, op: &IndexedAttentionOp<'_>) -> Result<()> {
        gpu.hip.device_synchronize().map_err(err)?;
        let start = op.state.position;
        let rows = op.rows;
        let end = start + rows;
        let capacity = op.state.selected_capacity;
        let width = op.heads * op.head_dim;
        let arena_valid = op.state.f16_arena.as_ref().map(|a| a.valid_len.get());

        let observed = Geo {
            index_dim: op.index_dim,
            compress: op.compress,
            capacity,
            heads: op.heads,
            kv_heads: op.kv_heads,
            head_dim: op.head_dim,
            index_dtype: op.state.raw_index_keys.dtype,
            kv_format: op.state.format,
            slots: slot + 1,
            arena: arena_valid.is_some(),
        };
        let mut changed = false;
        match self.geo.as_mut() {
            None => self.geo = Some(observed.clone()),
            Some(g) => {
                g.slots = g.slots.max(slot + 1);
                changed = g.index_dim != observed.index_dim
                    || g.capacity != observed.capacity
                    || g.kv_heads != observed.kv_heads
                    || g.head_dim != observed.head_dim
                    || format!("{:?}", g.index_dtype) != format!("{:?}", observed.index_dtype);
            }
        }
        if changed {
            self.fail(format!("QSA geometry changed at layer {slot}"));
        }
        if observed.kv_format != QsaKvFormat::F32 {
            self.fail(format!(
                "QSA K/V format {} is not the F32 reference",
                observed.kv_format.name()
            ));
        }
        if rows > 1 && observed.arena != self.want_arena {
            self.fail(format!(
                "layer {slot}: derived F16 arena present={} but HIPFIRE_QWEN4_QSA_F16_ARENA says {}",
                observed.arena, self.want_arena
            ));
        }
        if rows == 1 && observed.arena && self.derived_end != Some(end) {
            self.fail(format!(
                "decode layer {slot} consumed a derived arena instead of F32"
            ));
        }
        if observed.index_dtype != DType::F32 {
            self.fail(format!(
                "layer {slot}: F32 QSA index dtype changed to {:?}",
                observed.index_dtype
            ));
        }
        if let Some(valid) = arena_valid {
            if valid > end {
                self.fail(format!(
                    "layer {slot} position {start}: derived arena claims {valid} valid rows beyond {end}"
                ));
            }
        }

        // Selection bytes, every row.
        let mut sel = Dual::new();
        let (mut live, mut out_of_range) = (0usize, 0usize);
        let mut r0 = 0;
        while r0 < rows {
            let n = (rows - r0).min(256);
            let bytes = read_bytes(
                gpu,
                op.selected_scratch,
                r0 * capacity * 4,
                n * capacity * 4,
            )?;
            for w in bytes.chunks_exact(4) {
                let t = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                if t >= 0 {
                    live += 1;
                    if t as usize >= end {
                        out_of_range += 1;
                    }
                }
            }
            sel.update(&bytes);
            r0 += n;
        }
        if out_of_range > 0 || live == 0 {
            self.fail(format!(
                "layer {slot} position {start}: selection has {live} live entries, {out_of_range} past position end {end}"
            ));
        }

        // Attention output bytes, every row.
        let mut att = Dual::new();
        let (mut nonzero, mut nonfinite) = (0usize, 0usize);
        let mut r0 = 0;
        while r0 < rows {
            let n = (rows - r0).min(256);
            let bytes = read_bytes(gpu, op.qsa_output, r0 * width * 4, n * width * 4)?;
            for w in bytes.chunks_exact(4) {
                let bits = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                nonzero += usize::from(bits != 0);
                nonfinite += usize::from(!f32::from_bits(bits).is_finite());
            }
            att.update(&bytes);
            r0 += n;
        }
        if nonzero == 0 || nonfinite > 0 {
            self.fail(format!(
                "layer {slot} position {start}: attention output has {nonzero} nonzero and {nonfinite} non-finite values"
            ));
        }

        let rec = TapRec {
            position: start,
            rows,
            slot,
            sel: sel.finish(),
            att: att.finish(),
            sel_live: live,
            att_nonzero: nonzero,
            arena_valid,
        };

        if let Some(valid) = arena_valid {
            let at_end = rows > 1 && self.derived_end == Some(end);
            if at_end {
                self.derived_check(gpu, slot, op, valid.min(end))?;
            }
        }
        self.records.push(rec);
        Ok(())
    }

    /// Compare the incremental arena with a fresh whole-prefix launch of the
    /// existing hipcc converter (also the certified PM producer's byte
    /// reference). Ignore unwritten V padding beyond the valid token end.
    fn derived_check(
        &mut self,
        gpu: &mut Gpu,
        slot: usize,
        op: &IndexedAttentionOp<'_>,
        valid: usize,
    ) -> Result<()> {
        let Some(arena) = op.state.f16_arena.as_ref() else {
            return Ok(());
        };
        let width = op.kv_heads * op.head_dim;
        if valid == 0 {
            return Ok(());
        }
        let rows = rdna_compute::tensor_ops::qsa_f16_arena_rows(valid);
        let keys = gpu.zeros(&[rows * width], DType::F16).map_err(err)?;
        let values = match gpu.zeros(&[rows * width], DType::F16) {
            Ok(values) => values,
            Err(e) => {
                let _ = gpu.free_tensor(keys);
                return Err(err(e));
            }
        };
        let compared = (|| -> Result<(usize, usize, Option<String>)> {
            let mark = rdna_compute::tensor_ops::QsaF16ValidLen::default();
            let reference = rdna_compute::tensor_ops::QsaF16ArenaRef {
                keys: &keys,
                values: &values,
                valid_len: &mark,
            };
            rdna_compute::tensor_ops::qsa_f16_arena_sync(
                gpu,
                op.state.full_keys,
                op.state.full_values,
                &reference,
                op.kv_heads,
                valid,
                valid,
                false,
            )
            .map_err(err)?;
            gpu.hip.device_synchronize().map_err(err)?;
            let (mut bad_k, mut bad_v) = (0usize, 0usize);
            let mut first = None;
            for t0 in (0..valid).step_by(8192) {
                let t1 = (t0 + 8192).min(valid);
                let k16 = read_bytes(gpu, arena.keys, t0 * width * 2, (t1 - t0) * width * 2)?;
                let expected = read_bytes(gpu, &keys, t0 * width * 2, (t1 - t0) * width * 2)?;
                for (i, (got, want)) in k16
                    .chunks_exact(2)
                    .zip(expected.chunks_exact(2))
                    .enumerate()
                {
                    if got != want {
                        bad_k += 1;
                        first.get_or_insert_with(|| {
                            format!("K token {} channel {}", t0 + i / width, i % width)
                        });
                    }
                }
                let (b0, b1) = (t0 / 4, t1.div_ceil(4));
                let v16 = read_bytes(gpu, arena.values, b0 * width * 8, (b1 - b0) * width * 8)?;
                let expected = read_bytes(gpu, &values, b0 * width * 8, (b1 - b0) * width * 8)?;
                for tok in t0..t1 {
                    for c in 0..width {
                        let idx = (((tok >> 2) - b0) * width + c) * 4 + (tok & 3);
                        if v16[2 * idx..2 * idx + 2] != expected[2 * idx..2 * idx + 2] {
                            bad_v += 1;
                            first.get_or_insert_with(|| format!("V token {tok} channel {c}"));
                        }
                    }
                }
            }
            Ok((bad_k, bad_v, first))
        })();
        let free_keys = gpu.free_tensor(keys).map_err(err);
        let free_values = gpu.free_tensor(values).map_err(err);
        let (bad_k, bad_v, first) = compared?;
        free_keys?;
        free_values?;
        self.checks.push(
            json!({"kind":"derived_arena","phase":self.phase,"slot":slot,
            "position":op.state.position,"valid_rows":valid,"bad_k":bad_k,"bad_v":bad_v}),
        );
        if bad_k + bad_v > 0 {
            let message = format!(
                "{} layer {slot} position {}: derived F16 arena differs from the existing whole-prefix convert in {bad_k} K and {bad_v} V elements (first {})",
                self.phase,
                op.state.position,
                first.unwrap_or_default()
            );
            self.fail(message);
        }
        Ok(())
    }
}

type SharedProbe = Arc<Mutex<Probe>>;

fn install_tap(bundle: &mut Qwen4Bundle, probe: &SharedProbe, on: bool) -> Result<()> {
    if on {
        let shared = Arc::clone(probe);
        let tap: Qwen4QsaTap =
            Box::new(move |gpu, slot, op| shared.lock().map_err(err)?.tap(gpu, slot, op));
        bundle.set_qsa_tap(Some(tap)).map_err(err)
    } else {
        bundle.set_qsa_tap(None).map_err(err)
    }
}

fn begin_phase(probe: &SharedProbe, phase: &str, derived_end: Option<usize>) {
    let mut p = probe.lock().unwrap();
    p.phase = phase.to_string();
    p.records.clear();
    p.derived_end = derived_end;
}

fn take_records(probe: &SharedProbe) -> Vec<TapRec> {
    std::mem::take(&mut probe.lock().unwrap().records)
}

// ------------------------------------------------- pooled BF16 mirror

/// What a layer's pooled BF16 mirror watermark must be at a check point.
enum MirrorRule {
    /// A prefill just ran: the mirror caught up to every pooled block.
    Caught,
    /// Restore or decode: any prefix of the canonical pooled rows.
    Within,
    /// Exactly these per-layer marks (rollback clamps to the restored length).
    Exact(Vec<usize>),
}

/// Per-layer mirror watermark in blocks (0 without a mirror).
fn pooled_marks(bundle: &Qwen4Bundle) -> Vec<usize> {
    bundle
        .state
        .qsa
        .iter()
        .map(|s| s.pooled_bf16.as_ref().map_or(0, |a| a.valid_blocks.get()))
        .collect()
}

/// Compare every layer's mirror prefix with the canonical pooled rows: the
/// mirror holds the high 16 bits of each F32, whose low 16 bits are zero.
fn pooled_check(
    bundle: &Qwen4Bundle,
    gpu: &Gpu,
    probe: &SharedProbe,
    geo: &Geo,
    phase: &str,
    want: bool,
    rule: &MirrorRule,
) -> Result<()> {
    gpu.hip.device_synchronize().map_err(err)?;
    let dim = geo.index_dim;
    let mut failures: Vec<String> = Vec::new();
    let (mut blocks, mut present) = (0usize, 0usize);
    let mut marks: Vec<usize> = Vec::new();
    for (slot, layer) in bundle.state.qsa.iter().enumerate() {
        let Some(arena) = layer.pooled_bf16.as_ref() else {
            if want {
                failures.push(format!("{phase} layer {slot}: pooled BF16 mirror missing"));
            }
            continue;
        };
        if !want {
            failures.push(format!(
                "{phase} layer {slot}: pooled BF16 mirror present but the flag is off"
            ));
            continue;
        }
        present += 1;
        let valid = arena.valid_blocks.get();
        marks.push(valid);
        if valid > layer.pooled_len {
            failures.push(format!(
                "{phase} layer {slot}: mirror claims {valid} valid blocks beyond the {} pooled blocks",
                layer.pooled_len
            ));
            continue;
        }
        match rule {
            MirrorRule::Caught if valid != layer.pooled_len => failures.push(format!(
                "{phase} layer {slot}: mirror valid {valid}, expected every one of {} pooled blocks",
                layer.pooled_len
            )),
            MirrorRule::Exact(expected) if valid != expected[slot] => failures.push(format!(
                "{phase} layer {slot}: mirror valid {valid}, expected {}",
                expected[slot]
            )),
            _ => {}
        }
        let (mut bad_high, mut bad_low) = (0usize, 0usize);
        let mut first: Option<String> = None;
        for b0 in (0..valid).step_by(4096) {
            let b1 = (b0 + 4096).min(valid);
            let pooled = read_bytes(gpu, &layer.pooled_keys, b0 * dim * 4, (b1 - b0) * dim * 4)?;
            let mirror = read_bytes(gpu, &arena.mirror, b0 * dim * 2, (b1 - b0) * dim * 2)?;
            for (i, (word, half)) in pooled
                .chunks_exact(4)
                .zip(mirror.chunks_exact(2))
                .enumerate()
            {
                if word[..2] != [0, 0] {
                    bad_low += 1;
                    first.get_or_insert_with(|| {
                        format!(
                            "block {} channel {} low half nonzero",
                            b0 + i / dim,
                            i % dim
                        )
                    });
                }
                if word[2..] != *half {
                    bad_high += 1;
                    first.get_or_insert_with(|| {
                        format!(
                            "block {} channel {} mirror differs from the F32 high half",
                            b0 + i / dim,
                            i % dim
                        )
                    });
                }
            }
        }
        blocks += valid;
        if bad_high + bad_low > 0 {
            failures.push(format!(
                "{phase} layer {slot}: pooled mirror prefix of {valid} blocks has {bad_high} high-half and {bad_low} low-half violations (first: {})",
                first.unwrap_or_default()
            ));
        }
    }
    let mut p = probe.lock().map_err(err)?;
    p.checks.push(
        json!({"kind":"pooled_mirror","phase":phase,"layers":present,"blocks":blocks,
        "marks":marks,"failures":failures.len()}),
    );
    for message in failures {
        p.fail(message);
    }
    Ok(())
}

// ---------------------------------------------------------- state hashing

struct StateRoll {
    duals: Vec<Dual>,
    done: Vec<usize>,
}

impl StateRoll {
    fn new(streams: usize) -> Self {
        Self {
            duals: vec![Dual::new(); streams],
            done: vec![0; streams],
        }
    }
}

#[derive(Clone)]
struct StateHash {
    position: usize,
    meta: Hx,
    fixed: Vec<Hx>,
    rows: Vec<Hx>,
    row_counts: Vec<usize>,
    combined: Hx,
}

impl StateHash {
    fn json(&self, label: &str) -> Value {
        json!({"kind":"state","label":label,"position":self.position,"combined":self.combined.json(),
            "meta":self.meta.json(),
            "fixed":self.fixed.iter().map(Hx::json).collect::<Vec<_>>(),
            "rows":self.rows.iter().map(Hx::json).collect::<Vec<_>>(),
            "row_counts":self.row_counts})
    }
}

fn feed_rows(
    gpu: &Gpu,
    buf: &DeviceBuffer,
    row_bytes: usize,
    from: usize,
    to: usize,
    dual: &mut Dual,
) -> Result<()> {
    let slice_rows = ((32usize << 20) / row_bytes.max(1)).max(1);
    let mut r0 = from;
    while r0 < to {
        let n = (to - r0).min(slice_rows);
        let mut host = vec![0u8; n * row_bytes];
        gpu.hip
            .memcpy_dtoh_at(&mut host, buf, r0 * row_bytes)
            .map_err(err)?;
        dual.update(&host);
        r0 += n;
    }
    Ok(())
}

/// The canonical full persistent state at `position`: metadata, every fixed
/// part, the valid rows of every row stream (K, V, raw index, pooled index per
/// QSA layer). With `roll`, only rows appended since the last call are read.
fn hash_state(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    geo: &Geo,
    position: usize,
    roll: Option<&mut StateRoll>,
) -> Result<StateHash> {
    let n_qsa = bundle.state.qsa.len();
    gpu.hip.device_synchronize().map_err(err)?;
    let SnapshotParts { meta, layout } =
        SessionState::snapshot_parts(bundle, SessionRoute::Ar, position).map_err(err)?;
    if layout.rows.len() != 4 * n_qsa {
        return Err(format!(
            "unexpected session layout: {} row streams for {n_qsa} QSA layers (want 4 per layer: K, V, raw index, pooled index)",
            layout.rows.len()
        ));
    }
    let kv_row = geo.kv_heads * geo.head_dim * 4;
    let meta_hash = hash_bytes(&meta);
    let mut fixed = Vec::with_capacity(layout.fixed.len());
    for part in &layout.fixed {
        let mut host = vec![0u8; part.bytes];
        gpu.hip
            .memcpy_dtoh_at(&mut host, part.buf, part.offset)
            .map_err(err)?;
        fixed.push(hash_bytes(&host));
    }
    let mut roll = roll;
    let mut rows = Vec::with_capacity(layout.rows.len());
    let mut row_counts = Vec::with_capacity(layout.rows.len());
    for (i, stream) in layout.rows.iter().enumerate() {
        let kind = i % 4;
        let expected_row = if kind >= 2 { geo.index_dim * 4 } else { kv_row };
        if stream.row_bytes != expected_row {
            return Err(format!(
                "row stream {i} has {} bytes per row, want {expected_row}",
                stream.row_bytes
            ));
        }
        if kind < 2 && stream.rows != position {
            return Err(format!(
                "row stream {i} holds {} rows at position {position}",
                stream.rows
            ));
        }
        let digest = match roll.as_deref_mut() {
            Some(r) => {
                if stream.rows < r.done[i] {
                    return Err(format!(
                        "row stream {i} shrank from {} to {} rows",
                        r.done[i], stream.rows
                    ));
                }
                feed_rows(
                    gpu,
                    stream.buf,
                    stream.row_bytes,
                    r.done[i],
                    stream.rows,
                    &mut r.duals[i],
                )?;
                r.done[i] = stream.rows;
                r.duals[i].peek()
            }
            None => {
                let mut dual = Dual::new();
                feed_rows(gpu, stream.buf, stream.row_bytes, 0, stream.rows, &mut dual)?;
                dual.finish()
            }
        };
        rows.push(digest);
        row_counts.push(stream.rows);
    }
    let mut items = vec![("meta".to_string(), meta_hash.clone())];
    items.extend(
        fixed
            .iter()
            .enumerate()
            .map(|(i, h)| (format!("fixed{i}"), h.clone())),
    );
    items.extend(
        rows.iter()
            .enumerate()
            .map(|(i, h)| (format!("rows{i}:{}", row_counts[i]), h.clone())),
    );
    Ok(StateHash {
        position,
        meta: meta_hash,
        fixed,
        rows,
        row_counts,
        combined: combine(&items),
    })
}

// ------------------------------------------------------------------ runs

#[derive(Clone)]
struct Step {
    fed: u32,
    logits: Hx,
    next: u32,
}

struct Run {
    records: Vec<TapRec>,
    boundaries: Vec<StateHash>,
    chunk_logits: Vec<Hx>,
    prefill_logits: Hx,
    prefill_argmax: u32,
    final_state: StateHash,
    steps: Vec<Step>,
    after_decode: StateHash,
}

impl Run {
    fn ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.steps.iter().map(|s| s.fed).collect();
        ids.extend(self.steps.last().map(|s| s.next));
        ids
    }
}

fn download_logits(gpu: &Gpu, logits: &GpuTensor) -> Result<Vec<f32>> {
    gpu.hip.device_synchronize().map_err(err)?;
    let values = gpu.download_f32(logits).map_err(err)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err("non-finite logits".into());
    }
    Ok(values)
}

fn decode_steps(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    first: u32,
    steps: usize,
) -> Result<Vec<Step>> {
    let mut out = Vec::with_capacity(steps);
    let mut next = first;
    for i in 0..steps {
        let valid_before: Vec<usize> = bundle.state.qsa.iter().map(|s| s.f16_valid_len()).collect();
        let mirror_before = pooled_marks(bundle);
        bundle.forward_token(gpu, next, logits, None).map_err(err)?;
        let valid_after: Vec<usize> = bundle.state.qsa.iter().map(|s| s.f16_valid_len()).collect();
        let mirror_after = pooled_marks(bundle);
        if valid_before != valid_after {
            return Err(format!(
                "decode step {i} advanced or invalidated the F16 watermark"
            ));
        }
        if mirror_before.iter().zip(&mirror_after).any(|(b, a)| a < b) {
            return Err(format!(
                "decode step {i} rewound or invalidated the pooled BF16 mirror watermark"
            ));
        }
        let values = download_logits(gpu, logits)?;
        let picked = argmax(&values) as u32;
        out.push(Step {
            fed: next,
            logits: hash_f32(&values),
            next: picked,
        });
        next = picked;
    }
    Ok(out)
}

fn geo_of(probe: &SharedProbe) -> Result<Geo> {
    probe
        .lock()
        .unwrap()
        .geo
        .clone()
        .ok_or_else(|| "the QSA tap never fired".to_string())
}

/// R0: cold prefill chunk by chunk through the tapped production forward.
fn run_cold_tapped(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    probe: &SharedProbe,
    logits: &GpuTensor,
    tokens: &[u32],
    decode: usize,
    want_pooled: bool,
    failures: &mut Vec<String>,
) -> Result<Run> {
    bundle.reset(gpu).map_err(err)?;
    begin_phase(probe, "cold", Some(tokens.len()));
    install_tap(bundle, probe, true)?;
    let mut roll = StateRoll::new(4 * bundle.state.qsa.len());
    let (mut boundaries, mut chunk_logits) = (Vec::new(), Vec::new());
    let mut pos = 0;
    let mut rollback = None;
    while pos < tokens.len() {
        let end = (pos + CHUNK).min(tokens.len());
        bundle
            .forward_chunk_final(gpu, &tokens[pos..end], logits, None)
            .map_err(err)?;
        let values = download_logits(gpu, logits)?;
        chunk_logits.push(hash_f32(&values));
        let geo = geo_of(probe)?;
        boundaries.push(hash_state(bundle, gpu, &geo, end, Some(&mut roll))?);
        pooled_check(
            bundle,
            gpu,
            probe,
            &geo,
            &format!("cold chunk ending {end}"),
            want_pooled,
            &MirrorRule::Caught,
        )?;
        pos = end;
        if end == CHUNK {
            rollback = Some(bundle.state.snapshot(gpu).map_err(err)?);
        }
    }
    let n = tokens.len();
    let values = download_logits(gpu, logits)?;
    let geo = geo_of(probe)?;
    let final_state = hash_state(bundle, gpu, &geo, n, None)?;
    if boundaries.last().map(|b| &b.combined) != Some(&final_state.combined) {
        failures.push("cold: rolling chunk-boundary state hash differs from a full re-read of the final state (an earlier row was rewritten)".into());
    }
    let first = argmax(&values) as u32;
    let steps = decode_steps(bundle, gpu, logits, first, decode)?;
    let after_decode = hash_state(bundle, gpu, &geo, n + decode, None)?;
    pooled_check(
        bundle,
        gpu,
        probe,
        &geo,
        "cold decode",
        want_pooled,
        &MirrorRule::Within,
    )?;
    install_tap(bundle, probe, false)?;
    let cold = Run {
        records: take_records(probe),
        boundaries,
        chunk_logits,
        prefill_logits: hash_f32(&values),
        prefill_argmax: first,
        final_state,
        steps,
        after_decode,
    };
    let before: Vec<usize> = bundle.state.qsa.iter().map(|s| s.f16_valid_len()).collect();
    let pooled_before = pooled_marks(bundle);
    if bundle.state.qsa.iter().any(|s| s.f16_arena.is_some())
        && !before.iter().any(|&valid| valid > CHUNK)
    {
        failures.push("rollback never exercised a converted prefix beyond its restored mark; use a longer prompt".into());
    }
    bundle
        .state
        .restore(
            gpu,
            rollback.ok_or("missing first-chunk rollback snapshot")?,
        )
        .map_err(err)?;
    for (slot, layer) in bundle.state.qsa.iter().enumerate() {
        if let Some(arena) = &layer.f16_arena {
            let valid = arena.valid_len.get();
            if valid != before[slot].min(CHUNK) {
                failures.push(format!(
                    "rollback layer {slot}: valid {valid}, expected {}",
                    before[slot].min(CHUNK)
                ));
            }
        }
    }
    let pooled_expected: Vec<usize> = bundle
        .state
        .qsa
        .iter()
        .zip(&pooled_before)
        .map(|(s, &valid)| valid.min(s.pooled_len))
        .collect();
    if want_pooled && pooled_expected == pooled_before {
        failures.push("rollback never clamped a pooled BF16 mirror beyond its restored length; use a longer prompt".into());
    }
    pooled_check(
        bundle,
        gpu,
        probe,
        &geo,
        "rollback restore",
        want_pooled,
        &MirrorRule::Exact(pooled_expected),
    )?;
    begin_phase(probe, "rollback", Some(n));
    install_tap(bundle, probe, true)?;
    let mut pos = CHUNK;
    while pos < n {
        let end = (pos + CHUNK).min(n);
        bundle
            .forward_chunk_final(gpu, &tokens[pos..end], logits, None)
            .map_err(err)?;
        pos = end;
    }
    let values = download_logits(gpu, logits)?;
    pooled_check(
        bundle,
        gpu,
        probe,
        &geo,
        "rollback continuation prefill",
        want_pooled,
        &MirrorRule::Caught,
    )?;
    let final_state = hash_state(bundle, gpu, &geo, n, None)?;
    let steps = decode_steps(bundle, gpu, logits, argmax(&values) as u32, decode)?;
    let after_decode = hash_state(bundle, gpu, &geo, n + decode, None)?;
    pooled_check(
        bundle,
        gpu,
        probe,
        &geo,
        "rollback continuation decode",
        want_pooled,
        &MirrorRule::Within,
    )?;
    install_tap(bundle, probe, false)?;
    let continued = Run {
        records: take_records(probe),
        boundaries: Vec::new(),
        chunk_logits: Vec::new(),
        prefill_logits: hash_f32(&values),
        prefill_argmax: argmax(&values) as u32,
        final_state,
        steps,
        after_decode,
    };
    compare_runs("rollback", &cold, &continued, CHUNK, failures);
    Ok(cold)
}

fn compare_runs(
    name: &str,
    cold: &Run,
    other: &Run,
    from_position: usize,
    failures: &mut Vec<String>,
) {
    let mut cold_by_key: HashMap<(usize, usize, usize), &TapRec> = HashMap::new();
    for rec in &cold.records {
        cold_by_key.insert(rec.key(), rec);
    }
    let mut compared = 0usize;
    for rec in &other.records {
        match cold_by_key.get(&rec.key()) {
            None => failures.push(format!(
                "{name}: tap record {} has no cold counterpart",
                rec.name()
            )),
            Some(c) => {
                compared += 1;
                if c.sel != rec.sel {
                    failures.push(format!(
                        "{name}: selection bytes differ from cold at {}",
                        rec.name()
                    ));
                }
                if c.att != rec.att {
                    failures.push(format!(
                        "{name}: attention output bytes differ from cold at {}",
                        rec.name()
                    ));
                }
            }
        }
    }
    let expected = cold
        .records
        .iter()
        .filter(|r| r.position >= from_position)
        .count();
    if compared != expected || other.records.len() != expected {
        failures.push(format!(
            "{name}: compared {compared} of {} tap records, expected {expected} at positions >= {from_position}",
            other.records.len()
        ));
    }
    if cold.final_state.combined != other.final_state.combined {
        failures.push(format!("{name}: final state differs from cold"));
    }
    if cold.prefill_logits != other.prefill_logits {
        failures.push(format!("{name}: final prefill logits differ from cold"));
    }
    if cold.ids() != other.ids() {
        failures.push(format!(
            "{name}: greedy ids {:?} differ from cold {:?}",
            other.ids(),
            cold.ids()
        ));
    }
    for (i, (a, b)) in cold.steps.iter().zip(&other.steps).enumerate() {
        if a.logits != b.logits {
            failures.push(format!("{name}: decode step {i} logits differ from cold"));
        }
    }
    if cold.after_decode.combined != other.after_decode.combined {
        failures.push(format!("{name}: state after decode differs from cold"));
    }
}

fn env_flag(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Err(_) => Ok(false),
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "off" | "no" => Ok(false),
            "1" | "true" | "on" | "yes" => Ok(true),
            other => Err(format!("{name}={other:?} is not a recognised boolean")),
        },
    }
}

/// The prompt tokens the serve request path produces for `text`: one user
/// message, `enable_thinking=false`, the model chat template, the daemon's
/// prompt normalisation (`batch_render_prompt_tokens` with no system and no
/// history), or a plain encode with `QSA_ORACLE_FRAME=raw`.
fn prompt_tokens(hfq: &HfqFile, tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    match std::env::var("QSA_ORACLE_FRAME")
        .as_deref()
        .unwrap_or("serve")
    {
        "raw" => Ok(tokenizer.encode(text)),
        "serve" => {
            let template = hfq
                .chat_template()
                .ok_or("the model has no chat_template")?;
            let user = hipfire_runtime::tokenizer::maybe_normalize_prompt(text);
            let frame = JinjaChatFrame {
                tokenizer,
                template: &template,
                system: None,
                user: &user,
                enable_thinking: false,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort: None,
            };
            let rendered = frame.render()?;
            Ok(tokenizer.encode(&rendered))
        }
        other => Err(format!("QSA_ORACLE_FRAME={other:?} is not serve or raw")),
    }
}

fn branch_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> Result<Vec<u32>> {
    if let Some(path) = std::env::var_os("QSA_ORACLE_BRANCH_FILE") {
        let text = std::fs::read_to_string(&path).map_err(err)?;
        let branch = tokenizer.encode(&text);
        if branch.len() <= CHUNK || branch.len() > tokens.len() {
            return Err(format!(
                "branch text must be {}..={} tokens, got {}",
                CHUNK + 1,
                tokens.len(),
                branch.len()
            ));
        }
        if branch[0] == tokens[0] {
            return Err("branch text starts with the prompt's first token".into());
        }
        return Ok(branch);
    }
    let mut offset = tokens.len() / 3;
    while offset < tokens.len() && tokens[offset] == tokens[0] {
        offset += 1;
    }
    let len = (tokens.len() - offset).min(2 * CHUNK + 300);
    if len <= CHUNK {
        return Err(format!(
            "prompt too short for an alternate real-text branch ({} tokens)",
            tokens.len()
        ));
    }
    Ok(tokens[offset..offset + len].to_vec())
}

fn emit(out: &mut std::fs::File, value: &Value) -> Result<()> {
    writeln!(out, "{value}").map_err(err)
}

fn run() -> Result<i32> {
    self_test_hashes()?;
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: qwen4_qsa_arena_check MODEL PROMPT_TEXT OUT.jsonl".into());
    }
    if env_flag("QSA_ORACLE_COMPARE")? {
        return compare_processes(&args);
    }
    let model_path = std::path::Path::new(&args[1]);
    let text = std::fs::read_to_string(&args[2]).map_err(err)?;
    let mut out = std::fs::File::create(&args[3]).map_err(err)?;
    let decode: usize = std::env::var("QSA_ORACLE_DECODE")
        .ok()
        .map(|v| v.parse::<usize>().map_err(err))
        .transpose()?
        .unwrap_or(8);
    if decode < 2 {
        return Err("QSA_ORACLE_DECODE must be at least 2".into());
    }
    let want_arena = env_flag("HIPFIRE_QWEN4_QSA_F16_ARENA")?;
    let want_pooled = env_flag("HIPFIRE_QWEN4_QSA_POOLED_BF16")?;

    let mut hfq = HfqFile::open(model_path).map_err(err)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(err)?;
    let tokens = prompt_tokens(&hfq, &tokenizer, &text)?;
    let n = tokens.len();
    if let Some(expect) = std::env::var("QSA_ORACLE_EXPECT_TOKENS").ok() {
        let expect: usize = expect.parse().map_err(err)?;
        if expect != n {
            return Err(format!(
                "prompt is {n} tokens, QSA_ORACLE_EXPECT_TOKENS says {expect}"
            ));
        }
    }
    if n % CHUNK == 0 || n <= CHUNK {
        return Err(format!(
            "prompt of {n} tokens needs at least one full chunk and a partial tail chunk"
        ));
    }
    let branch = branch_tokens(&tokenizer, &tokens)?;
    let mut gpu = Gpu::init().map_err(err)?;
    // Admission is GPU-arch aware: an expert-slab artifact admits only on a
    // supported arch. Initialising the GPU allocates no model memory.
    let receipt = admit_hfqm_artifact_on(&hfq, Some(&gpu.arch)).map_err(err)?;
    if !receipt.config.profile.is_flash_next() {
        return Err("arena oracle requires an admitted real Flash-Next model".into());
    }

    // Device proof.
    let pci = gpu.hip.device_pci_bus_id(gpu.device_id).map_err(err)?;
    if gpu.arch != "gfx1151" && !env_flag("QSA_ORACLE_ALLOW_ARCH")? {
        return Err(format!("device is {} ({pci}), not gfx1151", gpu.arch));
    }
    if let Ok(expect) = std::env::var("QSA_ORACLE_EXPECT_PCI") {
        if !expect.eq_ignore_ascii_case(&pci) {
            return Err(format!(
                "device PCI bus id is {pci}, QSA_ORACLE_EXPECT_PCI says {expect}"
            ));
        }
    }
    let model_bytes = std::fs::metadata(model_path).map_err(err)?.len();
    let device = json!({
        "arch": gpu.arch, "pci": pci, "device_id": gpu.device_id, "uma": gpu.is_uma(),
        "flags": {"HIPFIRE_QWEN4_QSA_F16_ARENA": std::env::var("HIPFIRE_QWEN4_QSA_F16_ARENA").ok(),
            "HIPFIRE_QWEN4_QSA_POOLED_BF16": std::env::var("HIPFIRE_QWEN4_QSA_POOLED_BF16").ok()},
        "visible": {"HIP_VISIBLE_DEVICES": std::env::var("HIP_VISIBLE_DEVICES").ok(),
            "ROCR_VISIBLE_DEVICES": std::env::var("ROCR_VISIBLE_DEVICES").ok()},
        "model": {"path": model_path.display().to_string(), "bytes": model_bytes,
            "md5_claimed": std::env::var("QSA_ORACLE_MODEL_MD5").ok()},
        "prompt": {"path": args[2], "bytes": text.len(), "tokens": n, "sha256_md5": hash_bytes(text.as_bytes()).json()},
        "branch_tokens": branch.len(), "chunk": CHUNK, "decode": decode,
    });
    println!("ARENA_DEVICE {device}");
    emit(&mut out, &json!({"kind":"device","device":device}))?;

    let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
        &gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(err)?;
    let scratch_label = format!("{scratch_mode:?}");
    let use_ranges = gpu.is_uma();
    let domain = CacheDomain::for_model(&hfq, &tokenizer, None, "qwen4", gpu.device_id);
    if use_ranges {
        hfq.drop_mmap();
    }
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
        &gpu,
        &receipt.config,
    )
    .map_err(err)?;
    if state_format.qsa != QsaKvFormat::F32 {
        return Err(format!(
            "QSA K/V format is {}; the arena oracle needs the F32 reference",
            state_format.qsa.name()
        ));
    }
    if state_format.qsa_f16_arena != want_arena {
        return Err("requested arena route is not admitted (requires gfx1151, F32 K/V, head_dim 256 and gathered WMMA enabled)".into());
    }
    if state_format.qsa_pooled_bf16 != want_pooled {
        return Err("requested pooled BF16 mirror route is not admitted (requires gfx1151, F32 K/V, index_dim 128 and the PM BF16 score module)".into());
    }
    if hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()
        .map_err(err)?
        .is_some()
    {
        return Err("host-mapped experts are not supported by the arena oracle".into());
    }
    let backend = match std::env::var("QSA_ORACLE_KV_BACKEND")
        .as_deref()
        .unwrap_or("auto")
    {
        "auto" => hipfire_arch_qwen4::Qwen4KvBackend::automatic(&gpu),
        "legacy" => hipfire_arch_qwen4::Qwen4KvBackend::Legacy,
        "vmm" if hipfire_arch_qwen4::kv_backend::qwen4_vmm_supported(&gpu) => {
            hipfire_arch_qwen4::Qwen4KvBackend::Vmm
        }
        other => return Err(format!("QSA_ORACLE_KV_BACKEND={other:?} unsupported")),
    };
    let n_ctx = n + decode + 64;
    let mesh = DeviceMesh::single().map_err(err)?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &receipt.manifest.weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                &source,
                &receipt.manifest.identity.recipe,
                entry,
            ) {
                return payload;
            }
            if use_ranges || entry.residency.is_external() {
                return source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name));
            }
            let (info, bytes) = source
                .tensor_data(&entry.name)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))?;
            Ok(SourcePayload::Borrowed { info, bytes })
        },
    )
    .map_err(err)?;
    let vocab = receipt.config.vocab_size;
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        n_ctx,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(err)?;
    bundle.set_turn_snapshots(false);
    bundle
        .attach_forward(&mut gpu, n_ctx, scratch_mode)
        .map_err(err)?;
    bundle.attach_session_cache(SessionCache::new(domain, u64::MAX >> 1));
    let chunk_rows = bundle.spec_chunk_rows().ok_or("no forward attached")?;
    if chunk_rows != CHUNK {
        return Err(format!(
            "prefill chunk is {chunk_rows} rows, the oracle runs at {CHUNK}"
        ));
    }
    let n_qsa = bundle.state.qsa.len();
    let logits = gpu.zeros(&[vocab], DType::F32).map_err(err)?;
    let mut failures: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let probe: SharedProbe = Arc::new(Mutex::new(Probe {
        records: Vec::new(),
        checks: Vec::new(),
        geo: None,
        failures: Vec::new(),
        want_arena,
        derived_end: None,
        phase: String::new(),
    }));

    // R0: cold, tapped, chunk by chunk.
    let cold = run_cold_tapped(
        &mut bundle,
        &mut gpu,
        &probe,
        &logits,
        &tokens,
        decode,
        want_pooled,
        &mut failures,
    )?;
    let geo = geo_of(&probe)?;
    if geo.slots != n_qsa {
        failures.push(format!(
            "tap saw {} QSA layers, the state holds {n_qsa}",
            geo.slots
        ));
    }
    let chunks = n.div_ceil(CHUNK);
    if cold.records.len() != (chunks + decode) * n_qsa {
        failures.push(format!(
            "cold: {} tap records, expected {} ({chunks} chunks + {decode} decode steps x {n_qsa} layers)",
            cold.records.len(),
            (chunks + decode) * n_qsa
        ));
    }
    let route = json!({
        "state_format": format!("{state_format:?}"), "qsa_kv": geo.kv_format.name(),
        "index_dtype": format!("{:?}", geo.index_dtype), "f16_arena_present": geo.arena,
        "pooled_bf16_layers": bundle.state.qsa.iter().filter(|s| s.pooled_bf16.is_some()).count(),
        "qsa_layers": n_qsa, "backend": backend.name(), "chunk_rows": chunk_rows,
        "scratch_mode": scratch_label,
        "wmma_gather": rdna_compute::tensor_ops::qsa_gathered_wmma_enabled(&gpu, state_format.qsa),
        "wmma_dense": gpu.arch_caps.has_wmma_w32()
            && std::env::var("HIPFIRE_QWEN4_F16_WMMA").map_or(true, |v| v.trim() != "0"),
        "env": std::env::vars().filter(|(k, _)| k.starts_with("HIPFIRE_") || k.starts_with("QSA_"))
            .collect::<std::collections::BTreeMap<_, _>>(),
    });
    println!("ARENA_ROUTE {route}");
    emit(&mut out, &json!({"kind":"route","route":route}))?;

    // R1: untapped production prefill (captures the session snapshots).
    install_tap(&mut bundle, &probe, false)?;
    bundle
        .prefill_final(&mut gpu, &tokens, 0, &logits)
        .map_err(err)?;
    let values = download_logits(&gpu, &logits)?;
    let control_state = hash_state(&mut bundle, &mut gpu, &geo, n, None)?;
    pooled_check(
        &bundle,
        &gpu,
        &probe,
        &geo,
        "untapped prefill",
        want_pooled,
        &MirrorRule::Caught,
    )?;
    let control_logits = hash_f32(&values);
    bundle.session_commit();
    let control_steps = decode_steps(
        &mut bundle,
        &mut gpu,
        &logits,
        argmax(&values) as u32,
        decode,
    )?;
    pooled_check(
        &bundle,
        &gpu,
        &probe,
        &geo,
        "untapped decode",
        want_pooled,
        &MirrorRule::Within,
    )?;
    if control_state.combined != cold.final_state.combined {
        failures
            .push("untapped production prefill state differs from the tapped cold prefill".into());
    }
    if control_logits != cold.prefill_logits {
        failures
            .push("untapped production prefill logits differ from the tapped cold prefill".into());
    }
    let control_ids: Vec<u32> = control_steps
        .iter()
        .map(|s| s.fed)
        .chain(control_steps.last().map(|s| s.next))
        .collect();
    if control_ids != cold.ids() {
        failures.push(format!(
            "untapped greedy ids {control_ids:?} differ from tapped {:?}",
            cold.ids()
        ));
    }
    for (i, (a, b)) in cold.steps.iter().zip(&control_steps).enumerate() {
        if a.logits != b.logits {
            warnings.push(format!("untapped decode step {i} logits differ from the tapped decode (retained-body route)"));
        }
    }

    // R2: overwrite the live state (and arenas) with an alternate real prefix.
    if bundle.session_plan(&branch, SessionRoute::Ar) != 0 {
        return Err("the alternate branch matches a cached prefix of the prompt".into());
    }
    bundle
        .prefill_final(&mut gpu, &branch, 0, &logits)
        .map_err(err)?;
    let branch_state = hash_state(&mut bundle, &mut gpu, &geo, branch.len(), None)?;
    pooled_check(
        &bundle,
        &gpu,
        &probe,
        &geo,
        "branch prefill",
        want_pooled,
        &MirrorRule::Caught,
    )?;
    bundle.session_commit();
    if branch_state.combined == cold.final_state.combined || branch_state.position == n {
        failures.push("branch prefill did not overwrite the live state".into());
    }

    // R3a: restore only.
    let reused = bundle.session_plan(&tokens, SessionRoute::Ar);
    let want_reused = n / CHUNK * CHUNK;
    if reused != want_reused {
        failures.push(format!("session plan offers {reused} reused tokens, expected the deepest chunk snapshot at {want_reused}"));
        return finish(
            &mut out,
            &cold,
            &control_state,
            &control_steps,
            &failures,
            &warnings,
            &probe,
            &tokens,
        );
    }
    bundle
        .prefill_final(&mut gpu, &tokens[..reused], reused, &logits)
        .map_err(err)?;
    let restored = hash_state(&mut bundle, &mut gpu, &geo, reused, None)?;
    pooled_check(
        &bundle,
        &gpu,
        &probe,
        &geo,
        "prefix restore",
        want_pooled,
        &MirrorRule::Within,
    )?;
    match cold.boundaries.iter().find(|b| b.position == reused) {
        Some(b) if b.combined == restored.combined => {}
        Some(_) => failures.push(format!(
            "state restored at {reused} differs from the cold state at that boundary"
        )),
        None => failures.push(format!("no cold boundary hash at {reused}")),
    }
    if want_arena {
        for (layer, state) in bundle.state.qsa.iter().enumerate() {
            match state.f16_arena.as_ref() {
                None => failures.push(format!(
                    "layer {layer}: derived arena missing after restore"
                )),
                Some(arena) => {
                    let valid = arena.valid_len.get();
                    if valid > reused {
                        failures.push(format!(
                            "layer {layer}: restore left the derived arena at {valid} valid rows beyond the restored {reused} (stale)"
                        ));
                    }
                }
            }
        }
    }
    bundle.session_commit();
    emit(&mut out, &restored.json("restored_boundary"))?;

    // R3: restore, tail chunk, decode, tapped.
    begin_phase(&probe, "restored", Some(n));
    install_tap(&mut bundle, &probe, true)?;
    bundle
        .prefill_final(&mut gpu, &tokens, reused, &logits)
        .map_err(err)?;
    let values = download_logits(&gpu, &logits)?;
    let restored_final = hash_state(&mut bundle, &mut gpu, &geo, n, None)?;
    pooled_check(
        &bundle,
        &gpu,
        &probe,
        &geo,
        "restored tail prefill",
        want_pooled,
        &MirrorRule::Caught,
    )?;
    let first = argmax(&values) as u32;
    let steps = decode_steps(&mut bundle, &mut gpu, &logits, first, decode)?;
    let after_decode = hash_state(&mut bundle, &mut gpu, &geo, n + decode, None)?;
    pooled_check(
        &bundle,
        &gpu,
        &probe,
        &geo,
        "restored decode",
        want_pooled,
        &MirrorRule::Within,
    )?;
    install_tap(&mut bundle, &probe, false)?;
    let warm = Run {
        records: take_records(&probe),
        boundaries: Vec::new(),
        chunk_logits: Vec::new(),
        prefill_logits: hash_f32(&values),
        prefill_argmax: first,
        final_state: restored_final,
        steps,
        after_decode,
    };
    if warm.prefill_argmax != cold.prefill_argmax {
        failures.push("restored argmax differs from cold".into());
    }
    compare_runs("restored", &cold, &warm, reused, &mut failures);
    emit(&mut out, &warm.final_state.json("restored_final"))?;

    for run_record in [&cold, &warm] {
        for rec in &run_record.records {
            emit(
                &mut out,
                &json!({"kind":"tap","run": if std::ptr::eq(run_record, &cold) {"cold"} else {"restored"},
                "position":rec.position,"rows":rec.rows,"slot":rec.slot,
                "selection":rec.sel.json(),"attention":rec.att.json(),
                "selection_live":rec.sel_live,"attention_nonzero":rec.att_nonzero,
                "arena_valid":rec.arena_valid}),
            )?;
        }
    }
    finish(
        &mut out,
        &cold,
        &control_state,
        &control_steps,
        &failures,
        &warnings,
        &probe,
        &tokens,
    )
}

/// Canonical digests, the records behind them, and the result line.
#[allow(clippy::too_many_arguments)]
fn finish(
    out: &mut std::fs::File,
    cold: &Run,
    control_state: &StateHash,
    control_steps: &[Step],
    failures: &[String],
    warnings: &[String],
    probe: &SharedProbe,
    tokens: &[u32],
) -> Result<i32> {
    let mut failures: Vec<String> = failures.to_vec();
    {
        let mut p = probe.lock().unwrap();
        failures.append(&mut p.failures);
        for check in &p.checks {
            emit(out, check)?;
        }
        let mirror_checks: Vec<&Value> = p
            .checks
            .iter()
            .filter(|c| c["kind"] == "pooled_mirror")
            .collect();
        let layers = mirror_checks
            .iter()
            .filter_map(|c| c["layers"].as_u64())
            .max()
            .unwrap_or(0);
        let blocks: u64 = mirror_checks
            .iter()
            .filter_map(|c| c["blocks"].as_u64())
            .sum();
        println!(
            "ARENA_MIRROR {}",
            json!({"present": layers > 0, "layers": layers,
            "checks": mirror_checks.len(), "blocks": blocks})
        );
    }
    for (i, b) in cold.boundaries.iter().enumerate() {
        emit(out, &b.json(&format!("cold_boundary_{i}")))?;
    }
    emit(out, &cold.final_state.json("cold_final"))?;
    emit(out, &cold.after_decode.json("cold_after_decode"))?;
    emit(out, &control_state.json("untapped_final"))?;

    let labelled = |f: &dyn Fn(&TapRec) -> Hx| -> Hx {
        combine(
            &cold
                .records
                .iter()
                .map(|r| (r.name(), f(r)))
                .collect::<Vec<_>>(),
        )
    };
    let selection = labelled(&|r| r.sel.clone());
    let attention = labelled(&|r| r.att.clone());
    let slots = cold
        .records
        .iter()
        .map(|r| r.slot)
        .max()
        .map_or(0, |s| s + 1);
    for slot in 0..slots {
        let digest = combine(
            &cold
                .records
                .iter()
                .filter(|r| r.slot == slot)
                .map(|r| (r.name(), r.att.clone()))
                .collect::<Vec<_>>(),
        );
        println!(
            "ARENA_LAYER_HASH slot={slot} sha256={} md5={}",
            digest.sha, digest.md5
        );
    }
    let mut state_items: Vec<(String, Hx)> = cold
        .boundaries
        .iter()
        .map(|b| (format!("boundary{}", b.position), b.combined.clone()))
        .collect();
    state_items.push(("final".into(), cold.final_state.combined.clone()));
    state_items.push(("after_decode".into(), cold.after_decode.combined.clone()));
    state_items.push(("untapped_final".into(), control_state.combined.clone()));
    let state = combine(&state_items);
    let mut logit_items: Vec<(String, Hx)> = cold
        .chunk_logits
        .iter()
        .enumerate()
        .map(|(i, h)| (format!("chunk{i}"), h.clone()))
        .collect();
    logit_items.push(("prefill".into(), cold.prefill_logits.clone()));
    logit_items.extend(
        cold.steps
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("step{i}"), s.logits.clone())),
    );
    logit_items.extend(
        control_steps
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("untapped{i}"), s.logits.clone())),
    );
    let logits = combine(&logit_items);
    let ids = {
        let mut d = Dual::new();
        for id in cold.ids() {
            d.update(&id.to_le_bytes());
        }
        d.finish()
    };
    let token_hash = {
        let mut d = Dual::new();
        d.update(&(tokens.len() as u64).to_le_bytes());
        for t in tokens {
            d.update(&t.to_le_bytes());
        }
        d.finish()
    };
    let canonical = combine(&[
        ("tokens".into(), token_hash.clone()),
        ("selection".into(), selection.clone()),
        ("attention".into(), attention.clone()),
        ("state".into(), state.clone()),
        ("logits".into(), logits.clone()),
        ("ids".into(), ids.clone()),
    ]);
    for (label, h) in [
        ("tokens", &token_hash),
        ("selection", &selection),
        ("attention", &attention),
        ("state", &state),
        ("logits", &logits),
        ("ids", &ids),
        ("canonical", &canonical),
    ] {
        println!("QSA_ARENA_HASH {label} sha256={} md5={}", h.sha, h.md5);
    }
    emit(
        out,
        &json!({"kind":"summary","tokens":tokens,"ids":cold.ids(),
        "tap_records":cold.records.len(),"failures":failures,"warnings":warnings,
        "canonical":canonical.json()}),
    )?;
    out.flush().map_err(err)?;
    for w in warnings {
        eprintln!("ARENA_WARN {w}");
    }
    if failures.is_empty() {
        println!("ARENA_RESULT PASS");
        Ok(0)
    } else {
        for f in &failures {
            println!("ARENA_FAIL {f}");
        }
        println!(
            "ARENA_RESULT FAIL {} internal comparison failure(s)",
            failures.len()
        );
        Ok(1)
    }
}

/// Controller mode never initializes HIP: each arm resolves load flags in a
/// fresh process. Keep the single-arm stdout protocol compatible with the
/// external driver, but do not report PASS until cross-arm bytes agree.
fn compare_processes(args: &[String]) -> Result<i32> {
    let exe = std::env::current_exe().map_err(err)?;
    // (name, F16 arena flag, pooled BF16 mirror flag)
    let mut arms: Vec<(&str, &str, &str)> = vec![("off", "0", "0"), ("on", "1", "0")];
    if env_flag("QSA_ORACLE_POOLED")? {
        arms.push(("pooled", "0", "1"));
        arms.push(("pooled_arena", "1", "1"));
    }
    let mut outputs = Vec::new();
    for (name, arena, pooled) in &arms {
        let jsonl = format!("{}.{}.jsonl", args[3], name);
        let child = std::process::Command::new(&exe)
            .args([&args[1], &args[2], &jsonl])
            .env("QSA_ORACLE_COMPARE", "0")
            .env("HIPFIRE_QWEN4_QSA_F16_ARENA", arena)
            .env("HIPFIRE_QWEN4_QSA_POOLED_BF16", pooled)
            .output()
            .map_err(err)?;
        let stdout = String::from_utf8(child.stdout).map_err(err)?;
        std::fs::write(format!("{}.{}.stdout.txt", args[3], name), &stdout).map_err(err)?;
        std::fs::write(format!("{}.{}.stderr.txt", args[3], name), child.stderr).map_err(err)?;
        if !child.status.success() || stdout.lines().last() != Some("ARENA_RESULT PASS") {
            return Err(format!(
                "{name} child failed; see {}.{name}.stdout.txt",
                args[3]
            ));
        }
        let hashes: Vec<String> = stdout
            .lines()
            .filter(|l| l.starts_with("QSA_ARENA_HASH ") || l.starts_with("ARENA_LAYER_HASH "))
            .map(str::to_owned)
            .collect();
        let canonical_count = hashes
            .iter()
            .filter(|l| l.starts_with("QSA_ARENA_HASH "))
            .count();
        if canonical_count != 7 || !hashes.iter().any(|l| l.starts_with("ARENA_LAYER_HASH ")) {
            return Err(format!(
                "{name} child has incomplete canonical/per-layer hashes"
            ));
        }
        let mirror: Value = stdout
            .lines()
            .find_map(|l| l.strip_prefix("ARENA_MIRROR "))
            .ok_or_else(|| format!("{name} child printed no ARENA_MIRROR line"))
            .and_then(|l| serde_json::from_str(l).map_err(err))?;
        let want = *pooled == "1";
        let exercised = mirror["checks"].as_u64().unwrap_or(0) > 0
            && mirror["blocks"].as_u64().unwrap_or(0) > 0;
        if mirror["present"].as_bool() != Some(want) || (want && !exercised) {
            return Err(format!(
                "{name} child pooled mirror evidence is wrong for pooled={want}: {mirror}"
            ));
        }
        outputs.push((stdout, hashes));
    }
    for (i, (name, _, _)) in arms.iter().enumerate().skip(1) {
        if outputs[0].1 != outputs[i].1 {
            return Err(format!(
                "off/{name} canonical or per-layer attention hashes differ"
            ));
        }
    }
    let last = outputs.len() - 1;
    for line in outputs[last]
        .0
        .lines()
        .filter(|l| *l != "ARENA_RESULT PASS")
    {
        println!("{line}");
    }
    let names: Vec<&str> = arms.iter().map(|a| a.0).collect();
    std::fs::write(
        &args[3],
        json!({"kind":"comparison","off":"0","on":"1","arms":names,"equal":true}).to_string(),
    )
    .map_err(err)?;
    println!("ARENA_ARMS_EQUAL");
    println!("ARENA_RESULT PASS");
    Ok(0)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            println!("ARENA_RESULT FAIL setup: {e}");
            eprintln!("setup error: {e}");
            std::process::exit(2);
        }
    }
}
