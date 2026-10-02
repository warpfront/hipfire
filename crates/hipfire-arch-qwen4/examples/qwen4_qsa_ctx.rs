// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QSA past-budget parity probe on the real model: GPU vs CPU reference for
//! every QSA layer at 2K..32K context, through the production forward.
//!
//! Per context length `L`: reset, prefill `L - DECODE` tokens of real text
//! (tiled chunks: the batched select and the dense-F16 / grouped attention
//! routes), then teacher-force the last `DECODE` tokens one at a time (the
//! single-row decode routes). A QSA tap reads each layer's scratch and state
//! right after its step and checks, for sampled rows:
//!
//! * pooled keys: the GPU arena against a CPU pool of the GPU's raw index keys
//!   (mean, BF16 RMS norm with the learned weight, HalfSplit RoPE);
//! * selected indices: the GPU row (capacity `budget + compress - 1`, block
//!   order by rank, causal tail, `-1` fill) against a CPU selection over the
//!   GPU's index query and pooled keys, ranked by (score desc, block asc).
//!   Exact equality is required. `fma` scores mirror the kernel's chained
//!   F32 FMA dot (the exact reference); `plain` (separate multiply/add) and
//!   `upstream` (the oracle's `sum(relu(dot)) / sqrt(d)` order) are reported
//!   with their block-set overlap to show how close near-ties sit;
//! * attention output: `ops::qsa_attention` (F32) over the GPU selection, the
//!   GPU's Q/gate row and K/V arenas, against the GPU's gated head output.
//!
//! With QSA_DUMP_DIR set, also retain every pp8192 Q/gate, source cache
//! codes/scales, selection/mirror, eager and per-head exact output. A repeated
//! identical-partition run checks instrumentation's logits/state byte identity.
//! QSA_ARTIFACT_ID should be the recorded model md5. Shared metrics and offline
//! f64/cast-oracle analysis are in `bench_qsa_indexed`; greedy text is secondary.
//!
//! usage: qwen4_qsa_ctx MODEL TEXT OUT.jsonl [CTX=2048,4096,8192,16384,32768]
//!        [STRIDE=256] [DECODE=4]

use hipfire_arch_qwen4::admit_hfqm_artifact;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::gpu_forward::Qwen4QsaTap;
use hipfire_arch_qwen4::ops;
use hipfire_dispatch::pipeline::IndexedAttentionOp;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::tensor_ops::QsaKvFormat;

#[allow(dead_code)]
#[path = "../../rdna-compute/examples/bench_qsa_indexed.rs"]
mod evidence;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

type Result<T, E = String> = std::result::Result<T, E>;

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

/// `len` bytes of `tensor` from byte `offset`.
fn read_bytes(gpu: &Gpu, tensor: &GpuTensor, offset: usize, len: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; len];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &tensor.buf.byte_view(offset, len))
        .map_err(err)?;
    Ok(bytes)
}

/// `len` values of an F32 or BF16 index arena from element `offset`, as F32
/// (the BF16 arenas hold the values the F32 ones do, already BF16-rounded).
fn read_index(gpu: &Gpu, tensor: &GpuTensor, offset: usize, len: usize) -> Result<Vec<f32>> {
    match tensor.dtype {
        DType::F32 => read_f32(gpu, tensor, offset, len),
        DType::BF16 => Ok(read_bytes(gpu, tensor, offset * 2, len * 2)?
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect()),
        other => Err(format!("unexpected index arena dtype {other:?}")),
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let man = (bits & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 => if man == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn e4m3_to_f32(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((code >> 3) & 0xf) as i32;
    let man = (code & 7) as f32;
    if exp == 15 && code & 7 == 7 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * man / 8.0 * 2f32.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// `count` K or V rows from token `start`, dequantized to F32 per the QSA
/// state format (`QsaKvFormat` row layouts).
fn read_kv_rows(
    gpu: &Gpu,
    tensor: &GpuTensor,
    format: QsaKvFormat,
    start: usize,
    count: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let row_bytes = format.kv_row_bytes(kv_heads, head_dim);
    let bytes = read_bytes(gpu, tensor, start * row_bytes, count * row_bytes)?;
    let width = kv_heads * head_dim;
    let mut out = Vec::with_capacity(count * width);
    for row in bytes.chunks_exact(row_bytes) {
        match format {
            QsaKvFormat::F32 => out.extend(row.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
            QsaKvFormat::Fp8 => {
                let (codes, scales) = row.split_at(width);
                for head in 0..kv_heads {
                    let scale = f16_bits_to_f32(u16::from_le_bytes([scales[2 * head], scales[2 * head + 1]]));
                    out.extend(codes[head * head_dim..(head + 1) * head_dim].iter().map(|&c| e4m3_to_f32(c) * scale));
                }
            }
        }
    }
    Ok(out)
}

fn read_f32(gpu: &Gpu, tensor: &GpuTensor, offset: usize, len: usize) -> Result<Vec<f32>> {
    gpu.download_f32(&tensor.sub_offset(offset, len)).map_err(err)
}

/// `len` elements of `tensor` from element `offset`, as raw bytes.
fn read_raw(gpu: &Gpu, tensor: &GpuTensor, offset: usize, len: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; len * tensor.dtype.size()];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &tensor.sub_offset(offset, len).buf)
        .map_err(err)?;
    Ok(bytes)
}

fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    f32::from_bits((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000)
}

#[derive(Clone, Copy, PartialEq)]
enum Score {
    /// Kernel order: per head a chained F32 FMA dot, `max(dot / sqrt(d), 0)`,
    /// heads summed in order.
    Fma,
    /// Same order with a separate multiply and add.
    Plain,
    /// Upstream oracle order: `sum(max(dot, 0)) / sqrt(d)`.
    Upstream,
}

fn block_scores(query: &[f32], pooled: &[f32], blocks: usize, heads: usize, dim: usize, mode: Score) -> Vec<f32> {
    let scale = (dim as f32).sqrt();
    (0..blocks)
        .map(|block| {
            let key = &pooled[block * dim..(block + 1) * dim];
            let mut score = 0.0f32;
            for head in 0..heads {
                let q = &query[head * dim..(head + 1) * dim];
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot = match mode {
                        Score::Fma => q[d].mul_add(key[d], dot),
                        _ => dot + q[d] * key[d],
                    };
                }
                score += match mode {
                    Score::Upstream => dot.max(0.0),
                    _ => (dot / scale).max(0.0),
                };
            }
            if mode == Score::Upstream {
                score / scale
            } else {
                score
            }
        })
        .collect()
}

/// Blocks by (score desc, block asc): the documented tie rule.
fn rank(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// The selected row the kernel contract defines: ranked blocks' tokens, then
/// the causal tail, `-1` in every other slot.
fn selection_row(order: &[usize], budget_blocks: usize, compress: usize, visible: usize, capacity: usize) -> Vec<i32> {
    let blocks = visible / compress;
    let chosen = budget_blocks.min(blocks);
    let mut row = vec![-1i32; capacity];
    for (slot, &block) in order.iter().take(chosen).enumerate() {
        for offset in 0..compress {
            let index = slot * compress + offset;
            let token = block * compress + offset;
            if index < capacity && token < visible {
                row[index] = token as i32;
            }
        }
    }
    let mut offset = chosen * compress;
    for token in blocks * compress..visible {
        if offset >= capacity {
            break;
        }
        row[offset] = token as i32;
        offset += 1;
    }
    row
}

fn chosen_set(row: &[i32], compress: usize) -> std::collections::BTreeSet<i32> {
    row.iter()
        .filter(|&&t| t >= 0)
        .map(|&t| t / compress as i32)
        .collect()
}

struct Probe {
    phase: &'static str,
    ctx: usize,
    stride: usize,
    rows: Vec<Value>,
    /// Per slot: the learned index-key norm weight, read once.
    k_norm: Vec<Option<Vec<f32>>>,
    /// Per slot: pooled blocks already checked this session.
    pooled_checked: Vec<usize>,
    wmma_dense: bool,
    /// `HIPFIRE_QWEN4_QSA_WMMA_GATHER` not `0` on the arch / state format it serves
    /// (gfx1151 F32, gfx1201 fp8): prefill chunks past the dense route run
    /// the gathered F16 WMMA kernel instead of hg4 (a route label only).
    wmma_gather: bool,
    /// Teacher-forced decode rows after the prefill (the prefill's final row
    /// is `ctx - decode`).
    decode: usize,
    dump_dir: Option<std::path::PathBuf>,
    artifact_id: String,
    expected_state: Vec<(String, String)>,
    validation_cursor: Option<usize>,
}

impl Probe {
    fn sampled(&self, rows: usize, start: usize, r: usize) -> bool {
        let visible = start + r + 1;
        rows == 1
            || visible + self.decode == self.ctx
            || visible % self.stride == 0
            || (visible > 2048 && visible <= 2052)
    }

    fn tap(&mut self, gpu: &mut Gpu, slot: usize, op: &IndexedAttentionOp<'_>) -> Result<()> {
        let waited = Instant::now();
        gpu.hip.device_synchronize().map_err(err)?;
        let gpu_s = waited.elapsed().as_secs_f64();
        let started = Instant::now();
        let result = self.tap_inner(gpu, slot, op);
        let cpu_s = started.elapsed().as_secs_f64();
        if gpu_s > 5.0 || cpu_s > 5.0 {
            eprintln!(
                "[slow tap] ctx {} slot {slot} rows {} position {} gpu wait {gpu_s:.1}s tap {cpu_s:.1}s",
                self.ctx, op.rows, op.state.position
            );
        }
        result
    }

    fn projection_evidence(&mut self, gpu: &mut Gpu, slot: usize, op: &IndexedAttentionOp<'_>) -> Result<()> {
        let Some(root) = self.dump_dir.as_ref() else { return Ok(()); };
        let key = format!("ctx-{}-{}-layer-{slot}-chunk-{}", self.ctx, self.phase, op.state.position);
        let index_stride = (op.index_heads + op.index_kv_heads) * op.index_dim;
        let tensors = [
            ("raw-qgate.f32", op.qgate_scratch, op.rows * op.heads * op.head_dim * 2),
            ("raw-index.f32", op.index_scratch, op.rows * index_stride),
            ("raw-k.f32", op.k_scratch, op.rows * op.kv_heads * op.head_dim),
            ("raw-v.f32", op.v_scratch, op.rows * op.kv_heads * op.head_dim),
            ("index-q-norm.source", op.indexer_q_norm, op.index_dim),
            ("index-k-norm.source", op.indexer_k_norm, op.index_dim),
            ("q-norm.source", op.q_norm, op.head_dim),
            ("k-norm.source", op.k_norm, op.head_dim),
            ("keys.source", op.state.full_keys, op.state.full_keys.numel()),
            ("values.source", op.state.full_values, op.state.full_values.numel()),
            ("raw-index.source", op.state.raw_index_keys, op.state.full_capacity * op.index_dim),
            ("pooled.source", op.state.pooled_keys, op.state.pooled_capacity * op.index_dim),
            ("selected-mirror.i32", op.state.selected_indices, op.state.selected_capacity * 4),
        ];
        let mut blobs = Vec::with_capacity(tensors.len());
        let mut files = serde_json::Map::new();
        for (name,tensor,elements) in tensors {
            let bytes = read_raw(gpu,tensor,0,elements)?;
            files.insert(name.into(),json!({"bytes":bytes.len(),"sha256":evidence::sha256(&bytes),
                "dtype":format!("{:?}",tensor.dtype),"elements":elements}));
            blobs.push((name,bytes));
        }
        let digest = evidence::sha256(serde_json::to_vec(&files).map_err(err)?.as_slice());
        let event = format!("pre-prologue:{key}");
        if let Some(cursor) = self.validation_cursor.as_mut() {
            if self.expected_state.get(*cursor) != Some(&(event,digest)) {
                return Err(format!("instrumented/uninstrumented raw projections differ at {key}"));
            }
            *cursor += 1;
            return Ok(());
        }
        self.expected_state.push((event,digest));
        let dir = root.join(&key).join("pre-prologue");
        std::fs::create_dir_all(&dir).map_err(err)?;
        for (name,bytes) in blobs { std::fs::write(dir.join(name),bytes).map_err(err)?; }
        let header = json!({"schema":"qsa-pre-prologue-v1","artifact":self.artifact_id,
            "phase":self.phase,"ctx":self.ctx,"layer":slot,"rows":op.rows,"position":op.state.position,
            "heads":op.heads,"kv_heads":op.kv_heads,"dim":op.head_dim,
            "index_heads":op.index_heads,"index_kv_heads":op.index_kv_heads,"index_dim":op.index_dim,
            "index_stride":index_stride,"compress":op.compress,"cache_format":op.state.format.name(),
            "full_capacity":op.state.full_capacity,"pooled_capacity":op.state.pooled_capacity,
            "selected_capacity":op.state.selected_capacity,"files":files,
            "capture_boundary":"after exact production project_weights; before any QSA norm/RoPE/cache append"});
        std::fs::write(dir.join("snapshot.json"),serde_json::to_vec_pretty(&header).map_err(err)?).map_err(err)?;
        Ok(())
    }

    fn full_evidence(&mut self, gpu: &mut Gpu, slot: usize, op: &IndexedAttentionOp<'_>) -> Result<()> {
        let Some(root) = self.dump_dir.as_ref() else { return Ok(()); };
        let g = evidence::Geometry {
            rows: op.rows, position_start: op.state.position, heads: op.heads,
            kv_heads: op.kv_heads, dim: op.head_dim, budget_blocks: op.budget / op.compress,
            compress: op.compress, capacity: op.state.selected_capacity,
            full_capacity: op.state.full_capacity, fp8: op.state.format == QsaKvFormat::Fp8,
        };
        let key = format!("ctx-{}-{}-layer-{slot}-chunk-{}", self.ctx, self.phase, g.position_start);
        let q = read_f32(gpu, op.qgate_scratch, 0, g.rows * g.heads * g.dim * 2)?;
        let eager = read_f32(gpu, op.qsa_output, 0, g.rows * g.heads * g.dim)?;
        let cache_bytes = g.full_capacity * op.state.format.kv_row_bytes(g.kv_heads, g.dim);
        let keys = read_bytes(gpu, op.state.full_keys, 0, cache_bytes)?;
        let values = read_bytes(gpu, op.state.full_values, 0, cache_bytes)?;
        let selected = read_bytes(gpu, op.selected_scratch, 0, g.rows * g.capacity * 4)?;
        let mirror = read_bytes(gpu, op.state.selected_indices, 0, g.capacity * 4)?;
        let raw = read_raw(gpu, op.state.raw_index_keys, 0, g.full_capacity * op.index_dim)?;
        let pooled = read_raw(gpu, op.state.pooled_keys, 0, op.state.pooled_capacity * op.index_dim)?;
        let index_stride = (op.index_heads + op.index_kv_heads) * op.index_dim;
        let index_projection = read_raw(gpu, op.index_scratch, 0, g.rows * index_stride)?;
        let index_norm = read_raw(gpu, op.indexer_k_norm, 0, op.index_dim)?;
        let qbytes = evidence::f32_bytes(&q);
        let eager_bytes = evidence::f32_bytes(&eager);
        let blobs = [
            ("qgate.f32", qbytes.as_slice()), ("keys.source", keys.as_slice()),
            ("values.source", values.as_slice()), ("selected.i32", selected.as_slice()),
            ("selected-mirror.i32", mirror.as_slice()), ("raw-index.source", raw.as_slice()),
            ("pooled.source", pooled.as_slice()), ("eager.f32", eager_bytes.as_slice()),
            ("index-projection.f32", index_projection.as_slice()),
            ("indexer-k-norm.source", index_norm.as_slice()),
        ];
        let digest = evidence::sha256(serde_json::to_string(
            &blobs.iter().map(|(name, b)| (*name, evidence::sha256(b))).collect::<Vec<_>>()
        ).map_err(err)?.as_bytes());
        if let Some(cursor) = self.validation_cursor.as_mut() {
            if self.expected_state.get(*cursor) != Some(&(key.clone(), digest.clone())) {
                return Err(format!("instrumented/uninstrumented state differs at {key} (event {cursor})"));
            }
            *cursor += 1;
            return Ok(());
        }
        self.expected_state.push((key.clone(), digest));
        let exact = gpu.zeros(&[eager.len()], DType::F32).map_err(err)?;
        let launch = rdna_compute::tensor_ops::indexed_attention_attention_batch_exact(
            gpu, &rdna_compute::tensor_ops::IndexedAttentionAttentionBatch {
                q_with_gate: op.qgate_scratch, full_keys: op.state.full_keys,
                full_values: op.state.full_values, selected: op.selected_scratch, output: &exact,
                rows: g.rows, position_start: g.position_start, n_heads: g.heads,
                n_kv_heads: g.kv_heads, head_dim: g.dim, budget_blocks: g.budget_blocks,
                compress: g.compress, capacity: g.capacity, full_capacity: g.full_capacity,
                format: op.state.format, shape_selected: g.capacity,
            },
        ).map_err(err);
        let reference = launch.and_then(|()| gpu.download_f32(&exact).map_err(err));
        gpu.free_tensor(exact).map_err(err)?;
        let reference = reference?;
        // The exact launch writes only its private output allocation. Check all
        // model-owned sources and outputs; do not restore or hide any mutation.
        for (name, original, tensor, bytes) in [
            ("qgate", qbytes.as_slice(), op.qgate_scratch, qbytes.len()),
            ("eager", eager_bytes.as_slice(), op.qsa_output, eager_bytes.len()),
            ("keys", keys.as_slice(), op.state.full_keys, keys.len()),
            ("values", values.as_slice(), op.state.full_values, values.len()),
            ("selected", selected.as_slice(), op.selected_scratch, selected.len()),
            ("mirror", mirror.as_slice(), op.state.selected_indices, mirror.len()),
            ("raw", raw.as_slice(), op.state.raw_index_keys, raw.len()),
            ("pooled", pooled.as_slice(), op.state.pooled_keys, pooled.len()),
            ("index-projection", index_projection.as_slice(), op.index_scratch, index_projection.len()),
            ("indexer-k-norm", index_norm.as_slice(), op.indexer_k_norm, index_norm.len()),
        ] {
            if read_bytes(gpu, tensor, 0, bytes)? != original {
                return Err(format!("exact-reference instrumentation mutated {key}/{name}"));
            }
        }
        let dir = root.join(&key);
        std::fs::create_dir_all(&dir).map_err(err)?;
        let stats = evidence::metrics(&reference, &eager, g.heads, g.dim, Some(&dir.join("G-vs-R.elements")))?;
        let exact_bytes = evidence::f32_bytes(&reference);
        let mut all_blobs = blobs.to_vec();
        all_blobs.push(("exact.f32", &exact_bytes));
        let snapshot = evidence::dump_snapshot(
            &dir, json!({"artifact": self.artifact_id, "layer": slot, "chunk": g.position_start,
                "phase": self.phase, "ctx": self.ctx, "arch": gpu.arch,
                "source_format": op.state.format.name(), "instrumentation_state_differing_bytes": 0,
                "selector":{"heads":op.index_heads,"kv_heads":op.index_kv_heads,"dim":op.index_dim,
                    "projection_stride":index_stride,"projection_dtype":format!("{:?}",op.index_scratch.dtype),
                    "raw_dtype":format!("{:?}",op.state.raw_index_keys.dtype),
                    "pooled_dtype":format!("{:?}",op.state.pooled_keys.dtype),
                    "norm_dtype":format!("{:?}",op.indexer_k_norm.dtype),"pooled_capacity":op.state.pooled_capacity}}),
            &g, &all_blobs, stats.clone(),
        )?;
        self.rows.push(json!({"kind":"full_evidence","slot":slot,"chunk":g.position_start,
            "phase":self.phase,"ctx":self.ctx,"snapshot":snapshot,"metrics":stats}));
        Ok(())
    }

    fn tap_inner(&mut self, gpu: &mut Gpu, slot: usize, op: &IndexedAttentionOp<'_>) -> Result<()> {
        let n = op.rows;
        let start = op.state.position;
        let end = start + n;
        let compress = op.compress;
        let dim = op.index_dim;
        let budget_blocks = op.budget / compress;
        let capacity = op.state.selected_capacity;
        let index_q_width = op.index_heads * dim;
        let index_width = index_q_width + op.index_kv_heads * dim;
        let q_width = op.heads * op.head_dim;
        let kv_width = op.kv_heads * op.head_dim;
        let blocks_end = end / compress;
        self.full_evidence(gpu, slot, op)?;
        if self.validation_cursor.is_some() { return Ok(()); }
        if self.k_norm.len() <= slot {
            self.k_norm.resize(slot + 1, None);
            self.pooled_checked.resize(slot + 1, 0);
        }
        if self.k_norm[slot].is_none() {
            let bytes = read_raw(gpu, op.indexer_k_norm, 0, dim)?;
            self.k_norm[slot] = Some(
                bytes
                    .chunks_exact(2)
                    .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                    .collect(),
            );
        }

        // Pooled arena vs a CPU pool of the GPU's raw keys (new blocks only).
        let first = self.pooled_checked[slot].min(blocks_end);
        if first < blocks_end {
            let pooled_new = read_index(gpu, op.state.pooled_keys, first * dim, (blocks_end - first) * dim)?;
            let raw = read_index(
                gpu,
                op.state.raw_index_keys,
                first * compress * dim,
                (blocks_end - first) * compress * dim,
            )?;
            let norm = self.k_norm[slot].as_ref().expect("norm cached");
            let (mut max_abs, mut max_rel, mut exact) = (0.0f32, 0.0f32, 0usize);
            for block in first..blocks_end {
                let base = (block - first) * compress * dim;
                let mut mean = vec![0.0f32; dim];
                for row in 0..compress {
                    for c in 0..dim {
                        mean[c] += raw[base + row * dim + c] / compress as f32;
                    }
                }
                let mean: Vec<f32> = mean.into_iter().map(bf16_round).collect();
                let sum: f32 = mean.iter().map(|&m| bf16_round(m * m)).sum();
                let inv = 1.0 / (sum / dim as f32 + 1.0e-6).sqrt();
                let mut value: Vec<f32> = (0..dim)
                    .map(|c| bf16_round(mean[c] * inv * (1.0 + norm[c])))
                    .collect();
                ops::rope_prefix_halfsplit(&mut value, block * compress, dim.min(64), 10_000_000.0)
                    .map_err(err)?;
                for c in 0..dim {
                    let cpu = bf16_round(value[c]);
                    let gpu_value = pooled_new[(block - first) * dim + c];
                    let diff = (cpu - gpu_value).abs();
                    max_abs = max_abs.max(diff);
                    max_rel = max_rel.max(diff / cpu.abs().max(1.0e-3));
                    exact += usize::from(cpu.to_bits() == gpu_value.to_bits());
                }
            }
            self.rows.push(json!({
                "kind": "pool", "phase": self.phase, "ctx": self.ctx, "slot": slot,
                "blocks": [first, blocks_end], "max_abs": max_abs, "max_rel": max_rel,
                "exact_frac": exact as f64 / ((blocks_end - first) * dim) as f64,
            }));
            self.pooled_checked[slot] = blocks_end;
        }

        let sampled: Vec<usize> = (0..n).filter(|&r| self.sampled(n, start, r)).collect();
        if sampled.is_empty() {
            return Ok(());
        }
        let pooled = read_index(gpu, op.state.pooled_keys, 0, blocks_end.max(1) * dim)?;
        let route = if n == 1 {
            "per_head"
        } else if self.wmma_dense && n >= 512 && blocks_end <= budget_blocks && end <= capacity {
            "dense_f16"
        } else if self.wmma_gather && n >= 512 {
            "gathered_f16"
        } else {
            "hg4"
        };
        for r in sampled {
            let position = start + r;
            let visible = position + 1;
            let blocks = visible / compress;
            let query = read_f32(gpu, op.index_scratch, r * index_width, index_q_width)?;
            let gpu_row: Vec<i32> = read_raw(gpu, op.selected_scratch, r * capacity * 4, capacity * 4)?
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let mut verdict = serde_json::Map::new();
            let mut fma_order = Vec::new();
            let mut fma_scores = Vec::new();
            for (label, mode) in [("fma", Score::Fma), ("plain", Score::Plain), ("upstream", Score::Upstream)] {
                let scores = block_scores(&query, &pooled, blocks, op.index_heads, dim, mode);
                let order = rank(&scores);
                let row = selection_row(&order, budget_blocks, compress, visible, capacity);
                let gpu_set = chosen_set(&gpu_row, compress);
                let cpu_set = chosen_set(&row, compress);
                let overlap = gpu_set.intersection(&cpu_set).count();
                let first_diff = row.iter().zip(&gpu_row).position(|(a, b)| a != b);
                verdict.insert(
                    label.to_string(),
                    json!({
                        "exact": first_diff.is_none(),
                        "first_diff_slot": first_diff,
                        "set_overlap": overlap,
                        "set_size": cpu_set.len(),
                    }),
                );
                if mode == Score::Fma {
                    fma_order = order;
                    fma_scores = scores;
                }
            }
            // Boundary structure of the exact (fma) ranking.
            let zero_blocks = fma_scores.iter().filter(|&&s| s == 0.0).count();
            let (boundary_tie, boundary_gap) = if blocks > budget_blocks {
                let last_in = fma_scores[fma_order[budget_blocks - 1]];
                let first_out = fma_scores[fma_order[budget_blocks]];
                (last_in == first_out, (last_in - first_out) as f64)
            } else {
                (false, f64::NAN)
            };

            // Attention over the GPU selection.
            let qgate = read_f32(gpu, op.qgate_scratch, r * 2 * q_width, 2 * q_width)?;
            let gpu_out = read_f32(gpu, op.qsa_output, r * q_width, q_width)?;
            let selected: Vec<usize> = gpu_row.iter().filter(|&&t| t >= 0).map(|&t| t as usize).collect();
            // Gather only the selected K/V rows (contiguous runs), in order.
            let (mut keys, mut values) = (Vec::with_capacity(selected.len() * kv_width), Vec::with_capacity(selected.len() * kv_width));
            let mut i = 0;
            while i < selected.len() {
                let mut j = i + 1;
                while j < selected.len() && selected[j] == selected[j - 1] + 1 {
                    j += 1;
                }
                let (start_token, count) = (selected[i], j - i);
                if start_token + count > visible {
                    return Err(format!("selected token {} past visible {visible}", start_token + count - 1));
                }
                keys.extend(read_kv_rows(gpu, op.state.full_keys, op.state.format, start_token, count, op.kv_heads, op.head_dim)?);
                values.extend(read_kv_rows(gpu, op.state.full_values, op.state.format, start_token, count, op.kv_heads, op.head_dim)?);
                i = j;
            }
            let compact: Vec<usize> = (0..selected.len()).collect();
            let mut cpu_out = vec![0.0f32; q_width];
            ops::qsa_attention(
                &qgate,
                &keys,
                &values,
                &compact,
                op.heads,
                op.kv_heads,
                op.head_dim,
                &mut cpu_out,
            )
            .map_err(err)?;
            let (mut max_abs, mut ref_max, mut dot, mut na, mut nb, mut err2) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in cpu_out.iter().zip(&gpu_out) {
                let (a, b) = (a as f64, b as f64);
                max_abs = max_abs.max((a - b).abs());
                ref_max = ref_max.max(a.abs());
                dot += a * b;
                na += a * a;
                nb += b * b;
                err2 += (a - b) * (a - b);
            }
            self.rows.push(json!({
                "kind": "row", "phase": self.phase, "ctx": self.ctx, "slot": slot,
                "position": position, "blocks": blocks, "past_budget": blocks > budget_blocks,
                "selected_len": selected.len(), "route": route, "select": Value::Object(verdict),
                "zero_score_blocks": zero_blocks, "boundary_tie": boundary_tie,
                "boundary_gap": boundary_gap,
                "attn_max_abs": max_abs, "attn_ref_max": ref_max,
                "attn_rel": max_abs / ref_max.max(1.0e-30),
                "attn_cos": dot / (na.sqrt() * nb.sqrt()).max(1.0e-30),
                "attn_rel_l2": (err2 / na.max(1.0e-30)).sqrt(),
            }));
        }
        Ok(())
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
        .0
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let max = v.iter().copied().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = v.iter().map(|&x| (x as f64 - max).exp()).sum();
    v.iter().map(|&x| x as f64 - max - sum.ln()).collect()
}

fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(Ordering::Equal));
    idx.truncate(k);
    idx
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: qwen4_qsa_ctx MODEL TEXT OUT.jsonl [CTX,..] [STRIDE] [DECODE]".into());
    }
    let path = std::path::Path::new(&args[1]);
    let text = std::fs::read_to_string(&args[2]).map_err(err)?;
    let out_path = &args[3];
    let contexts: Vec<usize> = args
        .get(4)
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![2048, 4096, 8192, 16384, 32768]);
    let stride: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(256);
    let decode: usize = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(4);
    // Room for the 24 greedy tokens after the largest context, within the
    // model's native 262,144 positions.
    let n_ctx = (contexts.iter().copied().max().unwrap_or(2048) + 64).min(262_144);
    if contexts.iter().any(|&c| c + 24 > n_ctx) {
        return Err(format!("a context leaves no room for 24 greedy tokens in {n_ctx}"));
    }

    let mut hfq = HfqFile::open(path).map_err(err)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(err)?;
    let receipt = admit_hfqm_artifact(&hfq).map_err(err)?;
    let mut gpu = Gpu::init().map_err(err)?;
    eprintln!("gpu {} uma={} n_ctx={n_ctx}", gpu.arch, gpu.is_uma());
    let use_ranges = gpu.is_uma();
    if use_ranges {
        hfq.drop_mmap();
    }
    // The shipped state formats for this device (HIPFIRE_KV_MODE /
    // HIPFIRE_STATE_QUANT; `bf16` + `fp32` is the exact reference arm).
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
        &gpu,
        &receipt.config,
    )
    .map_err(err)?;
    let qsa_format = state_format.qsa;
    eprintln!("state format qsa={} gdn={}", qsa_format.name(), state_format.gdn.name());
    #[allow(unused_mut)]
    let mut weights = receipt.manifest.weights.clone();
    use hipfire_arch_qwen4::expert_residency as residency;
    if let Some(policy) = residency::expert_vram_layers_from_env().map_err(err)? {
        if use_ranges { return Err("host-mapped experts are only for discrete GPUs".into()); }
        let vram_layers = match policy {
            residency::ExpertVramLayers::Layers(n) => n,
            residency::ExpertVramLayers::Auto => {
                let (free, _) = gpu.hip.get_vram_info().map_err(err)?;
                let (non_expert, per_layer) = residency::resident_split(&weights, |entry|
                    hfq.tensor_data(&entry.name).map(|(_, bytes)| bytes.len() as u64)).map_err(err)?;
                let gather = rdna_compute::tensor_ops::qsa_gathered_wmma_enabled(&gpu, qsa_format)
                    .then(|| rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(receipt.config.num_key_value_heads, n_ctx))
                    .flatten().map(|bytes| bytes as u64);
                let reserve = residency::auto_vram_reserve(&receipt.config, n_ctx,
                    hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(&gpu.arch, n_ctx), qsa_format, None, gather).map_err(err)?;
                if (free as u64) < non_expert.saturating_add(reserve) {
                    return Err("not enough free VRAM for non-expert weights plus auto reserve".into());
                }
                eprintln!("auto placement: {free} free bytes, {non_expert} non-expert bytes, {reserve} reserved bytes, gather={gather:?}");
                residency::auto_vram_layers(free as u64, non_expert, per_layer, receipt.config.num_hidden_layers, reserve)
            }
        };
        let moved = residency::place_routed_experts(&mut weights, vram_layers);
        eprintln!("routed experts: layers 0..{vram_layers} in VRAM, {moved} tensors host-mapped");
    }
    let mesh = DeviceMesh::single().map_err(err)?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            // The carrier's rule: ranges on UMA and for external rows,
            // borrowed mmap bytes for resident tensors on a discrete card.
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
        transaction,
        &receipt.placements,
        &mut gpu,
        n_ctx,
        receipt.ple,
        state_format,
    )
    .map_err(err)?;
    bundle.attach_forward(&mut gpu, n_ctx).map_err(err)?;
    let logits = gpu.zeros(&[vocab], DType::F32).map_err(err)?;
    let tokens = tokenizer.encode(&text);
    eprintln!("text: {} tokens", tokens.len());

    let probe = Arc::new(Mutex::new(Probe {
        phase: "prefill",
        ctx: 0,
        stride,
        rows: Vec::new(),
        k_norm: Vec::new(),
        pooled_checked: Vec::new(),
        decode,
        dump_dir: std::env::var_os("QSA_DUMP_DIR").map(std::path::PathBuf::from),
        artifact_id: std::env::var("QSA_ARTIFACT_ID").unwrap_or_else(|_| args[1].clone()),
        expected_state: Vec::new(),
        validation_cursor: None,
        wmma_dense: gpu.arch_caps.has_wmma_w32()
            && std::env::var("HIPFIRE_QWEN4_F16_WMMA").map_or(true, |v| v.trim() != "0"),
        wmma_gather: rdna_compute::tensor_ops::qsa_gathered_wmma_enabled(&gpu, qsa_format),
    }));
    let mut out = std::fs::File::create(out_path).map_err(err)?;
    let run_start = Instant::now();

    for &ctx in &contexts {
        if tokens.len() < ctx {
            return Err(format!("text has {} tokens, need {ctx}", tokens.len()));
        }
        let prompt = &tokens[..ctx];
        // Tapped: prefill then teacher-forced decode.
        bundle.reset(&mut gpu).map_err(err)?;
        {
            let mut p = probe.lock().unwrap();
            p.phase = "prefill";
            p.ctx = ctx;
            p.expected_state.clear();
            p.validation_cursor = None;
            p.pooled_checked.iter_mut().for_each(|c| *c = 0);
        }
        let tap_probe = Arc::clone(&probe);
        let tap: Qwen4QsaTap = Box::new(move |gpu, slot, op| tap_probe.lock().unwrap().tap(gpu, slot, op));
        bundle.set_qsa_tap(Some(tap)).map_err(err)?;
        if probe.lock().map_err(err)?.dump_dir.is_some() {
            let projection_probe = Arc::clone(&probe);
            bundle.set_qsa_projection_hook(Some(Box::new(move |gpu, slot, op|
                projection_probe.lock().map_err(err)?.projection_evidence(gpu, slot, op)))).map_err(err)?;
        }
        let phase = |name: &str| eprintln!("[phase] ctx {ctx} {name} at {:.1}s", run_start.elapsed().as_secs_f64());
        phase("tapped prefill");
        let t = Instant::now();
        bundle
            .forward_chunk_final(&mut gpu, &prompt[..ctx - decode], &logits, None)
            .map_err(err)?;
        let tapped_prefill_s = t.elapsed().as_secs_f64();
        probe.lock().unwrap().phase = "decode";
        phase("tapped decode");
        for &token in &prompt[ctx - decode..] {
            bundle.forward_token(&mut gpu, token, &logits, None).map_err(err)?;
        }
        gpu.hip.device_synchronize().map_err(err)?;
        let decode_logits = gpu.download_f32(&logits).map_err(err)?;
        bundle.set_qsa_tap(None).map_err(err)?;

        let has_dump = probe.lock().map_err(err)?.dump_dir.is_some();
        let instrumentation = if has_dump {
            // Same prefill/decode partition, with a read-only observational tap.
            // Comparing all-prefill instead would conflate instrumentation with
            // the known different chunk/reduction schedule.
            bundle.reset(&mut gpu).map_err(err)?;
            {
                let mut p = probe.lock().map_err(err)?;
                p.phase = "prefill";
                p.validation_cursor = Some(0);
            }
            let check_probe = Arc::clone(&probe);
            bundle.set_qsa_tap(Some(Box::new(move |gpu, slot, op|
                check_probe.lock().map_err(err)?.tap(gpu, slot, op)))).map_err(err)?;
            bundle.forward_chunk_final(&mut gpu, &prompt[..ctx - decode], &logits, None).map_err(err)?;
            probe.lock().map_err(err)?.phase = "decode";
            for &token in &prompt[ctx - decode..] {
                bundle.forward_token(&mut gpu, token, &logits, None).map_err(err)?;
            }
            let control = gpu.download_f32(&logits).map_err(err)?;
            if evidence::f32_bytes(&control) != evidence::f32_bytes(&decode_logits) {
                return Err(format!("instrumentation changes final logits at ctx {ctx}"));
            }
            bundle.set_qsa_tap(None).map_err(err)?;
            let mut p = probe.lock().map_err(err)?;
            if p.validation_cursor != Some(p.expected_state.len()) {
                return Err("instrumentation validation missed state events".into());
            }
            let events = p.expected_state.len();
            p.validation_cursor = None;
            p.expected_state.clear();
            json!({"logits_differing_bytes":0,"state_differing_bytes":0,"events":events})
        } else { Value::Null };

        // All-prefill: retain the actual pp8192 activation geometry (including
        // the final 512-row chunk), in addition to the split decode sanity run.
        bundle.reset(&mut gpu).map_err(err)?;
        gpu.hip.device_synchronize().map_err(err)?;
        phase("untapped prefill");
        if has_dump {
            probe.lock().map_err(err)?.phase = "all_prefill";
            let full_probe = Arc::clone(&probe);
            bundle.set_qsa_tap(Some(Box::new(move |gpu, slot, op|
                full_probe.lock().map_err(err)?.full_evidence(gpu, slot, op)))).map_err(err)?;
        }
        let t = Instant::now();
        bundle.forward_chunk_final(&mut gpu, prompt, &logits, None).map_err(err)?;
        gpu.hip.device_synchronize().map_err(err)?;
        let prefill_s = t.elapsed().as_secs_f64();
        let prefill_logits = gpu.download_f32(&logits).map_err(err)?;
        bundle.set_qsa_tap(None).map_err(err)?;
        bundle.set_qsa_projection_hook(None).map_err(err)?;
        // Final-row logits of the all-prefill route, for cross-arm KL (e.g.
        // compressed QSA state against the F32 reference).
        let bytes: Vec<u8> = prefill_logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{out_path}.logits-{ctx}.f32"), bytes).map_err(err)?;
        let max_diff = decode_logits
            .iter()
            .zip(&prefill_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let (lp, lq) = (log_softmax(&prefill_logits), log_softmax(&decode_logits));
        let kl: f64 = lp.iter().zip(&lq).map(|(&a, &b)| a.exp() * (a - b)).sum();
        let top_p = top_k(&prefill_logits, 5);
        let top_d = top_k(&decode_logits, 5);
        let top5_overlap = top_p.iter().filter(|t| top_d.contains(t)).count();

        // Greedy continuation from the all-prefill state.
        let mut generated = Vec::new();
        let mut next = argmax(&prefill_logits) as u32;
        phase("greedy");
        let t = Instant::now();
        for _ in 0..24 {
            generated.push(next);
            bundle.forward_token(&mut gpu, next, &logits, None).map_err(err)?;
            next = argmax(&gpu.download_f32(&logits).map_err(err)?) as u32;
        }
        let decode_tok_s = 24.0 / t.elapsed().as_secs_f64();
        let summary = json!({
            "kind": "logits", "ctx": ctx, "decode_rows": decode,
            "qsa_format": qsa_format.name(),
            "gdn_format": state_format.gdn.name(),
            "max_abs_diff": max_diff, "kl_prefill_decode": kl,
            "argmax_prefill": argmax(&prefill_logits), "argmax_decode": argmax(&decode_logits),
            "top5_overlap": top5_overlap,
            "instrumentation": instrumentation,
            "prefill_s": prefill_s, "prefill_tok_s": ctx as f64 / prefill_s,
            "tapped_prefill_s": tapped_prefill_s, "greedy_tok_s": decode_tok_s,
            "greedy_text": tokenizer.decode(&generated),
        });
        eprintln!("{summary}");
        let rows = std::mem::take(&mut probe.lock().unwrap().rows);
        for row in rows.iter().chain(std::iter::once(&summary)) {
            writeln!(out, "{row}").map_err(err)?;
        }
        out.flush().map_err(err)?;
        let checked = rows.iter().filter(|r| r["kind"] == "row").count();
        let fma_exact = rows
            .iter()
            .filter(|r| r["kind"] == "row" && r["select"]["fma"]["exact"] == true)
            .count();
        let worst_rel = rows
            .iter()
            .filter_map(|r| r["attn_rel"].as_f64())
            .fold(0.0f64, f64::max);
        eprintln!("ctx {ctx}: {checked} rows checked, fma-exact {fma_exact}, worst attn rel {worst_rel:.3e}");
    }
    Ok(())
}
