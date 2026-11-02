// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Developer-only logical observation and comparison support for the sealed-MoE
//! baseline.  This module is deliberately behind `moe-oracle`; production
//! builds do not compile any of the collector, host readback, or serialization
//! code and the forward hooks are removed by `cfg` at their call sites.

use super::config::Qwen35Config;
use super::forward::Qwen35Scratch;
use super::weights::{DeltaNetState, LayerWeights, MoeFfnWeights, Qwen35Weights};
use hipfire_runtime::llama::KvCache;
use rdna_compute::{Gpu, GpuTensor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

pub const SCHEMA_VERSION: u32 = 1;
pub const SCHEMA_FIELDS: [&str; 16] = [
    "router_scores",
    "topk_ids",
    "topk_weights",
    "gate",
    "up",
    "activated",
    "down",
    "routed",
    "shared",
    "residual",
    "logits",
    "kv",
    "recurrent",
    "convolution",
    "position",
    "token_ids",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayRef {
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

impl ArrayRef {
    fn empty() -> Self {
        Self {
            file: String::new(),
            dtype: "empty".to_string(),
            shape: Vec::new(),
            bytes: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub router_scores: ArrayRef,
    pub topk_ids: ArrayRef,
    pub topk_weights: ArrayRef,
    pub gate: ArrayRef,
    pub up: ArrayRef,
    pub activated: ArrayRef,
    pub down: ArrayRef,
    pub routed: ArrayRef,
    pub shared: ArrayRef,
    pub residual: ArrayRef,
    pub logits: ArrayRef,
    pub kv: ArrayRef,
    pub recurrent: ArrayRef,
    pub convolution: ArrayRef,
    pub position: ArrayRef,
    pub token_ids: ArrayRef,
}

impl Snapshot {
    fn empty() -> Self {
        Self {
            router_scores: ArrayRef::empty(),
            topk_ids: ArrayRef::empty(),
            topk_weights: ArrayRef::empty(),
            gate: ArrayRef::empty(),
            up: ArrayRef::empty(),
            activated: ArrayRef::empty(),
            down: ArrayRef::empty(),
            routed: ArrayRef::empty(),
            shared: ArrayRef::empty(),
            residual: ArrayRef::empty(),
            logits: ArrayRef::empty(),
            kv: ArrayRef::empty(),
            recurrent: ArrayRef::empty(),
            convolution: ArrayRef::empty(),
            position: ArrayRef::empty(),
            token_ids: ArrayRef::empty(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub sequence: String,
    pub kind: String,
    pub layer: usize,
    #[serde(rename = "position_index")]
    pub position: usize,
    #[serde(flatten)]
    pub snapshot: Snapshot,
}

#[derive(Debug)]
struct Pending {
    sequence: String,
    kind: String,
    layer: usize,
    positions: Vec<usize>,
    hidden: usize,
    k: usize,
    router_scores: Option<Vec<f32>>,
    before: Vec<f32>,
}

#[derive(Debug)]
struct Collector {
    dir: PathBuf,
    arrays_dir: PathBuf,
    identity: Value,
    sequence: String,
    decode_position: usize,
    prefill_start: usize,
    event: u64,
    observations: Vec<Observation>,
    pending: HashMap<(String, String, usize, usize), Pending>,
    ownership: Vec<Value>,
}

static ACTIVE: LazyLock<Mutex<Option<Collector>>> = LazyLock::new(|| Mutex::new(None));

fn active() -> &'static Mutex<Option<Collector>> {
    &ACTIVE
}

fn err<E: std::fmt::Display>(context: &str, error: E) -> String {
    format!("moe-oracle: {context}: {error}")
}

fn with_collector<T>(f: impl FnOnce(&mut Collector) -> Result<T, String>) -> Result<T, String> {
    let mut guard = active()
        .lock()
        .map_err(|_| "collector mutex poisoned".to_string())?;
    let collector = guard
        .as_mut()
        .ok_or_else(|| "collector is not active".to_string())?;
    f(collector)
}

pub fn begin(dir: &Path, identity: Value) -> Result<(), String> {
    if !identity.is_object() {
        return Err("report identity must be a JSON object".to_string());
    }
    if dir.exists() {
        if !dir.is_dir() {
            return Err(format!("report path is not a directory: {}", dir.display()));
        }
    } else {
        fs::create_dir_all(dir).map_err(|e| err("create report directory", e))?;
    }
    let arrays_dir = dir.join("arrays");
    fs::create_dir_all(&arrays_dir).map_err(|e| err("create arrays directory", e))?;
    let mut guard = active()
        .lock()
        .map_err(|_| "collector mutex poisoned".to_string())?;
    if guard.is_some() {
        return Err("another report collector is active".to_string());
    }
    *guard = Some(Collector {
        dir: dir.to_path_buf(),
        arrays_dir,
        identity,
        sequence: "primary".to_string(),
        decode_position: 0,
        prefill_start: 0,
        event: 0,
        observations: Vec::new(),
        pending: HashMap::new(),
        ownership: Vec::new(),
    });
    Ok(())
}

pub fn set_sequence(label: &str) -> Result<(), String> {
    if label.trim().is_empty() || label.contains('/') || label.contains('\\') {
        return Err(format!("invalid sequence label {label:?}"));
    }
    with_collector(|collector| {
        collector.sequence = label.to_string();
        Ok(())
    })
}

pub fn set_decode_position(position: usize) -> Result<(), String> {
    let mut guard = active()
        .lock()
        .map_err(|_| "collector mutex poisoned".to_string())?;
    if let Some(collector) = guard.as_mut() {
        collector.decode_position = position;
    }
    Ok(())
}

pub fn set_prefill_start(position: usize) -> Result<(), String> {
    let mut guard = active()
        .lock()
        .map_err(|_| "collector mutex poisoned".to_string())?;
    if let Some(collector) = guard.as_mut() {
        collector.prefill_start = position;
    }
    Ok(())
}

pub fn decode_position() -> Result<usize, String> {
    with_collector(|collector| Ok(collector.decode_position))
}

pub fn prefill_start() -> Result<usize, String> {
    with_collector(|collector| Ok(collector.prefill_start))
}

fn safe_label(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl Collector {
    fn write_array<T: Copy>(
        &mut self,
        field: &str,
        dtype: &str,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<ArrayRef, String> {
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, n| acc.checked_mul(*n))
            .ok_or_else(|| format!("{field}: shape element count overflow"))?;
        let element_size = match dtype {
            "f32" | "u32" => 4,
            "u8" => 1,
            _ => return Err(format!("{field}: unsupported array dtype {dtype}")),
        };
        let expected_bytes = expected
            .checked_mul(element_size)
            .ok_or_else(|| format!("{field}: byte count overflow"))?;
        if expected_bytes != bytes.len() {
            return Err(format!(
                "{field}: shape expects {expected_bytes} bytes, received {}",
                bytes.len()
            ));
        }
        let file = format!(
            "arrays/{:016}_{}_{}.bin",
            self.event,
            safe_label(field),
            dtype
        );
        self.event = self
            .event
            .checked_add(1)
            .ok_or_else(|| "array event counter overflow".to_string())?;
        let path = self.dir.join(&file);
        let mut out = File::create(&path).map_err(|e| err("create array", e))?;
        out.write_all(bytes).map_err(|e| err("write array", e))?;
        out.sync_all().map_err(|e| err("sync array", e))?;
        Ok(ArrayRef {
            file,
            dtype: dtype.to_string(),
            shape,
            bytes: bytes.len(),
        })
    }

    fn f32_array(
        &mut self,
        field: &str,
        values: &[f32],
        shape: Vec<usize>,
    ) -> Result<ArrayRef, String> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("{field}: nonfinite value in logical observation"));
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        self.write_array::<f32>(field, "f32", shape, &bytes)
    }

    fn u32_array(
        &mut self,
        field: &str,
        values: &[u32],
        shape: Vec<usize>,
    ) -> Result<ArrayRef, String> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.write_array::<u32>(field, "u32", shape, &bytes)
    }

    fn raw_array(&mut self, field: &str, bytes: &[u8]) -> Result<ArrayRef, String> {
        self.write_array::<u8>(field, "u8", vec![bytes.len()], bytes)
    }

    fn make_moe_snapshot(
        &mut self,
        field_prefix: &str,
        layer: usize,
        position: usize,
        hidden: usize,
        k: usize,
        router_scores: &[f32],
        topk_ids: &[u32],
        topk_weights: &[f32],
        gate: &[f32],
        up: &[f32],
        activated: &[f32],
        down: &[f32],
        before: &[f32],
        after: &[f32],
    ) -> Result<Snapshot, String> {
        if before.len() != hidden || after.len() != hidden {
            return Err(format!(
                "layer {layer} position {position}: residual length mismatch"
            ));
        }
        if topk_ids.len() != k || topk_weights.len() != k || down.len() != k * hidden {
            return Err(format!(
                "layer {layer} position {position}: top-k/down shape mismatch"
            ));
        }
        let mut routed = vec![0.0f32; hidden];
        for slot in 0..k {
            for j in 0..hidden {
                routed[j] += topk_weights[slot] * down[slot * hidden + j];
            }
        }
        let mut shared = vec![0.0f32; hidden];
        let mut residual = vec![0.0f32; hidden];
        for j in 0..hidden {
            residual[j] = after[j];
            shared[j] = after[j] - before[j] - routed[j];
        }
        let p = format!("{field_prefix}_l{layer}_p{position}");
        Ok(Snapshot {
            router_scores: self.f32_array(
                &format!("{p}_router_scores"),
                router_scores,
                vec![router_scores.len()],
            )?,
            topk_ids: self.u32_array(&format!("{p}_topk_ids"), topk_ids, vec![k])?,
            topk_weights: self.f32_array(&format!("{p}_topk_weights"), topk_weights, vec![k])?,
            gate: self.f32_array(&format!("{p}_gate"), gate, vec![gate.len()])?,
            up: self.f32_array(&format!("{p}_up"), up, vec![up.len()])?,
            activated: self.f32_array(
                &format!("{p}_activated"),
                activated,
                vec![activated.len()],
            )?,
            down: self.f32_array(&format!("{p}_down"), down, vec![k, hidden])?,
            routed: self.f32_array(&format!("{p}_routed"), &routed, vec![hidden])?,
            shared: self.f32_array(&format!("{p}_shared"), &shared, vec![hidden])?,
            residual: self.f32_array(&format!("{p}_residual"), &residual, vec![hidden])?,
            ..Snapshot::empty()
        })
    }
}

fn download_f32(gpu: &Gpu, tensor: &GpuTensor, field: &str) -> Result<Vec<f32>, String> {
    gpu.download_f32(tensor)
        .map_err(|e| err(&format!("download {field}"), e))
}

fn download_bytes(
    gpu: &Gpu,
    tensor: &GpuTensor,
    field: &str,
    limit: Option<usize>,
) -> Result<Vec<u8>, String> {
    gpu.bind_thread()
        .map_err(|e| err("bind GPU for observation", e))?;
    let bytes = limit
        .unwrap_or_else(|| tensor.buf.size())
        .min(tensor.buf.size());
    let mut out = vec![0u8; bytes];
    gpu.hip
        .memcpy_dtoh(&mut out, &tensor.buf)
        .map_err(|e| err(&format!("download {field}"), e))?;
    Ok(out)
}

fn prefix(values: Vec<f32>, count: usize, field: &str) -> Result<Vec<f32>, String> {
    if values.len() < count {
        return Err(format!(
            "{field}: buffer has {} values, need {count}",
            values.len()
        ));
    }
    Ok(values[..count].to_vec())
}

fn decode_key(sequence: &str, layer: usize, position: usize) -> (String, String, usize, usize) {
    (sequence.to_string(), "decode".to_string(), layer, position)
}

fn prefill_key(sequence: &str, layer: usize, start: usize) -> (String, String, usize, usize) {
    (sequence.to_string(), "prefill".to_string(), layer, start)
}

/// Capture the pre-MoE residual. Router scores are captured in `decode_after`
/// because the existing dispatch owns the router GEMV and top-k producer.
pub fn decode_before(
    gpu: &Gpu,
    ffn: &MoeFfnWeights,
    config: &Qwen35Config,
    x_residual: &GpuTensor,
    position: usize,
) -> Result<(), String> {
    with_collector(|collector| {
        let before = prefix(
            download_f32(gpu, x_residual, "decode residual before")?,
            config.dim,
            "decode residual before",
        )?;
        let key = decode_key(&collector.sequence, ffn.layer_idx as usize, position);
        if collector.pending.contains_key(&key) {
            return Err(format!(
                "duplicate decode observation for layer {} position {position}",
                ffn.layer_idx
            ));
        }
        collector.pending.insert(
            key,
            Pending {
                sequence: collector.sequence.clone(),
                kind: "decode".to_string(),
                layer: ffn.layer_idx as usize,
                positions: vec![position],
                hidden: config.dim,
                k: config.num_experts_per_tok,
                router_scores: None,
                before,
            },
        );
        Ok(())
    })
}

pub fn decode_after(
    gpu: &Gpu,
    ffn: &MoeFfnWeights,
    config: &Qwen35Config,
    router_scores: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    activated: &GpuTensor,
    down: &GpuTensor,
    x_residual: &GpuTensor,
    position: usize,
) -> Result<(), String> {
    with_collector(|collector| {
        let key = decode_key(&collector.sequence, ffn.layer_idx as usize, position);
        let pending = collector.pending.remove(&key).ok_or_else(|| {
            format!(
                "missing decode pre-observation for layer {} position {position}",
                ffn.layer_idx
            )
        })?;
        let scores = download_f32(gpu, router_scores, "decode router scores")?;
        let ids_raw = download_f32(gpu, topk_indices, "decode top-k ids")?;
        let weights = download_f32(gpu, topk_weights, "decode top-k weights")?;
        let ids: Vec<u32> = ids_raw
            .iter()
            .take(config.num_experts_per_tok)
            .map(|v| v.to_bits())
            .collect();
        if ids.len() != config.num_experts_per_tok || weights.len() < config.num_experts_per_tok {
            return Err(format!(
                "layer {} position {position}: incomplete decode top-k",
                ffn.layer_idx
            ));
        }
        if ids.iter().any(|id| (*id as usize) >= config.num_experts) {
            return Err(format!(
                "layer {} position {position}: top-k expert id out of range",
                ffn.layer_idx
            ));
        }
        let weights = weights[..config.num_experts_per_tok].to_vec();
        let gate = prefix(
            download_f32(gpu, gate, "decode gate")?,
            config.num_experts_per_tok * config.moe_intermediate_size,
            "decode gate",
        )?;
        let up = prefix(
            download_f32(gpu, up, "decode up")?,
            config.num_experts_per_tok * config.moe_intermediate_size,
            "decode up",
        )?;
        let activated = prefix(
            download_f32(gpu, activated, "decode activated")?,
            config.num_experts_per_tok * config.moe_intermediate_size,
            "decode activated",
        )?;
        let down = prefix(
            download_f32(gpu, down, "decode down")?,
            config.num_experts_per_tok * config.dim,
            "decode down",
        )?;
        let after = prefix(
            download_f32(gpu, x_residual, "decode residual after")?,
            config.dim,
            "decode residual after",
        )?;
        let snapshot = collector.make_moe_snapshot(
            "decode",
            pending.layer,
            position,
            pending.hidden,
            pending.k,
            &scores,
            &ids,
            &weights,
            &gate,
            &up,
            &activated,
            &down,
            &pending.before,
            &after,
        )?;
        collector.observations.push(Observation {
            sequence: pending.sequence,
            kind: pending.kind,
            layer: pending.layer,
            position,
            snapshot,
        });
        Ok(())
    })
}

/// Capture a batched prefill residual before the shared expert mutates it.
pub fn prefill_before(
    gpu: &Gpu,
    ffn: &MoeFfnWeights,
    config: &Qwen35Config,
    x_batch: &GpuTensor,
    n: usize,
    start_position: usize,
) -> Result<(), String> {
    with_collector(|collector| {
        if n == 0 {
            return Err("prefill batch must be non-empty".to_string());
        }
        let before = prefix(
            download_f32(gpu, x_batch, "prefill residual before")?,
            n.checked_mul(config.dim)
                .ok_or_else(|| "prefill residual size overflow".to_string())?,
            "prefill residual before",
        )?;
        let key = prefill_key(&collector.sequence, ffn.layer_idx as usize, start_position);
        if collector.pending.contains_key(&key) {
            return Err(format!(
                "duplicate prefill observation for layer {} start {start_position}",
                ffn.layer_idx
            ));
        }
        collector.pending.insert(
            key,
            Pending {
                sequence: collector.sequence.clone(),
                kind: "prefill".to_string(),
                layer: ffn.layer_idx as usize,
                positions: (0..n).map(|i| start_position + i).collect(),
                hidden: config.dim,
                k: config.num_experts_per_tok,
                router_scores: None,
                before,
            },
        );
        Ok(())
    })
}

pub fn prefill_after(
    gpu: &Gpu,
    ffn: &MoeFfnWeights,
    config: &Qwen35Config,
    x_batch: &GpuTensor,
    router_scores: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    activated: &GpuTensor,
    down: &GpuTensor,
    n: usize,
    start_position: usize,
) -> Result<(), String> {
    with_collector(|collector| {
        let key = prefill_key(&collector.sequence, ffn.layer_idx as usize, start_position);
        let pending = collector.pending.remove(&key).ok_or_else(|| {
            format!(
                "missing prefill pre-observation for layer {} start {start_position}",
                ffn.layer_idx
            )
        })?;
        let hidden = pending.hidden;
        let k = pending.k;
        let scores = prefix(
            download_f32(gpu, router_scores, "prefill router scores")?,
            n.checked_mul(config.num_experts)
                .ok_or_else(|| "prefill router size overflow".to_string())?,
            "prefill router scores",
        )?;
        let ids_raw = prefix(
            download_f32(gpu, topk_indices, "prefill top-k ids")?,
            n.checked_mul(k)
                .ok_or_else(|| "prefill top-k size overflow".to_string())?,
            "prefill top-k ids",
        )?;
        let weights = prefix(
            download_f32(gpu, topk_weights, "prefill top-k weights")?,
            n.checked_mul(k)
                .ok_or_else(|| "prefill top-k size overflow".to_string())?,
            "prefill top-k weights",
        )?;
        let ids: Vec<u32> = ids_raw.iter().map(|v| v.to_bits()).collect();
        if ids.iter().any(|id| (*id as usize) >= config.num_experts) {
            return Err(format!(
                "layer {} start {start_position}: top-k expert id out of range",
                ffn.layer_idx
            ));
        }
        let gate = prefix(
            download_f32(gpu, gate, "prefill gate")?,
            n * k * config.moe_intermediate_size,
            "prefill gate",
        )?;
        let up = prefix(
            download_f32(gpu, up, "prefill up")?,
            n * k * config.moe_intermediate_size,
            "prefill up",
        )?;
        let activated = prefix(
            download_f32(gpu, activated, "prefill activated")?,
            n * k * config.moe_intermediate_size,
            "prefill activated",
        )?;
        let down = prefix(
            download_f32(gpu, down, "prefill down")?,
            n * k * hidden,
            "prefill down",
        )?;
        let after = prefix(
            download_f32(gpu, x_batch, "prefill residual after")?,
            n * hidden,
            "prefill residual after",
        )?;
        for row in 0..n {
            let row_scores = &scores[row * config.num_experts..(row + 1) * config.num_experts];
            let row_ids = &ids[row * k..(row + 1) * k];
            let row_weights = &weights[row * k..(row + 1) * k];
            let row_gate = &gate[row * k * config.moe_intermediate_size
                ..(row + 1) * k * config.moe_intermediate_size];
            let row_up = &up[row * k * config.moe_intermediate_size
                ..(row + 1) * k * config.moe_intermediate_size];
            let row_activated = &activated[row * k * config.moe_intermediate_size
                ..(row + 1) * k * config.moe_intermediate_size];
            let row_down = &down[row * k * hidden..(row + 1) * k * hidden];
            let row_before = &pending.before[row * hidden..(row + 1) * hidden];
            let row_after = &after[row * hidden..(row + 1) * hidden];
            let snapshot = collector.make_moe_snapshot(
                "prefill",
                pending.layer,
                start_position + row,
                hidden,
                k,
                row_scores,
                row_ids,
                row_weights,
                row_gate,
                row_up,
                row_activated,
                row_down,
                row_before,
                row_after,
            )?;
            collector.observations.push(Observation {
                sequence: pending.sequence.clone(),
                kind: pending.kind.clone(),
                layer: pending.layer,
                position: start_position + row,
                snapshot,
            });
        }
        Ok(())
    })
}

fn kv_logical_bytes(cache: &KvCache, tensor: &GpuTensor, positions: usize, is_key: bool) -> usize {
    let q8_per_position = cache.n_kv_heads * (cache.head_dim / 32) * 34;
    let value_per_position = if cache.v_mode.bits() == 8 {
        q8_per_position
    } else {
        let bits = cache.v_mode.bits().max(0) as usize;
        cache.n_kv_heads * (4 + (cache.head_dim * bits) / 8)
    };
    let per_position = if cache.quant_q8 {
        if is_key {
            q8_per_position
        } else {
            value_per_position
        }
    } else if cache.quant_hfq4 {
        cache.n_kv_heads * (8 + cache.head_dim / 2)
    } else if cache.quant_asym4 {
        if is_key {
            cache.n_kv_heads * (4 + cache.head_dim / 2)
        } else {
            value_per_position
        }
    } else if cache.quant_asym3 {
        if is_key {
            cache.n_kv_heads * (4 + (cache.head_dim * 3) / 8)
        } else {
            value_per_position
        }
    } else if cache.quant_asym2 {
        if is_key {
            cache.n_kv_heads * (4 + cache.head_dim / 4)
        } else {
            value_per_position
        }
    } else if cache.quant_bf16 {
        cache.kv_dim * 2
    } else {
        cache.kv_dim * 4
    };
    tensor
        .buf
        .size()
        .min(per_position.saturating_mul(positions.min(cache.physical_cap)))
}

fn capture_state(
    collector: &mut Collector,
    gpu: &Gpu,
    kv: &KvCache,
    recurrent: &DeltaNetState,
    position: usize,
) -> Result<(ArrayRef, ArrayRef, ArrayRef), String> {
    let positions = position
        .checked_add(1)
        .ok_or_else(|| "state position overflow".to_string())?;
    let mut kv_bytes = Vec::new();
    for tensor in &kv.k_gpu {
        let n = kv_logical_bytes(kv, tensor, positions, true);
        kv_bytes.extend_from_slice(&download_bytes(gpu, tensor, "kv", Some(n))?);
    }
    for tensor in &kv.v_gpu {
        let n = kv_logical_bytes(kv, tensor, positions, false);
        kv_bytes.extend_from_slice(&download_bytes(gpu, tensor, "kv", Some(n))?);
    }
    let mut recurrent_bytes = Vec::new();
    for tensor in recurrent
        .s_matrices
        .iter()
        .chain(recurrent.s_scales.iter())
        .chain(recurrent.s_ef_residual.iter())
    {
        recurrent_bytes.extend_from_slice(&download_bytes(gpu, tensor, "recurrent", None)?);
    }
    let mut convolution_bytes = Vec::new();
    for tensor in &recurrent.conv_states {
        convolution_bytes.extend_from_slice(&download_bytes(gpu, tensor, "convolution", None)?);
    }
    Ok((
        collector.raw_array("kv", &kv_bytes)?,
        collector.raw_array("recurrent", &recurrent_bytes)?,
        collector.raw_array("convolution", &convolution_bytes)?,
    ))
}

/// Record the final logical outputs and state for one consumed token.  This is
/// deliberately called by the example after each public forward operation so
/// logits/state remain outside the arithmetic hooks.
pub fn record_step(
    gpu: &Gpu,
    scratch: &Qwen35Scratch,
    kv: &KvCache,
    recurrent: &DeltaNetState,
    token: u32,
    position: usize,
) -> Result<(), String> {
    with_collector(|collector| {
        let logits = collector.f32_array(
            &format!("step_p{position}_logits"),
            &download_f32(gpu, &scratch.logits, "logits")?,
            vec![scratch.logits.numel()],
        )?;
        let (kv_ref, recurrent_ref, convolution_ref) =
            capture_state(collector, gpu, kv, recurrent, position)?;
        let position_ref = collector.u32_array("position", &[position as u32], vec![1])?;
        let token_ref = collector.u32_array("token_ids", &[token], vec![1])?;
        let mut snapshot = Snapshot::empty();
        snapshot.logits = logits;
        snapshot.kv = kv_ref;
        snapshot.recurrent = recurrent_ref;
        snapshot.convolution = convolution_ref;
        snapshot.position = position_ref;
        snapshot.token_ids = token_ref;
        collector.observations.push(Observation {
            sequence: collector.sequence.clone(),
            kind: "step".to_string(),
            layer: usize::MAX,
            position,
            snapshot,
        });
        Ok(())
    })
}

pub fn record_ownership(gpu: &Gpu, weights: &Qwen35Weights) -> Result<(), String> {
    with_collector(|collector| {
        if !collector.ownership.is_empty() {
            return Ok(());
        }
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let ffn = match layer {
                LayerWeights::DeltaNetMoe(layer) => &layer.ffn,
                LayerWeights::FullAttnMoe(layer) => &layer.ffn,
                _ => continue,
            };
            let plan = &ffn.expert_execution_plan;
            let dimensions = plan.dimensions();
            for record in plan.experts() {
                let expert = ffn.experts.get(record.global_expert_id).ok_or_else(|| {
                    format!(
                        "layer {layer_idx} sealed expert {} has no live weight owner",
                        record.global_expert_id
                    )
                })?;
                let source = |binding: &hipfire_runtime::sealed_moe::ExpertProjectionBinding| {
                    json!({
                        "name": binding.source_name,
                        "fingerprint": binding.fingerprint,
                        "logical_shape": binding.logical_shape,
                        "dtype": format!("{:?}", binding.dtype),
                        "projection_bytes": binding.projection_bytes,
                        "encoded_bytes": binding.encoded_bytes,
                        "row_stride": binding.row_stride,
                        "alignment": binding.alignment,
                        "quant_tag": binding.quant_tag,
                        "basis": binding.basis,
                        "sidecars": binding.sidecar_source_names,
                        "alias_owner": binding.alias_owner,
                        "alias_byte_offset": binding.alias_byte_offset,
                    })
                };
                let physical_device = plan
                    .rank_ownership()
                    .get(record.owner_rank)
                    .map(|ownership| ownership.physical_device)
                    .ok_or_else(|| {
                        format!(
                            "layer {layer_idx} expert {} owner rank {} is absent",
                            record.global_expert_id, record.owner_rank
                        )
                    })?;
                collector.ownership.push(json!({
                    "global_expert_id": record.global_expert_id,
                    "layer": layer_idx,
                    "owner_rank": record.owner_rank,
                    "local_slot": record.local_slot,
                    "logical_projection_dimensions": {
                        "global_gate_rows": dimensions.global_gate_rows,
                        "global_up_rows": dimensions.global_up_rows,
                        "global_down_cols": dimensions.global_down_cols,
                        "local_gate_rows": dimensions.local_gate_rows,
                        "local_up_rows": dimensions.local_up_rows,
                        "local_down_cols": dimensions.local_down_cols,
                        "local_down_rows": dimensions.local_down_rows,
                    },
                    "physical_device": physical_device,
                    "observed_device": gpu.device_id,
                    "manifest_generation": plan.mesh_epoch().as_u64(),
                    "source_fingerprint": plan.source_fingerprint(),
                    "source": {
                        "gate": source(&record.gate),
                        "up": source(&record.up),
                        "down": source(&record.down),
                        "sidecars": record.sidecars.iter().map(source).collect::<Vec<_>>(),
                        "live_gate_up_bytes": expert.gate_up.buf.buf.size(),
                        "live_down_bytes": expert.down.buf.buf.size(),
                    },
                    "collective": plan.collective_rows().iter().map(|row| json!({
                        "name": row.name,
                        "layer": row.layer,
                        "kind": format!("{:?}", row.hint),
                    })).collect::<Vec<_>>(),
                }));
            }
        }
        Ok(())
    })
}

pub fn finish(lifecycle: Value) -> Result<PathBuf, String> {
    let mut guard = active()
        .lock()
        .map_err(|_| "collector mutex poisoned".to_string())?;
    let collector = guard
        .take()
        .ok_or_else(|| "collector is not active".to_string())?;
    if !collector.pending.is_empty() {
        return Err(format!(
            "{} incomplete MoE observations remain",
            collector.pending.len()
        ));
    }
    let mut root = collector
        .identity
        .as_object()
        .cloned()
        .ok_or_else(|| "report identity is not an object".to_string())?;
    root.insert("schema_version".to_string(), json!(SCHEMA_VERSION));
    root.insert("schema_fields".to_string(), json!(SCHEMA_FIELDS));
    root.insert("ownership".to_string(), Value::Array(collector.ownership));
    root.insert(
        "collectives".to_string(),
        json!({ "kind": "none", "physical_count": 0, "members": [] }),
    );
    root.insert("lifecycle".to_string(), lifecycle);
    root.insert(
        "observations".to_string(),
        serde_json::to_value(&collector.observations)
            .map_err(|e| err("serialize observations", e))?,
    );
    let out_path = collector.dir.join("report.json");
    let tmp_path = collector.dir.join("report.json.tmp");
    let bytes =
        serde_json::to_vec_pretty(&Value::Object(root)).map_err(|e| err("serialize report", e))?;
    fs::write(&tmp_path, bytes).map_err(|e| err("write report", e))?;
    fs::rename(&tmp_path, &out_path).map_err(|e| err("publish report", e))?;
    Ok(out_path)
}

#[derive(Debug, Deserialize)]
struct DiskReport {
    schema_version: u32,
    schema_fields: Vec<String>,
    fixture: Value,
    source: Value,
    config: Value,
    gpu: Value,
    collectives: Value,
    observations: Vec<Observation>,
}

fn read_report(dir: &Path) -> Result<(DiskReport, Value), String> {
    let path = dir.join("report.json");
    let bytes = fs::read(&path).map_err(|e| err("read report", e))?;
    let root: Value = serde_json::from_slice(&bytes).map_err(|e| err("parse report JSON", e))?;
    let report: DiskReport =
        serde_json::from_value(root.clone()).map_err(|e| err("decode report schema", e))?;
    Ok((report, root))
}

fn check_identity(left: &DiskReport, right: &DiskReport) -> Result<(), String> {
    if left.schema_version != SCHEMA_VERSION || right.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "report schema version mismatch (left {}, right {}, expected {})",
            left.schema_version, right.schema_version, SCHEMA_VERSION
        ));
    }
    let fields: Vec<String> = SCHEMA_FIELDS.iter().map(|s| (*s).to_string()).collect();
    if left.schema_fields != fields || right.schema_fields != fields {
        return Err("report schema fields are not the exact moe-oracle schema".to_string());
    }
    for (name, a, b) in [
        ("fixture", &left.fixture, &right.fixture),
        ("config", &left.config, &right.config),
        ("gpu", &left.gpu, &right.gpu),
        ("collectives", &left.collectives, &right.collectives),
    ] {
        if a != b {
            return Err(format!("identity mismatch in {name}"));
        }
    }
    for key in [
        "model_sha256",
        "metadata_sha256",
        "prompt_sha256",
        "token_ids",
        "token_ids_sha256",
    ] {
        if left.source.get(key) != right.source.get(key) {
            return Err(format!("identity mismatch in source.{key}"));
        }
    }
    Ok(())
}

fn load_array(dir: &Path, reference: &ArrayRef, field: &str) -> Result<Vec<u8>, String> {
    if reference.dtype == "empty" {
        if !reference.file.is_empty() || reference.bytes != 0 || !reference.shape.is_empty() {
            return Err(format!("{field}: malformed empty array descriptor"));
        }
        return Ok(Vec::new());
    }
    if reference.file.is_empty()
        || reference.file.contains("..")
        || Path::new(&reference.file).is_absolute()
    {
        return Err(format!("{field}: unsafe array path"));
    }
    let path = dir.join(&reference.file);
    let bytes = fs::read(&path).map_err(|e| err(&format!("read {field} array"), e))?;
    if bytes.len() != reference.bytes {
        return Err(format!(
            "{field}: descriptor says {} bytes, file has {}",
            reference.bytes,
            bytes.len()
        ));
    }
    let item_size = match reference.dtype.as_str() {
        "f32" | "u32" => 4,
        "u8" => 1,
        other => return Err(format!("{field}: unknown dtype {other}")),
    };
    let elements = reference
        .shape
        .iter()
        .try_fold(1usize, |acc, n| acc.checked_mul(*n))
        .ok_or_else(|| format!("{field}: shape overflow"))?;
    if elements.checked_mul(item_size) != Some(bytes.len()) {
        return Err(format!("{field}: shape/byte length mismatch"));
    }
    if reference.dtype == "f32" {
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            let value = f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap()));
            if !value.is_finite() {
                return Err(format!("{field}: nonfinite element {i}"));
            }
        }
    }
    Ok(bytes)
}

fn snapshot_fields(snapshot: &Snapshot) -> [(&'static str, &ArrayRef); 16] {
    [
        ("router_scores", &snapshot.router_scores),
        ("topk_ids", &snapshot.topk_ids),
        ("topk_weights", &snapshot.topk_weights),
        ("gate", &snapshot.gate),
        ("up", &snapshot.up),
        ("activated", &snapshot.activated),
        ("down", &snapshot.down),
        ("routed", &snapshot.routed),
        ("shared", &snapshot.shared),
        ("residual", &snapshot.residual),
        ("logits", &snapshot.logits),
        ("kv", &snapshot.kv),
        ("recurrent", &snapshot.recurrent),
        ("convolution", &snapshot.convolution),
        ("position", &snapshot.position),
        ("token_ids", &snapshot.token_ids),
    ]
}

fn compare_array(
    left_dir: &Path,
    right_dir: &Path,
    field: &str,
    left: &ArrayRef,
    right: &ArrayRef,
    sequence: &str,
    layer: usize,
    position: usize,
) -> Result<(), String> {
    if left.dtype != right.dtype || left.shape != right.shape || left.bytes != right.bytes {
        return Err(format!(
            "mismatch sequence={sequence} layer={layer} position={position} field={field}: descriptor differs"
        ));
    }
    let a = load_array(left_dir, left, field)?;
    let b = load_array(right_dir, right, field)?;
    if a.len() != b.len() {
        return Err(format!(
            "mismatch sequence={sequence} layer={layer} position={position} field={field}: byte length differs"
        ));
    }
    let width = if left.dtype == "u8" { 1 } else { 4 };
    for element in 0..(a.len() / width) {
        let start = element * width;
        if a[start..start + width] != b[start..start + width] {
            let av = if width == 4 {
                u32::from_le_bytes(a[start..start + 4].try_into().unwrap()) as u64
            } else {
                a[start] as u64
            };
            let bv = if width == 4 {
                u32::from_le_bytes(b[start..start + 4].try_into().unwrap()) as u64
            } else {
                b[start] as u64
            };
            return Err(format!(
                "mismatch sequence={sequence} layer={layer} position={position} field={field} element={element}: left=0x{av:08x} right=0x{bv:08x}"
            ));
        }
    }
    Ok(())
}

/// Compare final logits and logical state for two sequences collected in one
/// process.  Warm-prefix execution intentionally has different per-layer
/// prefill event boundaries, so the reusable-state proof compares its committed
/// step snapshots (which carry logits, state, position, and token IDs).
pub fn compare_sequences(left_sequence: &str, right_sequence: &str) -> Result<(), String> {
    let guard = active()
        .lock()
        .map_err(|_| "collector mutex poisoned".to_string())?;
    let collector = guard
        .as_ref()
        .ok_or_else(|| "collector is not active".to_string())?;
    let left: Vec<&Observation> = collector
        .observations
        .iter()
        .filter(|observation| observation.sequence == left_sequence && observation.kind == "step")
        .collect();
    let right: Vec<&Observation> = collector
        .observations
        .iter()
        .filter(|observation| observation.sequence == right_sequence && observation.kind == "step")
        .collect();
    if left.is_empty() || right.is_empty() {
        return Err(format!(
            "warm-prefix comparison has no step observations ({left_sequence}: {}, {right_sequence}: {})",
            left.len(),
            right.len()
        ));
    }
    if left.len() != right.len() {
        return Err(format!(
            "warm-prefix step count mismatch ({left_sequence}: {}, {right_sequence}: {})",
            left.len(),
            right.len()
        ));
    }
    for (left_observation, right_observation) in left.iter().zip(right.iter()) {
        if left_observation.position != right_observation.position {
            return Err(format!(
                "warm-prefix position mismatch (fresh {}, warm {})",
                left_observation.position, right_observation.position
            ));
        }
        for (field, a, b) in snapshot_fields(&left_observation.snapshot)
            .into_iter()
            .zip(snapshot_fields(&right_observation.snapshot))
            .map(|((field, a), (_, b))| (field, a, b))
        {
            compare_array(
                &collector.dir,
                &collector.dir,
                field,
                a,
                b,
                right_sequence,
                usize::MAX,
                left_observation.position,
            )?;
        }
    }
    Ok(())
}

pub fn compare(left_dir: &Path, right_dir: &Path) -> Result<(), String> {
    let (left, _) = read_report(left_dir)?;
    let (right, _) = read_report(right_dir)?;
    check_identity(&left, &right)?;
    if left.observations.len() != right.observations.len() {
        return Err(format!(
            "observation count mismatch (left {}, right {})",
            left.observations.len(),
            right.observations.len()
        ));
    }
    for (left_observation, right_observation) in left.observations.iter().zip(&right.observations) {
        if left_observation.sequence != right_observation.sequence
            || left_observation.kind != right_observation.kind
            || left_observation.layer != right_observation.layer
            || left_observation.position != right_observation.position
        {
            return Err(format!(
                "mismatch observation identity at sequence={} layer={} position={}",
                left_observation.sequence, left_observation.layer, left_observation.position
            ));
        }
        for (field, a, b) in snapshot_fields(&left_observation.snapshot)
            .into_iter()
            .zip(snapshot_fields(&right_observation.snapshot))
            .map(|((field, a), (_, b))| (field, a, b))
        {
            compare_array(
                left_dir,
                right_dir,
                field,
                a,
                b,
                &left_observation.sequence,
                left_observation.layer,
                left_observation.position,
            )?;
        }
    }
    Ok(())
}
