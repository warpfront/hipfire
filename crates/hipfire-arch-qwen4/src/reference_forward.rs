// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU/reference Qwen4 trunk forward equations.
//!
//! Production trunk execution lives in [`crate::gpu_forward`].
//!
//! The compact CPU/reference executor in this module is deliberately typed
//! like the ordinary-HIP path: one resident layer record, one recurrent/cache
//! state owner, and one shared token/chunk entry point.  It is used for fixture
//! parity and compact end-to-end probes; it does not read checkpoint files or
//! dequantize resident matrices.

use crate::config::LayerType;
use crate::ops::{hc_read, hc_write, ple_gate, rope_prefix_halfsplit, sigmoid, silu};
use std::fmt;

const EPS: f32 = 1.0e-6;

#[derive(Debug, Clone, PartialEq)]
pub enum ForwardError {
    Shape(&'static str),
    Invalid(String),
    LayerCount { expected: usize, actual: usize },
    Unsupported(&'static str),
}

impl fmt::Display for ForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(name) => write!(f, "Qwen4 forward shape error: {name}"),
            Self::Invalid(message) => write!(f, "Qwen4 forward input: {message}"),
            Self::LayerCount { expected, actual } => {
                write!(
                    f,
                    "Qwen4 forward layer count expected {expected}, got {actual}"
                )
            }
            Self::Unsupported(name) => write!(f, "Qwen4 forward unsupported: {name}"),
        }
    }
}

impl std::error::Error for ForwardError {}

/// Geometry for a compact typed forward probe.  The production pinned values
/// are accepted, while small values make deterministic CPU/GPU fixture probes
/// cheap and keep the operation ordering visible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactForwardConfig {
    pub hidden: usize,
    pub hc_count: usize,
    pub hc_rank: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub index_heads: usize,
    pub index_kv_heads: usize,
    pub index_dim: usize,
    pub qsa_budget: usize,
    pub qsa_compress: usize,
    pub gdn_key_heads: usize,
    pub gdn_value_heads: usize,
    pub gdn_key_dim: usize,
    pub gdn_value_dim: usize,
    pub gdn_conv_kernel: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    pub ple_layer: Option<usize>,
    pub ple_kernel: usize,
}

impl CompactForwardConfig {
    pub fn validate(&self) -> Result<(), ForwardError> {
        if self.hidden == 0 || self.hc_count == 0 || self.hc_rank == 0 {
            return Err(ForwardError::Invalid("HC geometry is zero".into()));
        }
        if self.q_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.q_heads % self.kv_heads != 0
        {
            return Err(ForwardError::Invalid("QSA head geometry is invalid".into()));
        }
        if self.index_heads == 0
            || self.index_kv_heads == 0
            || self.index_dim == 0
            || self.index_heads % self.index_kv_heads != 0
        {
            return Err(ForwardError::Invalid(
                "QSA index geometry is invalid".into(),
            ));
        }
        if self.qsa_budget == 0 || self.qsa_compress == 0 {
            return Err(ForwardError::Invalid("QSA budget/compress is zero".into()));
        }
        if self.gdn_key_heads == 0
            || self.gdn_value_heads == 0
            || self.gdn_key_dim == 0
            || self.gdn_value_dim == 0
            || self.gdn_value_heads % self.gdn_key_heads != 0
            || self.gdn_conv_kernel == 0
        {
            return Err(ForwardError::Invalid("GDN geometry is invalid".into()));
        }
        if self.experts < self.top_k || self.top_k == 0 || self.moe_intermediate == 0 {
            return Err(ForwardError::Invalid(
                "MoE top-k geometry is invalid".into(),
            ));
        }
        if self.shared_intermediate == 0 {
            return Err(ForwardError::Invalid("shared expert width is zero".into()));
        }
        if self.ple_layer.is_some() && (self.ple_kernel == 0 || self.ple_kernel % 2 == 0) {
            return Err(ForwardError::Invalid(
                "PLE kernel must be odd and non-zero".into(),
            ));
        }
        Ok(())
    }

    #[inline]
    pub fn wide(&self) -> usize {
        self.hidden * self.hc_count
    }

    #[inline]
    pub fn q_width(&self) -> usize {
        self.q_heads * self.head_dim
    }

    #[inline]
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    #[inline]
    pub fn index_width(&self) -> usize {
        (self.index_heads + self.index_kv_heads) * self.index_dim
    }

    #[inline]
    pub fn gdn_qk_width(&self) -> usize {
        self.gdn_key_heads * self.gdn_key_dim
    }

    #[inline]
    pub fn gdn_v_width(&self) -> usize {
        self.gdn_value_heads * self.gdn_value_dim
    }

    #[inline]
    pub fn selected_capacity(&self) -> usize {
        self.qsa_budget + self.qsa_compress - 1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceHyperWeights {
    pub block_inject: Vec<f32>,
    pub hc_norm: Vec<f32>,
    pub input_mix_down: Vec<f32>,
    pub input_mix_up: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceGdnWeights {
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub in_proj_a: Vec<f32>,
    pub in_proj_b: Vec<f32>,
    pub qkv: Vec<f32>,
    pub z: Vec<f32>,
    pub norm: Vec<f32>,
    pub conv: Vec<f32>,
    pub output: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceQsaWeights {
    pub index_qk: Vec<f32>,
    pub indexer_k_norm: Vec<f32>,
    pub indexer_q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub k: Vec<f32>,
    pub output: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub v: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferencePleWeights {
    pub conv: Vec<f32>,
    pub key: Vec<f32>,
    pub norm_conv: Vec<f32>,
    pub norm_key: Vec<f32>,
    pub norm_query: Vec<f32>,
    pub value: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceMoeWeights {
    pub router: Vec<f32>,
    pub experts_gate_up: Vec<f32>,
    pub experts_down: Vec<f32>,
    pub shared_gate: Vec<f32>,
    pub shared_up: Vec<f32>,
    pub shared_down: Vec<f32>,
    pub shared_gate_scalar: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceLayerWeights {
    pub kind: LayerType,
    pub attn_hyper: ReferenceHyperWeights,
    pub mlp_hyper: ReferenceHyperWeights,
    pub gdn: Option<ReferenceGdnWeights>,
    pub qsa: Option<ReferenceQsaWeights>,
    pub ple: Option<ReferencePleWeights>,
    pub moe: ReferenceMoeWeights,
}

#[derive(Clone, Debug, PartialEq)]
struct CompactGdnState {
    recurrent: Vec<f32>,
    conv_history: Vec<f32>,
    conv_cursor: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct CompactQsaState {
    full_keys: Vec<f32>,
    full_values: Vec<f32>,
    raw_index_keys: Vec<f32>,
    position: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct CompactPleState {
    history: Vec<f32>,
    cursor: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceForwardState {
    gdn: Vec<CompactGdnState>,
    qsa: Vec<CompactQsaState>,
    ple: Option<CompactPleState>,
    position: usize,
}

impl ReferenceForwardState {
    fn new(config: &CompactForwardConfig, layers: &[ReferenceLayerWeights]) -> Self {
        let gdn_count = layers
            .iter()
            .filter(|layer| layer.kind == LayerType::LinearAttention)
            .count();
        let qsa_count = layers
            .iter()
            .filter(|layer| layer.kind == LayerType::FullAttention)
            .count();
        let qk = config.gdn_key_heads * config.gdn_key_dim;
        let vd = config.gdn_value_heads * config.gdn_value_dim;
        let conv_channels = 2 * qk + vd;
        let conv_rows = config.gdn_conv_kernel.saturating_sub(1);
        let mut gdn = Vec::with_capacity(gdn_count);
        for _ in 0..gdn_count {
            gdn.push(CompactGdnState {
                recurrent: vec![
                    0.0;
                    config.gdn_value_heads * config.gdn_key_dim * config.gdn_value_dim
                ],
                conv_history: vec![0.0; conv_rows * conv_channels],
                conv_cursor: 0,
            });
        }
        let mut qsa = Vec::with_capacity(qsa_count);
        for _ in 0..qsa_count {
            qsa.push(CompactQsaState {
                full_keys: Vec::new(),
                full_values: Vec::new(),
                raw_index_keys: Vec::new(),
                position: 0,
            });
        }
        let ple = config.ple_layer.map(|_| CompactPleState {
            history: vec![0.0; config.ple_kernel.saturating_sub(1) * config.wide()],
            cursor: 0,
        });
        Self {
            gdn,
            qsa,
            ple,
            position: 0,
        }
    }

    pub fn reset(&mut self) {
        for state in &mut self.gdn {
            state.recurrent.fill(0.0);
            state.conv_history.fill(0.0);
            state.conv_cursor = 0;
        }
        for state in &mut self.qsa {
            state.full_keys.clear();
            state.full_values.clear();
            state.raw_index_keys.clear();
            state.position = 0;
        }
        if let Some(state) = self.ple.as_mut() {
            state.history.fill(0.0);
            state.cursor = 0;
        }
        self.position = 0;
    }
}

/// A compact, typed CPU/reference trunk executor.  It consumes already-resident
/// activation/weight arrays and follows the same token loop for one-token and
/// bounded-chunk calls, which makes chunk parity a direct invariant.
#[derive(Clone, Debug)]
pub struct ReferenceQwen4Forward {
    pub config: CompactForwardConfig,
    pub state: ReferenceForwardState,
}

impl ReferenceQwen4Forward {
    pub fn new(
        config: CompactForwardConfig,
        layers: &[ReferenceLayerWeights],
    ) -> Result<Self, ForwardError> {
        config.validate()?;
        Ok(Self {
            state: ReferenceForwardState::new(&config, layers),
            config,
        })
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn forward_token(
        &mut self,
        embedding: &[f32],
        layers: &[ReferenceLayerWeights],
        ple_embedding: Option<&[f32]>,
    ) -> Result<Vec<f32>, ForwardError> {
        let output = self.forward_chunk(embedding, 1, layers, ple_embedding)?;
        Ok(output)
    }

    /// Run `tokens` rows stored contiguously as `[tokens, hidden]` and return
    /// the final collapsed hidden row.  The operation order is shared with
    /// `forward_token`; no per-token alternate implementation exists.
    pub fn forward_chunk(
        &mut self,
        embeddings: &[f32],
        tokens: usize,
        layers: &[ReferenceLayerWeights],
        ple_embeddings: Option<&[f32]>,
    ) -> Result<Vec<f32>, ForwardError> {
        if tokens == 0 || embeddings.len() != tokens * self.config.hidden {
            return Err(ForwardError::Shape("embedding chunk"));
        }
        if layers.is_empty() {
            return Err(ForwardError::Invalid("forward has no layers".into()));
        }
        if let Some(rows) = ple_embeddings {
            if rows.len() != embeddings.len() {
                return Err(ForwardError::Shape("PLE embedding chunk"));
            }
        }
        let gdn_count = layers
            .iter()
            .filter(|l| l.kind == LayerType::LinearAttention)
            .count();
        let qsa_count = layers
            .iter()
            .filter(|l| l.kind == LayerType::FullAttention)
            .count();
        if gdn_count != self.state.gdn.len() || qsa_count != self.state.qsa.len() {
            return Err(ForwardError::LayerCount {
                expected: self.state.gdn.len() + self.state.qsa.len(),
                actual: layers.len(),
            });
        }
        let mut last = vec![0.0f32; self.config.hidden];
        for token in 0..tokens {
            let mut streams = vec![0.0f32; self.config.wide()];
            for branch in 0..self.config.hc_count {
                streams[branch * self.config.hidden..(branch + 1) * self.config.hidden]
                    .copy_from_slice(
                        &embeddings[token * self.config.hidden..(token + 1) * self.config.hidden],
                    );
            }
            let ple_row = ple_embeddings
                .map(|rows| &rows[token * self.config.hidden..(token + 1) * self.config.hidden]);
            let mut gdn_slot = 0usize;
            let mut qsa_slot = 0usize;
            for (layer_index, layer) in layers.iter().enumerate() {
                if Some(layer_index) == self.config.ple_layer {
                    if let (Some(ple), Some(row)) = (&layer.ple, ple_row) {
                        if let Some(mut state) = self.state.ple.take() {
                            self.apply_ple(&mut state, ple, row, &mut streams)?;
                            self.state.ple = Some(state);
                        }
                    }
                }
                let mut normalized = vec![0.0f32; self.config.wide()];
                let mut attn_input = vec![0.0f32; self.config.hidden];
                hc_read(
                    &streams,
                    &layer.attn_hyper.input_mix_down,
                    &layer.attn_hyper.input_mix_up,
                    &layer.attn_hyper.hc_norm,
                    self.config.hc_count,
                    self.config.hidden,
                    self.config.hc_rank,
                    EPS,
                    &mut normalized,
                    &mut attn_input,
                )
                .map_err(|_| ForwardError::Shape("attention HC read"))?;
                let attn_output = match layer.kind {
                    LayerType::LinearAttention => {
                        let slot = gdn_slot;
                        gdn_slot += 1;
                        let mut state = self.state.gdn.remove(slot);
                        let result = self.gdn_layer(
                            &mut state,
                            layer
                                .gdn
                                .as_ref()
                                .ok_or(ForwardError::Shape("GDN weights"))?,
                            &attn_input,
                        );
                        self.state.gdn.insert(slot, state);
                        result?
                    }
                    LayerType::FullAttention => {
                        let slot = qsa_slot;
                        qsa_slot += 1;
                        let mut state = self.state.qsa.remove(slot);
                        let result = self.qsa_layer(
                            &mut state,
                            layer
                                .qsa
                                .as_ref()
                                .ok_or(ForwardError::Shape("QSA weights"))?,
                            &attn_input,
                        );
                        self.state.qsa.insert(slot, state);
                        result?
                    }
                };
                let mut after_attn = vec![0.0f32; self.config.wide()];
                hc_write(
                    &streams,
                    &normalized,
                    &attn_output,
                    &layer.attn_hyper.block_inject,
                    self.config.hc_count,
                    self.config.hidden,
                    &mut after_attn,
                )
                .map_err(|_| ForwardError::Shape("attention HC write"))?;
                streams = after_attn;

                let mut mlp_norm = vec![0.0f32; self.config.wide()];
                let mut mlp_input = vec![0.0f32; self.config.hidden];
                hc_read(
                    &streams,
                    &layer.mlp_hyper.input_mix_down,
                    &layer.mlp_hyper.input_mix_up,
                    &layer.mlp_hyper.hc_norm,
                    self.config.hc_count,
                    self.config.hidden,
                    self.config.hc_rank,
                    EPS,
                    &mut mlp_norm,
                    &mut mlp_input,
                )
                .map_err(|_| ForwardError::Shape("MLP HC read"))?;
                let mlp_output = self.moe_layer(&layer.moe, &mlp_input)?;
                let mut after_mlp = vec![0.0f32; self.config.wide()];
                hc_write(
                    &streams,
                    &mlp_norm,
                    &mlp_output,
                    &layer.mlp_hyper.block_inject,
                    self.config.hc_count,
                    self.config.hidden,
                    &mut after_mlp,
                )
                .map_err(|_| ForwardError::Shape("MLP HC write"))?;
                streams = after_mlp;
            }
            self.final_mix(&streams, &mut last)?;
            self.state.position += 1;
        }
        Ok(last)
    }

    fn apply_ple(
        &self,
        state: &mut CompactPleState,
        weights: &ReferencePleWeights,
        row: &[f32],
        streams: &mut [f32],
    ) -> Result<(), ForwardError> {
        let wide = self.config.wide();
        if row.len() != self.config.hidden
            || streams.len() != wide
            || weights.key.len() != wide * self.config.hidden
            || weights.value.len() != self.config.hidden * self.config.hidden
            || weights.norm_key.len() != wide
            || weights.norm_query.len() != wide
            || weights.norm_conv.len() != wide
            || weights.conv.len() != wide * self.config.ple_kernel
        {
            return Err(ForwardError::Shape("PLE weights"));
        }
        let key = matvec(&weights.key, wide, self.config.hidden, row)?;
        let value = matvec(&weights.value, self.config.hidden, self.config.hidden, row)?;
        let key = rms_scaled(&key, &weights.norm_key);
        let query = rms_scaled(streams, &weights.norm_query);
        let mut gated = vec![0.0f32; wide];
        for branch in 0..self.config.hc_count {
            let start = branch * self.config.hidden;
            let dot = key[start..start + self.config.hidden]
                .iter()
                .zip(&query[start..start + self.config.hidden])
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                / (self.config.hidden as f32).sqrt();
            let gate = ple_gate(dot);
            for j in 0..self.config.hidden {
                gated[start + j] = gate * value[j];
            }
        }
        let gated_norm = rms_scaled(&gated, &weights.norm_conv);
        let rows = self.config.ple_kernel.saturating_sub(1);
        let mut conv = vec![0.0f32; wide];
        for channel in 0..wide {
            let mut sum = weights.conv[channel] * gated_norm[channel];
            for tap in 1..self.config.ple_kernel {
                let delay = tap * 3;
                if rows > 0 {
                    let history_row = (state.cursor + rows - (delay % rows)) % rows;
                    sum += weights.conv[tap * wide + channel]
                        * state.history[history_row * wide + channel];
                }
            }
            conv[channel] = silu(sum);
        }
        if rows > 0 {
            let offset = state.cursor * wide;
            state.history[offset..offset + wide].copy_from_slice(&gated_norm);
            state.cursor = (state.cursor + 1) % rows;
        }
        for (stream, (&base, &filtered)) in streams.iter_mut().zip(gated.iter().zip(conv.iter())) {
            *stream += base + filtered;
        }
        Ok(())
    }

    fn gdn_layer(
        &self,
        state: &mut CompactGdnState,
        weights: &ReferenceGdnWeights,
        input: &[f32],
    ) -> Result<Vec<f32>, ForwardError> {
        let qk = self.config.gdn_qk_width();
        let vw = self.config.gdn_v_width();
        let qkv = matvec(&weights.qkv, 2 * qk + vw, self.config.hidden, input)?;
        let a = matvec(
            &weights.in_proj_a,
            self.config.gdn_value_heads,
            self.config.hidden,
            input,
        )?;
        let b = matvec(
            &weights.in_proj_b,
            self.config.gdn_value_heads,
            self.config.hidden,
            input,
        )?;
        if weights.conv.len() != (2 * qk + vw) * self.config.gdn_conv_kernel {
            return Err(ForwardError::Shape("GDN convolution weights"));
        }
        let conv_rows = self.config.gdn_conv_kernel.saturating_sub(1);
        let mut mixed = vec![0.0f32; 2 * qk + vw];
        for channel in 0..mixed.len() {
            let mut value =
                weights.conv[channel * self.config.gdn_conv_kernel + conv_rows] * qkv[channel];
            for tap in 0..conv_rows {
                if conv_rows > 0 {
                    let row = (state.conv_cursor + tap) % conv_rows;
                    value += weights.conv[channel * self.config.gdn_conv_kernel + tap]
                        * state.conv_history[row * mixed.len() + channel];
                }
            }
            mixed[channel] = silu(value);
        }
        if conv_rows > 0 {
            let offset = state.conv_cursor * mixed.len();
            state.conv_history[offset..offset + mixed.len()].copy_from_slice(&qkv);
            state.conv_cursor = (state.conv_cursor + 1) % conv_rows;
        }
        let mut q_expanded = vec![0.0f32; self.config.gdn_value_heads * self.config.gdn_key_dim];
        let mut k_expanded = q_expanded.clone();
        let repeat = self.config.gdn_value_heads / self.config.gdn_key_heads;
        for vh in 0..self.config.gdn_value_heads {
            let kh = vh / repeat;
            let source = kh * self.config.gdn_key_dim;
            let target = vh * self.config.gdn_key_dim;
            q_expanded[target..target + self.config.gdn_key_dim]
                .copy_from_slice(&mixed[source..source + self.config.gdn_key_dim]);
            k_expanded[target..target + self.config.gdn_key_dim]
                .copy_from_slice(&mixed[qk + source..qk + source + self.config.gdn_key_dim]);
        }
        let mut output = vec![0.0f32; vw];
        for vh in 0..self.config.gdn_value_heads {
            let qn = l2_normalize(
                &q_expanded[vh * self.config.gdn_key_dim..(vh + 1) * self.config.gdn_key_dim],
            );
            let kn = l2_normalize(
                &k_expanded[vh * self.config.gdn_key_dim..(vh + 1) * self.config.gdn_key_dim],
            );
            let decay = -weights.a_log[vh].exp() * softplus(a[vh] + weights.dt_bias[vh]);
            let base = vh * self.config.gdn_key_dim * self.config.gdn_value_dim;
            for idx in 0..self.config.gdn_key_dim * self.config.gdn_value_dim {
                state.recurrent[base + idx] *= decay.exp();
            }
            let vbase = 2 * qk + vh * self.config.gdn_value_dim;
            for vd in 0..self.config.gdn_value_dim {
                let kv_mem = (0..self.config.gdn_key_dim)
                    .map(|kd| state.recurrent[base + kd * self.config.gdn_value_dim + vd] * kn[kd])
                    .sum::<f32>();
                let delta = (mixed[vbase + vd] - kv_mem) * sigmoid(b[vh]);
                for kd in 0..self.config.gdn_key_dim {
                    state.recurrent[base + kd * self.config.gdn_value_dim + vd] += kn[kd] * delta;
                }
                output[vh * self.config.gdn_value_dim + vd] = (0..self.config.gdn_key_dim)
                    .map(|kd| state.recurrent[base + kd * self.config.gdn_value_dim + vd] * qn[kd])
                    .sum::<f32>();
            }
        }
        let z = matvec(&weights.z, vw, self.config.hidden, input)?;
        for vh in 0..self.config.gdn_value_heads {
            let norm = &weights.norm[..self.config.gdn_value_dim];
            let start = vh * self.config.gdn_value_dim;
            let inv = (output[start..start + self.config.gdn_value_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / self.config.gdn_value_dim as f32
                + EPS)
                .sqrt()
                .recip();
            for vd in 0..self.config.gdn_value_dim {
                output[start + vd] = output[start + vd] * inv * norm[vd] * sigmoid(z[start + vd]);
            }
        }
        matvec(&weights.output, self.config.hidden, vw, &output)
    }

    fn qsa_layer(
        &self,
        state: &mut CompactQsaState,
        weights: &ReferenceQsaWeights,
        input: &[f32],
    ) -> Result<Vec<f32>, ForwardError> {
        let index_width = self.config.index_width();
        let index = matvec(&weights.index_qk, index_width, self.config.hidden, input)?;
        let mut index_query = index[..self.config.index_heads * self.config.index_dim].to_vec();
        let mut raw_key = index[self.config.index_heads * self.config.index_dim..].to_vec();
        for head in 0..self.config.index_heads {
            let start = head * self.config.index_dim;
            let norm = rms_scaled(
                &index_query[start..start + self.config.index_dim],
                &weights.indexer_q_norm,
            );
            index_query[start..start + self.config.index_dim].copy_from_slice(&norm);
            rope_prefix_halfsplit(
                &mut index_query[start..start + self.config.index_dim],
                state.position,
                self.config.index_dim.min(64),
                10_000_000.0,
            )
            .map_err(|_| ForwardError::Shape("QSA index query RoPE"))?;
        }
        for head in 0..self.config.index_kv_heads {
            let start = head * self.config.index_dim;
            let norm = rms_scaled(
                &raw_key[start..start + self.config.index_dim],
                &weights.indexer_k_norm,
            );
            raw_key[start..start + self.config.index_dim].copy_from_slice(&norm);
        }
        state.raw_index_keys.extend_from_slice(&raw_key);
        let q_proj = matvec(
            &weights.q,
            2 * self.config.q_width(),
            self.config.hidden,
            input,
        )?;
        let mut query = q_proj[..self.config.q_width()].to_vec();
        let gate = &q_proj[self.config.q_width()..];
        let mut key = matvec(
            &weights.k,
            self.config.kv_width(),
            self.config.hidden,
            input,
        )?;
        let value = matvec(
            &weights.v,
            self.config.kv_width(),
            self.config.hidden,
            input,
        )?;
        for head in 0..self.config.q_heads {
            let start = head * self.config.head_dim;
            let norm = rms_scaled(&query[start..start + self.config.head_dim], &weights.q_norm);
            query[start..start + self.config.head_dim].copy_from_slice(&norm);
            rope_prefix_halfsplit(
                &mut query[start..start + self.config.head_dim],
                state.position,
                64,
                10_000_000.0,
            )
            .map_err(|_| ForwardError::Shape("QSA query RoPE"))?;
        }
        for head in 0..self.config.kv_heads {
            let start = head * self.config.head_dim;
            let norm = rms_scaled(&key[start..start + self.config.head_dim], &weights.k_norm);
            key[start..start + self.config.head_dim].copy_from_slice(&norm);
            rope_prefix_halfsplit(
                &mut key[start..start + self.config.head_dim],
                state.position,
                64,
                10_000_000.0,
            )
            .map_err(|_| ForwardError::Shape("QSA key RoPE"))?;
        }
        state.full_keys.extend_from_slice(&key);
        state.full_values.extend_from_slice(&value);
        let visible = state.position + 1;
        let complete = visible / self.config.qsa_compress;
        let mut pooled = Vec::with_capacity(complete * self.config.index_dim);
        let mut positions = Vec::with_capacity(complete);
        for block in 0..complete {
            let start = block * self.config.qsa_compress * self.config.index_dim;
            let end = start + self.config.qsa_compress * self.config.index_dim;
            pooled.extend_from_slice(&pool_index_block(
                &state.raw_index_keys[start..end],
                self.config.qsa_compress,
                block * self.config.qsa_compress,
                self.config.index_dim,
            ));
            positions.push(block * self.config.qsa_compress);
        }
        let mut selected = select_index_tokens(
            &index_query,
            &pooled,
            &positions,
            self.config.index_heads,
            self.config.index_dim,
            self.config.qsa_budget / self.config.qsa_compress,
            self.config.qsa_compress,
            visible,
        );
        let tail_start = complete * self.config.qsa_compress;
        selected.extend(tail_start..visible);
        if selected.len() > self.config.selected_capacity() {
            selected.truncate(self.config.selected_capacity());
        }
        let mut heads = vec![0.0f32; self.config.q_width()];
        for qh in 0..self.config.q_heads {
            let kvh = qh / (self.config.q_heads / self.config.kv_heads);
            let qstart = qh * self.config.head_dim;
            let mut scores = Vec::with_capacity(selected.len());
            for &token in &selected {
                let kstart = token * self.config.kv_width() + kvh * self.config.head_dim;
                let dot = query[qstart..qstart + self.config.head_dim]
                    .iter()
                    .zip(&state.full_keys[kstart..kstart + self.config.head_dim])
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
                    / (self.config.head_dim as f32).sqrt();
                scores.push(dot);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|v| (*v - max).exp())
                .sum::<f32>()
                .max(EPS);
            for channel in 0..self.config.head_dim {
                let mut value_sum = 0.0f32;
                for (slot, &token) in selected.iter().enumerate() {
                    let vstart = token * self.config.kv_width() + kvh * self.config.head_dim;
                    value_sum +=
                        (scores[slot] - max).exp() / denom * state.full_values[vstart + channel];
                }
                heads[qstart + channel] = value_sum * sigmoid(gate[qstart + channel]);
            }
        }
        state.position += 1;
        matvec(
            &weights.output,
            self.config.hidden,
            self.config.q_width(),
            &heads,
        )
    }

    fn moe_layer(
        &self,
        weights: &ReferenceMoeWeights,
        input: &[f32],
    ) -> Result<Vec<f32>, ForwardError> {
        let h = self.config.hidden;
        let mi = self.config.moe_intermediate;
        let smi = self.config.shared_intermediate;
        let logits = matvec(&weights.router, self.config.experts, h, input)?;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = logits.iter().map(|v| (*v - max).exp()).collect::<Vec<_>>();
        let total = probs.iter().sum::<f32>().max(EPS);
        for p in &mut probs {
            *p /= total;
        }
        let mut order = (0..self.config.experts).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        order.truncate(self.config.top_k);
        let selected_total = order.iter().map(|&i| probs[i]).sum::<f32>().max(EPS);
        let mut output = vec![0.0f32; h];
        for &expert in &order {
            let gate = &weights.experts_gate_up[expert * 2 * mi * h..(expert + 1) * 2 * mi * h];
            let mut hidden = vec![0.0f32; mi];
            for j in 0..mi {
                let gate_value = dot(&gate[j * h..(j + 1) * h], input);
                let up_value = dot(&gate[(mi + j) * h..(mi + j + 1) * h], input);
                hidden[j] = silu(gate_value) * up_value;
            }
            let down = &weights.experts_down[expert * h * mi..(expert + 1) * h * mi];
            let route = probs[expert] / selected_total;
            for i in 0..h {
                output[i] += route * dot(&down[i * mi..(i + 1) * mi], &hidden);
            }
        }
        let shared_gate = matvec(&weights.shared_gate, smi, h, input)?;
        let shared_up = matvec(&weights.shared_up, smi, h, input)?;
        let mut shared_hidden = vec![0.0f32; smi];
        for i in 0..smi {
            shared_hidden[i] = silu(shared_gate[i]) * shared_up[i];
        }
        let shared = matvec(&weights.shared_down, h, smi, &shared_hidden)?;
        for i in 0..h {
            let scalar = weights
                .shared_gate_scalar
                .get(i)
                .copied()
                .ok_or(ForwardError::Shape("shared gate scalar"))?;
            output[i] += sigmoid(scalar) * shared[i];
        }
        Ok(output)
    }

    fn final_mix(&self, streams: &[f32], output: &mut [f32]) -> Result<(), ForwardError> {
        if streams.len() != self.config.wide() || output.len() != self.config.hidden {
            return Err(ForwardError::Shape("final HC collapse"));
        }
        for j in 0..self.config.hidden {
            output[j] = (0..self.config.hc_count)
                .map(|branch| streams[branch * self.config.hidden + j])
                .sum::<f32>()
                / self.config.hc_count as f32;
        }
        Ok(())
    }
}

fn matvec(
    weights: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, ForwardError> {
    if weights.len() != rows * cols || input.len() != cols {
        return Err(ForwardError::Shape("matrix/vector projection"));
    }
    let mut output = vec![0.0f32; rows];
    for row in 0..rows {
        output[row] = dot(&weights[row * cols..(row + 1) * cols], input);
    }
    Ok(output)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn rms_scaled(input: &[f32], weight: &[f32]) -> Vec<f32> {
    let inv = (input.iter().map(|v| v * v).sum::<f32>() / input.len().max(1) as f32 + EPS)
        .sqrt()
        .recip();
    input
        .iter()
        .enumerate()
        .map(|(i, &value)| value * inv * (1.0 + weight[i % weight.len()]))
        .collect()
}

fn l2_normalize(input: &[f32]) -> Vec<f32> {
    let inv = (input.iter().map(|v| v * v).sum::<f32>() + EPS)
        .sqrt()
        .recip();
    input.iter().map(|value| value * inv).collect()
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn pool_index_block(keys: &[f32], block: usize, start: usize, dim: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; dim];
    for row in keys.chunks_exact(dim) {
        for (dst, &value) in pooled.iter_mut().zip(row) {
            *dst += value / block as f32;
        }
    }
    let inv = (pooled.iter().map(|v| v * v).sum::<f32>() / dim as f32 + EPS)
        .sqrt()
        .recip();
    for value in &mut pooled {
        *value *= inv;
    }
    let _ = rope_prefix_halfsplit(&mut pooled, start, dim.min(64), 10_000_000.0);
    pooled
}

fn select_index_tokens(
    query: &[f32],
    pooled: &[f32],
    positions: &[usize],
    heads: usize,
    dim: usize,
    budget_blocks: usize,
    block: usize,
    visible: usize,
) -> Vec<usize> {
    let mut scored = positions
        .iter()
        .enumerate()
        .map(|(index, &position)| {
            let key = &pooled[index * dim..(index + 1) * dim];
            let score = (0..heads)
                .map(|head| dot(&query[head * dim..(head + 1) * dim], key).max(0.0))
                .sum::<f32>()
                / (dim as f32).sqrt();
            (score, position)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|(a, ap), (b, bp)| {
        b.partial_cmp(a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ap.cmp(bp))
    });
    let mut selected = Vec::new();
    for &(_, start) in scored.iter().take(budget_blocks) {
        selected.extend((start..(start + block)).filter(|&position| position < visible));
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> CompactForwardConfig {
        CompactForwardConfig {
            hidden: 8,
            hc_count: 2,
            hc_rank: 3,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 4,
            index_heads: 1,
            index_kv_heads: 1,
            index_dim: 4,
            qsa_budget: 4,
            qsa_compress: 2,
            gdn_key_heads: 1,
            gdn_value_heads: 2,
            gdn_key_dim: 4,
            gdn_value_dim: 4,
            gdn_conv_kernel: 3,
            experts: 4,
            top_k: 2,
            moe_intermediate: 3,
            shared_intermediate: 3,
            ple_layer: None,
            ple_kernel: 4,
        }
    }

    fn hc(config: &CompactForwardConfig) -> ReferenceHyperWeights {
        ReferenceHyperWeights {
            block_inject: vec![0.0; config.wide() * config.hc_count],
            hc_norm: vec![0.0; config.wide()],
            input_mix_down: vec![0.01; config.hc_rank * config.wide()],
            input_mix_up: vec![0.01; config.wide() * config.hc_rank],
        }
    }

    fn moe(config: &CompactForwardConfig) -> ReferenceMoeWeights {
        ReferenceMoeWeights {
            router: vec![0.01; config.experts * config.hidden],
            experts_gate_up: vec![
                0.01;
                config.experts * 2 * config.moe_intermediate * config.hidden
            ],
            experts_down: vec![0.01; config.experts * config.hidden * config.moe_intermediate],
            shared_gate: vec![0.01; config.shared_intermediate * config.hidden],
            shared_up: vec![0.01; config.shared_intermediate * config.hidden],
            shared_down: vec![0.01; config.hidden * config.shared_intermediate],
            shared_gate_scalar: vec![0.0; config.hidden],
        }
    }

    #[test]
    fn compact_gdn_chunk_matches_token_path() {
        let config = tiny_config();
        let layers = vec![ReferenceLayerWeights {
            kind: LayerType::LinearAttention,
            attn_hyper: hc(&config),
            mlp_hyper: hc(&config),
            gdn: Some(ReferenceGdnWeights {
                a_log: vec![0.0; config.gdn_value_heads],
                dt_bias: vec![0.0; config.gdn_value_heads],
                in_proj_a: vec![0.01; config.gdn_value_heads * config.hidden],
                in_proj_b: vec![0.01; config.gdn_value_heads * config.hidden],
                qkv: vec![0.01; (2 * config.gdn_qk_width() + config.gdn_v_width()) * config.hidden],
                z: vec![0.01; config.gdn_v_width() * config.hidden],
                norm: vec![0.0; config.gdn_value_dim],
                conv: vec![
                    0.01;
                    (2 * config.gdn_qk_width() + config.gdn_v_width())
                        * config.gdn_conv_kernel
                ],
                output: vec![0.01; config.hidden * config.gdn_v_width()],
            }),
            qsa: None,
            ple: None,
            moe: moe(&config),
        }];
        let embeddings = (0..16).map(|i| i as f32 / 16.0).collect::<Vec<_>>();
        let mut token = ReferenceQwen4Forward::new(config.clone(), &layers).unwrap();
        let first = token
            .forward_token(&embeddings[..8], &layers, None)
            .unwrap();
        let second = token
            .forward_token(&embeddings[8..], &layers, None)
            .unwrap();
        let mut chunk = ReferenceQwen4Forward::new(config, &layers).unwrap();
        let combined = chunk.forward_chunk(&embeddings, 2, &layers, None).unwrap();
        assert_eq!(second.len(), combined.len());
        assert!(second
            .iter()
            .zip(combined.iter())
            .all(|(a, b)| (a - b).abs() < 1.0e-5));
        assert!(first.iter().all(|value| value.is_finite()));
    }
}
