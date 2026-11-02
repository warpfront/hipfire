// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-request GDN scratch for the encoder: one FP32 S matrix and one
//! causal-conv ring per linear-attention layer.
//!
//! This is the `DeltaNetState` layout from `hipfire-arch-qwen35` (FP32
//! quant, fresh sequence) without the Q8/Q4/lane machinery an encoder never
//! uses: `s_matrices[l]` is `[n_v_heads × k_dim × v_dim]` f32 and
//! `conv_states[l]` is `[conv_dim × (kernel − 1)]` f32, both zeroed. Every
//! encode starts from zero state — an embedding model has no cross-request
//! recurrence — so [`GdnEncodeState::reset`] runs before each forward.

use crate::config::EmbedConfig;
use rdna_compute::{DType, Gpu, GpuTensor};

pub struct GdnEncodeState {
    pub s_matrices: Vec<GpuTensor>,
    pub conv_states: Vec<GpuTensor>,
    pub n_linear_layers: usize,
}

impl GdnEncodeState {
    pub fn new(gpu: &mut Gpu, cfg: &EmbedConfig) -> Result<Self, String> {
        let n_linear = cfg
            .layer_types
            .iter()
            .filter(|t| **t == crate::config::LayerType::LinearAttention)
            .count();
        let s_elems =
            cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim;
        let conv_elems = (cfg.gdn_key_dim() * 2 + cfg.gdn_value_dim()) * (cfg.conv_kernel_dim - 1);
        let mut s_matrices = Vec::with_capacity(n_linear);
        let mut conv_states = Vec::with_capacity(n_linear);
        for _ in 0..n_linear {
            let s = gpu
                .alloc_tensor(&[s_elems], DType::F32)
                .map_err(|e| format!("qwen35-embed: alloc S matrix: {e:?}"))?;
            gpu.zero_f32(&s)
                .map_err(|e| format!("qwen35-embed: zero S matrix: {e:?}"))?;
            let c = gpu
                .alloc_tensor(&[conv_elems], DType::F32)
                .map_err(|e| format!("qwen35-embed: alloc conv ring: {e:?}"))?;
            gpu.zero_f32(&c)
                .map_err(|e| format!("qwen35-embed: zero conv ring: {e:?}"))?;
            s_matrices.push(s);
            conv_states.push(c);
        }
        Ok(Self {
            s_matrices,
            conv_states,
            n_linear_layers: n_linear,
        })
    }

    /// Zero every S matrix and conv ring. Runs before each encode.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        for i in 0..self.n_linear_layers {
            self.reset_layer(gpu, i)?;
        }
        Ok(())
    }

    /// Zero one layer's S matrix and conv ring.
    pub fn reset_layer(&mut self, gpu: &mut Gpu, i: usize) -> Result<(), String> {
        self.reset_conv(gpu, i)?;
        self.reset_s(gpu, i)
    }

    /// Zero one layer's causal-conv ring.
    ///
    /// Split from [`Self::reset_s`] because the two are consumed in different
    /// passes: a batched encode runs every segment's conv, then a row-independent
    /// L2 norm over the whole row block, then every segment's delta-rule scan.
    /// Each pass must zero the state *it* reads at each segment's start — the
    /// scan zeroes S immediately before its own segment, not back when that
    /// segment's conv ran, or segment k would start from segment k-1's state.
    /// A single sequence needs neither: [`Self::reset`] once on entry is enough.
    pub fn reset_conv(&mut self, gpu: &mut Gpu, i: usize) -> Result<(), String> {
        gpu.zero_f32(&self.conv_states[i])
            .map_err(|e| format!("gdn reset: {e:?}"))
    }

    /// Zero one layer's delta-rule S matrix. See [`Self::reset_conv`].
    pub fn reset_s(&mut self, gpu: &mut Gpu, i: usize) -> Result<(), String> {
        gpu.zero_f32(&self.s_matrices[i])
            .map_err(|e| format!("gdn reset: {e:?}"))
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self
            .s_matrices
            .into_iter()
            .chain(self.conv_states.into_iter())
        {
            let _ = gpu.free_tensor(t);
        }
    }
}
