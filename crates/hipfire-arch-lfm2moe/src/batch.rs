// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Independent continuous-batch state for dense LFM2.5 models.
//!
//! Mirrors Qwen35DecodeBatchState but for dense LFM2.5 (arch_id 11, num_experts==0).
//! Each lane owns an independent KV slice and an independent conv ring buffer.
//! Steady-state forward uses batched kernels once per layer with batch=B, not a per-lane loop.

use hip_bridge::{HipError, HipResult};
use hipfire_runtime::llama::KvCache;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::{Lfm2MoeConfig, MixerKind};

/// Dense LFM2.5 independent continuous-batch decode state.
///
/// Owns the batched KV cache, per-layer conv rings, batched scratch buffers,
/// and per-row sampling state. Every lane reset is range-checked and clears
/// only that lane's slices.
pub struct Lfm2DecodeBatchState {
    pub max_batch: usize,
    pub lane_capacity: usize,
    pub sample_repeat_capacity: usize,
    pub kv: KvCache,
    pub conv_states: Vec<GpuTensor>,

    // batched scratch
    pub x_batch: GpuTensor,
    pub tmp_batch: GpuTensor,
    pub conv_bcx_batch: GpuTensor,
    pub conv_y_batch: GpuTensor,
    pub fa_q_batch: GpuTensor,
    pub fa_k_batch: GpuTensor,
    pub fa_v_batch: GpuTensor,
    pub fa_attn_out_batch: GpuTensor,
    pub ffn_tmp_batch: GpuTensor,
    pub dense_gate_batch: GpuTensor,
    pub dense_up_batch: GpuTensor,
    pub dense_act_batch: GpuTensor,
    pub final_hidden: GpuTensor,
    pub logits: GpuTensor,
    pub lm_rot: GpuTensor,
    pub proj_rot: GpuTensor,
    pub sample_out: GpuTensor,
    pub sample_repeat_tokens: GpuTensor,
    pub sample_repeat_lengths: GpuTensor,
    pub sample_rng_states: GpuTensor,
    pub positions: GpuTensor,
    pub tokens: GpuTensor,
}

impl Lfm2DecodeBatchState {
    /// Create a new dense batch state. Fails closed on zero/overflow geometry,
    /// MoE configs, or lane capacity that exceeds the GPU's independent-
    /// attention LDS limit.
    pub fn new(
        gpu: &mut Gpu,
        cfg: &Lfm2MoeConfig,
        max_batch: usize,
        lane_capacity: usize,
        sample_repeat_capacity: usize,
    ) -> HipResult<Self> {
        if max_batch == 0 || lane_capacity == 0 || sample_repeat_capacity == 0 {
            return Err(HipError::new(
                0,
                "decode batch size, lane capacity, and repeat capacity must be non-zero",
            ));
        }
        if cfg.num_experts != 0 {
            return Err(HipError::new(
                0,
                "Lfm2DecodeBatchState: MoE configs not supported (dense only, num_experts==0)",
            ));
        }
        if cfg.hidden_size == 0
            || cfg.num_hidden_layers == 0
            || cfg.vocab_size == 0
            || cfg.head_dim == 0
            || cfg.num_attention_heads == 0
            || cfg.num_key_value_heads == 0
            || cfg.conv_kernel_size < 2
        {
            return Err(HipError::new(
                0,
                "Lfm2DecodeBatchState: zero or degenerate geometry",
            ));
        }
        // LDS check and full overflow validation BEFORE any GPU allocation.
        gpu.ensure_attention_q8_0_kv_independent_lds(lane_capacity, cfg.head_dim)?;

        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let dense_inter = cfg.intermediate_size;
        if dense_inter == 0 {
            return Err(HipError::new(
                0,
                "Lfm2DecodeBatchState: dense intermediate_size must be non-zero for dense model",
            ));
        }
        // All max_batch products that will be allocated as [N] tensors — fail fast before any GPU alloc.
        let total_capacity = max_batch
            .checked_mul(lane_capacity)
            .ok_or_else(|| HipError::new(0, "decode batch KV capacity multiplication overflow"))?;
        let repeat_tokens_len = max_batch
            .checked_mul(sample_repeat_capacity)
            .ok_or_else(|| {
                HipError::new(0, "decode batch repeat capacity multiplication overflow")
            })?;
        // Validate that layer counts are self-consistent; total_capacity already checked.
        let _ = total_capacity
            .checked_mul(cfg.num_key_value_heads)
            .ok_or_else(|| HipError::new(0, "KV head capacity overflow"))?;
        let _ = total_capacity
            .checked_mul(cfg.head_dim)
            .ok_or_else(|| HipError::new(0, "KV dim capacity overflow"))?;
        // Conv hist.
        let k = cfg.conv_kernel_size;
        let hist = hidden
            .checked_mul(k - 1)
            .ok_or_else(|| HipError::new(0, "conv hist size overflow"))?;
        let conv_hist_per_layer = max_batch
            .checked_mul(hist)
            .ok_or_else(|| HipError::new(0, "conv batch hist size overflow"))?;
        // All batched scratch products — checked before any allocation.
        let x_len = max_batch
            .checked_mul(hidden)
            .ok_or_else(|| HipError::new(0, "x_batch size overflow"))?;
        let bcx_len = max_batch
            .checked_mul(3)
            .and_then(|v| v.checked_mul(hidden))
            .ok_or_else(|| HipError::new(0, "conv_bcx size overflow"))?;
        let q_len = max_batch
            .checked_mul(q_dim)
            .ok_or_else(|| HipError::new(0, "fa_q size overflow"))?;
        let kv_len = max_batch
            .checked_mul(kv_dim)
            .ok_or_else(|| HipError::new(0, "fa_kv size overflow"))?;
        let gate_len = max_batch
            .checked_mul(dense_inter)
            .ok_or_else(|| HipError::new(0, "dense gate size overflow"))?;
        let logits_len = max_batch
            .checked_mul(cfg.vocab_size)
            .ok_or_else(|| HipError::new(0, "logits size overflow"))?;
        let sample_out_len = max_batch
            .checked_mul(2)
            .ok_or_else(|| HipError::new(0, "sample_out size overflow"))?;
        // Touch each computed len to silence unused warning and prove validation.
        let _ = (
            x_len,
            bcx_len,
            q_len,
            kv_len,
            gate_len,
            logits_len,
            sample_out_len,
        );
        let _ = conv_hist_per_layer;
        let _ = repeat_tokens_len;

        // `AttnWeights::kv_idx` is compact over attention layers only. Keep the
        // batch cache in that same index space; global mixer-layer placeholders
        // would let a compact index land on a one-element Conv placeholder.
        let is_kv_layer = batch_kv_slot_mask(&cfg.layer_types);

        let kv = match KvCache::new_gpu_q8_filtered(
            gpu,
            &is_kv_layer,
            cfg.num_key_value_heads,
            cfg.head_dim,
            total_capacity,
        ) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        // Conv states: per conv layer [conv_hist_per_layer] F32 zeroed (already validated).
        let n_conv = cfg.num_conv_layers();
        let mut conv_states: Vec<GpuTensor> = Vec::with_capacity(n_conv);
        for idx in 0..n_conv {
            let t = match gpu.zeros(&[conv_hist_per_layer], DType::F32) {
                Ok(v) => v,
                Err(e) => {
                    for prev in conv_states.drain(..) {
                        let _ = gpu.free_tensor(prev);
                    }
                    let _ = kv.free_gpu(gpu);
                    return Err(e);
                }
            };
            conv_states.push(t);
            let _ = idx;
        }
        // Allocation ledger for the remaining tensors (alias on error, drop on success).
        let mut ledger: Vec<GpuTensor> = Vec::with_capacity(24);
        macro_rules! alloc_zeros {
            ($shape:expr) => {
                match gpu.zeros($shape, DType::F32) {
                    Ok(t) => {
                        ledger.push(GpuTensor {
                            buf: unsafe { t.buf.alias() },
                            shape: t.shape.clone(),
                            dtype: t.dtype,
                        });
                        t
                    }
                    Err(e) => {
                        for prev in ledger.drain(..) {
                            let _ = gpu.free_tensor(prev);
                        }
                        for cs in conv_states.drain(..) {
                            let _ = gpu.free_tensor(cs);
                        }
                        let _ = kv.free_gpu(gpu);
                        return Err(e);
                    }
                }
            };
        }

        let x_batch = alloc_zeros!(&[max_batch * hidden]);
        let tmp_batch = alloc_zeros!(&[max_batch * hidden]);
        let conv_bcx_batch = alloc_zeros!(&[max_batch * 3 * hidden]);
        let conv_y_batch = alloc_zeros!(&[max_batch * hidden]);
        let fa_q_batch = alloc_zeros!(&[max_batch * q_dim]);
        let fa_k_batch = alloc_zeros!(&[max_batch * kv_dim]);
        let fa_v_batch = alloc_zeros!(&[max_batch * kv_dim]);
        let fa_attn_out_batch = alloc_zeros!(&[max_batch * q_dim]);
        let ffn_tmp_batch = alloc_zeros!(&[max_batch * hidden]);
        let dense_gate_batch = alloc_zeros!(&[max_batch * dense_inter]);
        let dense_up_batch = alloc_zeros!(&[max_batch * dense_inter]);
        let dense_act_batch = alloc_zeros!(&[max_batch * dense_inter]);
        let final_hidden = alloc_zeros!(&[max_batch * hidden]);
        let logits = alloc_zeros!(&[max_batch * cfg.vocab_size]);
        let lm_rot = alloc_zeros!(&[max_batch * hidden]);
        let max_proj_k = hidden.max(dense_inter);
        let proj_rot_len = max_batch
            .checked_mul(max_proj_k)
            .ok_or_else(|| HipError::new(0, "proj_rot size overflow"))?;
        let proj_rot = alloc_zeros!(&[proj_rot_len]);
        let sample_out = alloc_zeros!(&[max_batch * 2]);
        let sample_repeat_tokens = alloc_zeros!(&[repeat_tokens_len]);
        let sample_repeat_lengths = alloc_zeros!(&[max_batch]);
        let sample_rng_states = alloc_zeros!(&[max_batch]);
        let positions = alloc_zeros!(&[max_batch]);
        let tokens = alloc_zeros!(&[max_batch]);

        Ok(Self {
            max_batch,
            lane_capacity,
            sample_repeat_capacity,
            kv,
            conv_states,
            x_batch,
            tmp_batch,
            conv_bcx_batch,
            conv_y_batch,
            fa_q_batch,
            fa_k_batch,
            fa_v_batch,
            fa_attn_out_batch,
            ffn_tmp_batch,
            dense_gate_batch,
            dense_up_batch,
            dense_act_batch,
            final_hidden,
            logits,
            lm_rot,
            proj_rot,
            sample_out,
            sample_repeat_tokens,
            sample_repeat_lengths,
            sample_rng_states,
            positions,
            tokens,
        })
    }

    /// Clear all lanes and scratch.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        self.kv.clear_gpu(gpu)?;
        for cs in &self.conv_states {
            gpu.hip.memset(&cs.buf, 0, cs.buf.size())?;
        }
        for t in [
            &self.x_batch,
            &self.tmp_batch,
            &self.conv_bcx_batch,
            &self.conv_y_batch,
            &self.fa_q_batch,
            &self.fa_k_batch,
            &self.fa_v_batch,
            &self.fa_attn_out_batch,
            &self.ffn_tmp_batch,
            &self.dense_gate_batch,
            &self.dense_up_batch,
            &self.dense_act_batch,
            &self.final_hidden,
            &self.logits,
            &self.lm_rot,
            &self.proj_rot,
            &self.sample_out,
            &self.sample_repeat_tokens,
            &self.sample_repeat_lengths,
            &self.sample_rng_states,
            &self.positions,
            &self.tokens,
        ] {
            gpu.hip.memset(&t.buf, 0, t.buf.size())?;
        }
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv.free_gpu(gpu);
        for cs in self.conv_states {
            let _ = gpu.free_tensor(cs);
        }
        let _ = gpu.free_tensor(self.x_batch);
        let _ = gpu.free_tensor(self.tmp_batch);
        let _ = gpu.free_tensor(self.conv_bcx_batch);
        let _ = gpu.free_tensor(self.conv_y_batch);
        let _ = gpu.free_tensor(self.fa_q_batch);
        let _ = gpu.free_tensor(self.fa_k_batch);
        let _ = gpu.free_tensor(self.fa_v_batch);
        let _ = gpu.free_tensor(self.fa_attn_out_batch);
        let _ = gpu.free_tensor(self.ffn_tmp_batch);
        let _ = gpu.free_tensor(self.dense_gate_batch);
        let _ = gpu.free_tensor(self.dense_up_batch);
        let _ = gpu.free_tensor(self.dense_act_batch);
        let _ = gpu.free_tensor(self.final_hidden);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.lm_rot);
        let _ = gpu.free_tensor(self.proj_rot);
        let _ = gpu.free_tensor(self.sample_out);
        let _ = gpu.free_tensor(self.sample_repeat_tokens);
        let _ = gpu.free_tensor(self.sample_repeat_lengths);
        let _ = gpu.free_tensor(self.sample_rng_states);
        let _ = gpu.free_tensor(self.positions);
        let _ = gpu.free_tensor(self.tokens);
    }
}

fn batch_kv_slot_mask(layer_types: &[MixerKind]) -> Vec<bool> {
    let attention_slots = layer_types
        .iter()
        .filter(|&&kind| kind == MixerKind::Attention)
        .count()
        .max(1);
    vec![true; attention_slots]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_kv_slots_match_compact_attention_indices() {
        let layer_types = [
            MixerKind::Conv,
            MixerKind::Attention,
            MixerKind::Conv,
            MixerKind::Attention,
            MixerKind::Conv,
        ];
        let mask = batch_kv_slot_mask(&layer_types);
        assert_eq!(mask.len(), 2);
        assert!(mask.into_iter().all(|enabled| enabled));
    }

    #[test]
    fn batch_kv_slots_keep_zero_attention_shape_allocatable() {
        assert_eq!(batch_kv_slot_mask(&[MixerKind::Conv]), vec![true]);
    }
}
