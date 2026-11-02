// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor-name resolution and source access for the qwen35-embed checkpoint.
//!
//! Two source kinds, two access conventions:
//!
//! - **A safetensors dir** (`ModelSource::Dir`) reads through the
//!   `ModelSource` trait, which returns a borrowed slice into the shard mmap.
//! - **An `.hfq`** must use `HfqFile`'s **native** methods: the trait impl
//!   deliberately returns `None` from `tensor_data`/`tensor_info` ("the HFQ
//!   path continues to use HfqFile's native methods"), so a loader that only
//!   knows the trait silently sees every tensor as missing. And the native
//!   `tensor_data` itself returns `None` once the mmap is dropped (the UMA
//!   path), so the read must fall back to the owned `tensor_data_vec`.
//!
//! [`read`] is the single entry point that hides both: it tries the
//! candidates in order and always returns owned-or-borrowed bytes plus the
//! storage descriptor, so callers never branch on the source kind.
//!
//! The pplx-embed-v2 release stores the text backbone under
//! `language_model.*` and the tower under `visual.*` (no `model.` wrapper —
//! `Qwen3_5Model` is the bare composite, unlike `Qwen3_5ForCausalLM` whose
//! tensors live under `model.*`). Exports that do carry the `model.` wrapper
//! are accepted too, so the candidate lists try both prefixes in order.

use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::model_source::ModelSource as _;
use std::borrow::Cow;

/// Checkpoint-relative name of the sentence-transformers Dense weight.
pub const DENSE_TENSOR: &str = "linear.weight";
/// Dir-relative path of the Dense weight's safetensors shard.
pub const DENSE_SHARD: &str = "1_Dense/model.safetensors";

/// One tensor's storage descriptor + bytes. `bytes` is borrowed when the
/// source can lend them (a live mmap) and owned when it must copy (a dropped
/// mmap / the pread path).
pub struct TensorBytes<'a> {
    /// HFQ quant-type byte; `0xFF` for a safetensors dir (dtype-driven).
    pub quant_type: u8,
    /// Safetensors dtype string (`"F32"`/`"F16"`/`"BF16"`); empty for an
    /// `.hfq`, where the quant-type byte carries the same information.
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Cow<'a, [u8]>,
}

impl TensorBytes<'_> {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Candidate source names for a backbone tensor given its bare
/// `language_model.`-stripped name (`embed_tokens.weight`,
/// `layers.0.self_attn.q_proj.weight`, …), most-specific first.
pub fn tensor_candidates(name: &str) -> Vec<String> {
    vec![
        format!("language_model.{name}"),
        format!("model.language_model.{name}"),
        name.to_string(),
    ]
}

/// Candidate source names for a vision-tower tensor given its bare
/// `visual.`-stripped name (`blocks.0.attn.qkv.weight`, …).
pub fn vision_tensor_candidates(name: &str) -> Vec<String> {
    vec![
        format!("visual.{name}"),
        format!("model.visual.{name}"),
        name.to_string(),
    ]
}

/// Read a tensor by bare name from either source kind (see module docs).
///
/// The caller must decode/scale the returned bytes BEFORE calling
/// [`release`] — the release madvise drops the mmap's resident pages, so a
/// release inside this function (before the decode has touched them) would
/// throw away nothing and force a re-fault of every byte the decode reads.
pub fn read<'a>(src: &'a ModelSource, name: &str, vision: bool) -> Result<TensorBytes<'a>, String> {
    let candidates = if vision {
        vision_tensor_candidates(name)
    } else {
        tensor_candidates(name)
    };
    match src {
        ModelSource::Dir(d) => {
            for cand in &candidates {
                if let Some((info, bytes)) = d.tensor_data(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: info.dtype.clone(),
                        shape: info.shape.clone(),
                        bytes: Cow::Borrowed(bytes),
                    });
                }
            }
        }
        ModelSource::Hfq(h) => {
            for cand in &candidates {
                // Borrowed first (cheap); the owned read is the documented
                // fallback when the mmap is gone.
                if let Some((info, bytes)) = h.tensor_data(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: String::new(),
                        shape: info.shape.iter().map(|v| *v as usize).collect(),
                        bytes: Cow::Borrowed(bytes),
                    });
                }
                if let Some((info, vec)) = h.tensor_data_vec(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: String::new(),
                        shape: info.shape.iter().map(|v| *v as usize).collect(),
                        bytes: Cow::Owned(vec),
                    });
                }
            }
        }
    }
    Err(format!(
        "qwen35-embed: missing tensor `{name}` (tried {candidates:?})"
    ))
}

/// Release the pages a [`read`] pulled in, once the caller has consumed the
/// bytes. Mirrors `read`'s candidate order; the first name the source holds
/// is released — the same one `read` would have returned. Best-effort and
/// advisory: this is `MADV_DONTNEED` on a read-only file mapping.
pub fn release(src: &ModelSource, name: &str, vision: bool) {
    let candidates = if vision {
        vision_tensor_candidates(name)
    } else {
        tensor_candidates(name)
    };
    let source: &dyn hipfire_runtime::model_source::ModelSource = match src {
        ModelSource::Dir(d) => d,
        // HfqSource's release is a no-op today (borrowed mmap / owned vec);
        // routing it through the same call keeps a future impl effective.
        ModelSource::Hfq(h) => h,
    };
    for cand in &candidates {
        if source.tensor_info(cand).is_some() {
            source.release_tensor_pages(cand);
            return;
        }
    }
}

/// Bare backbone names every checkpoint must provide (layer index elided).
pub fn required_text_tensors() -> Vec<String> {
    let mut out = vec!["embed_tokens.weight".to_string(), "norm.weight".to_string()];
    for n in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ] {
        out.push(format!("layers.N.{n}"));
    }
    for n in [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.conv1d.weight",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
    ] {
        out.push(format!("layers.N.{n}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_prefer_bare_composite_prefix() {
        assert_eq!(
            tensor_candidates("layers.0.self_attn.q_proj.weight"),
            vec![
                "language_model.layers.0.self_attn.q_proj.weight".to_string(),
                "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
                "layers.0.self_attn.q_proj.weight".to_string(),
            ]
        );
        assert_eq!(
            vision_tensor_candidates("patch_embed.proj.weight")[0],
            "visual.patch_embed.proj.weight"
        );
    }
}
