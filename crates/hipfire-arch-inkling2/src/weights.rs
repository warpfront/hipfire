// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validated view of the Inkling-2 weights inside an HFQ file.
//!
//! Phase 1 stores physical weights once (24 layers) with per-expert tensors
//! for the routed MoE (128 experts × gate_up/down per layer). This module
//! indexes and validates that set by name/shape, then hands raw `quant_type`
//! + bytes to the forward/GPU upload layer. No numerics live here.

use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use std::collections::HashMap;

/// A source of decoded f32 tensors for the CPU forward. Implemented by the
/// quantized HFQ index and by the raw (unquantized) safetensors checkpoint.
pub trait WeightSource {
    fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String>;
}

/// A single raw HFQ tensor: metadata plus a copy of its encoded bytes.
#[derive(Clone)]
pub struct RawTensor {
    pub info: HfqTensorInfo,
    pub data: Vec<u8>,
}

/// Index of every Inkling-2 tensor in the trunk HFQ, keyed by verbatim HFQ name.
pub struct Inkling2Weights {
    pub tensors: HashMap<String, RawTensor>,
    /// Lazily decoded f32 weights (a 48-visit forward reuses each tensor many times).
    f32_cache: std::cell::RefCell<HashMap<String, (Vec<usize>, Vec<f32>)>>,
}

impl WeightSource for Inkling2Weights {
    fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        Inkling2Weights::f32(self, name)
    }
}

impl Inkling2Weights {
    /// Load and index all tensors, validating the expected physical-layer and
    /// per-expert structure against `cfg`.
    pub fn load(hfq: &HfqFile, cfg: &crate::config::Inkling2Config) -> Result<Self, String> {
        let mut tensors = HashMap::new();
        for info in hfq.tensor_infos() {
            let (i, data) = hfq
                .tensor_data_vec(&info.name)
                .ok_or_else(|| format!("inkling2: cannot read tensor {}", info.name))?;
            tensors.insert(
                info.name.clone(),
                RawTensor {
                    info: i.clone(),
                    data: data.to_vec(),
                },
            );
        }

        // Structural validation: every physical layer must carry its router,
        // norms and the full 128-expert set. Fail closed on a partial artifact.
        for il in 0..cfg.n_phys {
            let p = format!("model.llm.layers.{il}.");
            // Every layer has router, MLP norm and attention output norm;
            // SWA layers use `attn_norm`, full layers use three input norms.
            for suffix in ["mlp.gate.weight", "mlp_norm.weight"] {
                let name = format!("{p}{suffix}");
                if !tensors.contains_key(&name) {
                    return Err(format!("inkling2: missing {name}"));
                }
            }
            let attn_norm = if cfg.swa_layers[il] {
                "attn_norm.weight"
            } else {
                "q_input_norm.weight"
            };
            let name = format!("{p}{attn_norm}");
            if !tensors.contains_key(&name) {
                return Err(format!("inkling2: missing {name}"));
            }
            for e in 0..cfg.n_experts {
                for leaf in ["gate_up_proj.weight", "down_proj.weight"] {
                    let name = format!("{p}mlp.experts.{e}.{leaf}");
                    let t = tensors
                        .get(&name)
                        .ok_or_else(|| format!("inkling2: missing {name}"))?;
                    if t.info.shape.len() != 2 {
                        return Err(format!("inkling2: {name} not 2-D: {:?}", t.info.shape));
                    }
                }
            }
        }
        for name in ["model.llm.embed.weight", "model.llm.norm.weight"] {
            if !tensors.contains_key(name) {
                return Err(format!("inkling2: missing {name}"));
            }
        }

        Ok(Self {
            tensors,
            f32_cache: std::cell::RefCell::new(HashMap::new()),
        })
    }

    pub fn get(&self, name: &str) -> Option<&RawTensor> {
        self.tensors.get(name)
    }

    /// Decode a named weight to true (un-rotated) f32 values. Uses hipfire's
    /// public HFQ decoder, which inverse-FWHTs the MQ4G256/G256V2 formats so
    /// the result is the original weight matrix (row-major, `shape`).
    pub fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        if let Some(cached) = self.f32_cache.borrow().get(name) {
            return Ok(cached.clone());
        }
        let t = self
            .get(name)
            .ok_or_else(|| format!("inkling2: tensor not found: {name}"))?;
        let shape: Vec<usize> = t.info.shape.iter().map(|&d| d as usize).collect();
        let n: usize = shape.iter().product();
        let vals = hipfire_runtime::weight_backend::dequantize_weight_to_f32(
            t.info.quant_type,
            &t.data,
            n,
        );
        if vals.len() != n {
            return Err(format!(
                "inkling2: decoded {name} to {} values, expected {n}",
                vals.len()
            ));
        }
        let result = (shape, vals);
        self.f32_cache
            .borrow_mut()
            .insert(name.to_string(), result.clone());
        Ok(result)
    }

    /// Expected expert tensor count: n_phys × n_experts × 2 (gate_up + down).
    pub fn expected_expert_tensors(cfg: &crate::config::Inkling2Config) -> usize {
        cfg.n_phys * cfg.n_experts * 2
    }

    pub fn expert_tensor_count(&self) -> usize {
        self.tensors
            .keys()
            .filter(|n| n.contains(".mlp.experts.") && n.ends_with(".weight"))
            .count()
    }
}

/// A weight source backed directly by the raw HF safetensors checkpoint
/// (BF16/F32, no quantization). Used for the decisive parity experiment:
/// run the CPU forward on unquantized weights and compare to llama.cpp.
pub struct RawWeights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl RawWeights {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
        let st = safetensors::SafeTensors::deserialize(&mmap).map_err(|e| format!("parse: {e}"))?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let n: usize = shape.iter().product();
            let vals: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::BF16 => view
                    .data()
                    .chunks_exact(2)
                    .map(|c| {
                        let b = u16::from_le_bytes([c[0], c[1]]);
                        half::bf16::from_bits(b).to_f32()
                    })
                    .collect(),
                safetensors::Dtype::F16 => view
                    .data()
                    .chunks_exact(2)
                    .map(|c| {
                        let b = u16::from_le_bytes([c[0], c[1]]);
                        half::f16::from_bits(b).to_f32()
                    })
                    .collect(),
                other => return Err(format!("unsupported dtype {other} for {name}")),
            };
            if vals.len() != n {
                return Err(format!("{name}: {} values != {n}", vals.len()));
            }
            tensors.insert(name, (shape, vals));
        }
        Ok(Self { tensors })
    }
}

impl WeightSource for RawWeights {
    fn f32(&self, hfq_name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        // Per-expert HFQ names `...experts.E.gate_up_proj.weight` /
        // `down_proj.weight` correspond to stacked checkpoint tensors
        // `...experts.w13_weight` [E, 2*inter, hidden] and
        // `...experts.w2_weight` [E, hidden, inter]. Slice out one expert and
        // regroup w13 interleaved [g0,u0,g1,u1,...] to [g...,u...].
        if let Some(rest) = hfq_name.strip_suffix(".gate_up_proj.weight") {
            if let Some(pos) = rest.rfind(".experts.") {
                let e: usize = rest[pos + ".experts.".len()..]
                    .parse()
                    .map_err(|_| "bad expert idx")?;
                let base = &rest[..pos + ".experts.".len()];
                let (shape, v) = self
                    .tensors
                    .get(&format!("{base}w13_weight"))
                    .ok_or("missing w13_weight")?;
                let ne = shape[0];
                let rows = shape[1];
                let cols = shape[2];
                let inter = rows / 2;
                let mut out = vec![0.0f32; rows * cols];
                let off = e * rows * cols;
                for j in 0..inter {
                    for c in 0..cols {
                        out[j * cols + c] = v[off + (2 * j) * cols + c];
                        out[(inter + j) * cols + c] = v[off + (2 * j + 1) * cols + c];
                    }
                }
                return Ok((vec![rows, cols], out));
            }
        }
        if let Some(rest) = hfq_name.strip_suffix(".down_proj.weight") {
            if let Some(pos) = rest.rfind(".experts.") {
                let e: usize = rest[pos + ".experts.".len()..]
                    .parse()
                    .map_err(|_| "bad expert idx")?;
                let base = &rest[..pos + ".experts.".len()];
                let (shape, v) = self
                    .tensors
                    .get(&format!("{base}w2_weight"))
                    .ok_or("missing w2_weight")?;
                let ne = shape[0];
                let per = shape[1] * shape[2];
                let out = v[e * per..(e + 1) * per].to_vec();
                let _ = ne;
                return Ok((vec![shape[1], shape[2]], out));
            }
        }
        self.tensors
            .get(hfq_name)
            .cloned()
            .ok_or_else(|| format!("raw: tensor not found: {hfq_name}"))
    }
}

/// Weight source backed by the F32 GGUF exported to a simple binary
/// (`gguf_to_f32_bin.py`): `[n][ (u32 name_len, name, u32 ndim, dims, f32 data) ]`.
/// All tensors are already dequantized F32 — the parity oracle's exact weights.
pub struct GgufF32 {
    path: std::path::PathBuf,
    /// name -> (shape, byte offset of data in file)
    index: HashMap<String, (Vec<usize>, u64)>,
}

impl GgufF32 {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf).map_err(|e| format!("read count: {e}"))?;
        let n = u32::from_le_bytes(buf);
        let mut index = HashMap::new();
        for _ in 0..n {
            f.read_exact(&mut buf).map_err(|e| format!("read name len: {e}"))?;
            let nl = u32::from_le_bytes(buf) as usize;
            let mut nb = vec![0u8; nl];
            f.read_exact(&mut nb).map_err(|e| format!("read name: {e}"))?;
            let name = String::from_utf8(nb).map_err(|e| format!("name: {e}"))?;
            f.read_exact(&mut buf).map_err(|e| format!("read ndim: {e}"))?;
            let nd = u32::from_le_bytes(buf) as usize;
            let mut db = vec![0u8; nd * 4];
            f.read_exact(&mut db).map_err(|e| format!("read dims: {e}"))?;
            let shape: Vec<usize> = db.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]]) as usize).collect();
            let len: usize = shape.iter().product();
            let off = f.stream_position().map_err(|e| format!("pos: {e}"))?;
            f.seek(SeekFrom::Current((len * 4) as i64)).map_err(|e| format!("seek: {e}"))?;
            index.insert(name, (shape, off));
        }
        Ok(Self { path: path.to_path_buf(), index })
    }

    fn read(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        use std::io::{Read, Seek, SeekFrom};
        let (shape, off) = self.index.get(name).cloned()
            .ok_or_else(|| format!("gguf: missing {name}"))?;
        let len: usize = shape.iter().product();
        let mut f = std::fs::File::open(&self.path).map_err(|e| format!("open: {e}"))?;
        f.seek(SeekFrom::Start(off)).map_err(|e| format!("seek: {e}"))?;
        let mut fb = vec![0u8; len * 4];
        f.read_exact(&mut fb).map_err(|e| format!("read {name}: {e}"))?;
        let vals: Vec<f32> = fb.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        Ok((shape, vals))
    }
}

impl WeightSource for GgufF32 {
    fn f32(&self, hfq: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        // map HFQ (checkpoint) name -> GGUF name
        let get = |n: &str| -> Result<(Vec<usize>, Vec<f32>), String> { self.read(n) };
        if hfq == "model.llm.embed.weight" {
            return get("token_embd.weight");
        }
        if hfq == "model.llm.embed_norm.weight" {
            return self.read("token_embd_norm.weight")
                .or_else(|_| self.read("token_norm.weight"));
        }
        if hfq == "model.llm.norm.weight" {
            return get("output_norm.weight");
        }
        if hfq == "model.llm.unembed.weight" {
            return get("output.weight");
        }

        // per-expert
        if let Some(rest) = hfq.strip_prefix("model.llm.layers.") {
            // layers.N....
            let dot = rest.find('.').ok_or("bad layer name")?;
            let layer: usize = rest[..dot].parse().map_err(|_| "bad layer")?;
            let suffix = &rest[dot + 1..];
            let b = format!("blk.{layer}.");
            if let Some(e) = suffix.strip_prefix("mlp.experts.") {
                // mlp.experts.E.gate_up_proj.weight / down_proj.weight
                let dot2 = e.find('.').ok_or("bad expert")?;
                let ex: usize = e[..dot2].parse().map_err(|_| "bad expert idx")?;
                if e.ends_with("gate_up_proj.weight") {
                    // GGUF [2*inter, hidden, n_exp], expert on last axis.
                    let (shape, v) = get(&format!("{b}ffn_gate_up_exps.weight"))?;
                    let rows = shape[0];
                    let cols = shape[1];
                    let nexp = shape[2];
                    let mut out = vec![0.0f32; rows * cols];
                    for i in 0..rows { for j in 0..cols {
                        out[i * cols + j] = v[(i * cols + j) * nexp + ex];
                    }}
                    return Ok((vec![rows, cols], out));
                }
                if e.ends_with("down_proj.weight") {
                    // GGUF [inter, hidden, n_exp] expert-last -> transpose to
                    // [hidden, inter] for the matvec.
                    let (shape, v) = get(&format!("{b}ffn_down_exps.weight"))?;
                    let inter = shape[0];
                    let hidden = shape[1];
                    let nexp = shape[2];
                    let mut out = vec![0.0f32; hidden * inter];
                    for i in 0..inter { for j in 0..hidden {
                        out[j * inter + i] = v[(i * hidden + j) * nexp + ex];
                    }}
                    return Ok((vec![hidden, inter], out));
                }
            }
            let map = |s: &str| -> String {
                // controllers.{attn,k,v,mlp}.{down,gain,forget}_weight
                for (short, tag) in [("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "ffn")] {
                    for (kind, ktag) in [("down", "down"), ("gain", "gain"), ("forget", "forget")] {
                        let key = format!("controllers.{short}.{kind}_weight");
                        if s == key {
                            return format!("{b}loop_{tag}_ctrl_{ktag}.weight");
                        }
                    }
                }
                match s {
                    "attn.wq_du.weight" => format!("{b}attn_q.weight"),
                    "attn.wk_dv.weight" => format!("{b}attn_k.weight"),
                    "attn.wv_dv.weight" => format!("{b}attn_v.weight"),
                    "attn.wr_du.weight" => format!("{b}attn_rel.weight"),
                    "attn.gate_attn.weight" => format!("{b}attn_gate.weight"),
                    "attn.wo_ud.weight" => format!("{b}attn_output.weight"),
                    "attn.attn_sink_proj.weight" => format!("{b}attn_sink_proj.weight"),
                    "attn.rel_logits_proj.proj" => format!("{b}attn_rel_bias.weight"),
                    "attn.q_l2_clip.scale" => format!("{b}attn_q_clip.weight"),
                    "attn.rel_l2_clip.scale" => format!("{b}attn_rel_clip.weight"),
                    "attn_norm.weight" => format!("{b}attn_norm.weight"),
                    "attn.k_sconv.weight" | "attn.k_sconv.static_w" => {
                        format!("{b}attn_k_conv.weight")
                    }
                    "attn.v_sconv.weight" | "attn.v_sconv.static_w" => {
                        format!("{b}attn_v_conv.weight")
                    }
                    "attn_sconv.weight" => format!("{b}attn_post_conv.weight"),
                    "mlp_sconv.weight" => format!("{b}ffn_post_conv.weight"),
                    "mlp.gate.weight" => format!("{b}ffn_gate_inp.weight"),
                    "mlp.gate.bias" => format!("{b}exp_probs_b.bias"),
                    "mlp_norm.weight" => format!("{b}ffn_norm.weight"),
                    _ => format!("{b}{s}"),
                }
            };
            return get(&map(suffix));
        }
        // Cross-pass globals: GGUF stores per-layer loop_msg_query [n_embd,4]
        // for core layers 6..18. Reassemble the two (24,n_embd) globals that
        // pack_message_queries expects: q rows = [msg[..0], msg[..3]], kv rows
        // = [msg[..1], msg[..2]] per core layer.
        if hfq == "model.llm.middle_cycle_cross_pass_queries"
            || hfq == "model.llm.middle_cycle_cross_pass_kv_queries"
        {
            let is_q = hfq.ends_with("_queries");
            let e = 2304usize;
            let mut q_rows: Vec<f32> = Vec::with_capacity(24 * e);
            let mut kv_rows: Vec<f32> = Vec::with_capacity(24 * e);
            for layer in 6..18 {
                let (_, m) = get(&format!("blk.{layer}.loop_msg_query.weight"))?;
                // m layout [n_embd, 4] row-major => col c at m[c*e..(c+1)*e]
                q_rows.extend_from_slice(&m[0..e]);
                q_rows.extend_from_slice(&m[3 * e..4 * e]);
                kv_rows.extend_from_slice(&m[1 * e..2 * e]);
                kv_rows.extend_from_slice(&m[2 * e..3 * e]);
            }
            return Ok((vec![24, e], if is_q { q_rows } else { kv_rows }));
        }
        // global attn_res queries: attn_queries.{i} -> blk.{i+1}.attn_res_query
        if let Some(rest) = hfq.strip_prefix("model.llm.attn_res.attn_queries.") {
            let i: usize = rest.parse().map_err(|_| "bad query idx")?;
            return get(&format!("blk.{}.attn_res_query.weight", i + 1));
        }
        if let Some(rest) = hfq.strip_prefix("model.llm.attn_res.mlp_queries.") {
            let i: usize = rest.parse().map_err(|_| "bad query idx")?;
            return get(&format!("blk.{i}.ffn_res_query.weight"));
        }
        get(hfq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_expert_count_for_mini() {
        let cfg = crate::config::LoopConfig::mini();
        // Use the typed config shape indirectly: 24 phys × 128 experts × 2.
        assert_eq!(
            24 * 128 * 2,
            Inkling2Weights::expected_expert_tensors(&tiny_cfg())
        );
        let _ = cfg;
    }

    fn tiny_cfg() -> crate::config::Inkling2Config {
        // Reuse the metadata parser test's config by parsing the same JSON.
        crate::config::Inkling2Config::from_metadata_json(TEST_META).unwrap()
    }

    pub(crate) fn test_meta() -> &'static str {
        TEST_META
    }

    const TEST_META: &str = r#"{"config":{"eos_token_id":200006,"text_config":{
        "num_hidden_layers":24,"hidden_size":2304,"vocab_size":201024,
        "unpadded_vocab_size":200058,"num_attention_heads":24,"num_key_value_heads":8,
        "head_dim":128,"sliding_window_size":1024,"d_rel":16,"rel_extent":1024,
        "qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,
        "log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"sconv_kernel_size":4,
        "n_layers_per_attn_res_block":6,"final_logit_softcapping":30.0,
        "n_routed_experts":128,"num_experts_per_tok":8,"intermediate_size":1152,
        "route_scale":8.0,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"rms_norm_eps":1e-6,
        "local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
        "loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64,"is_varloop":false}
    }}}"#;

    #[test]
    #[ignore = "print decoded qt44 stats to compare with GGUF"]
    fn inspect_qt44_decode() {
        use hipfire_runtime::hfq::HfqFile;
        let hfq = HfqFile::open(std::path::Path::new(
            r"D:\hipfire-models\inkling2-mini-mq4.hfq",
        ))
        .expect("open");
        let w = Inkling2Weights::load(
            &hfq,
            &crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap(),
        )
        .unwrap();
        for name in [
            "model.llm.layers.0.attn.wk_dv.weight",
            "model.llm.layers.0.attn.wq_du.weight",
        ] {
            let (shape, v) = w.f32(name).unwrap();
            let min = v.iter().copied().fold(f32::INFINITY, f32::min);
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
            eprintln!("{name}: shape={shape:?} min={min} max={max} mean={mean}");
            let first: Vec<String> = v[..8].iter().map(|x| format!("{:.5}", x)).collect();
            eprintln!("  first8={:?}", first);
            if name.ends_with("wk_dv.weight") {
                let mut s = String::new();
                for x in &v {
                    s.push_str(&format!("{}\n", x.to_bits()));
                }
                std::fs::write(r"D:\hipfire-models\decoded-wk.txt", s).unwrap();
            }
        }
    }

    #[test]
    #[ignore = "requires the 14 GB inkling2-mini-mq4.hfq artifact"]
    fn loads_real_artifact() {
        let hfq = HfqFile::open(std::path::Path::new(
            r"D:\hipfire-models\inkling2-mini-mq4.hfq",
        ))
        .expect("open artifact");
        let cfg = tiny_cfg();
        let w = Inkling2Weights::load(&hfq, &cfg).expect("load weights");
        assert_eq!(
            w.expert_tensor_count(),
            Inkling2Weights::expected_expert_tensors(&cfg)
        );
        assert!(w.get("model.llm.embed.weight").is_some());
    }
}
