// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Test-only synthetic GGUF writer and the 2-layer dense Qwen3.5 fixture
//! (layer 0 = GatedDeltaNet, layer 1 = gated full attention, optional MTP
//! block). Names and shapes are written out literally here, independently of
//! the name map under test.

use hipfire_runtime::gguf::GgmlType;
use std::io::Write;
use std::path::Path;

pub const DIM: usize = 256;
pub const FF: usize = 512;
pub const VOCAB: usize = 512;

#[derive(Clone, Debug)]
pub enum Kv {
    U32(u32),
    F32(f32),
    Str(String),
    Bool(bool),
    I32Arr(Vec<i32>),
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    /// ggml `ne` order (fastest first).
    pub ne: Vec<usize>,
    /// Raw ggml type id (so a test can write an unknown one).
    pub ty: u32,
}

impl Spec {
    pub fn new(name: &str, ne: &[usize], ty: GgmlType) -> Self {
        Self {
            name: name.to_string(),
            ne: ne.to_vec(),
            ty: ty.ggml_id(),
        }
    }

    /// Stored bytes (0 for an id this reader does not know).
    pub fn bytes(&self) -> usize {
        match GgmlType::from_u32(self.ty) {
            Some(t) => {
                let row = t.row_bytes(self.ne[0]).unwrap_or(0);
                self.ne[1..].iter().fold(row, |a, &d| a * d)
            }
            None => 0,
        }
    }

    pub fn params(&self) -> usize {
        self.ne.iter().product()
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_kv_value(out: &mut Vec<u8>, v: &Kv) {
    match v {
        Kv::U32(x) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Kv::F32(x) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Kv::Bool(x) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(u8::from(*x));
        }
        Kv::Str(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            put_str(out, s);
        }
        Kv::I32Arr(a) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&(a.len() as u64).to_le_bytes());
            for x in a {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
}

/// Write a GGUF v3 file. `with_data = false` writes the header only (the
/// shape of the S0 header fixtures): offsets stay valid but the file ends
/// before the tensor data.
pub fn write_gguf(path: &Path, kvs: &[(String, Kv)], tensors: &[Spec], with_data: bool) {
    const ALIGN: usize = 32;
    let mut head = Vec::new();
    head.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        put_str(&mut head, k);
        put_kv_value(&mut head, v);
    }
    let mut offset = 0usize;
    for t in tensors {
        put_str(&mut head, &t.name);
        head.extend_from_slice(&(t.ne.len() as u32).to_le_bytes());
        for d in &t.ne {
            head.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        head.extend_from_slice(&t.ty.to_le_bytes());
        head.extend_from_slice(&(offset as u64).to_le_bytes());
        offset = (offset + t.bytes()).div_ceil(ALIGN) * ALIGN;
    }
    let padded = head.len().div_ceil(ALIGN) * ALIGN;
    head.resize(padded, 0);
    let mut f = std::fs::File::create(path).expect("create gguf");
    f.write_all(&head).unwrap();
    if with_data {
        f.write_all(&vec![0u8; offset]).unwrap();
    }
}

fn kv(k: &str, v: Kv) -> (String, Kv) {
    (k.to_string(), v)
}

/// KV metadata of the 2-layer fixture (`general.architecture = qwen35`).
pub fn base_kvs(arch: &str, nextn: u32) -> Vec<(String, Kv)> {
    let mut v = vec![
        kv("general.architecture", Kv::Str(arch.to_string())),
        kv("qwen35.block_count", Kv::U32(2 + nextn)),
        kv("qwen35.embedding_length", Kv::U32(DIM as u32)),
        kv("qwen35.feed_forward_length", Kv::U32(FF as u32)),
        kv("qwen35.attention.head_count", Kv::U32(4)),
        kv("qwen35.attention.head_count_kv", Kv::U32(2)),
        kv("qwen35.attention.key_length", Kv::U32(64)),
        kv("qwen35.attention.value_length", Kv::U32(64)),
        kv("qwen35.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
        kv("qwen35.rope.dimension_count", Kv::U32(16)),
        kv("qwen35.rope.freq_base", Kv::F32(10_000_000.0)),
        kv(
            "qwen35.rope.dimension_sections",
            Kv::I32Arr(vec![2, 3, 3, 0]),
        ),
        kv("qwen35.ssm.conv_kernel", Kv::U32(4)),
        kv("qwen35.ssm.state_size", Kv::U32(32)),
        kv("qwen35.ssm.group_count", Kv::U32(2)),
        kv("qwen35.ssm.time_step_rank", Kv::U32(4)),
        kv("qwen35.ssm.inner_size", Kv::U32(128)),
        kv("qwen35.full_attention_interval", Kv::U32(2)),
        kv("tokenizer.ggml.eos_token_id", Kv::U32(7)),
    ];
    if nextn > 0 {
        v.push(kv("qwen35.nextn_predict_layers", Kv::U32(nextn)));
    }
    v
}

fn s(name: &str, ne: &[usize], ty: GgmlType) -> Spec {
    Spec::new(name, ne, ty)
}

/// The canonical tensor list of the 2-layer fixture, with mixed types the way
/// a per-tensor-mixed ("UD") file has them. `output.weight` is included.
pub fn base_tensors() -> Vec<Spec> {
    use GgmlType::*;
    vec![
        s("token_embd.weight", &[DIM, VOCAB], Q8_0),
        s("output_norm.weight", &[DIM], F32),
        s("output.weight", &[DIM, VOCAB], Q6K),
        // layer 0: GatedDeltaNet
        s("blk.0.attn_norm.weight", &[DIM], F32),
        s("blk.0.post_attention_norm.weight", &[DIM], F32),
        s("blk.0.attn_qkv.weight", &[DIM, 256], Q4K),
        s("blk.0.attn_gate.weight", &[DIM, 128], IQ4XS),
        s("blk.0.ssm_conv1d.weight", &[4, 256], F32),
        s("blk.0.ssm_dt.bias", &[4], F32),
        s("blk.0.ssm_a", &[4], F32),
        s("blk.0.ssm_beta.weight", &[DIM, 4], F16),
        s("blk.0.ssm_alpha.weight", &[DIM, 4], F16),
        s("blk.0.ssm_norm.weight", &[32], F32),
        s("blk.0.ssm_out.weight", &[128, DIM], Q8_0),
        s("blk.0.ffn_gate.weight", &[DIM, FF], Q4K),
        s("blk.0.ffn_up.weight", &[DIM, FF], Q4K),
        s("blk.0.ffn_down.weight", &[FF, DIM], Q5K),
        // layer 1: gated full attention
        s("blk.1.attn_norm.weight", &[DIM], F32),
        s("blk.1.post_attention_norm.weight", &[DIM], F32),
        s("blk.1.attn_q.weight", &[DIM, 512], Q4K),
        s("blk.1.attn_k.weight", &[DIM, 128], Q4K),
        s("blk.1.attn_v.weight", &[DIM, 128], Q6K),
        s("blk.1.attn_output.weight", &[256, DIM], Q5K),
        s("blk.1.attn_q_norm.weight", &[64], F32),
        s("blk.1.attn_k_norm.weight", &[64], F32),
        s("blk.1.ffn_gate.weight", &[DIM, FF], Q4K),
        s("blk.1.ffn_up.weight", &[DIM, FF], Q4K),
        s("blk.1.ffn_down.weight", &[FF, DIM], Q5K),
    ]
}

/// The MTP block (`blk.2`): a full-attention block plus the NextN tensors.
pub fn mtp_tensors() -> Vec<Spec> {
    use GgmlType::*;
    vec![
        s("blk.2.attn_norm.weight", &[DIM], F32),
        s("blk.2.post_attention_norm.weight", &[DIM], F32),
        s("blk.2.attn_q.weight", &[DIM, 512], Q4K),
        s("blk.2.attn_k.weight", &[DIM, 128], Q4K),
        s("blk.2.attn_v.weight", &[DIM, 128], Q6K),
        s("blk.2.attn_output.weight", &[256, DIM], Q5K),
        s("blk.2.attn_q_norm.weight", &[64], F32),
        s("blk.2.attn_k_norm.weight", &[64], F32),
        s("blk.2.ffn_gate.weight", &[DIM, FF], Q4K),
        s("blk.2.ffn_up.weight", &[DIM, FF], Q4K),
        s("blk.2.ffn_down.weight", &[FF, DIM], Q5K),
        s("blk.2.nextn.eh_proj.weight", &[2 * DIM, DIM], Q8_0),
        s("blk.2.nextn.enorm.weight", &[DIM], F32),
        s("blk.2.nextn.hnorm.weight", &[DIM], F32),
        s("blk.2.nextn.shared_head_norm.weight", &[DIM], F32),
    ]
}
