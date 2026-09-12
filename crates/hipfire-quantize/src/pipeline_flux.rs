// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FLUX diffusers-pipe → per-component HFQM packer.
//!
//! Turns a FLUX pipeline directory into per-component HFQ files with the arch
//! ids from `docs/architecture-ids.md`. Two family profiles, detected from
//! `transformer/config.json` (`_class_name Flux2Transformer2DModel` /
//! `model_type flux2` → klein):
//!
//! FLUX.1 (schnell/dev):
//!
//! | component   | pipe dir          | arch_id |
//! |-------------|-------------------|---------|
//! | transformer | `transformer/`      | 40      |
//! | t5          | `text_encoder_2/`   | 41      |
//! | clip        | `text_encoder/`     | 42      |
//! | vae         | `vae/`              | 43      |
//!
//! FLUX.2 (klein):
//!
//! | component   | pipe dir          | arch_id |
//! |-------------|-------------------|---------|
//! | transformer | `transformer/`      | 45      |
//! | qwen3       | `text_encoder/`     | 46      |
//! | vae         | `vae/`              | 43      |
//!
//! Dtype policy (matches what the arch loaders dispatch on — `decode_dtype` /
//! `F16Stage`: F32/BF16/F16):
//! - `.weight` (and every other GEMM operand + embedding table) → **F16**,
//!   converted once here with the crate's `f32_to_f16` (identical values to a
//!   device `(_Float16)` cast under the same rounding).
//! - `.bias` / `.scale` → **F32** untouched (they feed `bias_add_f32` /
//!   `rmsnorm_batched`, which read f32).
//!
//! Tensor NAMES pass through unchanged (BFL `double_blocks.*` or diffusers
//! `transformer_blocks.*`): the loader's `FluxPlan::detect` maps either layout.
//! Each HFQ file's metadata envelope carries its component's `config.json`
//! verbatim; the transformer file additionally embeds `scheduler_config.json`
//! and the T5/CLIP tokenizer blobs (they live on disk only in a pipe). The
//! loader (`hipfire-arch-diffusion` `load_pipe_hfq`) rebuilds the bundle from
//! exactly these metadata fields.
//!
//! T5 + VAE are pack-shared across FLUX.1 variants: pack them once and point
//! multiple registry entries (schnell/dev) at the same files, matching the
//! `t5`/`clip`/`vae` sidecar slots on `ModelEntry`.

use crate::hfq::{maybe_spill, write_hfq, HfqTensor, QuantType, TensorSpill};
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_runtime::model_source::ModelSource as _;
use hipfire_runtime::safetensors_source::{derive_arch_id, SafetensorsSource};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Image-generation component `arch_id`s. See `docs/architecture-ids.md`
/// § Image-generation component ids. Crate-local like `ARCH_ID_MAPLE`, so the
/// CPU-only packer stays free of the arch crates.
const ARCH_FLUX1_TRUNK: u32 = 40;
const ARCH_T5_SIDECAR: u32 = 41;
const ARCH_CLIP_SIDECAR: u32 = 42;
const ARCH_VAE_SIDECAR: u32 = 43;
const ARCH_FLUX2_TRUNK: u32 = 45;
const ARCH_QWEN3_TEXT_ENCODER: u32 = 46;

/// Spill in-memory tensor data to the spill file past this many bytes, so a
/// 24 GB transformer pack never holds two copies of the model.
const SPILL_THRESHOLD: usize = 2 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FluxComponent {
    Transformer,
    T5,
    Clip,
    Qwen3,
    Vae,
}

impl FluxComponent {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "transformer" => Some(Self::Transformer),
            "t5" => Some(Self::T5),
            "clip" => Some(Self::Clip),
            "qwen3" => Some(Self::Qwen3),
            "vae" => Some(Self::Vae),
            _ => None,
        }
    }

    fn subdir(self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::T5 => "text_encoder_2",
            Self::Clip | Self::Qwen3 => "text_encoder",
            Self::Vae => "vae",
        }
    }

    fn arch_id(self) -> u32 {
        match self {
            Self::Transformer => ARCH_FLUX1_TRUNK,
            Self::T5 => ARCH_T5_SIDECAR,
            Self::Clip => ARCH_CLIP_SIDECAR,
            Self::Qwen3 => ARCH_QWEN3_TEXT_ENCODER,
            Self::Vae => ARCH_VAE_SIDECAR,
        }
    }

    fn stem(self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::T5 => "t5",
            Self::Clip => "clip",
            Self::Qwen3 => "qwen3",
            Self::Vae => "vae",
        }
    }
}

/// FLUX family detected from the pipe's `transformer/config.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipeFamily {
    Flux1,
    Flux2,
}

impl PipeFamily {
    fn detect(pipe: &Path) -> Result<Self, String> {
        let cfg_path = pipe.join("transformer/config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .map_err(|e| format!("flux pack: {cfg_path:?}: {e}"))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("flux pack: {cfg_path:?} invalid: {e}"))?;
        let is_flux2 = v.get("_class_name").and_then(|c| c.as_str())
            == Some("Flux2Transformer2DModel")
            || v.get("model_type").and_then(|m| m.as_str()) == Some("flux2");
        Ok(if is_flux2 { Self::Flux2 } else { Self::Flux1 })
    }

    fn components(self) -> Vec<FluxComponent> {
        match self {
            Self::Flux1 => vec![
                FluxComponent::Transformer,
                FluxComponent::T5,
                FluxComponent::Clip,
                FluxComponent::Vae,
            ],
            // klein: no T5 / CLIP — a Qwen3 text encoder (arch 46) instead.
            Self::Flux2 => vec![
                FluxComponent::Transformer,
                FluxComponent::Qwen3,
                FluxComponent::Vae,
            ],
        }
    }
}

/// CLI entry: `--flux-pipe <dir> --output <base> [--flux-component all|…]`.
pub(crate) fn run_flux_pack(pipe: &Path, component: &str, output: &Path) -> Result<(), String> {
    if !pipe.is_dir() {
        return Err(format!("--flux-pipe {} is not a directory", pipe.display()));
    }
    let family = PipeFamily::detect(pipe)?;
    let components: Vec<FluxComponent> = match component {
        "all" => family.components(),
        other => vec![FluxComponent::from_name(other).ok_or_else(|| {
            format!(
                "unknown --flux-component '{other}' \
                 (expected transformer|t5|clip|qwen3|vae|all)"
            )
        })?],
    };
    for c in components {
        let out = if component == "all" {
            let stem = output
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("flux");
            output.with_file_name(format!("{stem}-{}.hfq", c.stem()))
        } else {
            output.to_path_buf()
        };
        pack_component(pipe, family, c, &out)?;
        eprintln!("flux pack: wrote {}", out.display());
    }
    Ok(())
}

fn pack_component(
    pipe: &Path,
    family: PipeFamily,
    component: FluxComponent,
    out: &Path,
) -> Result<(), String> {
    let sub = component.subdir();
    let dir = pipe.join(sub);
    let src =
        SafetensorsSource::open(&dir).map_err(|e| format!("flux pack: open {sub}/: {e:?}"))?;

    // The single pipe-level check that makes a mis-pack fail at pack time:
    // the parsed config family must match the detected profile (a FLUX.1
    // pack on a klein pipe or vice versa is a naming/rename bug, not a load
    // story we want discovered on the daemon).
    if component == FluxComponent::Transformer {
        let cfg: serde_json::Value = serde_json::from_str(src.metadata_json())
            .map_err(|e| format!("flux pack: transformer metadata invalid: {e}"))?;
        let cfg = cfg.get("config").cloned().unwrap_or(cfg);
        let parsed_is_flux2 = match derive_arch_id(&cfg) {
            ARCH_FLUX1_TRUNK => false,
            ARCH_FLUX2_TRUNK => true,
            other => {
                return Err(format!(
                    "flux pack: transformer/config.json does not resolve to a FLUX trunk \
                     (arch {other}); expected {ARCH_FLUX1_TRUNK} or {ARCH_FLUX2_TRUNK}"
                ))
            }
        };
        let expected_is_flux2 = family == PipeFamily::Flux2;
        if parsed_is_flux2 != expected_is_flux2 {
            return Err(format!(
                "flux pack: pipe family mismatch — transformer/config.json parses as {}, \
                 but the pipe layout is {} (arch {} vs {}); refusing to mis-pack",
                if parsed_is_flux2 {
                    "FLUX.2 (klein)"
                } else {
                    "FLUX.1"
                },
                if expected_is_flux2 {
                    "FLUX.2 (klein)"
                } else {
                    "FLUX.1"
                },
                if parsed_is_flux2 {
                    ARCH_FLUX2_TRUNK
                } else {
                    ARCH_FLUX1_TRUNK
                },
                if expected_is_flux2 {
                    ARCH_FLUX2_TRUNK
                } else {
                    ARCH_FLUX1_TRUNK
                },
            ));
        }
    }

    let names = src.tensor_names();
    if names.is_empty() {
        return Err(format!("flux pack: {sub}/ contains no tensors"));
    }
    let metadata = component_metadata(pipe, family, component, &src)?;

    let out_dir = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("flux pack: create {}: {e}", out_dir.display()))?;
    let mut spill = TensorSpill::new(&out_dir)
        .map_err(|e| format!("flux pack: spill in {}: {e}", out_dir.display()))?;

    let arch_id = match component {
        FluxComponent::Transformer if family == PipeFamily::Flux2 => ARCH_FLUX2_TRUNK,
        _ => component.arch_id(),
    };
    let mut tensors: Vec<HfqTensor> = Vec::new();
    for name in &names {
        let (info, bytes) = src
            .tensor_data(name)
            .ok_or_else(|| format!("flux pack: {sub}/ tensor {name} disappeared"))?;
        let shape: Result<Vec<u32>, String> = info
            .shape
            .iter()
            .map(|&s| u32::try_from(s).map_err(|_| format!("flux pack: {name} dim too large")))
            .collect();
        let shape = shape?;
        // BatchNorm bookkeeping (the FLUX.2 VAE's latent-norm `bn`): an I64
        // step counter that inference never reads. Skip it rather than refuse
        // the pack.
        if name.ends_with(".num_batches_tracked") {
            continue;
        }
        // Small f32 vectors the loaders read as f32: biases, RMSNorm scales,
        // and the BatchNorm statistics.
        let keep_f32 = name.ends_with(".bias")
            || name.ends_with(".scale")
            || name.ends_with(".running_mean")
            || name.ends_with(".running_var");
        let (quant_type, data) = if keep_f32 {
            (QuantType::F32, to_f32_bytes(info, bytes)?)
        } else {
            (QuantType::F16, to_f16_bytes(info, bytes)?)
        };
        tensors.push(HfqTensor {
            name: name.to_string(),
            quant_type,
            shape,
            group_size: 0,
            data,
            spilled_len: 0,
        });
        maybe_spill(&mut tensors, &mut spill, SPILL_THRESHOLD);
    }

    write_hfq(out, arch_id, &metadata, &tensors, Some(&mut spill))
        .map_err(|e| format!("flux pack: write {}: {e}", out.display()))?;
    Ok(())
}

/// F32 target: `.bias` / `.scale` stay exactly f32 (source F32 passes
/// through byte-identical; BF16/F16 widen).
fn to_f32_bytes(
    info: &hipfire_runtime::model_source::TensorInfo,
    bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let info_name = &info.name;
    match info.dtype.as_str() {
        "F32" => Ok(bytes.to_vec()),
        "BF16" => Ok(bytes
            .par_chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .flat_map(|f| f.to_le_bytes())
            .collect()),
        "F16" => Ok(bytes
            .par_chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .flat_map(|f| f.to_le_bytes())
            .collect()),
        other => Err(format!(
            "flux pack: unsupported source dtype `{other}` for {info_name} (F32/BF16/F16 only)"
        )),
    }
}

/// F16 target: GEMM operands. BF16/F32 convert once, here, with `f32_to_f16`
/// (RNE over the widened value); F16 passes through unchanged.
fn to_f16_bytes(
    info: &hipfire_runtime::model_source::TensorInfo,
    bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let info_name = &info.name;
    match info.dtype.as_str() {
        "F16" => Ok(bytes.to_vec()),
        "BF16" => Ok(bytes
            .par_chunks_exact(2)
            .map(|c| f32_to_f16(bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))))
            .flat_map(|f| f.to_le_bytes())
            .collect()),
        "F32" => Ok(bytes
            .par_chunks_exact(4)
            .map(|c| f32_to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .flat_map(|f| f.to_le_bytes())
            .collect()),
        other => Err(format!(
            "flux pack: unsupported source dtype `{other}` for {info_name} (F32/BF16/F16 only)"
        )),
    }
}

/// Per-component metadata envelope. `config` is the component's `config.json`
/// (transported through `SafetensorsSource::metadata_json`, which wraps it as
/// `{"config": …}` — the same shape the load side un-wraps). The transformer
/// file additionally embeds the scheduler config and the T5/CLIP tokenizer
/// blobs, the only pipe files that otherwise would not survive packing.
fn component_metadata(
    pipe: &Path,
    family: PipeFamily,
    component: FluxComponent,
    src: &SafetensorsSource,
) -> Result<String, String> {
    let src_meta: serde_json::Value = serde_json::from_str(src.metadata_json())
        .map_err(|e| format!("flux pack: {}/config.json invalid: {e}", component.subdir()))?;
    let mut m = serde_json::Map::new();
    m.insert("component".into(), serde_json::json!(component.stem()));
    m.insert(
        "family".into(),
        serde_json::json!(if family == PipeFamily::Flux2 {
            "flux2"
        } else {
            "flux1"
        }),
    );
    m.insert(
        "config".into(),
        src_meta.get("config").cloned().unwrap_or(src_meta),
    );
    if component == FluxComponent::Transformer {
        let sched = std::fs::read_to_string(pipe.join("scheduler/scheduler_config.json"))
            .map_err(|e| format!("flux pack: scheduler/scheduler_config.json: {e}"))?;
        m.insert(
            "scheduler_config".into(),
            serde_json::from_str(&sched)
                .map_err(|e| format!("flux pack: scheduler_config.json invalid: {e}"))?,
        );
        let tok = if family == PipeFamily::Flux1 {
            let t5_tok = std::fs::read_to_string(pipe.join("tokenizer_2/tokenizer.json"))
                .map_err(|e| format!("flux pack: tokenizer_2/tokenizer.json: {e}"))?;
            let clip_vocab = std::fs::read_to_string(pipe.join("tokenizer/vocab.json"))
                .map_err(|e| format!("flux pack: tokenizer/vocab.json: {e}"))?;
            let clip_merges = std::fs::read_to_string(pipe.join("tokenizer/merges.txt"))
                .map_err(|e| format!("flux pack: tokenizer/merges.txt: {e}"))?;
            serde_json::json!({
                "t5": serde_json::from_str::<serde_json::Value>(&t5_tok)
                    .map_err(|e| format!("flux pack: t5 tokenizer.json invalid: {e}"))?,
                "clip_vocab": serde_json::from_str::<serde_json::Value>(&clip_vocab)
                    .map_err(|e| format!("flux pack: clip vocab.json invalid: {e}"))?,
                "clip_merges": clip_merges,
            })
        } else {
            // klein: the Qwen3 BPE tokenizer, kept as raw text
            // (`Tokenizer::from_hf_json` consumes exactly this JSON).
            let qwen_tok = std::fs::read_to_string(pipe.join("tokenizer/tokenizer.json"))
                .map_err(|e| format!("flux pack: tokenizer/tokenizer.json: {e}"))?;
            serde_json::json!({ "qwen": qwen_tok })
        };
        m.insert("tokenizer".into(), tok);
    }
    serde_json::to_string_pretty(&serde_json::Value::Object(m))
        .map_err(|e| format!("flux pack: serialize metadata: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// One safetensors entry: `(name, little-endian bytes, dtype, shape)`.
    type NamedTensor = (String, Vec<u8>, String, Vec<usize>);

    /// Minimal hand-rolled safetensors writer (8-byte LE header length + JSON
    /// header + concatenated tensor bytes) — mirrors the fixture helper in
    /// `hipfire-arch-diffusion`'s flux tests; no writer dependency needed.
    fn write_safetensors(path: &Path, tensors: &[NamedTensor]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        let mut blobs: Vec<(&str, Vec<u8>)> = Vec::new();
        for (name, data, dtype, shape) in tensors {
            let mut meta = serde_json::Map::new();
            meta.insert("dtype".into(), dtype.clone().into());
            meta.insert(
                "shape".into(),
                serde_json::Value::Array(shape.iter().map(|&s| s.into()).collect()),
            );
            meta.insert(
                "data_offsets".into(),
                serde_json::json!([offset, offset + data.len()]),
            );
            header.insert(name.clone(), meta.into());
            blobs.push((name, data.clone()));
            offset += data.len();
        }
        let header_json = serde_json::json!(header).to_string();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(header_json.as_bytes());
        for (_, data) in blobs {
            out.extend_from_slice(&data);
        }
        std::fs::write(path, out).unwrap();
    }

    fn bf16(v: f32) -> [u8; 2] {
        // bf16 = top 16 bits of the f32.
        ((v.to_bits() >> 16) as u16).to_le_bytes()
    }

    fn f16(v: f32) -> [u8; 2] {
        // Round-trip through the lib crate's own converter.
        hipfire_quantize::float16::f32_to_f16(v).to_le_bytes()
    }

    fn f32w(v: f32) -> [u8; 4] {
        v.to_le_bytes()
    }

    /// A tiny-but-real FLUX.1-shaped pipe: schnell geometry for the config
    /// parse, two tensors per component for the pack/reopen assertions.
    fn write_tiny_flux_pipe(dir: &Path) {
        use std::fs;
        for sub in ["transformer", "text_encoder_2", "text_encoder", "vae"] {
            fs::create_dir_all(dir.join(sub)).unwrap();
        }
        fs::create_dir_all(dir.join("scheduler")).unwrap();
        fs::create_dir_all(dir.join("tokenizer_2")).unwrap();
        fs::create_dir_all(dir.join("tokenizer")).unwrap();

        // transformer: schnell-shaped config (parses through
        // FluxDiffusionConfig::from_json: axes sum to head_dim).
        let tx_cfg = serde_json::json!({
            "architectures": ["FluxTransformer2DModel"],
            "guidance_embed_dim": 0,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "head_dim": 128,
            "num_layers": 19,
            "num_single_layers": 38,
            "patch_size": 2,
            "latent_channels": 16,
            "axes_dim": [16, 56, 56, 0],
            "theta": 10000.0,
            "qk_norm": true,
            "norm_type": "rms_norm",
            "pooled_projection_dim": 768,
            "joint_attention_dim": 4096,
            "mlp_ratio": 4.0,
            "bias": true
        });
        fs::write(
            dir.join("transformer/config.json"),
            serde_json::to_string_pretty(&tx_cfg).unwrap(),
        )
        .unwrap();
        write_safetensors(
            &dir.join("transformer/diffusion_pytorch_model.safetensors"),
            &[
                (
                    "double_blocks.0.img_attn.qkv.weight".into(),
                    vec![bf16(1.0); 384 * 128].concat(),
                    "BF16".into(),
                    vec![384, 128],
                ),
                (
                    "single_blocks.0.modulation.lin.bias".into(),
                    vec![f32w(0.5); 384].concat(),
                    "F32".into(),
                    vec![384],
                ),
            ],
        );

        // T5 (text_encoder_2): two shards, merged by name by the packer.
        fs::write(
            dir.join("text_encoder_2/config.json"),
            r#"{"architectures":["T5EncoderModel"],"d_model":32,"d_ff":37,"d_kv":8,"num_heads":4,"num_layers":2}"#,
        )
        .unwrap();
        write_safetensors(
            &dir.join("text_encoder_2/model-00001-of-00002.safetensors"),
            &[(
                "shared.weight".into(),
                vec![bf16(1.0); 32 * 32].concat(),
                "BF16".into(),
                vec![32, 32],
            )],
        );
        write_safetensors(
            &dir.join("text_encoder_2/model-00002-of-00002.safetensors"),
            &[(
                "encoder.block.0.layer.0.SelfAttention.q.weight".into(),
                vec![bf16(1.0); 32 * 32].concat(),
                "BF16".into(),
                vec![32, 32],
            )],
        );

        // CLIP (text_encoder).
        fs::write(
            dir.join("text_encoder/config.json"),
            r#"{"architectures":["CLIPTextModel"],"hidden_size":64,"vocab_size":3}"#,
        )
        .unwrap();
        write_safetensors(
            &dir.join("text_encoder/model.safetensors"),
            &[(
                "text_model.embeddings.token_embedding.weight".into(),
                vec![f16(1.0); 3 * 64].concat(),
                "F16".into(),
                vec![3, 64],
            )],
        );

        // VAE.
        fs::write(
            dir.join("vae/config.json"),
            r#"{"architectures":["AutoencoderKL"],"in_channels":16,"latent_channels":16,"out_channels":3,"latent_patch":[1,1],"channels":[4,4,4,4]}"#,
        )
        .unwrap();
        write_safetensors(
            &dir.join("vae/diffusion_pytorch_model.safetensors"),
            &[(
                "decoder.conv_in.weight".into(),
                vec![bf16(1.0); 4 * 4].concat(),
                "BF16".into(),
                vec![4, 4],
            )],
        );

        fs::write(
            dir.join("scheduler/scheduler_config.json"),
            r#"{"num_train_timesteps":1000,"shift":1.0,"base_image_seq_len":256,"max_image_seq_len":4096,"base_shift":0.5,"max_shift":1.15}"#,
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer_2/tokenizer.json"),
            r#"{"model":{"unk_id":2,"vocab":[["<pad>",0.0],["</s>",0.0],["<unk>",0.0],["a",0.0]]}}"#,
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer/vocab.json"),
            r#"{"<|startoftext|>":0,"<|endoftext|>":1,"a":2}"#,
        )
        .unwrap();
        fs::write(dir.join("tokenizer/merges.txt"), "#version: 0.2\n").unwrap();
    }

    /// Pack the tiny pipe (all four components) and reopen each output with
    /// the runtime reader: names unchanged, weights F16 / bias F32, trunk
    /// metadata carrying the embedded scheduler + tokenizer blobs.
    #[test]
    fn flux_pack_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        write_tiny_flux_pipe(tmp.path());
        let out_base = tmp.path().join("pack/flux-tiny.hfq");
        run_flux_pack(tmp.path(), "all", &out_base).unwrap();

        let reopen = |name: &str, arch: u32| {
            let p = tmp.path().join("pack").join(name);
            let hfq = hipfire_runtime::hfq::HfqFile::open(&p)
                .map_err(|e| e.to_string())
                .unwrap();
            assert_eq!(hfq.arch_id, arch, "{name}: arch id");
            hfq
        };
        let trunk = reopen("flux-tiny-transformer.hfq", 40);
        assert_eq!(trunk.tensor_names().len(), 2);
        let names: Vec<&str> = trunk.tensor_names();
        assert!(names.contains(&"double_blocks.0.img_attn.qkv.weight"));
        assert!(names.contains(&"single_blocks.0.modulation.lin.bias"));
        let (wi, wb) = trunk
            .tensor_data("double_blocks.0.img_attn.qkv.weight")
            .unwrap();
        assert_eq!(wi.quant_type, 1, "weight packs as F16");
        assert_eq!(wb.len(), 384 * 128 * 2);
        let (bi, bb) = trunk
            .tensor_data("single_blocks.0.modulation.lin.bias")
            .unwrap();
        assert_eq!(bi.quant_type, 2, "bias stays F32");
        assert_eq!(bb.len(), 384 * 4);
        let meta: serde_json::Value = serde_json::from_str(trunk.metadata_json()).unwrap();
        assert_eq!(meta["component"], "transformer");
        assert!(meta["config"]["num_single_layers"] == 38);
        assert!(meta["scheduler_config"]["shift"] == 1.0);
        assert!(meta["tokenizer"]["clip_vocab"]["<|endoftext|>"] == 1);
        assert!(meta["tokenizer"]["clip_merges"]
            .as_str()
            .unwrap()
            .starts_with("#version"));

        let t5 = reopen("flux-tiny-t5.hfq", 41);
        assert_eq!(t5.tensor_names().len(), 2, "both T5 shards merged");
        let (_, _) = t5.tensor_data("shared.weight").unwrap();
        let (_, _) = t5
            .tensor_data("encoder.block.0.layer.0.SelfAttention.q.weight")
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(t5.metadata_json()).unwrap();
        assert_eq!(meta["component"], "t5");
        assert!(meta["tokenizer"].is_null() || meta.get("scheduler_config").is_none());

        let clip = reopen("flux-tiny-clip.hfq", 42);
        assert_eq!(clip.tensor_names().len(), 1);
        let (ci, cb) = clip
            .tensor_data("text_model.embeddings.token_embedding.weight")
            .unwrap();
        assert_eq!(ci.quant_type, 1);
        assert_eq!(cb.len(), 3 * 64 * 2, "F16 source passes through unchanged");

        let vae = reopen("flux-tiny-vae.hfq", 43);
        assert_eq!(vae.tensor_names().len(), 1);
        let (vi, _) = vae.tensor_data("decoder.conv_in.weight").unwrap();
        assert_eq!(vi.quant_type, 1);

        // The single-component form writes exactly to --output.
        let single = tmp.path().join("pack/clip-only.hfq");
        run_flux_pack(tmp.path(), "clip", &single).unwrap();
        let c = hipfire_runtime::hfq::HfqFile::open(&single).unwrap();
        assert_eq!(c.arch_id, 42);
    }
}
