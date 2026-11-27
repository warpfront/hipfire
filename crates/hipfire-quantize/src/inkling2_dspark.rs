//! Inkling-2-Mini native DSpark sidecar packer (`--format inkling2-dspark-bf16`).
//!
//! Reads exactly the pinned `mtp.safetensors` and `config.json` of the source
//! checkpoint plus the paired trunk HFQ (legacy or XTS layout), and writes
//! `<trunk-stem>-dspark.hfq`: the 54 source tensors with their names and BF16
//! bytes unchanged, then the `hidden_out` fold into that trunk's effective
//! unembed. Format, binding and validation are owned by
//! `hipfire_arch_inkling2::dspark_weights`; the packer reloads its output with
//! the same strict reader before reporting success.

use crate::hfq::{write_hfq_streamed, HfqIndexEntry};
use hipfire_arch_inkling2::dspark_weights::{
    sha256_hex, sidecar_metadata, InklingDsparkConfig, InklingDsparkFold, InklingDsparkSource,
    InklingDsparkTargetIdentity, InklingDsparkTensorDigest, InklingDsparkWeights, FOLD_BIAS,
    FOLD_RECIPE, FOLD_WEIGHT, QT_BF16, TRUNK_UNEMBED,
};
use hipfire_quantize::safetensors_file::SafetensorsFile;
use hipfire_runtime::hfq::HfqFile;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// `--format` value selecting this producer.
pub(crate) const FORMAT: &str = "inkling2-dspark-bf16";
/// Source draft file inside the checkpoint directory.
const DRAFT_FILE: &str = "mtp.safetensors";
/// Unembed rows decoded per fold work item.
const FOLD_TILE_ROWS: usize = 256;

#[derive(Debug)]
pub(crate) struct Inkling2DsparkSummary {
    pub(crate) output: PathBuf,
    pub(crate) layout: String,
    pub(crate) tensors: usize,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

/// CLI entry: the pinned Inkling-2-Mini source.
pub(crate) fn run_cli(
    source_dir: &Path,
    trunk: &Path,
    trunk_sha256: Option<&str>,
    output: Option<&Path>,
) -> Result<Inkling2DsparkSummary, String> {
    pack(
        source_dir,
        trunk,
        trunk_sha256,
        output,
        &InklingDsparkSource::mini(),
    )
}

/// `<stem>-dspark.<ext>` beside the trunk (the loader's sibling convention).
pub(crate) fn sibling_path(trunk: &Path) -> Result<PathBuf, String> {
    match (trunk.parent(), trunk.file_stem(), trunk.extension()) {
        (Some(parent), Some(stem), Some(ext)) => Ok(parent.join(format!(
            "{}-dspark.{}",
            stem.to_string_lossy(),
            ext.to_string_lossy()
        ))),
        _ => Err(format!(
            "cannot derive a -dspark sibling name from {}",
            trunk.display()
        )),
    }
}

pub(crate) fn pack(
    source_dir: &Path,
    trunk_path: &Path,
    trunk_sha256: Option<&str>,
    output: Option<&Path>,
    pin: &InklingDsparkSource,
) -> Result<Inkling2DsparkSummary, String> {
    let output = match output {
        Some(p) => p.to_path_buf(),
        None => sibling_path(trunk_path)?,
    };
    let draft_path = source_dir.join(DRAFT_FILE);
    for input in [trunk_path, draft_path.as_path()] {
        if same_file(input, &output) {
            return Err(format!(
                "output {} would overwrite an input",
                output.display()
            ));
        }
    }

    // ── Source provenance ───────────────────────────────────────────────────
    let draft_sha = file_sha256(&draft_path)?;
    let header_sha = sha256_hex(&safetensors_header(&draft_path)?);
    if draft_sha != pin.draft_sha256 || header_sha != pin.draft_header_sha256 {
        return Err(format!(
            "wrong draft checkpoint {}: sha256 {draft_sha} (header {header_sha}), \
             pinned {} (header {}) for {}@{}",
            draft_path.display(),
            pin.draft_sha256,
            pin.draft_header_sha256,
            pin.repo,
            pin.revision
        ));
    }
    let source_trunk = source_dir.join(&pin.trunk_file);
    let source_trunk_sha = file_sha256(&source_trunk)?;
    if source_trunk_sha != pin.trunk_sha256 {
        return Err(format!(
            "wrong source: {} sha256 {source_trunk_sha}, pinned {} for {}@{}",
            source_trunk.display(),
            pin.trunk_sha256,
            pin.repo,
            pin.revision
        ));
    }
    let config_path = source_dir.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("read {}: {e}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&config_text)
        .map_err(|e| format!("parse {}: {e}", config_path.display()))?;
    let cfg = InklingDsparkConfig::from_source_config(&config)?;
    let source_loop = hipfire_arch_inkling2::config::Inkling2Config::from_metadata_json(
        &serde_json::json!({ "config": config }).to_string(),
    )?
    .loop_cfg
    .visit_schedule();

    // ── Paired trunk ────────────────────────────────────────────────────────
    if let Some(want) = trunk_sha256 {
        let got = file_sha256(trunk_path)?;
        if !got.eq_ignore_ascii_case(want) {
            return Err(format!(
                "trunk {} sha256 {got} is not the expected {want}",
                trunk_path.display()
            ));
        }
    }
    let trunk = HfqFile::open(trunk_path)
        .map_err(|e| format!("open trunk {}: {e}", trunk_path.display()))?;
    let identity = InklingDsparkTargetIdentity::from_trunk(&trunk, pin.clone())?;
    if identity.draft_config != cfg {
        return Err(format!(
            "trunk {} mtp_config {:?} differs from the source config.json {cfg:?}",
            trunk_path.display(),
            identity.draft_config
        ));
    }
    if identity.loop_schedule != source_loop {
        return Err(format!(
            "trunk {} loop schedule differs from the source config.json",
            trunk_path.display()
        ));
    }

    // ── Strict source manifest ──────────────────────────────────────────────
    let manifest = cfg.manifest();
    let st = SafetensorsFile::open(&draft_path)
        .map_err(|e| format!("open {}: {e}", draft_path.display()))?;
    check_source_layout(&st, &draft_path, &manifest)?;
    let source_bytes = |name: &str| -> &[u8] { st.tensor_data(name).expect("checked name").1 };

    // ── Fold into the trunk's effective unembed ─────────────────────────────
    let (u_info, u_bytes) = trunk
        .tensor_data(TRUNK_UNEMBED)
        .map(|(i, b)| (i.clone(), std::borrow::Cow::Borrowed(b)))
        .or_else(|| {
            trunk
                .tensor_data_vec(TRUNK_UNEMBED)
                .map(|(i, v)| (i.clone(), std::borrow::Cow::Owned(v)))
        })
        .ok_or_else(|| format!("cannot read trunk {TRUNK_UNEMBED}"))?;
    if u_info.shape != [cfg.vocab_size as u32, cfg.hidden_size as u32] {
        return Err(format!(
            "trunk {TRUNK_UNEMBED} shape {:?} is not [{}, {}]",
            u_info.shape, cfg.vocab_size, cfg.hidden_size
        ));
    }
    let ho_w = source_bytes("markov_head.hidden_out.weight");
    let ho_b = source_bytes("markov_head.hidden_out.bias");
    let (fold_w, fold_b) = fold_hidden_out(
        u_info.quant_type,
        &u_bytes,
        cfg.vocab_size,
        cfg.hidden_size,
        &bf16_to_f64(ho_w),
        &bf16_to_f64(ho_b),
        cfg.markov_rank,
    )?;
    let fold = InklingDsparkFold {
        recipe: FOLD_RECIPE.to_string(),
        unembed: identity.unembed.clone(),
        hidden_out_weight_sha256: sha256_hex(ho_w),
        hidden_out_bias_sha256: sha256_hex(ho_b),
        weight_sha256: sha256_hex(&fold_w),
        bias_sha256: sha256_hex(&fold_b),
    };
    debug_assert_eq!(
        fold.unembed,
        InklingDsparkTensorDigest::of(TRUNK_UNEMBED, u_info.quant_type, &u_info.shape, &u_bytes)
    );
    drop(u_bytes);

    let mut payload = Sha256::new();
    for (name, _) in &manifest {
        payload.update(source_bytes(name));
    }
    let payload_sha = hipfire_hex(&payload.finalize());
    let metadata = sidecar_metadata(&identity, &payload_sha, &fold);

    // ── Write ───────────────────────────────────────────────────────────────
    let sidecar = cfg.sidecar_manifest();
    let entries: Vec<HfqIndexEntry> = sidecar
        .iter()
        .map(|(name, shape)| HfqIndexEntry {
            name: name.clone(),
            quant_type: QT_BF16,
            shape: shape.clone(),
            group_size: 0,
            data_len: 2 * shape.iter().map(|&d| d as u64).product::<u64>(),
        })
        .collect();
    let n_source = manifest.len();
    write_hfq_streamed(
        &output,
        hipfire_arch_inkling2::config::ARCH_ID,
        &metadata,
        &entries,
        |i, f| {
            let bytes: &[u8] = if i < n_source {
                source_bytes(&entries[i].name)
            } else if entries[i].name == FOLD_WEIGHT {
                &fold_w
            } else {
                debug_assert_eq!(entries[i].name, FOLD_BIAS);
                &fold_b
            };
            f.write_all(bytes)
        },
    )
    .map_err(|e| format!("write {}: {e}", output.display()))?;

    // ── Self-check with the strict reader ───────────────────────────────────
    let hfq = HfqFile::open(&output).map_err(|e| format!("reopen {}: {e}", output.display()))?;
    let loaded = InklingDsparkWeights::load(&hfq, &identity)?;
    for (name, _) in &manifest {
        if loaded.get(name).map(|t| t.data.as_slice()) != Some(source_bytes(name)) {
            return Err(format!(
                "{}: {name} bytes differ from the source",
                output.display()
            ));
        }
    }
    Ok(Inkling2DsparkSummary {
        layout: identity.layout,
        tensors: entries.len(),
        bytes: hfq.file_len(),
        sha256: file_sha256(&output)?,
        output,
    })
}

/// Exact names, BF16 dtype, shapes, and a contiguous, fully consumed data
/// section in the source safetensors.
fn check_source_layout(
    st: &SafetensorsFile,
    path: &Path,
    manifest: &[(String, Vec<u32>)],
) -> Result<(), String> {
    let err = |m: String| Err(format!("{}: {m}", path.display()));
    for name in st.tensors.keys() {
        if !manifest.iter().any(|(n, _)| n == name) {
            return err(format!("unexpected tensor {name}"));
        }
    }
    for (name, shape) in manifest {
        let Some(meta) = st.tensors.get(name) else {
            return err(format!("missing tensor {name}"));
        };
        let want: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if meta.dtype != "BF16" || meta.shape != want {
            return err(format!(
                "{name}: {} {:?} (expected BF16 {want:?})",
                meta.dtype, meta.shape
            ));
        }
        let numel: usize = want.iter().product();
        if meta.data_offsets[1] - meta.data_offsets[0] != 2 * numel {
            return err(format!("{name}: byte length disagrees with its shape"));
        }
    }
    let header_len = safetensors_header(path)?.len();
    let mut spans: Vec<[usize; 2]> = st.tensors.values().map(|m| m.data_offsets).collect();
    spans.sort_unstable();
    let mut cursor = 8 + header_len;
    for s in &spans {
        if s[0] != cursor {
            return err(format!("data span {s:?} does not start at {cursor}"));
        }
        cursor = s[1];
    }
    let len = std::fs::metadata(path)
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len() as usize;
    if cursor != len {
        return err(format!("tensor data ends at {cursor}, file is {len} bytes"));
    }
    Ok(())
}

/// `fold[v, r] = Σ_h U[v, h] · W[h, r]`, `bias[v] = Σ_h U[v, h] · b[h]` with
/// `U` the unembed decoded to F32, F64 accumulation over ascending `h`, one
/// nearest-even rounding to BF16 ([`FOLD_RECIPE`]). Decodes the unembed one
/// row tile at a time.
pub(crate) fn fold_hidden_out(
    quant_type: u8,
    unembed: &[u8],
    vocab: usize,
    hidden: usize,
    w: &[f64],
    b: &[f64],
    rank: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    if !hipfire_runtime::weight_backend::dequant_f32_supported(quant_type) {
        return Err(format!(
            "unembed quant type {quant_type} has no host decoder"
        ));
    }
    if w.len() != hidden * rank || b.len() != hidden || unembed.len() % vocab != 0 {
        return Err("fold operand sizes disagree".into());
    }
    let row_bytes = unembed.len() / vocab;
    let mut fold_w = vec![0u8; vocab * rank * 2];
    let mut fold_b = vec![0u8; vocab * 2];
    fold_w
        .par_chunks_mut(FOLD_TILE_ROWS * rank * 2)
        .zip(fold_b.par_chunks_mut(FOLD_TILE_ROWS * 2))
        .enumerate()
        .try_for_each(|(tile, (out_w, out_b))| -> Result<(), String> {
            let rows = out_b.len() / 2;
            let start = tile * FOLD_TILE_ROWS * row_bytes;
            let u = hipfire_runtime::weight_backend::dequantize_weight_to_f32(
                quant_type,
                &unembed[start..start + rows * row_bytes],
                rows * hidden,
            );
            if u.len() != rows * hidden {
                return Err(format!("unembed tile {tile} decoded to {} values", u.len()));
            }
            let mut acc = vec![0f64; rank];
            for row in 0..rows {
                acc.fill(0.0);
                let mut acc_b = 0f64;
                for (h, &x) in u[row * hidden..(row + 1) * hidden].iter().enumerate() {
                    let x = x as f64;
                    acc_b += x * b[h];
                    for (a, &wr) in acc.iter_mut().zip(&w[h * rank..(h + 1) * rank]) {
                        *a += x * wr;
                    }
                }
                for (r, &a) in acc.iter().enumerate() {
                    let o = (row * rank + r) * 2;
                    out_w[o..o + 2].copy_from_slice(&f64_to_bf16_rne(a).to_le_bytes());
                }
                out_b[row * 2..row * 2 + 2].copy_from_slice(&f64_to_bf16_rne(acc_b).to_le_bytes());
            }
            Ok(())
        })?;
    Ok((fold_w, fold_b))
}

fn bf16_to_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
        .collect()
}

/// Single nearest-even rounding of an F64 to BF16: round to odd at F32
/// (exact for the 16 low bits dropped next), then nearest-even to BF16.
pub(crate) fn f64_to_bf16_rne(v: f64) -> u16 {
    if v.is_nan() {
        return 0x7FC0 | (((v.to_bits() >> 48) as u16) & 0x8000);
    }
    let mut f = v as f32;
    if f.is_finite() && f as f64 != v {
        // Step to the F32 neighbour toward zero, then set the sticky LSB.
        if (f as f64).abs() > v.abs() {
            f = f32::from_bits(f.to_bits() - 1);
        }
        f = f32::from_bits(f.to_bits() | 1);
    }
    let bits = f.to_bits();
    if f.is_infinite() {
        return (bits >> 16) as u16;
    }
    let bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits + bias) >> 16) as u16
}

fn hipfire_hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hipfire_hex(&h.finalize()))
}

/// The JSON header bytes of a safetensors file.
fn safetensors_header(path: &Path) -> Result<Vec<u8>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let n = u64::from_le_bytes(len);
    if n > 100 << 20 {
        return Err(format!("{}: header length {n}", path.display()));
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(header)
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hfq::{write_hfq, HfqTensor, QuantType};
    use crate::quant_fwht::{gen_fwht_signs, quantize_mq4g256v2, quantize_mq4g256v2_symmetric};
    use hipfire_arch_inkling2::dspark_weights::{TRUNK_LAYOUT_KEY, TRUNK_LAYOUT_XTS};
    use serde_json::{json, Value};

    const HIDDEN: usize = 256;
    const VOCAB: usize = 64;

    fn rng_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|&x| f64_to_bf16_rne(x as f64).to_le_bytes())
            .collect()
    }

    fn config(num_passes: usize) -> Value {
        json!({
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "eos_token_id": 3,
            "text_config": {
                "num_hidden_layers": 24, "hidden_size": HIDDEN, "vocab_size": VOCAB,
                "unpadded_vocab_size": 60, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 128, "sliding_window_size": 1024, "d_rel": 16, "rel_extent": 1024,
                "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0,
                "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "sconv_kernel_size": 4,
                "n_layers_per_attn_res_block": 6, "final_logit_softcapping": 30.0,
                "n_routed_experts": 4, "num_experts_per_tok": 2, "intermediate_size": 128,
                "route_scale": 8.0, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
                "rms_norm_eps": 1e-6, "local_layer_ids": [0, 1, 2, 3, 4],
                "loop_config": {"core_start": 6, "core_end": 18, "num_passes": num_passes,
                                "gate_rank": 64, "is_varloop": false}
            },
            "mtp_config": {
                "hidden_size": HIDDEN, "num_hidden_layers": 3, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 64, "intermediate_size": 512,
                "max_position_embeddings": 1048576, "q_bias": false, "o_bias": false,
                "rms_norm_eps": 1e-6, "rope_theta": 1e6, "local_layer_ids": [0, 2],
                "sliding_window_size": 1024, "block_size": 8, "markov_rank": 16,
                "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48]
            }
        })
    }

    struct Fixture {
        dir: tempfile::TempDir,
        pin: InklingDsparkSource,
    }

    impl Fixture {
        fn p(&self, name: &str) -> PathBuf {
            self.dir.path().join(name)
        }
    }

    /// Writes a safetensors file with tensors laid out in the given order.
    fn write_safetensors(path: &Path, tensors: &[(String, Vec<u32>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut off = 0usize;
        for (name, shape, data) in tensors {
            header.insert(
                name.clone(),
                json!({"dtype": "BF16", "shape": shape, "data_offsets": [off, off + data.len()]}),
            );
            off += data.len();
        }
        let mut h = serde_json::to_vec(&Value::Object(header)).unwrap();
        while h.len() % 8 != 0 {
            h.push(b' ');
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        for (_, _, d) in tensors {
            f.write_all(d).unwrap();
        }
    }

    fn source_tensors() -> Vec<(String, Vec<u32>, Vec<u8>)> {
        let cfg = InklingDsparkConfig::from_source_config(&config(3)).unwrap();
        cfg.manifest()
            .into_iter()
            .enumerate()
            .map(|(i, (name, shape))| {
                let n: usize = shape.iter().map(|&d| d as usize).product();
                let data = bf16_bytes(&rng_f32(100 + i as u64, n, 0.5));
                (name, shape, data)
            })
            .collect()
    }

    fn fixture_with(tensors: &[(String, Vec<u32>, Vec<u8>)]) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("source");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("config.json"), config(3).to_string()).unwrap();
        std::fs::write(
            src.join("model-00001-of-00001.safetensors"),
            b"source trunk",
        )
        .unwrap();
        write_safetensors(&src.join(DRAFT_FILE), tensors);
        let pin = InklingDsparkSource {
            repo: "fixture/Inkling".into(),
            revision: "0123".into(),
            trunk_file: "model-00001-of-00001.safetensors".into(),
            trunk_sha256: file_sha256(&src.join("model-00001-of-00001.safetensors")).unwrap(),
            draft_sha256: file_sha256(&src.join(DRAFT_FILE)).unwrap(),
            draft_header_sha256: sha256_hex(&safetensors_header(&src.join(DRAFT_FILE)).unwrap()),
        };
        let fx = Fixture { dir, pin };
        write_trunk(&fx.p("legacy.hfq"), false, 3, "tok-a", 7);
        write_trunk(&fx.p("xts.hfq"), true, 3, "tok-a", 7);
        fx
    }

    fn fixture() -> Fixture {
        fixture_with(&source_tensors())
    }

    /// A minimal Inkling trunk: metadata plus embed, embed norm, unembed.
    /// XTS: layout tag and a symmetric unembed; embed bytes match legacy.
    /// (No `mq4v2.symmetric` marker: this tree's HfqFile checks it on the
    /// affine embed too; the binding keys on the layout tag alone.)
    fn write_trunk(path: &Path, xts: bool, num_passes: usize, tokenizer: &str, useed: u64) {
        let mut meta = json!({
            "architecture": "inkling2_loop",
            "config": config(num_passes),
            "tokenizer": {"model": {"type": "BPE", "vocab": {"a": 0}}, "fixture": tokenizer},
            "tokenizer_config": {"bos_token": "<s>"},
        });
        if xts {
            meta[TRUNK_LAYOUT_KEY] = json!(TRUNK_LAYOUT_XTS);
        }
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let embed = quantize_mq4g256v2(&rng_f32(1, VOCAB * HIDDEN, 1.0), VOCAB, HIDDEN, &s1, &s2);
        let u = rng_f32(useed, VOCAB * HIDDEN, 0.2);
        let unembed = if xts {
            quantize_mq4g256v2_symmetric(&u, VOCAB, HIDDEN, &s1, &s2)
        } else {
            quantize_mq4g256v2(&u, VOCAB, HIDDEN, &s1, &s2)
        };
        let norm: Vec<u8> = rng_f32(2, HIDDEN, 1.0)
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect();
        let t = |name: &str, qt, shape: Vec<u32>, gs, data| HfqTensor {
            name: name.into(),
            quant_type: qt,
            shape,
            group_size: gs,
            data,
            spilled_len: 0,
        };
        let tensors = vec![
            t(
                "model.llm.embed.weight",
                QuantType::MQ4G256V2,
                vec![VOCAB as u32, HIDDEN as u32],
                256,
                embed,
            ),
            t(
                "model.llm.embed_norm.weight",
                QuantType::F16,
                vec![HIDDEN as u32],
                0,
                norm,
            ),
            t(
                "model.llm.unembed.weight",
                QuantType::MQ4G256V2,
                vec![VOCAB as u32, HIDDEN as u32],
                256,
                unembed,
            ),
        ];
        write_hfq(path, 24, &meta.to_string(), &tensors, None).unwrap();
    }

    fn identity(fx: &Fixture, trunk: &str) -> InklingDsparkTargetIdentity {
        let hfq = HfqFile::open(&fx.p(trunk)).unwrap();
        InklingDsparkTargetIdentity::from_trunk(&hfq, fx.pin.clone()).unwrap()
    }

    fn pack_for(fx: &Fixture, trunk: &str) -> Inkling2DsparkSummary {
        pack(&fx.p("source"), &fx.p(trunk), None, None, &fx.pin).unwrap()
    }

    fn load(path: &Path, id: &InklingDsparkTargetIdentity) -> Result<InklingDsparkWeights, String> {
        InklingDsparkWeights::load(&HfqFile::open(path).unwrap(), id)
    }

    fn load_err(path: &Path, id: &InklingDsparkTargetIdentity) -> String {
        match load(path, id) {
            Ok(_) => panic!("{} loaded", path.display()),
            Err(e) => e,
        }
    }

    /// Rewrite a sidecar through `edit` (tensor list and metadata).
    fn rewrite(src: &Path, dst: &Path, edit: impl FnOnce(&mut Vec<HfqTensor>, &mut Value)) {
        let hfq = HfqFile::open(src).unwrap();
        let mut meta: Value = serde_json::from_str(&hfq.metadata_json).unwrap();
        let mut tensors: Vec<HfqTensor> = hfq
            .tensor_infos()
            .iter()
            .map(|i| HfqTensor {
                name: i.name.clone(),
                quant_type: QuantType::BF16,
                shape: i.shape.clone(),
                group_size: i.group_size,
                data: hfq.tensor_data(&i.name).unwrap().1.to_vec(),
                spilled_len: 0,
            })
            .collect();
        edit(&mut tensors, &mut meta);
        write_hfq(dst, hfq.arch_id, &meta.to_string(), &tensors, None).unwrap();
    }

    fn reference_fold(trunk: &Path) -> (Vec<u8>, Vec<u8>) {
        let hfq = HfqFile::open(trunk).unwrap();
        let (i, data) = hfq.tensor_data(TRUNK_UNEMBED).unwrap();
        let u = hipfire_runtime::weight_backend::dequantize_weight_to_f32(
            i.quant_type,
            data,
            VOCAB * HIDDEN,
        );
        let src = source_tensors();
        let get = |n: &str| bf16_to_f64(&src.iter().find(|t| t.0 == n).unwrap().2);
        let (w, b) = (
            get("markov_head.hidden_out.weight"),
            get("markov_head.hidden_out.bias"),
        );
        let rank = 16;
        let mut fw = Vec::new();
        let mut fb = Vec::new();
        for v in 0..VOCAB {
            for r in 0..rank {
                let mut a = 0f64;
                for h in 0..HIDDEN {
                    a += u[v * HIDDEN + h] as f64 * w[h * rank + r];
                }
                fw.extend(f64_to_bf16_rne(a).to_le_bytes());
            }
            let mut a = 0f64;
            for h in 0..HIDDEN {
                a += u[v * HIDDEN + h] as f64 * b[h];
            }
            fb.extend(f64_to_bf16_rne(a).to_le_bytes());
        }
        (fw, fb)
    }

    #[test]
    fn bf16_rounding_is_single_nearest_even() {
        assert_eq!(f64_to_bf16_rne(1.0), 0x3F80);
        // Exactly halfway between 1.0 and the next BF16: ties to even (1.0).
        assert_eq!(f64_to_bf16_rne(1.0 + 2f64.powi(-8)), 0x3F80);
        // A hair above the tie rounds up, even though a direct F32 rounding
        // would land on the tie and then round down (double rounding).
        assert_eq!(
            f64_to_bf16_rne(1.0 + 2f64.powi(-8) + 2f64.powi(-40)),
            0x3F81
        );
        assert_eq!(
            f64_to_bf16_rne(-1.0 - 2f64.powi(-8) - 2f64.powi(-40)),
            0xBF81
        );
        assert_eq!(f64_to_bf16_rne(1.0 + 3.0 * 2f64.powi(-8)), 0x3F82);
        assert_eq!(f64_to_bf16_rne(f64::INFINITY), 0x7F80);
        assert_eq!(f64_to_bf16_rne(1e300), 0x7F80);
        assert_eq!(f64_to_bf16_rne(0.0), 0);
    }

    #[test]
    fn packs_legacy_and_xts_with_exact_source_entries_and_trunk_fold() {
        let fx = fixture();
        let src = source_tensors();
        let mut folds = Vec::new();
        for (trunk, layout) in [("legacy.hfq", "legacy"), ("xts.hfq", TRUNK_LAYOUT_XTS)] {
            let trunk_before = file_sha256(&fx.p(trunk)).unwrap();
            let s = pack_for(&fx, trunk);
            assert_eq!(s.output, fx.p(&trunk.replace(".hfq", "-dspark.hfq")));
            assert_eq!((s.layout.as_str(), s.tensors), (layout, src.len() + 2));
            // The paired trunk is read only.
            assert_eq!(file_sha256(&fx.p(trunk)).unwrap(), trunk_before);
            let w = load(&s.output, &identity(&fx, trunk)).unwrap();
            assert_eq!(w.tensors.len(), src.len() + 2);
            for ((name, shape, data), (got_name, got)) in src.iter().zip(&w.tensors) {
                assert_eq!(got_name, name);
                assert_eq!(got.info.quant_type, QT_BF16);
                assert_eq!(&got.info.shape, shape);
                assert_eq!(&got.data, data, "{name}");
            }
            let (fw, fb) = reference_fold(&fx.p(trunk));
            assert_eq!(w.get(FOLD_WEIGHT).unwrap().data, fw);
            assert_eq!(w.get(FOLD_BIAS).unwrap().data, fb);
            folds.push(fw);
            // Deterministic: a second pack is byte-identical.
            let again = pack(
                &fx.p("source"),
                &fx.p(trunk),
                None,
                Some(&fx.p("again.hfq")),
                &fx.pin,
            )
            .unwrap();
            assert_eq!(again.sha256, s.sha256);
        }
        assert_ne!(folds[0], folds[1], "each trunk gets its own fold");
    }

    #[test]
    fn source_manifest_rejects_missing_extra_and_mistyped_tensors() {
        let mut src = source_tensors();
        src.pop();
        let fx = fixture_with(&src);
        let e = pack(&fx.p("source"), &fx.p("legacy.hfq"), None, None, &fx.pin).unwrap_err();
        assert!(e.contains("missing tensor norm.weight"), "{e}");

        let mut src = source_tensors();
        src.push(("markov_head.markov_w2.weight".into(), vec![4], vec![0; 8]));
        let fx = fixture_with(&src);
        let e = pack(&fx.p("source"), &fx.p("legacy.hfq"), None, None, &fx.pin).unwrap_err();
        assert!(
            e.contains("unexpected tensor markov_head.markov_w2.weight"),
            "{e}"
        );

        let mut src = source_tensors();
        src[0].1 = vec![2, 8];
        let fx = fixture_with(&src);
        let e = pack(&fx.p("source"), &fx.p("legacy.hfq"), None, None, &fx.pin).unwrap_err();
        assert!(e.contains("fuse_weight"), "{e}");
    }

    #[test]
    fn pack_rejects_wrong_checkpoint_source_and_trunk() {
        let fx = fixture();
        let mut pin = fx.pin.clone();
        pin.draft_sha256 = "00".repeat(32);
        let e = pack(&fx.p("source"), &fx.p("legacy.hfq"), None, None, &pin).unwrap_err();
        assert!(e.starts_with("wrong draft checkpoint"), "{e}");
        let mut pin = fx.pin.clone();
        pin.trunk_sha256 = "00".repeat(32);
        let e = pack(&fx.p("source"), &fx.p("legacy.hfq"), None, None, &pin).unwrap_err();
        assert!(e.starts_with("wrong source"), "{e}");
        let e = pack(
            &fx.p("source"),
            &fx.p("legacy.hfq"),
            Some(&"00".repeat(32)),
            None,
            &fx.pin,
        )
        .unwrap_err();
        assert!(e.contains("is not the expected"), "{e}");
        write_trunk(&fx.p("loop2.hfq"), false, 2, "tok-a", 7);
        let e = pack(&fx.p("source"), &fx.p("loop2.hfq"), None, None, &fx.pin).unwrap_err();
        assert!(e.contains("loop schedule"), "{e}");
    }

    #[test]
    fn reader_rejects_wrong_source_tokenizer_checkpoint_loop_and_layout() {
        let fx = fixture();
        let out = pack_for(&fx, "legacy.hfq").output;
        let id = identity(&fx, "legacy.hfq");

        let mut bad = id.clone();
        bad.source.revision = "main".into();
        assert!(load_err(&out, &bad).contains("wrong source"));
        let mut bad = id.clone();
        bad.source.trunk_sha256 = "11".repeat(32);
        assert!(load_err(&out, &bad).contains("wrong source"));
        let mut bad = id.clone();
        bad.source.draft_sha256 = "22".repeat(32);
        assert!(load_err(&out, &bad).contains("wrong draft checkpoint"));

        write_trunk(&fx.p("tok.hfq"), false, 3, "tok-b", 7);
        let e = load_err(&out, &identity(&fx, "tok.hfq"));
        assert!(e.contains("wrong tokenizer"), "{e}");

        write_trunk(&fx.p("loop2.hfq"), false, 2, "tok-a", 7);
        let e = load_err(&out, &identity(&fx, "loop2.hfq"));
        assert!(e.contains("wrong loop schedule"), "{e}");

        let e = load_err(&out, &identity(&fx, "xts.hfq"));
        assert!(e.contains("trunk layout"), "{e}");

        let mut bad = id.clone();
        bad.embed.sha256 = "33".repeat(32);
        assert!(load_err(&out, &bad).contains("bound embed"));
    }

    #[test]
    fn reader_rejects_stale_fold() {
        let fx = fixture();
        let out = pack_for(&fx, "legacy.hfq").output;
        // Same layout, tokenizer and embed; a different unembed.
        write_trunk(&fx.p("requant.hfq"), false, 3, "tok-a", 8);
        let e = load_err(&out, &identity(&fx, "requant.hfq"));
        assert!(e.contains("stale fold"), "{e}");

        // A re-trained hidden_out with the old fold: payload digest updated,
        // fold key not.
        let stale = fx.p("stale-dspark.hfq");
        rewrite(&out, &stale, |t, meta| {
            let ho = t
                .iter_mut()
                .find(|t| t.name == "markov_head.hidden_out.weight")
                .unwrap();
            ho.data[0] ^= 0x40;
            let mut h = Sha256::new();
            for t in t.iter().take(t.len() - 2) {
                h.update(&t.data);
            }
            meta["hipfire_inkling2_dspark"]["payload_sha256"] = json!(hipfire_hex(&h.finalize()));
        });
        let e = load_err(&stale, &identity(&fx, "legacy.hfq"));
        assert!(e.contains("stale fold: hidden_out.weight"), "{e}");
    }

    #[test]
    fn reader_rejects_corrupted_layout_and_missing_or_extra_tensors() {
        let fx = fixture();
        let out = pack_for(&fx, "legacy.hfq").output;
        let id = identity(&fx, "legacy.hfq");
        let case = |name: &str, edit: &dyn Fn(&mut Vec<HfqTensor>, &mut Value)| {
            let p = fx.p(name);
            rewrite(&out, &p, |t, m| edit(t, m));
            load_err(&p, &id)
        };
        let e = case("flip.hfq", &|t, _| t[5].data[3] ^= 1);
        assert!(e.contains("source tensor bytes sha256"), "{e}");
        let e = case("foldflip.hfq", &|t, _| {
            let n = t.len();
            t[n - 1].data[0] ^= 1
        });
        assert!(e.contains("fold bias sha256"), "{e}");
        let e = case("missing.hfq", &|t, _| {
            t.remove(10);
        });
        assert!(e.contains("tensors (expected"), "{e}");
        let e = case("extra.hfq", &|t, _| {
            t.push(HfqTensor {
                name: "markov_head.markov_w2.weight".into(),
                quant_type: QuantType::BF16,
                shape: vec![4],
                group_size: 0,
                data: vec![0; 8],
                spilled_len: 0,
            })
        });
        assert!(e.contains("tensors (expected"), "{e}");
        let e = case("renamed.hfq", &|t, _| {
            t[1].name = "layers.0.dflare_norm.bias".into()
        });
        assert!(e.contains("is expected"), "{e}");
        let e = case("swapped.hfq", &|t, _| t.swap(1, 2));
        assert!(e.contains("is expected"), "{e}");
        let e = case("reshaped.hfq", &|t, _| t[0].shape = vec![2, 8]);
        assert!(e.contains("fuse_weight: qt"), "{e}");
        let e = case("retyped.hfq", &|t, _| t[1].quant_type = QuantType::F16);
        assert!(e.contains("qt 1"), "{e}");
        let e = case("nobody.hfq", &|_, m| {
            m["config"].as_object_mut().unwrap().remove("dspark_body");
        });
        assert!(e.contains("not an Inkling DSpark sidecar"), "{e}");
        let e = case("capacity.hfq", &|_, m| {
            m["config"]["dspark_proposal_capacity"] = json!(8)
        });
        assert!(e.contains("proposals"), "{e}");

        // Trailing bytes after the last tensor.
        let p = fx.p("trailing.hfq");
        std::fs::copy(&out, &p).unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&p)
            .unwrap()
            .write_all(&[0; 16])
            .unwrap();
        let e = load_err(&p, &id);
        assert!(e.contains("tensor data ends at"), "{e}");

        // A trunk is never accepted as a sidecar, nor a sidecar as a trunk.
        let e = load_err(&fx.p("legacy.hfq"), &id);
        assert!(e.contains("not an Inkling DSpark sidecar"), "{e}");
        let side = HfqFile::open(&out).unwrap();
        assert!(InklingDsparkTargetIdentity::from_trunk(&side, fx.pin.clone()).is_err());
    }

    /// Real artifacts: `INKLING_DSPARK_SOURCE` (pinned checkpoint dir),
    /// `INKLING_DSPARK_TRUNK` and `INKLING_DSPARK_OUT`.
    #[test]
    #[ignore = "requires the pinned Inkling-2-Mini checkpoint and a trunk HFQ"]
    fn real_mini_sidecar_matches_mtp_safetensors() {
        let var = |k: &str| PathBuf::from(std::env::var(k).unwrap());
        let source = var("INKLING_DSPARK_SOURCE");
        let trunk = var("INKLING_DSPARK_TRUNK");
        let out = var("INKLING_DSPARK_OUT");
        let s = run_cli(&source, &trunk, None, Some(&out)).unwrap();
        let id = InklingDsparkTargetIdentity::from_trunk(
            &HfqFile::open(&trunk).unwrap(),
            InklingDsparkSource::mini(),
        )
        .unwrap();
        let w = load(&out, &id).unwrap();
        let st = SafetensorsFile::open(&source.join(DRAFT_FILE)).unwrap();
        assert_eq!(st.tensors.len(), 54);
        assert_eq!(w.tensors.len(), 56);
        for (name, t) in &w.tensors[..54] {
            let (meta, data) = st.tensor_data(name).unwrap();
            assert_eq!(meta.dtype, "BF16");
            let shape: Vec<u32> = meta.shape.iter().map(|&d| d as u32).collect();
            assert_eq!(
                (t.info.quant_type, &t.info.shape),
                (QT_BF16, &shape),
                "{name}"
            );
            assert_eq!(t.data, data, "{name}");
        }
        eprintln!(
            "{} {} {} sha256={}",
            s.output.display(),
            s.layout,
            s.bytes,
            s.sha256
        );
    }
}
