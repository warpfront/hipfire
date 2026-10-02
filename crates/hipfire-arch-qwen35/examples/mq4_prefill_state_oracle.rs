// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Test-only cross-process state oracle; not a serving/performance acceptance test.
//! dump MODEL.hfq PROMPT.txt ROWS CONTINUATION_STEPS NEW_OUTPUT_DIR
//! compare BASELINE_DIR CANDIDATE_DIR
//! Set chunk/experimental flags externally BEFORE each fresh process. Suggested
//! row counts: 513, 1025, 8193. Prompt tokens repeat/truncate to exact ROWS;
//! continuation is teacher-forced from the same token stream, never greedy.
//! Q8 VMM KV, Q8 DeltaNet with EF; full meaningful prefix only, no unused KV.
//! Raw dumps preserve codes/scales/EF. compare emits numeric metrics per layer
//! (Q8 KV is dequantized), finite counts, and first/max difference indices.
//! No tolerance is silently declared a pass. A finite result is not coherence.

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, LayerType, Qwen35Config, Qwen35Scratch, StateQuant,
};
use hipfire_runtime::{hfq::HfqFile, llama::KvCache, tokenizer::Tokenizer};
use rdna_compute::{Gpu, GpuTensor};
use serde::{Deserialize, Serialize};
use std::{error::Error, fs, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Serialize, Deserialize)]
struct Entry {
    name: String,
    kind: String,
    file: String,
    bytes: usize,
}
#[derive(Serialize, Deserialize)]
struct Manifest {
    model: String,
    arch: String,
    tokens: Vec<u32>,
    prefill_rows: usize,
    admitted_ceiling: usize,
    state_head_dim: usize,
    entries: Vec<Entry>,
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (bits >> 10) & 31;
    let m = bits & 1023;
    match e {
        0 => sign * (m as f32) * 2f32.powi(-24),
        31 if m == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

fn decode(kind: &str, bytes: &[u8]) -> Vec<f32> {
    match kind {
        "f32" => {
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        "f16" => {
            assert_eq!(bytes.len() % 2, 0);
            bytes
                .chunks_exact(2)
                .map(|b| half(u16::from_le_bytes(b.try_into().unwrap())))
                .collect()
        }
        "i8" => bytes.iter().map(|&b| (b as i8) as f32).collect(),
        "q8_0" => {
            assert_eq!(bytes.len() % 34, 0);
            bytes
                .chunks_exact(34)
                .flat_map(|b| {
                    let d = half(u16::from_le_bytes([b[0], b[1]]));
                    b[2..].iter().map(move |&q| (q as i8) as f32 * d)
                })
                .collect()
        }
        _ => panic!("unsupported dump encoding {kind}"),
    }
}

fn save(
    gpu: &Gpu,
    dir: &Path,
    entries: &mut Vec<Entry>,
    name: String,
    kind: &str,
    tensor: &GpuTensor,
    bytes: usize,
) -> Result<()> {
    assert!(
        bytes <= tensor.buf.size(),
        "{name}: download exceeds allocation"
    );
    let mut raw = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut raw, &tensor.buf)?;
    let values = decode(kind, &raw);
    let nonfinite = values.iter().filter(|v| !v.is_finite()).count();
    eprintln!(
        "dump {name}: {} values, nonfinite={nonfinite}",
        values.len()
    );
    let file = format!("{:04}.bin", entries.len());
    fs::write(dir.join(&file), raw)?;
    entries.push(Entry {
        name,
        kind: kind.into(),
        file,
        bytes,
    });
    if nonfinite != 0 {
        return Err("nonfinite state/logits (partial dump retained)".into());
    }
    Ok(())
}

fn snapshot(
    gpu: &Gpu,
    dir: &Path,
    entries: &mut Vec<Entry>,
    stage: &str,
    rows: usize,
    config: &Qwen35Config,
    dn: &DeltaNetState,
    kv: &KvCache,
    scratch: &Qwen35Scratch,
) -> Result<()> {
    gpu.hip.device_synchronize()?;
    save(
        gpu,
        dir,
        entries,
        format!("{stage}/logits"),
        "f32",
        &scratch.logits,
        config.vocab_size * 4,
    )?;
    let mut la = 0;
    for (layer, ty) in config.layer_types.iter().enumerate() {
        let prefix = format!("{stage}/layer_{layer:02}");
        if *ty == LayerType::LinearAttention {
            let n = config.linear_num_value_heads * config.linear_value_head_dim.pow(2);
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/state_codes"),
                "i8",
                &dn.s_matrices[la],
                n,
            )?;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/state_scales"),
                "f32",
                &dn.s_scales[la],
                dn.s_scales[la].numel() * 4,
            )?;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/ef"),
                "f16",
                &dn.s_ef_residual[la],
                n * 2,
            )?;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/conv"),
                "f32",
                &dn.conv_states[la],
                dn.conv_states[la].numel() * 4,
            )?;
            la += 1;
        } else {
            let bytes = rows * config.n_kv_heads * (config.head_dim / 32) * 34;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/key"),
                "q8_0",
                &kv.k_gpu[layer],
                bytes,
            )?;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/value"),
                "q8_0",
                &kv.v_gpu[layer],
                bytes,
            )?;
        }
    }
    Ok(())
}

fn dump(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        return Err("dump MODEL PROMPT.txt ROWS CONTINUATION_STEPS NEW_OUTPUT_DIR".into());
    }
    let rows: usize = args[2].parse()?;
    let continuation: usize = args[3].parse()?;
    if rows < 2 {
        return Err("ROWS must be >=2".into());
    }
    let total = rows.checked_add(continuation).ok_or("row overflow")?;
    let cap = total.checked_add(16).ok_or("capacity overflow")?.max(512);
    let dir = Path::new(&args[4]);
    fs::create_dir(dir)?; // fail rather than overwrite an existing run
    let mut hfq = HfqFile::open(Path::new(&args[0]))?;
    let config = qwen35::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    let seed = tokenizer.encode(&fs::read_to_string(&args[1])?);
    if seed.is_empty() {
        return Err("prompt tokenized to empty sequence".into());
    }
    let tokens: Vec<u32> = seed.iter().copied().cycle().take(total).collect();
    let mut gpu = Gpu::init()?;
    let mut source = qwen35::HfqSource::new(&mut hfq, &config);
    let weights = qwen35::load_weights(
        &mut source,
        std::slice::from_mut(&mut gpu),
        &qwen35::Layout::single(config.n_layers),
    )?;
    let mask: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let mut kv = KvCache::new_gpu_q8_vmm_capped_filtered(
        &mut gpu,
        &mask,
        config.n_kv_heads,
        config.head_dim,
        cap,
        cap,
    )?;
    let mut dn = DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::Q8)?;
    if dn.s_ef_residual.len() != dn.s_matrices.len() {
        return Err("Q8 EF must be enabled".into());
    }
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cap)?;
    // Final executed ceiling is also printed by the production prefill receipt;
    // this pre-call query may differ after KV mapping and is labeled as such.
    let ceiling = qwen35::ordinary_prefill_chunk_limit(&gpu, &weights, &config, &dn, &kv, None)?;
    let mut manifest = Manifest {
        model: args[0].clone(),
        arch: gpu.arch.clone(),
        tokens,
        prefill_rows: rows,
        admitted_ceiling: ceiling,
        state_head_dim: config.linear_value_head_dim,
        entries: vec![],
    };
    eprintln!("oracle pre-call admitted ceiling={ceiling}; consult executed prefill receipt");
    qwen35::forward_prefill_batch(
        &mut gpu,
        &weights,
        &config,
        &manifest.tokens[..rows],
        0,
        &mut kv,
        &mut dn,
        &scratch,
        None,
        None,
        None,
        None,
    )?;
    snapshot(
        &gpu,
        dir,
        &mut manifest.entries,
        "prefill",
        rows,
        &config,
        &dn,
        &kv,
        &scratch,
    )?;
    for pos in rows..total {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            manifest.tokens[pos],
            pos,
            &mut kv,
            &mut dn,
            &scratch,
        )?;
        gpu.hip.device_synchronize()?;
        save(
            &gpu,
            dir,
            &mut manifest.entries,
            format!("decode_{}/logits", pos - rows),
            "f32",
            &scratch.logits,
            config.vocab_size * 4,
        )?;
    }
    if continuation > 0 {
        snapshot(
            &gpu,
            dir,
            &mut manifest.entries,
            "continued",
            total,
            &config,
            &dn,
            &kv,
            &scratch,
        )?;
    }
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    // Process isolation is intentional; OS teardown releases the single-run GPU.
    Ok(())
}

fn metrics(name: &str, a: &[f32], b: &[f32], raw_mismatches: Option<usize>) {
    assert_eq!(a.len(), b.len(), "{name}: element counts differ");
    let mut ss = 0f64;
    let mut reference = 0f64;
    let mut max = 0f64;
    let mut max_index = 0;
    let mut first = None;
    let mut finite_pairs = 0;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() && first.is_none() {
            first = Some(i);
        }
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        finite_pairs += 1;
        let d = (x as f64 - y as f64).abs();
        ss += d * d;
        reference += (x as f64).powi(2);
        if d > max {
            max = d;
            max_index = i;
        }
    }
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    };
    println!(
        "{}",
        serde_json::json!({"tensor": name, "elements": a.len(),
        "raw_byte_mismatches": raw_mismatches, "first_diff_index": first,
        "max_abs": max, "max_abs_index": max_index,
        "rmse": (ss / (finite_pairs.max(1) as f64)).sqrt(),
        "relative_l2": if reference > 0.0 { Some((ss/reference).sqrt()) } else { None },
        "nonfinite_a": a.iter().filter(|v| !v.is_finite()).count(),
        "nonfinite_b": b.iter().filter(|v| !v.is_finite()).count(),
        "argmax_a": if name.ends_with("logits") { argmax(a) } else { None },
        "argmax_b": if name.ends_with("logits") { argmax(b) } else { None }})
    );
}

fn compare(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        return Err("compare BASELINE_DIR CANDIDATE_DIR".into());
    }
    let load = |dir: &str| -> Result<Manifest> {
        Ok(serde_json::from_slice(&fs::read(
            Path::new(dir).join("manifest.json"),
        )?)?)
    };
    let a = load(&args[0])?;
    let b = load(&args[1])?;
    if a.tokens != b.tokens
        || a.prefill_rows != b.prefill_rows
        || a.state_head_dim != b.state_head_dim
        || a.entries.len() != b.entries.len()
        || a.model != b.model
        || a.arch != b.arch
    {
        return Err(
            "incompatible manifests (model/path, arch, tokens, rows, or state shape)".into(),
        );
    }
    for (x, y) in a.entries.iter().zip(&b.entries) {
        if x.name != y.name || x.kind != y.kind || x.bytes != y.bytes {
            return Err("tensor metadata mismatch".into());
        }
        let xb = fs::read(Path::new(&args[0]).join(&x.file))?;
        let yb = fs::read(Path::new(&args[1]).join(&y.file))?;
        if xb.len() != x.bytes || yb.len() != y.bytes {
            return Err("truncated tensor dump".into());
        }
        metrics(
            &x.name,
            &decode(&x.kind, &xb),
            &decode(&y.kind, &yb),
            Some(xb.iter().zip(&yb).filter(|(u, v)| u != v).count()),
        );
        if let Some(prefix) = x.name.strip_suffix("/state_codes") {
            let reconstruct =
                |manifest: &Manifest, dir: &str, codes: &[u8]| -> Result<(Vec<f32>, Vec<f32>)> {
                    let component = |suffix: &str| -> Result<Vec<f32>> {
                        let name = format!("{prefix}/{suffix}");
                        let entry = manifest
                            .entries
                            .iter()
                            .find(|e| e.name == name)
                            .ok_or("missing state component")?;
                        let raw = fs::read(Path::new(dir).join(&entry.file))?;
                        if raw.len() != entry.bytes {
                            return Err("truncated state component".into());
                        }
                        Ok(decode(&entry.kind, &raw))
                    };
                    let scales = component("state_scales")?;
                    let ef = component("ef")?;
                    if ef.len() != codes.len()
                        || scales.len() * manifest.state_head_dim != codes.len()
                    {
                        return Err("invalid rowwise Q8 state geometry".into());
                    }
                    let state: Vec<f32> = codes
                        .iter()
                        .enumerate()
                        .map(|(i, &q)| (q as i8) as f32 * scales[i / manifest.state_head_dim])
                        .collect();
                    let effective = state.iter().zip(ef).map(|(s, e)| s + e).collect();
                    Ok((state, effective))
                };
            let (sa, ea) = reconstruct(&a, &args[0], &xb)?;
            let (sb, eb) = reconstruct(&b, &args[1], &yb)?;
            metrics(&format!("{prefix}/state_dequantized"), &sa, &sb, None);
            metrics(&format!("{prefix}/state_plus_ef"), &ea, &eb, None);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("dump") => dump(&args[1..]),
        Some("compare") => compare(&args[1..]),
        _ => Err("use dump MODEL PROMPT.txt ROWS CONTINUATION_STEPS NEW_OUTPUT_DIR | compare BASE CANDIDATE".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_specials_and_subnormals() {
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xbc00), -1.0);
        assert_eq!(half(1), 2f32.powi(-24));
        assert!(half(0x7c00).is_infinite());
        assert!(half(0x7e00).is_nan());
    }
    #[test]
    fn q8_block_decodes_signed_codes_and_scale() {
        let mut block = vec![0u8; 34];
        block[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
        block[2] = 127;
        block[3] = (-127i8) as u8;
        let decoded = decode("q8_0", &block);
        assert_eq!(decoded.len(), 32);
        assert_eq!(&decoded[..3], &[63.5, -63.5, 0.0]);
    }
}
