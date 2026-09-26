// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_kld — KLD reference builder and candidate scorer for Qwen4
//! (Qwen3.8 Flash-Next), on the saddle-quant HFKLDR v3 / HFKSEQ v2 formats.
//!
//! `ref` and `eval` run the production `Qwen4Bundle::forward_chunk` (all-row
//! logits, per-chunk `reset`) so an engine teacher and the candidate share one
//! route and differ only in weights.
//!
//! The engine can only run Qwen4 with MQ4 routed experts, so an engine
//! teacher (e.g. a BF16-trunk HFQM) measures trunk / head / PLE damage only.
//! The source-precision teacher is the BF16 checkpoint itself, run by
//! `reference_oracle/kld_teacher.py` over a template reference's tokens;
//! `import` turns its logits into a reference with the same top-k, NLL and
//! plausibility gate as `ref`.
//!
//! Usage:
//!   qwen4_kld ref    --model TEACHER.hfq --slice TEXT --output REF.kldref \
//!                    [--n-ctx 512] [--top-k 256] [--max-chunks N]
//!   qwen4_kld import --logits TEACHER.logits --ref TEMPLATE.kldref \
//!                    --teacher ID_FILE --output REF.kldref [--top-k 256]
//!   qwen4_kld eval   --model CANDIDATE.hfq --ref REF.kldref --output OUT.kldseq \
//!                    [--max-chunks N]
//! Reduce with `saddle-quant reduce DIR`.

use hipfire_arch_qwen4::admit_hfqm_artifact;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use saddle_quant::eval::estimator::{kld_from_block, nll_of, topk_block, TopKBlock};
use saddle_quant::eval::teacher::{evaluate, TeacherGate};
use saddle_quant::format::kldref::{KldRef, RefHeader, RefWriter};
use saddle_quant::format::kldseq::{self, ChunkScore};
use saddle_quant::{ArtifactId, Estimator, OracleStats, TeacherVerdict, WindowSpec};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage:\n  qwen4_kld ref    --model TEACHER.hfq --slice TEXT --output REF.kldref [--n-ctx 512] [--top-k 256] [--max-chunks N]\n  qwen4_kld import --logits TEACHER.logits --ref TEMPLATE.kldref --teacher ID_FILE --output REF.kldref [--top-k 256]\n  qwen4_kld eval   --model CANDIDATE.hfq --ref REF.kldref --output OUT.kldseq [--max-chunks N]";

struct Model {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    logits: GpuTensor,
    vocab: usize,
    arch_id: u32,
    tokenizer: Tokenizer,
}

impl Model {
    /// Same range-loaded path as the serve loader (`hipfire-loader` carriers).
    fn load(path: &Path, n_ctx: usize) -> Result<Self, String> {
        let mut hfq =
            HfqFile::open(path).map_err(|e| format!("open HFQM {}: {e}", path.display()))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let arch_id = hfq.arch_id;
        let receipt = admit_hfqm_artifact(&hfq).map_err(|e| format!("qwen4 admission: {e}"))?;
        let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
        if gpu.is_uma() {
            hfq.drop_mmap();
        }
        let mesh = DeviceMesh::single().map_err(|e| format!("qwen4 mesh: {e}"))?;
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let source = HfqModelSource::from_hfq(hfq);
        let transaction = fulfill_manifest_from_payloads(
            &receipt.manifest.weights,
            &mesh,
            receipt.config.num_hidden_layers,
            &mut gpu,
            expected,
            |entry| {
                source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name))
            },
        )
        .map_err(|e| format!("qwen4 manifest fulfillment: {e}"))?;
        let vocab = receipt.config.vocab_size;
        let mut bundle = Qwen4Bundle::assemble_with_metadata(
            receipt.config,
            transaction,
            &receipt.placements,
            &mut gpu,
            n_ctx,
            receipt.ple,
        )
        .map_err(|e| format!("qwen4 bundle assembly: {e}"))?;
        bundle
            .attach_forward(&mut gpu, n_ctx)
            .map_err(|e| format!("qwen4 forward setup: {e}"))?;
        let logits = gpu
            .zeros(&[n_ctx * vocab], DType::F32)
            .map_err(|e| e.to_string())?;
        eprintln!(
            "qwen4_kld: loaded {} on {} (vocab {vocab})",
            path.display(),
            gpu.arch
        );
        Ok(Self {
            gpu,
            bundle,
            logits,
            vocab,
            arch_id,
            tokenizer,
        })
    }

    /// All-row logits for one chunk from a fresh state.
    fn chunk_logits(&mut self, chunk: &[u32]) -> Result<Vec<f32>, String> {
        self.bundle
            .reset(&mut self.gpu)
            .map_err(|e| e.to_string())?;
        self.bundle
            .forward_chunk(&mut self.gpu, chunk, &self.logits, None)
            .map_err(|e| e.to_string())?;
        let rows = self
            .gpu
            .download_f32(&self.logits)
            .map_err(|e| e.to_string())?;
        if rows.iter().any(|v| !v.is_finite()) {
            return Err("non-finite logits".into());
        }
        Ok(rows)
    }
}

fn artifact_id(path: &Path) -> Result<ArtifactId, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buf = vec![0u8; 16 << 20];
    let mut bytes = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
        bytes += n as u64;
    }
    Ok(ArtifactId {
        path: path.display().to_string(),
        sha256: format!("{:x}", digest.finalize()),
        bytes,
    })
}

fn engine_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Header facts that depend on the teacher source, not on the scored data.
struct RefMeta {
    arch_id: u32,
    n_ctx: usize,
    vocab: usize,
    window: WindowSpec,
    teacher: ArtifactId,
    corpus: ArtifactId,
    arch: String,
}

/// Top-k blocks plus summed target NLL for one chunk. `rows` holds only the
/// chunk's scored rows, `window.score_from..window.score_to`.
fn score_teacher_chunk(
    rows: &[f32],
    chunk: &[u32],
    meta: &RefMeta,
    top_k: usize,
    blocks: &mut Vec<TopKBlock>,
) -> Result<f64, String> {
    let mut nll = 0.0f64;
    for (row, pos) in rows
        .chunks_exact(meta.vocab)
        .zip(meta.window.score_from..meta.window.score_to)
    {
        blocks.push(topk_block(row, top_k).map_err(|e| e.to_string())?);
        nll += nll_of(row, chunk[pos + 1] as usize).map_err(|e| e.to_string())?;
    }
    Ok(nll)
}

/// Gate the teacher on its own PPL, then write the HFKLDR v3 reference.
fn write_ref(
    output: &Path,
    meta: RefMeta,
    top_k: usize,
    tokens: &[u32],
    blocks: &[TopKBlock],
    nll_sum: f64,
) -> Result<(), String> {
    let mean_nll = nll_sum / blocks.len() as f64;
    let oracle = OracleStats {
        mean_nll,
        ppl: mean_nll.exp(),
        n_scored: blocks.len(),
    };
    if let TeacherVerdict::Implausible { reason, .. } =
        evaluate(&TeacherGate::for_vocab(meta.vocab), oracle)
    {
        return Err(format!("teacher implausible: {reason}"));
    }
    let header = RefHeader {
        version: 3,
        arch_id: meta.arch_id,
        n_ctx: meta.n_ctx,
        n_chunk: tokens.len() / meta.n_ctx,
        n_vocab: meta.vocab,
        estimator: Estimator::TopK {
            k: top_k as u32,
            bias_vs_full: None,
        },
        window: meta.window,
        teacher: meta.teacher,
        corpus: meta.corpus,
        oracle,
        engine_commit: engine_commit(),
        arch: meta.arch,
    };
    let mut writer = RefWriter::create(output, header).map_err(|e| e.to_string())?;
    writer.push_tokens(tokens).map_err(|e| e.to_string())?;
    for block in blocks {
        writer
            .push_block(block.residual_logprob, &block.top)
            .map_err(|e| e.to_string())?;
    }
    writer.finish().map_err(|e| e.to_string())?;
    eprintln!(
        "qwen4_kld: wrote {}  teacher PPL {:.4} (mean NLL {mean_nll:.6}, {} tokens)",
        output.display(),
        oracle.ppl,
        oracle.n_scored
    );
    Ok(())
}

fn build_ref(
    model: &Path,
    slice: &Path,
    output: &Path,
    n_ctx: usize,
    top_k: usize,
    max_chunks: Option<usize>,
) -> Result<(), String> {
    let mut m = Model::load(model, n_ctx)?;
    let text =
        std::fs::read_to_string(slice).map_err(|e| format!("read {}: {e}", slice.display()))?;
    // Qwen tokenizers add no BOS: chunks are plain n_ctx windows of the stream.
    let stream = m.tokenizer.encode(&text);
    let n_chunk = (stream.len() / n_ctx).min(max_chunks.unwrap_or(usize::MAX));
    if n_chunk == 0 {
        return Err(format!(
            "slice has {} tokens, fewer than n_ctx {n_ctx}",
            stream.len()
        ));
    }
    let tokens = &stream[..n_chunk * n_ctx];
    let window = WindowSpec::legacy_half(n_ctx);
    eprintln!(
        "qwen4_kld ref: {n_chunk} chunks x {n_ctx}, scoring [{}, {})",
        window.score_from, window.score_to
    );
    let meta = RefMeta {
        arch_id: m.arch_id,
        n_ctx,
        vocab: m.vocab,
        window,
        teacher: artifact_id(model)?,
        corpus: artifact_id(slice)?,
        arch: m.gpu.arch.clone(),
    };

    // Blocks are small (n_chunk * scored * (8 + 8k) bytes), so buffer them and
    // write once the oracle statistics the header carries are known.
    let started = Instant::now();
    let mut blocks = Vec::with_capacity(n_chunk * window.scored_per_chunk());
    let mut nll_sum = 0.0f64;
    for (c, chunk) in tokens.chunks_exact(n_ctx).enumerate() {
        let rows = m.chunk_logits(chunk)?;
        let scored = &rows[window.score_from * m.vocab..window.score_to * m.vocab];
        nll_sum += score_teacher_chunk(scored, chunk, &meta, top_k, &mut blocks)?;
        eprintln!(
            "  chunk {}/{n_chunk}  running PPL {:.4}  {:.0}s",
            c + 1,
            (nll_sum / blocks.len() as f64).exp(),
            started.elapsed().as_secs_f64()
        );
    }
    write_ref(output, meta, top_k, tokens, &blocks, nll_sum)
}

/// Build a reference from externally computed teacher logits (the source
/// checkpoint teacher, `reference_oracle/kld_teacher.py`): F32
/// `[chunk][scored position][vocab]` over the token stream of `template`.
/// `teacher` is the file that identifies the teacher weights.
fn import_ref(
    logits: &Path,
    template: &Path,
    teacher: &Path,
    output: &Path,
    top_k: usize,
) -> Result<(), String> {
    let reference = KldRef::open(template).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(template).map_err(|e| e.to_string())?;
    // SAFETY: read-only maps of files this process does not modify.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
    let lfile =
        std::fs::File::open(logits).map_err(|e| format!("open {}: {e}", logits.display()))?;
    let lmap = unsafe { memmap2::Mmap::map(&lfile) }.map_err(|e| e.to_string())?;
    let h = &reference.header;
    let chunk_rows = h.window.scored_per_chunk() * h.n_vocab;
    // SAFETY: any bit pattern is a valid f32; the map is page-aligned and the
    // file is little-endian F32, which is this host's layout.
    let (head, rows, tail) = unsafe { lmap.align_to::<f32>() };
    if !head.is_empty() || !tail.is_empty() || rows.is_empty() || rows.len() % chunk_rows != 0 {
        return Err(format!(
            "{}: {} bytes is not a whole number of {}-row x {} chunks",
            logits.display(),
            lmap.len(),
            h.window.scored_per_chunk(),
            h.n_vocab
        ));
    }
    let n_chunk = rows.len() / chunk_rows;
    if n_chunk > h.n_chunk {
        return Err(format!(
            "{n_chunk} logit chunks but template has {}",
            h.n_chunk
        ));
    }
    let tokens = &reference.tokens(&mmap).map_err(|e| e.to_string())?[..n_chunk * h.n_ctx];
    let meta = RefMeta {
        arch_id: h.arch_id,
        n_ctx: h.n_ctx,
        vocab: h.n_vocab,
        window: h.window,
        teacher: artifact_id(teacher)?,
        corpus: h.corpus.clone(),
        arch: "pytorch-source".into(),
    };
    let mut blocks = Vec::with_capacity(n_chunk * h.window.scored_per_chunk());
    let mut nll_sum = 0.0f64;
    for (chunk_logits, chunk) in rows
        .chunks_exact(chunk_rows)
        .zip(tokens.chunks_exact(h.n_ctx))
    {
        nll_sum += score_teacher_chunk(chunk_logits, chunk, &meta, top_k, &mut blocks)?;
    }
    write_ref(output, meta, top_k, tokens, &blocks, nll_sum)
}

fn score(
    model: &Path,
    ref_path: &Path,
    output: &Path,
    max_chunks: Option<usize>,
) -> Result<(), String> {
    let reference = KldRef::open(ref_path).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(ref_path).map_err(|e| e.to_string())?;
    // SAFETY: read-only map of a file this process does not modify.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
    let h = &reference.header;
    let (n_ctx, window) = (h.n_ctx, h.window);
    let n_chunk = h.n_chunk.min(max_chunks.unwrap_or(usize::MAX));
    let tokens = reference.tokens(&mmap).map_err(|e| e.to_string())?;
    let mut m = Model::load(model, n_ctx)?;
    if m.vocab != h.n_vocab {
        return Err(format!(
            "candidate vocab {} != reference vocab {}",
            m.vocab, h.n_vocab
        ));
    }
    eprintln!(
        "qwen4_kld eval: {n_chunk} chunks x {n_ctx}; teacher PPL {:.4} ({})",
        h.oracle.ppl, h.teacher.path
    );

    let started = Instant::now();
    let mut chunks = Vec::with_capacity(n_chunk);
    let (mut kld_sum, mut nll_sum, mut top1_hits, mut scored) = (0.0f64, 0.0f64, 0usize, 0usize);
    for c in 0..n_chunk {
        let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
        let rows = m.chunk_logits(chunk)?;
        let mut klds = Vec::with_capacity(window.scored_per_chunk());
        let mut chunk_nll = 0.0f64;
        for (i, pos) in (window.score_from..window.score_to).enumerate() {
            let row = &rows[pos * m.vocab..(pos + 1) * m.vocab];
            let block = reference.block(&mmap, c, i).map_err(|e| e.to_string())?;
            let block = TopKBlock {
                residual_logprob: block.residual_logprob,
                top: block.top.to_vec(),
            };
            klds.push(kld_from_block(&block, row).map_err(|e| e.to_string())?);
            chunk_nll += nll_of(row, chunk[pos + 1] as usize).map_err(|e| e.to_string())?;
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32);
            top1_hits += usize::from(argmax == Some(block.top[0].0));
        }
        let n = klds.len();
        kld_sum += klds.iter().sum::<f64>();
        nll_sum += chunk_nll;
        scored += n;
        let mean_kld = klds.iter().sum::<f64>() / n as f64;
        klds.sort_by(f64::total_cmp);
        let p99_kld = klds[((n as f64 * 0.99).ceil() as usize).clamp(1, n) - 1];
        chunks.push(ChunkScore {
            mean_kld,
            p99_kld,
            mean_nll: chunk_nll / n as f64,
        });
        eprintln!(
            "  chunk {}/{n_chunk}  KLD {mean_kld:.6}  running KLD {:.6}  {:.0}s",
            c + 1,
            kld_sum / scored as f64,
            started.elapsed().as_secs_f64()
        );
    }
    kldseq::write(output, &chunks).map_err(|e| e.to_string())?;
    let mean_nll = nll_sum / scored as f64;
    eprintln!(
        "qwen4_kld eval: mean KLD = {:.6}  mean NLL = {mean_nll:.6}  PPL = {:.4}  top1 = {:.4}  ({scored} tokens) -> {}",
        kld_sum / scored as f64,
        mean_nll.exp(),
        top1_hits as f64 / scored as f64,
        output.display()
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or(USAGE)?;
    let (mut model, mut slice, mut ref_path, mut output) = (None, None, None, None);
    let (mut logits, mut teacher) = (None, None);
    let (mut n_ctx, mut top_k, mut max_chunks) = (512usize, 256usize, None);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        let number = |v: String| v.parse::<usize>().map_err(|e| format!("{flag}: {e}"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--slice" => slice = Some(PathBuf::from(value()?)),
            "--ref" => ref_path = Some(PathBuf::from(value()?)),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--logits" => logits = Some(PathBuf::from(value()?)),
            "--teacher" => teacher = Some(PathBuf::from(value()?)),
            "--n-ctx" => n_ctx = number(value()?)?,
            "--top-k" => top_k = number(value()?)?,
            "--max-chunks" => max_chunks = Some(number(value()?)?),
            _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
    }
    match (mode.as_str(), model, slice, ref_path, output) {
        ("ref", Some(model), Some(slice), None, Some(output)) => {
            build_ref(&model, &slice, &output, n_ctx, top_k, max_chunks)
        }
        ("eval", Some(model), None, Some(ref_path), Some(output)) => {
            score(&model, &ref_path, &output, max_chunks)
        }
        ("import", None, None, Some(template), Some(output)) => match (logits, teacher) {
            (Some(logits), Some(teacher)) => {
                import_ref(&logits, &template, &teacher, &output, top_k)
            }
            _ => Err(USAGE.into()),
        },
        _ => Err(USAGE.into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("qwen4_kld: {error}");
        std::process::exit(1);
    }
}
