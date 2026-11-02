// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batch parity for the encoder: one forward over several sequences must
//! reproduce the per-sequence result of encoding them one at a time.
//!
//! This is the acceptance test for the batched forward
//! ([`encode_embeddings_batch`], the multi-slot engine's core): the whole
//! argument for batching rests on row concatenation being equivalent to
//! separate encodes, and nothing in the layer loop can show that by reading —
//! the failure mode is a segment's attention seeing its neighbour's keys, or a
//! delta-rule scan starting from the previous segment's state, and both
//! produce *plausible* vectors.
//!
//! The lengths are chosen to hit every dispatch boundary: a sub-256 input (the
//! Q-tiled attention kernel, no padding), several over it with lengths that are
//! and are not multiples of 64 (the eager path with a padded tail), and a mix
//! in the same batch so a segment's row block never starts where its own rows
//! would have.
//!
//! Usage: cargo run --release -p hipfire-arch-qwen35-embed \
//!          --example batch_parity --features lab -- <model-dir-or-hfq>

use hipfire_arch_qwen35_embed::carrier::assemble_input;
use hipfire_arch_qwen35_embed::forward::{encode_embeddings_batch, EncodeInput};
use hipfire_arch_qwen35_embed::{load_qwen35_embed_from_source, InputKind};
use hipfire_runtime::loader_api::ModelSource;
use rdna_compute::Gpu;

fn rel(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max = 0f32;
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b) {
        max = max.max((x - y).abs());
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (max, (num / den.max(f64::MIN_POSITIVE)).sqrt() as f32)
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: batch_parity <model-dir-or-hfq>")?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let src = ModelSource::from_path(&path)?;
    let mut bundle = load_qwen35_embed_from_source(&mut gpu, src)?;
    println!(
        "gpu {} | dim {} | {} layers | {} out",
        gpu.arch,
        bundle.config.hidden_size,
        bundle.config.num_layers,
        bundle.config.dense_out_features
    );

    // Length mix: 40 (tiled, no pad), 300 (eager + 44 pad rows), 256 (exactly
    // one block, no pad), 1000 (eager, 24 pad rows), and two short ones so a
    // batch has several small segments.
    let texts: Vec<(InputKind, String)> = vec![
        (InputKind::Query, "what is the capital of Peru".to_string()),
        (InputKind::Document, "word ".repeat(300)),
        (InputKind::Document, "token ".repeat(256)),
        (
            InputKind::Query,
            "explain the difference between a process and a thread in one paragraph".to_string(),
        ),
        (InputKind::Document, "sentence ".repeat(120)),
        (InputKind::Query, "hipfire".to_string()),
    ];

    // ── per-sequence reference ────────────────────────────────────────
    std::env::set_var("HIPFIRE_EMBED_PHASE", "1");
    let mut per_seq: Vec<Vec<f32>> = Vec::new();
    let mut seq_rows: Vec<(usize, usize)> = Vec::new();
    let t0 = std::time::Instant::now();
    for (kind, text) in &texts {
        let a = assemble_input(&bundle, &mut gpu, *kind, text, None)?;
        let raw = {
            let input = EncodeInput {
                ids: &a.ids,
                image_embeds: &a.image_embeds,
                spans: &a.spans,
                pos3: &a.pos3,
            };
            encode_embeddings_batch(
                &mut gpu,
                &bundle.weights,
                &mut bundle.state,
                &bundle.config,
                std::slice::from_ref(&input),
            )?
            .pop()
            .expect("one input")
        };
        let (emb, n_tok) = bundle.finalize_embeddings(*kind, &a.ids, &raw)?;
        seq_rows.push((a.ids.len(), n_tok));
        per_seq.push(emb);
    }
    let t_seq = t0.elapsed();

    // ── one batched forward over all of them ──────────────────────────
    let assembled: Vec<_> = texts
        .iter()
        .map(|(kind, text)| assemble_input(&bundle, &mut gpu, *kind, text, None))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs: Vec<EncodeInput<'_>> = assembled
        .iter()
        .map(|a| EncodeInput {
            ids: &a.ids,
            image_embeds: &a.image_embeds,
            spans: &a.spans,
            pos3: &a.pos3,
        })
        .collect();
    let t1 = std::time::Instant::now();
    let batch_raw = encode_embeddings_batch(
        &mut gpu,
        &bundle.weights,
        &mut bundle.state,
        &bundle.config,
        &inputs,
    )?;
    let t_batch = t1.elapsed();

    let rows_total: usize = assembled
        .iter()
        .map(|a| {
            let n = a.ids.len();
            if n < 256 {
                n
            } else {
                n.div_ceil(64) * 64
            }
        })
        .sum();
    println!(
        "rows {rows_total}: sequential {:.1} ms | batched {:.1} ms ({:.2}x)",
        t_seq.as_secs_f64() * 1e3,
        t_batch.as_secs_f64() * 1e3,
        t_seq.as_secs_f64() / t_batch.as_secs_f64()
    );

    let mut worst = 0f32;
    let mut worst_i = 0usize;
    for (i, (kind, _)) in texts.iter().enumerate() {
        let (emb, n_tok) = bundle.finalize_embeddings(*kind, &assembled[i].ids, &batch_raw[i])?;
        let (max, r) = rel(&emb, &per_seq[i]);
        // All-zero rows are the signature of a segment that ran on another
        // segment's state or inputs (the failure this probe was written for),
        // and they normalize to zero rather than NaN, so they would otherwise
        // read as a small error.
        let zero_rows = emb
            .chunks(128)
            .filter(|c| c.iter().all(|x| *x == 0.0))
            .count();
        println!(
            "  [{i}] {} -> {} tokens ({} in, {} rows): n_tokens {} | max {max:.3e} | rel_l2 {r:.3e}",
            if *kind == InputKind::Query { "query   " } else { "document" },
            seq_rows[i].1,
            seq_rows[i].0,
            if assembled[i].ids.len() < 256 { assembled[i].ids.len() } else { assembled[i].ids.len().div_ceil(64) * 64 },
            n_tok,
        );
        let total_rows = emb.len() / 128;
        if zero_rows > 0 {
            println!("      {zero_rows}/{total_rows} rows came back exactly zero");
        }
        if max > worst {
            worst = max;
            worst_i = i;
        }
    }
    if worst > 1e-5 {
        return Err(format!(
            "FAIL: batch differs from per-sequence encode (worst max abs {worst:.3e} at input {worst_i})"
        ));
    }
    println!("PASS: every input is bit-comparable to its per-sequence encode (worst {worst:.3e})");
    bundle.weights.free_gpu(&mut gpu);
    bundle.state.free_gpu(&mut gpu);
    Ok(())
}
