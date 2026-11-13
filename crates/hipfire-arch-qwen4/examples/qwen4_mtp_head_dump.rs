// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G3 head-logit oracle (stock-FP8 MTP sidecar, gfx942): dump the production
//! MTP head's logits at `--positions` consecutive positions together with the
//! exact inputs the head consumed, so an independent PyTorch run of `mtp.*`
//! (`beat/M/g3/mtp_head_ref.py`) can be compared with `KL(ref || hipfire)`.
//!
//! What one evaluated position `n` is (all of it is the production path, see
//! `beat/M/g3/README.md` for the file:line evidence):
//!   1. `Qwen4MtpDrafter::mtp_prefill(tokens[..n])` from a cold state. It runs
//!      the trunk over the prompt and appends one head row per prompt token
//!      (row `p` = token `p` paired with the trunk hidden of position `p - 1`;
//!      row 0 pairs with its own hidden), exactly the serve prompt fill
//!      (`mtp_spec.rs` `mtp_append_prefill_chunk`). It leaves `pending` = the
//!      trunk's `[hc_count * hidden]` F32 output streams (pre final mixer) of
//!      position `n - 1`.
//!   2. One production head Predict step (`Qwen4Bundle::mtp_forward_token`
//!      through `wide_verify_oracle::dev_head_chain`, depth 1): token
//!      `tokens[n]` at head position `n`, backbone hidden `pending`, fresh QSA
//!      selection. The draft head's full-vocabulary logit row is read back
//!      (`Qwen4Bundle::mtp_draft_read`); it is the model's own `lm_head`
//!      applied to the head's collapsed hidden, which needs
//!      `HIPFIRE_MTP_DRAFT_HEAD=none` (the default `mq2r` ranks a 2-bit copy and
//!      only its top 8 are exact; the example refuses that).
//!
//! Every prefill `n = 1..` also yields `pending` = trunk hidden of position
//! `n - 1`; all of them are dumped (`trunk_hidden`), because the reference has
//! to rebuild the head's K/V rows for every prior position from them.
//!
//! The model is loaded with its `<model>.mtp` sidecar bound through
//! `bind_stock_mtp_sidecar` and merged at admission
//! (`admit_hfqm_artifact_with(.., Some(&bound))`); the `mtp.*` payloads are
//! served `Borrowed` from the sidecar's own mapping, every other tensor from
//! the base, as the product carrier does.
//!
//! usage: qwen4_mtp_head_dump --model M.hfq (--prompt-file P.txt | --ids 1,2,3,..)
//!                            --out DIR [--prefix 48] [--positions 16]
//!                            [--n-ctx 2048] [--max-k 3] [--sidecar PATH]
//!
//! Needs `tokens >= prefix + positions + 1` (seed token and teacher-forced
//! next token of every evaluated position). Output files, all little endian:
//! `index.json`, `tokens.u32.bin`, `trunk_hidden.f32.bin`, `head_logits.f32.bin`, and
//! for a stock-FP8 head `head_router.f32.bin`, `head_topk_ids.i32.bin`,
//! `head_topk_w.f32.bin` (the head MoE's own routing at each evaluated position,
//! read right after its Predict step; `index.json` `head_routing` says what they hold).

use hipfire_arch_qwen4::artifact::{admit_hfqm_artifact_with, source_forward_scratch_mode};
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter;
use hipfire_arch_qwen4::sampled_lane::DraftRead;
use hipfire_arch_qwen4::stock_mtp_sidecar::{
    bind_stock_mtp_sidecar, qwen4_stock_mtp_sidecar_path, stock_mtp_attached_line,
    stock_mtp_refusal_line,
};
use hipfire_arch_qwen4::wide_verify_oracle as oracle;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::spec::MtpDrafter;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::Gpu;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, String>;

const USAGE: &str = "usage: qwen4_mtp_head_dump --model M.hfq (--prompt-file P.txt | --ids 1,2,3,..) --out DIR [--prefix 48] [--positions 16] [--n-ctx 2048] [--max-k 3] [--sidecar PATH]";

fn es<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

struct Args {
    model: PathBuf,
    prompt_file: Option<PathBuf>,
    ids: Option<Vec<u32>>,
    out: PathBuf,
    prefix: usize,
    positions: usize,
    n_ctx: usize,
    max_k: usize,
    sidecar: Option<PathBuf>,
}

fn parse_args() -> R<Args> {
    let mut model = None;
    let mut prompt_file = None;
    let mut ids = None;
    let mut out = None;
    let mut prefix = 48usize;
    let mut positions = 16usize;
    let mut n_ctx = 2048usize;
    let mut max_k = 3usize;
    let mut sidecar = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value("--model")?)),
            "--prompt-file" => prompt_file = Some(PathBuf::from(value("--prompt-file")?)),
            "--ids" => {
                let list = value("--ids")?;
                ids = Some(
                    list.split(',')
                        .map(|s| s.trim().parse::<u32>().map_err(es))
                        .collect::<R<Vec<u32>>>()?,
                );
            }
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--prefix" => prefix = value("--prefix")?.parse().map_err(es)?,
            "--positions" => positions = value("--positions")?.parse().map_err(es)?,
            "--n-ctx" => n_ctx = value("--n-ctx")?.parse().map_err(es)?,
            "--max-k" => max_k = value("--max-k")?.parse().map_err(es)?,
            "--sidecar" => sidecar = Some(PathBuf::from(value("--sidecar")?)),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    let model = model.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    let out = out.ok_or_else(|| format!("--out is required\n{USAGE}"))?;
    if prompt_file.is_none() == ids.is_none() {
        return Err(format!(
            "give exactly one of --prompt-file / --ids\n{USAGE}"
        ));
    }
    if prefix == 0 || positions == 0 {
        return Err("--prefix and --positions must be >= 1".into());
    }
    if prefix + positions + 1 > n_ctx {
        return Err(format!(
            "--n-ctx {n_ctx} is too small for prefix {prefix} + positions {positions} + 1"
        ));
    }
    Ok(Args {
        model,
        prompt_file,
        ids,
        out,
        prefix,
        positions,
        n_ctx,
        max_k,
        sidecar,
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn f32_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &x) in row.iter().enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0 as u32
}

fn main() {
    if let Err(message) = run() {
        eprintln!("qwen4_mtp_head_dump: {message}");
        std::process::exit(2);
    }
}

fn run() -> R<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out).map_err(es)?;

    // ── base + sidecar ──────────────────────────────────────────────────
    let mut hfq = HfqFile::open(&args.model).map_err(es)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(es)?;
    let sidecar_path = args
        .sidecar
        .clone()
        .unwrap_or_else(|| qwen4_stock_mtp_sidecar_path(&args.model));
    let bound = match bind_stock_mtp_sidecar(&hfq, &sidecar_path) {
        Ok(bound) => bound,
        Err(reason) => {
            return Err(stock_mtp_refusal_line(&sidecar_path, &reason));
        }
    };
    let attached_line = stock_mtp_attached_line(&bound);
    println!("{attached_line}");
    let sidecar_index_sha256 = hex(&bound.index_sha256);
    let sidecar_payload_bytes = bound.payload_bytes;

    // ── tokens ──────────────────────────────────────────────────────────
    let tokens: Vec<u32> = match (&args.prompt_file, &args.ids) {
        (Some(path), _) => {
            let text = std::fs::read_to_string(path).map_err(es)?;
            tokenizer.encode(&text)
        }
        (_, Some(ids)) => ids.clone(),
        _ => unreachable!("parse_args admits exactly one source"),
    };
    let need = args.prefix + args.positions + 1;
    if tokens.len() < need {
        return Err(format!(
            "the prompt has {} tokens, prefix {} + positions {} + 1 = {need} are needed",
            tokens.len(),
            args.prefix,
            args.positions
        ));
    }
    let tokens = &tokens[..need];

    // The head's draft row must be the full vocabulary of the source head.
    let draft_env = std::env::var("HIPFIRE_MTP_DRAFT_HEAD").unwrap_or_default();
    if draft_env != "none" {
        return Err(
            "set HIPFIRE_MTP_DRAFT_HEAD=none: the default `mq2r` draft head ranks a 2-bit copy of \
             lm_head and only its top 8 logits are exact, so no full-vocabulary row exists to compare"
                .into(),
        );
    }

    // ── load (qwen4_rows / fn_lane_oracle loader, two-file source) ───────
    hipfire_loader::prepare_host_memory_for(&args.model);
    let mut gpu = Gpu::init().map_err(es)?;
    let receipt =
        admit_hfqm_artifact_with(&hfq, Some(&gpu.arch), false, Some(&bound)).map_err(es)?;
    let profile_id = receipt.config.profile.id().to_string();
    let hidden_size = receipt.config.hidden_size;
    let hc_count = receipt.config.hc_count;
    let vocab = receipt.config.vocab_size;
    let wide = hidden_size * hc_count;
    println!(
        "qwen4_mtp_head_dump: admitted profile {profile_id} (head {}; mtp sidecar receipt {})",
        if receipt.config.mtp.is_some() {
            "declared"
        } else {
            "ABSENT"
        },
        receipt
            .mtp_sidecar
            .as_ref()
            .map(|s| format!(
                "{} bytes, index {}",
                s.payload_bytes,
                hex(&s.index_sha256)[..12].to_string()
            ))
            .unwrap_or_else(|| "none".to_string()),
    );
    if receipt.config.mtp.is_none() || receipt.mtp_sidecar.is_none() {
        return Err("admission did not attach the MTP head sidecar".into());
    }
    let scratch_mode =
        source_forward_scratch_mode(&gpu, &hfq, &receipt.config, &receipt.manifest).map_err(es)?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(es)?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    // The sidecar is served Borrowed from its own mapping (never a range: a second
    // backing file would fail the direct reader's identity check).
    let mut side = HfqModelSource::from_hfq(bound.hfq);
    side.set_prefer_ranges(false);
    let weights = {
        let mut weights = receipt.manifest.weights.clone();
        match hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()? {
            None => {}
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Layers(layers)) => {
                if gpu.is_uma() {
                    return Err("host-mapped experts are only for discrete GPUs".into());
                }
                let moved = hipfire_arch_qwen4::expert_residency::place_routed_experts(
                    &mut weights,
                    layers,
                );
                eprintln!(
                    "qwen4_mtp_head_dump: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                );
            }
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Auto) => {
                return Err(
                    "qwen4_mtp_head_dump takes a fixed expert layer count, not auto".into(),
                );
            }
        }
        weights
    };
    let load_started = Instant::now();
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            // The one total routing rule: `mtp.*` is the sidecar's, everything else the base's.
            if entry.name.starts_with("mtp.") {
                let (info, bytes) = side
                    .tensor_data(&entry.name)
                    .ok_or_else(|| format!("missing sidecar tensor '{}'", entry.name))?;
                return Ok(SourcePayload::Borrowed { info, bytes });
            }
            if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                &source,
                &receipt.manifest.identity.recipe,
                entry,
            ) {
                return payload;
            }
            if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped {
                let (info, bytes) = source
                    .tensor_data(&entry.name)
                    .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                return Ok(SourcePayload::Borrowed { info, bytes });
            }
            source
                .tensor_range(&entry.name)
                .map_err(es)?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(es)?;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
        &gpu,
        &receipt.config,
    )?;
    let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        args.n_ctx,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(es)?;
    bundle
        .attach_forward(&mut gpu, args.n_ctx, scratch_mode)
        .map_err(es)?;
    bundle.attach_mtp(&mut gpu, args.n_ctx).map_err(es)?;
    // The loader settles the int8 routed-down verdict right after its head-attach step.
    bundle.settle_moe_down_a8_head(&mut gpu, true).map_err(es)?;
    if !oracle::dev_head_attached(&bundle) {
        return Err("the bundle has no attached MTP head after attach_mtp".into());
    }
    println!(
        "qwen4_mtp_head_dump: head attached on {} (profile {profile_id}; sidecar {} payload bytes; \
         load {:.1} s; head position {})",
        gpu.arch,
        sidecar_payload_bytes,
        load_started.elapsed().as_secs_f64(),
        oracle::dev_head_position(&bundle)?,
    );

    // ── the evaluation loop ─────────────────────────────────────────────
    let mut drafter = Qwen4MtpDrafter::new(args.max_k, args.n_ctx, None);
    let last_prefill = args.prefix + args.positions - 1;
    let mut trunk_hidden: Vec<f32> = Vec::with_capacity(last_prefill * wide);
    let mut head_logits: Vec<f32> = Vec::with_capacity(args.positions * vocab);
    let mut eval_positions = Vec::with_capacity(args.positions);
    let mut hipfire_top1 = Vec::with_capacity(args.positions);
    let mut head_router: Vec<f32> = Vec::new();
    let mut head_topk_ids: Vec<i32> = Vec::new();
    let mut head_topk_w: Vec<f32> = Vec::new();
    let mut routing_rows = 0usize;
    let body = (|| -> R<()> {
        for n in 1..=last_prefill {
            // 1. cold prefill of tokens[..n]: trunk + head rows 0..n-1, pending = h_{n-1}.
            let prompt = &tokens[..n];
            drafter
                .mtp_prefill(&mut gpu, &mut bundle, prompt, prompt, 0, false, &|| false)?
                .into_first_token()?;
            let position = oracle::dev_position(&bundle);
            let head_position = oracle::dev_head_position(&bundle)?;
            if position != n || head_position != n {
                return Err(format!(
                    "prefill of {n} tokens left the target at {position} and the head at {head_position}"
                ));
            }
            let bytes = drafter.lab_pending_hidden_bytes(&gpu)?;
            if bytes.len() != wide * 4 {
                return Err(format!(
                    "pending hidden is {} bytes, expected {} ([{hc_count} x {hidden_size}] F32)",
                    bytes.len(),
                    wide * 4
                ));
            }
            let pending_host: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if pending_host.iter().any(|v| !v.is_finite()) {
                return Err(format!("trunk hidden of position {} is not finite", n - 1));
            }
            trunk_hidden.extend_from_slice(&pending_host);
            if n < args.prefix {
                continue;
            }
            // 2. production head Predict step: token tokens[n] at head position n on `pending`.
            let pending = gpu.upload_f32(&pending_host, &[wide]).map_err(es)?;
            let drafted =
                oracle::dev_head_chain(&mut bundle, &mut gpu, tokens[n], &pending, position, 1);
            let freed = gpu.free_tensor(pending).map_err(es);
            let drafted = drafted?;
            freed?;
            gpu.hip.device_synchronize().map_err(es)?;
            let row =
                match bundle.mtp_draft_read(&gpu).map_err(es)? {
                    DraftRead::FullRow(row) => row,
                    DraftRead::Candidates(_) => return Err(
                        "the draft head re-scored a quantized copy (candidates only); run with \
                         HIPFIRE_MTP_DRAFT_HEAD=none"
                            .into(),
                    ),
                };
            if row.len() != vocab {
                return Err(format!(
                    "head logit row has {} entries, vocab is {vocab}",
                    row.len()
                ));
            }
            // The head MoE's own routing of that Predict step. Nothing but a Predict step
            // writes these buffers (the cold prefill's Append steps skip the FFN, the draft
            // read above is the lm_head only), so they are this position's.
            if let Some(routing) = bundle.mtp_lab_fp8_head_routing(&gpu).map_err(es)? {
                if routing.router.iter().any(|v| !v.is_finite())
                    || routing.topk_ids.iter().any(|&e| e < 0)
                    || routing.topk_weights.iter().any(|v| !v.is_finite())
                {
                    return Err(format!(
                        "head router buffers at position {n} are not finite / hold a failed row \
                         (ids {:?})",
                        routing.topk_ids
                    ));
                }
                head_router.extend_from_slice(&routing.router);
                head_topk_ids.extend_from_slice(&routing.topk_ids);
                head_topk_w.extend_from_slice(&routing.topk_weights);
                routing_rows += 1;
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(format!("head logits at position {n} are not finite"));
            }
            let top1 = argmax(&row);
            if drafted.first() != Some(&top1) {
                return Err(format!(
                    "head draft {:?} is not the argmax {top1} of its own logit row at position {n}",
                    drafted
                ));
            }
            println!(
                "qwen4_mtp_head_dump: position {n}: seed {} -> head top1 {top1} (teacher next {})",
                tokens[n],
                tokens[n + 1]
            );
            eval_positions.push(n);
            hipfire_top1.push(top1);
            head_logits.extend_from_slice(&row);
        }
        Ok(())
    })();
    Box::new(drafter).mtp_free(&mut gpu);
    body?;

    // ── dump ────────────────────────────────────────────────────────────
    let rows = trunk_hidden.len() / wide;
    std::fs::write(
        args.out.join("tokens.u32.bin"),
        tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .map_err(es)?;
    std::fs::write(args.out.join("trunk_hidden.f32.bin"), f32_le(&trunk_hidden)).map_err(es)?;
    std::fs::write(args.out.join("head_logits.f32.bin"), f32_le(&head_logits)).map_err(es)?;
    let routing_top_k = if routing_rows > 0 {
        head_topk_ids.len() / routing_rows
    } else {
        0
    };
    let routing_experts = if routing_rows > 0 {
        head_router.len() / routing_rows
    } else {
        0
    };
    if routing_rows != 0 {
        if routing_rows != eval_positions.len() {
            return Err(format!(
                "head routing was read at {routing_rows} of {} positions",
                eval_positions.len()
            ));
        }
        std::fs::write(args.out.join("head_router.f32.bin"), f32_le(&head_router)).map_err(es)?;
        std::fs::write(
            args.out.join("head_topk_ids.i32.bin"),
            head_topk_ids
                .iter()
                .flat_map(|e| e.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .map_err(es)?;
        std::fs::write(args.out.join("head_topk_w.f32.bin"), f32_le(&head_topk_w)).map_err(es)?;
    }
    let mut index = json!({
        "schema": "qwen4_mtp_head_dump/1",
        "model": args.model.display().to_string(),
        "sidecar": sidecar_path.display().to_string(),
        "sidecar_attached_line": attached_line,
        "sidecar_index_sha256": sidecar_index_sha256,
        "sidecar_payload_bytes": sidecar_payload_bytes,
        "profile": profile_id,
        "gpu_arch": gpu.arch,
        "hidden_size": hidden_size,
        "hc_count": hc_count,
        "vocab_size": vocab,
        "draft_head": "none (source lm_head, full vocabulary, token order)",
        "prefix": args.prefix,
        "positions": args.positions,
        "eval_positions": eval_positions,
        "hipfire_top1": hipfire_top1,
        "teacher_next": eval_positions.iter().map(|&n| tokens[n + 1]).collect::<Vec<u32>>(),
        "pairing": "head row p = (embed(tokens[p]), trunk_hidden[max(p-1,0)]); evaluated position n = head row n with trunk_hidden[n-1] and tokens[n]; logits predict tokens[n+1]",
        "tokens": {"file": "tokens.u32.bin", "dtype": "u32", "shape": [tokens.len()]},
        "trunk_hidden": {
            "file": "trunk_hidden.f32.bin",
            "dtype": "f32",
            "shape": [rows, wide],
            "meaning": "row j = the trunk's final-layer output hyper-connection streams [hc_count, hidden] of position j, before the final mixer (captured by the prefill of tokens[..j+1])"
        },
        "head_logits": {
            "file": "head_logits.f32.bin",
            "dtype": "f32",
            "shape": [args.positions, vocab],
            "meaning": "row i = production head logits (lm_head over the head's collapsed hidden) at eval_positions[i]"
        },
    });
    if routing_rows != 0 {
        index["head_routing"] = json!({
            "files": {
                "router": "head_router.f32.bin",
                "topk_ids": "head_topk_ids.i32.bin",
                "topk_w": "head_topk_w.f32.bin"
            },
            "dtypes": {"router": "f32", "topk_ids": "i32", "topk_w": "f32"},
            "shapes": {
                "router": [routing_rows, routing_experts],
                "topk_ids": [routing_rows, routing_top_k],
                "topk_w": [routing_rows, routing_top_k]
            },
            "top_k": routing_top_k,
            "router_kind": "logits",
            "router_selection_rounding": "bf16_rne",
            "router_meaning": "router row i = the head MoE's RAW F32 router logits (x @ W_router^T of the head MoE input; neither softmax probabilities nor BF16-rounded; the router kernel moe_router_softmax_top10_f32 reads them const) at eval_positions[i]; the runtime selects on bf16_rne(logit), ties by (value desc, expert index asc). topk_ids row i = the 10 selected expert ids, best first. topk_w row i = softmax over all 512 BF16-rounded logits, renormalised over the selected 10 (norm_topk_prob=true), each stored BF16-rounded, in topk_ids order",
        });
    }
    std::fs::write(
        args.out.join("index.json"),
        serde_json::to_vec_pretty(&index).map_err(es)?,
    )
    .map_err(es)?;
    println!(
        "qwen4_mtp_head_dump: wrote {} ({} positions x {vocab} logits, {rows} trunk rows)",
        Path::new(&args.out).display(),
        args.positions
    );
    Ok(())
}
