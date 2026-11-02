// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_greedy_ids — greedy (argmax) token-id generation for Qwen4 HFQM
//! artifacts, one prompt at a time from a fresh production state.
//!
//! Identity harness driver: two builds of the engine (or two artifacts that
//! must agree bit-for-bit) run the same JSONL and their output files are
//! compared byte for byte.
//!
//! Model load follows `qwen4_kld`: admission on the GPU architecture
//! (`admit_hfqm_artifact_on`), slab layer arenas served by
//! `expert_arena::recipe_arena_payload`, and the routed-expert placement: unset (or
//! `auto`) `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` resolves the product loader's automatic
//! placement through the shared `expert_residency::incumbent_auto_resident_layers`; a
//! fixed `N` is an optional diagnostic override keeping layers 0..N resident. The
//! rest is host-mapped and borrowed from the resident source bytes.
//!
//! Usage:
//!   qwen4_greedy_ids --model MODEL.hfq --input PROMPTS.jsonl --output IDS.jsonl \
//!                    --max-new-tokens N [--max-seq 8192] [--thinking on|off]
//!
//! S1 slow-link validation: `--receipt RECEIPTS.jsonl` writes one line per
//! record, `{"id": .., "slowlink": [<route receipt per prefill call>]}`, from
//! `Qwen4Bundle::slowlink_receipt`; the ids file is unchanged, so an
//! incumbent arm and a layer-major arm (`HIPFIRE_QWEN4_SLOWLINK_PREFILL=0|1`)
//! still compare byte for byte, and the receipts prove which route each ran.
//! A requested receipt that is missing fails the run. An id record may carry
//! `"resume_at": N` (0 < N < len): the first N ids are prefilled, then the
//! rest as a second consecutive call entering at position N (the irregular
//! resumed-prefix fixture).
//!
//! `--input` (alias `--prompts`) / `--output` default to stdin / stdout (`-`
//! selects them too).
//!
//! Input, one JSON object per line (blank lines are skipped, unknown keys are
//! refused):
//!   {"id": <any JSON value>, "messages": [{"role": "user", "content": "..."}, ...],
//!    "tools": [..]?, "category": ".."?}
//!   {"id": <any JSON value>, "prompt_ids": [151644, 872, ...]}
//! Exactly one of `messages` / `prompt_ids`. `tools` (OpenAI tool definitions,
//! only with `messages`; an empty list means none, as in the daemon) is passed
//! to the template. `category` is a label for the caller and is ignored.
//!
//! `messages` are rendered the way the daemon's Qwen4 AR route renders a
//! request that carries a message history (`generate_qwen4_ar` →
//! `render_qwen4_prompt_cold`): each non-empty content goes through
//! `maybe_normalize_prompt` (`parse_generate_messages`), the artifact's own
//! `chat_template` renders them via `JinjaChatFrame::render_messages`
//! (generation prompt included; no system override, no reasoning effort,
//! `enable_thinking` from `--thinking`, default `off`), and the text is encoded
//! with the artifact tokenizer. A render failure aborts the run (the AR route
//! is fail-closed). `prompt_ids` bypass the template and tokenizer entirely.
//! A template that reads the current date (`current_date` / `strftime_now`) is
//! refused unless `HIPFIRE_CHAT_CURRENT_DATE` pins it: no prompt may depend on
//! the wall clock.
//!
//! Output, one line per input record, flushed as it is produced:
//!   {"id": <input id verbatim>, "token_ids": [..], "finish_reason": "stop"|"length",
//!    "prompt_tokens": N}
//!
//! Generation reproduces the daemon's Qwen4 AR producer with a greedy
//! request: `Qwen4Bundle::prefill_final` of the whole prompt from a fresh
//! state (no session reuse), the first token is the host argmax of the final
//! prefill row (`llama::argmax`), every later token is the GPU argmax of the
//! previous decode forward (`forward_token_or_argmax`). No sampling, no
//! penalties, no think-budget forcing, no stop sequences.
//!
//! EOS: a token that is the config `eos_token_id` or a tokenizer terminator
//! (`<|im_end|>` / `<|endoftext|>`) ends the record, IS the last element of
//! `token_ids` (the daemon commits and emits it before it stops) and gives
//! `finish_reason` "stop". Otherwise generation ends after exactly
//! `--max-new-tokens` tokens with "length". A prompt whose
//! `len + max_new_tokens + 1` exceeds `--max-seq` is an error (the daemon
//! would silently shrink the budget; an identity run must not).

use hipfire_arch_qwen4::admit_hfqm_artifact_on;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::ple_sidecar::{resolve_and_attach_ple_rows, PleRowsOrigin, ResolvePolicy};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message};
use hipfire_runtime::tokenizer::{maybe_normalize_prompt, Tokenizer};
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::Value;
use std::borrow::Cow;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: qwen4_greedy_ids --model MODEL.hfq --max-new-tokens N [--input|--prompts PROMPTS.jsonl|-] [--output IDS.jsonl|-] [--max-seq 8192] [--thinking on|off] [--receipt RECEIPTS.jsonl]\ninput line: {\"id\": ..., \"messages\": [{\"role\": \"user\", \"content\": \"...\"}], \"tools\": [..]?, \"category\": ..?} | {\"id\": ..., \"prompt_ids\": [..], \"resume_at\": N?}\noutput line: {\"id\": ..., \"token_ids\": [..], \"finish_reason\": \"stop\"|\"length\", \"prompt_tokens\": N}";

const DEFAULT_MAX_SEQ: usize = 8192;

/// Where the input comes from / the output goes to.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Stream {
    Std,
    File(PathBuf),
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    input: Stream,
    output: Stream,
    max_new_tokens: usize,
    max_seq: usize,
    thinking: bool,
    /// Optional JSONL of each record's S1 route receipt (`--receipt`). The
    /// ids file is unchanged, so two arms still compare byte for byte.
    receipt: Option<PathBuf>,
}

fn parse_stream(value: String) -> Stream {
    if value == "-" {
        Stream::Std
    } else {
        Stream::File(PathBuf::from(value))
    }
}

fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Args, String> {
    let mut args = args.into_iter();
    let (mut model, mut max_new_tokens) = (None, None);
    let (mut input, mut output) = (Stream::Std, Stream::Std);
    let (mut max_seq, mut thinking) = (DEFAULT_MAX_SEQ, false);
    let mut receipt = None;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--input" | "--prompts" => input = parse_stream(value()?),
            "--output" => output = parse_stream(value()?),
            "--max-new-tokens" => {
                max_new_tokens = Some(
                    value()?
                        .parse::<usize>()
                        .map_err(|e| format!("--max-new-tokens: {e}"))?,
                )
            }
            "--max-seq" => {
                max_seq = value()?
                    .parse::<usize>()
                    .map_err(|e| format!("--max-seq: {e}"))?
            }
            "--thinking" => {
                thinking = match value()?.as_str() {
                    "on" => true,
                    "off" => false,
                    other => return Err(format!("--thinking: {other:?} is not on or off")),
                }
            }
            "--receipt" => receipt = Some(PathBuf::from(value()?)),
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    let model = model.ok_or("--model is required")?;
    let max_new_tokens = max_new_tokens.ok_or("--max-new-tokens is required")?;
    if max_new_tokens == 0 {
        return Err("--max-new-tokens must be > 0".into());
    }
    if max_seq == 0 {
        return Err("--max-seq must be > 0".into());
    }
    Ok(Args {
        model,
        input,
        output,
        max_new_tokens,
        max_seq,
        thinking,
        receipt,
    })
}

enum Prompt {
    Ids(Vec<u32>),
    Messages {
        messages: Vec<Message>,
        tools: Option<Vec<Value>>,
    },
}

struct Record {
    id: Value,
    prompt: Prompt,
    /// `prompt_ids` only: prefill the first `resume_at` ids, then the rest as
    /// a second consecutive call (a resumed prefix).
    resume_at: Option<usize>,
}

/// Content normalization applied to every message (`parse_generate_messages`).
type Normalizer = for<'a> fn(&'a str) -> Cow<'a, str>;

/// One JSONL input line. `line_no` is 1-based, for errors only.
fn parse_record(line: &str, line_no: usize, normalize: Normalizer) -> Result<Record, String> {
    let at = |e: String| format!("input line {line_no}: {e}");
    let value: Value = serde_json::from_str(line).map_err(|e| at(e.to_string()))?;
    let Value::Object(mut object) = value else {
        return Err(at("not a JSON object".into()));
    };
    let id = object
        .remove("id")
        .ok_or_else(|| at("missing \"id\"".into()))?;
    let messages = object.remove("messages");
    let prompt_ids = object.remove("prompt_ids");
    let tools = object.remove("tools");
    // A caller-side label.
    object.remove("category");
    let resume_at = object.remove("resume_at");
    if let Some(key) = object.keys().next() {
        return Err(at(format!("unknown key {key:?}")));
    }
    let prompt = match (messages, prompt_ids) {
        (Some(_), Some(_)) => {
            return Err(at("both \"messages\" and \"prompt_ids\" given".into()));
        }
        (None, None) => return Err(at("needs \"messages\" or \"prompt_ids\"".into())),
        (Some(messages), None) => {
            let mut messages: Vec<Message> = serde_json::from_value(messages)
                .map_err(|e| at(format!("invalid messages: {e}")))?;
            if messages.is_empty() {
                return Err(at("\"messages\" is empty".into()));
            }
            for entry in &mut messages {
                if !entry.content.is_empty() {
                    if let Cow::Owned(normalized) = normalize(&entry.content) {
                        entry.content = normalized;
                    }
                }
            }
            let tools = match tools {
                None | Some(Value::Null) => None,
                Some(tools) => {
                    let tools: Vec<Value> = serde_json::from_value(tools)
                        .map_err(|e| at(format!("invalid tools: {e}")))?;
                    (!tools.is_empty()).then_some(tools)
                }
            };
            Prompt::Messages { messages, tools }
        }
        (None, Some(ids)) => {
            if tools.is_some() {
                return Err(at("\"tools\" needs \"messages\"".into()));
            }
            let ids: Vec<u32> =
                serde_json::from_value(ids).map_err(|e| at(format!("invalid prompt_ids: {e}")))?;
            if ids.is_empty() {
                return Err(at("\"prompt_ids\" is empty".into()));
            }
            Prompt::Ids(ids)
        }
    };
    let resume_at = match resume_at {
        None => None,
        Some(value) => {
            let Prompt::Ids(ids) = &prompt else {
                return Err(at("\"resume_at\" needs \"prompt_ids\"".into()));
            };
            let row = value
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| at("\"resume_at\" is not an unsigned integer".into()))?;
            if row == 0 || row >= ids.len() {
                return Err(at(format!(
                    "\"resume_at\" {row} is not interior to {} prompt_ids",
                    ids.len()
                )));
            }
            Some(row)
        }
    };
    Ok(Record {
        id,
        prompt,
        resume_at,
    })
}

fn read_records(input: &Stream) -> Result<Vec<Record>, String> {
    let reader: Box<dyn BufRead> = match input {
        Stream::Std => Box::new(std::io::stdin().lock()),
        Stream::File(path) => Box::new(std::io::BufReader::new(
            std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?,
        )),
    };
    let mut records = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read input line {}: {e}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(parse_record(&line, i + 1, maybe_normalize_prompt)?);
    }
    if records.is_empty() {
        return Err("input has no records".into());
    }
    Ok(records)
}

/// Whether a chat template reads the render-time date. `render_messages`
/// injects `current_date` from the wall clock onto a leading system message
/// unless `HIPFIRE_CHAT_CURRENT_DATE` pins it; a template that reads it (or
/// `strftime_now`) would make the prompt time-dependent.
fn template_reads_date(template: &str) -> bool {
    template.contains("current_date") || template.contains("strftime_now")
}

/// One output line (no trailing newline).
fn output_line(id: &Value, token_ids: &[u32], finish_reason: &str, prompt_tokens: usize) -> String {
    serde_json::json!({
        "id": id,
        "token_ids": token_ids,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
    })
    .to_string()
}

/// One receipt line: the record id and the route receipt of each prefill
/// call. A missing receipt (no verdict from the selector) is an error.
fn receipt_line(id: &Value, route: &[Option<Value>]) -> Result<String, String> {
    if route.is_empty() {
        return Err("no slowlink receipt: the prefill made no call".into());
    }
    let mut receipts = Vec::with_capacity(route.len());
    for (call, receipt) in route.iter().enumerate() {
        receipts.push(
            receipt
                .clone()
                .ok_or_else(|| format!("no slowlink receipt for prefill call {call}"))?,
        );
    }
    Ok(serde_json::json!({"id": id, "slowlink": receipts}).to_string())
}

struct Model {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    /// Final prefill row.
    prompt_logits: GpuTensor,
    /// Row every decode forward writes and the next argmax reads.
    decode_logits: GpuTensor,
    vocab: usize,
    eos_token: u32,
    max_seq: usize,
    tokenizer: Tokenizer,
    template: Option<String>,
}

impl Model {
    /// Same range-loaded path as the serve loader (`hipfire-loader` carrier),
    /// without MTP, the session cache or the PLE hot store: none of them
    /// changes the numerics of a cold AR request.
    fn load(path: &Path, max_seq: usize) -> Result<Self, String> {
        // S1 selector (`HIPFIRE_QWEN4_SLOWLINK_PREFILL` unset or `1`): the
        // shared admission-first loader runs ONLY when the same selector the
        // product uses chooses S1 for this load (forced `1`, or a later
        // eligible automatic selection); otherwise (selector `0`, unset and
        // declined, fast/unified/unmeasured link, a pinned
        // `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`) it returns before any allocation
        // and the loader below runs, placement and all, exactly as before.
        let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
        if hipfire_arch_qwen4::slowlink_admission::SlowlinkMode::from_env()
            != hipfire_arch_qwen4::slowlink_admission::SlowlinkMode::Off
        {
            if let Some(loaded) = hipfire_arch_qwen4::slowlink_load::slowlink_load_if_selected(
                &mut gpu, path, max_seq, None,
            )? {
                let prompt_logits = gpu
                    .zeros(&[loaded.vocab], DType::F32)
                    .map_err(|e| e.to_string())?;
                let decode_logits = gpu
                    .zeros(&[loaded.vocab], DType::F32)
                    .map_err(|e| e.to_string())?;
                eprintln!(
                    "qwen4_greedy_ids: loaded {} on {} via the slowlink loader (vocab {}, max_seq {max_seq}, ple rows {})",
                    path.display(),
                    gpu.arch,
                    loaded.vocab,
                    loaded.ple_format
                );
                return Ok(Self {
                    gpu,
                    bundle: loaded.bundle,
                    prompt_logits,
                    decode_logits,
                    vocab: loaded.vocab,
                    eos_token: loaded.eos_token,
                    max_seq,
                    tokenizer: loaded.tokenizer,
                    template: loaded.chat_template,
                });
            }
        }
        let mut hfq =
            HfqFile::open(path).map_err(|e| format!("open HFQM {}: {e}", path.display()))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let template = hfq.chat_template();
        // Never derives a missing q4 PLE table: derive it with
        // `hipfire ple derive` (or a serve load) first.
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            path,
            hipfire_config::memory::ple_format(),
            &gpu.arch,
            ResolvePolicy {
                derive_missing_q4: false,
                progress: None,
            },
        )
        .map_err(|e| format!("qwen4 PLE rows: {e}"))?;
        let receipt = admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
            .map_err(|e| format!("qwen4 admission: {e}"))?;
        let trunk_overlay =
            hipfire_arch_qwen4::resolve_trunk_overlay(&hfq, path, &gpu.arch, &receipt.config)
                .map_err(|e| format!("qwen4 trunk overlay: {e}"))?;
        eprintln!(
            "qwen4_greedy_ids: ple rows: {} from {}",
            choice.format.name(),
            match &choice.origin {
                PleRowsOrigin::Embedded => "embedded model file".to_string(),
                PleRowsOrigin::Sidecar(sidecar) => sidecar.display().to_string(),
            }
        );
        if let Some(note) = &choice.note {
            eprintln!("qwen4_greedy_ids: ple rows: {note}");
        }
        let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
            &gpu,
            &hfq,
            &receipt.config,
            &receipt.manifest,
        )
        .map_err(|e| e.to_string())?;
        if gpu.is_uma() {
            hfq.drop_mmap();
        }
        let mesh = DeviceMesh::single().map_err(|e| format!("qwen4 mesh: {e}"))?;
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let npu_identity = hipfire_arch_qwen4::npu_trunk::model_index_identity(&hfq, &gpu.arch);
        let source = HfqModelSource::from_hfq(hfq);
        // Resolved once, before placement: the product's automatic placement is sized
        // from them and the bundle is assembled with the same values.
        let state_format = hipfire_arch_qwen4::resolve_state_format(
            &hipfire_runtime::config::get().kv_mode,
            // Qwen3.5's DeltaNet `state_quant` knob: auto (q8) or fp32.
            &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
            &gpu,
            &receipt.config,
        )?;
        let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(&gpu);
        // `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` keeps trunk layers 0..N resident
        // and host-maps the rest (an optional diagnostic override). Unset (or `auto`)
        // resolves the product loader's automatic placement through the one shared
        // `expert_residency::incumbent_auto_resident_layers`.
        let weights = {
            use hipfire_arch_qwen4::expert_residency as residency;
            let mut weights = receipt.manifest.weights.clone();
            match residency::expert_vram_layers_from_env()? {
                Some(residency::ExpertVramLayers::Layers(layers)) => {
                    if gpu.is_uma() {
                        return Err("host-mapped experts are only for discrete GPUs".into());
                    }
                    let moved = residency::place_routed_experts(&mut weights, layers);
                    eprintln!(
                        "qwen4_greedy_ids: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                    );
                }
                None | Some(residency::ExpertVramLayers::Auto) => {
                    let recipe = &receipt.manifest.identity.recipe;
                    let slab_recipe = matches!(
                        recipe,
                        hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. }
                    );
                    // Entry sizes as the product loader sizes them: a slab layer arena from
                    // the recipe's plan, anything else from the mapped tensor.
                    let bytes_of =
                        |entry: &hipfire_runtime::weight_manifest::WeightEntry| -> Option<u64> {
                            if let hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab {
                                arenas,
                                ..
                            } = recipe
                            {
                                if entry
                                    .name
                                    .ends_with(hipfire_arch_qwen4::expert_arena::ARENA_SUFFIX)
                                {
                                    return entry
                                        .layer
                                        .and_then(|layer| arenas.layer(layer))
                                        .map(|layer| layer.bytes as u64);
                                }
                            }
                            source
                                .tensor_data(&entry.name)
                                .map(|(_, bytes)| bytes.len() as u64)
                        };
                    let vmm_granularity = match backend {
                        hipfire_arch_qwen4::Qwen4KvBackend::Vmm => gpu
                            .vmm_recommended_granularity()
                            .map_err(|e| format!("qwen4: VMM granularity: {e}"))?,
                        hipfire_arch_qwen4::Qwen4KvBackend::Legacy => 1,
                    };
                    let resolved = residency::incumbent_auto_resident_layers(
                        &gpu,
                        &receipt.config,
                        &weights,
                        &bytes_of,
                        backend,
                        max_seq,
                        vmm_granularity,
                        state_format,
                        scratch_mode,
                        residency::dedicated_slab_stage(
                            residency::expert_stage_placement_enabled(&gpu.arch),
                            slab_recipe,
                            false,
                        ),
                        0,
                    )?;
                    match resolved {
                        Some(layers) => {
                            let moved = residency::place_routed_experts(&mut weights, layers);
                            eprintln!(
                                "qwen4_greedy_ids: routed experts: product automatic placement (expert_residency::incumbent_auto_resident_layers): layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                            );
                        }
                        None => eprintln!(
                            "qwen4_greedy_ids: routed experts: product automatic placement keeps every expert resident"
                        ),
                    }
                }
            }
            weights
        };
        let transaction = fulfill_manifest_from_payloads(
            &weights,
            &mesh,
            receipt.config.num_hidden_layers,
            &mut gpu,
            expected,
            |entry| {
                if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                    &source,
                    &receipt.manifest.identity.recipe,
                    entry,
                ) {
                    return payload;
                }
                // The store rejects a host-mapped entry as a range: borrow it
                // from the resident source bytes, as the product loader does.
                if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped
                {
                    let (info, bytes) = source
                        .tensor_data(&entry.name)
                        .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                    return Ok(SourcePayload::Borrowed { info, bytes });
                }
                source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name))
            },
        )
        .map_err(|e| format!("qwen4 manifest fulfillment: {e}"))?;
        let vocab = receipt.config.vocab_size;
        let eos_token = receipt.config.eos_token_id;
        let mut bundle = Qwen4Bundle::assemble_with_ple_rows(
            receipt.config,
            &receipt.manifest.identity.recipe,
            transaction,
            &receipt.placements,
            &mut gpu,
            max_seq,
            receipt.ple,
            receipt.ple_rows,
            state_format,
            backend,
            None,
        )
        .map_err(|e| format!("qwen4 bundle assembly: {e}"))?;
        bundle.attach_npu_trunk_identity(npu_identity);
        if let Some(overlay) = trunk_overlay {
            bundle
                .attach_trunk_overlay(overlay)
                .map_err(|e| format!("qwen4 trunk overlay: {e}"))?;
        }
        bundle
            .attach_forward(&mut gpu, max_seq, scratch_mode)
            .map_err(|e| format!("qwen4 forward setup: {e}"))?;
        let prompt_logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
        let decode_logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
        eprintln!(
            "qwen4_greedy_ids: loaded {} on {} (vocab {vocab}, max_seq {max_seq})",
            path.display(),
            gpu.arch
        );
        Ok(Self {
            gpu,
            bundle,
            prompt_logits,
            decode_logits,
            vocab,
            eos_token,
            max_seq,
            tokenizer,
            template,
        })
    }

    /// The prompt ids of one record.
    fn prompt_ids(&self, prompt: &Prompt, thinking: bool) -> Result<Vec<u32>, String> {
        match prompt {
            Prompt::Ids(ids) => {
                if let Some(bad) = ids.iter().find(|&&id| id as usize >= self.vocab) {
                    return Err(format!(
                        "prompt id {bad} is outside the vocabulary ({})",
                        self.vocab
                    ));
                }
                Ok(ids.clone())
            }
            Prompt::Messages { messages, tools } => {
                let template = self
                    .template
                    .as_deref()
                    .ok_or("the artifact has no chat_template; use prompt_ids")?;
                if template_reads_date(template)
                    && hipfire_config::developer_var("HIPFIRE_CHAT_CURRENT_DATE")
                        .map_or(true, |date| date.trim().is_empty())
                {
                    return Err("the chat_template reads the current date, so the render depends on the wall clock; pin HIPFIRE_CHAT_CURRENT_DATE=YYYY-MM-DD or use prompt_ids".into());
                }
                let frame = JinjaChatFrame {
                    tokenizer: &self.tokenizer,
                    template,
                    // Ignored by `render_messages`.
                    system: None,
                    user: "",
                    enable_thinking: thinking,
                    bos_token: None,
                    reasoning_strength: None,
                    reasoning_effort: None,
                };
                let rendered = frame
                    .render_messages(messages, tools.as_deref(), None)
                    .map_err(|e| format!("qwen4 Jinja render failed: {e}"))?;
                Ok(self.tokenizer.encode(&rendered))
            }
        }
    }

    /// Greedy ids for `prompt` from a fresh state, why they ended (the first
    /// terminator, included, "stop"; or `max_new` tokens, "length"), the
    /// prefill wall seconds (`prefill_final` through the prompt-logits
    /// download, which synchronizes the device) and the S1 route receipt of
    /// each prefill call (one for a whole prompt, two for a `resume_at`
    /// prompt). `resume_at = Some(n)` prefills the first `n` ids and then
    /// the rest as two consecutive `forward_chunk_final` calls (a resumed
    /// prefix: the second call enters at position `n`).
    #[allow(clippy::type_complexity)]
    fn generate(
        &mut self,
        prompt: &[u32],
        resume_at: Option<usize>,
        max_new: usize,
    ) -> Result<(Vec<u32>, &'static str, f64, Vec<Option<Value>>), String> {
        if prompt.is_empty() {
            return Err("prompt rendered to zero tokens".into());
        }
        let required = prompt
            .len()
            .checked_add(max_new)
            .and_then(|n| n.checked_add(1));
        if required.is_none_or(|n| n > self.max_seq) {
            return Err(format!(
                "request exceeds context window: prompt={} + max_new_tokens={max_new} + trailer=1 > max_seq={}; raise --max-seq",
                prompt.len(),
                self.max_seq
            ));
        }
        let prefill_started = std::time::Instant::now();
        let mut route = Vec::new();
        match resume_at {
            None => {
                // `reused = 0` with no session cache attached resets every owner.
                self.bundle
                    .prefill_final(&mut self.gpu, prompt, 0, &self.prompt_logits)
                    .map_err(|e| format!("prefill: {e}"))?;
                route.push(self.bundle.slowlink_receipt());
            }
            Some(at) => {
                if at == 0 || at >= prompt.len() {
                    return Err(format!(
                        "resume_at {at} is not interior to {} prompt ids",
                        prompt.len()
                    ));
                }
                self.bundle
                    .reset(&mut self.gpu)
                    .map_err(|e| format!("reset: {e}"))?;
                for part in [&prompt[..at], &prompt[at..]] {
                    self.bundle
                        .forward_chunk_final(&mut self.gpu, part, &self.prompt_logits, None)
                        .map_err(|e| format!("resumed prefill: {e}"))?;
                    route.push(self.bundle.slowlink_receipt());
                }
            }
        }
        let row = self
            .gpu
            .download_f32(&self.prompt_logits)
            .map_err(|e| e.to_string())?;
        let prefill_s = prefill_started.elapsed().as_secs_f64();
        if row.iter().any(|v| !v.is_finite()) {
            return Err("non-finite prefill logits".into());
        }
        let mut pending = Some(hipfire_runtime::llama::argmax(&row));
        let mut ids = Vec::with_capacity(max_new);
        while ids.len() < max_new {
            // Runs the pending token (first step) or the argmax of the
            // previous decode row, and returns the token it ran.
            let token = self
                .bundle
                .forward_token_or_argmax(&mut self.gpu, pending.take(), &self.decode_logits)
                .map_err(|e| format!("decode: {e}"))?;
            ids.push(token);
            if token == self.eos_token || self.tokenizer.is_terminator(token) {
                return Ok((ids, "stop", prefill_s, route));
            }
        }
        Ok((ids, "length", prefill_s, route))
    }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let args = parse_args(argv).map_err(|e| format!("{e}\n{USAGE}"))?;
    // Parse everything before the expensive model load.
    let records = read_records(&args.input)?;
    let mut out: Box<dyn Write> = match &args.output {
        Stream::Std => Box::new(std::io::stdout().lock()),
        Stream::File(path) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?,
        )),
    };
    let mut model = Model::load(&args.model, args.max_seq)?;
    let mut receipts: Option<Box<dyn Write>> = match &args.receipt {
        Some(path) => Some(Box::new(std::io::BufWriter::new(
            std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?,
        ))),
        None => None,
    };
    let started = std::time::Instant::now();
    let total = records.len();
    for (i, record) in records.iter().enumerate() {
        let label = record.id.to_string();
        let prompt = model
            .prompt_ids(&record.prompt, args.thinking)
            .map_err(|e| format!("record {label}: {e}"))?;
        let (ids, finish, prefill_s, route) = model
            .generate(&prompt, record.resume_at, args.max_new_tokens)
            .map_err(|e| format!("record {label}: {e}"))?;
        writeln!(
            out,
            "{}",
            output_line(&record.id, &ids, finish, prompt.len())
        )
        .and_then(|()| out.flush())
        .map_err(|e| format!("write output: {e}"))?;
        if let Some(sink) = receipts.as_mut() {
            // A requested receipt that is missing is a failed run: the arms
            // are compared by their ids, and the route each took is proof.
            let line =
                receipt_line(&record.id, &route).map_err(|e| format!("record {label}: {e}"))?;
            writeln!(sink, "{line}")
                .and_then(|()| sink.flush())
                .map_err(|e| format!("write receipt: {e}"))?;
        }
        eprintln!(
            "qwen4_greedy_ids: {}/{total} id {label}: prompt {} tokens, generated {}  {:.0}s  prefill {:.3}s",
            i + 1,
            prompt.len(),
            ids.len(),
            started.elapsed().as_secs_f64(),
            prefill_s
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("qwen4_greedy_ids: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::prompt_frame::Role;
    use hipfire_runtime::tokenizer::normalize_prompt_with;

    #[test]
    fn date_reading_templates_are_detected() {
        assert!(template_reads_date(
            "{% if system_message.current_date %}x{% endif %}"
        ));
        assert!(template_reads_date("{{ strftime_now('%Y') }}"));
        assert!(!template_reads_date("<|im_start|>{{ message.role }}\n"));
    }

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn normalizing(s: &str) -> Cow<'_, str> {
        normalize_prompt_with(s, true)
    }

    fn record(line: &str) -> Result<Record, String> {
        parse_record(line, 7, normalizing)
    }

    #[test]
    fn args_defaults_and_required() {
        let a = parse_args(argv("--model m.hfq --max-new-tokens 64")).unwrap();
        assert_eq!(
            a,
            Args {
                model: "m.hfq".into(),
                input: Stream::Std,
                output: Stream::Std,
                max_new_tokens: 64,
                max_seq: DEFAULT_MAX_SEQ,
                thinking: false,
                receipt: None,
            }
        );
        assert!(parse_args(argv("--max-new-tokens 4")).is_err());
        assert!(parse_args(argv("--model m.hfq")).is_err());
    }

    #[test]
    fn args_receipt_is_an_optional_output_path() {
        let a = parse_args(argv(
            "--model m.hfq --max-new-tokens 8 --receipt /tmp/slowlink-receipts.jsonl",
        ))
        .unwrap();
        assert_eq!(a.receipt, Some("/tmp/slowlink-receipts.jsonl".into()));
        assert!(parse_args(argv("--model m.hfq --max-new-tokens 8 --receipt")).is_err());
    }

    #[test]
    fn record_resume_at_splits_an_id_prompt_at_an_interior_row() {
        let r = record(r#"{"id": "r", "prompt_ids": [1, 2, 3, 4, 5], "resume_at": 3}"#).unwrap();
        assert_eq!(r.resume_at, Some(3));
        assert_eq!(
            record(r#"{"id": "r", "prompt_ids": [1, 2]}"#)
                .unwrap()
                .resume_at,
            None
        );
        for bad in [
            r#"{"id": 1, "prompt_ids": [1, 2, 3], "resume_at": 0}"#,
            r#"{"id": 1, "prompt_ids": [1, 2, 3], "resume_at": 3}"#,
            r#"{"id": 1, "prompt_ids": [1, 2, 3], "resume_at": -1}"#,
            r#"{"id": 1, "prompt_ids": [1, 2, 3], "resume_at": "x"}"#,
            r#"{"id": 1, "messages": [{"role":"user","content":"x"}], "resume_at": 1}"#,
        ] {
            assert!(record(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn receipt_line_fails_closed_without_a_route_receipt() {
        let id = serde_json::json!("a");
        let receipt = serde_json::json!({"route": "layer-major"});
        let line: Value =
            serde_json::from_str(&receipt_line(&id, &[Some(receipt.clone())]).unwrap()).unwrap();
        assert_eq!(line, serde_json::json!({"id": "a", "slowlink": [receipt]}));
        assert!(receipt_line(&id, &[None])
            .unwrap_err()
            .contains("no slowlink receipt"));
        assert!(receipt_line(&id, &[]).is_err());
    }

    #[test]
    fn args_full() {
        let a = parse_args(argv(
            "--model m.hfq --input in.jsonl --output - --max-new-tokens 8 --max-seq 4096 --thinking on",
        ))
        .unwrap();
        assert_eq!(a.input, Stream::File("in.jsonl".into()));
        assert_eq!(a.output, Stream::Std);
        assert_eq!((a.max_new_tokens, a.max_seq, a.thinking), (8, 4096, true));
    }

    #[test]
    fn args_reject_bad_values() {
        for bad in [
            "--model m --max-new-tokens 0",
            "--model m --max-new-tokens -1",
            "--model m --max-new-tokens x",
            "--model m --max-new-tokens 4 --max-seq 0",
            "--model m --max-new-tokens 4 --thinking maybe",
            "--model m --max-new-tokens 4 --bogus 1",
            "--model m --max-new-tokens",
        ] {
            assert!(parse_args(argv(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn record_prompt_ids() {
        let r = record(r#"{"id": "a", "prompt_ids": [1, 2, 3]}"#).unwrap();
        assert_eq!(r.id, Value::from("a"));
        match r.prompt {
            Prompt::Ids(ids) => assert_eq!(ids, vec![1, 2, 3]),
            Prompt::Messages { .. } => panic!("expected ids"),
        }
    }

    #[test]
    fn record_id_is_kept_verbatim() {
        let r = record(r#"{"id": {"k": [1, "x"]}, "prompt_ids": [5]}"#).unwrap();
        let out: Value = serde_json::from_str(&output_line(&r.id, &[9, 8], "stop", 3)).unwrap();
        assert_eq!(
            out,
            serde_json::json!({
                "id": {"k": [1, "x"]},
                "token_ids": [9, 8],
                "finish_reason": "stop",
                "prompt_tokens": 3,
            })
        );
        let r = record(r#"{"id": 12, "prompt_ids": [5]}"#).unwrap();
        let out: Value = serde_json::from_str(&output_line(&r.id, &[], "length", 1)).unwrap();
        assert_eq!(
            out,
            serde_json::json!({"id": 12, "token_ids": [], "finish_reason": "length", "prompt_tokens": 1})
        );
    }

    #[test]
    fn record_messages_are_normalized_like_the_daemon() {
        let r = record(
            r#"{"id": 1, "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "a  \n\n\n\nb"},
                {"role": "assistant", "content": "ok"}
            ]}"#,
        )
        .unwrap();
        let Prompt::Messages { messages: m, tools } = r.prompt else {
            panic!("expected messages")
        };
        assert!(tools.is_none());
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].role, Role::System);
        assert_eq!(m[0].content, "");
        assert_eq!(m[1].role, Role::User);
        assert_eq!(m[1].content, "a\n\nb");
        assert_eq!(m[2].role, Role::Assistant);
        assert_eq!(m[2].content, "ok");
    }

    #[test]
    fn record_tools_and_category() {
        let line = |tools: &str| {
            format!(
                r#"{{"id": 1, "category": "tool", "messages": [{{"role":"user","content":"x"}}]{tools}}}"#
            )
        };
        let get = |tools: &str| match record(&line(tools)).unwrap().prompt {
            Prompt::Messages { tools, .. } => tools,
            Prompt::Ids(_) => panic!("expected messages"),
        };
        assert!(get("").is_none());
        assert!(get(r#", "tools": []"#).is_none());
        assert!(get(r#", "tools": null"#).is_none());
        let tools = get(r#", "tools": [{"type": "function", "function": {"name": "f"}}]"#).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "f");
        assert!(record(&line(r#", "tools": {"a": 1}"#)).is_err());
    }

    #[test]
    fn record_rejects_malformed_lines() {
        for (bad, needle) in [
            ("not json", "input line 7"),
            ("[1]", "not a JSON object"),
            (r#"{"prompt_ids": [1]}"#, "missing \"id\""),
            (r#"{"id": 1}"#, "needs"),
            (
                r#"{"id": 1, "messages": [{"role":"user","content":"x"}], "prompt_ids": [1]}"#,
                "both",
            ),
            (r#"{"id": 1, "prompt_ids": []}"#, "empty"),
            (r#"{"id": 1, "prompt_ids": [1, -2]}"#, "invalid prompt_ids"),
            (r#"{"id": 1, "prompt_ids": [1.5]}"#, "invalid prompt_ids"),
            (r#"{"id": 1, "messages": []}"#, "empty"),
            (
                r#"{"id": 1, "messages": [{"role":"robot","content":"x"}]}"#,
                "invalid messages",
            ),
            (r#"{"id": 1, "prompt": "hi"}"#, "unknown key"),
            (
                r#"{"id": 1, "prompt_ids": [1], "tools": []}"#,
                "needs \"messages\"",
            ),
        ] {
            let error = record(bad)
                .err()
                .unwrap_or_else(|| panic!("accepted {bad}"));
            assert!(error.contains(needle), "{bad}: {error}");
        }
    }
}
