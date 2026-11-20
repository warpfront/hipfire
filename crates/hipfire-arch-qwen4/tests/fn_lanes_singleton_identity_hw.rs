// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Single-card Flash-Next fn-lanes (`serve.vmm_batch` on one GPU) against the
//! singleton route, the contract of `docs/SERVE.md` ("Flash-Next exact lane
//! batching"):
//!
//! - the request that prefilled alone on the singleton and was promoted into a
//!   lane decodes byte-identically to the same request run alone;
//! - a request whose prompt runs as lane prefill tiles
//!   (`Qwen4Bundle::lane_prefill_tile`) decodes correctly but is not
//!   byte-identical (the tile's GDN state is not the singleton prefill's), so it
//!   must only run to its token budget, or stop where the singleton stops, with
//!   no blank-line run the singleton does not have. That is the class of #1180:
//!   every tile-prefilled lane emitted blank lines and stopped after 15-23 tokens.
//!
//! `#[ignore]`d: it needs one real HIP GPU (no `topology.ep` / `topology.pp`) and
//! the canonical `qwen3.8-flash-next-gptq3.mq4` named by
//! `HIPFIRE_FN_LANES_IDENTITY_MODEL`.
//!
//! The model loads through the product loader with the serve's
//! `serve.vmm_batch` lane reservation (`BatchLoadRequest`, MTP off), and the
//! AR lane store is staged as the daemon stages it (`stage_lanes_with_copy`, no
//! spec, no copy lane). The prompts are the first eight `measure` prompts of
//! `hipfire bench --backend batch`, rendered as `hipfire serve` renders one user
//! message with thinking off. Each runs alone on the singleton first (`reset`,
//! `prefill_final`, `TOKENS` x `forward_token_or_argmax`). Then two bursts run as
//! the serve runs them: the first request prefills on the singleton and is
//! promoted at its first token (`prepare_promotion` / `commit_promotion`), the
//! rest are admitted behind it and prefill as lane tiles, every lane driven in
//! lockstep (every prefilling lane's next tile, every decoding lane's AR row, one
//! executor step per tick) until it has `TOKENS` ids or commits an end token.
//! C4: prompt 0 promoted, prompts 1..=3 tiles. C8: prompt 7 promoted, prompts
//! 0..=6 tiles (other slots).
//!
//! Each test loads Flash-Next (tens of GiB of pinned host memory): run it only
//! inside a fabric hold that declares that memory and holds the Flash-Next host
//! lock, one test thread, e.g. on an R9700:
//!
//! ```text
//! fabric-run.sh --class correct --card A --mem-gb 30 --hostmem-gb 48 --hostmem-lock \
//!   --max-secs 1800 -- fn_lanes_singleton_identity_hw --ignored --test-threads=1
//! ```

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind, RowRange, SlotBatch,
};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::{Path, PathBuf};

#[path = "common/hw_identity.rs"]
mod hw_identity;

const MODEL_ENV: &str = "HIPFIRE_FN_LANES_IDENTITY_MODEL";
const MAX_SEQ: usize = 32768;
/// Lanes staged (the C8 burst; the C4 burst uses four of them).
const LANES: usize = 8;
/// Ids per request: the first pick, then the AR rows (the repro's `max_tokens`).
const TOKENS: usize = 128;
/// Four newlines in a row: the blank-line run of the #1180 garbage.
const BLANK_RUN: &str = "\n\n\n\n";

/// The first eight `measure` prompts of `crates/hipfire-cli/src/bench_concurrency_prompts.json`.
const PROMPTS: [&str; LANES] = [
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring, type hints, and a set of pytest unit tests covering edge cases. Then explain its complexity in detail.",
    "Write a Rust function that parses a CSV line into fields, handling quoted commas and escaped quotes. Include unit tests and a detailed explanation of the state machine.",
    "Implement an LRU cache class in Python with get and put in O(1), using a dict and a doubly linked list written from scratch. Include tests and explain each method.",
    "Write a C function that reverses a singly linked list in place, plus a small test harness in main() that builds, prints, reverses and frees a list. Explain the pointer manipulation step by step.",
    "Write a TypeScript debounce function with generics, a matching throttle function, and Jest tests for both. Explain the difference between them in detail.",
    "Write a Go function that computes the longest common subsequence of two strings using dynamic programming, reconstructs the subsequence, and includes table-driven tests. Explain the recurrence.",
    "Write a SQL schema for employees and departments, insert sample data, then write a query that returns the top 3 salaries per department using window functions, and explain every clause.",
    "Implement binary search in Python, prove its loop invariant, then add lower_bound and upper_bound variants with tests for duplicates and empty lists.",
];

/// The bursts: (promoted prompt, tile-prefilled prompts in admission order).
const BURSTS: [(&str, usize, &[usize]); 2] =
    [("C4", 0, &[1, 2, 3]), ("C8", 7, &[0, 1, 2, 3, 4, 5, 6])];

/// The serve's request reservation of a single-card fn-lanes load: `LANES` AR
/// lanes at `LANES` rows, MTP off.
fn load(model: &Path, gpu: &mut Gpu) -> Result<hipfire_loader::LoadedModel, String> {
    use hipfire_loader::admission::{
        admit_source_with_options, legacy_warning, qwen4_kv_adaptive_requested, BatchLoadRequest,
        KvBackendHints, KvBackendRequest, SourceAdmissionOptions,
    };
    let path = model
        .to_str()
        .ok_or_else(|| format!("{} is not valid UTF-8", model.display()))?;
    let (state_request, _) = hw_identity::gdn_test_state_quant()?;
    let spec = hipfire_runtime::loader_api::SpecLoadCfg {
        mtp: Some(false),
        ..Default::default()
    };
    let mut admission = admit_source_with_options(
        path,
        1,
        1,
        KvBackendRequest::Automatic,
        None,
        gpu.arch.as_str(),
        None,
        "auto",
        None,
        MAX_SEQ,
        KvBackendHints {
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            qwen_default_q8: true,
            kv_adaptive: None,
            deepseek4_heterogeneous: false,
            vmm_runtime_refusal: gpu.vmm_runtime_refusal().as_deref(),
            free_vram_bytes: gpu.device_mem_info().ok().map(|(free, _)| free),
            unified_memory: gpu.is_uma(),
        },
        SourceAdmissionOptions {
            spec,
            kv_adaptive: qwen4_kv_adaptive_requested(None),
            ..Default::default()
        },
    )?;
    admission.batch_request = BatchLoadRequest {
        vmm_batch: true,
        continuous_batch_size: LANES,
        row_budget: LANES,
        spec_k: None,
        spec_copy: false,
    };
    let warning = legacy_warning(admission.kv_backend, admission.kv_backend_reason.as_deref());
    let loaded = hipfire_loader::load_admitted_with_gemma4_drafter(
        admission,
        path,
        MAX_SEQ,
        None,
        hipfire_config::Deepseek4ComputePlacement::Single,
        None,
        None,
        hipfire_loader::GEMMA4_EAGLE_DRAFT_LEN,
        None,
        None,
        None,
        None,
        Some(state_request),
        1,
        spec,
        gpu,
    )?;
    if let Some(warning) = warning {
        eprintln!("{warning}");
    }
    Ok(loaded)
}

/// `text` as `hipfire serve` renders one user message with thinking off.
fn render(tokenizer: &Tokenizer, template: &str, text: &'static str) -> Result<Vec<u32>, String> {
    let frame = JinjaChatFrame {
        tokenizer,
        template,
        system: None,
        user: text,
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let user = Message {
        role: Role::User,
        content: text.to_string(),
        reasoning_content: None,
        name: None,
        rendered_name: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_plan: String::new(),
    };
    Ok(tokenizer.encode(&frame.render_messages(&[user], None, None)?))
}

/// The greedy pick of the F32 logits row `logits` (the lowest index of the maximum).
fn host_argmax(gpu: &mut Gpu, logits: &GpuTensor) -> Result<u32, String> {
    let mut bytes = vec![0u8; logits.byte_size()];
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    gpu.hip
        .memcpy_dtoh(&mut bytes, &logits.buf)
        .map_err(|e| e.to_string())?;
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(chunk.try_into().expect("4 bytes"));
        if value > best.1 {
            best = (i, value);
        }
    }
    Ok(best.0 as u32)
}

/// The singleton run of `prompt`: its `TOKENS` greedy ids.
fn singleton(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    prompt: &[u32],
) -> Result<Vec<u32>, String> {
    bundle.reset(gpu).map_err(|e| e.to_string())?;
    bundle
        .prefill_final(gpu, prompt, 0, logits)
        .map_err(|e| e.to_string())?;
    let ids = (0..TOKENS)
        .map(|_| {
            bundle
                .forward_token_or_argmax(gpu, None, logits)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    bundle.reset(gpu).map_err(|e| e.to_string())?;
    Ok(ids)
}

/// One request's rows of a step.
struct RowSpec {
    slot: usize,
    epoch: RequestEpoch,
    tokens: Vec<u32>,
    start: usize,
    kind: RequestStepKind,
}

/// One step's plan: per-slot rows for every slot (idle slots empty), requests
/// in slot order, absolute positions.
fn build_plan(mut rows: Vec<RowSpec>) -> BatchStepPlan {
    rows.sort_by_key(|r| r.slot);
    let mut per: Vec<(SlotId, Vec<u32>, usize)> =
        (0..LANES).map(|s| (SlotId(s), Vec::new(), 0)).collect();
    for r in &rows {
        per[r.slot] = (SlotId(r.slot), r.tokens.clone(), r.start);
    }
    let triples: Vec<(SlotId, &[u32], usize)> =
        per.iter().map(|(s, t, p)| (*s, t.as_slice(), *p)).collect();
    let mut plan = BatchStepPlan {
        batch: SlotBatch::build(&triples),
        ..Default::default()
    };
    let mut begin = 0;
    for r in &rows {
        plan.requests.push(RequestRows {
            epoch: r.epoch,
            rows: RowRange {
                begin,
                len: r.tokens.len(),
            },
            kind: r.kind,
        });
        match r.kind {
            RequestStepKind::Prefill => plan.prefill_rows += r.tokens.len(),
            _ => plan.decode_rows += r.tokens.len(),
        }
        begin += r.tokens.len();
    }
    plan
}

/// provision -> forward -> commit of one plan; a failure after provisioning
/// aborts the step.
fn exec_step(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    plan: &BatchStepPlan,
) -> Result<Vec<RequestAdvance>, String> {
    let mut ex = bundle.lane_executor().map_err(|e| e.to_string())?;
    ex.provision_step(gpu, plan)
        .map_err(|m| format!("provision_step failed: {m}"))?;
    let out = match ex.forward_step(gpu, plan) {
        Ok(out) => out,
        Err(m) => {
            ex.abort_step(plan);
            return Err(format!("forward_step failed: {m}"));
        }
    };
    match ex.commit_step(gpu, plan, out) {
        Ok(advances) => Ok(advances),
        Err(m) => {
            ex.abort_step(plan);
            Err(format!("commit_step failed: {m}"))
        }
    }
}

struct LaneReq {
    prompt: usize,
    epoch: RequestEpoch,
    slot: usize,
    prefilled: usize,
    ids: Vec<u32>,
    done: bool,
}

fn next_epoch(tag: &mut u64) -> RequestEpoch {
    *tag += 1;
    RequestEpoch {
        request_tag: *tag,
        owner_generation: 1,
    }
}

/// One burst as the serve runs it: `promoted` prefills on the singleton and is
/// promoted into a lane at its first pick, then `tiles` are admitted behind it;
/// every lane runs in lockstep until it has `TOKENS` ids or commits one of
/// `stops`. Returns each request's ids, indexed by prompt.
fn burst(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    prompts: &[Vec<u32>],
    (promoted, tiles): (usize, &[usize]),
    stops: &[u32],
    tag: &mut u64,
) -> Result<Vec<Option<Vec<u32>>>, String> {
    // The singleton prefill and its first pick, then the promotion at
    // generated = 0 (`promote_qwen4_singleton` with the seed not yet emitted).
    let prompt = &prompts[promoted];
    bundle.reset(gpu).map_err(|e| e.to_string())?;
    bundle
        .prefill_final(gpu, prompt, 0, logits)
        .map_err(|e| e.to_string())?;
    let seed = host_argmax(gpu, logits)?;
    let epoch = next_epoch(tag);
    let slot = bundle
        .prepare_promotion(
            gpu,
            epoch,
            prompt.clone(),
            vec![seed],
            seed,
            prompt.len(),
            None,
        )
        .map_err(|e| e.to_string())?;
    bundle
        .commit_promotion(gpu, &epoch)
        .map_err(|e| e.to_string())?;
    let mut reqs = vec![LaneReq {
        prompt: promoted,
        epoch,
        slot,
        prefilled: prompt.len(),
        ids: vec![seed],
        done: stops.contains(&seed),
    }];
    for &prompt in tiles {
        let epoch = next_epoch(tag);
        let slot = bundle
            .admit_lane(gpu, epoch, prompts[prompt].clone())
            .map_err(|e| e.to_string())?;
        reqs.push(LaneReq {
            prompt,
            epoch,
            slot,
            prefilled: 0,
            ids: Vec::new(),
            done: false,
        });
    }
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
    for _tick in 0..longest + TOKENS + 8 {
        let mut specs = Vec::new();
        for req in reqs.iter().filter(|r| !r.done) {
            let prompt = &prompts[req.prompt];
            if req.prefilled < prompt.len() {
                let len = bundle
                    .lane_executor()
                    .map_err(|e| e.to_string())?
                    .exact_prefill_chunk_len(&req.epoch)?;
                if len == 0 || req.prefilled + len > prompt.len() {
                    return Err(format!(
                        "prompt {}: prefill tile of {len} rows at {}/{}",
                        req.prompt,
                        req.prefilled,
                        prompt.len()
                    ));
                }
                specs.push(RowSpec {
                    slot: req.slot,
                    epoch: req.epoch,
                    tokens: prompt[req.prefilled..req.prefilled + len].to_vec(),
                    start: req.prefilled,
                    kind: RequestStepKind::Prefill,
                });
            } else {
                let seed = *req.ids.last().ok_or("a decoding lane without a seed")?;
                specs.push(RowSpec {
                    slot: req.slot,
                    epoch: req.epoch,
                    tokens: vec![seed],
                    start: prompt.len() + req.ids.len() - 1,
                    kind: RequestStepKind::Ar,
                });
            }
        }
        if specs.is_empty() {
            let mut out = vec![None; prompts.len()];
            for req in reqs {
                bundle.release_lane(&req.epoch).map_err(|e| e.to_string())?;
                out[req.prompt] = Some(req.ids);
            }
            return Ok(out);
        }
        let plan = build_plan(specs);
        let advances = exec_step(bundle, gpu, &plan)?;
        if advances.len() != plan.requests.len() {
            return Err(format!(
                "{} advances for {} plan requests",
                advances.len(),
                plan.requests.len()
            ));
        }
        for (row, advance) in plan.requests.iter().zip(&advances) {
            let req = reqs
                .iter_mut()
                .find(|r| r.epoch == row.epoch && advance.epoch == row.epoch)
                .ok_or("an advance names no request of this burst")?;
            if row.kind == RequestStepKind::Prefill {
                req.prefilled += row.rows.len;
                if req.prefilled < prompts[req.prompt].len() {
                    if !advance.committed_ids.is_empty() {
                        return Err("a non-final prefill tile committed ids".into());
                    }
                    continue;
                }
            }
            let [id] = advance.committed_ids.as_slice() else {
                return Err(format!(
                    "prompt {}: a {:?} row committed {:?}, expected one id",
                    req.prompt, row.kind, advance.committed_ids
                ));
            };
            req.ids.push(*id);
            req.done = req.ids.len() == TOKENS || stops.contains(id);
        }
    }
    Err("lanes still running after the tick cap".into())
}

/// Why a tile-prefilled lane's `ids` break the contract against the singleton's
/// `reference`, if they do: it stopped where the singleton did not (truncation),
/// or its text has a blank-line run the singleton's does not.
fn tile_lane_fault(
    tokenizer: &Tokenizer,
    stops: &[u32],
    reference: &[u32],
    ids: &[u32],
) -> Option<String> {
    let reference_stops = reference.iter().any(|id| stops.contains(id));
    if ids.len() < TOKENS && !reference_stops {
        return Some(format!(
            "stopped after {} of {TOKENS} tokens; the singleton runs the full budget",
            ids.len()
        ));
    }
    let text = tokenizer.decode(ids);
    if text.contains(BLANK_RUN) && !tokenizer.decode(reference).contains(BLANK_RUN) {
        return Some(format!("blank-line run in {text:?}"));
    }
    None
}

#[test]
#[ignore = "needs one HIP GPU and HIPFIRE_FN_LANES_IDENTITY_MODEL (Flash-Next), inside a fabric hold"]
fn single_card_lanes_keep_the_singleton_contract_on_flash_next() {
    assert!(
        std::env::var_os("FABRIC_JOB").is_some(),
        "refusing to load Flash-Next outside a fabric hold (see the module docs)"
    );
    let model = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name qwen3.8-flash-next-gptq3.mq4"));
    hipfire_loader::prepare_host_memory_for(&model);
    let mut gpu = Gpu::init().expect("GPU");
    let mut loaded = load(&model, &mut gpu).expect("load");
    let tokenizer = loaded.tokenizer.take().expect("the load has no tokenizer");
    let template = loaded
        .chat_template
        .take()
        .expect("the artifact has no chat template");
    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|text| render(&tokenizer, &template, text).expect("render prompt"))
        .collect();
    let im_end = tokenizer.encode("<|im_end|>");
    let bundle = loaded.qwen4_mut().expect("the load is not a Qwen4 bundle");
    assert!(
        bundle.ep_mesh().is_none(),
        "the load is an EP mesh: this test is the single-card route"
    );
    let mut stops = vec![bundle.config.eos_token_id];
    if let [id] = im_end.as_slice() {
        stops.push(*id);
    }
    bundle
        .stage_lanes_with_copy(&mut gpu, LANES, LANES, None, None)
        .expect("stage lanes");
    let logits = gpu
        .zeros(&[bundle.config.vocab_size], DType::F32)
        .expect("logits");
    let reference: Vec<Vec<u32>> = prompts
        .iter()
        .map(|prompt| singleton(bundle, &mut gpu, &logits, prompt).expect("singleton run"))
        .collect();
    let mut tag = 0;
    let mut failures = Vec::new();
    for (name, promoted, tiles) in BURSTS {
        let run = burst(
            bundle,
            &mut gpu,
            &logits,
            &prompts,
            (promoted, tiles),
            &stops,
            &mut tag,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        // The promoted lane is the singleton's request: byte-identical up to
        // the singleton's own end.
        let got = run[promoted].as_deref().expect("the promoted lane ran");
        let want = &reference[promoted][..got.len()];
        let promoted_ok =
            got == want && (got.len() == TOKENS || got.last().is_some_and(|id| stops.contains(id)));
        println!(
            "{name} promoted prompt {promoted}: {} ids, {}",
            got.len(),
            if promoted_ok {
                "== singleton"
            } else {
                "DIFFERS"
            }
        );
        if !promoted_ok {
            failures.push(format!(
                "{name} promoted prompt {promoted}: lane {got:?} != singleton {:?}",
                reference[promoted]
            ));
        }
        for &prompt in tiles {
            let ids = run[prompt].as_deref().expect("the tile lane ran");
            let fault = tile_lane_fault(&tokenizer, &stops, &reference[prompt], ids);
            println!(
                "{name} tile prompt {prompt}: {} ids, {}, {}",
                ids.len(),
                if ids == reference[prompt].as_slice() {
                    "== singleton"
                } else {
                    "differs from the singleton (allowed)"
                },
                fault
                    .as_deref()
                    .unwrap_or("runs to its budget, no blank-line run")
            );
            if let Some(fault) = fault {
                failures.push(format!("{name} tile prompt {prompt}: {fault}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "lanes broke the single-card contract:\n{}",
        failures.join("\n")
    );
}
