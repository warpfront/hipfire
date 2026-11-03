// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3 (arch_id 24) generation bodies.
//!
//! Lifted from the pre-merge `hipfire-runtime/examples/daemon.rs` (lfm3-port)
//! into `hipfire-generate`, matching the other per-arch bodies.

use std::io::Write;
use std::time::Instant;

use hipfire_arch_deepseek4::sampling;
use hipfire_engine::emit::{emit_error_with_id, emit_gen_start, render_tail_opens_think, write_error};
use hipfire_engine::terminal::{
    active_attempt_id, await_client_terminal_commit, emit_staged_terminal_done,
    ClientTerminalDecision,
};
use hipfire_loader::LoadedModel;
use std::path::Path;
use crate::vision::{GenerateVLParams, ImageSource};
use hipfire_arch_qwen35_vl::image;
use hipfire_arch_lfm3_vl::lfm3_vl;
use hipfire_arch_lfm3_audio::{lfm3_audio, mel as lfm3_mel, wav as lfm3_wav};

/// Emit an active-attempt error (thin alias to the dense-route helper).
fn emit_active_attempt_error(
    stdout: &mut impl Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    crate::dense::emit_active_attempt_error(stdout, id, message, class, retryable, rolled_back);
}

pub fn generate_lfm3(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    max_think_tokens: usize,
    // Engine-authoritative reasoning toggle (from `qwen_jinja_reasoning`).
    // LFM3's template keys its think opener on this; without it LFM3 always
    // thought, so `--thinking off` was a no-op and short caps were burned
    // inside the chain-of-thought.
    enable_thinking: bool,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) {
    // THINK-1: derive started_in_think from the frame actually rendered, not
    // from `max_think_tokens != 1`. That flag is correct for the Jinja path
    // (it keys the template's think opener) but wrong for the Plain fallback
    // (HIPFIRE_JINJA_CHAT=0 or a render failure): Plain emits no think
    // opener, so flagging started_in_think=true there mislabels the whole
    // answer as reasoning_content. Same derivation as the lfm2moe/qwen routes
    // (render_tail_opens_think on the rendered template; false for Plain).
    // The prompt build emits no events, so it can run before the contract
    // opener (which must stay the first event on the stream).
    let prompt_build: Option<(Vec<u32>, bool)> = m.tokenizer.as_ref().map(|tokenizer| {
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
                let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                    match messages_history {
                        Some(h) => h,
                        None => {
                            let mut v = Vec::new();
                            if let Some(sys) = system_prompt {
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::System,
                                    content: sys.to_string(),
                                    reasoning_content: None,
                                    name: None,
                                    rendered_name: None,
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                    tool_plan: String::new(),
                                });
                            }
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::User,
                                content: prompt.to_string(),
                                reasoning_content: None,
                                name: None,
                                rendered_name: None,
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
                            });
                            synthesized = v;
                            &synthesized
                        }
                    };
                frame.render_messages(messages_slice, tools, None)
            } else {
                frame.render()
            };
            match render_result {
                Ok(rendered) => {
                    let opened = render_tail_opens_think(&rendered);
                    (tokenizer.encode(&rendered), opened)
                }
                Err(e) => {
                    eprintln!(
                        "[daemon] jinja render failed in lfm3 path ({e}) — falling back to Plain"
                    );
                    (
                        hipfire_runtime::prompt_frame::ChatFrame {
                            tokenizer,
                            system: system_prompt,
                            user: prompt,
                            assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                            raw: false,
                        }
                        .build(),
                        false,
                    )
                }
            }
        } else {
            (
                hipfire_runtime::prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: prompt,
                    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                    raw: false,
                }
                .build(),
                false,
            )
        }
    });
    let (prompt_ids, started_in_think) = prompt_build.unwrap_or((Vec::new(), false));
    // Contract opener: the CLI serve's StreamContractGate rejects every
    // pre-gen_start event (PreStartEvent → client abort → drain deadlock), so
    // this path MUST emit gen_start before any token/error event. Matches the
    // main generate() opener (line ~15614); arch 13 latches the legacy
    // contract (gen_start_contract_version_for_arch returns None for 13).
    emit_gen_start(
        stdout,
        id,
        started_in_think,
        crate::common::gen_start_contract_version_for_arch(m.arch_id),
    );
    if m.tokenizer.is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "tokenizer not loaded",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    if m.lfm3().is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "lfm3 bundle missing on arch_id=13 generate",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // ── Prompt build (hoisted above the contract opener — see THINK-1) ──
    // `prompt_ids` and `started_in_think` were computed before gen_start; the
    // tokenizer / bundle validation below still gates every event path.

    if prompt_ids.is_empty() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "empty prompt after tokenize",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let eos_tok = m.lfm3().unwrap().eos_tok;

    // LFM3 (like LFM2.5) emits multiple EOS-class tokens inconsistently
    // (`<|im_end|>`, ``, rarely `</s>`). Catch any whose literal string
    // round-trips through `encode` to a single id; the rest are caught by the
    // string-level guard in the decode loop below.
    let stop_toks: Vec<u32> = {
        let tk = m.tokenizer.as_ref().unwrap();
        let mut v = vec![eos_tok];
        for s in ["", "</s>", "<|im_end|>"] {
            let ids = tk.encode(s);
            if ids.len() == 1 && !v.contains(&ids[0]) {
                v.push(ids[0]);
            }
        }
        v
    };

    // Cross-conversation reset: LFM3's hybrid conv+GQA state can't be cheaply
    // rewound to an arbitrary prefix (the conv window chains back to token 0),
    // so cold-rebuild every turn from position 0 — same as LFM2.5.
    let _ = m.lfm3_mut().unwrap().state.reset(gpu);
    m.seq_pos = 0;
    m.conversation_tokens.clear();

    let cap = m.lfm3().unwrap().state.max_seq;
    if prompt_ids.len().saturating_add(max_tokens) > cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!("prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq", prompt_ids.len(), max_tokens, cap),
            "context_length",
            false,
            false
        );
        let _ = stdout.flush();
        return;
    }

    let t0 = Instant::now();

    // ── Prefill: decode_step per prompt token. The LAST decode_step's logits
    // are the predictions for the first generated token. ──
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let b = m.lfm3_mut().unwrap();
        let cfg = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut position = state.n_tokens as u32;
        for &tok in &prompt_ids {
            match hipfire_arch_lfm3::forward::decode_step(cfg, weights, state, gpu, tok, position) {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("lfm3 prefill failed: {e:?}"));
                    return;
                }
            }
            position += 1;
        }
    }
    for &tok in &prompt_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // ── Decode loop. Sample host-side from the running logits vector. ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = sampling::Xorshift::new(seed);

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    loop {
        if generated_count >= max_tokens {
            break;
        }
        let next_tok = sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        if stop_toks.contains(&next_tok) {
            break;
        }

        let frag = {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            tokenizer.decode(&[next_tok])
        };
        // String-level EOS-class guard: exact EOS-class strings only. A
        // whitespace-only fragment (" ", "\n") is LEGITIMATE model output —
        // LFM3 thinking emits standalone space/newline tokens constantly —
        // so this must NOT trim() before matching: trim-to-empty killed
        // generation on the first whitespace token (surfaced through serve as
        // truncated/"zero content" answers). Exact "" is kept: an EOS variant
        // that decodes to the empty string doesn't id-round-trip into
        // stop_toks, so the string form is the only catch.
        if matches!(frag.as_str(), "" | "</s>" | "<|im_end|>") {
            break;
        }
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
            "attempt_id": active_attempt_id(),
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        let step = {
            let b = m.lfm3_mut().unwrap();
            let cfg = &b.config;
            let weights = &b.weights;
            let state = &mut b.state;
            let position = state.n_tokens as u32;
            hipfire_arch_lfm3::forward::decode_step(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("lfm3 decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.lfm3().unwrap().state.n_tokens;

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    // The loop's only breaks are the max_tokens cap, stop_toks, and the
    // string-level EOS guard (all "stop"), so reaching the cap is the sole
    // non-stop cause. The CLI defaults a missing finish_reason to "stop",
    // which mislabeled token-cap truncation as a natural stop.
    let finish_reason = if generated_count >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "finish_reason": finish_reason,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => {
            let _ = emit_staged_terminal_done(stdout, &pending_done);
        }
        // Client cancelled (or the handshake failed closed): emit the
        // correlated cancellation transcript so the CLI serve's
        // abort_and_drain terminates instead of waiting forever while holding
        // the serve admission slot (queue-wide stall).
        ClientTerminalDecision::Abort => crate::ar::emit_active_route_cancel(stdout, id, generated_count),
    }
}

pub fn generate_lfm3_vl(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
    use base64::Engine as _;
    use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame};

    let GenerateVLParams {
        id,
        prompt,
        system_prompt,
        image_source: ref image_source,
        temp,
        top_p,
        max_tokens,
        max_think_tokens: _,
        repeat_penalty: _,
        repeat_window: _,
        ..
    } = *params;

    // Reasoning toggle. The engine lowers "thinking off" to a closed-think
    // assistant prefix; every other prefix (open_think / plain) keeps LFM3's
    // default-on reasoning. `max_think_tokens` stays an independent cap.
    let enable_thinking = !matches!(params.assistant_prefix, AssistantPrefix::ClosedThink);

    // Contract opener (see generate_lfm3): gen_start must precede every other
    // event or the CLI serve's StreamContractGate rejects the stream. The VL
    // Jinja frame below renders with enable_thinking: true, so the stream
    // starts inside <think>.
    emit_gen_start(
        stdout,
        id,
        enable_thinking,
        crate::common::gen_start_contract_version_for_arch(m.arch_id),
    );
    if m.tokenizer.is_none() || m.lfm3().is_none() || m.lfm3_vision_weights().is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "lfm3-vl: tokenizer / lfm3 bundle / vision weights missing",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let vision_config = m.lfm3_vision_config().unwrap();
    let image_pad_id = vision_config.image_token_id;
    let vision_start_id = vision_config.vision_start_token_id;
    let vision_end_id = vision_config.vision_end_token_id;
    let sms = vision_config.spatial_merge_size;

    // ── Image preprocessing (CPU decode + smart resize + patch extract) ──
    // VL-RESIZE-1: LFM3-VL has its own pixel budget (HF `Lfm3VlImageProcessor`
    // size {shortest_edge: 65_536, longest_edge: 16_777_216}); the shared
    // preprocessor's Qwen defaults silently changed the patch grid for
    // out-of-overlap images.
    let (pixels, img_h, img_w) = match image_source {
        ImageSource::Path(path) => {
            eprintln!("[lfm3-vl] preprocessing image: path: {}", path);
            match image::load_and_preprocess_with_pixels(
                Path::new(path),
                vision_config.patch_size,
                vision_config.spatial_merge_size,
                image::LFM3_VL_MIN_PIXELS,
                image::LFM3_VL_MAX_PIXELS,
            ) {
                Ok(r) => r,
                Err(e) => {
                    write_error(stdout, id, &e);
                    return;
                }
            }
        }
        ImageSource::Base64(b64) => {
            let raw_b64 = if let Some(rest) = b64.strip_prefix("data:") {
                match rest.split_once(',') {
                    Some((_, after)) => after,
                    None => {
                        write_error(stdout, id, "malformed data URL: missing ',' separator");
                        return;
                    }
                }
            } else {
                b64
            };
            let bytes = match base64::engine::general_purpose::STANDARD.decode(raw_b64) {
                Ok(b) => b,
                Err(e) => {
                    write_error(
                        stdout,
                        id,
                        &format!("failed to decode base64 image data: {e}"),
                    );
                    return;
                }
            };
            match image::load_and_preprocess_from_bytes_with_pixels(
                &bytes,
                vision_config.patch_size,
                vision_config.spatial_merge_size,
                image::LFM3_VL_MIN_PIXELS,
                image::LFM3_VL_MAX_PIXELS,
            ) {
                Ok(r) => r,
                Err(e) => {
                    write_error(stdout, id, &e);
                    return;
                }
            }
        }
    };
    let grid_h = img_h / vision_config.patch_size;
    let grid_w = img_w / vision_config.patch_size;
    let n_visual_tokens = grid_h * grid_w / (sms * sms);

    let patches = image::extract_patches(
        &pixels,
        3,
        img_h,
        img_w,
        vision_config.patch_size,
        vision_config.temporal_patch_size,
        vision_config.spatial_merge_size,
    );

    // ── Vision tower + projector → projected visual tokens ──
    let visual_tokens = {
        let vw = m.lfm3_vision_weights().unwrap();
        match lfm3_vl::vision_forward(gpu, vw, vision_config, &patches, grid_h, grid_w) {
            Ok(v) => v,
            Err(e) => {
                write_error(
                    stdout,
                    id,
                    &format!("lfm3-vl: vision_forward failed: {e:?}"),
                );
                return;
            }
        }
    };
    if visual_tokens.len() != n_visual_tokens * m.lfm3().unwrap().config.hidden_size {
        write_error(
            stdout,
            id,
            &format!(
                "lfm3-vl: visual token count mismatch: got {} expected {}x{}",
                visual_tokens.len(),
                n_visual_tokens,
                m.lfm3().unwrap().config.hidden_size
            ),
        );
        return;
    }

    // ── Prompt build: render the LFM3 chat template (exact framing — system /
    // user / assistant `<|im_start|>assistant\n<think></think>`), then splice the vision
    // tokens [vision_start, image_pad×N, vision_end, \n] before the user
    // content. This matches generate_lfm3's Jinja path (which produces long
    // coherent thinking); the ChatFrame scaffold + OpenThink approximation
    // mismatched LFM3's `<think></think>` (no trailing \n) and truncated early.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let nl = tokenizer.encode("\n");
    let q_tokens = tokenizer.encode(prompt);

    let mut vision_seq: Vec<u32> = Vec::with_capacity(n_visual_tokens + 4);
    vision_seq.push(vision_start_id);
    for _ in 0..n_visual_tokens {
        vision_seq.push(image_pad_id);
    }
    vision_seq.push(vision_end_id);
    vision_seq.extend_from_slice(&nl);

    let prompt_tokens: Vec<u32> = if let Some(template) = m.chat_template.as_ref() {
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort: None,
        };
        match frame.render() {
            Ok(rendered) => {
                let mut toks = tokenizer.encode(&rendered);
                // Insert the vision sequence immediately before the user
                // content (the prompt tokens) in the rendered stream.
                if !q_tokens.is_empty() {
                    if let Some(pos) = toks.windows(q_tokens.len()).position(|w| w == q_tokens) {
                        toks.splice(pos..pos, vision_seq.clone());
                    } else {
                        eprintln!(
                            "[lfm3-vl] warning: could not locate user content for vision splice"
                        );
                    }
                }
                toks
            }
            Err(e) => {
                eprintln!(
                    "[daemon] jinja render failed in lfm3-vl path ({e}) — falling back to ChatFrame"
                );
                let mut body = vision_seq.clone();
                body.extend_from_slice(&q_tokens);
                ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix: AssistantPrefix::OpenThink,
                    raw: false,
                }
                .build_with_user_tokens(&body)
            }
        }
    } else {
        let mut body = vision_seq.clone();
        body.extend_from_slice(&q_tokens);
        ChatFrame {
            tokenizer,
            system: system_prompt,
            user: "",
            assistant_prefix: AssistantPrefix::OpenThink,
            raw: false,
        }
        .build_with_user_tokens(&body)
    };

    if prompt_tokens.is_empty() {
        write_error(stdout, id, "lfm3-vl: empty prompt after tokenize");
        return;
    }
    // SPLICE-1: prefill embeds vision rows only at image_pad positions, so a
    // prompt that lost its splice (anchor not found in the rendered template)
    // would prefill with zero vision embeddings and "succeed" ungrounded.
    // Hard-error instead of continuing past the warning above.
    let n_pads = prompt_tokens.iter().filter(|&&t| t == image_pad_id).count();
    if n_pads != n_visual_tokens {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "lfm3-vl: vision splice failed — prompt carries {n_pads} image_pad placeholders, expected {n_visual_tokens} (user content not found in rendered template)"
            ),
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let eos_tok = m.lfm3().unwrap().eos_tok;
    let stop_toks: Vec<u32> = {
        let tk = m.tokenizer.as_ref().unwrap();
        let mut v = vec![eos_tok];
        for s in ["", "</s>", "<|im_end|>"] {
            let ids = tk.encode(s);
            if ids.len() == 1 && !v.contains(&ids[0]) {
                v.push(ids[0]);
            }
        }
        v
    };

    // Cross-conversation reset (LFM3 hybrid conv+GQA state can't rewind).
    let _ = m.lfm3_mut().unwrap().state.reset(gpu);
    m.seq_pos = 0;
    m.conversation_tokens.clear();

    let cap = m.lfm3().unwrap().state.max_seq;
    if prompt_tokens.len().saturating_add(max_tokens) > cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "lfm3-vl: prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={}",
                prompt_tokens.len(),
                max_tokens,
                cap
            ),
            "context_length",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let text_hidden = m.lfm3().unwrap().config.hidden_size;
    let t0 = Instant::now();

    // ── Prefill: per-token decode_step, splicing visual embeddings at
    //    image_pad positions via decode_step_with_embed. ──
    let mut last_logits: Vec<f32> = Vec::new();
    let mut visual_idx = 0usize;
    {
        let b = m.lfm3_mut().unwrap();
        let cfg = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut position = state.n_tokens as u32;
        for &tok in &prompt_tokens {
            let res = if tok == image_pad_id && visual_idx < n_visual_tokens {
                let emb = &visual_tokens[visual_idx * text_hidden..(visual_idx + 1) * text_hidden];
                visual_idx += 1;
                hipfire_arch_lfm3::forward::decode_step_with_embed(cfg, weights, state, gpu, emb, position)
            } else {
                hipfire_arch_lfm3::forward::decode_step(cfg, weights, state, gpu, tok, position)
            };
            match res {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("lfm3-vl prefill failed: {e:?}"));
                    return;
                }
            }
            position += 1;
        }
    }
    for &tok in &prompt_tokens {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // ── Decode loop (host-side sampling, same as generate_lfm3). ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = sampling::Xorshift::new(seed);
    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    loop {
        if generated_count >= max_tokens {
            break;
        }
        let next_tok = sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        if stop_toks.contains(&next_tok) {
            break;
        }
        let frag = {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            tokenizer.decode(&[next_tok])
        };
        // Only break on a real EOS / turn-close token. The bare `""` frag case
        // (whitespace / the LFM3 think-close marker) is intentionally NOT a stop
        // here — the model thinks (<think>…</think>) then emits the visible
        // answer, and breaking on whitespace would truncate mid-think before the
        // answer. stop_toks (eos / <|im_end|>) handles the real turn end.
        if matches!(frag.trim(), "</s>" | "<|im_end|>") {
            break;
        }
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
            "attempt_id": active_attempt_id(),
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        let step = {
            let b = m.lfm3_mut().unwrap();
            let cfg = &b.config;
            let weights = &b.weights;
            let state = &mut b.state;
            let position = state.n_tokens as u32;
            hipfire_arch_lfm3::forward::decode_step(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("lfm3-vl decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.lfm3().unwrap().state.n_tokens;
    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    // The loop's only breaks are the max_tokens cap, stop_toks, and the
    // string-level EOS guard (all "stop"), so reaching the cap is the sole
    // non-stop cause. The CLI defaults a missing finish_reason to "stop",
    // which mislabeled token-cap truncation as a natural stop.
    let finish_reason = if generated_count >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "finish_reason": finish_reason,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => { let _ = emit_staged_terminal_done(stdout, &pending_done); }
        ClientTerminalDecision::Abort => crate::ar::emit_active_route_cancel(stdout, id, generated_count),
    }
}

/// Splice position for an EMPTY text prompt on the LFM3-Audio ASR path
/// (Phase 5b). The audio tokens belong inside the user turn — where the HF
/// processor would have expanded the `<|audio|>` placeholder as the user
/// content — so anchor on the LAST `<|im_start|>user\n` header in the rendered
/// template and return the position right after it (start of user content,
/// matching the with-text path's audio-before-text order). Fallback: the
/// position right before the assistant generation header (a template that
/// dropped the empty user turn entirely). None when neither anchor exists.
pub fn lfm3_audio_empty_prompt_splice(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    toks: &[u32],
) -> Option<usize> {
    let user_hdr = tokenizer.encode("<|im_start|>user\n");
    if !user_hdr.is_empty() {
        if let Some(pos) = toks.windows(user_hdr.len()).rposition(|w| w == user_hdr) {
            return Some(pos + user_hdr.len());
        }
    }
    let asst_hdr = tokenizer.encode("<|im_start|>assistant");
    if !asst_hdr.is_empty() {
        if let Some(pos) = toks.windows(asst_hdr.len()).rposition(|w| w == asst_hdr) {
            return Some(pos);
        }
    }
    None
}

/// LFM3-Audio ASR path (Phase 3). Mirrors `generate_lfm3_vl`'s shape (render
/// chat template → splice placeholder tokens → prefill with
/// `decode_step_with_embed` at placeholder positions → host-side decode loop),
/// but the multimodal input is audio: decode WAV → log-mel → conv2d downsampler
/// → 17-layer LFM2 encoder → projector → `[n_audio, text_hidden]` audio tokens,
/// spliced at `<|audio|>` (id 125031) positions. One `<|audio|>` per encoder
/// token (one per 8 mel frames), matching the HF processor's
/// `replace_audio_token` expansion.
pub fn generate_lfm3_audio(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    // Reasoning toggle for ASR. LFM3-Audio supports thinking, but the ASR
    // transcript must land in `content` (not `reasoning_content`), so the
    // dispatch keeps the historical default OFF and only an explicit
    // open-think request turns it on.
    enable_thinking: bool,
    audio_bytes: &[u8],
) {
    use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame};

    // Contract opener (see generate_lfm3): gen_start must precede every other
    // event or the CLI serve's StreamContractGate rejects the stream. The ASR
    // prompt uses a CLOSED think span (see the framing note at the prompt
    // build — Phase 5b), so the stream starts in answer mode.
    emit_gen_start(
        stdout,
        id,
        enable_thinking,
        crate::common::gen_start_contract_version_for_arch(m.arch_id),
    );
    if m.tokenizer.is_none() || m.lfm3().is_none() || m.lfm3_audio_weights().is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "lfm3-audio: tokenizer / lfm3 bundle / audio weights missing",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let audio_config = m.lfm3_audio_config().unwrap();
    let sr = audio_config.mel.sample_rate;

    // ── Audio decode (WAV → mono F32 @ 16 kHz) ──
    // Any input rate is accepted: a mismatching WAV is linearly resampled to
    // the model's rate (the HF `load_audio` backend resamples too; 24 kHz
    // browser/TTS audio no longer has to be pre-converted).
    let waveform = match lfm3_wav::decode_wav_any(audio_bytes) {
        Ok(w) if w.sample_rate == sr => w,
        Ok(w) => {
            let n = ((w.samples.len() as u64) * (sr as u64) / (w.sample_rate as u64)) as usize;
            let mut out = vec![0.0f32; n.max(1)];
            let ratio = w.sample_rate as f64 / sr as f64;
            for (i, o) in out.iter_mut().enumerate() {
                let src = i as f64 * ratio;
                let i0 = src as usize;
                let i1 = (i0 + 1).min(w.samples.len() - 1);
                let f = (src - i0 as f64) as f32;
                *o = w.samples[i0] * (1.0 - f) + w.samples[i1] * f;
            }
            eprintln!(
                "[lfm3-audio] resampled {} samples @ {}Hz → {} samples @ {}Hz (linear)",
                w.samples.len(),
                w.sample_rate,
                out.len(),
                sr
            );
            lfm3_wav::Waveform { samples: out, sample_rate: sr }
        }
        Err(e) => {
            write_error(stdout, id, &format!("lfm3-audio: WAV decode failed: {e}"));
            return;
        }
    };
    eprintln!(
        "[lfm3-audio] decoded {} samples ({:.1}s @ {}Hz)",
        waveform.samples.len(),
        waveform.samples.len() as f64 / sr as f64,
        sr
    );

    // ── Mel + encoder + projector → projected audio tokens ──
    let mel = lfm3_mel::log_mel_from_waveform(&waveform.samples, &audio_config.mel);
    let num_frames = mel.len() / audio_config.mel.n_mels;
    // AUDIO-DBG: FNV-1a hash of a stage's bytes — run-to-run identical requests
    // must produce identical hashes; a mismatch pins the nondeterministic stage.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
    let dbg = std::env::var("HIPFIRE_AUDIO_DEBUG").ok().as_deref() == Some("1");
    if dbg {
        let mel_bytes: Vec<u8> = mel.iter().flat_map(|f| f.to_le_bytes()).collect();
        eprintln!(
            "[lfm3-audio][dbg] mel hash {:016x} ({} f32)",
            fnv1a(&mel_bytes),
            mel.len()
        );
    }
    let audio_tokens = {
        let aw = m.lfm3_audio_weights().unwrap();
        match lfm3_audio::audio_forward(gpu, aw, audio_config, &mel, num_frames) {
            Ok(v) => v,
            Err(e) => {
                write_error(
                    stdout,
                    id,
                    &format!("lfm3-audio: audio_forward failed: {e:?}"),
                );
                return;
            }
        }
    };
    let text_hidden = m.lfm3().unwrap().config.hidden_size;
    let n_audio_tokens = audio_tokens.len() / text_hidden;
    if dbg {
        let tok_bytes: Vec<u8> = audio_tokens.iter().flat_map(|f| f.to_le_bytes()).collect();
        eprintln!(
            "[lfm3-audio][dbg] audio_tokens hash {:016x} ({} tok x {} dims)",
            fnv1a(&tok_bytes),
            n_audio_tokens,
            text_hidden
        );
    }
    eprintln!(
        "[lfm3-audio] encoder produced {n_audio_tokens} audio tokens x {text_hidden} dims (from {num_frames} mel frames)"
    );

    // ── Resolve the <|audio|> placeholder token id (default 125031). ──
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let audio_pad_id = {
        let ids = tokenizer.encode("<|audio|>");
        eprintln!(
            "[lfm3-audio] <|audio|> encodes to {:?} (using {})",
            ids,
            if ids.len() == 1 {
                ids[0].to_string()
            } else {
                "125031 (fallback)".to_string()
            }
        );
        if ids.len() == 1 {
            ids[0]
        } else {
            125_031
        }
    };

    // Audio-SPAN fidelity (Phase 5d fix): the HF processor expands an audio
    // placeholder to `<|vision_start|><|audio|>×N<|vision_end|>` (processor
    // config audio_markers=true — the same span tokens the Omni SFT data
    // renders). The bare audio_pad splice (Phase 3) produced coherent
    // acoustic descriptions on LFM3-Audio-2B but degenerate transcripts on
    // LFM3-Omni-2B — the Omni SFT expects the span wrapper. Resolve the ids
    // through the tokenizer like <|audio|> above (processor config pins
    // <|vision_start|>/<|vision_end|>); the VL route (validated Phase 2)
    // wraps its image pads the same way.
    let vision_start_id = {
        let ids = tokenizer.encode("<|vision_start|>");
        if ids.len() == 1 {
            ids[0]
        } else {
            125_009
        }
    };
    let vision_end_id = {
        let ids = tokenizer.encode("<|vision_end|>");
        if ids.len() == 1 {
            ids[0]
        } else {
            125_010
        }
    };

    // ── Prompt build: render the LFM3 chat template, then splice n_audio
    //    <|audio|> tokens before the user content (mirrors the VL vision
    //    splice; the HF processor expands <|audio|> once per encoder token).
    //
    //    THINK FRAMING (Phase 5b): the assistant prefix closes the think span
    //    (`<think></think>`, thinking OFF). With an OPEN `<think>` (Phase 3's
    //    OpenThink) the model answered INSIDE the span — the CLI's
    //    ThinkChannelRouter routed the whole answer to reasoning_content and
    //    delta.content stayed empty, so Liquid Agent's transcribe() (content
    //    only) got "". ASR wants the transcript spoken as the answer. ──
    let q_tokens = tokenizer.encode(prompt);
    let mut audio_seq: Vec<u32> = Vec::with_capacity(n_audio_tokens + 2);
    audio_seq.push(vision_start_id);
    audio_seq.extend(std::iter::repeat(audio_pad_id).take(n_audio_tokens));
    audio_seq.push(vision_end_id);

    let prompt_tokens: Vec<u32> = if let Some(template) = m.chat_template.as_ref() {
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort: None,
        };
        match frame.render() {
            Ok(rendered) => {
                let mut toks = tokenizer.encode(&rendered);
                // Splice position: when the request carries text, immediately
                // before the user-content tokens (audio first, then text — the
                // HF processor's expansion order). When the prompt is EMPTY
                // (Liquid Agent transcribe(): an audio-only content part, no
                // text part) there are no user tokens to search for, so anchor
                // on the user-turn boundary of the rendered template instead
                // (Phase 5b — previously the splice was SKIPPED and the
                // SPLICE-1 guard below hard-errored: "no audio tokens in
                // context").
                let splice_pos = if !q_tokens.is_empty() {
                    toks.windows(q_tokens.len()).position(|w| w == q_tokens)
                } else {
                    lfm3_audio_empty_prompt_splice(tokenizer, &toks)
                };
                match splice_pos {
                    Some(pos) => {
                        toks.splice(pos..pos, audio_seq.clone());
                    }
                    None => {
                        eprintln!(
                            "[lfm3-audio] warning: could not locate user content for audio splice"
                        );
                    }
                }
                toks
            }
            Err(e) => {
                eprintln!("[daemon] jinja render failed in lfm3-audio path ({e}) — falling back to ChatFrame");
                let mut body = audio_seq.clone();
                body.extend_from_slice(&q_tokens);
                ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix: AssistantPrefix::ClosedThink,
                    raw: false,
                }
                .build_with_user_tokens(&body)
            }
        }
    } else {
        let mut body = audio_seq.clone();
        body.extend_from_slice(&q_tokens);
        ChatFrame {
            tokenizer,
            system: system_prompt,
            user: "",
            assistant_prefix: AssistantPrefix::ClosedThink,
            raw: false,
        }
        .build_with_user_tokens(&body)
    };

    if prompt_tokens.is_empty() {
        write_error(stdout, id, "lfm3-audio: empty prompt after tokenize");
        return;
    }
    // SPLICE-1: prefill embeds audio rows only at audio_pad positions, so a
    // prompt that lost its splice would prefill with zero audio embeddings
    // and "succeed" ungrounded. Hard-error instead of logging and continuing.
    let n_pad_in_prompt = prompt_tokens.iter().filter(|&&t| t == audio_pad_id).count();
    if n_pad_in_prompt != n_audio_tokens {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "lfm3-audio: audio splice failed — prompt carries {n_pad_in_prompt} audio_pad placeholders, expected {n_audio_tokens} (user content / user-turn anchor not found in rendered template)"
            ),
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    eprintln!(
        "[lfm3-audio] prompt_tokens={} (audio_pad {n_pad_in_prompt} expected {n_audio_tokens}, audio_pad_id={audio_pad_id})",
        prompt_tokens.len()
    );

    let eos_tok = m.lfm3().unwrap().eos_tok;
    let stop_toks: Vec<u32> = {
        let tk = m.tokenizer.as_ref().unwrap();
        let mut v = vec![eos_tok];
        for s in ["", "</s>", "<|im_end|>"] {
            let ids = tk.encode(s);
            if ids.len() == 1 && !v.contains(&ids[0]) {
                v.push(ids[0]);
            }
        }
        v
    };

    // Cross-conversation reset (LFM3 hybrid conv+GQA state can't rewind).
    let _ = m.lfm3_mut().unwrap().state.reset(gpu);
    m.seq_pos = 0;
    m.conversation_tokens.clear();

    let cap = m.lfm3().unwrap().state.max_seq;
    if prompt_tokens.len().saturating_add(max_tokens) > cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "lfm3-audio: prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={}",
                prompt_tokens.len(),
                max_tokens,
                cap
            ),
            "context_length",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let t0 = Instant::now();

    // ── Prefill: per-token decode_step, splicing audio embeddings at
    //    audio_pad positions via decode_step_with_embed. ──
    let mut last_logits: Vec<f32> = Vec::new();
    let mut audio_idx = 0usize;
    {
        let b = m.lfm3_mut().unwrap();
        let cfg = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut position = state.n_tokens as u32;
        for &tok in &prompt_tokens {
            let res = if tok == audio_pad_id && audio_idx < n_audio_tokens {
                let emb = &audio_tokens[audio_idx * text_hidden..(audio_idx + 1) * text_hidden];
                audio_idx += 1;
                hipfire_arch_lfm3::forward::decode_step_with_embed(cfg, weights, state, gpu, emb, position)
            } else {
                hipfire_arch_lfm3::forward::decode_step(cfg, weights, state, gpu, tok, position)
            };
            match res {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("lfm3-audio prefill failed: {e:?}"));
                    return;
                }
            }
            position += 1;
        }
    }
    for &tok in &prompt_tokens {
        m.conversation_tokens.push(tok);
    }
    eprintln!("[lfm3-audio] prefill spliced {audio_idx}/{n_audio_tokens} audio embeddings");
    let prefill_ms = t0.elapsed().as_millis();

    // ── Decode loop (host-side sampling, same as generate_lfm3_vl). ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = sampling::Xorshift::new(seed);
    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    loop {
        if generated_count >= max_tokens {
            break;
        }
        let next_tok = sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        if stop_toks.contains(&next_tok) {
            break;
        }
        let frag = {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            tokenizer.decode(&[next_tok])
        };
        if matches!(frag.trim(), "</s>" | "<|im_end|>") {
            break;
        }
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
            "attempt_id": active_attempt_id(),
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        let step = {
            let b = m.lfm3_mut().unwrap();
            let cfg = &b.config;
            let weights = &b.weights;
            let state = &mut b.state;
            let position = state.n_tokens as u32;
            hipfire_arch_lfm3::forward::decode_step(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("lfm3-audio decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.lfm3().unwrap().state.n_tokens;
    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    // The loop's only breaks are the max_tokens cap, stop_toks, and the
    // string-level EOS guard (all "stop"), so reaching the cap is the sole
    // non-stop cause. The CLI defaults a missing finish_reason to "stop",
    // which mislabeled token-cap truncation as a natural stop.
    let finish_reason = if generated_count >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "finish_reason": finish_reason,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => { let _ = emit_staged_terminal_done(stdout, &pending_done); }
        ClientTerminalDecision::Abort => crate::ar::emit_active_route_cancel(stdout, id, generated_count),
    }
}
/// LFM3-Omni TTS (audio output) path — Phase 4c. Generates audio OUTPUT: a
/// ChatML prompt + `<|audio_start|>` → `omni_generate_audio` (LM decode loop +
/// depth decoder MaskGIT + codebook_embedding feedback) → `codec_decode` →
/// 24 kHz WAV → base64 `audio` event.
///
/// Triggered by `"tts": true` on an LFM3-Omni model (depth decoder present).
/// The codec loads from `HIPFIRE_OMNI_CODEC_DIR` (the `audio_tokenizer/` dir).
/// AUDIO mode only here; TEXT/INTERLEAVED + the eos refinement are open.
/// Omni TTS seed: HIPFIRE_OMNI_SEED for reproducible runs (batch-vs-streaming
/// equivalence testing), else time-based.
pub fn omni_seed() -> u64 {
    std::env::var("HIPFIRE_OMNI_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        })
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn generate_lfm3_omni(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    system_voice: Option<&str>,
    _temp: f32,
    _top_p: f32,
    max_tokens: usize,
    mode: &str,
    stream: bool,
    max_steps_override: Option<usize>,
) {
    use hipfire_arch_lfm3_omni as omni;
    use hipfire_runtime::safetensors_source::SafetensorsSource;

    // Contract opener (see generate_lfm3): gen_start must precede every other
    // event or the CLI serve's StreamContractGate rejects the stream. The
    // omni prompt ends with the thinking-END marker (<think></think>) before
    // the mode marker, so the stream starts in answer mode.
    emit_gen_start(
        stdout,
        id,
        false,
        crate::common::gen_start_contract_version_for_arch(m.arch_id),
    );
    if m.tokenizer.is_none() || m.lfm3().is_none() || m.lfm3_depth_weights().is_none() {
        write_error(
            stdout,
            id,
            "lfm3-omni: tokenizer / lfm3 bundle / depth decoder missing",
        );
        return;
    }
    let codec_dir = match std::env::var("HIPFIRE_OMNI_CODEC_DIR") {
        Ok(d) => d,
        Err(_) => {
            write_error(
                stdout,
                id,
                "lfm3-omni: HIPFIRE_OMNI_CODEC_DIR not set (point at the audio_tokenizer/ dir)",
            );
            return;
        }
    };

    // Cross-conversation reset (same as generate_lfm3_audio): the omni loop
    // always builds its own ChatML prompt from position 0, and the LFM3 hybrid
    // conv+GQA state can't rewind — so a previous request's conv/KV state on
    // this serve process must not leak into the TTS prefill. Phase 5 runs
    // chat + /tts turns against ONE daemon, so this reset is load-bearing.
    // (The early check above guarantees the Lfm3 bundle is present.)
    if let Some(b) = m.lfm3_mut() {
        let _ = b.state.reset(gpu);
    }
    m.seq_pos = 0;
    m.conversation_tokens.clear();

    // Build the prompt: BOS + ChatML + assistant prefix + mode marker.
    // "audio" → <|audio_start|> (verbatim TTS); "interleaved" → <|mixed_start|>
    // (text+audio blocks, a conversational reply).
    //
    // VERBATIM-TTS (AUDIO mode): the /tts contract is "speak THIS text" — the
    // input must be SPOKEN, not answered. The old framing (mode=interleaved,
    // no system prompt) asked the model to CONVERSE about the text: it
    // generated its own reply text first (the daemon logged 100+ text tokens
    // for a one-line input) and then spoke what IT had written — a paraphrase.
    // The reference contract (checkpoint README "Speech synthesis") is a
    // "Perform TTS in <language> using a <gender> voice." system turn, the
    // text as the USER message, and generation_mode="audio" — the assistant
    // frame opens with <|audio_start|> and speaks the user's text verbatim,
    // ending at the audio eos (which fires in this framing; the Phase-4c
    // "pure AUDIO never terminates" observation was made WITHOUT the TTS
    // system prompt — the model had no text-terminus context then).
    let marker = if mode == "interleaved" {
        omni::MIXED_START_TOKEN
    } else {
        omni::AUDIO_START_TOKEN
    };
    let tok = m.tokenizer.as_ref().unwrap();
    let chatml = if mode == "interleaved" {
        match system_prompt.filter(|s| !s.trim().is_empty()) {
            Some(sys) => format!(
                "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>"
            ),
            None => format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>"),
        }
    } else {
        let voice = system_voice.unwrap_or("female");
        let tts_system = system_prompt
            .filter(|s| !s.trim().is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("Perform TTS in American English using a {voice} voice."));
        format!(
            "<|im_start|>system\n{tts_system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>"
        )
    };
    let mut prompt_tokens = tok.encode(&chatml);
    prompt_tokens.insert(0, 124_894); // BOS <|startoftext|>
    prompt_tokens.push(marker);
    eprintln!(
        "[lfm3-omni] tts prompt (mode={mode}): {} tokens, marker={marker}",
        prompt_tokens.len()
    );

    // max_tokens bounds the generation steps (1 LM step per text token or audio
    // frame). Default clamp 200 (≈13 s of speech); the request `max_steps`
    // field (Phase 5a `/tts`) raises it for long speech. Both are guarded
    // against model capacity right after the prompt is built.
    const OMNI_MAX_STEPS_CAP: usize = 4096;
    let max_steps = max_steps_override
        .unwrap_or_else(|| max_tokens.max(1))
        .clamp(1, OMNI_MAX_STEPS_CAP);

    // Capacity guard (the STREAM-1 class, at the LM level): every step writes
    // one KV row, so prompt + max_steps must fit state.max_seq or the loop
    // would run past the KV allocation. Clean error instead of silent OOB.
    {
        let cap = m.lfm3().unwrap().state.max_seq;
        if prompt_tokens.len().saturating_add(max_steps) > cap {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!(
                    "lfm3-omni: prompt + step budget exceeds context capacity: prompt={} + max_steps={} > capacity={} (lower max_steps or raise max_seq)",
                    prompt_tokens.len(),
                    max_steps,
                    cap
                ),
                "context_length",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    }
    let depth_cfg = m.lfm3_depth_config().unwrap().clone();

    // ── Streaming TTS path: emit audio chunks as frames are generated ──
    if stream {
        let t0 = std::time::Instant::now();
        let seed = omni_seed();
        // Prefill once.
        let mut stream_state = {
            let bundle = match m.state.as_mut() {
                Some(s) => (s.as_mut() as &mut dyn std::any::Any).downcast_mut::<hipfire_loader::Lfm3Bundle>().expect("not lfm3"),
                _ => {
                    write_error(stdout, id, "lfm3-omni: not an lfm3 model");
                    return;
                }
            };
            match omni::omni_stream_prefill(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                &prompt_tokens,
            ) {
                Ok(s) => s,
                Err(e) => {
                    write_error(stdout, id, &format!("lfm3-omni: stream prefill: {e}"));
                    return;
                }
            }
        };
        // Load the codec + create the streaming cache.
        let source = match SafetensorsSource::open(std::path::Path::new(&codec_dir)) {
            Ok(s) => s,
            Err(e) => {
                write_error(stdout, id, &format!("lfm3-omni: open codec dir: {e}"));
                return;
            }
        };
        let cfg_val: serde_json::Value =
            match std::fs::read_to_string(format!("{codec_dir}/config.json"))
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
            {
                Some(v) => v,
                None => {
                    write_error(stdout, id, "lfm3-omni: read codec config.json failed");
                    return;
                }
            };
        let codec_cfg = omni::CodecConfig::from_config_value(&cfg_val);
        let codec_weights = match omni::load_codec_weights(&source, &codec_cfg, gpu) {
            Ok(w) => w,
            Err(e) => {
                write_error(stdout, id, &format!("lfm3-omni: load codec: {e}"));
                return;
            }
        };
        let mut codec_cache = match omni::CodecStreamingCache::new(gpu, &codec_cfg, max_steps) {
            Ok(c) => c,
            Err(e) => {
                write_error(stdout, id, &format!("lfm3-omni: codec cache: {e}"));
                return;
            }
        };
        let mut rng = omni::DepthRng::new(seed);
        let chunk_steps = 12usize; // generate ~12 frames/steps per chunk
        let mut total_frames = 0usize;
        let mut n_chunks = 0usize;
        // STREAM-1: total-step budget. `finished` only flips on audio-eos
        // (AUDIO) or <|im_end|> (INTERLEAVED) and pure AUDIO may never reach
        // eos, so without a cap this loop ran unbounded: KV/history grew past
        // max_seq (silent GPU OOB) and the request never emitted a terminal
        // done. The budget is the same max_steps the CodecStreamingCache was
        // sized for, so total frames can never exceed the cache's capacity —
        // the previous mismatch (12-frame chunks vs a ≤200-frame cache) is
        // closed by construction. The last chunk is CLAMPED to the remainder
        // (`min(chunk_steps, steps_left)`): subtracting a full chunk while
        // still generating a full chunk overshot the cache by up to
        // chunk_steps-1 frames (max_steps=200, chunk=12 → 204 frames = 1224
        // decoder tokens vs a 1200-token cache → out-of-range KV write, the
        // exact class this budget was added to close).
        let mut steps_left = max_steps;
        let mut hit_step_budget = false;
        while !stream_state.finished {
            if steps_left == 0 {
                hit_step_budget = true;
                break;
            }
            let this_chunk = chunk_steps.min(steps_left);
            steps_left -= this_chunk;
            let step_out = {
                let bundle = match m.state.as_mut() {
                    Some(s) => (s.as_mut() as &mut dyn std::any::Any).downcast_mut::<hipfire_loader::Lfm3Bundle>().expect("not lfm3"),
                    _ => {
                        // LEAK-1: free the codec GPU buffers on the error path.
                        codec_cache.free_gpu(gpu);
                        codec_weights.free_gpu(gpu);
                        write_error(stdout, id, "lfm3-omni: not an lfm3 model");
                        return;
                    }
                };
                match omni::omni_stream_step(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    &depth_cfg,
                    bundle.depth_weights.as_ref().unwrap(),
                    gpu,
                    &mut stream_state,
                    this_chunk,
                    &mut rng,
                ) {
                    Ok(o) => o,
                    Err(e) => {
                        // LEAK-1: free the codec GPU buffers on the error path.
                        codec_cache.free_gpu(gpu);
                        codec_weights.free_gpu(gpu);
                        write_error(stdout, id, &format!("lfm3-omni: stream step: {e}"));
                        return;
                    }
                }
            };
            // Emit any text tokens.
            if !step_out.text_tokens.is_empty() {
                let tok = m.tokenizer.as_ref().unwrap();
                let text_ids: Vec<u32> = step_out
                    .text_tokens
                    .iter()
                    .filter(|&&t| t != omni::AUDIO_OUT_TOKEN)
                    .copied()
                    .collect();
                let text = tok.decode(&text_ids);
                if !text.is_empty() {
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({"type":"token","id":id,"text":text,"attempt_id":active_attempt_id()})
                    );
                    let _ = stdout.flush();
                }
            }
            // Decode + emit the audio chunk.
            if !step_out.audio_codes.is_empty() {
                let finalize = step_out.finished;
                let audio_chunk = match omni::codec_decode_streaming(
                    &codec_cfg,
                    &codec_weights,
                    gpu,
                    &step_out.audio_codes,
                    &mut codec_cache,
                    finalize,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        // LEAK-1: free the codec GPU buffers on the error path.
                        codec_cache.free_gpu(gpu);
                        codec_weights.free_gpu(gpu);
                        write_error(stdout, id, &format!("lfm3-omni: stream decode: {e}"));
                        return;
                    }
                };
                if !audio_chunk.is_empty() {
                    let wav = omni::wav_bytes(&audio_chunk, 24_000);
                    use base64::Engine as _;
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({"type":"audio_chunk","id":id,"audio":b64,"sample_rate":24000,"frames":step_out.n_audio_frames,"final":finalize,"attempt_id":active_attempt_id()})
                    );
                    let _ = stdout.flush();
                    n_chunks += 1;
                }
                total_frames += step_out.n_audio_frames;
            }
        }
        // Flush the codec's held-back tail if not already finalized.
        // LEAK-1 was also here: a failed tail flush used to be swallowed
        // silently via `if let Ok` — surface it (the stream still terminates,
        // but the operator sees why audio may be truncated).
        if !codec_cache.finished {
            match omni::codec_decode_streaming(
                &codec_cfg,
                &codec_weights,
                gpu,
                &[],
                &mut codec_cache,
                true,
            ) {
                Ok(tail) => {
                    if !tail.is_empty() {
                        let wav = omni::wav_bytes(&tail, 24_000);
                        use base64::Engine as _;
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::json!({"type":"audio_chunk","id":id,"audio":b64,"sample_rate":24000,"frames":0,"final":true,"attempt_id":active_attempt_id()})
                        );
                        let _ = stdout.flush();
                        n_chunks += 1;
                    }
                }
                Err(e) => eprintln!("[lfm3-omni] warning: codec tail flush failed: {e}"),
            }
        }
        codec_cache.free_gpu(gpu);
        codec_weights.free_gpu(gpu);
        let gen_ms = t0.elapsed().as_millis();
        eprintln!(
            "[lfm3-omni] streamed {total_frames} audio frames in {n_chunks} chunks in {gen_ms}ms (finish_reason={})",
            if hit_step_budget { "length" } else { "stop" }
        );
        // FINISH-1: distinguish budget-capped streams (length) from a natural
        // eos/im_end finish (stop) — same contract as the text routes.
        let finish_reason = if hit_step_budget { "length" } else { "stop" };
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type":"done","id":id,"tokens":total_frames,"total_ms":gen_ms,"streamed":true,"finish_reason":finish_reason,"attempt_id":active_attempt_id()})
        );
        let _ = stdout.flush();
        return;
    }

    // Omni generate loop (AUDIO or INTERLEAVED, inferred from the prompt marker).
    let t0 = std::time::Instant::now();
    let out = {
        let bundle = match m.state.as_mut() {
            Some(s) => (s.as_mut() as &mut dyn std::any::Any).downcast_mut::<hipfire_loader::Lfm3Bundle>().expect("not lfm3"),
            _ => {
                write_error(stdout, id, "lfm3-omni: not an lfm3 model");
                return;
            }
        };
        let seed = omni_seed();
        let mut rng = omni::DepthRng::new(seed);
        match omni::omni_generate(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            &depth_cfg,
            bundle.depth_weights.as_ref().unwrap(),
            gpu,
            &prompt_tokens,
            max_steps,
            &mut rng,
        ) {
            Ok(v) => v,
            Err(e) => {
                write_error(stdout, id, &format!("lfm3-omni: generate failed: {e}"));
                return;
            }
        }
    };
    let gen_ms = t0.elapsed().as_millis();
    // FINISH-1 (omni batch): same contract as the streaming path and the text
    // routes — a natural stop (audio-eos / <|im_end|> / post-eos text_end)
    // reports "stop"; exhausting max_steps reports "length". `finished` is
    // exactly the natural-stop flag in both omni generate loops.
    let finish_reason = if out.finished { "stop" } else { "length" };
    let n_emitted = out.n_audio_frames;
    eprintln!(
        "[lfm3-omni] generated {} text tokens + {n_emitted} audio frames (eos={}, finished={}) in {gen_ms}ms",
        out.text_tokens.len(),
        out.hit_eos,
        out.finished,
    );

    // Decode the text tokens (skip audio_out placeholders) for the response.
    let text_out: String = if out.text_tokens.is_empty() {
        String::new()
    } else {
        let tok = m.tokenizer.as_ref().unwrap();
        let text_ids: Vec<u32> = out
            .text_tokens
            .iter()
            .filter(|&&t| t != omni::AUDIO_OUT_TOKEN)
            .copied()
            .collect();
        tok.decode(&text_ids)
    };

    if out.audio_codes.is_empty() {
        // Text-only result (e.g., interleaved produced no audio before finishing).
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type":"token","id":id,"text":text_out,"attempt_id":active_attempt_id()})
        );
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type":"done","id":id,"tokens":out.text_tokens.len(),"total_ms":gen_ms,"finish_reason":finish_reason,"attempt_id":active_attempt_id()})
        );
        let _ = stdout.flush();
        return;
    }

    // Load the codec + decode the audio frames.
    let source = match SafetensorsSource::open(std::path::Path::new(&codec_dir)) {
        Ok(s) => s,
        Err(e) => {
            write_error(stdout, id, &format!("lfm3-omni: open codec dir: {e}"));
            return;
        }
    };
    let cfg_val: serde_json::Value =
        match std::fs::read_to_string(format!("{codec_dir}/config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
        {
            Some(v) => v,
            None => {
                write_error(stdout, id, "lfm3-omni: read codec config.json failed");
                return;
            }
        };
    let codec_cfg = omni::CodecConfig::from_config_value(&cfg_val);
    let codec_weights = match omni::load_codec_weights(&source, &codec_cfg, gpu) {
        Ok(w) => w,
        Err(e) => {
            write_error(stdout, id, &format!("lfm3-omni: load codec: {e}"));
            return;
        }
    };
    let (waveform, sr) = match omni::codec_decode(&codec_cfg, &codec_weights, gpu, &out.audio_codes)
    {
        Ok(v) => v,
        Err(e) => {
            // LEAK-1: free the codec weights on the error path.
            codec_weights.free_gpu(gpu);
            write_error(stdout, id, &format!("lfm3-omni: codec decode: {e}"));
            return;
        }
    };
    codec_weights.free_gpu(gpu);

    // Emit the text (if any) + audio (base64 WAV) + done.
    if !text_out.is_empty() {
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type":"token","id":id,"text":text_out,"attempt_id":active_attempt_id()})
        );
    }
    let wav = omni::wav_bytes(&waveform, sr);
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
    let _ = writeln!(
        stdout,
        "{}",
        serde_json::json!({
            "type": "audio",
            "id": id,
            "audio": b64,
            "sample_rate": sr,
            "frames": n_emitted,
            "duration_s": (waveform.len() as f64 / sr as f64 * 100.0).round() / 100.0,
            "eos": out.hit_eos,
            "attempt_id": active_attempt_id(),
        })
    );
    let _ = writeln!(
        stdout,
        "{}",
        serde_json::json!({"type":"done","id":id,"tokens":out.text_tokens.len(),"total_ms":gen_ms,"finish_reason":finish_reason,"attempt_id":active_attempt_id()})
    );
    let _ = stdout.flush();
}

