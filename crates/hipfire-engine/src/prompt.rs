// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Prompt-frame helpers — architecture-neutral Jinja/ChatFrame rendering.
//!
//! Relocated verbatim from `crates/hipfire-daemon/src/main.rs` (wave 3).

/// Pure helper: derive Jinja `enable_thinking` and `reasoning_effort` from
/// an explicit `thinking_enabled` authority when present, otherwise from
/// legacy `raw_effort`/`max_think` inference. `thinking_enabled` is the
/// typed contract authority: `Some(false)` disables regardless of effort or
/// cap, `Some(true)` enables with `None`/`"auto"` => undefined and all other
/// exact strings passed verbatim (no lowercasing, no empty-drop). When
/// `None`, legacy direct-JSONL fallback is preserved: `none`/`off`/`chat`
/// => disabled+undefined, `max_think==1` => disabled, otherwise
/// `None`/`"auto"` => enabled+undefined else enabled+verbatim.
/// `max_think_tokens` remains an independent explicit cap (0 = uncapped) and
/// is never derived from effort.
pub fn qwen_jinja_reasoning(
    thinking_enabled: Option<bool>,
    raw_effort: Option<&str>,
    max_think_tokens: usize,
) -> (bool, Option<String>) {
    if let Some(enabled) = thinking_enabled {
        if !enabled {
            return (false, None);
        }
        return match raw_effort {
            None | Some("auto") => (true, None),
            Some(s) => (true, Some(s.to_string())),
        };
    }
    let is_disable = matches!(raw_effort, Some("none") | Some("off") | Some("chat"));
    let enable = max_think_tokens != 1 && !is_disable;
    if !enable {
        return (false, None);
    }
    match raw_effort {
        None | Some("auto") => (true, None),
        Some(s) => (true, Some(s.to_string())),
    }
}

/// A generate request's `messages`, each non-empty content normalized the
/// way the daemon normalizes `prompt` (`maybe_normalize_prompt`: strip
/// trailing whitespace before `\n`, collapse 3+ newlines to 2, ...).
/// Without it turn N's `prompt`-encoded user tokens diverge from turn N+1's
/// history tokens and break the prefix-cache LCP. The singleton dispatch and
/// the VMM batch admission share it so both render identical bytes.
/// `Err` carries the serde error text.
pub fn parse_generate_messages(
    msg: &serde_json::Value,
) -> Result<Option<Vec<hipfire_runtime::prompt_frame::Message>>, String> {
    let Some(v) = msg.get("messages") else {
        return Ok(None);
    };
    let mut messages: Vec<hipfire_runtime::prompt_frame::Message> =
        serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
    for entry in &mut messages {
        if !entry.content.is_empty() {
            let normalized = hipfire_runtime::tokenizer::maybe_normalize_prompt(&entry.content);
            if matches!(normalized, std::borrow::Cow::Owned(_)) {
                entry.content = normalized.into_owned();
            }
        }
    }
    Ok(Some(messages))
}

/// Stateless prompt rendering for a batch lane, reusing the production
/// `ChatFrame`/`JinjaChatFrame` path. Called with `seq_pos=0`, no tools/
/// messages/PFlash, retains `started_in_think` for barrier gating.
/// Plain fallback on Jinja render failure is preserved only when no explicit
/// `reasoning_effort` was supplied; explicit effort render errors are
/// surfaced as `Err` (request validation) instead of hidden by Plain.
/// `max_think_tokens` is an independent explicit cap (0 = uncapped, 1 = immediate close)
/// and is no longer tied to `enable_thinking`; explicit `thinking_enabled` may be
/// true even with cap 1 (budgeted immediate close), so no assert on cap != 1.
pub fn batch_render_prompt_tokens(
    prompt: &str,
    system: Option<&str>,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    chat_template: Option<&String>,
    max_think_tokens: usize,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) -> Result<(Vec<u32>, bool), String> {
    let jinja_enabled = hipfire_config::developer_var("HIPFIRE_JINJA_CHAT")
        .ok()
        .as_deref()
        != Some("0");
    let try_jinja = jinja_enabled && chat_template.is_some();
    let q_tokens = tokenizer.encode(prompt);
    let system_prompt = system;
    let mut started_in_think = matches!(
        assistant_prefix,
        hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
    );
    let new_tokens = if try_jinja {
        let template = chat_template.unwrap();
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort,
        };
        let render_result = if let Some(messages) = messages_history {
            frame.render_messages(messages, None, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => {
                started_in_think = crate::emit::render_tail_opens_think(&rendered);
                tokenizer.encode(&rendered)
            }
            Err(e) => {
                if reasoning_effort.is_some() {
                    return Err(e);
                }
                eprintln!("[daemon] jinja render failed ({e}) — falling back to Plain");
                hipfire_runtime::prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix,
                    raw: false,
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        hipfire_runtime::prompt_frame::ChatFrame {
            tokenizer,
            system: system_prompt,
            user: "",
            assistant_prefix,
            raw: false,
        }
        .build_with_user_tokens(&q_tokens)
    };
    Ok((new_tokens, started_in_think))
}
