// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Static explainer text for Settings keys (5e). A plain, GPU-independent data
//! table: selecting a key in the Settings tab shows what it is, what it does,
//! its tradeoff, the default, when to use it, and any load-bearing interaction
//! (e.g. thinking↔dflash, kv_cache:auto = inherit). Authored as a const table so
//! it is testable without a GPU or a daemon and can be cross-checked against the
//! config defaults.

use hipfire_config::{
    developer_env_for_key,
    effective::{quality_choices, ArchCtx, Choice, Effective, Origin},
    evidence::{evidence_for, format_closer, format_slower, two_sig},
    field,
};

use crate::hipfire::writer::FieldKind;

// ---------------------------------------------------------------------------
// Curated Settings rows (spec §3.1, §3.7)
// ---------------------------------------------------------------------------

/// Id of the Quality group's virtual Prompt processing row.
pub const PROMPT_PROCESSING_ID: &str = "virtual:prompt_processing";

/// Widest an explainer line may be at 80 columns (spec §3.7).
pub const EXPLAINER_COLS: usize = 76;

/// The honesty line every explainer pane ends with until serve reports the
/// values it admitted (spec §3.3.4).
pub const PREDICTION_NOTE: &str =
    "Predicted from your config, the model card and GPU rules; serve decides at load (see Logs).";

/// Where line 2 (the trade-off) of an explainer comes from.
#[derive(Clone, Copy, Debug)]
pub enum Tradeoff {
    /// A measured pair from `hipfire_config::evidence`, formatted at runtime:
    /// `line` is the evidence line id, `side` names the choice the numbers
    /// describe ("Accurate"). Missing numbers read "not measured on this GPU".
    Evidence {
        line: &'static str,
        side: &'static str,
        /// A second evidence line quoted after the first when it fits
        /// (sidecar disk sizes).
        extra: Option<&'static str>,
    },
    /// Prose with no number in it.
    Text(&'static str),
}

/// How an entry is explained.
#[derive(Clone, Copy, Debug)]
pub enum KnobKind {
    /// A §3.1 curated row: the four-line explainer, line 2 the trade-off.
    Curated(Tradeoff),
    /// An Advanced row with plain-language help: `what` replaces the schema
    /// help, and the why pane adds "When", "Note" and the schema default.
    Advanced {
        /// When to use / leave it.
        when: &'static str,
        /// A load-bearing interaction or caveat, if any.
        note: Option<&'static str>,
    },
}

/// Plain-language entry for one Settings row, keyed by canonical key (or
/// [`PROMPT_PROCESSING_ID`]). Line 3 (why this value, when it applies) and
/// line 4 (key, env, controls) are computed from the row at runtime.
#[derive(Clone, Copy, Debug)]
pub struct SettingKnob {
    pub id: &'static str,
    /// Plain title shown in the Setting column.
    pub title: &'static str,
    /// Selectable values and their plain labels (empty for numbers and free
    /// text). Bool rows use `"true"`/`"false"`. The Prompt processing row's
    /// real labels come from `quality_choices`; these are the generic ones.
    pub options: &'static [(&'static str, &'static str)],
    /// What each value does, shown beside it under "Options on this GPU".
    pub option_help: &'static [(&'static str, &'static str)],
    /// Explainer line 1: what it is.
    pub what: &'static str,
    pub kind: KnobKind,
}

impl SettingKnob {
    const fn with_option_help(
        mut self,
        option_help: &'static [(&'static str, &'static str)],
    ) -> Self {
        self.option_help = option_help;
        self
    }
}

const fn text(
    id: &'static str,
    title: &'static str,
    options: &'static [(&'static str, &'static str)],
    what: &'static str,
    tradeoff: &'static str,
) -> SettingKnob {
    SettingKnob {
        id,
        title,
        options,
        option_help: &[],
        what,
        kind: KnobKind::Curated(Tradeoff::Text(tradeoff)),
    }
}

const fn advanced(
    id: &'static str,
    title: &'static str,
    what: &'static str,
    when: &'static str,
    note: Option<&'static str>,
) -> SettingKnob {
    SettingKnob {
        id,
        title,
        options: &[],
        option_help: &[],
        what,
        kind: KnobKind::Advanced { when, note },
    }
}

const AUTO_ON_OFF_OPTIONS: &[(&str, &str)] = &[("auto", "auto"), ("on", "on"), ("off", "off")];
const VISION_MODE_OPTIONS: &[(&str, &str)] = &[
    ("auto", "auto"),
    ("resident", "resident"),
    ("off", "off"),
    ("on", "on"),
];

/// Every §3.1 curated row, in group order, then the Advanced rows that have
/// plain-language help beyond the schema's ([`KnobKind::Advanced`]).
pub const CURATED: &[SettingKnob] = &[
    // Quality
    SettingKnob {
        id: PROMPT_PROCESSING_ID,
        title: "Prompt processing",
        options: &[("fast", "Fast"), ("accurate", "Accurate")],
        what: "How the model reads your prompt before it starts to answer.",
        option_help: &[],
        kind: KnobKind::Curated(Tradeoff::Evidence {
            line: hipfire_config::evidence::PROMPT_PROCESSING,
            side: "Accurate",
            extra: None,
        }),
    },
    SettingKnob {
        id: "developer.qwen4_moe_sym_iu4",
        title: "Flash-Next experts",
        options: &[("true", "Fast"), ("false", "Accurate")],
        what: "Math for the experts of Flash-Next models while reading prompts.",
        option_help: &[],
        kind: KnobKind::Curated(Tradeoff::Evidence {
            line: hipfire_config::evidence::FLASH_NEXT_EXPERTS,
            side: "Accurate",
            extra: None,
        }),
    },
    SettingKnob {
        id: "memory.ple_format",
        title: "Flash-Next word table",
        options: &[("auto", "auto"), ("q4", "q4 (small)"), ("q8", "q8 (exact)")],
        what: "Which word-table rows Flash-Next loads: small 4-bit or exact 8-bit.",
        option_help: &[
            ("auto", "q4 on gfx1151, q8 elsewhere; uses the other format when only it is on disk."),
            ("q4", "4-bit Lloyd-Max rows (<model>.ple-q4); the load refuses when missing."),
            ("q8", "Exact Q8 rows (embedded, or the <model>.ple-q8 sidecar)."),
        ],
        kind: KnobKind::Curated(Tradeoff::Evidence {
            line: hipfire_config::evidence::FLASH_NEXT_PLE,
            side: "q8",
            extra: Some(hipfire_config::evidence::FLASH_NEXT_PLE_DISK),
        }),
    },
    text(
        "reasoning.mode",
        "Thinking",
        &[("on", "on"), ("off", "off")],
        "Whether the model thinks before it answers.",
        "Off answers sooner, but hard questions may get worse answers.",
    )
    .with_option_help(&[
        ("on", "Emit a hidden think block — stronger reasoning, more tokens, no spec-decode."),
        ("off", "No think block — faster, leaner, spec-decode eligible."),
    ]),
    text(
        "reasoning.effort",
        "Thinking effort",
        &[
            ("auto", "auto"),
            ("none", "none"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
        ],
        "How hard the model is told to think, in its own vocabulary.",
        "Lower effort is shorter thinking; its quality cost is not measured.",
    )
    .with_option_help(&[
        ("auto", "Leave the template's own default in force (Qwen3.8: xhigh)."),
        ("none", "No thinking — closed think block; see the thinking knob."),
        ("low", "Brief, focused reasoning; straight to the conclusion."),
        ("medium", "The model's unsteered baseline — no effort instruction."),
        ("high", "Projected up to the model's next real rung if it has no 'high'."),
        ("xhigh", "Thorough analysis: validate assumptions, weigh alternatives."),
        ("max", "Saturates at the model's top rung."),
    ]),
    text(
        "reasoning.budget",
        "Thinking cap",
        &[
            ("off", "off"),
            ("low", "low"),
            ("med", "med"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
            ("uncapped", "uncapped"),
        ],
        "A hard limit on thinking length per turn. It cuts thinking short.",
        "A low cap can stop mid-thought; prefer effort where the model has one.",
    )
    .with_option_help(&[
        ("low", "512 tokens — quick answers, minimal reasoning."),
        ("med", "2048 tokens — the default; balanced reasoning vs latency."),
        ("high", "8192 tokens — deeper reasoning for harder prompts."),
        ("xhigh", "24576 tokens — very deep reasoning; watch latency."),
        ("max", "32768 tokens — the ceiling; longest bounded reasoning."),
        ("uncapped", "0 = no limit — model reasons until it stops (watch latency)."),
    ]),
    // Speed
    text(
        "experimental.graph.forward",
        "Reuse recorded GPU work",
        AUTO_ON_OFF_OPTIONS,
        "Records the GPU work for a token once and replays it.",
        "Cuts launch overhead; auto follows the model's own policy.",
    ),
    text(
        "fusions.npu_trunk",
        "Use the NPU for prompts",
        &[
            ("off", "off"),
            ("auto", "auto"),
            ("on", "on"),
            ("emulate", "emulate"),
        ],
        "Runs part of Flash-Next prompt reading on the Strix Halo NPU.",
        "Calibrated for gfx1151 only; off is the GPU-only kill switch.",
    ),
    text(
        "prefill.chunk_rows",
        "Prompt chunk size",
        &[],
        "How many prompt rows are read per step.",
        "Bigger chunks read prompts faster but need more memory.",
    ),
    text(
        "serve.multi_slot",
        "Serve several chats at once",
        &[("true", "on"), ("false", "off")],
        "Lets serve answer several requests at the same time.",
        "Better for many clients; each chat may get less memory and speed.",
    ),
    text(
        "attention.flash",
        "Attention kernel",
        &[("auto", "auto"), ("always", "always"), ("never", "never")],
        "Which attention kernel reads the conversation memory.",
        "Leave on auto; forcing a kernel can fail where it is not supported.",
    )
    .with_option_help(&[
        (
            "auto",
            "Engine decides per arch — flash on where it helps (already on by default on gfx11/gfx12).",
        ),
        ("always", "Force flash attention on."),
        ("never", "Disable flash attention (FWHT modes are flash-only)."),
    ]),
    // Memory & context
    text(
        "memory.max_seq",
        "Context length",
        &[],
        "How many tokens of conversation the model can hold.",
        "Longer context reserves more memory up front; shorter leaves room.",
    ),
    text(
        "memory.kv_cache",
        "KV cache precision",
        &[
            ("auto", "auto"),
            ("q8", "q8"),
            ("fwht4", "fwht4"),
            ("fwht3", "fwht3"),
            ("fwht2", "fwht2"),
            ("fp8", "fp8"),
            ("bf16", "bf16"),
            ("f16", "f16"),
            ("f32", "f32"),
        ],
        "Format of the conversation memory (KV cache). Smaller fits longer chats.",
        "Smaller formats save memory; exact formats keep more detail.",
    )
    .with_option_help(&[
        ("auto", "Inherit: non-q8 registry default, else architecture default, else q8."),
        ("q8", "Near-reference, ~2x vs fp16. The default. DFlash-safe."),
        ("fwht4", "4-bit FWHT-rotated K + Q8 V. DFlash-safe."),
        ("fwht3", "3-bit FWHT-rotated, ~5.5x. Best compressed-yet-DFlash-safe."),
        ("fwht2", "2-bit FWHT. Smallest; for genuinely tight VRAM."),
    ]),
    text(
        "memory.kv_adaptive",
        "KV fit-to-memory",
        &[
            ("off", "off"),
            ("conservative", "conservative"),
            ("balanced", "balanced"),
            ("aggressive", "aggressive"),
        ],
        "Lowers KV precision as the chat grows instead of a fixed format.",
        "Fits long chats in tight memory; needs an FWHT key format.",
    )
    .with_option_help(&[
        ("off", "No adaptive downshift — fixed KV precision."),
        ("conservative", "Gentle floor: K=fwht4, V=lloyd4 — downshifts least."),
        ("balanced", "Middle floor: K=fwht3, V=lloyd3."),
        ("aggressive", "Deep floor: K=fwht2, V=lloyd2 — saves the most VRAM."),
        (
            "advanced:k=…,v=…",
            "Pick the K (fwht4/3/2) and V (lloyd4/3/2) floor tiers explicitly — 9 combos.",
        ),
    ]),
    text(
        "memory.gpu_layer_budget",
        "Layers kept on GPU",
        &[],
        "How many layers stay on the GPU; the rest spill to system memory.",
        "Frees GPU memory for longer chats, but typing gets much slower.",
    ),
    text(
        "memory.moe_expert_budget",
        "Expert layers kept on GPU",
        &[],
        "Mixture-of-Experts models: how many layers keep experts on the GPU.",
        "Frees the most memory; a spilled expert costs time only when chosen.",
    ),
    text(
        "memory.offload_exec",
        "Who multiplies spilled layers",
        &[
            ("pcie", "over PCIe"),
            ("cpu", "on CPU"),
            ("passback", "passback (CPU + GPU)"),
        ],
        "Which engine multiplies the weights of layers spilled to memory.",
        "PCIe reads over the link, CPU computes in place, passback does both.",
    )
    .with_option_help(&[
        (
            "pcie",
            "'over PCIe' — `memory.offload_exec = pcie`: the GPU kernels read the host-mapped weights over the link and multiply.",
        ),
        (
            "cpu",
            "'on CPU' — `memory.offload_exec = cpu`: the CPU executes those GEMVs from system RAM; the GPU gets the result back.",
        ),
        (
            "passback",
            "'passback' — `memory.offload_exec = passback`: both do — the GPU takes a share of each step's output rows (see the 'Pass-back GPU share' row) while the CPU takes the rest, on the same weight bytes.",
        ),
    ]),
    text(
        "memory.offload_passback_share",
        "Pass-back GPU share",
        &[],
        "Share of each spilled step the GPU takes back (passback mode only).",
        "Auto schedules it from measured speeds; pin a number only to compare.",
    ),
    text(
        "memory.oom_guard",
        "Out-of-memory guard",
        AUTO_ON_OFF_OPTIONS,
        "Checks memory before a load so a too-big model fails early.",
        "Costs a quick check; off lets a too-big load hit a real out-of-memory.",
    ),
    text(
        "memory.prompt_cache_capacity",
        "Saved prompt cache size",
        &[],
        "How many past answers are kept so their text is not re-read.",
        "More entries save time on long chats and use a little more memory.",
    ),
    text(
        "vision.mode",
        "Vision tower",
        VISION_MODE_OPTIONS,
        "Vision tower: auto streams it per image, resident keeps it in VRAM.",
        "Resident is fastest on the first image; off is text-only, on needs a tower.",
    ),
    // Speculation
    text(
        "speculation.mode",
        "Speculation",
        &[
            ("off", "off"),
            ("auto", "auto"),
            ("ngram", "ngram"),
            ("dflash", "dflash"),
            ("mtp", "mtp"),
            ("dspark", "dspark"),
        ],
        "Which method guesses ahead so the model types faster.",
        "Meant to keep answers identical; a bad guess only wastes a little time.",
    )
    .with_option_help(&[
        ("off", "Plain autoregressive decode, no speculation."),
        ("auto", "Cascade by availability and eligibility (the default)."),
        ("ngram", "Prompt n-gram drafting; needs no draft model."),
        ("dflash", "Force the model's DFlash draft; refuses the load without one."),
        ("mtp", "Force the model's native MTP head."),
        ("dspark", "Force the DSpark sidecar drafter."),
    ]),
    text(
        "speculation.mtp",
        "MTP",
        AUTO_ON_OFF_OPTIONS,
        "Guesses ahead with the model's own multi-token head, when it has one.",
        "Faster typing when guesses land; answers are not changed.",
    )
    .with_option_help(&[
        ("off", "No multi-token prediction."),
        ("on", "Force the MTP head where the model has one."),
        (
            "auto",
            "Use MTP only where the producer has a measured AR floor and the model is not host-spilled; else single-token.",
        ),
    ]),
    text(
        "speculation.mtp_k",
        "MTP draft window",
        &[],
        "How many tokens MTP guesses per step.",
        "A longer window pays when guesses land and wastes work when not.",
    ),
    text(
        "speculation.dflash",
        "DFlash",
        AUTO_ON_OFF_OPTIONS,
        "Guesses ahead with a small paired draft model, when installed.",
        "Faster typing on supported pairs; needs the draft file installed.",
    )
    .with_option_help(&[
        ("off", "No DFlash speculative decoding (plain autoregressive)."),
        (
            "on",
            "Use the DFlash draft whenever one is attached + the gate allows; fails the load without a draft.",
        ),
        ("auto", "Engage DFlash when eligible (greedy, thinking-off, arch)."),
    ]),
    text(
        "speculation.ngram",
        "N-gram drafting",
        &[("auto", "auto"), ("on", "on"), ("off", "off")],
        "Guesses ahead by copying repeated text from the chat; no extra model.",
        "Helps repetitive text such as code edits; little help elsewhere.",
    ),
    // Hardware & devices
    text(
        "hardware.devices",
        "GPUs to use",
        &[],
        "Which GPUs this serve uses, in order.",
        "More cards fit bigger models; unset uses the visible GPUs.",
    ),
    text(
        "hardware.allow_mixed_arch",
        "Allow mixed GPU types",
        &[("true", "on"), ("false", "off")],
        "Lets one model span GPUs of different architectures.",
        "Fits more, but the slowest card sets the pace.",
    ),
    text(
        "hardware.tp_use_rccl",
        "Multi-GPU transport",
        &[("true", "on"), ("false", "off")],
        "Uses RCCL to combine results across GPUs.",
        "RCCL is usually faster between GPUs; off uses the built-in path.",
    ),
    // Advanced rows with plain-language help beyond the schema's.
    advanced(
        "serve.default_model",
        "Default model",
        concat!(
            "The model serve pre-warms and chat uses when no model is named.",
            " ",
            "Picks which weights load. No quality/speed tradeoff by itself — it just selects the model."
        ),
        "Set it to the model you use most. Change it from the Models tab.",
        None,
    ),
    advanced(
        "memory.kv_backend",
        "KV cache layout",
        concat!(
            "How the conversation memory (KV cache) is laid out in GPU memory.",
            " ",
            "vmm reserves the whole context window and commits physical pages on demand; legacy allocates one contiguous block up front."
        ),
        "Leave on vmm; switch to legacy only to rule out a paging problem.",
        None,
    )
    .with_option_help(&[
        ("legacy", "Physically contiguous allocation of the full context."),
        ("vmm", "Reserve the logical window, commit pages as the chat grows."),
    ]),
    advanced(
        "memory.kv_k",
        "KV key format",
        concat!(
            "Qwen-only override of the key (K) half of the KV cache format.",
            " ",
            "Empty leaves the format the KV cache precision row selects; a value forces just the K axis."
        ),
        "Only for experiments on Qwen models; normally follow the KV cache precision row.",
        None,
    ),
    advanced(
        "memory.kv_v",
        "KV value format",
        concat!(
            "Qwen-only override of the value (V) half of the KV cache format.",
            " ",
            "Empty leaves the format the KV cache precision row selects; a value forces just the V axis."
        ),
        "Only for experiments on Qwen models; normally follow the KV cache precision row.",
        None,
    ),
    advanced(
        "memory.mmq.screen",
        "MMQ screening",
        concat!(
            "Mixed-precision matmul screening for quantized GEMMs.",
            " ",
            "Detects Q8_1 outlier rows and falls back to WMMA. Arch-gated to gfx906 + RDNA3/3.5 (gfx1100-1103/1150-1152) and only fires when MMQ is active."
        ),
        "Leave on auto unless profiling kernels.",
        None,
    )
    .with_option_help(&[
        ("off", "No outlier sweep — max speed, risks tool-call corruption (#87)."),
        ("on", "Force the per-weight outlier sweep."),
        ("auto", "Screen on the supported arches (gfx906 + RDNA3/3.5) when MMQ is active."),
    ]),
    advanced(
        "speculation.dflash_adaptive_b",
        "DFlash adaptive batch",
        concat!(
            "Lets DFlash adapt its draft verify-block width to recent acceptance.",
            " ",
            "Follows the trailing 8-cycle acceptance depth (τ̂+2 rows), shrunk only past 2k context where the fixed full block loses throughput. This is not output-identical: realized tokens can differ from fixed-B (observed even at temp 0) because a narrower verify window moves window boundaries; per-position sampling stays target-lossless."
        ),
        "Opt in for long-context DFlash sessions after measuring your workload; off keeps the fixed full block. Mutually exclusive with the retained PM4 verify route (PM4 loads keep the fixed B=16 shape).",
        Some("HIPFIRE_DFLASH_ADAPTIVE_B=0 forces the fixed block even when enabled. Short contexts always get the full block."),
    )
    .with_option_help(&[
        ("true", "Size the draft batch to recent acceptance (context-gated, seeded at full)."),
        ("false", "Fixed draft batch size (default)."),
    ]),
    advanced(
        "prompt.normalize",
        "Prompt normalization",
        concat!(
            "Collapses 3+ consecutive newlines to 2 before tokenizing.",
            " ",
            "Stabilizes tokenization so whitespace differences don't swing performance; no meaning change for normal text."
        ),
        "Leave on. Turn off only when raw repeated newlines are semantically load-bearing.",
        None,
    )
    .with_option_help(&[
        ("true", "Collapse 3+ newlines to 2 before tokenizing."),
        ("false", "Leave whitespace as-is (when newlines are load-bearing)."),
    ]),
    advanced(
        "prompt.default_chatml",
        "Default ChatML",
        concat!(
            "Whether to apply a ChatML scaffold when a model has no embedded chat template.",
            " ",
            "Improves chat formatting for templateless models; harmless when a model carries its own template."
        ),
        "Leave on; turn off only for raw/base models where a ChatML scaffold hurts.",
        None,
    )
    .with_option_help(&[
        ("true", "Use a ChatML scaffold as the fallback when a model has no embedded template."),
        (
            "false",
            "No fallback scaffold — a templateless model gets none (models with their own template are unaffected either way).",
        ),
    ]),
    advanced(
        "prompt.chat_template",
        "Chat template",
        concat!(
            "Path to a custom Jinja chat template (.j2/.jinja). Empty = model default.",
            " ",
            "Overrides how messages are rendered into the prompt. Wrong templates degrade output."
        ),
        "Set only when you need a specific template; leave empty to use the model's own.",
        Some("Must point to a file that exists; an invalid path is rejected on save."),
    ),
    advanced(
        "generation.temperature",
        "Temperature",
        concat!(
            "Sampling randomness for generation.",
            " ",
            "Lower = more deterministic/repeatable; higher = more varied/creative but less reliable."
        ),
        "Low (<=0.3) for code/factual; higher (0.7-1.0) for brainstorming.",
        Some("Greedy (~0) is what spec-decode runs at; sampling adds cost."),
    ),
    advanced(
        "generation.top_p",
        "Top-p (nucleus)",
        concat!(
            "Caps sampling to the smallest set of tokens whose probability sums to p.",
            " ",
            "Lower = safer, more focused; higher = more diverse. Interacts with temperature."
        ),
        "Leave near default; lower it to tighten outputs.",
        None,
    ),
    advanced(
        "serve.host",
        "Serve endpoint",
        concat!(
            "The bind address (and port) of the OpenAI-compatible serve API.",
            " ",
            "127.0.0.1 is local-only (the default); 0.0.0.0 listens on all interfaces. Chat and API clients connect here."
        ),
        "Set 0.0.0.0 to expose serve on the network; the default keeps it private to this machine.",
        Some("Paired with port (default 11435). Changes take effect on the next serve/restart."),
    ),
    advanced(
        "generation.max_tokens",
        "Max tokens",
        concat!(
            "Upper bound on tokens generated per response.",
            " ",
            "Caps response length and runaway generation. No quality effect within the cap."
        ),
        "Raise for long outputs; lower to bound latency/cost.",
        None,
    ),
];

/// The entry for `id` (canonical key or [`PROMPT_PROCESSING_ID`]), curated
/// row or Advanced help.
pub fn knob(id: &str) -> Option<&'static SettingKnob> {
    CURATED.iter().find(|knob| knob.id == id)
}

/// The curated row entry for `id`; `None` for Advanced help entries.
pub fn curated_knob(id: &str) -> Option<&'static SettingKnob> {
    knob(id).filter(|knob| matches!(knob.kind, KnobKind::Curated(_)))
}

/// What an explainer is rendered for: the row's computed state.
#[derive(Clone, Copy)]
pub struct ExplainCtx<'a> {
    pub arch: &'a ArchCtx,
    pub effective: &'a Effective,
    pub origin: &'a Origin,
    /// The row's lock text (`set by HIPFIRE_KV_MODE; unset it to change`).
    pub locked: Option<&'a str>,
    /// The "Using now" text of the row (the prediction).
    pub display: &'a str,
    /// The value serve reports for the resident model, when the row has one.
    pub live: Option<&'a str>,
    /// `config.toml` is newer than serve and the row is read at load/start.
    pub restart: bool,
}

/// Line 3 of an explainer: why the row has this value and when it applies.
/// A lock outranks everything; a live serve value replaces the prediction.
fn why_line(ctx: &ExplainCtx<'_>) -> String {
    let applies = if ctx.restart {
        "↻ restart to apply".to_owned()
    } else {
        format!("Applies: {}", ctx.effective.applies.label())
    };
    match (ctx.locked, ctx.live) {
        (Some(lock), _) => capitalize(lock),
        (None, Some(live)) => {
            format!(
                "Live from serve: {live}; config predicts {}. {applies}.",
                ctx.display
            )
        }
        (None, None) => {
            let reason = match (&ctx.effective.reason, ctx.effective.used_by_model) {
                (_, Some(false)) => "The active model does not read this setting".to_owned(),
                (Some(reason), _) => capitalize(reason),
                (None, _) => origin_phrase(ctx.origin),
            };
            format!("{reason}. {applies}.")
        }
    }
}

/// Dedicated line for a key with no curated entry (Advanced rows): the
/// schema help over two lines, then why/when, then the key and its controls.
pub fn schema_explainer(key: &str, ctx: &ExplainCtx<'_>) -> [String; 4] {
    let text = match (knob(key), field(key)) {
        (Some(knob), _) => knob.what.to_owned(),
        (None, Some(schema)) => schema.help.to_owned(),
        (None, None) => format!("{key} has no schema help."),
    };
    let mut wrapped = wrap_cols(&text, EXPLAINER_COLS);
    let (first, second) = match wrapped.len() {
        0 => (String::new(), String::new()),
        1 => (wrapped.remove(0), String::new()),
        _ => {
            let first = wrapped.remove(0);
            let mut second = wrapped.remove(0);
            if !wrapped.is_empty() {
                second = second.chars().take(EXPLAINER_COLS - 1).collect();
                second.push('…');
            }
            (first, second)
        }
    };
    let controls = match crate::hipfire::writer::field_kind(key) {
        Some(FieldKind::Enum(_)) => "←/→ preview, Enter saves",
        Some(FieldKind::Bool) => "←/→ toggles",
        Some(_) => "Enter edits",
        None => "edit with hipfire config set",
    };
    let keys_line = match env_name(key) {
        Some(env) => format!("{key} · {env} · {controls}"),
        None => format!("{key} · {controls}"),
    };
    [fit(first), fit(second), fit(why_line(ctx)), fit(keys_line)]
}

/// Greedy word wrap at `cols` columns.
fn wrap_cols(text: &str, cols: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let extra = usize::from(!line.is_empty()) + word.chars().count();
        if !line.is_empty() && line.chars().count() + extra > cols {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

/// The four-line explainer of a curated row (what / trade-off / why it has
/// this value and when it applies / key · env · controls), each at most
/// [`EXPLAINER_COLS`] columns. `None` when `id` has no curated entry.
pub fn explainer(id: &str, ctx: &ExplainCtx<'_>) -> Option<[String; 4]> {
    let knob = curated_knob(id)?;
    let choices = (id == PROMPT_PROCESSING_ID).then(|| quality_choices(ctx.arch));
    let no_route = choices.as_ref().is_some_and(Vec::is_empty);

    let tradeoff = if no_route {
        "This GPU has no choice here; the raw keys stay in Advanced.".to_owned()
    } else {
        match knob.kind {
            KnobKind::Curated(Tradeoff::Text(text)) => text.to_owned(),
            KnobKind::Curated(Tradeoff::Evidence { line, side, extra }) => {
                evidence_text(line, side, extra, ctx.arch)
            }
            KnobKind::Advanced { .. } => return None,
        }
    };

    let why = why_line(ctx);

    let (key_text, env_text, controls) = match &choices {
        Some(choices) => {
            let keys = choice_keys(choices);
            let env = keys.first().and_then(|key| env_name(key));
            let shown = if keys.is_empty() {
                "kernel.iu4_prefill".to_owned()
            } else {
                keys.join(", ")
            };
            (
                shown,
                env,
                if no_route {
                    "Edit raw keys in Advanced"
                } else {
                    "←/→ choose, Enter saves"
                },
            )
        }
        None => (
            id.to_owned(),
            env_name(id),
            match crate::hipfire::writer::field_kind(id) {
                Some(FieldKind::Enum(_)) => "←/→ preview, Enter saves",
                Some(FieldKind::Bool) => "←/→ toggles",
                Some(_) => "Enter edits",
                None => "edit with hipfire config set",
            },
        ),
    };
    let keys_line = match env_text {
        Some(env) => format!("{key_text} · {env} · {controls}"),
        None => format!("{key_text} · {controls}"),
    };

    Some([
        fit(knob.what.to_owned()),
        fit(tradeoff),
        fit(why),
        fit(keys_line),
    ])
}

fn evidence_text(line: &str, side: &str, extra: Option<&str>, arch: &ArchCtx) -> String {
    let gpu = arch.gpu_arch.as_deref().unwrap_or("");
    let Some(row) = evidence_for(line, gpu, arch.a8_prefill) else {
        return format!("{side}: not measured on this GPU.");
    };
    let mut parts = Vec::new();
    if let Some(closer) = format_closer(row) {
        parts.push(format!("{closer} to the full model"));
    }
    let prompts = row.speed_metric.is_some_and(|metric| metric.contains("pp"));
    if let Some(slower) = format_slower(row) {
        parts.push(format!(
            "{} {slower}",
            if prompts { "prompts" } else { "typing" }
        ));
    }
    if parts.is_empty() {
        return format!("{side}: not measured on this GPU.");
    }
    let mut text = format!("{side}: {}.", parts.join(", "));
    // Optional tails, only while the line still fits.
    let mut tails = Vec::new();
    if prompts
        && row
            .typing
            .is_some_and(|typing| typing.starts_with("unchanged"))
    {
        tails.push("typing unchanged".to_owned());
    }
    if let Some((small, exact)) = extra
        .and_then(|extra| evidence_for(extra, gpu, arch.a8_prefill))
        .and_then(|disk| disk.disk_gb)
    {
        tails.push(format!("disk {} vs {} GB", two_sig(small), two_sig(exact)));
    }
    for tail in tails {
        let candidate = format!("{}, {tail}.", text.trim_end_matches('.'));
        if candidate.chars().count() <= EXPLAINER_COLS {
            text = candidate;
        }
    }
    text
}

fn origin_phrase(origin: &Origin) -> String {
    match origin {
        Origin::Default => "Built-in default".to_owned(),
        Origin::DefaultArch(arch) => format!("Built-in default for {arch}"),
        Origin::Registry => "Set by the model card".to_owned(),
        Origin::Profile(name) => format!("Set by profile {name}"),
        Origin::Preset => "Set by the applied tuning preset".to_owned(),
        Origin::You => "Set by you for all models".to_owned(),
        Origin::ThisModel => "Set by this model's own value".to_owned(),
        Origin::Env(name) => format!("Set by {name}"),
        Origin::OneShot => "Set on the command line".to_owned(),
    }
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Distinct keys a virtual row's choices write, in first-seen order.
fn choice_keys(choices: &[Choice]) -> Vec<&'static str> {
    let mut keys: Vec<&'static str> = Vec::new();
    for choice in choices {
        for (key, _) in &choice.writes {
            if !keys.contains(key) {
                keys.push(key);
            }
        }
    }
    keys
}

/// The `HIPFIRE_*` variable that sets `key`, if it has one.
pub fn env_name(key: &str) -> Option<String> {
    field(key)
        .and_then(|schema| schema.env_compat.map(str::to_owned))
        .or_else(|| developer_env_for_key(key))
}

pub fn fit(text: String) -> String {
    if text.chars().count() <= EXPLAINER_COLS {
        return text;
    }
    let mut cut: String = text.chars().take(EXPLAINER_COLS - 1).collect();
    cut.push('…');
    cut
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hipfire::writer::EDITABLE_FIELDS;

    /// Canonical key of an `EDITABLE_FIELDS` entry (keyed by legacy key).
    fn canonical(key: &str) -> &'static str {
        field(key)
            .unwrap_or_else(|| panic!("{key} is not a schema key"))
            .key
    }

    #[test]
    fn lookup_separates_curated_rows_from_advanced_help() {
        assert_eq!(
            knob("speculation.dflash").expect("dflash entry").title,
            "DFlash"
        );
        assert!(curated_knob("speculation.dflash").is_some());
        let temperature = knob("generation.temperature").expect("temperature help");
        assert!(matches!(temperature.kind, KnobKind::Advanced { .. }));
        assert!(curated_knob("generation.temperature").is_none());
        assert!(knob("does_not_exist").is_none());
    }

    #[test]
    fn every_entry_is_a_schema_key_listed_once() {
        for (index, knob) in CURATED.iter().enumerate() {
            assert!(
                knob.id == PROMPT_PROCESSING_ID
                    || hipfire_config::canonical_config_key(knob.id).as_deref() == Some(knob.id),
                "{} is not a canonical schema key",
                knob.id
            );
            assert!(
                CURATED[..index].iter().all(|other| other.id != knob.id),
                "{} is listed twice",
                knob.id
            );
        }
    }

    #[test]
    fn dflash_help_does_not_promise_an_mtp_fallback() {
        // docs/CONFIG.md: `dflash_mode=on` requires a DFlash draft and never
        // enables MTP; the general selector is `speculation`.
        let help = field("speculation.dflash").unwrap().help;
        assert!(
            !help.contains("else its MTP head"),
            "dflash help must not promise an MTP fallback: {help}"
        );
        assert!(help.contains("never falls back"));
        let spec = knob("speculation.mode").expect("speculation entry");
        let values: Vec<&str> = spec.option_help.iter().map(|(v, _)| *v).collect();
        assert_eq!(values, hipfire_config::SPECULATION_MODES);
    }

    #[test]
    fn kv_adaptive_has_per_option_help() {
        let k = knob("memory.kv_adaptive").expect("kv_adaptive entry");
        // 4 presets + the advanced-family entry.
        assert_eq!(k.option_help.len(), 5);
        assert!(k.option_help.iter().any(|(v, _)| *v == "aggressive"));
        assert!(k.option_help.iter().any(|(v, _)| v.starts_with("advanced")));
        assert!(
            field("memory.kv_adaptive")
                .unwrap()
                .help
                .to_lowercase()
                .contains("fwht"),
            "help explains the FWHT-K requirement"
        );
    }

    #[test]
    fn enum_and_bool_knobs_carry_per_option_help() {
        // Every editable enum/bool key's explainer lists what each value does, so
        // the option semantics show in the in-tab pane (no tab switching).
        use crate::hipfire::writer::FieldKind;
        for f in EDITABLE_FIELDS {
            if matches!(f.kind, FieldKind::Enum(_) | FieldKind::Bool) {
                let key = canonical(f.key);
                let info = knob(key).unwrap_or_else(|| panic!("no explainer for {key}"));
                assert!(
                    !info.option_help.is_empty(),
                    "enum/bool key {key} has no per-option help"
                );
            }
        }
    }

    #[test]
    fn every_editable_field_has_help() {
        // 5e: selecting any editable key must show an explainer.
        for f in EDITABLE_FIELDS {
            let key = canonical(f.key);
            assert!(knob(key).is_some(), "no explainer for editable key {key}");
        }
    }

    #[test]
    fn load_bearing_interactions_are_documented() {
        // The two interactions this phase exists to surface.
        assert!(
            field("speculation.dflash")
                .unwrap()
                .help
                .contains("thinking"),
            "dflash help must mention the thinking interaction"
        );
        assert!(
            field("reasoning.mode")
                .unwrap()
                .help
                .contains("spec-decode"),
            "thinking help must mention the dflash interaction"
        );
        assert!(
            field("memory.kv_cache").unwrap().help.contains("inherit"),
            "kv_cache help must explain auto = inherit"
        );
    }

    fn arch(gpu: Option<&str>, a8: hipfire_config::effective::A8Prefill) -> ArchCtx {
        ArchCtx {
            gpu_arch: gpu.map(str::to_owned),
            gpu_count: 1,
            model_arch_id: None,
            registry_tag: Some("qwen3.8:27b".into()),
            a8_prefill: a8,
        }
    }

    fn explain(id: &str, arch: &ArchCtx) -> [String; 4] {
        let resolved = hipfire_config::resolve(Vec::new()).unwrap();
        let key = if id == PROMPT_PROCESSING_ID {
            "kernel.iu4_prefill"
        } else {
            id
        };
        let effective = hipfire_config::effective::effective(key, &resolved, arch);
        let origin = Origin::Default;
        explainer(
            id,
            &ExplainCtx {
                arch,
                effective: &effective,
                origin: &origin,
                locked: None,
                display: "",
                live: None,
                restart: false,
            },
        )
        .unwrap_or_else(|| panic!("no explainer for {id}"))
    }

    fn archs() -> Vec<ArchCtx> {
        use hipfire_config::effective::A8Prefill::*;
        let mut out = vec![arch(None, NotLanded)];
        for gpu in ["gfx1151", "gfx1201", "gfx1100", "gfx1030"] {
            for a8 in [NotLanded, OptIn, DefaultOn] {
                out.push(arch(Some(gpu), a8));
            }
        }
        out
    }

    #[test]
    fn every_curated_row_has_an_entry() {
        use hipfire_config::intent::Intent;
        assert!(curated_knob(PROMPT_PROCESSING_ID).is_some());
        for intent in Intent::ALL {
            for key in intent.curated_present() {
                // The raw prefill keys are replaced by the virtual row.
                if matches!(key, "kernel.iu4_prefill" | "kernel.gfx11_a8_prefill") {
                    continue;
                }
                assert!(curated_knob(key).is_some(), "no curated entry for {key}");
            }
        }
        for knob in CURATED {
            assert!(!knob.title.is_empty() && !knob.what.is_empty());
        }
    }

    #[test]
    fn explainers_are_four_lines_of_at_most_76_columns() {
        for arch in archs() {
            for knob in CURATED
                .iter()
                .filter(|knob| curated_knob(knob.id).is_some())
            {
                let lines = explain(knob.id, &arch);
                for line in &lines {
                    let width = line.chars().count();
                    assert!(
                        width <= EXPLAINER_COLS,
                        "{} on {:?}: {width} cols: {line}",
                        knob.id,
                        arch.gpu_arch
                    );
                    assert!(
                        !line.ends_with('…'),
                        "{} on {:?} was truncated: {line}",
                        knob.id,
                        arch.gpu_arch
                    );
                    assert!(!line.is_empty(), "{}", knob.id);
                }
            }
        }
    }

    #[test]
    fn trade_off_numbers_come_from_the_evidence_table() {
        use hipfire_config::effective::A8Prefill::*;
        let line = |id: &str, gpu: &str, a8| explain(id, &arch(Some(gpu), a8))[1].clone();
        // The "typing unchanged" tail is dropped when it would not fit 76 columns.
        assert_eq!(
            line(PROMPT_PROCESSING_ID, "gfx1151", OptIn),
            "Accurate: 43% closer to the full model, prompts 1.6× slower."
        );
        assert!(line(PROMPT_PROCESSING_ID, "gfx1151", NotLanded).contains("42% closer"));
        assert!(line(PROMPT_PROCESSING_ID, "gfx1151", NotLanded).contains("2.8× slower"));
        assert!(line(PROMPT_PROCESSING_ID, "gfx1201", NotLanded).contains("40% closer"));
        // Never another arch's row; a GPU with no choices says so.
        assert!(line(PROMPT_PROCESSING_ID, "gfx1030", NotLanded).contains("no choice here"));
        assert_eq!(
            line("developer.qwen4_moe_sym_iu4", "gfx1030", NotLanded),
            "Accurate: not measured on this GPU."
        );
        assert!(line("developer.qwen4_moe_sym_iu4", "gfx1151", NotLanded).contains("36% closer"));
        // Strix Halo and the 7900 XTX were measured for Flash-Next experts.
        assert!(line("developer.qwen4_moe_sym_iu4", "gfx1100", NotLanded).contains("closer"));
        assert_eq!(
            line("developer.qwen4_moe_sym_iu4", "gfx1201", NotLanded),
            "Accurate: not measured on this GPU."
        );
        let ple = line("memory.ple_format", "gfx1151", NotLanded);
        assert!(
            ple.contains("1.5% closer") && ple.contains("typing 1.1× slower"),
            "{ple}"
        );
        assert!(ple.contains("disk 29 vs 54 GB"), "{ple}");
    }

    #[test]
    fn no_percentage_is_hard_coded_in_the_curated_section() {
        let source = include_str!("knobs.rs");
        let section = source
            .split("// Curated Settings rows")
            .nth(1)
            .and_then(|rest| rest.split("#[cfg(test)]").next())
            .expect("curated section");
        assert!(!section.contains('%'), "a percentage is hard-coded");
    }

    #[test]
    fn lock_and_unused_states_reach_line_three() {
        let arch = arch(
            Some("gfx1151"),
            hipfire_config::effective::A8Prefill::NotLanded,
        );
        let resolved = hipfire_config::resolve(Vec::new()).unwrap();
        let effective = hipfire_config::effective::effective("memory.kv_cache", &resolved, &arch);
        let origin = Origin::Env("HIPFIRE_KV_MODE".into());
        let lines = explainer(
            "memory.kv_cache",
            &ExplainCtx {
                arch: &arch,
                effective: &effective,
                origin: &origin,
                locked: Some("set by HIPFIRE_KV_MODE; unset it to change"),
                display: "q8",
                live: None,
                restart: false,
            },
        )
        .unwrap();
        assert_eq!(lines[2], "Set by HIPFIRE_KV_MODE; unset it to change");
        assert!(lines[3].starts_with("memory.kv_cache · HIPFIRE_KV_MODE"));
    }
}
