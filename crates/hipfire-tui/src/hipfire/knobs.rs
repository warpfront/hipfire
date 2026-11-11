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
    field, is_deprecated_key,
};

use crate::hipfire::writer::FieldKind;

/// One key's help. Every field is a short, honest sentence (no marketing).
#[derive(Clone, Copy, Debug)]
pub struct KnobInfo {
    pub key: &'static str,
    /// Human-friendly title.
    pub title: &'static str,
    /// What it is (one line).
    pub summary: &'static str,
    /// What it does + the tradeoff (pros/cons).
    pub effect: &'static str,
    /// The default value, as a string (cross-checked against config defaults).
    pub default: &'static str,
    /// When to use / leave it.
    pub when: &'static str,
    /// A load-bearing interaction or caveat, if any.
    pub note: Option<&'static str>,
    /// For enum / bool keys: each selectable value and what it does. Empty for
    /// numeric / free-string keys. Rendered in the explainer so users see what
    /// every option means without leaving the Settings tab.
    pub options: &'static [(&'static str, &'static str)],
}

/// Help for `key`, if a curated entry exists.
pub fn knob_info(key: &str) -> Option<&'static KnobInfo> {
    KNOBS.iter().find(|k| k.key == key)
}

/// Curated, honest help for the Settings keys. Order is not significant (lookup
/// is by key). Defaults here mirror `config::defaults()` — see the cross-check
/// test below so the two can't silently drift.
pub const KNOBS: &[KnobInfo] = &[
    KnobInfo {
        key: "default_model",
        title: "Default model",
        summary: "The model serve pre-warms and chat uses when no model is named.",
        effect: "Picks which weights load. No quality/speed tradeoff by itself — it just selects the model.",
        default: "qwen3.5:9b",
        when: "Set it to the model you use most. Change it from the Models tab.",
        note: None,
        options: &[],
    },
    KnobInfo {
        key: "max_seq",
        title: "Context (KV cache capacity)",
        summary: "Maximum tokens the KV cache is allocated to hold at load.",
        effect: "Larger = longer context, but more VRAM reserved up front. Smaller = less VRAM, shorter ceiling.",
        default: "32768",
        when: "Raise for long documents/agents; lower to fit a tight VRAM budget.",
        note: Some("Allocated at load — changing it takes effect on the next serve/restart."),
        options: &[],
    },
    KnobInfo {
        key: "gpu_layer_budget",
        title: "Layers kept on the GPU",
        summary: "How many of the model's layers stay on the GPU. The rest spill to system RAM.",
        effect: "Frees VRAM for a longer context, but decode slows sharply: a spilled layer's weights are read over PCIe every step instead of from VRAM. Load also takes much longer.",
        default: "",
        when: "Leave it unset for normal use — every model stays fully on the GPU, including MoE. Set a count or explicitly opt into 'auto' when the model or context will not otherwise fit.",
        note: Some("The number counts layers ON THE GPU, not layers offloaded: 3 on a 64-layer model keeps 3 on the GPU and spills the other 61. Explicit 'auto' (-1) places by fit: the smallest spill that loads, routed experts first, and a model that already fits is left fully resident."),
        options: &[],
    },
    KnobInfo {
        key: "moe_expert_budget",
        title: "Expert layers kept on the GPU",
        summary: "For a Mixture-of-Experts model: how many layers keep their routed experts on the GPU. The rest spill to system RAM.",
        effect: "Frees the VRAM a MoE model's experts dominate. A spilled expert is read over PCIe only when the router selects it, so a decode step pays for its few routed experts rather than the whole layer.",
        default: "",
        when: "Leave it unset for normal use — every routed expert stays on the GPU. Set a count or explicitly opt into 'auto' when a MoE model's experts do not fit, or when the row above still leaves too little VRAM.",
        note: Some("Counts layers whose routed experts stay ON the GPU; llama.cpp's --n-cpu-moe K is this value written as n_layers - K. Only routed experts move — attention, router, shared expert and KV stay on the card. Needs the packed-expert layout (uniform, packable MQ4 experts). No effect on a dense model, and none on an architecture that cannot spill experts: a configured spill fails that model's load rather than being ignored."),
        options: &[],
    },
    KnobInfo {
        key: "offload_exec",
        title: "Who multiplies a spilled layer",
        summary: "Which engine executes the ops that read a spilled layer's weights: the GPU over PCIe, the CPU, or both at once.",
        effect: "The CPU arm replaces a per-step read of every spilled weight over the link with a host-side GEMV, which is what makes a spill pay instead of costing. It is a per-step host sync (no hipGraph), and the GPU kernels are the faster engine once the bytes are resident. The pass-back ('both') splits each step's output rows between the two engines, so the GPU does useful work during the CPU arm's host time.",
        default: "pcie",
        when: "Leave on PCIe unless a spilled layer's per-step link read is the bottleneck; 'cpu' is the arm the CPU-exec offload path exists for, and 'passback' is the arm for a host whose link and CPU can stream the same weight bytes concurrently.",
        note: Some("Decides who multiplies, never what is spilled: placement stays the 'GPU layers' row, and the KV cache stays in VRAM either way. Refused together with a retained-replay (Redline) backend, and unset/empty/unknown all fall back to pcie. Read once at load — changing it takes effect on the next serve/restart."),
        options: &[
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
        ],
    },
    KnobInfo {
        key: "offload_passback_share",
        title: "Pass-back GPU share",
        summary: "How much of a spilled step the GPU takes back, as a share of its output rows.",
        effect: "Higher hands more of each step to the faster-per-byte engine (usually the link, on a narrow PCIe link) and shortens the CPU arm; lower does the reverse. Only meaningful when 'Who multiplies a spilled layer' is 'passback'.",
        default: "auto",
        when: "Leave on auto: the engine schedules the share from this host's own measured engine rates and refines it as it decodes. Pin a number only when a measurement disagrees — `hipfire offload-bench` reports this host's numbers.",
        note: Some("0 disables the pass-back, so 'passback' then behaves exactly as 'cpu' (the same bytes, the same order) — the A/B baseline. Ignored unless the mode is 'passback'; a share set in another mode prints one informational line at load. Up to 0.5, because beyond half the rows the choice is better expressed by switching the mode to 'pcie'. Read once at load."),
        options: &[],
    },
    KnobInfo {
        key: "kv_cache",
        title: "KV cache precision",
        summary: "Precision/memory tradeoff for the attention key/value cache.",
        effect: "Lower precision saves VRAM and can speed decode, at some quality risk. Higher keeps fidelity.",
        default: "auto",
        when: "Leave on auto unless you need to fit a larger context or are A/B-testing precision.",
        note: Some("auto is the inherit sentinel — it resolves to a non-q8 registry default_kv_mode when present, else the architecture default (fp8 on exact single-GPU gfx1201 Qwen), else q8. A model can ship a compressed default via the registry; a q8 default falls through to the architecture default, and nothing is guessed from the GPU arch alone."),
        options: &[
            ("auto", "Inherit: non-q8 registry default, else architecture default, else q8."),
            ("q8", "Near-reference, ~2x vs fp16. The default. DFlash-safe."),
            ("fwht4", "4-bit FWHT-rotated K + Q8 V. DFlash-safe."),
            ("fwht3", "3-bit FWHT-rotated, ~5.5x. Best compressed-yet-DFlash-safe."),
            ("fwht2", "2-bit FWHT. Smallest; for genuinely tight VRAM."),
            // lifecycle: deprecated since 0.4.0, removal 0.5.0 — Givens asym KV and the asymN/turboN aliases are superseded by fwht3
            ("asym4", "Deprecated (removal in 0.5.0): 4-bit Givens-era name; use fwht4."),
            ("asym3", "Deprecated (removal in 0.5.0): 3-bit Givens-era name; use fwht3."),
            ("asym2", "Deprecated (removal in 0.5.0): 2-bit Givens-era name; use fwht2."),
            ("turbo / turbo2-4", "Deprecated (removal in 0.5.0) aliases; use fwhtN."),
        ],
    },
    KnobInfo {
        key: "kv_adaptive",
        title: "Adaptive KV downshift",
        summary: "Dynamically drops KV precision as context grows, instead of a fixed mode.",
        effect: "Saves VRAM on long sessions by spending bits where they matter. Off = fixed precision. Requires an FWHT K mode (kv_cache fwht*).",
        default: "off",
        when: "Leave off unless you run long contexts on a tight VRAM budget and want graceful downshift.",
        note: Some("Requires an FWHT K mode to take effect. The `advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>` combos set explicit K/V floor tiers — all 9 are cyclable here (and via the CLI)."),
        options: &[
            ("off", "No adaptive downshift — fixed KV precision."),
            ("conservative", "Gentle floor: K=fwht4, V=lloyd4 — downshifts least."),
            ("balanced", "Middle floor: K=fwht3, V=lloyd3."),
            ("aggressive", "Deep floor: K=fwht2, V=lloyd2 — saves the most VRAM."),
            ("advanced:k=…,v=…", "Pick the K (fwht4/3/2) and V (lloyd4/3/2) floor tiers explicitly — 9 combos."),
        ],
    },
    KnobInfo {
        key: "kv_backend",
        title: "KV cache layout",
        summary: "How the conversation memory (KV cache) is laid out in GPU memory.",
        effect: "vmm reserves the whole context window and commits physical pages on demand; legacy allocates one contiguous block up front.",
        default: "vmm",
        when: "Leave on vmm; switch to legacy only to rule out a paging problem.",
        note: None,
        options: &[
            ("legacy", "Physically contiguous allocation of the full context."),
            ("vmm", "Reserve the logical window, commit pages as the chat grows."),
        ],
    },
    KnobInfo {
        key: "kv_k",
        title: "KV key format",
        summary: "Qwen-only override of the key (K) half of the KV cache format.",
        effect: "Empty leaves the format the KV cache precision row selects; a value forces just the K axis.",
        default: "",
        when: "Only for experiments on Qwen models; normally follow the KV cache precision row.",
        note: Some("asymN and turboN names are deprecated (removal 0.5.0); use fwhtN or q8."),
        options: &[],
    },
    KnobInfo {
        key: "kv_v",
        title: "KV value format",
        summary: "Qwen-only override of the value (V) half of the KV cache format.",
        effect: "Empty leaves the format the KV cache precision row selects; a value forces just the V axis.",
        default: "",
        when: "Only for experiments on Qwen models; normally follow the KV cache precision row.",
        note: None,
        options: &[],
    },
    KnobInfo {
        key: "dflash_mode",
        title: "DFlash eligibility",
        summary: "Eligibility policy for DFlash speculative decoding: a paired DFlash draft the target verifies in batches. DFlash needs its own draft; it never falls back to the MTP head.",
        effect: "Big decode speedup on predictable output (code, tool-calls, structured: ~4-5x). On free-form prose the acceptance rate is ~1, so it is a slight loss. Correctness is unchanged (the target verifies every token).",
        default: "off",
        when: "Turn on for code/agentic workloads when the model has a DFlash draft; `on` fails the load without one, even when an MTP head is present. Use the Speculation row (`speculation = mtp`) or mtp_mode to request the MTP head. Leave off for open-ended chat.",
        note: Some("A thinking BUDGET routes a DFlash draft to plain decode: the dispatch gate bails when max_think_tokens > 0 and the think block isn't closed (the default thinking-on path sets a budget). Turn thinking off — or set its budget to 0 — and use greedy to actually get spec-decode."),
        options: &[
            ("off", "No DFlash speculative decoding (plain autoregressive)."),
            ("on", "Use the DFlash draft whenever one is attached + the gate allows; fails the load without a draft."),
            ("auto", "Engage DFlash when eligible (greedy, thinking-off, arch)."),
        ],
    },
    KnobInfo {
        key: "speculation",
        title: "Speculation",
        summary: "The general speculative-decoding selector (speculation.mode): which mechanism drafts tokens the target then verifies.",
        effect: "auto picks by availability and eligibility (DFlash draft, native MTP head, n-gram, DSpark sidecar); the other values force one mechanism, which refuses the load when its own prerequisite is missing. Output is unchanged; only decode speed differs.",
        default: "auto",
        when: "Leave on auto. Force a mechanism to A/B-test it, or `off` for plain autoregressive decode.",
        note: Some("Forcing `dflash` needs a DFlash draft and never aliases to the MTP head; use `mtp` for the head. The legacy dflash_mode / mtp_mode rows still filter each mechanism under auto."),
        options: &[
            ("off", "Plain autoregressive decode, no speculation."),
            ("auto", "Cascade by availability and eligibility (the default)."),
            ("ngram", "Prompt n-gram drafting; needs no draft model."),
            ("dflash", "Force the model's DFlash draft; refuses the load without one."),
            ("mtp", "Force the model's native MTP head."),
            ("dspark", "Force the DSpark sidecar drafter."),
        ],
    },
    KnobInfo {
        key: "ple_format",
        title: "PLE row format",
        summary: "Which per-layer-embedding (PLE) row table a Flash-Next (Qwen4) load reads.",
        effect: "q4 reads the 4-bit Lloyd-Max rows (much smaller on disk and in RAM, tiny quality cost); q8 reads the exact Q8 rows. Only affects PLE-split Qwen4 models; other models ignore it.",
        default: "auto",
        when: "Leave on auto (q4 on gfx1151 unified-memory Strix Halo, q8 elsewhere). Force q4 or q8 to pin the table.",
        note: Some("A load never derives or downloads a missing table: q4 needs <model>.ple-q4 (hipfire pull, or `hipfire ple derive --format q4`). auto uses whichever format is on disk when only one exists."),
        options: &[
            ("auto", "q4 on gfx1151, q8 elsewhere; uses the other format when only it is on disk."),
            ("q4", "4-bit Lloyd-Max rows (<model>.ple-q4); the load refuses when missing."),
            ("q8", "Exact Q8 rows (embedded, or the <model>.ple-q8 sidecar)."),
        ],
    },
    KnobInfo {
        key: "thinking",
        title: "Thinking (reasoning block)",
        summary: "Whether reasoning models emit a hidden <think> block before the answer.",
        effect: "On = better reasoning on hard prompts, but more tokens/latency and it suppresses spec-decode. Off = faster, leaner, spec-decode eligible.",
        default: "on",
        when: "On for hard reasoning/math; off for throughput, code, or to enable DFlash.",
        note: Some("Interacts with dflash_mode: an active thinking budget (max_think_tokens > 0) makes DFlash fall back to plain decode. Turn thinking off to use spec-decode."),
        options: &[
            ("on", "Emit a hidden think block — stronger reasoning, more tokens, no spec-decode."),
            ("off", "No think block — faster, leaner, spec-decode eligible."),
        ],
    },
    KnobInfo {
        key: "reasoning_effort",
        title: "Reasoning effort",
        summary: "How hard the model is told to think, in its own vocabulary.",
        effect: "Shapes the reasoning up front by setting the model's native effort level, rather than truncating it after the fact. On Qwen3.8 this injects the card's effort instruction; medium is deliberately the unsteered baseline (no instruction at all).",
        default: "auto",
        when: "Prefer this over thinking_budget on any model that has it — lowering effort actually shortens the reasoning, where lowering the budget just cuts it off.",
        note: Some("The offered levels come from the MODEL, probed from its chat template at load, so they differ per model: Qwen3.8 has low/medium/xhigh and rejects high/max outright. A level the model lacks is projected UP to its next real one (high -> xhigh), never down. When a model has no native effort, this falls back to the thinking_budget ladder."),
        options: &[
            ("auto", "Leave the template's own default in force (Qwen3.8: xhigh)."),
            ("none", "No thinking — closed think block; see the thinking knob."),
            ("low", "Brief, focused reasoning; straight to the conclusion."),
            ("medium", "The model's unsteered baseline — no effort instruction."),
            ("high", "Projected up to the model's next real rung if it has no 'high'."),
            ("xhigh", "Thorough analysis: validate assumptions, weigh alternatives."),
            ("max", "Saturates at the model's top rung."),
        ],
    },
    KnobInfo {
        key: "thinking_budget",
        title: "Reasoning budget",
        summary: "Named per-turn cap on tokens the model may spend inside <think>.",
        effect: "Higher budgets allow deeper reasoning at the cost of latency; lower budgets force the model to commit to an answer sooner. Picks a preset instead of a raw token count.",
        default: "med",
        when: "med for everyday use; low for snappy chat; high/xhigh/max for hard reasoning; uncapped only if latency is no concern.",
        note: Some("A CAP, not a reasoning dial — it truncates, it does not shape. On a model with a native reasoning.effort (Qwen3.8, DeepSeek4) set the effort instead: a budget below what the chosen effort spends force-closes <think> mid-thought, so you pay the deep reasoning and lose the conclusion. Effort-native models therefore leave this uncapped by default. Drives max_think_tokens unless a raw max_think_tokens override is set (the advanced knob wins). Only applies when thinking is on; thinking=off caps reasoning to a single token regardless."),
        options: &[
            ("low", "512 tokens — quick answers, minimal reasoning."),
            ("med", "2048 tokens — the default; balanced reasoning vs latency."),
            ("high", "8192 tokens — deeper reasoning for harder prompts."),
            ("xhigh", "24576 tokens — very deep reasoning; watch latency."),
            ("max", "32768 tokens — the ceiling; longest bounded reasoning."),
            ("uncapped", "0 = no limit — model reasons until it stops (watch latency)."),
        ],
    },
    KnobInfo {
        key: "mtp_mode",
        title: "Multi-token prediction (MTP)",
        summary: "Uses a model's MTP head to predict several tokens per step where its producer has a measured autoregressive floor.",
        effect: "Raises decode throughput on models whose MTP drafter declares a measured AR floor (native Flash-Next) when the head is usable at the loaded placement; Qwen3.5/3.6 and DeepSeek V4 keep plain AR under `auto` and load no head — use `on` (or --spec mtp) to opt in. A no-op on models without an MTP head.",
        default: "auto",
        when: "Leave on auto; force on/off only when A/B-testing throughput. `on` arms the head on a producer without a measured AR floor (Qwen3.5/3.6, DeepSeek V4) and overrides `auto`'s decline of a host-mapped routed-expert spill.",
        note: Some("`auto` declines a host-mapped routed-expert spill (memory.moe_expert_budget, either offload_exec seam): the verify re-reads or recomputes the experts, and a head that is unverified there has produced garbled windows (`qwen3.6-35b-a3b.mq6` — mtp windows at tau 0.04-0.13 while its single-row AR decodes coherently). The load prints one line and decodes AR; set on (or --spec mtp) to opt back in."),
        options: &[
            ("off", "No multi-token prediction."),
            ("on", "Force the MTP head where the model has one."),
            ("auto", "Use MTP only where the producer has a measured AR floor and the model is not host-spilled; else single-token."),
        ],
    },
    KnobInfo {
        key: "mtp_k",
        title: "MTP depth (K)",
        summary: "How many tokens the MTP head proposes per step.",
        effect: "Higher K can accept more per step (faster) but risks lower acceptance on hard text. Only used when MTP is active.",
        default: "auto",
        when: "Tune only when measuring MTP throughput; `auto` is 4 for Flash-Next on gfx1151 and 3 elsewhere.",
        note: None,
        options: &[],
    },
    KnobInfo {
        key: "flash_mode",
        title: "Flash attention mode",
        summary: "Controls the flash-attention path for the Q8 KV cache.",
        effect: "auto picks per arch; always/never force it. Mostly a performance knob; the asym KV modes are flash-only.",
        default: "auto",
        when: "Leave on auto unless debugging an attention path.",
        note: None,
        options: &[
            ("auto", "Engine decides per arch — flash on where it helps (already on by default on gfx11/gfx12)."),
            ("always", "Force flash attention on."),
            ("never", "Disable flash attention (asym modes are flash-only)."),
        ],
    },
    KnobInfo {
        key: "mmq_screen",
        title: "MMQ screening",
        summary: "Mixed-precision matmul screening for quantized GEMMs.",
        effect: "Detects Q8_1 outlier rows and falls back to WMMA. Arch-gated to gfx906 + RDNA3/3.5 (gfx1100-1103/1150-1152) and only fires when MMQ is active.",
        default: "auto",
        when: "Leave on auto unless profiling kernels.",
        note: None,
        options: &[
            ("off", "No outlier sweep — max speed, risks tool-call corruption (#87)."),
            ("on", "Force the per-weight outlier sweep."),
            ("auto", "Screen on the supported arches (gfx906 + RDNA3/3.5) when MMQ is active."),
        ],
    },
    KnobInfo {
        key: "dflash_adaptive_b",
        title: "DFlash adaptive batch",
        summary: "Lets DFlash adapt its draft verify-block width to recent acceptance.",
        effect: "Follows the trailing 8-cycle acceptance depth (τ̂+2 rows), shrunk only past 2k context where the fixed full block loses throughput. This is not output-identical: realized tokens can differ from fixed-B (observed even at temp 0) because a narrower verify window moves window boundaries; per-position sampling stays target-lossless.",
        default: "false",
        when: "Opt in for long-context DFlash sessions after measuring your workload; off keeps the fixed full block. Mutually exclusive with the retained PM4 verify route (PM4 loads keep the fixed B=16 shape).",
        note: Some("HIPFIRE_DFLASH_ADAPTIVE_B=0 forces the fixed block even when enabled. Short contexts always get the full block."),
        options: &[
            ("true", "Size the draft batch to recent acceptance (context-gated, seeded at full)."),
            ("false", "Fixed draft batch size (default)."),
        ],
    },
    KnobInfo {
        key: "prompt_normalize",
        title: "Prompt normalization",
        summary: "Collapses 3+ consecutive newlines to 2 before tokenizing.",
        effect: "Stabilizes tokenization so whitespace differences don't swing performance; no meaning change for normal text.",
        default: "true",
        when: "Leave on. Turn off only when raw repeated newlines are semantically load-bearing.",
        note: None,
        options: &[
            ("true", "Collapse 3+ newlines to 2 before tokenizing."),
            ("false", "Leave whitespace as-is (when newlines are load-bearing)."),
        ],
    },
    KnobInfo {
        key: "default_chatml",
        title: "Default ChatML",
        summary: "Whether to apply a ChatML scaffold when a model has no embedded chat template.",
        effect: "Improves chat formatting for templateless models; harmless when a model carries its own template.",
        default: "true",
        when: "Leave on; turn off only for raw/base models where a ChatML scaffold hurts.",
        note: None,
        options: &[
            ("true", "Use a ChatML scaffold as the fallback when a model has no embedded template."),
            ("false", "No fallback scaffold — a templateless model gets none (models with their own template are unaffected either way)."),
        ],
    },
    KnobInfo {
        key: "chat_template",
        title: "Chat template",
        summary: "Path to a custom Jinja chat template (.j2/.jinja). Empty = model default.",
        effect: "Overrides how messages are rendered into the prompt. Wrong templates degrade output.",
        default: "",
        when: "Set only when you need a specific template; leave empty to use the model's own.",
        note: Some("Must point to a file that exists; an invalid path is rejected on save."),
        options: &[],
    },
    KnobInfo {
        key: "temperature",
        title: "Temperature",
        summary: "Sampling randomness for generation.",
        effect: "Lower = more deterministic/repeatable; higher = more varied/creative but less reliable.",
        default: "0.3",
        when: "Low (<=0.3) for code/factual; higher (0.7-1.0) for brainstorming.",
        note: Some("Greedy (~0) is what spec-decode runs at; sampling adds cost."),
        options: &[],
    },
    KnobInfo {
        key: "top_p",
        title: "Top-p (nucleus)",
        summary: "Caps sampling to the smallest set of tokens whose probability sums to p.",
        effect: "Lower = safer, more focused; higher = more diverse. Interacts with temperature.",
        default: "0.8",
        when: "Leave near default; lower it to tighten outputs.",
        note: None,
        options: &[],
    },
    KnobInfo {
        key: "host",
        title: "Serve endpoint",
        summary: "The bind address (and port) of the OpenAI-compatible serve API.",
        effect: "127.0.0.1 is local-only (the default); 0.0.0.0 listens on all interfaces. Chat and API clients connect here.",
        // Must match `serve.host`'s config default — the serve surface is
        // unauthenticated, so a LAN bind is an explicit operator choice.
        // `defaults_match_config_defaults` pins this equality.
        default: "127.0.0.1",
        when: "Set 0.0.0.0 to expose serve on the network; the default keeps it private to this machine.",
        note: Some("Paired with port (default 11435). Changes take effect on the next serve/restart."),
        options: &[],
    },
    KnobInfo {
        key: "max_tokens",
        title: "Max tokens",
        summary: "Upper bound on tokens generated per response.",
        effect: "Caps response length and runaway generation. No quality effect within the cap.",
        default: "4096",
        when: "Raise for long outputs; lower to bound latency/cost.",
        note: None,
        options: &[],
    },
];

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
    /// Explainer line 1: what it is.
    pub what: &'static str,
    /// Explainer line 2: the trade-off.
    pub tradeoff: Tradeoff,
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
        what,
        tradeoff: Tradeoff::Text(tradeoff),
    }
}

const AUTO_ON_OFF_OPTIONS: &[(&str, &str)] = &[("auto", "auto"), ("on", "on"), ("off", "off")];
const VISION_MODE_OPTIONS: &[(&str, &str)] = &[
    ("auto", "auto"),
    ("resident", "resident"),
    ("off", "off"),
    ("on", "on"),
];

/// Every §3.1 curated row, in group order.
pub const CURATED: &[SettingKnob] = &[
    // Quality
    SettingKnob {
        id: PROMPT_PROCESSING_ID,
        title: "Prompt processing",
        options: &[("fast", "Fast"), ("accurate", "Accurate")],
        what: "How the model reads your prompt before it starts to answer.",
        tradeoff: Tradeoff::Evidence {
            line: hipfire_config::evidence::PROMPT_PROCESSING,
            side: "Accurate",
            extra: None,
        },
    },
    SettingKnob {
        id: "developer.qwen4_moe_sym_iu4",
        title: "Flash-Next experts",
        options: &[("true", "Fast"), ("false", "Accurate")],
        what: "Math for the experts of Flash-Next models while reading prompts.",
        tradeoff: Tradeoff::Evidence {
            line: hipfire_config::evidence::FLASH_NEXT_EXPERTS,
            side: "Accurate",
            extra: None,
        },
    },
    SettingKnob {
        id: "memory.ple_format",
        title: "Flash-Next word table",
        options: &[("auto", "auto"), ("q4", "q4 (small)"), ("q8", "q8 (exact)")],
        what: "Which word-table rows Flash-Next loads: small 4-bit or exact 8-bit.",
        tradeoff: Tradeoff::Evidence {
            line: hipfire_config::evidence::FLASH_NEXT_PLE,
            side: "q8",
            extra: Some(hipfire_config::evidence::FLASH_NEXT_PLE_DISK),
        },
    },
    text(
        "reasoning.mode",
        "Thinking",
        &[("on", "on"), ("off", "off")],
        "Whether the model thinks before it answers.",
        "Off answers sooner, but hard questions may get worse answers.",
    ),
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
    ),
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
    ),
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
    ),
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
    ),
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
    ),
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
    ),
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
    ),
    text(
        "speculation.mtp",
        "MTP",
        AUTO_ON_OFF_OPTIONS,
        "Guesses ahead with the model's own multi-token head, when it has one.",
        "Faster typing when guesses land; answers are not changed.",
    ),
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
    ),
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
];

/// The curated entry for `id` (canonical key or [`PROMPT_PROCESSING_ID`]).
pub fn curated_knob(id: &str) -> Option<&'static SettingKnob> {
    CURATED.iter().find(|knob| knob.id == id)
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
    let schema = field(key);
    let legacy = schema.and_then(|schema| knob_info(schema.legacy_key));
    let mut text = match (legacy, schema) {
        (Some(info), _) => format!("{} {}", info.summary, info.effect),
        (None, Some(schema)) => schema.help.to_owned(),
        (None, None) => format!("{key} has no schema help."),
    };
    if schema.is_some_and(|schema| is_deprecated_key(schema.key)) {
        text = format!("Deprecated (removal 0.5.0), read-only here. {text}");
    }
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
        _ if schema.is_some_and(|schema| is_deprecated_key(schema.key)) => {
            "read-only; hipfire config set"
        }
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
        match knob.tradeoff {
            Tradeoff::Text(text) => text.to_owned(),
            Tradeoff::Evidence { line, side, extra } => evidence_text(line, side, extra, ctx.arch),
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

    #[test]
    fn lookup_returns_curated_entry() {
        let dflash = knob_info("dflash_mode").expect("dflash entry");
        assert_eq!(dflash.title, "DFlash eligibility");
        assert!(knob_info("does_not_exist").is_none());
    }

    #[test]
    fn dflash_explainer_does_not_promise_an_mtp_fallback() {
        // docs/CONFIG.md: `dflash_mode=on` requires a DFlash draft and never
        // enables MTP; the general selector is `speculation`.
        let dflash = knob_info("dflash_mode").expect("dflash entry");
        for field in [dflash.summary, dflash.when] {
            assert!(
                !field.contains("else its MTP head"),
                "dflash copy must not promise an MTP fallback: {field}"
            );
        }
        assert!(dflash.summary.contains("never falls back"));
        let spec = knob_info("speculation").expect("speculation entry");
        assert_eq!(spec.default, "auto");
        let values: Vec<&str> = spec.options.iter().map(|(v, _)| *v).collect();
        assert_eq!(values, hipfire_config::SPECULATION_MODES);
    }

    #[test]
    fn deprecated_keys_have_no_curated_entry() {
        // They fall back to the schema's own deprecation help text.
        for key in [
            "cask",
            "prefill_compression",
            "prefill_drafter",
            "prefill_threshold",
        ] {
            assert!(knob_info(key).is_none(), "{key}");
        }
    }

    #[test]
    fn kv_adaptive_has_explainer_with_options() {
        let k = knob_info("kv_adaptive").expect("kv_adaptive explainer");
        // 4 presets + the advanced-family entry.
        assert_eq!(k.options.len(), 5);
        assert!(k.options.iter().any(|(v, _)| *v == "aggressive"));
        assert!(k.options.iter().any(|(v, _)| v.starts_with("advanced")));
        assert!(
            k.note.unwrap().to_lowercase().contains("fwht"),
            "note explains the FWHT-K requirement"
        );
    }

    #[test]
    fn enum_and_bool_knobs_carry_per_option_help() {
        // Every editable enum/bool key's explainer lists what each value does, so
        // the option semantics show in the in-tab pane (no tab switching).
        use crate::hipfire::writer::FieldKind;
        for f in EDITABLE_FIELDS {
            if matches!(f.kind, FieldKind::Enum(_) | FieldKind::Bool) {
                let info = knob_info(f.key).unwrap_or_else(|| panic!("no explainer for {}", f.key));
                assert!(
                    !info.options.is_empty(),
                    "enum/bool key {} has no per-option help",
                    f.key
                );
            }
        }
    }

    #[test]
    fn every_editable_field_has_help() {
        // 5e: selecting any editable key must show an explainer.
        for f in EDITABLE_FIELDS {
            assert!(
                knob_info(f.key).is_some(),
                "no explainer for editable key {}",
                f.key
            );
        }
    }

    #[test]
    fn defaults_match_config_defaults() {
        // STRICT: every knob's `default` must exist in config::defaults() AND
        // match it — no silent skip. config::defaults() resolves the shared
        // hipfire-config schema, so this also pins knob defaults to the real
        // defaults. A knob without a config default fails loudly so it can't
        // drift unnoticed (this is exactly the gap that hid default_chatml).
        let cfg = crate::hipfire::config::ConfigState::defaults_for_test();
        for k in KNOBS {
            let expected = cfg.get(k.key).unwrap_or_else(|| {
                panic!(
                    "knob {} has no config default to cross-check — add it to config::defaults()",
                    k.key
                )
            });
            assert_eq!(
                &k.default, expected,
                "knob default for {} drifted from config defaults ({} vs {})",
                k.key, k.default, expected
            );
        }
    }

    #[test]
    fn load_bearing_interactions_are_documented() {
        // The two interactions this phase exists to surface.
        assert!(
            knob_info("dflash_mode")
                .unwrap()
                .note
                .unwrap()
                .contains("thinking"),
            "dflash note must mention the thinking interaction"
        );
        assert!(
            knob_info("thinking")
                .unwrap()
                .note
                .unwrap()
                .to_lowercase()
                .contains("dflash")
                || knob_info("thinking")
                    .unwrap()
                    .note
                    .unwrap()
                    .contains("spec-decode"),
            "thinking note must mention the dflash interaction"
        );
        assert!(
            knob_info("kv_cache")
                .unwrap()
                .note
                .unwrap()
                .contains("inherit"),
            "kv_cache note must explain auto = inherit"
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
            for knob in CURATED {
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
