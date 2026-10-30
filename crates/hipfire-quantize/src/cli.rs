// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Routed-expert `down_proj` codec override (`--moe-expert-down`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum MoeExpertDown {
    /// Symmetric MQ4G128V2 (qt53): row-local per-128 groups, 68 B each.
    #[value(name = "mq4g128v2-symmetric")]
    Mq4g128v2Symmetric,
}

#[derive(Debug, Parser)]
#[command(
    name = "hipfire-quantize",
    version,
    about = "Quantize Hugging Face safetensors (or deprecated GGUF weights) into Hipfire HFQ",
    after_help = "SUBCOMMANDS:\n  ple-hot-rank  Rank Qwen4 PLE rows over a corpus and write the <model>.plehot \
                  hot-row sidecar (run `hipfire-quantize ple-hot-rank --help`)\n  \
                  ple-hot-pack  Cut the <model>.plehotrows hot-row pack from a model and its \
                  .plehot sidecar (run `hipfire-quantize ple-hot-pack --help`)\n  \
                  trunk-overlay  Build the <model>.trunk-overlay QT44 overlay sidecar for a base \
                  model (run `hipfire-quantize trunk-overlay --help`)"
)]
pub(crate) struct QuantizeArgs {
    /// Hugging Face model directory or model ID. A GGUF weight file is
    /// deprecated (removal in 0.5.0): GGUF→mqN is lossy double quantization; use llama.cpp for GGUF. For
    /// `--qwen4-flash-next`, use a local directory/file or the immutable
    /// remote form `hf://OWNER/REPO@40_HEX_REVISION`; floating refs such as
    /// `main`, tags, and short revisions are rejected. Not used by
    /// `--flux-pipe`, which names its own input.
    #[arg(
        long,
        value_name = "PATH_OR_MODEL_ID",
        required_unless_present = "flux_pipe"
    )]
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — GGUF→mqN is lossy double quantization; use llama.cpp for GGUF (GGUF weight input only; imatrix.gguf stays)
    pub input: Option<String>,

    /// Produce the native Qwen4/Qwen3.8-Flash-Next streaming artifact.  This
    /// transactional path always includes typed PLE metadata, all PLE shards,
    /// and native MTP experts.  It reads no legacy recipe flags: imatrix, AWQ,
    /// symmetric-requant, tier and Hessian controls are refused, not ignored.
    #[arg(long, conflicts_with = "flux_pipe")]
    pub qwen4_flash_next: bool,
    /// Explicit non-production bounded fixture mode for exercising the full
    /// transactional writer with a compact local component. Production remains
    /// the default and keeps the pinned checkpoint admission counts.
    #[arg(long, value_name = "MODE", default_value = "production",
          value_parser = ["production", "compact-fixture"])]
    pub qwen4_component_mode: String,
    /// Select a dedicated recipe converter.  `stock-fp8` converts the stock
    /// Qwen3.8-Flash-Next FP8 checkpoint (a directory holding `fp8.sha256`)
    /// into an FP8-passthrough Qwen4 artifact plus `<output>.receipt.json`;
    /// it selects its own route, so `--qwen4-flash-next` is not given.
    /// `--qwen4-component-mode compact-fixture` is the only other flag read;
    /// legacy recipe flags, `--format` and the other routes are refused.
    #[arg(long, value_name = "RECIPE", value_parser = ["stock-fp8"],
          conflicts_with_all = ["qwen4_flash_next", "flux_pipe", "mq4v2_final_codes",
                                "mq3v2_final_codes", "mq2v2_final_codes", "reap_overlay"])]
    pub recipe: Option<String>,
    /// Developer-only (build feature `qwen4-dev-profiles`): produce a COMPLETE
    /// tiny local Qwen4 checkpoint from a local safetensors directory whose
    /// config.json carries a `hipfire_profile` developer marker.
    /// `source-exact` stores every tensor as its BF16 source bytes; `packed`
    /// quantizes each role to the format its runtime consumer supports.  The
    /// pinned Flash-Next production profile and `hf://` inputs are refused.
    /// Exclusive with `--qwen4-flash-next` and `--qwen4-component-mode`.
    #[cfg(feature = "qwen4-dev-profiles")]
    #[arg(long, value_name = "ENCODING",
          conflicts_with_all = ["qwen4_flash_next", "qwen4_component_mode", "flux_pipe", "recipe"],
          value_parser = ["source-exact", "packed"])]
    pub qwen4_dev_tiny: Option<String>,

    /// Destination HFQ file.
    #[arg(long, value_name = "PATH")]
    pub output: String,

    /// Quantization recipe or wire format.
    #[arg(long, default_value = "q8f16")]
    pub format: String,

    /// Rayon worker threads (defaults to 80% of available cores).
    #[arg(long, env = "HIPFIRE_QUANT_THREADS", value_name = "N")]
    pub threads: Option<usize>,

    /// `--format maple` only: carrier for `lm_head.weight`.
    ///
    /// Maple's head is dense over the full 151,936-row vocab and is read in its
    /// ENTIRETY every decoded token — 622 MB of BF16 at 205 GB/s, 90% of this
    /// box's achievable DRAM bandwidth and 36% of the decode token. It is the
    /// only bandwidth-bound part of the model, so this carrier is a decode-speed
    /// decision, not a fidelity one.
    ///
    /// Default is `q8` (331 MB). Measured on gfx1151 against a bf16 reference
    /// (2048 teacher-forced tokens): q8 and bf16 heads give the IDENTICAL mean
    /// KL of 0.0511, but q8 decodes 23% faster (144.6 vs 117.6 tok/s). A bf16
    /// head is therefore strictly dominated -- it costs throughput and buys
    /// exactly zero accuracy. mq4v2 (qt=44, 4.25 bpw, FWHT-rotated) is +10.4%
    /// decode over q8 but +51% mean KL and -2.7pp top-1, which is a poor trade
    /// on this stack.
    ///
    /// `mq4` (qt=30) is DEPRECATED and no longer selectable: mq4v2 (qt=44)
    /// beats it on every axis -- lower KL (0.0744 vs 0.0772), faster (165.8 vs
    /// 161.8 tok/s) and 15% smaller (4.25 vs 5.0 bpw). Existing .hfq files with
    /// a qt=30 head still LOAD; only producing new ones is removed.
    #[arg(long, value_name = "MODE", default_value = "q8",
          value_parser = ["bf16", "q8", "mq4v2", "q4k"])]
    pub head_quant: String,

    /// `--format maple` only: emit a HEAD-ONLY `.hfq` containing just
    /// `lm_head.weight` at `--head-quant`, instead of a full model.
    ///
    /// The result is a load-time overlay for a full build: same arch_id, same
    /// logical shape, differing only in the head's quant tier. Shipping heads
    /// this way avoids duplicating the identical 6.17 GB body per carrier —
    /// three head variants cost 7.30 GB rather than 19.63 GB, and switching
    /// heads is a 175 MB download instead of 6.5 GB.
    #[arg(long, default_value_t = false)]
    pub head_only: bool,

    /// Override the architecture ID stamped into the HFQ header.
    #[arg(long, value_name = "ID")]
    pub arch_id: Option<u32>,

    /// Allow an architecture override to move Qwen3 off its pillar IDs.
    #[arg(long)]
    pub force_arch_id: bool,

    /// Pack a FLUX.1 or FLUX.2 Klein diffusers pipe into per-component HFQ files instead of
    /// running the quantize pipeline (`--format` is ignored on this path).
    /// The pipe is a dir root holding `transformer/`, `vae/`, `scheduler/`, the
    /// text encoder dirs and `tokenizer*/`. The packs are the only form the
    /// daemon loads; see docs/QUANTIZE.md.
    #[arg(long, value_name = "PIPE_DIR")]
    pub flux_pipe: Option<String>,

    /// `--flux-pipe` only: one component to pack, or `all` (default) to write
    /// every component derived from `--output`: `<stem>-transformer.hfq`,
    /// `<stem>-t5.hfq`, `<stem>-clip.hfq`, `<stem>-vae.hfq` for FLUX.1;
    /// `<stem>-transformer.hfq`, `<stem>-qwen3.hfq`, `<stem>-vae.hfq` for
    /// FLUX.2 Klein. A single component writes exactly to `--output`.
    #[arg(long, value_name = "COMPONENT", default_value = "all")]
    pub flux_component: String,

    /// Upstream URL recorded in the output's `hipfire_provenance`.
    #[arg(long, value_name = "URL")]
    pub source_url: Option<String>,

    /// SPDX license recorded in the output's `hipfire_provenance`.
    #[arg(long, value_name = "SPDX")]
    pub license: Option<String>,

    /// Emit only tensors selected by a REAP plan.
    #[arg(long, value_name = "PLAN_DIR", conflicts_with = "reap_bake")]
    pub reap_overlay: Option<String>,

    /// Apply a REAP plan while baking a complete model.
    #[arg(long, value_name = "PLAN_DIR", conflicts_with = "reap_overlay")]
    pub reap_bake: Option<String>,

    /// Output path for a REAP overlay or baked model.
    #[arg(long, value_name = "PATH")]
    pub reap_out: Option<String>,

    /// Architecture family used to interpret a REAP plan.
    #[arg(long, value_name = "ARCH")]
    pub reap_arch: Option<String>,

    /// llama.cpp imatrix GGUF used for activation-aware quantization.
    #[arg(long, value_name = "PATH")]
    pub imatrix: Option<PathBuf>,

    /// Per-tensor Hessian directory used by GPTQ-E8 recipes.
    #[arg(long, value_name = "DIR")]
    pub hessian_dir: Option<PathBuf>,

    /// Fraction of hot layers assigned the higher-precision Lloyd tier.
    #[arg(long, env = "HIPFIRE_TIER_RATIO", default_value_t = 0.30)]
    pub tier_ratio: f64,

    /// Force router tensors to Q8.
    #[arg(long)]
    pub q8_router: bool,

    /// Disable the default Q8 protection for conv1d tensors.
    #[arg(long)]
    pub no_q8_conv1d: bool,

    /// Let the FIXED tier (attention / lm_head / embed / router) follow
    /// `--format` instead of being pinned to Q8F16. The fixed tier is ~66% of
    /// per-token decode bytes on a3b, so pinning it at Q8 (1.0625 B/w) rather
    /// than MQ4 (0.53125) doubles the dominant term — this is why `.mq2` reads
    /// 45% MORE bytes/token than `.mq4r` despite being 7 GB smaller on disk.
    /// Required to reproduce `.mq4r`.
    #[arg(long)]
    pub no_q8_router: bool,

    /// Disable K-map precision promotion.
    #[arg(long)]
    pub no_kmap: bool,

    /// Alias for --no-kmap for uniform quantization.
    #[arg(long)]
    pub uniform: bool,

    /// Enable AWQ pre-scaling with the default alpha.
    #[arg(long)]
    pub awq: bool,

    /// Enable AWQ pre-scaling with an explicit alpha.
    #[arg(long, value_name = "ALPHA")]
    pub awq_alpha: Option<f32>,
    /// Correct Qwen3.8 linear-attention out_proj imatrix head order before
    /// fitting AWQ scales. Does not change the alpha or any other tensor.
    #[arg(long, requires = "imatrix")]
    pub awq_fix_la_head_order: bool,

    /// Reproduce the historical A4-aware alpha search (four A4 candidates,
    /// asymmetric W surrogate, positive imatrix-RMS activation).
    #[arg(long, conflicts_with = "awq_alpha")]
    pub awq_a4_aware: bool,

    /// Search shared AWQ alpha with the gfx1201 fused c2 A4 recipe and
    /// symmetric qt44 W writer for every eligible layer. Without captured
    /// rows, retains the legacy imatrix RMS activation statistic.
    #[arg(long, requires = "mq4v2_symmetric", conflicts_with = "awq_a4_aware")]
    pub awq_a4_route_c2: bool,
    /// Optionally replace the imatrix RMS statistic with signed QAT producer
    /// rows on captured sites; uncaptured sites still use c2/symmetric objective.
    #[arg(long, value_name = "CAPTURE_DIR", requires_all = ["awq_a4_route_c2", "awq_a4_source_sha"])]
    pub awq_a4_signed_capture: Option<PathBuf>,

    /// SHA-256 of the BF16 parent recorded in the signed QAT capture manifest.
    #[arg(long, value_name = "SHA256", requires = "awq_a4_signed_capture")]
    pub awq_a4_source_sha: Option<String>,

    /// Encode MQ4V2 with a per-128 symmetric grid while retaining the existing
    /// affine header layout. The stored zero is `-8*d`, so code 8 maps to zero.
    #[arg(long)]
    pub mq4v2_symmetric: bool,

    /// Encode MQ3V2 with a per-128 symmetric grid while retaining the existing
    /// affine header layout. The stored zero is `-4*d`, so code 4 maps to zero.
    #[arg(long)]
    pub mq3v2_symmetric: bool,

    /// Encode MQ2V2 with a per-128 symmetric grid while retaining the existing
    /// affine header layout. The stored zero is `-2*d`, so code 2 maps to zero.
    #[arg(long)]
    pub mq2v2_symmetric: bool,

    /// Routed-expert down_proj codec override. `mq4g128v2-symmetric` encodes
    /// every routed-expert down_proj (stacked `experts.down_proj` slices and
    /// un-stacked `mlp.experts.N.down_proj`) as symmetric MQ4G128V2 (qt53,
    /// 68 B per 128 weights) while gate_up and everything else keep the
    /// recipe's codec. Requires uniform symmetric MQ4: `--format mq4
    /// --uniform --mq4v2-symmetric`.
    #[arg(long, value_enum, value_name = "VALUE")]
    pub moe_expert_down: Option<MoeExpertDown>,

    /// Replace selected MQ4V2-XT tensors in an existing HFQ with trained final
    /// codes. PATH is one frozen-record `.safetensors` file or a directory of
    /// them; each record carries metadata `name,M,K,qt,source_sha` and tensors
    /// `S_f16`, `d_z_f16`, `codes_u8`. `--input` must be the source HFQ named
    /// by `source_sha`; unselected tensor/index/metadata bytes are copied intact.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["flux_pipe", "reap_overlay", "reap_bake"]
    )]
    pub mq4v2_final_codes: Option<PathBuf>,

    /// Replace only packed MQ3V2 code bytes in an existing HFQ using frozen-grid
    /// safetensors records. The source HFQ SHA, tensor shape, qt=49, and
    /// unchanged fp16 grid are checked before writing.
    #[arg(long, value_name = "DIR", conflicts_with_all = ["mq4v2_final_codes", "mq2v2_final_codes", "flux_pipe", "reap_overlay", "reap_bake"])]
    pub mq3v2_final_codes: Option<PathBuf>,

    /// Replace only packed MQ2V2 code bytes in an existing HFQ using frozen-grid
    /// safetensors records (qt=50).
    #[arg(long, value_name = "DIR", conflicts_with_all = ["mq4v2_final_codes", "mq3v2_final_codes", "flux_pipe", "reap_overlay", "reap_bake"])]
    pub mq2v2_final_codes: Option<PathBuf>,

    /// Enable K-map promotion for dense models.
    #[arg(long)]
    pub kmap_dense: bool,

    /// K-map policy: full, alternating/alt, typed, or typed-gemma4 (0-3).
    #[arg(long, default_value = "alternating", value_name = "MODE",
          value_parser = ["full", "alternating", "alt", "typed", "typed-gemma4", "0", "1", "2", "3"])]
    pub kmap_mode: String,

    /// Permit research-only uniform MQ2 output.
    #[arg(long)]
    pub allow_mq2: bool,

    /// Permit research-only MQ2-Lloyd output.
    #[arg(long)]
    pub allow_mq2_lloyd: bool,

    /// Permit research-only MQ3-Lloyd output.
    #[arg(long)]
    pub allow_mq3_lloyd: bool,

    /// Permit research-only MQ4-Lloyd output.
    #[arg(long)]
    pub allow_mq4_lloyd: bool,

    /// Include vision tensors that are skipped by default.
    #[arg(long)]
    pub include_vision: bool,

    /// Quantization recipe for included vision tensors.
    #[arg(long, default_value = "", value_name = "FORMAT")]
    pub vision_quant: String,

    /// Ingest only tensors whose names start with this prefix.
    #[arg(long, value_name = "PREFIX")]
    pub include_prefix: Option<String>,

    /// Vision-tower-only sidecar shorthand: `--include-vision` plus
    /// `--include-prefix model.visual.` (an explicit `--include-prefix`
    /// still wins). Emits the shared `qwen3.8-27b-vision.hfq` sidecar.
    #[arg(long)]
    pub vision_only: bool,

    /// Product tier for Qwen3.8 ladder: xt keeps lm_head at base codec, base lifts lm_head, pro also lifts ssm_out (linear_attn.out_proj).
    /// embed_tokens and linear_attn.conv1d.weight remain Q8 at every rung; structural tensors remain F16.
    #[arg(long, value_name = "TIER")]
    pub tier: Option<String>,

    /// Per-class fixed-tier codec overrides, e.g. lm_head:mq6v2,ssm_out:mq5v2.
    /// Accepted classes: lm_head,embed,router,attn,ssm_out; accepted dtypes: q8,mq2v2,mq3v2,mq4v2,mq5v2,mq6v2.
    #[arg(long, value_name = "SPEC")]
    pub fixed_tier: Option<String>,
}

/// Refuse an `--arch-id` override that strips a qwen3* model (auto-detected
/// arch 5 or 6) off the pillar arches, unless `--force-arch-id` is given.
/// Keeps the froggeric chat-template pillar + qwen35-crate dispatch intact by
/// construction. No-op for non-qwen models or overrides that stay in {5,6}.
pub(crate) fn guard_qwen3_arch_override(auto_arch_id: u32, arch_id: u32, force_arch_id: bool) {
    let auto_is_qwen3 = matches!(auto_arch_id, 5 | 6);
    let override_off_pillar = !matches!(arch_id, 5 | 6);
    if auto_is_qwen3 && arch_id != auto_arch_id && override_off_pillar && !force_arch_id {
        eprintln!(
            "error: --arch-id {arch_id} moves an auto-detected qwen3* model (arch {auto_arch_id}) \
             OFF the pillar arches {{5,6}}; this would break the froggeric chat-template pillar \
             and the qwen35-crate dispatch. Pass --force-arch-id to override anyway."
        );
        std::process::exit(1);
    }
}

/// The mutually exclusive paths `run()` dispatches to. Each reads a different
/// subset of the CLI; a flag set for a path that never reads it is an error
/// rather than a silent no-op (see [`reject_unreachable`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Route {
    /// `--flux-pipe` component packing.
    Flux,
    /// `--qwen4-flash-next` streaming artifact.
    Qwen4,
    /// `--recipe stock-fp8` stock FP8 passthrough converter.
    Qwen4StockFp8,
    /// `--qwen4-dev-tiny` complete developer tiny checkpoint (feature-gated).
    #[cfg(feature = "qwen4-dev-profiles")]
    Qwen4DevTiny,
    /// `--mq{4,3,2}v2-final-codes` import into an existing HFQ.
    FinalCodes,
    /// `--format maple|ds4-dense-e8soa-overlay|ds4-dspark-e8soa|qwen3-dspark-q8`.
    EarlySpecial,
    /// Deprecated GGUF weight input.
    Gguf,
    /// Raon-1-mini (`model_type: raon1_mini`) symmetric qt44 control artifact.
    Raon,
    /// `--reap-overlay` selective re-quant.
    ReapOverlay,
    /// The safetensors quantize pipeline (including `--reap-bake`).
    Main,
}

use Route::*;

#[cfg(not(feature = "qwen4-dev-profiles"))]
const EVERY: &[Route] = &[
    Flux,
    Qwen4,
    Qwen4StockFp8,
    FinalCodes,
    EarlySpecial,
    Gguf,
    Raon,
    ReapOverlay,
    Main,
];
#[cfg(feature = "qwen4-dev-profiles")]
const EVERY: &[Route] = &[
    Flux,
    Qwen4,
    Qwen4StockFp8,
    Qwen4DevTiny,
    FinalCodes,
    EarlySpecial,
    Gguf,
    Raon,
    ReapOverlay,
    Main,
];

/// Clap argument id → the routes that read it. Every argument of
/// [`QuantizeArgs`] must appear (enforced by a test).
pub(crate) const FLAG_ROUTES: &[(&str, &[Route])] = &[
    #[cfg(not(feature = "qwen4-dev-profiles"))]
    (
        "input",
        &[
            Qwen4,
            Qwen4StockFp8,
            FinalCodes,
            EarlySpecial,
            Gguf,
            Raon,
            ReapOverlay,
            Main,
        ],
    ),
    #[cfg(feature = "qwen4-dev-profiles")]
    (
        "input",
        &[
            Qwen4,
            Qwen4StockFp8,
            Qwen4DevTiny,
            FinalCodes,
            EarlySpecial,
            Gguf,
            Raon,
            ReapOverlay,
            Main,
        ],
    ),
    ("output", EVERY),
    ("threads", EVERY),
    ("qwen4_flash_next", &[Qwen4]),
    ("qwen4_component_mode", &[Qwen4, Qwen4StockFp8]),
    ("recipe", &[Qwen4StockFp8]),
    #[cfg(feature = "qwen4-dev-profiles")]
    ("qwen4_dev_tiny", &[Qwen4DevTiny]),
    ("format", &[EarlySpecial, Gguf, Raon, Main]),
    ("head_quant", &[EarlySpecial]),
    ("head_only", &[EarlySpecial]),
    ("arch_id", &[Gguf, ReapOverlay, Main]),
    ("force_arch_id", &[Gguf, ReapOverlay, Main]),
    ("flux_pipe", &[Flux]),
    ("flux_component", &[Flux]),
    ("source_url", &[ReapOverlay, Main]),
    ("license", &[ReapOverlay, Main]),
    ("reap_overlay", &[ReapOverlay]),
    ("reap_bake", &[Main]),
    ("reap_out", &[ReapOverlay, Main]),
    ("reap_arch", &[ReapOverlay, Main]),
    // The Qwen4 producers read none of the legacy recipe controls: their
    // quantization recipe is fixed by the artifact profile.  Do not list a
    // flag for a Qwen4 route until `crate::qwen4` actually reads it.
    ("imatrix", &[Main]),
    ("awq_alpha", &[Main]),
    ("mq4v2_symmetric", &[Raon, Main]),
    ("hessian_dir", &[Main]),
    ("tier_ratio", &[Main]),
    ("q8_router", &[Main]),
    ("no_q8_conv1d", &[Main]),
    ("no_q8_router", &[Main]),
    ("no_kmap", &[Gguf, Main]),
    ("uniform", &[Gguf, Main]),
    ("awq", &[Main]),
    ("awq_fix_la_head_order", &[Main]),
    ("awq_a4_aware", &[Main]),
    ("awq_a4_route_c2", &[Main]),
    ("awq_a4_signed_capture", &[Main]),
    ("awq_a4_source_sha", &[Main]),
    ("mq3v2_symmetric", &[Main]),
    ("mq2v2_symmetric", &[Main]),
    ("moe_expert_down", &[Main]),
    ("mq4v2_final_codes", &[FinalCodes]),
    ("mq3v2_final_codes", &[FinalCodes]),
    ("mq2v2_final_codes", &[FinalCodes]),
    ("kmap_dense", &[Gguf, Main]),
    ("kmap_mode", &[Gguf, Main]),
    ("allow_mq2", &[Gguf, Main]),
    ("allow_mq2_lloyd", &[Gguf, Main]),
    ("allow_mq3_lloyd", &[Gguf, Main]),
    ("allow_mq4_lloyd", &[Gguf, Main]),
    ("include_vision", &[Main]),
    ("vision_quant", &[Main]),
    ("include_prefix", &[Main]),
    ("vision_only", &[Main]),
    ("tier", &[Gguf, Raon, Main]),
    ("fixed_tier", &[Gguf, Main]),
];

fn route_label(route: Route) -> &'static str {
    match route {
        Flux => "--flux-pipe",
        Qwen4 => "--qwen4-flash-next",
        Qwen4StockFp8 => "--recipe stock-fp8",
        #[cfg(feature = "qwen4-dev-profiles")]
        Qwen4DevTiny => "--qwen4-dev-tiny",
        FinalCodes => "--mq*v2-final-codes",
        EarlySpecial => "this --format",
        Gguf => "GGUF weight input",
        Raon => "the Raon-1-mini converter",
        ReapOverlay => "--reap-overlay",
        Main => "the safetensors quantize path",
    }
}

/// Err for the first flag given on the command line (clap defaults and
/// environment values do not count) that `route` never reads.
pub(crate) fn reject_unreachable(matches: &clap::ArgMatches, route: Route) -> Result<(), String> {
    for (id, routes) in FLAG_ROUTES {
        if routes.contains(&route) {
            continue;
        }
        if matches.value_source(id) == Some(clap::parser::ValueSource::CommandLine) {
            return Err(format!(
                "--{} has no effect with {}; remove it",
                id.replace('_', "-"),
                route_label(route)
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod route_tests {
    use super::*;
    use clap::CommandFactory;

    fn matches(argv: &[&str]) -> clap::ArgMatches {
        QuantizeArgs::command()
            .try_get_matches_from(std::iter::once("hipfire-quantize").chain(argv.iter().copied()))
            .unwrap()
    }

    #[test]
    fn every_cli_flag_has_a_route() {
        for arg in QuantizeArgs::command().get_arguments() {
            let id = arg.get_id().as_str();
            if id == "help" || id == "version" {
                continue;
            }
            assert!(
                FLAG_ROUTES.iter().any(|(k, _)| *k == id),
                "--{id} has no FLAG_ROUTES entry; declare which routes read it"
            );
        }
        for (id, _) in FLAG_ROUTES {
            assert!(
                QuantizeArgs::command()
                    .get_arguments()
                    .any(|a| a.get_id() == *id),
                "FLAG_ROUTES names unknown argument {id}"
            );
        }
    }

    #[test]
    fn each_route_rejects_a_flag_it_never_reads() {
        let cases: &[(Route, &[&str], &str)] = &[
            (
                Flux,
                &["--flux-pipe", "p", "--output", "o", "--format", "mq4"],
                "--format",
            ),
            (
                Qwen4,
                &[
                    "--qwen4-flash-next",
                    "--input",
                    "i",
                    "--output",
                    "o",
                    "--tier",
                    "xt",
                ],
                "--tier",
            ),
            (
                FinalCodes,
                &[
                    "--mq4v2-final-codes",
                    "r",
                    "--input",
                    "i",
                    "--output",
                    "o",
                    "--awq",
                ],
                "--awq",
            ),
            (
                EarlySpecial,
                &[
                    "--format",
                    "maple",
                    "--input",
                    "i",
                    "--output",
                    "o",
                    "--mq4v2-symmetric",
                ],
                "--mq4v2-symmetric",
            ),
            (
                Gguf,
                &["--input", "m.gguf", "--output", "o", "--imatrix", "x"],
                "--imatrix",
            ),
            (
                ReapOverlay,
                &[
                    "--reap-overlay",
                    "p",
                    "--input",
                    "i",
                    "--output",
                    "o",
                    "--format",
                    "mq4",
                ],
                "--format",
            ),
            (
                Main,
                &["--input", "i", "--output", "o", "--head-only"],
                "--head-only",
            ),
        ];
        for (route, argv, flag) in cases {
            let err = reject_unreachable(&matches(argv), *route).unwrap_err();
            assert!(
                err.starts_with(&format!("{flag} has no effect")),
                "{route:?}: {err}"
            );
        }
    }

    #[test]
    fn moe_expert_down_parses_only_its_value_and_is_main_only() {
        let parse = |value: &str| {
            QuantizeArgs::command().try_get_matches_from([
                "hipfire-quantize",
                "--input",
                "i",
                "--output",
                "o",
                "--moe-expert-down",
                value,
            ])
        };
        let parsed = parse("mq4g128v2-symmetric").unwrap();
        assert_eq!(
            parsed.get_one::<MoeExpertDown>("moe_expert_down"),
            Some(&MoeExpertDown::Mq4g128v2Symmetric)
        );
        for bad in [
            "mq4g128v2",
            "mq4g128v2-asymmetric",
            "mq4v2-symmetric",
            "",
            "MQ4G128V2-SYMMETRIC",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} must not parse");
        }
        assert!(QuantizeArgs::command()
            .try_get_matches_from([
                "hipfire-quantize",
                "--input",
                "i",
                "--output",
                "o",
                "--moe-expert-down"
            ])
            .is_err());
        let argv = [
            "--input",
            "i",
            "--output",
            "o",
            "--moe-expert-down",
            "mq4g128v2-symmetric",
        ];
        reject_unreachable(&matches(&argv), Main).unwrap();
        // Flux reads `--flux-pipe` instead of `--input` (and refuses `--input`
        // first), so give each route its own minimal source argument.
        for route in EVERY.iter().copied().filter(|route| *route != Main) {
            let source: &[&str] = if route == Flux {
                &["--flux-pipe", "p"]
            } else {
                &["--input", "i"]
            };
            let mut probe = source.to_vec();
            probe.extend(["--output", "o", "--moe-expert-down", "mq4g128v2-symmetric"]);
            let err = reject_unreachable(&matches(&probe), route).unwrap_err();
            assert!(
                err.starts_with("--moe-expert-down has no effect"),
                "{route:?}: {err}"
            );
        }
    }

    #[test]
    fn stock_fp8_recipe_selects_its_route_and_refuses_what_it_never_reads() {
        let base = ["--recipe", "stock-fp8", "--input", "i", "--output", "o"];
        reject_unreachable(&matches(&base), Qwen4StockFp8).unwrap();
        let mut fixture = base.to_vec();
        fixture.extend(["--qwen4-component-mode", "compact-fixture"]);
        reject_unreachable(&matches(&fixture), Qwen4StockFp8).unwrap();
        for (flag, value) in [
            ("--format", Some("mq4")),
            ("--imatrix", Some("x")),
            ("--awq-alpha", Some("0.5")),
            ("--tier", Some("xt")),
            ("--hessian-dir", Some("d")),
            ("--awq", None),
            ("--mq4v2-symmetric", None),
        ] {
            let mut argv = base.to_vec();
            argv.push(flag);
            argv.extend(value);
            let err = reject_unreachable(&matches(&argv), Qwen4StockFp8).unwrap_err();
            assert!(
                err.starts_with(&format!("{flag} has no effect")),
                "{flag}: {err}"
            );
        }
        // No other route reads --recipe.
        for route in [Qwen4, Main, Gguf, EarlySpecial] {
            let err = reject_unreachable(&matches(&base), route).unwrap_err();
            assert!(
                err.starts_with("--recipe has no effect"),
                "{route:?}: {err}"
            );
        }
    }

    #[test]
    fn stock_fp8_recipe_is_exclusive_with_other_routes() {
        for extra in [
            &["--qwen4-flash-next"][..],
            &["--flux-pipe", "p"][..],
            &["--reap-overlay", "p"][..],
            &["--mq4v2-final-codes", "r"][..],
        ] {
            let mut argv = vec!["--recipe", "stock-fp8", "--input", "i", "--output", "o"];
            argv.extend_from_slice(extra);
            assert!(
                QuantizeArgs::command()
                    .try_get_matches_from(std::iter::once("hipfire-quantize").chain(argv))
                    .is_err(),
                "{extra:?}"
            );
        }
        assert!(QuantizeArgs::command()
            .try_get_matches_from([
                "hipfire-quantize",
                "--recipe",
                "other",
                "--input",
                "i",
                "--output",
                "o"
            ])
            .is_err());
    }

    #[test]
    fn defaults_and_readable_flags_pass() {
        // `--format` has a default and `--kmap-mode` too: unset ⇒ not rejected.
        reject_unreachable(&matches(&["--flux-pipe", "p", "--output", "o"]), Flux).unwrap();
        reject_unreachable(
            &matches(&[
                "--input",
                "i",
                "--output",
                "o",
                "--format",
                "mq4",
                "--awq",
                "--imatrix",
                "x",
            ]),
            Main,
        )
        .unwrap();
    }

    #[test]
    fn qwen4_refuses_legacy_recipe_flags_it_never_reads() {
        for (flag, value) in [("--imatrix", "x"), ("--awq-alpha", "0.5")] {
            let argv = [
                "--qwen4-flash-next",
                "--input",
                "i",
                "--output",
                "o",
                flag,
                value,
            ];
            let err = reject_unreachable(&matches(&argv), Qwen4).unwrap_err();
            assert!(
                err.starts_with(&format!("{flag} has no effect")),
                "{flag}: {err}"
            );
        }
        for flag in ["--mq4v2-symmetric", "--awq", "--hessian-dir"] {
            let argv: &[&str] = if flag == "--hessian-dir" {
                &[
                    "--qwen4-flash-next",
                    "--input",
                    "i",
                    "--output",
                    "o",
                    flag,
                    "d",
                ]
            } else {
                &["--qwen4-flash-next", "--input", "i", "--output", "o", flag]
            };
            let err = reject_unreachable(&matches(argv), Qwen4).unwrap_err();
            assert!(
                err.starts_with(&format!("{flag} has no effect")),
                "{flag}: {err}"
            );
        }
        reject_unreachable(
            &matches(&[
                "--qwen4-flash-next",
                "--input",
                "i",
                "--output",
                "o",
                "--qwen4-component-mode",
                "compact-fixture",
            ]),
            Qwen4,
        )
        .unwrap();
    }

    #[cfg(feature = "qwen4-dev-profiles")]
    #[test]
    fn qwen4_dev_tiny_reads_only_its_encoding() {
        for encoding in ["source-exact", "packed"] {
            let argv = [
                "--qwen4-dev-tiny",
                encoding,
                "--input",
                "i",
                "--output",
                "o",
            ];
            reject_unreachable(&matches(&argv), Qwen4DevTiny).unwrap();
            let argv = [
                "--qwen4-dev-tiny",
                encoding,
                "--input",
                "i",
                "--output",
                "o",
                "--imatrix",
                "x",
            ];
            let err = reject_unreachable(&matches(&argv), Qwen4DevTiny).unwrap_err();
            assert!(err.starts_with("--imatrix has no effect"), "{err}");
        }
        // FN-only and recipe controls never ride along with the developer route.
        let argv = [
            "--qwen4-dev-tiny",
            "packed",
            "--input",
            "i",
            "--output",
            "o",
            "--tier",
            "xt",
        ];
        let err = reject_unreachable(&matches(&argv), Qwen4DevTiny).unwrap_err();
        assert!(err.starts_with("--tier has no effect"), "{err}");
        // The encoding is meaningless for the FN route.
        let argv = ["--qwen4-flash-next", "--input", "i", "--output", "o"];
        reject_unreachable(&matches(&argv), Qwen4).unwrap();
        let err = reject_unreachable(
            &matches(&[
                "--qwen4-dev-tiny",
                "packed",
                "--input",
                "i",
                "--output",
                "o",
            ]),
            Qwen4,
        )
        .unwrap_err();
        assert!(err.starts_with("--qwen4-dev-tiny has no effect"), "{err}");
    }

    #[cfg(feature = "qwen4-dev-profiles")]
    #[test]
    fn qwen4_dev_tiny_is_exclusive_with_fn_route_flags() {
        for extra in [
            &["--qwen4-flash-next"][..],
            &["--qwen4-component-mode", "compact-fixture"][..],
        ] {
            let mut argv = vec![
                "--qwen4-dev-tiny",
                "packed",
                "--input",
                "i",
                "--output",
                "o",
            ];
            argv.extend_from_slice(extra);
            assert!(
                QuantizeArgs::command()
                    .try_get_matches_from(std::iter::once("hipfire-quantize").chain(argv))
                    .is_err(),
                "{extra:?}"
            );
        }
    }
}
