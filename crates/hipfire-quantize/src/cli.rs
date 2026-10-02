// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "hipfire-quantize",
    version,
    about = "Quantize Hugging Face safetensors (or deprecated GGUF weights) into Hipfire HFQ"
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
    /// and native MTP experts; legacy recipe flags are ignored.
    #[arg(long, conflicts_with = "flux_pipe")]
    pub qwen4_flash_next: bool,
    /// Explicit non-production bounded fixture mode for exercising the full
    /// transactional writer with a compact local component. Production remains
    /// the default and keeps the pinned checkpoint admission counts.
    #[arg(long, value_name = "MODE", default_value = "production",
          value_parser = ["production", "compact-fixture"])]
    pub qwen4_component_mode: String,

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
    /// `--mq{4,3,2}v2-final-codes` import into an existing HFQ.
    FinalCodes,
    /// `--format maple|ds4-dense-e8soa-overlay|ds4-dspark-e8soa|qwen3-dspark-q8`.
    EarlySpecial,
    /// Deprecated GGUF weight input.
    Gguf,
    /// `--reap-overlay` selective re-quant.
    ReapOverlay,
    /// The safetensors quantize pipeline (including `--reap-bake`).
    Main,
}

use Route::*;

const EVERY: &[Route] = &[Flux, Qwen4, FinalCodes, EarlySpecial, Gguf, ReapOverlay, Main];

/// Clap argument id → the routes that read it. Every argument of
/// [`QuantizeArgs`] must appear (enforced by a test).
pub(crate) const FLAG_ROUTES: &[(&str, &[Route])] = &[
    ("input", &[Qwen4, FinalCodes, EarlySpecial, Gguf, ReapOverlay, Main]),
    ("output", EVERY),
    ("threads", EVERY),
    ("qwen4_flash_next", &[Qwen4]),
    ("qwen4_component_mode", &[Qwen4]),
    ("format", &[EarlySpecial, Gguf, Main]),
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
    // Qwen4 reads these once the symmetric requant lands on that path.
    ("imatrix", &[Qwen4, Main]),
    ("awq_alpha", &[Qwen4, Main]),
    ("mq4v2_symmetric", &[Qwen4, Main]),
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
    ("tier", &[Gguf, Main]),
    ("fixed_tier", &[Gguf, Main]),
];

fn route_label(route: Route) -> &'static str {
    match route {
        Flux => "--flux-pipe",
        Qwen4 => "--qwen4-flash-next",
        FinalCodes => "--mq*v2-final-codes",
        EarlySpecial => "this --format",
        Gguf => "GGUF weight input",
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
                QuantizeArgs::command().get_arguments().any(|a| a.get_id() == *id),
                "FLAG_ROUTES names unknown argument {id}"
            );
        }
    }

    #[test]
    fn each_route_rejects_a_flag_it_never_reads() {
        let cases: &[(Route, &[&str], &str)] = &[
            (Flux, &["--flux-pipe", "p", "--output", "o", "--format", "mq4"], "--format"),
            (Qwen4, &["--qwen4-flash-next", "--input", "i", "--output", "o", "--tier", "xt"], "--tier"),
            (FinalCodes, &["--mq4v2-final-codes", "r", "--input", "i", "--output", "o", "--awq"], "--awq"),
            (EarlySpecial, &["--format", "maple", "--input", "i", "--output", "o", "--mq4v2-symmetric"], "--mq4v2-symmetric"),
            (Gguf, &["--input", "m.gguf", "--output", "o", "--imatrix", "x"], "--imatrix"),
            (ReapOverlay, &["--reap-overlay", "p", "--input", "i", "--output", "o", "--format", "mq4"], "--format"),
            (Main, &["--input", "i", "--output", "o", "--head-only"], "--head-only"),
        ];
        for (route, argv, flag) in cases {
            let err = reject_unreachable(&matches(argv), *route).unwrap_err();
            assert!(err.starts_with(&format!("{flag} has no effect")), "{route:?}: {err}");
        }
    }

    #[test]
    fn defaults_and_readable_flags_pass() {
        // `--format` has a default and `--kmap-mode` too: unset ⇒ not rejected.
        reject_unreachable(&matches(&["--flux-pipe", "p", "--output", "o"]), Flux).unwrap();
        reject_unreachable(
            &matches(&["--input", "i", "--output", "o", "--format", "mq4", "--awq", "--imatrix", "x"]),
            Main,
        )
        .unwrap();
    }
}
