// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native hipfire control plane.
//!
//! This binary owns hipfire's operator surface and never shells out to a
//! JavaScript or TypeScript runtime.

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use hipfire_client::{
    complete_openai_chat, http_get_json, probe_host, service_ready, service_url,
    stream_openai_chat, Engine, OpenAiSseEvent, ServiceEndpoint,
};
use hipfire_config::{
    apply_config_profile, canonical_config_key, create_config_profile, field, fields,
    is_developer_key, load_catalog, load_env_layer, load_global, resolve, unknown_key,
    write_catalog_toml, write_global_toml, CatalogFormat, ConfigFormat, ConfigLayer, ConfigPaths,
    ConfigSource, NamedLayer, RemoveModelOverride, ValueRule, CONFIG_SCHEMA_VERSION,
};
use hipfire_registry::{
    load as load_registry, ExpertHotsetPin, GrandfatheredMonolithic, LoadedRegistry, ModelEntry,
    RegistryPaths, RegistrySource, RegistryV1, RetiredModel, Sidecar, TrunkOverlayPin,
    EXPERT_HOTSET_PINS, GRANDFATHERED_MONOLITHICS, RETIRED_MODELS, TRUNK_OVERLAY_PINS,
};
use hipfire_runtime::offload_calibrate::{CalibrateOptions, CalibrationReport};
use hipfire_runtime::prompt_frame::ToolCall;
use saddle_core::caps::ReasoningContract;
use saddle_core::kv::KvBackend;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    fs,
    io::{self, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

mod bench_concurrency;
mod bench_power;
mod calibrate;
mod config_tuning;
mod config_view;
#[cfg(target_os = "linux")]
mod doctor;
mod emu;
mod fit;
mod inspect;
mod kernel_pack;
mod npu_trunk;
mod ple;
mod prefill_progress;
mod process_identity;
mod pull_select;
mod serve;
mod setup;
mod spill;
use crate::serve::complete::next_attempt_id;
use crate::serve::http::request_id;
use crate::serve::{detach_serve, parse_host_port, parse_pid_record, ServePidRecord};
use hipfire_runtime::verified_cache::{self, VerifiedCache};
use setup::setup_command;

use hipfire_registry::{is_listable_model_file, MODEL_SUFFIXES};
const BUILD_COMMIT: &str = env!("HIPFIRE_BUILD_COMMIT");
const BUILD_REF: &str = env!("HIPFIRE_BUILD_REF");
const BUILD_DIRTY: &str = env!("HIPFIRE_BUILD_DIRTY");
const BUILD_TARGET: &str = env!("HIPFIRE_BUILD_TARGET");

#[derive(Parser, Debug)]
#[command(
    name = "hipfire",
    version = env!("HIPFIRE_BUILD_VERSION"),
    about = "LLM inference for AMD GPUs",
    long_about = "Native Rust control plane for hipfire. Configuration, registry, model lifecycle, serving, chat, and diagnostics are implemented without a JavaScript runtime."
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
    /// View or change settings (global or per-model); no arguments opens the TUI.
    Config(ConfigArgs),
    /// Inspect or refresh the signed-model-registry migration surface.
    Registry(RegistryArgs),
    /// List local models and optionally the remote registry.
    List(ListArgs),
    /// Download and verify a registered model and its sidecars.
    Pull(PullArgs),
    /// Remove a local model and registered sidecars.
    #[command(alias = "remove")]
    Rm(RmArgs),
    /// Launch the Rust terminal UI.
    Tui(TuiArgs),
    /// Report local GPU/runtime/model/control-plane readiness.
    Diag(OutputArgs),
    /// Show the native service process and loaded model.
    Ps(OutputArgs),
    /// Benchmark a model through the native daemon protocol.
    Bench(BenchArgs),
    /// SKU capacity emulation: list profiles, validate a profile, or estimate a target.
    Emu(emu::EmuArgs),
    /// Measure this host's GPU share for a split offload step.
    OffloadBench(OffloadBenchArgs),
    /// Measure this host's link, CPU and GPU rates and save them for the engine.
    Calibrate(calibrate::CalibrateArgs),
    /// Report compiled kernel inventory for the detected architecture.
    Profile(ProfileArgs),
    /// Print build, source-checkout, and installed-daemon identity.
    Version(OutputArgs),
    /// Update to a branch, tag, or commit and rebuild the native control plane.
    Update(UpdateArgs),
    /// Install or repair this machine's hipfire runtime.
    Setup(SetupArgs),
    /// Install a prebuilt release kernel pack (no device compiler needed).
    KernelPack(KernelPackArgs),
    /// Quantize a Hugging Face or local model with the Rust quantizer.
    Quantize(QuantizeArgs),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    /// Deprecated (CASK; removal in 0.5.0): generate a TriAttention calibration sidecar.
    SidecarGen(SidecarArgs),
    /// Generate text through a fresh native daemon process.
    Run(RunArgs),
    /// Generate an image (txt2img) through a fresh native daemon process.
    Img(ImgArgs),
    /// Encode text and/or images with an embedding model; optionally score a
    /// query against a document with late-interaction MaxSim.
    Embed(EmbedArgs),
    /// Start an interactive conversation through the native HTTP service.
    Chat(ChatArgs),
    /// Start the native OpenAI-compatible HTTP service.
    Serve(ServeArgs),
    /// Stop a detached native serve process.
    Stop(StopArgs),
    /// Stop and start the native HTTP service.
    Restart(ServeArgs),
    /// Derive or split the Flash-Next PLE row sidecars (`<model>.ple-q4`, `<base>.ple-q8`).
    Ple(ple::PleArgs),
    /// Write a keyed NPU trunk calibration from offline measurement inputs.
    CalibrateNpuTrunk(npu_trunk::CalibrateArgs),
    /// Runtime preflight: GPU driver, firmware, ROCm and install mismatches, with fixes (--npu: the Strix Halo NPU trunk route).
    #[cfg(target_os = "linux")]
    Doctor(doctor::DoctorArgs),
    /// What fits on this machine: context, placement, route and expected speed for a model.
    Fit(fit::FitArgs),
    /// Read a GGUF header: architecture, tensor types, bits per weight and whether hipfire can serve it.
    Inspect(inspect::InspectArgs),
}

#[derive(Args, Debug)]
#[command(subcommand_precedence_over_arg = true)]
struct ConfigArgs {
    /// Optional model tag, alias, filename, or local catalog identity.
    #[arg(value_name = "MODEL")]
    model: Option<String>,
    #[command(subcommand)]
    action: Option<ConfigAction>,
}

#[derive(Subcommand, Debug)]
enum ConfigAction {
    /// Print every setting by intent group with its effective value and origin.
    List(ConfigListArgs),
    /// Print one effective value.
    Get {
        key: String,
        #[command(flatten)]
        output: OutputArgs,
    },
    /// Persist one global user override to config.toml.
    Set { key: String, value: String },
    /// Remove one override, or all overrides when no key is supplied.
    Reset { key: Option<String> },
    /// Explain a key's type, scope, default, effective value, and provenance.
    Explain {
        /// Key to explain; omit to list every effective setting with its source
        /// and audit the HIPFIRE_* environment.
        key: Option<String>,
        /// Explain for one model (same as `hipfire config MODEL explain KEY`).
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,
        #[command(flatten)]
        output: OutputArgs,
    },
    /// List every per-model value in models.toml next to the all-models value.
    Overrides(OutputArgs),
    /// Print the authoritative typed configuration schema.
    Schema(OutputArgs),
    /// Convert legacy config.json to sparse config.toml without deleting JSON.
    Migrate,
    /// Select or create named configuration profiles.
    Profile {
        #[command(subcommand)]
        action: Option<ConfigProfileAction>,
    },
    /// Preview, apply and undo tuning presets (speed vs. answer quality); with
    /// no subcommand, opens the TUI on the preset chooser.
    Tuning {
        #[command(subcommand)]
        action: Option<ConfigTuningAction>,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigProfileAction {
    /// Merge a built-in or custom profile into the global sparse config.
    Set {
        /// Built-in (`default`, `dev`, `hip`, `redline`) or custom profile name.
        name: String,
    },
    /// Snapshot the current global sparse config as a new custom profile.
    Create {
        /// New custom profile name (not a built-in).
        name: String,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigTuningAction {
    /// Show which tuning preset is applied and which settings it owns.
    Status(OutputArgs),
    /// Preview a preset for this GPU (or another with --arch) without writing.
    Show {
        /// balanced, quality-first (alias quality) or low-power.
        preset: hipfire_config::tuning::Preset,
        /// Preview the plan for another GPU arch (for example gfx1201).
        #[arg(long, value_name = "GFX")]
        arch: Option<String>,
        #[command(flatten)]
        output: OutputArgs,
    },
    /// Apply a preset to config.toml after showing the change.
    Set {
        /// balanced, quality-first (alias quality) or low-power.
        preset: hipfire_config::tuning::Preset,
        /// Leave one preview line out (prompt-processing, flash-next-experts,
        /// flash-next-ple, shorter-thinking); repeatable.
        #[arg(long, value_name = "LINE")]
        skip: Vec<String>,
        /// Must equal this machine's GPU arch; use `show --arch` to preview others.
        #[arg(long, value_name = "GFX")]
        arch: Option<String>,
        /// Save without asking (required when stdin is not a terminal).
        #[arg(short, long)]
        yes: bool,
        #[command(flatten)]
        output: OutputArgs,
    },
    /// Undo the last tuning action.
    Undo {
        /// Undo without asking (required when stdin is not a terminal).
        #[arg(short, long)]
        yes: bool,
    },
}

#[derive(Args, Debug, Clone, Copy)]
struct OutputArgs {
    /// Emit machine-readable JSON.
    #[arg(short, long)]
    json: bool,
}

#[derive(Args, Debug, Clone, Default)]
struct ConfigListArgs {
    /// Show one intent group: quality, speed, memory, speculation, hardware, advanced.
    #[arg(long, value_name = "GROUP")]
    group: Option<String>,
    /// Only settings set by something other than the defaults or the model card.
    #[arg(long)]
    changed: bool,
    /// Only settings matching TEXT (key, legacy key, help, env var, category, scope).
    #[arg(long, value_name = "TEXT")]
    search: Option<String>,
    /// List for one model (same as `hipfire config MODEL list`).
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,
    /// Emit machine-readable JSON.
    #[arg(short, long)]
    json: bool,
}

#[derive(Args, Debug, Default)]
struct UpdateArgs {
    /// Branch, tag, or commit to install. A leading '@' is optional.
    #[arg(
        value_name = "REF",
        conflicts_with_all = ["branch", "tag", "commit"]
    )]
    reference: Option<String>,
    /// Install the tip of a named remote branch.
    #[arg(long, value_name = "NAME", conflicts_with_all = ["tag", "commit"])]
    branch: Option<String>,
    /// Install a named git tag in detached/pinned mode.
    #[arg(long, value_name = "TAG", conflicts_with = "commit")]
    tag: Option<String>,
    /// Install an exact git commit in detached/pinned mode.
    #[arg(long, value_name = "SHA")]
    commit: Option<String>,
}

#[derive(Args, Debug, Default)]
struct SetupArgs {
    /// Source checkout to build from (set by scripts/install.sh).
    #[arg(long, value_name = "PATH")]
    source: PathBuf,
    #[arg(long, value_name = "PATH")]
    rocm_root: Option<PathBuf>,
    /// Explicit device compiler (hipcc/amdclang++) when it lives in a different
    /// prefix than the runtime. Also set via HIPFIRE_HIPCC.
    #[arg(long, value_name = "PATH")]
    hipcc: Option<PathBuf>,
    /// Disable cross-root compiler fallback; require the compiler under the
    /// selected root. Also set via HIPFIRE_ROCM_STRICT=1.
    #[arg(long)]
    strict_rocm: bool,
    #[arg(long, value_name = "ARCH")]
    gpu_arch: Option<String>,
    /// auto (default) leaves replay.backend=auto so .mq4r models select Redline.
    #[arg(long, value_parser = ["auto", "hip", "redline"])]
    profile: Option<String>,
    #[arg(long, short = 'y', visible_alias = "non-interactive")]
    yes: bool,
    /// Requested revision ref forwarded by scripts/install.sh for install.json.
    /// A tag-named ref also selects that release's kernel pack.
    #[arg(
        long = "ref",
        value_name = "REF",
        hide = true,
        conflicts_with_all = ["branch", "tag", "commit"]
    )]
    reference: Option<String>,
    /// Requested branch forwarded by scripts/install.sh for install.json.
    #[arg(
        long,
        value_name = "NAME",
        hide = true,
        conflicts_with_all = ["tag", "commit"]
    )]
    branch: Option<String>,
    /// Requested tag forwarded by scripts/install.sh for install.json; also
    /// selects that release's prebuilt kernel pack.
    #[arg(long, value_name = "TAG", hide = true, conflicts_with = "commit")]
    tag: Option<String>,
    /// Requested commit forwarded by scripts/install.sh for install.json.
    #[arg(long, value_name = "SHA", hide = true)]
    commit: Option<String>,
    /// Compile kernels locally with hipcc even when a release kernel pack exists.
    #[arg(long)]
    compile_kernels: bool,
    /// Directory or URL (https://, file://, or a path) holding the release
    /// kernel pack assets. Default: the tag's GitHub release downloads.
    #[arg(long, value_name = "URL")]
    kernel_pack_url: Option<String>,
}

#[derive(Args, Debug)]
struct KernelPackArgs {
    #[command(subcommand)]
    action: KernelPackAction,
}

#[derive(Subcommand, Debug)]
enum KernelPackAction {
    /// Download, verify and install the pack for TAG and ARCH.
    Install(KernelPackInstallArgs),
}

#[derive(Args, Debug)]
pub(crate) struct KernelPackInstallArgs {
    /// Release tag whose pack to install.
    #[arg(long)]
    tag: String,
    /// GPU architecture, e.g. gfx1201.
    #[arg(long, value_name = "ARCH")]
    arch: String,
    /// Source checkout the installed daemon was built from; its HEAD must be
    /// the pack's commit.
    #[arg(long, value_name = "PATH")]
    source: PathBuf,
    /// Installed kernel directory to replace.
    /// Default: ~/.hipfire/bin/kernels/compiled/ARCH.
    #[arg(long, value_name = "DIR")]
    dest: Option<PathBuf>,
    /// Directory or URL holding the release assets. Default: the tag's GitHub release.
    #[arg(long, value_name = "URL")]
    url: Option<String>,
    /// Local ROCm version checked against the pack's supported range.
    /// Default: the resolved ROCm root's .info/version.
    #[arg(long, value_name = "VERSION")]
    rocm_version: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RevisionKind {
    Auto,
    Branch,
    Tag,
    Commit,
}

impl RevisionKind {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "ref",
            Self::Branch => "branch",
            Self::Tag => "tag",
            Self::Commit => "commit",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RevisionSelector {
    value: String,
    kind: RevisionKind,
}

#[derive(Debug)]
struct ResolvedRevision {
    selector: RevisionSelector,
    commit: String,
    tracking_ref: Option<String>,
}

#[derive(Args, Debug)]
struct RegistryArgs {
    #[command(subcommand)]
    action: RegistryAction,
}

#[derive(Subcommand, Debug)]
enum RegistryAction {
    /// Show registry source, revision, cache path, and warnings.
    Status(OutputArgs),
    /// List registered model tags.
    List(OutputArgs),
    /// Show one canonical registry entry.
    Show {
        tag: String,
        #[command(flatten)]
        output: OutputArgs,
    },
    /// Refresh the dynamic registry cache.
    Update(OutputArgs),
    /// Validate the bundled registry or an explicit v1 JSON file.
    Verify {
        path: Option<PathBuf>,
        #[command(flatten)]
        output: OutputArgs,
    },
}

#[derive(Args, Debug)]
struct ListArgs {
    /// Include registry models that are not downloaded.
    #[arg(short, long)]
    remote: bool,
    /// Emit machine-readable JSON.
    #[arg(short, long)]
    json: bool,
}

#[derive(Args, Debug)]
pub(crate) struct PullArgs {
    model: String,
    /// Replace an existing target after downloading and verifying a new copy.
    #[arg(long)]
    force: bool,
    /// Which PLE row sidecar a split model pulls next to its base: `auto`
    /// (memory.ple_format, else q4 on gfx1151 and q8 elsewhere; the union over
    /// every detected GPU arch), `q4`, `q8`,
    /// `all`, or `none` (fetched on first load). Omitted: asked on a terminal,
    /// else `auto`.
    #[arg(long, value_enum)]
    ple: Option<ple::PleSelection>,
    /// Fetch the registry's draft/MTP sidecars without asking (the default).
    #[arg(long, overrides_with = "no_draft")]
    draft: bool,
    /// Skip the registry's draft/MTP sidecars (MTP, DSpark, DFlash).
    #[arg(long, overrides_with = "draft")]
    no_draft: bool,
    /// Take the registry defaults without asking. A pull whose stdin is not a
    /// terminal never asks either.
    #[arg(short, long)]
    yes: bool,
}

impl PullArgs {
    /// A pull another command starts for a missing model: the requested tag
    /// with the registry defaults, never a prompt.
    pub(crate) fn defaults(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            force: false,
            ple: None,
            draft: false,
            no_draft: false,
            yes: true,
        }
    }

    /// `--draft` / `--no-draft`; `None` when neither is given.
    pub(crate) fn draft_flag(&self) -> Option<bool> {
        if self.no_draft {
            Some(false)
        } else if self.draft {
            Some(true)
        } else {
            None
        }
    }
}

#[derive(Args, Debug)]
struct RmArgs {
    model: String,
    /// Skip the interactive confirmation.
    #[arg(short, long)]
    yes: bool,
}

#[derive(Args, Debug, Default)]
struct TuiArgs {
    /// Arguments forwarded to hipfire-tui, such as --check.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    arguments: Vec<String>,
}

#[derive(Args, Debug, Clone)]
struct RunArgs {
    /// Registry tag, local alias, filename, or model path.
    model: String,
    /// Prompt words. Quote the prompt to preserve exact whitespace.
    #[arg(num_args = 0..)]
    prompt: Vec<String>,
    #[arg(short = 't', long)]
    /// Sampling temperature in 0..=2.
    temp: Option<f64>,
    #[arg(long)]
    /// Nucleus probability in (0, 1].
    top_p: Option<f64>,
    #[arg(long)]
    /// Multiplicative repetition penalty.
    repeat_penalty: Option<f64>,
    #[arg(short = 'n', long)]
    /// Maximum generated tokens.
    max_tokens: Option<u64>,
    #[arg(long)]
    /// One-shot KV format override for this model load.
    kv_mode: Option<String>,
    #[arg(long = "kv-k")]
    /// Qwen-only K-format override (e.g. `fwht3`, `q8`; `legacy-asymN`, `asymN`
    /// and `turboN` are deprecated, removal in 0.5.0).
    kv_k: Option<String>,
    #[arg(long = "kv-v")]
    /// Qwen-only V-format override (`q8`, `lloyd2`, `lloyd3`, `lloyd4`).
    kv_v: Option<String>,
    #[arg(long)]
    /// Select a published lm_head variant (see the registry's `heads`), e.g.
    /// `--head q4k`. The overlay shadows the model's own head at load time;
    /// omitting this uses the head baked into the model file.
    head: Option<String>,
    #[arg(long, value_parser = parse_kv_backend_arg, value_name = "legacy|vmm")]
    /// One-shot KV storage backend override for this model load (`legacy` or `vmm`).
    kv_backend: Option<String>,
    /// Tensor/expert-parallel degree for this model load (same admission as `serve --tp` / `bench --tp`).
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=64))]
    tp: Option<u64>,
    /// KV context (`max_seq`) for this model load, within the `memory.max_seq`
    /// bounds; overrides the configured/automatic context.
    #[arg(long, value_parser = parse_max_seq_arg, value_name = "N")]
    max_seq: Option<u64>,
    /// Select one speculative mechanism: off, auto, ngram, dflash, mtp, or dspark.
    #[arg(long = "spec", alias = "speculation")]
    speculation: Option<String>,
    /// Explicit DFlash draft model.
    #[arg(long, alias = "md")]
    model_draft: Option<PathBuf>,
    /// Explicit vision-tower sidecar (overrides the registry `vision` slot
    /// and `HIPFIRE_VISION_SIDECAR`; skipped while `vision_mode=off`).
    #[arg(long)]
    vision: Option<PathBuf>,
    /// Override the active MTP/n-gram draft window.
    #[arg(long, alias = "draft")]
    draft_max: Option<u64>,
    /// DSpark confidence cutoff.
    #[arg(long)]
    dspark_conf_threshold: Option<f64>,
    #[arg(long)]
    /// Override the resolved system prompt.
    system: Option<String>,
    #[arg(long)]
    /// Local image path for a vision-language model.
    image: Option<PathBuf>,
    #[arg(short = 'j', long)]
    /// Emit one JSON result object.
    json: bool,
    #[arg(long)]
    /// Print authoritative generation metrics on stderr (except with --json).
    stats: bool,
    #[arg(long)]
    /// Buffer visible output instead of streaming it.
    no_stream: bool,
}

#[derive(Args, Debug)]
#[command(after_help = "NOTE: --image must be given before the prompt words \
    (e.g. `hipfire img --image a.png --image b.png my-flux2-klein-pipe edit this photo`); \
    the prompt is a greedy trailing positional and swallows anything after it.")]
struct ImgArgs {
    /// Model tag, alias, filename, or path to an HFQ trunk pack
    /// (`<base>-transformer.hfq`, with its sidecar packs next to it).
    model: String,
    /// Prompt words. Quote the prompt to preserve exact whitespace.
    #[arg(num_args = 0..)]
    prompt: Vec<String>,
    /// Reference image for FLUX.2 Klein edit (repeatable, up to 4). Requires
    /// an arch-45 pipe; a FLUX.1 pipe refuses reference images.
    #[arg(long = "image", value_name = "PATH")]
    image: Vec<PathBuf>,
    /// Output PNG path (default: hipfire-<seed>.png in the working directory).
    #[arg(short = 'o', long)]
    out: Option<PathBuf>,
    /// Pixel width (must be divisible by the VAE compression factor). Defaults
    /// to 1024 when no `--image` is given; with `--image`, defaults to the
    /// reference image's own size.
    #[arg(long)]
    width: Option<u64>,
    /// Pixel height (must be divisible by the VAE compression factor).
    /// Defaults to 1024 when no `--image` is given; with `--image`, defaults
    /// to the reference image's own size.
    #[arg(long)]
    height: Option<u64>,
    /// Denoise steps; defaults to the model's architecture default (4 for
    /// step-distilled schnell, 28 for guidance-distilled dev).
    #[arg(long)]
    steps: Option<u64>,
    /// Noise seed; same seed → byte-identical PNG.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Execution backend for the transformer forward: `gpu` (the HIP MMDiT
    /// forward) or `cpu` (the f32 reference oracle). Defaults to `gpu` when
    /// a GPU is available, else `cpu`.
    #[arg(long)]
    backend: Option<String>,
    /// Emit one JSON result object instead of the progress lines.
    #[arg(short = 'j', long)]
    json: bool,
}

#[derive(Args, Debug)]
struct EmbedArgs {
    /// Model tag, alias, or path to an embedding checkpoint dir (HF
    /// snapshot of e.g. `pplx-embed-v2-late-0.6b`) or a raw-dtype `.hfq`.
    model: String,
    /// Text to encode (query- or document-side per `--kind`). Quote it to
    /// preserve exact whitespace.
    #[arg(num_args = 0..)]
    text: Vec<String>,
    /// Query-side text. With `--doc`, prints the MaxSim score of query
    /// against document.
    #[arg(long)]
    query: Option<String>,
    /// Document-side text (with `--query`).
    #[arg(long)]
    doc: Option<String>,
    /// Side for the positional text: `query` or `document` (default
    /// `document`).
    #[arg(long)]
    kind: Option<String>,
    /// Image input (PNG/JPEG); encoded with the checkpoint's vision tower.
    /// Repeatable — the images are appended after the text, in flag order
    /// (each takes the place of one `<|image|>` marker written in the text,
    /// or, with no marker, its run lands before EOS).
    #[arg(long = "image", value_name = "PATH")]
    image: Vec<PathBuf>,
    /// Audio input; encoded with the checkpoint's audio tower. Repeatable —
    /// clips are appended after the text and images, in flag order. Takes
    /// 16 kHz mono/stereo WAV/RIFF (PCM16 or float32); anything else is
    /// refused here with the conversion to run.
    #[arg(long = "audio", value_name = "PATH")]
    audio: Vec<PathBuf>,
    /// Video input (arch 18); encoded with the checkpoint's vision tower,
    /// which runs once per frame. One clip per flag: give that clip's
    /// FRAMES, comma separated, in display order — the daemon applies the
    /// video processor's sampler (`max_frames` 32) itself, so pass a decoded
    /// clip at whatever rate you extracted it. Repeat the flag for several
    /// clips; clip order is flag order and each clip's frames are spliced at
    /// its own `<|video|>` marker. Frames are PNG/JPEG; a container (MP4,
    /// MKV, …) is refused here with the `ffmpeg` extraction to run.
    #[arg(long = "video", value_name = "FRAME[,FRAME…]")]
    video: Vec<String>,
    /// Emit the full JSON result (token-embedding matrix / score object)
    /// instead of the human summary.
    #[arg(short = 'j', long)]
    json: bool,
}

#[derive(Args, Debug)]
struct ChatArgs {
    /// Model tag, alias, filename, or local catalog identity.
    model: Option<String>,
    #[arg(short = 't', long)]
    temp: Option<f64>,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(short = 'n', long)]
    max_tokens: Option<u64>,
    #[arg(long)]
    system: Option<String>,
    /// Accepted for compatibility; native chat does not emit ANSI colors.
    #[arg(long)]
    no_color: bool,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct BenchArgs {
    model: String,
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Tensor/expert-parallel degree for the benchmark daemon (same admission as `serve --tp`).
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=64))]
    tp: Option<u64>,
    #[arg(short = 'j', long)]
    json: bool,
    /// Compare the five RDNA2 kernel variants in isolated daemon processes.
    #[arg(long)]
    exp: bool,
    /// Run deterministic synthetic prefill/decode rows.
    #[arg(long)]
    matrix: bool,
    /// Client-side time-to-first-token: per run, generate one token on the
    /// standard prompt and time request-send to first streamed token.
    /// Reports prompt_tokens / TTFT, the competitor-comparable prefill
    /// number. Standard generate path only (not --matrix).
    #[arg(long)]
    ttft: bool,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "128,512,2048,4096,8192,20000"
    )]
    pp: Vec<usize>,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "128,512,2048,4096,8192,20000"
    )]
    ctx: Vec<usize>,
    #[arg(long, default_value_t = 128)]
    tg: usize,
    /// Generated tokens per measured request: default 128 on the standard
    /// bench, 256 on the `--concurrency` `steady` shape.
    #[arg(long)]
    max_tokens: Option<usize>,
    #[arg(long)]
    sustained_tg: Option<usize>,
    #[arg(long, value_delimiter = ',', default_value = "128,8192")]
    sustained_ctx: Vec<usize>,
    #[arg(long, default_value_t = 10)]
    warmups: usize,
    #[arg(long)]
    kv_mode: Option<String>,
    #[arg(long = "kv-k")]
    /// Qwen-only K-format override (e.g. `fwht3`, `q8`; `legacy-asymN`, `asymN`
    /// and `turboN` are deprecated, removal in 0.5.0).
    kv_k: Option<String>,
    #[arg(long = "kv-v")]
    /// Qwen-only V-format override (`q8`, `lloyd2`, `lloyd3`, `lloyd4`).
    kv_v: Option<String>,
    #[arg(long, value_parser = parse_kv_backend_arg, value_name = "legacy|vmm")]
    /// KV storage backend override (`legacy` or `vmm`).
    kv_backend: Option<String>,
    /// KV context (`max_seq`) for the benchmark load, within the
    /// `memory.max_seq` bounds. Omitted, `--matrix`/`--redline` size the load
    /// to the run (largest `--pp`, or `--ctx` + `--tg`, plus 32) and every
    /// other path keeps the configured/automatic context.
    #[arg(long, value_parser = parse_max_seq_arg, value_name = "N")]
    max_seq: Option<u64>,
    #[arg(long)]
    redline: bool,
    /// Speculation mode to benchmark (off, dflash, mtp, ngram, dspark, or auto).
    #[arg(long = "spec")]
    speculation: Option<String>,
    /// Let the model think during the benchmark. Off by default: a reasoning
    /// model cannot close its `<think>` span inside the benchmark's token
    /// budget, and the daemon fails such a turn closed as a validation error.
    /// Pair this with `--max-tokens` large enough for the span to close.
    #[arg(long = "reasoning-on")]
    reasoning_on: bool,
    /// Sweep closed-loop concurrency levels, e.g. `1,2,4,8`: at each level K,
    /// K client lanes drain the shape's N requests. Absent leaves bench on
    /// its single-stream path, unchanged.
    #[arg(long)]
    concurrency: Option<String>,
    /// Which backend to drive: slots (multi-slot engine), noslots (one
    /// request at a time, lanes queue), batch (the VMM continuous-batching
    /// route of `hipfire serve` with `serve.vmm_batch`), or both (all three).
    #[arg(long, value_parser = ["slots", "noslots", "batch", "both"], default_value = "both")]
    backend: String,
    /// Which workload arm to run: stateless, multiturn (two-turn chats, the
    /// follow-up sent on the same lane), or both.
    #[arg(long, value_parser = ["stateless", "multiturn", "both"], default_value = "both")]
    workload: String,
    /// `--concurrency` request population: `steady` (64 long-answer code
    /// prompts, fixed --max-tokens, greedy) or `betterbench` (BetterBench
    /// 0.6.0's concurrency workload over --bb-corpus).
    #[arg(long, value_parser = ["steady", "betterbench"], default_value = "steady")]
    bench_shape: String,
    /// BetterBench corpus directory (`*.jsonl` per category) for
    /// `--bench-shape betterbench`.
    #[arg(long, value_name = "DIR")]
    bb_corpus: Option<PathBuf>,
    /// `--concurrency` measured requests per sample (default 64 `steady`,
    /// 48 `betterbench`); request i is prompt i modulo the shape's list.
    #[arg(long, value_name = "N")]
    num_prompts: Option<usize>,
    /// `--concurrency` sampling temperature (default 0 `steady`, 0.7
    /// `betterbench`).
    #[arg(long)]
    temperature: Option<f64>,
    /// Prompt words for the standard benchmark.
    #[arg(num_args = 0..)]
    prompt: Vec<String>,
    /// Read the standard-benchmark prompt verbatim from a file (raw bytes,
    /// no trimming). Mutually exclusive with positional PROMPT words.
    #[arg(long, conflicts_with = "prompt")]
    prompt_file: Option<PathBuf>,
    /// SKU capacity-emulation profile (`hipfire emu list`) or `off` for every
    /// spawned bench daemon. Overrides `HIPFIRE_EMU`, including `off`. The host
    /// still executes at full speed; only application memory capacity is capped.
    #[arg(long, value_name = "ID|off")]
    emu: Option<String>,
}

impl BenchArgs {
    /// `--max-tokens` on the standard (single-stream) bench, default 128.
    fn standard_max_tokens(&self) -> usize {
        self.max_tokens.unwrap_or(128)
    }
}

#[derive(Args, Debug)]
struct OffloadBenchArgs {
    /// Emit machine-readable JSON.
    #[arg(short = 'j', long)]
    json: bool,
    /// Measure one format instead of the default matrix: a quant name (`mq4`,
    /// `mq4g256v2`, …) or a `.hfq` wire quant_type number.
    #[arg(long, value_name = "NAME|QT")]
    format: Option<String>,
    /// Activation width to measure at (default: the 9B trunk's hidden size).
    #[arg(long, value_name = "N")]
    k: Option<usize>,
    /// Weight megabytes per format.
    #[arg(long, value_name = "MB")]
    buffer_mb: Option<usize>,
    /// Timed reps per engine; the median is reported.
    #[arg(long, value_name = "N")]
    reps: Option<usize>,
    /// Persist the recommended share (the median over the largest measurements).
    #[arg(long)]
    write: bool,
}

#[derive(Args, Debug)]
struct ProfileArgs {
    model: Option<String>,
    #[arg(long)]
    kernel: Option<String>,
    #[arg(short = 'j', long)]
    json: bool,
}

#[derive(Args, Debug)]
struct QuantizeArgs {
    /// Hugging Face model ID or local safetensors directory (a GGUF file is
    /// deprecated, removal in 0.5.0: GGUF→mqN is lossy double quantization; use llama.cpp for GGUF).
    input: String,
    #[arg(long = "format")]
    /// Repeatable output format: mq4, mq6, q8, q8f16, hf4, or hf6.
    formats: Vec<String>,
    #[arg(long)]
    /// Produce both MQ4 and MQ6.
    both: bool,
    #[arg(short = 'o', long)]
    /// Exact output path; valid with one format only.
    output: Option<PathBuf>,
    #[arg(long)]
    /// Output directory for one or more formats.
    output_dir: Option<PathBuf>,
    #[arg(long)]
    /// Override the output filename stem.
    stem: Option<String>,
    #[arg(long)]
    /// Symmetric per-128 MQ4V2 grid (qt44, header zero = f16(-8*d)); forwarded to
    /// `hipfire-quantize --mq4v2-symmetric`. Requires `--format mq4`. Models with a
    /// dedicated recipe (Raon-1-mini) require `--mq4v2-symmetric --tier xt`.
    mq4v2_symmetric: bool,
    #[arg(long, value_parser = ["xt", "base", "pro"])]
    /// Product tier forwarded to `hipfire-quantize --tier`. Requires `--format mq4`.
    tier: Option<String>,
    #[arg(long)]
    /// Upload completed artifacts to owner/repo on Hugging Face.
    upload: Option<String>,
    #[arg(long)]
    /// Create the Hugging Face model repository if needed.
    create_repo: bool,
    #[arg(long)]
    /// Copy completed artifacts into ~/.hipfire/models.
    install: bool,
    #[arg(long)]
    /// Register a local model alias in models.toml.
    register: Option<String>,
}

#[derive(Args, Debug)]
struct SidecarArgs {
    model: String,
    #[arg(long)]
    corpus: Option<PathBuf>,
    #[arg(long, default_value_t = 4000)]
    max_tokens: usize,
    #[arg(long, default_value_t = 256)]
    chunk_len: usize,
    #[arg(long, conflicts_with = "cpu_calib")]
    gpu_calib: bool,
    #[arg(long)]
    cpu_calib: bool,
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    #[arg(long)]
    skip_validation: bool,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct ServeArgs {
    /// Optional model, host, host:port, and/or port in legacy-compatible order.
    #[arg(value_name = "MODEL_HOST_OR_PORT", num_args = 0..=3)]
    positionals: Vec<String>,
    /// Model tag/path to pre-warm for this process.
    #[arg(long)]
    model: Option<String>,
    /// Run in the background and log to ~/.hipfire/serve.log.
    #[arg(short = 'd', long, alias = "background")]
    detach: bool,
    /// Do not load the configured default model before accepting requests.
    #[arg(long)]
    no_prewarm: bool,
    /// KV cache mode for models loaded by this service.
    #[arg(long)]
    kv_mode: Option<String>,
    /// Qwen-only K-format override for models loaded by this service.
    #[arg(long = "kv-k")]
    kv_k: Option<String>,
    /// Qwen-only V-format override for models loaded by this service.
    #[arg(long = "kv-v")]
    kv_v: Option<String>,
    /// KV storage backend for models loaded by this service (`legacy` or `vmm`).
    #[arg(long, value_parser = parse_kv_backend_arg, value_name = "legacy|vmm")]
    kv_backend: Option<String>,
    /// Vision-tower sidecar wired into every model load (`params["vision"]`);
    /// overrides the registry `vision` slot and `HIPFIRE_VISION_SIDECAR`;
    /// skipped while `vision_mode=off`.
    #[arg(long)]
    vision: Option<PathBuf>,
    /// Idle model-unload timeout in seconds; zero disables eviction.
    #[arg(long, value_parser = clap::value_parser!(u64).range(0..=86400))]
    idle_timeout: Option<u64>,
    /// Tensor/expert-parallel degree.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=64))]
    tp: Option<u64>,
    /// Speculation mode for models loaded by this service: off, auto, ngram,
    /// dflash, mtp, or dspark. Overrides `speculation.mode` and per-model
    /// speculation settings; `--spec off` disables MTP/DFlash.
    #[arg(
        long = "spec",
        alias = "speculation",
        value_parser = ["off", "auto", "ngram", "dflash", "mtp", "dspark"]
    )]
    speculation: Option<String>,
    /// KV context (`max_seq`) for models loaded by this service, within the
    /// `memory.max_seq` bounds; overrides the configured/automatic context.
    #[arg(long, value_parser = parse_max_seq_arg, value_name = "N")]
    max_seq: Option<u64>,
    /// Multi-slot serve: host-RAM ceiling for parked swap snapshots, in
    /// bytes; zero disables the host tier so snapshots spill to disk.
    #[arg(long, value_parser = clap::value_parser!(u64).range(0..=1_099_511_627_776), value_name = "BYTES")]
    swap_host_bytes: Option<u64>,
    /// Multi-slot serve: disk ceiling for parked swap snapshots, in bytes;
    /// zero disables disk spill. LRU snapshots are evicted at the ceiling.
    #[arg(long, value_parser = clap::value_parser!(u64).range(0..=1_099_511_627_776), value_name = "BYTES")]
    swap_disk_bytes: Option<u64>,
    /// Multi-slot serve: directory for swap spill files; empty uses
    /// `$XDG_RUNTIME_DIR/hipfire/swap/<pid>`, falling back to
    /// `$TMPDIR/hipfire-swap-<pid>`.
    #[arg(long, value_name = "DIR")]
    swap_dir: Option<PathBuf>,
    /// Maximum concurrent eligible batched lanes; 1 preserves sequential behavior.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=hipfire_config::MAX_CONTINUOUS_BATCH_SIZE))]
    continuous_batch_size: Option<u64>,
    /// Serve the embedded chat UI at /ui on the serve listener.
    #[arg(long)]
    ui: bool,
    /// Open the chat UI in the default browser once serving (implies --ui).
    #[arg(long)]
    open: bool,
    /// Absolute path of a file holding the bearer token clients must present
    /// (`Authorization: Bearer <token>`); overrides `serve.api_key_file`.
    /// Optional; recommended when binding a non-loopback address.
    #[arg(long, value_name = "ABSOLUTE_PATH", value_parser = crate::serve::parse_api_key_file_arg)]
    api_key_file: Option<PathBuf>,
    /// Internal marker used by the detached child.
    #[arg(long, hide = true)]
    foreground_child: bool,
}

#[derive(Args, Debug, Clone, Copy)]
pub(crate) struct StopArgs {
    /// Port to free when --force or --all is used.
    port: Option<u16>,
    /// Reap orphan daemon processes and free the configured port.
    #[arg(long)]
    force: bool,
    /// Also reap native quantizer processes.
    #[arg(long)]
    all: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Paths {
    root: PathBuf,
    models: PathBuf,
    config: ConfigPaths,
    registry: RegistryPaths,
}

impl Paths {
    /// A missing home directory is a clean CLI error, not a panic.
    fn discover() -> Result<Self> {
        let config = ConfigPaths::try_discover()?;
        let root = config.root.clone();
        Ok(Self {
            models: config.models.clone(),
            registry: RegistryPaths {
                cache: root.join("registry.cache.json"),
            },
            root,
            config,
        })
    }
}

fn main() {
    finish_command(run());
}

fn finish_command(result: Result<()>) {
    if BENCH_INTERRUPT.load(Ordering::SeqCst) {
        // Killing the daemon wakes its reader, so run() may return before the
        // handler does. Finish reaping before either thread exits the process.
        Engine::terminate_spawned();
        std::process::exit(BENCH_INTERRUPTED_EXIT);
    }
    if let Err(error) = result {
        eprintln!("hipfire: {error:#}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let cli = Cli::parse_from(env::args_os().map(|argument| {
        if argument == "-md" {
            OsString::from("--model-draft")
        } else {
            argument
        }
    }));
    // Print-only commands die quietly on a closed stdout (`hipfire list -r |
    // head -1`) like any Unix filter instead of panicking in `println!`.
    // Commands that drive a daemon, server or child over pipes keep Rust's
    // ignored SIGPIPE so a dead peer is an EPIPE error, not a silent exit.
    #[cfg(unix)]
    if matches!(
        cli.command,
        Some(
            Commands::List(_)
                | Commands::Registry(_)
                | Commands::Config(_)
                | Commands::Ps(_)
                | Commands::Version(_)
                | Commands::Rm(_)
                | Commands::Inspect(_)
        )
    ) {
        // SAFETY: restores the default disposition before any thread starts.
        unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
    }
    let paths = Paths::discover()?;
    match cli.command {
        None => launch_tui(&paths, &[]),
        Some(Commands::Tui(args)) => launch_tui(&paths, &args.arguments),
        Some(Commands::Inspect(args)) => inspect::inspect_command(args),
        Some(Commands::Config(args)) => config_command(&paths, args),
        Some(Commands::Registry(args)) => registry_command(&paths, args),
        Some(Commands::List(args)) => list_command(&paths, args),
        Some(Commands::Pull(args)) => pull_command(&paths, args),
        Some(Commands::Ple(args)) => ple::ple_command(&paths, args),
        Some(Commands::CalibrateNpuTrunk(args)) => npu_trunk::calibrate_command(args),
        #[cfg(target_os = "linux")]
        Some(Commands::Doctor(args)) => doctor::doctor_command(&paths, args),
        Some(Commands::Fit(args)) => fit::fit_command(&paths, args),
        Some(Commands::Rm(args)) => rm_command(&paths, args),
        Some(Commands::Diag(output)) => diag_command(&paths, output),
        Some(Commands::Ps(output)) => ps_command(&paths, output),
        Some(Commands::Bench(args)) => bench_command(&paths, args),
        Some(Commands::Emu(args)) => emu::emu_command(&paths, args),
        Some(Commands::OffloadBench(args)) => offload_bench_command(&paths, args),
        Some(Commands::Calibrate(args)) => calibrate::calibrate_command(&paths, args),
        Some(Commands::Profile(args)) => profile_command(&paths, args),
        Some(Commands::Version(output)) => version_command(&paths, output),
        Some(Commands::Update(args)) => update_command(&paths, args),
        Some(Commands::Setup(args)) => setup_command(&paths, args),
        Some(Commands::KernelPack(args)) => match args.action {
            KernelPackAction::Install(args) => kernel_pack::install_command(&paths, args),
        },
        Some(Commands::Quantize(args)) => quantize_command(&paths, args),
        Some(Commands::SidecarGen(args)) => sidecar_command(&paths, args),
        Some(Commands::Run(args)) => run_command(&paths, args),
        Some(Commands::Img(args)) => img_command(&paths, args),
        Some(Commands::Embed(args)) => embed_command(&paths, args),
        Some(Commands::Chat(args)) => chat_command(&paths, args),
        Some(Commands::Serve(args)) => crate::serve::serve_command(&paths, args),
        Some(Commands::Stop(args)) => crate::serve::stop_command(&paths, args),
        Some(Commands::Restart(args)) => {
            let port = args.positionals.iter().find_map(|value| {
                value.parse::<u16>().ok().or_else(|| {
                    crate::serve::parse_host_port(value)
                        .ok()
                        .flatten()
                        .map(|(_, port)| port)
                })
            });
            let _ = crate::serve::stop_command(
                &paths,
                StopArgs {
                    port,
                    force: true,
                    all: false,
                },
            );
            crate::serve::serve_command(&paths, args)
        }
    }
}

/// Persist one global user override and return what `hipfire config set` prints.
/// `doctor --npu --apply` shares this so both write the same way.
pub(crate) fn set_global_config(paths: &Paths, key: &str, value: &str) -> Result<String> {
    let mut loaded = load_global(&paths.config)?;
    loaded.layer.set_cli(key, value)?;
    write_global_toml(&paths.config, &loaded.layer)?;
    let canonical = canonical_config_key(key).ok_or_else(|| unknown_key(key))?;
    let value = loaded
        .layer
        .get(&canonical)
        .ok_or_else(|| anyhow!("configuration key '{canonical}' is not set"))?;
    let mut line = format!("{canonical} = {value}");
    if loaded.format == ConfigFormat::LegacyJson {
        line.push_str(&format!(
            "\nmigrated active configuration to {}; preserved {} as a rollback copy",
            paths.config.config_toml.display(),
            paths.config.config_json.display()
        ));
    }
    Ok(line)
}

fn config_command(paths: &Paths, args: ConfigArgs) -> Result<()> {
    let flag_model = match &args.action {
        Some(ConfigAction::List(list)) => list.model.clone(),
        Some(ConfigAction::Explain { model, .. }) => model.clone(),
        _ => None,
    };
    let model = match (args.model, flag_model) {
        (Some(positional), Some(flag)) if positional != flag => {
            bail!("model given twice: '{positional}' and --model '{flag}'")
        }
        (positional, flag) => positional.or(flag),
    };
    if let Some(model) = model {
        return model_config_command(paths, &model, args.action);
    }
    let Some(action) = args.action else {
        // `hipfire config` is the config TUI: open it on the Settings tab.
        return launch_tui(paths, &["--settings".to_owned()]);
    };
    match action {
        ConfigAction::List(list) => {
            config_view::list_global_to(paths, &list, &mut std::io::stdout())
        }
        ConfigAction::Get { key, output } => {
            let (_, resolved) = resolved_global(paths, true)?;
            let canonical = canonical_config_key(&key).ok_or_else(|| unknown_key(&key))?;
            let schema = field(&canonical);
            let value = resolved
                .get(&canonical)
                .ok_or_else(|| anyhow!("configuration key '{canonical}' is not set"))?;
            if output.json {
                println!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "key": canonical,
                        "legacy_key": schema.map(|schema| schema.legacy_key),
                        "value": value.value,
                        "source": value.source,
                    }))?
                );
            } else {
                println!("{}", value.value);
            }
            Ok(())
        }
        ConfigAction::Set { key, value } => {
            println!("{}", set_global_config(paths, &key, &value)?);
            Ok(())
        }
        ConfigAction::Reset { key } => {
            let mut loaded = load_global(&paths.config)?;
            if let Some(key) = key {
                let canonical = canonical_config_key(&key).ok_or_else(|| unknown_key(&key))?;
                let existed = loaded.layer.remove(&canonical)?.is_some();
                write_global_toml(&paths.config, &loaded.layer)?;
                if existed {
                    println!("{canonical} override removed");
                } else {
                    println!("{canonical} was already inherited");
                }
            } else {
                write_global_toml(&paths.config, &ConfigLayer::default())?;
                println!("all global overrides removed");
            }
            Ok(())
        }
        ConfigAction::Explain { key, output, .. } => match key {
            Some(key) => {
                config_view::explain_global_to(paths, &key, output.json, &mut std::io::stdout())
            }
            None => config_view::explain_all_global_to(
                paths,
                output.json,
                &hipfire_config::env_audit::audit_process_env(),
                &mut std::io::stdout(),
            ),
        },
        ConfigAction::Overrides(output) => {
            config_view::overrides_to(paths, output.json, &mut std::io::stdout())
        }
        ConfigAction::Schema(output) => {
            let schema = fields()
                .iter()
                .map(|field| {
                    serde_json::json!({
                        "key": field.key,
                        "legacy_key": field.legacy_key,
                        "category": field.category,
                        "scope": field.scope,
                        "default": config_default_value(field),
                        "rule": config_rule_json(field.rule),
                        "registry_allowed": field.registry_allowed,
                        "experimental": field.experimental,
                        "env_compat": field.env_compat,
                        "help": field.help,
                    })
                })
                .collect::<Vec<_>>();
            if output.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "schema_version": CONFIG_SCHEMA_VERSION,
                        "fields": schema,
                        "developer_namespace": {
                            "prefix": "developer.",
                            "scope": "process",
                            "registry_allowed": false,
                            "experimental": true,
                            "value_types": ["boolean", "integer", "number", "string"],
                            "legacy_mapping": "HIPFIRE_FOO -> developer.foo"
                        },
                    }))?
                );
            } else {
                println!("Configuration schema v{CONFIG_SCHEMA_VERSION}");
                for field in fields() {
                    println!(
                        "  {:<48} {:<18} {:<12} {}",
                        field.key,
                        format_default(field),
                        config_rule_label(field.rule),
                        field.help
                    );
                }
                println!(
                    "  {:<48} {:<18} {:<12} Experimental process-scoped overrides (HIPFIRE_FOO -> developer.foo).",
                    "developer.<name>", "unset", "scalar"
                );
            }
            Ok(())
        }
        ConfigAction::Migrate => {
            let loaded = load_global(&paths.config)?;
            match loaded.format {
                ConfigFormat::Toml => {
                    println!("already using {}", paths.config.config_toml.display());
                }
                ConfigFormat::LegacyJson => {
                    write_global_toml(&paths.config, &loaded.layer)?;
                    println!(
                        "wrote {}; preserved {} unchanged",
                        paths.config.config_toml.display(),
                        paths.config.config_json.display()
                    );
                    for warning in loaded.warnings {
                        eprintln!("warning: {warning}");
                    }
                }
                ConfigFormat::Empty => {
                    write_global_toml(&paths.config, &ConfigLayer::default())?;
                    println!("wrote {}", paths.config.config_toml.display());
                }
            }
            let catalog = load_catalog(&paths.config)?;
            match catalog.format {
                CatalogFormat::Toml => {
                    println!("already using {}", paths.config.models_toml.display());
                }
                CatalogFormat::LegacyJson => {
                    write_catalog_toml(&paths.config, &catalog.catalog)?;
                    println!(
                        "wrote {}; preserved {} and {} unchanged",
                        paths.config.models_toml.display(),
                        paths.config.models_json.display(),
                        paths.config.legacy_per_model_json.display()
                    );
                    for warning in catalog.warnings {
                        eprintln!("warning: {warning}");
                    }
                }
                CatalogFormat::Empty => {
                    write_catalog_toml(&paths.config, &catalog.catalog)?;
                    println!("wrote {}", paths.config.models_toml.display());
                }
            }
            Ok(())
        }
        ConfigAction::Profile { action } => config_profile_command(paths, action),
        ConfigAction::Tuning { action } => match action {
            Some(action) => config_tuning::config_tuning_command(paths, action),
            None => launch_tui(paths, &config_tuning::tui_args()),
        },
    }
}

fn config_profile_command(paths: &Paths, action: Option<ConfigProfileAction>) -> Result<()> {
    let Some(action) = action else {
        return launch_tui(paths, &["--config-profile-wizard".to_owned()]);
    };
    match action {
        ConfigProfileAction::Set { name } => {
            let mut loaded = load_global(&paths.config)?;
            let set = apply_config_profile(&mut loaded.layer, &paths.config, &name)?;
            write_global_toml(&paths.config, &loaded.layer)?;
            println!(
                "applied configuration profile '{name}' (set {set} key{})",
                if set == 1 { "" } else { "s" }
            );
            if loaded.format == ConfigFormat::LegacyJson {
                println!(
                    "migrated active configuration to {}; preserved {} as a rollback copy",
                    paths.config.config_toml.display(),
                    paths.config.config_json.display()
                );
            }
            Ok(())
        }
        ConfigProfileAction::Create { name } => {
            let loaded = load_global(&paths.config)?;
            let path = create_config_profile(&paths.config, &name, &loaded.layer)?;
            println!(
                "created configuration profile '{name}' at {}",
                path.display()
            );
            Ok(())
        }
    }
}

fn model_config_command(
    paths: &Paths,
    model_name: &str,
    action: Option<ConfigAction>,
) -> Result<()> {
    let registry = load_registry(&paths.registry).registry;
    model_config_command_to(paths, &registry, model_name, action, &mut std::io::stdout())
}

fn model_config_command_to(
    paths: &Paths,
    registry: &RegistryV1,
    model_name: &str,
    action: Option<ConfigAction>,
    out: &mut dyn std::io::Write,
) -> Result<()> {
    let card = model_card_by_name(registry, model_name);
    let (tag, entry) = (card.tag().map(str::to_owned), card.entry());
    let action = action.unwrap_or(ConfigAction::List(ConfigListArgs::default()));
    if matches!(
        action,
        ConfigAction::Migrate
            | ConfigAction::Schema(_)
            | ConfigAction::Profile { .. }
            | ConfigAction::Overrides(_)
            | ConfigAction::Tuning { .. }
    ) {
        bail!("config migrate/schema/profile/overrides/tuning are global; omit the model argument");
    }

    match action {
        ConfigAction::List(list) => {
            config_view::list_model_to(paths, registry, model_name, &list, out)
        }
        ConfigAction::Get { key, output } => {
            if is_developer_key(&key) {
                bail!("developer configuration is global process policy; omit the model argument");
            }
            let resolved = resolved_for_model(paths, model_name, tag.as_deref(), entry)?;
            let schema = field(&key).ok_or_else(|| unknown_key(&key))?;
            let value = resolved
                .get(schema.key)
                .ok_or_else(|| anyhow!("configuration key '{}' is not set", schema.key))?;
            if output.json {
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "model": model_name,
                        "key": schema.key,
                        "legacy_key": schema.legacy_key,
                        "value": value.value,
                        "source": value.source,
                    }))?
                )?;
            } else {
                writeln!(out, "{}", value.value)?;
            }
            Ok(())
        }
        ConfigAction::Set { key, value } => {
            if is_developer_key(&key) {
                bail!("developer configuration is global process policy; omit the model argument");
            }
            let saved = hipfire_config::set_model_override(
                &paths.config,
                tag.as_deref(),
                find_model_path(paths, registry, model_name),
                model_name,
                &key,
                &value,
            )?;
            writeln!(out, "{} {} = {}", saved.id, saved.key, saved.value)?;
            if saved.migrated_legacy_json {
                writeln!(
                    out,
                    "migrated model catalog to {}; preserved legacy JSON as rollback copies",
                    paths.config.models_toml.display()
                )?;
            }
            Ok(())
        }
        ConfigAction::Reset { key } => {
            if key.as_deref().is_some_and(is_developer_key) {
                bail!("developer configuration is global process policy; omit the model argument");
            }
            match hipfire_config::remove_model_override(&paths.config, model_name, key.as_deref())?
            {
                RemoveModelOverride::NoOverrides => {
                    writeln!(out, "{model_name} has no per-model overrides")?
                }
                RemoveModelOverride::KeyRemoved { id, key } => {
                    writeln!(out, "{id} {key} override removed")?
                }
                RemoveModelOverride::KeyInherited { id, key } => {
                    writeln!(out, "{id} {key} was already inherited")?
                }
                RemoveModelOverride::AllRemoved { id } => {
                    writeln!(out, "all {id} overrides removed")?
                }
            }
            Ok(())
        }
        ConfigAction::Explain {
            key: None, output, ..
        } => config_view::explain_all_model_to(
            paths,
            registry,
            model_name,
            output.json,
            &hipfire_config::env_audit::audit_process_env(),
            out,
        ),
        ConfigAction::Explain {
            key: Some(key),
            output,
            ..
        } => config_view::explain_model_to(paths, registry, model_name, &key, output.json, out),
        ConfigAction::Migrate
        | ConfigAction::Schema(_)
        | ConfigAction::Profile { .. }
        | ConfigAction::Overrides(_)
        | ConfigAction::Tuning { .. } => {
            unreachable!()
        }
    }
}

pub(crate) fn resolved_global(
    paths: &Paths,
    include_env: bool,
) -> Result<(hipfire_config::LoadedConfig, hipfire_config::ResolvedConfig)> {
    let loaded = load_global(&paths.config)?;
    let mut layers = vec![NamedLayer {
        source: ConfigSource::GlobalUser {
            path: loaded.path.clone(),
        },
        layer: loaded.layer.clone(),
    }];
    if include_env {
        let env_layer = load_env_layer()?;
        if !env_layer.values.is_empty() {
            // Field-level env names remain available from schema metadata. A
            // single layer source makes effective output concise while explain
            // still names the compatibility surface.
            layers.push(NamedLayer {
                source: ConfigSource::LegacyEnv {
                    name: "HIPFIRE_*".into(),
                },
                layer: env_layer,
            });
        }
    }
    Ok((loaded, resolve(layers)?))
}

fn registry_command(paths: &Paths, args: RegistryArgs) -> Result<()> {
    match args.action {
        RegistryAction::Verify { path, output } => {
            let registry = if let Some(path) = path.as_deref() {
                let raw = fs::read_to_string(path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                RegistryV1::parse(&raw, path.display().to_string())?
            } else {
                hipfire_registry::bundled()?
            };
            if output.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "valid": true,
                        "schema_version": registry.schema_version,
                        "generated_at": registry.generated_at,
                        "models": registry.models.len(),
                        "aliases": registry.aliases.len(),
                    }))?
                );
            } else {
                println!(
                    "registry valid: schema v{}, {} models, {} aliases, generated {}",
                    registry.schema_version,
                    registry.models.len(),
                    registry.aliases.len(),
                    registry.generated_at
                );
            }
            Ok(())
        }
        action => {
            let loaded = load_registry(&paths.registry);
            match action {
                RegistryAction::Status(output) | RegistryAction::Update(output) => {
                    if output.json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&registry_status_json(paths, &loaded))?
                        );
                    } else {
                        println!("source:       {}", registry_source(loaded.source));
                        println!("schema:       v{}", loaded.registry.schema_version);
                        println!("generated:    {}", loaded.registry.generated_at);
                        println!("models:       {}", loaded.registry.models.len());
                        println!("aliases:      {}", loaded.registry.aliases.len());
                        println!("cache:        {}", paths.registry.cache.display());
                        for warning in &loaded.warnings {
                            eprintln!("warning: {warning}");
                        }
                    }
                    Ok(())
                }
                RegistryAction::List(output) => print_registry_list(&loaded, output.json),
                RegistryAction::Show { tag, output } => {
                    let (canonical, entry) = loaded
                        .registry
                        .model(&tag)
                        .ok_or_else(|| unknown_model_error(&loaded.registry, &tag))?;
                    if output.json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&serde_json::json!({
                                "tag": canonical,
                                "entry": entry,
                                "registry_source": registry_source(loaded.source),
                            }))?
                        );
                    } else {
                        println!("tag:          {canonical}");
                        println!("repo:         {}", entry.repo);
                        println!("file:         {}", entry.file);
                        println!("size:         {:.3} GB", entry.size_gb);
                        println!("minimum VRAM: {:.3} GB", entry.min_vram_gb);
                        if let Some(hash) = &entry.sha256 {
                            println!("sha256:       {hash}");
                        }
                        if let Some(quant) = &entry.quant {
                            println!("quant:        {quant}");
                        }
                        println!("about:        {}", entry.desc);
                    }
                    Ok(())
                }
                RegistryAction::Verify { .. } => unreachable!(),
            }
        }
    }
}

fn registry_status_json(paths: &Paths, loaded: &LoadedRegistry) -> serde_json::Value {
    serde_json::json!({
        "source": registry_source(loaded.source),
        "schema_version": loaded.registry.schema_version,
        "generated_at": loaded.registry.generated_at,
        "models": loaded.registry.models.len(),
        "aliases": loaded.registry.aliases.len(),
        "cache_path": paths.registry.cache,
        "warnings": loaded.warnings,
    })
}

fn print_registry_list(loaded: &LoadedRegistry, json: bool) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "source": registry_source(loaded.source),
                "models": loaded.registry.models,
                "aliases": loaded.registry.aliases,
            }))?
        );
    } else {
        for (tag, model) in &loaded.registry.models {
            println!("  {:<32} {:>7.2} GB  {}", tag, model.size_gb, model.desc);
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct LocalModel {
    name: String,
    path: PathBuf,
    size_bytes: u64,
    /// File mtime as a unix timestamp, the closest thing a local artifact has
    /// to OpenAI's required `created`. `0` when the platform cannot report one.
    created: u64,
    registry_tag: Option<String>,
    /// The registry no longer publishes this file's tag; the client still runs
    /// it from a verified local copy ([`RetiredModel`]).
    retired: bool,
}

/// One `hipfire list` row of a local model: file, size, registry tag, and a
/// `[retired]` marker when the registry no longer publishes the file's tag.
fn local_model_line(model: &LocalModel) -> String {
    let tag = model
        .registry_tag
        .as_deref()
        .map(|tag| format!(" ({tag})"))
        .unwrap_or_default();
    format!(
        "  {:<42} {:>7.2} GB{}{}",
        model.name,
        model.size_bytes as f64 / 1e9,
        tag,
        if model.retired { " [retired]" } else { "" }
    )
}

fn list_command(paths: &Paths, args: ListArgs) -> Result<()> {
    let loaded = load_registry(&paths.registry);
    let local = list_local_models(paths, &loaded.registry)?;
    let local_files = local
        .iter()
        .map(|model| model.name.as_str())
        .collect::<BTreeSet<_>>();
    if args.json {
        let registry = loaded
            .registry
            .models
            .iter()
            .map(|(tag, entry)| {
                serde_json::json!({
                    "tag": tag,
                    "name": entry.file,
                    "size_bytes": entry.size_bytes.unwrap_or_else(|| (entry.size_gb * 1e9).round() as u64),
                    "quant": entry.quant,
                    "downloaded": local_files.contains(entry.file.as_str())
                        || grandfather_pin(entry).is_some_and(|pin| {
                            local_files.contains(pin.canonical_file)
                                || local_files.contains(pin.legacy_file)
                        }),
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "models": local,
                "registry": registry,
                "registry_source": registry_source(loaded.source),
            }))?
        );
        return Ok(());
    }
    if local.is_empty() {
        println!("No local models. Pull one:\n  hipfire pull qwen3.5:4b");
    } else {
        println!("Local models:\n");
        for model in &local {
            println!("{}", local_model_line(model));
        }
    }
    if args.remote || local.is_empty() {
        println!("\nAvailable models:\n");
        for (tag, entry) in &loaded.registry.models {
            let status = if local_files.contains(entry.file.as_str()) {
                " [downloaded]"
            } else {
                ""
            };
            println!(
                "  {:<32} {:>7.2} GB  {}{}",
                tag, entry.size_gb, entry.desc, status
            );
        }
    }
    Ok(())
}

pub(crate) fn list_local_models(paths: &Paths, registry: &RegistryV1) -> Result<Vec<LocalModel>> {
    let mut candidates = local_model_paths(paths, registry)?;
    let mut catalog_tags: std::collections::HashMap<PathBuf, String> =
        std::collections::HashMap::new();
    if let Ok(catalog) = load_catalog(&paths.config) {
        for model in catalog.catalog.models.values() {
            if let (Some(path), Some(tag)) = (&model.path, &model.registry_tag) {
                if let Ok(canonical) = fs::canonicalize(path) {
                    catalog_tags.insert(canonical, tag.clone());
                }
            }
        }
        candidates.extend(
            catalog
                .catalog
                .models
                .values()
                .filter_map(|model| model.path.clone())
                .filter(|path| path.is_file()),
        );
    }
    let retired_pins = retired_table();
    let mut seen = BTreeSet::new();
    let mut models = Vec::new();
    for path in candidates {
        let canonical = fs::canonicalize(&path).unwrap_or(path);
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let metadata = fs::metadata(&canonical)?;
        let name = canonical
            .file_name()
            .and_then(|file| file.to_str())
            .unwrap_or_default()
            .to_owned();
        // Local listings (`hipfire list`, `hipfire diag`) keep sidecars: a
        // pulled draft or a vision tower is a real file the operator may want
        // to see and `hipfire rm`. The gate is `is_listable_model_file` — a
        // model suffix OR a registry entry's own `file`, which admits both
        // kinds. `/v1/models` narrows it further with `is_standalone_model` —
        // see `serve::http`.
        if !is_listable_model_file(&name, registry, &grandfather_table()) {
            continue;
        }
        // A retired model is the installed artifact itself (same file as
        // `models/<pin.file>`), never a same-named lookalike elsewhere.
        let retired = retired_pins
            .iter()
            .find(|pin| same_installed_file(&paths.models.join(pin.file), &canonical))
            .copied();
        let registry_tag = registry
            .models
            .iter()
            .find_map(|(tag, model)| {
                (model.file == name
                    || grandfather_pin(model)
                        .is_some_and(|pin| name == pin.canonical_file || name == pin.legacy_file))
                .then(|| tag.clone())
            })
            .or_else(|| retired.map(|pin| pin.tag.to_owned()))
            .or_else(|| catalog_tags.get(&canonical).cloned());
        let retired = retired.is_some_and(|pin| registry_tag.as_deref() == Some(pin.tag));
        let created = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0);
        models.push(LocalModel {
            name,
            path: canonical,
            size_bytes: metadata.len(),
            created,
            registry_tag,
            retired,
        });
    }
    models.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(models)
}

/// Enumerate the model files in the models dir, plus the same one level down
/// inside subdirectories.
///
/// The test is [`is_listable_model_file`], not a bare suffix match: several
/// registry tiers ship files whose suffix is absent from `MODEL_SUFFIXES`
/// (`qwen3.8-27b.mq3-xt`, `-pro`, `.bq1`, `.tq2`), and a suffix-only scan drops
/// exactly those models — including the tier a client is looking for.
///
/// Sidecars are NOT filtered here. This scan backs `hipfire list`, `hipfire
/// diag`, and name resolution, all of which should still see a pulled draft or
/// a vision tower. The serve discovery surface narrows it — see
/// [`crate::serve::http`] and [`hipfire_registry::is_standalone_model`].
pub(crate) fn local_model_paths(paths: &Paths, registry: &RegistryV1) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(&paths.models) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error).context("failed to list model directory"),
    };
    let mut models = Vec::new();
    let pins = grandfather_table();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if path
                .file_name()
                .and_then(|file| file.to_str())
                .is_some_and(|file| is_listable_model_file(file, registry, &pins))
            {
                models.push(path);
            }
            continue;
        }
        if !path.is_dir() {
            continue;
        }
        let Ok(children) = fs::read_dir(path) else {
            continue;
        };
        models.extend(children.flatten().map(|entry| entry.path()).filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|file| file.to_str())
                    .is_some_and(|file| is_listable_model_file(file, registry, &pins))
        }));
    }
    Ok(models)
}

pub(crate) fn pull_command(paths: &Paths, args: PullArgs) -> Result<()> {
    let pulled = pull_with_sidecar_pins(
        paths,
        args,
        TRUNK_OVERLAY_PINS,
        EXPERT_HOTSET_PINS,
        &grandfather_table(),
    )?;
    if let Some(model) = pulled {
        println!("{}", model.display());
    }
    Ok(())
}

/// [`pull_command`] with explicit client-side pin tables (tests inject fixture
/// pins). Returns the path the loader opens (the split base, or the served
/// grandfathered monolithic); `None` when the user cancelled.
fn pull_with_sidecar_pins(
    paths: &Paths,
    args: PullArgs,
    overlay_pins: &[TrunkOverlayPin],
    hotset_pins: &[ExpertHotsetPin],
    grandfather_pins: &[GrandfatheredMonolithic],
) -> Result<Option<PathBuf>> {
    // A retired tag is refused before any credential check or network work,
    // with or without --force: the registry no longer publishes it.
    if let Some(pin) = hipfire_registry::retired_model_named(&args.model, &retired_table()) {
        bail!("{}", pin.refusal());
    }
    // A malformed HF_TOKEN would otherwise surface only as the HTTP client's
    // generic "failed to parse header value" after the registry work, or as a
    // mere warning on an optional sidecar: refuse it before anything is fetched.
    hipfire_config::http_auth::hf_authorization()?;
    let loaded = load_registry(&paths.registry);
    let (tag, entry) = loaded
        .registry
        .model(&args.model)
        .ok_or_else(|| unknown_model_error(&loaded.registry, &args.model))?;
    if entry.repo.is_empty() {
        bail!(
            "cannot pull {tag}: registry entry is local-only; place {} in {}",
            entry.file,
            paths.models.display()
        );
    }
    fs::create_dir_all(&paths.models)
        .with_context(|| format!("failed to create {}", paths.models.display()))?;
    let flags = pull_select::PullFlags {
        ple: args.ple,
        draft: args.draft_flag(),
        force: args.force,
    };
    let choice = if pull_select::is_interactive(io::stdin().is_terminal(), args.yes) {
        let host = pull_select::HostFacts::detect(paths);
        let present = |file: &str, size: Option<u64>| {
            pull_select::file_present(&paths.models.join(file), size)
        };
        let context = pull_select::PullContext {
            registry: &loaded.registry,
            host: &host,
            overlay_pins,
            hotset_pins,
            grandfather_pins,
            present: &present,
        };
        let mut prompt = pull_select::LinePrompt::new(io::stdin().lock(), io::stderr());
        match pull_select::select_pull(&context, tag, entry, &flags, &mut prompt)? {
            Some(choice) => choice,
            None => {
                eprintln!("Pull cancelled.");
                return Ok(None);
            }
        }
    } else {
        pull_select::PullChoice::defaults(tag, entry, &flags)
    };
    let (tag, entry) = (choice.tag, choice.entry);
    let destination = paths.models.join(&entry.file);
    let migrated = if args.force {
        None
    } else {
        migrate_flash_next_xts(paths, entry)?
    };
    // A verified local monolithic that serves this split entry (never under --force).
    let mut grandfathered: Option<(&GrandfatheredMonolithic, PathBuf)> = None;
    let needs_base = if args.force {
        true
    } else if let Some(path) = migrated {
        eprintln!("Already downloaded: {}", path.display());
        false
    } else if destination.exists()
        && existing_artifact_valid(
            &VerifiedCache::in_home(&paths.root),
            &destination,
            entry.sha256.as_deref(),
            entry.size_bytes,
        )
    {
        eprintln!("Already downloaded: {}", destination.display());
        false
    } else {
        grandfathered = grandfathered_local(paths, entry, grandfather_pins);
        if let Some((_, served)) = &grandfathered {
            eprintln!(
                "Already satisfied (monolithic): {} (serves {tag} without the split base)",
                served.display()
            );
            false
        } else {
            if destination.exists() {
                eprintln!("Refreshing stale artifact: {}", destination.display());
            }
            true
        }
    };
    // The artifact the loader opens: the served monolithic, else the split base.
    let model_path = grandfathered
        .as_ref()
        .map(|(_, served)| served.clone())
        .unwrap_or_else(|| destination.clone());
    let ple_plan = entry.has_ple_sidecars().then(|| {
        let base = match &grandfathered {
            Some((pin, served)) => ple::PullBase::Served {
                pin,
                path: served.as_path(),
            },
            None if needs_base => ple::PullBase::Download {
                monolithic: hipfire_registry::grandfathered_monolithic_in(entry, grandfather_pins),
            },
            None => ple::PullBase::Present,
        };
        ple::announce_pull_plan(paths, entry, choice.ple, base)
    });
    if needs_base {
        let url = artifact_url(entry, &entry.file);
        eprintln!("Pulling {tag} ({:.2} GB)...", entry.size_gb);
        download_verified(
            &VerifiedCache::in_home(&paths.root),
            &url,
            &destination,
            entry.sha256.as_deref(),
            entry.size_bytes,
            false,
        )?;
        if entry.is_flash_next_xts() {
            // The download has already verified the canonical payload.
            let legacy = paths
                .models
                .join(hipfire_registry::FLASH_NEXT_XTS_LEGACY_FILE);
            if install_verified_link(&destination, &legacy)?
                && !same_installed_file(&destination, &legacy)
            {
                verified_metadata(
                    &VerifiedCache::in_home(&paths.root),
                    &legacy,
                    entry.sha256.as_deref().expect("pinned digest"),
                    entry.size_bytes.expect("pinned size"),
                )?;
            }
        }
    }
    if let Some(plan) = &ple_plan {
        for (format, sidecar) in &plan.sidecars {
            ple::fetch_ple_sidecar(
                &VerifiedCache::in_home(&paths.root),
                entry,
                *format,
                sidecar,
                &paths.models,
                &model_path,
                args.force,
            )?;
        }
    }
    for declared in pull_select::declared_sidecars(entry) {
        if declared.spec && !choice.draft {
            eprintln!(
                "  Skipping {} sidecar (draft/MTP not selected): {}",
                declared.label, declared.sidecar.file
            );
            continue;
        }
        fetch_registry_sidecar(
            &VerifiedCache::in_home(&paths.root),
            declared.label,
            &entry.repo,
            declared.sidecar,
            &paths.models.join(&declared.sidecar.file),
            args.force,
        );
    }
    // The trunk overlay and expert hot set are not registry fields: the client
    // carries their pins, keyed by the exact base artifact (`TRUNK_OVERLAY_PINS`,
    // `EXPERT_HOTSET_PINS`). A served grandfathered monolithic is a different
    // artifact from the split base, so its pins are looked up by its own identity
    // and the files install beside it.
    let served_identity = grandfathered.as_ref().map(|(pin, _)| ModelEntry {
        file: pin.canonical_file.to_owned(),
        sha256: Some(pin.sha256.to_owned()),
        size_bytes: Some(pin.size_bytes),
        ple: BTreeMap::new(),
        ..entry.clone()
    });
    let identity = served_identity.as_ref().unwrap_or(entry);
    if let Some(overlay) = hipfire_registry::trunk_overlay_in(identity, overlay_pins) {
        if pull_select::trunk_overlay_wanted(&ple::local_gpu_arches()) {
            fetch_registry_sidecar(
                &VerifiedCache::in_home(&paths.root),
                "Trunk overlay",
                overlay.repo,
                &overlay.sidecar,
                &hipfire_config::memory::trunk_overlay_sidecar_path(&model_path),
                args.force,
            );
        } else {
            eprintln!(
                "  Skipping Trunk overlay sidecar (no gfx1151 GPU detected; the loader applies it only on gfx1151, or anywhere with HIPFIRE_QWEN4_TRUNK_OVERLAY=1): {}",
                overlay.sidecar.file
            );
        }
    }
    if choice.expert_hotset {
        if let Some(hotset) = hipfire_registry::expert_hotset_in(identity, hotset_pins) {
            fetch_registry_sidecar(
                &VerifiedCache::in_home(&paths.root),
                "Expert hot set",
                hotset.repo,
                &hotset.sidecar,
                &hipfire_config::memory::expert_hotset_sidecar_path(&model_path),
                args.force,
            );
        }
    }
    for (name, sidecar) in &entry.heads {
        let destination = paths.models.join(&sidecar.file);
        if destination.exists() && !args.force {
            if existing_artifact_valid(
                &VerifiedCache::in_home(&paths.root),
                &destination,
                sidecar.sha256.as_deref(),
                sidecar.size_bytes,
            ) {
                eprintln!("  head {name} already present: {}", sidecar.file);
                continue;
            }
            eprintln!("  head {name} stale, refreshing: {}", sidecar.file);
        }
        eprintln!("  Fetching head {name}: {}", sidecar.file);
        download_verified(
            &VerifiedCache::in_home(&paths.root),
            &artifact_url(entry, &sidecar.file),
            &destination,
            sidecar.sha256.as_deref(),
            sidecar.size_bytes,
            true,
        )?;
    }
    Ok(Some(model_path))
}

/// Fetch one optional sidecar to its local destination: an existing valid file is
/// kept, a stale one (or any under `force`) is refreshed through
/// [`download_verified`]. A failed download only warns; the base stays usable.
fn fetch_registry_sidecar(
    cache: &VerifiedCache,
    label: &str,
    repo: &str,
    sidecar: &Sidecar,
    destination: &Path,
    force: bool,
) {
    if destination.exists() && !force {
        if existing_artifact_valid(
            cache,
            destination,
            sidecar.sha256.as_deref(),
            sidecar.size_bytes,
        ) {
            eprintln!("  {label} sidecar already present: {}", sidecar.file);
            return;
        }
        eprintln!("  {label} sidecar stale, refreshing: {}", sidecar.file);
    }
    eprintln!("  Fetching {label} sidecar: {}", sidecar.file);
    if let Err(error) = download_verified(
        cache,
        &artifact_url_in(repo, &sidecar.file),
        destination,
        sidecar.sha256.as_deref(),
        sidecar.size_bytes,
        true,
    ) {
        eprintln!("  warning: {label} sidecar unavailable: {error:#}");
    }
}

pub(crate) fn artifact_url(entry: &ModelEntry, file: &str) -> String {
    artifact_url_in(&entry.repo, file)
}

fn artifact_url_in(repo: &str, file: &str) -> String {
    let base = env::var("HIPFIRE_HF_BASE")
        .or_else(|_| env::var("HF_ENDPOINT"))
        .unwrap_or_else(|_| "https://huggingface.co".into());
    format!(
        "{}/{}/resolve/main/{}",
        base.trim_end_matches('/'),
        repo,
        file
    )
}

#[cfg(test)]
thread_local! { static DOWNLOAD_INVOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

pub(crate) fn download_verified(
    cache: &VerifiedCache,
    url: &str,
    destination: &Path,
    expected_sha256: Option<&str>,
    expected_size: Option<u64>,
    quiet: bool,
) -> Result<()> {
    #[cfg(test)]
    DOWNLOAD_INVOCATIONS.with(|count| count.set(count.get() + 1));
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(24 * 60 * 60)))
        .http_status_as_error(false)
        .build()
        .into();
    let mut request = agent.get(url);
    if let Some(authorization) = hipfire_config::http_auth::hf_authorization()? {
        request = request.header("Authorization", &authorization);
    }
    let mut response = request
        .call()
        .map_err(|error| anyhow!("download request failed: {error}"))?;
    if !response.status().is_success() {
        bail!("download returned HTTP {} for {url}", response.status());
    }
    let announced = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let temporary = destination.with_extension(format!("part.{}", std::process::id()));
    let mut output = fs::File::create(&temporary)
        .with_context(|| format!("failed to create {}", temporary.display()))?;
    let mut reader = response.body_mut().as_reader();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut downloaded = 0_u64;
    let started = Instant::now();
    let mut last_report = Instant::now();
    let result = (|| -> Result<()> {
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            output.write_all(&buffer[..count])?;
            hasher.update(&buffer[..count]);
            downloaded += count as u64;
            if !quiet && last_report.elapsed() >= Duration::from_millis(500) {
                report_progress(downloaded, announced.or(expected_size), started.elapsed());
                last_report = Instant::now();
            }
        }
        output.sync_all()?;
        if !quiet {
            report_progress(downloaded, announced.or(expected_size), started.elapsed());
            eprintln!();
        }
        if let Some(expected) = expected_size {
            if downloaded != expected {
                bail!("size mismatch: expected {expected} bytes, received {downloaded}");
            }
        }
        let actual = format!("{:x}", hasher.finalize());
        if let Some(expected) = expected_sha256 {
            if !actual.eq_ignore_ascii_case(expected) {
                bail!("SHA-256 mismatch: expected {expected}, received {actual}");
            }
        }
        fs::rename(&temporary, destination).with_context(|| {
            format!(
                "failed to install {} as {}",
                temporary.display(),
                destination.display()
            )
        })?;
        // The digest above is the proof: record it against the installed file's
        // stat so later loads and proof-requiring checks never re-hash it.
        if expected_sha256.is_some() {
            if let Ok(installed) = fs::metadata(destination) {
                cache.remember(
                    destination,
                    verified_cache::StatKey::of(&installed),
                    &actual,
                );
            }
        }
        eprintln!(
            "  Saved: {} ({:.3} GB, sha256 {})",
            destination.display(),
            downloaded as f64 / 1e9,
            actual
        );
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn report_progress(downloaded: u64, total: Option<u64>, elapsed: Duration) {
    let rate = downloaded as f64 / elapsed.as_secs_f64().max(0.001);
    if let Some(total) = total.filter(|total| *total > 0) {
        let percent = downloaded as f64 / total as f64 * 100.0;
        let remaining = total.saturating_sub(downloaded) as f64 / rate.max(1.0);
        eprint!(
            "\r  {:>6.2}%  {:.2}/{:.2} GB  {:.1} MB/s  ETA {:.0}s",
            percent,
            downloaded as f64 / 1e9,
            total as f64 / 1e9,
            rate / 1e6,
            remaining
        );
    } else {
        eprint!(
            "\r  {:.2} GB  {:.1} MB/s",
            downloaded as f64 / 1e9,
            rate / 1e6
        );
    }
    let _ = std::io::stderr().flush();
}

/// True when `path` is the pinned artifact: size, and SHA-256 when pinned. A
/// file the cache already proved (same stat, same digest) is not hashed again;
/// a file hashed here is recorded, so one verification serves every later
/// load, pull and proof-requiring check ([`verified_metadata`]).
pub(crate) fn existing_artifact_valid(
    cache: &VerifiedCache,
    path: &Path,
    expected_sha256: Option<&str>,
    expected_size: Option<u64>,
) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if let Some(expected) = expected_size {
        if metadata.len() != expected {
            return false;
        }
    }
    match expected_sha256 {
        Some(expected) => verified_metadata(cache, path, expected, metadata.len()).is_ok(),
        None => true,
    }
}

fn rm_command(paths: &Paths, args: RmArgs) -> Result<()> {
    let loaded = load_registry(&paths.registry);
    rm_with_registry(paths, &loaded.registry, args)
}

/// Remove one model and its sidecars. A declared DFlash draft sidecar is
/// shared: several registry entries can name the same `dflash.file` (e.g.
/// `qwen3.8:27b`, `qwen3.8:27b-mq4-pro`, and `qwen3.8:27b-mq4-xt` all declare
/// `qwen38-27b-dflash-mq4.hfq`). Deleting it while a sibling declarer is
/// still on disk leaves those siblings without their paired draft under `dflash_mode=auto`
/// or refusing to load under `on`, so the sidecar is kept — with one stderr
/// line — whenever any OTHER entry declaring the same file still has its own
/// target file present in the models dir. The `vision` tower sidecar follows
/// the same shared-keeper rule (every `qwen3.8:27b*` tier declares the one
/// `qwen3.8-27b-vision.hfq`).
fn rm_with_registry(paths: &Paths, registry: &RegistryV1, args: RmArgs) -> Result<()> {
    rm_with_sidecar_pins(
        paths,
        registry,
        TRUNK_OVERLAY_PINS,
        EXPERT_HOTSET_PINS,
        args,
    )
}

/// [`rm_with_registry`] with explicit client-side pin tables (tests inject fixture pins).
fn rm_with_sidecar_pins(
    paths: &Paths,
    registry: &RegistryV1,
    overlay_pins: &[TrunkOverlayPin],
    hotset_pins: &[ExpertHotsetPin],
    args: RmArgs,
) -> Result<()> {
    let card = model_card(paths, registry, &args.model);
    let resolved = card.pair();
    // Removal acts on the artifact itself, so it resolves symlinks (the
    // selected path `find_local_model_path` returns keeps them for sidecar lookup).
    let path = find_local_model_path(paths, registry, &args.model)
        .map(|selected| fs::canonicalize(&selected).unwrap_or(selected))
        .unwrap_or_else(|| paths.models.join(&args.model));
    if !path.is_file() {
        if let Some((tag, _)) = card.pair() {
            bail!("{tag} is not downloaded; nothing to remove");
        }
        if !args.model.contains(std::path::MAIN_SEPARATOR) {
            return Err(unknown_model_error(registry, &args.model));
        }
        bail!("model not found: {}", path.display());
    }
    let mut targets = BTreeSet::from([path.clone()]);
    // The other spelling of a pinned monolithic is a hard link of the same
    // inode: leaving it would keep the tag resolving (and the disk full).
    targets.extend(ple::monolithic_twins(&path));
    // A shared DFlash sidecar that must survive this removal: (file, keepers).
    let mut kept_sidecar: Option<(String, String)> = None;
    // A shared vision-tower sidecar under the same rule: every `qwen3.8:27b*`
    // tier declares `qwen3.8-27b-vision.hfq`, so the keeper check is copied
    // from DFlash exactly.
    let mut kept_vision: Option<(String, String)> = None;
    // An XDNA spillover archive under the same shared-keeper rule: archives
    // bind their model hash, so sharing across entries is unlikely, but a
    // shared file must survive exactly like a shared DFlash draft.
    let mut kept_xdna: Option<(String, String)> = None;
    // A shared MTP head under the same rule: every `qwen3.8:27b*` tier declares
    // `qwen3.8-27b.mtp`, and the stem sweep below would otherwise catch it too.
    let mut kept_mtp: Option<(String, String)> = None;
    if let Some((tag, entry)) = resolved {
        targets.extend(
            [
                &entry.triattn,
                &entry.dspark,
                &entry.t5,
                &entry.clip,
                &entry.vae,
            ]
            .into_iter()
            .flatten()
            .map(|sidecar| paths.models.join(&sidecar.file))
            .filter(|path| path.is_file()),
        );
        if let Some(overlay) = hipfire_registry::trunk_overlay_in(entry, overlay_pins) {
            let overlay_path = paths.models.join(&overlay.sidecar.file);
            if overlay_path.is_file() {
                targets.insert(overlay_path);
            }
        }
        if let Some(hotset) = hipfire_registry::expert_hotset_in(entry, hotset_pins) {
            let hotset_path = paths.models.join(&hotset.sidecar.file);
            if hotset_path.is_file() {
                targets.insert(hotset_path);
            }
        }
        if let Some(sidecar) = entry.dflash.as_ref() {
            let sidecar_path = paths.models.join(&sidecar.file);
            if sidecar_path.is_file() {
                // `models` is a BTreeMap, so keepers list in sorted tag order.
                let keepers: Vec<&str> = registry
                    .models
                    .iter()
                    .filter(|(other_tag, other)| {
                        other_tag.as_str() != tag
                            && other.file != entry.file
                            && other
                                .dflash
                                .as_ref()
                                .is_some_and(|other_sidecar| other_sidecar.file == sidecar.file)
                            && paths.models.join(&other.file).is_file()
                    })
                    .map(|(other_tag, _)| other_tag.as_str())
                    .collect();
                if keepers.is_empty() {
                    targets.insert(sidecar_path);
                } else {
                    kept_sidecar = Some((sidecar.file.clone(), keepers.join(", ")));
                }
            }
        }
        if let Some(sidecar) = entry.vision.as_ref() {
            let sidecar_path = paths.models.join(&sidecar.file);
            if sidecar_path.is_file() {
                // `models` is a BTreeMap, so keepers list in sorted tag order.
                let keepers: Vec<&str> = registry
                    .models
                    .iter()
                    .filter(|(other_tag, other)| {
                        other_tag.as_str() != tag
                            && other.file != entry.file
                            && other
                                .vision
                                .as_ref()
                                .is_some_and(|other_sidecar| other_sidecar.file == sidecar.file)
                            && paths.models.join(&other.file).is_file()
                    })
                    .map(|(other_tag, _)| other_tag.as_str())
                    .collect();
                if keepers.is_empty() {
                    targets.insert(sidecar_path);
                } else {
                    kept_vision = Some((sidecar.file.clone(), keepers.join(", ")));
                }
            }
        }
        if let Some(sidecar) = entry.xdna.as_ref() {
            let sidecar_path = paths.models.join(&sidecar.file);
            if sidecar_path.is_file() {
                // `models` is a BTreeMap, so keepers list in sorted tag order.
                let keepers: Vec<&str> = registry
                    .models
                    .iter()
                    .filter(|(other_tag, other)| {
                        other_tag.as_str() != tag
                            && other.file != entry.file
                            && other
                                .xdna
                                .as_ref()
                                .is_some_and(|other_sidecar| other_sidecar.file == sidecar.file)
                            && paths.models.join(&other.file).is_file()
                    })
                    .map(|(other_tag, _)| other_tag.as_str())
                    .collect();
                if keepers.is_empty() {
                    targets.insert(sidecar_path);
                } else {
                    kept_xdna = Some((sidecar.file.clone(), keepers.join(", ")));
                }
            }
        }
        if let Some(sidecar) = entry.mtp.as_ref() {
            let sidecar_path = paths.models.join(&sidecar.file);
            if sidecar_path.is_file() {
                // `models` is a BTreeMap, so keepers list in sorted tag order.
                let keepers: Vec<&str> = registry
                    .models
                    .iter()
                    .filter(|(other_tag, other)| {
                        other_tag.as_str() != tag
                            && other.file != entry.file
                            && other
                                .mtp
                                .as_ref()
                                .is_some_and(|other_sidecar| other_sidecar.file == sidecar.file)
                            && paths.models.join(&other.file).is_file()
                    })
                    .map(|(other_tag, _)| other_tag.as_str())
                    .collect();
                if keepers.is_empty() {
                    targets.insert(sidecar_path);
                } else {
                    kept_mtp = Some((sidecar.file.clone(), keepers.join(", ")));
                }
            }
        }
        targets.extend(
            entry
                .heads
                .values()
                .map(|sidecar| paths.models.join(&sidecar.file))
                .filter(|path| path.is_file()),
        );
    }
    if let (Some(parent), Some(file)) = (
        path.parent(),
        path.file_name().and_then(|file| file.to_str()),
    ) {
        let stem = file.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(file);
        if let Ok(entries) = fs::read_dir(parent) {
            targets.extend(
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|candidate| {
                        let Some(name) = candidate.file_name().and_then(|name| name.to_str())
                        else {
                            return false;
                        };
                        candidate.is_file()
                            && name != file
                            && ((name.starts_with(&format!("{stem}.triattn"))
                                && name.ends_with(".bin"))
                                || (name.starts_with(stem)
                                    && (name.ends_with(".mtp")
                                        || name.contains("-mtp.")
                                        || name.contains("-dspark."))))
                    }),
            );
        }
    }
    // The PLE row sidecars (and their hot-row packs) ride with the model file
    // that was resolved, whichever layout it is.
    targets.extend(
        ple::ple_sidecar_paths(&path, resolved.map(|(_, entry)| entry))
            .into_iter()
            .filter(|sidecar| sidecar.is_file()),
    );
    // A family-shared PLE file (`qwen3.8-flash-next.ple-q4`) is the link target
    // of every Flash-Next split base's `<base>.ple-qN` (and of a served
    // grandfathered monolithic's `<monolithic>.ple-q4`): it goes with the last
    // file that still links it, like the other shared sidecars.
    let mut kept_ple: Vec<(String, String)> = Vec::new();
    if let Some((tag, entry)) = resolved.filter(|(_, entry)| {
        let name = path.file_name();
        name.is_some_and(|name| name == std::ffi::OsStr::new(&entry.file))
            || grandfather_pin(entry).is_some_and(|pin| {
                name.is_some_and(|name| {
                    name == std::ffi::OsStr::new(pin.canonical_file)
                        || name == std::ffi::OsStr::new(pin.legacy_file)
                })
            })
    }) {
        for (key, sidecar) in &entry.ple {
            if !ple::is_shared_slot(entry, key, sidecar) {
                continue;
            }
            let shared = paths.models.join(&sidecar.file);
            if !shared.is_file() {
                continue;
            }
            let mut keepers: Vec<&str> = registry
                .models
                .iter()
                .filter(|(other_tag, other)| {
                    other_tag.as_str() != tag
                        && other.file != entry.file
                        && other
                            .ple_sidecar(key)
                            .is_some_and(|other_sidecar| other_sidecar.file == sidecar.file)
                        && ple::ple_sidecar_path_for_key(&paths.models.join(&other.file), key)
                            .is_file()
                })
                .map(|(other_tag, _)| other_tag.as_str())
                .collect();
            // A served monolithic's own split base may be installed next to it.
            let own_base = paths.models.join(&entry.file);
            let removing_base = path
                .file_name()
                .is_some_and(|name| name == std::ffi::OsStr::new(&entry.file));
            if !removing_base
                && own_base.is_file()
                && ple::ple_sidecar_path_for_key(&own_base, key).is_file()
            {
                keepers.push(tag);
            }
            if keepers.is_empty() {
                targets.insert(shared);
            } else {
                kept_ple.push((sidecar.file.clone(), keepers.join(", ")));
            }
        }
    }
    // A trunk overlay rides with its base: the pinned overlay file (also
    // reached through a legacy-named link) and `<resolved file>.trunk-overlay`.
    let overlay = hipfire_config::memory::trunk_overlay_sidecar_path(&path);
    if overlay.is_file() {
        targets.insert(overlay);
    }
    let hotset = hipfire_config::memory::expert_hotset_sidecar_path(&path);
    if hotset.is_file() {
        targets.insert(hotset);
    }
    if let Some((file, _)) = kept_mtp.as_ref() {
        // The stem sweep matched the shared head beside the trunk; keep it.
        let kept = paths.models.join(file);
        if let Ok(canonical) = fs::canonicalize(&kept) {
            targets.remove(&canonical);
        }
        targets.remove(&kept);
    }
    if !args.yes {
        eprint!("Remove {} file(s)? [y/N] ", targets.len());
        std::io::stderr().flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            println!("cancelled");
            return Ok(());
        }
    }
    for target in targets {
        fs::remove_file(&target)
            .with_context(|| format!("failed to remove {}", target.display()))?;
        println!("removed {}", target.display());
    }
    if let Some((file, keepers)) = kept_sidecar {
        eprintln!("keeping DFlash sidecar {file}: still declared by {keepers}");
    }
    if let Some((file, keepers)) = kept_vision {
        eprintln!("keeping Vision sidecar {file}: still declared by {keepers}");
    }
    if let Some((file, keepers)) = kept_mtp {
        eprintln!("keeping MTP sidecar {file}: still declared by {keepers}");
    }
    if let Some((file, keepers)) = kept_xdna {
        eprintln!("keeping XDNA sidecar {file}: still declared by {keepers}");
    }
    for (file, keepers) in kept_ple {
        eprintln!("keeping shared PLE sidecar {file}: still linked by {keepers}");
    }
    Ok(())
}

fn run_command(paths: &Paths, args: RunArgs) -> Result<()> {
    let loaded_registry = load_registry(&paths.registry);
    let registry = &loaded_registry.registry;
    let card = model_card(paths, registry, &args.model);
    let (canonical, entry) = (card.tag().map(str::to_owned), card.entry());
    let mut model_path = resolve_model_path(paths, registry, &args.model)?;
    if model_path.is_none() {
        if card.pullable().is_some() {
            eprintln!(
                "Model not found locally. Pulling {}...",
                canonical.as_deref().unwrap_or(&args.model)
            );
            pull_command(paths, PullArgs::defaults(args.model.clone()))?;
            // pull already verified either the downloaded canonical or legacy fallback.
            model_path = find_model_path(paths, registry, &args.model);
        }
    }
    let model_path =
        model_path.ok_or_else(|| model_not_found_error(paths, registry, &args.model))?;
    if let Some(image) = &args.image {
        if !image.is_file() {
            bail!("image not found: {}", image.display());
        }
    }
    if let Some(draft) = &args.model_draft {
        if !draft.is_file() {
            let fix = match card.pullable().filter(|entry| entry.dflash.is_some()).and(card.tag()) {
                Some(tag) => format!(
                    "fetch the registry draft: hipfire pull {tag} (then drop --model-draft to use it)"
                ),
                None => "pass an existing draft with -md <path>, or run without it: --spec off"
                    .to_owned(),
            };
            bail!("DFlash draft not found: {}\n  {fix}", draft.display());
        }
    }
    if let Some(vision) = &args.vision {
        if !vision.is_file() {
            let pull = match card
                .pullable()
                .filter(|entry| entry.vision.is_some())
                .and(card.tag())
            {
                Some(tag) => format!(", fetch the registry vision tower: hipfire pull {tag}"),
                None => String::new(),
            };
            bail!(
                "vision sidecar not found: {}\n  pass an existing --vision <path>{pull}, or disable \
                 vision: HIPFIRE_VISION_MODE=off (hipfire config set vision.mode off)",
                vision.display()
            );
        }
    }
    if args
        .dspark_conf_threshold
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        bail!("--dspark-conf-threshold must be between 0 and 1");
    }

    let resolved = resolved_for_model(paths, &args.model, canonical.as_deref(), entry)?;
    let configured_max_tokens = config_u64(&resolved, "generation.max_tokens")?;
    let max_tokens = args.max_tokens.unwrap_or(configured_max_tokens);
    if max_tokens == 0 || max_tokens > 393_216 {
        bail!("--max-tokens must be between 1 and 393216");
    }
    let temperature = request_f64(&resolved, "generation.temperature", args.temp)?;
    let top_p = request_f64(&resolved, "generation.top_p", args.top_p)?;
    let top_k = request_u64(&resolved, "generation.top_k", None)?;
    let min_p = request_f64(&resolved, "generation.min_p", None)?;
    let presence_penalty = request_f64(&resolved, "generation.presence_penalty", None)?;
    let repeat_penalty = request_f64(&resolved, "generation.repeat_penalty", args.repeat_penalty)?;
    let system_prompt = request_string(&resolved, "prompt.system", args.system.clone())?
        .filter(|value| !value.is_empty());
    if temperature.is_some_and(|value| !(0.0..=2.0).contains(&value)) {
        bail!("--temp must be between 0 and 2");
    }
    if top_p.is_some_and(|value| !(0.0 < value && value <= 1.0)) {
        bail!("--top-p must be in (0, 1]");
    }
    if repeat_penalty.is_some_and(|value| !(1.0..=3.0).contains(&value)) {
        bail!("--repeat-penalty must be between 1 and 3");
    }

    let prompt = if args.prompt.is_empty() {
        if args.image.is_some() {
            "Describe this image.".to_owned()
        } else {
            "Hello".to_owned()
        }
    } else {
        args.prompt.join(" ")
    };
    let host = config_string(&resolved, "serve.host")?;
    let port = serve_probe_port(paths, config_u64(&resolved, "serve.port")? as u16);
    let endpoint = local_service_endpoint(&resolved, &host, port)?;
    // A resolved `topology.pp` above 1 (or one that cannot be read) must not be
    // answered by a resident service that may hold a different topology; the
    // local daemon load then applies it or refuses it by name.
    let force_local = run_should_force_local(&args)
        || config_u64(&resolved, "topology.pp").map_or(true, |pp| pp > 1);
    if !force_local && service_ready(&endpoint, Duration::from_millis(150)) {
        return run_via_http(
            &endpoint,
            &args.model,
            &prompt,
            system_prompt.as_deref(),
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            repeat_penalty,
            max_tokens,
            args.json,
            args.no_stream,
            args.stats,
        );
    }
    ple::ensure_ple_sidecar(
        entry,
        canonical.as_deref(),
        &model_path,
        ple::request_from_config(&resolved),
        &ple::load_target_arch(&resolved),
        true,
    )?;

    let daemon = find_daemon(paths).ok_or_else(|| daemon_not_found_error(paths))?;
    let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
    let mut engine = Engine::spawn_configured(
        &daemon,
        &BTreeMap::new(),
        &process_config,
        Some(&model_path),
    )?;
    engine.ping()?;
    let mut params = load_params_with_overrides(
        &resolved,
        entry,
        &paths.models,
        &model_path,
        max_tokens,
        canonical.as_deref(),
        &LoadOverrides {
            kv_mode: args.kv_mode.as_deref(),
            kv_backend: args.kv_backend.as_deref(),
            kv_k: args.kv_k.as_deref(),
            kv_v: args.kv_v.as_deref(),
            speculation: args.speculation.as_deref(),
            head: args.head.as_deref(),
            explicit_draft: args.model_draft.is_some(),
        },
    )?;
    // Tensor-parallel load: forward `tp` like `bench --tp` / `serve --tp` so the
    // daemon takes the EP/dense-TP route instead of the single-GPU one.
    if let Some(tp) = args.tp.filter(|&tp| tp > 1) {
        params["tp"] = serde_json::json!(tp);
    }
    // `--max-seq` replaces any configured value, exactly as `bench --max-seq`.
    if let Some(max_seq) = args.max_seq {
        params["max_seq"] = serde_json::json!(max_seq);
    }
    if let Some(draft) = &args.model_draft {
        params["draft"] = serde_json::json!(draft.display().to_string());
        if args.speculation.is_none() {
            apply_speculation_selector(&mut params, "dflash")?;
        }
    }
    // Registry sidecar for a final auto/on selector the config-time
    // load_params could not see (config-off + `run --spec dflash`); a
    // no-op when load_params already resolved or an explicit draft won.
    resolve_dflash_sidecar(
        &mut params,
        entry,
        &paths.models,
        &model_path,
        canonical.as_deref(),
    )?;
    resolve_mtp_sidecar(
        &mut params,
        entry,
        &paths.models,
        &model_path,
        &args.model,
        canonical.as_deref(),
    );
    if let Some(vision) = &args.vision {
        // Forwarded in every mode; the daemon's `vision_mode=off` gate decides
        // and can then name the sidecar it declined on an image request.
        params["vision"] = serde_json::json!(vision.display().to_string());
    }
    if let Some(window) = args.draft_max {
        if !(1..=32).contains(&window) {
            bail!("--draft-max must be between 1 and 32");
        }
        match args.speculation.as_deref().unwrap_or("auto") {
            "ngram" => params["ngram_k"] = serde_json::json!(window),
            "mtp" => params["mtp_k"] = serde_json::json!(window),
            _ => {
                params["mtp_k"] = serde_json::json!(window);
                params["ngram_k"] = serde_json::json!(window);
            }
        }
    }
    if let Some(value) = args.dspark_conf_threshold {
        params["dspark_conf_threshold"] = serde_json::json!(value);
    }
    let loaded = engine.load(&model_path, params)?;
    let mut spill = spill::attach_for_engine(
        &engine,
        process_config.legacy_value("HIPFIRE_DEVICES").as_deref(),
    );
    if !args.json {
        eprintln!(
            "[{}] {}d {}L {} vocab",
            loaded
                .get("arch")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown"),
            loaded
                .get("dim")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            loaded
                .get("layers")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            loaded
                .get("vocab")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        );
    }

    let mut request = serde_json::json!({
        "type": "generate",
        "id": "run",
        "prompt": prompt,
        "max_tokens": max_tokens,
        // `Engine::generate` rejects a request without `attempt_id`
        // (hipfire-client lib.rs:557 -> "generate request missing attempt_id"),
        // and `hipfire run` never set one, so EVERY `hipfire run` failed with a
        // daemon protocol error. `run` is a one-shot, non-retrying caller, so a
        // literal 1 is correct — same as `bench_generate_request` (main.rs:6407).
        // The retrying serve path threads a real counter instead (main.rs:4234).
        "attempt_id": 1,
    });
    insert_optional_f64(&mut request, "temperature", temperature);
    insert_optional_f64(&mut request, "top_p", top_p);
    insert_optional_u64(&mut request, "top_k", top_k);
    insert_optional_f64(&mut request, "min_p", min_p);
    insert_optional_f64(&mut request, "presence_penalty", presence_penalty);
    insert_optional_f64(&mut request, "repeat_penalty", repeat_penalty);
    if args.max_tokens.is_none() {
        // The configured default is only a ceiling: the daemon fits it to the
        // context left after the prompt, as serve does for an omitted value.
        request["max_tokens_fit"] = serde_json::Value::Bool(true);
    }
    if let Some(system) = system_prompt {
        request["system"] = serde_json::Value::String(system);
    }
    if let Some(image) = args.image {
        request["image"] = serde_json::Value::String(image.display().to_string());
    }
    let contract = loaded
        .get("reasoning_contract")
        .and_then(serde_json::Value::as_str)
        .and_then(ReasoningContract::from_wire_name)
        .unwrap_or(ReasoningContract::Unsupported);
    let effort_native = loaded
        .get("reasoning_effort_native")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let supported_efforts = loaded
        .get("reasoning_efforts")
        .and_then(serde_json::Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(|value| value.as_str().map(|string| string.to_owned()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let _ = apply_http_reasoning_request(
        &serde_json::json!({}),
        &resolved,
        &mut request,
        contract,
        effort_native,
        &supported_efforts,
    )?;

    let mut content = String::new();
    let stream = !args.no_stream && !args.json;
    // End the streamed answer's line at `commit_ready`, before the commit is
    // sent, so later daemon stderr never lands on the answer's line.
    let mut line_ended = false;
    // Prefill progress redraws one stderr line, only on a terminal and never
    // under `--json`; it is erased before the first token is written.
    let mut prefill = prefill_progress::PrefillLine::new(
        !args.json && std::io::stderr().is_terminal(),
        Instant::now(),
    );
    let generated = engine.generate(&request, |event| {
        match event.get("type").and_then(serde_json::Value::as_str) {
            Some("prefill_progress") => {
                if let (Some(done), Some(total)) = (
                    event.get("done").and_then(serde_json::Value::as_u64),
                    event.get("total").and_then(serde_json::Value::as_u64),
                ) {
                    if let Some(draw) = prefill.update(done, total, Instant::now()) {
                        eprint!("{draw}");
                    }
                }
            }
            Some("token") => {
                if let Some(erase) = prefill.clear() {
                    eprint!("{erase}");
                }
                if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                    content.push_str(text);
                    if stream {
                        print!("{text}");
                        std::io::stdout().flush()?;
                    }
                }
            }
            Some("commit_ready") if stream && !line_ended => {
                if let Some(erase) = prefill.clear() {
                    eprint!("{erase}");
                }
                println!();
                std::io::stdout().flush()?;
                line_ended = true;
            }
            _ => {}
        }
        Ok(())
    });
    // Also covers an error or cancel before any token.
    if let Some(erase) = prefill.clear() {
        eprint!("{erase}");
    }
    let done = generated?;
    if let Some(message) = spill.as_mut().and_then(spill::SpillMonitor::check) {
        spill::emit(&message);
    }
    if args.json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "content": content,
                "tokens": done.get("tokens").and_then(serde_json::Value::as_u64),
                "tok_s": done.get("tok_s").and_then(serde_json::Value::as_f64),
                "finish_reason": done.get("finish_reason"),
            }))?
        );
    } else if args.no_stream {
        println!("{content}");
    } else if !line_ended {
        println!();
    }
    if args.stats && !args.json {
        eprintln!(
            "{}",
            format_run_stats(
                done.get("tokens").and_then(serde_json::Value::as_u64),
                &done
            )
        );
    }
    let _ = engine.unload();
    Ok(())
}

/// PIDs named by a live `~/.hipfire/daemon*.pid`.
///
/// Advisory by construction: it cannot prevent a daemon started mid-benchmark, but
/// it stops the common case (a serve instance already running), which would make
/// both engine rates meaningless. A real fix would move the measurement behind the
/// daemon protocol, which is deliberately not done here.
fn live_daemon_pids(paths: &Paths) -> Vec<i32> {
    let mut live = Vec::new();
    let Ok(entries) = std::fs::read_dir(&paths.root) else {
        return live;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("daemon") || !name.ends_with(".pid") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Ok(pid) = raw.trim().parse::<i32>() else {
            continue;
        };
        let alive = process_identity::pid_signalable(pid);
        if alive {
            live.push(pid);
        }
    }
    live.sort_unstable();
    live.dedup();
    live
}

/// The share `offload-bench` recommends from a measurement matrix.
///
/// The recommendation comes from the largest measurements: a real spilled
/// projection is MBs of weight bytes, and the matrix's smallest entry is where the
/// copies and the launch overhead distort the balance most. Every format is
/// measured over the same `buffer_mb` by construction, so "the largest" is usually
/// the whole matrix — a *set* — and the **median** share over it is the honest
/// single number (a singleton when `--format` pinned one format). Split out from
/// the command so the tie behavior is pinned by a test rather than by
/// `max_by_key`'s iteration order, which silently picked whichever format came
/// last.
fn recommended_share(reports: &[CalibrationReport]) -> Option<(String, f64)> {
    let largest = reports.iter().map(|r| r.bytes).max()?;
    let biggest: Vec<&CalibrationReport> = reports.iter().filter(|r| r.bytes == largest).collect();
    let mut shares: Vec<f64> = biggest.iter().map(|r| r.share).collect();
    shares.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let label = if biggest.len() == 1 {
        biggest[0].format.clone()
    } else {
        format!("the {} largest measured formats", biggest.len())
    };
    Some((label, shares[shares.len() / 2]))
}

/// `hipfire offload-bench`: measure this host's GPU share for a split offload step
/// (`memory.offload_exec=passback`, `memory.offload_passback_share`).
///
/// A raw device benchmark — no model, no daemon — so it refuses while a daemon pid
/// file names a live process.
fn offload_bench_command(paths: &Paths, args: OffloadBenchArgs) -> Result<()> {
    let live = live_daemon_pids(paths);
    if !live.is_empty() {
        bail!(
            "a hipfire daemon is live (pid {}); stop it first (`hipfire stop`) — offload-bench \
             measures the device directly, and a concurrent daemon makes both engine rates \
             meaningless",
            live.iter()
                .map(|pid| pid.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let mut gpu = rdna_compute::Gpu::init()
        .map_err(|e| anyhow!("offload-bench needs a GPU: {} [code {}]", e.message, e.code))?;
    let opts = CalibrateOptions {
        format: args.format.clone(),
        k: args.k,
        buffer_mb: args
            .buffer_mb
            .unwrap_or(hipfire_runtime::offload_calibrate::DEFAULT_BUFFER_MB),
        reps: args
            .reps
            .unwrap_or(hipfire_runtime::offload_calibrate::DEFAULT_REPS),
    };
    let reports = hipfire_runtime::offload_calibrate::calibrate(&mut gpu, &opts)?;
    let recommended = recommended_share(&reports);
    if args.json {
        let rows: Vec<serde_json::Value> = reports
            .iter()
            .map(|r| {
                serde_json::json!({
                    "format": r.format,
                    "m": r.m,
                    "k": r.k,
                    "bytes": r.bytes,
                    "cpu_bytes_per_s": r.cpu_bytes_per_s,
                    "gpu_bytes_per_s": r.gpu_bytes_per_s,
                    "share": r.share,
                    "predicted_speedup": r.predicted_speedup,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "arch": gpu.arch,
                "recommended": recommended
                    .as_ref()
                    .map(|(format, share)| serde_json::json!({ "format": format, "share": share })),
                "formats": rows,
            }))?
        );
    } else {
        println!(
            "arch {} — GPU share for a split offload step (memory.offload_exec=passback)",
            gpu.arch
        );
        println!(
            "{:<12} {:>8} {:>6} {:>8} {:>12} {:>12} {:>7} {:>9}",
            "format", "m", "k", "MiB", "cpu GB/s", "gpu GB/s", "share", "speedup"
        );
        for r in &reports {
            println!(
                "{:<12} {:>8} {:>6} {:>8.1} {:>12.1} {:>12.1} {:>7.3} {:>8.2}x",
                r.format,
                r.m,
                r.k,
                r.bytes as f64 / (1 << 20) as f64,
                r.cpu_bytes_per_s / 1e9,
                r.gpu_bytes_per_s / 1e9,
                r.share,
                r.predicted_speedup,
            );
        }
        if let Some((format, share)) = recommended.as_ref() {
            println!();
            println!("recommended for {format}: memory.offload_passback_share = {share:.3}");
        }
    }
    if args.write {
        let Some((_, share)) = recommended.as_ref() else {
            bail!("offload-bench: nothing measured, nothing to write");
        };
        let mut loaded = load_global(&paths.config)?;
        loaded
            .layer
            .set_cli("memory.offload_passback_share", &format!("{share:.3}"))?;
        write_global_toml(&paths.config, &loaded.layer)?;
        println!(
            "wrote memory.offload_passback_share = {share:.3} to {}",
            paths.config.config_toml.display()
        );
    }
    Ok(())
}

/// Default `hipfire img --backend`: `"gpu"` when a GPU is available
/// (`rdna_compute::Gpu::init()` succeeds), else `"cpu"`. Probed once per
/// process and cached — never re-probes the GPU per invocation. An explicit
/// `--backend` always overrides this default. A failed probe prints one
/// stderr line with the underlying HIP error so a fixable driver problem
/// isn't silently mistaken for "no GPU present".
fn default_img_backend() -> &'static str {
    static GPU_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *GPU_AVAILABLE.get_or_init(|| match rdna_compute::Gpu::init() {
        Ok(_) => true,
        Err(e) => {
            eprintln!(
                "hipfire: no GPU detected ({} [code {}]); defaulting --backend to cpu \
                 (run `hipfire diag` for a full environment report if a GPU should be present)",
                e.message, e.code
            );
            false
        }
    }) {
        "gpu"
    } else {
        "cpu"
    }
}

/// `hipfire img <model> <prompt…>`: one-shot txt2img through a fresh native
/// daemon process. Mirrors `run`'s model resolution and
/// local-spawn posture; prints per-step progress to stderr, writes the PNG,
/// prints its path (or the JSON result object with `--json`).
fn img_command(paths: &Paths, args: ImgArgs) -> Result<()> {
    if args.steps.is_some_and(|s| s == 0 || s > 128) {
        bail!("--steps must be between 1 and 128 (omit for the model default)");
    }
    let backend = args
        .backend
        .clone()
        .unwrap_or_else(|| default_img_backend().to_owned());
    if backend != "cpu" && backend != "gpu" {
        bail!("--backend must be \"cpu\" or \"gpu\", got {:?}", backend);
    }
    for (name, v) in [("--width", args.width), ("--height", args.height)] {
        if let Some(v) = v {
            if v == 0 || v > 8192 {
                bail!("{name} must be between 1 and 8192");
            }
        }
    }
    if args.image.len() > 4 {
        bail!("at most 4 --image references (got {})", args.image.len());
    }
    // Width/height default to 1024 only for plain txt2img (no reference
    // images); a reference edit leaves them unset so the daemon defaults to
    // the reference image's own size.
    let (width, height) = if args.image.is_empty() {
        (
            Some(args.width.unwrap_or(1024)),
            Some(args.height.unwrap_or(1024)),
        )
    } else {
        (args.width, args.height)
    };
    // The CLI reads the reference files itself and ships their bytes; the
    // daemon never opens a client-named path.
    let images = args
        .image
        .iter()
        .map(|p| {
            use base64::Engine as _;
            let bytes = std::fs::read(p)
                .with_context(|| format!("--image {}: cannot read file", p.display()))?;
            Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
        })
        .collect::<Result<Vec<String>>>()?;
    let prompt = if args.prompt.is_empty() {
        "a tiny cat sitting on a tiny table".to_owned()
    } else {
        args.prompt.join(" ")
    };
    let loaded_registry = load_registry(&paths.registry);
    let registry = &loaded_registry.registry;
    let (canonical, entry) = registry
        .model(&args.model)
        .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry)))
        .unwrap_or((None, None));
    let mut model_path = find_model_path(paths, registry, &args.model);
    if model_path.is_none() {
        if let Some(entry) = entry {
            eprintln!(
                "Model not found locally. Pulling {}...",
                canonical.as_deref().unwrap_or(&args.model)
            );
            pull_command(paths, PullArgs::defaults(args.model.clone()))?;
            model_path = Some(paths.models.join(&entry.file));
        }
    }
    let model_path = model_path.ok_or_else(|| {
        if std::path::Path::new(&args.model).is_dir() {
            anyhow!(
                "model not found: {0} is a directory; a diffusers pipe is not a model — pack it \
                 with `hipfire-quantize --flux-pipe {0} --output <base>.hfq` and pass \
                 `<base>-transformer.hfq`",
                args.model
            )
        } else {
            model_not_found_error(paths, registry, &args.model)
        }
    })?;
    let resolved = resolved_for_model(paths, &args.model, canonical.as_deref(), entry)?;
    let daemon = find_daemon(paths).ok_or_else(|| daemon_not_found_error(paths))?;
    let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
    let engine = Engine::spawn_configured(
        &daemon,
        &BTreeMap::new(),
        &process_config,
        Some(&model_path),
    )?;
    engine.ping()?;
    let _loaded = engine.load(&model_path, serde_json::json!({}))?;
    let mut request = serde_json::json!({
        "type": "img_generate",
        "id": "img",
        "prompt": prompt,
        "seed": args.seed,
        "backend": backend,
    });
    // Absent means the architecture default; the daemon refuses an explicit 0.
    if let Some(s) = args.steps {
        request["steps"] = serde_json::json!(s);
    }
    if let Some(w) = width {
        request["width"] = serde_json::json!(w);
    }
    if let Some(h) = height {
        request["height"] = serde_json::json!(h);
    }
    if !images.is_empty() {
        request["images"] = serde_json::json!(images);
    }
    let mut progress = 0u64;
    let done = engine.img_generate(&request, |event| {
        if event.get("type").and_then(serde_json::Value::as_str) == Some("img_progress") {
            let step = event
                .get("step")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let total = event
                .get("total")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if !args.json && step > progress {
                eprintln!("[img] step {step}/{total}");
                progress = step;
            }
        }
        Ok(())
    })?;
    let _ = engine.unload();
    let png_b64 = done
        .get("png_b64")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("daemon img_done missing png_b64"))?;
    let png = {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(png_b64)
            .context("daemon returned invalid base64 PNG")?
    };
    let out_path = args.out.clone().unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(format!("hipfire-{}.png", args.seed))
    });
    std::fs::write(&out_path, &png)
        .with_context(|| format!("failed to write {}", out_path.display()))?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "path": out_path,
                "bytes": png.len(),
                "width": done.get("width").and_then(serde_json::Value::as_u64),
                "height": done.get("height").and_then(serde_json::Value::as_u64),
                "steps": done.get("steps").and_then(serde_json::Value::as_u64),
                "seed": done.get("seed").and_then(serde_json::Value::as_u64),
                "ms": done.get("ms").and_then(serde_json::Value::as_u64),
                "model": done.get("model"),
            }))?
        );
    } else {
        println!("{}", out_path.display());
    }
    Ok(())
}

/// `hipfire embed`: encode text (or an image) with an embedding model, or —
/// with both `--query` and `--doc` — score the pair by late-interaction
/// MaxSim. Mirrors `img_command`'s engine lifecycle (fresh daemon, load,
/// request, unload).
fn embed_command(paths: &Paths, args: EmbedArgs) -> Result<()> {
    let kind = args.kind.as_deref().unwrap_or("document");
    if !matches!(kind, "query" | "document") {
        bail!("--kind must be \"query\" or \"document\", got {kind:?}");
    }
    if args.query.is_none()
        && args.doc.is_none()
        && args.text.is_empty()
        && args.image.is_empty()
        && args.audio.is_empty()
        && args.video.is_empty()
    {
        bail!(
            "nothing to encode: pass text (positional or --query/--doc) and/or \
             --image/--audio/--video (e.g. `hipfire embed <model> --query \"what is rust\" \
             --doc \"Rust is …\"`)"
        );
    }
    if args.doc.is_some() && args.query.is_none() {
        bail!("--doc without --query: encode the document alone with the positional text + --kind document");
    }
    // Every `--image` in flag order. The daemon decodes them; the base64 here
    // is the wire's one media spelling (`images_base64`, a list).
    let images_b64: Vec<String> = args
        .image
        .iter()
        .map(|p| -> Result<String> {
            use base64::Engine as _;
            let bytes = std::fs::read(p)
                .with_context(|| format!("--image {}: cannot read file", p.display()))?;
            Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
        })
        .collect::<Result<_>>()?;
    // Same for the clips. The container is checked here because the caller can
    // fix it without a model: the tower takes 16 kHz WAV/RIFF only.
    let audios_b64: Vec<String> = args
        .audio
        .iter()
        .map(|p| -> Result<String> {
            use base64::Engine as _;
            let bytes = std::fs::read(p)
                .with_context(|| format!("--audio {}: cannot read file", p.display()))?;
            if !is_wav(&bytes) {
                bail!(
                    "--audio {}: not a WAV file (RIFF/WAVE) — convert it first: \
                     ffmpeg -i {} -ar 16000 -ac 1 {}.wav",
                    p.display(),
                    p.display(),
                    p.display()
                );
            }
            Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
        })
        .collect::<Result<_>>()?;
    // Videos: one clip per `--video`, frames comma separated. The daemon runs
    // the sampler (`max_frames` 32) and the tower, so this layer only reads
    // the frames and checks each one's container — a clip container (MP4 …)
    // is refused here with the extraction to run, never decoded by guessing.
    let videos_b64: Vec<Vec<String>> = args
        .video
        .iter()
        .enumerate()
        .map(|(v, spec)| -> Result<Vec<String>> {
            use base64::Engine as _;
            let frames: Vec<&str> = spec
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            if frames.is_empty() {
                bail!("--video #{v} ({spec:?}) names no frames");
            }
            frames
                .iter()
                .enumerate()
                .map(|(f, path)| -> Result<String> {
                    let bytes = std::fs::read(path).with_context(|| {
                        format!("--video #{v} frame {f} ({path}): cannot read file")
                    })?;
                    if !is_frame_image(&bytes) {
                        bail!(
                            "--video #{v} frame {f} ({path}): not a PNG/JPEG image — this \
                             flag takes a clip's decoded FRAMES, not a container; extract \
                             them first: ffmpeg -i {path} -vf fps=1 frame_%04d.png"
                        );
                    }
                    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
                })
                .collect()
        })
        .collect::<Result<_>>()?;

    let loaded_registry = load_registry(&paths.registry);
    let registry = &loaded_registry.registry;
    // Embedding checkpoints are directories (an HF snapshot), which
    // `find_model_path` does not accept (it is file-only for the registry's
    // HFQ entries). Take a direct existing file OR directory first, then fall
    // back to the registry.
    let model_path = {
        let direct = PathBuf::from(&args.model);
        if direct.exists() {
            std::fs::canonicalize(&direct).unwrap_or(direct)
        } else {
            find_model_path(paths, registry, &args.model)
                .ok_or_else(|| model_not_found_error(paths, registry, &args.model))?
        }
    };
    let resolved = resolved_for_model(paths, &args.model, None, None)?;
    let daemon = find_daemon(paths).ok_or_else(|| daemon_not_found_error(paths))?;
    let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
    let engine = Engine::spawn_configured(
        &daemon,
        &BTreeMap::new(),
        &process_config,
        Some(&model_path),
    )?;
    engine.ping()?;
    // The vision ladder (and an authored `max_seq`) reaches the daemon through
    // these params: without them `vision.mode` never left the client, so a
    // configured `auto` still loaded text-only and refused every media input.
    let (canonical, entry) = registry_entry_for_path(paths, registry, &args.model)
        .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry)))
        .unwrap_or((None, None));
    let params = embed_load_params(
        &resolved,
        entry,
        &paths.models,
        &model_path,
        canonical.as_deref(),
    )?;
    let _loaded = engine.load(&model_path, params)?;

    let encode = |id: &str,
                  kind: &str,
                  text: &str,
                  images: &[String],
                  audios: &[String],
                  videos: &[Vec<String>]|
     -> Result<serde_json::Value> {
        let mut input = serde_json::Map::new();
        if !text.is_empty() {
            input.insert("text".into(), serde_json::json!(text));
        }
        if !images.is_empty() {
            input.insert("images_base64".into(), serde_json::json!(images));
        }
        if !audios.is_empty() {
            input.insert("audios_base64".into(), serde_json::json!(audios));
        }
        if !videos.is_empty() {
            input.insert("videos_base64".into(), serde_json::json!(videos));
        }
        let request = serde_json::json!({
            "type": "embed",
            "id": id,
            "kind": kind,
            "input": input,
        });
        engine
            .embed(&request)
            .map_err(|e| anyhow!("embed request failed: {e}"))
    };

    let positional = args.text.join(" ");
    let mut result_json = None;
    if let (Some(q), Some(d)) = (&args.query, args.doc.as_deref()) {
        let q_res = encode("q", "query", q, &images_b64, &audios_b64, &videos_b64)?;
        let d_res = encode("d", "document", d, &[], &[], &[])?;
        let dim = q_res
            .get("dim")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let q_tokens = q_res
            .get("n_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let d_tokens = d_res
            .get("n_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let q_emb = q_res
            .get("embeddings")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64().map(|x| x as f32))
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_default();
        let d_emb = d_res
            .get("embeddings")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64().map(|x| x as f32))
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_default();
        // Late interaction, same math as hipfire-arch-qwen35-embed::maxsim
        // (kept local: the CLI deliberately declares no arch-crate dep).
        let score = (0..q_tokens)
            .map(|qi| {
                (0..d_tokens)
                    .map(|di| {
                        q_emb[qi * dim..(qi + 1) * dim]
                            .iter()
                            .zip(&d_emb[di * dim..(di + 1) * dim])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                    })
                    .fold(f32::MIN, f32::max)
            })
            .sum::<f32>();
        if args.json {
            result_json = Some(serde_json::json!({
                "score": score,
                "query_tokens": q_tokens,
                "doc_tokens": d_tokens,
                "dim": dim,
            }));
        } else {
            println!("{score:.4}");
        }
    } else {
        let text = if args.query.is_some() {
            args.query.clone().unwrap_or_default()
        } else {
            positional
        };
        let kind = if args.query.is_some() { "query" } else { kind };
        let res = encode("e", kind, &text, &images_b64, &audios_b64, &videos_b64)?;
        if args.json {
            result_json = Some(res);
        } else {
            println!(
                "{} token embeddings ({} dims), {} input tokens — pass --json for the matrix",
                res.get("n_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                res.get("dim")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                res.get("n_input_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
            );
        }
    }
    let _ = engine.unload();
    if let Some(j) = result_json {
        println!("{}", serde_json::to_string(&j)?);
    }
    Ok(())
}

/// RIFF/WAVE magic — the only container the embeddinggemma2 audio tower takes
/// (`--audio` needs nothing else from the bytes; the daemon decodes them).
fn is_wav(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

/// PNG/JPEG magic — the only frames the embeddinggemma2 video leg takes
/// (`--video` names decoded frames, never a container).
fn is_frame_image(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n") || bytes.starts_with(b"\xff\xd8\xff")
}

fn process_truthy(name: &str) -> bool {
    hipfire_config::process_value(name).is_some_and(|value| {
        !matches!(
            value.to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        )
    })
}

pub(crate) fn run_should_force_local(args: &RunArgs) -> bool {
    process_truthy("HIPFIRE_LOCAL")
        || args.image.is_some()
        || args.kv_mode.is_some()
        || args.kv_k.is_some()
        || args.kv_v.is_some()
        || args.kv_backend.is_some()
        || args.tp.is_some_and(|tp| tp > 1)
        || args.max_seq.is_some()
        || args.head.is_some()
        || args.speculation.is_some()
        || args.model_draft.is_some()
        || args.vision.is_some()
        || args.draft_max.is_some()
        || args.dspark_conf_threshold.is_some()
}

fn format_run_stats(tokens: Option<u64>, timings: &serde_json::Value) -> String {
    let metric = |key: &str, precision: usize| {
        timings
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(|value| format!("{value:.precision$}"))
            .unwrap_or_else(|| "n/a".to_owned())
    };
    let mut footer = format!(
        "[stats] tokens={} decode={} tok/s ttft={} ms",
        tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_owned()),
        metric("decode_tok_s", 2),
        metric("ttft_ms", 1)
    );
    let routes = ["dflash", "mtp", "mtp_ngram"];
    let has_route = routes.iter().any(|key| {
        timings
            .get(key)
            .and_then(serde_json::Value::as_bool)
            .is_some()
    });
    let speculative = routes
        .iter()
        .any(|key| timings.get(key).and_then(serde_json::Value::as_bool) == Some(true));
    if (!has_route || speculative) && timings.get("tau").is_some_and(|value| !value.is_null()) {
        footer.push_str(&format!(" tau={}", metric("tau", 2)));
    }
    footer
}

#[allow(clippy::too_many_arguments)]
fn run_via_http(
    endpoint: &ServiceEndpoint,
    model: &str,
    prompt: &str,
    system: Option<&str>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    min_p: Option<f64>,
    presence_penalty: Option<f64>,
    repeat_penalty: Option<f64>,
    max_tokens: u64,
    json: bool,
    no_stream: bool,
    stats: bool,
) -> Result<()> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    messages.push(serde_json::json!({ "role": "user", "content": prompt }));
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    });
    insert_optional_f64(&mut body, "temperature", temperature);
    insert_optional_f64(&mut body, "top_p", top_p);
    insert_optional_u64(&mut body, "top_k", top_k);
    insert_optional_f64(&mut body, "min_p", min_p);
    insert_optional_f64(&mut body, "presence_penalty", presence_penalty);
    insert_optional_f64(&mut body, "repeat_penalty", repeat_penalty);
    let timeout = Duration::from_secs(60 * 60);
    let stats = stats && !json;
    if stats && !no_stream {
        body["stream_options"] = serde_json::json!({"include_usage": true});
    }
    if json || no_stream {
        let response = complete_openai_chat(endpoint, body, timeout)?;
        let content = response
            .pointer("/choices/0/message/content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "content": content,
                    "tokens": response.pointer("/usage/completion_tokens"),
                    "tok_s": response.pointer("/hipfire/tok_s"),
                    "finish_reason": response.pointer("/choices/0/finish_reason"),
                }))?
            );
        } else {
            println!("{content}");
            if stats {
                eprintln!(
                    "{}",
                    format_run_stats(
                        response
                            .pointer("/usage/completion_tokens")
                            .and_then(serde_json::Value::as_u64),
                        &response["timings"],
                    )
                );
            }
        }
        return Ok(());
    }

    let mut tokens = None;
    let mut timings = serde_json::Value::Null;
    let mut prefill =
        prefill_progress::PrefillLine::new(std::io::stderr().is_terminal(), Instant::now());
    let streamed = stream_openai_chat(
        endpoint,
        body,
        timeout,
        |event| {
            match event {
                OpenAiSseEvent::PrefillProgress { done, total } => {
                    if let Some(draw) = prefill.update(done, total, Instant::now()) {
                        eprint!("{draw}");
                    }
                }
                OpenAiSseEvent::Reasoning { text } | OpenAiSseEvent::Content { text } => {
                    if let Some(erase) = prefill.clear() {
                        eprint!("{erase}");
                    }
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
                OpenAiSseEvent::Finish { timings: value, .. } if stats => {
                    timings = value.unwrap_or(serde_json::Value::Null);
                }
                OpenAiSseEvent::Usage { usage } if stats => {
                    tokens = usage
                        .get("completion_tokens")
                        .and_then(serde_json::Value::as_u64);
                }
                _ => {}
            }
            Ok(())
        },
        || false,
    );
    if let Some(erase) = prefill.clear() {
        eprint!("{erase}");
    }
    streamed?;
    println!();
    if stats {
        eprintln!("{}", format_run_stats(tokens, &timings));
    }
    Ok(())
}

fn chat_command(paths: &Paths, args: ChatArgs) -> Result<()> {
    let (_, resolved) = resolved_global(paths, true)?;
    let host = config_string(&resolved, "serve.host")?;
    let port = config_u64(&resolved, "serve.port")? as u16;
    let endpoint = local_service_endpoint(&resolved, &host, port)?;
    let model = args
        .model
        .unwrap_or(config_string(&resolved, "serve.default_model")?);
    let max_tokens = args
        .max_tokens
        .unwrap_or(config_u64(&resolved, "generation.max_tokens")?);
    if max_tokens == 0 || max_tokens > 393_216 {
        bail!("--max-tokens must be between 1 and 393216");
    }
    if let Some(value) = args.temp {
        if !(0.0..=2.0).contains(&value) {
            bail!("--temp must be between 0 and 2");
        }
    }
    if let Some(value) = args.top_p {
        if !(0.0 < value && value <= 1.0) {
            bail!("--top-p must be in (0, 1]");
        }
    }

    if !service_ready(&endpoint, Duration::from_millis(150)) {
        let serve_args = ServeArgs {
            positionals: vec![host.clone(), port.to_string()],
            model: None,
            detach: true,
            no_prewarm: true,
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            kv_backend: None,
            vision: None,
            idle_timeout: None,
            tp: None,
            speculation: None,
            max_seq: None,
            swap_host_bytes: None,
            swap_disk_bytes: None,
            swap_dir: None,
            continuous_batch_size: None,
            foreground_child: false,
            ui: false,
            open: false,
            api_key_file: None,
        };
        detach_serve(paths, &serve_args, &host, port)?;
    }
    eprintln!("Interactive chat with {model}. Commands: /clear, /exit");
    let mut messages = Vec::new();
    if let Some(system) = args.system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    let stdin = std::io::stdin();
    loop {
        eprint!("you> ");
        std::io::stderr().flush()?;
        let mut input = String::new();
        if stdin.read_line(&mut input)? == 0 {
            println!();
            break;
        }
        let input = input.trim_end_matches(['\r', '\n']);
        match input.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/clear" => {
                messages.retain(|message| {
                    message.get("role").and_then(serde_json::Value::as_str) == Some("system")
                });
                eprintln!("conversation cleared");
                continue;
            }
            _ => {}
        }
        messages.push(serde_json::json!({ "role": "user", "content": input }));
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        });
        if let Some(value) = args.temp {
            body["temperature"] = serde_json::json!(value);
        }
        if let Some(value) = args.top_p {
            body["top_p"] = serde_json::json!(value);
        }
        print!("assistant> ");
        std::io::stdout().flush()?;
        let mut assistant_reasoning = String::new();
        let mut assistant_content = String::new();
        let result = stream_openai_chat(
            &endpoint,
            body,
            Duration::from_secs(60 * 60),
            |event| {
                match event {
                    OpenAiSseEvent::Reasoning { text } => {
                        assistant_reasoning.push_str(&text);
                        print!("{text}");
                        std::io::stdout().flush()?;
                    }
                    OpenAiSseEvent::Content { text } => {
                        assistant_content.push_str(&text);
                        print!("{text}");
                        std::io::stdout().flush()?;
                    }
                    OpenAiSseEvent::Role { .. }
                    | OpenAiSseEvent::ToolCall { .. }
                    | OpenAiSseEvent::Finish { .. }
                    | OpenAiSseEvent::Usage { .. }
                    | OpenAiSseEvent::PrefillProgress { .. }
                    | OpenAiSseEvent::Done => {}
                }
                Ok(())
            },
            || false,
        );
        println!();
        if let Err(error) = result {
            messages.pop();
            return Err(error.into());
        }
        let mut assistant_msg =
            serde_json::json!({ "role": "assistant", "content": assistant_content });
        if !assistant_reasoning.is_empty() {
            assistant_msg["reasoning_content"] = serde_json::Value::String(assistant_reasoning);
        }
        messages.push(assistant_msg);
    }
    let _ = args.no_color;
    Ok(())
}

pub(crate) fn resolved_for_model(
    paths: &Paths,
    model_name: &str,
    tag: Option<&str>,
    entry: Option<&ModelEntry>,
) -> Result<hipfire_config::ResolvedConfig> {
    let resolved = hipfire_registry::resolve_for_model(
        &paths.config,
        model_name,
        tag.zip(entry),
        load_env_layer()?,
    )?;
    // config.toml warnings name retired keys the load ignored. serve resolves
    // per model load, so report them once per process; legacy JSON migration
    // notes stay with the `config` commands.
    static REPORTED: std::sync::Once = std::sync::Once::new();
    REPORTED.call_once(|| {
        for warning in &resolved.warnings {
            eprintln!("warning: {warning}");
        }
    });
    Ok(resolved.resolved)
}

/// How [`scan_local_models`] compares an input against on-disk filenames.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchMode {
    /// Compare the lowercased strings as written.
    Literal,
    /// Additionally strip `-`, `.` and `_` from both sides, so spellings that
    /// differ only in separators compare equal (`ornith-1.5` ↔ `ornith1.5`).
    IgnoreSeparators,
}

fn scan_local_models(local: &[PathBuf], search: &str, mode: MatchMode) -> Vec<PathBuf> {
    let normalize = |value: &str| -> String {
        let lower = value.to_ascii_lowercase();
        match mode {
            MatchMode::Literal => lower,
            MatchMode::IgnoreSeparators => lower
                .chars()
                .filter(|c| !matches!(c, '-' | '.' | '_'))
                .collect(),
        }
    };
    let needle = normalize(search);
    // A needle that normalizes to nothing (an input of only separators) would
    // make `contains` true for every file and hand back an arbitrary model.
    if needle.is_empty() {
        return Vec::new();
    }
    local
        .iter()
        .filter(|path| {
            let name = normalize(
                path.file_name()
                    .and_then(|file| file.to_str())
                    .unwrap_or_default(),
            );
            name == needle || name.contains(&needle)
        })
        .cloned()
        .collect()
}

/// Resolve a user-supplied model input to its registry entry, if it has one.
///
/// A path-form input receives registry identity (sidecars, kv/max_seq policy,
/// rm targets) only when it IS the installed artifact: `canonicalize(input)`
/// must equal `canonicalize(models_dir.join(entry.file))` for some entry.
/// Both sides are canonicalized so a symlink inside the models directory
/// (e.g. `qwen3.8-27b.mq4-xt` pointing out at `~/qcal`) still matches by
/// target. A path that merely shares a basename with an entry file gets no
/// entry and loads as (or removes as) a bare artifact.
///
/// Non-path inputs (tags, aliases, bare file names with no on-disk doppelganger
/// in the current directory) go through [`RegistryV1::model`], which keeps
/// tag/alias/`qwen3.5:` normalization and the bare `entry.file` match.
pub(crate) fn registry_entry_for_path<'registry>(
    paths: &Paths,
    registry: &'registry RegistryV1,
    input: &str,
) -> Option<(&'registry str, &'registry ModelEntry)> {
    let candidate = Path::new(input);
    if input.contains('/') || input.contains('\\') || candidate.is_file() {
        let canonical_input = fs::canonicalize(candidate).ok()?;
        return registry.models.iter().find_map(|(tag, entry)| {
            let installed = paths.models.join(&entry.file);
            let legacy = paths
                .models
                .join(hipfire_registry::FLASH_NEXT_XTS_LEGACY_FILE);
            // A split entry also owns the verified monolithic it was cut from.
            let grandfathered = grandfather_pin(entry).is_some_and(|pin| {
                [pin.canonical_file, pin.legacy_file]
                    .iter()
                    .any(|file| same_installed_file(&paths.models.join(file), &canonical_input))
            });
            (same_installed_file(&installed, &canonical_input)
                || (entry.is_flash_next_xts() && same_installed_file(&legacy, &canonical_input))
                || grandfathered)
                .then(|| (tag.as_str(), entry))
        });
    }
    registry.model(input)
}

pub(crate) fn same_installed_file(a: &Path, b: &Path) -> bool {
    match (fs::metadata(a), fs::metadata(b)) {
        (Ok(a), Ok(b)) => same_artifact_metadata(&a, &b),
        _ => false,
    }
}

/// Fixture retired pins, process-wide so a serve harness's request threads see
/// them. [`RetiredOverride`] serializes the tests that set it.
#[cfg(test)]
static RETIRED_OVERRIDE: std::sync::Mutex<Option<Vec<RetiredModel>>> = std::sync::Mutex::new(None);

#[cfg(test)]
static RETIRED_OVERRIDE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Test guard: [`retired_table`] returns the fixture `pins` (on every thread)
/// until dropped. Holding it excludes other retired-fixture tests.
#[cfg(test)]
pub(crate) struct RetiredOverride(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

#[cfg(test)]
impl RetiredOverride {
    pub(crate) fn set(pins: &[RetiredModel]) -> Self {
        let guard = RETIRED_OVERRIDE_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *RETIRED_OVERRIDE
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(pins.to_vec());
        Self(guard)
    }
}

#[cfg(test)]
impl Drop for RetiredOverride {
    fn drop(&mut self) {
        *RETIRED_OVERRIDE
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
    }
}

/// The retired models this client still runs from a verified local copy: the
/// compiled [`RETIRED_MODELS`] (tests substitute small fixture pins).
pub(crate) fn retired_table() -> std::borrow::Cow<'static, [RetiredModel]> {
    #[cfg(test)]
    if let Some(table) = RETIRED_OVERRIDE
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
    {
        return std::borrow::Cow::Owned(table);
    }
    std::borrow::Cow::Borrowed(RETIRED_MODELS)
}

/// What a model input names: a catalog entry, a retired model, or neither.
///
/// A retired model keeps the card it had while published ([`RetiredModel::entry`]),
/// so a verified local copy loads with the same sampling defaults and
/// sidecar pins. It is not in [`RegistryV1`]: it is never offered, never
/// pulled ([`ModelCard::pullable`]), and only runs once
/// [`resolve_model_path`] has verified its bytes.
// The retired card is borrowed from the compiled pins' one-time cache
// (`RetiredModel::card`), so no variant allocates; the size gap is only stack.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ModelCard<'r> {
    Unregistered,
    Registry {
        tag: &'r str,
        entry: &'r ModelEntry,
    },
    Retired {
        pin: RetiredModel,
        entry: std::borrow::Cow<'r, ModelEntry>,
    },
}

impl ModelCard<'_> {
    /// The tag and card entry, whether published or retired.
    pub(crate) fn pair(&self) -> Option<(&str, &ModelEntry)> {
        match self {
            Self::Unregistered => None,
            Self::Registry { tag, entry } => Some((*tag, *entry)),
            Self::Retired { pin, entry } => Some((pin.tag, entry.as_ref())),
        }
    }

    pub(crate) fn tag(&self) -> Option<&str> {
        self.pair().map(|(tag, _)| tag)
    }

    pub(crate) fn entry(&self) -> Option<&ModelEntry> {
        self.pair().map(|(_, entry)| entry)
    }

    /// The entry a missing local file may be downloaded for. A retired model
    /// has none: it is never pulled, not even implicitly by `run` or `serve`.
    pub(crate) fn pullable(&self) -> Option<&ModelEntry> {
        match self {
            Self::Registry { entry, .. } => Some(entry),
            Self::Unregistered | Self::Retired { .. } => None,
        }
    }

    pub(crate) fn retired(&self) -> Option<&RetiredModel> {
        match self {
            Self::Retired { pin, .. } => Some(pin),
            _ => None,
        }
    }
}

/// [`registry_entry_for_path`] widened to the retired models: the card a
/// path-form or name input receives. A path is a retired model only when it IS
/// the installed artifact (`models/<pin.file>`, compared by file identity); a
/// same-named lookalike elsewhere is an unregistered file.
pub(crate) fn model_card<'r>(
    paths: &Paths,
    registry: &'r RegistryV1,
    input: &str,
) -> ModelCard<'r> {
    if let Some((tag, entry)) = registry_entry_for_path(paths, registry, input) {
        return ModelCard::Registry { tag, entry };
    }
    let candidate = Path::new(input);
    let table = retired_table();
    let pin = if input.contains('/') || input.contains('\\') || candidate.is_file() {
        fs::canonicalize(candidate).ok().and_then(|canonical| {
            table
                .iter()
                .find(|pin| same_installed_file(&paths.models.join(pin.file), &canonical))
        })
    } else {
        hipfire_registry::retired_model_named(input, &table)
    };
    match pin {
        Some(pin) => ModelCard::Retired {
            pin: *pin,
            entry: pin.card(),
        },
        None => ModelCard::Unregistered,
    }
}

/// The card of a tag, alias or bare file name: [`model_card`] without path
/// identity, for callers that never receive paths.
pub(crate) fn model_card_by_name<'r>(registry: &'r RegistryV1, input: &str) -> ModelCard<'r> {
    if let Some((tag, entry)) = registry.model(input) {
        return ModelCard::Registry { tag, entry };
    }
    let table = retired_table();
    match hipfire_registry::retired_model_named(input, &table) {
        Some(pin) => ModelCard::Retired {
            pin: *pin,
            entry: pin.card(),
        },
        None => ModelCard::Unregistered,
    }
}

/// A retired model's pinned file is not in the models directory. The client
/// never downloads it; `serve` reports it as a missing model.
#[derive(Debug)]
pub(crate) struct RetiredMissing(pub(crate) String);

impl std::fmt::Display for RetiredMissing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RetiredMissing {}

/// Prove `file` is the artifact `pin` names: SHA-256 and size (a previously
/// verified, unchanged file is not hashed again, see [`VerifiedCache`]).
fn verify_retired_file(paths: &Paths, pin: &RetiredModel, file: &Path) -> Result<()> {
    verified_metadata(
        &VerifiedCache::in_home(&paths.root),
        file,
        pin.sha256,
        pin.size_bytes,
    )
    .map(|_| ())
    .map_err(|error| {
        anyhow!(
            "{} is not the pinned {}; {} ({error:#}); move it aside or use {}",
            file.display(),
            pin.tag,
            pin.refusal(),
            pin.replacement
        )
    })
}

/// The installed file of a retired model, once its bytes verify against the
/// pin. `selected` is the path the user named, kept as written so sidecar
/// lookup sees the spelling they chose. Nothing is downloaded or written.
fn resolve_retired_path(
    paths: &Paths,
    pin: &RetiredModel,
    selected: Option<PathBuf>,
) -> Result<PathBuf> {
    let installed = paths.models.join(pin.file);
    match fs::symlink_metadata(&installed) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(RetiredMissing(format!(
                "{}; {} is not in {} and is never downloaded",
                pin.refusal(),
                pin.file,
                paths.models.display()
            ))
            .into());
        }
        Err(error) => return Err(error.into()),
    }
    verify_retired_file(paths, pin, &installed)?;
    Ok(selected.unwrap_or(installed))
}

/// [`find_model_path`] plus the retired models' installed files. Existence
/// only, with no verification: for callers that remove, list, compare or probe
/// a model. Anything that loads one goes through [`resolve_model_path`].
pub(crate) fn find_local_model_path(
    paths: &Paths,
    registry: &RegistryV1,
    model: &str,
) -> Option<PathBuf> {
    find_model_path(paths, registry, model).or_else(|| match model_card(paths, registry, model) {
        ModelCard::Retired { pin, .. } => {
            Some(paths.models.join(pin.file)).filter(|path| path.is_file())
        }
        _ => None,
    })
}

/// Fallible pinned migration stays outside generic, cheap local discovery.
pub(crate) fn resolve_model_path(
    paths: &Paths,
    registry: &RegistryV1,
    model: &str,
) -> Result<Option<PathBuf>> {
    let explicit = Path::new(model);
    // A directory is a valid model source (an HF safetensors snapshot — the
    // loader's `ModelSource::Dir`), not just an HFQ file. Resolve it here so
    // every entry point (serve prewarm, run, embed) accepts one path spelling.
    if explicit.is_dir() {
        return Ok(fs::canonicalize(explicit)
            .ok()
            .or_else(|| Some(explicit.to_path_buf())));
    }
    let catalog_file = load_catalog(&paths.config).ok().and_then(|catalog| {
        catalog
            .catalog
            .model(model)
            .and_then(|(_, record)| record.path.clone())
            .filter(|path| path.is_file())
    });
    let catalog_path = catalog_file.is_some();
    // A retired model is no longer in the catalog, but its pinned local copy
    // still runs once its bytes verify. It is never downloaded: a missing or
    // mismatching file is an error, not a pull. A catalog path registered
    // under the retired identity is held to the same pin.
    if let ModelCard::Retired { pin, .. } = model_card(paths, registry, model) {
        return match catalog_file {
            Some(file) => {
                verify_retired_file(paths, &pin, &file)?;
                Ok(Some(lexical_absolute(&file)))
            }
            None => {
                let selected = explicit.is_file().then(|| lexical_absolute(explicit));
                resolve_retired_path(paths, &pin, selected).map(Some)
            }
        };
    }
    if !explicit.is_file() && !model.contains('/') && !model.contains('\\') && !catalog_path {
        if let Some((_, entry)) = registry.model(model) {
            if entry.is_flash_next_xts() {
                return migrate_flash_next_xts(paths, entry);
            }
            // The split base is absent: verify (and link the legacy spelling of)
            // the pinned monolithic it was cut from, exactly as its own tag does.
            if entry.has_ple_sidecars() && !paths.models.join(&entry.file).is_file() {
                if let Some(pin) = grandfather_pin(entry) {
                    if let Some(path) = migrate_grandfathered(paths, &pin)? {
                        return Ok(Some(path));
                    }
                }
            }
        }
    }
    Ok(find_model_path(paths, registry, model))
}

fn migrate_flash_next_xts(paths: &Paths, entry: &ModelEntry) -> Result<Option<PathBuf>> {
    if !entry.is_flash_next_xts() {
        return Ok(None);
    }
    migrate_verified_artifact(
        &VerifiedCache::in_home(&paths.root),
        &paths
            .models
            .join(hipfire_registry::FLASH_NEXT_XTS_LEGACY_FILE),
        &paths
            .models
            .join(hipfire_registry::FLASH_NEXT_XTS_CANONICAL_FILE),
        entry.sha256.as_deref().expect("pinned digest"),
        entry.size_bytes.expect("pinned size"),
    )
}

#[cfg(test)]
thread_local! {
    static GRANDFATHER_OVERRIDE: std::cell::RefCell<Option<Vec<GrandfatheredMonolithic>>> =
        const { std::cell::RefCell::new(None) };
}

/// Test guard: [`grandfather_table`] returns the fixture `pins` on this thread
/// until dropped.
#[cfg(test)]
pub(crate) struct GrandfatherOverride;

#[cfg(test)]
impl GrandfatherOverride {
    pub(crate) fn set(pins: &[GrandfatheredMonolithic]) -> Self {
        GRANDFATHER_OVERRIDE.with(|cell| *cell.borrow_mut() = Some(pins.to_vec()));
        Self
    }
}

#[cfg(test)]
impl Drop for GrandfatherOverride {
    fn drop(&mut self) {
        GRANDFATHER_OVERRIDE.with(|cell| *cell.borrow_mut() = None);
    }
}

/// The monoliths this client keeps serving under their split tags: the compiled
/// [`GRANDFATHERED_MONOLITHICS`] (tests substitute a small fixture pin).
pub(crate) fn grandfather_table() -> std::borrow::Cow<'static, [GrandfatheredMonolithic]> {
    #[cfg(test)]
    if let Some(table) = GRANDFATHER_OVERRIDE.with(|cell| cell.borrow().clone()) {
        return std::borrow::Cow::Owned(table);
    }
    std::borrow::Cow::Borrowed(GRANDFATHERED_MONOLITHICS)
}

/// The monolithic pin `entry` is grandfathered under, if any.
pub(crate) fn grandfather_pin(entry: &ModelEntry) -> Option<GrandfatheredMonolithic> {
    hipfire_registry::grandfathered_monolithic_in(entry, &grandfather_table()).copied()
}

/// Verify the local monolithic of `pin` (canonical spelling, else the legacy
/// one, which gets a canonical hard link): `Ok(None)` when neither exists,
/// `Err` when one exists but its bytes do not verify. A previously verified,
/// unchanged file is not hashed again ([`VerifiedCache`]).
fn migrate_grandfathered(paths: &Paths, pin: &GrandfatheredMonolithic) -> Result<Option<PathBuf>> {
    migrate_verified_artifact(
        &VerifiedCache::in_home(&paths.root),
        &paths.models.join(pin.legacy_file),
        &paths.models.join(pin.canonical_file),
        pin.sha256,
        pin.size_bytes,
    )
}

/// The verified local monolithic that serves `entry` without its split base, and
/// the pin it verified against. A monolithic that fails verification is not
/// served: the pull falls back to the split base.
fn grandfathered_local<'p>(
    paths: &Paths,
    entry: &ModelEntry,
    pins: &'p [GrandfatheredMonolithic],
) -> Option<(&'p GrandfatheredMonolithic, PathBuf)> {
    let pin = hipfire_registry::grandfathered_monolithic_in(entry, pins)?;
    match migrate_grandfathered(paths, pin) {
        Ok(served) => served.map(|path| (pin, path)),
        Err(error) => {
            eprintln!(
                "  note: the local {} does not verify as the pinned monolithic ({error:#}); \
                 pulling the split base",
                pin.canonical_file
            );
            None
        }
    }
}

fn same_artifact_metadata(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        a.dev() == b.dev()
            && a.ino() == b.ino()
            && a.len() == b.len()
            && a.mtime() == b.mtime()
            && a.mtime_nsec() == b.mtime_nsec()
    }
    #[cfg(not(unix))]
    {
        a.len() == b.len() && a.modified().ok() == b.modified().ok()
    }
}

#[cfg(test)]
thread_local! { static HASH_INVOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

/// Open `path` and prove it is the artifact pinned by (`digest`, `size`). The
/// full-file SHA-256 is skipped when `cache` holds a record whose stat equals
/// the file's current stat and whose digest equals `digest`; otherwise the file
/// is hashed (a 125 GB monolithic takes minutes) and the result recorded.
fn verified_metadata(
    cache: &VerifiedCache,
    path: &Path,
    digest: &str,
    size: u64,
) -> Result<(fs::File, fs::Metadata)> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let before = file.metadata()?;
    if !before.is_file() || before.len() != size {
        bail!("size mismatch: {}", path.display());
    }
    let stat = verified_cache::StatKey::of(&before);
    if cache.satisfies(path, &stat, digest) {
        return Ok((file, before));
    }
    #[cfg(test)]
    HASH_INVOCATIONS.with(|count| count.set(count.get() + 1));
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if !actual.eq_ignore_ascii_case(digest) {
        bail!("SHA-256 mismatch: {}", path.display());
    }
    let after = file.metadata()?;
    let current = fs::metadata(path)?;
    if !same_artifact_metadata(&before, &after) || !same_artifact_metadata(&before, &current) {
        bail!("artifact changed during verification: {}", path.display());
    }
    cache.remember(path, stat, &actual);
    Ok((file, before))
}

// Never copy or move payloads. Unsupported/read-only link installation leaves
// the verified source usable. Existing destinations are handled by the caller.
fn install_verified_link(source: &Path, destination: &Path) -> Result<bool> {
    match fs::hard_link(source, destination) {
        Ok(()) => return Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(true),
        Err(_) => {}
    }
    #[cfg(unix)]
    {
        let target = fs::canonicalize(source)?;
        match std::os::unix::fs::symlink(target, destination) {
            Ok(()) => return Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(true),
            Err(_) => {}
        }
    }
    Ok(false)
}

fn migrate_verified_artifact(
    cache: &VerifiedCache,
    old: &Path,
    canonical: &Path,
    digest: &str,
    size: u64,
) -> Result<Option<PathBuf>> {
    match fs::symlink_metadata(canonical) {
        Ok(_) => {
            verified_metadata(cache, canonical, digest, size)?;
            return Ok(Some(canonical.to_owned()));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    match fs::symlink_metadata(old) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    // Keep the opened inode alive until installation and stability checks finish.
    let (_file, verified) = verified_metadata(cache, old, digest, size)?;
    if install_verified_link(old, canonical)? {
        let installed = fs::metadata(canonical)?;
        if same_artifact_metadata(&verified, &installed) {
            // The hard link is the same inode: the verification covers it.
            cache.remember(canonical, verified_cache::StatKey::of(&installed), digest);
        } else {
            // A concurrent installer won: verify its bytes, never clobber it.
            verified_metadata(cache, canonical, digest, size)?;
        }
        if !same_artifact_metadata(&verified, &fs::metadata(old)?) {
            bail!("artifact changed during migration: {}", old.display());
        }
        return Ok(Some(canonical.to_owned()));
    }
    if !same_artifact_metadata(&verified, &fs::metadata(old)?) {
        bail!("artifact changed during migration: {}", old.display());
    }
    Ok(Some(old.to_owned()))
}

/// `path` made absolute without resolving symlinks: the daemon may run from
/// another cwd, but the spelling the user selected (a symlink beside which
/// sidecars are published) must survive.
fn lexical_absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// `model_path` as selected, then its canonical target when that differs: the
/// locations a sidecar of a possibly symlinked trunk may sit beside.
fn trunk_locations(model_path: &Path) -> Vec<PathBuf> {
    let mut locations = vec![model_path.to_path_buf()];
    if let Ok(canonical) = fs::canonicalize(model_path) {
        if canonical != model_path {
            locations.push(canonical);
        }
    }
    locations
}

/// Lowercased `name` without `-`, `.` and `_`: the separator-insensitive key
/// [`find_model_path`] compares a registry tag's file against disk by.
fn strip_name_separators(name: &str) -> String {
    name.to_ascii_lowercase()
        .chars()
        .filter(|c| !matches!(c, '-' | '.' | '_'))
        .collect()
}

/// Resolve a model name or path to a local file. A path the user named (or a
/// catalog path) is returned as selected, made absolute but with symlinks
/// intact: the PLE sidecar lookup probes `<selected>.ple-*` first and the
/// canonical target second, and a canonical path would lose the former.
pub(crate) fn find_model_path(
    paths: &Paths,
    registry: &RegistryV1,
    model: &str,
) -> Option<PathBuf> {
    let direct = PathBuf::from(model);
    if direct.is_file() {
        return Some(lexical_absolute(&direct));
    }
    if let Ok(catalog) = load_catalog(&paths.config) {
        if let Some((_, record)) = catalog.catalog.model(model) {
            if let Some(path) = record.path.as_ref().filter(|path| path.is_file()) {
                return Some(lexical_absolute(path));
            }
        }
    }
    if let Some((_, entry)) = registry
        .model(model)
        .filter(|(_, entry)| entry.is_flash_next_xts())
    {
        return [
            entry.file.as_str(),
            hipfire_registry::FLASH_NEXT_XTS_LEGACY_FILE,
        ]
        .into_iter()
        .map(|file| paths.models.join(file))
        .find(|path| path.is_file());
    }
    if let Some((_, entry)) = registry
        .model(model)
        .filter(|(_, entry)| entry.has_ple_sidecars())
    {
        // A split tag serves its base, else the grandfathered monolithic it was
        // cut from, so nobody is forced to re-download after the registry switch.
        return hipfire_registry::split_model_files_in(entry, &grandfather_table())
            .iter()
            .map(|file| paths.models.join(file))
            .find(|path| path.is_file());
    }
    // An exact on-disk spelling outranks a registry alias that would rewrite it.
    // The registry maps a model NAME to its canonical FILE, which is what makes
    // `qwen3.8:27b` resolve to `qwen3.8-27b.mq4`. But when a legacy alias points at
    // a RENAMED artifact and both spellings are present, that rewrite silently
    // serves weights the user did not name -- `ornith1.5:35b-a3b` reaching
    // `ornith-1.5-35b-a3b.mq4` while `ornith1.5-35b-a3b.mq4` sits right there.
    //
    // The comparison is on the exact stem, deliberately NOT the looser `contains`
    // used by scan_local_models: `qwen3.8:27b` would `contains`-match
    // `qwen3.8-27b.mq4r` and `qwen3.8-27b.mq4-xt` as well, and the quant sort below
    // ranks `.mq4r` ABOVE `.mq4`, so a looser rule here would silently move users
    // onto a different tier. An exact stem can only ever match the one file the
    // user actually spelled.
    let exact_stem = model.replace(':', "-").to_ascii_lowercase();
    if let Ok(local) = local_model_paths(paths, registry) {
        if let Some(hit) = local.iter().find(|path| {
            let name = path
                .file_name()
                .and_then(|file| file.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            MODEL_SUFFIXES
                .iter()
                .any(|suffix| name == format!("{exact_stem}{suffix}"))
        }) {
            return Some(hit.clone());
        }
    }
    if let Some((_, entry)) = registry.model(model) {
        let path = paths.models.join(&entry.file);
        if path.is_file() {
            return Some(path);
        }
    }
    let path = paths.models.join(model);
    if path.is_file() {
        return Some(path);
    }
    let search = model.replace(':', "-").to_ascii_lowercase();
    let explicit_quant = MODEL_SUFFIXES.iter().any(|suffix| search.ends_with(suffix));
    let local = local_model_paths(paths, registry).ok()?;
    // A registry tag names exactly one file. When that file is missing (or a
    // dangling symlink) the substring scans below would hand back any sibling
    // whose name merely contains the tag: `qwen3.8:27b-mq3` reached
    // `qwen3.8-27b.mq3-xt`, a different tier, and loaded it silently. Only a
    // spelling of the SAME file qualifies (equal once `-`, `.`, `_` are ignored:
    // the renamed-artifact case); otherwise report it absent so the caller pulls
    // the canonical file.
    if let Some((_, entry)) = registry.model(model) {
        let wanted = strip_name_separators(&entry.file);
        return local.into_iter().find(|path| {
            path.file_name()
                .and_then(|file| file.to_str())
                .is_some_and(|name| strip_name_separators(name) == wanted)
        });
    }
    // Two passes. The first matches the literal spelling and is what has always
    // run. The second retries with `-`, `.` and `_` stripped from both sides, so
    // an input and an on-disk file that differ only in separators still meet:
    // `ornith-1.5:35b-a3b` finds the `ornith1.5-35b-a3b.mq4` left on disk by
    // anyone who downloaded before the artifacts were renamed.
    //
    // The fallback is strictly second — it runs only when the literal pass found
    // nothing, so it can add a match where today there is none but can never
    // change one that already resolves. That ordering is the safety argument:
    // separator-stripping is a looser `contains`, and running it first would let
    // it outrank an exact hit.
    let mut candidates = scan_local_models(&local, &search, MatchMode::Literal);
    if candidates.is_empty() {
        candidates = scan_local_models(&local, &search, MatchMode::IgnoreSeparators);
    }
    candidates.sort_by_key(|path| {
        let name = path
            .file_name()
            .and_then(|file| file.to_str())
            .unwrap_or_default();
        if explicit_quant || name.ends_with(".mq4r") {
            0
        } else if name.ends_with(".mq4") {
            1
        } else if name.ends_with(".hf4") || name.ends_with(".hfq") {
            2
        } else {
            3
        }
    });
    candidates.into_iter().next()
}

/// Clap value parser for `--kv-backend`: shared `KvBackend::from_str` so the old
/// `contiguous` spelling returns the migration error naming `legacy`.
fn parse_kv_backend_arg(raw: &str) -> std::result::Result<String, String> {
    raw.parse::<KvBackend>()
        .map(|backend| backend.as_str().to_owned())
        .map_err(|err| err.to_string())
}

/// Clap value parser for `--max-seq`: the `memory.max_seq` schema field parses
/// and bounds it, so serve/bench flags and the config key accept the same values.
fn parse_max_seq_arg(raw: &str) -> std::result::Result<u64, String> {
    match field("memory.max_seq")
        .expect("schema field")
        .parse_cli(raw)
        .map_err(|err| err.to_string())?
    {
        hipfire_config::ConfigValue::Integer(value) => {
            u64::try_from(value).map_err(|_| format!("max_seq {value} is negative"))
        }
        other => Err(format!("max_seq must be an integer, got {other:?}")),
    }
}

/// Layer an authored typed K/V axis from resolved config. Built-in empty defaults
/// and registry pins are omitted; user/one-shot/env sources emit non-empty values.
fn authored_kv_axis(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<Option<String>> {
    let Some(item) = resolved.get(key) else {
        return Ok(None);
    };
    match &item.source {
        ConfigSource::GlobalUser { .. }
        | ConfigSource::ModelUser { .. }
        | ConfigSource::OneShot { .. }
        | ConfigSource::LegacyEnv { .. } => {}
        _ => return Ok(None),
    }
    match &item.value {
        hipfire_config::ConfigValue::String(value) if !value.is_empty() => Ok(Some(value.clone())),
        _ => Ok(None),
    }
}

/// Apply CLI `--kv-k` / `--kv-v` onto load params (CLI wins over config axes).
/// Empty strings are ignored. Validates through the typed schema when present.
pub(crate) fn apply_kv_axis_overrides(
    params: &mut serde_json::Value,
    kv_k: Option<&str>,
    kv_v: Option<&str>,
) -> Result<()> {
    if let Some(raw) = kv_k.map(str::trim).filter(|value| !value.is_empty()) {
        if let Some(schema) = field("memory.kv_k") {
            schema
                .parse_cli(raw)
                .map_err(|err| anyhow!("--kv-k {raw}: {err}"))?;
        }
        params["kv_k"] = serde_json::json!(raw);
    }
    if let Some(raw) = kv_v.map(str::trim).filter(|value| !value.is_empty()) {
        if let Some(schema) = field("memory.kv_v") {
            schema
                .parse_cli(raw)
                .map_err(|err| anyhow!("--kv-v {raw}: {err}"))?;
        }
        params["kv_v"] = serde_json::json!(raw);
    }
    Ok(())
}

/// Copy effective KV mode, backend fields and the effective pipeline degree
/// (`pp`, a number — distinct from `bench --pp` prompt sizes) from the
/// validated loaded ACK.
fn with_loaded_kv_backend_fields(
    mut report: serde_json::Value,
    loaded: &serde_json::Value,
) -> serde_json::Value {
    let Some(object) = report.as_object_mut() else {
        return report;
    };
    for key in [
        "kv_backend",
        "kv_backend_request",
        "kv_backend_reason",
        "kv_backend_legacy",
        "kv_backend_warning",
        "kv_mode",
        "pp",
    ] {
        object.insert(
            key.to_owned(),
            loaded.get(key).cloned().unwrap_or(serde_json::Value::Null),
        );
    }
    report
}

/// `speculation.ngram` → the typed `params.ngram_draft` flag. The schema keeps
/// the authored spelling (`on`/`off`/`auto` and the `1`/`0` aliases), so the
/// shared strict tri-state parser resolves it; an unrecognized spelling is a
/// config error, never a silent off. `auto` arms the model-free speculator
/// wherever the loader has one for the arch (unchanged).
fn ngram_draft_param(mode: &str) -> Result<bool> {
    let armed = hipfire_config::parse_tri_state_alias(mode)
        .map_err(|error| anyhow!("speculation.ngram: {error}"))?;
    Ok(armed.unwrap_or(true))
}

pub(crate) fn load_params(
    resolved: &hipfire_config::ResolvedConfig,
    entry: Option<&ModelEntry>,
    models_dir: &Path,
    model_path: &Path,
    max_tokens: u64,
    kv_override: Option<&str>,
    kv_backend_override: Option<&str>,
    tag: Option<&str>,
    explicit_draft: bool,
    head_override: Option<&str>,
) -> Result<serde_json::Value> {
    // max_seq is omitted for built-in/registry provenance so the daemon can
    // derive min(model context, card capacity). Explicit user sources are
    // forwarded unchanged (no max_tokens+1024 inflation).
    let configured_kv = config_string(resolved, "memory.kv_cache")?;
    let kv_mode = kv_override
        .map(str::to_owned)
        .filter(|value| !value.is_empty())
        .unwrap_or(configured_kv);
    // Validate through the shared schema. `auto` is preserved so architecture
    // (maple vs qwen) can select BF16 vs Q8; do not substitute q8 here.
    field("memory.kv_cache")
        .expect("schema field")
        .parse_cli(&kv_mode)?;
    let configured_backend = config_string(resolved, "memory.kv_backend")?;
    let backend_raw = kv_backend_override
        .filter(|value| !value.is_empty())
        .unwrap_or(configured_backend.as_str());
    let kv_backend = backend_raw
        .parse::<KvBackend>()
        .map_err(|err| anyhow!("{err}"))?
        .as_str()
        .to_owned();
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    let mut cask_sidecar = config_string(resolved, "memory.cask.sidecar")?;
    if cask_sidecar.is_empty() && config_bool(resolved, "memory.cask.auto_attach")? {
        if let Some(sidecar) = entry.and_then(|entry| entry.triattn.as_ref()) {
            let candidate = model_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&sidecar.file);
            if candidate.is_file() {
                cask_sidecar = candidate.display().to_string();
            }
        }
    }
    // Resolve --head <name> against the registry's `heads` map. The overlay
    // lives beside the model file, exactly like the triattn sidecar. Refuse
    // rather than fall back: a silent fall-back would serve the base's head
    // and answer a different question than the operator asked.
    let head_file = match head_override.filter(|s| !s.is_empty()) {
        None => String::new(),
        // A direct path is accepted as well as a registry name: loading a
        // model BY PATH has no registry entry, so names cannot resolve there
        // and only a path can work.
        Some(name) if Path::new(name).is_file() => name.to_string(),
        Some(name) => {
            let heads = entry.map(|e| &e.heads);
            let sidecar = heads.and_then(|h| h.get(name)).ok_or_else(|| {
                let known: Vec<&str> = heads
                    .map(|h| h.keys().map(String::as_str).collect())
                    .unwrap_or_default();
                anyhow!(
                    "--head {name}: not a file, and this model has no such head variant{}",
                    if known.is_empty() {
                        " (it publishes none)".to_string()
                    } else {
                        format!(" (available: {})", known.join(", "))
                    }
                )
            })?;
            let candidate = model_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&sidecar.file);
            if !candidate.is_file() {
                bail!(
                    "--head {name}: overlay {} not found — fetch it with \
                     `hipfire pull` or place it beside the model",
                    candidate.display()
                );
            }
            candidate.display().to_string()
        }
    };
    let mut params = serde_json::json!({
        "deepseek4_compute_placement": config_string(
            resolved,
            "hardware.deepseek4_compute_placement",
        )?,
        "kv_mode": kv_mode,
        "kv_adaptive": config_string(resolved, "memory.kv_adaptive")?,
        "dflash_mode": config_string(resolved, "speculation.dflash")?,
        "vision_mode": config_string(resolved, "vision.mode")?,
        "dflash_adaptive_b": config_bool(resolved, "speculation.dflash_adaptive_b")?,
        // `auto` or a fixed width (schema-validated); the daemon resolves
        // `auto` against the route it actually stages.
        "dflash_block": match config_value(resolved, "speculation.dflash_block")? {
            hipfire_config::ConfigValue::Integer(b) => serde_json::json!(b),
            _ => serde_json::json!("auto"),
        },
        "mtp_mode": config_string(resolved, "speculation.mtp")?,
        // `auto` or an explicit 1-10 (schema-validated); the daemon/loader
        // resolves `auto` per model arch and GPU.
        "mtp_k": match config_value(resolved, "speculation.mtp_k")? {
            hipfire_config::ConfigValue::Integer(k) => serde_json::json!(k),
            _ => serde_json::json!("auto"),
        },
        "ngram_draft": ngram_draft_param(&config_string(resolved, "speculation.ngram")?)?,
        "ngram_k": config_u64(resolved, "speculation.ngram_k")?,
        "ngram_min_count": config_u64(resolved, "speculation.ngram_min_count")?,
        "ddtree_budget": config_u64(resolved, "speculation.ddtree_budget")?,
        "ddtree_topk": config_u64(resolved, "speculation.ddtree_topk")?,
        "cask_sidecar": cask_sidecar,
        "head": head_file,
        "cask": config_bool(resolved, "memory.cask.enabled")?,
        "cask_budget": config_u64(resolved, "memory.cask.budget")?,
        "cask_beta": config_u64(resolved, "memory.cask.beta")?,
        "cask_handoff_tokens": config_u64(resolved, "memory.cask.handoff_tokens")?,
        "cask_core_frac": config_f64(resolved, "memory.cask.core_fraction")?,
        "cask_fold_m": config_u64(resolved, "memory.cask.fold")?,
        // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
        "prefill_compression": config_string(resolved, "speculation.prefill.mode")?,
        "prefill_threshold": config_u64(resolved, "speculation.prefill.threshold")?,
        "prefill_keep_ratio": config_f64(resolved, "speculation.prefill.keep_ratio")?,
        "prefill_alpha": config_f64(resolved, "speculation.prefill.alpha")?,
        "prefill_min_keep": config_u64(resolved, "speculation.prefill.min_keep")?,
        "prefill_sink": config_u64(resolved, "speculation.prefill.sink")?,
        "prefill_recent": config_u64(resolved, "speculation.prefill.recent")?,
        "prefill_block": config_u64(resolved, "speculation.prefill.block")?,
        "prefill_drafter": config_string(resolved, "speculation.prefill.drafter")?,
        "prefill_drafter_device": config_i64(resolved, "speculation.prefill.drafter_device")?,
        "prefill_sparse_threshold": config_u64(resolved, "speculation.prefill.sparse_threshold")?,
        "speculation": config_string(resolved, "speculation.mode")?,
        "continuous_batch_size": config_u64(resolved, "serve.continuous_batch_size")?,
    });
    let backend_source = &resolved
        .get("memory.kv_backend")
        .expect("schema field")
        .source;
    if kv_backend_override.is_some()
        || matches!(
            backend_source,
            ConfigSource::GlobalUser { .. }
                | ConfigSource::ModelUser { .. }
                | ConfigSource::OneShot { .. }
        )
    {
        params["kv_backend"] = serde_json::json!(kv_backend);
    }
    // GDN/DeltaNet state precision: forwarded only when authored (user,
    // per-model, one-shot or `HIPFIRE_STATE_QUANT`), so the daemon applies
    // its per-model/arch `auto` for built-in and registry provenance.
    let state_quant_source = &resolved
        .get("memory.state_quant")
        .expect("schema field")
        .source;
    if matches!(
        state_quant_source,
        ConfigSource::GlobalUser { .. }
            | ConfigSource::ModelUser { .. }
            | ConfigSource::OneShot { .. }
            | ConfigSource::LegacyEnv { .. }
    ) {
        params["state_quant"] = serde_json::json!(config_string(resolved, "memory.state_quant")?);
    }
    // Authored K/V axes only: CLI is applied by callers via apply_kv_axis_overrides
    // after this returns so serve can share the same helper. Empty/BuiltIn/registry
    // defaults are omitted from IPC.
    if let Some(kv_k) = authored_kv_axis(resolved, "memory.kv_k")? {
        params["kv_k"] = serde_json::json!(kv_k);
    }
    if let Some(kv_v) = authored_kv_axis(resolved, "memory.kv_v")? {
        params["kv_v"] = serde_json::json!(kv_v);
    }
    let max_seq_source = &resolved.get("memory.max_seq").expect("schema field").source;
    if matches!(
        max_seq_source,
        ConfigSource::GlobalUser { .. }
            | ConfigSource::ModelUser { .. }
            | ConfigSource::OneShot { .. }
            | ConfigSource::LegacyEnv { .. }
    ) {
        let max_seq = config_u64(resolved, "memory.max_seq")?;
        params["max_seq"] = serde_json::json!(max_seq);
    }
    // Resolved `topology.pp` (global, per-model, env): forwarded as an explicit
    // request when above 1, so serve's per-model resolution is not lost to its
    // process-wide snapshot. Absent/1 leaves the daemon's `topology.pp`.
    let pp = config_u64(resolved, "topology.pp")?;
    if pp > 1 {
        params["pp"] = serde_json::json!(pp);
    }
    if let Some(experts_per_token) =
        config_optional_u64(resolved, "model.deepseek4_experts_per_token")?
    {
        params["deepseek4_experts_per_token"] = serde_json::json!(experts_per_token);
    }
    let selector = config_string(resolved, "speculation.mode")?;
    apply_speculation_selector(&mut params, &selector)?;
    project_dflash_draft(&mut params, developer_dflash_draft(resolved));
    if !explicit_draft {
        // A CLI `--model-draft` (projected by the caller after this returns)
        // always wins, so skip sidecar resolution — and its `on` fail-closed
        // bail — when one was given.
        resolve_dflash_sidecar(&mut params, entry, models_dir, model_path, tag)?;
    }
    resolve_vision_sidecar(&mut params, entry, models_dir, model_path, tag)?;
    resolve_xdna_sidecar(&mut params, resolved, entry, models_dir, model_path, tag)?;
    Ok(params)
}

/// One-shot per-load overrides layered over the resolved config: the flags
/// `hipfire run` and `hipfire fit` share, so a dry-run admission and a real
/// load build their params through one sequence.
#[derive(Clone, Copy, Default)]
pub(crate) struct LoadOverrides<'a> {
    pub kv_mode: Option<&'a str>,
    pub kv_backend: Option<&'a str>,
    pub kv_k: Option<&'a str>,
    pub kv_v: Option<&'a str>,
    /// `--spec`; absent means the resolved `speculation.mode`.
    pub speculation: Option<&'a str>,
    pub head: Option<&'a str>,
    /// A CLI `--model-draft` will be projected by the caller.
    pub explicit_draft: bool,
}

/// The speculation selector one load applies: the one-shot `--spec` value,
/// else the resolved `speculation.mode`.
pub(crate) fn effective_speculation_selector(
    resolved: &hipfire_config::ResolvedConfig,
    speculation: Option<&str>,
) -> Result<String> {
    match speculation {
        Some(selector) => Ok(selector.to_owned()),
        None => config_string(resolved, "speculation.mode"),
    }
}

/// `load_params` plus the one-shot overrides `hipfire run` applies before it
/// hands the params to the daemon: `--kv-k`/`--kv-v` axes, the effective
/// speculation selector, and the DFlash draft projection that selector
/// leaves enabled (config-off + `--spec dflash` must still carry the draft).
/// Registry sidecar resolution for the final selector stays with the caller
/// ([`resolve_dflash_sidecar`], [`resolve_mtp_sidecar`]).
pub(crate) fn load_params_with_overrides(
    resolved: &hipfire_config::ResolvedConfig,
    entry: Option<&ModelEntry>,
    models_dir: &Path,
    model_path: &Path,
    max_tokens: u64,
    tag: Option<&str>,
    overrides: &LoadOverrides<'_>,
) -> Result<serde_json::Value> {
    let mut params = load_params(
        resolved,
        entry,
        models_dir,
        model_path,
        max_tokens,
        overrides.kv_mode,
        overrides.kv_backend,
        tag,
        overrides.explicit_draft,
        overrides.head,
    )?;
    apply_kv_axis_overrides(&mut params, overrides.kv_k, overrides.kv_v)?;
    let selector = effective_speculation_selector(resolved, overrides.speculation)?;
    apply_speculation_selector(&mut params, &selector)?;
    // Final effective selector wins: re-project inherited draft only when DFlash
    // remains enabled.
    project_dflash_draft(&mut params, developer_dflash_draft(resolved));
    Ok(params)
}

/// Load params for the non-chat `embed` verb.
///
/// The encoder consumes the vision ladder and an authored sequence bound;
/// generation-only policy (DFlash/MTP/KV tiers, speculation selectors) has no
/// meaning for a checkpoint with no LM head, so it is deliberately not
/// projected — forwarding it would make an encoder load depend on it.
///
/// This exists because `embed` used to load with `{}`, so `vision.mode` never
/// reached the daemon: a configured `vision_mode=auto` was ignored and every
/// image/audio/video input was refused with the tower-less error, while the
/// same model served media through the daemon protocol.
fn embed_load_params(
    resolved: &hipfire_config::ResolvedConfig,
    entry: Option<&ModelEntry>,
    models_dir: &Path,
    model_path: &Path,
    tag: Option<&str>,
) -> Result<serde_json::Value> {
    let mut params = serde_json::json!({
        "vision_mode": config_string(resolved, "vision.mode")?,
    });
    // Same rule as `load_params`: forward an authored `max_seq` (global, model
    // overlay, one-shot flag or legacy env) and let the daemon derive one for
    // built-in/registry provenance.
    let max_seq_source = &resolved.get("memory.max_seq").expect("schema field").source;
    if matches!(
        max_seq_source,
        ConfigSource::GlobalUser { .. }
            | ConfigSource::ModelUser { .. }
            | ConfigSource::OneShot { .. }
            | ConfigSource::LegacyEnv { .. }
    ) {
        let max_seq = config_u64(resolved, "memory.max_seq")?;
        params["max_seq"] = serde_json::json!(max_seq);
    }
    resolve_vision_sidecar(&mut params, entry, models_dir, model_path, tag)?;
    Ok(params)
}

/// Resolve a registry-declared DFlash sidecar into `params["draft"]`.
///
/// Call only once the final `dflash_mode` is known. When the mode is `auto`
/// or `on`, no explicit draft is set (`params["draft"]`, e.g. from
/// `developer.dflash_draft`), and `entry.dflash` names a pulled file, wire
/// it: `on` without the file fails closed, `auto` logs one line and falls
/// back to the remaining speculation policy (MTP `auto` when a head exists).
/// With no entry at all the artifact is not registry-managed (e.g. a path
/// that merely shares a basename with an entry file): `auto` wires no draft
/// for a bare artifact, but `on` fails closed instead of silently running AR.
/// The sidecar is looked up in `models_dir` first — `find_model_path`
/// keeps the selected (possibly symlinked) path, so the artifact's real
/// directory is probed too, not only the models directory the draft was
/// pulled into — then next to the selected path, then next to its canonical
/// target. An explicit draft always wins; a final `off` never
/// carries a draft (`project_dflash_draft` strips it) and returns early.
pub(crate) fn resolve_dflash_sidecar(
    params: &mut serde_json::Value,
    entry: Option<&ModelEntry>,
    models_dir: &Path,
    model_path: &Path,
    tag: Option<&str>,
) -> Result<()> {
    match locate_dflash_sidecar(params, entry, models_dir, model_path)? {
        DflashSidecar::Unresolved => {}
        DflashSidecar::Found(candidate) => {
            params["draft"] = serde_json::json!(candidate.display().to_string());
        }
        DflashSidecar::NotPulled(file) => {
            if params["dflash_mode"].as_str() == Some("on") {
                bail!(dflash_draft_not_pulled(&file, tag));
            }
            eprintln!(
                "[hipfire] DFlash draft {file} not pulled; no paired draft, speculation falls back — `hipfire pull {}`",
                tag.unwrap_or("<model>")
            );
        }
    }
    Ok(())
}

/// What [`locate_dflash_sidecar`] found for the final DFlash mode.
pub(crate) enum DflashSidecar {
    /// Nothing to wire: the final mode is `off`, a draft is already
    /// projected, or the artifact declares no sidecar under `auto`.
    Unresolved,
    /// The declared sidecar is on disk.
    Found(PathBuf),
    /// The registry declares this sidecar file but it is not pulled.
    NotPulled(String),
}

/// The `on`-mode refusal for a declared DFlash draft that is not pulled, worded
/// once for the load and for `hipfire fit`.
pub(crate) fn dflash_draft_not_pulled(file: &str, tag: Option<&str>) -> String {
    format!(
        "DFlash draft {file} is not pulled; run `hipfire pull {}` or set developer.dflash_draft",
        tag.unwrap_or("<model>")
    )
}

/// The lookup half of [`resolve_dflash_sidecar`], with no side effects: where
/// the registry DFlash draft is (or that it is declared but not pulled) for
/// the params' final `dflash_mode`. Errs exactly where the load refuses a
/// DFlash `on` that has no declared draft at all.
pub(crate) fn locate_dflash_sidecar(
    params: &serde_json::Value,
    entry: Option<&ModelEntry>,
    models_dir: &Path,
    model_path: &Path,
) -> Result<DflashSidecar> {
    if !matches!(params["dflash_mode"].as_str(), Some("auto" | "on")) {
        return Ok(DflashSidecar::Unresolved);
    }
    if params
        .get("draft")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|draft| !draft.is_empty())
    {
        return Ok(DflashSidecar::Unresolved);
    }
    let Some(sidecar) = entry.and_then(|entry| entry.dflash.as_ref()) else {
        if params["dflash_mode"].as_str() == Some("on") && model_path.is_file() {
            bail!(
                "DFlash needs a draft model: {} is not a registry-managed artifact; add --model-draft <draft.hfq>, or run the model by its registry tag (which pairs its draft)",
                model_path.display()
            );
        }
        return Ok(DflashSidecar::Unresolved);
    };
    let beside_target: Vec<PathBuf> = trunk_locations(model_path)
        .iter()
        .map(|location| {
            location
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        })
        .collect();
    let candidate = std::iter::once(models_dir)
        .chain(beside_target.iter().map(PathBuf::as_path))
        .map(|dir| dir.join(&sidecar.file))
        .find(|candidate| candidate.is_file());
    Ok(match candidate {
        Some(candidate) => DflashSidecar::Found(candidate),
        None => DflashSidecar::NotPulled(sidecar.file.clone()),
    })
}

/// Resolve the Qwen MTP head sidecar into `params["mtp"]`.
///
/// Call once the final `mtp_mode` is known (after the effective speculation
/// selector); `off` never carries a head. Otherwise the first existing file
/// wins: the registry `entry.mtp` file in `models_dir`, then `<entry
/// file>.mtp` in `models_dir`, then `.mtp` beside `requested` as typed (when
/// it names a file), then `.mtp` beside `model_path` as selected, then beside
/// its canonical target. `find_model_path` keeps a symlinked trunk's own
/// path, so a head beside the link or beside the target is found either way.
/// With no hit
/// `params["mtp"]` stays unset and the loader keeps its bundled-trailer and
/// `<trunk>.mtp` lookup (and `on` fails there); a registry-declared head that
/// is not pulled gets a one-line `hipfire pull` hint.
pub(crate) fn resolve_mtp_sidecar(
    params: &mut serde_json::Value,
    entry: Option<&ModelEntry>,
    models_dir: &Path,
    model_path: &Path,
    requested: &str,
    tag: Option<&str>,
) {
    if let Some(obj) = params.as_object_mut() {
        obj.remove("mtp");
    }
    if params["mtp_mode"].as_str() == Some("off") {
        return;
    }
    let mut candidates = Vec::new();
    if let Some(entry) = entry {
        if let Some(sidecar) = entry.mtp.as_ref() {
            candidates.push(models_dir.join(&sidecar.file));
        }
        candidates.push(models_dir.join(Path::new(&entry.file).with_extension("mtp")));
    }
    let requested = Path::new(requested);
    if requested.is_file() {
        // Absolute but not canonical: the daemon may run from another cwd,
        // and the point is to keep the symlink's own directory.
        let requested = std::path::absolute(requested).unwrap_or_else(|_| requested.to_path_buf());
        candidates.push(requested.with_extension("mtp"));
    }
    candidates.extend(
        trunk_locations(model_path)
            .iter()
            .map(|location| location.with_extension("mtp")),
    );
    if let Some(hit) = candidates.into_iter().find(|candidate| candidate.is_file()) {
        params["mtp"] = serde_json::json!(hit.display().to_string());
        return;
    }
    if let Some(sidecar) = entry.and_then(|entry| entry.mtp.as_ref()) {
        eprintln!(
            "[hipfire] MTP head {} not pulled; `hipfire pull {}` enables MTP speculation",
            sidecar.file,
            tag.unwrap_or("<model>")
        );
    }
}

/// Resolve a registry-declared XDNA spillover archive into `params["xdna"]`.
///
/// The `kernel.npu_spillover` process flag is the opt-in: when it is off this
/// is a no-op and no `xdna` param is projected, so flag-off loads can never
/// observe an NPU path. When it is on, the entry's `xdna` slot must name a
/// pulled `.xdna.zip` whose manifest verifies (`load_verified` checks shape,
/// exact `gfx1151`/`npu5` identity, and every payload hash); the verified
/// archive path is projected as `params["xdna"]` for the daemon, which
/// re-verifies at load and admits it into `LoadCtx.xdna` only on an exact
/// `gfx1151` host (see `admit_for_arch`).
///
/// Fail-closed only where an explicit opt-in names an artifact: flag on plus
/// a declared-but-missing or corrupt archive bails with a `hipfire pull`
/// hint. Flag on with no entry (a bare path that merely shares a basename)
/// or no `xdna` slot logs one line and stays GPU-only — erroring there would
/// make a process-global opt-in unusable for models that simply have no NPU
/// artifact. Lookup order (models dir, then beside the target) mirrors the
/// DFlash draft resolution, including the symlinked-target rationale.
fn resolve_xdna_sidecar(
    params: &mut serde_json::Value,
    resolved: &hipfire_config::ResolvedConfig,
    entry: Option<&ModelEntry>,
    models_dir: &Path,
    model_path: &Path,
    tag: Option<&str>,
) -> Result<()> {
    if !config_bool(resolved, "kernel.npu_spillover")? {
        return Ok(());
    }
    let Some(sidecar) = entry.and_then(|entry| entry.xdna.as_ref()) else {
        let what = tag
            .map(str::to_owned)
            .unwrap_or_else(|| model_path.display().to_string());
        eprintln!(
            "[hipfire] NPU spillover opted in but {what} declares no xdna sidecar; staying GPU-only",
        );
        return Ok(());
    };
    let beside_target: Vec<PathBuf> = trunk_locations(model_path)
        .iter()
        .map(|location| {
            location
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        })
        .collect();
    let candidate = std::iter::once(models_dir)
        .chain(beside_target.iter().map(PathBuf::as_path))
        .map(|dir| dir.join(&sidecar.file))
        .find(|candidate| candidate.is_file());
    let Some(candidate) = candidate else {
        bail!(
            "NPU spillover opted in but {} is not pulled; run `hipfire pull {}` (or turn kernel.npu_spillover off)",
            sidecar.file,
            tag.unwrap_or("<model>"),
        );
    };
    if let Err(error) = hipfire_registry::XdnaSidecarDescriptor::load_verified(&candidate) {
        bail!(
            "NPU spillover archive {} failed verification ({error:#}); re-pull with `hipfire pull {}`",
            candidate.display(),
            tag.unwrap_or("<model>"),
        );
    }
    params["xdna"] = serde_json::json!(candidate.display().to_string());
    Ok(())
}

/// Resolve a registry-declared vision-tower sidecar into `params["vision"]`.
///
/// Call only once the final `vision_mode` is known (`load_params` projects it
/// from `vision.mode`). `off` is a hard override mirroring the daemon's
/// `dflash_mode=off` guard: never carry a tower, even an explicitly projected
/// one. `auto` and `resident` use the registry/sibling sidecar when present
/// and run silently text-only when absent (they differ only in tower
/// residency, decided at load). `on` requires the declared sidecar and
/// fails the load closed when it cannot be resolved (same shape as the
/// DFlash `on` refusal). A trunk with an embedded tower declares no sidecar
/// and is unaffected by this key under every mode.
///
/// Priority under `auto`/`resident`/`on`: an already-projected `params["vision"]`
/// (e.g. `run --vision`) always wins; then `HIPFIRE_VISION_SIDECAR` (empty
/// string opts out, the same semantics as `HIPFIRE_DFLASH_DRAFT`); then
/// `entry.vision`, looked up in `models_dir` first — `find_model_path`
/// keeps the selected (possibly symlinked) path, so the artifact's real
/// directory is probed too, not only the models directory the sidecar was
/// pulled into — then next to the selected path and its canonical target,
/// then as the `<trunk-stem>-vision.hfq` sibling
/// convention beside any trunk.
fn resolve_vision_sidecar(
    params: &mut serde_json::Value,
    entry: Option<&ModelEntry>,
    models_dir: &Path,
    model_path: &Path,
    tag: Option<&str>,
) -> Result<()> {
    // Resolve in every mode. The daemon's `vision_mode=off` gate is the hard
    // override and the only place that decides; leaving the resolved path in
    // the params lets it tell an image request which sidecar it declined.
    let mode = params["vision_mode"]
        .as_str()
        .unwrap_or(hipfire_config::vision_mode_default());
    if !matches!(mode, "off" | "auto" | "resident" | "on") {
        return Ok(());
    }
    if let Some(projected) = params.get("vision").and_then(serde_json::Value::as_str) {
        if projected.is_empty() {
            if let Some(obj) = params.as_object_mut() {
                obj.remove("vision");
            }
            return Ok(());
        }
        return Ok(());
    }
    if let Ok(env) = hipfire_config::developer_var("HIPFIRE_VISION_SIDECAR") {
        if env.is_empty() {
            if let Some(obj) = params.as_object_mut() {
                obj.remove("vision");
            }
            return Ok(());
        }
        params["vision"] = serde_json::json!(env);
        return Ok(());
    }
    let locations = trunk_locations(model_path);
    let mut candidates = Vec::new();
    if let Some(sidecar) = entry.and_then(|entry| entry.vision.as_ref()) {
        candidates.push(models_dir.join(&sidecar.file));
        for location in &locations {
            candidates.push(
                location
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(&sidecar.file),
            );
        }
    }
    for location in &locations {
        let Some(stem) = location
            .file_name()
            .and_then(|file| file.to_str())
            .map(|file| file.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(file))
        else {
            continue;
        };
        candidates.push(
            location
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(format!("{stem}-vision.hfq")),
        );
    }
    if let Some(hit) = candidates.into_iter().find(|candidate| candidate.is_file()) {
        params["vision"] = serde_json::json!(hit.display().to_string());
        return Ok(());
    }
    if mode == "on" {
        if let Some(sidecar) = entry.and_then(|entry| entry.vision.as_ref()) {
            let tag = tag.unwrap_or("<model>");
            bail!(
                "Vision tower {} is not pulled; run `hipfire pull {tag}` or pass --vision",
                sidecar.file
            );
        }
    }
    Ok(())
}

/// Project snapshotted `developer.dflash_draft` after the effective speculation selector.
///
/// Call only once final `dflash_mode` is known. Config-off must not carry a draft;
/// a later CLI selector (e.g. `run --spec dflash`) can opt back in here.
fn project_dflash_draft(params: &mut serde_json::Value, draft: Option<&str>) {
    if params["dflash_mode"].as_str() == Some("off") {
        if let Some(obj) = params.as_object_mut() {
            obj.remove("draft");
        }
        return;
    }
    if let Some(draft) = draft {
        if !draft.is_empty() {
            params["draft"] = serde_json::json!(draft);
        }
    }
}

/// Optional draft path from resolved `developer.dflash_draft` (legacy HIPFIRE_DFLASH_DRAFT).
fn developer_dflash_draft(resolved: &hipfire_config::ResolvedConfig) -> Option<&str> {
    match resolved
        .get("developer.dflash_draft")
        .map(|item| &item.value)
    {
        Some(hipfire_config::ConfigValue::String(value)) => Some(value.as_str()),
        _ => None,
    }
}

pub(crate) fn apply_speculation_selector(
    params: &mut serde_json::Value,
    selector: &str,
) -> Result<()> {
    match selector {
        "off" => {
            params["dflash_mode"] = serde_json::json!("off");
            params["mtp_mode"] = serde_json::json!("off");
            params["ngram_draft"] = serde_json::json!(false);
            params["dspark_mode"] = serde_json::json!("off");
        }
        "dflash" => {
            params["dflash_mode"] = serde_json::json!("on");
            params["mtp_mode"] = serde_json::json!("off");
            params["ngram_draft"] = serde_json::json!(false);
            params["dspark_mode"] = serde_json::json!("off");
        }
        "mtp" => {
            params["dflash_mode"] = serde_json::json!("off");
            params["mtp_mode"] = serde_json::json!("on");
            params["ngram_draft"] = serde_json::json!(false);
            params["dspark_mode"] = serde_json::json!("off");
        }
        "ngram" => {
            params["dflash_mode"] = serde_json::json!("off");
            params["mtp_mode"] = serde_json::json!("off");
            params["ngram_draft"] = serde_json::json!(true);
            params["dspark_mode"] = serde_json::json!("off");
        }
        "dspark" => {
            params["dflash_mode"] = serde_json::json!("off");
            params["mtp_mode"] = serde_json::json!("off");
            params["ngram_draft"] = serde_json::json!(false);
            params["dspark_mode"] = serde_json::json!("on");
        }
        "auto" => {
            params["dspark_mode"] = serde_json::json!("auto");
        }
        other => bail!("unknown speculation selector '{other}'"),
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReasoningResolution {
    pub effective_mode: String,
    pub effective_effort: Option<String>,
    pub effective_cap: Option<u64>,
    pub cap_source: String,
    pub contract: ReasoningContract,
    pub warnings: Vec<String>,
}

/// Concrete effort resolved for effort-native QwenJinja when the request/config
/// leaves effort unset or `auto`. The raw Jinja templates still default to
/// `xhigh`; this CLI resolution layer supplies the concrete value.
const QWEN_JINJA_DEFAULT_EFFORT: &str = "medium";

pub(crate) fn apply_http_reasoning_request(
    body: &serde_json::Value,
    resolved: &hipfire_config::ResolvedConfig,
    request: &mut serde_json::Value,
    contract: ReasoningContract,
    effort_native: bool,
    supported_efforts: &[String],
) -> Result<ReasoningResolution> {
    let mut warnings: Vec<String> = Vec::new();
    let mut push_warn = |msg: String| {
        eprintln!("[WARN: INVALID CONFIG] {}", msg);
        warnings.push(msg);
    };
    if let Some(object) = request.as_object_mut() {
        for key in [
            "thinking_enabled",
            "assistant_prefix",
            "reasoning_effort",
            "max_think_tokens",
        ] {
            object.remove(key);
        }
    }
    if body.get("enable_thinking").is_some() {
        if let Some(value) = body.get("enable_thinking") {
            if !value.is_boolean() && !value.is_null() {
                serve::bail_invalid!("enable_thinking must be a boolean");
            }
        }
    }
    let top_enable = body
        .get("enable_thinking")
        .and_then(serde_json::Value::as_bool);
    if body
        .pointer("/chat_template_kwargs/enable_thinking")
        .is_some()
    {
        if let Some(value) = body.pointer("/chat_template_kwargs/enable_thinking") {
            if !value.is_boolean() && !value.is_null() {
                serve::bail_invalid!("chat_template_kwargs.enable_thinking must be a boolean");
            }
        }
    }
    let kwargs_enable = body
        .pointer("/chat_template_kwargs/enable_thinking")
        .and_then(serde_json::Value::as_bool);
    let thinking_type_raw = body
        .pointer("/thinking/type")
        .or_else(|| body.get("thinking").and_then(|value| value.get("type")));
    let mut thinking_type_str: Option<&str> = None;
    if let Some(raw) = thinking_type_raw {
        if !raw.is_string() {
            serve::bail_invalid!("thinking.type must be enabled or disabled");
        } else {
            let s = raw.as_str().unwrap();
            if s == "enabled" || s == "disabled" {
                thinking_type_str = Some(s);
            } else {
                push_warn(format!(
                    "thinking.type '{}' dropped: expected enabled|disabled",
                    s
                ));
                thinking_type_str = None;
            }
        }
    }
    let effort_raw = body
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            body.pointer("/reasoning/effort")
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            body.pointer("/chat_template_kwargs/reasoning_effort")
                .and_then(serde_json::Value::as_str)
        });
    let effort_present = body.get("reasoning_effort").is_some()
        || body.pointer("/reasoning/effort").is_some()
        || body
            .pointer("/chat_template_kwargs/reasoning_effort")
            .is_some();
    if effort_present && effort_raw.is_none() {
        serve::bail_invalid!("reasoning_effort must be a string");
    }
    let body_budget_present = body.get("thinking_budget").is_some();
    let body_budget_str = body
        .get("thinking_budget")
        .and_then(serde_json::Value::as_str);
    if body_budget_present && body_budget_str.is_none() {
        serve::bail_invalid!("thinking_budget must be a string preset");
    }
    let body_top_max_present = body.get("max_think_tokens").is_some();
    let body_nested_max_present = body.pointer("/reasoning/max_tokens").is_some();
    let parse_body_think_cap = |value: &serde_json::Value, field: &str| -> Result<u64> {
        match value {
            serde_json::Value::Number(number) => {
                if let Some(parsed) = number.as_u64() {
                    if parsed > 393_216 {
                        serve::bail_invalid!("{field} must be between 0 and 393216");
                    }
                    Ok(parsed)
                } else {
                    serve::bail_invalid!("{field} must be between 0 and 393216");
                }
            }
            _ => serve::bail_invalid!("{field} must be between 0 and 393216"),
        }
    };
    let top_max_opt = if body_top_max_present {
        Some(parse_body_think_cap(
            body.get("max_think_tokens").unwrap(),
            "max_think_tokens",
        )?)
    } else {
        None
    };
    let nested_max_opt = if body_nested_max_present {
        Some(parse_body_think_cap(
            body.pointer("/reasoning/max_tokens").unwrap(),
            "reasoning.max_tokens",
        )?)
    } else {
        None
    };
    let (max_opt, body_max_source) = match (top_max_opt, nested_max_opt) {
        (Some(top), Some(nested)) => {
            if top != nested {
                push_warn(
                    "reasoning.max_tokens dropped because explicit max_think_tokens takes precedence"
                        .to_string(),
                );
            }
            (Some(top), "explicit:body:max_think_tokens")
        }
        (Some(top), None) => (Some(top), "explicit:body:max_think_tokens"),
        (None, Some(nested)) => (Some(nested), "explicit:body:reasoning.max_tokens"),
        (None, None) => (None, ""),
    };
    let has_explicit_body_max = body_top_max_present || body_nested_max_present;
    let config_max_entry = resolved.get("reasoning.max_tokens").filter(|value| {
        !matches!(value.source, hipfire_config::ConfigSource::BuiltIn)
            && !matches!(value.value, hipfire_config::ConfigValue::Null)
    });
    let mut config_max_opt: Option<u64> = None;
    if let Some(entry) = config_max_entry {
        match entry.value {
            hipfire_config::ConfigValue::Integer(value) if value >= 0 => {
                config_max_opt = Some(value as u64);
            }
            _ => bail!("reasoning.max_tokens resolved to a non-negative integer"),
        }
    }
    let has_explicit_config_max = config_max_opt.is_some();
    let config_budget_entry = resolved
        .get("reasoning.budget")
        .filter(|value| !matches!(value.source, hipfire_config::ConfigSource::BuiltIn));
    let has_explicit_config_budget = config_budget_entry.is_some();
    let config_budget_str: Option<String> = if has_explicit_config_budget {
        Some(config_string(resolved, "reasoning.budget")?)
    } else {
        None
    };
    let has_explicit_effort = effort_raw.is_some();
    let has_explicit_toggle =
        top_enable.is_some() || kwargs_enable.is_some() || thinking_type_str.is_some();
    let is_effort_native = match contract {
        ReasoningContract::QwenJinja => effort_native,
        ReasoningContract::DeepSeek4 => true,
        ReasoningContract::MuseGlimmer => true,
        _ => false,
    };
    if matches!(contract, ReasoningContract::Unsupported) {
        if has_explicit_toggle {
            push_warn(
                "reasoning controls dropped for unsupported contract: thinking toggle ignored"
                    .to_string(),
            );
        }
        if has_explicit_effort {
            push_warn(format!(
                "reasoning_effort '{}' dropped for unsupported contract",
                effort_raw.unwrap_or("unknown")
            ));
        }
        if has_explicit_body_max
            || body_budget_present
            || has_explicit_config_max
            || has_explicit_config_budget
        {
            push_warn("max_think_tokens/budget dropped for unsupported contract".to_string());
        }
        let resolution = ReasoningResolution {
            effective_mode: "disabled".to_string(),
            effective_effort: None,
            effective_cap: None,
            cap_source: "none".to_string(),
            contract,
            warnings: warnings.clone(),
        };
        return Ok(resolution);
    }
    if matches!(contract, ReasoningContract::GemmaBoolean) {
        if has_explicit_effort {
            push_warn(format!(
                "reasoning_effort '{}' dropped for gemma_boolean: use thinking toggle only",
                effort_raw.unwrap()
            ));
        }
        if body_budget_present {
            push_warn(format!(
                "thinking_budget '{}' dropped for gemma_boolean",
                body_budget_str.unwrap()
            ));
        }
        if has_explicit_body_max {
            push_warn(format!(
                "{} {} dropped for gemma_boolean: use thinking toggle only",
                if body_top_max_present {
                    "max_think_tokens"
                } else {
                    "reasoning.max_tokens"
                },
                max_opt.unwrap()
            ));
        }
        if has_explicit_config_max {
            push_warn(format!(
                "reasoning.max_tokens {} dropped for gemma_boolean: use thinking toggle only",
                config_max_opt.unwrap()
            ));
        }
        if has_explicit_config_budget {
            push_warn(format!(
                "thinking_budget '{}' dropped for gemma_boolean",
                config_budget_str.clone().unwrap()
            ));
        }
    }
    let thinking_type_toggle = thinking_type_str.map(|value| value == "enabled");
    let toggle_values = [top_enable, kwargs_enable, thinking_type_toggle];
    let saw_enabled = toggle_values.iter().flatten().any(|value| *value);
    let saw_disabled = toggle_values.iter().flatten().any(|value| !*value);
    if saw_enabled && saw_disabled {
        push_warn("conflicting thinking toggles normalized: disabled wins".to_string());
    }
    let mut toggle_opt = toggle_values
        .into_iter()
        .flatten()
        .reduce(|previous, value| previous && value);
    let is_off_effort = matches!(
        contract,
        ReasoningContract::QwenJinja | ReasoningContract::DeepSeek4
    ) && matches!(effort_raw, Some("none") | Some("off") | Some("chat"));
    if is_off_effort {
        if toggle_opt == Some(true) {
            push_warn(
                "thinking enabled conflicts with off/none effort; thinking disabled wins"
                    .to_string(),
            );
        }
        toggle_opt = Some(false);
    }
    let is_off_budget = !is_effort_native
        && !matches!(contract, ReasoningContract::GemmaBoolean)
        && body_budget_str == Some("off");
    if is_off_budget {
        if toggle_opt == Some(true) {
            push_warn(
                "thinking enabled conflicts with legacy budget off; thinking disabled wins"
                    .to_string(),
            );
        }
        toggle_opt = Some(false);
    }
    // `reasoning_budget_tokens == 0` is the wire spelling hag and other OpenAI
    // clients use for "do not think". Absent from master but genuine: without
    // it a small max_tokens budget is burned inside <think> and the client gets
    // empty content. Route through toggle reduction so it inherits contract
    // gating and disabled-wins warnings, like is_off_budget.
    let is_off_reasoning_budget_tokens = !is_effort_native
        && !matches!(contract, ReasoningContract::GemmaBoolean)
        && body
            .get("reasoning_budget_tokens")
            .and_then(serde_json::Value::as_u64)
            == Some(0);
    if is_off_reasoning_budget_tokens {
        if toggle_opt == Some(true) {
            push_warn(
                "thinking enabled conflicts with reasoning_budget_tokens 0; thinking disabled wins"
                    .to_string(),
            );
        }
        toggle_opt = Some(false);
    }
    // `max_think_tokens == 1` is the engine's own lowered encoding of off
    // (alongside assistant_prefix=closed_think). Incoming 1 must not be a
    // contract-blind short-circuit: a client legitimately requesting a 1-token
    // cap would be silently reinterpreted. Gate on contract and participate in
    // disabled-wins, so every off-path is contract-aware.
    let is_off_max_think_one = matches!(contract, ReasoningContract::QwenJinja)
        && body
            .get("max_think_tokens")
            .and_then(serde_json::Value::as_u64)
            == Some(1);
    if is_off_max_think_one {
        if toggle_opt == Some(true) {
            push_warn(
                "thinking enabled conflicts with max_think_tokens 1; thinking disabled wins"
                    .to_string(),
            );
        }
        toggle_opt = Some(false);
    }
    let config_mode = config_string(resolved, "reasoning.mode").unwrap_or_else(|_| "on".into());
    let config_mode_is_explicit = resolved
        .get("reasoning.mode")
        .is_some_and(|value| !matches!(value.source, hipfire_config::ConfigSource::BuiltIn));
    let config_effort =
        config_string(resolved, "reasoning.effort").unwrap_or_else(|_| "auto".into());
    let config_effort_is_explicit = resolved
        .get("reasoning.effort")
        .is_some_and(|value| !matches!(value.source, hipfire_config::ConfigSource::BuiltIn));
    let config_budget =
        config_string(resolved, "reasoning.budget").unwrap_or_else(|_| "uncapped".into());
    let config_budget_is_explicit = resolved
        .get("reasoning.budget")
        .is_some_and(|value| !matches!(value.source, hipfire_config::ConfigSource::BuiltIn));
    let configured_off = (config_mode_is_explicit && config_mode == "off")
        || (config_effort_is_explicit && config_effort == "none")
        || (!is_effort_native
            && !matches!(contract, ReasoningContract::GemmaBoolean)
            && config_budget_is_explicit
            && config_budget == "off");
    let family_default = !matches!(contract, ReasoningContract::GemmaBoolean);
    let mut thinking_enabled = toggle_opt.unwrap_or_else(|| {
        if configured_off {
            false
        } else if config_mode_is_explicit {
            config_mode != "off"
        } else {
            family_default
        }
    });
    if matches!(contract, ReasoningContract::MuseGlimmer) && !thinking_enabled {
        push_warn("reasoning off dropped for muse_glimmer: always-on reasoning".to_string());
        thinking_enabled = true;
    }
    request["thinking_enabled"] = serde_json::json!(thinking_enabled);
    let prefix = match contract {
        ReasoningContract::QwenJinja | ReasoningContract::DeepSeek4 => {
            if thinking_enabled {
                "open_think"
            } else {
                "closed_think"
            }
        }
        ReasoningContract::GemmaBoolean | ReasoningContract::MuseGlimmer => "plain",
        ReasoningContract::Unsupported => "plain",
    };
    request["assistant_prefix"] = serde_json::json!(prefix);
    if !thinking_enabled {
        if has_explicit_effort {
            push_warn(format!(
                "reasoning_effort '{}' dropped: thinking disabled",
                effort_raw.unwrap()
            ));
        }
        if config_effort_is_explicit && config_effort != "auto" && config_effort != "none" {
            push_warn(format!(
                "reasoning.effort '{}' dropped: thinking disabled",
                config_effort
            ));
        }
        let cap_present = has_explicit_body_max
            || body_budget_present
            || has_explicit_config_max
            || has_explicit_config_budget;
        if cap_present {
            push_warn("max_think_tokens/budget dropped: thinking disabled".to_string());
        }
        let resolution = ReasoningResolution {
            effective_mode: "disabled".to_string(),
            effective_effort: None,
            effective_cap: None,
            cap_source: "none".to_string(),
            contract,
            warnings: warnings.clone(),
        };
        return Ok(resolution);
    }
    if matches!(contract, ReasoningContract::QwenJinja) && is_effort_native && body_budget_present {
        push_warn(format!(
            "thinking_budget '{}' ignored for effort-native contract {}: use explicit max_think_tokens for cap",
            body_budget_str.unwrap(),
            contract.wire_name()
        ));
    }
    if matches!(contract, ReasoningContract::QwenJinja)
        && is_effort_native
        && has_explicit_config_budget
    {
        push_warn(format!(
            "thinking_budget '{}' ignored for effort-native contract {}: use explicit max_think_tokens for cap",
            config_budget_str.clone().unwrap(),
            contract.wire_name()
        ));
    }

    let (mut effective_cap, mut cap_source) = if matches!(contract, ReasoningContract::GemmaBoolean)
    {
        (None, "none".to_string())
    } else if matches!(
        contract,
        ReasoningContract::DeepSeek4 | ReasoningContract::MuseGlimmer
    ) && (has_explicit_body_max
        || has_explicit_config_max
        || body_budget_present
        || has_explicit_config_budget)
    {
        if has_explicit_body_max {
            push_warn(format!(
                "{} {} dropped for {}: use reasoning_effort only",
                if body_top_max_present {
                    "max_think_tokens"
                } else {
                    "reasoning.max_tokens"
                },
                max_opt.unwrap(),
                contract.wire_name()
            ));
        }
        if has_explicit_config_max {
            push_warn(format!(
                "reasoning.max_tokens {} dropped for {}: use reasoning_effort only",
                config_max_opt.unwrap(),
                contract.wire_name()
            ));
        }
        if body_budget_present {
            push_warn(format!(
                "thinking_budget '{}' dropped for {}: use reasoning_effort only",
                body_budget_str.unwrap(),
                contract.wire_name()
            ));
        }
        if has_explicit_config_budget {
            push_warn(format!(
                "thinking_budget '{}' dropped for {}: use reasoning_effort only",
                config_budget_str.clone().unwrap(),
                contract.wire_name()
            ));
        }
        (None, "none".to_string())
    } else if has_explicit_body_max {
        if body_budget_present {
            push_warn(
                "thinking_budget dropped because explicit max_think_tokens takes precedence"
                    .to_string(),
            );
        }
        (max_opt, body_max_source.to_string())
    } else if !is_effort_native && body_budget_present {
        let budget_str = body_budget_str.unwrap();
        let mapped = match budget_str {
            "off" => None,
            "low" => Some(512),
            "med" => Some(2048),
            "high" => Some(8192),
            "xhigh" => Some(24576),
            "max" => Some(32768),
            "uncapped" => Some(0),
            other => {
                push_warn(format!(
                    "thinking_budget '{}' dropped: unknown preset",
                    other
                ));
                None
            }
        };
        if mapped.is_none() && budget_str != "off" && {
            let known = ["low", "med", "high", "xhigh", "max", "uncapped"];
            !known.contains(&budget_str)
        } {
            (None, "none".to_string())
        } else {
            (mapped, "explicit:body:thinking_budget".to_string())
        }
    } else if has_explicit_config_max {
        (config_max_opt, "config:reasoning.max_tokens".to_string())
    } else if !is_effort_native && has_explicit_config_budget {
        let budget_str = config_budget_str.as_deref().unwrap();
        let mapped = match budget_str {
            "off" => None,
            "low" => Some(512),
            "med" => Some(2048),
            "high" => Some(8192),
            "xhigh" => Some(24576),
            "max" => Some(32768),
            "uncapped" => Some(0),
            other => {
                push_warn(format!(
                    "thinking_budget '{}' dropped: unknown preset",
                    other
                ));
                None
            }
        };
        if mapped.is_none() && budget_str != "off" && {
            let known = ["low", "med", "high", "xhigh", "max", "uncapped"];
            !known.contains(&budget_str)
        } {
            (None, "none".to_string())
        } else {
            (mapped, "config:reasoning.budget".to_string())
        }
    } else {
        (None, "none".to_string())
    };
    if effective_cap == Some(0) {
        effective_cap = None;
        cap_source = "none".to_string();
    }
    if let Some(value) = effective_cap {
        request["max_think_tokens"] = serde_json::json!(value);
    }
    let mut effective_effort: Option<String> = None;
    match contract {
        ReasoningContract::QwenJinja => {
            if !effort_native {
                if has_explicit_effort {
                    push_warn(format!(
                        "reasoning_effort '{}' dropped: loaded chat template does not natively support effort, so native effort is unavailable for this model; use thinking_budget or max_think_tokens to cap thinking",
                        effort_raw.unwrap()
                    ));
                }
                if config_effort_is_explicit && config_effort != "auto" {
                    push_warn(format!(
                        "reasoning.effort '{}' dropped: loaded chat template does not natively support effort, so native effort is unavailable for this model; use thinking_budget or max_think_tokens to cap thinking",
                        config_effort
                    ));
                }
                effective_effort = None;
            } else if has_explicit_effort || config_effort_is_explicit {
                let raw = effort_raw
                    .filter(|value| *value != "auto")
                    .map(str::to_owned)
                    .unwrap_or_else(|| config_effort.clone());
                if raw == "auto" {
                    effective_effort = Some(QWEN_JINJA_DEFAULT_EFFORT.to_string());
                    request["reasoning_effort"] = serde_json::json!(QWEN_JINJA_DEFAULT_EFFORT);
                } else {
                    match raw.as_str() {
                        "low" | "medium" | "xhigh" => {
                            if !supported_efforts.is_empty() && !supported_efforts.contains(&raw) {
                                push_warn(format!(
                                    "reasoning_effort '{}' dropped: not in supported {:?}",
                                    raw, supported_efforts
                                ));
                                effective_effort = Some(QWEN_JINJA_DEFAULT_EFFORT.to_string());
                                request["reasoning_effort"] =
                                    serde_json::json!(QWEN_JINJA_DEFAULT_EFFORT);
                            } else {
                                effective_effort = Some(raw.clone());
                                request["reasoning_effort"] = serde_json::json!(raw);
                            }
                        }
                        "high" | "max" | "minimal" | "med" => {
                            // Off-ladder aliases normalize to the nearest
                            // supported tier instead of dropping to the
                            // default, so intent-ordered clients
                            // (minimal<low<medium<high<xhigh<max) keep their
                            // ordering. When the model's advertised ladder
                            // excludes the nearest tier, fall as before.
                            let nearest: &str = match raw.as_str() {
                                "minimal" => "low",
                                "med" => "medium",
                                _ => "xhigh",
                            };
                            let target = if !supported_efforts.is_empty()
                                && !supported_efforts.iter().any(|e| e == nearest)
                            {
                                QWEN_JINJA_DEFAULT_EFFORT
                            } else {
                                nearest
                            };
                            push_warn(format!(
                                "reasoning_effort '{}' normalized to {target} (qwen_jinja ladder is low|medium|xhigh)",
                                raw
                            ));
                            effective_effort = Some(target.to_string());
                            request["reasoning_effort"] = serde_json::json!(target);
                        }
                        "native" => {
                            push_warn(format!(
                                "reasoning_effort 'native' is not a supported effort value for effort-native qwen_jinja: expected low|medium|xhigh; using default {}",
                                QWEN_JINJA_DEFAULT_EFFORT
                            ));
                            effective_effort = Some(QWEN_JINJA_DEFAULT_EFFORT.to_string());
                            request["reasoning_effort"] =
                                serde_json::json!(QWEN_JINJA_DEFAULT_EFFORT);
                        }
                        other => {
                            push_warn(format!(
                                "reasoning_effort '{}' normalized to qwen_jinja default {}",
                                other, QWEN_JINJA_DEFAULT_EFFORT
                            ));
                            effective_effort = Some(QWEN_JINJA_DEFAULT_EFFORT.to_string());
                            request["reasoning_effort"] =
                                serde_json::json!(QWEN_JINJA_DEFAULT_EFFORT);
                        }
                    }
                }
            } else {
                let default = QWEN_JINJA_DEFAULT_EFFORT.to_string();
                effective_effort = Some(default.clone());
                request["reasoning_effort"] = serde_json::json!(default);
            }
        }
        ReasoningContract::DeepSeek4 => {
            let raw = if let Some(value) = effort_raw {
                value.to_owned()
            } else {
                let cfg = config_string(resolved, "reasoning.effort")
                    .unwrap_or_else(|_| "auto".to_string());
                if cfg != "auto" {
                    cfg
                } else {
                    "high".to_string()
                }
            };
            let normalized = match raw.as_str() {
                "minimal" => "low",
                "low" => "low",
                "medium" | "med" => "high",
                "xhigh" => "high",
                "high" => "high",
                "max" => "max",
                other => {
                    push_warn(format!(
                        "reasoning_effort '{}' normalized to deepseek4 default high",
                        other
                    ));
                    "high"
                }
            };
            effective_effort = Some(normalized.to_string());
            request["reasoning_effort"] = serde_json::json!(normalized);
        }
        ReasoningContract::MuseGlimmer => {
            let raw = if let Some(value) = effort_raw {
                value.to_owned()
            } else {
                let cfg = config_string(resolved, "reasoning.effort")
                    .unwrap_or_else(|_| "auto".to_string());
                if cfg != "auto" {
                    match cfg.as_str() {
                        "low" | "medium" | "high" | "xhigh" | "max" => cfg,
                        other => {
                            push_warn(format!(
                                "reasoning.effort '{}' normalized to muse_glimmer default high",
                                other
                            ));
                            "high".to_string()
                        }
                    }
                } else {
                    "high".to_string()
                }
            };
            let normalized = match raw.as_str() {
                "low" => "low",
                "medium" | "med" => "medium",
                "high" => "high",
                "xhigh" => "xhigh",
                "max" => "xhigh",
                other => {
                    push_warn(format!(
                        "reasoning_effort '{}' normalized to muse_glimmer default high",
                        other
                    ));
                    "high"
                }
            };
            effective_effort = Some(normalized.to_string());
            request["reasoning_effort"] = serde_json::json!(normalized);
        }
        ReasoningContract::GemmaBoolean => {
            effective_effort = None;
        }
        ReasoningContract::Unsupported => {
            effective_effort = None;
        }
    }
    let resolution = ReasoningResolution {
        effective_mode: if thinking_enabled {
            "enabled".to_string()
        } else {
            "disabled".to_string()
        },
        effective_effort,
        effective_cap,
        cap_source,
        contract,
        warnings: warnings.clone(),
    };
    Ok(resolution)
}

pub(crate) fn config_value<'a>(
    resolved: &'a hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<&'a hipfire_config::ConfigValue> {
    resolved
        .get(key)
        .map(|value| &value.value)
        .ok_or_else(|| anyhow!("missing resolved configuration key {key}"))
}

pub(crate) fn config_string(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<String> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::String(value) => Ok(value.clone()),
        value => bail!("{key} resolved as {}, expected string", value.kind()),
    }
}

pub(crate) fn config_bool(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<bool> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Bool(value) => Ok(*value),
        value => bail!("{key} resolved as {}, expected bool", value.kind()),
    }
}

pub(crate) fn config_i64(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<i64> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Integer(value) => Ok(*value),
        value => bail!("{key} resolved as {}, expected integer", value.kind()),
    }
}

pub(crate) fn config_u64(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<u64> {
    let value = config_i64(resolved, key)?;
    u64::try_from(value).map_err(|_| anyhow!("{key} cannot be negative"))
}

pub(crate) fn config_optional_u64(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<Option<u64>> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Null => Ok(None),
        hipfire_config::ConfigValue::Integer(value) => u64::try_from(*value)
            .map(Some)
            .map_err(|_| anyhow!("{key} cannot be negative")),
        value => bail!(
            "{key} resolved as {}, expected integer or null",
            value.kind()
        ),
    }
}

pub(crate) fn config_f64(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<f64> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Integer(value) => Ok(*value as f64),
        hipfire_config::ConfigValue::Float(value) => Ok(*value),
        value => bail!("{key} resolved as {}, expected number", value.kind()),
    }
}

fn launch_tui(paths: &Paths, arguments: &[String]) -> Result<()> {
    let executable = env::var_os("HIPFIRE_TUI_BIN")
        .map(PathBuf::from)
        .or_else(|| {
            let installed = paths.root.join("bin/hipfire-tui");
            installed.is_file().then_some(installed)
        })
        .or_else(|| {
            let workspace =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/release/hipfire-tui");
            workspace.is_file().then_some(workspace)
        })
        .or_else(|| find_on_path("hipfire-tui"))
        .ok_or_else(|| {
            anyhow!(
                "hipfire-tui is not installed; build it with `cargo build --release -p hipfire-tui`"
            )
        })?;
    let status = Command::new(&executable)
        .args(arguments)
        .status()
        .with_context(|| format!("failed to launch {}", executable.display()))?;
    if status.success() {
        Ok(())
    } else {
        bail!("hipfire-tui exited with {status}")
    }
}

#[derive(Debug, Serialize)]
struct ProcessRecord {
    pid: u32,
    rss_mb: u64,
    command: String,
}

fn scan_auxiliary_processes() -> Result<(Vec<ProcessRecord>, Vec<ProcessRecord>), String> {
    let snapshots = process_identity::scan_processes(|command| {
        command.contains("hf upload")
            || command.contains("hipfire-quantize")
            || command.contains("hipfire quantize")
    })?;
    let mut quantize = Vec::new();
    let mut uploads = Vec::new();
    for snapshot in snapshots {
        let record = ProcessRecord {
            pid: snapshot.pid,
            rss_mb: snapshot.rss_mb,
            command: snapshot.command,
        };
        if record.command.contains("hf upload") {
            uploads.push(record);
        } else {
            quantize.push(record);
        }
    }
    Ok((quantize, uploads))
}

fn ps_command(paths: &Paths, output: OutputArgs) -> Result<()> {
    let (_, resolved) = resolved_global(paths, true)?;
    let host = config_string(&resolved, "serve.host")?;
    let port = serve_probe_port(paths, config_u64(&resolved, "serve.port")? as u16);
    let endpoint = local_service_endpoint(&resolved, &host, port)?;
    let pid_path = paths.root.join("serve.pid");
    let pid_record = fs::read_to_string(&pid_path)
        .ok()
        .and_then(|raw| parse_pid_record(&raw));
    let pid = pid_record.as_ref().map(|record| record.pid);
    let alive = pid.is_some_and(process_identity::pid_alive);
    let health = http_get_json(&endpoint, "/health", Duration::from_secs(1)).ok();
    let stats = http_get_json(&endpoint, "/stats", Duration::from_secs(1)).ok();
    let (quantize, uploads, auxiliary_unavailable) = match scan_auxiliary_processes() {
        Ok((quantize, uploads)) => (quantize, uploads, None),
        Err(reason) => (Vec::new(), Vec::new(), Some(reason)),
    };
    let mut report = serde_json::json!({
        "running": health.is_some(),
        "pid": pid,
        "pid_record": pid_record.as_ref().map(ServePidRecord::to_public_json),
        "pid_alive": alive,
        "endpoint": service_url(&endpoint, ""),
        "health": health,
        "stats": stats,
        "quantize": quantize,
        "uploads": uploads,
    });
    if let Some(reason) = &auxiliary_unavailable {
        report["auxiliary_scan_unavailable"] = serde_json::json!(reason);
    }
    if output.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if report["running"].as_bool() == Some(true) {
        println!(
            "hipfire serve is online at {}",
            report["endpoint"].as_str().unwrap()
        );
        println!(
            "  pid:       {}",
            pid.map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".into())
        );
        println!(
            "  model:     {}",
            report
                .pointer("/health/model")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("idle")
        );
        println!(
            "  requests:  {}",
            report
                .pointer("/stats/requests_served")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
        );
        if let Some(tok_s) = report
            .pointer("/stats/recent_tok_s")
            .and_then(serde_json::Value::as_f64)
        {
            println!("  recent:    {tok_s:.2} tok/s");
        }
    } else if alive {
        println!(
            "hipfire serve PID {} is alive but HTTP is not ready",
            pid.unwrap()
        );
    } else {
        println!("hipfire serve is not running");
    }
    for (label, records) in [("quantize", &quantize), ("HF upload", &uploads)] {
        for process in records {
            println!(
                "{label}: PID {}  {} MB  {}",
                process.pid, process.rss_mb, process.command
            );
        }
    }
    if !output.json {
        if let Some(reason) = &auxiliary_unavailable {
            println!("note: {reason}");
        }
    }
    Ok(())
}

pub(crate) fn local_service_endpoint(
    resolved: &hipfire_config::ResolvedConfig,
    host: &str,
    port: u16,
) -> Result<ServiceEndpoint> {
    let key_path = config_string(resolved, "serve.api_key_file")?;
    let bearer = if key_path.is_empty() {
        None
    } else {
        Some(hipfire_config::secret::BearerToken::load(Path::new(
            &key_path,
        ))?)
    };
    Ok(ServiceEndpoint::local(host, port, bearer)?)
}

#[derive(Clone, Copy, Debug, Serialize)]
struct SampleStats {
    median: f64,
    mean: f64,
    min: f64,
    max: f64,
    stdev: f64,
}

fn sample_stats(values: &[f64]) -> Option<SampleStats> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / sorted.len() as f64;
    Some(SampleStats {
        median,
        mean,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        stdev: variance.sqrt(),
    })
}

/// Default standard-bench prompt. Historical numbers depend on its exact
/// bytes; do not change it.
const BENCH_DEFAULT_PROMPT: &str = "Explain the theory of general relativity in simple terms.";

/// Below this tokenized prompt length the measured `prefill_tok_s` is launch
/// overhead, not prefill throughput (363 tok/s at ~24 tokens vs 886 at 4.4k
/// on a 7900 XTX with the same binary).
const BENCH_PREFILL_EVIDENCE_TOKENS: u64 = 256;

/// Resolve the standard-benchmark prompt: `--prompt-file` reads the file
/// verbatim (raw bytes, no trimming — one newline can move τ by 17%), else
/// the positional words joined with spaces, else the historical default.
/// The clap `conflicts_with` on `--prompt-file` covers CLI parsing; the
/// explicit check here covers programmatically built args.
fn resolve_bench_prompt(args: &BenchArgs) -> Result<String> {
    if let Some(path) = args.prompt_file.as_deref() {
        if !args.prompt.is_empty() {
            bail!("--prompt-file cannot be combined with a positional prompt");
        }
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read --prompt-file {}", path.display()))?;
        return String::from_utf8(bytes)
            .with_context(|| format!("--prompt-file {} is not valid UTF-8", path.display()));
    }
    Ok(if args.prompt.is_empty() {
        BENCH_DEFAULT_PROMPT.to_owned()
    } else {
        args.prompt.join(" ")
    })
}

/// Hex md5 of the exact prompt bytes sent, so two bench numbers are only
/// compared when their prompts are byte-identical.
fn bench_prompt_md5(prompt: &str) -> String {
    format!("{:x}", md5::compute(prompt.as_bytes()))
}

/// Short-prompt caveat: below 256 prompt tokens `prefill_tok_s` measures
/// launch overhead, not prefill throughput.
fn bench_prompt_warning(prompt_tokens: u64) -> Option<String> {
    (prompt_tokens < BENCH_PREFILL_EVIDENCE_TOKENS).then(|| {
        format!(
            "prompt is {prompt_tokens} tokens; prefill_tok_s at this length measures launch overhead, not prefill throughput — use --prompt-file with ≥256 tokens for a prefill number"
        )
    })
}

/// Prompt length as the daemon reports it on the `done` event:
/// `prefill_tokens` (rows actually prefilled) plus `cached_tokens` (prefix
/// served from the prompt cache). `None` when the event carries neither.
fn bench_prompt_tokens_from_done(done: &serde_json::Value) -> Option<u64> {
    let prefill = done
        .get("prefill_tokens")
        .and_then(serde_json::Value::as_u64);
    let cached = done
        .get("cached_tokens")
        .and_then(serde_json::Value::as_u64);
    match (prefill, cached) {
        (None, None) => None,
        (p, c) => Some(p.unwrap_or(0) + c.unwrap_or(0)),
    }
}

/// Exit status of a `hipfire bench` stopped by SIGINT/SIGTERM/SIGHUP (the
/// conventional 128 + SIGINT; `ctrlc` does not report which signal arrived).
const BENCH_INTERRUPTED_EXIT: i32 = 130;

// Publish before killing a daemon: its stdout EOF can make the command return
// an error on the main thread before the interrupt handler finishes reaping.
static BENCH_INTERRUPT: AtomicBool = AtomicBool::new(false);

/// Make an interrupted bench take its daemon with it. A process stopped by a
/// signal never runs `Drop`, so the default disposition would leave the daemon
/// (spawned through `Engine`) running: still loading or compiling kernels, and
/// holding the machine-wide GPU `flock`, so the next claim of those cards
/// fails with "already reserved by holder PID …". A harness timeout that
/// SIGTERMs only this process is exactly that case. The handler kills and
/// reaps every daemon *this process* spawned (never an unrelated one) before
/// exiting; reaping returns only after the daemon's teardown has released the
/// lock, so once `hipfire bench` has exited the cards are claimable again.
/// An already-installed process handler is left in place (`AlreadyExists`).
fn install_bench_interrupt_handler() {
    let _ = ctrlc::set_handler(|| {
        BENCH_INTERRUPT.store(true, Ordering::SeqCst);
        let reaped = Engine::terminate_spawned();
        eprintln!("hipfire bench: interrupted; terminated {reaped} owned daemon(s)");
        std::process::exit(BENCH_INTERRUPTED_EXIT);
    });
}

fn bench_command(paths: &Paths, args: BenchArgs) -> Result<()> {
    // An explicit `--emu` is validated before anything else (model lookup, pull,
    // daemon spawn): an unknown id or a UMA-knob conflict is a hard error here.
    emu::preflight_bench(paths, args.emu.as_deref())?;
    install_bench_interrupt_handler();
    if args.runs == 0 {
        bail!("--runs must be positive");
    }
    if args.exp && (args.matrix || args.redline) {
        bail!("--exp cannot be combined with matrix or Redline options");
    }
    // The matrix is the synthetic `bench_prefill`/`bench_decode` protocol:
    // its prefill rows are fixed token ids on the plain prefill route, never a
    // prompt and never the speculative prompt prefill `serve` runs. Refuse a
    // prompt instead of silently measuring something else.
    if (args.matrix || args.redline) && (args.prompt_file.is_some() || !args.prompt.is_empty()) {
        bail!(
            "--matrix/--redline measure synthetic token ids (bench_prefill) and cannot take a prompt; \
             for a prompt's prefill with the configured speculation use the standard bench \
             (`hipfire bench MODEL --prompt-file FILE --max-tokens 1`)"
        );
    }
    if args.ttft && args.matrix {
        bail!("--ttft cannot be combined with --matrix (it measures the standard generate path)");
    }
    if args.exp && args.json {
        bail!("--json is not supported with --exp");
    }
    // --exp runs a fixed 128-token protocol across five RDNA2 variants and
    // ignores --max-tokens, so it can never give a think span room to close.
    // Reject the combination rather than silently drop the flag and abort
    // later on an open-think terminal.
    if args.exp && args.reasoning_on {
        bail!("--reasoning-on is not supported with --exp (its token budget is fixed at 128)");
    }
    if let Some(spec) = args.concurrency.clone() {
        if args.json {
            bail!(
                "--json is not supported with --concurrency; omit --json for the concurrency table, \
                 or run without --concurrency for standard bench JSON (including kv_backend fields)"
            );
        }
        return bench_concurrency_command(paths, &args, &spec);
    }
    for (name, values) in [
        ("--pp", &args.pp),
        ("--ctx", &args.ctx),
        ("--sustained-ctx", &args.sustained_ctx),
    ] {
        if values.is_empty() || values.contains(&0) {
            bail!("{name} values must be positive");
        }
    }
    if args.tg == 0 || args.sustained_tg == Some(0) {
        bail!("decode lengths must be positive");
    }
    if let Some(mode) = args.kv_mode.as_deref() {
        // Validate against the canonical `memory.kv_cache` schema instead of a
        // local subset. The old hardcoded list accepted only q8/fwht{2,3,4} and
        // so rejected `f32`/`f16` — the only KV formats DeepSeek V4 implements,
        // and precisely what the loader tells you to pass when it falls back
        // ("Pass --kv f32 for the golden configuration"). That made the advised
        // configuration unreachable through `bench`.
        let field = hipfire_config::field("memory.kv_cache")
            .ok_or_else(|| anyhow!("missing memory.kv_cache configuration field"))?;
        field
            .validate(&hipfire_config::ConfigValue::String(mode.to_owned()))
            .map_err(|err| anyhow!("--kv-mode {mode}: {err}"))?;
    }

    if args.exp {
        return bench_experimental(paths, &args);
    }
    let (mut engine, loaded, pre_diag, post_diag, devices_spec, max_seq, load_speculation) =
        open_bench_engine(paths, &args, None, &BenchLoadOpts::default())?;
    let max_seq_json = max_seq.json(&loaded);
    let prompt = resolve_bench_prompt(&args)?;
    let prompt_md5 = bench_prompt_md5(&prompt);
    let prompt_chars = prompt.chars().count() as u64;
    eprintln!("hipfire bench");
    eprintln!("  model:  {}", args.model);
    eprintln!(
        "  arch:   {}",
        loaded
            .get("arch")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
    );
    eprintln!(
        "  gpu:    {}",
        post_diag
            .get("arch")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
    );
    eprintln!("  runs:   {}", args.runs);
    eprintln!("  max_tokens: {}", args.standard_max_tokens());
    eprintln!("  prompt_md5: {prompt_md5}");
    eprintln!("  prompt_chars: {prompt_chars}");
    print_bench_max_seq(&max_seq, &loaded);
    let power = bench_power_probe(devices_spec.as_deref(), &post_diag);
    let mut spill = power.device().and_then(spill::SpillMonitor::attach);
    if args.matrix || args.redline {
        bench_matrix(
            &mut engine,
            &args,
            &loaded,
            &post_diag,
            &power,
            &max_seq_json,
            &load_speculation,
        )
    } else if args.ttft {
        bench_ttft(
            &mut engine,
            &args,
            &prompt,
            &prompt_md5,
            prompt_chars,
            &loaded,
            &post_diag,
            &power,
            &max_seq_json,
        )
    } else {
        // Warmups are discarded requests with the measured prompt, token
        // budget and reasoning mode, so the first measured run meets the same
        // prefill shape, kernels and caches the measured runs use. `--warmups
        // 0` runs none.
        let warmups_run = run_standard_bench_warmups(&args, &prompt, |request| {
            engine.generate(request, |_| Ok(()))?;
            Ok(())
        })?;
        if let Some(message) = spill.as_mut().and_then(spill::SpillMonitor::check) {
            spill::emit(&message);
        }
        let mut spill_runs = Vec::new();
        let mut decode = Vec::new();
        let mut prefill = Vec::new();
        let mut wall = Vec::new();
        let mut ttft = Vec::new();
        let mut prompt_tokens: Option<u64> = None;
        let mut spec_routes: Vec<Option<String>> = Vec::new();
        let mut spec_tau: Vec<Option<f64>> = Vec::new();
        let mut spec_mtp: Vec<Option<serde_json::Value>> = Vec::new();
        let mut power_runs = Vec::new();
        for _ in 0..args.runs {
            let sampler = power.start_run();
            let done = bench_generate_with_reasoning(
                &mut engine,
                &prompt,
                args.standard_max_tokens() as u64,
                args.reasoning_on,
            )?;
            power_runs.push(sampler.finish());
            if let Some(monitor) = spill.as_mut() {
                if let Some(message) = monitor.check() {
                    eprintln!();
                    spill::emit(&message);
                }
                spill_runs.push(monitor.report_json());
            }
            // Every run uses the same prompt, so the daemon's tokenized
            // prompt length is run-invariant; keep the first report. The
            // done event reports the prompt as `prefill_tokens` (rows the
            // engine actually prefilled) plus `cached_tokens` (prefix served
            // from the prompt cache); the prompt is their sum.
            if prompt_tokens.is_none() {
                prompt_tokens = bench_prompt_tokens_from_done(&done);
            }
            let route = bench_spec_route(&done);
            spec_tau.push(
                route
                    .as_ref()
                    .and_then(|_| done.get("tau").and_then(serde_json::Value::as_f64)),
            );
            spec_routes.push(route);
            spec_mtp.push(bench_spec_mtp(&done));
            if let Some(value) = done.get("decode_tok_s").and_then(serde_json::Value::as_f64) {
                decode.push(value);
            }
            if let Some(value) = done
                .get("prefill_tok_s")
                .and_then(serde_json::Value::as_f64)
            {
                prefill.push(value);
            }
            if let Some(value) = done.get("tok_s").and_then(serde_json::Value::as_f64) {
                wall.push(value);
            }
            if let Some(value) = done.get("ttft_ms").and_then(serde_json::Value::as_f64) {
                ttft.push(value);
            }
            eprint!(".");
            std::io::stderr().flush()?;
        }
        eprintln!();
        // Below 256 prompt tokens prefill_tok_s is launch overhead, not
        // prefill throughput — say so in the report instead of leaving a
        // bare number that invites a wrong comparison.
        let warnings: Vec<String> = prompt_tokens
            .and_then(bench_prompt_warning)
            .into_iter()
            .chain(bench_power::throttle_warning(&power_runs))
            .chain(bench_spec_warning(
                args.speculation.as_deref(),
                &spec_routes,
            ))
            .collect();
        if let Some(tokens) = prompt_tokens {
            eprintln!("  prompt_tokens: {tokens}");
        }
        for warning in &warnings {
            eprintln!("  warning: {warning}");
        }
        let report = with_loaded_kv_backend_fields(
            serde_json::json!({
                "protocol": "native-generate-v1",
                "model": args.model,
                "loaded": loaded,
                "gpu": post_diag,
                "vram_free_before_mb": pre_diag.get("vram_free_mb"),
                "max_seq": max_seq_json,
                "max_tokens": args.standard_max_tokens(),
                "runs": args.runs,
                "batch": 1,
                "prompt_tokens": prompt_tokens,
                "prompt_md5": prompt_md5,
                "prompt_chars": prompt_chars,
                "warnings": warnings,
                "power_device": power.device_json(),
                "power": bench_power::runs_json(&power_runs),
                "spill": spill_runs,
                "decode_tok_s": sample_stats(&decode),
                "prefill_tok_s": sample_stats(&prefill),
                "wall_tok_s": sample_stats(&wall),
                "ttft_ms": sample_stats(&ttft),
                "spec_requested": args.speculation,
                "load_speculation": load_speculation,
                "spec_route": spec_routes,
                "spec_tau": spec_tau,
                "spec_mtp": spec_mtp,
                "samples": { "decode": decode, "prefill": prefill, "wall": wall, "ttft_ms": ttft },
            }),
            &loaded,
        );
        let report = with_warmups_run(report, warmups_run);
        if args.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            print_sample_row("decode", sample_stats(&decode));
            print_sample_row("prefill", sample_stats(&prefill));
            print_sample_row("wall", sample_stats(&wall));
            print_sample_row("ttft ms", sample_stats(&ttft));
        }
        Ok(())
    }
}

/// Closed-loop concurrency sweep (see `bench_concurrency`). Only reached
/// when `--concurrency` is given; the single-stream path above is untouched.
fn bench_concurrency_command(paths: &Paths, args: &BenchArgs, spec: &str) -> Result<()> {
    use crate::bench_concurrency::{
        betterbench_workload, parse_concurrency, render_table, steady_workload, sweep_backend,
        BackendSel, BenchShape, Driver, Point, WorkloadSel, STEADY_DEFAULT_MAX_TOKENS,
    };

    let points = parse_concurrency(spec)?;
    let max_k = *points.iter().max().expect("non-empty");
    let backend_sel = match args.backend.as_str() {
        "slots" => BackendSel::Slots,
        "noslots" => BackendSel::Sequential,
        "batch" => BackendSel::Batch,
        _ => BackendSel::Both,
    };
    let arms: Vec<WorkloadSel> = match args.workload.as_str() {
        "stateless" => vec![WorkloadSel::Stateless],
        "multiturn" => vec![WorkloadSel::Multiturn],
        _ => vec![WorkloadSel::Stateless, WorkloadSel::Multiturn],
    };
    let shape = match args.bench_shape.as_str() {
        "betterbench" => BenchShape::BetterBench,
        _ => BenchShape::Steady,
    };
    let mut workload = match shape {
        BenchShape::Steady => {
            if args.bb_corpus.is_some() {
                bail!("--bb-corpus needs --bench-shape betterbench");
            }
            let max_tokens = args
                .max_tokens
                .map_or(STEADY_DEFAULT_MAX_TOKENS, |n| n as u64);
            steady_workload(args.num_prompts, max_tokens)?
        }
        BenchShape::BetterBench => {
            if args.max_tokens.is_some() {
                bail!(
                    "--max-tokens does not apply to --bench-shape betterbench: \
                     every corpus prompt carries its own max_tokens"
                );
            }
            let dir = args.bb_corpus.as_deref().ok_or_else(|| {
                anyhow!("--bench-shape betterbench needs --bb-corpus <BetterBench corpus dir>")
            })?;
            betterbench_workload(dir, args.num_prompts)?
        }
    };
    if let Some(t) = args.temperature {
        workload.sampling.temperature = t;
    }
    // The load-time knobs `hipfire serve` reads from the same configuration:
    // its default system prompt and the VMM batch route parameters.
    let registry = load_registry(&paths.registry).registry;
    let (tag, entry) = registry_entry_for_path(paths, &registry, &args.model)
        .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry.clone())))
        .unwrap_or((None, None));
    let resolved = resolved_for_model(paths, &args.model, tag.as_deref(), entry.as_ref())?;
    if let Some(system) = request_string(&resolved, "prompt.system", None)? {
        workload.inject_system(&system);
    }

    eprintln!("hipfire bench — concurrency sweep (closed loop)");
    eprintln!("  model:       {}", args.model);
    eprintln!("  concurrency: {points:?}");
    eprintln!("  runs/point:  {}", args.runs);
    eprintln!(
        "  shape:       {} ({} requests/sample, source md5 {}, {} warmup requests of {} tokens/sample)",
        shape.label(),
        workload.measure.len(),
        workload.source_md5,
        workload.warmup.len(),
        crate::bench_concurrency::WARMUP_TOKENS,
    );
    if shape == BenchShape::Steady {
        eprintln!("  max_tokens:  {}", workload.measure[0].max_tokens);
    }
    eprintln!("  sampling:    {:?}", workload.sampling);

    let mut out: Vec<Point> = Vec::new();

    // The backends run STRICTLY SEQUENTIALLY, and the scopes below enforce
    // it. Each holds a full copy of the model's weights -- 18.7 GB for a
    // 35B-A3B mq4r -- so overlapping them doubles resident footprint. On a
    // box with no swap that is not "slower", it is an OOM kill. Dropping a
    // driver drops its engine, which stops that daemon before the next loads.
    if matches!(backend_sel, BackendSel::Slots | BackendSel::Both) {
        preflight_headroom_for_model(paths, &args.model)?;
        let mut slot_args = args.clone();
        slot_args.concurrency = None;
        // Small per-slot arenas, not the serve default of 8192: the steady
        // shape's turns stay under 1K tokens and BetterBench's under 4K, and a
        // larger arena multiplies per-slot KV for nothing.
        let slot_ctx = match shape {
            BenchShape::Steady => 2048,
            BenchShape::BetterBench => 4096,
        };
        let (engine, loaded, _, _, _, max_seq, _) =
            open_bench_engine_slots(paths, &slot_args, max_k, slot_ctx)?;
        print_bench_max_seq(&max_seq, &loaded);
        let slot_capable = loaded
            .get("experimental_multi_slot")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        match Driver::slots(engine, max_k, slot_capable, args.reasoning_on) {
            Ok(mut d) => {
                eprintln!("  slots backend up ({max_k} slots)");
                let r = sweep_backend(&mut d, &workload, &arms, &points, args.runs, &mut out);
                // Free the weights before the next backend loads its own copy,
                // even on the error path.
                drop(d);
                r?;
            }
            // A backend that cannot run this model is a RESULT, not a crash.
            Err(e) => eprintln!("  slots backend unavailable: {e}"),
        }
    }

    // No-batching baseline: the daemon serves one request at a time and the
    // other lanes queue client-side, as `hipfire serve` queues them with
    // batching off.
    if matches!(backend_sel, BackendSel::Sequential | BackendSel::Both) {
        preflight_headroom_for_model(paths, &args.model)?;
        let mut seq_args = args.clone();
        seq_args.concurrency = None;
        let (engine, loaded, _, _, _, max_seq, _) =
            open_bench_engine(paths, &seq_args, None, &BenchLoadOpts::default())?;
        print_bench_max_seq(&max_seq, &loaded);
        let mut d = Driver::sequential(engine, max_k, args.reasoning_on);
        eprintln!("  noslots backend up (one request at a time)");
        let r = sweep_backend(&mut d, &workload, &arms, &points, args.runs, &mut out);
        drop(d);
        r?;
    }

    if matches!(backend_sel, BackendSel::Batch | BackendSel::Both) {
        preflight_headroom_for_model(paths, &args.model)?;
        let mut batch_args = args.clone();
        batch_args.concurrency = None;
        // The VMM route needs at least two lanes; a lone request at k=1
        // takes the singleton route either way, as it does under serve.
        let lanes = max_k.max(2);
        let vmm = crate::serve::VmmBatchFlags::from_config(&resolved)?;
        let vmm_params = vmm.load_params(config_u64(&resolved, "serve.max_batch_tokens")?);
        let (engine, loaded, _, _, _, max_seq, _) =
            open_bench_engine_batched(paths, &batch_args, lanes, vmm_params)?;
        print_bench_max_seq(&max_seq, &loaded);
        match Driver::batch(engine, max_k, &loaded, args.reasoning_on) {
            Ok(mut d) => {
                let field = |key: &str| loaded.get(key).cloned().unwrap_or_default();
                eprintln!(
                    "  batch backend up (route={}, continuous_batch_size={lanes}, exact={}, spec_mode={}, row_budget={})",
                    field("continuous_batch_route"),
                    field("continuous_batch_exact"),
                    field("continuous_batch_spec_mode"),
                    field("continuous_batch_row_budget"),
                );
                sweep_backend(&mut d, &workload, &arms, &points, args.runs, &mut out)?;
            }
            Err(e) => eprintln!("  batch backend unavailable: {e}"),
        }
    }

    println!("{}", render_table(&out, shape));
    Ok(())
}

/// Refuse to load a model that will not fit in available host memory.
///
/// The GPU allocates from system RAM on this class of box and there is no
/// swap, so an overcommit is an OOM kill of the whole machine rather than a
/// slow run. Checked between the two backends because that is precisely where
/// a leaked first model would show up: if the slots engine did not actually
/// release its weights, available physical memory is still depressed here and
/// this stops the sweep instead of taking the box down.
///
/// `memory.oom_guard` (default `auto`) opts out or forces the check on: this
/// process never initializes a GPU, so `auto` falls back to host swap state —
/// with swap an overcommit degrades rather than kills and the check stands
/// down; without swap it stays up. A discrete-GPU box that wants the check
/// anyway pins `memory.oom_guard=true`.
fn preflight_headroom_for_model(paths: &Paths, model: &str) -> Result<()> {
    if !hipfire_config::oom_guard_effective(None) {
        eprintln!("memory headroom guard inactive (memory.oom_guard); continuing sweep");
        return Ok(());
    }
    let registry = load_registry(&paths.registry).registry;
    let Some(path) = find_model_path(paths, &registry, model) else {
        return Ok(());
    };
    let Ok(meta) = std::fs::metadata(&path) else {
        return Ok(());
    };
    let need = meta.len();
    // No host memory source means no check.
    let Some(avail) = sweep_available_bytes() else {
        return Ok(());
    };
    let want = sweep_headroom_want_bytes(need);
    if avail < want {
        bail!(
            "refusing to load {}: needs ~{:.1} GB with headroom, only {:.1} GB available. \
             A previous backend may not have released its weights; stop it with `hipfire stop` or free system memory, then retry.",
            path.display(),
            want as f64 / 1e9,
            avail as f64 / 1e9
        );
    }
    Ok(())
}

/// Host memory available for sweep sizing. Linux nets the CMA region out of
/// `MemAvailable` (`rdna_compute::uma_memory`): CMA pageblocks (including KHO
/// scratch) are counted as available but are unusable for the unmovable
/// allocations a model load makes.
/// Other platforms keep the raw host snapshot.
#[cfg(target_os = "linux")]
fn sweep_available_bytes() -> Option<u64> {
    rdna_compute::uma_memory::mem_available_bytes()
}

#[cfg(not(target_os = "linux"))]
fn sweep_available_bytes() -> Option<u64> {
    hipfire_config::host_memory().map(|host| host.available_physical_bytes)
}

/// Raw weight bytes plus 20% headroom for KV arenas and scratch.
fn sweep_headroom_want_bytes(need: u64) -> u64 {
    need + need / 5
}

/// `open_bench_engine`, loading the VMM continuous-batching route the way
/// `hipfire serve` does with `serve.vmm_batch` on: `continuous_batch_size`
/// lanes plus the `serve_vmm_batch` parameter object. Without the latter the
/// daemon would batch on the legacy fixed-slot route serve never runs. The
/// lane count is fixed per load, which is why the sweep holds it at max.
fn open_bench_engine_batched(
    paths: &Paths,
    args: &BenchArgs,
    batch_size: usize,
    vmm_batch: serde_json::Value,
) -> Result<BenchEngine> {
    open_bench_engine(
        paths,
        args,
        None,
        &BenchLoadOpts {
            continuous_batch: Some((batch_size, vmm_batch)),
            ..Default::default()
        },
    )
}

/// Spawn a daemon and load the model with `experimental_multi_slot=true` so
/// the slot backend owns the GPU. The slot count and per-slot context cap are
/// fixed per load, which is why the sweep holds them at max.
fn open_bench_engine_slots(
    paths: &Paths,
    args: &BenchArgs,
    slots: usize,
    ctx: usize,
) -> Result<BenchEngine> {
    open_bench_engine(
        paths,
        args,
        None,
        &BenchLoadOpts {
            multi_slot: Some((slots, ctx)),
            ..Default::default()
        },
    )
}

/// `open_bench_engine`'s result: the engine, its `loaded` reply, `diag`
/// before and after the load, the `hardware.devices` selector the daemon
/// was configured with (the bench's power sampler resolves the same card),
/// how the load's `max_seq` was chosen, and the speculation the load asked
/// for ([`bench_load_speculation`]).
type BenchEngine = (
    Engine,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
    Option<String>,
    BenchMaxSeq,
    serde_json::Value,
);

/// The speculation selectors of a bench load's params (`mtp_mode`,
/// `dflash_mode`, `dspark_mode`, `ngram_draft`; null when the params leave
/// one unset), so a report states which speculative route the loaded model
/// could take.
fn bench_load_speculation(params: &serde_json::Value) -> serde_json::Value {
    let field = |key: &str| params.get(key).cloned().unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "mtp_mode": field("mtp_mode"),
        "dflash_mode": field("dflash_mode"),
        "dspark_mode": field("dspark_mode"),
        "ngram_draft": field("ngram_draft"),
    })
}

/// Add the daemon's `loaded` ack `mtp_mode_effective` to a bench load's
/// speculation selectors as `mtp_effective`: the MTP mode the loaded model
/// serves after its producer's capability gate. `mtp_mode` stays what the
/// params asked for (`auto` on a producer without a measured AR floor is
/// requested `auto` but effective `off`); `null` when the ack carries no
/// verdict (an older daemon, or a route that does not report one).
fn bench_load_speculation_effective(
    mut requested: serde_json::Value,
    loaded: &serde_json::Value,
) -> serde_json::Value {
    requested["mtp_effective"] = loaded
        .get("mtp_mode_effective")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    requested
}

/// Positions past the measured ones that the daemon's `bench_prefill` and
/// `bench_decode` guards require (`tokens + 32 <= physical_cap`).
const BENCH_SEQ_HEADROOM: u64 = 32;

/// Where a bench load's `max_seq` came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchMaxSeqSource {
    /// `--max-seq N`.
    Flag,
    /// Sized to the `--matrix`/`--redline` shape.
    RunShape,
    /// An authored `memory.max_seq` forwarded by `load_params`.
    Config,
    /// Not sent; the daemon's admission chooses the context.
    Automatic,
}

impl BenchMaxSeqSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Flag => "flag",
            Self::RunShape => "run_shape",
            Self::Config => "config",
            Self::Automatic => "automatic",
        }
    }
}

/// How a bench load's `max_seq` was chosen; reported beside the effective
/// value the daemon loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BenchMaxSeq {
    /// The value sent with the load; `None` leaves it to admission.
    requested: Option<u64>,
    source: BenchMaxSeqSource,
    /// Positions the matrix touches plus headroom; `None` off the matrix.
    run_need: Option<u64>,
    /// The model's declared context, when it was read.
    model_limit: Option<u64>,
}

impl BenchMaxSeq {
    /// The report's `max_seq` object. `effective` is what the daemon loaded.
    fn json(&self, loaded: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "effective": loaded.get("max_seq"),
            "requested": self.requested,
            "source": self.source.as_str(),
            "run_need": self.run_need,
            "model_limit": self.model_limit,
        })
    }
}

/// Context the `--matrix`/`--redline` run touches: the largest prefill, the
/// largest decode context plus its decode length (sustained rows only when
/// `--sustained-tg` runs them), plus the daemon's probe headroom.
fn bench_matrix_seq_need(
    pp: &[usize],
    ctx: &[usize],
    tg: usize,
    sustained: Option<(usize, &[usize])>,
) -> u64 {
    let longest = |values: &[usize], decode: usize| {
        values
            .iter()
            .map(|&value| value as u64 + decode as u64)
            .max()
            .unwrap_or(0)
    };
    let sustained = sustained.map_or(0, |(tg, ctx)| longest(ctx, tg));
    longest(pp, 0)
        .max(longest(ctx, tg))
        .max(sustained)
        .saturating_add(BENCH_SEQ_HEADROOM)
}

/// The model's declared context window: `max_position_embeddings` under
/// `config.text_config`, `config`, or the top level of the source metadata,
/// extended by YaRN RoPE scaling (checkpoint or `attention.yarn_factor`), the
/// same lookup admission uses for a model's trained context.
fn declared_context_from_metadata(metadata_json: &str) -> Option<u64> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    let config = meta.get("config").unwrap_or(&meta);
    let text = config.get("text_config").unwrap_or(config);
    let native = text
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .filter(|&context| context > 0)?;
    let scaling = hipfire_runtime::rope::rope_scaling(text).unwrap_or_default();
    Some(scaling.context_limit(native as usize) as u64)
}

/// [`declared_context_from_metadata`] for a model file or directory. `None`
/// when the source cannot be opened or declares no context; the daemon's
/// admission is then the only bound.
fn bench_declared_context(path: &Path) -> Option<u64> {
    let source = hipfire_runtime::model_source::open_model(path).ok()?;
    declared_context_from_metadata(source.metadata_json())
}

/// Choose the bench load's `max_seq` before the daemon starts.
///
/// `--max-seq` wins but must hold the run. Without it a matrix run is sized
/// to `run_need` (raised to the `memory.max_seq` minimum, never above the
/// model's declared context) instead of the configured global value; any other
/// path stays automatic. A matrix shape past the model's declared context
/// fails here, naming both numbers, instead of at load or mid-run.
fn plan_bench_max_seq(
    flag: Option<u64>,
    run_need: Option<u64>,
    model_limit: Option<u64>,
    model: &str,
) -> Result<BenchMaxSeq> {
    if let (Some(need), Some(limit)) = (run_need, model_limit) {
        if need > limit {
            bail!(
                "bench run needs max_seq {need} (largest --pp, or --ctx + --tg, plus \
                 {BENCH_SEQ_HEADROOM} headroom) but {model} declares a maximum context of \
                 {limit}; shrink --pp/--ctx/--tg"
            );
        }
    }
    let (requested, source) = match (flag, run_need) {
        (Some(value), Some(need)) if value < need => bail!(
            "--max-seq {value} is below this run's need of {need} (largest --pp, or --ctx + \
             --tg, plus {BENCH_SEQ_HEADROOM} headroom); raise --max-seq or shrink the matrix"
        ),
        (Some(value), _) => (Some(value), BenchMaxSeqSource::Flag),
        (None, Some(need)) => {
            let ValueRule::Integer { min, .. } =
                field("memory.max_seq").expect("schema field").rule
            else {
                unreachable!("memory.max_seq is an integer field");
            };
            let sized = need.max(min as u64);
            let sized = model_limit.map_or(sized, |limit| sized.min(limit));
            (Some(sized), BenchMaxSeqSource::RunShape)
        }
        (None, None) => (None, BenchMaxSeqSource::Automatic),
    };
    Ok(BenchMaxSeq {
        requested,
        source,
        run_need,
        model_limit,
    })
}

/// Print the effective `max_seq` and how it was chosen.
fn print_bench_max_seq(max_seq: &BenchMaxSeq, loaded: &serde_json::Value) {
    let effective = loaded
        .get("max_seq")
        .and_then(serde_json::Value::as_u64)
        .map_or_else(|| "unknown".to_owned(), |value| value.to_string());
    let mut detail = max_seq.source.as_str().to_owned();
    if let Some(need) = max_seq.run_need {
        detail.push_str(&format!("; run needs {need}"));
    }
    if let Some(limit) = max_seq.model_limit {
        detail.push_str(&format!("; model declares {limit}"));
    }
    eprintln!("  max_seq: {effective} ({detail})");
}

/// Optional load-time knobs for `open_bench_engine`, threaded explicitly
/// rather than via `std::env::set_var` — `hipfire_config::developer_var`
/// reads the OnceLock process-config snapshot, not the ambient environment,
/// so a set_var inside the process is invisible once the snapshot is taken.
#[derive(Default)]
struct BenchLoadOpts {
    multi_slot: Option<(usize, usize)>,
    /// `continuous_batch_size` and the `serve_vmm_batch` load parameter.
    continuous_batch: Option<(usize, serde_json::Value)>,
}

/// `bench` measures text generation and has no image/audio/video input, so
/// its load is text-only whatever `vision.mode` resolves to. Under the schema
/// default `auto` a pulled `qwen3.8-27b-vision.hfq` would otherwise be wired
/// into every bench load: ~1.05 GB of resident tower VRAM that shifts KV and
/// admission headroom (and, on cards where `max_seq` is derived from free
/// memory, the context the run is sized to) relative to earlier releases.
/// Projects `off` and drops the resolved sidecar so no `skipping tower
/// sidecar` line fires either.
fn pin_bench_text_only(params: &mut serde_json::Value) {
    params["vision_mode"] = serde_json::json!("off");
    if let Some(obj) = params.as_object_mut() {
        obj.remove("vision");
    }
}

fn open_bench_engine(
    paths: &Paths,
    args: &BenchArgs,
    rdna2_variant: Option<u8>,
    load_opts: &BenchLoadOpts,
) -> Result<BenchEngine> {
    let registry = load_registry(&paths.registry).registry;
    let (tag, entry) = registry_entry_for_path(paths, &registry, &args.model)
        .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry.clone())))
        .unwrap_or((None, None));
    let mut path = resolve_model_path(paths, &registry, &args.model)?;
    if path.is_none() && entry.is_some() {
        pull_command(paths, PullArgs::defaults(args.model.clone()))?;
        // Do not hash a payload twice after a verified pull.
        path = find_model_path(paths, &registry, &args.model);
    }
    let path = path.ok_or_else(|| model_not_found_error(paths, &registry, &args.model))?;
    let resolved = resolved_for_model(paths, &args.model, tag.as_deref(), entry.as_ref())?;
    ple::ensure_ple_sidecar(
        entry.as_ref(),
        tag.as_deref(),
        &path,
        ple::request_from_config(&resolved),
        &ple::load_target_arch(&resolved),
        true,
    )?;
    let run_need = (args.matrix || args.redline).then(|| {
        bench_matrix_seq_need(
            &args.pp,
            &args.ctx,
            args.tg,
            args.sustained_tg
                .map(|tg| (tg, args.sustained_ctx.as_slice())),
        )
    });
    let model_limit = run_need.and_then(|_| bench_declared_context(&path));
    let mut max_seq = plan_bench_max_seq(args.max_seq, run_need, model_limit, &args.model)?;
    let daemon = find_daemon(paths).ok_or_else(|| daemon_not_found_error(paths))?;
    let environment = BTreeMap::new();
    let mut process_config = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
    if args.redline {
        process_config.values.set_cli("replay.backend", "redline")?;
        process_config.values.set_cli("replay.transport", "pm4")?;
        process_config
            .values
            .set_cli("experimental.graph.ar", "true")?;
        process_config
            .values
            .set_cli("experimental.graph.forward", "true")?;
    }
    if let Some(variant) = rdna2_variant {
        process_config
            .values
            .set_cli("diagnostic.kernel.rdna2_variant", &variant.to_string())?;
    }
    // `--emu` overrides HIPFIRE_EMU in this child's process snapshot (including
    // `off`). Every bench arm (standard, matrix, ttft, concurrency, exp) opens its
    // daemon here, so every spawned arm gets the same profile.
    let effective_emu = emu::apply_cli_override(&mut process_config, args.emu.as_deref())?;
    if let Some(line) = effective_emu.log_line() {
        eprintln!("  {line}");
    }
    let mut engine = Engine::spawn_configured(daemon, &environment, &process_config, Some(&path))?;
    engine.ping()?;
    let pre_diag = engine.request(&serde_json::json!({ "type": "diag" }))?;
    let max_tokens = config_u64(&resolved, "generation.max_tokens")?;
    let mut params = load_params(
        &resolved,
        entry.as_ref(),
        &paths.models,
        &path,
        max_tokens,
        args.kv_mode.as_deref(),
        args.kv_backend.as_deref(),
        tag.as_deref(),
        false,
        // No --head on this path yet; the model's own head is used.
        None,
    )?;
    pin_bench_text_only(&mut params);
    apply_kv_axis_overrides(&mut params, args.kv_k.as_deref(), args.kv_v.as_deref())?;
    if let Some(selector) = args.speculation.as_deref() {
        apply_speculation_selector(&mut params, selector)?;
    }
    // Registry sidecar for a final auto/on selector the config-time
    // load_params could not see (config-off + `bench --spec dflash`).
    resolve_dflash_sidecar(
        &mut params,
        entry.as_ref(),
        &paths.models,
        &path,
        tag.as_deref(),
    )?;
    resolve_mtp_sidecar(
        &mut params,
        entry.as_ref(),
        &paths.models,
        &path,
        &args.model,
        tag.as_deref(),
    );
    match max_seq.requested {
        // The flag and the matrix sizing replace any configured value.
        Some(value) => params["max_seq"] = serde_json::json!(value),
        None => {
            max_seq.requested = params.get("max_seq").and_then(serde_json::Value::as_u64);
            if max_seq.requested.is_some() {
                max_seq.source = BenchMaxSeqSource::Config;
            }
        }
    }
    if let Some((n, vmm_batch)) = &load_opts.continuous_batch {
        params["continuous_batch_size"] = serde_json::json!(*n as u64);
        params["serve_vmm_batch"] = vmm_batch.clone();
    }
    if let Some(tp) = args.tp.filter(|&tp| tp > 1) {
        params["tp"] = serde_json::json!(tp);
    }
    if let Some((slots, ctx)) = load_opts.multi_slot {
        params["experimental_multi_slot"] = serde_json::json!(true);
        params["experimental_multi_slot_slots"] = serde_json::json!(slots as u64);
        params["experimental_multi_slot_ctx"] = serde_json::json!(ctx as u64);
    }
    let requested_speculation = bench_load_speculation(&params);
    let loaded = engine.load(&path, params)?;
    let load_speculation = bench_load_speculation_effective(requested_speculation, &loaded);
    let post_diag = engine.request(&serde_json::json!({ "type": "diag" }))?;
    let devices_spec = process_config.legacy_value("HIPFIRE_DEVICES");
    Ok((
        engine,
        loaded,
        pre_diag,
        post_diag,
        devices_spec,
        max_seq,
        load_speculation,
    ))
}

/// The standard benchmark generate: greedy, fixed budget, and **answer mode**.
///
/// Answer mode is the default rather than an opt-in so every model runs the
/// same turn shape. A reasoning model (any Qwen3.6 SKU, for one) opens
/// `<think>` within its first tokens and would spend the benchmark's whole
/// budget — 128 tokens in a measured run — inside it,
/// ending every sample mid-thought at `finish_reason=length`.
///
/// Benchmarks measure tokens per second and never read the text, so asking for
/// answer mode costs nothing and keeps the measured path independent of how
/// long a model thinks. `--reasoning-on` restores the thinking turn for anyone
/// who wants to measure that path.
fn bench_generate_request(prompt: &str, max_tokens: u64) -> serde_json::Value {
    bench_generate_request_reasoning(prompt, max_tokens, false)
}

fn bench_generate_request_reasoning(
    prompt: &str,
    max_tokens: u64,
    reasoning_on: bool,
) -> serde_json::Value {
    let mut request = serde_json::json!({
        "type": "generate",
        "id": request_id(),
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        // Neutral on every path: any non-neutral penalty routes the daemon to
        // the plain AR decoder, which made `--spec` silently measure AR, and a
        // shared value keeps `--spec off` and speculative runs the same workload.
        "repeat_penalty": 1.0,
        "max_tokens": max_tokens,
        "attempt_id": 1,
    });
    if !reasoning_on {
        request["max_think_tokens"] = serde_json::json!(1);
        request["assistant_prefix"] = serde_json::json!("closed_think");
        request["reasoning_effort"] = serde_json::json!("none");
    }
    request
}

/// The discarded warmup request of the standard bench: the measured request
/// shape (prompt, `--max-tokens`, `--reasoning-on`), never a shorter stand-in.
fn standard_bench_warmup_request(args: &BenchArgs, prompt: &str) -> serde_json::Value {
    bench_generate_request_reasoning(prompt, args.standard_max_tokens() as u64, args.reasoning_on)
}

/// Issue exactly `args.warmups` warmup requests through `generate`, stopping at
/// the first error, and return how many ran (`0` for `--warmups 0`). The
/// request is built once; each warmup after the first gets a fresh `id`, as
/// every measured generate does.
fn run_standard_bench_warmups(
    args: &BenchArgs,
    prompt: &str,
    mut generate: impl FnMut(&serde_json::Value) -> Result<()>,
) -> Result<usize> {
    if args.warmups == 0 {
        return Ok(0);
    }
    let mut request = standard_bench_warmup_request(args, prompt);
    for i in 0..args.warmups {
        if i > 0 {
            request["id"] = serde_json::json!(request_id());
        }
        generate(&request)?;
    }
    Ok(args.warmups)
}

/// Record in the standard bench report how many warmup requests actually ran
/// (the count `run_standard_bench_warmups` returned, not the flag).
fn with_warmups_run(mut report: serde_json::Value, warmups_run: usize) -> serde_json::Value {
    report["warmups_run"] = serde_json::json!(warmups_run);
    report
}

/// The speculation route a generate `done` event reports, or `None` when the
/// plain AR decoder ran (no `tau` stats, or zero speculative cycles).
fn bench_spec_route(done: &serde_json::Value) -> Option<String> {
    let cycles = done
        .get("cycles")
        .or_else(|| done.get("rounds"))
        .and_then(serde_json::Value::as_u64);
    if done.get("tau").is_none() || cycles == Some(0) {
        return None;
    }
    if let Some(spec) = done.get("spec").and_then(serde_json::Value::as_str) {
        return Some(spec.to_string());
    }
    let flag = |key: &str| done.get(key).and_then(serde_json::Value::as_bool) == Some(true);
    let route = if flag("mtp_ngram") {
        "mtp+ngram"
    } else if flag("mtp") {
        "mtp"
    } else if flag("dflash") {
        "dflash"
    } else {
        "speculative"
    };
    Some(route.to_string())
}

/// The native-MTP window split of a generate `done` event, `None` unless
/// native MTP speculated: `windows_mtp` (the windows before any AR-floor
/// retirement: native and n-gram takeover windows plus the floor's timed AR
/// tokens), `windows_floor_ar` (those timed AR tokens: the calibration token
/// and the probes of a would-be retirement), `windows_ar_retired`
/// (head-free AR tokens after a sticky retirement), `retired_at` (index of
/// the first retired window; `null` when the request never retired) and
/// `tau_mtp`, accepted drafts per drafting window (`windows_mtp -
/// windows_floor_ar`; `null` from a daemon that reports no `accepted` or
/// `floor_ar_windows`). AR tokens accept nothing, so `tau`, averaged over
/// every window, reads near zero for a retired request whatever its drafts
/// accepted.
fn bench_spec_mtp(done: &serde_json::Value) -> Option<serde_json::Value> {
    bench_spec_route(done)?;
    if done.get("mtp").and_then(serde_json::Value::as_bool) != Some(true) {
        return None;
    }
    let count = |key: &str| done.get(key).and_then(serde_json::Value::as_u64);
    let cycles = count("cycles")?;
    let ar_retired = count("ar_windows").unwrap_or(0);
    let windows = cycles.saturating_sub(ar_retired);
    let floor_ar = count("floor_ar_windows");
    let retired = done.get("mtp_retired").and_then(serde_json::Value::as_bool) == Some(true);
    let accepted = count("accepted");
    let drafting = floor_ar.map(|floor_ar| windows.saturating_sub(floor_ar));
    let tau_mtp = accepted
        .zip(drafting.filter(|&drafting| drafting > 0))
        .map(|(accepted, drafting)| (accepted as f64 / drafting as f64 * 100.0).round() / 100.0);
    Some(serde_json::json!({
        "tau_mtp": tau_mtp,
        "windows_mtp": windows,
        "windows_floor_ar": floor_ar,
        "windows_ar_retired": ar_retired,
        "retired_at": retired.then_some(windows),
        "accepted": accepted,
    }))
}

/// Warning when a speculative selector was requested but every measured run
/// went through AR.
fn bench_spec_warning(selector: Option<&str>, routes: &[Option<String>]) -> Option<String> {
    let selector = selector?;
    if selector == "off" || routes.is_empty() || routes.iter().any(Option::is_some) {
        return None;
    }
    Some(format!(
        "--spec {selector} requested but no speculation ran; these numbers measure AR decode"
    ))
}

fn bench_generate(engine: &mut Engine, prompt: &str, max_tokens: u64) -> Result<serde_json::Value> {
    Ok(engine.generate(&bench_generate_request(prompt, max_tokens), |_| Ok(()))?)
}

fn bench_generate_with_reasoning(
    engine: &mut Engine,
    prompt: &str,
    max_tokens: u64,
    reasoning_on: bool,
) -> Result<serde_json::Value> {
    let request = bench_generate_request_reasoning(prompt, max_tokens, reasoning_on);
    Ok(engine.generate(&request, |_| Ok(()))?)
}

fn bench_probe(
    engine: &mut Engine,
    message: serde_json::Value,
    expected: &str,
) -> Result<serde_json::Value> {
    let response = engine.request(&message)?;
    match response.get("type").and_then(serde_json::Value::as_str) {
        Some(actual) if actual == expected => Ok(response),
        Some("error") => bail!(
            "{}",
            response
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("benchmark probe failed")
        ),
        other => bail!(
            "expected {expected}, received {}",
            other.unwrap_or("missing type")
        ),
    }
}

/// [`bench_probe`] for one measured run, sampling the GPU power state across
/// it into `runs`.
fn bench_probe_sampled(
    engine: &mut Engine,
    power: &bench_power::PowerProbe,
    runs: &mut Vec<bench_power::RunPower>,
    message: serde_json::Value,
    expected: &str,
) -> Result<serde_json::Value> {
    let sampler = power.start_run();
    let response = bench_probe(engine, message, expected)?;
    runs.push(sampler.finish());
    Ok(response)
}

/// The bench's power sampler, aimed at the card the daemon was given:
/// `hardware.devices`, else the inherited `ROCR_VISIBLE_DEVICES` the daemon
/// also sees, else the first ROCr agent of the daemon's reported arch.
fn bench_power_probe(
    devices_spec: Option<&str>,
    diag: &serde_json::Value,
) -> bench_power::PowerProbe {
    let rocr_visible = env::var("ROCR_VISIBLE_DEVICES").ok();
    let probe = bench_power::PowerProbe::new(
        devices_spec,
        rocr_visible.as_deref(),
        diag.get("arch").and_then(serde_json::Value::as_str),
    );
    eprintln!("  power:  {}", probe.describe());
    probe
}

/// Client-side TTFT on the standard generate path: per run, generate a
/// single token with thinking off and time wall clock from request send to
/// the first streamed `token` event. `pp_tok_s = prompt_tokens / ttft_s` is
/// the competitor-comparable prefill number (radiance reports
/// prompt_tokens/TTFT p50 at an HTTP client). Warmups run the same shape
/// and are discarded. `--reasoning-on` is intentionally ignored here: the
/// metric is defined with thinking off.
#[allow(clippy::too_many_arguments)]
fn bench_ttft(
    engine: &mut Engine,
    args: &BenchArgs,
    prompt: &str,
    prompt_md5: &str,
    prompt_chars: u64,
    loaded: &serde_json::Value,
    diag: &serde_json::Value,
    power: &bench_power::PowerProbe,
    max_seq: &serde_json::Value,
) -> Result<()> {
    // The shared bench banner prints args.max_tokens; the ttft path always
    // generates exactly one token with thinking off, so say so explicitly.
    eprintln!("  ttft shape: max_tokens=1, thinking off");
    for _ in 0..args.warmups {
        let _ = bench_generate(engine, prompt, 1)?;
    }
    let mut ttft_ms_samples = Vec::new();
    let mut prompt_tokens: Option<u64> = None;
    let mut power_runs = Vec::new();
    for _ in 0..args.runs {
        let start = Instant::now();
        let mut first: Option<Duration> = None;
        let request = bench_generate_request(prompt, 1);
        let sampler = power.start_run();
        let done = engine.generate(&request, |event| {
            if first.is_none()
                && event.get("type").and_then(serde_json::Value::as_str) == Some("token")
            {
                first = Some(start.elapsed());
            }
            Ok(())
        })?;
        let elapsed = first.ok_or_else(|| {
            anyhow!("no streamed token observed; cannot measure client-side TTFT")
        })?;
        power_runs.push(sampler.finish());
        ttft_ms_samples.push(elapsed.as_secs_f64() * 1000.0);
        if prompt_tokens.is_none() {
            prompt_tokens = bench_prompt_tokens_from_done(&done);
        }
        eprint!(".");
        std::io::stderr().flush()?;
    }
    eprintln!();
    if let Some(tokens) = prompt_tokens {
        eprintln!("  prompt_tokens: {tokens}");
    }
    let warnings: Vec<String> = bench_power::throttle_warning(&power_runs)
        .into_iter()
        .collect();
    for warning in &warnings {
        eprintln!("  warning: {warning}");
    }
    let pp_samples: Vec<f64> = match prompt_tokens {
        Some(tokens) if tokens > 0 => ttft_ms_samples
            .iter()
            .map(|ms| tokens as f64 / (ms / 1000.0))
            .collect(),
        _ => Vec::new(),
    };
    let ttft_stats = sample_stats(&ttft_ms_samples);
    let pp_stats = sample_stats(&pp_samples);
    let report = with_loaded_kv_backend_fields(
        serde_json::json!({
            "protocol": "client-ttft-v1",
            "model": args.model,
            "loaded": loaded,
            "gpu": diag,
            "max_seq": max_seq,
            "max_tokens": 1,
            "runs": args.runs,
            "warmups": args.warmups,
            "batch": 1,
            "prompt_tokens": prompt_tokens,
            "prompt_md5": prompt_md5,
            "prompt_chars": prompt_chars,
            "ttft": {
                "samples": ttft_ms_samples,
                "median": ttft_stats.map(|stats| stats.median),
                "stdev": ttft_stats.map(|stats| stats.stdev),
                "prompt_tokens": prompt_tokens,
                "prompt_md5": prompt_md5,
            },
            "pp_tok_s": pp_stats,
            "pp_tok_s_samples": pp_samples,
            "warnings": warnings,
            "power_device": power.device_json(),
            "power": bench_power::runs_json(&power_runs),
        }),
        loaded,
    );
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_sample_row("ttft ms", ttft_stats);
        print_sample_row("pp tok/s", pp_stats);
    }
    Ok(())
}

/// The loaded parallel topology as the daemon's `bench_prefill` /
/// `bench_decode` answers report it: pipeline stages, tensor ranks, expert
/// ranks and the identifiers of the cards actually holding the model. A
/// single-card load is `{pp:1, tp:1, ep:1, cards:["<id>"]}`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BenchTopology {
    pp: u64,
    tp: u64,
    ep: u64,
    cards: Vec<String>,
}

impl BenchTopology {
    /// Read `response.topology`. `Ok(None)` when the daemon sent none (the
    /// report then says `null`, never a guessed single card); an error when
    /// it is present but `pp`/`tp`/`ep` are not positive integers or `cards`
    /// is not a non-empty array of strings.
    fn from_response(response: &serde_json::Value) -> Result<Option<Self>> {
        let Some(topology) = response.get("topology") else {
            return Ok(None);
        };
        let field = |name: &str| -> Result<u64> {
            topology
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .filter(|value| *value >= 1)
                .ok_or_else(|| {
                    anyhow!("daemon bench topology {topology} has no positive integer `{name}`")
                })
        };
        let cards = topology
            .get("cards")
            .and_then(serde_json::Value::as_array)
            .filter(|cards| !cards.is_empty())
            .and_then(|cards| {
                cards
                    .iter()
                    .map(|card| card.as_str().map(str::to_owned))
                    .collect::<Option<Vec<String>>>()
            })
            .ok_or_else(|| {
                anyhow!("daemon bench topology {topology} has no non-empty string array `cards`")
            })?;
        Ok(Some(Self {
            pp: field("pp")?,
            tp: field("tp")?,
            ep: field("ep")?,
            cards,
        }))
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "pp": self.pp, "tp": self.tp, "ep": self.ep, "cards": self.cards })
    }
}

/// Collects the topology reported by every probe of one matrix run. All rows
/// of a run measure one loaded model, so they must agree.
#[derive(Debug, Default)]
struct BenchTopologyTracker {
    seen: Option<BenchTopology>,
}

impl BenchTopologyTracker {
    fn observe(&mut self, response: &serde_json::Value) -> Result<()> {
        let Some(topology) = BenchTopology::from_response(response)? else {
            return Ok(());
        };
        if let Some(first) = &self.seen {
            if *first != topology {
                bail!(
                    "daemon bench rows disagree on the loaded topology: {} then {}",
                    first.to_json(),
                    topology.to_json()
                );
            }
        }
        self.seen = Some(topology);
        Ok(())
    }

    /// The report's `topology` value, checked against the load's own `pp`
    /// so a pipeline or expert-parallel load is never labeled single-card.
    fn finish(self, loaded: &serde_json::Value) -> Result<serde_json::Value> {
        let Some(topology) = self.seen else {
            return Ok(serde_json::Value::Null);
        };
        if let Some(loaded_pp) = loaded.get("pp").and_then(serde_json::Value::as_u64) {
            if loaded_pp != topology.pp {
                bail!(
                    "daemon bench topology {} disagrees with the loaded pp={loaded_pp}",
                    topology.to_json()
                );
            }
        }
        Ok(topology.to_json())
    }
}

fn bench_matrix(
    engine: &mut Engine,
    args: &BenchArgs,
    loaded: &serde_json::Value,
    diag: &serde_json::Value,
    power: &bench_power::PowerProbe,
    max_seq: &serde_json::Value,
    load_speculation: &serde_json::Value,
) -> Result<()> {
    let mut topology = BenchTopologyTracker::default();
    for size in &args.pp {
        let response = bench_probe(
            engine,
            serde_json::json!({ "type": "bench_prefill", "tokens": size }),
            "prefill_result",
        )?;
        topology.observe(&response)?;
    }
    let warm_context = args.ctx[0];
    for _ in 0..args.warmups {
        let response = bench_probe(
            engine,
            serde_json::json!({ "type": "bench_decode", "context_tokens": warm_context, "iterations": args.tg }),
            "decode_result",
        )?;
        topology.observe(&response)?;
    }
    let mut pp_rows = Vec::new();
    let mut all_power = Vec::new();
    for size in &args.pp {
        let mut samples = Vec::new();
        let mut row_power = Vec::new();
        for _ in 0..args.runs {
            let result = bench_probe_sampled(
                engine,
                power,
                &mut row_power,
                serde_json::json!({ "type": "bench_prefill", "tokens": size }),
                "prefill_result",
            )?;
            topology.observe(&result)?;
            samples.push(
                result
                    .get("tok_s")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
            );
        }
        eprintln!(
            "  pp{size}: {:.2} tok/s median",
            sample_stats(&samples).unwrap().median
        );
        pp_rows.push(serde_json::json!({ "tokens": size, "stats": sample_stats(&samples), "samples": samples, "power": bench_power::runs_json(&row_power) }));
        all_power.extend(row_power);
    }
    let mut decode_rows = Vec::new();
    for context in &args.ctx {
        let response = bench_probe(
            engine,
            serde_json::json!({ "type": "bench_decode", "context_tokens": context, "iterations": args.tg }),
            "decode_result",
        )?;
        topology.observe(&response)?;
        let mut samples = Vec::new();
        let mut row_power = Vec::new();
        for _ in 0..args.runs {
            let result = bench_probe_sampled(
                engine,
                power,
                &mut row_power,
                serde_json::json!({ "type": "bench_decode", "context_tokens": context, "iterations": args.tg }),
                "decode_result",
            )?;
            topology.observe(&result)?;
            samples.push(
                result
                    .get("tok_s")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
            );
        }
        eprintln!(
            "  tg{}@{}: {:.2} tok/s median",
            args.tg,
            context,
            sample_stats(&samples).unwrap().median
        );
        decode_rows.push(serde_json::json!({ "context": context, "tokens": args.tg, "stats": sample_stats(&samples), "samples": samples, "power": bench_power::runs_json(&row_power) }));
        all_power.extend(row_power);
    }
    let mut sustained_rows = Vec::new();
    if let Some(tg) = args.sustained_tg {
        for context in &args.sustained_ctx {
            let response = bench_probe(
                engine,
                serde_json::json!({ "type": "bench_decode", "context_tokens": context, "iterations": tg }),
                "decode_result",
            )?;
            topology.observe(&response)?;
            let mut samples = Vec::new();
            let mut row_power = Vec::new();
            for _ in 0..args.runs {
                let result = bench_probe_sampled(
                    engine,
                    power,
                    &mut row_power,
                    serde_json::json!({ "type": "bench_decode", "context_tokens": context, "iterations": tg }),
                    "decode_result",
                )?;
                topology.observe(&result)?;
                samples.push(
                    result
                        .get("tok_s")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0),
                );
            }
            eprintln!(
                "  tg{tg}@{context}: {:.2} tok/s median",
                sample_stats(&samples).unwrap().median
            );
            sustained_rows.push(serde_json::json!({ "context": context, "tokens": tg, "stats": sample_stats(&samples), "samples": samples, "power": bench_power::runs_json(&row_power) }));
            all_power.extend(row_power);
        }
    }
    let warnings: Vec<String> = bench_power::throttle_warning(&all_power)
        .into_iter()
        .collect();
    for warning in &warnings {
        eprintln!("  warning: {warning}");
    }
    let topology = topology.finish(loaded)?;
    let report = with_loaded_kv_backend_fields(
        serde_json::json!({
            "protocol": "synthetic-pp-tg-matrix-v1",
            "model": args.model,
            "loaded": loaded,
            "gpu": diag,
            "max_seq": max_seq,
            "redline_pm4": args.redline,
            "kv_mode": args.kv_mode,
            "spec_requested": args.speculation,
            "load_speculation": load_speculation,
            // `bench_prefill` never drafts: the prompt prefill of a
            // speculative load (MTP's head fill) is not what these rows time.
            "prefill_route": "bench_prefill (synthetic ids, no speculative prompt fill)",
            "runs": args.runs,
            "warnings": warnings,
            "power_device": power.device_json(),
            "prefill": pp_rows,
            "decode": decode_rows,
            "sustained": sustained_rows,
            "topology": topology,
        }),
        loaded,
    );
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}

fn bench_experimental(paths: &Paths, args: &BenchArgs) -> Result<()> {
    let mut rows = Vec::new();
    for variant in 1..=5 {
        let (mut engine, _, _, diag, _, _, _) =
            open_bench_engine(paths, args, Some(variant), &BenchLoadOpts::default())?;
        let arch = diag
            .get("arch")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        if !matches!(arch, "gfx1030" | "gfx1031") {
            bail!("--exp requires RDNA2 (gfx1030/gfx1031), detected {arch}");
        }
        let _ = bench_generate(&mut engine, "Hello", 16)?;
        let prompt = resolve_bench_prompt(args)?;
        let mut samples = Vec::new();
        for _ in 0..args.runs {
            let done = bench_generate(&mut engine, &prompt, 128)?;
            if let Some(value) = done.get("decode_tok_s").and_then(serde_json::Value::as_f64) {
                samples.push(value);
            }
        }
        let stats = sample_stats(&samples)
            .ok_or_else(|| anyhow!("variant {variant} produced no measurements"))?;
        println!(
            "v{variant}: median {:.2}, mean {:.2}, range {:.2}-{:.2} tok/s",
            stats.median, stats.mean, stats.min, stats.max
        );
        rows.push((variant, stats));
    }
    if let Some((variant, stats)) = rows.iter().max_by(|a, b| a.1.median.total_cmp(&b.1.median)) {
        println!("best: v{variant} at {:.2} tok/s median", stats.median);
    }
    Ok(())
}

fn print_sample_row(label: &str, stats: Option<SampleStats>) {
    if let Some(stats) = stats {
        println!(
            "  {label:<10} median {:>9.2}  mean {:>9.2}  range {:>9.2}-{:>9.2}  sd {:>7.2}",
            stats.median, stats.mean, stats.min, stats.max, stats.stdev
        );
    }
}

fn profile_command(paths: &Paths, args: ProfileArgs) -> Result<()> {
    let mut engine = if let Some(model) = args.model.as_deref() {
        eprintln!("loading {model} once so its kernels are present in the inventory...");
        let bench = BenchArgs {
            model: model.to_owned(),
            runs: 1,
            tp: None,
            json: false,
            exp: false,
            matrix: false,
            ttft: false,
            pp: vec![128],
            ctx: vec![128],
            tg: 1,
            max_tokens: None,
            sustained_tg: None,
            sustained_ctx: vec![128],
            warmups: 1,
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            kv_backend: None,
            max_seq: None,
            redline: false,
            speculation: None,
            reasoning_on: false,
            concurrency: None,
            backend: "both".to_owned(),
            workload: "both".to_owned(),
            bench_shape: "steady".to_owned(),
            bb_corpus: None,
            num_prompts: None,
            temperature: None,
            prompt: Vec::new(),
            prompt_file: None,
            emu: None,
        };
        let (mut engine, _, _, _, _, _, _) =
            open_bench_engine(paths, &bench, None, &BenchLoadOpts::default())?;
        let _ = bench_generate(&mut engine, "Hello", 1)?;
        engine
    } else {
        let (_, resolved) = resolved_global(paths, true)?;
        let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
        let daemon = find_daemon(paths).ok_or_else(|| daemon_not_found_error(paths))?;
        let mut engine =
            Engine::spawn_configured(&daemon, &BTreeMap::new(), &process_config, None)?;
        engine.ping()?;
        engine
    };
    let mut report = engine.request(&serde_json::json!({ "type": "profile" }))?;
    if report.get("type").and_then(serde_json::Value::as_str) != Some("profile") {
        bail!(
            "daemon profile failed: {}",
            report
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unexpected response")
        );
    }
    if let Some(filter) = args.kernel.as_deref() {
        let filtered = report
            .get("kernels")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter(|kernel| {
                kernel
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|name| name.contains(filter))
            })
            .cloned()
            .collect::<Vec<_>>();
        report["kernels"] = serde_json::Value::Array(filtered);
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        let gpu = &report["gpu"];
        println!(
            "GPU: {} ({})",
            gpu.get("arch")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown"),
            gpu.get("generation")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
        );
        println!(
            "{} CUs | peak BW {:.0} GB/s | boost {} MHz | ridge {:.1} FLOP/byte",
            gpu.get("cu_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            gpu.get("peak_bw_gbs")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
            gpu.get("boost_clock_mhz")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            gpu.get("ridge_point")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
        );
        let kernels = report["kernels"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        println!("\nKernel report ({} kernels):", kernels.len());
        println!(
            "  {:<38} {:>5} {:>5} {:>8} {:>10}  limiter",
            "kernel", "VGPR", "SGPR", "LDS", "occupancy"
        );
        for kernel in kernels {
            println!(
                "  {:<38} {:>5} {:>5} {:>8} {:>9.1}%  {}",
                kernel
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown"),
                kernel
                    .get("vgprs")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                kernel
                    .get("sgprs")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                kernel
                    .get("lds_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                kernel
                    .pointer("/occupancy/pct")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
                kernel
                    .pointer("/occupancy/limiter")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown"),
            );
        }
        println!("\nFor phase-aware ISA fit evidence, run python3 scripts/kernel_atlas.py render-fit --row <atlas-row.json>.");
    }
    Ok(())
}

fn version_command(paths: &Paths, output: OutputArgs) -> Result<()> {
    let installed = paths.root.join("src");
    let (source_kind, source) = if installed.join("Cargo.toml").is_file() {
        ("managed", installed)
    } else {
        (
            "build checkout",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        )
    };
    let source = fs::canonicalize(&source).unwrap_or(source);
    let source_commit = git_output(&source, &["rev-parse", "--verify", "HEAD"]).ok();
    let source_ref = git_output(&source, &["describe", "--tags", "--exact-match", "HEAD"])
        .ok()
        .or_else(|| git_output(&source, &["symbolic-ref", "--short", "HEAD"]).ok());
    let source_dirty = git_output(&source, &["status", "--porcelain"])
        .ok()
        .map(|status| !status.is_empty());
    let source_matches_build = source_commit
        .as_deref()
        .filter(|_| BUILD_COMMIT != "unknown")
        .map(|commit| commit == BUILD_COMMIT);
    let daemon = ["daemon", "daemon.exe"]
        .into_iter()
        .map(|name| paths.root.join("bin").join(name))
        .find(|path| path.is_file());
    let daemon_sha256 = daemon
        .as_deref()
        .map(sha256_path)
        .transpose()
        .context("failed to hash installed daemon")?;
    let value = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "build": {
            "commit": BUILD_COMMIT,
            "ref": BUILD_REF,
            "dirty": BUILD_DIRTY == "true",
            "target": BUILD_TARGET,
        },
        "source": {
            "kind": source_kind,
            "path": source,
            "commit": source_commit,
            "ref": source_ref,
            "dirty": source_dirty,
            "matches_build": source_matches_build,
        },
        "daemon": {
            "path": daemon,
            "sha256": daemon_sha256,
        },
        "config_schema_version": CONFIG_SCHEMA_VERSION,
    });
    if output.json {
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }

    println!("hipfire {}", env!("CARGO_PKG_VERSION"));
    println!("  build commit: {BUILD_COMMIT}");
    println!(
        "  build ref:    {BUILD_REF}{}",
        if BUILD_DIRTY == "true" {
            " (dirty)"
        } else {
            ""
        }
    );
    println!("  build target: {BUILD_TARGET}");
    println!("  source:       {source_kind} {}", source.display());
    println!(
        "  source ref:   {}",
        source_ref.as_deref().unwrap_or("unknown")
    );
    println!(
        "  source commit: {}",
        source_commit
            .as_deref()
            .map(str::to_owned)
            .unwrap_or_else(|| "unknown".into())
    );
    println!(
        "  source state: {}",
        match source_dirty {
            Some(true) => "dirty",
            Some(false) => "clean",
            None => "unknown",
        }
    );
    println!(
        "  source/build: {}",
        match source_matches_build {
            Some(true) => "match",
            Some(false) => "MISMATCH",
            None => "unknown",
        }
    );
    if let (Some(path), Some(digest)) = (daemon, daemon_sha256) {
        println!("  daemon:       {}", path.display());
        println!("  daemon sha256: {digest}");
    } else {
        println!("  daemon:       not installed");
    }
    Ok(())
}

/// Cooperative cancel flag for `hipfire update`. SIGINT/SIGTERM set this;
/// handlers never call `process::exit` so the armed rollback guard can run.
static UPDATE_INTERRUPT: AtomicBool = AtomicBool::new(false);

fn install_update_interrupt_handler() {
    // `termination` enables SIGTERM alongside SIGINT. Ignore AlreadyExists so a
    // pre-installed process handler does not abort update.
    let _ = ctrlc::set_handler(|| {
        UPDATE_INTERRUPT.store(true, Ordering::SeqCst);
    });
}

fn update_interrupted() -> bool {
    UPDATE_INTERRUPT.load(Ordering::SeqCst)
}

fn ensure_update_not_interrupted() -> Result<()> {
    if update_interrupted() {
        bail!("update interrupted");
    }
    Ok(())
}

/// Why `hipfire update` cannot run on `os` (a `std::env::consts::OS` value);
/// `None` where it is supported. Pure so the policy is testable on any host.
fn update_unsupported_message(os: &str, version: &str) -> Option<String> {
    match os {
        "linux" => None,
        "windows" => Some(format!(
            "hipfire update is not available on Windows (installed hipfire {version}); no files were changed. \
             Windows updates require re-running scripts/install.ps1 after stopping the engine. \
             For current master run in PowerShell:\n\
             iex (irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1)\n\
             For a specific revision, use the reviewed, pinned install in docs/GETTING_STARTED.md."
        )),
        other => Some(format!(
            "hipfire update is Linux-only (installed hipfire {version}, running on {other}); \
             re-run the platform installer with a revision selector on this OS"
        )),
    }
}

fn update_command(paths: &Paths, args: UpdateArgs) -> Result<()> {
    if let Some(message) =
        update_unsupported_message(std::env::consts::OS, env!("CARGO_PKG_VERSION"))
    {
        bail!("{message}");
    }
    // Install before any fetch/mutation so SIGINT cannot race past the guard.
    install_update_interrupt_handler();
    UPDATE_INTERRUPT.store(false, Ordering::SeqCst);

    let requested = parse_revision_selector(&args)?;
    let installed = paths.root.join("src");
    let managed = installed.join("Cargo.toml").is_file();
    let repo = if managed {
        installed
    } else {
        if requested.is_some() {
            bail!(
                "revision switching is limited to managed installs under {}; \
                 run install.sh --ref <ref> once to create one",
                paths.root.join("src").display()
            );
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    };
    let repo = fs::canonicalize(&repo).unwrap_or(repo);
    let current_branch = git_output(&repo, &["symbolic-ref", "--short", "HEAD"]).ok();
    if !managed && current_branch.as_deref() != Some("master") {
        bail!(
            "this binary was built from an unmanaged '{}' checkout; \
             update it with git or install a managed copy",
            current_branch.as_deref().unwrap_or("detached")
        );
    }
    let selector = requested
        .or_else(|| {
            current_branch.as_ref().map(|branch| RevisionSelector {
                value: branch.clone(),
                kind: RevisionKind::Branch,
            })
        })
        .ok_or_else(|| {
            anyhow!(
                "this installation is pinned at a detached commit; \
                 choose a target such as `hipfire update @master`"
            )
        })?;

    ensure_update_not_interrupted()?;
    eprintln!(
        "fetching {} '{}' from origin...",
        selector.kind.label(),
        selector.value
    );
    let resolved = fetch_revision(&repo, selector)?;
    ensure_update_not_interrupted()?;
    let previous_head = git_output(&repo, &["rev-parse", "--verify", "HEAD"])?;
    let previous_branch = git_output(&repo, &["symbolic-ref", "--short", "HEAD"]).ok();
    let short = previous_head.get(..12).unwrap_or(&previous_head);
    let backup_ref = format!(
        "refs/hipfire/backups/pre-update-{}-{short}",
        unix_timestamp()
    );
    run_checked(
        Command::new("git")
            .current_dir(&repo)
            .args(["update-ref", &backup_ref, &previous_head]),
        "git update-ref backup",
    )?;
    eprintln!("previous source retained at {backup_ref}");

    let mut checkpoint = UpdateCheckpoint {
        head: previous_head,
        branch: previous_branch,
        stash_sha: None,
    };

    let dirty = !git_output(&repo, &["status", "--porcelain"])?.is_empty();
    if dirty {
        let message = format!("hipfire-update-{}", unix_timestamp());
        eprintln!("local modifications detected; stashing as {message}");
        run_checked(
            Command::new("git").current_dir(&repo).args([
                "stash",
                "push",
                "--include-untracked",
                "-m",
                &message,
            ]),
            "git stash",
        )?;
        checkpoint.stash_sha = Some(git_output(&repo, &["rev-parse", "stash@{0}"])?);
        eprintln!("recover later with: git -C {} stash pop", repo.display());
    }

    // Armed after clean/stashed checkpoint and before checkout. Drop/error
    // rolls back unless explicitly committed after installer exit 0.
    let mut guard = UpdateRollbackGuard::arm(repo.clone(), checkpoint);
    ensure_update_not_interrupted()?;

    if let Err(err) = checkout_revision(&repo, &resolved) {
        return Err(guard.fail(err));
    }
    ensure_update_not_interrupted()?;

    match run_update_installer(&repo, paths, &resolved) {
        Ok(()) => {
            guard.commit();
            println!(
                "hipfire updated to {} '{}' ({})",
                resolved.selector.kind.label(),
                resolved.selector.value,
                resolved.commit
            );
            println!("verify with: hipfire version");
            Ok(())
        }
        Err(err) => Err(guard.fail(err)),
    }
}

#[derive(Debug, Clone)]
struct UpdateCheckpoint {
    head: String,
    branch: Option<String>,
    stash_sha: Option<String>,
}

/// Restores pre-update checkout/stash unless [`Self::commit`] is called after
/// a successful installer handoff. Drop and explicit fail both roll back.
struct UpdateRollbackGuard {
    repo: PathBuf,
    checkpoint: UpdateCheckpoint,
    armed: bool,
}

impl UpdateRollbackGuard {
    fn arm(repo: PathBuf, checkpoint: UpdateCheckpoint) -> Self {
        Self {
            repo,
            checkpoint,
            armed: true,
        }
    }

    fn commit(&mut self) {
        self.armed = false;
    }

    #[cfg(test)]
    fn is_armed(&self) -> bool {
        self.armed
    }

    fn fail(mut self, err: anyhow::Error) -> anyhow::Error {
        if let Err(restore_err) = self.rollback() {
            eprintln!(
                "WARNING: failed to restore pre-update checkout after failure: {restore_err}"
            );
            return err.context(format!("pre-update restore also failed: {restore_err}"));
        }
        err
    }

    fn rollback(&mut self) -> Result<()> {
        if !self.armed {
            return Ok(());
        }
        self.armed = false;
        restore_update_checkpoint(&self.repo, &self.checkpoint)
    }
}

impl Drop for UpdateRollbackGuard {
    fn drop(&mut self) {
        if let Err(err) = self.rollback() {
            eprintln!("WARNING: failed to restore pre-update checkout on drop: {err}");
        }
    }
}

fn run_update_installer(repo: &Path, paths: &Paths, resolved: &ResolvedRevision) -> Result<()> {
    let installer = repo.join("scripts/install.sh");
    if !installer.is_file() {
        bail!("updated checkout has no {}", installer.display());
    }
    let recorded = recorded_install_metadata(&paths.root);
    let mut installer_cmd = Command::new("bash");
    installer_cmd
        .arg(&installer)
        .current_dir(repo)
        .env("HIPFIRE_FORCE_REBUILD", "1");
    #[cfg(unix)]
    {
        // Own process group so SIGTERM/KILL can reach the whole installer tree.
        installer_cmd.process_group(0);
    }
    for arg in installer_handoff_args(
        &resolved.selector,
        recorded.rocm_root.as_deref(),
        recorded.gpu_arch.as_deref(),
        recorded.hipcc.as_deref(),
        recorded.strict_rocm,
    ) {
        installer_cmd.arg(arg);
    }
    run_update_installer_child(installer_cmd)
}

/// Spawn the installer, poll-wait, and on interrupt TERM then KILL the group.
fn run_update_installer_child(mut installer_cmd: Command) -> Result<()> {
    let mut child = installer_cmd
        .spawn()
        .context("failed to start native installer")?;
    let status = wait_update_installer_child(&mut child)?;
    if update_interrupted() {
        bail!("update interrupted");
    }
    if status.success() {
        Ok(())
    } else {
        bail!("native installer failed with {status}")
    }
}

fn wait_update_installer_child(child: &mut Child) -> Result<std::process::ExitStatus> {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {
                if update_interrupted() {
                    terminate_update_installer_group(child);
                    return child
                        .wait()
                        .context("failed to reap interrupted native installer");
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(err) => {
                return Err(err).context("failed to wait for native installer");
            }
        }
    }
}

/// TERM the installer process group, then bounded KILL fallback; always wait/reap.
fn terminate_update_installer_group(child: &mut Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        if pid > 0 {
            // Negative pid targets the process group created via process_group(0).
            unsafe {
                libc::kill(-pid, libc::SIGTERM);
            }
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                match child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) => thread::sleep(Duration::from_millis(50)),
                    Err(_) => break,
                }
            }
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
}

/// Args forwarded to scripts/install.sh during noninteractive update handoff.
fn installer_handoff_args(
    selector: &RevisionSelector,
    rocm_root: Option<&Path>,
    gpu_arch: Option<&str>,
    hipcc: Option<&Path>,
    strict_rocm: bool,
) -> Vec<String> {
    let mut args = vec!["--yes".to_owned()];
    match selector.kind {
        RevisionKind::Auto => {
            args.push("--ref".to_owned());
            args.push(selector.value.clone());
        }
        RevisionKind::Branch => {
            args.push("--branch".to_owned());
            args.push(selector.value.clone());
        }
        RevisionKind::Tag => {
            args.push("--tag".to_owned());
            args.push(selector.value.clone());
        }
        RevisionKind::Commit => {
            args.push("--commit".to_owned());
            args.push(selector.value.clone());
        }
    }
    if let Some(root) = rocm_root {
        args.push("--rocm-root".to_owned());
        args.push(root.to_string_lossy().into_owned());
    }
    if let Some(arch) = gpu_arch.map(str::trim).filter(|arch| !arch.is_empty()) {
        args.push("--gpu-arch".to_owned());
        args.push(arch.to_owned());
    }
    if let Some(hipcc) = hipcc
        .map(|p| p.to_string_lossy().into_owned())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
    {
        args.push("--hipcc".to_owned());
        args.push(hipcc);
    }
    if strict_rocm {
        args.push("--strict-rocm".to_owned());
    }
    args
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct RecordedInstallMetadata {
    rocm_root: Option<PathBuf>,
    gpu_arch: Option<String>,
    hipcc: Option<PathBuf>,
    strict_rocm: bool,
}
fn recorded_install_metadata(install_home: &Path) -> RecordedInstallMetadata {
    let text = match fs::read_to_string(install_home.join("install.json")) {
        Ok(text) => text,
        Err(_) => return RecordedInstallMetadata::default(),
    };
    let value: serde_json::Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(_) => return RecordedInstallMetadata::default(),
    };
    let rocm_root = value
        .get("rocm_root")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|root| !root.is_empty())
        .map(PathBuf::from);
    let gpu_arch = value
        .get("gpu_arch")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|arch| !arch.is_empty())
        .map(str::to_owned);
    let hipcc = value
        .get("hipcc")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(PathBuf::from);
    let strict_rocm = match value.get("strict_rocm") {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::String(s)) => {
            let s = s.trim();
            s == "1" || s.eq_ignore_ascii_case("true")
        }
        Some(serde_json::Value::Number(n)) => n.as_u64().is_some_and(|v| v != 0),
        _ => false,
    };
    RecordedInstallMetadata {
        rocm_root,
        gpu_arch,
        hipcc,
        strict_rocm,
    }
}

fn restore_update_checkpoint(repo: &Path, checkpoint: &UpdateCheckpoint) -> Result<()> {
    // Failed-target Cargo/source mutations are wiped only after the original
    // user work is already in the update stash, so restore cannot be blocked.
    run_checked(
        Command::new("git")
            .current_dir(repo)
            .args(["reset", "--hard"]),
        "git reset failed target",
    )?;
    run_checked(
        Command::new("git").current_dir(repo).args(["clean", "-fd"]),
        "git clean failed target",
    )?;
    if let Some(branch) = checkpoint.branch.as_deref() {
        run_checked(
            Command::new("git").current_dir(repo).args([
                "checkout",
                "-B",
                branch,
                &checkpoint.head,
            ]),
            "git restore previous branch",
        )?;
    } else {
        run_checked(
            Command::new("git")
                .current_dir(repo)
                .args(["checkout", "--detach", &checkpoint.head]),
            "git restore previous commit",
        )?;
    }
    if let Some(stash_sha) = checkpoint.stash_sha.as_deref() {
        reapply_update_stash(repo, stash_sha)?;
    }
    Ok(())
}

fn reapply_update_stash(repo: &Path, stash_sha: &str) -> Result<()> {
    // Preserve staged index state from the original dirty tree.
    if let Err(err) = run_checked(
        Command::new("git")
            .current_dir(repo)
            .args(["stash", "apply", "--index", stash_sha]),
        "git stash apply --index",
    ) {
        bail!(
            "failed to restore pre-update stash {stash_sha} (kept for recovery: \
             git -C {} stash apply --index {stash_sha}): {err}",
            repo.display()
        );
    }
    // Drop only after successful apply so a conflicted restore keeps the stash.
    if let Ok(list) = git_output(repo, &["stash", "list", "--format=%gd %H"]) {
        for line in list.lines() {
            let mut parts = line.split_whitespace();
            let Some(gd) = parts.next() else {
                continue;
            };
            let Some(hash) = parts.next() else {
                continue;
            };
            if hash == stash_sha || stash_sha.starts_with(hash) || hash.starts_with(stash_sha) {
                let _ = run_checked(
                    Command::new("git")
                        .current_dir(repo)
                        .args(["stash", "drop", gd]),
                    "git stash drop",
                );
                break;
            }
        }
    }
    Ok(())
}

fn parse_revision_selector(args: &UpdateArgs) -> Result<Option<RevisionSelector>> {
    let candidates = [
        args.reference
            .as_ref()
            .map(|value| (value.as_str(), RevisionKind::Auto)),
        args.branch
            .as_ref()
            .map(|value| (value.as_str(), RevisionKind::Branch)),
        args.tag
            .as_ref()
            .map(|value| (value.as_str(), RevisionKind::Tag)),
        args.commit
            .as_ref()
            .map(|value| (value.as_str(), RevisionKind::Commit)),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    if candidates.len() > 1 {
        bail!("choose only one update ref, --branch, --tag, or --commit");
    }
    let Some((raw, mut kind)) = candidates.first().copied() else {
        return Ok(None);
    };
    let mut value = raw.trim().trim_start_matches('@');
    if let Some(branch) = value.strip_prefix("refs/heads/") {
        value = branch;
        kind = RevisionKind::Branch;
    } else if let Some(tag) = value.strip_prefix("refs/tags/") {
        value = tag;
        kind = RevisionKind::Tag;
    } else if let Some(branch) = value.strip_prefix("origin/") {
        value = branch;
        if kind == RevisionKind::Auto {
            kind = RevisionKind::Branch;
        }
    }
    validate_revision(value, kind)?;
    Ok(Some(RevisionSelector {
        value: value.to_owned(),
        kind,
    }))
}

fn validate_revision(value: &str, kind: RevisionKind) -> Result<()> {
    let invalid = value.is_empty()
        || value.starts_with(['-', '.', '/'])
        || value.ends_with(['.', '/'])
        || value.contains("..")
        || value.contains("@{")
        || value.contains("//")
        || value.chars().any(|character| {
            character.is_whitespace()
                || character.is_control()
                || matches!(character, '\\' | ':' | '?' | '*' | '[' | '^' | '~')
        });
    if invalid {
        bail!("unsafe or invalid git revision {value:?}");
    }
    if kind == RevisionKind::Commit
        && (!(7..=40).contains(&value.len())
            || !value.chars().all(|character| character.is_ascii_hexdigit()))
    {
        bail!("--commit requires a 7-40 character hexadecimal git commit");
    }
    Ok(())
}

fn fetch_revision(repo: &Path, mut selector: RevisionSelector) -> Result<ResolvedRevision> {
    if selector.kind == RevisionKind::Auto {
        selector.kind = if remote_ref_exists(repo, &format!("refs/heads/{}", selector.value))? {
            RevisionKind::Branch
        } else if remote_ref_exists(repo, &format!("refs/tags/{}", selector.value))? {
            RevisionKind::Tag
        } else {
            RevisionKind::Commit
        };
    }

    match selector.kind {
        RevisionKind::Branch => {
            let remote = format!("refs/heads/{}", selector.value);
            if !remote_ref_exists(repo, &remote)? {
                bail!("origin has no branch '{}'", selector.value);
            }
            let tracking = format!("refs/remotes/origin/{}", selector.value);
            let refspec = format!("+{remote}:{tracking}");
            run_checked(
                Command::new("git")
                    .current_dir(repo)
                    .args(["fetch", "origin", &refspec]),
                "git fetch branch",
            )?;
            let commit = git_output(repo, &["rev-parse", "--verify", &tracking])?;
            Ok(ResolvedRevision {
                selector,
                commit,
                tracking_ref: Some(tracking),
            })
        }
        RevisionKind::Tag => {
            let remote = format!("refs/tags/{}", selector.value);
            if !remote_ref_exists(repo, &remote)? {
                bail!("origin has no tag '{}'", selector.value);
            }
            run_checked(
                Command::new("git")
                    .current_dir(repo)
                    .args(["fetch", "--depth", "1", "origin", &remote]),
                "git fetch tag",
            )?;
            let commit = git_output(repo, &["rev-parse", "--verify", "FETCH_HEAD^{commit}"])?;
            Ok(ResolvedRevision {
                selector,
                commit,
                tracking_ref: None,
            })
        }
        RevisionKind::Commit => {
            run_checked(
                Command::new("git").current_dir(repo).args([
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    &selector.value,
                ]),
                "git fetch commit",
            )?;
            let commit = git_output(repo, &["rev-parse", "--verify", "FETCH_HEAD^{commit}"])?;
            Ok(ResolvedRevision {
                selector,
                commit,
                tracking_ref: None,
            })
        }
        RevisionKind::Auto => unreachable!("auto revisions are resolved before fetch"),
    }
}

fn checkout_revision(repo: &Path, resolved: &ResolvedRevision) -> Result<()> {
    if let Some(tracking) = &resolved.tracking_ref {
        refuse_unpushed_branch_commits(repo, &resolved.selector.value, tracking)?;
        run_checked(
            Command::new("git").current_dir(repo).args([
                "checkout",
                "-B",
                &resolved.selector.value,
                tracking,
            ]),
            "git checkout branch",
        )
    } else {
        run_checked(
            Command::new("git")
                .current_dir(repo)
                .args(["checkout", "--detach", &resolved.commit]),
            "git checkout pinned revision",
        )
    }
}

/// Refuse to reset a local branch that still has commits not present on the
/// resolved remote-tracking tip. Channel switches onto a different branch are
/// unaffected when that target branch is not ahead.
fn refuse_unpushed_branch_commits(repo: &Path, branch: &str, tracking: &str) -> Result<()> {
    let local_ref = format!("refs/heads/{branch}");
    let local_tip = match git_output(repo, &["rev-parse", "--verify", &local_ref]) {
        Ok(tip) => tip,
        Err(_) => return Ok(()),
    };
    let remote_tip = git_output(repo, &["rev-parse", "--verify", tracking])?;
    if local_tip == remote_tip {
        return Ok(());
    }
    // Behind (or equal ancestry): remote contains local tip → safe to fast-forward reset.
    if is_ancestor(repo, &local_tip, &remote_tip)? {
        return Ok(());
    }
    let ahead = git_output(
        repo,
        &["rev-list", "--count", &format!("{tracking}..{local_ref}")],
    )
    .ok()
    .and_then(|count| count.parse::<u64>().ok())
    .unwrap_or(1);
    bail!(
        "refusing to update branch '{branch}': {ahead} local commit(s) ahead of {tracking}; \
         push or move them before updating"
    );
}

fn is_ancestor(repo: &Path, maybe_ancestor: &str, commit: &str) -> Result<bool> {
    let status = Command::new("git")
        .current_dir(repo)
        .args(["merge-base", "--is-ancestor", maybe_ancestor, commit])
        .status()
        .with_context(|| format!("failed to compare git ancestry {maybe_ancestor} vs {commit}"))?;
    Ok(status.success())
}

fn remote_ref_exists(repo: &Path, reference: &str) -> Result<bool> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(["ls-remote", "--exit-code", "origin", reference])
        .output()
        .with_context(|| format!("failed to query origin for {reference}"))?;
    match output.status.code() {
        Some(0) => Ok(true),
        Some(2) => Ok(false),
        _ => bail!(
            "git ls-remote failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ),
    }
}

fn sha256_path(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn git_output(repo: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn run_checked(command: &mut Command, label: &str) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("failed to start {label}"))?;
    if status.success() {
        Ok(())
    } else {
        bail!("{label} failed with {status}")
    }
}

pub(crate) fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Files the quantizer writes BESIDE its trunk output, which an install or an
/// upload has to carry along: today the arch-18 `.vl` tower pack (the vision
/// and audio tower groups). Located with the loader's own probe, so an install
/// and a later load can never disagree about which file that is.
fn derived_packs(trunk: &std::path::Path) -> Vec<std::path::PathBuf> {
    hipfire_runtime::sidecar::find_vl_sidecar(trunk)
        .into_iter()
        .collect()
}

/// Arguments the public `hipfire quantize` passes to `hipfire-quantize`: the
/// historical `--input/--output/--format`, plus the recipe controls when given.
fn quantizer_argv(
    input: &str,
    output: &Path,
    format: &str,
    mq4v2_symmetric: bool,
    tier: Option<&str>,
) -> Vec<std::ffi::OsString> {
    let mut argv: Vec<std::ffi::OsString> = vec![
        "--input".into(),
        input.into(),
        "--output".into(),
        output.as_os_str().to_owned(),
        "--format".into(),
        format.into(),
    ];
    if mq4v2_symmetric {
        argv.push("--mq4v2-symmetric".into());
    }
    if let Some(tier) = tier {
        argv.push("--tier".into());
        argv.push(tier.into());
    }
    argv
}

fn quantize_command(paths: &Paths, mut args: QuantizeArgs) -> Result<()> {
    let quantizer = find_workspace_binary(paths, "hipfire-quantize").ok_or_else(|| {
        anyhow!(
            "hipfire-quantize is not installed; build `cargo build --release -p hipfire-quantize`"
        )
    })?;
    if args.both {
        args.formats.extend(["mq4".into(), "mq6".into()]);
    }
    let input_path = PathBuf::from(&args.input);
    let is_gguf = input_path.is_file()
        && input_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("gguf"));
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — GGUF→mqN is lossy double quantization; use llama.cpp for GGUF
    // (the one-line warning is printed by hipfire-quantize itself)
    if args.formats.is_empty() {
        args.formats
            .push(if is_gguf { "hf4".into() } else { "mq4".into() });
    }
    for format in &mut args.formats {
        *format = match format.as_str() {
            "hfq4" | "hfq4g256" => "hf4".into(),
            "hfq6" | "hfq6g256" => "hf6".into(),
            _ => format.clone(),
        };
    }
    let mut seen = BTreeSet::new();
    args.formats.retain(|format| seen.insert(format.clone()));
    let valid = ["mq4", "mq6", "q8", "q8f16", "hf4", "hf6"];
    for format in &args.formats {
        if !valid.contains(&format.as_str()) {
            bail!(
                "unsupported format {format}; supported: {}",
                valid.join(", ")
            );
        }
        if is_gguf && !matches!(format.as_str(), "hf4" | "hf6" | "mq4" | "mq6") {
            bail!("GGUF input supports hf4, hf6, mq4, or mq6");
        }
    }
    if (args.mq4v2_symmetric || args.tier.is_some()) && args.formats.iter().any(|f| f != "mq4") {
        bail!("--mq4v2-symmetric and --tier apply to --format mq4 only");
    }
    if args.output.is_some() && args.formats.len() != 1 {
        bail!("--output requires exactly one format; use --output-dir for multiple formats");
    }
    if let Some(repo) = args.upload.as_deref() {
        if repo.split('/').count() != 2 {
            bail!("--upload requires owner/repo");
        }
    }
    let input = if input_path.exists() {
        fs::canonicalize(input_path)?.display().to_string()
    } else {
        args.input.clone()
    };
    let stem = args.stem.unwrap_or_else(|| {
        Path::new(&args.input)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(&args.input)
            .trim_end_matches(".gguf")
            .to_owned()
    });
    let output_dir = args.output_dir.unwrap_or(env::current_dir()?);
    fs::create_dir_all(&output_dir)?;
    let mut produced = Vec::new();
    for format in &args.formats {
        let output = args
            .output
            .clone()
            .unwrap_or_else(|| output_dir.join(format!("{stem}.{format}")));
        eprintln!("quantizing {input} -> {} ({format})", output.display());
        run_checked(
            Command::new(&quantizer).args(quantizer_argv(
                &input,
                &output,
                format,
                args.mq4v2_symmetric,
                args.tier.as_deref(),
            )),
            "hipfire-quantize",
        )?;
        if !output.is_file() {
            bail!(
                "quantizer reported success but {} was not created",
                output.display()
            );
        }
        produced.push((format.clone(), fs::canonicalize(&output).unwrap_or(output)));
    }
    if args.install {
        fs::create_dir_all(&paths.models)?;
        for (_, output) in &mut produced {
            // Packs the quantizer wrote BESIDE the trunk (the arch-18 `.vl`
            // tower sidecar) must be located before `output` becomes the
            // installed path.
            let packs = derived_packs(output);
            let file = output
                .file_name()
                .ok_or_else(|| anyhow!("invalid output path {}", output.display()))?;
            let destination = paths.models.join(file);
            if fs::canonicalize(&destination).ok().as_ref()
                != fs::canonicalize(&*output).ok().as_ref()
            {
                fs::copy(&*output, &destination)?;
            }
            *output = fs::canonicalize(&destination).unwrap_or(destination);
            eprintln!("installed {}", output.display());
            // Installing the trunk alone would leave a 1 GB tower pack in the
            // caller's working directory and hand the user a text-only install
            // (media refused with no tower): carry each derived pack over under
            // its own name so it lands as the trunk's sibling.
            for pack in packs {
                let pack_file = pack
                    .file_name()
                    .ok_or_else(|| anyhow!("invalid sidecar path {}", pack.display()))?;
                let pack_dst = paths.models.join(pack_file);
                if fs::canonicalize(&pack_dst).ok().as_ref()
                    != fs::canonicalize(&pack).ok().as_ref()
                {
                    fs::copy(&pack, &pack_dst)?;
                }
                eprintln!(
                    "installed {}",
                    fs::canonicalize(&pack_dst).unwrap_or(pack_dst).display()
                );
            }
        }
    }
    if let Some(repo) = args.upload.as_deref() {
        if args.create_repo {
            run_checked(
                Command::new("hf").args(["repos", "create", repo, "--type", "model", "--exist-ok"]),
                "hf repos create",
            )?;
        }
        for (_, output) in &produced {
            // The trunk plus everything the quantizer wrote beside it (the
            // arch-18 `.vl` tower pack), so an upload is not a trunk-only
            // artifact that loads text-only on the far side.
            for path in std::iter::once(output.clone()).chain(derived_packs(output)) {
                let name = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| anyhow!("invalid upload filename"))?;
                run_checked(
                    Command::new("hf")
                        .arg("upload")
                        .arg(repo)
                        .arg(&path)
                        .arg(name),
                    "hf upload",
                )?;
            }
        }
    }
    if let Some(alias) = args.register {
        let (_, primary) = produced
            .iter()
            .find(|(format, _)| format == "mq4")
            .or_else(|| produced.first())
            .ok_or_else(|| anyhow!("no quantized artifact produced"))?;
        let mut loaded = load_catalog(&paths.config)?;
        let id = primary
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| anyhow!("invalid output filename"))?
            .to_owned();
        loaded.catalog.models.insert(
            id.clone(),
            hipfire_config::LocalModelConfig {
                path: Some(primary.clone()),
                registry_tag: None,
                overrides: ConfigLayer::default(),
            },
        );
        loaded.catalog.aliases.insert(alias.clone(), id);
        write_catalog_toml(&paths.config, &loaded.catalog)?;
        eprintln!("registered {alias} -> {}", primary.display());
    }
    Ok(())
}

// lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
fn sidecar_command(paths: &Paths, args: SidecarArgs) -> Result<()> {
    eprintln!(
        "warning: CASK is deprecated and will be removed in 0.5.0; not supported (sidecar-gen)"
    );
    if !(1..=1_000_000).contains(&args.max_tokens) {
        bail!("--max-tokens must be between 1 and 1000000");
    }
    if !(1..=16_384).contains(&args.chunk_len) {
        bail!("--chunk-len must be between 1 and 16384");
    }
    if let Some(corpus) = args.corpus.as_ref().filter(|path| !path.is_file()) {
        bail!("corpus not found: {}", corpus.display());
    }
    let registry = load_registry(&paths.registry).registry;
    let model = find_model_path(paths, &registry, &args.model)
        .ok_or_else(|| model_not_found_error(paths, &registry, &args.model))?;
    let output = args
        .output
        .unwrap_or_else(|| PathBuf::from(format!("{}.triattn.bin", model.display())));
    let binary = find_workspace_example(paths, "triattn_validate").ok_or_else(|| anyhow!(
        "triattn_validate is not installed; build `cargo build --release --features deltanet -p hipfire-runtime --example triattn_validate`"
    ))?;
    let mut command = Command::new(binary);
    command
        .arg(&model)
        .arg("--sidecar")
        .arg(&output)
        .arg("--max-tokens")
        .arg(args.max_tokens.to_string())
        .arg("--chunk-len")
        .arg(args.chunk_len.to_string());
    if let Some(corpus) = args.corpus {
        command.arg("--corpus").arg(corpus);
    }
    if args.cpu_calib {
        command.arg("--cpu-calib");
    }
    if args.skip_validation {
        command.arg("--val-prompt").arg("");
    }
    let _ = args.gpu_calib;
    run_checked(&mut command, "triattn_validate")?;
    if !output.is_file() {
        bail!(
            "sidecar generator reported success but {} was not created",
            output.display()
        );
    }
    println!("{}", output.display());
    Ok(())
}

fn find_workspace_binary(paths: &Paths, name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    };
    [
        paths.root.join("bin").join(&exe),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/release")
            .join(&exe),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .or_else(|| find_on_path(&exe))
}

fn find_workspace_example(paths: &Paths, name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    };
    [
        paths.root.join("bin").join(&exe),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/release/examples")
            .join(&exe),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .or_else(|| find_on_path(&exe))
}

fn diag_command(paths: &Paths, output: OutputArgs) -> Result<()> {
    let loaded_registry = load_registry(&paths.registry);
    let models = list_local_models(paths, &loaded_registry.registry)?;
    let loaded_config = load_global(&paths.config)?;
    let platform = format!("{}-{}", env::consts::OS, env::consts::ARCH);
    let kfd = Path::new("/dev/kfd").exists();
    let amdgpu_loaded = Path::new("/sys/module/amdgpu").exists();
    let gpu_arches = detect_gpu_arches();
    let gpus = detect_amd_drm_cards();
    let hipcc = command_version("hipcc", "--version");
    // Per-root component inventory. A working `hipcc` says nothing about the
    // HIP headers or runtime — they are separate packages — so reporting only
    // the hipcc version made a half-installed ROCm look healthy here while
    // every kernel compile and dlopen failed elsewhere.
    let rocm_roots = hipfire_config::rocm::roots()
        .iter()
        .filter(|root| root.is_dir())
        .map(|root| {
            let missing = hipfire_config::rocm::missing_components(root);
            serde_json::json!({
                "path": root.display().to_string(),
                "device_compiler": hipfire_config::rocm::DEVICE_COMPILERS
                    .iter()
                    .find_map(|name| hipfire_config::rocm::tool_from_selected_root(root, name)),
                "hip_headers": hipfire_config::rocm::is_complete_root(root),
                "hip_runtime": hipfire_config::rocm::runtime_library(root)
                    .map(|p| p.display().to_string()),
                "missing": missing.iter().map(|m| m.what).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let rocm_selected = hipfire_config::rocm::root().map(|p| p.display().to_string());
    let daemon_path = find_daemon(paths);
    let daemon = daemon_path.as_ref().map(|path| path.display().to_string());
    let live_gpu = daemon_path.as_ref().and_then(|daemon| {
        let (_, resolved) = resolved_global(paths, true).ok()?;
        let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved).ok()?;
        let mut engine =
            Engine::spawn_configured(daemon, &BTreeMap::new(), &process_config, None).ok()?;
        engine.ping().ok()?;
        engine.request(&serde_json::json!({ "type": "diag" })).ok()
    });
    let gpu = gpu_arches
        .first()
        .map(|arch| serde_json::json!({ "arch": arch }))
        .unwrap_or_else(
            || serde_json::json!({ "error": "no gfx target detected in KFD topology" }),
        );
    let config_overrides = loaded_config
        .layer
        .values
        .iter()
        .map(|(key, value)| (key.clone(), serde_json::json!(value)))
        .collect::<serde_json::Map<_, _>>();
    // One DXGI query serves both the JSON and the text report. Failures stay
    // as text so the report names them instead of showing an empty adapter list.
    #[cfg(windows)]
    let gpu_budget_query =
        hipfire_config::host_memory::dxgi_adapter_budgets().map_err(|error| error.to_string());
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut report = serde_json::json!({
        "registry": registry_source(loaded_registry.source),
        "platform": platform,
        "hardware_probe": if cfg!(target_os = "linux") { "linux" } else { "limited" },
        "gpus": gpus,
        "dri_nodes": list_dri_nodes(),
        "kfd": kfd,
        "amdgpu_loaded": amdgpu_loaded,
        "rocm": { "hipcc": hipcc, "selected_root": rocm_selected, "roots": rocm_roots },
        "daemon": daemon,
        "live_gpu": live_gpu,
        "models": models,
        "gpu": gpu,
        "config_path": loaded_config.path,
        "config_format": format!("{:?}", loaded_config.format).to_lowercase(),
        "config_overrides": config_overrides,
        "warnings": loaded_registry.warnings,
        "host_memory": hipfire_config::host_memory(),
        "kho_scratch": kho_scratch_check(kho_scratch_probe()),
    });
    #[cfg(windows)]
    {
        report["gpu_memory_budget"] = gpu_memory_budget_json(
            std::process::id(),
            gpu_budget_query
                .as_ref()
                .map(Vec::as_slice)
                .map_err(String::as_str),
        );
    }
    if output.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("hipfire diagnostics\n");
        println!(
            "registry:      {}",
            report["registry"].as_str().unwrap_or("unknown")
        );
        println!("platform:      {platform}");
        println!(
            "amdgpu:       {}",
            if amdgpu_loaded {
                "loaded"
            } else {
                "not loaded"
            }
        );
        println!("/dev/kfd:      {}", if kfd { "present" } else { "missing" });
        println!(
            "GPU targets:   {}",
            if gpu_arches.is_empty() {
                "none".into()
            } else {
                gpu_arches.join(", ")
            }
        );
        println!("local models:  {}", models.len());
        println!(
            "ROCm root:     {}",
            rocm_selected.as_deref().unwrap_or("none found")
        );
        // Only actionable for a root that HAS a compiler: one without is a shim
        // directory (the /opt/rocm of a split-tree install), so its "missing"
        // components are expected rather than a problem to fix.
        let mut incomplete_toolchain = false;
        for root in &rocm_roots {
            let s = |k: &str| root[k].as_str().map(str::to_owned);
            println!(
                "  {}\n    compiler: {}   headers: {}   runtime: {}",
                root["path"].as_str().unwrap_or("?"),
                s("device_compiler").unwrap_or_else(|| "MISSING".into()),
                if root["hip_headers"].as_bool().unwrap_or(false) {
                    "yes"
                } else {
                    "MISSING"
                },
                s("hip_runtime").unwrap_or_else(|| "MISSING".into()),
            );
            let missing = root["missing"].as_array().map(Vec::len).unwrap_or(0);
            incomplete_toolchain |= missing > 0 && s("device_compiler").is_some();
        }
        if incomplete_toolchain {
            println!("  a ROCm root above has a compiler but no HIP runtime/headers:");
            for line in hipfire_config::rocm::install_guidance() {
                println!("    {line}");
            }
        }
        println!(
            "config:        {} ({:?})",
            loaded_config.path.display(),
            loaded_config.format
        );
        println!("daemon:        {}", daemon.as_deref().unwrap_or("missing"));
        println!(
            "KHO scratch:   {} ({})",
            report["kho_scratch"]["status"].as_str().unwrap_or("pass"),
            report["kho_scratch"]["detail"].as_str().unwrap_or("")
        );
        if let Some(live) = report.get("live_gpu").filter(|value| !value.is_null()) {
            println!(
                "HIP GPU:       {} (HIP {})",
                live.get("arch")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown"),
                live.get("hip_version")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
            );
            println!(
                "VRAM:          {} MB free / {} MB total",
                live.get("vram_free_mb")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                live.get("vram_total_mb")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
            );
            // Windows daemon only: the amdhip64*.dll that actually loaded (a
            // System32 driver copy can shadow an SDK's) and the raw versions.
            // Fields are null/absent when the daemon could not read them.
            #[cfg(windows)]
            {
                let raw = |key: &str| {
                    live.get(key)
                        .and_then(serde_json::Value::as_i64)
                        .map_or_else(|| "unknown".to_owned(), |value| value.to_string())
                };
                println!(
                    "HIP runtime:   {} (runtime {}, driver {})",
                    live.get("hip_runtime_library")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown"),
                    raw("hip_runtime_version_raw"),
                    raw("hip_driver_version_raw")
                );
            }
            if matches!(
                live.get("arch").and_then(serde_json::Value::as_str),
                Some("gfx1150" | "gfx1151" | "gfx1152")
            ) && live
                .get("hip_version")
                .and_then(serde_json::Value::as_str)
                .and_then(parse_major_minor)
                .is_some_and(|version| version < (7, 2))
            {
                println!("WARNING: RDNA 3.5 requires ROCm/HIP 7.2 or newer.");
            }
        } else if daemon.is_some() {
            println!("HIP probe:     failed (run the daemon directly for detailed startup errors)");
        }
        #[cfg(windows)]
        for line in gpu_memory_budget_text(
            std::process::id(),
            gpu_budget_query
                .as_ref()
                .map(Vec::as_slice)
                .map_err(String::as_str),
        ) {
            println!("{line}");
        }
    }
    Ok(())
}

/// Live KHO scratch bytes (`rdna_compute::uma_memory`); `None` off Linux.
#[cfg(target_os = "linux")]
fn kho_scratch_probe() -> Option<u64> {
    rdna_compute::uma_memory::kho_scratch_bytes()
}

#[cfg(not(target_os = "linux"))]
fn kho_scratch_probe() -> Option<u64> {
    None
}

/// `hipfire diag` host-memory check for Kexec HandOver scratch: `pass` when
/// there is none (or it is under the warning threshold), `warn` (never a
/// failure) with the shared warning text at 1 GiB or more. `scratch_bytes` is
/// the CMA pageblock total outside any registered CMA area.
fn kho_scratch_check(scratch_bytes: Option<u64>) -> serde_json::Value {
    let bytes = scratch_bytes.unwrap_or(0);
    let warning = rdna_compute::uma_memory::kho_scratch_warning(bytes);
    let detail = match (&warning, bytes) {
        (Some(text), _) => text.clone(),
        (None, 0) => "none".to_owned(),
        (None, bytes) => format!(
            "{:.1} MiB of CMA pageblocks outside any CMA area (below the warning threshold)",
            bytes as f64 / (1024.0 * 1024.0)
        ),
    };
    serde_json::json!({
        "name": "kho_scratch",
        "status": if warning.is_some() { "warn" } else { "pass" },
        "scratch_bytes": bytes,
        "detail": detail,
    })
}

/// Provenance of every figure in the diag DXGI section. These are budgets as
/// seen by the calling process (the CLI), not an admission budget: the engine
/// has to query DXGI itself from its own process before it admits a load.
#[cfg(any(windows, test))]
const GPU_MEMORY_BUDGET_SOURCE: &str = hipfire_config::host_memory::DXGI_BUDGET_SOURCE;

#[cfg(any(windows, test))]
fn dxgi_adapter_is_amd(adapter: &hipfire_config::host_memory::DxgiAdapterBudget) -> bool {
    hipfire_config::host_memory::amd_adapters(std::slice::from_ref(adapter))
        .next()
        .is_some()
}

#[cfg(any(windows, test))]
fn dxgi_luid_hex(luid: &[u8; 8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(16);
    for &byte in luid {
        text.push(HEX[(byte >> 4) as usize] as char);
        text.push(HEX[(byte & 0xf) as usize] as char);
    }
    text
}

#[cfg(any(windows, test))]
fn dxgi_mib_text(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// JSON for `hipfire diag --json` key `gpu_memory_budget`. `result` is the
/// single DXGI query: the adapter list, or the failure rendered as text so a
/// failed query is reported rather than shown as "no adapters".
#[cfg(any(windows, test))]
fn gpu_memory_budget_json(
    pid: u32,
    result: Result<&[hipfire_config::host_memory::DxgiAdapterBudget], &str>,
) -> serde_json::Value {
    let (adapters, error) = match result {
        Ok(adapters) => (adapters, None),
        Err(error) => (&[][..], Some(error)),
    };
    let adapters = adapters
        .iter()
        .map(|adapter| {
            let mut entry = match serde_json::to_value(adapter) {
                Ok(entry) => entry,
                Err(error) => {
                    return serde_json::json!({ "serialization_error": error.to_string() })
                }
            };
            if let Some(map) = entry.as_object_mut() {
                let amd = dxgi_adapter_is_amd(adapter);
                map.insert("amd".into(), serde_json::json!(amd));
                map.insert(
                    "vendor_label".into(),
                    serde_json::json!(if amd { "AMD" } else { "non-AMD" }),
                );
                map.insert(
                    "luid_hex".into(),
                    serde_json::json!(dxgi_luid_hex(&adapter.luid)),
                );
            }
            entry
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "source": GPU_MEMORY_BUDGET_SOURCE,
        "calling_process_pid": pid,
        "adapters": adapters,
        "error": error,
    })
}

/// Text lines for the "GPU memory budget (DXGI, this process)" diag section.
#[cfg(any(windows, test))]
fn gpu_memory_budget_text(
    pid: u32,
    result: Result<&[hipfire_config::host_memory::DxgiAdapterBudget], &str>,
) -> Vec<String> {
    let mut lines = vec![
        "GPU memory budget (DXGI, this process):".to_string(),
        format!(
            "  source: {GPU_MEMORY_BUDGET_SOURCE}; calling process pid {pid}; \
             not an admission budget"
        ),
    ];
    match result {
        Err(error) => lines.push(format!("  error: {error}")),
        Ok([]) => lines
            .push("  no hardware adapters reported (software adapters are skipped)".to_string()),
        Ok(adapters) => {
            for (index, adapter) in adapters.iter().enumerate() {
                let label = if dxgi_adapter_is_amd(adapter) {
                    "AMD"
                } else {
                    "non-AMD"
                };
                lines.push(format!(
                    "  adapter {index}: {} [{label}]  vendor 0x{:04x}  device 0x{:04x}  \
                     LUID {}  flags 0x{:x}  node {}",
                    adapter.description,
                    adapter.vendor_id,
                    adapter.device_id,
                    dxgi_luid_hex(&adapter.luid),
                    adapter.flags,
                    adapter.node_index,
                ));
                lines.push(format!(
                    "    dedicated {}   shared {}",
                    dxgi_mib_text(adapter.dedicated_video_memory_bytes),
                    dxgi_mib_text(adapter.shared_system_memory_bytes),
                ));
                for (name, segment) in [
                    ("local:", &adapter.local),
                    ("non-local:", &adapter.non_local),
                ] {
                    lines.push(format!(
                        "    {name:<10} budget {}  usage {}  available-for-reservation {}  reservation {}",
                        dxgi_mib_text(segment.budget_bytes),
                        dxgi_mib_text(segment.current_usage_bytes),
                        dxgi_mib_text(segment.available_for_reservation_bytes),
                        dxgi_mib_text(segment.current_reservation_bytes),
                    ));
                }
            }
        }
    }
    lines
}

fn parse_major_minor(value: &str) -> Option<(u64, u64)> {
    let mut parts = value.split('.');
    Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
}

fn detect_gpu_arches() -> Vec<String> {
    let root = Path::new("/sys/class/kfd/kfd/topology/nodes");
    let mut arches = Vec::new();
    let Ok(nodes) = fs::read_dir(root) else {
        return arches;
    };
    for node in nodes.flatten() {
        let Ok(properties) = fs::read_to_string(node.path().join("properties")) else {
            continue;
        };
        let Some(version) = properties.lines().find_map(|line| {
            line.split_whitespace()
                .collect::<Vec<_>>()
                .as_slice()
                .strip_prefix(&["gfx_target_version"])
                .and_then(|rest| rest.first())
                .and_then(|value| value.parse::<u32>().ok())
        }) else {
            continue;
        };
        if let Some(arch) = gfx_version_to_arch(version) {
            if !arches.iter().any(|candidate| candidate == arch) {
                arches.push(arch.to_owned());
            }
        }
    }
    arches
}

fn gfx_version_to_arch(version: u32) -> Option<&'static str> {
    match version {
        90006 => Some("gfx906"),
        90008 => Some("gfx908"),
        90010 => Some("gfx90a"),
        90400..=90402 => Some("gfx94x"),
        100100 => Some("gfx1010"),
        100300 | 100302 => Some("gfx1030"),
        110000..=110002 => Some("gfx1100"),
        110500 => Some("gfx1150"),
        110501 => Some("gfx1151"),
        120000 => Some("gfx1200"),
        120001 => Some("gfx1201"),
        _ => None,
    }
}

fn detect_amd_drm_cards() -> Vec<String> {
    let mut cards = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return cards;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || !name[4..].bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let vendor = fs::read_to_string(entry.path().join("device/vendor")).unwrap_or_default();
        if vendor.trim() == "0x1002" {
            cards.push(name);
        }
    }
    cards.sort();
    cards
}

fn list_dri_nodes() -> Vec<String> {
    let Ok(entries) = fs::read_dir("/dev/dri") else {
        return Vec::new();
    };
    let mut nodes = entries
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    nodes.sort();
    nodes
}

fn command_version(command: &str, argument: &str) -> Option<String> {
    Command::new(command)
        .arg(argument)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or_default()
                .trim()
                .to_owned()
        })
        .filter(|line| !line.is_empty())
}

/// The refusal for a missing daemon binary: the build command and the install
/// path [`find_daemon_in`] probes for the daemon (its other candidates are the
/// running executable's directory and the source-tree `target/`).
pub(crate) fn daemon_not_found_error(paths: &Paths) -> anyhow::Error {
    let installed = paths
        .root
        .join("bin")
        .join(daemon_bin_names(cfg!(windows))[0]);
    anyhow!(
        "daemon binary not found; build `cargo build --release -p hipfire-daemon` (expected at \
         {}, or set HIPFIRE_DAEMON_BIN)",
        installed.display()
    )
}

pub(crate) fn find_daemon(paths: &Paths) -> Option<PathBuf> {
    if let Some(path) = env::var_os("HIPFIRE_DAEMON_BIN").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
    find_daemon_in(paths, exe_dir.as_deref(), &workspace, cfg!(windows))
}

/// Daemon binary name candidates, most preferred first.
///
/// Windows ships the daemon as `daemon.exe`; ELF platforms ship an
/// extensionless `daemon`. The bare spelling is kept as a fallback so a
/// future extensionless shim still wins. `windows` is a pure parameter
/// (mirroring `hipfire_config::rocm::tool_filename_candidates`) so the
/// policy is unit-testable on any host without process-global env.
fn daemon_bin_names(windows: bool) -> &'static [&'static str] {
    if windows {
        &["daemon.exe", "daemon"]
    } else {
        &["daemon"]
    }
}

/// Candidate lookup shared by [`find_daemon`] and its platform-shaped tests:
/// probe the directory of the running executable (`exe_dir`, so an extracted
/// binary archive finds its own `bin/daemon` from anywhere), the install root
/// (`~/.hipfire/bin/`) and the source-tree target dir (`release/`, then
/// `debug/`), in that order, for each candidate name.
fn find_daemon_in(
    paths: &Paths,
    exe_dir: Option<&std::path::Path>,
    workspace: &std::path::Path,
    windows: bool,
) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for name in daemon_bin_names(windows) {
        if let Some(dir) = exe_dir {
            candidates.push(dir.join(name));
        }
        candidates.push(paths.root.join("bin").join(name));
        candidates.push(workspace.join("release").join(name));
        candidates.push(workspace.join("debug").join(name));
    }
    candidates.into_iter().find(|path| path.is_file())
}

/// A process whose `argv[0]` is `daemon` (this install's daemon binary) and
/// whose hipfire home is `root`. Linux only: other OSes cannot read another
/// process's environment, so ownership is unprovable there.
#[cfg(target_os = "linux")]
fn is_install_daemon(pid: u32, daemon: &Path, root: &Path) -> bool {
    process_identity::process_image_matches(pid, daemon)
        && process_identity::process_hipfire_home(pid)
            .is_some_and(|home| process_identity::same_file_path(&home, root))
}

/// Explanation for daemon candidates `stop --force` refuses to terminate on an
/// OS that cannot prove a process's `HIPFIRE_HOME`: `matching` are live PIDs
/// whose image is this install's daemon binary. `None` when there are none.
#[cfg_attr(target_os = "linux", allow(dead_code))]
fn unproven_reap_notice(root: &Path, matching: &[u32], windows: bool) -> Option<String> {
    if matching.is_empty() {
        return None;
    }
    let pids = matching
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let stop = if windows {
        "stop it yourself after confirming it belongs to this install (e.g. `Stop-Process -Id <pid>`)"
    } else {
        "stop it yourself after confirming it belongs to this install"
    };
    Some(format!(
        "not terminating daemon PID(s) {pids}: the image matches this install's daemon, but this OS \
         cannot prove the process's HIPFIRE_HOME is {}, so it could belong to another install; {stop}",
        root.display()
    ))
}

/// PIDs this install may own a daemon under: the daemon pidfiles
/// (`daemon*.pid` in the home and in `$HOME/.hipfire`, where the daemon writes
/// them), the GPU-lock holders, and the children of the recorded serve PID.
/// Every candidate is still verified by [`is_install_daemon`].
fn install_daemon_candidates(paths: &Paths) -> BTreeSet<u32> {
    let mut candidates = BTreeSet::new();
    let first_pid = |raw: &str| raw.split_whitespace().next()?.parse::<u32>().ok();
    let mut pid_dirs = vec![paths.root.clone()];
    if let Some(home) = env::var_os("HOME") {
        pid_dirs.push(PathBuf::from(home).join(".hipfire"));
    }
    let lock_dirs = match env::var_os("HIPFIRE_LOCK_DIR") {
        Some(dir) => vec![PathBuf::from(dir)],
        None => vec![
            PathBuf::from("/run/lock/hipfire"),
            PathBuf::from("/tmp/hipfire-locks"),
        ],
    };
    let scan = |dir: &Path, prefix: &str, suffix: &str, out: &mut BTreeSet<u32>| {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(prefix) && name.ends_with(suffix) {
                if let Some(pid) = fs::read_to_string(entry.path())
                    .ok()
                    .and_then(|raw| first_pid(&raw))
                {
                    out.insert(pid);
                }
            }
        }
    };
    for dir in &pid_dirs {
        scan(dir, "daemon", ".pid", &mut candidates);
    }
    for dir in &lock_dirs {
        scan(dir, "gpu-", ".lock", &mut candidates);
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(serve) = fs::read_to_string(paths.root.join("serve.pid"))
            .ok()
            .and_then(|raw| parse_pid_record(&raw))
        {
            candidates.extend(process_identity::child_pids(serve.pid));
        }
    }
    candidates
}

/// `stop --force`: stop only this install's daemon processes — candidates from
/// its pidfiles/GPU locks whose `argv[0]` is this install's daemon binary and
/// whose hipfire home is this home. SIGTERM, then SIGKILL after 5 s for the
/// ones still running (re-verified first). Returns the signalled PIDs.
#[cfg(target_os = "linux")]
pub(crate) fn reap_install_daemons(paths: &Paths) -> Result<Vec<u32>> {
    let Some(daemon) = find_daemon(paths) else {
        return Ok(Vec::new());
    };
    let own = std::process::id();
    let targets = install_daemon_candidates(paths)
        .into_iter()
        .filter(|&pid| pid != own && is_install_daemon(pid, &daemon, &paths.root))
        .collect::<Vec<_>>();
    for &pid in &targets {
        // SAFETY: plain kill(2) on a verified positive PID.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline
        && !targets
            .iter()
            .all(|&pid| process_identity::process_gone(pid))
    {
        thread::sleep(Duration::from_millis(100));
    }
    for &pid in &targets {
        if !process_identity::process_gone(pid) && is_install_daemon(pid, &daemon, &paths.root) {
            eprintln!("daemon PID {pid} ignored SIGTERM for 5 s; sending SIGKILL");
            // SAFETY: as above, re-verified immediately before signalling.
            unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
        }
    }
    Ok(targets)
}

/// `stop --force` off Linux terminates nothing: a process's `HIPFIRE_HOME`
/// cannot be proven here, and an image match alone may be another install's
/// daemon. Live candidates matching this install's daemon image are reported
/// with the reason instead.
#[cfg(not(target_os = "linux"))]
pub(crate) fn reap_install_daemons(paths: &Paths) -> Result<Vec<u32>> {
    let Some(daemon) = find_daemon(paths) else {
        return Ok(Vec::new());
    };
    let own = std::process::id();
    let matching = install_daemon_candidates(paths)
        .into_iter()
        .filter(|&pid| {
            pid != own
                && process_identity::pid_alive(pid)
                && process_identity::process_image_matches(pid, &daemon)
        })
        .collect::<Vec<_>>();
    if let Some(notice) = unproven_reap_notice(&paths.root, &matching, cfg!(windows)) {
        eprintln!("{notice}");
    }
    Ok(Vec::new())
}

/// Something accepts TCP connections on `host:port`.
pub(crate) fn port_in_use(host: &str, port: u16) -> bool {
    use std::net::ToSocketAddrs;
    (host, port).to_socket_addrs().is_ok_and(|mut addrs| {
        addrs.any(|addr| {
            std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
        })
    })
}

/// Port of the running serve: the live `serve.pid` record's port (a serve
/// started with a host:port positional or `-d` on another port), else
/// `serve.port`.
fn serve_probe_port(paths: &Paths, configured: u16) -> u16 {
    fs::read_to_string(paths.root.join("serve.pid"))
        .ok()
        .and_then(|raw| parse_pid_record(&raw))
        .filter(|record| process_identity::pid_alive(record.pid))
        .and_then(|record| record.port)
        .unwrap_or(configured)
}

/// Levenshtein distance over chars.
fn edit_distance(a: &str, b: &str) -> usize {
    let b = b.chars().collect::<Vec<_>>();
    let mut row = (0..=b.len()).collect::<Vec<_>>();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != *cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

/// Up to three registry tags/aliases close to `input` (case-insensitive edit
/// distance), then a pointer to the full list. Empty for path-like input.
fn model_suggestions(registry: &RegistryV1, input: &str) -> String {
    if input.contains(std::path::MAIN_SEPARATOR) {
        return String::new();
    }
    let mut message = String::new();
    let close = close_model_tags(registry, input);
    if !close.is_empty() {
        message.push_str(&format!("\n  did you mean: {}?", close.join(", ")));
    }
    message.push_str("\n  see: hipfire list -r");
    message
}

pub(crate) fn unknown_model_error(registry: &RegistryV1, input: &str) -> anyhow::Error {
    anyhow!(
        "unknown model '{input}'{}",
        model_suggestions(registry, input)
    )
}

/// Tags and aliases close to `input` (case-insensitive edit distance), nearest
/// first, at most three.
fn close_model_tags<'r>(registry: &'r RegistryV1, input: &str) -> Vec<&'r str> {
    let needle = input.to_lowercase();
    let limit = (needle.chars().count() / 4).max(2);
    let mut close = registry
        .models
        .keys()
        .chain(registry.aliases.keys())
        .map(|tag| (edit_distance(&needle, &tag.to_lowercase()), tag.as_str()))
        .filter(|(distance, _)| *distance <= limit)
        .collect::<Vec<_>>();
    close.sort_unstable();
    close.dedup_by(|a, b| a.1 == b.1);
    close.into_iter().take(3).map(|(_, tag)| tag).collect()
}

/// Whether a model argument names a file rather than a registry tag: it has a
/// path separator or ends in a model extension.
fn looks_like_model_path(input: &str) -> bool {
    const EXTENSIONS: &[&str] = &[
        ".hfq",
        ".hf4",
        ".mq4",
        ".mq3",
        ".mq6",
        ".mq8",
        ".mq4r",
        ".mq4-xt",
        ".gguf",
        ".safetensors",
    ];
    let lower = input.to_ascii_lowercase();
    input.contains('/')
        || input.contains('\\')
        || EXTENSIONS
            .iter()
            .any(|extension| lower.ends_with(extension))
}

/// The refusal for a model argument that resolved to no local file, naming the
/// expected file and the exact next command.
///
/// - a registry (or retired) entry: the file expected in the models dir, and
///   `hipfire pull <tag>` when the entry is downloadable, else where to place it;
/// - a path: the missing path and how to name a registry tag instead;
/// - anything else: an unknown tag, with the registry's did-you-mean list and
///   the pull command for the closest tag.
pub(crate) fn model_not_found_error(
    paths: &Paths,
    registry: &RegistryV1,
    input: &str,
) -> anyhow::Error {
    let card = model_card(paths, registry, input);
    if let Some((tag, entry)) = card.pair() {
        let expected = paths.models.join(&entry.file);
        let fix = if card.pullable().is_some_and(|entry| !entry.repo.is_empty()) {
            format!("fetch it: hipfire pull {tag}")
        } else {
            format!("place {} in {}", entry.file, paths.models.display())
        };
        return anyhow!(
            "model not found: {tag} — expected {}, which is not on disk.\n  {fix}",
            expected.display()
        );
    }
    if looks_like_model_path(input) {
        return anyhow!(
            "model not found: {input} (no such file)\n  check the path, or use a registry tag: \
             hipfire list -r"
        );
    }
    let then = close_model_tags(registry, input)
        .first()
        .map(|tag| format!("\n  then: hipfire pull {tag}"))
        .unwrap_or_default();
    anyhow!(
        "model not found: {input}{}{then}",
        model_suggestions(registry, input)
    )
}

pub(crate) fn request_f64(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
    explicit: Option<f64>,
) -> Result<Option<f64>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    request_config_value(resolved, key)?
        .map(|value| config_value_f64(value, key))
        .transpose()
}

pub(crate) fn request_u64(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
    explicit: Option<u64>,
) -> Result<Option<u64>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    request_config_value(resolved, key)?
        .map(|value| config_value_u64(value, key))
        .transpose()
}

pub(crate) fn request_string(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
    explicit: Option<String>,
) -> Result<Option<String>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    request_config_value(resolved, key)?
        .map(|value| match value {
            hipfire_config::ConfigValue::String(value) => Ok(value.clone()),
            value => bail!(
                "configuration key '{key}' resolved as {}, expected string",
                value.kind()
            ),
        })
        .transpose()
}

pub(crate) fn request_config_value<'a>(
    resolved: &'a hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<Option<&'a hipfire_config::ConfigValue>> {
    let value = resolved
        .get(key)
        .ok_or_else(|| anyhow!("configuration key '{key}' is not resolved"))?;
    match &value.source {
        ConfigSource::BuiltIn => Ok(None),
        ConfigSource::GlobalUser { .. } => Ok(value
            .shadowed
            .iter()
            .rev()
            .find(|candidate| {
                matches!(
                    candidate.source,
                    ConfigSource::RegistryModel { .. } | ConfigSource::RegistryTarget { .. }
                )
            })
            .map(|candidate| &candidate.value)),
        _ => Ok(Some(&value.value)),
    }
}

pub(crate) fn config_value_f64(value: &hipfire_config::ConfigValue, key: &str) -> Result<f64> {
    match value {
        hipfire_config::ConfigValue::Float(value) => Ok(*value),
        hipfire_config::ConfigValue::Integer(value) => Ok(*value as f64),
        _ => bail!("configuration key '{key}' did not resolve to a number"),
    }
}

pub(crate) fn config_value_u64(value: &hipfire_config::ConfigValue, key: &str) -> Result<u64> {
    match value {
        hipfire_config::ConfigValue::Integer(value) => u64::try_from(*value)
            .map_err(|_| anyhow!("configuration key '{key}' cannot be negative")),
        value => bail!(
            "configuration key '{key}' resolved as {}, expected integer",
            value.kind()
        ),
    }
}

pub(crate) fn insert_optional_f64(target: &mut serde_json::Value, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        target[key] = serde_json::json!(value);
    }
}

pub(crate) fn insert_optional_u64(target: &mut serde_json::Value, key: &str, value: Option<u64>) {
    if let Some(value) = value {
        target[key] = serde_json::json!(value);
    }
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|directory| directory.join(name))
            .find(|candidate| candidate.is_file())
    })
}

fn source_label(source: &ConfigSource) -> String {
    match source {
        ConfigSource::BuiltIn => "built-in".into(),
        ConfigSource::RegistryModel { tag, revision } => {
            format!("registry model {tag}@{revision}")
        }
        ConfigSource::RegistryTarget {
            tag,
            arch,
            revision,
        } => format!("registry target {tag}/{arch}@{revision}"),
        ConfigSource::GlobalUser { path } => format!("global user ({})", path.display()),
        ConfigSource::ModelUser { model, path } => {
            format!("model user {model} ({})", path.display())
        }
        ConfigSource::LegacyEnv { name } => format!("legacy env {name}"),
        ConfigSource::OneShot { argument } => format!("one-shot {argument}"),
    }
}

fn config_rule_json(rule: ValueRule) -> serde_json::Value {
    match rule {
        ValueRule::Bool => serde_json::json!({ "type": "boolean" }),
        ValueRule::Integer { min, max } => {
            serde_json::json!({ "type": "integer", "minimum": min, "maximum": max })
        }
        ValueRule::Float {
            min,
            max,
            min_inclusive,
        } => serde_json::json!({
            "type": "number",
            "minimum": min,
            "maximum": max,
            "minimum_inclusive": min_inclusive,
        }),
        ValueRule::String => serde_json::json!({ "type": "string" }),
        ValueRule::NonEmptyString => {
            serde_json::json!({ "type": "string", "min_length": 1 })
        }
        ValueRule::Host => serde_json::json!({ "type": "string", "format": "host" }),
        ValueRule::PathOrEmpty => {
            serde_json::json!({ "type": "string", "format": "existing-path-or-empty" })
        }
        ValueRule::AbsolutePathOrEmpty => {
            serde_json::json!({ "type": "string", "format": "absolute-path-or-empty" })
        }
        ValueRule::Enum(values) => {
            serde_json::json!({ "type": "string", "enum": values })
        }
        ValueRule::KvBackend => {
            serde_json::json!({ "type": "string", "enum": ["legacy", "vmm"] })
        }
        ValueRule::AutoInteger { min, max } => serde_json::json!({
            "type": ["integer", "string"],
            "enum": ["auto"],
            "minimum": min,
            "maximum": max,
        }),
        ValueRule::AutoBool => serde_json::json!({
            "type": ["boolean", "string"],
            "enum": [true, false, "auto"],
        }),
        ValueRule::NullableString => {
            serde_json::json!({ "type": ["string", "null"] })
        }
        ValueRule::NullableEnum(values) => serde_json::json!({
            "type": ["string", "null"],
            "enum": values,
            "nullable": true,
        }),
        ValueRule::NullableInteger { min, max } => serde_json::json!({
            "type": ["integer", "null"],
            "minimum": min,
            "maximum": max,
        }),
        ValueRule::NullableIntegerOrAuto { min, max } => serde_json::json!({
            "type": ["integer", "string", "null"],
            "minimum": min,
            "maximum": max,
            "enum": ["auto"],
            "nullable": true,
        }),
        ValueRule::NullableFloat { min, max } => serde_json::json!({
            "type": ["number", "null"],
            "minimum": min,
            "maximum": max,
        }),
        ValueRule::KvAdaptive => serde_json::json!({
            "type": "string",
            "format": "kv-adaptive-policy",
        }),
        ValueRule::PassbackShare => serde_json::json!({
            "type": "string",
            "format": "auto-or-share",
        }),
        ValueRule::Deepseek4Placement => serde_json::json!({
            "type": "string",
            "format": "deepseek4-compute-placement",
        }),
    }
}

fn config_rule_label(rule: ValueRule) -> &'static str {
    match rule {
        ValueRule::Bool => "bool",
        ValueRule::Integer { .. } => "integer",
        ValueRule::Float { .. } => "number",
        ValueRule::String => "string",
        ValueRule::NonEmptyString => "nonempty-string",
        ValueRule::Host => "host",
        ValueRule::PathOrEmpty => "path-or-empty",
        ValueRule::AbsolutePathOrEmpty => "absolute-path-or-empty",
        ValueRule::Enum(_) => "enum",
        ValueRule::KvBackend => "kv-backend",
        ValueRule::AutoBool => "auto-bool",
        ValueRule::AutoInteger { .. } => "auto|integer",
        ValueRule::NullableString => "string|null",
        ValueRule::NullableEnum(_) => "enum|null",
        ValueRule::NullableInteger { .. } => "integer|null",
        ValueRule::NullableIntegerOrAuto { .. } => "integer|auto|null",
        ValueRule::NullableFloat { .. } => "number|null",
        ValueRule::KvAdaptive => "kv-adaptive",
        ValueRule::PassbackShare => "auto-or-share",
        ValueRule::Deepseek4Placement => "deepseek4-placement",
    }
}

fn config_default_value(schema: &hipfire_config::ConfigField) -> hipfire_config::ConfigValue {
    // Resolve one empty layer set so the config crate remains the only place
    // that turns the private DefaultValue representation into a public value.
    resolve(Vec::<NamedLayer>::new())
        .expect("built-in schema validates")
        .get(schema.key)
        .expect("schema key resolved")
        .value
        .clone()
}

fn format_default(schema: &hipfire_config::ConfigField) -> String {
    config_default_value(schema).to_string()
}

fn registry_source(source: RegistrySource) -> &'static str {
    match source {
        RegistrySource::Cache => "cache",
        RegistrySource::Network => "network",
        RegistrySource::StaleCache => "stale-cache",
        RegistrySource::Bundled => "bundled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--video` takes ONE clip per occurrence: its frames, comma separated,
    /// in display order. The comma is the only frame separator, so the
    /// parsing (and the frame-container check) is pinned here rather than
    /// only end-to-end.
    #[test]
    fn the_video_flag_carries_one_clip_per_occurrence() {
        let cli = Cli::try_parse_from([
            "hipfire",
            "embed",
            "model",
            "a",
            "clip",
            "of",
            "--video",
            "f0.png,f1.png",
            "--video",
            "g0.png",
        ])
        .expect("parses");
        let Some(Commands::Embed(args)) = cli.command else {
            panic!("expected the embed verb");
        };
        assert_eq!(args.text, vec!["a", "clip", "of"]);
        assert_eq!(
            args.video,
            vec!["f0.png,f1.png".to_string(), "g0.png".to_string()]
        );
        assert!(args.image.is_empty() && args.audio.is_empty());

        // No --video: the list is empty, and the flag is not implied by text.
        let cli = Cli::try_parse_from(["hipfire", "embed", "model", "text"]).expect("parses");
        let Some(Commands::Embed(args)) = cli.command else {
            panic!("expected the embed verb");
        };
        assert!(args.video.is_empty());
    }

    #[test]
    fn only_png_and_jpeg_bytes_count_as_frames() {
        assert!(is_frame_image(b"\x89PNG\r\n\x1a\n\x00"));
        assert!(is_frame_image(b"\xff\xd8\xff\xe0\x00\x10"));
        assert!(!is_frame_image(b"RIFF"));
        assert!(!is_frame_image(b"\x00\x00\x00\x18ftypmp42"));
        assert!(!is_frame_image(b"\x1aE\xdf\xa3"));
        assert!(!is_frame_image(b""));
    }
    use crate::serve::complete::{
        forward_think_fragments, include_reasoning_content, inject_default_system_message,
        normalize_openai_messages, Completion, ThinkFragment,
    };
    use hipfire_registry::{is_standalone_model, registry_sidecar_files};

    use crate::serve::{serve_instance_token, Admission, ServeMeta, ServeRuntime, ServeShared};
    use hipfire_config::CONFIG_PROFILE_NAMES;

    fn test_dxgi_adapter(
        description: &str,
        vendor_id: u32,
        luid: [u8; 8],
    ) -> hipfire_config::host_memory::DxgiAdapterBudget {
        use hipfire_config::host_memory::{DxgiAdapterBudget, SegmentBudget};
        DxgiAdapterBudget {
            description: description.to_string(),
            vendor_id,
            device_id: 0x744c,
            luid,
            dedicated_video_memory_bytes: 8 * 1024 * 1024 * 1024,
            shared_system_memory_bytes: 16 * 1024 * 1024 * 1024,
            flags: 0,
            source: GPU_MEMORY_BUDGET_SOURCE,
            node_index: 0,
            local: SegmentBudget {
                budget_bytes: 7 * 1024 * 1024 * 1024,
                current_usage_bytes: 1024 * 1024 * 1024,
                available_for_reservation_bytes: 4 * 1024 * 1024 * 1024,
                current_reservation_bytes: 512 * 1024 * 1024,
            },
            non_local: SegmentBudget {
                budget_bytes: 15 * 1024 * 1024 * 1024,
                current_usage_bytes: 2 * 1024 * 1024 * 1024,
                available_for_reservation_bytes: 8 * 1024 * 1024 * 1024,
                current_reservation_bytes: 256 * 1024 * 1024,
            },
        }
    }

    #[test]
    fn gpu_memory_budget_json_marks_amd_and_non_amd_adapters() {
        let adapters = vec![
            test_dxgi_adapter("AMD Radeon", 0x1002, [1, 0, 0, 0, 0, 0, 0, 0]),
            test_dxgi_adapter("Other GPU", 0x10de, [2, 0, 0, 0, 0, 0, 0, 0]),
        ];
        let json = gpu_memory_budget_json(4242, Ok(&adapters));
        assert_eq!(json["source"], GPU_MEMORY_BUDGET_SOURCE);
        assert_eq!(json["calling_process_pid"], 4242);
        assert!(json["error"].is_null());
        let listed = json["adapters"].as_array().unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0]["amd"], true);
        assert_eq!(listed[0]["vendor_label"], "AMD");
        assert_eq!(listed[0]["description"], "AMD Radeon");
        assert_eq!(listed[0]["luid_hex"], "0100000000000000");
        assert_eq!(
            listed[0]["local"]["budget_bytes"],
            7u64 * 1024 * 1024 * 1024
        );
        assert_eq!(
            listed[0]["non_local"]["current_reservation_bytes"],
            256u64 * 1024 * 1024
        );
        assert_eq!(listed[1]["amd"], false);
        assert_eq!(listed[1]["vendor_label"], "non-AMD");
    }

    #[test]
    fn kho_scratch_check_passes_when_none_and_warns_from_one_gib() {
        for none in [None, Some(0), Some(512 << 20)] {
            let check = kho_scratch_check(none);
            assert_eq!(check["name"], "kho_scratch");
            assert_eq!(check["status"], "pass");
        }
        assert_eq!(kho_scratch_check(None)["scratch_bytes"], 0);
        let warn = kho_scratch_check(Some(13_536 << 20));
        assert_eq!(warn["status"], "warn");
        assert_eq!(warn["scratch_bytes"], 13_536u64 << 20);
        assert_eq!(
            warn["detail"],
            "KHO scratch reserves 13.2 GiB unusable by the GPU; boot with kho=off to reclaim it"
        );
    }

    #[test]
    fn gpu_memory_budget_json_reports_error_and_empty_explicitly() {
        let failed = gpu_memory_budget_json(7, Err("CreateDXGIFactory1 failed: 0x887A0004"));
        assert_eq!(failed["error"], "CreateDXGIFactory1 failed: 0x887A0004");
        assert_eq!(failed["adapters"].as_array().unwrap().len(), 0);
        assert_eq!(failed["calling_process_pid"], 7);
        let empty = gpu_memory_budget_json(7, Ok(&[]));
        assert!(empty["error"].is_null());
        assert_eq!(empty["adapters"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn gpu_memory_budget_text_lists_every_segment_field() {
        let adapters = vec![
            test_dxgi_adapter("AMD Radeon", 0x1002, [1, 0, 0, 0, 0, 0, 0, 0]),
            test_dxgi_adapter("Other GPU", 0x10de, [2, 0, 0, 0, 0, 0, 0, 0]),
        ];
        let text = gpu_memory_budget_text(4242, Ok(&adapters)).join("\n");
        assert!(text.starts_with("GPU memory budget (DXGI, this process)"));
        assert!(text.contains("pid 4242"));
        assert!(text.contains("not an admission budget"));
        assert!(text.contains("AMD Radeon [AMD]"));
        assert!(text.contains("Other GPU [non-AMD]"));
        assert!(text.contains("dedicated 8192.0 MiB"));
        assert!(text.contains("shared 16384.0 MiB"));
        assert!(text.contains("non-local:"));
        assert!(text.contains("budget 7168.0 MiB"));
        assert!(text.contains("usage 1024.0 MiB"));
        assert!(text.contains("available-for-reservation 4096.0 MiB"));
        assert!(text.contains("reservation 512.0 MiB"));
        assert!(text.contains("budget 15360.0 MiB"));
        assert!(text.contains("reservation 256.0 MiB"));
    }

    #[test]
    fn gpu_memory_budget_text_reports_error_and_empty_explicitly() {
        let failed = gpu_memory_budget_text(7, Err("QueryVideoMemoryInfo failed")).join("\n");
        assert!(failed.starts_with("GPU memory budget (DXGI, this process)"));
        assert!(failed.contains("error: QueryVideoMemoryInfo failed"));
        let empty = gpu_memory_budget_text(7, Ok(&[])).join("\n");
        assert!(empty.contains("no hardware adapters reported"));
        assert!(empty.contains("software adapters are skipped"));
    }

    /// Points `launch_tui` at the checked-in `tests/fixtures/fake_tui.sh`,
    /// executed in place and never written or chmod-ed at runtime, and names
    /// the record file it fills. The env guards must outlive the launch.
    #[cfg(unix)]
    fn fake_tui_fixture(paths: &Paths) -> (PathBuf, EnvGuard, EnvGuard) {
        fs::create_dir_all(&paths.root).unwrap();
        let record = paths.root.join("tui-args.txt");
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_tui.sh");
        let tui = EnvGuard::set("HIPFIRE_TUI_BIN", script.to_str().unwrap());
        let args = EnvGuard::set("FAKE_TUI_ARGS", record.to_str().unwrap());
        (record, tui, args)
    }

    #[cfg(unix)]
    #[test]
    fn config_without_action_launches_the_tui_on_settings() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let paths = test_paths("config-tui-settings");
        let (record, _tui, _args) = fake_tui_fixture(&paths);
        config_command(
            &paths,
            ConfigArgs {
                model: None,
                action: None,
            },
        )
        .unwrap();
        assert_eq!(fs::read_to_string(&record).unwrap(), "--settings\n");
        let _ = fs::remove_dir_all(&paths.root);
    }

    /// The race behind the flaky ETXTBSY, made deterministic. The test used to
    /// write a fresh `<root>/fake-tui.sh`, chmod it and exec it. A concurrent
    /// test forking a child at that moment copies the still-open write
    /// descriptor into the child; `O_CLOEXEC` closes the copy only at the
    /// child's exec, so until then the inode has a live writer and exec of the
    /// script fails ETXTBSY even though this thread already closed its own fd.
    /// Renaming does not help (the writer pins the inode, not the name).
    ///
    /// Here a decoy writer is opened on that historical path, a child is
    /// forked while it is open and parked in `pre_exec` (before exec, so the
    /// copy is live), and the parent's writer is closed. The negative control
    /// then repeats the old write+chmod+launch on the same pinned inode and
    /// must fail ETXTBSY every time; the immutable fixture launched through
    /// the same helper and `config_command` call must still succeed.
    #[cfg(target_os = "linux")]
    #[test]
    fn config_tui_launch_is_unaffected_by_a_child_pinning_the_historical_script_inode() {
        use std::os::unix::fs::PermissionsExt;

        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let paths = test_paths("config-tui-pinned-writer");
        fs::create_dir_all(&paths.root).unwrap();
        let historical = paths.root.join("fake-tui.sh");
        let writer = fs::File::create(&historical).unwrap();

        let (mut ready_r, ready_w) = io::pipe().unwrap();
        let (go_r, go_w) = io::pipe().unwrap();
        let (ready_fd, go_fd, go_w_fd) = (
            std::os::fd::AsRawFd::as_raw_fd(&ready_w),
            std::os::fd::AsRawFd::as_raw_fd(&go_r),
            std::os::fd::AsRawFd::as_raw_fd(&go_w),
        );
        let mut command = Command::new("/bin/true");
        // SAFETY: the hook runs in the forked child before exec and only calls
        // the async-signal-safe `close`, `write` and `read`.
        unsafe {
            command.pre_exec(move || {
                libc::close(go_w_fd);
                libc::write(ready_fd, [1u8].as_ptr().cast(), 1);
                let mut byte = 0u8;
                libc::read(go_fd, (&mut byte as *mut u8).cast(), 1);
                Ok(())
            });
        }
        // `spawn` returns only after the child execs, so it runs on its own
        // thread, like a concurrent test starting a process.
        let spawner = std::thread::spawn(move || {
            let child = command.spawn();
            drop(ready_w);
            child
        });
        let mut byte = [0u8];
        assert_eq!(ready_r.read(&mut byte).unwrap(), 1, "child never forked");
        // The child now holds a copy of `writer`; closing ours leaves the
        // historical inode write-pinned until that child execs.
        drop(writer);

        let (record, tui, _args) = fake_tui_fixture(&paths);
        let fixed = config_command(
            &paths,
            ConfigArgs {
                model: None,
                action: None,
            },
        );
        let fixed_record = fs::read_to_string(&record);

        // Negative control: the original runtime fs::write + chmod + launch
        // at the historical destination, on the same pinned inode.
        let old_body = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\n",
            record.display()
        );
        fs::write(&historical, old_body).unwrap();
        fs::set_permissions(&historical, fs::Permissions::from_mode(0o755)).unwrap();
        let old_tui = EnvGuard::set("HIPFIRE_TUI_BIN", historical.to_str().unwrap());
        let control = config_command(
            &paths,
            ConfigArgs {
                model: None,
                action: None,
            },
        );

        // Release, join and reap the parked child before any assertion.
        drop(go_w);
        let child = spawner.join().unwrap();
        let reaped = child.and_then(|mut child| child.wait());
        drop(old_tui);
        drop(tui);
        let _ = fs::remove_dir_all(&paths.root);

        reaped.expect("parked /bin/true child failed");
        fixed.expect("immutable fixture launch must not depend on the historical inode");
        assert_eq!(fixed_record.unwrap(), "--settings\n");
        let err =
            control.expect_err("old runtime-written script must fail while its inode is pinned");
        assert_eq!(
            err.root_cause()
                .downcast_ref::<io::Error>()
                .and_then(io::Error::raw_os_error),
            Some(libc::ETXTBSY),
            "unexpected launch error: {err:#}"
        );
    }

    fn test_paths(label: &str) -> Paths {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-cli-{label}-{}-{nonce}",
            std::process::id()
        ));
        let config = ConfigPaths::under(&root);
        Paths {
            models: config.models.clone(),
            registry: RegistryPaths {
                cache: root.join("registry.cache.json"),
            },
            root,
            config,
        }
    }

    /// Run `quantize_command` against a recording stand-in for `hipfire-quantize`
    /// and return the argv it was launched with.
    #[cfg(unix)]
    fn recorded_quantizer_argv(label: &str, extra: &[&str]) -> Result<Vec<String>> {
        use std::os::unix::fs::PermissionsExt;
        let paths = test_paths(label);
        let bin = paths.root.join("bin");
        fs::create_dir_all(&bin)?;
        let record = paths.root.join("argv.txt");
        let script = bin.join("hipfire-quantize");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nwhile [ $# -gt 0 ]; do\n  if [ \"$1\" = --output ]; then : > \"$2\"; fi\n  shift\ndone\n",
                record.display()
            ),
        )?;
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755))?;
        let input = paths.root.join("model");
        fs::create_dir_all(&input)?;
        let out = paths.root.join("out.hfq");
        let mut argv = vec![
            "hipfire".to_string(),
            "quantize".into(),
            input.display().to_string(),
            "--output".into(),
            out.display().to_string(),
        ];
        argv.extend(extra.iter().map(|s| s.to_string()));
        // The script was just written: a sibling test's fork may still hold its
        // write fd, so exec can fail ETXTBSY. Bounded retry like the fake-daemon
        // spawns.
        let mut attempt = 0u64;
        loop {
            let cli = Cli::try_parse_from(&argv)?;
            let Some(Commands::Quantize(args)) = cli.command else {
                panic!("expected quantize subcommand");
            };
            match quantize_command(&paths, args) {
                Err(error)
                    if attempt < 8
                        && error
                            .root_cause()
                            .downcast_ref::<io::Error>()
                            .and_then(io::Error::raw_os_error)
                            == Some(libc::ETXTBSY) =>
                {
                    attempt += 1;
                    thread::sleep(Duration::from_millis(5 * attempt));
                }
                other => break other?,
            }
        }
        let lines = fs::read_to_string(&record)?
            .lines()
            .map(str::to_owned)
            .collect();
        let _ = fs::remove_dir_all(&paths.root);
        Ok(lines)
    }

    #[cfg(unix)]
    #[test]
    fn quantize_forwards_symmetric_and_tier_recipe_controls() {
        let argv = recorded_quantizer_argv("q-recipe", &["--mq4v2-symmetric", "--tier", "xt"])
            .expect("quantize with recipe controls");
        let tail = &argv[argv.len() - 5..];
        assert_eq!(
            tail,
            ["--format", "mq4", "--mq4v2-symmetric", "--tier", "xt"],
            "argv: {argv:?}"
        );
        assert_eq!(&argv[0], "--input");
        assert_eq!(&argv[2], "--output");

        // Without the controls the historical three-flag forwarding is unchanged.
        let plain = recorded_quantizer_argv("q-plain", &[]).expect("plain quantize");
        assert_eq!(plain.len(), 6, "argv: {plain:?}");
        assert_eq!(&plain[4], "--format");
        assert!(!plain
            .iter()
            .any(|a| a == "--mq4v2-symmetric" || a == "--tier"));
    }

    #[cfg(unix)]
    #[test]
    fn quantize_recipe_controls_require_mq4_and_a_known_tier() {
        let err = recorded_quantizer_argv("q-badfmt", &["--format", "q8", "--mq4v2-symmetric"])
            .unwrap_err();
        assert!(err.to_string().contains("--format mq4 only"), "{err:#}");
        assert!(Cli::try_parse_from(["hipfire", "quantize", "m", "--tier", "bogus"]).is_err());
    }
    #[test]
    fn run_stats_are_authoritative_and_missing_is_not_wall_rate() {
        assert_eq!(
            format_run_stats(
                Some(256),
                &serde_json::json!({
                    "decode_tok_s":123.3, "ttft_ms":42.1, "tau":7.15, "dflash":true
                })
            ),
            "[stats] tokens=256 decode=123.30 tok/s ttft=42.1 ms tau=7.15"
        );
        assert_eq!(
            format_run_stats(None, &serde_json::json!({"tok_s":99.0})),
            "[stats] tokens=n/a decode=n/a tok/s ttft=n/a ms"
        );
        assert_eq!(
            format_run_stats(
                Some(0),
                &serde_json::json!({"decode_tok_s":-1,"ttft_ms":null,"tau":"bad"})
            ),
            "[stats] tokens=0 decode=n/a tok/s ttft=n/a ms tau=n/a"
        );
        assert_eq!(
            format_run_stats(
                None,
                &serde_json::json!({
                    "decode_tok_s": f64::INFINITY, "ttft_ms": f64::NAN
                })
            ),
            "[stats] tokens=n/a decode=n/a tok/s ttft=n/a ms"
        );
        assert_eq!(
            format_run_stats(
                Some(1),
                &serde_json::json!({
                    "tau":1, "dflash":false, "mtp":false
                })
            ),
            "[stats] tokens=1 decode=n/a tok/s ttft=n/a ms"
        );
    }

    #[test]
    fn verified_migration_zero_transfer_race_and_corruption() {
        let paths = test_paths("verified-migration");
        fs::create_dir_all(&paths.models).unwrap();
        let old = paths.models.join("old");
        let new = paths.models.join("new");
        let sidecar = paths.models.join("sidecar");
        let bytes = b"tiny XTS fixture";
        let digest = format!("{:x}", Sha256::digest(bytes));
        fs::write(&old, bytes).unwrap();
        fs::write(&sidecar, b"preserved").unwrap();
        let mut joins = Vec::new();
        for _ in 0..4 {
            let (old, new, digest) = (old.clone(), new.clone(), digest.clone());
            joins.push(std::thread::spawn(move || {
                DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
                let result = migrate_verified_artifact(
                    &VerifiedCache::disabled(),
                    &old,
                    &new,
                    &digest,
                    bytes.len() as u64,
                );
                DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 0));
                result
            }));
        }
        for join in joins {
            assert_eq!(join.join().unwrap().unwrap(), Some(new.clone()));
        }
        assert!(same_installed_file(&old, &new));
        assert_eq!(fs::read(&sidecar).unwrap(), b"preserved");
        assert_eq!(
            migrate_verified_artifact(
                &VerifiedCache::disabled(),
                &old,
                &new,
                &digest,
                bytes.len() as u64
            )
            .unwrap(),
            Some(new.clone())
        );
        fs::remove_file(&new).unwrap();
        fs::write(&old, vec![b'x'; bytes.len()]).unwrap();
        assert!(migrate_verified_artifact(
            &VerifiedCache::disabled(),
            &old,
            &new,
            &digest,
            bytes.len() as u64
        )
        .is_err());
        assert!(!new.exists());
        fs::write(&old, bytes).unwrap();
        fs::write(&new, vec![b'x'; bytes.len()]).unwrap();
        assert!(migrate_verified_artifact(
            &VerifiedCache::disabled(),
            &old,
            &new,
            &digest,
            bytes.len() as u64
        )
        .is_err());
        fs::remove_dir_all(paths.root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn verified_migration_preserves_external_symlink() {
        let paths = test_paths("migration-symlink");
        fs::create_dir_all(&paths.models).unwrap();
        let external = paths.root.join("external");
        let old = paths.models.join("old");
        let new = paths.models.join("new");
        fs::write(&external, b"fixture").unwrap();
        std::os::unix::fs::symlink(&external, &old).unwrap();
        let digest = format!("{:x}", Sha256::digest(b"fixture"));
        assert_eq!(
            migrate_verified_artifact(&VerifiedCache::disabled(), &old, &new, &digest, 7).unwrap(),
            Some(new.clone())
        );
        assert!(fs::symlink_metadata(&old).unwrap().file_type().is_symlink());
        assert_eq!(fs::read(new).unwrap(), b"fixture");
        fs::remove_dir_all(paths.root).unwrap();
    }
    #[test]
    fn pinned_migration_never_selects_rtn_and_link_failure_keeps_old_path() {
        let paths = test_paths("migration-fallback");
        fs::create_dir_all(&paths.models).unwrap();
        let old = paths.models.join("old");
        fs::write(&old, b"fixture").unwrap();
        let unavailable = paths.root.join("missing-parent/canonical");
        let digest = format!("{:x}", Sha256::digest(b"fixture"));
        assert_eq!(
            migrate_verified_artifact(&VerifiedCache::disabled(), &old, &unavailable, &digest, 7)
                .unwrap(),
            Some(old.clone())
        );
        #[cfg(target_os = "linux")]
        {
            // procfs does not permit arbitrary new links even for root.
            let read_only = Path::new("/proc").join(paths.root.file_name().unwrap());
            assert_eq!(
                migrate_verified_artifact(&VerifiedCache::disabled(), &old, &read_only, &digest, 7)
                    .unwrap(),
                Some(old.clone())
            );
            assert_eq!(fs::read(&old).unwrap(), b"fixture");
        }
        let registry = hipfire_registry::bundled().unwrap();
        let (_, entry) = registry.model("qwen3.8:flash-next").unwrap();
        // The stock tag is the split base; the compiled pin grandfathers the monolithic.
        assert!(!entry.is_flash_next_xts());
        let pin = grandfather_pin(entry).expect("the stock MQ4 XTS tag is grandfathered");
        let rtn = paths.models.join("qwen3.8-flash-next.mq4");
        fs::write(&rtn, b"historical RTN").unwrap();
        assert_eq!(migrate_flash_next_xts(&paths, entry).unwrap(), None);
        assert_eq!(migrate_grandfathered(&paths, &pin).unwrap(), None);
        assert_eq!(
            resolve_model_path(&paths, &registry, "qwen3.8:flash-next").unwrap(),
            None
        );
        assert!(registry.model("qwen3.8:flash-next-rtn-asym").is_none());
        let rtn_entry = RETIRED_MODELS[0].entry();
        assert!(!rtn_entry.is_flash_next_xts());
        assert_eq!(migrate_flash_next_xts(&paths, &rtn_entry).unwrap(), None);
        fs::remove_dir_all(paths.root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn verified_migration_cross_device_uses_symlink_without_copy() {
        use std::os::unix::fs::MetadataExt;
        let paths = test_paths("migration-cross-device");
        fs::create_dir_all(&paths.models).unwrap();
        let shared = Path::new("/dev/shm");
        if !shared.is_dir()
            || fs::metadata(shared).unwrap().dev() == fs::metadata(&paths.root).unwrap().dev()
        {
            fs::remove_dir_all(paths.root).unwrap();
            return;
        }
        let old = paths.models.join("old");
        fs::write(&old, b"fixture").unwrap();
        let canonical = shared.join(paths.root.file_name().unwrap());
        let digest = format!("{:x}", Sha256::digest(b"fixture"));
        assert_eq!(
            migrate_verified_artifact(&VerifiedCache::disabled(), &old, &canonical, &digest, 7)
                .unwrap(),
            Some(canonical.clone())
        );
        assert!(fs::symlink_metadata(&canonical)
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(same_installed_file(&old, &canonical));
        fs::remove_file(canonical).unwrap();
        fs::remove_dir_all(paths.root).unwrap();
    }

    fn idle_test_meta() -> ServeMeta {
        ServeMeta {
            current_model: Some("model.hfq".to_owned()),
            loading_model: Some("model.hfq".to_owned()),
            last_activity: Instant::now() - Duration::from_secs(600),
            ..ServeMeta::new("test".to_owned())
        }
    }

    /// `offload-bench` is a raw device benchmark: it must name a live daemon and
    /// measure nothing. The predicate behind that refusal is pure filesystem +
    /// `kill(pid, 0)`, so it is testable without a GPU or a real daemon — and it is
    /// the only thing standing between a user and two engines sharing the device.
    #[test]
    fn live_daemon_pids_sees_live_pid_files_only() {
        let paths = test_paths("offload-bench-pids");
        std::fs::create_dir_all(&paths.root).unwrap();
        // Our own pid is alive; a pid above the kernel's `pid_max` cannot be.
        std::fs::write(
            paths.root.join("daemon.pid"),
            format!("{}\n", std::process::id()),
        )
        .unwrap();
        std::fs::write(
            paths.root.join("daemon-GPU-0abc_pci-0000:14:00.0.pid"),
            "999999999\n",
        )
        .unwrap();
        // Not pid files, and not a pid: ignored, never an error.
        std::fs::write(paths.root.join("daemon.pid.bak"), "1\n").unwrap();
        std::fs::write(paths.root.join("daemon-garbage.pid"), "not-a-pid\n").unwrap();
        std::fs::write(
            paths.root.join("serve.pid"),
            format!("{}\n", std::process::id()),
        )
        .unwrap();

        assert_eq!(live_daemon_pids(&paths), vec![std::process::id() as i32]);

        // The same file naming a dead process is no longer a live daemon.
        std::fs::write(paths.root.join("daemon.pid"), "999999999\n").unwrap();
        assert!(live_daemon_pids(&paths).is_empty());

        // A missing root is empty, not a panic.
        let _ = std::fs::remove_dir_all(&paths.root);
        assert!(live_daemon_pids(&paths).is_empty());
    }

    #[test]
    fn update_policy_is_linux_only_with_windows_guidance() {
        assert_eq!(update_unsupported_message("linux", "0.4.3"), None);
        let windows = update_unsupported_message("windows", "0.4.3").unwrap();
        assert!(windows.contains("hipfire 0.4.3"));
        assert!(windows.contains("no files were changed"));
        assert!(windows.contains(
            "iex (irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1)"
        ));
        let other = update_unsupported_message("macos", "0.4.3").unwrap();
        assert!(other.contains("Linux-only") && other.contains("0.4.3") && other.contains("macos"));
    }

    #[test]
    fn unproven_reap_notice_names_pids_and_refuses_to_kill() {
        let root = Path::new("/h/.hipfire");
        assert_eq!(unproven_reap_notice(root, &[], true), None);
        let notice = unproven_reap_notice(root, &[11, 22], true).unwrap();
        assert!(notice.contains("11, 22"));
        assert!(notice.contains("not terminating"));
        assert!(notice.contains("HIPFIRE_HOME"));
        assert!(notice.contains("Stop-Process"));
        let unix = unproven_reap_notice(root, &[5], false).unwrap();
        assert!(!unix.contains("Stop-Process"));
    }

    /// The recommendation a user pins: a singleton when one format was measured,
    /// otherwise the **median** of the largest measurements. The tie is the point —
    /// every format is measured over the same `buffer_mb`, so picking by
    /// `max_by_key(bytes)` silently returned whichever format came last in the
    /// matrix (q8_0), whose rate profile is not the model's.
    #[test]
    fn recommended_share_takes_the_median_of_the_largest() {
        let report = |format: &str, bytes: usize, share: f64| CalibrationReport {
            arch: "gfx1201".into(),
            format: format.into(),
            m: bytes / 2720,
            k: 5120,
            bytes,
            cpu_bytes_per_s: 4.0e10,
            gpu_bytes_per_s: 2.5e10,
            share,
            predicted_speedup: 1.5,
        };

        assert!(recommended_share(&[]).is_none());

        // One pinned format: itself, named.
        let one = vec![report("mq4g256", 201_328_960, 0.375)];
        assert_eq!(
            recommended_share(&one),
            Some(("mq4g256".to_string(), 0.375))
        );

        // A tie on size takes the median share of the tied set, and says so.
        let tied = vec![
            report("a", 100, 0.10),
            report("b", 100, 0.20),
            report("c", 100, 0.50),
            report("small", 50, 0.99),
        ];
        assert_eq!(
            recommended_share(&tied),
            Some(("the 3 largest measured formats".to_string(), 0.20))
        );

        // A strictly larger entry wins outright, wherever it sits in the matrix.
        let mixed = vec![report("small", 50, 0.99), report("big", 200, 0.30)];
        assert_eq!(recommended_share(&mixed), Some(("big".to_string(), 0.30)));
    }

    /// A minimal real HFQ container carrying `arch_id`, so the discovery
    /// predicate's header probe runs against the format it will meet on disk
    /// rather than a stub.
    fn write_hfq_fixture(path: &Path, arch_id: u32) {
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
        write_hfqm_package_mem(
            path,
            arch_id,
            "{}",
            &[HfqMemTensor {
                name: "model.embed_tokens.weight".to_owned(),
                quant_type: 1,
                shape: vec![4, 4],
                group_size: 0,
                data: vec![0u8; 32],
            }],
        )
        .unwrap();
    }

    /// The two surfaces deliberately disagree, and that is the point:
    /// `hipfire list` shows what is on disk; `/v1/models` shows what can serve.
    #[test]
    fn local_listing_keeps_sidecars_that_serve_discovery_drops() {
        let paths = test_paths("listing-sidecars");
        fs::create_dir_all(&paths.models).unwrap();
        for file in [
            "qwen3.8-27b.mq3-xt",
            "qwen3.8-27b-vision.hfq",
            "qwen38-27b-dflash-mq3.hfq",
            "qwen3.6-27b.mq4",
        ] {
            fs::write(paths.models.join(file), b"fixture").unwrap();
        }
        // Unregistered artifacts, so the header probe decides: a local trunk
        // stays advertised, a local draft does not.
        write_hfq_fixture(&paths.models.join("local-trunk.mq4"), 5);
        write_hfq_fixture(&paths.models.join("qwen3-8b-dflash.hfq"), 20);
        let registry = hipfire_registry::bundled().unwrap();

        let models = list_local_models(&paths, &registry).unwrap();
        let listed: Vec<String> = models.iter().map(|model| model.name.clone()).collect();
        // Local listing: everything on disk is visible and rm-able.
        for file in [
            "qwen3.8-27b.mq3-xt",
            "qwen3.8-27b-vision.hfq",
            "qwen38-27b-dflash-mq3.hfq",
            "qwen3.6-27b.mq4",
            "local-trunk.mq4",
            "qwen3-8b-dflash.hfq",
        ] {
            assert!(
                listed.contains(&file.to_owned()),
                "{file} missing: {listed:?}"
            );
        }

        // Serve discovery: the XT tier survives, sidecars do not.
        let served: Vec<String> = models
            .iter()
            .filter(|model| {
                is_standalone_model(&model.path, &model.name, &registry, &grandfather_table())
            })
            .map(|model| model.name.clone())
            .collect();
        assert!(
            served.contains(&"qwen3.8-27b.mq3-xt".to_owned()),
            "{served:?}"
        );
        assert!(served.contains(&"qwen3.6-27b.mq4".to_owned()), "{served:?}");
        assert!(
            served.contains(&"local-trunk.mq4".to_owned()),
            "a local trunk must stay advertised: {served:?}"
        );
        assert!(
            !served.contains(&"qwen3.8-27b-vision.hfq".to_owned()),
            "tower sidecar must not be offered as a model: {served:?}"
        );
        assert!(
            !served.contains(&"qwen38-27b-dflash-mq3.hfq".to_owned()),
            "DFlash draft must not be offered as a model: {served:?}"
        );
        assert!(
            !served.contains(&"qwen3-8b-dflash.hfq".to_owned()),
            "an unregistered DFlash draft must not be offered as a model: {served:?}"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// The Ornith artifacts shipped briefly as `ornith1.5-*` before being
    /// renamed to `ornith-1.5-*`. Anyone who downloaded during that window has
    /// the old filename on disk, and the registry now points at the new one —
    /// so the canonical tag must still find their file rather than silently
    /// re-downloading 19 GB.
    #[test]
    fn separator_spellings_find_an_already_downloaded_file() {
        let paths = test_paths("separator-spellings");
        let legacy = paths.models.join("ornith1.5-35b-a3b.mq4");
        fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        fs::write(&legacy, b"fixture").unwrap();
        let registry = hipfire_registry::bundled().unwrap();

        // The new canonical spelling reaches the old file only via the
        // separator-insensitive fallback — this is the case the rename broke.
        assert_eq!(
            find_model_path(&paths, &registry, "ornith-1.5:35b-a3b"),
            Some(legacy.clone()),
            "hyphenated tag must find the unhyphenated file"
        );
        // The old spelling already worked through the literal pass; pin it so
        // the fallback cannot regress it.
        assert_eq!(
            find_model_path(&paths, &registry, "ornith1.5:35b-a3b"),
            Some(legacy.clone()),
            "legacy tag must keep working"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// The fallback must not outrank a literal hit. With both spellings present
    /// on disk, each input resolves to its own file, not to whichever the
    /// looser comparison happened to reach first.
    #[test]
    fn literal_match_wins_over_the_separator_fallback() {
        let paths = test_paths("separator-precedence");
        let legacy = paths.models.join("ornith1.5-35b-a3b.mq4");
        let renamed = paths.models.join("ornith-1.5-35b-a3b.mq4");
        fs::create_dir_all(paths.models.as_path()).unwrap();
        fs::write(&legacy, b"fixture").unwrap();
        fs::write(&renamed, b"fixture").unwrap();
        let registry = hipfire_registry::bundled().unwrap();

        assert_eq!(
            find_model_path(&paths, &registry, "ornith-1.5:35b-a3b"),
            Some(renamed),
            "exact spelling must win when it exists"
        );
        assert_eq!(
            find_model_path(&paths, &registry, "ornith1.5:35b-a3b"),
            Some(legacy),
            "exact spelling must win when it exists"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// An input that normalizes to the empty string must not match everything.
    #[test]
    fn separator_only_input_matches_nothing() {
        let paths = test_paths("separator-empty");
        let model = paths.models.join("qwen3.6-35b-a3b.mq4");
        fs::create_dir_all(model.parent().unwrap()).unwrap();
        fs::write(&model, b"fixture").unwrap();
        let registry = hipfire_registry::bundled().unwrap();

        assert_eq!(find_model_path(&paths, &registry, "-.-"), None);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// A tag whose canonical file is a dangling symlink must not resolve to a
    /// sibling that merely shares its prefix (`qwen3.8-27b.mq3-xt`/`-pro` for
    /// `qwen3.8:27b-mq3`): it is absent, so callers pull the real file.
    #[cfg(unix)]
    #[test]
    fn missing_or_dangling_tag_file_never_resolves_to_a_sibling_tier() {
        let paths = test_paths("dangling-tag-sibling");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = hipfire_registry::bundled().unwrap();
        let (_, entry) = registry.model("qwen3.8:27b-mq3").unwrap();
        assert_eq!(entry.file, "qwen3.8-27b.mq3");
        for sibling in ["qwen3.8-27b.mq3-xt", "qwen3.8-27b.mq3-pro"] {
            fs::write(paths.models.join(sibling), b"sibling").unwrap();
        }
        let canonical = paths.models.join(&entry.file);

        // Canonical file absent.
        assert_eq!(find_model_path(&paths, &registry, "qwen3.8:27b-mq3"), None);
        // Canonical file a dangling symlink.
        std::os::unix::fs::symlink(paths.root.join("gone.hfq"), &canonical).unwrap();
        assert_eq!(find_model_path(&paths, &registry, "qwen3.8:27b-mq3"), None);
        assert_eq!(
            resolve_model_path(&paths, &registry, "qwen3.8:27b-mq3").unwrap(),
            None
        );
        // The siblings still resolve under their own tags.
        assert_eq!(
            find_model_path(&paths, &registry, "qwen3.8:27b-mq3-xt"),
            Some(paths.models.join("qwen3.8-27b.mq3-xt"))
        );
        // A real canonical file wins again.
        fs::remove_file(&canonical).unwrap();
        fs::write(&canonical, b"canonical").unwrap();
        assert_eq!(
            find_model_path(&paths, &registry, "qwen3.8:27b-mq3"),
            Some(canonical)
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn find_model_path_keeps_the_selected_symlink_for_sidecar_lookup() {
        // The PLE loader probes `<given>.ple-*` first and the canonical target
        // second; a canonicalized path would lose the sidecars published beside
        // the link.
        let paths = test_paths("find-model-symlink");
        let store = paths.root.join("store");
        fs::create_dir_all(&store).unwrap();
        fs::create_dir_all(&paths.models).unwrap();
        let target = store.join("flash-next-base.mq4-xts");
        fs::write(&target, b"fixture").unwrap();
        let link = paths.models.join("flash-next-base.mq4-xts");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let registry = hipfire_registry::bundled().unwrap();

        let found = find_model_path(&paths, &registry, link.to_str().unwrap()).unwrap();
        assert_eq!(found, link);
        assert_ne!(found, fs::canonicalize(&link).unwrap());
        // A plain file is returned as given (already absolute and canonical).
        let found = find_model_path(&paths, &registry, target.to_str().unwrap()).unwrap();
        assert_eq!(found, target);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn mtp_head_is_found_beside_the_selected_link_then_beside_its_canonical_target() {
        let paths = test_paths("mtp-link-then-target");
        let store = paths.root.join("store");
        fs::create_dir_all(&store).unwrap();
        fs::create_dir_all(&paths.models).unwrap();
        let target = store.join("trunk.mq4");
        fs::write(&target, b"trunk").unwrap();
        let link = paths.models.join("trunk.mq4");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let resolve = || {
            let mut params = serde_json::json!({"mtp_mode": "auto"});
            resolve_mtp_sidecar(&mut params, None, &paths.models, &link, "", None);
            params["mtp"].as_str().map(PathBuf::from)
        };

        assert_eq!(resolve(), None);
        fs::write(target.with_extension("mtp"), b"head").unwrap();
        assert_eq!(resolve(), Some(target.with_extension("mtp")));
        fs::write(link.with_extension("mtp"), b"head").unwrap();
        assert_eq!(resolve(), Some(link.with_extension("mtp")));
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn nested_model_discovery_matches_native_registry_layout() {
        let paths = test_paths("nested-models");
        let nested = paths.models.join("community").join("example-model.mq4r");
        fs::create_dir_all(nested.parent().unwrap()).unwrap();
        fs::write(&nested, b"fixture").unwrap();
        let registry = hipfire_registry::bundled().unwrap();

        assert_eq!(
            find_model_path(&paths, &registry, "example-model"),
            Some(fs::canonicalize(&nested).unwrap())
        );
        assert!(list_local_models(&paths, &registry)
            .unwrap()
            .iter()
            .any(|model| model.path == fs::canonicalize(&nested).unwrap()));
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn daemon_discovery_prefers_windows_exe_spelling() {
        // Windows-shaped policy (runs on any host, like the rocm.rs HIPCC
        // suffix tests): daemon.exe is probed before the bare name so an
        // install or source-tree build is found on Windows.
        assert_eq!(daemon_bin_names(true), &["daemon.exe", "daemon"]);
        assert_eq!(daemon_bin_names(false), &["daemon"]);
    }

    #[test]
    fn find_daemon_discovers_daemon_exe_under_windows_shaped_policy() {
        // Only the .exe spelling exists — exactly the Windows install layout.
        let paths = test_paths("daemon-exe");
        let bin = paths.root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("daemon.exe"), b"").unwrap();
        let workspace = paths.root.join("target");
        fs::create_dir_all(&workspace).unwrap();
        assert_eq!(
            find_daemon_in(&paths, None, &workspace, true),
            Some(bin.join("daemon.exe"))
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn find_daemon_windows_policy_accepts_extensionless_shim() {
        // The bare spelling stays a fallback on Windows for a future shim.
        let paths = test_paths("daemon-shim");
        let bin = paths.root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("daemon"), b"").unwrap();
        let workspace = paths.root.join("target");
        fs::create_dir_all(&workspace).unwrap();
        assert_eq!(
            find_daemon_in(&paths, None, &workspace, true),
            Some(bin.join("daemon"))
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn find_daemon_prefers_install_dir_over_source_tree() {
        // Install root (~/.hipfire/bin) wins over the source-tree target dir
        // even when both carry a candidate (real host: Windows + dev build).
        let paths = test_paths("daemon-install-vs-target");
        let bin = paths.root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("daemon.exe"), b"install").unwrap();
        let workspace = paths.root.join("target");
        fs::create_dir_all(workspace.join("release")).unwrap();
        fs::write(workspace.join("release").join("daemon.exe"), b"dev").unwrap();
        assert_eq!(
            find_daemon_in(&paths, None, &workspace, true),
            Some(bin.join("daemon.exe"))
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn find_daemon_falls_back_to_bare_spelling_for_unix_shaped_policy() {
        // Unix-shaped policy: only the extensionless daemon is probed.
        let paths = test_paths("daemon-bare");
        let release = paths.root.join("target").join("release");
        fs::create_dir_all(&release).unwrap();
        fs::write(release.join("daemon"), b"").unwrap();
        let workspace = paths.root.join("target");
        assert_eq!(
            find_daemon_in(&paths, None, &workspace, false),
            Some(release.join("daemon"))
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn find_daemon_prefers_daemon_beside_the_executable_over_hipfire_home() {
        // A prebuilt archive (bin/hipfire + bin/daemon) works from any
        // directory: the sibling daemon wins over ~/.hipfire/bin and the
        // source tree, and is found under either spelling on Windows.
        let paths = test_paths("daemon-beside-exe");
        let bin = paths.root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("daemon"), b"install").unwrap();
        let archive = paths.root.join("archive").join("bin");
        fs::create_dir_all(&archive).unwrap();
        fs::write(archive.join("daemon"), b"beside").unwrap();
        fs::write(archive.join("daemon.exe"), b"beside").unwrap();
        let workspace = paths.root.join("target");
        fs::create_dir_all(&workspace).unwrap();
        assert_eq!(
            find_daemon_in(&paths, Some(&archive), &workspace, false),
            Some(archive.join("daemon"))
        );
        assert_eq!(
            find_daemon_in(&paths, Some(&archive), &workspace, true),
            Some(archive.join("daemon.exe"))
        );
        // No sibling daemon: falls through to the hipfire home.
        let empty = paths.root.join("elsewhere");
        fs::create_dir_all(&empty).unwrap();
        assert_eq!(
            find_daemon_in(&paths, Some(&empty), &workspace, false),
            Some(bin.join("daemon"))
        );
        assert_eq!(
            find_daemon_in(&paths, None, &workspace, false),
            Some(bin.join("daemon"))
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn cask_triattn_and_pflash_remain_opt_in_at_load() {
        let paths = test_paths("experimental-defaults");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = hipfire_registry::bundled().unwrap();
        let entry = registry
            .models
            .values()
            .find(|entry| entry.triattn.is_some())
            .expect("bundled registry should retain a TriAttention sidecar");
        let model_path = paths.models.join(&entry.file);
        fs::write(&model_path, b"model").unwrap();
        let triattn = entry.triattn.as_ref().unwrap();
        let sidecar_path = paths.models.join(&triattn.file);
        fs::write(&sidecar_path, b"sidecar").unwrap();

        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let params = load_params(
            &defaults,
            Some(entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["cask"], false);
        assert_eq!(params["cask_handoff_tokens"], 0);
        assert_eq!(params["cask_sidecar"], "");
        assert_eq!(params["prefill_compression"], "off");

        let mut explicit = ConfigLayer::default();
        explicit.set_cli("memory.cask.auto_attach", "true").unwrap();
        let enabled = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "memory.cask.auto_attach=true".into(),
            },
            layer: explicit,
        }])
        .unwrap();
        let params = load_params(
            &enabled,
            Some(entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["cask"], false);
        assert_eq!(params["cask_sidecar"], sidecar_path.display().to_string());
        assert_eq!(params["prefill_compression"], "off");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    fn ngram_draft_params_for(raw: &str) -> serde_json::Value {
        let mut layer = ConfigLayer::default();
        layer.set_cli("speculation.ngram", raw).unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::LegacyEnv {
                name: "HIPFIRE_NGRAM_DRAFT".into(),
            },
            layer,
        }])
        .unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        load_params(
            &resolved,
            None,
            model_path.parent().unwrap(),
            &model_path,
            64,
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap()
    }

    #[test]
    fn ngram_draft_one_and_on_arm_zero_and_off_disarm() {
        // HIPFIRE_NGRAM_DRAFT=1 is preserved as the string "1"; it once
        // lowered to `ngram_draft = false`, silently disabling the speculator.
        for raw in ["1", "on", "true"] {
            assert_eq!(ngram_draft_params_for(raw)["ngram_draft"], true, "{raw}");
        }
        for raw in ["0", "off", "false"] {
            assert_eq!(ngram_draft_params_for(raw)["ngram_draft"], false, "{raw}");
        }
        // `auto` keeps arming where the loader has an n-gram speculator.
        assert_eq!(ngram_draft_params_for("auto")["ngram_draft"], true);
        // The built-in default stays off.
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let params = load_params(
            &defaults,
            None,
            model_path.parent().unwrap(),
            &model_path,
            64,
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["ngram_draft"], false);
    }

    #[test]
    fn ngram_draft_param_accepts_true_false_and_rejects_unknown_values() {
        assert!(ngram_draft_param("true").unwrap());
        assert!(!ngram_draft_param("false").unwrap());
        for bad in ["", "maybe", "yes", "ON", "2", "enabled"] {
            let error = ngram_draft_param(bad).unwrap_err().to_string();
            assert!(error.contains("speculation.ngram"), "{bad:?}: {error}");
            assert!(error.contains(&format!("{bad:?}")), "{bad:?}: {error}");
        }
        // The schema agrees: true/false are accepted, other spellings refused.
        for raw in ["true", "false"] {
            let mut layer = ConfigLayer::default();
            layer.set_cli("speculation.ngram", raw).unwrap();
        }
        let mut layer = ConfigLayer::default();
        assert!(layer.set_cli("speculation.ngram", "maybe").is_err());
    }

    #[test]
    pub(crate) fn load_params_forwards_explicit_vmm_backend() {
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let params = load_params(
            &defaults,
            None,
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            Some("vmm"),
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["kv_backend"], "vmm");
    }

    #[test]
    pub(crate) fn load_params_omits_automatic_kv_backend_for_path() {
        // Absolute path with only schema/built-in backend/max_seq is automatic:
        // omit both so admission can pick VMM and the daemon can derive max_seq.
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let params = load_params(
            &defaults,
            None,
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            params.get("kv_backend").is_none(),
            "absolute path automatic backend must omit kv_backend"
        );
        assert!(
            params.get("max_seq").is_none(),
            "absolute path automatic max_seq must be omitted for runtime default"
        );
    }

    #[test]
    pub(crate) fn resolved_for_model_applies_qwen_tag_policy_and_excludes_original_and_sidecars() {
        let paths = test_paths("registry-qwen-tag-policy");
        fs::create_dir_all(&paths.root).unwrap();
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.5:4b":{"repo":"x","file":"qwen3.5-4b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"},
                "qwen3.6:35b-a3b":{"repo":"x","file":"qwen3.6-35b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"},
                "qwen3.8:27b-fast":{"repo":"x","file":"qwen3.8-27b.mq4r","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"},
                "qwen3:8b":{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"},
                "qwen3.5:9b-draft":{"repo":"x","file":"qwen35-9b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.6:27b-dflash":{"repo":"x","file":"qwen36-27b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}
            },
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();

        // Exact Qwen families keep max_tokens only; registry no longer pins
        // max_seq or kv_backend. load_params omits automatic max_seq/backend.
        for tag in [
            "qwen3.5:4b",
            "qwen3.6:35b-a3b",
            "qwen3.8:27b",
            "qwen3.8:27b-fast",
        ] {
            let (_, entry) = registry.model(tag).unwrap();
            let resolved = resolved_for_model(&paths, tag, Some(tag), Some(entry)).unwrap();
            let direct = hipfire_registry::config_layer_for_tag(tag, entry).unwrap();
            assert!(
                direct.get("memory.kv_backend").is_none(),
                "{tag} registry layer must not write kv_backend"
            );
            assert!(
                direct.get("memory.max_seq").is_none(),
                "{tag} registry layer must not write max_seq"
            );
            assert_eq!(
                config_u64(&resolved, "memory.max_seq").unwrap(),
                32768,
                "{tag} resolved max_seq is schema built-in only"
            );
            assert_eq!(
                config_u64(&resolved, "generation.max_tokens").unwrap(),
                81920,
                "{tag}"
            );
            let model_path = PathBuf::from("/tmp/test-model.mq4");
            let params = load_params(
                &resolved,
                Some(entry),
                &model_path.parent().unwrap(),
                &model_path,
                64,
                Some("q8"),
                None,
                None,
                false,
                None,
            )
            .unwrap();
            assert!(
                params.get("kv_backend").is_none(),
                "{tag} load_params must omit automatic kv_backend"
            );
            assert!(
                params.get("max_seq").is_none(),
                "{tag} load_params must omit automatic max_seq"
            );
        }

        // Original qwen3:* has no tag policy — backend/max_seq stay automatic.
        let (_, entry) = registry.model("qwen3:8b").unwrap();
        let resolved =
            resolved_for_model(&paths, "qwen3:8b", Some("qwen3:8b"), Some(entry)).unwrap();
        assert_eq!(config_u64(&resolved, "memory.max_seq").unwrap(), 32768);
        assert_eq!(
            config_u64(&resolved, "generation.max_tokens").unwrap(),
            4096
        );
        // More directly, check the helper layer itself has no policy.
        let direct = hipfire_registry::config_layer_for_tag("qwen3:8b", entry).unwrap();
        assert!(direct.get("memory.kv_backend").is_none());
        assert!(direct.get("memory.max_seq").is_none());
        assert!(direct.get("generation.max_tokens").is_none());
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let params = load_params(
            &resolved,
            Some(entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            params.get("kv_backend").is_none(),
            "original qwen3 automatic backend must omit kv_backend"
        );
        assert!(
            params.get("max_seq").is_none(),
            "original qwen3 automatic max_seq must be omitted"
        );

        // Draft/dflash sidecars do not get the Qwen policy even though family matches.
        for tag in ["qwen3.5:9b-draft", "qwen3.6:27b-dflash"] {
            let (_, entry) = registry.model(tag).unwrap();
            let direct = hipfire_registry::config_layer_for_tag(tag, entry).unwrap();
            assert!(
                direct.get("memory.kv_backend").is_none(),
                "{tag} sidecar must not get vmm"
            );
            assert!(direct.get("memory.max_seq").is_none());
            assert!(direct.get("generation.max_tokens").is_none());
        }
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn resolved_for_model_applies_glimmer_and_deepseek_targets() {
        let paths = test_paths("registry-glimmer-deepseek-tag-policy");
        fs::create_dir_all(&paths.root).unwrap();
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:fast":{"repo":"x","file":"muse-glimmer-30b.mq4r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:draft":{"repo":"x","file":"muse-glimmer-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash":{"repo":"x","file":"deepseek-v4-flash-0731.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:mq2lloyd":{"repo":"x","file":"deepseek-v4-flash-0731.mq2lloyd","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash-preview":{"repo":"x","file":"deepseek-v4-flash-preview.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:draft":{"repo":"x","file":"deepseek-v4-flash-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "other:model":{"repo":"x","file":"other.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}
            },
            "aliases":{
                "deepseek4":"deepseek-v4-flash",
                "ds4":"deepseek-v4-flash",
                "deepseek4:preview":"deepseek-v4-flash-preview",
                "muse-glimmer:quality":"muse-glimmer"
            }
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();

        // Muse Glimmer tags no longer pin max_seq; backend remains automatic.
        for tag in ["muse-glimmer", "muse-glimmer:fast"] {
            let (_, entry) = registry.model(tag).unwrap();
            let resolved = resolved_for_model(&paths, tag, Some(tag), Some(entry)).unwrap();
            assert_eq!(
                config_string(&resolved, "memory.kv_backend").unwrap(),
                "vmm",
                "{tag}"
            );
            assert_eq!(
                config_u64(&resolved, "memory.max_seq").unwrap(),
                32768,
                "{tag} resolved max_seq is schema built-in only"
            );
            let direct = hipfire_registry::config_layer_for_tag(tag, entry).unwrap();
            assert!(direct.get("memory.kv_backend").is_none());
            assert!(
                direct.get("memory.max_seq").is_none(),
                "{tag} registry must not write max_seq"
            );
            assert!(
                direct.get("generation.max_tokens").is_none(),
                "{tag} must not get max_tokens"
            );
            let model_path = PathBuf::from("/tmp/test-model.mq4");
            let params = load_params(
                &resolved,
                Some(entry),
                &model_path.parent().unwrap(),
                &model_path,
                64,
                None,
                None,
                None,
                false,
                None,
            )
            .unwrap();
            assert!(
                params.get("max_seq").is_none(),
                "{tag} load_params must omit automatic max_seq"
            );
        }
        // quality alias lands on trunk (still no max_seq policy).
        let (resolved_tag, entry) = registry.model("muse-glimmer:quality").unwrap();
        assert_eq!(resolved_tag, "muse-glimmer");
        let direct = hipfire_registry::config_layer_for_tag(resolved_tag, entry).unwrap();
        assert!(direct.get("memory.max_seq").is_none());

        // Muse Glimmer draft receives none.
        let (_, entry) = registry.model("muse-glimmer:draft").unwrap();
        let direct = hipfire_registry::config_layer_for_tag("muse-glimmer:draft", entry).unwrap();
        assert!(direct.get("memory.kv_backend").is_none());
        assert!(direct.get("memory.max_seq").is_none());
        assert!(direct.get("generation.max_tokens").is_none());
        let resolved = resolved_for_model(
            &paths,
            "muse-glimmer:draft",
            Some("muse-glimmer:draft"),
            Some(entry),
        )
        .unwrap();
        assert_eq!(
            config_string(&resolved, "memory.kv_backend").unwrap(),
            "vmm"
        );

        // DeepSeek tags keep generation.max_tokens only; no max_seq/backend pin.
        for tag in [
            "deepseek-v4-flash",
            "deepseek-v4-flash:mq2lloyd",
            "deepseek-v4-flash-preview",
        ] {
            let (resolved_tag, entry) = registry.model(tag).unwrap();
            let resolved =
                resolved_for_model(&paths, resolved_tag, Some(resolved_tag), Some(entry)).unwrap();
            assert_eq!(
                config_string(&resolved, "memory.kv_backend").unwrap(),
                "vmm",
                "{tag}"
            );
            assert_eq!(
                config_u64(&resolved, "memory.max_seq").unwrap(),
                32768,
                "{tag} resolved max_seq is schema built-in only"
            );
            assert_eq!(
                config_u64(&resolved, "generation.max_tokens").unwrap(),
                393216,
                "{tag}"
            );
            let direct = hipfire_registry::config_layer_for_tag(resolved_tag, entry).unwrap();
            assert!(direct.get("memory.kv_backend").is_none());
            assert!(
                direct.get("memory.max_seq").is_none(),
                "{tag} registry must not write max_seq"
            );
            assert_eq!(
                direct.get("generation.max_tokens"),
                Some(&hipfire_config::ConfigValue::Integer(393216))
            );
            let model_path = PathBuf::from("/tmp/test-model.mq4");
            let params = load_params(
                &resolved,
                Some(entry),
                &model_path.parent().unwrap(),
                &model_path,
                64,
                None,
                None,
                None,
                false,
                None,
            )
            .unwrap();
            assert!(
                params.get("max_seq").is_none(),
                "{tag} load_params must omit automatic max_seq"
            );
        }
        for alias in ["deepseek4", "ds4", "deepseek4:preview"] {
            let (resolved_tag, entry) = registry.model(alias).unwrap();
            let direct = hipfire_registry::config_layer_for_tag(resolved_tag, entry).unwrap();
            assert!(
                direct.get("memory.max_seq").is_none(),
                "{alias}->{resolved_tag}"
            );
            assert_eq!(
                direct.get("generation.max_tokens"),
                Some(&hipfire_config::ConfigValue::Integer(393216)),
                "{alias}->{resolved_tag}"
            );
        }
        // DeepSeek draft sidecar receives none.
        let (_, entry) = registry.model("deepseek-v4-flash:draft").unwrap();
        let direct =
            hipfire_registry::config_layer_for_tag("deepseek-v4-flash:draft", entry).unwrap();
        assert!(direct.get("memory.kv_backend").is_none());
        assert!(direct.get("memory.max_seq").is_none());
        assert!(direct.get("generation.max_tokens").is_none());

        // Absent policy: unrelated model gets no automatic policy.
        let (_, entry) = registry.model("other:model").unwrap();
        let direct = hipfire_registry::config_layer_for_tag("other:model", entry).unwrap();
        assert!(direct.get("memory.kv_backend").is_none());
        assert!(direct.get("memory.max_seq").is_none());
        assert!(direct.get("generation.max_tokens").is_none());

        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// Drives `config <model> {list,set,reset}` against a fixture home and
    /// records stdout plus the `models.toml` bytes after each step.
    fn model_config_golden_transcript() -> String {
        let paths = test_paths("model-config-golden");
        fs::create_dir_all(&paths.root).unwrap();
        fs::write(
            &paths.config.config_toml,
            "[generation]\nmax_tokens = 1234\n",
        )
        .unwrap();
        let registry = RegistryV1::parse(
            r#"{
                "schema_version":1,
                "generated_at":"2026-09-01T00:00:00Z",
                "models":{
                    "qwen3:8b":{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"}
                },
                "aliases":{}
            }"#,
            "test",
        )
        .unwrap();
        let root = paths.root.display().to_string();
        let mut transcript = String::new();
        let mut step = |label: &str, model: &str, action: ConfigAction| {
            let mut out = Vec::new();
            let result = model_config_command_to(&paths, &registry, model, Some(action), &mut out);
            transcript.push_str(&format!("== {label}\n"));
            transcript.push_str(&match result {
                Ok(()) if label.starts_with("list") => {
                    // The full list tracks the whole schema; pin the header and
                    // the rows these steps change or inherit, each under its
                    // group title.
                    const PINNED: [&str; 4] = [
                        "generation.max_tokens",
                        "generation.temperature",
                        "memory.max_seq",
                        "reasoning.effort",
                    ];
                    let titles = hipfire_config::intent::Intent::ALL.map(|group| group.title());
                    let mut kept = String::new();
                    let mut title: Option<&str> = None;
                    let text = String::from_utf8(out).unwrap();
                    for line in text.lines() {
                        if titles.contains(&line) {
                            title = Some(line);
                        } else if line.starts_with("  ") {
                            let key = line.split_whitespace().next().unwrap_or("");
                            if PINNED.contains(&key) {
                                if let Some(title) = title.take() {
                                    kept.push_str(&format!("\n{title}\n"));
                                }
                                kept.push_str(line);
                                kept.push('\n');
                            }
                        } else if !line.is_empty() {
                            kept.push_str(line);
                            kept.push('\n');
                        }
                    }
                    kept
                }
                Ok(()) => String::from_utf8(out).unwrap(),
                Err(error) => format!("error: {error}\n"),
            });
            transcript.push_str("-- models.toml\n");
            transcript.push_str(&fs::read_to_string(&paths.config.models_toml).unwrap_or_default());
        };
        let list = || ConfigAction::List(ConfigListArgs::default());
        step("list", "qwen3:8b", list());
        step(
            "set reasoning.effort high",
            "qwen3:8b",
            ConfigAction::Set {
                key: "reasoning.effort".into(),
                value: "high".into(),
            },
        );
        step("list after set", "qwen3:8b", list());
        step(
            "set alias key",
            "qwen3:8b",
            ConfigAction::Set {
                key: "max_seq".into(),
                value: "4096".into(),
            },
        );
        step(
            "set local model",
            "local-thing.mq4",
            ConfigAction::Set {
                key: "generation.temperature".into(),
                value: "0.5".into(),
            },
        );
        step(
            "set developer key",
            "qwen3:8b",
            ConfigAction::Set {
                key: "developer.qwen4_moe_sym_iu4".into(),
                value: "0".into(),
            },
        );
        step(
            "set process key",
            "qwen3:8b",
            ConfigAction::Set {
                key: "kernel.iu4_prefill".into(),
                value: "0".into(),
            },
        );
        step(
            "reset reasoning.effort",
            "qwen3:8b",
            ConfigAction::Reset {
                key: Some("reasoning.effort".into()),
            },
        );
        step(
            "reset reasoning.effort again",
            "qwen3:8b",
            ConfigAction::Reset {
                key: Some("reasoning.effort".into()),
            },
        );
        step(
            "reset unknown key",
            "qwen3:8b",
            ConfigAction::Reset {
                key: Some("no.such.key".into()),
            },
        );
        step("reset all", "qwen3:8b", ConfigAction::Reset { key: None });
        step(
            "reset model without overrides",
            "never-seen",
            ConfigAction::Reset { key: None },
        );
        step("list local after sets", "local-thing.mq4", list());
        let transcript = transcript.replace(&root, "<ROOT>");
        fs::remove_dir_all(&paths.root).unwrap();
        transcript
    }

    #[test]
    fn model_config_list_set_reset_output_is_stable() {
        let transcript = model_config_golden_transcript();
        // set/reset output is the original capture; the list blocks follow the
        // grouped settings view (using now / from / applies), changed
        // intentionally with `config list` redesign.
        assert_eq!(
            transcript,
            include_str!("../tests/fixtures/model_config_transcript.txt")
        );
    }

    /// A fixture HOME: optional `config.toml`, `models.toml` and
    /// `install.json` under a fresh root.
    fn settings_fixture(
        label: &str,
        config_toml: Option<&str>,
        models_toml: Option<&str>,
        install_json: Option<&str>,
    ) -> Paths {
        let paths = test_paths(label);
        fs::create_dir_all(&paths.root).unwrap();
        if let Some(text) = config_toml {
            fs::write(&paths.config.config_toml, text).unwrap();
        }
        if let Some(text) = models_toml {
            fs::write(&paths.config.models_toml, text).unwrap();
        }
        if let Some(text) = install_json {
            fs::write(paths.config.root.join("install.json"), text).unwrap();
        }
        paths
    }

    fn squash(line: &str) -> String {
        line.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn list_global_text(paths: &Paths, args: ConfigListArgs) -> String {
        let mut out = Vec::new();
        config_view::list_global_to(paths, &args, &mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    const LOW_EFFORT: &str = "[reasoning]\neffort = \"low\"\n";
    const GFX1201: &str = "{\"gpu_arch\":\"gfx1201\"}";

    #[test]
    fn config_list_group_quality_shows_value_origin_and_applies() {
        let paths = settings_fixture("cfg-list-quality", Some(LOW_EFFORT), None, Some(GFX1201));
        let text = list_global_text(
            &paths,
            ConfigListArgs {
                group: Some("QUALITY".into()),
                ..ConfigListArgs::default()
            },
        );
        assert!(text.contains("GPU: gfx1201 (install.json)"), "{text}");
        assert!(text.contains("\nQuality\n"), "{text}");
        for other in ["Speed", "Memory & context", "Speculation", "Advanced"] {
            assert!(!text.contains(&format!("\n{other}\n")), "{text}");
        }
        let rows: Vec<String> = text.lines().map(squash).collect();
        assert!(
            rows.iter()
                .any(|row| row == "reasoning.effort low you next message"),
            "{text}"
        );
        assert!(!text.contains("memory.kv_cache"), "{text}");
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn config_list_unknown_group_lists_valid_names_and_aliases_work() {
        let paths = settings_fixture("cfg-list-group-names", None, None, None);
        let mut out = Vec::new();
        let error = config_view::list_global_to(
            &paths,
            &ConfigListArgs {
                group: Some("nope".into()),
                ..ConfigListArgs::default()
            },
            &mut out,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unknown group 'nope'"), "{error}");
        for name in [
            "quality",
            "speed",
            "memory",
            "speculation",
            "hardware",
            "advanced",
        ] {
            assert!(error.contains(name), "{error}");
        }
        for (alias, title) in [
            ("memory-context", "Memory & context"),
            ("Memory", "Memory & context"),
            ("hardware-devices", "Hardware & devices"),
            ("hardware", "Hardware & devices"),
        ] {
            let text = list_global_text(
                &paths,
                ConfigListArgs {
                    group: Some(alias.into()),
                    ..ConfigListArgs::default()
                },
            );
            assert!(text.contains(&format!("\n{title}\n")), "{alias}: {text}");
        }
        assert!(list_global_text(
            &paths,
            ConfigListArgs {
                group: Some("memory".into()),
                ..ConfigListArgs::default()
            }
        )
        .contains("GPU: unknown (no gpu_arch in install.json)"));
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn config_list_changed_hides_default_rows_and_search_filters() {
        let paths = settings_fixture("cfg-list-changed", Some(LOW_EFFORT), None, Some(GFX1201));
        let changed = list_global_text(
            &paths,
            ConfigListArgs {
                changed: true,
                ..ConfigListArgs::default()
            },
        );
        // The process may carry HIPFIRE_* variables (env rows are marked too);
        // every kept row must be marked and the default rows must be gone.
        let rows: Vec<String> = changed
            .lines()
            .filter(|line| line.starts_with("  "))
            .map(squash)
            .collect();
        assert!(
            rows.iter()
                .any(|row| row == "reasoning.effort low you next message"),
            "{changed}"
        );
        assert!(
            rows.iter()
                .all(|row| !row.contains(" default ") && !row.contains(" registry ")),
            "{changed}"
        );
        assert!(!changed.contains("memory.kv_cache"), "{changed}");

        let searched = list_global_text(
            &paths,
            ConfigListArgs {
                search: Some("KV".into()),
                ..ConfigListArgs::default()
            },
        );
        assert!(searched.contains("memory.kv_cache"), "{searched}");
        assert!(!searched.contains("reasoning.effort"), "{searched}");
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn config_list_json_adds_view_fields_and_keeps_existing_ones() {
        let paths = settings_fixture("cfg-list-json", Some(LOW_EFFORT), None, Some(GFX1201));
        let text = {
            let mut out = Vec::new();
            config_view::list_global_to(
                &paths,
                &ConfigListArgs {
                    json: true,
                    ..ConfigListArgs::default()
                },
                &mut out,
            )
            .unwrap();
            String::from_utf8(out).unwrap()
        };
        let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(doc["scope"], "global");
        assert_eq!(doc["gpu_arch"], "gfx1201");
        let effort = &doc["values"]["reasoning.effort"];
        assert_eq!(effort["value"], "low");
        assert_eq!(effort["overridden"], true);
        assert!(effort["legacy_key"].is_string());
        assert!(effort["default"].is_string());
        assert!(effort["source"].is_object() || effort["source"].is_string());
        assert_eq!(effort["group"], "quality");
        assert_eq!(effort["effective"]["value"], "low");
        assert_eq!(effort["origin"], "you");
        assert_eq!(effort["from"], "you");
        assert_eq!(effort["applies"], "next message");
        let kv = &doc["values"]["memory.kv_cache"];
        assert_eq!(kv["group"], "memory-context");
        assert_eq!(kv["effective"]["value"], "fp8");
        assert_eq!(kv["origin"], "default");
        assert_eq!(kv["from"], "default · gfx1201");
        assert_eq!(kv["overridden"], false);

        // Filters apply to JSON too.
        let mut out = Vec::new();
        config_view::list_global_to(
            &paths,
            &ConfigListArgs {
                json: true,
                changed: true,
                ..ConfigListArgs::default()
            },
            &mut out,
        )
        .unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&out).unwrap();
        // The process may carry HIPFIRE_* variables (env rows are marked too);
        // every kept row must be marked and the default rows must be gone.
        let rows = doc["values"].as_object().unwrap();
        assert!(rows.contains_key("reasoning.effort"));
        assert!(!rows.contains_key("memory.kv_cache"));
        assert!(rows
            .values()
            .all(|row| !matches!(row["origin"].as_str(), Some("default" | "registry"))));
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn config_model_list_marks_rows_the_model_does_not_use() {
        let paths = settings_fixture(
            "cfg-list-model",
            None,
            None,
            Some("{\"gpu_arch\":\"gfx1151\"}"),
        );
        let registry = RegistryV1::parse(
            r#"{
                "schema_version":1,
                "generated_at":"2026-09-01T00:00:00Z",
                "models":{
                    "dense:8b":{"repo":"x","file":"dense-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5}
                },
                "aliases":{}
            }"#,
            "test",
        )
        .unwrap();
        let mut out = Vec::new();
        model_config_command_to(
            &paths,
            &registry,
            "dense:8b",
            Some(ConfigAction::List(ConfigListArgs {
                group: Some("quality".into()),
                ..ConfigListArgs::default()
            })),
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("Model configuration: dense:8b\n"),
            "{text}"
        );
        assert!(text.contains("GPU: gfx1151 (install.json)"), "{text}");
        let ple = text
            .lines()
            .find(|line| line.trim_start().starts_with("memory.ple_format"))
            .unwrap();
        assert!(ple.ends_with("   (not used by this model)"), "{ple}");
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn config_list_model_flag_routes_to_the_model_form() {
        let parsed = Cli::try_parse_from([
            "hipfire",
            "config",
            "list",
            "--model",
            "qwen:test",
            "--group",
            "quality",
        ])
        .unwrap();
        let Some(Commands::Config(args)) = parsed.command else {
            panic!("expected config command")
        };
        assert!(args.model.is_none());
        assert!(matches!(
            &args.action,
            Some(ConfigAction::List(ConfigListArgs { model: Some(m), .. })) if m == "qwen:test"
        ));

        let paths = settings_fixture("cfg-model-twice", None, None, None);
        let error = config_command(
            &paths,
            ConfigArgs {
                model: Some("a".into()),
                action: Some(ConfigAction::List(ConfigListArgs {
                    model: Some("b".into()),
                    ..ConfigListArgs::default()
                })),
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("model given twice"), "{error}");
        let _ = fs::remove_dir_all(&paths.root);
    }

    const THREE_OVERRIDES: &str = r#"schema_version = 1

[models."local-thing.mq4".overrides.generation]
temperature = 0.5

[models."qwen3:8b"]
registry_tag = "qwen3:8b"

[models."qwen3:8b".overrides.memory]
max_seq = 4096

[models."qwen3:8b".overrides.reasoning]
effort = "high"
"#;

    #[test]
    fn config_overrides_text_and_json_compare_with_all_models() {
        let paths = settings_fixture(
            "cfg-overrides",
            Some(LOW_EFFORT),
            Some(THREE_OVERRIDES),
            None,
        );
        let mut out = Vec::new();
        config_view::overrides_to(&paths, false, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let rows: Vec<String> = text.lines().map(squash).collect();
        assert_eq!(
            rows,
            [
                "local-thing.mq4 generation.temperature 0.5 (all models: 0.3)",
                "qwen3:8b memory.max_seq 4096 (all models: 32768)",
                "qwen3:8b reasoning.effort high (all models: low)",
                "",
                "3 overrides on 2 models.",
            ],
            "{text}"
        );
        // The documented column layout.
        assert!(text.contains(&format!(
            "  {:<22} {:<32} {:<10} (all models: low)",
            "qwen3:8b", "reasoning.effort", "high"
        )));

        let mut out = Vec::new();
        config_view::overrides_to(&paths, true, &mut out).unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(doc["count"], 3);
        assert_eq!(doc["models"], 2);
        let list = doc["overrides"].as_array().unwrap();
        assert_eq!(list.len(), 3);
        assert!(list.iter().any(|row| row["model"] == "qwen3:8b"
            && row["key"] == "reasoning.effort"
            && row["value"] == "high"
            && row["all_models"] == "low"));
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn config_overrides_reports_none_when_models_toml_is_empty() {
        let paths = settings_fixture("cfg-overrides-empty", None, None, None);
        let mut out = Vec::new();
        config_view::overrides_to(&paths, false, &mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "No per-model overrides (~/.hipfire/models.toml has none).\n"
        );
        let mut out = Vec::new();
        config_view::overrides_to(&paths, true, &mut out).unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(doc["count"], 0);
        assert_eq!(doc["overrides"], serde_json::json!([]));
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn config_explain_shows_effective_value_origin_group_and_applies() {
        let paths = settings_fixture("cfg-explain", None, None, Some(GFX1201));
        let mut out = Vec::new();
        config_view::explain_global_to(&paths, "memory.kv_cache", false, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let effective = text
            .lines()
            .find(|line| line.starts_with("  effective:   "))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(
            effective.starts_with("  effective:   fp8 (auto"),
            "{effective}"
        );
        assert!(
            text.contains("\n  from:        default · gfx1201\n"),
            "{text}"
        );
        assert!(
            text.contains("\n  group:       Memory & context\n"),
            "{text}"
        );
        assert!(
            text.contains("\n  applies:     next model load\n"),
            "{text}"
        );

        // A plain default with no rule has no `effective:` line.
        let mut out = Vec::new();
        config_view::explain_global_to(&paths, "generation.temperature", false, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("effective:"), "{text}");
        assert!(text.contains("\n  from:        default\n"), "{text}");

        let mut out = Vec::new();
        config_view::explain_global_to(&paths, "memory.kv_cache", true, &mut out).unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(doc["effective"]["value"], "fp8");
        assert_eq!(doc["origin"], "default");
        assert_eq!(doc["from"], "default · gfx1201");
        assert_eq!(doc["group"], "memory-context");
        assert!(doc["applies"].is_string());
        assert!(doc["shadowed"].is_array(), "existing fields stay");
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn config_explain_without_key_lists_every_setting_source_and_env_audit() {
        let paths = settings_fixture("cfg-explain-all", None, None, Some(GFX1201));
        // Fixture names are assembled at runtime so check-lifecycle.py does not
        // read the typo / secret fixtures as variables the CLI reads.
        let env_name = |suffix: &str| format!("HIPFIRE_{suffix}");
        let env = hipfire_config::env_audit::audit_env([
            (env_name("KVMODE"), "q8".to_owned()),
            (env_name("DFLASH_CTX_CAP"), "1".to_owned()),
            (env_name("API_KEY"), "hunter2".to_owned()),
        ]);
        let mut out = Vec::new();
        config_view::explain_all_global_to(&paths, false, &env, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("Effective configuration (config.toml: "),
            "{text}"
        );
        let kv = text
            .lines()
            .find(|line| line.trim_start().starts_with("memory.kv_cache "))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(kv.ends_with("default · gfx1201"), "{kv}");
        assert!(text.contains("\nenvironment\n"), "{text}");
        assert!(
            text.contains("UNKNOWN: not a hipfire variable (typo? did you mean HIPFIRE_KV_MODE)"),
            "{text}"
        );
        assert!(
            text.contains("DEPRECATED since 0.4.0, removed in 0.5.0: unset it"),
            "{text}"
        );
        assert!(
            text.contains("(set)") && !text.contains("hunter2"),
            "{text}"
        );
        assert!(
            text.ends_with("explain one key: hipfire config explain <key>\n"),
            "{text}"
        );

        let mut out = Vec::new();
        config_view::explain_all_global_to(&paths, true, &env, &mut out).unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&out).unwrap();
        let settings = doc["settings"].as_array().unwrap();
        let kv = settings
            .iter()
            .find(|row| row["key"] == "memory.kv_cache")
            .unwrap();
        assert_eq!(kv["source"]["kind"], "default-arch");
        assert_eq!(kv["source"]["label"], "default · gfx1201");
        assert_eq!(doc["env"].as_array().unwrap().len(), 3);
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    pub(crate) fn resolved_for_model_tag_policy_is_overridable_by_user() {
        let paths = test_paths("registry-tag-policy-override");
        fs::create_dir_all(&paths.root).unwrap();
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let resolved = resolved_for_model(&paths, tag, Some(tag), Some(entry)).unwrap();
        assert_eq!(
            config_string(&resolved, "memory.kv_backend").unwrap(),
            "vmm"
        );
        assert_eq!(config_u64(&resolved, "memory.max_seq").unwrap(), 32768);
        assert_eq!(
            config_u64(&resolved, "generation.max_tokens").unwrap(),
            81920
        );

        // Global user override wins over registry tag policy (registry below global).
        let mut user_layer = ConfigLayer::default();
        user_layer.set_cli("memory.kv_backend", "legacy").unwrap();
        user_layer.set_cli("memory.max_seq", "32768").unwrap();
        user_layer.set_cli("generation.max_tokens", "1024").unwrap();
        let overridden = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: hipfire_registry::config_layer_for_tag(tag, entry).unwrap(),
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test.toml"),
                },
                layer: user_layer,
            },
        ])
        .unwrap();
        assert_eq!(
            config_string(&overridden, "memory.kv_backend").unwrap(),
            "legacy"
        );
        assert_eq!(config_u64(&overridden, "memory.max_seq").unwrap(), 32768);
        assert_eq!(
            config_u64(&overridden, "generation.max_tokens").unwrap(),
            1024
        );

        // Explicit CLI flag is emitted (flag wins over automatic resolved backend).
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let params = load_params(
            &resolved,
            Some(entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            Some("legacy"),
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["kv_backend"], "legacy");
        // Automatic registry/built-in: omit kv_backend and max_seq.
        let params2 = load_params(
            &resolved,
            Some(entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            params2.get("kv_backend").is_none(),
            "registry-tag automatic backend must omit kv_backend"
        );
        assert!(
            params2.get("max_seq").is_none(),
            "registry-tag automatic max_seq must be omitted"
        );

        // Global user config is an explicit choice and is emitted.
        let params_global = load_params(
            &overridden,
            Some(entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params_global["kv_backend"], "legacy");
        assert_eq!(params_global["max_seq"], 32768);

        // Precedence: flag > model config > global config.
        let mut global_layer = ConfigLayer::default();
        global_layer.set_cli("memory.kv_backend", "legacy").unwrap();
        let mut model_layer = ConfigLayer::default();
        model_layer.set_cli("memory.kv_backend", "vmm").unwrap();
        let model_over_global = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/global.toml"),
                },
                layer: global_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::ModelUser {
                    model: tag.to_owned(),
                    path: std::path::PathBuf::from("/tmp/model.toml"),
                },
                layer: model_layer,
            },
        ])
        .unwrap();
        assert_eq!(
            config_string(&model_over_global, "memory.kv_backend").unwrap(),
            "vmm",
            "model user config must beat global user"
        );
        let params_model = load_params(
            &model_over_global,
            Some(entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params_model["kv_backend"], "vmm");
        let params_flag = load_params(
            &model_over_global,
            Some(entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            Some("legacy"),
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            params_flag["kv_backend"], "legacy",
            "CLI flag must beat model user config"
        );

        // Glimmer target likewise overridable (backend + max_seq).
        let raw2 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry2 = RegistryV1::parse(raw2, "test").unwrap();
        let (g_tag, g_entry) = registry2.model("muse-glimmer").unwrap();
        let g_layer = hipfire_registry::config_layer_for_tag(g_tag, g_entry).unwrap();
        assert!(g_layer.get("memory.kv_backend").is_none());
        assert!(g_layer.get("memory.max_seq").is_none());
        assert!(g_layer.get("generation.max_tokens").is_none());
        let mut g_user = ConfigLayer::default();
        g_user.set_cli("memory.kv_backend", "legacy").unwrap();
        g_user.set_cli("memory.max_seq", "8192").unwrap();
        let g_resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: g_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: g_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test2.toml"),
                },
                layer: g_user,
            },
        ])
        .unwrap();
        assert_eq!(
            config_string(&g_resolved, "memory.kv_backend").unwrap(),
            "legacy"
        );
        assert_eq!(config_u64(&g_resolved, "memory.max_seq").unwrap(), 8192);
        let model_path_g = PathBuf::from("/tmp/test-model.mq4");
        let g_params = load_params(
            &g_resolved,
            Some(g_entry),
            &model_path_g.parent().unwrap(),
            &model_path_g,
            64,
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(g_params["kv_backend"], "legacy");
        assert_eq!(g_params["max_seq"], 8192);

        // DeepSeek target: generation policy remains; user max_seq still wins.
        let raw3 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"deepseek-v4-flash":{"repo":"x","file":"ds4.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry3 = RegistryV1::parse(raw3, "test").unwrap();
        let (d_tag, d_entry) = registry3.model("deepseek-v4-flash").unwrap();
        let d_resolved = resolved_for_model(&paths, d_tag, Some(d_tag), Some(d_entry)).unwrap();
        assert_eq!(
            config_string(&d_resolved, "memory.kv_backend").unwrap(),
            "vmm"
        );
        assert_eq!(config_u64(&d_resolved, "memory.max_seq").unwrap(), 32768);
        assert_eq!(
            config_u64(&d_resolved, "generation.max_tokens").unwrap(),
            393216
        );
        let mut d_user = ConfigLayer::default();
        d_user.set_cli("memory.max_seq", "65536").unwrap();
        d_user.set_cli("generation.max_tokens", "2048").unwrap();
        let d_overridden = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: d_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: hipfire_registry::config_layer_for_tag(d_tag, d_entry).unwrap(),
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test3.toml"),
                },
                layer: d_user,
            },
        ])
        .unwrap();
        assert_eq!(config_u64(&d_overridden, "memory.max_seq").unwrap(), 65536);
        assert_eq!(
            config_u64(&d_overridden, "generation.max_tokens").unwrap(),
            2048
        );
        let d_params = load_params(
            &d_overridden,
            Some(d_entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(d_params["max_seq"], 65536);

        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_only_forwards_explicit_deepseek4_expert_fanout() {
        let model_path = PathBuf::from("/tmp/test-model.mq2r");
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let params = load_params(
            &defaults,
            None,
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["deepseek4_compute_placement"], "single");
        assert!(params.get("deepseek4_experts_per_token").is_none());

        let mut explicit = ConfigLayer::default();
        explicit
            .set_cli("model.deepseek4_experts_per_token", "4")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "model.deepseek4_experts_per_token=4".into(),
            },
            layer: explicit,
        }])
        .unwrap();
        let params = load_params(
            &resolved,
            None,
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["deepseek4_experts_per_token"], 4);
    }

    #[test]
    pub(crate) fn load_params_forwards_explicit_state_quant() {
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let load = |resolved: &hipfire_config::ResolvedConfig| {
            load_params(
                resolved,
                None,
                model_path.parent().unwrap(),
                &model_path,
                64,
                Some("q8"),
                None,
                None,
                false,
                None,
            )
            .unwrap()
        };
        // The built-in `auto` default is not forwarded: the daemon applies its
        // per-model/arch default.
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        assert!(load(&defaults).get("state_quant").is_none());

        let sources = [
            (
                "q8",
                ConfigSource::OneShot {
                    argument: "memory.state_quant=q8".into(),
                },
            ),
            (
                "fp32",
                ConfigSource::GlobalUser {
                    path: PathBuf::from("/tmp/global.toml"),
                },
            ),
            (
                "auto",
                ConfigSource::ModelUser {
                    model: "test-model".into(),
                    path: PathBuf::from("/tmp/model.toml"),
                },
            ),
            (
                "f32",
                ConfigSource::LegacyEnv {
                    name: "HIPFIRE_*".into(),
                },
            ),
        ];
        for (value, source) in sources {
            let mut explicit = ConfigLayer::default();
            explicit.set_cli("memory.state_quant", value).unwrap();
            let resolved = resolve([NamedLayer {
                source,
                layer: explicit,
            }])
            .unwrap();
            assert_eq!(load(&resolved)["state_quant"], value, "{value}");
        }
    }

    #[test]
    pub(crate) fn load_params_forwards_typed_deepseek4_compute_placement() {
        let raw = "dense-expert-split(dense=arch:gfx1100,experts=arch:gfx1151)";
        let mut explicit = ConfigLayer::default();
        explicit
            .set_cli("hardware.deepseek4_compute_placement", raw)
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: format!("hardware.deepseek4_compute_placement={raw}"),
            },
            layer: explicit,
        }])
        .unwrap();
        let params = load_params(
            &resolved,
            None,
            Path::new("/tmp/test-model.mq2r").parent().unwrap(),
            Path::new("/tmp/test-model.mq2r"),
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["deepseek4_compute_placement"], raw);
    }

    #[test]
    pub(crate) fn load_params_forwards_dflash_draft_from_environment() {
        let draft = "/tmp/qwen35-9b-dflash-mq4.hfq";

        let mut explicit = ConfigLayer::default();
        explicit.set_cli("speculation.mode", "dflash").unwrap();
        explicit.set_cli("developer.dflash_draft", draft).unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "speculation.mode=dflash".into(),
            },
            layer: explicit,
        }])
        .unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");

        let params = load_params(
            &resolved,
            None,
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["draft"], draft);
    }

    fn dflash_sidecar_entry(draft_file: &str) -> ModelEntry {
        ModelEntry {
            repo: "hipfire-models/qwen3.5-9b".into(),
            file: "qwen3.5-9b.mq4".into(),
            size_gb: 5.31,
            min_vram_gb: 6.8,
            desc: "test target".into(),
            dflash: Some(hipfire_registry::Sidecar {
                file: draft_file.into(),
                sha256: None,
                size_bytes: None,
            }),
            ..Default::default()
        }
    }

    fn vision_sidecar_entry(vision_file: &str) -> ModelEntry {
        ModelEntry {
            repo: "hipfire-models/qwen3.8-27b".into(),
            file: "qwen3.8-27b.mq4".into(),
            size_gb: 15.66,
            min_vram_gb: 17.0,
            desc: "test target".into(),
            vision: Some(hipfire_registry::Sidecar {
                file: vision_file.into(),
                sha256: None,
                size_bytes: None,
            }),
            ..Default::default()
        }
    }
    fn xdna_sidecar_entry(xdna_file: &str) -> ModelEntry {
        ModelEntry {
            repo: "hipfire-models/qwen3.8-27b".into(),
            file: "qwen3.8-27b.mq4".into(),
            size_gb: 15.66,
            min_vram_gb: 17.0,
            desc: "test target".into(),
            xdna: Some(hipfire_registry::Sidecar {
                file: xdna_file.into(),
                sha256: None,
                size_bytes: None,
            }),
            ..Default::default()
        }
    }

    fn resolved_with_npu_spillover(on: bool) -> hipfire_config::ResolvedConfig {
        let mut explicit = ConfigLayer::default();
        explicit
            .set_cli("kernel.npu_spillover", if on { "true" } else { "false" })
            .unwrap();
        resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: format!("kernel.npu_spillover={on}"),
            },
            layer: explicit,
        }])
        .unwrap()
    }

    /// Minimal stored-ZIP writer for XDNA sidecar fixtures (local headers +
    /// central directory + EOCD, method 0 throughout).
    fn write_stored_zip(path: &Path, entries: &[(&str, &[u8])]) {
        fn u16le(out: &mut Vec<u8>, value: u16) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        fn u32le(out: &mut Vec<u8>, value: u32) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let mut bytes = Vec::new();
        let mut central = Vec::new();
        for (name, data) in entries {
            let local_offset = bytes.len() as u32;
            bytes.extend_from_slice(b"PK\x03\x04");
            u16le(&mut bytes, 20);
            u16le(&mut bytes, 0);
            u16le(&mut bytes, 0);
            u16le(&mut bytes, 0);
            u16le(&mut bytes, 0);
            u32le(&mut bytes, 0);
            u32le(&mut bytes, data.len() as u32);
            u32le(&mut bytes, data.len() as u32);
            u16le(&mut bytes, name.len() as u16);
            u16le(&mut bytes, 0);
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(data);
            central.extend_from_slice(b"PK\x01\x02");
            u16le(&mut central, 20);
            u16le(&mut central, 20);
            u16le(&mut central, 0);
            u16le(&mut central, 0);
            u16le(&mut central, 0);
            u16le(&mut central, 0);
            u32le(&mut central, 0);
            u32le(&mut central, data.len() as u32);
            u32le(&mut central, data.len() as u32);
            u16le(&mut central, name.len() as u16);
            u16le(&mut central, 0);
            u16le(&mut central, 0);
            u16le(&mut central, 0);
            u16le(&mut central, 0);
            u32le(&mut central, 0);
            u32le(&mut central, local_offset);
            central.extend_from_slice(name.as_bytes());
        }
        let central_offset = bytes.len() as u32;
        bytes.extend_from_slice(&central);
        let central_size = central.len() as u32;
        bytes.extend_from_slice(b"PK\x05\x06");
        u16le(&mut bytes, 0);
        u16le(&mut bytes, 0);
        u16le(&mut bytes, entries.len() as u16);
        u16le(&mut bytes, entries.len() as u16);
        u32le(&mut bytes, central_size);
        u32le(&mut bytes, central_offset);
        u16le(&mut bytes, 0);
        fs::write(path, &bytes).unwrap();
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        digest
            .iter()
            .fold(String::with_capacity(64), |mut out, byte| {
                out.push(char::from_digit((byte >> 4) as u32, 16).unwrap());
                out.push(char::from_digit((byte & 0x0F) as u32, 16).unwrap());
                out
            })
    }

    /// A manifest-v1 `.xdna.zip` binding the given payloads, written to the
    /// models dir. Returns the archive path.
    fn write_xdna_fixture(models_dir: &Path, file: &str) -> PathBuf {
        let pdi = b"fixture-pdi-bytes";
        let insts = b"fixture-insts-bytes";
        let manifest = serde_json::json!({
            "version": 1,
            "model_sha256": "c".repeat(64),
            "quant": "mq4g256v2",
            "arch": "gfx1151",
            "npu": "npu5",
            "toolchain": {"mlir_aie": "f50bef7", "peano": "22.0.0.2026090701"},
            "profiles": [{
                "tensor_role": "gate_up",
                "M": 1024, "K": 5120, "N": 512, "row_count": 1024,
                "tile_m": 128, "tile_k": 64, "tile_n": 64,
                "cols": 8, "k_mt": 512,
                "pdi": "profiles/gate_up/main.pdi",
                "insts": "profiles/gate_up/insts.bin",
                "arg_layout": [{"name": "a", "offset": 0, "size": 64}],
                "sha256_pdi": sha256_hex(pdi),
                "sha256_insts": sha256_hex(insts)
            }]
        })
        .to_string();
        let path = models_dir.join(file);
        write_stored_zip(
            &path,
            &[
                ("manifest.json", manifest.as_bytes()),
                ("profiles/gate_up/main.pdi", pdi),
                ("profiles/gate_up/insts.bin", insts),
            ],
        );
        path
    }

    fn resolved_with_dflash_mode(
        mode: &str,
        draft: Option<&str>,
    ) -> hipfire_config::ResolvedConfig {
        let mut explicit = ConfigLayer::default();
        explicit.set_cli("speculation.dflash", mode).unwrap();
        if let Some(draft) = draft {
            explicit.set_cli("developer.dflash_draft", draft).unwrap();
        }
        resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: format!("speculation.dflash={mode}"),
            },
            layer: explicit,
        }])
        .unwrap()
    }

    fn resolved_with_vision_mode(mode: &str) -> hipfire_config::ResolvedConfig {
        let mut explicit = ConfigLayer::default();
        explicit.set_cli("vision.mode", mode).unwrap();
        resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: format!("vision.mode={mode}"),
            },
            layer: explicit,
        }])
        .unwrap()
    }

    #[test]
    pub(crate) fn load_params_resolves_registry_dflash_sidecar_when_present() {
        // (b) auto + pulled draft file → params["draft"] points at it.
        let paths = test_paths("dflash-sidecar-present");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.5-9b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let draft_path = paths.models.join("qwen35-9b-dflash-mq4.hfq");
        fs::write(&draft_path, b"draft").unwrap();
        let entry = dflash_sidecar_entry("qwen35-9b-dflash-mq4.hfq");
        let resolved = resolved_with_dflash_mode("auto", None);
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.5:9b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["dflash_mode"], "auto");
        assert_eq!(params["draft"], draft_path.display().to_string());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    pub(crate) fn load_params_finds_sidecar_in_models_dir_for_symlinked_target() {
        // A target symlinked out of the models dir has a parent with no draft
        // in it. The sidecar must be looked up in the models dir, not only
        // beside the target.
        // Measured 2026-09-03: serve --speculation dflash ran AR (tau=None)
        // on a symlinked qwen3.8-27b.mq5 while the tag form resolved.
        let paths = test_paths("dflash-sidecar-symlink");
        fs::create_dir_all(&paths.models).unwrap();
        let elsewhere = paths.root.join("artifacts");
        fs::create_dir_all(&elsewhere).unwrap();
        let real_model = elsewhere.join("qwen3.5-9b.mq4v2.base.hfq");
        fs::write(&real_model, b"model").unwrap();
        std::os::unix::fs::symlink(&real_model, paths.models.join("qwen3.5-9b.mq4")).unwrap();
        let draft_path = paths.models.join("qwen35-9b-dflash-mq4.hfq");
        fs::write(&draft_path, b"draft").unwrap();
        let entry = dflash_sidecar_entry("qwen35-9b-dflash-mq4.hfq");
        let resolved = resolved_with_dflash_mode("on", None);
        // What serve/run actually pass: the canonicalized path.
        let canonical = fs::canonicalize(paths.models.join("qwen3.5-9b.mq4")).unwrap();
        assert_eq!(
            canonical.parent().unwrap(),
            elsewhere.canonicalize().unwrap()
        );
        let params = load_params(
            &resolved,
            Some(&entry),
            &paths.models,
            &canonical,
            64,
            Some("q8"),
            None,
            Some("qwen3.5:9b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["draft"], draft_path.display().to_string());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_dflash_on_fails_closed_when_sidecar_missing() {
        // (c) on + missing file errors with a pull hint naming the tag.
        let paths = test_paths("dflash-sidecar-on-missing");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.5-9b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let entry = dflash_sidecar_entry("qwen35-9b-dflash-mq4.hfq");
        let resolved = resolved_with_dflash_mode("on", None);
        let error = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.5:9b"),
            false,
            None,
        )
        .expect_err("on without a pulled draft must fail closed");
        let message = format!("{error:#}");
        assert!(message.contains("qwen35-9b-dflash-mq4.hfq"), "{message}");
        assert!(message.contains("hipfire pull qwen3.5:9b"), "{message}");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_dflash_auto_runs_ar_when_sidecar_missing() {
        // (d) auto + missing file yields no draft and no error.
        let paths = test_paths("dflash-sidecar-auto-missing");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.5-9b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let entry = dflash_sidecar_entry("qwen35-9b-dflash-mq4.hfq");
        let resolved = resolved_with_dflash_mode("auto", None);
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.5:9b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["dflash_mode"], "auto");
        assert_eq!(params["mtp_mode"], "auto", "MTP auto stays in effect");
        assert!(
            params.get("draft").is_none(),
            "auto without a pulled draft wires no draft"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    fn qwen38_27b_dflash_registry() -> RegistryV1 {
        let dflash = r#""dflash":{"file":"qwen38-27b-dflash-mq4.hfq"}"#;
        let raw = format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{{
                "qwen3.8:27b":{{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq4-pro":{{"repo":"x","file":"qwen3.8-27b.mq4-pro","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq5":{{"repo":"x","file":"qwen3.8-27b.mq5","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5}},
                "qwen3.8:27boops":{{"repo":"x","file":"qwen3.8-27boops.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:flash-next":{{"repo":"x","file":"qwen3.8-flash-next.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":16,{dflash}}},
                "qwen3.8:27b-draft-mq4":{{"repo":"x","file":"qwen38-27b-draft.hfq","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":20,{dflash}}},
                "qwen3.6:27b":{{"repo":"x","file":"qwen3.6-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}},
                "qwen3.5:9b":{{"repo":"x","file":"qwen3.5-9b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}}
            }},
            "aliases":{{"qwen3.8":"qwen3.8:27b","qwen3.8:27b-mq4":"qwen3.8:27b"}}
        }}"#
        );
        RegistryV1::parse(&raw, "test").unwrap()
    }

    #[test]
    pub(crate) fn resolved_for_model_defaults_dflash_auto_only_for_paired_qwen38_27b() {
        // Registry layer lowers dflash=auto only for canonical qwen3.8:27b[-*]
        // trunks that declare a dflash sidecar; BuiltIn off and MTP auto
        // stay for everything else, including the no-registry path.
        let paths = test_paths("registry-dflash-default-scope");
        fs::create_dir_all(&paths.root).unwrap();
        let registry = qwen38_27b_dflash_registry();
        for (input, expected) in [
            ("qwen3.8:27b", "auto"),
            ("qwen3.8", "auto"),
            ("qwen3.8:27b-mq4", "auto"),
            ("qwen3.8:27b-mq4-pro", "auto"),
            ("qwen3.8:27b-mq5", "off"),
            ("qwen3.8:27boops", "off"),
            ("qwen3.8:flash-next", "off"),
            ("qwen3.8:27b-draft-mq4", "off"),
            ("qwen3.6:27b", "off"),
            ("qwen3.5:9b", "off"),
        ] {
            let (tag, entry) = registry.model(input).unwrap();
            let resolved = resolved_for_model(&paths, input, Some(tag), Some(entry)).unwrap();
            assert_eq!(
                config_string(&resolved, "speculation.dflash").unwrap(),
                expected,
                "{input}->{tag}"
            );
            assert_eq!(
                config_string(&resolved, "speculation.mtp").unwrap(),
                "auto",
                "{input}->{tag}"
            );
            assert_eq!(
                config_string(&resolved, "speculation.mode").unwrap(),
                "auto",
                "{input}->{tag}"
            );
        }
        // No registry tag/entry (bare artifact path): BuiltIn off.
        let bare = resolved_for_model(&paths, "qwen3.8-27b.mq4", None, None).unwrap();
        assert_eq!(config_string(&bare, "speculation.dflash").unwrap(), "off");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_wires_default_dflash_draft_for_qwen38_27b_only_when_pulled() {
        // Paired file present: auto + draft wired. Missing: no draft, MTP auto
        // unchanged. Non-eligible models never wire a draft even with the file
        // on disk.
        let paths = test_paths("registry-dflash-default-load");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = qwen38_27b_dflash_registry();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let resolved = resolved_for_model(&paths, tag, Some(tag), Some(entry)).unwrap();
        let load = |resolved: &hipfire_config::ResolvedConfig, tag: &str, entry: &ModelEntry| {
            load_params(
                resolved,
                Some(entry),
                &paths.models,
                &model_path,
                64,
                Some("q8"),
                None,
                Some(tag),
                false,
                None,
            )
            .unwrap()
        };

        let missing = load(&resolved, tag, entry);
        assert_eq!(missing["dflash_mode"], "auto");
        assert_eq!(missing["mtp_mode"], "auto");
        assert!(missing.get("draft").is_none(), "no pulled draft, no wiring");

        let draft_path = paths.models.join("qwen38-27b-dflash-mq4.hfq");
        fs::write(&draft_path, b"draft").unwrap();
        let present = load(&resolved, tag, entry);
        assert_eq!(present["dflash_mode"], "auto");
        assert_eq!(present["mtp_mode"], "auto");
        assert_eq!(present["draft"], draft_path.display().to_string());

        for input in [
            "qwen3.8:27b-mq5",
            "qwen3.8:27boops",
            "qwen3.8:flash-next",
            "qwen3.8:27b-draft-mq4",
            "qwen3.6:27b",
            "qwen3.5:9b",
        ] {
            let (tag, entry) = registry.model(input).unwrap();
            let resolved = resolved_for_model(&paths, input, Some(tag), Some(entry)).unwrap();
            let params = load(&resolved, tag, entry);
            assert_eq!(params["dflash_mode"], "off", "{input}");
            assert_eq!(params["mtp_mode"], "auto", "{input}");
            assert!(params.get("draft").is_none(), "{input}");
        }
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn qwen38_27b_default_dflash_yields_to_explicit_off_and_blocks_developer_draft() {
        // The registry layer sits below global/model/one-shot config: an
        // explicit dflash=off (or speculation.mode=off) wins and strips both
        // the paired sidecar and a developer.dflash_draft.
        let paths = test_paths("registry-dflash-default-off-wins");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = qwen38_27b_dflash_registry();
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let draft_path = paths.models.join("qwen38-27b-dflash-mq4.hfq");
        fs::write(&draft_path, b"draft").unwrap();
        let developer_draft = "/tmp/developer-draft.hfq";
        let registry_layer = || NamedLayer {
            source: ConfigSource::RegistryModel {
                tag: tag.to_owned(),
                revision: "v1".into(),
            },
            layer: hipfire_registry::config_layer_for_tag(tag, entry).unwrap(),
        };
        let load = |resolved: &hipfire_config::ResolvedConfig| {
            load_params(
                resolved,
                Some(entry),
                &paths.models,
                &model_path,
                64,
                Some("q8"),
                None,
                Some(tag),
                false,
                None,
            )
            .unwrap()
        };
        let global = |layer: ConfigLayer| NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("/tmp/global.toml"),
            },
            layer,
        };
        let model_user = |layer: ConfigLayer| NamedLayer {
            source: ConfigSource::ModelUser {
                model: tag.to_owned(),
                path: PathBuf::from("/tmp/model.toml"),
            },
            layer,
        };

        // Baseline: registry default is on (auto), the pulled draft is wired.
        let baseline = resolve(vec![registry_layer()]).unwrap();
        let params = load(&baseline);
        assert_eq!(params["dflash_mode"], "auto");
        assert_eq!(params["draft"], draft_path.display().to_string());

        // A developer draft with default auto still wins over the paired file.
        let mut dev_only = ConfigLayer::default();
        dev_only
            .set_cli("developer.dflash_draft", developer_draft)
            .unwrap();
        let params = load(&resolve(vec![registry_layer(), global(dev_only)]).unwrap());
        assert_eq!(params["dflash_mode"], "auto");
        assert_eq!(params["draft"], developer_draft);

        // Explicit global/model off wins and drops sidecar + developer draft.
        for scope in ["global", "model"] {
            let mut user = ConfigLayer::default();
            user.set_cli("speculation.dflash", "off").unwrap();
            user.set_cli("developer.dflash_draft", developer_draft)
                .unwrap();
            let user_layer = if scope == "global" {
                global(user)
            } else {
                model_user(user)
            };
            let resolved = resolve(vec![registry_layer(), user_layer]).unwrap();
            assert_eq!(
                config_string(&resolved, "speculation.dflash").unwrap(),
                "off",
                "{scope}"
            );
            let params = load(&resolved);
            assert_eq!(params["dflash_mode"], "off", "{scope}");
            assert!(
                params.get("draft").is_none(),
                "{scope}: off blocks sidecar and developer draft"
            );
        }

        // One-shot off (as `--set speculation.dflash=off`) wins over a model-user auto.
        let mut model_auto = ConfigLayer::default();
        model_auto.set_cli("speculation.dflash", "auto").unwrap();
        let params = load(
            &resolve(vec![
                registry_layer(),
                model_user(model_auto),
                NamedLayer {
                    source: ConfigSource::OneShot {
                        argument: "speculation.dflash=off".into(),
                    },
                    layer: {
                        let mut layer = ConfigLayer::default();
                        layer.set_cli("speculation.dflash", "off").unwrap();
                        layer
                    },
                },
            ])
            .unwrap(),
        );
        assert_eq!(params["dflash_mode"], "off");
        assert!(params.get("draft").is_none());

        // speculation.mode=off disables DFlash through the selector as well.
        let mut mode_off = ConfigLayer::default();
        mode_off.set_cli("speculation.mode", "off").unwrap();
        let params = load(&resolve(vec![registry_layer(), global(mode_off)]).unwrap());
        assert_eq!(params["dflash_mode"], "off");
        assert!(params.get("draft").is_none());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_explicit_draft_wins_over_dflash_sidecar() {
        // (e) developer.dflash_draft beats the sidecar even when pulled.
        let paths = test_paths("dflash-sidecar-explicit-wins");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.5-9b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let draft_path = paths.models.join("qwen35-9b-dflash-mq4.hfq");
        fs::write(&draft_path, b"draft").unwrap();
        let entry = dflash_sidecar_entry("qwen35-9b-dflash-mq4.hfq");
        let explicit = "/tmp/custom-draft.hfq";
        let resolved = resolved_with_dflash_mode("auto", Some(explicit));
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.5:9b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["draft"], explicit);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_final_off_drops_dflash_sidecar() {
        // (f) off never carries the sidecar, and a final off selector drops
        // a previously resolved one.
        let paths = test_paths("dflash-sidecar-off-drops");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.5-9b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let draft_path = paths.models.join("qwen35-9b-dflash-mq4.hfq");
        fs::write(&draft_path, b"draft").unwrap();
        let entry = dflash_sidecar_entry("qwen35-9b-dflash-mq4.hfq");
        let resolved = resolved_with_dflash_mode("off", None);
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.5:9b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["dflash_mode"], "off");
        assert!(
            params.get("draft").is_none(),
            "off must not resolve the sidecar"
        );

        // Resolve under auto, then a final off selector drops it.
        let resolved = resolved_with_dflash_mode("auto", None);
        let mut params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.5:9b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["draft"], draft_path.display().to_string());
        apply_speculation_selector(&mut params, "off").unwrap();
        project_dflash_draft(&mut params, developer_dflash_draft(&resolved));
        assert_eq!(params["dflash_mode"], "off");
        assert!(
            params.get("draft").is_none(),
            "final off must drop the sidecar draft"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_skips_sidecar_for_explicit_cli_draft() {
        // `run --model-draft` (projected by the caller after load_params)
        // always wins: even `on` must not fail closed on a missing sidecar.
        let paths = test_paths("dflash-sidecar-cli-explicit");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.5-9b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let entry = dflash_sidecar_entry("qwen35-9b-dflash-mq4.hfq");
        let resolved = resolved_with_dflash_mode("on", None);
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.5:9b"),
            true,
            None,
        )
        .unwrap();
        assert!(params.get("draft").is_none());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_resolves_registry_vision_sidecar_when_present() {
        // `auto` + pulled registry `vision.file` wires `params["vision"]`
        // for the daemon load.
        let paths = test_paths("vision-sidecar-present");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let vision_path = paths.models.join("qwen3.8-27b-vision.hfq");
        fs::write(&vision_path, b"vision").unwrap();
        let entry = vision_sidecar_entry("qwen3.8-27b-vision.hfq");
        let resolved = resolved_with_vision_mode("auto");
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["vision"], vision_path.display().to_string());
        fs::remove_dir_all(&paths.root).unwrap();
    }
    #[test]
    pub(crate) fn resident_vision_mode_projects_and_resolves_like_auto() {
        let paths = test_paths("vision-sidecar-resident");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let entry = vision_sidecar_entry("qwen3.8-27b-vision.hfq");
        let load = |resolved: &hipfire_config::ResolvedConfig| {
            load_params(
                resolved,
                Some(&entry),
                &model_path.parent().unwrap(),
                &model_path,
                64,
                Some("q8"),
                None,
                Some("qwen3.8:27b"),
                false,
                None,
            )
        };
        let resolved = resolved_with_vision_mode("resident");
        let missing = load(&resolved).unwrap();
        assert_eq!(missing["vision_mode"], "resident");
        assert!(
            missing.get("vision").is_none(),
            "missing sidecar stays text-only"
        );
        let vision_path = paths.models.join("qwen3.8-27b-vision.hfq");
        fs::write(&vision_path, b"vision").unwrap();
        let present = load(&resolved).unwrap();
        assert_eq!(present["vision_mode"], "resident");
        assert_eq!(present["vision"], vision_path.display().to_string());
        fs::remove_dir_all(&paths.root).unwrap();
    }
    #[test]
    pub(crate) fn derived_packs_finds_the_tower_pack_beside_the_trunk() {
        // `hipfire quantize --install` must carry the arch-18 `.vl` pack the
        // quantizer wrote beside the trunk: installing the trunk alone left the
        // 1 GB pack in the caller's CWD and produced a text-only install (media
        // refused with no tower), which is what the BYO flow hit.
        let paths = test_paths("derived-packs");
        fs::create_dir_all(&paths.root).unwrap();
        let trunk = paths.root.join("embeddinggemma-2-q8f16.hfq");
        fs::write(&trunk, b"trunk").unwrap();
        assert!(
            derived_packs(&trunk).is_empty(),
            "a trunk with no pack has nothing to carry"
        );
        let pack = paths.root.join("embeddinggemma-2-q8f16.vl");
        fs::write(&pack, b"towers").unwrap();
        assert_eq!(derived_packs(&trunk), vec![pack.clone()]);
        // A second stem's pack must not be picked up.
        fs::write(paths.root.join("other.q8f16"), b"t").unwrap();
        fs::write(paths.root.join("other.vl"), b"p").unwrap();
        assert_eq!(derived_packs(&trunk), vec![pack]);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn embed_load_params_projects_the_vision_ladder() {
        // Regression: `embed` used to load with `{}`, so `vision.mode` never
        // reached the daemon and every image/audio/video input was refused
        // text-only even with `vision_mode=auto` configured. The image/audio
        // harnesses passed because they set `params.vision_mode` themselves.
        let paths = test_paths("embed-vision-ladder");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("embeddinggemma-2");
        fs::create_dir_all(&model_path).unwrap();
        let pack = paths.models.join("embeddinggemma-2.vl");
        fs::write(&pack, b"towers").unwrap();
        let entry = vision_sidecar_entry("embeddinggemma-2.vl");
        for mode in ["auto", "resident", "on"] {
            let params = embed_load_params(
                &resolved_with_vision_mode(mode),
                Some(&entry),
                &paths.models,
                &model_path,
                Some("embeddinggemma-2"),
            )
            .unwrap();
            assert_eq!(params["vision_mode"], mode);
            assert_eq!(
                params["vision"],
                pack.display().to_string(),
                "{mode} must carry the resolved `.vl` pack"
            );
        }
        // `off` forwards the mode and, per `resolve_vision_sidecar`'s contract,
        // still carries the resolved path: the daemon's `vision_mode=off` gate
        // is the hard override and the only place that decides, and it names
        // the declined sidecar in a media refusal.
        let off = embed_load_params(
            &resolved_with_vision_mode("off"),
            Some(&entry),
            &paths.models,
            &model_path,
            Some("embeddinggemma-2"),
        )
        .unwrap();
        assert_eq!(off["vision_mode"], "off");
        // Generation-only policy must not leak into an encoder load: the
        // projected key set is the encoder's own.
        let keys: Vec<&str> = off
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert!(keys.contains(&"vision_mode"), "the ladder must always ride");
        for key in &keys {
            assert!(
                matches!(*key, "vision_mode" | "vision" | "max_seq"),
                "embed_load_params projected a non-encoder key: {key}"
            );
        }
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_xdna_off_by_default_projects_nothing() {
        // Default-off: even a declared AND pulled archive must not surface.
        let paths = test_paths("xdna-sidecar-off-default");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        write_xdna_fixture(&paths.models, "qwen38-27b.xdna.zip");
        let entry = xdna_sidecar_entry("qwen38-27b.xdna.zip");
        let resolved = resolved_with_npu_spillover(false);
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .unwrap();
        assert!(
            params.get("xdna").is_none(),
            "flag-off must not project xdna"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_xdna_on_projects_verified_archive() {
        // Flag on + pulled verified archive → params["xdna"] carries the path
        // for the daemon (which re-verifies and arch-gates into LoadCtx.xdna).
        let paths = test_paths("xdna-sidecar-on-present");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let xdna_path = write_xdna_fixture(&paths.models, "qwen38-27b.xdna.zip");
        let entry = xdna_sidecar_entry("qwen38-27b.xdna.zip");
        let resolved = resolved_with_npu_spillover(true);
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["xdna"], xdna_path.display().to_string());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_xdna_on_fails_closed_when_sidecar_missing() {
        // Flag on + declared but unpulled archive fails with a pull hint —
        // an explicit opt-in must not silently run GPU-only.
        let paths = test_paths("xdna-sidecar-on-missing");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let entry = xdna_sidecar_entry("qwen38-27b.xdna.zip");
        let resolved = resolved_with_npu_spillover(true);
        let error = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .expect_err("missing xdna archive must fail closed");
        let message = format!("{error:#}");
        assert!(message.contains("qwen38-27b.xdna.zip"), "{message}");
        assert!(message.contains("hipfire pull qwen3.8:27b"), "{message}");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_xdna_on_without_entry_stays_gpu_only() {
        // Flag on with a bare (non-registry) artifact logs one line and stays
        // GPU-only instead of failing every load for models with no artifact.
        let paths = test_paths("xdna-sidecar-on-no-entry");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let resolved = resolved_with_npu_spillover(true);
        let params = load_params(
            &resolved,
            None,
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .unwrap();
        assert!(params.get("xdna").is_none());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_skips_vision_sidecar_when_unpulled() {
        // `auto` + declared but unpulled sidecar is silently skipped: vision
        // never gates a text-only load.
        let paths = test_paths("vision-sidecar-absent");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let entry = vision_sidecar_entry("qwen3.8-27b-vision.hfq");
        let resolved = resolved_with_vision_mode("auto");
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .unwrap();
        assert!(params.get("vision").is_none());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_discovers_stem_vision_sibling_beside_trunk() {
        // `auto` with no registry identity at all: `<trunk-stem>-vision.hfq`
        // beside the trunk is still discovered.
        let paths = test_paths("vision-stem-sibling");
        let elsewhere = paths.root.join("artifacts");
        fs::create_dir_all(&elsewhere).unwrap();
        let model_path = elsewhere.join("qwen3.8-27b.mq5");
        fs::write(&model_path, b"model").unwrap();
        let vision_path = elsewhere.join("qwen3.8-27b-vision.hfq");
        fs::write(&vision_path, b"vision").unwrap();
        let resolved = resolved_with_vision_mode("auto");
        let params = load_params(
            &resolved,
            None,
            &paths.models,
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["vision"], vision_path.display().to_string());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn projected_vision_path_wins_over_registry_sidecar() {
        // `run --vision` (projected by the caller after load_params returns)
        // always wins: a preset `params["vision"]` is never re-resolved.
        let paths = test_paths("vision-projected-wins");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        fs::write(paths.models.join("qwen3.8-27b-vision.hfq"), b"vision").unwrap();
        let entry = vision_sidecar_entry("qwen3.8-27b-vision.hfq");
        let mut params = serde_json::json!({"vision_mode": "auto"});
        resolve_vision_sidecar(
            &mut params,
            Some(&entry),
            &paths.models,
            &model_path,
            Some("qwen3.8:27b"),
        )
        .unwrap();
        assert_eq!(
            params["vision"],
            paths
                .models
                .join("qwen3.8-27b-vision.hfq")
                .display()
                .to_string()
        );
        // An explicit override survives a second resolution pass.
        params["vision"] = serde_json::json!("/custom/tower.hfq");
        resolve_vision_sidecar(
            &mut params,
            Some(&entry),
            &paths.models,
            &model_path,
            Some("qwen3.8:27b"),
        )
        .unwrap();
        assert_eq!(params["vision"], "/custom/tower.hfq");
        // Empty string opts out: the key is dropped, never re-resolved.
        params["vision"] = serde_json::json!("");
        resolve_vision_sidecar(
            &mut params,
            Some(&entry),
            &paths.models,
            &model_path,
            Some("qwen3.8:27b"),
        )
        .unwrap();
        assert!(params.get("vision").is_none());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_vision_off_still_forwards_sidecar_for_daemon_gate() {
        // `off` is a hard override: a pulled sidecar is never wired, and an
        // explicitly projected path is stripped, mirroring `dflash_mode=off`.
        let paths = test_paths("vision-mode-off");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        fs::write(paths.models.join("qwen3.8-27b-vision.hfq"), b"vision").unwrap();
        let entry = vision_sidecar_entry("qwen3.8-27b-vision.hfq");
        let resolved = resolved_with_vision_mode("off");
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["vision_mode"], "off");
        // `off` still resolves the sidecar into the params: the daemon's gate
        // is the hard override and needs the path to name what it declined.
        assert_eq!(
            params["vision"],
            paths
                .models
                .join("qwen3.8-27b-vision.hfq")
                .display()
                .to_string()
        );
        // An explicitly projected path is forwarded untouched under `off`.
        let mut params = serde_json::json!({"vision_mode": "off", "vision": "/custom/tower.hfq"});
        resolve_vision_sidecar(
            &mut params,
            Some(&entry),
            &paths.models,
            &model_path,
            Some("qwen3.8:27b"),
        )
        .unwrap();
        assert_eq!(params["vision"], "/custom/tower.hfq");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn bench_load_is_text_only_under_the_default_vision_mode() {
        // The schema default is `auto`, which wires a pulled sidecar into
        // `run`/`serve`; `bench` has no media input and must not pay the tower.
        let paths = test_paths("bench-vision-pin");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        fs::write(paths.models.join("qwen3.8-27b-vision.hfq"), b"vision").unwrap();
        let entry = vision_sidecar_entry("qwen3.8-27b-vision.hfq");
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut params = load_params(
            &defaults,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["vision_mode"], "auto");
        assert!(params["vision"].is_string());
        pin_bench_text_only(&mut params);
        assert_eq!(params["vision_mode"], "off");
        assert!(params.get("vision").is_none());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_vision_on_wires_sidecar_when_present() {
        // `on` + pulled sidecar wires it like `auto`.
        let paths = test_paths("vision-mode-on-present");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let vision_path = paths.models.join("qwen3.8-27b-vision.hfq");
        fs::write(&vision_path, b"vision").unwrap();
        let entry = vision_sidecar_entry("qwen3.8-27b-vision.hfq");
        let resolved = resolved_with_vision_mode("on");
        let params = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["vision_mode"], "on");
        assert_eq!(params["vision"], vision_path.display().to_string());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_vision_on_fails_closed_when_sidecar_missing() {
        // `on` + declared but unpulled sidecar fails closed with a pull hint
        // naming the tag — same shape as the DFlash `on` refusal.
        let paths = test_paths("vision-mode-on-missing");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let entry = vision_sidecar_entry("qwen3.8-27b-vision.hfq");
        let resolved = resolved_with_vision_mode("on");
        let error = load_params(
            &resolved,
            Some(&entry),
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .expect_err("on without a pulled tower must fail closed");
        let message = format!("{error:#}");
        assert!(message.contains("qwen3.8-27b-vision.hfq"), "{message}");
        assert!(message.contains("hipfire pull qwen3.8:27b"), "{message}");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_vision_on_leaves_bare_trunk_alone() {
        // `on` with no declared sidecar never fails: a trunk with an embedded
        // tower is unaffected by this key, so the load proceeds without one.
        let paths = test_paths("vision-mode-on-bare");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&model_path, b"model").unwrap();
        let resolved = resolved_with_vision_mode("on");
        let params = load_params(
            &resolved,
            None,
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            Some("qwen3.8:27b"),
            false,
            None,
        )
        .unwrap();
        assert!(params.get("vision").is_none());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// Minimal in-memory registry for rm tests: (tag, target file, dflash file).
    fn rm_test_registry(entries: &[(&str, &str, Option<&str>)]) -> RegistryV1 {
        let mut models = BTreeMap::new();
        for (tag, file, dflash) in entries {
            models.insert(
                (*tag).to_owned(),
                ModelEntry {
                    repo: "test/repo".into(),
                    file: (*file).to_owned(),
                    size_gb: 1.0,
                    min_vram_gb: 1.0,
                    desc: "rm test".into(),
                    dflash: dflash.map(|draft| hipfire_registry::Sidecar {
                        file: draft.into(),
                        sha256: None,
                        size_bytes: None,
                    }),
                    ..Default::default()
                },
            );
        }
        RegistryV1 {
            schema_version: hipfire_registry::REGISTRY_SCHEMA_VERSION,
            generated_at: "test".into(),
            _comment: None,
            models,
            aliases: BTreeMap::new(),
        }
    }

    #[test]
    fn unknown_model_suggests_close_tags_and_list() {
        let registry = rm_test_registry(&[
            ("qwen3.5:4b", "qwen3.5-4b.mq4", None),
            ("qwen3.5:9b", "qwen3.5-9b.mq4", None),
            ("qwen3.8:27b-mq4", "qwen3.8-27b.mq4", None),
        ]);
        let message = unknown_model_error(&registry, "qwen3.5:4bb").to_string();
        assert!(
            message.starts_with("unknown model 'qwen3.5:4bb'"),
            "{message}"
        );
        assert!(message.contains("did you mean: qwen3.5:4b"), "{message}");
        assert!(message.ends_with("see: hipfire list -r"), "{message}");
        let message = unknown_model_error(&registry, "QWEN35:4B").to_string();
        assert!(message.contains("did you mean: qwen3.5:4b"), "{message}");
        let message = unknown_model_error(&registry, "llama3:8b").to_string();
        assert!(!message.contains("did you mean"), "{message}");
        assert!(message.contains("see: hipfire list -r"), "{message}");
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn model_not_found_names_the_expected_file_and_the_pull_command() {
        let paths = test_paths("model-not-found-tag");
        let registry = rm_test_registry(&[("qwen3.5:4b", "qwen3.5-4b.mq4", None)]);
        let message = model_not_found_error(&paths, &registry, "qwen3.5:4b").to_string();
        let expected = format!(
            "model not found: qwen3.5:4b — expected {}, which is not on disk.\n  fetch it: hipfire pull qwen3.5:4b",
            paths.models.join("qwen3.5-4b.mq4").display()
        );
        assert_eq!(message, expected);
        // The bare file name resolves to the same entry, so the tag is named.
        let message = model_not_found_error(&paths, &registry, "qwen3.5-4b.mq4").to_string();
        assert_eq!(message, expected);
    }

    #[test]
    fn model_not_found_for_a_local_only_entry_says_where_to_place_it() {
        let paths = test_paths("model-not-found-local-only");
        let mut registry = rm_test_registry(&[("local:tune", "local-tune.mq4", None)]);
        registry.models.get_mut("local:tune").unwrap().repo = String::new();
        let message = model_not_found_error(&paths, &registry, "local:tune").to_string();
        let expected = format!(
            "model not found: local:tune — expected {}, which is not on disk.\n  place local-tune.mq4 in {}",
            paths.models.join("local-tune.mq4").display(),
            paths.models.display()
        );
        assert_eq!(message, expected);
        assert!(!message.contains("hipfire pull"), "{message}");
    }

    #[test]
    fn model_not_found_for_a_path_points_at_the_registry_tags() {
        let paths = test_paths("model-not-found-path");
        let registry = rm_test_registry(&[("qwen3.5:4b", "qwen3.5-4b.mq4", None)]);
        for input in ["/nonexistent/dir/model.hfq", "no-separator.hfq"] {
            let message = model_not_found_error(&paths, &registry, input).to_string();
            assert_eq!(
                message,
                format!(
                    "model not found: {input} (no such file)\n  check the path, or use a registry tag: hipfire list -r"
                )
            );
        }
    }

    #[test]
    fn model_not_found_for_an_unknown_tag_keeps_the_suggestions_and_adds_the_pull() {
        let paths = test_paths("model-not-found-unknown");
        let registry = rm_test_registry(&[
            ("qwen3.5:4b", "qwen3.5-4b.mq4", None),
            ("qwen3.5:9b", "qwen3.5-9b.mq4", None),
        ]);
        let message = model_not_found_error(&paths, &registry, "qwen3.5:4bb").to_string();
        assert!(
            message.starts_with("model not found: qwen3.5:4bb"),
            "{message}"
        );
        assert!(message.contains("did you mean: qwen3.5:4b"), "{message}");
        assert!(message.contains("see: hipfire list -r"), "{message}");
        assert!(
            message.ends_with("then: hipfire pull qwen3.5:4b"),
            "{message}"
        );
        let message = model_not_found_error(&paths, &registry, "llama3:8b").to_string();
        assert!(!message.contains("did you mean"), "{message}");
        assert!(!message.contains("then:"), "{message}");
        assert!(message.ends_with("see: hipfire list -r"), "{message}");
    }

    #[test]
    fn daemon_not_found_names_the_build_command_and_the_install_path() {
        let paths = test_paths("daemon-not-found");
        let message = daemon_not_found_error(&paths).to_string();
        assert!(
            message.contains("build `cargo build --release -p hipfire-daemon`"),
            "{message}"
        );
        assert!(
            message.contains(&paths.root.join("bin").display().to_string()),
            "{message}"
        );
    }

    #[test]
    fn rm_of_known_tag_not_downloaded_says_so() {
        let paths = test_paths("rm-not-downloaded");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = rm_test_registry(&[("qwen3.8:27b", "qwen3.8-27b.mq4", None)]);
        let rm = |model: &str| {
            rm_with_registry(
                &paths,
                &registry,
                RmArgs {
                    model: model.into(),
                    yes: true,
                },
            )
            .unwrap_err()
            .to_string()
        };
        assert_eq!(
            rm("qwen3.8:27b"),
            "qwen3.8:27b is not downloaded; nothing to remove"
        );
        assert!(rm("qwen3.8:27bb").contains("did you mean: qwen3.8:27b"));
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn continuous_batch_size_schema_and_cli_share_maximum() {
        let max = hipfire_config::MAX_CONTINUOUS_BATCH_SIZE;
        let field = field("serve.continuous_batch_size").expect("continuous_batch_size schema");
        let hipfire_config::ValueRule::Integer {
            min: schema_min,
            max: schema_max,
        } = field.rule
        else {
            panic!("serve.continuous_batch_size must be an integer rule")
        };
        assert_eq!(schema_min, 1);
        assert_eq!(schema_max, i64::try_from(max).unwrap());
        for lanes in 1..=max {
            field
                .validate(&hipfire_config::ConfigValue::Integer(
                    i64::try_from(lanes).unwrap(),
                ))
                .unwrap_or_else(|err| panic!("schema rejected {lanes}: {err}"));
            let parsed = Cli::try_parse_from([
                "hipfire",
                "serve",
                "--continuous-batch-size",
                &lanes.to_string(),
            ])
            .unwrap_or_else(|err| panic!("CLI rejected {lanes}: {err}"));
            let Some(Commands::Serve(args)) = parsed.command else {
                panic!("expected serve command")
            };
            assert_eq!(args.continuous_batch_size, Some(lanes));
        }
        for lanes in [0, max + 1] {
            assert!(field
                .validate(&hipfire_config::ConfigValue::Integer(
                    i64::try_from(lanes).unwrap()
                ))
                .is_err());
            assert!(Cli::try_parse_from([
                "hipfire",
                "serve",
                "--continuous-batch-size",
                &lanes.to_string(),
            ])
            .is_err());
        }
    }

    #[test]
    fn serve_and_run_accept_spec_and_max_seq() {
        let parsed =
            Cli::try_parse_from(["hipfire", "serve", "--spec", "off", "--max-seq", "8192"])
                .unwrap();
        let Some(Commands::Serve(args)) = parsed.command else {
            panic!("expected serve command")
        };
        assert_eq!(args.speculation.as_deref(), Some("off"));
        assert_eq!(args.max_seq, Some(8192));
        assert!(Cli::try_parse_from(["hipfire", "serve", "--spec", "bogus"]).is_err());
        assert!(Cli::try_parse_from(["hipfire", "serve", "--max-seq", "1"]).is_err());
        let parsed =
            Cli::try_parse_from(["hipfire", "run", "qwen3.5:4b", "--max-seq", "4096", "hi"])
                .unwrap();
        let Some(Commands::Run(args)) = parsed.command else {
            panic!("expected run command")
        };
        assert_eq!(args.max_seq, Some(4096));
        assert!(
            run_should_force_local(&args),
            "--max-seq needs a local load"
        );
    }

    #[test]
    fn rm_keeps_shared_dflash_sidecar_while_sibling_target_present() {
        // The hw-gate regression on PR #686: `qwen3.8:27b`, `qwen3.8:27b-mq4-pro`,
        // and `qwen3.8:27b-mq4-xt` all declare `qwen38-27b-dflash-mq4.hfq`.
        // Removing one target must keep the sidecar while a sibling declarer's
        // target file is still on disk.
        let paths = test_paths("rm-shared-sidecar-kept");
        fs::create_dir_all(&paths.models).unwrap();
        for file in [
            "qwen3.8-27b.mq4",
            "qwen3.8-27b.mq4-pro",
            "qwen38-27b-dflash-mq4.hfq",
        ] {
            fs::write(paths.models.join(file), b"fixture").unwrap();
        }
        let registry = rm_test_registry(&[
            (
                "qwen3.8:27b",
                "qwen3.8-27b.mq4",
                Some("qwen38-27b-dflash-mq4.hfq"),
            ),
            (
                "qwen3.8:27b-mq4-pro",
                "qwen3.8-27b.mq4-pro",
                Some("qwen38-27b-dflash-mq4.hfq"),
            ),
            (
                "qwen3.8:27b-mq4-xt",
                "qwen3.8-27b.mq4-xt",
                Some("qwen38-27b-dflash-mq4.hfq"),
            ),
        ]);
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: "qwen3.8:27b-mq4-pro".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(
            !paths.models.join("qwen3.8-27b.mq4-pro").exists(),
            "removed target is gone"
        );
        assert!(
            paths.models.join("qwen3.8-27b.mq4").exists(),
            "sibling target stays"
        );
        assert!(
            paths.models.join("qwen38-27b-dflash-mq4.hfq").exists(),
            "shared sidecar is kept while a sibling declarer is on disk"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_removes_dflash_sidecar_with_last_declaring_target() {
        // `qwen3.8:27b` still declares the sidecar in the registry, but its
        // target file was never downloaded — a registry row alone must not pin
        // the sidecar once the last on-disk declarer is removed.
        let paths = test_paths("rm-shared-sidecar-last");
        fs::create_dir_all(&paths.models).unwrap();
        for file in ["qwen3.8-27b.mq4-pro", "qwen38-27b-dflash-mq4.hfq"] {
            fs::write(paths.models.join(file), b"fixture").unwrap();
        }
        let registry = rm_test_registry(&[
            (
                "qwen3.8:27b",
                "qwen3.8-27b.mq4",
                Some("qwen38-27b-dflash-mq4.hfq"),
            ),
            (
                "qwen3.8:27b-mq4-pro",
                "qwen3.8-27b.mq4-pro",
                Some("qwen38-27b-dflash-mq4.hfq"),
            ),
        ]);
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: "qwen3.8:27b-mq4-pro".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(
            !paths.models.join("qwen3.8-27b.mq4-pro").exists(),
            "removed target is gone"
        );
        assert!(
            !paths.models.join("qwen38-27b-dflash-mq4.hfq").exists(),
            "sidecar goes with the last on-disk declarer"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_keeps_shared_vision_sidecar_while_sibling_target_present() {
        // Every `qwen3.8:27b*` tier declares `qwen3.8-27b-vision.hfq`.
        // Removing one target must keep the sidecar while a sibling declarer's
        // target file is still on disk (same shared-keeper rule as DFlash).
        let paths = test_paths("rm-shared-vision-kept");
        fs::create_dir_all(&paths.models).unwrap();
        for file in [
            "qwen3.8-27b.mq4",
            "qwen3.8-27b.mq4-pro",
            "qwen3.8-27b-vision.hfq",
        ] {
            fs::write(paths.models.join(file), b"fixture").unwrap();
        }
        let mut registry = rm_test_registry(&[
            ("qwen3.8:27b", "qwen3.8-27b.mq4", None),
            ("qwen3.8:27b-mq4-pro", "qwen3.8-27b.mq4-pro", None),
        ]);
        for entry in registry.models.values_mut() {
            entry.vision = Some(hipfire_registry::Sidecar {
                file: "qwen3.8-27b-vision.hfq".into(),
                sha256: None,
                size_bytes: None,
            });
        }
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: "qwen3.8:27b-mq4-pro".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(
            !paths.models.join("qwen3.8-27b.mq4-pro").exists(),
            "removed target is gone"
        );
        assert!(
            paths.models.join("qwen3.8-27b.mq4").exists(),
            "sibling target stays"
        );
        assert!(
            paths.models.join("qwen3.8-27b-vision.hfq").exists(),
            "shared vision sidecar is kept while a sibling declarer is on disk"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_removes_vision_sidecar_with_last_declaring_target() {
        // A registry row alone must not pin the sidecar once the last
        // on-disk declarer is removed.
        let paths = test_paths("rm-shared-vision-last");
        fs::create_dir_all(&paths.models).unwrap();
        for file in ["qwen3.8-27b.mq4-pro", "qwen3.8-27b-vision.hfq"] {
            fs::write(paths.models.join(file), b"fixture").unwrap();
        }
        let mut registry = rm_test_registry(&[
            ("qwen3.8:27b", "qwen3.8-27b.mq4", None),
            ("qwen3.8:27b-mq4-pro", "qwen3.8-27b.mq4-pro", None),
        ]);
        for entry in registry.models.values_mut() {
            entry.vision = Some(hipfire_registry::Sidecar {
                file: "qwen3.8-27b-vision.hfq".into(),
                sha256: None,
                size_bytes: None,
            });
        }
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: "qwen3.8:27b-mq4-pro".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(
            !paths.models.join("qwen3.8-27b.mq4-pro").exists(),
            "removed target is gone"
        );
        assert!(
            !paths.models.join("qwen3.8-27b-vision.hfq").exists(),
            "vision sidecar goes with the last on-disk declarer"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    fn mtp_entry(file: &str, mtp: Option<&str>) -> ModelEntry {
        ModelEntry {
            repo: "test/repo".into(),
            file: file.into(),
            size_gb: 1.0,
            min_vram_gb: 1.0,
            desc: "mtp test".into(),
            mtp: mtp.map(|head| hipfire_registry::Sidecar {
                file: head.into(),
                sha256: None,
                size_bytes: None,
            }),
            ..Default::default()
        }
    }

    #[cfg(unix)]
    #[test]
    fn mtp_sidecar_resolves_beside_a_symlinked_trunk() {
        // B2: a daemon handed the symlink TARGET (a resolved or canonical trunk
        // path) must still find the head the user placed (or pulled) beside the
        // symlink: via the models dir for a registry tag, via the path as typed
        // for a bare path.
        let paths = test_paths("mtp-symlinked-trunk");
        let store = paths.root.join("store");
        fs::create_dir_all(&paths.models).unwrap();
        fs::create_dir_all(&store).unwrap();
        let target = store.join("h2.group-alpha-refit.hfq");
        fs::write(&target, b"trunk").unwrap();
        let link = paths.models.join("qwen3.8-27b.mq4-xts");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let head = paths.models.join("qwen3.8-27b.mtp");
        fs::write(&head, b"head").unwrap();
        let canonical = fs::canonicalize(&link).unwrap();
        assert_eq!(canonical, fs::canonicalize(&target).unwrap());

        // Registry tag with the declared slot.
        let entry = mtp_entry("qwen3.8-27b.mq4-xts", Some("qwen3.8-27b.mtp"));
        let mut params = serde_json::json!({ "mtp_mode": "auto" });
        resolve_mtp_sidecar(
            &mut params,
            Some(&entry),
            &paths.models,
            &canonical,
            "qwen3.8:27b-mq4-xts",
            Some("qwen3.8:27b-mq4-xts"),
        );
        assert_eq!(params["mtp"], head.display().to_string());

        // Bare path to a symlink outside the models dir, no registry entry.
        let elsewhere = paths.root.join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        let bare_link = elsewhere.join("mine.mq4-xts");
        std::os::unix::fs::symlink(&target, &bare_link).unwrap();
        let bare_head = elsewhere.join("mine.mtp");
        fs::write(&bare_head, b"head").unwrap();
        let mut params = serde_json::json!({ "mtp_mode": "on" });
        resolve_mtp_sidecar(
            &mut params,
            None,
            &paths.models,
            &canonical,
            bare_link.to_str().unwrap(),
            None,
        );
        assert_eq!(params["mtp"], bare_head.display().to_string());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn mtp_sidecar_prefers_models_dir_then_falls_back_beside_the_target() {
        let paths = test_paths("mtp-precedence");
        let store = paths.root.join("store");
        fs::create_dir_all(&paths.models).unwrap();
        fs::create_dir_all(&store).unwrap();
        let target = store.join("qwen3.8-27b.mq4");
        fs::write(&target, b"trunk").unwrap();
        let beside_target = store.join("qwen3.8-27b.mtp");
        fs::write(&beside_target, b"head").unwrap();
        let entry = mtp_entry("qwen3.8-27b.mq4", Some("qwen3.8-27b.mtp"));
        let resolve = |params: &mut serde_json::Value| {
            resolve_mtp_sidecar(
                params,
                Some(&entry),
                &paths.models,
                &target,
                "qwen3.8:27b",
                None,
            )
        };

        // Only the canonical sibling exists: today's loader behaviour is kept.
        let mut params = serde_json::json!({ "mtp_mode": "auto" });
        resolve(&mut params);
        assert_eq!(params["mtp"], beside_target.display().to_string());

        // A pulled head in the models dir wins over the canonical sibling.
        let pulled = paths.models.join("qwen3.8-27b.mtp");
        fs::write(&pulled, b"head").unwrap();
        resolve(&mut params);
        assert_eq!(params["mtp"], pulled.display().to_string());

        // Nothing anywhere: no param, so the loader's own lookup decides.
        fs::remove_file(&pulled).unwrap();
        fs::remove_file(&beside_target).unwrap();
        resolve(&mut params);
        assert!(params.get("mtp").is_none(), "{params}");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn mtp_sidecar_is_never_carried_when_mtp_is_off() {
        let paths = test_paths("mtp-off");
        fs::create_dir_all(&paths.models).unwrap();
        let target = paths.models.join("qwen3.8-27b.mq4");
        fs::write(&target, b"trunk").unwrap();
        fs::write(paths.models.join("qwen3.8-27b.mtp"), b"head").unwrap();
        let entry = mtp_entry("qwen3.8-27b.mq4", Some("qwen3.8-27b.mtp"));
        let resolve = |params: &mut serde_json::Value| {
            resolve_mtp_sidecar(
                params,
                Some(&entry),
                &paths.models,
                &target,
                "qwen3.8:27b",
                None,
            )
        };
        // A selector applied after an earlier resolution (e.g. `run --spec
        // dflash` sets mtp_mode=off) must strip the stale path.
        let mut params = serde_json::json!({ "mtp_mode": "auto" });
        resolve(&mut params);
        assert!(params.get("mtp").is_some());
        apply_speculation_selector(&mut params, "dflash").unwrap();
        resolve(&mut params);
        assert!(params.get("mtp").is_none(), "{params}");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_keeps_shared_mtp_head_until_the_last_declaring_target() {
        // Every `qwen3.8:27b*` tier declares `qwen3.8-27b.mtp`, whose name also
        // matches the `<stem>.mtp` sweep beside the trunk being removed.
        let paths = test_paths("rm-shared-mtp");
        fs::create_dir_all(&paths.models).unwrap();
        for file in ["qwen3.8-27b.mq4", "qwen3.8-27b.mq4-xts", "qwen3.8-27b.mtp"] {
            fs::write(paths.models.join(file), b"fixture").unwrap();
        }
        let mut registry = rm_test_registry(&[
            ("qwen3.8:27b", "qwen3.8-27b.mq4", None),
            ("qwen3.8:27b-mq4-xts", "qwen3.8-27b.mq4-xts", None),
        ]);
        for entry in registry.models.values_mut() {
            entry.mtp = Some(hipfire_registry::Sidecar {
                file: "qwen3.8-27b.mtp".into(),
                sha256: None,
                size_bytes: None,
            });
        }
        let rm = |model: &str| {
            rm_with_registry(
                &paths,
                &registry,
                RmArgs {
                    model: model.into(),
                    yes: true,
                },
            )
            .unwrap()
        };
        rm("qwen3.8:27b-mq4-xts");
        assert!(!paths.models.join("qwen3.8-27b.mq4-xts").exists());
        assert!(
            paths.models.join("qwen3.8-27b.mtp").exists(),
            "shared head is kept while a sibling declarer is on disk"
        );
        rm("qwen3.8:27b");
        assert!(!paths.models.join("qwen3.8-27b.mq4").exists());
        assert!(
            !paths.models.join("qwen3.8-27b.mtp").exists(),
            "head goes with the last on-disk declarer"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_without_dflash_declaration_leaves_draft_file_alone() {
        // A tag with no dflash declaration keeps master behaviour: its target
        // goes, and a draft file it never declared is not an rm target.
        let paths = test_paths("rm-no-dflash");
        fs::create_dir_all(&paths.models).unwrap();
        fs::write(paths.models.join("qwen3.8-27b.mq4"), b"fixture").unwrap();
        fs::write(paths.models.join("qwen38-27b-dflash-mq4.hfq"), b"draft").unwrap();
        let registry = rm_test_registry(&[("qwen3.8:27b", "qwen3.8-27b.mq4", None)]);
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: "qwen3.8:27b".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(
            !paths.models.join("qwen3.8-27b.mq4").exists(),
            "removed target is gone"
        );
        assert!(
            paths.models.join("qwen38-27b-dflash-mq4.hfq").exists(),
            "an undeclared draft file is never an rm target"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_foreign_same_basename_path_removes_only_that_file() {
        // PR #686 hw-gate regression: `hipfire rm /elsewhere/qwen3.6-27b.mq4`
        // basename-matched the `qwen3.6:27b` entry and deleted the installed
        // target plus its sidecars while the installed target stayed. A path
        // that merely shares a basename gets no registry identity: only that
        // file goes.
        let paths = test_paths("rm-foreign-basename");
        fs::create_dir_all(&paths.models).unwrap();
        for file in ["qwen3.6-27b.mq4", "qwen36-27b-dflash-mq4.hfq"] {
            fs::write(paths.models.join(file), b"fixture").unwrap();
        }
        let elsewhere = paths.root.join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        let foreign = elsewhere.join("qwen3.6-27b.mq4");
        fs::write(&foreign, b"lookalike").unwrap();
        let registry = rm_test_registry(&[(
            "qwen3.6:27b",
            "qwen3.6-27b.mq4",
            Some("qwen36-27b-dflash-mq4.hfq"),
        )]);
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: foreign.display().to_string(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!foreign.exists(), "the named foreign file is removed");
        assert!(
            paths.models.join("qwen3.6-27b.mq4").exists(),
            "installed target stays"
        );
        assert!(
            paths.models.join("qwen36-27b-dflash-mq4.hfq").exists(),
            "installed sidecars stay"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_installed_path_removes_target_and_sidecars() {
        // The same removal by installed path keeps master behaviour: target
        // plus declared sidecars go.
        let paths = test_paths("rm-installed-path");
        fs::create_dir_all(&paths.models).unwrap();
        for file in ["qwen3.6-27b.mq4", "qwen36-27b-dflash-mq4.hfq"] {
            fs::write(paths.models.join(file), b"fixture").unwrap();
        }
        let registry = rm_test_registry(&[(
            "qwen3.6:27b",
            "qwen3.6-27b.mq4",
            Some("qwen36-27b-dflash-mq4.hfq"),
        )]);
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: paths.models.join("qwen3.6-27b.mq4").display().to_string(),
                yes: true,
            },
        )
        .unwrap();
        assert!(
            !paths.models.join("qwen3.6-27b.mq4").exists(),
            "installed target is gone"
        );
        assert!(
            !paths.models.join("qwen36-27b-dflash-mq4.hfq").exists(),
            "declared sidecar goes with its target"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn registry_entry_for_path_matches_symlinked_artifact() {
        // `qwen3.8-27b.mq4-xt` is a symlink out of the models dir: the input
        // and the installed entry file canonicalize to the same target, so
        // the entry (and its sidecar) still resolve. A same-basename file
        // elsewhere canonicalizes elsewhere and gets no entry.
        let paths = test_paths("registry-entry-symlink");
        fs::create_dir_all(&paths.models).unwrap();
        let elsewhere = paths.root.join("qcal");
        fs::create_dir_all(&elsewhere).unwrap();
        let real = elsewhere.join("qwen3.8-27b-weights.mq4");
        fs::write(&real, b"weights").unwrap();
        std::os::unix::fs::symlink(&real, paths.models.join("qwen3.8-27b.mq4-xt")).unwrap();
        let foreign_dir = paths.root.join("foreign");
        fs::create_dir_all(&foreign_dir).unwrap();
        let foreign = foreign_dir.join("qwen3.8-27b.mq4-xt");
        fs::write(&foreign, b"lookalike").unwrap();
        let registry = rm_test_registry(&[("qwen3.8:27b-mq4-xt", "qwen3.8-27b.mq4-xt", None)]);
        let installed = paths
            .models
            .join("qwen3.8-27b.mq4-xt")
            .display()
            .to_string();
        let (tag, _) = registry_entry_for_path(&paths, &registry, &installed)
            .expect("symlinked installed artifact must match by canonical target");
        assert_eq!(tag, "qwen3.8:27b-mq4-xt");
        assert!(
            registry_entry_for_path(&paths, &registry, &foreign.display().to_string()).is_none(),
            "same-basename foreign file gets no registry entry"
        );
        assert!(
            registry_entry_for_path(&paths, &registry, "qwen3.8:27b-mq4-xt").is_some(),
            "tag form still resolves"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn load_params_dflash_on_fails_closed_for_non_registry_artifact() {
        // `on` + a path with no registry entry + no explicit draft fails
        // closed with the not-managed message instead of silently running AR
        // (Fable's earlier note on unregistered basenames). `auto` on the
        // same file wires no draft as a bare artifact.
        let paths = test_paths("dflash-on-foreign-path");
        fs::create_dir_all(&paths.models).unwrap();
        let foreign_dir = paths.root.join("elsewhere");
        fs::create_dir_all(&foreign_dir).unwrap();
        let foreign = foreign_dir.join("qwen3.6-27b.mq4");
        fs::write(&foreign, b"lookalike").unwrap();
        assert!(
            registry_entry_for_path(
                &paths,
                &rm_test_registry(&[("qwen3.6:27b", "qwen3.6-27b.mq4", None)]),
                &foreign.display().to_string()
            )
            .is_none(),
            "precondition: foreign path has no entry"
        );
        let resolved = resolved_with_dflash_mode("on", None);
        let error = load_params(
            &resolved,
            None,
            &paths.models,
            &foreign,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .expect_err("on without registry identity must fail closed");
        let message = format!("{error:#}");
        assert!(
            message.contains("not a registry-managed artifact"),
            "{message}"
        );
        assert!(message.contains("--model-draft"), "{message}");
        let resolved = resolved_with_dflash_mode("auto", None);
        let params = load_params(
            &resolved,
            None,
            &paths.models,
            &foreign,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["dflash_mode"], "auto");
        assert!(
            params.get("draft").is_none(),
            "auto on a bare artifact runs AR"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn dflash_on_refuses_without_draft_even_with_mtp_head() {
        let paths = test_paths("dflash-on-no-draft-with-mtp");
        fs::create_dir_all(&paths.models).unwrap();
        let model_path = paths.models.join("qwen3.6-35b-a3b.mq4p");
        fs::write(&model_path, b"trunk").unwrap();
        let head = paths.models.join("qwen3.6-35b-a3b.mtp");
        fs::write(&head, b"head").unwrap();
        let entry = ModelEntry {
            file: "qwen3.6-35b-a3b.mq4p".into(),
            mtp: Some(Sidecar {
                file: "qwen3.6-35b-a3b.mtp".into(),
                sha256: None,
                size_bytes: None,
            }),
            ..Default::default()
        };
        for mtp_mode in ["auto", "on", "off"] {
            let mut params = serde_json::json!({
                "dflash_mode": "on",
                "mtp_mode": mtp_mode,
            });
            let error = resolve_dflash_sidecar(
                &mut params,
                Some(&entry),
                &paths.models,
                &model_path,
                Some("qwen3.6:35b-a3b-mq4p"),
            )
            .expect_err("DFlash on requires a DFlash draft, not an MTP head");
            assert!(error.to_string().contains("--model-draft"), "{error:#}");
            assert_eq!(params["mtp_mode"], mtp_mode);
            assert!(params.get("draft").is_none());
        }
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn run_spec_dflash_projects_inherited_draft_after_config_off() {
        // Reviewer case: resolved config leaves DFlash off, but an inherited
        // developer.dflash_draft is present and `run --spec dflash` re-enables
        // DFlash after load_params. Draft must land on the final load params.
        let draft = "/tmp/qwen35-9b-dflash-mq4.hfq";

        let mut explicit = ConfigLayer::default();
        explicit.set_cli("speculation.mode", "off").unwrap();
        explicit.set_cli("developer.dflash_draft", draft).unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "speculation.mode=off".into(),
            },
            layer: explicit,
        }])
        .unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");

        // load_params alone must not carry the draft while config mode is off.
        let mut params = load_params(
            &resolved,
            None,
            &model_path.parent().unwrap(),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params["dflash_mode"], "off");
        assert!(
            params.get("draft").is_none(),
            "config-off load_params must not project developer.dflash_draft"
        );

        // Final run-path selector: CLI `--spec dflash` then project inherited draft.
        apply_speculation_selector(&mut params, "dflash").unwrap();
        project_dflash_draft(&mut params, developer_dflash_draft(&resolved));
        assert_eq!(params["dflash_mode"], "on");
        assert_eq!(params["draft"], draft);

        // Final off must clear any previously projected draft.
        apply_speculation_selector(&mut params, "off").unwrap();
        project_dflash_draft(&mut params, developer_dflash_draft(&resolved));
        assert_eq!(params["dflash_mode"], "off");
        assert!(
            params.get("draft").is_none(),
            "final off must drop projected developer.dflash_draft"
        );
    }

    #[test]
    pub(crate) fn artifact_urls_honor_endpoint_precedence() {
        struct EnvRestore(&'static str, Option<std::ffi::OsString>);

        impl Drop for EnvRestore {
            fn drop(&mut self) {
                match &self.1 {
                    Some(value) => env::set_var(self.0, value),
                    None => env::remove_var(self.0),
                }
            }
        }

        // `HIPFIRE_HF_BASE` is process-global: the pull tests point it at
        // their local servers under the same lock.
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _hf_base = EnvRestore("HIPFIRE_HF_BASE", env::var_os("HIPFIRE_HF_BASE"));
        let _hf_endpoint = EnvRestore("HF_ENDPOINT", env::var_os("HF_ENDPOINT"));
        let registry = hipfire_registry::bundled().unwrap();
        let (_, entry) = registry.model("qwen3.6:35b-a3b-mq4r").unwrap();
        let suffix = "hipfire-models/qwen3.6-35b-a3b/resolve/main/qwen3.6-35b-a3b.mq4r";

        env::remove_var("HIPFIRE_HF_BASE");
        env::remove_var("HF_ENDPOINT");
        assert_eq!(
            artifact_url(entry, &entry.file),
            format!("https://huggingface.co/{suffix}")
        );

        env::set_var("HF_ENDPOINT", "https://hf-mirror.example/");
        assert_eq!(
            artifact_url(entry, &entry.file),
            format!("https://hf-mirror.example/{suffix}")
        );

        env::set_var("HIPFIRE_HF_BASE", "https://hipfire-mirror.example///");
        assert_eq!(
            artifact_url(entry, &entry.file),
            format!("https://hipfire-mirror.example/{suffix}")
        );
    }

    #[test]
    fn native_help_exposes_migrated_command_families() {
        use clap::CommandFactory;
        let command = Cli::command();
        let names = command
            .get_subcommands()
            .map(|command| command.get_name())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("config"));
        assert!(names.contains("registry"));
        assert!(names.contains("pull"));
        assert!(names.contains("run"));
        assert!(names.contains("chat"));
        assert!(names.contains("serve"));
        assert!(names.contains("stop"));
        assert!(names.contains("restart"));
        assert!(names.contains("bench"));
        assert!(names.contains("profile"));
        assert!(names.contains("version"));
        assert!(names.contains("update"));
        assert!(names.contains("quantize"));
        assert!(names.contains("sidecar-gen"));
    }

    #[test]
    fn build_version_includes_commit_and_ref_identity() {
        use clap::error::ErrorKind;

        let error = Cli::try_parse_from(["hipfire", "--version"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayVersion);
        let rendered = error.to_string();
        assert!(rendered.contains(env!("CARGO_PKG_VERSION")));
        assert!(rendered.contains(BUILD_COMMIT.get(..12).unwrap_or(BUILD_COMMIT)));
        assert!(rendered.contains(BUILD_REF));
    }

    #[test]
    fn update_accepts_branch_tag_commit_and_at_shorthand() {
        let cases = [
            (
                UpdateArgs {
                    reference: Some("@beta".into()),
                    ..UpdateArgs::default()
                },
                RevisionSelector {
                    value: "beta".into(),
                    kind: RevisionKind::Auto,
                },
            ),
            (
                UpdateArgs {
                    reference: Some("@origin/beta".into()),
                    ..UpdateArgs::default()
                },
                RevisionSelector {
                    value: "beta".into(),
                    kind: RevisionKind::Branch,
                },
            ),
            (
                UpdateArgs {
                    tag: Some("v0.3.0".into()),
                    ..UpdateArgs::default()
                },
                RevisionSelector {
                    value: "v0.3.0".into(),
                    kind: RevisionKind::Tag,
                },
            ),
            (
                UpdateArgs {
                    commit: Some("0123456789abcdef".into()),
                    ..UpdateArgs::default()
                },
                RevisionSelector {
                    value: "0123456789abcdef".into(),
                    kind: RevisionKind::Commit,
                },
            ),
        ];
        for (args, expected) in cases {
            assert_eq!(parse_revision_selector(&args).unwrap(), Some(expected));
        }

        let cli = Cli::try_parse_from(["hipfire", "update", "@beta"]).unwrap();
        let Some(Commands::Update(args)) = cli.command else {
            panic!("expected update command");
        };
        assert_eq!(args.reference.as_deref(), Some("@beta"));
    }

    #[test]
    fn update_rejects_unsafe_or_ambiguous_revisions() {
        for value in ["../beta", "-beta", "beta^{tree}", "beta branch"] {
            let args = UpdateArgs {
                reference: Some(value.into()),
                ..UpdateArgs::default()
            };
            assert!(parse_revision_selector(&args).is_err(), "{value}");
        }
        let short_commit = UpdateArgs {
            commit: Some("123".into()),
            ..UpdateArgs::default()
        };
        assert!(parse_revision_selector(&short_commit).is_err());
        let ambiguous = UpdateArgs {
            branch: Some("beta".into()),
            tag: Some("v0.3.0".into()),
            ..UpdateArgs::default()
        };
        assert!(parse_revision_selector(&ambiguous).is_err());
    }

    #[test]
    fn update_fetches_and_checks_out_branch_from_local_origin() {
        fn git(repo: &Path, args: &[&str]) {
            let status = Command::new("git")
                .current_dir(repo)
                .args(args)
                .status()
                .unwrap();
            assert!(status.success(), "git {}", args.join(" "));
        }

        let root = env::temp_dir().join(format!(
            "hipfire-update-ref-test-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        let origin = root.join("origin.git");
        let seed = root.join("seed");
        let installed = root.join("installed");
        fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "--bare", origin.to_str().unwrap()]);
        fs::create_dir_all(&seed).unwrap();
        git(&seed, &["init"]);
        git(&seed, &["config", "user.name", "hipfire test"]);
        git(
            &seed,
            &["config", "user.email", "hipfire-test@example.invalid"],
        );
        fs::write(seed.join("channel"), "master\n").unwrap();
        git(&seed, &["add", "channel"]);
        git(&seed, &["commit", "-m", "master"]);
        git(&seed, &["branch", "-M", "master"]);
        git(
            &seed,
            &["remote", "add", "origin", origin.to_str().unwrap()],
        );
        git(&seed, &["push", "-u", "origin", "master"]);
        git(&seed, &["checkout", "-b", "beta"]);
        fs::write(seed.join("channel"), "beta\n").unwrap();
        git(&seed, &["commit", "-am", "beta"]);
        git(&seed, &["push", "-u", "origin", "beta"]);
        git(
            &root,
            &[
                "clone",
                "--branch",
                "master",
                origin.to_str().unwrap(),
                installed.to_str().unwrap(),
            ],
        );

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Auto,
            },
        )
        .unwrap();
        assert_eq!(resolved.selector.kind, RevisionKind::Branch);
        checkout_revision(&installed, &resolved).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );
        assert_eq!(
            fs::read_to_string(installed.join("channel")).unwrap(),
            "beta\n"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_refuses_branch_with_unpushed_commits() {
        fn git(repo: &Path, args: &[&str]) {
            let status = Command::new("git")
                .current_dir(repo)
                .args(args)
                .status()
                .unwrap();
            assert!(status.success(), "git {}", args.join(" "));
        }

        let root = env::temp_dir().join(format!(
            "hipfire-update-ahead-test-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        let origin = root.join("origin.git");
        let seed = root.join("seed");
        let installed = root.join("installed");
        fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "--bare", origin.to_str().unwrap()]);
        fs::create_dir_all(&seed).unwrap();
        git(&seed, &["init"]);
        git(&seed, &["config", "user.name", "hipfire test"]);
        git(
            &seed,
            &["config", "user.email", "hipfire-test@example.invalid"],
        );
        fs::write(seed.join("channel"), "master\n").unwrap();
        git(&seed, &["add", "channel"]);
        git(&seed, &["commit", "-m", "master"]);
        git(&seed, &["branch", "-M", "master"]);
        git(
            &seed,
            &["remote", "add", "origin", origin.to_str().unwrap()],
        );
        git(&seed, &["push", "-u", "origin", "master"]);
        git(
            &root,
            &[
                "clone",
                "--branch",
                "master",
                origin.to_str().unwrap(),
                installed.to_str().unwrap(),
            ],
        );
        git(&installed, &["config", "user.name", "hipfire test"]);
        git(
            &installed,
            &["config", "user.email", "hipfire-test@example.invalid"],
        );
        fs::write(installed.join("local_only.txt"), "keep-me\n").unwrap();
        git(&installed, &["add", "local_only.txt"]);
        git(&installed, &["commit", "-m", "local-only"]);
        let local_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "master".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        let err = checkout_revision(&installed, &resolved)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("ahead") && err.contains("master"),
            "unexpected error: {err}"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            local_head
        );
        assert_eq!(
            fs::read_to_string(installed.join("local_only.txt")).unwrap(),
            "keep-me\n"
        );
        fs::remove_dir_all(root).unwrap();
    }

    fn update_signal_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn git_test(repo: &Path, args: &[&str]) {
        let status = Command::new("git")
            .current_dir(repo)
            .args(args)
            .status()
            .unwrap();
        assert!(status.success(), "git {}", args.join(" "));
    }

    fn init_update_fixture(label: &str) -> (PathBuf, PathBuf) {
        let root = env::temp_dir().join(format!(
            "hipfire-update-{label}-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        let origin = root.join("origin.git");
        let seed = root.join("seed");
        let installed = root.join("installed");
        fs::create_dir_all(&root).unwrap();
        git_test(&root, &["init", "--bare", origin.to_str().unwrap()]);
        fs::create_dir_all(&seed).unwrap();
        git_test(&seed, &["init"]);
        git_test(&seed, &["config", "user.name", "hipfire test"]);
        git_test(
            &seed,
            &["config", "user.email", "hipfire-test@example.invalid"],
        );
        fs::write(seed.join("channel"), "master\n").unwrap();
        git_test(&seed, &["add", "channel"]);
        git_test(&seed, &["commit", "-m", "master"]);
        git_test(&seed, &["branch", "-M", "master"]);
        git_test(
            &seed,
            &["remote", "add", "origin", origin.to_str().unwrap()],
        );
        git_test(&seed, &["push", "-u", "origin", "master"]);
        git_test(&seed, &["checkout", "-b", "beta"]);
        fs::write(seed.join("channel"), "beta\n").unwrap();
        git_test(&seed, &["commit", "-am", "beta"]);
        git_test(&seed, &["push", "-u", "origin", "beta"]);
        git_test(
            &root,
            &[
                "clone",
                "--branch",
                "master",
                origin.to_str().unwrap(),
                installed.to_str().unwrap(),
            ],
        );
        git_test(&installed, &["config", "user.name", "hipfire test"]);
        git_test(
            &installed,
            &["config", "user.email", "hipfire-test@example.invalid"],
        );
        (root, installed)
    }

    #[test]
    fn update_handoff_forwards_recorded_rocm_root_and_gpu_arch() {
        let home = env::temp_dir().join(format!(
            "hipfire-update-rocm-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("install.json"),
            r#"{"commit":"abc","ref":"master","rocm_root":"/opt/rocm/core-7.14","gpu_arch":"gfx1201","profile":"auto","installed_at":1}"#,
        )
        .unwrap();
        let recorded = recorded_install_metadata(&home);
        assert_eq!(
            recorded.rocm_root.as_deref(),
            Some(Path::new("/opt/rocm/core-7.14"))
        );
        assert_eq!(recorded.gpu_arch.as_deref(), Some("gfx1201"));
        let args = installer_handoff_args(
            &RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
            recorded.rocm_root.as_deref(),
            recorded.gpu_arch.as_deref(),
            recorded.hipcc.as_deref(),
            recorded.strict_rocm,
        );
        assert_eq!(
            args,
            vec![
                "--yes".to_owned(),
                "--branch".to_owned(),
                "beta".to_owned(),
                "--rocm-root".to_owned(),
                "/opt/rocm/core-7.14".to_owned(),
                "--gpu-arch".to_owned(),
                "gfx1201".to_owned(),
            ]
        );

        fs::write(
            home.join("install.json"),
            r#"{"rocm_root":"  ","gpu_arch":"  "}"#,
        )
        .unwrap();
        let empty = recorded_install_metadata(&home);
        assert!(empty.rocm_root.is_none());
        assert!(empty.gpu_arch.is_none());
        assert!(empty.hipcc.is_none());
        assert!(!empty.strict_rocm);
        let bare = installer_handoff_args(
            &RevisionSelector {
                value: "deadbeef".into(),
                kind: RevisionKind::Commit,
            },
            None,
            None,
            None,
            false,
        );
        assert_eq!(
            bare,
            vec![
                "--yes".to_owned(),
                "--commit".to_owned(),
                "deadbeef".to_owned(),
            ]
        );

        // Selector remains before optional install metadata; --yes stays first.
        let arch_only = installer_handoff_args(
            &RevisionSelector {
                value: "master".into(),
                kind: RevisionKind::Auto,
            },
            None,
            Some("gfx1100"),
            None,
            false,
        );
        assert_eq!(
            arch_only,
            vec![
                "--yes".to_owned(),
                "--ref".to_owned(),
                "master".to_owned(),
                "--gpu-arch".to_owned(),
                "gfx1100".to_owned(),
            ]
        );
        fs::remove_dir_all(home).unwrap();
    }
    #[test]
    fn update_handoff_forwards_hipcc_and_strict_with_backward_compat() {
        let home = env::temp_dir().join(format!(
            "hipfire-update-hipcc-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        fs::create_dir_all(&home).unwrap();
        // New format with hipcc and strict_rocm.
        fs::write(
            home.join("install.json"),
            r#"{"rocm_root":"/opt/rocm","hipcc":"/usr/bin/hipcc","strict_rocm":true,"gpu_arch":"gfx1201"}"#,
        )
        .unwrap();
        let recorded = recorded_install_metadata(&home);
        assert_eq!(recorded.hipcc.as_deref(), Some(Path::new("/usr/bin/hipcc")));
        assert!(recorded.strict_rocm);
        assert_eq!(recorded.rocm_root.as_deref(), Some(Path::new("/opt/rocm")));
        let args = installer_handoff_args(
            &RevisionSelector {
                value: "master".into(),
                kind: RevisionKind::Auto,
            },
            recorded.rocm_root.as_deref(),
            recorded.gpu_arch.as_deref(),
            recorded.hipcc.as_deref(),
            recorded.strict_rocm,
        );
        assert!(args.contains(&"--hipcc".to_owned()));
        assert!(args.contains(&"/usr/bin/hipcc".to_owned()));
        assert!(args.contains(&"--strict-rocm".to_owned()));
        assert!(args.contains(&"--rocm-root".to_owned()));
        // Empty/whitespace hipcc is treated as None, like rocm_root.
        fs::write(
            home.join("install.json"),
            r#"{"hipcc":"  ","strict_rocm":false}"#,
        )
        .unwrap();
        let empty = recorded_install_metadata(&home);
        assert!(empty.hipcc.is_none());
        assert!(!empty.strict_rocm);
        let bare = installer_handoff_args(
            &RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
            None,
            None,
            empty.hipcc.as_deref(),
            empty.strict_rocm,
        );
        assert!(!bare.contains(&"--hipcc".to_owned()));
        assert!(!bare.contains(&"--strict-rocm".to_owned()));
        // Older file without hipcc key loads without error (backward compat).
        fs::write(
            home.join("install.json"),
            r#"{"rocm_root":"/opt/rocm","gpu_arch":"gfx1100"}"#,
        )
        .unwrap();
        let old = recorded_install_metadata(&home);
        assert_eq!(old.rocm_root.as_deref(), Some(Path::new("/opt/rocm")));
        assert!(old.hipcc.is_none());
        assert!(!old.strict_rocm);
        // Strict can be stored as string \"1\" or number 1 for compat.
        fs::write(home.join("install.json"), r#"{"strict_rocm":"1"}"#).unwrap();
        assert!(recorded_install_metadata(&home).strict_rocm);
        fs::write(home.join("install.json"), r#"{"strict_rocm":1}"#).unwrap();
        assert!(recorded_install_metadata(&home).strict_rocm);
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn update_restores_staged_unstaged_and_untracked_after_failed_handoff() {
        let (root, installed) = init_update_fixture("index-restore");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();

        // Tracked file with staged + unstaged split, plus untracked work.
        fs::write(installed.join("channel"), "staged-base\n").unwrap();
        git_test(&installed, &["add", "channel"]);
        fs::write(installed.join("channel"), "staged-base\nunstaged-tail\n").unwrap();
        fs::write(installed.join("scratch.txt"), "untracked-user\n").unwrap();

        run_checked(
            Command::new("git").current_dir(&installed).args([
                "stash",
                "push",
                "--include-untracked",
                "-m",
                "hipfire-update-index-test",
            ]),
            "git stash",
        )
        .unwrap();
        let stash_sha = git_output(&installed, &["rev-parse", "stash@{0}"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: Some(stash_sha),
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );

        // Installer dirties the failed target with tracked + untracked junk.
        fs::write(installed.join("channel"), "installer-mutated\n").unwrap();
        fs::write(installed.join("installer-junk.txt"), "leftover\n").unwrap();

        restore_update_checkpoint(&installed, &checkpoint).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );
        assert_eq!(
            fs::read_to_string(installed.join("channel")).unwrap(),
            "staged-base\nunstaged-tail\n"
        );
        assert_eq!(
            fs::read_to_string(installed.join("scratch.txt")).unwrap(),
            "untracked-user\n"
        );
        // Index holds the staged half; worktree holds the full dirty file.
        let cached = git_output(&installed, &["show", ":channel"]).unwrap();
        assert_eq!(cached, "staged-base");
        assert!(!installed.join("installer-junk.txt").exists());
        // Successful --index apply drops the update stash.
        let stash_list = git_output(&installed, &["stash", "list"]).unwrap_or_default();
        assert!(
            stash_list.is_empty(),
            "update stash should be dropped after successful apply: {stash_list}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_installer_mutations_cannot_block_checkout_restore() {
        let (root, installed) = init_update_fixture("dirty-target");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: None,
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();

        // Simulate cargo/installer tracked + untracked mutations on the target.
        fs::write(installed.join("channel"), "lockfile-like-mutation\n").unwrap();
        fs::write(installed.join("target-artifact.bin"), "blob\n").unwrap();

        restore_update_checkpoint(&installed, &checkpoint).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );
        assert_eq!(
            fs::read_to_string(installed.join("channel")).unwrap(),
            "master\n"
        );
        assert!(!installed.join("target-artifact.bin").exists());
        let porcelain = git_output(&installed, &["status", "--porcelain"]).unwrap_or_default();
        assert!(
            porcelain.is_empty(),
            "restored tree should be clean: {porcelain}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_rollback_guard_stays_armed_until_commit() {
        let (root, installed) = init_update_fixture("guard-arm");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: None,
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );

        {
            let mut guard = UpdateRollbackGuard::arm(installed.clone(), checkpoint.clone());
            assert!(guard.is_armed());
            // Drop without commit must restore master.
        }
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );

        // Success path: commit disarms so drop leaves the new revision alone.
        checkout_revision(&installed, &resolved).unwrap();
        {
            let mut guard = UpdateRollbackGuard::arm(installed.clone(), checkpoint);
            assert!(guard.is_armed());
            guard.commit();
            assert!(!guard.is_armed());
        }
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_interrupted_child_is_reaped_while_checkpoint_stays_armed() {
        let _lock = update_signal_test_lock();
        UPDATE_INTERRUPT.store(false, Ordering::SeqCst);

        let (root, installed) = init_update_fixture("interrupt-reap");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: None,
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();

        let mut guard = UpdateRollbackGuard::arm(installed.clone(), checkpoint);
        assert!(guard.is_armed());

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg("trap 'exit 0' TERM; while true; do sleep 0.05; done")
            .current_dir(&installed);
        #[cfg(unix)]
        {
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().unwrap();
        let child_pid = child.id();

        // Arm interrupt after spawn so the wait loop takes the TERM path.
        UPDATE_INTERRUPT.store(true, Ordering::SeqCst);
        let status = wait_update_installer_child(&mut child).unwrap();
        assert!(
            !status.success() || update_interrupted(),
            "interrupted wait should surface cancel state"
        );

        // Child must be reaped (no zombie); try_wait Ok(Some) or Err after wait.
        match child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) => panic!("installer child {child_pid} was not reaped"),
            Err(_) => {}
        }

        // Guard remains armed until explicit fail/drop performs rollback.
        assert!(guard.is_armed());
        let err = guard.fail(anyhow!("update interrupted"));
        assert!(
            err.to_string().contains("update interrupted"),
            "unexpected error: {err}"
        );
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );

        UPDATE_INTERRUPT.store(false, Ordering::SeqCst);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_restores_checkout_and_stash_after_failed_handoff() {
        let (root, installed) = init_update_fixture("restore-basic");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();
        fs::write(installed.join("dirty.txt"), "user-edit\n").unwrap();
        run_checked(
            Command::new("git").current_dir(&installed).args([
                "stash",
                "push",
                "--include-untracked",
                "-m",
                "hipfire-update-test",
            ]),
            "git stash",
        )
        .unwrap();
        let stash_sha = git_output(&installed, &["rev-parse", "stash@{0}"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: Some(stash_sha),
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );

        // Simulate installer handoff failure recovery.
        restore_update_checkpoint(&installed, &checkpoint).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );
        assert_eq!(
            fs::read_to_string(installed.join("dirty.txt")).unwrap(),
            "user-edit\n"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn run_options_after_prompt_and_tui_passthrough_parse() {
        let cli =
            Cli::try_parse_from(["hipfire", "run", "qwen:test", "hello", "--max-tokens", "7"])
                .unwrap();
        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.prompt, ["hello"]);
        assert_eq!(args.max_tokens, Some(7));

        let cli = Cli::try_parse_from(["hipfire", "tui", "--check"]).unwrap();
        let Some(Commands::Tui(args)) = cli.command else {
            panic!("expected tui command");
        };
        assert_eq!(args.arguments, ["--check"]);
    }

    #[test]
    fn registry_system_prompt_is_injected_only_when_client_omits_one() {
        let mut messages = normalize_openai_messages(
            Some(&serde_json::json!([
                { "role": "user", "content": "hello" }
            ])),
            false,
        );
        inject_default_system_message(&mut messages, Some("registry identity"));
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "registry identity");

        let mut messages = normalize_openai_messages(
            Some(&serde_json::json!([
                { "role": "developer", "content": "client policy" },
                { "role": "user", "content": "hello" }
            ])),
            false,
        );
        inject_default_system_message(&mut messages, Some("registry identity"));
        assert_eq!(messages.as_array().unwrap().len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "client policy");
    }

    #[test]
    fn normalize_reasoning_sources_with_flag_on_and_off() {
        // reasoning field takes precedence over reasoning_content and inline think
        let body = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "<think>inline</think>\nvisible",
                "reasoning": "explicit reasoning",
                "reasoning_content": "secondary"
            }]
        });
        let off = normalize_openai_messages(body.get("messages"), false);
        assert_eq!(off[0]["content"], "visible");
        assert_eq!(off[0]["tool_plan"], "explicit reasoning");
        assert!(off[0].get("reasoning_content").is_none());
        let on = normalize_openai_messages(body.get("messages"), true);
        assert_eq!(on[0]["content"], "visible");
        assert_eq!(on[0]["tool_plan"], "explicit reasoning");
        assert_eq!(on[0]["reasoning_content"], "explicit reasoning");
        assert_eq!(on[0]["reasoning_content"], on[0]["tool_plan"]);

        // reasoning_content when reasoning absent
        let body2 = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "visible only",
                "reasoning_content": "from content field"
            }]
        });
        let off2 = normalize_openai_messages(body2.get("messages"), false);
        assert_eq!(off2[0]["tool_plan"], "from content field");
        assert!(off2[0].get("reasoning_content").is_none());
        let on2 = normalize_openai_messages(body2.get("messages"), true);
        assert_eq!(on2[0]["reasoning_content"], "from content field");
        assert_eq!(on2[0]["tool_plan"], "from content field");
        assert_eq!(on2[0]["content"], "visible only");

        // inline <think> when neither reasoning field present
        let body3 = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "<think>inline think</think>\n\nvisible answer"
            }]
        });
        let off3 = normalize_openai_messages(body3.get("messages"), false);
        assert_eq!(off3[0]["content"], "visible answer");
        assert_eq!(off3[0]["tool_plan"], "inline think");
        assert!(off3[0].get("reasoning_content").is_none());
        let on3 = normalize_openai_messages(body3.get("messages"), true);
        assert_eq!(on3[0]["content"], "visible answer");
        assert_eq!(on3[0]["tool_plan"], "inline think");
        assert_eq!(on3[0]["reasoning_content"], "inline think");
        assert_eq!(on3[0]["reasoning_content"], on3[0]["tool_plan"]);
    }

    #[test]
    fn include_reasoning_content_arch_predicate() {
        assert!(include_reasoning_content(Some("muse_glimmer")));
        assert!(include_reasoning_content(Some("qwen35")));
        assert!(include_reasoning_content(Some("qwen35-vl")));
        assert!(include_reasoning_content(Some("qwen36")));
        assert!(include_reasoning_content(Some("Qwen3.6-A3B")));
        assert!(include_reasoning_content(Some("qwen3.5")));
        assert!(include_reasoning_content(Some("qwen4")));
        assert!(!include_reasoning_content(Some("deepseek4")));
        assert!(!include_reasoning_content(Some("llama")));
        assert!(!include_reasoning_content(Some("gemma4")));
        assert!(!include_reasoning_content(Some("qwen2")));
        assert!(!include_reasoning_content(None));
    }

    #[test]
    fn normalize_reasoning_emitted_for_qwen_include_flag() {
        let body = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "visible",
                "reasoning_content": "qwen chain of thought"
            }]
        });
        let include = include_reasoning_content(Some("qwen35"));
        assert!(include);
        let on = normalize_openai_messages(body.get("messages"), include);
        assert_eq!(on[0]["reasoning_content"], "qwen chain of thought");
        assert_eq!(on[0]["tool_plan"], "qwen chain of thought");
        assert_eq!(on[0]["content"], "visible");

        let off = normalize_openai_messages(
            body.get("messages"),
            include_reasoning_content(Some("llama")),
        );
        assert!(off[0].get("reasoning_content").is_none());
        assert_eq!(off[0]["tool_plan"], "qwen chain of thought");
    }

    #[test]
    fn normalize_tool_call_id_and_tool_result_name_survive() {
        let body = serde_json::json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [{
                        "id": "call_42",
                        "type": "function",
                        "function": { "name": "my_tool", "arguments": "{}" }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_42",
                    "name": "my_tool",
                    "content": "result"
                }
            ]
        });
        for flag in [false, true] {
            let normalized = normalize_openai_messages(body.get("messages"), flag);
            assert_eq!(normalized[0]["tool_calls"][0]["id"], "call_42");
            assert_eq!(normalized[0]["tool_calls"][0]["name"], "my_tool");
            assert_eq!(normalized[1]["tool_call_id"], "call_42");
            assert_eq!(normalized[1]["name"], "my_tool");
            assert_eq!(normalized[1]["content"], "result");
        }
    }

    #[test]
    fn normalize_glimmer_flag_rejects_non_object_arguments_string() {
        let body = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "x",
                "tool_calls": [{
                    "function": { "name": "t", "arguments": "not-json" }
                }]
            }]
        });
        let off = normalize_openai_messages(body.get("messages"), false);
        assert_eq!(
            off[0]["tool_calls"][0]["arguments"],
            serde_json::json!({ "_raw": "not-json" })
        );
        let on = normalize_openai_messages(body.get("messages"), true);
        assert_eq!(
            on[0]["tool_calls"][0]["arguments"],
            serde_json::Value::String("not-json".into())
        );
        // JSON string that parses to non-object (array) also surfaces as string under glimmer
        let body_arr = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "x",
                "tool_calls": [{
                    "function": { "name": "t", "arguments": "[1,2]" }
                }]
            }]
        });
        let on_arr = normalize_openai_messages(body_arr.get("messages"), true);
        assert_eq!(
            on_arr[0]["tool_calls"][0]["arguments"],
            serde_json::Value::String("[1,2]".into())
        );
        let off_arr = normalize_openai_messages(body_arr.get("messages"), false);
        // non-glimmer keeps parsed array (today's behaviour is to keep whatever parsed)
        assert_eq!(
            off_arr[0]["tool_calls"][0]["arguments"],
            serde_json::json!([1, 2])
        );
    }

    #[test]
    fn positional_model_config_scope_parses_without_stealing_global_actions() {
        let global = Cli::try_parse_from(["hipfire", "config", "list", "--json"]).unwrap();
        let Some(Commands::Config(global)) = global.command else {
            panic!("expected config command")
        };
        assert!(global.model.is_none());
        assert!(matches!(global.action, Some(ConfigAction::List(_))));

        let model =
            Cli::try_parse_from(["hipfire", "config", "qwen:test", "get", "memory.kv_cache"])
                .unwrap();
        let Some(Commands::Config(model)) = model.command else {
            panic!("expected config command")
        };
        assert_eq!(model.model.as_deref(), Some("qwen:test"));
        assert!(matches!(model.action, Some(ConfigAction::Get { .. })));

        let schema = Cli::try_parse_from(["hipfire", "config", "schema", "--json"]).unwrap();
        let Some(Commands::Config(schema)) = schema.command else {
            panic!("expected config command")
        };
        assert!(schema.model.is_none());
        assert!(matches!(
            schema.action,
            Some(ConfigAction::Schema(OutputArgs { json: true }))
        ));
    }

    #[test]
    fn config_profile_set_and_create_parse_as_dedicated_actions() {
        let set = Cli::try_parse_from(["hipfire", "config", "profile", "set", "dev"]).unwrap();
        let Some(Commands::Config(args)) = set.command else {
            panic!("expected config command")
        };
        assert!(args.model.is_none());
        assert!(matches!(
            args.action,
            Some(ConfigAction::Profile {
                action: Some(ConfigProfileAction::Set { ref name })
            }) if name == "dev"
        ));

        let create =
            Cli::try_parse_from(["hipfire", "config", "profile", "create", "lab"]).unwrap();
        let Some(Commands::Config(args)) = create.command else {
            panic!("expected config command")
        };
        assert!(matches!(
            args.action,
            Some(ConfigAction::Profile {
                action: Some(ConfigProfileAction::Create { ref name })
            }) if name == "lab"
        ));

        let bare = Cli::try_parse_from(["hipfire", "config", "profile"]).unwrap();
        let Some(Commands::Config(args)) = bare.command else {
            panic!("expected config command")
        };
        assert!(matches!(
            args.action,
            Some(ConfigAction::Profile { action: None })
        ));
    }

    #[test]
    fn config_profile_helpers_merge_layer_and_are_global_only() {
        assert_eq!(CONFIG_PROFILE_NAMES, &["default", "dev", "hip", "redline"]);
        let root = env::temp_dir().join(format!("hipfire-cli-profile-{}", std::process::id()));
        let config_paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer
            .set(
                "generation.temperature",
                hipfire_config::ConfigValue::Float(0.5),
            )
            .unwrap();
        let set = apply_config_profile(&mut layer, &config_paths, "redline").unwrap();
        let redline = hipfire_config::load_config_profile(&config_paths, "redline").unwrap();
        assert_eq!(set, redline.values.len());
        assert_eq!(
            layer.get("generation.temperature"),
            Some(&hipfire_config::ConfigValue::Float(0.5)),
            "profile merge keeps unrelated keys"
        );
        assert_eq!(layer.values.len(), redline.values.len() + 1);
        assert_eq!(
            layer.get("replay.backend"),
            Some(&hipfire_config::ConfigValue::String("redline".into()))
        );

        let model = Cli::try_parse_from([
            "hipfire",
            "config",
            "qwen:test",
            "profile",
            "set",
            "default",
        ])
        .unwrap();
        let Some(Commands::Config(args)) = model.command else {
            panic!("expected config command")
        };
        assert_eq!(args.model.as_deref(), Some("qwen:test"));
        assert!(matches!(
            args.action,
            Some(ConfigAction::Profile {
                action: Some(ConfigProfileAction::Set { .. })
            })
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn schema_json_preserves_default_types_and_validation_rules() {
        let bool_field = field("hardware.allow_mixed_arch").unwrap();
        assert_eq!(
            config_default_value(bool_field),
            hipfire_config::ConfigValue::Bool(false)
        );
        assert_eq!(config_rule_json(bool_field.rule)["type"], "boolean");

        let variant_field = field("diagnostic.kernel.rdna2_variant").unwrap();
        assert_eq!(
            config_default_value(variant_field),
            hipfire_config::ConfigValue::Null
        );
        assert_eq!(config_rule_json(variant_field.rule)["minimum"], 1);
        assert_eq!(config_rule_json(variant_field.rule)["maximum"], 5);
    }

    fn sample_completion(
        content: &str,
        tool_calls: Vec<ToolCall>,
        finish_reason: &str,
    ) -> Completion {
        Completion {
            id: "chatcmpl_test".into(),
            created: 42,
            model: "qwen:test".into(),
            content: content.into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls,
            done: serde_json::json!({
                "finish_reason": finish_reason,
                "prompt_tokens": 3,
                "tokens": 5,
                "cached_tokens": 1,
                "tok_s": 10.0,
            }),
            logprobs: None,
            reasoning: None,
        }
    }

    fn sample_tc(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: None,
            name: name.into(),
            arguments,
            rendered_body: None,
        }
    }

    /// Build a Completion whose done envelope has a non-string/missing finish_reason.
    fn sample_completion_with_done(
        content: &str,
        tool_calls: Vec<ToolCall>,
        done: serde_json::Value,
    ) -> Completion {
        Completion {
            id: "chatcmpl_test".into(),
            created: 42,
            model: "qwen:test".into(),
            content: content.into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls,
            done,
            logprobs: None,
            reasoning: None,
        }
    }

    fn sample_tool_call(name: &str) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "arguments": { "path": "README.md" }
        })
    }

    fn task15_daemon_err(class: &str, retryable: bool, attempt_id: u64) -> anyhow::Error {
        anyhow::Error::new(hipfire_client::ClientError::Daemon(
            hipfire_client::TypedDaemonError {
                message: format!("t15 {class}"),
                class: class.to_owned(),
                retryable,
                rolled_back: false,
                attempt_id,
                id: Some("req-t15".into()),
            },
        ))
    }

    #[test]
    fn task15_serve_retry_config_defaults_off() {
        let resolved = resolve(Vec::<NamedLayer>::new()).expect("resolve empty layers");
        let enabled = config_bool(&resolved, "serve.retry_enabled").expect("retry_enabled");
        let backoff = config_u64(&resolved, "serve.retry_backoff_ms").expect("retry_backoff_ms");
        assert!(!enabled, "serve.retry_enabled must default false");
        assert_eq!(backoff, 50);
    }

    // --- StreamContractGate / complete_request framing (fix round 2) ---

    // ── Task 6: canonical OpenAI tool-call adapter + endpoint registry ──

    #[test]
    fn forward_think_fragments_preserves_cancelled_callback_error() {
        let mut content = String::new();
        let mut reasoning = String::new();
        let err = forward_think_fragments(
            vec![ThinkFragment::Content("x".into())],
            &mut content,
            &mut reasoning,
            &mut |_| Err(hipfire_client::ClientError::Cancelled),
        )
        .expect_err("callback Cancelled must surface typed");
        assert!(matches!(err, hipfire_client::ClientError::Cancelled));
        // Fragment still applied before callback failure (accumulation is local).
        assert_eq!(content, "x");
    }

    // =========================================================================
    // Task 11 — no-GPU fake-daemon HTTP acceptance through real serve lowering
    // =========================================================================

    /// Unix-only JSONL fake daemon used by the Task 11 HTTP matrix.
    /// Scenario selection is driven by generate request prompt/model fixture tags.
    #[cfg(unix)]
    fn write_task11_fake_daemon(root: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let daemon = root.join("task11-fake-daemon.py");
        // Python keeps correlated id/attempt_id, full reset ack, and commit handshake.
        let script = include_str!("serve/fake_daemon.py");
        fs::write(&daemon, script).unwrap();
        fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
        daemon
    }

    /// In-process Hyper harness: Engine::spawn_configured → serve_listener_until.
    /// Does not touch HIPFIRE_DAEMON_BIN.
    #[cfg(unix)]
    struct Task11HttpHarness {
        paths: Paths,
        port: u16,
        model_name: String,
        shared: Arc<ServeShared>,
        shutdown: tokio_util::sync::CancellationToken,
        _join: Option<thread::JoinHandle<()>>,
    }

    #[cfg(unix)]
    impl Task11HttpHarness {
        fn spawn(label: &str) -> Self {
            Self::spawn_inner(label, false, Duration::from_millis(0), None)
        }

        /// Retry-enabled variant for the Task 15 one-retry scenarios.
        fn spawn_with_retry(label: &str, retry_backoff: Duration) -> Self {
            Self::spawn_inner(label, true, retry_backoff, None)
        }

        /// Keyed server: every route but the minimal `/health` needs `api_key`.
        fn spawn_with_api_key(label: &str, api_key: hipfire_config::secret::BearerToken) -> Self {
            Self::spawn_inner(label, false, Duration::from_millis(0), Some(api_key))
        }

        fn spawn_inner(
            label: &str,
            retry_enabled: bool,
            retry_backoff: Duration,
            api_key: Option<hipfire_config::secret::BearerToken>,
        ) -> Self {
            let paths = test_paths(label);
            fs::create_dir_all(&paths.models).unwrap();
            fs::create_dir_all(&paths.root).unwrap();

            let model_name = format!("t11-fixture-{label}.hfq");
            let model_path = paths.models.join(&model_name);
            fs::write(&model_path, b"task11-dummy-model").unwrap();

            let daemon = write_task11_fake_daemon(&paths.root);
            let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
            let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();

            // Bounded ETXTBSY retry like hipfire-client fake daemons.
            const ETXTBSY: i32 = 26;
            let mut engine = None;
            let mut last = None;
            for attempt in 0..8 {
                match Engine::spawn_configured(&daemon, &BTreeMap::new(), &process_config, None) {
                    Ok(e) => {
                        engine = Some(e);
                        break;
                    }
                    Err(hipfire_client::ClientError::Spawn { source, path })
                        if source.raw_os_error() == Some(ETXTBSY) =>
                    {
                        last = Some(format!("spawn {path:?}: {source}"));
                        thread::sleep(Duration::from_millis(
                            5u64.saturating_mul(1 + attempt as u64),
                        ));
                    }
                    Err(err) => panic!("Task11HttpHarness spawn non-retryable: {err}"),
                }
            }
            let mut engine = engine.unwrap_or_else(|| {
                panic!(
                    "Task11HttpHarness exhausted ETXTBSY retries: {}",
                    last.unwrap_or_default()
                )
            });
            engine.ping().expect("fake daemon ping");

            let registry = hipfire_registry::bundled().unwrap();
            let shared = Arc::new(ServeShared {
                capabilities: crate::serve::route_capabilities(
                    false,
                    4,
                    8192,
                    1024,
                    false,
                    0,
                    false,
                    4096,
                    1,
                    64,
                    268435456,
                    30000,
                    None,
                    64 << 20,
                    false,
                ),
                metrics: crate::serve::metrics::Metrics::default(),
                runtime: Mutex::new(ServeRuntime {
                    engine,
                    paths: paths.clone(),
                    registry,
                    current_path: None,
                    current_arch: None,
                    current_reasoning_contract: ReasoningContract::Unsupported,
                    current_reasoning_effort_native: false,
                    current_reasoning_efforts: Vec::new(),
                    continuous_batch_capable: false,
                    continuous_batch_history: false,
                    current_max_seq: 0,
                    cache_capable: false,
                    kv_override: None,
                    kv_k_override: None,
                    kv_v_override: None,
                    kv_backend_override: None,
                    vision_override: None,
                    tp: None,
                    speculation_override: None,
                    max_seq_override: None,
                    continuous_batch_size: 1,
                    vmm_batch: crate::serve::VmmBatchFlags::default(),
                    multi_slot_enabled: false,
                    multi_slot_slots: 4,
                    multi_slot_ctx: 8192,
                    multi_slot_prefill_chunk: 1024,
                    spawner: crate::serve::EngineSpawner {
                        daemon: daemon.clone(),
                        process_config: process_config.clone(),
                        attempts: 3,
                        backoff: Duration::from_millis(10),
                    },
                    resident_model: None,
                    request_policy: crate::serve::RequestModelPolicy {
                        allow_pull: false,
                        allow_paths: false,
                        operator_model: None,
                    },
                    max_batch_tokens: 4096,
                }),
                meta: Mutex::new(ServeMeta::new(serve_instance_token())),
                max_request_bytes: 8 * 1024 * 1024,
                admission: Arc::new(Admission::new(4, Duration::from_secs(5))),
                idle_timeout: Duration::from_secs(0),
                retry_enabled,
                retry_backoff,
                backoff_hook: Mutex::new(None),
                stream_stall_timeout: None,
                ui_enabled: false,
                api_key,
            });

            let std_listener =
                std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral serve port");
            std_listener
                .set_nonblocking(true)
                .expect("set nonblocking listener");
            let port = std_listener
                .local_addr()
                .expect("listener local addr")
                .port();

            let shutdown = tokio_util::sync::CancellationToken::new();
            let shutdown_loop = shutdown.clone();
            let shared_loop = Arc::clone(&shared);
            let join = thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("task11 runtime");
                rt.block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(std_listener)
                        .expect("tokio listener from std");
                    if let Err(error) = crate::serve::http::serve_listener_until(
                        listener,
                        shared_loop,
                        shutdown_loop,
                    )
                    .await
                    {
                        eprintln!("[task11-harness] serve failed: {error:#}");
                    }
                });
            });

            // Health probe — proves production Hyper service path is live.
            let endpoint = ServiceEndpoint::local("127.0.0.1", port, None).unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                if hipfire_client::service_ready(&endpoint, Duration::from_millis(200)) {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            assert!(
                hipfire_client::service_ready(&endpoint, Duration::from_millis(500)),
                "task11 harness never became ready on port {port}"
            );

            Self {
                paths,
                port,
                // Bare file name, resolved under `paths.models` like an
                // installed model; request-named paths are refused by default.
                model_name,
                shared,
                shutdown,
                _join: Some(join),
            }
        }

        fn port(&self) -> u16 {
            self.port
        }

        fn model(&self) -> &str {
            &self.model_name
        }

        fn base_body(&self, scenario_tag: &str, stream: bool) -> serde_json::Value {
            // Encode scenario in both model (direct path still resolves file) and
            // user prompt so the fake daemon can select without external deps.
            serde_json::json!({
                "model": self.model(),
                "stream": stream,
                "messages": [{
                    "role": "user",
                    "content": format!("{scenario_tag} please")
                }],
            })
        }

        fn tools_body(&self, scenario_tag: &str, stream: bool) -> serde_json::Value {
            let mut body = self.base_body(scenario_tag, stream);
            body["tools"] = serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read a file",
                    "parameters": {
                        "type": "object",
                        "properties": { "path": { "type": "string" } }
                    }
                }
            }]);
            body
        }

        fn requests_log_path(&self) -> PathBuf {
            self.paths.root.join("requests.log")
        }

        fn read_requests_log(&self) -> Vec<serde_json::Value> {
            let raw = fs::read_to_string(self.requests_log_path()).unwrap_or_default();
            raw.lines()
                .filter(|line| !line.trim().is_empty())
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect()
        }

        fn meta_retries(&self) -> (u64, u64) {
            let meta = self
                .shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            (meta.retries_attempted, meta.retries_succeeded)
        }

        fn set_backoff_hook<F>(&self, hook: F)
        where
            F: Fn(Duration) + Send + Sync + 'static,
        {
            let mut slot = self
                .shared
                .backoff_hook
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            *slot = Some(Arc::new(hook));
        }

        fn ops_of_type<'a>(log: &'a [serde_json::Value], ty: &str) -> Vec<&'a serde_json::Value> {
            log.iter()
                .filter(|row| row.get("type").and_then(|v| v.as_str()) == Some(ty))
                .collect()
        }
    }

    #[cfg(unix)]
    impl Drop for Task11HttpHarness {
        fn drop(&mut self) {
            self.shutdown.cancel();
            if let Some(join) = self._join.take() {
                let _ = join.join();
            }
            // Dropping Engine (inside ServeShared via Arc) kills the fake child.
            // ServeShared is held only by the server thread which has exited.
            let _ = fs::remove_dir_all(&self.paths.root);
        }
    }

    #[cfg(unix)]
    #[test]
    fn inference_router_cannot_shutdown_serve() {
        let harness = Task11HttpHarness::spawn("management-isolation");
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .proxy(None)
            .http_status_as_error(false)
            .build()
            .into();
        let response = agent
            .post(&format!(
                "http://127.0.0.1:{}/api/v1/shutdown",
                harness.port()
            ))
            .header("Authorization", "Bearer unrelated-management-token")
            .send_empty()
            .unwrap();
        assert_eq!(response.status().as_u16(), 404);
        assert!(!harness.shutdown.is_cancelled());
        assert_eq!(serve_health(harness.port()).0, 200);
    }

    // ── 0.4.0 serve lifecycle: max_tokens fit, respawn, request policy ──

    #[cfg(unix)]
    fn serve_health(port: u16) -> (u16, serde_json::Value) {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(5)))
            .http_status_as_error(false)
            .build()
            .into();
        let mut response = agent
            .get(&format!("http://127.0.0.1:{port}/health"))
            .call()
            .expect("GET /health");
        let status = response.status().as_u16();
        let text = response.body_mut().read_to_string().expect("/health body");
        let body = serde_json::from_str(&text).expect("/health JSON");
        (status, body)
    }

    #[cfg(unix)]
    fn post_status(port: u16, body: &serde_json::Value) -> (u16, serde_json::Value, String) {
        let (status, bytes) = raw_nonstream_post(port, body);
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let json = text
            .find('{')
            .and_then(|start| serde_json::from_str(&text[start..]).ok())
            .unwrap_or(serde_json::Value::Null);
        (status, json, text)
    }

    /// Writes an owner-only API key file; returns its path and the token text.
    #[cfg(unix)]
    fn write_test_api_key(label: &str) -> (PathBuf, String) {
        use std::os::unix::fs::OpenOptionsExt;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!(
            "hipfire-auth-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("api.key");
        let token = format!("hipfire-auth-test-token-{label}-0123456789abcdef");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .unwrap();
        file.write_all(format!("{token}\n").as_bytes()).unwrap();
        (path, token)
    }

    /// One request with an optional raw `Authorization` header value; returns
    /// the status, the `WWW-Authenticate` challenge, and the body text.
    #[cfg(unix)]
    fn auth_call(
        port: u16,
        method: &str,
        path: &str,
        authorization: Option<&str>,
        body: &str,
    ) -> (u16, Option<String>, String) {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(10)))
            .http_status_as_error(false)
            .build()
            .into();
        let mut request = ureq::http::Request::builder()
            .method(method)
            .uri(format!("http://127.0.0.1:{port}{path}"))
            .header("Content-Type", "application/json");
        if let Some(value) = authorization {
            request = request.header("Authorization", value);
        }
        let body = if method == "POST" {
            body.to_owned()
        } else {
            String::new()
        };
        let mut response = agent
            .run(request.body(body).unwrap())
            .unwrap_or_else(|error| panic!("{method} {path}: {error}"));
        let status = response.status().as_u16();
        let challenge = response
            .headers()
            .get("www-authenticate")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let text = response.body_mut().read_to_string().unwrap_or_default();
        (status, challenge, text)
    }

    #[cfg(unix)]
    #[test]
    fn serve_api_key_gates_every_route_and_redacts_public_health() {
        let (key_path, token) = write_test_api_key("gate");
        let bearer = hipfire_config::secret::BearerToken::load(&key_path).unwrap();
        let h = Task11HttpHarness::spawn_with_api_key("api-key-gate", bearer);
        let port = h.port();
        let chat = h.base_body("t11-stop-text", false).to_string();
        let good = format!("Bearer {token}");
        let lower = format!("bearer {token}");
        let wrong = format!("Bearer {token}x");
        let basic = format!("Basic {token}");
        let debug = format!("{:?}", h.shared.api_key);
        assert!(!debug.contains(&token), "{debug}");

        // Unauthenticated and wrong-credential /health: status and version only.
        for credential in [None, Some(wrong.as_str())] {
            let (status, challenge, text) = auth_call(port, "GET", "/health", credential, "");
            assert_eq!(status, 200, "{text}");
            assert_eq!(challenge, None);
            let json: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(json.as_object().unwrap().len(), 2, "{text}");
            assert_eq!(json["status"], crate::serve::EngineState::Up.as_str());
            assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
            assert!(!text.contains(&token), "{text}");
        }

        // Every other route, including the UI namespace and unknown paths,
        // answers one uniform 401 whatever the credential defect.
        let routes = [
            ("GET", "/stats"),
            ("GET", "/metrics"),
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/messages"),
            ("POST", "/v1/responses"),
            ("POST", "/v1/embeddings"),
            ("POST", "/v1/images/generations"),
            ("POST", "/v1/systemone"),
            ("GET", "/ui"),
            ("GET", "/"),
            ("GET", "/no-such-route"),
            ("OPTIONS", "/v1/chat/completions"),
            ("POST", "/health"),
            ("GET", "/config"),
        ];
        for (method, path) in routes {
            let mut bodies = Vec::new();
            for credential in [
                None,
                Some(wrong.as_str()),
                Some(basic.as_str()),
                Some("Bearer"),
            ] {
                let (status, challenge, text) = auth_call(port, method, path, credential, &chat);
                assert_eq!(status, 401, "{method} {path} {credential:?}: {text}");
                assert_eq!(challenge.as_deref(), Some("Bearer"), "{method} {path}");
                assert!(text.is_empty(), "{method} {path}: {text}");
                assert!(!text.contains(&token), "{method} {path}: {text}");
                bodies.push(text);
            }
            assert!(
                bodies.windows(2).all(|pair| pair[0] == pair[1]),
                "{method} {path} 401 bodies differ: {bodies:?}"
            );
        }
        // Reject before waiting for the advertised body or applying its limit.
        let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        write!(
            socket,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 999999999\r\n\r\n"
        ).unwrap();
        let mut status_line = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(socket), &mut status_line)
            .unwrap();
        assert!(status_line.starts_with("HTTP/1.1 401"), "{status_line}");
        let log = h.read_requests_log();
        assert!(Task11HttpHarness::ops_of_type(&log, "generate").is_empty());
        assert!(Task11HttpHarness::ops_of_type(&log, "load").is_empty());

        // The right token (scheme is case-insensitive) opens the routes.
        for credential in [good.as_str(), lower.as_str()] {
            for path in ["/stats", "/metrics", "/v1/models"] {
                let (status, challenge, text) = auth_call(port, "GET", path, Some(credential), "");
                assert_eq!(status, 200, "{path}: {text}");
                assert_eq!(challenge, None);
            }
        }
        let (status, _, text) = auth_call(port, "POST", "/v1/chat/completions", Some(&good), &chat);
        assert_eq!(status, 200, "{text}");
        assert!(!text.contains(&token), "{text}");

        // Authenticated /health keeps the full payload; its instance token is
        // a different secret from the API key.
        let (status, _, text) = auth_call(port, "GET", "/health", Some(&good), "");
        assert_eq!(status, 200, "{text}");
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(json["pid"].is_u64(), "{text}");
        assert!(json["capabilities"].is_object(), "{text}");
        assert_ne!(json["token"], serde_json::json!(token));
        assert!(!text.contains(&token), "{text}");

        // Keyed preflight is gated too; authenticated native clients can use it.
        let (status, _, text) = auth_call(port, "OPTIONS", "/v1/chat/completions", Some(&good), "");
        assert_eq!(status, 204, "{text}");

        let _ = fs::remove_dir_all(key_path.parent().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn serve_without_api_key_keeps_routes_and_full_health_open() {
        let h = Task11HttpHarness::spawn("api-key-none");
        let port = h.port();
        for credential in [None, Some("Bearer ignored-by-a-keyless-server")] {
            let (status, challenge, text) = auth_call(port, "GET", "/stats", credential, "");
            assert_eq!(status, 200, "{text}");
            assert_eq!(challenge, None);
        }
        let health = serve_health(port).1;
        assert!(health["pid"].is_u64(), "{health}");
        assert!(health["capabilities"].is_object(), "{health}");
        let (status, _, text) = post_status(port, &h.base_body("t11-stop-text", false));
        assert_eq!(status, 200, "{text}");
    }

    /// An omitted max_tokens (config default 4096 against the fake's 4096
    /// context) used to reload the model on every request, because the reload
    /// cannot grow max_seq. Now it is fitted by the daemon, and an explicit
    /// budget that cannot fit is a 400 with no reload.
    #[cfg(unix)]
    #[test]
    fn serve_omitted_max_tokens_fits_without_reload() {
        let h = Task11HttpHarness::spawn("fit-max-tokens");
        let body = h.base_body("t11-stop-text", false);
        for _ in 0..2 {
            let (status, _, text) = post_status(h.port(), &body);
            assert_eq!(status, 200, "{text}");
        }
        let mut explicit = body.clone();
        explicit["max_tokens"] = serde_json::json!(8000);
        let (status, _, text) = post_status(h.port(), &explicit);
        assert_eq!(status, 400, "{text}");
        assert!(text.contains("max_tokens=8000"), "{text}");
        let mut fits = body.clone();
        fits["max_tokens"] = serde_json::json!(100);
        let (status, _, text) = post_status(h.port(), &fits);
        assert_eq!(status, 200, "{text}");

        let log = h.read_requests_log();
        assert_eq!(Task11HttpHarness::ops_of_type(&log, "load").len(), 1);
        let generates = Task11HttpHarness::ops_of_type(&log, "generate");
        assert_eq!(generates.len(), 3);
        assert!(generates[..2]
            .iter()
            .all(|g| g["max_tokens_fit"] == serde_json::json!(true)));
        assert_eq!(generates[2]["max_tokens"], serde_json::json!(100));
        assert!(generates[2].get("max_tokens_fit").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn serve_max_seq_override_reaches_model_load() {
        let h = Task11HttpHarness::spawn("max-seq-override");
        h.shared.runtime.lock().unwrap().max_seq_override = Some(65_536);

        let (status, _, text) = post_status(h.port(), &h.base_body("t11-stop-text", false));
        assert_eq!(status, 200, "{text}");

        let second_model = "t11-fixture-max-seq-override-second.hfq";
        fs::write(h.paths.models.join(second_model), b"task11-dummy-model").unwrap();
        let mut second_body = h.base_body("t11-stop-text", false);
        second_body["model"] = serde_json::json!(second_model);
        let (status, _, text) = post_status(h.port(), &second_body);
        assert_eq!(status, 200, "{text}");

        let log = h.read_requests_log();
        let loads = Task11HttpHarness::ops_of_type(&log, "load");
        assert_eq!(loads.len(), 2);
        assert!(loads.iter().all(|load| load["params"]["max_seq"] == 65_536));
    }

    /// A request may not load a file outside the installed models, nor start
    /// a registry download; an installed model still loads.
    #[cfg(unix)]
    #[test]
    fn serve_refuses_request_file_paths_and_downloads() {
        let h = Task11HttpHarness::spawn("request-policy");
        let mut body = h.base_body("t11-stop-text", false);
        body["model"] = serde_json::json!("/etc/hostname");
        let (status, _, text) = post_status(h.port(), &body);
        assert_eq!(status, 404, "{text}");
        assert!(text.contains("serve.allow_request_paths"), "{text}");
        body["model"] = serde_json::json!("qwen3.5:0.8b");
        let (status, _, text) = post_status(h.port(), &body);
        assert_eq!(status, 404, "{text}");
        assert!(text.contains("hipfire pull qwen3.5:0.8b"), "{text}");
        assert!(Task11HttpHarness::ops_of_type(&h.read_requests_log(), "load").is_empty());

        let (status, _, text) = post_status(h.port(), &h.base_body("t11-stop-text", false));
        assert_eq!(status, 200, "{text}");
    }

    /// HA-13: a turn the model ended inside `<think>` reaches the client as
    /// reasoning only, `finish_reason: "stop"`, and no error, both streamed
    /// and non-streamed.
    #[cfg(unix)]
    #[test]
    fn reasoning_only_stop_streams_and_completes_without_error() {
        let harness = Task11HttpHarness::spawn("open-think-stop");
        let port = harness.port();
        let reasoning = "Schedule: \"0 8 * * *\" — every day at 8 AM.\n\nNow let's create the job.";

        let cap = capture_stream(port, harness.base_body("t11-open-think-stop", true))
            .expect("stream ends without an error event");
        assert_eq!(cap.reasoning, reasoning);
        assert!(cap.content.is_empty(), "no answer: {:?}", cap.content);
        assert!(cap.tool_calls.is_empty());
        assert_eq!(cap.finish.as_deref(), Some("stop"));
        assert!(cap.saw_done);

        let json = complete_nonstream(port, harness.base_body("t11-open-think-stop", false))
            .expect("non-stream completes with 200");
        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "stop");
        assert_eq!(choice["message"]["reasoning_content"], reasoning);
        assert!(
            choice["message"]["content"].is_null() || choice["message"]["content"] == "",
            "{json}"
        );
        assert!(choice["message"].get("tool_calls").is_none(), "{json}");
    }

    /// After a failed switch the daemon holds nothing (tp<=1 unloads first):
    /// `/health` must stop naming the old model and the next request for it
    /// must reload instead of generating against an empty daemon.
    #[cfg(unix)]
    #[test]
    fn serve_failed_model_switch_clears_residency() {
        let h = Task11HttpHarness::spawn("switch-fail");
        let (status, _, text) = post_status(h.port(), &h.base_body("t11-stop-text", false));
        assert_eq!(status, 200, "{text}");
        let health = serve_health(h.port()).1;
        assert_eq!(health["model"], serde_json::json!(h.model()));
        assert_eq!(
            health["n_ctx"],
            serde_json::json!(4096),
            "the load ack's max_seq"
        );

        fs::write(h.paths.models.join("t20-load-fail.hfq"), b"x").unwrap();
        let mut bad = h.base_body("t11-stop-text", false);
        bad["model"] = serde_json::json!("t20-load-fail.hfq");
        let (status, _, text) = post_status(h.port(), &bad);
        assert_ne!(status, 200, "{text}");
        let health = serve_health(h.port()).1;
        assert!(health["model"].is_null());
        assert!(
            health["n_ctx"].is_null(),
            "no context is advertised with nothing resident"
        );

        let (status, _, text) = post_status(h.port(), &h.base_body("t11-stop-text", false));
        assert_eq!(status, 200, "{text}");
        let loads = Task11HttpHarness::ops_of_type(&h.read_requests_log(), "load").len();
        assert_eq!(
            loads, 3,
            "initial load, failed switch, reload of the old model"
        );
    }

    /// A daemon that exits (crash, or a sticky GPU fault after which it exits
    /// 75) is respawned and the next request is served.
    #[cfg(unix)]
    #[test]
    fn serve_respawns_a_daemon_that_exited() {
        let h = Task11HttpHarness::spawn("respawn");
        let ok = h.base_body("t11-stop-text", false);
        assert_eq!(post_status(h.port(), &ok).0, 200);
        for fault in ["t11-premature-eof", "t20-gpu-poison"] {
            let (status, _, text) = post_status(h.port(), &h.base_body(fault, false));
            assert_ne!(status, 200, "{fault}: {text}");
            let (status, _, text) = post_status(h.port(), &ok);
            assert_eq!(status, 200, "after {fault}: {text}");
        }
        let log = h.read_requests_log();
        assert_eq!(Task11HttpHarness::ops_of_type(&log, "configure").len(), 3);
        assert_eq!(Task11HttpHarness::ops_of_type(&log, "load").len(), 3);
        assert_eq!(serve_health(h.port()).0, 200);
    }

    /// `/health` is 503 while the daemon is dead and cannot be respawned, and
    /// the supervisor brings it back with the resident model reloaded.
    #[cfg(unix)]
    #[test]
    fn serve_health_is_unhealthy_until_the_daemon_is_back() {
        let h = Task11HttpHarness::spawn("respawn-health");
        assert_eq!(
            post_status(h.port(), &h.base_body("t11-stop-text", false)).0,
            200
        );
        let daemon = {
            let mut runtime = h.shared.runtime.lock().unwrap();
            std::mem::replace(
                &mut runtime.spawner.daemon,
                h.paths.root.join("missing-daemon"),
            )
        };
        let (status, _, text) = post_status(h.port(), &h.base_body("t11-premature-eof", false));
        assert_ne!(status, 200, "{text}");

        assert!(crate::serve::supervise_engine(&h.shared).is_err());
        let (status, health) = serve_health(h.port());
        assert_eq!(status, 503);
        assert_eq!(health["status"], "unhealthy");

        h.shared.runtime.lock().unwrap().spawner.daemon = daemon;
        crate::serve::supervise_engine(&h.shared).unwrap();
        let (status, health) = serve_health(h.port());
        assert_eq!(status, 200);
        assert_eq!(health["status"], "ok");
        assert_eq!(health["model"], serde_json::json!(h.model()));
        assert_eq!(
            post_status(h.port(), &h.base_body("t11-stop-text", false)).0,
            200
        );
        // The supervisor reloaded the model; the request did not load again.
        let loads = Task11HttpHarness::ops_of_type(&h.read_requests_log(), "load").len();
        assert_eq!(loads, 2);
    }

    /// A failed pre-warm leaves single-slot serve up to load on the next
    /// request, but fails multi-slot serve closed: `/health` goes 503
    /// `unhealthy` (never 200 `ok` with every request failing), the daemon is
    /// stopped, and the load error comes back for serve to exit on.
    #[cfg(unix)]
    #[test]
    fn serve_prewarm_failure_fails_closed_only_on_multi_slot() {
        let single = Task11HttpHarness::spawn("prewarm-fail-single");
        fs::write(single.paths.models.join("t20-load-fail.hfq"), b"x").unwrap();
        {
            let mut runtime = single.shared.runtime.lock().unwrap();
            crate::serve::prewarm(&mut runtime, &single.shared.meta, "t20-load-fail.hfq").unwrap();
        }
        let (status, health) = serve_health(single.port());
        assert_eq!(status, 200);
        assert_eq!(health["status"], "ok");
        assert!(health["model"].is_null());
        let ok = single.base_body("t11-stop-text", false);
        assert_eq!(post_status(single.port(), &ok).0, 200);

        let multi = Task11HttpHarness::spawn("prewarm-fail-multi");
        fs::write(multi.paths.models.join("t20-load-fail.hfq"), b"x").unwrap();
        let mut runtime = multi.shared.runtime.lock().unwrap();
        runtime.multi_slot_enabled = true;
        let error = crate::serve::prewarm(&mut runtime, &multi.shared.meta, "t20-load-fail.hfq")
            .unwrap_err();
        assert!(runtime.engine.exited(), "the daemon is stopped");
        drop(runtime);
        let message = format!("{error:#}");
        assert!(
            message.contains("multi-slot pre-warm of t20-load-fail.hfq failed")
                && message.contains("load failed: fixture"),
            "{message}"
        );
        let (status, health) = serve_health(multi.port());
        assert_eq!(status, 503);
        assert_eq!(health["status"], "unhealthy");
        assert!(health["model"].is_null());
        assert!(health["loading_model"].is_null());
    }

    /// The reload after a daemon respawn follows the same rule: single-slot
    /// serve comes back up without the model, multi-slot serve fails closed.
    #[cfg(unix)]
    #[test]
    fn serve_reload_failure_after_respawn_fails_closed_only_on_multi_slot() {
        for multi_slot in [false, true] {
            let h = Task11HttpHarness::spawn(&format!("reload-fail-{multi_slot}"));
            fs::write(h.paths.models.join("t20-load-fail.hfq"), b"x").unwrap();
            {
                let mut runtime = h.shared.runtime.lock().unwrap();
                runtime.multi_slot_enabled = multi_slot;
                runtime.resident_model = Some("t20-load-fail.hfq".to_owned());
                runtime.engine.terminate();
            }
            let supervised = crate::serve::supervise_engine(&h.shared);
            let (status, health) = serve_health(h.port());
            assert!(health["model"].is_null());
            if multi_slot {
                let message = format!("{:#}", supervised.unwrap_err());
                assert!(
                    message.contains(
                        "multi-slot reload of t20-load-fail.hfq after the daemon restart failed"
                    ),
                    "{message}"
                );
                assert_eq!(status, 503);
                assert_eq!(health["status"], "unhealthy");
            } else {
                supervised.unwrap();
                assert_eq!(status, 200);
                assert_eq!(health["status"], "ok");
            }
        }
    }

    /// DeepSeek V4 (legacy contract) stages tool calls only on the terminal;
    /// they must reach the client (#593).
    #[cfg(unix)]
    #[test]
    fn serve_legacy_staged_tool_calls_reach_the_client() {
        let h = Task11HttpHarness::spawn("legacy-tools");
        let (status, json, text) =
            post_status(h.port(), &h.tools_body("t20-legacy-staged-tool", false));
        assert_eq!(status, 200, "{text}");
        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{text}");
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["name"], "read_file",
            "{text}"
        );
    }

    /// Gemma 4 request: the OpenAI `tools` shape for `get_temperature`.
    #[cfg(unix)]
    fn gemma4_tool_body(h: &Task11HttpHarness, stream: bool) -> serde_json::Value {
        let mut body = h.base_body("t22-gemma4-tool", stream);
        body["tools"] = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "current temperature in a city",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } }
                }
            }
        }]);
        body
    }

    /// Gemma 4 (legacy contract, two-phase terminal) must reach an HTTP
    /// body with content, finish reason and usage on a non-stream request.
    #[cfg(unix)]
    #[test]
    fn gemma4_legacy_nonstream_returns_body_with_usage() {
        let h = Task11HttpHarness::spawn("gemma4-nonstream");
        let (status, json, text) = post_status(h.port(), &h.base_body("t22-gemma4-stop", false));
        assert_eq!(status, 200, "{text}");
        let choice = &json["choices"][0];
        assert_eq!(
            choice["message"]["content"], "391\n\nExplanation.",
            "{text}"
        );
        assert_eq!(choice["finish_reason"], "stop", "{text}");
        assert_eq!(json["usage"]["prompt_tokens"], 31, "{text}");
        assert_eq!(json["usage"]["completion_tokens"], 5, "{text}");
        assert_eq!(json["usage"]["total_tokens"], 36, "{text}");
    }

    /// Streaming ends with content deltas, a finish chunk, the usage chunk
    /// (`choices: []`) and `[DONE]`.
    #[cfg(unix)]
    #[test]
    fn gemma4_legacy_stream_ends_with_usage_chunk() {
        let h = Task11HttpHarness::spawn("gemma4-stream");
        let mut body = h.base_body("t22-gemma4-stop", true);
        body["stream_options"] = serde_json::json!({ "include_usage": true });

        // Raw wire order: finish chunk, then usage chunk, then [DONE].
        let (status, bytes) = raw_nonstream_post(h.port(), &body);
        let wire = String::from_utf8(bytes).expect("SSE body is UTF-8");
        assert_eq!(status, 200, "{wire}");
        let chunks: Vec<serde_json::Value> = wire
            .split("\n\n")
            .filter_map(|frame| frame.trim().strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).expect("SSE data is JSON"))
            .collect();
        assert!(
            wire.trim_end().ends_with("data: [DONE]"),
            "stream must end with [DONE]: {wire}"
        );
        let finish_idx = chunks
            .iter()
            .position(|c| c["choices"][0]["finish_reason"] == "stop")
            .unwrap_or_else(|| panic!("no finish_reason=stop chunk: {wire}"));
        let usage_chunk = chunks.last().expect("usage chunk");
        assert!(
            finish_idx + 1 < chunks.len(),
            "usage follows finish: {wire}"
        );
        assert_eq!(usage_chunk["choices"], serde_json::json!([]), "{wire}");
        assert_eq!(usage_chunk["usage"]["prompt_tokens"], 31, "{wire}");
        assert_eq!(usage_chunk["usage"]["completion_tokens"], 5, "{wire}");

        let cap = capture_stream(h.port(), body).expect("gemma4 stream completes");
        assert!(cap.saw_role);
        assert_eq!(cap.content, "391\n\nExplanation.");
        assert!(!cap.content_deltas.is_empty());
        assert_eq!(cap.finish.as_deref(), Some("stop"));
        let usage = cap.usage.expect("usage chunk delivered");
        assert_eq!(usage["prompt_tokens"], 31);
        assert_eq!(usage["completion_tokens"], 5);
        assert_eq!(usage["total_tokens"], 36);
        assert!(cap.saw_done);
    }

    #[cfg(unix)]
    #[test]
    fn gemma4_legacy_length_is_reported() {
        let h = Task11HttpHarness::spawn("gemma4-length");
        let (status, json, text) = post_status(h.port(), &h.base_body("t22-gemma4-length", false));
        assert_eq!(status, 200, "{text}");
        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "length", "{text}");
        assert_eq!(
            choice["message"]["content"], "391\n\nExplanation.",
            "{text}"
        );
        assert_eq!(json["usage"]["completion_tokens"], 5, "{text}");
    }

    /// Gemma 4 tool calls ride the `commit_ready`/`done` terminal and are
    /// routed to the client as structured OpenAI tool_calls (non-stream and
    /// stream), with usage that includes the cached prefix.
    #[cfg(unix)]
    #[test]
    fn gemma4_legacy_tool_call_is_routed() {
        let h = Task11HttpHarness::spawn("gemma4-tools");
        let (status, json, text) = post_status(h.port(), &gemma4_tool_body(&h, false));
        assert_eq!(status, 200, "{text}");
        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{text}");
        // Pure tool turn: content is JSON null (serve convention).
        assert!(choice["message"]["content"].is_null(), "{text}");
        let calls = choice["message"]["tool_calls"]
            .as_array()
            .expect("tool_calls");
        assert_eq!(calls.len(), 1, "{text}");
        assert_eq!(calls[0]["type"], "function", "{text}");
        assert_eq!(calls[0]["function"]["name"], "get_temperature", "{text}");
        let arguments: serde_json::Value = serde_json::from_str(
            calls[0]["function"]["arguments"]
                .as_str()
                .expect("arguments string"),
        )
        .expect("arguments are JSON");
        assert_eq!(arguments, serde_json::json!({ "city": "Oslo" }));
        assert_eq!(json["usage"]["prompt_tokens"], 88, "{text}");
        assert_eq!(
            json["usage"]["prompt_tokens_details"]["cached_tokens"], 28,
            "{text}"
        );

        // The daemon received the tools array.
        let generate = last_generate_without_ids(&h);
        let tools = generate["tools"]
            .as_array()
            .expect("generate carries tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "get_temperature");

        let mut stream_body = gemma4_tool_body(&h, true);
        stream_body["stream_options"] = serde_json::json!({ "include_usage": true });
        let cap = capture_stream(h.port(), stream_body).expect("gemma4 tool stream completes");
        assert!(
            cap.content.is_empty(),
            "no content on a pure tool turn: {:?}",
            cap.content
        );
        assert_eq!(cap.tool_calls.len(), 1, "{:?}", cap.tool_calls);
        let (index, _id, name, arguments) = &cap.tool_calls[0];
        assert_eq!(*index, 0);
        assert_eq!(name.as_deref(), Some("get_temperature"));
        let streamed: serde_json::Value =
            serde_json::from_str(arguments.as_deref().expect("streamed arguments"))
                .expect("streamed arguments are JSON");
        assert_eq!(streamed, serde_json::json!({ "city": "Oslo" }));
        assert_eq!(cap.finish.as_deref(), Some("tool_calls"));
        let usage = cap.usage.expect("usage chunk delivered");
        assert_eq!(usage["prompt_tokens"], 88);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 28);
        assert!(cap.saw_done);
    }

    #[cfg(unix)]
    #[derive(Debug, Default)]
    struct StreamCapture {
        content: String,
        reasoning: String,
        /// Individual content deltas (for per-chunk leak assertions).
        content_deltas: Vec<String>,
        /// Individual reasoning deltas (for per-chunk leak assertions).
        reasoning_deltas: Vec<String>,
        tool_calls: Vec<(u32, Option<String>, Option<String>, Option<String>)>,
        finish: Option<String>,
        usage: Option<serde_json::Value>,
        saw_done: bool,
        saw_role: bool,
    }

    /// Tool-call protocol markers that must never appear in valid-path content/reasoning.
    #[cfg(unix)]
    const TASK11_TOOL_PROTOCOL_MARKERS: &[&str] = &[
        "<tool_call>",
        "</tool_call>",
        "<tool_calls>",
        "</tool_calls>",
        "<|tool_call|>",
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "call tool",
        "invoke tool",
    ];

    /// Assert visible text from a *valid structured-call* path has zero protocol
    /// markers and zero JSON argument fragments belonging to structured calls.
    #[cfg(unix)]
    fn assert_valid_path_text_clean(label: &str, text: &str, forbidden_arg_frags: &[&str]) {
        for marker in TASK11_TOOL_PROTOCOL_MARKERS {
            assert!(
                !text.contains(marker),
                "{label}: content/reasoning leaked tool protocol marker {marker:?} in {text:?}"
            );
        }
        for frag in forbidden_arg_frags {
            if frag.is_empty() {
                continue;
            }
            assert!(
                !text.contains(frag),
                "{label}: content/reasoning leaked structured-call argument fragment {frag:?} in {text:?}"
            );
        }
    }

    #[cfg(unix)]
    fn assert_nonstream_valid_structured_clean(
        label: &str,
        json: &serde_json::Value,
        forbidden_arg_frags: &[&str],
    ) {
        let message = &json["choices"][0]["message"];
        match message.get("content") {
            None | Some(serde_json::Value::Null) => {}
            Some(serde_json::Value::String(content)) => {
                assert_valid_path_text_clean(
                    &format!("{label}/nonstream.content"),
                    content,
                    forbidden_arg_frags,
                );
            }
            Some(other) => panic!("{label}: unexpected content shape {other}"),
        }
        if let Some(reasoning) = message
            .get("reasoning_content")
            .and_then(serde_json::Value::as_str)
        {
            assert_valid_path_text_clean(
                &format!("{label}/nonstream.reasoning"),
                reasoning,
                forbidden_arg_frags,
            );
        }
        // Calls/arguments may appear only under message.tool_calls.
        if let Some(calls) = message
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
        {
            assert!(
                !calls.is_empty(),
                "{label}: empty tool_calls array is not a structured release"
            );
            for call in calls {
                assert!(
                    call.get("function").and_then(|f| f.get("name")).is_some(),
                    "{label}: structured tool_calls entry missing function.name"
                );
            }
        }
    }

    #[cfg(unix)]
    fn assert_stream_valid_structured_clean(
        label: &str,
        cap: &StreamCapture,
        forbidden_arg_frags: &[&str],
    ) {
        assert_valid_path_text_clean(
            &format!("{label}/stream.content"),
            &cap.content,
            forbidden_arg_frags,
        );
        assert_valid_path_text_clean(
            &format!("{label}/stream.reasoning"),
            &cap.reasoning,
            forbidden_arg_frags,
        );
        for (i, delta) in cap.content_deltas.iter().enumerate() {
            assert_valid_path_text_clean(
                &format!("{label}/stream.content_delta[{i}]"),
                delta,
                forbidden_arg_frags,
            );
        }
        for (i, delta) in cap.reasoning_deltas.iter().enumerate() {
            assert_valid_path_text_clean(
                &format!("{label}/stream.reasoning_delta[{i}]"),
                delta,
                forbidden_arg_frags,
            );
        }
    }

    #[cfg(unix)]
    fn capture_stream(
        port: u16,
        body: serde_json::Value,
    ) -> std::result::Result<StreamCapture, hipfire_client::ClientError> {
        let mut cap = StreamCapture::default();
        stream_openai_chat(
            &ServiceEndpoint::local("127.0.0.1", port, None)?,
            body,
            Duration::from_secs(10),
            |event| {
                match event {
                    OpenAiSseEvent::Role { .. } => cap.saw_role = true,
                    OpenAiSseEvent::Content { text } => {
                        cap.content_deltas.push(text.clone());
                        cap.content.push_str(&text);
                    }
                    OpenAiSseEvent::Reasoning { text } => {
                        cap.reasoning_deltas.push(text.clone());
                        cap.reasoning.push_str(&text);
                    }
                    OpenAiSseEvent::ToolCall {
                        index,
                        id,
                        name,
                        arguments,
                    } => cap.tool_calls.push((index, id, name, arguments)),
                    OpenAiSseEvent::Finish { reason, .. } => cap.finish = Some(reason),
                    OpenAiSseEvent::Usage { usage } => cap.usage = Some(usage),
                    OpenAiSseEvent::Done => cap.saw_done = true,
                    OpenAiSseEvent::PrefillProgress { .. } => {}
                }
                Ok(())
            },
            || false,
        )?;
        Ok(cap)
    }

    #[cfg(unix)]
    fn complete_nonstream(
        port: u16,
        body: serde_json::Value,
    ) -> std::result::Result<serde_json::Value, hipfire_client::ClientError> {
        complete_openai_chat(
            &ServiceEndpoint::local("127.0.0.1", port, None)?,
            body,
            Duration::from_secs(10),
        )
    }

    /// Raw non-stream POST; returns HTTP status and body bytes (no client parse).
    #[cfg(unix)]
    fn raw_nonstream_post(port: u16, body: &serde_json::Value) -> (u16, Vec<u8>) {
        raw_post(port, "/v1/chat/completions", body)
    }

    /// Raw `POST path` with `Connection: close`: HTTP status and the
    /// de-chunked body (a whole SSE stream for streaming requests).
    #[cfg(unix)]
    fn raw_post(port: u16, path: &str, body: &serde_json::Value) -> (u16, Vec<u8>) {
        use std::net::TcpStream;
        let payload = body.to_string();
        let request = format!(
            "POST {path} HTTP/1.1\r\n\
             Host: 127.0.0.1:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {payload}",
            payload.len(),
        );
        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect serve");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read timeout");
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .expect("write timeout");
        stream.write_all(request.as_bytes()).expect("write request");
        let mut buf = Vec::new();
        let _ = stream.read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf);
        let status = text
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok())
            .unwrap_or(0);
        let body_start = text
            .find("\r\n\r\n")
            .map(|idx| idx + 4)
            .or_else(|| text.find("\n\n").map(|idx| idx + 2))
            .unwrap_or(buf.len());
        let raw_body = buf.get(body_start..).unwrap_or(&[]);
        let chunked = text[..body_start]
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked");
        if !chunked {
            return (status, raw_body.to_vec());
        }
        let mut decoded = Vec::new();
        let mut remaining = raw_body;
        loop {
            let Some(line_end) = remaining.windows(2).position(|bytes| bytes == b"\r\n") else {
                panic!("malformed chunk header");
            };
            let size = usize::from_str_radix(
                std::str::from_utf8(&remaining[..line_end])
                    .expect("chunk size utf8")
                    .split(';')
                    .next()
                    .expect("chunk size"),
                16,
            )
            .expect("chunk size hex");
            remaining = &remaining[line_end + 2..];
            if size == 0 {
                break;
            }
            assert!(remaining.len() >= size + 2, "truncated chunk body");
            decoded.extend_from_slice(&remaining[..size]);
            assert_eq!(&remaining[size..size + 2], b"\r\n");
            remaining = &remaining[size + 2..];
        }
        (status, decoded)
    }

    /// `(event, data)` pairs of a named-event SSE body (keepalives skipped).
    #[cfg(unix)]
    fn named_sse_events(bytes: &[u8]) -> Vec<(String, serde_json::Value)> {
        String::from_utf8(bytes.to_vec())
            .expect("SSE body is UTF-8")
            .split("\n\n")
            .filter(|frame| !frame.trim().is_empty() && !frame.starts_with(':'))
            .map(|frame| {
                let mut event = None;
                let mut data = None;
                for line in frame.lines() {
                    if let Some(name) = line.strip_prefix("event: ") {
                        event = Some(name.to_owned());
                    } else if let Some(json) = line.strip_prefix("data: ") {
                        data = Some(serde_json::from_str(json).expect("SSE data is JSON"));
                    }
                }
                (
                    event.unwrap_or_else(|| panic!("frame without event: {frame}")),
                    data.unwrap_or_else(|| panic!("frame without data: {frame}")),
                )
            })
            .collect()
    }

    /// The last `generate` the fake daemon received, minus its per-request
    /// wire ids, so two HTTP requests can be compared as daemon requests.
    #[cfg(unix)]
    fn last_generate_without_ids(h: &Task11HttpHarness) -> serde_json::Value {
        let log = h.read_requests_log();
        let mut generate = (*Task11HttpHarness::ops_of_type(&log, "generate")
            .last()
            .expect("a generate reached the daemon"))
        .clone();
        let obj = generate.as_object_mut().unwrap();
        obj.remove("id");
        obj.remove("attempt_id");
        generate
    }

    /// A Messages or Responses request reaches the daemon as exactly the
    /// generate its chat-completions twin sends: same messages, sampling,
    /// stops and reasoning controls. Admission, batching eligibility and the
    /// session cache all key on that request, so they behave identically.
    #[cfg(unix)]
    #[test]
    fn messages_and_responses_reach_the_daemon_as_their_chat_twin() {
        let h = Task11HttpHarness::spawn("adapter-twins");
        let chat = serde_json::json!({
            "model": h.model(),
            "max_tokens": 64,
            "messages": [
                { "role": "system", "content": "Be brief." },
                { "role": "user", "content": "t11-stop-text please" },
            ],
            "stop": ["END"],
            "temperature": 0.2,
            "top_k": 20,
            "thinking": { "type": "disabled" },
        });
        assert_eq!(raw_post(h.port(), "/v1/chat/completions", &chat).0, 200);
        let chat_generate = last_generate_without_ids(&h);

        let messages = serde_json::json!({
            "model": h.model(),
            "max_tokens": 64,
            "system": "Be brief.",
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "t11-stop-text please" }] }],
            "stop_sequences": ["END"],
            "temperature": 0.2,
            "top_k": 20,
        });
        assert_eq!(raw_post(h.port(), "/v1/messages", &messages).0, 200);
        assert_eq!(last_generate_without_ids(&h), chat_generate);

        let mut chat_effort = chat.clone();
        chat_effort.as_object_mut().unwrap().remove("thinking");
        chat_effort["reasoning_effort"] = serde_json::json!("low");
        assert_eq!(
            raw_post(h.port(), "/v1/chat/completions", &chat_effort).0,
            200
        );
        let chat_generate = last_generate_without_ids(&h);
        let responses = serde_json::json!({
            "model": h.model(),
            "max_output_tokens": 64,
            "instructions": "Be brief.",
            "input": [{ "role": "user", "content": [{ "type": "input_text", "text": "t11-stop-text please" }] }],
            "stop": ["END"],
            "temperature": 0.2,
            "top_k": 20,
            "reasoning": { "effort": "low" },
        });
        assert_eq!(raw_post(h.port(), "/v1/responses", &responses).0, 200);
        assert_eq!(last_generate_without_ids(&h), chat_generate);
    }

    #[cfg(unix)]
    #[test]
    fn anthropic_messages_nonstream_and_stream_shapes() {
        let h = Task11HttpHarness::spawn("anthropic-messages");
        let body = serde_json::json!({
            "model": h.model(),
            "max_tokens": 64,
            "thinking": { "type": "enabled", "budget_tokens": 1024 },
            "messages": [{ "role": "user", "content": "t21-think-text please" }],
        });
        let (status, bytes) = raw_post(h.port(), "/v1/messages", &body);
        let message: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, 200, "{message}");
        assert_eq!(message["type"], "message");
        assert_eq!(message["role"], "assistant");
        assert!(message["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(
            message["content"],
            serde_json::json!([
                { "type": "thinking", "thinking": "plan it", "signature": "" },
                { "type": "text", "text": "answer text" },
            ])
        );
        assert_eq!(message["stop_reason"], "end_turn");
        assert_eq!(
            message["usage"],
            serde_json::json!({ "input_tokens": 9, "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 2, "output_tokens": 5 })
        );

        let mut stream_body = body.clone();
        stream_body["stream"] = serde_json::json!(true);
        let (status, bytes) = raw_post(h.port(), "/v1/messages", &stream_body);
        assert_eq!(status, 200);
        let events = named_sse_events(&bytes);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let thinking: String = events
            .iter()
            .filter_map(|(_, data)| data.pointer("/delta/thinking").and_then(|t| t.as_str()))
            .collect();
        let text: String = events
            .iter()
            .filter_map(|(_, data)| data.pointer("/delta/text").and_then(|t| t.as_str()))
            .collect();
        assert_eq!(
            (thinking.as_str(), text.as_str()),
            ("plan it", "answer text")
        );
        assert_eq!(events[9].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(events[9].1["usage"]["output_tokens"], 5);

        // Errors keep the Anthropic shape and the chat route's status mapping.
        let (status, bytes) = raw_post(
            h.port(),
            "/v1/messages",
            &serde_json::json!({ "model": h.model(), "messages": [] }),
        );
        let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, 400);
        assert_eq!(error["type"], "error");
        assert_eq!(error["error"]["type"], "invalid_request_error");
        let mut unknown = body.clone();
        unknown["model"] = serde_json::json!("qwen3.5:0.8b");
        let (status, bytes) = raw_post(h.port(), "/v1/messages", &unknown);
        let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, 404);
        assert_eq!(error["error"]["type"], "not_found_error");
    }

    #[cfg(unix)]
    #[test]
    fn anthropic_messages_tool_use_round_trip() {
        let h = Task11HttpHarness::spawn("anthropic-tools");
        let tools = serde_json::json!([{
            "name": "read_file",
            "description": "read a file",
            "input_schema": { "type": "object", "properties": { "path": { "type": "string" } } },
        }]);
        let first = serde_json::json!({
            "model": h.model(),
            "max_tokens": 64,
            "tools": tools,
            "messages": [{ "role": "user", "content": "t11-mixed-tool please" }],
        });
        let (status, bytes) = raw_post(h.port(), "/v1/messages", &first);
        let message: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, 200, "{message}");
        assert_eq!(message["stop_reason"], "tool_use");
        let content = message["content"].as_array().unwrap();
        assert_eq!(
            content[0],
            serde_json::json!({ "type": "text", "text": "I'll look that up." })
        );
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["name"], "read_file");
        assert_eq!(
            content[1]["input"],
            serde_json::json!({ "path": "mixed.rs" })
        );
        let tool_id = content[1]["id"].as_str().unwrap().to_owned();

        let mut stream_body = first.clone();
        stream_body["stream"] = serde_json::json!(true);
        let (status, bytes) = raw_post(h.port(), "/v1/messages", &stream_body);
        assert_eq!(status, 200);
        let events = named_sse_events(&bytes);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(events[4].1["content_block"]["type"], "tool_use");
        assert_eq!(events[4].1["index"], 1);
        assert_eq!(
            events[5].1["delta"],
            serde_json::json!({ "type": "input_json_delta", "partial_json": "{\"path\":\"mixed.rs\"}" })
        );
        assert_eq!(events[7].1["delta"]["stop_reason"], "tool_use");

        // Send the tool result back: the daemon gets the chat tool turn.
        let second = serde_json::json!({
            "model": h.model(),
            "max_tokens": 64,
            "tools": tools,
            "messages": [
                { "role": "user", "content": "t11-mixed-tool please" },
                { "role": "assistant", "content": message["content"] },
                { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": tool_id, "content": "fn main() {}" }] },
            ],
        });
        assert_eq!(raw_post(h.port(), "/v1/messages", &second).0, 200);
        let generate = last_generate_without_ids(&h);
        let sent = generate["messages"].as_array().unwrap();
        assert_eq!(sent[1]["role"], "assistant");
        assert_eq!(sent[1]["tool_calls"][0]["name"], "read_file");
        assert_eq!(
            sent[1]["tool_calls"][0]["arguments"],
            serde_json::json!({ "path": "mixed.rs" })
        );
        assert_eq!(sent[1]["tool_calls"][0]["id"], serde_json::json!(tool_id));
        assert_eq!(sent[2]["role"], "tool");
        assert_eq!(sent[2]["content"], "fn main() {}");
        assert_eq!(sent[2]["tool_call_id"], serde_json::json!(tool_id));
    }

    #[cfg(unix)]
    #[test]
    fn openai_responses_shapes_and_function_call_round_trip() {
        let h = Task11HttpHarness::spawn("openai-responses");
        let body = serde_json::json!({
            "model": h.model(),
            "input": "t21-think-text please",
            "max_output_tokens": 64,
        });
        let (status, bytes) = raw_post(h.port(), "/v1/responses", &body);
        let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, 200, "{response}");
        assert_eq!(response["object"], "response");
        assert_eq!(response["status"], "completed");
        assert!(response["id"].as_str().unwrap().starts_with("resp_"));
        let output = response["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "plan it");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "answer text");
        assert_eq!(
            response["usage"],
            serde_json::json!({ "input_tokens": 11, "input_tokens_details": { "cached_tokens": 2 },
                                "output_tokens": 5, "total_tokens": 16 })
        );

        let mut stream_body = body.clone();
        stream_body["stream"] = serde_json::json!(true);
        let (status, bytes) = raw_post(h.port(), "/v1/responses", &stream_body);
        assert_eq!(status, 200);
        let events = named_sse_events(&bytes);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (seq, (_, data)) in events.iter().enumerate() {
            assert_eq!(data["sequence_number"], serde_json::json!(seq));
        }
        assert_eq!(events[11].1["text"], "answer text");

        let tools = serde_json::json!([{
            "type": "function",
            "name": "read_file",
            "description": "read a file",
            "parameters": { "type": "object", "properties": { "path": { "type": "string" } } },
        }]);
        let call = serde_json::json!({ "model": h.model(), "input": "t11-pure-tool please", "tools": tools });
        let (status, bytes) = raw_post(h.port(), "/v1/responses", &call);
        let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, 200, "{response}");
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["name"], "read_file");
        assert_eq!(output[0]["arguments"], "{\"path\":\"a.rs\"}");
        let call_id = output[0]["call_id"].as_str().unwrap().to_owned();

        let mut stream_call = call.clone();
        stream_call["stream"] = serde_json::json!(true);
        let (status, bytes) = raw_post(h.port(), "/v1/responses", &stream_call);
        assert_eq!(status, 200);
        let names: Vec<String> = named_sse_events(&bytes)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );

        let mut input =
            vec![serde_json::json!({ "role": "user", "content": "t11-pure-tool please" })];
        input.extend(output.iter().cloned());
        input.push(serde_json::json!({ "type": "function_call_output", "call_id": call_id, "output": "fn main() {}" }));
        let round_trip = serde_json::json!({ "model": h.model(), "input": input, "tools": tools });
        assert_eq!(raw_post(h.port(), "/v1/responses", &round_trip).0, 200);
        let generate = last_generate_without_ids(&h);
        let sent = generate["messages"].as_array().unwrap();
        assert_eq!(
            sent[1]["tool_calls"][0]["arguments"],
            serde_json::json!({ "path": "a.rs" })
        );
        assert_eq!(sent[1]["tool_calls"][0]["id"], serde_json::json!(call_id));
        assert_eq!(sent[2]["role"], "tool");
        assert_eq!(sent[2]["tool_call_id"], serde_json::json!(call_id));

        let (status, bytes) = raw_post(
            h.port(),
            "/v1/responses",
            &serde_json::json!({ "model": h.model(), "input": "x", "previous_response_id": "resp_1" }),
        );
        let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, 400);
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }

    /// Write a non-stream request and return the live TCP stream without reading status.
    #[cfg(unix)]
    fn open_nonstream_request(port: u16, body: &serde_json::Value) -> std::net::TcpStream {
        use std::net::TcpStream;
        let payload = body.to_string();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: 127.0.0.1:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {payload}",
            payload.len(),
        );
        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect serve");
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .expect("write timeout");
        // Short read timeout so accidental reads fail fast; tests close without status.
        stream
            .set_read_timeout(Some(Duration::from_millis(50)))
            .expect("read timeout");
        stream.write_all(request.as_bytes()).expect("write request");
        let _ = stream.flush();
        stream
    }

    #[cfg(unix)]
    fn wait_fixture_ready(harness: &Task11HttpHarness, deadline: Instant) -> serde_json::Value {
        let ready_path = harness.paths.root.join("fixture-ready.log");
        while Instant::now() < deadline {
            if let Ok(raw) = fs::read_to_string(&ready_path) {
                for line in raw.lines().filter(|l| !l.trim().is_empty()) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                        if val.get("type").and_then(|v| v.as_str()) == Some("fixture_ready") {
                            return val;
                        }
                    }
                }
            }
            // Fallback: requests.log also records fixture_ready.
            if let Some(row) = harness
                .read_requests_log()
                .into_iter()
                .find(|row| row.get("type").and_then(|v| v.as_str()) == Some("fixture_ready"))
            {
                return row;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!(
            "fixture-ready never appeared; log={:?}",
            harness.read_requests_log()
        );
    }

    #[cfg(unix)]
    #[test]
    fn oversized_declared_body_is_rejected_before_body_read() {
        let harness = Task11HttpHarness::spawn("oversized-content-length");
        let mut stream =
            std::net::TcpStream::connect(("127.0.0.1", harness.port())).expect("connect serve");
        stream
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        stream
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\n\
                  Host: localhost\r\n\
                  Content-Type: application/json\r\n\
                  Content-Length: 8388609\r\n\
                  Connection: close\r\n\r\n",
            )
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(
            response.starts_with("HTTP/1.1 413"),
            "oversized declaration was not rejected immediately: {response:?}"
        );
    }

    #[cfg(unix)]
    fn post_systemone(port: u16, body: &serde_json::Value) -> (u16, String, String) {
        let payload = serde_json::to_vec(body).unwrap();
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect serve");
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        let head = format!(
            "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            payload.len()
        );
        stream.write_all(head.as_bytes()).unwrap();
        stream.write_all(&payload).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let status: u16 = response[9..12].parse().unwrap();
        let (headers, body) = response.split_once("\r\n\r\n").unwrap();
        (status, headers.to_string(), body.to_string())
    }

    #[cfg(unix)]
    #[test]
    fn systemone_named_model_returns_jev_shape() {
        let harness = Task11HttpHarness::spawn("systemone-named");
        let (status, headers, body) = post_systemone(
            harness.port(),
            &serde_json::json!({
                "model": harness.model(),
                "state": "hello",
                "questions": {"q": {"type": "choice", "instructions": "i",
                                    "criteria": {"a": "x", "b": "y"}}}
            }),
        );
        assert_eq!(status, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["answers"]["q"]["type"], "choice");
        assert_eq!(v["usage"]["output_tokens"], 0);
        assert!(
            v.get("timing").is_none(),
            "timing must not leak into the Jev body"
        );
        assert!(
            headers.to_ascii_lowercase().contains("x-hipfire-timing:"),
            "{headers}"
        );
        let mut keys: Vec<&str> = v
            .as_object()
            .expect("Jev body is an object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec!["answers", "model", "usage"],
            "Jev body must carry exactly {{model, answers, usage}}, no extras: {body}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn systemone_jev_latest_without_loaded_model_is_503() {
        let harness = Task11HttpHarness::spawn("systemone-unloaded");
        let (status, _, body) = post_systemone(
            harness.port(),
            &serde_json::json!({
                "model": "jev-latest", "state": "s",
                "questions": {"q": {"type": "noul", "instructions": "i"}}
            }),
        );
        assert_eq!(status, 503, "{body}");
    }

    #[cfg(unix)]
    #[test]
    fn systemone_jev_latest_uses_loaded_model() {
        let harness = Task11HttpHarness::spawn("systemone-loaded");
        let named = serde_json::json!({"model": harness.model(), "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let (named_status, _, named_body) = post_systemone(harness.port(), &named);
        assert_eq!(named_status, 200, "{named_body}");
        let named_v: serde_json::Value = serde_json::from_str(&named_body).unwrap();
        let (status, _, body) = post_systemone(
            harness.port(),
            &serde_json::json!({
                "model": "jev-latest", "state": "s",
                "questions": {"q": {"type": "noul", "instructions": "i"}}
            }),
        );
        assert_eq!(status, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_ne!(v["model"], "jev-latest");
        assert_eq!(
            v["model"], named_v["model"],
            "jev-latest must echo the same model the named request loaded"
        );
    }

    #[cfg(unix)]
    #[test]
    fn systemone_daemon_validation_error_maps_to_422() {
        let harness = Task11HttpHarness::spawn("systemone-422");
        let (status, _, body) = post_systemone(
            harness.port(),
            &serde_json::json!({
                "model": harness.model(), "state": 5, "questions": {}
            }),
        );
        assert_eq!(status, 422, "{body}");
    }

    #[cfg(unix)]
    fn timing_header(headers: &str) -> serde_json::Value {
        let raw = headers
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.trim()
                    .eq_ignore_ascii_case("x-hipfire-timing")
                    .then(|| v.trim().to_string())
            })
            .expect("x-hipfire-timing header");
        serde_json::from_str(&raw).unwrap()
    }

    #[cfg(unix)]
    fn session_body(model: &str, tool_choice: Option<&str>) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "yo"}],
            "tools": [{"type": "function", "function": {
                "name": "lookup", "description": "d",
                "parameters": {"type": "object", "properties": {}}}}],
            "questions": {"q": {"type": "noul", "instructions": "i"}}
        });
        if let Some(tc) = tool_choice {
            body["tool_choice"] = serde_json::json!(tc);
        }
        body
    }

    /// Session mode (spec §12.6): serve forwards the chat projection of
    /// `messages` and `tools`, and the Jev body shape is unchanged.
    #[cfg(unix)]
    #[test]
    fn systemone_session_forwards_projected_messages_and_tools() {
        let harness = Task11HttpHarness::spawn("systemone-session");
        let (status, headers, body) =
            post_systemone(harness.port(), &session_body(harness.model(), None));
        assert_eq!(status, 200, "{body}");
        let t = timing_header(&headers);
        assert_eq!(t["mode"], "session", "{t}");
        let roles: Vec<&str> = t["session_roles"]
            .as_array()
            .expect("session_roles")
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        assert!(roles.ends_with(&["user", "assistant"]), "{roles:?}");
        assert_eq!(t["session_tools"], 1, "{t}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["answers"]["q"]["type"], "noul");
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["answers", "model", "usage"], "{body}");
    }

    /// `tool_choice: "none"` drops tools exactly as the chat projection
    /// does, proving serve projects rather than passes `tools` through.
    #[cfg(unix)]
    #[test]
    fn systemone_session_applies_tool_choice_projection() {
        let harness = Task11HttpHarness::spawn("systemone-session-tc");
        let (status, headers, body) =
            post_systemone(harness.port(), &session_body(harness.model(), Some("none")));
        assert_eq!(status, 200, "{body}");
        assert_eq!(timing_header(&headers)["session_tools"], 0);
    }

    #[cfg(unix)]
    #[test]
    fn systemone_state_and_messages_together_is_422() {
        let harness = Task11HttpHarness::spawn("systemone-both");
        let mut body = session_body(harness.model(), None);
        body["state"] = serde_json::json!("s");
        let (status, _, body) = post_systemone(harness.port(), &body);
        assert_eq!(status, 422, "{body}");
    }

    /// Serve refuses a session `messages` that is not a non-empty array, that
    /// the chat projection would reduce to only the injected default system
    /// message, or that carries an image part (spec §12.6), all as 422 and
    /// before any daemon round trip. The harness model gets a per-model
    /// `prompt.system` override, so the default system message really is
    /// injected (the control request shows it).
    #[cfg(unix)]
    #[test]
    fn systemone_session_refuses_messages_the_projection_would_empty() {
        let harness = Task11HttpHarness::spawn("systemone-session-empty");
        let mut overrides = ConfigLayer::default();
        overrides
            .set_cli("prompt.system", "injected default system")
            .unwrap();
        let mut catalog = hipfire_config::ModelCatalog::default();
        catalog.models.insert(
            "t11-session".into(),
            hipfire_config::LocalModelConfig {
                path: Some(PathBuf::from(harness.model())),
                registry_tag: None,
                overrides,
            },
        );
        write_catalog_toml(&harness.paths.config, &catalog).unwrap();
        let mut body = session_body(harness.model(), None);
        let (status, headers, text) = post_systemone(harness.port(), &body);
        assert_eq!(status, 200, "{text}");
        let roles = timing_header(&headers)["session_roles"].clone();
        assert_eq!(roles, serde_json::json!(["system", "user", "assistant"]));
        let image = serde_json::json!([{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aa"}}]}]);
        let cases = [
            serde_json::json!([]),
            serde_json::json!("hi"),
            serde_json::json!([{"role": "bogus", "content": "x"}]),
            image,
        ];
        for messages in cases {
            body["messages"] = messages.clone();
            let (status, _, text) = post_systemone(harness.port(), &body);
            assert_eq!(status, 422, "{messages}: {text}");
            assert!(text.contains("messages"), "{messages}: {text}");
        }
        let log = harness.read_requests_log();
        assert_eq!(
            Task11HttpHarness::ops_of_type(&log, "decide").len(),
            1,
            "only the control request reached the daemon; log={log:?}"
        );
    }

    /// A daemon 422 with `required_max_seq` is returned to the client as-is,
    /// `required_max_seq` included, without any reload: serve never grows
    /// `max_seq` on a request's behalf.
    #[cfg(unix)]
    #[test]
    fn systemone_required_max_seq_returns_422_without_reload() {
        let harness = Task11HttpHarness::spawn("systemone-huge-ctx");
        let (status, _, body) = post_systemone(
            harness.port(),
            &serde_json::json!({
                "model": harness.model(), "state": "t-needs-huge-ctx",
                "questions": {"q": {"type": "noul", "instructions": "i"}}
            }),
        );
        assert_eq!(status, 422, "{body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["required_max_seq"], 10_000_000u64, "{body}");
        let log = harness.read_requests_log();
        let loads = Task11HttpHarness::ops_of_type(&log, "load");
        assert_eq!(
            loads.len(),
            1,
            "only the initial load; no reload; log={log:?}"
        );
        assert_eq!(
            Task11HttpHarness::ops_of_type(&log, "decide").len(),
            1,
            "no retry past the ceiling; log={log:?}"
        );
    }

    /// `_debug_no_snapshot` is a daemon-only gate knob: serve must not
    /// forward it, only `state` and `questions`.
    #[cfg(unix)]
    #[test]
    fn systemone_does_not_forward_debug_no_snapshot() {
        let harness = Task11HttpHarness::spawn("systemone-no-debug");
        let (status, _, body) = post_systemone(
            harness.port(),
            &serde_json::json!({
                "model": harness.model(), "state": "s", "_debug_no_snapshot": true,
                "questions": {"q": {"type": "noul", "instructions": "i"}}
            }),
        );
        assert_eq!(status, 200, "{body}");
        let log = harness.read_requests_log();
        let decides = Task11HttpHarness::ops_of_type(&log, "decide");
        assert_eq!(decides.len(), 1, "log={log:?}");
        assert!(
            decides[0].get("_debug_no_snapshot").is_none(),
            "serve forwarded _debug_no_snapshot: {:?}",
            decides[0]
        );
        assert_eq!(decides[0]["state"], "s");
    }

    /// Spec §13.2: write a per-model `decide.calibration` override for the
    /// harness fixture into the harness's own models.toml.
    #[cfg(unix)]
    fn write_decide_calibration(harness: &Task11HttpHarness, table: &str) {
        let catalog = format!(
            "schema_version = 1\n\n[models.\"fixture\"]\npath = {:?}\n\n\
             [models.\"fixture\".overrides.decide.calibration]\n{table}\n",
            harness.model()
        );
        fs::write(&harness.paths.config.models_toml, catalog).unwrap();
    }

    /// POST a decide (must succeed) and return the message serve forwarded.
    #[cfg(unix)]
    fn forwarded_decide(
        harness: &Task11HttpHarness,
        body: &serde_json::Value,
    ) -> serde_json::Value {
        let (status, _, text) = post_systemone(harness.port(), body);
        assert_eq!(status, 200, "{text}");
        let log = harness.read_requests_log();
        let decides = Task11HttpHarness::ops_of_type(&log, "decide");
        (*decides.last().expect("a forwarded decide")).clone()
    }

    #[cfg(unix)]
    #[test]
    fn systemone_forwards_the_per_model_calibration() {
        let harness = Task11HttpHarness::spawn("systemone-cal");
        write_decide_calibration(&harness, "choice = 1.5\nnoul = 0.8");
        let body = serde_json::json!({"model": harness.model(), "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let d = forwarded_decide(&harness, &body);
        assert_eq!(
            d["calibration"],
            serde_json::json!({"choice": 1.5, "score": 1.0, "noul": 0.8})
        );
        // Session mode forwards it too.
        let d = forwarded_decide(&harness, &session_body(harness.model(), None));
        assert_eq!(d["calibration"]["choice"], 1.5);
    }

    #[cfg(unix)]
    #[test]
    fn systemone_sends_no_calibration_without_an_override() {
        let harness = Task11HttpHarness::spawn("systemone-cal-none");
        let body = serde_json::json!({"model": harness.model(), "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let d = forwarded_decide(&harness, &body);
        assert!(d.get("calibration").is_none(), "{d}");
    }

    #[cfg(unix)]
    #[test]
    fn systemone_client_cannot_set_calibration() {
        let harness = Task11HttpHarness::spawn("systemone-cal-client");
        let body = serde_json::json!({"model": harness.model(), "state": "s",
            "calibration": {"choice": 9.0},
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let d = forwarded_decide(&harness, &body);
        assert!(
            d.get("calibration").is_none(),
            "client calibration forwarded: {d}"
        );
        write_decide_calibration(&harness, "noul = 0.8");
        let d = forwarded_decide(&harness, &body);
        assert_eq!(
            d["calibration"],
            serde_json::json!({"choice": 1.0, "score": 1.0, "noul": 0.8})
        );
    }

    /// A `jev-latest`-style request (no local match, spec §12.5's "use
    /// whatever is loaded" fallback) resolves `runtime.current_path` — the
    /// path `find_model_path` returned for the originally named request —
    /// back to the per-model catalog record for a PATH-ONLY model (no
    /// registry tag). The override is written through the real `hipfire
    /// config <model> set` code path (`config_command`), not hand-rolled
    /// TOML, so the catalog's `path` field and table key are exactly what
    /// production writes. The harness label deliberately avoids "jev" and
    /// "latest": `find_model_path`'s separator-stripped fuzzy match would
    /// otherwise resolve the literal string "jev-latest" straight to a
    /// same-labelled fixture file and never exercise the fallback branch
    /// at all (caught empirically while writing this test).
    #[cfg(unix)]
    #[test]
    fn systemone_forwards_calibration_for_jev_latest_on_a_path_only_model() {
        let harness = Task11HttpHarness::spawn("systemone-cal-fallback");
        config_command(
            &harness.paths,
            ConfigArgs {
                model: Some(harness.model().to_string()),
                action: Some(ConfigAction::Set {
                    key: "decide.calibration.noul".to_string(),
                    value: "0.8".to_string(),
                }),
            },
        )
        .unwrap();

        // A named request first, so the model actually loads and
        // `runtime.current_path` is populated.
        let named = serde_json::json!({"model": harness.model(), "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let d = forwarded_decide(&harness, &named);
        assert_eq!(
            d["calibration"],
            serde_json::json!({"choice": 1.0, "score": 1.0, "noul": 0.8}),
            "named request must resolve the override: {d}"
        );

        // `jev-latest` is not itself a local model: is_local_model is
        // false, so run_decide falls back to runtime.current_path.
        let latest = serde_json::json!({"model": "jev-latest", "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let d = forwarded_decide(&harness, &latest);
        assert_eq!(
            d["calibration"],
            serde_json::json!({"choice": 1.0, "score": 1.0, "noul": 0.8}),
            "jev-latest fallback must resolve the same per-model override: {d}"
        );
    }

    /// The response `model` is the served model name chat and `/health`
    /// report (`meta.current_model`, the tag when a tag was requested), not
    /// the resolved filesystem path.
    #[cfg(unix)]
    #[test]
    fn systemone_model_echo_is_served_model_name() {
        let harness = Task11HttpHarness::spawn("systemone-echo");
        let req = |model: &str| {
            serde_json::json!({"model": model, "state": "s",
                "questions": {"q": {"type": "noul", "instructions": "i"}}})
        };
        let (status, _, body) = post_systemone(harness.port(), &req(harness.model()));
        assert_eq!(status, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let served = harness
            .shared
            .meta
            .lock()
            .unwrap()
            .current_model
            .clone()
            .expect("a model is loaded");
        assert_eq!(v["model"], served.as_str());
        // Stand in a tag-style served name for the loaded model (what a tag
        // request records): the echo must follow it, not the path.
        harness.shared.meta.lock().unwrap().current_model = Some("qwen3.5:4b".to_owned());
        let (status, _, body) = post_systemone(harness.port(), &req("jev-latest"));
        assert_eq!(status, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "qwen3.5:4b", "{body}");
    }

    /// Write a `/v1/systemone` request and return the live TCP stream
    /// without reading the response — mirrors `open_nonstream_request`.
    #[cfg(unix)]
    fn open_systemone_request(port: u16, body: &serde_json::Value) -> std::net::TcpStream {
        use std::net::TcpStream;
        let payload = body.to_string();
        let request = format!(
            "POST /v1/systemone HTTP/1.1\r\n\
             Host: 127.0.0.1:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {payload}",
            payload.len(),
        );
        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect serve");
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .expect("write timeout");
        stream
            .set_read_timeout(Some(Duration::from_millis(50)))
            .expect("read timeout");
        stream.write_all(request.as_bytes()).expect("write request");
        let _ = stream.flush();
        stream
    }

    /// A client disconnect mid-decide must not release the admission slot
    /// before the daemon's (deliberately slow) `decided` reply lands.
    /// Decide's daemon round trip (`Engine::request`) has no cancellation
    /// hook, so releasing the slot the instant hyper notices the client is
    /// gone would let a second decide race the daemon while the first is
    /// still, invisibly, in flight — corrupting the single-daemon
    /// serialisation `Admission` exists to guarantee.
    ///
    /// This checks `harness.shared.admission.inflight()` directly (the same
    /// technique `nonstream_client_disconnect_aborts_and_releases_admission`
    /// uses), not the fake daemon's request log: the fake daemon is a
    /// single-threaded Python script blocked in `time.sleep(1)` while
    /// processing the slow decide, so it would not read/log a second
    /// request's bytes until it finishes regardless of when serve released
    /// admission — log-ordering alone cannot distinguish early-release from
    /// held-to-completion here, only `inflight()` can.
    #[cfg(unix)]
    #[test]
    fn systemone_client_disconnect_does_not_release_admission_early() {
        let harness = Task11HttpHarness::spawn("systemone-disconnect");
        let port = harness.port();
        let slow_body = serde_json::json!({
            "model": harness.model(), "state": "t-slow",
            "questions": {"q": {"type": "noul", "instructions": "i"}}
        });

        let stream = open_systemone_request(port, &slow_body);

        // Wait for the daemon to actually receive the slow decide (proving
        // the request is admitted and dispatched) before disconnecting, so
        // the guard is provably held mid-flight.
        let ready_deadline = Instant::now() + Duration::from_secs(5);
        let slow_decide_seen = |harness: &Task11HttpHarness| {
            Task11HttpHarness::ops_of_type(&harness.read_requests_log(), "decide")
                .iter()
                .any(|row| row.get("state").and_then(|v| v.as_str()) == Some("t-slow"))
        };
        while Instant::now() < ready_deadline && !slow_decide_seen(&harness) {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            slow_decide_seen(&harness),
            "slow decide never reached the fake daemon within 5s; log={:?}",
            harness.read_requests_log()
        );
        assert_eq!(
            harness.shared.admission.inflight(),
            1,
            "precondition: exactly one decide holds admission"
        );

        drop(stream);

        // Poll admission for up to 300ms right after disconnecting. If the
        // guard were released the instant hyper notices the client is gone
        // (the bug), inflight would drop to 0 within this short window, well
        // before the fake daemon's 1s sleep can possibly have elapsed.
        let short_deadline = Instant::now() + Duration::from_millis(300);
        let mut dropped_early = false;
        while Instant::now() < short_deadline {
            if harness.shared.admission.inflight() == 0 {
                dropped_early = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !dropped_early,
            "admission released within 300ms of disconnect — the slot must stay held for the \
             full ~1s daemon round trip, not release the instant the client vanishes"
        );

        // It must still release once the daemon's slow decide actually
        // completes, within a generous bound past the 1s sleep.
        let release_deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < release_deadline && harness.shared.admission.inflight() != 0 {
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            harness.shared.admission.inflight(),
            0,
            "admission must eventually release once the slow daemon call completes"
        );
    }

    /// Silent non-stream disconnect holds admission until the correlated done
    /// drains, independently of when the fixture writes its side-channel log.
    #[cfg(unix)]
    #[test]
    fn nonstream_client_disconnect_aborts_and_releases_admission() {
        let harness = Task11HttpHarness::spawn("ns-disconnect-abort");
        let port = harness.port();
        let body = harness.base_body("t11-gated-nonstream", false);

        assert_eq!(
            harness.shared.admission.inflight(),
            0,
            "precondition: admission idle"
        );

        let stream = open_nonstream_request(port, &body);
        let ready_deadline = Instant::now() + Duration::from_secs(5);
        let ready = wait_fixture_ready(&harness, ready_deadline);
        let gen_id = ready
            .get("id")
            .cloned()
            .expect("fixture_ready carries generate id");
        let gen_aid = ready
            .get("attempt_id")
            .and_then(|v| v.as_u64())
            .expect("fixture_ready carries attempt_id");

        // Confirm generate is in the wire log with the same correlation pair.
        let request_log = harness.read_requests_log();
        let generates = Task11HttpHarness::ops_of_type(&request_log, "generate");
        assert!(
            generates.iter().any(|g| {
                g.get("id") == Some(&gen_id)
                    && g.get("attempt_id").and_then(|v| v.as_u64()) == Some(gen_aid)
            }),
            "generate missing for fixture pair: {:?}",
            harness.read_requests_log()
        );

        // Client disconnect without ever reading an HTTP status line.
        drop(stream);

        let correlated = |row: &serde_json::Value, kind: &str| {
            row.get("type").and_then(|v| v.as_str()) == Some(kind)
                && row.get("id") == Some(&gen_id)
                && row.get("attempt_id").and_then(|v| v.as_u64()) == Some(gen_aid)
        };
        let wait_for_marker = |kind: &str| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let log = harness.read_requests_log();
                if log.iter().any(|row| correlated(row, kind)) {
                    break;
                }
                assert!(Instant::now() < deadline, "missing {kind}; log={log:?}");
                thread::sleep(Duration::from_millis(20));
            }
        };
        let release = |kind: &str| {
            harness
                .shared
                .runtime
                .lock()
                .unwrap()
                .engine
                .send(&serde_json::json!({
                    "type": kind,
                    "id": gen_id,
                    "attempt_id": gen_aid,
                }))
                .unwrap();
        };

        wait_for_marker("abort");
        wait_for_marker("daemon_aborted");
        // The fixture cannot emit done until we release it. This checks the
        // drain contract without racing two independently sampled observers.
        assert_eq!(
            harness.shared.admission.inflight(),
            1,
            "admission must remain held while done is withheld"
        );
        release("fixture_release_done");
        let deadline = Instant::now() + Duration::from_secs(5);
        while harness.shared.admission.inflight() != 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(harness.shared.admission.inflight(), 0, "done must drain");
        // Force the old false-positive schedule: the real done has drained,
        // but the fixture's later side-channel marker is still withheld.
        assert!(
            !harness
                .read_requests_log()
                .iter()
                .any(|row| correlated(row, "daemon_done_aborted")),
            "the fixture must still be holding its done log"
        );
        release("fixture_release_done_log");
        wait_for_marker("daemon_done_aborted");
        assert_eq!(
            harness.shared.admission.inflight(),
            0,
            "admission must be fully released after abort drain"
        );

        // No commit may have been sent for the cancelled attempt.
        let log = harness.read_requests_log();
        let commits = Task11HttpHarness::ops_of_type(&log, "commit");
        assert!(
            commits.iter().all(|c| {
                !(c.get("id") == Some(&gen_id)
                    && c.get("attempt_id").and_then(|v| v.as_u64()) == Some(gen_aid))
            }),
            "cancelled attempt must not commit: {commits:?}"
        );

        // Immediate follow-up must admit and return a single JSON success.
        let follow = complete_nonstream(port, harness.base_body("t11-stop-text", false))
            .expect("follow-up after cancel must return 200 JSON");
        assert_eq!(follow["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            follow["choices"][0]["message"]["content"],
            "hello from fake daemon"
        );
        let release_deadline = Instant::now() + Duration::from_millis(500);
        while harness.shared.admission.inflight() != 0 && Instant::now() < release_deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            harness.shared.admission.inflight(),
            0,
            "follow-up admission guard did not release after flushed response"
        );
    }

    /// Connected pre-terminal daemon error keeps non-200 OpenAI error JSON.
    #[cfg(unix)]
    #[test]
    fn nonstream_connected_error_preserves_status() {
        let harness = Task11HttpHarness::spawn("ns-connected-err");
        let port = harness.port();
        let body = harness.base_body("t15-class-validation", false);
        let (status, raw_body) = raw_nonstream_post(port, &body);
        assert!(
            status >= 400 && status != 200,
            "preterminal error must keep non-200 status, got {status}"
        );
        let text = String::from_utf8_lossy(&raw_body);
        // Strip optional chunked framing / trailing whitespace for JSON parse.
        let json_start = text.find('{').expect("OpenAI error JSON object");
        let json_text = &text[json_start..];
        let err: serde_json::Value =
            serde_json::from_str(json_text.trim_end()).expect("OpenAI error JSON");
        let message = err
            .pointer("/error/message")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            !message.is_empty(),
            "error.message required in OpenAI error body: {err}"
        );
        assert!(
            err.pointer("/choices").is_none(),
            "error response must not look like a completion: {err}"
        );
        // Exactly one top-level JSON value.
        let mut it = serde_json::Deserializer::from_str(json_text.trim_end())
            .into_iter::<serde_json::Value>();
        let _first = it.next().expect("one value").expect("valid JSON");
        assert!(
            it.next().is_none(),
            "error body must be exactly one JSON value"
        );
    }

    /// Connected non-stream success is exactly one JSON value (no SSE/trailer).
    #[cfg(unix)]
    #[test]
    fn nonstream_success_is_exactly_one_json() {
        let harness = Task11HttpHarness::spawn("ns-one-json");
        let port = harness.port();
        let body = harness.base_body("t11-stop-text", false);
        let (status, raw_body) = raw_nonstream_post(port, &body);
        assert_eq!(status, 200, "success status");
        let text = String::from_utf8_lossy(&raw_body);
        let json_start = text.find('{').expect("JSON object in body");
        let json_text = text[json_start..].trim_end();
        let mut it = serde_json::Deserializer::from_str(json_text).into_iter::<serde_json::Value>();
        let value = it
            .next()
            .expect("exactly one JSON value")
            .expect("valid JSON");
        assert!(
            it.next().is_none(),
            "non-stream success must not append a second JSON/SSE value; body={text:?}"
        );
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            value["choices"][0]["message"]["content"],
            "hello from fake daemon"
        );
        assert!(
            !text.contains("data:"),
            "non-stream body must not carry SSE framing"
        );
    }

    /// Close-before terminal → abort and no commit; close-after full JSON → commit, no abort.
    #[cfg(unix)]
    #[test]
    fn nonstream_close_before_vs_after_terminal_commit_race() {
        let harness = Task11HttpHarness::spawn("ns-commit-race");
        let port = harness.port();

        // --- before terminal: disconnect during long silent generation ---
        {
            let body = harness.base_body("t11-long-nonstream", false);
            let stream = open_nonstream_request(port, &body);
            let ready = wait_fixture_ready(&harness, Instant::now() + Duration::from_secs(5));
            let gen_id = ready.get("id").cloned().expect("id");
            let gen_aid = ready
                .get("attempt_id")
                .and_then(|v| v.as_u64())
                .expect("attempt_id");
            drop(stream);

            let deadline = Instant::now() + Duration::from_secs(5);
            let mut saw_abort = false;
            while Instant::now() < deadline {
                let log = harness.read_requests_log();
                saw_abort = log.iter().any(|row| {
                    row.get("type").and_then(|v| v.as_str()) == Some("abort")
                        && row.get("id") == Some(&gen_id)
                        && row.get("attempt_id").and_then(|v| v.as_u64()) == Some(gen_aid)
                });
                let done = log.iter().any(|row| {
                    row.get("type").and_then(|v| v.as_str()) == Some("daemon_done_aborted")
                        && row.get("id") == Some(&gen_id)
                });
                if saw_abort && done && harness.shared.admission.inflight() == 0 {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            assert!(saw_abort, "close-before-terminal must abort");
            let log = harness.read_requests_log();
            assert!(
                Task11HttpHarness::ops_of_type(&log, "commit")
                    .iter()
                    .all(|c| c.get("id") != Some(&gen_id)),
                "close-before-terminal must not commit: {log:?}"
            );
        }

        // --- after terminal: full success response consumed; commit, no abort ---
        {
            let before_aborts =
                Task11HttpHarness::ops_of_type(&harness.read_requests_log(), "abort").len();
            let body = harness.base_body("t11-stop-text", false);
            let (status, raw_body) = raw_nonstream_post(port, &body);
            assert_eq!(status, 200);
            let text = String::from_utf8_lossy(&raw_body);
            let json_start = text.find('{').expect("json");
            let value: serde_json::Value =
                serde_json::from_str(text[json_start..].trim_end()).expect("json body");
            assert_eq!(value["choices"][0]["finish_reason"], "stop");

            let deadline = Instant::now() + Duration::from_secs(3);
            let mut saw_commit = false;
            while Instant::now() < deadline {
                let log = harness.read_requests_log();
                // Latest generate should have a matching commit.
                let generates = Task11HttpHarness::ops_of_type(&log, "generate");
                let last_gen = generates.last().expect("generate for success path");
                let gid = last_gen.get("id").cloned();
                let gaid = last_gen.get("attempt_id").and_then(|v| v.as_u64());
                saw_commit = Task11HttpHarness::ops_of_type(&log, "commit")
                    .iter()
                    .any(|c| {
                        c.get("id") == gid.as_ref()
                            && c.get("attempt_id").and_then(|v| v.as_u64()) == gaid
                    });
                if saw_commit {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            assert!(saw_commit, "close-after-terminal must commit");
            let after_aborts =
                Task11HttpHarness::ops_of_type(&harness.read_requests_log(), "abort").len();
            assert_eq!(
                after_aborts, before_aborts,
                "close-after full JSON must not emit a new abort"
            );
        }
    }

    /// Paired stream/nonstream matrix rows for stop, pure-tool, mixed-tool,
    /// two-tools, length-withhold, and usage ordering.
    ///
    /// Valid structured-call rows hard-fail on any tool-protocol marker or
    /// structured-argument JSON fragment leaking into content/reasoning.
    /// Invalid-producer dirty-marker diagnostics live in a separate test and
    /// must not weaken these valid-path assertions.
    /// (Matrix body lives with the broader Task11 suite; streaming helpers above
    /// remain the stable surface for parity checks.)
    #[cfg(unix)]

    /// Invalid-producer diagnostic (authority violation): dirty marker token text
    /// stays byte-verbatim content and never becomes structured tool_calls.
    /// Kept separate so it cannot weaken valid structured-call leak assertions.
    #[cfg(unix)]

    /// Premature daemon EOF after gen_start/token without done → client/HTTP failure.
    #[cfg(unix)]

    /// Capability denial: daemon typed error on tools request → no completion/tool payload.
    #[cfg(unix)]
    // --- Task 15: server-owned one-retry (disabled-by-default) ---
    #[test]
    fn bench_generate_request_includes_numeric_first_attempt() {
        let req = bench_generate_request("bench prompt", 37);
        assert_eq!(req.get("type").and_then(|v| v.as_str()), Some("generate"));
        assert_eq!(req.get("attempt_id").and_then(|v| v.as_u64()), Some(1));
        let id = req.get("id").and_then(|v| v.as_str()).unwrap_or("");
        assert!(!id.is_empty(), "id must be a non-empty string");
        assert_eq!(
            req.get("prompt").and_then(|v| v.as_str()),
            Some("bench prompt")
        );
        assert_eq!(req.get("max_tokens").and_then(|v| v.as_u64()), Some(37));
    }

    /// Every benchmark generate must ask for answer mode.
    ///
    /// A reasoning model opens `<think>` in its first tokens and would spend a
    /// benchmark's fixed budget (16 tokens for the warmup, 128 for the measured
    /// runs) inside it, so the measured turn would never reach the answer path.
    #[test]
    fn bench_generate_request_is_answer_mode_by_default() {
        let req = bench_generate_request("bench prompt", 128);
        assert_eq!(
            req.get("max_think_tokens").and_then(|v| v.as_u64()),
            Some(1),
            "benchmark generates must cap thinking"
        );
        assert_eq!(
            req.get("assistant_prefix").and_then(|v| v.as_str()),
            Some("closed_think"),
            "benchmark generates must start in answer mode"
        );
        assert_eq!(
            req.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("none")
        );
    }

    /// `--reasoning-on` restores the thinking turn for anyone who wants to
    /// benchmark that path, and must produce a request carrying none of the
    /// answer-mode fields.
    #[test]
    fn bench_reasoning_on_opts_back_into_thinking() {
        let req = bench_generate_request_reasoning("bench prompt", 128, true);
        assert!(req.get("max_think_tokens").is_none());
        assert!(req.get("assistant_prefix").is_none());
        assert!(req.get("reasoning_effort").is_none());
        // Still an ordinary benchmark generate otherwise.
        assert_eq!(req.get("max_tokens").and_then(|v| v.as_u64()), Some(128));
    }

    /// Any non-neutral repeat penalty routes the daemon to the plain AR
    /// decoder, so every bench generate must carry exactly 1.0.
    #[test]
    fn bench_generate_request_uses_neutral_repeat_penalty() {
        for req in [
            bench_generate_request("p", 1),
            bench_generate_request_reasoning("p", 128, false),
            bench_generate_request_reasoning("p", 128, true),
        ] {
            assert_eq!(
                req.get("repeat_penalty").and_then(|v| v.as_f64()),
                Some(1.0)
            );
        }
    }

    #[test]
    fn bench_spec_route_reads_done_stats() {
        let ar = serde_json::json!({"type": "done", "decode_tok_s": 35.0});
        assert_eq!(bench_spec_route(&ar), None);
        let mtp = serde_json::json!({"mtp": true, "tau": 2.4, "cycles": 40});
        assert_eq!(bench_spec_route(&mtp).as_deref(), Some("mtp"));
        let dflash = serde_json::json!({"dflash": true, "tau": 3.1, "cycles": 9});
        assert_eq!(bench_spec_route(&dflash).as_deref(), Some("dflash"));
        let idle = serde_json::json!({"dflash": true, "tau": 0.0, "cycles": 0});
        assert_eq!(bench_spec_route(&idle), None);
        let eagle = serde_json::json!({"spec": "gemma4_eagle", "tau": 1.8, "rounds": 7});
        assert_eq!(bench_spec_route(&eagle).as_deref(), Some("gemma4_eagle"));
    }

    #[test]
    fn bench_load_speculation_reports_the_effective_mtp_mode_beside_the_requested_one() {
        let params = serde_json::json!({
            "mtp_mode": "auto", "dflash_mode": "auto", "dspark_mode": "auto",
            "ngram_draft": false,
        });
        let requested = bench_load_speculation(&params);
        // A producer without a measured AR floor: requested auto, effective off.
        let off = bench_load_speculation_effective(
            requested.clone(),
            &serde_json::json!({ "type": "loaded", "mtp_mode_effective": "off" }),
        );
        assert_eq!(off["mtp_mode"], "auto");
        assert_eq!(off["mtp_effective"], "off");
        // The DFlash/DSpark/n-gram selectors pass through untouched.
        assert_eq!(off["dflash_mode"], "auto");
        assert_eq!(off["dspark_mode"], "auto");
        assert_eq!(off["ngram_draft"], false);
        // A producer with the floor reports `auto`.
        let on = bench_load_speculation_effective(
            requested.clone(),
            &serde_json::json!({ "mtp_mode_effective": "auto" }),
        );
        assert_eq!(on["mtp_effective"], "auto");
        // An ack with no verdict is null, never a guess.
        let none =
            bench_load_speculation_effective(requested, &serde_json::json!({ "type": "loaded" }));
        assert_eq!(none["mtp_effective"], serde_json::Value::Null);
    }

    #[test]
    fn speculation_selector_lowering_is_unchanged_by_the_mtp_floor_gate() {
        // The producer-capability gate lives on the consumers, not here: `auto`
        // leaves the registry's `mtp_mode = auto` and DFlash selectors alone,
        // while every explicit selector pins its modes exactly as before.
        let lowered = |selector: &str| {
            let mut params = serde_json::json!({
                "mtp_mode": "auto", "dflash_mode": "auto", "dspark_mode": "auto",
            });
            apply_speculation_selector(&mut params, selector).unwrap();
            params
        };
        let auto = lowered("auto");
        assert_eq!(auto["mtp_mode"], "auto");
        assert_eq!(auto["dflash_mode"], "auto");
        assert_eq!(auto["dspark_mode"], "auto");
        for (selector, dflash, mtp, dspark) in [
            ("off", "off", "off", "off"),
            ("dflash", "on", "off", "off"),
            ("mtp", "off", "on", "off"),
            ("ngram", "off", "off", "off"),
            ("dspark", "off", "off", "on"),
        ] {
            let params = lowered(selector);
            assert_eq!(params["dflash_mode"], dflash, "{selector}");
            assert_eq!(params["mtp_mode"], mtp, "{selector}");
            assert_eq!(params["dspark_mode"], dspark, "{selector}");
        }
    }

    /// A request the AR floor retired must not read as rejected drafts. The
    /// 7900 XTX Flash-Next one-card offload run (j1428, humaneval_0, 128
    /// tokens): the calibration token, 16 native windows accepting 22 drafts
    /// and 6 AR probes, then 82 retired AR tokens; done `tau` 0.21.
    #[test]
    fn bench_spec_mtp_splits_retired_ar_windows_from_drafting_windows() {
        let retired = serde_json::json!({
            "mtp": true, "tau": 0.21, "cycles": 105, "accepted": 22, "mtp_windows": 23,
            "floor_ar_windows": 7, "ar_windows": 82, "mtp_retired": true,
        });
        assert_eq!(
            bench_spec_mtp(&retired),
            Some(serde_json::json!({
                "tau_mtp": 1.38, "windows_mtp": 23, "windows_floor_ar": 7,
                "windows_ar_retired": 82, "retired_at": 23, "accepted": 22,
            }))
        );
        // Never retired (n-gram composition on): only the calibration token
        // drafts nothing; the n-gram takeovers are drafting windows.
        let live = serde_json::json!({
            "mtp": true, "mtp_ngram": true, "tau": 2.34, "cycles": 53, "accepted": 124,
            "mtp_windows": 27, "floor_ar_windows": 1, "ngram_mod_windows": 26,
            "ar_windows": 0, "mtp_retired": false,
        });
        assert_eq!(
            bench_spec_mtp(&live),
            Some(serde_json::json!({
                "tau_mtp": 2.38, "windows_mtp": 53, "windows_floor_ar": 1,
                "windows_ar_retired": 0, "retired_at": null, "accepted": 124,
            }))
        );
        // A daemon without `accepted` or `floor_ar_windows` still splits the
        // windows but prices no drafting window.
        for old in [
            serde_json::json!({
                "mtp": true, "tau": 0.21, "cycles": 105, "ar_windows": 82, "mtp_retired": true,
            }),
            serde_json::json!({
                "mtp": true, "tau": 0.21, "cycles": 105, "accepted": 22, "mtp_windows": 23,
                "ar_windows": 82, "mtp_retired": true,
            }),
        ] {
            let split = bench_spec_mtp(&old).unwrap();
            assert_eq!(split["tau_mtp"], serde_json::Value::Null, "{old}");
            assert_eq!(split["retired_at"], 23, "{old}");
        }
        // AR, DFlash and idle runs carry no MTP split.
        for done in [
            serde_json::json!({"type": "done", "decode_tok_s": 35.0}),
            serde_json::json!({"dflash": true, "tau": 3.1, "cycles": 9}),
            serde_json::json!({"mtp": true, "tau": 0.0, "cycles": 0}),
        ] {
            assert_eq!(bench_spec_mtp(&done), None, "{done}");
        }
    }

    #[test]
    fn bench_spec_warning_only_when_spec_requested_and_never_ran() {
        let ar = vec![None, None];
        assert!(bench_spec_warning(Some("mtp"), &ar).is_some());
        assert!(bench_spec_warning(Some("off"), &ar).is_none());
        assert!(bench_spec_warning(None, &ar).is_none());
        assert!(bench_spec_warning(Some("auto"), &[None, Some("mtp".into())]).is_none());
    }

    fn bench_args_for_test(prompt: Vec<String>, prompt_file: Option<PathBuf>) -> BenchArgs {
        BenchArgs {
            model: "qwen:test".to_owned(),
            runs: 1,
            tp: None,
            json: true,
            exp: false,
            matrix: false,
            ttft: false,
            pp: vec![128],
            ctx: vec![128],
            tg: 128,
            max_tokens: None,
            sustained_tg: None,
            sustained_ctx: vec![128],
            warmups: 1,
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            kv_backend: None,
            max_seq: None,
            redline: false,
            speculation: None,
            reasoning_on: false,
            concurrency: None,
            backend: "both".to_owned(),
            workload: "both".to_owned(),
            bench_shape: "steady".to_owned(),
            bb_corpus: None,
            num_prompts: None,
            temperature: None,
            prompt,
            prompt_file,
            emu: None,
        }
    }

    fn parse_bench_args(extra: &[&str]) -> BenchArgs {
        let mut argv = vec!["hipfire", "bench", "qwen:test"];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv).unwrap().command {
            Some(Commands::Bench(args)) => args,
            other => panic!("expected bench command, got {other:?}"),
        }
    }

    fn collect_standard_warmups(args: &BenchArgs, prompt: &str) -> (usize, Vec<serde_json::Value>) {
        let mut requests = Vec::new();
        let ran = run_standard_bench_warmups(args, prompt, |request| {
            requests.push(request.clone());
            Ok(())
        })
        .unwrap();
        (ran, requests)
    }

    #[test]
    fn bench_warmups_flag_defaults_to_ten_and_accepts_zero() {
        assert_eq!(parse_bench_args(&[]).warmups, 10);
        assert_eq!(parse_bench_args(&["--warmups", "3"]).warmups, 3);
        assert_eq!(parse_bench_args(&["--warmups", "0"]).warmups, 0);
    }

    #[test]
    fn standard_bench_zero_warmups_issue_no_requests() {
        let args = parse_bench_args(&["--warmups", "0"]);
        let (ran, requests) = collect_standard_warmups(&args, "measured prompt");
        assert_eq!(ran, 0);
        assert!(requests.is_empty());
    }

    #[test]
    fn standard_bench_runs_exactly_the_requested_warmups() {
        for (extra, expected) in [
            (vec!["--warmups", "1"], 1),
            (vec!["--warmups", "3"], 3),
            (vec![], 10),
        ] {
            let args = parse_bench_args(&extra);
            let (ran, requests) = collect_standard_warmups(&args, "measured prompt");
            assert_eq!(ran, expected, "{extra:?}");
            assert_eq!(requests.len(), expected, "{extra:?}");
        }
    }

    #[test]
    fn standard_bench_warmups_match_measured_prompt_and_max_tokens() {
        // Default budget (128) and an explicit one, never the old 16-token
        // "Hello" stand-in.
        for (extra, max_tokens) in [
            (vec!["--warmups", "2"], 128u64),
            (vec!["--warmups", "2", "--max-tokens", "37"], 37),
        ] {
            let args = parse_bench_args(&extra);
            let (_, requests) = collect_standard_warmups(&args, "measured prompt");
            assert_eq!(requests.len(), 2);
            for request in &requests {
                assert_eq!(request["type"], "generate");
                assert_eq!(request["prompt"], "measured prompt");
                assert_eq!(request["max_tokens"].as_u64(), Some(max_tokens));
            }
            let measured = bench_generate_request_reasoning(
                "measured prompt",
                args.standard_max_tokens() as u64,
                args.reasoning_on,
            );
            for key in [
                "type",
                "prompt",
                "temperature",
                "top_p",
                "repeat_penalty",
                "max_tokens",
                "attempt_id",
                "max_think_tokens",
                "assistant_prefix",
                "reasoning_effort",
            ] {
                assert_eq!(requests[0][key], measured[key], "{key}");
            }
        }
    }

    #[test]
    fn standard_bench_warmups_follow_reasoning_mode_of_measured_runs() {
        let args = parse_bench_args(&["--warmups", "1", "--reasoning-on", "--max-tokens", "512"]);
        let (_, requests) = collect_standard_warmups(&args, "p");
        assert_eq!(requests[0]["max_tokens"].as_u64(), Some(512));
        assert!(requests[0].get("max_think_tokens").is_none());
        assert!(requests[0].get("assistant_prefix").is_none());
        assert!(requests[0].get("reasoning_effort").is_none());
    }

    #[test]
    fn standard_bench_warmups_share_shape_with_distinct_ids() {
        let args = parse_bench_args(&["--warmups", "3"]);
        let (_, requests) = collect_standard_warmups(&args, "p");
        assert_eq!(requests.len(), 3);
        let ids: std::collections::HashSet<&str> = requests
            .iter()
            .map(|request| request["id"].as_str().expect("string id"))
            .collect();
        assert_eq!(ids.len(), 3, "every warmup needs its own request id");
        let strip = |request: &serde_json::Value| {
            let mut request = request.clone();
            request.as_object_mut().unwrap().remove("id");
            request
        };
        assert!(requests
            .iter()
            .all(|request| strip(request) == strip(&requests[0])));
    }

    #[test]
    fn standard_bench_report_records_actual_warmups_run() {
        for extra in [vec!["--warmups", "0"], vec!["--warmups", "4"], vec![]] {
            let args = parse_bench_args(&extra);
            let (ran, _) = collect_standard_warmups(&args, "p");
            let report = with_warmups_run(
                serde_json::json!({ "protocol": "native-generate-v1", "runs": args.runs }),
                ran,
            );
            assert_eq!(report["warmups_run"].as_u64(), Some(args.warmups as u64));
            assert_eq!(report["protocol"], "native-generate-v1");
            assert_eq!(report["runs"].as_u64(), Some(args.runs as u64));
        }
        // The recorded value is the count that ran, not the flag.
        let report = with_warmups_run(serde_json::json!({}), 2);
        assert_eq!(report["warmups_run"].as_u64(), Some(2));
    }

    #[test]
    fn ttft_and_matrix_keep_the_warmups_flag() {
        let ttft = parse_bench_args(&["--ttft", "--warmups", "3"]);
        assert!(ttft.ttft && !ttft.matrix);
        assert_eq!(ttft.warmups, 3);
        let matrix = parse_bench_args(&["--matrix", "--warmups", "1"]);
        assert!(matrix.matrix && !matrix.ttft);
        assert_eq!(matrix.warmups, 1);
        assert_eq!(parse_bench_args(&["--ttft"]).warmups, 10);
        assert_eq!(parse_bench_args(&["--matrix"]).warmups, 10);
    }

    #[test]
    fn standard_bench_warmup_error_stops_the_loop() {
        let args = parse_bench_args(&["--warmups", "5"]);
        let mut calls = 0;
        let result = run_standard_bench_warmups(&args, "p", |_| {
            calls += 1;
            if calls == 2 {
                bail!("daemon failed")
            }
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(calls, 2);
    }

    #[test]
    fn bench_prompt_file_conflicts_with_positional_prompt() {
        let err = Cli::try_parse_from([
            "hipfire",
            "bench",
            "qwen:test",
            "--prompt-file",
            "prompt.txt",
            "hello",
        ])
        .unwrap_err();
        assert!(
            err.to_string().contains("--prompt-file"),
            "conflict error should name the flag: {err}"
        );
    }

    #[test]
    fn bench_matrix_refuses_a_prompt() {
        for (prompt, prompt_file) in [
            (Vec::new(), Some(PathBuf::from("prompt.txt"))),
            (vec!["hello".to_owned()], None),
        ] {
            for redline in [false, true] {
                let mut args = bench_args_for_test(prompt.clone(), prompt_file.clone());
                args.matrix = !redline;
                args.redline = redline;
                let err = bench_command(&test_paths("bench-matrix-prompt"), args).unwrap_err();
                assert!(
                    err.to_string().contains("cannot take a prompt"),
                    "matrix with a prompt must be refused before any load: {err}"
                );
            }
        }
    }

    #[test]
    fn bench_max_seq_flag_takes_the_memory_max_seq_bounds() {
        let parse = |value: &str| -> std::result::Result<Option<u64>, clap::Error> {
            let cli = Cli::try_parse_from(["hipfire", "bench", "qwen:test", "--max-seq", value])?;
            match cli.command {
                Some(Commands::Bench(args)) => Ok(args.max_seq),
                other => panic!("expected bench, got {other:?}"),
            }
        };
        assert_eq!(parse("512").unwrap(), Some(512));
        assert_eq!(parse("1048576").unwrap(), Some(1_048_576));
        assert!(parse("511").is_err(), "below the memory.max_seq minimum");
        assert!(
            parse("1048577").is_err(),
            "above the memory.max_seq maximum"
        );
        assert!(parse("8k").is_err());
    }

    #[test]
    fn sweep_headroom_want_is_weights_plus_a_fifth() {
        assert_eq!(sweep_headroom_want_bytes(0), 0);
        assert_eq!(sweep_headroom_want_bytes(10_000_000_000), 12_000_000_000);
    }

    #[test]
    fn bench_matrix_seq_need_is_the_longest_probe_plus_headroom() {
        // Decode rows dominate the default-like shape: 20000 + 128 + 32.
        assert_eq!(
            bench_matrix_seq_need(&[128, 8192], &[128, 20000], 128, None),
            20_160
        );
        // A long prefill dominates a short decode row.
        assert_eq!(bench_matrix_seq_need(&[8192], &[128], 32, None), 8_224);
        // Sustained rows count with their own decode length...
        assert_eq!(
            bench_matrix_seq_need(&[512], &[128], 32, Some((256, &[8192, 30000]))),
            30_288
        );
        // ...and not at all when `--sustained-tg` does not run them.
        assert_eq!(bench_matrix_seq_need(&[512], &[128], 32, None), 544);
    }

    #[test]
    fn bench_matrix_topology_is_the_daemons_actual_pp_tp_ep_cards() {
        let probe = |pp: u64, tp: u64, ep: u64, cards: &[&str]| {
            serde_json::json!({
                "type": "prefill_result",
                "tokens": 128,
                "topology": { "pp": pp, "tp": tp, "ep": ep, "cards": cards },
            })
        };
        let report = |responses: &[serde_json::Value], loaded_pp: u64| {
            let mut tracker = BenchTopologyTracker::default();
            for response in responses {
                tracker.observe(response)?;
            }
            tracker.finish(&serde_json::json!({ "pp": loaded_pp }))
        };
        // Single, PP, dense TP and EP each report their own shape and card
        // identifiers; TP and EP stay distinct and none collapses to Single.
        let shapes: [(u64, u64, u64, &[&str]); 4] = [
            (1, 1, 1, &["0"]),
            (3, 1, 1, &["0", "1", "2"]),
            (1, 2, 1, &["0", "1"]),
            (1, 1, 3, &["0", "1", "2"]),
        ];
        for (pp, tp, ep, cards) in shapes {
            let value = report(&[probe(pp, tp, ep, cards), probe(pp, tp, ep, cards)], pp).unwrap();
            assert_eq!(
                value,
                serde_json::json!({ "pp": pp, "tp": tp, "ep": ep, "cards": cards })
            );
        }
        // A daemon that reports no topology yields null, never a guessed Single.
        let bare = serde_json::json!({ "type": "decode_result", "tok_s": 1.0 });
        assert_eq!(report(&[bare.clone()], 1).unwrap(), serde_json::Value::Null);
        // A decode answer without it does not erase an earlier prefill answer.
        assert_eq!(
            report(&[probe(3, 1, 1, &["0", "1", "2"]), bare], 3).unwrap(),
            serde_json::json!({ "pp": 3, "tp": 1, "ep": 1, "cards": ["0", "1", "2"] })
        );
        // Rows that disagree (including on the card list), a malformed
        // object, a numeric/empty card list, and a topology that contradicts
        // the load's pp are all errors.
        let err = report(
            &[probe(1, 1, 1, &["0"]), probe(3, 1, 1, &["0", "1", "2"])],
            1,
        )
        .unwrap_err();
        assert!(err.to_string().contains("disagree"), "{err}");
        let err = report(&[probe(1, 1, 1, &["0"]), probe(1, 1, 1, &["1"])], 1).unwrap_err();
        assert!(err.to_string().contains("disagree"), "{err}");
        let malformed = serde_json::json!({
            "type": "prefill_result",
            "topology": { "pp": 1, "tp": 1, "ep": 0, "cards": ["0"] },
        });
        let err = report(&[malformed], 1).unwrap_err();
        assert!(err.to_string().contains("`ep`"), "{err}");
        for cards in [
            serde_json::json!(1),
            serde_json::json!([]),
            serde_json::json!([0]),
        ] {
            let bad = serde_json::json!({
                "type": "prefill_result",
                "topology": { "pp": 1, "tp": 1, "ep": 1, "cards": cards },
            });
            let err = report(&[bad], 1).unwrap_err();
            assert!(err.to_string().contains("`cards`"), "{err}");
        }
        let err = report(&[probe(1, 1, 1, &["0"])], 3).unwrap_err();
        assert!(err.to_string().contains("loaded pp=3"), "{err}");
    }

    #[test]
    fn bench_max_seq_sizes_a_matrix_run_within_the_model_limit() {
        let plan = plan_bench_max_seq(None, Some(8_224), Some(262_144), "m").unwrap();
        assert_eq!(
            plan,
            BenchMaxSeq {
                requested: Some(8_224),
                source: BenchMaxSeqSource::RunShape,
                run_need: Some(8_224),
                model_limit: Some(262_144),
            }
        );
        // A tiny matrix is raised to the memory.max_seq minimum...
        let plan = plan_bench_max_seq(None, Some(160), None, "m").unwrap();
        assert_eq!(plan.requested, Some(512));
        // ...but never past what the model declares.
        let plan = plan_bench_max_seq(None, Some(160), Some(400), "m").unwrap();
        assert_eq!(plan.requested, Some(400));
        // The need itself may sit exactly at the limit.
        let plan = plan_bench_max_seq(None, Some(4_096), Some(4_096), "m").unwrap();
        assert_eq!(plan.requested, Some(4_096));
    }

    #[test]
    fn bench_max_seq_refuses_a_shape_past_the_model_limit() {
        for flag in [None, Some(1_048_576)] {
            let err = plan_bench_max_seq(flag, Some(262_177), Some(262_144), "flash")
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("262177") && err.contains("262144") && err.contains("flash"),
                "must name the need, the limit and the model: {err}"
            );
        }
    }

    #[test]
    fn bench_max_seq_flag_wins_but_must_hold_the_run() {
        let err = plan_bench_max_seq(Some(8_192), Some(8_224), Some(262_144), "m")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("8192") && err.contains("8224"),
            "must name the flag and the need: {err}"
        );
        let plan = plan_bench_max_seq(Some(8_224), Some(8_224), None, "m").unwrap();
        assert_eq!(
            (plan.requested, plan.source),
            (Some(8_224), BenchMaxSeqSource::Flag)
        );
        // The declared context bounds the run shape, not the flag: past it,
        // the daemon's admission decides.
        let plan = plan_bench_max_seq(Some(300_000), Some(8_224), Some(262_144), "m").unwrap();
        assert_eq!(plan.requested, Some(300_000));
        // Off the matrix the flag is sent as given.
        let plan = plan_bench_max_seq(Some(4_096), None, None, "m").unwrap();
        assert_eq!(
            (plan.requested, plan.source),
            (Some(4_096), BenchMaxSeqSource::Flag)
        );
    }

    #[test]
    fn bench_max_seq_stays_automatic_off_the_matrix_without_a_flag() {
        let plan = plan_bench_max_seq(None, None, None, "m").unwrap();
        assert_eq!(
            (plan.requested, plan.source),
            (None, BenchMaxSeqSource::Automatic)
        );
    }

    #[test]
    fn declared_context_reads_max_position_embeddings_like_admission() {
        let read = |json: &str| declared_context_from_metadata(json);
        assert_eq!(
            read(
                r#"{"config":{"max_position_embeddings":4096,"text_config":{"max_position_embeddings":262144}}}"#
            ),
            Some(262_144),
            "text_config wins over the outer config"
        );
        assert_eq!(
            read(r#"{"config":{"max_position_embeddings":131072}}"#),
            Some(131_072)
        );
        assert_eq!(read(r#"{"max_position_embeddings":40960}"#), Some(40_960));
        assert_eq!(read(r#"{"config":{"max_position_embeddings":0}}"#), None);
        assert_eq!(read(r#"{"config":{}}"#), None);
        assert_eq!(read("not json"), None);
    }

    #[test]
    fn bench_resolve_prompt_keeps_historical_default() {
        let args = bench_args_for_test(Vec::new(), None);
        assert_eq!(
            resolve_bench_prompt(&args).unwrap(),
            "Explain the theory of general relativity in simple terms."
        );
    }

    #[test]
    fn bench_resolve_prompt_joins_positional_words() {
        let args = bench_args_for_test(vec!["hello".to_owned(), "world".to_owned()], None);
        assert_eq!(resolve_bench_prompt(&args).unwrap(), "hello world");
    }

    #[test]
    fn bench_resolve_prompt_file_is_verbatim() {
        let path = std::env::temp_dir().join("hipfire-bench-prompt-verbatim.txt");
        // Trailing newline included: the file is read as raw bytes, never trimmed.
        std::fs::write(&path, "repeat after me\n").unwrap();
        let args = bench_args_for_test(Vec::new(), Some(path.clone()));
        let prompt = resolve_bench_prompt(&args).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(prompt, "repeat after me\n");
        assert_eq!(
            bench_prompt_md5(&prompt),
            format!("{:x}", md5::compute(b"repeat after me\n"))
        );
    }

    #[test]
    fn bench_resolve_prompt_rejects_file_and_positional() {
        let args = bench_args_for_test(vec!["hello".to_owned()], Some(PathBuf::from("prompt.txt")));
        assert!(resolve_bench_prompt(&args).is_err());
    }

    #[test]
    fn bench_prompt_warning_threshold() {
        // The default short prompt must warn; 256+ tokens must not.
        let short = bench_prompt_warning(24).expect("24 tokens must warn");
        assert!(
            short.contains("launch overhead"),
            "unexpected text: {short}"
        );
        assert!(
            short.contains("24"),
            "warning should name the count: {short}"
        );
        assert!(bench_prompt_warning(255).is_some());
        assert!(bench_prompt_warning(256).is_none());
        assert!(bench_prompt_warning(4400).is_none());
    }

    #[test]
    fn bench_prompt_tokens_come_from_prefill_plus_cached() {
        // The daemon's done event names the prompt as prefill_tokens (+ any
        // prompt-cache hit in cached_tokens); there is no prompt_tokens key.
        let done = serde_json::json!({"prefill_tokens": 4400, "cached_tokens": 8});
        assert_eq!(bench_prompt_tokens_from_done(&done), Some(4408));
        let no_cache = serde_json::json!({"prefill_tokens": 24});
        assert_eq!(bench_prompt_tokens_from_done(&no_cache), Some(24));
        let neither = serde_json::json!({"tokens": 128, "prompt_tokens": 99});
        assert_eq!(bench_prompt_tokens_from_done(&neither), None);
    }

    #[test]
    fn bench_prompt_md5_is_hex_of_prompt_bytes() {
        // md5("abc") is a fixed vector; guards against swapping in sha256.
        assert_eq!(bench_prompt_md5("abc"), "900150983cd24fb0d6963f7d28e17f72");
    }

    /// Gate for the Low-power "Shorter thinking" tuning line: a chat request
    /// that names no effort picks up the global `reasoning.effort = "low"`
    /// and forwards it for an effort-native model. If this stops holding the
    /// line must leave `hipfire_config::tuning::plan`.
    #[test]
    fn global_reasoning_effort_low_reaches_an_effort_native_request() {
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        let mut layer = ConfigLayer::default();
        layer
            .set(
                "reasoning.effort",
                hipfire_config::ConfigValue::String("low".into()),
            )
            .unwrap();
        let configured = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let mut request = serde_json::json!({});
        let resolution = apply_http_reasoning_request(
            &serde_json::json!({ "messages": [{ "role": "user", "content": "hi" }] }),
            &configured,
            &mut request,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(request["reasoning_effort"], "low");
        assert_eq!(resolution.effective_effort.as_deref(), Some("low"));
        assert!(resolution.warnings.is_empty(), "{:?}", resolution.warnings);

        // Without the global value the same request does not get "low".
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut baseline = serde_json::json!({});
        let baseline_resolution = apply_http_reasoning_request(
            &serde_json::json!({ "messages": [{ "role": "user", "content": "hi" }] }),
            &defaults,
            &mut baseline,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_ne!(baseline_resolution.effective_effort.as_deref(), Some("low"));

        // An explicit request effort still wins over the config value.
        let mut explicit = serde_json::json!({});
        let explicit_resolution = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "xhigh" }),
            &configured,
            &mut explicit,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(
            explicit_resolution.effective_effort.as_deref(),
            Some("xhigh")
        );
    }

    #[test]
    fn http_reasoning_nested_max_tokens_alias_resolves_cap_source() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "low",
                "reasoning": { "max_tokens": 2048 }
            }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req["max_think_tokens"], 2048);
        assert_eq!(res.effective_cap, Some(2048));
        assert_eq!(res.cap_source, "explicit:body:reasoning.max_tokens");
        assert!(res.warnings.is_empty());
    }

    #[test]
    fn http_reasoning_top_level_max_think_tokens_precedes_nested_alias() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        let mut conflicting = serde_json::json!({});
        let res_conflict = apply_http_reasoning_request(
            &serde_json::json!({
                "max_think_tokens": 4096,
                "reasoning": { "max_tokens": 2048 }
            }),
            &resolved,
            &mut conflicting,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(conflicting["max_think_tokens"], 4096);
        assert_eq!(res_conflict.effective_cap, Some(4096));
        assert_eq!(res_conflict.cap_source, "explicit:body:max_think_tokens");
        assert!(res_conflict.warnings.iter().any(|warning| {
            warning.contains("reasoning.max_tokens")
                && warning.contains("max_think_tokens")
                && warning.contains("precedence")
        }));

        let mut equal = serde_json::json!({});
        let res_equal = apply_http_reasoning_request(
            &serde_json::json!({
                "max_think_tokens": 2048,
                "reasoning": { "max_tokens": 2048 }
            }),
            &resolved,
            &mut equal,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(equal["max_think_tokens"], 2048);
        assert_eq!(res_equal.effective_cap, Some(2048));
        assert_eq!(res_equal.cap_source, "explicit:body:max_think_tokens");
        assert!(
            res_equal
                .warnings
                .iter()
                .all(|warning| !warning.contains("reasoning.max_tokens")),
            "equal duplicate caps must not warn"
        );
    }

    #[test]
    fn http_reasoning_malformed_nested_max_tokens_is_hard_error() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        for bad in [
            serde_json::json!({ "reasoning": { "max_tokens": -1 } }),
            serde_json::json!({ "reasoning": { "max_tokens": 1.5 } }),
            serde_json::json!({ "reasoning": { "max_tokens": "2048" } }),
            serde_json::json!({ "reasoning": { "max_tokens": 393217 } }),
        ] {
            let mut req = serde_json::json!({});
            let err = apply_http_reasoning_request(
                &bad,
                &resolved,
                &mut req,
                ReasoningContract::QwenJinja,
                true,
                &supported,
            )
            .expect_err("malformed nested reasoning.max_tokens must hard-error");
            let message = format!("{err:#}");
            assert!(
                message.contains("reasoning.max_tokens")
                    && message.contains("must be between 0 and 393216"),
                "unexpected error wording: {message}"
            );
        }
    }

    #[test]
    fn http_reasoning_three_toggle_sources_disabled_wins_once() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        // pairwise: top true vs kwargs false
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "chat_template_kwargs": { "enable_thinking": false }
            }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req["thinking_enabled"], false);
        assert_eq!(res.effective_mode, "disabled");
        let warns: Vec<_> = res
            .warnings
            .iter()
            .filter(|w| w.contains("conflicting thinking toggles"))
            .collect();
        assert_eq!(warns.len(), 1, "pairwise conflict must warn once");

        // pairwise: kwargs true vs thinking.type disabled
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "thinking": { "type": "disabled" }
            }),
            &resolved,
            &mut req2,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req2["thinking_enabled"], false);
        assert_eq!(res2.effective_mode, "disabled");
        assert_eq!(
            res2.warnings
                .iter()
                .filter(|w| w.contains("conflicting thinking toggles"))
                .count(),
            1
        );

        // all three: top true, kwargs true, thinking disabled => disabled wins, single warning
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "chat_template_kwargs": { "enable_thinking": true },
                "thinking": { "type": "disabled" }
            }),
            &resolved,
            &mut req3,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req3["thinking_enabled"], false);
        assert_eq!(res3.effective_mode, "disabled");
        assert_eq!(
            res3.warnings
                .iter()
                .filter(|w| w.contains("conflicting thinking toggles"))
                .count(),
            1
        );

        // all three agree true => no conflict warning, enabled
        let mut req4 = serde_json::json!({});
        let res4 = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "chat_template_kwargs": { "enable_thinking": true },
                "thinking": { "type": "enabled" }
            }),
            &resolved,
            &mut req4,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req4["thinking_enabled"], true);
        assert_eq!(res4.effective_mode, "enabled");
        assert_eq!(
            res4.warnings
                .iter()
                .filter(|w| w.contains("conflicting thinking toggles"))
                .count(),
            0
        );
    }

    #[test]
    fn http_reasoning_gemma_enabled_with_cap_and_budget_dropped() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        // Gemma explicit enable true must remain enabled and report no cap, even with caps present
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "max_think_tokens": 1234,
                "thinking_budget": "high",
                "reasoning_effort": "low"
            }),
            &resolved,
            &mut req,
            ReasoningContract::GemmaBoolean,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(req["thinking_enabled"], true);
        assert_eq!(res.effective_mode, "enabled");
        assert!(
            req.get("max_think_tokens").is_none(),
            "Gemma must not send cap"
        );
        assert!(res.effective_cap.is_none());
        assert_eq!(res.cap_source, "none");
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("gemma_boolean") && w.contains("reasoning_effort")));
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("gemma_boolean") && w.contains("max_think_tokens")));
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("gemma_boolean") && w.contains("thinking_budget")));

        // Config caps also dropped for Gemma when thinking enabled
        let mut layer = ConfigLayer::default();
        layer.set_cli("reasoning.max_tokens", "4096").unwrap();
        layer.set_cli("reasoning.budget", "low").unwrap();
        let cfg_resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "test".into(),
            },
            layer,
        }])
        .unwrap();
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "enable_thinking": true }),
            &cfg_resolved,
            &mut req2,
            ReasoningContract::GemmaBoolean,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(req2["thinking_enabled"], true);
        assert!(req2.get("max_think_tokens").is_none());
        assert!(res2.effective_cap.is_none());
        assert!(res2
            .warnings
            .iter()
            .any(|w| w.contains("reasoning.max_tokens")));
    }

    #[test]
    fn http_reasoning_gemma_budget_off_does_not_disable() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        // Gemma with enable true + budget off must stay enabled (budget off ignored for Gemma)
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "thinking_budget": "off"
            }),
            &resolved,
            &mut req,
            ReasoningContract::GemmaBoolean,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(req["thinking_enabled"], true);
        assert_eq!(res.effective_mode, "enabled");
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("gemma_boolean") && w.contains("thinking_budget")));
        // Gemma default disabled without explicit enable, budget off should not change that (still disabled via default, not via budget)
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking_budget": "off" }),
            &resolved,
            &mut req2,
            ReasoningContract::GemmaBoolean,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(res2.effective_mode, "disabled");
        // budget off for non-Gemma Qwen should disable
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "thinking_budget": "off"
            }),
            &resolved,
            &mut req3,
            ReasoningContract::QwenJinja,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(req3["thinking_enabled"], false);
        assert_eq!(res3.effective_mode, "disabled");
    }

    #[test]
    fn http_reasoning_invalid_enum_warns_not_hard_error_and_malformed_hard_errors() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        // unknown thinking.type string -> warn+drop, not error, results in default enabled for Qwen
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({ "thinking": { "type": "maybe" } }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert!(!res.warnings.is_empty());
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("thinking.type") && w.contains("dropped")));
        assert_eq!(res.effective_mode, "enabled"); // default for Qwen

        // unknown non-native thinking_budget -> warn+drop, not error
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking_budget": "turbo" }),
            &resolved,
            &mut req2,
            ReasoningContract::QwenJinja,
            false,
            &supported,
        )
        .unwrap();
        assert!(req2.get("max_think_tokens").is_none());
        assert!(res2.effective_cap.is_none());
        assert!(res2
            .warnings
            .iter()
            .any(|w| w.contains("thinking_budget") && w.contains("dropped")));

        // wrong JSON type for enable_thinking -> hard error
        let mut req3 = serde_json::json!({});
        let err = apply_http_reasoning_request(
            &serde_json::json!({ "enable_thinking": "true" }),
            &resolved,
            &mut req3,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("enable_thinking must be a boolean"));

        // wrong JSON type for thinking.type -> hard error
        let mut req4 = serde_json::json!({});
        let err2 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking": { "type": 123 } }),
            &resolved,
            &mut req4,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap_err();
        assert!(format!("{err2}").contains("thinking.type must be enabled or disabled"));

        // cap range violation -> hard error (body)
        let mut req5 = serde_json::json!({});
        let err3 = apply_http_reasoning_request(
            &serde_json::json!({ "max_think_tokens": 999999 }),
            &resolved,
            &mut req5,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap_err();
        assert!(format!("{err3}").contains("must be between 0 and 393216"));
    }

    #[test]
    fn http_reasoning_qwen_jinja_off_ladder_effort_normalizes_to_nearest() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        for (raw, want) in [
            ("high", "xhigh"),
            ("max", "xhigh"),
            ("med", "medium"),
            ("minimal", "low"),
        ] {
            let mut req = serde_json::json!({});
            let res = apply_http_reasoning_request(
                &serde_json::json!({ "reasoning_effort": raw }),
                &resolved,
                &mut req,
                ReasoningContract::QwenJinja,
                true,
                &supported,
            )
            .unwrap();
            assert_eq!(req["reasoning_effort"], want, "{raw}");
            assert_eq!(res.effective_effort.as_deref(), Some(want), "{raw}");
            assert!(
                res.warnings.iter().any(|w| w.contains("normalized")),
                "{raw}"
            );
        }
        // A restricted ladder that excludes the nearest tier still falls to the
        // qwen_jinja default.
        let restricted = vec!["low".to_string(), "medium".to_string()];
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "high" }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &restricted,
        )
        .unwrap();
        assert_eq!(req["reasoning_effort"], "medium");
        assert_eq!(res.effective_effort.as_deref(), Some("medium"));
    }

    #[test]
    fn http_reasoning_nested_max_tokens_and_qwen_deepseek_glimmer_contracts_intact() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        // Nested reasoning.max_tokens still works
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning": { "max_tokens": 2048 } }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &["low".to_string(), "medium".to_string(), "xhigh".to_string()],
        )
        .unwrap();
        assert_eq!(req["max_think_tokens"], 2048);
        assert_eq!(res.effective_cap, Some(2048));
        assert_eq!(res.cap_source, "explicit:body:reasoning.max_tokens");

        // Qwen non-native still drops effort
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "low" }),
            &resolved,
            &mut req2,
            ReasoningContract::QwenJinja,
            false,
            &[],
        )
        .unwrap();
        assert!(req2.get("reasoning_effort").is_none());
        assert!(res2
            .warnings
            .iter()
            .any(|w| w.contains("does not natively support effort")));

        // DeepSeek effort mapping intact
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "medium" }),
            &resolved,
            &mut req3,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert_eq!(req3["reasoning_effort"], "high");
        assert_eq!(res3.effective_effort.as_deref(), Some("high"));

        // Glimmer always-on intact
        let mut req4 = serde_json::json!({});
        let res4 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking": { "type": "disabled" } }),
            &resolved,
            &mut req4,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap();
        assert_eq!(req4["thinking_enabled"], true);
        assert_eq!(res4.effective_mode, "enabled");
    }

    #[test]
    fn http_reasoning_qwen_non_native_effort_warning_is_template_based_and_keeps_caps() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "low",
                "max_think_tokens": 2048
            }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            false,
            &[],
        )
        .unwrap();
        assert!(req.get("reasoning_effort").is_none());
        assert_eq!(res.effective_effort, None);
        assert_eq!(req["max_think_tokens"], 2048);
        assert_eq!(res.effective_cap, Some(2048));
        assert_eq!(res.cap_source, "explicit:body:max_think_tokens");
        let warning = res
            .warnings
            .iter()
            .find(|w| w.contains("reasoning_effort 'low'"))
            .expect("non-native effort warning");
        assert!(warning.contains("loaded chat template does not natively support effort"));
        assert!(warning.contains("native effort is unavailable for this model"));
        assert!(warning.contains("thinking_budget"));
        assert!(warning.contains("max_think_tokens"));
        assert!(!warning.contains("Qwen3.6"));

        let mut layer = ConfigLayer::default();
        layer.set_cli("reasoning.effort", "high").unwrap();
        layer.set_cli("reasoning.budget", "high").unwrap();
        let cfg = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "test".into(),
            },
            layer,
        }])
        .unwrap();
        let mut cfg_req = serde_json::json!({});
        let cfg_res = apply_http_reasoning_request(
            &serde_json::json!({}),
            &cfg,
            &mut cfg_req,
            ReasoningContract::QwenJinja,
            false,
            &[],
        )
        .unwrap();
        assert!(cfg_req.get("reasoning_effort").is_none());
        assert_eq!(cfg_res.effective_effort, None);
        assert_eq!(cfg_req["max_think_tokens"], 8192);
        assert_eq!(cfg_res.effective_cap, Some(8192));
        assert_eq!(cfg_res.cap_source, "config:reasoning.budget");
        let cfg_warning = cfg_res
            .warnings
            .iter()
            .find(|w| w.contains("reasoning.effort 'high'"))
            .expect("non-native config effort warning");
        assert!(cfg_warning.contains("loaded chat template does not natively support effort"));
        assert!(cfg_warning.contains("native effort is unavailable for this model"));
        assert!(cfg_warning.contains("thinking_budget"));
        assert!(cfg_warning.contains("max_think_tokens"));
        assert!(!cfg_warning.contains("Qwen3.6"));
    }

    #[test]
    fn http_reasoning_qwen_native_effort_value_native_warns_and_falls_back_to_medium() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = ["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "native",
                "max_think_tokens": 2048
            }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req["reasoning_effort"], "medium");
        assert_eq!(res.effective_effort.as_deref(), Some("medium"));
        assert_eq!(req["max_think_tokens"], 2048);
        assert_eq!(res.effective_cap, Some(2048));
        assert_eq!(res.cap_source, "explicit:body:max_think_tokens");
        let warning = res
            .warnings
            .iter()
            .find(|w| w.contains("reasoning_effort 'native'"))
            .expect("native effort warning");
        assert!(warning.contains("not a supported effort value"));
        assert!(warning.contains("low|medium|xhigh"));
        assert!(warning.contains("medium"));
    }

    #[test]
    fn http_reasoning_deepseek_explicit_caps_dropped() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "high",
                "max_think_tokens": 4096
            }),
            &resolved,
            &mut req,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert!(req.get("max_think_tokens").is_none());
        assert!(res.effective_cap.is_none());
        assert_eq!(res.cap_source, "none");
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("max_think_tokens") && w.contains("deepseek4")));
        assert_eq!(req["reasoning_effort"], "high");
        assert_eq!(req["thinking_enabled"], true);
        assert_eq!(res.effective_mode, "enabled");
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning": { "max_tokens": 2048 } }),
            &resolved,
            &mut req2,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert!(req2.get("max_think_tokens").is_none());
        assert!(res2.effective_cap.is_none());
        assert!(res2
            .warnings
            .iter()
            .any(|w| w.contains("reasoning.max_tokens")));
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking_budget": "high" }),
            &resolved,
            &mut req3,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert!(req3.get("max_think_tokens").is_none());
        assert!(res3
            .warnings
            .iter()
            .any(|w| w.contains("thinking_budget") && w.contains("deepseek4")));
        let mut layer = ConfigLayer::default();
        layer.set_cli("reasoning.max_tokens", "8192").unwrap();
        let cfg = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "test".into(),
            },
            layer,
        }])
        .unwrap();
        let mut req4 = serde_json::json!({});
        let res4 = apply_http_reasoning_request(
            &serde_json::json!({}),
            &cfg,
            &mut req4,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert!(req4.get("max_think_tokens").is_none());
        assert!(res4.effective_cap.is_none());
        assert!(res4
            .warnings
            .iter()
            .any(|w| w.contains("reasoning.max_tokens")));
        let mut bad = serde_json::json!({});
        let err = apply_http_reasoning_request(
            &serde_json::json!({ "max_think_tokens": "bad" }),
            &resolved,
            &mut bad,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap_err();
        assert!(format!("{err}").contains("must be between 0 and 393216"));
    }

    #[test]
    fn http_reasoning_glimmer_explicit_caps_dropped() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "low",
                "max_think_tokens": 512
            }),
            &resolved,
            &mut req,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap();
        assert!(req.get("max_think_tokens").is_none());
        assert!(res.effective_cap.is_none());
        assert_eq!(res.cap_source, "none");
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("max_think_tokens") && w.contains("muse_glimmer")));
        assert_eq!(req["reasoning_effort"], "low");
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning": { "max_tokens": 1024 },
                "thinking_budget": "low"
            }),
            &resolved,
            &mut req2,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap();
        assert!(req2.get("max_think_tokens").is_none());
        assert!(res2.effective_cap.is_none());
        assert_eq!(res2.cap_source, "none");
        assert!(res2.warnings.iter().any(|w| w.contains("muse_glimmer")));
        let mut layer = ConfigLayer::default();
        layer.set_cli("reasoning.budget", "high").unwrap();
        let cfg = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "test".into(),
            },
            layer,
        }])
        .unwrap();
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({}),
            &cfg,
            &mut req3,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap();
        assert!(req3.get("max_think_tokens").is_none());
        assert!(res3.effective_cap.is_none());
        assert!(res3
            .warnings
            .iter()
            .any(|w| w.contains("muse_glimmer") || w.contains("thinking_budget")));
        let mut bad = serde_json::json!({});
        let err = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning": { "max_tokens": 500000 } }),
            &resolved,
            &mut bad,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap_err();
        assert!(format!("{err}").contains("must be between 0 and 393216"));
    }

    #[test]
    fn head_forces_local_even_when_service_would_be_ready() {
        let with_head = RunArgs {
            model: "maple-preview".into(),
            prompt: vec![],
            temp: None,
            top_p: None,
            repeat_penalty: None,
            max_tokens: None,
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            head: Some("q4k".into()),
            kv_backend: None,
            tp: None,
            max_seq: None,
            speculation: None,
            model_draft: None,
            vision: None,
            draft_max: None,
            dspark_conf_threshold: None,
            system: None,
            image: None,
            json: false,
            stats: false,
            no_stream: false,
        };
        let without_head = RunArgs {
            head: None,
            ..with_head.clone()
        };
        // --head must force local; without head should not force local by itself
        assert!(
            run_should_force_local(&with_head),
            "--head must force local load path"
        );
        assert!(
            !run_should_force_local(&without_head),
            "without head and no other flags should not force local"
        );
    }

    #[test]
    fn existing_artifact_valid_detects_fresh_and_stale() {
        use sha2::{Digest, Sha256};
        let dir = env::temp_dir().join(format!(
            "hipfire-artifact-valid-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.mq4");
        let content = b"fresh content";
        fs::write(&path, content).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(content);
        let sha = format!("{:x}", hasher.finalize());
        let size = content.len() as u64;
        let off = VerifiedCache::disabled();
        assert!(existing_artifact_valid(&off, &path, Some(&sha), Some(size)));
        assert!(!existing_artifact_valid(
            &off,
            &path,
            Some(&sha),
            Some(size + 1)
        ));
        assert!(!existing_artifact_valid(
            &off,
            &path,
            Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
            Some(size)
        ));
        // No expectations means existence alone is valid
        assert!(existing_artifact_valid(&off, &path, None, None));
        // Missing file is invalid
        assert!(!existing_artifact_valid(
            &off,
            &dir.join("missing"),
            Some(&sha),
            Some(size)
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_params_preserves_auto_for_direct_path_and_lowers_registry_bf16() {
        // Direct-path load: no registry entry, config is auto -> must stay auto.
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        assert_eq!(config_string(&defaults, "memory.kv_cache").unwrap(), "auto");
        let direct_path = PathBuf::from("/tmp/direct-model.mq4");
        let params = load_params(
            &defaults,
            None,
            &direct_path.parent().unwrap(),
            &direct_path,
            64,
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            params["kv_mode"], "auto",
            "direct-path auto must survive to architecture"
        );

        // A non-q8 registry default is a model-specific opinion and must lower
        // into the resolved config before load parameters are built.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{
                "maple-preview":{"repo":"x","file":"maple-preview.mq2lloydu","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"bf16"}
            },
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (_, entry) = registry.model("maple-preview").unwrap();
        let resolved = resolve(vec![NamedLayer {
            source: ConfigSource::RegistryModel {
                tag: "maple-preview".into(),
                revision: "test".into(),
            },
            layer: entry.config_layer().unwrap(),
        }])
        .unwrap();
        let params2 = load_params(
            &resolved,
            Some(entry),
            &direct_path.parent().unwrap(),
            &direct_path,
            64,
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            params2["kv_mode"], "bf16",
            "non-q8 registry default must resolve to BF16"
        );

        // Explicit override still wins.
        let params3 = load_params(
            &resolved,
            Some(entry),
            &direct_path.parent().unwrap(),
            &direct_path,
            64,
            Some("q8"),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(params3["kv_mode"], "q8");
    }

    fn write_test_registry_cache(paths: &Paths, raw: &str) {
        let registry = RegistryV1::parse(raw, "test-cache").unwrap();
        let url = env::var("HIPFIRE_REGISTRY_URL")
            .unwrap_or_else(|_| "https://example.com/test.json".into());
        let cache = serde_json::json!({
            "fetched_at": unix_timestamp() * 1000,
            "url": url,
            "registry": registry
        });
        fs::create_dir_all(paths.registry.cache.parent().unwrap()).unwrap();
        fs::write(
            &paths.registry.cache,
            serde_json::to_string(&cache).unwrap(),
        )
        .unwrap();
    }

    fn tiny_http_server(
        files: std::collections::HashMap<String, Vec<u8>>,
    ) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base = format!("http://127.0.0.1:{}", addr.port());
        let expected = files.len();
        let handle = std::thread::spawn(move || {
            let mut served = 0usize;
            listener.set_nonblocking(false).unwrap();
            for stream in listener.incoming() {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let mut buf = [0u8; 4096];
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                // Extract file name from request path: /{repo}/resolve/main/{file}
                let mut body: Option<Vec<u8>> = None;
                for (name, data) in &files {
                    if req.contains(name) {
                        body = Some(data.clone());
                        break;
                    }
                }
                if let Some(data) = body {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        data.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&data);
                } else {
                    let header =
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = stream.write_all(header.as_bytes());
                }
                let _ = stream.flush();
                served += 1;
                if served >= expected {
                    break;
                }
            }
        });
        // small pause to let listener start
        std::thread::sleep(Duration::from_millis(50));
        (base, handle)
    }

    #[test]
    fn pull_fresh_downloads_heads_with_hash_verification() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("pull-fresh-heads");
        // Build tiny artifacts
        let base_content = b"base-model-content";
        let head_q4k_content = b"head-q4k-content";
        let head_bf16_content = b"head-bf16-content";
        use sha2::{Digest, Sha256};
        let sha = |data: &[u8]| {
            let mut h = Sha256::new();
            h.update(data);
            format!("{:x}", h.finalize())
        };
        let raw = format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{{
                "test-model":{{
                    "repo":"test/repo",
                    "file":"test-model.mq4",
                    "size_gb":0.001,
                    "min_vram_gb":1,
                    "desc":"x",
                    "default_kv_mode":"bf16",
                    "heads":{{
                        "q4k":{{"file":"test-model-head-q4k.hfq","sha256":"{}","size_bytes":{}}},
                        "bf16":{{"file":"test-model-head-bf16.hfq","sha256":"{}","size_bytes":{}}}
                    }},
                    "sha256":"{}",
                    "size_bytes":{}
                }}
            }},
            "aliases":{{}}
        }}"#,
            sha(head_q4k_content),
            head_q4k_content.len(),
            sha(head_bf16_content),
            head_bf16_content.len(),
            sha(base_content),
            base_content.len()
        );
        write_test_registry_cache(&paths, &raw);
        let mut files = std::collections::HashMap::new();
        files.insert("test-model.mq4".to_string(), base_content.to_vec());
        files.insert(
            "test-model-head-q4k.hfq".to_string(),
            head_q4k_content.to_vec(),
        );
        files.insert(
            "test-model-head-bf16.hfq".to_string(),
            head_bf16_content.to_vec(),
        );
        let (base, handle) = tiny_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        // Pull should download base + both heads
        pull_command(&paths, PullArgs::defaults("test-model")).unwrap();
        assert_eq!(
            fs::read(paths.models.join("test-model.mq4")).unwrap(),
            base_content
        );
        assert_eq!(
            fs::read(paths.models.join("test-model-head-q4k.hfq")).unwrap(),
            head_q4k_content
        );
        assert_eq!(
            fs::read(paths.models.join("test-model-head-bf16.hfq")).unwrap(),
            head_bf16_content
        );
        let _ = handle.join();
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn pull_stale_same_name_artifact_refreshes_atomically() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("pull-stale-refresh");
        let fresh = b"fresh-content";
        let stale = b"stale-old-content";
        use sha2::{Digest, Sha256};
        let sha = |data: &[u8]| {
            let mut h = Sha256::new();
            h.update(data);
            format!("{:x}", h.finalize())
        };
        let raw = format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{{
                "test-model":{{
                    "repo":"test/repo",
                    "file":"test-model.mq4",
                    "size_gb":0.001,
                    "min_vram_gb":1,
                    "desc":"x",
                    "sha256":"{}",
                    "size_bytes":{}
                }}
            }},
            "aliases":{{}}
        }}"#,
            sha(fresh),
            fresh.len()
        );
        write_test_registry_cache(&paths, &raw);
        fs::create_dir_all(&paths.models).unwrap();
        // Place stale artifact with same name but wrong hash/size
        fs::write(paths.models.join("test-model.mq4"), stale).unwrap();
        assert!(!existing_artifact_valid(
            &VerifiedCache::disabled(),
            &paths.models.join("test-model.mq4"),
            Some(&sha(fresh)),
            Some(fresh.len() as u64)
        ));
        let mut files = std::collections::HashMap::new();
        files.insert("test-model.mq4".to_string(), fresh.to_vec());
        let (base, handle) = tiny_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        // Without --force, stale should still be detected and refreshed
        pull_command(&paths, PullArgs::defaults("test-model")).unwrap();
        assert_eq!(
            fs::read(paths.models.join("test-model.mq4")).unwrap(),
            fresh
        );
        let _ = handle.join();
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_removes_heads_alongside_base() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("rm-heads");
        let valid_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let raw = format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{{
                "test-model":{{
                    "repo":"test/repo",
                    "file":"test-model.mq4",
                    "size_gb":0.001,
                    "min_vram_gb":1,
                    "desc":"x",
                    "heads":{{
                        "q4k":{{"file":"test-model-head-q4k.hfq","sha256":"{sha}","size_bytes":3}},
                        "bf16":{{"file":"test-model-head-bf16.hfq","sha256":"{sha}","size_bytes":3}}
                    }},
                    "sha256":"{sha}",
                    "size_bytes":3
                }}
            }},
            "aliases":{{}}
        }}"#,
            sha = valid_sha
        );
        write_test_registry_cache(&paths, &raw);
        fs::create_dir_all(&paths.models).unwrap();
        fs::write(paths.models.join("test-model.mq4"), b"base").unwrap();
        fs::write(paths.models.join("test-model-head-q4k.hfq"), b"q4k").unwrap();
        fs::write(paths.models.join("test-model-head-bf16.hfq"), b"bf16").unwrap();
        assert!(paths.models.join("test-model-head-q4k.hfq").is_file());
        rm_command(
            &paths,
            RmArgs {
                model: "test-model".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!paths.models.join("test-model.mq4").exists());
        assert!(!paths.models.join("test-model-head-q4k.hfq").exists());
        assert!(!paths.models.join("test-model-head-bf16.hfq").exists());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// A malformed `HF_TOKEN` is refused before the registry is read or any
    /// request is sent, with an error naming the variable (issue #655).
    #[test]
    fn pull_rejects_malformed_hf_token_before_any_request() {
        let _env_lock = TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let paths = test_paths("pull-bad-hf-token");
        // An unreachable endpoint: any attempted download would fail differently.
        let _base = EnvGuard::set("HIPFIRE_HF_BASE", "http://127.0.0.1:1");
        for (value, what) in [
            ("hf_secret\n", "a newline"),
            ("hf_secret ", "a space"),
            ("\"hf_secret\"", "a double quote"),
            ("hf_s\u{e9}cret", "a non-ASCII character"),
        ] {
            let _token = EnvGuard::set("HF_TOKEN", value);
            let error = pull_command(&paths, PullArgs::defaults("test-model"))
                .expect_err("malformed token must be refused");
            let text = format!("{error:#}");
            assert!(text.contains("HF_TOKEN"), "{text}");
            assert!(text.contains(what), "{text}");
            assert!(!text.contains("secret"), "{text}");
            assert!(!text.contains("download request failed"), "{text}");
        }
        assert!(
            !paths.models.exists(),
            "nothing is created before the check"
        );
    }

    #[test]
    fn download_verified_rejects_malformed_token_without_creating_files() {
        let _env_lock = TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _token = EnvGuard::set("HF_TOKEN", "hf_bad token");
        let dir = std::env::temp_dir().join(format!("hipfire-bad-token-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let destination = dir.join("model.mq4");
        let error = download_verified(
            &VerifiedCache::disabled(),
            "http://127.0.0.1:1/model.mq4",
            &destination,
            None,
            None,
            true,
        )
        .expect_err("malformed token must be refused");
        let text = format!("{error:#}");
        assert!(text.contains("environment variable HF_TOKEN"), "{text}");
        assert!(text.contains("a space at character 7"), "{text}");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir_all(&dir).unwrap();
    }

    static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvGuard {
        key: String,
        prev: Option<std::ffi::OsString>,
    }
    impl EnvGuard {
        fn set(key: &str, val: &str) -> Self {
            let prev = env::var_os(key);
            env::set_var(key, val);
            Self {
                key: key.to_string(),
                prev,
            }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(v) = &self.prev {
                env::set_var(&self.key, v);
            } else {
                env::remove_var(&self.key);
            }
        }
    }

    /// Raw `POST /v1/chat/completions` with `Connection: close`: status, the
    /// lowercased response head, and the de-chunked body. Panics if a chunked
    /// body ends without its terminating zero-size chunk (a torn stream).
    #[cfg(unix)]
    fn raw_chat_post(port: u16, body: &serde_json::Value) -> (u16, String, Vec<u8>) {
        let payload = body.to_string();
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect serve");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read timeout");
        write!(
            stream,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{payload}",
            payload.len()
        )
        .expect("write request");
        let mut raw = Vec::new();
        let _ = stream.read_to_end(&mut raw);
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("response head");
        let head = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .expect("status code");
        let mut rest = &raw[split + 4..];
        if !head.contains("transfer-encoding: chunked") {
            return (status, head, rest.to_vec());
        }
        let mut body = Vec::new();
        loop {
            let line_end = rest
                .windows(2)
                .position(|w| w == b"\r\n")
                .unwrap_or_else(|| {
                    panic!(
                        "torn chunked body after {:?}",
                        String::from_utf8_lossy(&body)
                    )
                });
            let size = usize::from_str_radix(
                std::str::from_utf8(&rest[..line_end]).expect("chunk size"),
                16,
            )
            .expect("chunk size hex");
            rest = &rest[line_end + 2..];
            if size == 0 {
                return (status, head, body);
            }
            assert!(
                rest.len() >= size + 2,
                "torn chunked body after {:?}",
                String::from_utf8_lossy(&body)
            );
            body.extend_from_slice(&rest[..size]);
            rest = &rest[size + 2..];
        }
    }

    /// `data:` payloads of an SSE body, in order.
    #[cfg(unix)]
    fn sse_payloads(body: &[u8]) -> Vec<String> {
        String::from_utf8_lossy(body)
            .split("\n\n")
            .filter_map(|frame| frame.strip_prefix("data: "))
            .map(str::to_owned)
            .collect()
    }

    /// Error status comes from the daemon's typed class, and a stream that
    /// fails before its first frame is an ordinary JSON error, not a 200.
    #[cfg(unix)]
    #[test]
    fn typed_errors_keep_their_status_stream_and_nonstream() {
        let harness = Task11HttpHarness::spawn("error-status");
        let port = harness.port();
        for stream in [false, true] {
            for (tag, status) in [
                ("t15-class-validation", 400),
                ("t15-class-context", 400),
                ("t15-class-unsupported", 400),
                ("t15-class-internal", 500),
                ("t15-transient-always", 503),
            ] {
                let (got, head, body) = raw_chat_post(port, &harness.base_body(tag, stream));
                assert_eq!(got, status, "{tag} stream={stream}: {head}");
                assert!(
                    head.contains("content-type: application/json"),
                    "{tag}: {head}"
                );
                let err: serde_json::Value = serde_json::from_slice(&body)
                    .unwrap_or_else(|e| panic!("{tag}: one JSON error body ({e}): {body:?}"));
                assert!(
                    err.pointer("/error/message")
                        .and_then(|v| v.as_str())
                        .is_some(),
                    "{tag}: {err}"
                );
                assert_eq!(
                    head.contains("retry-after: 1"),
                    status == 503,
                    "{tag}: Retry-After only on 503: {head}"
                );
            }
            // Gateway validation (no daemon class) also fails before commit.
            let mut body = harness.base_body("t11-stop-text", stream);
            body["max_tokens"] = serde_json::json!(0);
            let (got, _, raw) = raw_chat_post(port, &body);
            assert_eq!(got, 400, "max_tokens=0 stream={stream}");
            assert!(String::from_utf8_lossy(&raw).contains("max_tokens"));
        }
    }

    /// Daemon prefill progress precedes the first token, i.e. the commit: it
    /// must not commit the stream early (an error right after it is still a
    /// JSON error with its real status) and must not leak into a committed
    /// body that has not yet needed a keepalive.
    #[cfg(unix)]
    #[test]
    fn prefill_progress_does_not_commit_or_corrupt_responses() {
        let harness = Task11HttpHarness::spawn("prefill-progress");
        let port = harness.port();
        for stream in [false, true] {
            let (status, head, body) = raw_chat_post(
                port,
                &harness.base_body("t43-prefill-progress-error", stream),
            );
            assert_eq!(status, 400, "stream={stream}: {head}");
            assert!(head.contains("content-type: application/json"), "{head}");
            let err: serde_json::Value = serde_json::from_slice(&body)
                .unwrap_or_else(|e| panic!("one JSON error body ({e}): {body:?}"));
            assert!(
                err.pointer("/error/message")
                    .and_then(|v| v.as_str())
                    .is_some_and(|m| m.contains("validation failed after prefill")),
                "{err}"
            );
        }

        let (status, _, body) =
            raw_chat_post(port, &harness.base_body("t43-prefill-progress", false));
        assert_eq!(status, 200);
        let reply: serde_json::Value = serde_json::from_slice(&body).expect("one JSON body");
        assert_eq!(
            reply.pointer("/choices/0/message/content"),
            Some(&serde_json::json!("hello from fake daemon"))
        );

        let (status, _, body) =
            raw_chat_post(port, &harness.base_body("t43-prefill-progress", true));
        assert_eq!(status, 200);
        let text = String::from_utf8_lossy(&body);
        assert!(
            !text.contains(": prefill"),
            "progress predates commit: {text}"
        );
        let payloads = sse_payloads(&body);
        assert!(
            payloads[1].contains("hello from fake daemon"),
            "{payloads:?}"
        );
        assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
        let cap = capture_stream(port, harness.base_body("t43-prefill-progress", true))
            .expect("stream parses");
        assert_eq!(cap.content, "hello from fake daemon");
    }

    /// A stream that fails after it committed ends with an OpenAI error event
    /// and `[DONE]` inside a cleanly terminated body, never a torn one.
    #[cfg(unix)]
    #[test]
    fn stream_failure_after_first_token_ends_with_error_event_and_done() {
        let harness = Task11HttpHarness::spawn("midstream-error");
        let port = harness.port();
        let (status, _, body) = raw_chat_post(port, &harness.base_body("t15-visible-token", true));
        assert_eq!(status, 200);
        let payloads = sse_payloads(&body);
        let [role, token, error, done] = payloads.as_slice() else {
            panic!("role, token, error, [DONE] expected: {payloads:?}");
        };
        assert!(role.contains(r#""role":"assistant""#), "{role}");
        assert!(token.contains("visible-before-fail"), "{token}");
        let error: serde_json::Value = serde_json::from_str(error).expect("error event JSON");
        assert_eq!(error["error"]["type"], "server_error", "{error}");
        assert!(
            error["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("transient after visible token")),
            "{error}"
        );
        assert_eq!(done, "[DONE]");

        // The OpenAI-compatible client reports the server's message.
        match capture_stream(port, harness.base_body("t15-visible-token", true)) {
            Err(hipfire_client::ClientError::Http(message)) => {
                assert!(
                    message.contains("transient after visible token"),
                    "{message}"
                )
            }
            other => panic!("expected the stream's error event, got {other:?}"),
        }

        // Daemon death mid-stream is reported the same way.
        let (status, _, body) = raw_chat_post(port, &harness.base_body("t11-premature-eof", true));
        assert_eq!(status, 200);
        let payloads = sse_payloads(&body);
        assert!(payloads[1].contains("partial-before-eof"), "{payloads:?}");
        assert!(payloads[2].contains(r#""error""#), "{payloads:?}");
        assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
    }

    /// Force the post-kill EOF error to finish on the main thread before the
    /// signal handler exits. Isolate the process-wide flag and exit in a child.
    #[test]
    fn bench_interrupt_wins_over_daemon_eof_error() {
        const CHILD_ENV: &str = "BENCH_INTERRUPT_EXIT_TEST_CHILD";
        if env::var_os(CHILD_ENV).is_some() {
            BENCH_INTERRUPT.store(true, Ordering::SeqCst);
            finish_command(Err(anyhow!(
                "daemon stdout closed during interrupt cleanup"
            )));
            panic!("interrupted command returned");
        }
        let output = Command::new(env::current_exe().unwrap())
            .args([
                "tests::bench_interrupt_wins_over_daemon_eof_error",
                "--exact",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(BENCH_INTERRUPTED_EXIT),
            "daemon EOF must not replace an interrupt with exit 1: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// Marks the child run of `accept_errors_do_not_stop_serving` (not a
    /// product setting, so no `HIPFIRE_` prefix).
    #[cfg(unix)]
    const EMFILE_CHILD_ENV: &str = "T11_EMFILE_CHILD";

    /// A failed `accept` must not end the server. A child process runs the
    /// serve harness and then takes every free descriptor, so the kernel
    /// completes this test's connect but serve's `accept` fails with EMFILE.
    /// Once the child gives the descriptors back, the same connection must
    /// be served.
    #[cfg(unix)]
    #[test]
    fn accept_errors_do_not_stop_serving() {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};
        if env::var_os(EMFILE_CHILD_ENV).is_some() {
            return emfile_child();
        }
        let mut child = Command::new(env::current_exe().unwrap())
            .args([
                "tests::accept_errors_do_not_stop_serving",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(EMFILE_CHILD_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn child test");
        // Keep the read end open until the child exits: a closed pipe would
        // make its test output fail.
        let mut child_stdout = BufReader::new(child.stdout.take().unwrap());
        let port: u16 = loop {
            let mut line = String::new();
            assert!(
                child_stdout.read_line(&mut line).unwrap() > 0,
                "child exited before it held every descriptor"
            );
            // libtest prints "test <name> ... " on the same line first.
            if let Some((_, port)) = line.trim().split_once("EMFILE_READY ") {
                break port.parse().unwrap();
            }
        };
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut child_stdin = child.stdin.take().unwrap();
        writeln!(child_stdin, "connected").unwrap();
        let mut response = String::new();
        let _ = stream.read_to_string(&mut response);
        drop(child_stdin);
        let output = child.wait_with_output().expect("child test");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("accept failed"),
            "serve never hit an accept error: {stderr}"
        );
        assert!(
            response.starts_with("HTTP/1.1 200"),
            "connection not served after EMFILE: {response:?}\n{stderr}"
        );
        assert!(output.status.success(), "child test failed: {stderr}");
        drop(child_stdout);
    }

    #[cfg(unix)]
    fn emfile_child() {
        let harness = Task11HttpHarness::spawn("emfile");
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
            0
        );
        limit.rlim_cur = limit.rlim_cur.min(256);
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);
        let mut hogs = Vec::new();
        while let Ok(file) = fs::File::open("/dev/null") {
            hogs.push(file);
        }
        println!("EMFILE_READY {}", harness.port());
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        // Serve fails its accept and backs off meanwhile.
        thread::sleep(Duration::from_millis(300));
        drop(hogs);
        // The parent closes stdin once it has its response.
        let _ = std::io::stdin().read_line(&mut line);
    }

    /// One `/metrics` sample value, by exact series name.
    #[cfg(unix)]
    fn metric(port: u16, name: &str) -> u64 {
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut text = String::new();
        stream.read_to_string(&mut text).unwrap();
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{name} ")))
            .unwrap_or_else(|| panic!("{name} missing from /metrics: {text}"))
            .parse()
            .unwrap()
    }

    /// Failed generations and admission rejections reach `/metrics`, each in
    /// its own counter; successes reach neither.
    #[cfg(unix)]
    #[test]
    fn failures_and_admission_rejections_are_counted() {
        let harness = Task11HttpHarness::spawn("metrics-counters");
        let port = harness.port();
        let failed = "hipfire_requests_failed_total";
        let rejected = "hipfire_admission_rejected_total";

        // JSON error, stream rejected before commit, stream failing after it.
        for (tag, stream, status) in [
            ("t15-class-validation", false, 400),
            ("t15-class-internal", true, 500),
            ("t15-visible-token", true, 200),
            ("t11-stop-text", false, 200),
            ("t11-stop-text", true, 200),
        ] {
            let (got, _, _) = raw_chat_post(port, &harness.base_body(tag, stream));
            assert_eq!(got, status, "{tag} stream={stream}");
        }
        assert_eq!(metric(port, failed), 3);
        assert_eq!(metric(port, rejected), 0);

        // Hold the only slot and fill the four-deep queue: the next request
        // is refused by admission.
        let held = harness.shared.admission.acquire().unwrap();
        let queued: Vec<_> = (0..4)
            .map(|_| {
                let body = harness.base_body("t11-stop-text", false);
                thread::spawn(move || raw_chat_post(port, &body).0)
            })
            .collect();
        let deadline = Instant::now() + Duration::from_secs(5);
        while harness.shared.admission.inflight() < 5 {
            assert!(Instant::now() < deadline, "queue never filled");
            thread::sleep(Duration::from_millis(10));
        }
        let (status, head, _) = raw_chat_post(port, &harness.base_body("t11-stop-text", false));
        assert_eq!(status, 503, "{head}");
        drop(held);
        for waiter in queued {
            assert_eq!(waiter.join().unwrap(), 200);
        }
        assert_eq!(metric(port, rejected), 1);
        assert_eq!(metric(port, failed), 3);
    }

    /// An image request queued for admission must not block the server: the
    /// harness serves on one runtime thread, which a blocking wait would hold.
    #[cfg(unix)]
    #[test]
    fn queued_image_request_does_not_block_the_server() {
        let harness = Task11HttpHarness::spawn("images-admission");
        let port = harness.port();
        let held = harness.shared.admission.acquire().unwrap();
        let image = thread::spawn(move || {
            // `images` is refused only after admission, so this never
            // reaches the daemon once it leaves the queue.
            let payload = r#"{"prompt":"a red cube","images":["x"]}"#;
            let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            write!(
                stream,
                "POST /v1/images/generations HTTP/1.1\r\nHost: 127.0.0.1\r\n\
                 Content-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{payload}",
                payload.len()
            )
            .unwrap();
            let mut response = String::new();
            let _ = stream.read_to_string(&mut response);
            response
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while harness.shared.admission.inflight() < 2 {
            assert!(Instant::now() < deadline, "image request never queued");
            thread::sleep(Duration::from_millis(10));
        }

        let mut health = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        health
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        health
            .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut response = String::new();
        let _ = health.read_to_string(&mut response);
        assert!(
            response.starts_with("HTTP/1.1 200"),
            "/health stalled behind a queued image request: {response:?}"
        );

        drop(held);
        let response = image.join().unwrap();
        assert!(
            response.starts_with("HTTP/1.1 400") && response.contains("not a field"),
            "{response}"
        );
    }

    /// One HTTP/1.1 response from a kept-alive connection: status and body,
    /// de-chunked or by Content-Length. Fails if the response does not end.
    #[cfg(unix)]
    fn read_keepalive_response(
        reader: &mut std::io::BufReader<std::net::TcpStream>,
    ) -> (u16, Vec<u8>) {
        use std::io::BufRead;
        let mut line = String::new();
        reader.read_line(&mut line).expect("status line");
        let status = line
            .split_whitespace()
            .nth(1)
            .expect("status")
            .parse()
            .unwrap();
        let (mut length, mut chunked) = (None, false);
        loop {
            line.clear();
            reader.read_line(&mut line).expect("header");
            let header = line.trim().to_ascii_lowercase();
            if header.is_empty() {
                break;
            }
            if let Some(value) = header.strip_prefix("content-length:") {
                length = Some(value.trim().parse::<usize>().unwrap());
            }
            chunked |= header == "transfer-encoding: chunked";
        }
        let mut body = Vec::new();
        if !chunked {
            body.resize(length.expect("content-length"), 0);
            reader.read_exact(&mut body).expect("body");
            return (status, body);
        }
        loop {
            line.clear();
            reader
                .read_line(&mut line)
                .expect("chunk size (response never ended)");
            let size = usize::from_str_radix(line.trim(), 16).expect("chunk size hex");
            let mut chunk = vec![0; size + 2];
            reader.read_exact(&mut chunk).expect("chunk");
            if size == 0 {
                return (status, body);
            }
            body.extend_from_slice(&chunk[..size]);
        }
    }

    /// A client that keeps its connection gets complete, prompt responses to
    /// a stream, then a non-stream, then a stream request on it; and a client
    /// that closes as soon as it reads `[DONE]` does not hold the server up.
    #[cfg(unix)]
    #[test]
    fn keepalive_connection_serves_stream_and_nonstream_in_turn() {
        let harness = Task11HttpHarness::spawn("keepalive-reuse");
        let port = harness.port();
        let stream = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut writer = stream.try_clone().unwrap();
        let mut reader = std::io::BufReader::new(stream);
        for is_stream in [true, false, true] {
            let payload = harness.base_body("t11-stop-text", is_stream).to_string();
            write!(
                writer,
                "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n\
                 Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
                payload.len()
            )
            .unwrap();
            let started = Instant::now();
            let (status, body) = read_keepalive_response(&mut reader);
            assert_eq!(status, 200, "stream={is_stream}");
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "stream={is_stream} took {:?}",
                started.elapsed()
            );
            if is_stream {
                let payloads = sse_payloads(&body);
                assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
                assert!(payloads
                    .iter()
                    .any(|p| p.contains("hello from fake daemon")));
            } else {
                let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(value["choices"][0]["finish_reason"], "stop");
            }
        }

        // Close as soon as `[DONE]` arrives, like serve_harness and most SSE
        // readers, before the chunked terminator.
        let mut early = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect");
        early
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let payload = harness.base_body("t11-stop-text", true).to_string();
        write!(
            early,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
            payload.len()
        )
        .unwrap();
        let mut seen = Vec::new();
        let mut buf = [0u8; 4096];
        while !String::from_utf8_lossy(&seen).contains("data: [DONE]") {
            let n = early.read(&mut buf).expect("read until [DONE]");
            assert!(n > 0, "closed before [DONE]");
            seen.extend_from_slice(&buf[..n]);
        }
        drop(early);
        let (status, _, _) = raw_chat_post(port, &harness.base_body("t11-stop-text", false));
        assert_eq!(status, 200);
    }

    // ---- PLE split layout: pull, on-demand, tag fallback, rm, listings --------

    fn ple_sha(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(data))
    }

    /// Serves exactly `files.len()` requests, matching the LAST path segment
    /// exactly (the shared `tiny_http_server` matches substrings, which cannot
    /// tell `x-base.mq4` from `x-base.mq4.ple-q8`).
    fn exact_http_server(
        files: std::collections::HashMap<String, Vec<u8>>,
    ) -> (String, std::thread::JoinHandle<()>) {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let expected = files.len();
        let handle = std::thread::spawn(move || {
            for _ in 0..expected {
                let (mut stream, _) = listener.accept().unwrap();
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]).into_owned();
                let target = request
                    .split_whitespace()
                    .nth(1)
                    .and_then(|path| path.rsplit('/').next())
                    .unwrap_or_default()
                    .to_owned();
                match files.get(&target) {
                    Some(data) => {
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            data.len()
                        );
                        let _ = stream.write_all(header.as_bytes());
                        let _ = stream.write_all(data);
                    }
                    None => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        );
                    }
                }
                let _ = stream.flush();
            }
        });
        (base, handle)
    }

    const PLE_BASE: &[u8] = b"split-base-bytes";
    const PLE_Q8: &[u8] = b"q8-rows-bytes";
    const PLE_Q4: &[u8] = b"q4-rows";

    /// `split-model` is the split entry (base + both slots).
    fn split_registry_raw() -> String {
        format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{{
                "split-model":{{
                    "repo":"test/repo","file":"split-model-base.mq4","size_gb":0.001,
                    "min_vram_gb":1,"desc":"x","sha256":"{}","size_bytes":{},
                    "ple":{{
                        "q8":{{"file":"split-model-base.mq4.ple-q8","sha256":"{}","size_bytes":{}}},
                        "q4":{{"file":"split-model-base.mq4.ple-q4","sha256":"{}","size_bytes":{}}}
                    }}
                }}
            }},
            "aliases":{{}}
        }}"#,
            ple_sha(PLE_BASE),
            PLE_BASE.len(),
            ple_sha(PLE_Q8),
            PLE_Q8.len(),
            ple_sha(PLE_Q4),
            PLE_Q4.len(),
        )
    }

    #[test]
    fn pull_downloads_the_base_and_only_the_selected_ple_sidecars() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let cases = [
            (ple::PleSelection::Q8, false, true),
            (ple::PleSelection::Q4, true, false),
            (ple::PleSelection::All, true, true),
            (ple::PleSelection::Skip, false, false),
        ];
        for (index, (selection, want_q4, want_q8)) in cases.into_iter().enumerate() {
            let paths = test_paths(&format!("ple-pull-{index}"));
            write_test_registry_cache(&paths, &split_registry_raw());
            let mut files = std::collections::HashMap::new();
            files.insert("split-model-base.mq4".to_owned(), PLE_BASE.to_vec());
            if want_q8 {
                files.insert("split-model-base.mq4.ple-q8".to_owned(), PLE_Q8.to_vec());
            }
            if want_q4 {
                files.insert("split-model-base.mq4.ple-q4".to_owned(), PLE_Q4.to_vec());
            }
            let (base, handle) = exact_http_server(files);
            let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
            pull_command(
                &paths,
                PullArgs {
                    ple: Some(selection),
                    ..PullArgs::defaults("split-model")
                },
            )
            .unwrap();
            handle.join().unwrap();
            let models = &paths.models;
            assert_eq!(
                fs::read(models.join("split-model-base.mq4")).unwrap(),
                PLE_BASE
            );
            assert_eq!(
                fs::read(models.join("split-model-base.mq4.ple-q8")).ok(),
                want_q8.then(|| PLE_Q8.to_vec()),
                "{selection:?}"
            );
            assert_eq!(
                fs::read(models.join("split-model-base.mq4.ple-q4")).ok(),
                want_q4.then(|| PLE_Q4.to_vec()),
                "{selection:?}"
            );
            fs::remove_dir_all(&paths.root).unwrap();
        }
    }

    #[test]
    fn pull_rejects_a_ple_sidecar_that_fails_verification() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("ple-pull-corrupt");
        write_test_registry_cache(&paths, &split_registry_raw());
        let mut files = std::collections::HashMap::new();
        files.insert("split-model-base.mq4".to_owned(), PLE_BASE.to_vec());
        files.insert(
            "split-model-base.mq4.ple-q8".to_owned(),
            b"tampered-rows".to_vec(),
        );
        let (base, handle) = exact_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        let error = pull_command(
            &paths,
            PullArgs {
                ple: Some(ple::PleSelection::Q8),
                ..PullArgs::defaults("split-model")
            },
        )
        .unwrap_err();
        handle.join().unwrap();
        assert!(format!("{error:#}").contains("mismatch"), "{error:#}");
        assert!(
            !paths.models.join("split-model-base.mq4.ple-q8").exists(),
            "a failed verification must not leave the sidecar installed"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn pull_fetches_draft_sidecars_by_default_and_skips_them_with_no_draft() {
        const BASE: &[u8] = b"draft-base";
        const MTP: &[u8] = b"mtp-head";
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let raw = format!(
            r#"{{"schema_version":1,"generated_at":"2099-01-01T00:00:00Z","models":{{
                "draft-model":{{"repo":"test/repo","file":"draft-model.mq4","size_gb":0.001,
                    "min_vram_gb":1,"desc":"x","sha256":"{}","size_bytes":{},
                    "mtp":{{"file":"draft-model.mtp","sha256":"{}","size_bytes":{}}}}}
            }},"aliases":{{}}}}"#,
            ple_sha(BASE),
            BASE.len(),
            ple_sha(MTP),
            MTP.len()
        );
        for (index, no_draft) in [false, true].into_iter().enumerate() {
            let paths = test_paths(&format!("draft-pull-{index}"));
            write_test_registry_cache(&paths, &raw);
            let mut files = std::collections::HashMap::new();
            files.insert("draft-model.mq4".to_owned(), BASE.to_vec());
            if !no_draft {
                files.insert("draft-model.mtp".to_owned(), MTP.to_vec());
            }
            let (base, handle) = exact_http_server(files);
            let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
            DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
            pull_command(
                &paths,
                PullArgs {
                    no_draft,
                    ..PullArgs::defaults("draft-model")
                },
            )
            .unwrap();
            handle.join().unwrap();
            DOWNLOAD_INVOCATIONS
                .with(|count| assert_eq!(count.get(), if no_draft { 1 } else { 2 }));
            assert_eq!(
                fs::read(paths.models.join("draft-model.mtp")).ok(),
                (!no_draft).then(|| MTP.to_vec())
            );
            fs::remove_dir_all(&paths.root).unwrap();
        }
    }

    #[test]
    fn on_demand_downloads_the_selected_sidecar_once_and_leaves_usable_files_alone() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let paths = test_paths("ple-on-demand");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = RegistryV1::parse(&split_registry_raw(), "ple on-demand").unwrap();
        let (tag, entry) = registry.model("split-model").unwrap();
        let model = paths.models.join(&entry.file);
        fs::write(&model, PLE_BASE).unwrap();
        let q8 = paths.models.join("split-model-base.mq4.ple-q8");
        let q4 = paths.models.join("split-model-base.mq4.ple-q4");

        // q8 requested and absent: one verified download.
        let mut files = std::collections::HashMap::new();
        files.insert("split-model-base.mq4.ple-q8".to_owned(), PLE_Q8.to_vec());
        let (base, handle) = exact_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        ple::ensure_ple_sidecar(
            Some(entry),
            Some(tag),
            &model,
            hipfire_config::memory::PleFormatRequest::Q8,
            "",
            true,
        )
        .unwrap();
        handle.join().unwrap();
        assert_eq!(fs::read(&q8).unwrap(), PLE_Q8);
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 1));

        // Present now: q8 and q4 requests, and auto, download nothing (no server).
        for request in [
            hipfire_config::memory::PleFormatRequest::Q8,
            hipfire_config::memory::PleFormatRequest::Q4,
            hipfire_config::memory::PleFormatRequest::Auto,
        ] {
            ple::ensure_ple_sidecar(Some(entry), Some(tag), &model, request, "", true).unwrap();
        }
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 1));
        assert!(
            !q4.exists(),
            "q4 is derived locally from the q8 rows (`hipfire ple derive`), never downloaded"
        );

        // q4 requested with neither file present: the q4 sidecar is fetched.
        fs::remove_file(&q8).unwrap();
        let mut files = std::collections::HashMap::new();
        files.insert("split-model-base.mq4.ple-q4".to_owned(), PLE_Q4.to_vec());
        let (base, handle) = exact_http_server(files);
        let _hf2 = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        ple::ensure_ple_sidecar(
            Some(entry),
            Some(tag),
            &model,
            hipfire_config::memory::PleFormatRequest::Q4,
            "",
            true,
        )
        .unwrap();
        handle.join().unwrap();
        assert_eq!(fs::read(&q4).unwrap(), PLE_Q4);
        assert!(!q8.exists());
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 2));
        fs::remove_dir_all(&paths.root).unwrap();
    }

    // ---- Shared Flash-Next PLE pair: one family file, per-base link aliases ---

    const SHARED_BASE_A: &[u8] = b"shared-base-a-mq4";
    const SHARED_BASE_B: &[u8] = b"shared-base-b-mq3";
    const SHARED_FILE_A: &str = "qwen3.8-flash-next-base.mq4-xts";
    const SHARED_FILE_B: &str = "qwen3.8-flash-next-base.mq3-xts";
    const SHARED_Q8_FILE: &str = "qwen3.8-flash-next.ple-q8";
    const SHARED_Q4_FILE: &str = "qwen3.8-flash-next.ple-q4";

    /// Two Flash-Next split bases (`flash-a`, `flash-b`) naming ONE shared pair.
    fn shared_registry_raw() -> String {
        let slots = format!(
            r#""ple":{{"q8":{{"file":"{SHARED_Q8_FILE}","sha256":"{}","size_bytes":{}}},
               "q4":{{"file":"{SHARED_Q4_FILE}","sha256":"{}","size_bytes":{}}}}}"#,
            ple_sha(PLE_Q8),
            PLE_Q8.len(),
            ple_sha(PLE_Q4),
            PLE_Q4.len(),
        );
        format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{{
                "flash-a":{{
                    "repo":"test/repo","file":"{SHARED_FILE_A}","size_gb":0.001,
                    "min_vram_gb":1,"desc":"x","sha256":"{}","size_bytes":{},{slots}
                }},
                "flash-b":{{
                    "repo":"test/repo","file":"{SHARED_FILE_B}","size_gb":0.001,
                    "min_vram_gb":1,"desc":"x","sha256":"{}","size_bytes":{},{slots}
                }}
            }},
            "aliases":{{}}
        }}"#,
            ple_sha(SHARED_BASE_A),
            SHARED_BASE_A.len(),
            ple_sha(SHARED_BASE_B),
            SHARED_BASE_B.len(),
        )
    }

    fn same_inode(a: &Path, b: &Path) -> bool {
        use std::os::unix::fs::MetadataExt;
        let (a, b) = (fs::metadata(a).unwrap(), fs::metadata(b).unwrap());
        a.dev() == b.dev() && a.ino() == b.ino()
    }

    fn loader_path(
        paths: &Paths,
        base: &str,
        format: hipfire_config::memory::PleFormat,
    ) -> PathBuf {
        hipfire_config::memory::ple_sidecar_path(&paths.models.join(base), format)
    }

    fn pull_shared(paths: &Paths, model: &str, selection: ple::PleSelection) -> Result<()> {
        pull_command(
            paths,
            PullArgs {
                ple: Some(selection),
                ..PullArgs::defaults(model)
            },
        )
    }

    fn serve_files(entries: &[(&str, &[u8])]) -> (String, std::thread::JoinHandle<()>) {
        exact_http_server(
            entries
                .iter()
                .map(|(name, data)| ((*name).to_owned(), data.to_vec()))
                .collect(),
        )
    }

    #[test]
    fn pull_downloads_the_shared_pair_once_and_links_each_base_alias() {
        use hipfire_config::memory::PleFormat;
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("ple-shared-pull");
        write_test_registry_cache(&paths, &shared_registry_raw());
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        let canonical_q8 = paths.models.join(SHARED_Q8_FILE);
        let canonical_q4 = paths.models.join(SHARED_Q4_FILE);

        // flash-a --ple q8: base + the shared q8; the loader's name is a link.
        let (base, handle) =
            serve_files(&[(SHARED_FILE_A, SHARED_BASE_A), (SHARED_Q8_FILE, PLE_Q8)]);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_shared(&paths, "flash-a", ple::PleSelection::Q8).unwrap();
        handle.join().unwrap();
        let a_q8 = loader_path(&paths, SHARED_FILE_A, PleFormat::Q8);
        assert_eq!(fs::read(&canonical_q8).unwrap(), PLE_Q8);
        assert_eq!(fs::read(&a_q8).unwrap(), PLE_Q8);
        assert!(
            same_inode(&a_q8, &canonical_q8),
            "alias is a hard link, not a copy"
        );
        assert!(!canonical_q4.exists(), "--ple q8 leaves q4 alone");
        assert!(!loader_path(&paths, SHARED_FILE_A, PleFormat::Q4).exists());
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 2));

        // flash-b --ple all: its base and the shared q4 download; the q8 is reused.
        let (base, handle) =
            serve_files(&[(SHARED_FILE_B, SHARED_BASE_B), (SHARED_Q4_FILE, PLE_Q4)]);
        let _hf2 = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_shared(&paths, "flash-b", ple::PleSelection::All).unwrap();
        handle.join().unwrap();
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 4));
        let b_q8 = loader_path(&paths, SHARED_FILE_B, PleFormat::Q8);
        let b_q4 = loader_path(&paths, SHARED_FILE_B, PleFormat::Q4);
        assert_eq!(fs::read(&b_q8).unwrap(), PLE_Q8);
        assert_eq!(fs::read(&b_q4).unwrap(), PLE_Q4);
        assert!(same_inode(&b_q8, &canonical_q8));
        assert!(same_inode(&b_q4, &canonical_q4));
        assert!(same_inode(&a_q8, &b_q8), "both bases serve the same bytes");

        // flash-a --ple all afterwards downloads nothing (no server is running).
        let _dead = EnvGuard::set("HIPFIRE_HF_BASE", "http://127.0.0.1:9");
        pull_shared(&paths, "flash-a", ple::PleSelection::All).unwrap();
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 4));
        let a_q4 = loader_path(&paths, SHARED_FILE_A, PleFormat::Q4);
        assert_eq!(fs::read(&a_q4).unwrap(), PLE_Q4);
        assert!(same_inode(&a_q4, &canonical_q4));
        // `--ple none` touches no PLE file.
        pull_shared(&paths, "flash-a", ple::PleSelection::Skip).unwrap();
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn pull_refreshes_a_stale_alias_after_a_corrupt_canonical_is_redownloaded() {
        use hipfire_config::memory::PleFormat;
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("ple-shared-stale");
        write_test_registry_cache(&paths, &shared_registry_raw());
        let canonical = paths.models.join(SHARED_Q8_FILE);
        let alias = loader_path(&paths, SHARED_FILE_A, PleFormat::Q8);
        let (base, handle) =
            serve_files(&[(SHARED_FILE_A, SHARED_BASE_A), (SHARED_Q8_FILE, PLE_Q8)]);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_shared(&paths, "flash-a", ple::PleSelection::Q8).unwrap();
        handle.join().unwrap();

        // The canonical file is replaced by a corrupt one (new inode): the alias
        // still names the old, good inode.
        fs::remove_file(&canonical).unwrap();
        fs::write(&canonical, b"corrupt-rows").unwrap();
        assert!(!same_inode(&alias, &canonical));
        let (base, handle) = serve_files(&[(SHARED_Q8_FILE, PLE_Q8)]);
        let _hf2 = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_shared(&paths, "flash-a", ple::PleSelection::Q8).unwrap();
        handle.join().unwrap();
        assert_eq!(fs::read(&canonical).unwrap(), PLE_Q8);
        assert_eq!(fs::read(&alias).unwrap(), PLE_Q8);
        assert!(
            same_inode(&alias, &canonical),
            "the alias follows the refreshed canonical file"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn pull_never_overwrites_an_unverified_alias_and_rejects_a_tampered_download() {
        use hipfire_config::memory::PleFormat;
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("ple-shared-unverified");
        write_test_registry_cache(&paths, &shared_registry_raw());
        let canonical = paths.models.join(SHARED_Q8_FILE);
        let alias = loader_path(&paths, SHARED_FILE_A, PleFormat::Q8);

        // A tampered shared download fails verification and installs nothing.
        let (base, handle) = serve_files(&[
            (SHARED_FILE_A, SHARED_BASE_A),
            (SHARED_Q8_FILE, &b"tampered-rows"[..]),
        ]);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        let error = pull_shared(&paths, "flash-a", ple::PleSelection::Q8).unwrap_err();
        handle.join().unwrap();
        assert!(format!("{error:#}").contains("mismatch"), "{error:#}");
        assert!(!canonical.exists() && !alias.exists());

        // A private, unverified file under the loader's name is left untouched
        // (e.g. an old `hipfire ple split` output).
        fs::write(&alias, b"private-rows").unwrap();
        let (base, handle) = serve_files(&[(SHARED_Q8_FILE, PLE_Q8)]);
        let _hf2 = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        let error = pull_shared(&paths, "flash-a", ple::PleSelection::Q8).unwrap_err();
        handle.join().unwrap();
        assert!(
            format!("{error:#}").contains("not the verified shared"),
            "{error:#}"
        );
        assert_eq!(fs::read(&alias).unwrap(), b"private-rows");
        assert_eq!(fs::read(&canonical).unwrap(), PLE_Q8);
        let leftovers: Vec<_> = fs::read_dir(&paths.models)
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().contains(".link."))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");

        // A dangling symlink under that name is replaced.
        fs::remove_file(&alias).unwrap();
        std::os::unix::fs::symlink(paths.models.join("gone"), &alias).unwrap();
        pull_shared(&paths, "flash-a", ple::PleSelection::Q8).unwrap();
        assert!(same_inode(&alias, &canonical));
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn on_demand_links_the_shared_file_for_a_second_base_without_downloading() {
        use hipfire_config::memory::{PleFormat, PleFormatRequest};
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let paths = test_paths("ple-shared-on-demand");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = RegistryV1::parse(&shared_registry_raw(), "ple shared").unwrap();
        let (tag_a, entry_a) = registry.model("flash-a").unwrap();
        let (tag_b, entry_b) = registry.model("flash-b").unwrap();
        let model_a = paths.models.join(SHARED_FILE_A);
        let model_b = paths.models.join(SHARED_FILE_B);
        fs::write(&model_a, SHARED_BASE_A).unwrap();
        fs::write(&model_b, SHARED_BASE_B).unwrap();
        let canonical = paths.models.join(SHARED_Q8_FILE);

        let (base, handle) = serve_files(&[(SHARED_Q8_FILE, PLE_Q8)]);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        ple::ensure_ple_sidecar(
            Some(entry_a),
            Some(tag_a),
            &model_a,
            PleFormatRequest::Q8,
            "",
            true,
        )
        .unwrap();
        handle.join().unwrap();
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 1));
        assert_eq!(fs::read(&canonical).unwrap(), PLE_Q8);
        let alias_a = loader_path(&paths, SHARED_FILE_A, PleFormat::Q8);
        assert!(same_inode(&alias_a, &canonical));

        // The second base only links: no server, downloads off.
        let _dead = EnvGuard::set("HIPFIRE_HF_BASE", "http://127.0.0.1:9");
        ple::ensure_ple_sidecar(
            Some(entry_b),
            Some(tag_b),
            &model_b,
            PleFormatRequest::Q8,
            "",
            false,
        )
        .unwrap();
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 1));
        let alias_b = loader_path(&paths, SHARED_FILE_B, PleFormat::Q8);
        assert_eq!(fs::read(&alias_b).unwrap(), PLE_Q8);
        assert!(same_inode(&alias_b, &canonical));

        // Present aliases are a no-op for every request, q4 included.
        for request in [
            PleFormatRequest::Q8,
            PleFormatRequest::Q4,
            PleFormatRequest::Auto,
        ] {
            ple::ensure_ple_sidecar(Some(entry_b), Some(tag_b), &model_b, request, "", false)
                .unwrap();
        }

        // A corrupt canonical never gets linked: downloads off -> refuse.
        fs::remove_file(&alias_b).unwrap();
        fs::remove_file(&canonical).unwrap();
        fs::write(&canonical, b"corrupt-rows").unwrap();
        let error = ple::ensure_ple_sidecar(
            Some(entry_b),
            Some(tag_b),
            &model_b,
            PleFormatRequest::Q8,
            "",
            false,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("fails verification"),
            "{error:#}"
        );
        assert!(!alias_b.exists());
        // Downloads on: the stale canonical is refreshed and linked.
        let (base, handle) = serve_files(&[(SHARED_Q8_FILE, PLE_Q8)]);
        let _hf2 = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        ple::ensure_ple_sidecar(
            Some(entry_b),
            Some(tag_b),
            &model_b,
            PleFormatRequest::Q8,
            "",
            true,
        )
        .unwrap();
        handle.join().unwrap();
        assert_eq!(fs::read(&canonical).unwrap(), PLE_Q8);
        assert!(same_inode(&alias_b, &canonical));
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_keeps_the_shared_pair_until_the_last_base_alias_goes() {
        use hipfire_config::memory::PleFormat;
        let paths = test_paths("ple-shared-rm");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = RegistryV1::parse(&shared_registry_raw(), "ple shared rm").unwrap();
        let canonical = paths.models.join(SHARED_Q8_FILE);
        fs::write(&canonical, PLE_Q8).unwrap();
        for base in [SHARED_FILE_A, SHARED_FILE_B] {
            fs::write(paths.models.join(base), base.as_bytes()).unwrap();
            fs::hard_link(&canonical, loader_path(&paths, base, PleFormat::Q8)).unwrap();
        }
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: "flash-a".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!paths.models.join(SHARED_FILE_A).exists());
        assert!(!loader_path(&paths, SHARED_FILE_A, PleFormat::Q8).exists());
        assert!(canonical.is_file(), "flash-b still links the shared file");
        assert_eq!(
            fs::read(loader_path(&paths, SHARED_FILE_B, PleFormat::Q8)).unwrap(),
            PLE_Q8
        );
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: "flash-b".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(
            !canonical.exists(),
            "the last base takes the shared file along"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    const GF_MONO_BYTES: &[u8] = b"grandfathered-monolithic";
    const GF_MONO_SHA: &str = "deec35dcefcc729df481caa629f305913b5aa65b9319d5aef17f418887901acd";
    const GF_OVERLAY_BYTES: &[u8] = b"overlay-bytes-mono";
    const GF_OVERLAY_SHA: &str = "9cbb2a1860c1a0c022c617bd757e0fff4c93045dce1e9648f81976636ca29f7b";
    const GF_HOTSET_BYTES: &[u8] = b"hotset-bytes-mono";
    const GF_HOTSET_SHA: &str = "cdda2be78a526bc24a498f9ac6a977faa1a0bf0e03dc07a69c130c1652f9232a";
    const GF_BASE_A_SHA: &str = "a752e8e68d16af0c8033e291afe95fb29c3fdee505374ef76fef735ab883746e";
    const GF_CANONICAL: &str = "qwen3.8-flash-next.mq4-xts";
    const GF_LEGACY: &str = "qwen3.8-flash-next-gptq3.mq4";
    const GF_TAGS: [&str; 5] = [
        "qwen3.8:flash-next",
        "qwen3.8:flash-next-mq4",
        "qwen3.8:flash-next-gptq3",
        "qwen3.8:flash-next-mq4-xts",
        "qwen3.8:flash",
    ];

    /// A small stand-in for the compiled grandfather pin of `split-model`.
    fn split_model_pin() -> hipfire_registry::GrandfatheredMonolithic {
        hipfire_registry::GrandfatheredMonolithic {
            repo: "test/repo",
            split_file: "split-model-base.mq4",
            split_sha256: "137eeb8a7ecee9b04833bddde9f43959dffd7c6d78383b6d77eded45f843f0c8",
            split_size_bytes: PLE_BASE.len() as u64,
            canonical_file: "split-model.mq4",
            legacy_file: "split-model-legacy.mq4",
            sha256: GF_MONO_SHA,
            size_bytes: GF_MONO_BYTES.len() as u64,
        }
    }

    /// A small stand-in for the compiled grandfather pin of the Flash-Next MQ4
    /// XTS split base (`SHARED_FILE_A`).
    fn gf_pin() -> hipfire_registry::GrandfatheredMonolithic {
        hipfire_registry::GrandfatheredMonolithic {
            repo: "test/repo",
            split_file: SHARED_FILE_A,
            split_sha256: GF_BASE_A_SHA,
            split_size_bytes: SHARED_BASE_A.len() as u64,
            canonical_file: GF_CANONICAL,
            legacy_file: GF_LEGACY,
            sha256: GF_MONO_SHA,
            size_bytes: GF_MONO_BYTES.len() as u64,
        }
    }

    /// The four stock MQ4 tags (+ alias) on the split base `SHARED_FILE_A` with
    /// the shared pair, and a second shared-pair base (`SHARED_FILE_B`).
    fn gf_registry_raw() -> String {
        let slots = format!(
            r#""ple":{{"q8":{{"file":"{SHARED_Q8_FILE}","sha256":"{}","size_bytes":{}}},
               "q4":{{"file":"{SHARED_Q4_FILE}","sha256":"{}","size_bytes":{}}}}}"#,
            ple_sha(PLE_Q8),
            PLE_Q8.len(),
            ple_sha(PLE_Q4),
            PLE_Q4.len(),
        );
        let stock = |file: &str, base: &[u8]| {
            format!(
                r#"{{"repo":"test/repo","file":"{file}","size_gb":0.001,"min_vram_gb":1,
                   "desc":"MQ4 XTS split base","sha256":"{}","size_bytes":{},{slots}}}"#,
                ple_sha(base),
                base.len()
            )
        };
        let a = stock(SHARED_FILE_A, SHARED_BASE_A);
        let b = stock(SHARED_FILE_B, SHARED_BASE_B);
        format!(
            r#"{{"schema_version":1,"generated_at":"2099-01-01T00:00:00Z","models":{{
                "qwen3.8:flash-next":{a},"qwen3.8:flash-next-mq4":{a},
                "qwen3.8:flash-next-gptq3":{a},"qwen3.8:flash-next-mq4-xts":{a},
                "qwen3.8:flash-next-mq3-xts":{b}
            }},"aliases":{{"qwen3.8:flash":"qwen3.8:flash-next"}}}}"#
        )
    }

    fn gf_overlay_pins() -> Vec<TrunkOverlayPin> {
        vec![
            TrunkOverlayPin {
                base_sha256: GF_MONO_SHA,
                base_size: GF_MONO_BYTES.len() as u64,
                repo: "test/repo",
                file: "qwen3.8-flash-next.mq4-xts.trunk-overlay",
                sha256: GF_OVERLAY_SHA,
                size_bytes: GF_OVERLAY_BYTES.len() as u64,
            },
            // The split base's own overlay must NOT be fetched for a served monolithic.
            TrunkOverlayPin {
                base_sha256: GF_BASE_A_SHA,
                base_size: SHARED_BASE_A.len() as u64,
                repo: "test/repo",
                file: "qwen3.8-flash-next-base.mq4-xts.trunk-overlay",
                sha256: GF_OVERLAY_SHA,
                size_bytes: GF_OVERLAY_BYTES.len() as u64,
            },
        ]
    }

    fn gf_hotset_pins() -> Vec<ExpertHotsetPin> {
        vec![
            ExpertHotsetPin {
                base_sha256: GF_MONO_SHA,
                base_size: GF_MONO_BYTES.len() as u64,
                repo: "test/repo",
                file: "qwen3.8-flash-next.mq4-xts.expert-hotset",
                sha256: GF_HOTSET_SHA,
                size_bytes: GF_HOTSET_BYTES.len() as u64,
            },
            ExpertHotsetPin {
                base_sha256: GF_BASE_A_SHA,
                base_size: SHARED_BASE_A.len() as u64,
                repo: "test/repo",
                file: "qwen3.8-flash-next-base.mq4-xts.expert-hotset",
                sha256: GF_HOTSET_SHA,
                size_bytes: GF_HOTSET_BYTES.len() as u64,
            },
        ]
    }

    #[test]
    fn split_tag_serves_a_verified_grandfathered_monolithic_instead_of_forcing_a_download() {
        let paths = test_paths("ple-tag-fallback");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = RegistryV1::parse(&split_registry_raw(), "ple fallback").unwrap();
        let _pins = GrandfatherOverride::set(&[split_model_pin()]);
        assert_eq!(
            resolve_model_path(&paths, &registry, "split-model").unwrap(),
            None
        );
        assert_eq!(find_model_path(&paths, &registry, "split-model"), None);

        let mono = paths.models.join("split-model.mq4");
        fs::write(&mono, GF_MONO_BYTES).unwrap();
        assert_eq!(
            resolve_model_path(&paths, &registry, "split-model").unwrap(),
            Some(mono.clone()),
            "the split tag falls back to the verified local monolithic"
        );
        assert_eq!(
            find_model_path(&paths, &registry, "split-model"),
            Some(mono.clone())
        );

        // Once the split base is installed it wins.
        let base = paths.models.join("split-model-base.mq4");
        fs::write(&base, PLE_BASE).unwrap();
        assert_eq!(
            resolve_model_path(&paths, &registry, "split-model").unwrap(),
            Some(base.clone())
        );
        assert_eq!(
            find_model_path(&paths, &registry, "split-model"),
            Some(base)
        );
        // The split tag never fuzzy-matches an unrelated local file.
        fs::remove_file(paths.models.join("split-model-base.mq4")).unwrap();
        fs::remove_file(&mono).unwrap();
        fs::write(paths.models.join("split-model.mq4r"), b"other tier").unwrap();
        assert_eq!(
            resolve_model_path(&paths, &registry, "split-model").unwrap(),
            None
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn every_stock_tag_resolves_the_grandfathered_monolithic_in_both_spellings() {
        let paths = test_paths("gf-resolve");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = RegistryV1::parse(&gf_registry_raw(), "gf").unwrap();
        let _pins = GrandfatherOverride::set(&[gf_pin()]);
        let canonical = paths.models.join(GF_CANONICAL);
        let legacy = paths.models.join(GF_LEGACY);
        for tag in GF_TAGS {
            assert_eq!(resolve_model_path(&paths, &registry, tag).unwrap(), None);
            assert_eq!(find_model_path(&paths, &registry, tag), None);
        }

        // Canonical spelling.
        fs::write(&canonical, GF_MONO_BYTES).unwrap();
        for tag in GF_TAGS {
            assert_eq!(
                resolve_model_path(&paths, &registry, tag).unwrap(),
                Some(canonical.clone()),
                "{tag}"
            );
            assert_eq!(
                find_model_path(&paths, &registry, tag),
                Some(canonical.clone()),
                "{tag}"
            );
        }
        let (tag, entry) =
            registry_entry_for_path(&paths, &registry, canonical.to_str().unwrap()).unwrap();
        assert_eq!(tag, "qwen3.8:flash-next");
        assert_eq!(entry.file, SHARED_FILE_A);
        // The second base does not own the monolithic.
        assert!(grandfather_pin(&registry.models["qwen3.8:flash-next-mq3-xts"]).is_none());
        assert_eq!(
            ple::classify_layout(entry, GF_CANONICAL),
            ple::ModelLayout::Monolithic
        );

        // Legacy spelling only: the first resolve verifies it and links the
        // canonical name beside it (never copying the payload).
        fs::remove_file(&canonical).unwrap();
        fs::write(&legacy, GF_MONO_BYTES).unwrap();
        assert_eq!(
            find_model_path(&paths, &registry, "qwen3.8:flash-next"),
            Some(legacy.clone())
        );
        assert_eq!(
            registry_entry_for_path(&paths, &registry, legacy.to_str().unwrap())
                .map(|(tag, _)| tag),
            Some("qwen3.8:flash-next")
        );
        for tag in GF_TAGS {
            assert_eq!(
                resolve_model_path(&paths, &registry, tag).unwrap(),
                Some(canonical.clone()),
                "{tag}"
            );
        }
        assert!(same_inode(&canonical, &legacy), "a link, not a copy");

        // Bytes that do not verify are never served: the resolver refuses.
        for name in [GF_CANONICAL, GF_LEGACY] {
            fs::remove_file(&canonical).ok();
            fs::remove_file(&legacy).ok();
            let path = paths.models.join(name);
            fs::write(&path, vec![b'x'; GF_MONO_BYTES.len()]).unwrap();
            let error = resolve_model_path(&paths, &registry, "qwen3.8:flash-next")
                .unwrap_err()
                .to_string();
            assert!(error.contains("mismatch"), "{name}: {error}");
            fs::remove_file(&path).unwrap();
        }
        // With the split base installed the monolithic is never touched.
        fs::write(&canonical, b"not the pinned bytes").unwrap();
        let base = paths.models.join(SHARED_FILE_A);
        fs::write(&base, SHARED_BASE_A).unwrap();
        assert_eq!(
            resolve_model_path(&paths, &registry, "qwen3.8:flash").unwrap(),
            Some(base)
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn pull_serves_a_verified_monolithic_without_downloading_the_base() {
        use hipfire_config::memory::PleFormat;
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let _pins = GrandfatherOverride::set(&[gf_pin()]);
        for legacy_spelling in [false, true] {
            let paths = test_paths("gf-pull");
            write_test_registry_cache(&paths, &gf_registry_raw());
            fs::create_dir_all(&paths.models).unwrap();
            let canonical = paths.models.join(GF_CANONICAL);
            let legacy = paths.models.join(GF_LEGACY);
            fs::write(
                if legacy_spelling { &legacy } else { &canonical },
                GF_MONO_BYTES,
            )
            .unwrap();
            let overlay_pins = gf_overlay_pins();
            let hotset_pins = gf_hotset_pins();
            let pull = |paths: &Paths, tag: &str| {
                pull_with_sidecar_pins(
                    paths,
                    PullArgs::defaults(tag),
                    &overlay_pins,
                    &hotset_pins,
                    &grandfather_table(),
                )
            };

            // gfx1151 prefers q4: the shared q4, the monolithic's overlay and hot
            // set are fetched; the base never is.
            let _arch = ple::GpuArchOverride::set(&["gfx1151"]);
            let (base_url, handle) = serve_files(&[
                (SHARED_Q4_FILE, PLE_Q4),
                ("qwen3.8-flash-next.mq4-xts.trunk-overlay", GF_OVERLAY_BYTES),
                ("qwen3.8-flash-next.mq4-xts.expert-hotset", GF_HOTSET_BYTES),
            ]);
            let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base_url);
            DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
            let served = pull(&paths, "qwen3.8:flash-next").unwrap().unwrap();
            handle.join().unwrap();
            DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 3, "q4 + overlay + hot set"));
            assert_eq!(
                served, canonical,
                "the served monolithic is the printed path"
            );
            assert!(
                !paths.models.join(SHARED_FILE_A).exists(),
                "no base download"
            );
            let shared_q4 = paths.models.join(SHARED_Q4_FILE);
            let linked_q4 = hipfire_config::memory::ple_sidecar_path(&served, PleFormat::Q4);
            assert_eq!(fs::read(&linked_q4).unwrap(), PLE_Q4);
            assert!(same_inode(&linked_q4, &shared_q4), "linked, not copied");
            assert!(
                !paths.models.join(SHARED_Q8_FILE).exists(),
                "q8 is embedded"
            );
            assert!(!hipfire_config::memory::ple_sidecar_path(&served, PleFormat::Q8).exists());
            assert_eq!(
                fs::read(hipfire_config::memory::trunk_overlay_sidecar_path(&served)).unwrap(),
                GF_OVERLAY_BYTES
            );
            assert_eq!(
                fs::read(hipfire_config::memory::expert_hotset_sidecar_path(&served)).unwrap(),
                GF_HOTSET_BYTES
            );
            assert!(
                !paths
                    .models
                    .join("qwen3.8-flash-next-base.mq4-xts.trunk-overlay")
                    .exists(),
                "the split base's overlay is not the served artifact's"
            );

            // Every other stock tag and the alias: nothing left to download.
            let _dead = EnvGuard::set("HIPFIRE_HF_BASE", "http://127.0.0.1:9");
            for tag in GF_TAGS {
                let again = pull(&paths, tag).unwrap().unwrap();
                assert_eq!(again, canonical, "{tag}");
            }
            DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 3));
            if legacy_spelling {
                assert!(
                    same_inode(&canonical, &legacy),
                    "the legacy file is linked, not copied"
                );
            }
            fs::remove_dir_all(&paths.root).unwrap();
        }
    }

    #[test]
    fn pull_of_a_served_monolithic_skips_q8_and_fetches_nothing_on_rdna4() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let _pins = GrandfatherOverride::set(&[gf_pin()]);
        let paths = test_paths("gf-pull-q8");
        write_test_registry_cache(&paths, &gf_registry_raw());
        fs::create_dir_all(&paths.models).unwrap();
        let canonical = paths.models.join(GF_CANONICAL);
        fs::write(&canonical, GF_MONO_BYTES).unwrap();
        // No server is reachable: any download would fail the pull's PLE step.
        let _dead = EnvGuard::set("HIPFIRE_HF_BASE", "http://127.0.0.1:9");
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        for (arch, selection) in [
            ("gfx1201", None),
            ("", None),
            ("gfx1151", Some(ple::PleSelection::Q8)),
            ("gfx1151", Some(ple::PleSelection::Skip)),
        ] {
            let _arch = ple::GpuArchOverride::set(&[arch]);
            let served = pull_with_sidecar_pins(
                &paths,
                PullArgs {
                    ple: selection,
                    ..PullArgs::defaults("qwen3.8:flash")
                },
                &[],
                &[],
                &grandfather_table(),
            )
            .unwrap();
            assert_eq!(served, Some(canonical.clone()), "{arch} {selection:?}");
        }
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 0));
        assert!(!paths.models.join(SHARED_Q8_FILE).exists());
        assert!(!paths.models.join(SHARED_Q4_FILE).exists());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn pull_pulls_the_split_base_when_the_monolithic_does_not_verify_or_under_force() {
        use hipfire_config::memory::PleFormat;
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let _pins = GrandfatherOverride::set(&[gf_pin()]);
        let _arch = ple::GpuArchOverride::set(&["gfx1151"]);
        // (monolithic bytes, force): wrong bytes of the right size, then valid bytes + --force.
        let cases: [(Vec<u8>, bool); 2] = [
            (vec![b'x'; GF_MONO_BYTES.len()], false),
            (GF_MONO_BYTES.to_vec(), true),
        ];
        for (index, (bytes, force)) in cases.into_iter().enumerate() {
            let paths = test_paths(&format!("gf-pull-split-{index}"));
            write_test_registry_cache(&paths, &gf_registry_raw());
            fs::create_dir_all(&paths.models).unwrap();
            let canonical = paths.models.join(GF_CANONICAL);
            fs::write(&canonical, &bytes).unwrap();
            let (base_url, handle) =
                serve_files(&[(SHARED_FILE_A, SHARED_BASE_A), (SHARED_Q4_FILE, PLE_Q4)]);
            let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base_url);
            DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
            let served = pull_with_sidecar_pins(
                &paths,
                PullArgs {
                    force,
                    ..PullArgs::defaults("qwen3.8:flash-next")
                },
                &[],
                &[],
                &grandfather_table(),
            )
            .unwrap()
            .unwrap();
            handle.join().unwrap();
            DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 2, "base + shared q4"));
            let base = paths.models.join(SHARED_FILE_A);
            assert_eq!(served, base, "case {index}");
            assert_eq!(fs::read(&base).unwrap(), SHARED_BASE_A);
            let linked = hipfire_config::memory::ple_sidecar_path(&base, PleFormat::Q4);
            assert_eq!(fs::read(&linked).unwrap(), PLE_Q4);
            assert!(same_inode(&linked, &paths.models.join(SHARED_Q4_FILE)));
            assert_eq!(
                fs::read(&canonical).unwrap(),
                bytes,
                "the monolithic is untouched"
            );
            // The base now wins for every tag.
            let registry = RegistryV1::parse(&gf_registry_raw(), "gf").unwrap();
            for tag in GF_TAGS {
                assert_eq!(
                    resolve_model_path(&paths, &registry, tag).unwrap(),
                    Some(base.clone())
                );
            }
            fs::remove_dir_all(&paths.root).unwrap();
        }
    }

    /// A pull records the verification proof of every artifact it verifies (the
    /// split base, its PLE sidecar, the trunk overlay, the expert hot set), with
    /// the identity `verified_metadata` records for a retired monolithic. Nothing
    /// is hashed again afterwards: not by the validity check, not by a second
    /// pull, not by the loader's read-only proof lookup.
    #[test]
    fn pull_records_a_proof_for_the_split_base_and_every_sidecar() {
        use hipfire_config::memory::{expert_hotset_sidecar_path, trunk_overlay_sidecar_path};
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let _pins = GrandfatherOverride::set(&[]);
        let _arch = ple::GpuArchOverride::set(&["gfx1151"]);
        let paths = test_paths("pull-proof");
        write_test_registry_cache(&paths, &gf_registry_raw());
        let overlay_pins = gf_overlay_pins();
        let hotset_pins = gf_hotset_pins();
        let pull = |paths: &Paths| {
            pull_with_sidecar_pins(
                paths,
                PullArgs::defaults("qwen3.8:flash-next"),
                &overlay_pins,
                &hotset_pins,
                &grandfather_table(),
            )
        };
        let (base_url, handle) = serve_files(&[
            (SHARED_FILE_A, SHARED_BASE_A),
            (SHARED_Q4_FILE, PLE_Q4),
            (
                "qwen3.8-flash-next-base.mq4-xts.trunk-overlay",
                GF_OVERLAY_BYTES,
            ),
            (
                "qwen3.8-flash-next-base.mq4-xts.expert-hotset",
                GF_HOTSET_BYTES,
            ),
        ]);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base_url);
        HASH_INVOCATIONS.with(|count| count.set(0));
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        let base = pull(&paths).unwrap().unwrap();
        handle.join().unwrap();
        DOWNLOAD_INVOCATIONS
            .with(|count| assert_eq!(count.get(), 4, "base + q4 + overlay + hotset"));
        assert_eq!(base, paths.models.join(SHARED_FILE_A));

        let artifacts = [
            (base.clone(), SHARED_BASE_A),
            (paths.models.join(SHARED_Q4_FILE), PLE_Q4),
            (trunk_overlay_sidecar_path(&base), GF_OVERLAY_BYTES),
            (expert_hotset_sidecar_path(&base), GF_HOTSET_BYTES),
        ];
        let cache = VerifiedCache::in_home(&paths.root);
        for (path, bytes) in artifacts.clone() {
            assert_eq!(fs::read(&path).unwrap(), bytes, "{}", path.display());
            let record = cached_record(&paths, &path)
                .unwrap_or_else(|| panic!("no proof recorded for {}", path.display()));
            let stat = verified_cache::StatKey::of(&fs::metadata(&path).unwrap());
            let expected = format!("{:x}", Sha256::digest(bytes));
            assert_eq!(record["sha256"], expected.as_str());
            assert_eq!(record["size"], bytes.len() as u64);
            assert!(record["verified_at"].as_u64().unwrap() > 0);
            // The proof the loader and the identity gates read.
            assert_eq!(cache.verified_digest(&path, &stat), Some(expected.clone()));
            assert!(cache.satisfies(&path, &stat, &expected));
        }
        assert_eq!(hash_invocations(), 0, "the download hashed in stream");

        // A validity check hashes nothing: the proof stands.
        for (path, bytes) in artifacts {
            for _ in 0..2 {
                assert!(existing_artifact_valid(
                    &cache,
                    &path,
                    Some(&format!("{:x}", Sha256::digest(bytes))),
                    Some(bytes.len() as u64),
                ));
            }
        }
        assert_eq!(hash_invocations(), 0);

        // A second pull finds everything present and re-hashes nothing.
        let _dead = EnvGuard::set("HIPFIRE_HF_BASE", "http://127.0.0.1:9");
        assert_eq!(pull(&paths).unwrap().unwrap(), base);
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 4));
        assert_eq!(hash_invocations(), 0);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// An artifact already on disk (an older pull, a copied-in file) is hashed by
    /// the first pull that finds it, the proof is recorded, and no later check
    /// hashes it again until its stat changes.
    #[test]
    fn existing_artifact_valid_records_its_proof_and_hashes_once_per_stat() {
        let paths = test_paths("valid-records");
        fs::create_dir_all(&paths.models).unwrap();
        let cache = VerifiedCache::in_home(&paths.root);
        let path = paths.models.join("split-base.mq4-xts");
        let bytes = b"synthetic split base";
        fs::write(&path, bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(bytes));
        let size = bytes.len() as u64;
        HASH_INVOCATIONS.with(|count| count.set(0));

        assert!(existing_artifact_valid(
            &cache,
            &path,
            Some(&sha),
            Some(size)
        ));
        assert_eq!(hash_invocations(), 1);
        assert!(cached_record(&paths, &path).is_some(), "proof recorded");
        assert!(existing_artifact_valid(
            &cache,
            &path,
            Some(&sha),
            Some(size)
        ));
        assert_eq!(hash_invocations(), 1, "second check hits the proof");

        // Same size, new bytes: the stat moves, the old proof no longer applies,
        // and wrong bytes never earn one.
        fs::write(&path, b"synthetic split BASE").unwrap();
        assert!(!existing_artifact_valid(
            &cache,
            &path,
            Some(&sha),
            Some(size)
        ));
        assert_eq!(hash_invocations(), 2);
        assert!(!existing_artifact_valid(
            &cache,
            &path,
            Some(&sha),
            Some(size)
        ));
        assert_eq!(hash_invocations(), 3, "a failed proof is never remembered");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// A local HTTP server that serves `files` for any number of requests and
    /// records which file names were requested (404 for the rest).
    struct RecordingServer {
        base: String,
        seen: std::sync::mpsc::Receiver<String>,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl RecordingServer {
        fn spawn(files: std::collections::HashMap<String, Vec<u8>>) -> Self {
            use std::sync::atomic::Ordering;
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
            let (seen_thread, seen) = std::sync::mpsc::channel::<String>();
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stop_thread = stop.clone();
            let handle = std::thread::spawn(move || {
                while !stop_thread.load(Ordering::SeqCst) {
                    let Ok((mut stream, _)) = listener.accept() else {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    };
                    stream.set_nonblocking(false).unwrap();
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                    let mut buf = [0u8; 4096];
                    let n = stream.read(&mut buf).unwrap_or(0);
                    let request = String::from_utf8_lossy(&buf[..n]).into_owned();
                    let target = request
                        .split_whitespace()
                        .nth(1)
                        .and_then(|path| path.rsplit('/').next())
                        .unwrap_or_default()
                        .to_owned();
                    let _ = seen_thread.send(target.clone());
                    match files.get(&target) {
                        Some(data) => {
                            let header = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                data.len()
                            );
                            let _ = stream.write_all(header.as_bytes());
                            let _ = stream.write_all(data);
                        }
                        None => {
                            let _ = stream.write_all(
                                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            );
                        }
                    }
                    let _ = stream.flush();
                }
            });
            Self {
                base,
                seen,
                stop,
                handle: Some(handle),
            }
        }

        /// Stop serving; the sorted names of every requested file.
        fn finish(mut self) -> Vec<String> {
            self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
            self.handle.take().unwrap().join().unwrap();
            let mut seen: Vec<String> = self.seen.try_iter().collect();
            seen.sort();
            seen
        }
    }

    /// Where the verified Flash-Next MQ4 XTS monolithic sits locally.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum LocalMono {
        /// `qwen3.8-flash-next.mq4-xts`.
        Canonical,
        /// Only the old `qwen3.8-flash-next-gptq3.mq4` spelling.
        GptqAlias,
        Nothing,
    }

    /// The CPU resolution matrix of `hipfire pull qwen3.8:flash-next` (a split
    /// base grandfathering a monolithic) over local state x GPU arches x `--ple`:
    /// whether the base downloads, which PLE rows are fetched (and their
    /// sizes), which links are made, and what the monolithic-bound overlay and
    /// hot set do.
    #[test]
    fn pull_resolution_matrix_over_local_monolithic_gpu_arches_and_ple_selection() {
        use hipfire_config::memory::PleFormat;
        use ple::PleSelection;
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let _pins = GrandfatherOverride::set(&[gf_pin()]);
        let overlay_pins = gf_overlay_pins();
        let hotset_pins = gf_hotset_pins();
        // (label, detected arches, --ple, the formats the plan wants).
        type Row = (
            &'static str,
            &'static [&'static str],
            Option<PleSelection>,
            &'static [PleFormat],
        );
        let rows: [Row; 11] = [
            ("gfx1151 auto", &["gfx1151"], None, &[PleFormat::Q4]),
            ("gfx1201 auto", &["gfx1201"], None, &[PleFormat::Q8]),
            ("gfx1100 auto", &["gfx1100"], None, &[PleFormat::Q8]),
            (
                "gfx1100+gfx1151 auto",
                &["gfx1100", "gfx1151"],
                None,
                &[PleFormat::Q4, PleFormat::Q8],
            ),
            ("no gpu auto", &[], None, &[PleFormat::Q8]),
            ("unknown arch auto", &["gfx906"], None, &[PleFormat::Q8]),
            (
                "gfx1151 --ple auto",
                &["gfx1151"],
                Some(PleSelection::Auto),
                &[PleFormat::Q4],
            ),
            (
                "gfx1151 --ple q8",
                &["gfx1151"],
                Some(PleSelection::Q8),
                &[PleFormat::Q8],
            ),
            (
                "gfx1151 --ple q4",
                &["gfx1151"],
                Some(PleSelection::Q4),
                &[PleFormat::Q4],
            ),
            (
                "gfx1201 --ple q4",
                &["gfx1201"],
                Some(PleSelection::Q4),
                &[PleFormat::Q4],
            ),
            (
                "gfx1100+gfx1151 --ple q4",
                &["gfx1100", "gfx1151"],
                Some(PleSelection::Q4),
                &[PleFormat::Q4],
            ),
        ];
        let shared_file = |format: PleFormat| match format {
            PleFormat::Q4 => SHARED_Q4_FILE,
            PleFormat::Q8 => SHARED_Q8_FILE,
        };
        let rows_bytes = |format: PleFormat| match format {
            PleFormat::Q4 => PLE_Q4,
            PleFormat::Q8 => PLE_Q8,
        };
        let overlay_mono = "qwen3.8-flash-next.mq4-xts.trunk-overlay";
        let hotset_mono = "qwen3.8-flash-next.mq4-xts.expert-hotset";
        let overlay_base = "qwen3.8-flash-next-base.mq4-xts.trunk-overlay";
        let hotset_base = "qwen3.8-flash-next-base.mq4-xts.expert-hotset";
        let everything: std::collections::HashMap<String, Vec<u8>> = [
            (SHARED_FILE_A, SHARED_BASE_A),
            (SHARED_Q4_FILE, PLE_Q4),
            (SHARED_Q8_FILE, PLE_Q8),
            (overlay_mono, GF_OVERLAY_BYTES),
            (hotset_mono, GF_HOTSET_BYTES),
            (overlay_base, GF_OVERLAY_BYTES),
            (hotset_base, GF_HOTSET_BYTES),
        ]
        .into_iter()
        .map(|(name, data)| (name.to_owned(), data.to_vec()))
        .collect();

        for local in [
            LocalMono::Canonical,
            LocalMono::GptqAlias,
            LocalMono::Nothing,
        ] {
            for (label, arches, selection, wanted) in rows {
                let case = format!("{local:?} / {label}");
                let paths = test_paths("gf-matrix");
                write_test_registry_cache(&paths, &gf_registry_raw());
                fs::create_dir_all(&paths.models).unwrap();
                let canonical = paths.models.join(GF_CANONICAL);
                match local {
                    LocalMono::Canonical => fs::write(&canonical, GF_MONO_BYTES).unwrap(),
                    LocalMono::GptqAlias => {
                        fs::write(paths.models.join(GF_LEGACY), GF_MONO_BYTES).unwrap()
                    }
                    LocalMono::Nothing => {}
                }
                let mono_present = local != LocalMono::Nothing;
                let _arch = ple::GpuArchOverride::set(arches);
                let server = RecordingServer::spawn(everything.clone());
                let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &server.base);
                DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
                let served = pull_with_sidecar_pins(
                    &paths,
                    PullArgs {
                        ple: selection,
                        ..PullArgs::defaults("qwen3.8:flash-next")
                    },
                    &overlay_pins,
                    &hotset_pins,
                    &grandfather_table(),
                )
                .unwrap()
                .unwrap();
                let seen = server.finish();
                DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), seen.len(), "{case}"));

                // The exact q8 rows are embedded in a present monolithic.
                let fetched: Vec<PleFormat> = wanted
                    .iter()
                    .copied()
                    .filter(|format| !(mono_present && *format == PleFormat::Q8))
                    .collect();
                let mut expected: Vec<String> = fetched
                    .iter()
                    .map(|format| shared_file(*format).to_owned())
                    .collect();
                // The base downloads only with nothing local; the overlay and hot
                // set are the ones bound to whichever artifact serves the tags.
                // The overlay is fetched only where the loader applies it
                // (gfx1151, or no detected GPU).
                let overlay_fetched = arches.is_empty() || arches.contains(&"gfx1151");
                if mono_present {
                    expected.push(hotset_mono.to_owned());
                    if overlay_fetched {
                        expected.push(overlay_mono.to_owned());
                    }
                } else {
                    expected.extend([SHARED_FILE_A.to_owned(), hotset_base.to_owned()]);
                    if overlay_fetched {
                        expected.push(overlay_base.to_owned());
                    }
                }
                expected.sort();
                assert_eq!(seen, expected, "{case}");
                assert_eq!(
                    seen.iter().any(|name| name == SHARED_FILE_A),
                    !mono_present,
                    "{case}: base download"
                );
                let model = paths.models.join(if mono_present {
                    GF_CANONICAL
                } else {
                    SHARED_FILE_A
                });
                assert_eq!(served, model, "{case}");
                if mono_present {
                    assert!(!paths.models.join(SHARED_FILE_A).exists(), "{case}");
                    assert!(same_installed_file(&canonical, &model), "{case}");
                } else {
                    assert_eq!(fs::read(&model).unwrap(), SHARED_BASE_A, "{case}");
                }

                // The fetched rows: size, bytes and the link under the loader's name.
                for format in [PleFormat::Q4, PleFormat::Q8] {
                    let shared = paths.models.join(shared_file(format));
                    let link = hipfire_config::memory::ple_sidecar_path(&model, format);
                    if fetched.contains(&format) {
                        let data = rows_bytes(format);
                        assert_eq!(
                            fs::metadata(&shared).unwrap().len(),
                            data.len() as u64,
                            "{case}"
                        );
                        assert_eq!(fs::read(&shared).unwrap(), data, "{case}");
                        assert!(
                            same_inode(&link, &shared),
                            "{case}: {format:?} linked, not copied"
                        );
                    } else {
                        assert!(!shared.exists(), "{case}: {format:?} must not be fetched");
                        assert!(!link.exists(), "{case}: {format:?} must not be linked");
                    }
                }
                if local == LocalMono::GptqAlias {
                    assert!(
                        same_inode(&canonical, &paths.models.join(GF_LEGACY)),
                        "{case}: canonical name is a link of the legacy file"
                    );
                }
                fs::remove_dir_all(&paths.root).unwrap();
            }
        }
    }

    /// A monolithic present with `--ple q8` on Halo: the exact rows are embedded,
    /// so no PLE sidecar is fetched or linked at all.
    #[test]
    fn q8_on_halo_with_a_monolithic_fetches_no_ple_rows() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let _pins = GrandfatherOverride::set(&[gf_pin()]);
        let paths = test_paths("gf-halo-q8");
        write_test_registry_cache(&paths, &gf_registry_raw());
        fs::create_dir_all(&paths.models).unwrap();
        let canonical = paths.models.join(GF_CANONICAL);
        fs::write(&canonical, GF_MONO_BYTES).unwrap();
        let _arch = ple::GpuArchOverride::set(&["gfx1151"]);
        let server = RecordingServer::spawn(Default::default());
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &server.base);
        let served = pull_with_sidecar_pins(
            &paths,
            PullArgs {
                ple: Some(ple::PleSelection::Q8),
                ..PullArgs::defaults("qwen3.8:flash-next")
            },
            &[],
            &[],
            &grandfather_table(),
        )
        .unwrap();
        assert_eq!(server.finish(), Vec::<String>::new());
        assert_eq!(served, Some(canonical.clone()));
        let leftovers: Vec<String> = fs::read_dir(&paths.models)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(leftovers, [GF_CANONICAL]);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    fn hash_invocations() -> usize {
        HASH_INVOCATIONS.with(|count| count.get())
    }

    /// Verification-cache record of `path` as written to the hipfire home.
    fn cached_record(paths: &Paths, path: &Path) -> Option<serde_json::Value> {
        let raw = fs::read(paths.root.join("cache/verified-artifacts.json")).ok()?;
        let cache: serde_json::Value = serde_json::from_slice(&raw).ok()?;
        cache["records"]
            .get(path.to_string_lossy().as_ref())
            .cloned()
    }

    #[test]
    fn resolving_a_verified_monolithic_hashes_it_once_per_stat() {
        let paths = test_paths("verify-cache");
        fs::create_dir_all(&paths.models).unwrap();
        let cache = VerifiedCache::in_home(&paths.root);
        let canonical = paths.models.join("mono.mq4-xts");
        let legacy = paths.models.join("mono-legacy.mq4");
        let digest = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        let bytes = b"pinned monolithic payload";
        let pin = |bytes: &[u8]| (digest(bytes), bytes.len() as u64);
        let resolve = |digest: &str, size: u64| {
            migrate_verified_artifact(&cache, &legacy, &canonical, digest, size)
        };
        fs::write(&canonical, bytes).unwrap();
        let (sha, size) = pin(bytes);
        HASH_INVOCATIONS.with(|count| count.set(0));

        // First resolve hashes and records dev/ino/size/mtime/sha/verified_at.
        assert_eq!(resolve(&sha, size).unwrap(), Some(canonical.clone()));
        assert_eq!(hash_invocations(), 1);
        let record = cached_record(&paths, &canonical).expect("record written");
        let metadata = fs::metadata(&canonical).unwrap();
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(record["dev"], metadata.dev());
            assert_eq!(record["ino"], metadata.ino());
            assert_eq!(record["mtime_sec"], metadata.mtime());
            assert_eq!(record["mtime_nsec"], metadata.mtime_nsec());
        }
        assert_eq!(record["size"], size);
        assert_eq!(record["sha256"], sha);
        assert!(record["verified_at"].as_u64().unwrap() > 0);

        // The second resolve (and every later load) does not hash again.
        for _ in 0..3 {
            assert_eq!(resolve(&sha, size).unwrap(), Some(canonical.clone()));
        }
        assert_eq!(hash_invocations(), 1, "a cache hit must not hash");

        // mtime alone changes: re-hash, then hits again.
        let original_mtime = metadata.modified().unwrap();
        let set_mtime = |path: &Path, time: std::time::SystemTime| {
            fs::OpenOptions::new()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(time)
                .unwrap();
        };
        set_mtime(&canonical, original_mtime + Duration::from_secs(5));
        assert_eq!(resolve(&sha, size).unwrap(), Some(canonical.clone()));
        assert_eq!(hash_invocations(), 2, "a changed mtime re-hashes");
        resolve(&sha, size).unwrap();
        assert_eq!(hash_invocations(), 2);
        let mtime = fs::metadata(&canonical).unwrap().modified().unwrap();

        // Size alone changes (same inode, same mtime restored): re-hash.
        let longer = b"pinned monolithic payload, longer";
        fs::write(&canonical, longer).unwrap();
        set_mtime(&canonical, mtime);
        let (longer_sha, longer_size) = pin(longer);
        assert_eq!(
            resolve(&longer_sha, longer_size).unwrap(),
            Some(canonical.clone())
        );
        assert_eq!(hash_invocations(), 3, "a changed size re-hashes");
        resolve(&longer_sha, longer_size).unwrap();
        assert_eq!(hash_invocations(), 3);
        assert_eq!(
            cached_record(&paths, &canonical).unwrap()["size"],
            longer_size
        );

        // Inode alone changes (same bytes, size and mtime): re-hash.
        let replacement = paths.models.join("replacement");
        fs::write(&replacement, longer).unwrap();
        set_mtime(
            &replacement,
            fs::metadata(&canonical).unwrap().modified().unwrap(),
        );
        let old_inode = {
            use std::os::unix::fs::MetadataExt;
            fs::metadata(&canonical).unwrap().ino()
        };
        fs::rename(&replacement, &canonical).unwrap();
        let new_inode = {
            use std::os::unix::fs::MetadataExt;
            fs::metadata(&canonical).unwrap().ino()
        };
        assert_ne!(old_inode, new_inode);
        resolve(&longer_sha, longer_size).unwrap();
        assert_eq!(hash_invocations(), 4, "a changed inode re-hashes");
        resolve(&longer_sha, longer_size).unwrap();
        assert_eq!(hash_invocations(), 4);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn a_cache_record_for_another_sha_never_satisfies_and_a_corrupt_cache_is_tolerated() {
        let paths = test_paths("verify-cache-sha");
        fs::create_dir_all(&paths.models).unwrap();
        let cache = VerifiedCache::in_home(&paths.root);
        let canonical = paths.models.join("mono.mq4-xts");
        let legacy = paths.models.join("mono-legacy.mq4");
        let bytes = b"pinned monolithic payload";
        let sha = format!("{:x}", Sha256::digest(bytes));
        let size = bytes.len() as u64;
        let other_sha = "b".repeat(64);
        fs::write(&canonical, bytes).unwrap();
        let resolve =
            |digest: &str| migrate_verified_artifact(&cache, &legacy, &canonical, digest, size);
        HASH_INVOCATIONS.with(|count| count.set(0));

        // The cache says these exact bytes verified against OTHER_SHA: that is
        // not the pin, so the file hashes (and now verifies for real).
        let stat = verified_cache::StatKey::of(&fs::metadata(&canonical).unwrap());
        cache.remember(&canonical, stat, &other_sha);
        assert_eq!(resolve(&sha).unwrap(), Some(canonical.clone()));
        assert_eq!(hash_invocations(), 1, "a record of another sha is no proof");
        assert_eq!(cached_record(&paths, &canonical).unwrap()["sha256"], sha);
        resolve(&sha).unwrap();
        assert_eq!(hash_invocations(), 1);

        // Asking for a different pin against the same bytes hashes, fails, and
        // leaves the good record alone.
        let error = resolve(&other_sha).unwrap_err().to_string();
        assert!(error.contains("SHA-256 mismatch"), "{error}");
        assert_eq!(hash_invocations(), 2);
        assert_eq!(cached_record(&paths, &canonical).unwrap()["sha256"], sha);
        resolve(&sha).unwrap();
        assert_eq!(hash_invocations(), 2);

        // A corrupt cache file is ignored, never fatal; the next verification
        // replaces it with a valid one.
        let cache_file = paths.root.join("cache/verified-artifacts.json");
        for garbage in [&b"{not json"[..], b"", b"\x00\xff\xfe", b"[1,2,3]"] {
            fs::write(&cache_file, garbage).unwrap();
            let before = hash_invocations();
            assert_eq!(resolve(&sha).unwrap(), Some(canonical.clone()));
            assert_eq!(hash_invocations(), before + 1);
            assert_eq!(cached_record(&paths, &canonical).unwrap()["sha256"], sha);
            resolve(&sha).unwrap();
            assert_eq!(hash_invocations(), before + 1);
        }

        // An unwritable cache location never fails the resolve either.
        let blocker = paths.root.join("blocked-home");
        fs::write(&blocker, b"a file, not a directory").unwrap();
        let broken = VerifiedCache::in_home(&blocker);
        let before = hash_invocations();
        assert_eq!(
            migrate_verified_artifact(&broken, &legacy, &canonical, &sha, size).unwrap(),
            Some(canonical.clone())
        );
        assert_eq!(hash_invocations(), before + 1);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn a_pull_of_a_served_monolithic_does_not_rehash_it_on_the_next_resolve() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let _pins = GrandfatherOverride::set(&[gf_pin()]);
        let paths = test_paths("gf-verify-cache");
        write_test_registry_cache(&paths, &gf_registry_raw());
        fs::create_dir_all(&paths.models).unwrap();
        let legacy = paths.models.join(GF_LEGACY);
        fs::write(&legacy, GF_MONO_BYTES).unwrap();
        let registry = RegistryV1::parse(&gf_registry_raw(), "gf").unwrap();
        HASH_INVOCATIONS.with(|count| count.set(0));
        // Legacy spelling only: one hash verifies it and links the canonical name.
        let first = resolve_model_path(&paths, &registry, "qwen3.8:flash-next")
            .unwrap()
            .unwrap();
        assert_eq!(first, paths.models.join(GF_CANONICAL));
        assert_eq!(hash_invocations(), 1);
        // Every later resolve (any tag, canonical or legacy name) is a cache hit.
        for tag in GF_TAGS {
            assert_eq!(
                resolve_model_path(&paths, &registry, tag).unwrap(),
                Some(first.clone()),
                "{tag}"
            );
        }
        assert_eq!(
            hash_invocations(),
            1,
            "the link covers the canonical name too"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_removes_a_grandfathered_monolithic_with_its_twin_and_sidecars_but_not_a_shared_pair_another_base_links(
    ) {
        use hipfire_config::memory::PleFormat;
        let _pins = GrandfatherOverride::set(&[gf_pin()]);
        let registry = RegistryV1::parse(&gf_registry_raw(), "gf rm").unwrap();
        for keep_second_base in [true, false] {
            let paths = test_paths("gf-rm");
            fs::create_dir_all(&paths.models).unwrap();
            let canonical = paths.models.join(GF_CANONICAL);
            let legacy = paths.models.join(GF_LEGACY);
            fs::write(&canonical, GF_MONO_BYTES).unwrap();
            fs::hard_link(&canonical, &legacy).unwrap();
            let shared_q4 = paths.models.join(SHARED_Q4_FILE);
            fs::write(&shared_q4, PLE_Q4).unwrap();
            let mono_q4 = hipfire_config::memory::ple_sidecar_path(&canonical, PleFormat::Q4);
            fs::hard_link(&shared_q4, &mono_q4).unwrap();
            let overlay = hipfire_config::memory::trunk_overlay_sidecar_path(&canonical);
            let hotset = hipfire_config::memory::expert_hotset_sidecar_path(&canonical);
            fs::write(&overlay, GF_OVERLAY_BYTES).unwrap();
            fs::write(&hotset, GF_HOTSET_BYTES).unwrap();
            let second = paths.models.join(SHARED_FILE_B);
            if keep_second_base {
                fs::write(&second, SHARED_BASE_B).unwrap();
                let second_q4 = hipfire_config::memory::ple_sidecar_path(&second, PleFormat::Q4);
                fs::hard_link(&shared_q4, &second_q4).unwrap();
            }
            rm_with_registry(
                &paths,
                &registry,
                RmArgs {
                    model: "qwen3.8:flash-next".into(),
                    yes: true,
                },
            )
            .unwrap();
            for gone in [&canonical, &legacy, &mono_q4, &overlay, &hotset] {
                assert!(!gone.exists(), "{} must be removed", gone.display());
            }
            assert_eq!(
                shared_q4.exists(),
                keep_second_base,
                "the shared q4 stays only while another base links it"
            );
            assert_eq!(second.exists(), keep_second_base);
            fs::remove_dir_all(&paths.root).unwrap();
        }
    }

    #[test]
    fn rm_removes_ple_sidecars_with_either_layout() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("ple-rm");
        write_test_registry_cache(&paths, &split_registry_raw());
        fs::create_dir_all(&paths.models).unwrap();
        let split_files = [
            "split-model-base.mq4",
            "split-model-base.mq4.ple-q4",
            "split-model-base.mq4.ple-q8",
            "split-model-base.mq4.ple-q4.plehotrows",
            "split-model-base.mq4.ple-q8.plehotrows",
            "split-model-base.mq4.plehot",
        ];
        for file in split_files {
            fs::write(paths.models.join(file), b"x").unwrap();
        }
        fs::write(paths.models.join("untouched.mq4"), b"x").unwrap();
        rm_command(
            &paths,
            RmArgs {
                model: "split-model".into(),
                yes: true,
            },
        )
        .unwrap();
        for file in split_files {
            assert!(!paths.models.join(file).exists(), "{file} must be removed");
        }
        assert!(paths.models.join("untouched.mq4").exists());

        // A monolithic served under the split tag takes its derived q4 with it.
        let _pins = GrandfatherOverride::set(&[split_model_pin()]);
        for file in ["split-model.mq4", "split-model.mq4.ple-q4"] {
            fs::write(paths.models.join(file), b"x").unwrap();
        }
        rm_command(
            &paths,
            RmArgs {
                model: "split-model".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!paths.models.join("split-model.mq4").exists());
        assert!(!paths.models.join("split-model.mq4.ple-q4").exists());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn ple_sidecars_are_never_listed_or_served_as_models() {
        let paths = test_paths("ple-listing");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = RegistryV1::parse(&split_registry_raw(), "ple listing").unwrap();
        let _pins = GrandfatherOverride::set(&[split_model_pin()]);
        let sidecars = [
            "split-model-base.mq4.ple-q4",
            "split-model-base.mq4.ple-q8",
            "split-model-base.mq4.plehot",
            "split-model-base.mq4.plehotrows",
            "split-model-base.mq4.ple-q8.plehotrows",
            "split-model.mq4.ple-q4",
            "unregistered-base.mq4.ple-q8",
        ];
        for name in sidecars {
            assert!(
                !is_listable_model_file(name, &registry, &grandfather_table()),
                "{name}"
            );
            assert!(
                !is_standalone_model(
                    &paths.models.join(name),
                    name,
                    &registry,
                    &grandfather_table()
                ),
                "{name}"
            );
        }
        for name in ["split-model-base.mq4", "split-model.mq4"] {
            assert!(
                is_listable_model_file(name, &registry, &grandfather_table()),
                "{name}"
            );
            assert!(
                is_standalone_model(
                    &paths.models.join(name),
                    name,
                    &registry,
                    &grandfather_table()
                ),
                "{name}"
            );
        }
        for name in sidecars
            .iter()
            .chain(&["split-model-base.mq4", "split-model.mq4"])
        {
            fs::write(paths.models.join(name), b"x").unwrap();
        }
        let mut listed: Vec<String> = local_model_paths(&paths, &registry)
            .unwrap()
            .iter()
            .filter_map(|path| path.file_name()?.to_str().map(str::to_owned))
            .collect();
        listed.sort();
        assert_eq!(listed, ["split-model-base.mq4", "split-model.mq4"]);
        assert!(registry_sidecar_files(&registry).contains("split-model-base.mq4.ple-q8"));
        assert!(registry_sidecar_files(&registry).contains("split-model-base.mq4.ple-q4"));
        fs::remove_dir_all(&paths.root).unwrap();
    }

    // ---- Flash-Next trunk overlay sidecar: pull, rm, listings -----------------

    const OVERLAY_BASE: &[u8] = b"overlay-base-bytes";
    const OVERLAY_BYTES: &[u8] = b"pre-encoded-qt44-overlay-bytes";

    /// The registry entry carries no overlay field: the overlay comes only from
    /// the injected pin table, keyed by the base's sha256/size.
    fn overlay_registry_raw() -> String {
        format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{{
                "overlay-model":{{
                    "repo":"test/repo","file":"overlay-model.mq4","size_gb":0.001,
                    "min_vram_gb":1,"desc":"x","sha256":"{}","size_bytes":{}
                }}
            }},
            "aliases":{{}}
        }}"#,
            ple_sha(OVERLAY_BASE),
            OVERLAY_BASE.len(),
        )
    }

    fn overlay_pins() -> Vec<TrunkOverlayPin> {
        vec![TrunkOverlayPin {
            base_sha256: ple_sha(OVERLAY_BASE).leak(),
            base_size: OVERLAY_BASE.len() as u64,
            repo: "test/repo",
            file: "overlay-model.mq4.trunk-overlay",
            sha256: ple_sha(OVERLAY_BYTES).leak(),
            size_bytes: OVERLAY_BYTES.len() as u64,
        }]
    }

    /// `PullArgs::defaults("overlay-model")` on a host with the given GPU arches.
    fn pull_overlay_model_on(paths: &Paths, force: bool, arches: &[&str]) {
        let _arch = ple::GpuArchOverride::set(arches);
        pull_with_sidecar_pins(
            paths,
            PullArgs {
                force,
                ..PullArgs::defaults("overlay-model")
            },
            &overlay_pins(),
            &[],
            &grandfather_table(),
        )
        .unwrap();
    }

    fn pull_overlay_model(paths: &Paths, force: bool) {
        pull_overlay_model_on(paths, force, &["gfx1151"]);
    }

    #[test]
    fn pull_skips_the_trunk_overlay_where_no_gpu_applies_it() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        for (arches, fetched) in [
            (&["gfx1201"][..], false),
            (&["gfx1100"][..], false),
            (&["gfx1100", "gfx1201"][..], false),
            (&["gfx1151"][..], true),
            (&["gfx1100", "gfx1151"][..], true),
            (&[][..], true),
        ] {
            let paths = test_paths("overlay-arch");
            write_test_registry_cache(&paths, &overlay_registry_raw());
            let mut files = std::collections::HashMap::new();
            files.insert("overlay-model.mq4".to_owned(), OVERLAY_BASE.to_vec());
            if fetched {
                files.insert(
                    "overlay-model.mq4.trunk-overlay".to_owned(),
                    OVERLAY_BYTES.to_vec(),
                );
            }
            // The server holds only what the pull should fetch; the download
            // counter below proves a skipped overlay was never requested.
            let (base, handle) = exact_http_server(files);
            let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
            DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
            pull_overlay_model_on(&paths, false, arches);
            handle.join().unwrap();
            DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 1 + usize::from(fetched)));
            assert_eq!(
                paths
                    .models
                    .join("overlay-model.mq4.trunk-overlay")
                    .exists(),
                fetched,
                "{arches:?}"
            );
            fs::remove_dir_all(&paths.root).unwrap();
        }
    }

    #[test]
    fn pull_downloads_and_verifies_the_trunk_overlay_once() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("overlay-pull");
        write_test_registry_cache(&paths, &overlay_registry_raw());
        let overlay = paths.models.join("overlay-model.mq4.trunk-overlay");

        let mut files = std::collections::HashMap::new();
        files.insert("overlay-model.mq4".to_owned(), OVERLAY_BASE.to_vec());
        files.insert(
            "overlay-model.mq4.trunk-overlay".to_owned(),
            OVERLAY_BYTES.to_vec(),
        );
        let (base, handle) = exact_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        pull_overlay_model(&paths, false);
        handle.join().unwrap();
        assert_eq!(
            fs::read(paths.models.join("overlay-model.mq4")).unwrap(),
            OVERLAY_BASE
        );
        assert_eq!(fs::read(&overlay).unwrap(), OVERLAY_BYTES);
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 2));

        // A valid file on disk is not downloaded again (an empty server would
        // refuse any request; the invocation counter proves none was made).
        let (base, handle) = exact_http_server(std::collections::HashMap::new());
        let _hf2 = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_overlay_model(&paths, false);
        handle.join().unwrap();
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 2));
        assert_eq!(fs::read(&overlay).unwrap(), OVERLAY_BYTES);

        // A stale overlay is refreshed and re-verified.
        fs::write(&overlay, b"stale").unwrap();
        let mut files = std::collections::HashMap::new();
        files.insert(
            "overlay-model.mq4.trunk-overlay".to_owned(),
            OVERLAY_BYTES.to_vec(),
        );
        let (base, handle) = exact_http_server(files);
        let _hf3 = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_overlay_model(&paths, false);
        handle.join().unwrap();
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 3));
        assert_eq!(fs::read(&overlay).unwrap(), OVERLAY_BYTES);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn pull_never_installs_a_trunk_overlay_that_fails_verification() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("overlay-pull-corrupt");
        write_test_registry_cache(&paths, &overlay_registry_raw());
        let mut files = std::collections::HashMap::new();
        files.insert("overlay-model.mq4".to_owned(), OVERLAY_BASE.to_vec());
        // Same length as the pinned payload, different bytes: the digest rejects it.
        files.insert(
            "overlay-model.mq4.trunk-overlay".to_owned(),
            vec![b'x'; OVERLAY_BYTES.len()],
        );
        let (base, handle) = exact_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_overlay_model(&paths, false);
        handle.join().unwrap();
        assert!(paths.models.join("overlay-model.mq4").is_file());
        assert!(
            !paths
                .models
                .join("overlay-model.mq4.trunk-overlay")
                .exists(),
            "a failed verification must not leave the overlay installed"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_removes_the_trunk_overlay_with_its_base() {
        let paths = test_paths("overlay-rm");
        let registry = RegistryV1::parse(&overlay_registry_raw(), "overlay rm").unwrap();
        fs::create_dir_all(&paths.models).unwrap();
        for file in [
            "overlay-model.mq4",
            "overlay-model.mq4.trunk-overlay",
            "untouched.mq4",
        ] {
            fs::write(paths.models.join(file), b"x").unwrap();
        }
        rm_with_sidecar_pins(
            &paths,
            &registry,
            &overlay_pins(),
            &[],
            RmArgs {
                model: "overlay-model".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!paths.models.join("overlay-model.mq4").exists());
        assert!(!paths
            .models
            .join("overlay-model.mq4.trunk-overlay")
            .exists());
        assert!(paths.models.join("untouched.mq4").exists());

        // An unregistered base takes its `<file>.trunk-overlay` with it too.
        for file in ["local.mq4", "local.mq4.trunk-overlay"] {
            fs::write(paths.models.join(file), b"x").unwrap();
        }
        rm_with_sidecar_pins(
            &paths,
            &registry,
            &overlay_pins(),
            &[],
            RmArgs {
                model: "local.mq4".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!paths.models.join("local.mq4").exists());
        assert!(!paths.models.join("local.mq4.trunk-overlay").exists());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn trunk_overlays_are_never_listed_or_served_as_models() {
        let paths = test_paths("overlay-listing");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = RegistryV1::parse(&overlay_registry_raw(), "overlay listing").unwrap();
        let overlays = [
            "overlay-model.mq4.trunk-overlay",
            "qwen3.8-flash-next.mq4-xts.trunk-overlay",
            "unregistered.mq4.trunk-overlay",
        ];
        for name in overlays {
            assert!(
                !is_listable_model_file(name, &registry, &grandfather_table()),
                "{name}"
            );
            assert!(
                !is_standalone_model(
                    &paths.models.join(name),
                    name,
                    &registry,
                    &grandfather_table()
                ),
                "{name}"
            );
        }
        assert!(registry_sidecar_files(&registry).contains(TRUNK_OVERLAY_PINS[0].file));
        for name in overlays.iter().chain(&["overlay-model.mq4"]) {
            fs::write(paths.models.join(name), b"x").unwrap();
        }
        let listed: Vec<String> = local_model_paths(&paths, &registry)
            .unwrap()
            .iter()
            .filter_map(|path| path.file_name()?.to_str().map(str::to_owned))
            .collect();
        assert_eq!(listed, ["overlay-model.mq4"]);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    // ---- Flash-Next expert hot set: client pin, verified pull, rm, discovery --

    const HOTSET_BYTES: &[u8] = br#"{"kind":"hipfire.qwen4.expert_hotset","version":1}"#;
    const HOTSET_FILE: &str = "overlay-model.mq4.expert-hotset";

    fn hotset_pins() -> Vec<ExpertHotsetPin> {
        vec![ExpertHotsetPin {
            base_sha256: ple_sha(OVERLAY_BASE).leak(),
            base_size: OVERLAY_BASE.len() as u64,
            repo: "test/repo",
            file: HOTSET_FILE,
            sha256: ple_sha(HOTSET_BYTES).leak(),
            size_bytes: HOTSET_BYTES.len() as u64,
        }]
    }

    fn pull_hotset_model(paths: &Paths, force: bool) {
        pull_with_sidecar_pins(
            paths,
            PullArgs {
                force,
                // A hot set is independent of speculative-decode sidecars.
                no_draft: true,
                ..PullArgs::defaults("overlay-model")
            },
            &[],
            &hotset_pins(),
            &grandfather_table(),
        )
        .unwrap();
    }

    #[test]
    fn pull_downloads_and_verifies_the_expert_hotset_once() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("hotset-pull");
        write_test_registry_cache(&paths, &overlay_registry_raw());
        let hotset = hipfire_config::memory::expert_hotset_sidecar_path(
            &paths.models.join("overlay-model.mq4"),
        );
        let mut files = std::collections::HashMap::new();
        files.insert("overlay-model.mq4".to_owned(), OVERLAY_BASE.to_vec());
        files.insert(HOTSET_FILE.to_owned(), HOTSET_BYTES.to_vec());
        let (base, handle) = exact_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        pull_hotset_model(&paths, false);
        handle.join().unwrap();
        assert_eq!(fs::read(&hotset).unwrap(), HOTSET_BYTES);
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 2));

        let (base, handle) = exact_http_server(std::collections::HashMap::new());
        let _hf2 = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_hotset_model(&paths, false);
        handle.join().unwrap();
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 2));

        fs::write(&hotset, b"stale").unwrap();
        let mut files = std::collections::HashMap::new();
        files.insert(HOTSET_FILE.to_owned(), HOTSET_BYTES.to_vec());
        let (base, handle) = exact_http_server(files);
        let _hf3 = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_hotset_model(&paths, false);
        handle.join().unwrap();
        DOWNLOAD_INVOCATIONS.with(|count| assert_eq!(count.get(), 3));
        assert_eq!(fs::read(&hotset).unwrap(), HOTSET_BYTES);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn pull_never_installs_an_expert_hotset_that_fails_verification() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        for (suffix, corrupt) in [
            ("digest", vec![b'x'; HOTSET_BYTES.len()]),
            ("size", b"short".to_vec()),
        ] {
            let paths = test_paths(&format!("hotset-corrupt-{suffix}"));
            write_test_registry_cache(&paths, &overlay_registry_raw());
            let mut files = std::collections::HashMap::new();
            files.insert("overlay-model.mq4".to_owned(), OVERLAY_BASE.to_vec());
            files.insert(HOTSET_FILE.to_owned(), corrupt);
            let (base, handle) = exact_http_server(files);
            let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
            pull_hotset_model(&paths, false);
            handle.join().unwrap();
            assert!(paths.models.join("overlay-model.mq4").is_file());
            assert!(
                !paths.models.join(HOTSET_FILE).exists(),
                "{suffix} verification failure must not install a hot set"
            );
            fs::remove_dir_all(&paths.root).unwrap();
        }
    }

    #[test]
    fn pull_installs_expert_hotset_beside_the_local_base_filename() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("hotset-renamed-base");
        let raw = overlay_registry_raw().replace("overlay-model.mq4", "renamed.mq4");
        write_test_registry_cache(&paths, &raw);
        let mut files = std::collections::HashMap::new();
        files.insert("renamed.mq4".to_owned(), OVERLAY_BASE.to_vec());
        // The remote pin's filename does not change when the local base does.
        files.insert(HOTSET_FILE.to_owned(), HOTSET_BYTES.to_vec());
        let (base, handle) = exact_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        pull_hotset_model(&paths, false);
        handle.join().unwrap();
        assert_eq!(
            fs::read(paths.models.join("renamed.mq4.expert-hotset")).unwrap(),
            HOTSET_BYTES
        );
        assert!(!paths.models.join(HOTSET_FILE).exists());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_removes_the_expert_hotset_with_its_base() {
        let paths = test_paths("hotset-rm");
        let registry = RegistryV1::parse(&overlay_registry_raw(), "hotset rm").unwrap();
        fs::create_dir_all(&paths.models).unwrap();
        for file in ["overlay-model.mq4", HOTSET_FILE, "untouched.mq4"] {
            fs::write(paths.models.join(file), b"x").unwrap();
        }
        rm_with_sidecar_pins(
            &paths,
            &registry,
            &[],
            &hotset_pins(),
            RmArgs {
                model: "overlay-model".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!paths.models.join("overlay-model.mq4").exists());
        assert!(!paths.models.join(HOTSET_FILE).exists());
        assert!(paths.models.join("untouched.mq4").exists());

        for file in ["local.mq4", "local.mq4.expert-hotset"] {
            fs::write(paths.models.join(file), b"x").unwrap();
        }
        rm_with_sidecar_pins(
            &paths,
            &registry,
            &[],
            &hotset_pins(),
            RmArgs {
                model: "local.mq4".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!paths.models.join("local.mq4").exists());
        assert!(!paths.models.join("local.mq4.expert-hotset").exists());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn expert_hotsets_are_never_listed_or_served_as_models() {
        let paths = test_paths("hotset-listing");
        fs::create_dir_all(&paths.models).unwrap();
        let mut registry = RegistryV1::parse(&overlay_registry_raw(), "hotset listing").unwrap();
        let hotsets = [
            HOTSET_FILE,
            EXPERT_HOTSET_PINS[0].file,
            "unregistered.mq4.expert-hotset",
            "uppercase.mq4.EXPERT-HOTSET",
        ];
        for name in hotsets {
            assert!(
                !is_listable_model_file(name, &registry, &grandfather_table()),
                "{name}"
            );
            assert!(
                !is_standalone_model(
                    &paths.models.join(name),
                    name,
                    &registry,
                    &grandfather_table()
                ),
                "{name}"
            );
            fs::write(paths.models.join(name), b"x").unwrap();
        }
        // Even a registry entry cannot turn placement metadata into a model.
        registry.models.get_mut("overlay-model").unwrap().file = HOTSET_FILE.into();
        assert!(!is_listable_model_file(
            HOTSET_FILE,
            &registry,
            &grandfather_table()
        ));
        assert!(!is_standalone_model(
            &paths.models.join(HOTSET_FILE),
            HOTSET_FILE,
            &registry,
            &grandfather_table(),
        ));
        assert!(registry_sidecar_files(&registry).contains(EXPERT_HOTSET_PINS[0].file));
        fs::write(paths.models.join("real-model.mq4"), b"x").unwrap();
        let listed: Vec<String> = local_model_paths(&paths, &registry)
            .unwrap()
            .iter()
            .filter_map(|path| path.file_name()?.to_str().map(str::to_owned))
            .collect();
        assert_eq!(listed, ["real-model.mq4"]);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    // ── Retired Flash-Next tags (rtn-asym, mq6q8-pleq8) ──

    const RETIRED_RTN_BYTES: &[u8] = b"retired rtn-asym fixture bytes";
    const RETIRED_Q8_BYTES: &[u8] = b"retired mq6q8-pleq8 fixture bytes";

    static RETIRED_RTN_SHA: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| format!("{:x}", Sha256::digest(RETIRED_RTN_BYTES)));

    static RETIRED_Q8_SHA: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| format!("{:x}", Sha256::digest(RETIRED_Q8_BYTES)));

    fn retired_rtn_sha() -> &'static str {
        RETIRED_RTN_SHA.as_str()
    }

    fn retired_q8_sha() -> &'static str {
        RETIRED_Q8_SHA.as_str()
    }

    /// The two compiled retired identities (real tags and file names) pinned
    /// to tiny fixture payloads instead of the 125 GB originals.
    fn retired_fixture_pins() -> Vec<RetiredModel> {
        vec![
            RetiredModel {
                sha256: retired_rtn_sha(),
                size_bytes: RETIRED_RTN_BYTES.len() as u64,
                ..RETIRED_MODELS[0]
            },
            RetiredModel {
                sha256: retired_q8_sha(),
                size_bytes: RETIRED_Q8_BYTES.len() as u64,
                ..RETIRED_MODELS[1]
            },
        ]
    }

    fn hash_count() -> usize {
        HASH_INVOCATIONS.with(|count| count.get())
    }

    fn download_count() -> usize {
        DOWNLOAD_INVOCATIONS.with(|count| count.get())
    }

    #[test]
    fn retired_tags_refuse_pull_with_one_line_even_forced_before_any_work() {
        let paths = test_paths("retired-pull");
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        for pin in RETIRED_MODELS {
            for model in [pin.tag, pin.file] {
                for force in [false, true] {
                    let mut args = PullArgs::defaults(model);
                    args.force = force;
                    let error = pull_command(&paths, args).unwrap_err();
                    assert_eq!(error.to_string(), pin.refusal(), "{model} force={force}");
                    assert!(!error.to_string().contains('\n'));
                    assert!(error.to_string().contains(
                        "retired; the file still runs if you have it; replacement qwen3.8:flash-next"
                    ));
                }
            }
        }
        assert_eq!(download_count(), 0);
        assert!(!paths.root.exists(), "a refused pull creates nothing");
    }

    #[test]
    fn only_the_two_retired_tags_are_retired_and_every_other_flash_next_tag_is_unchanged() {
        let registry = hipfire_registry::bundled().unwrap();
        for pin in RETIRED_MODELS {
            assert!(registry.model(pin.tag).is_none(), "{}", pin.tag);
            assert!(registry.model(pin.file).is_none(), "{}", pin.file);
        }
        // Shipped split / MQ3 / MQ2 tags still resolve to their own files.
        let expected = [
            ("qwen3.8:flash-next", "qwen3.8-flash-next-base.mq4-xts"),
            ("qwen3.8:flash-next-mq4", "qwen3.8-flash-next-base.mq4-xts"),
            (
                "qwen3.8:flash-next-gptq3",
                "qwen3.8-flash-next-base.mq4-xts",
            ),
            (
                "qwen3.8:flash-next-mq4-xts",
                "qwen3.8-flash-next-base.mq4-xts",
            ),
            (
                "qwen3.8:flash-next-mq3-xts",
                "qwen3.8-flash-next-base.mq3-xts",
            ),
            (
                "qwen3.8:flash-next-mq2-xts",
                "qwen3.8-flash-next-base.mq2-xts",
            ),
            (
                "qwen3.8:flash-next-mq2-xts-4500",
                "qwen3.8-flash-next-base.mq2-xts-4500",
            ),
            ("qwen3.8:flash", "qwen3.8-flash-next-base.mq4-xts"),
        ];
        // The obsolete local-only ZAB tier is removed outright: no catalog row, no
        // retired pin (its trunk tier ships as the default overlay on the stock tags).
        let zab = "qwen3.8:flash-next-mq4-xts-zab";
        assert!(registry.model(zab).is_none());
        assert!(registry.model("qwen3.8-flash-next.mq4-xts-zab").is_none());
        assert!(hipfire_registry::retired_model_named(zab, RETIRED_MODELS).is_none());
        for (tag, file) in expected {
            let (resolved, entry) = registry.model(tag).unwrap_or_else(|| panic!("{tag}"));
            assert_eq!(entry.file, file, "{tag} ({resolved})");
            assert!(hipfire_registry::retired_model_named(tag, RETIRED_MODELS).is_none());
            assert!(
                matches!(
                    model_card_by_name(&registry, tag),
                    ModelCard::Registry { .. }
                ),
                "{tag}"
            );
        }
        for tag in [
            "qwen3.8:flash-next",
            "qwen3.8:flash-next-mq4",
            "qwen3.8:flash-next-gptq3",
            "qwen3.8:flash-next-mq4-xts",
        ] {
            let (_, entry) = registry.model(tag).unwrap();
            assert!(
                grandfather_pin(entry).is_some(),
                "{tag} stays grandfathered"
            );
            assert!(entry.trunk_overlay().is_some() && entry.expert_hotset().is_some());
        }
    }

    #[test]
    fn verified_retired_tag_file_and_path_resolve_without_download_and_reuse_the_cache() {
        let paths = test_paths("retired-resolve");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = hipfire_registry::bundled().unwrap();
        let _pins = RetiredOverride::set(&retired_fixture_pins());
        let pin = RETIRED_MODELS[0];
        let installed = paths.models.join(pin.file);
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        HASH_INVOCATIONS.with(|count| count.set(0));

        // Not installed: a clear retired/missing error, never a download.
        for input in [pin.tag, pin.file] {
            let error = resolve_model_path(&paths, &registry, input).unwrap_err();
            assert!(error.downcast_ref::<RetiredMissing>().is_some(), "{input}");
            assert!(error.to_string().contains(&pin.refusal()), "{error}");
            assert!(error.to_string().contains("never downloaded"), "{error}");
            let card = model_card(&paths, &registry, input);
            assert_eq!(card.tag(), Some(pin.tag));
            assert!(card.pullable().is_none(), "a retired model is never pulled");
        }
        assert_eq!(find_model_path(&paths, &registry, pin.tag), None);
        assert_eq!(find_local_model_path(&paths, &registry, pin.tag), None);

        fs::write(&installed, RETIRED_RTN_BYTES).unwrap();
        let link = paths.root.join("link.mq4");
        std::os::unix::fs::symlink(&installed, &link).unwrap();
        let card = model_card(&paths, &registry, pin.tag);
        assert_eq!(card.retired().map(|retired| retired.tag), Some(pin.tag));
        let entry = card.entry().unwrap();
        assert_eq!(entry.file, pin.file);
        assert_eq!(entry.arch_id, Some(16));
        assert_eq!(
            entry.recommended_settings.as_ref().and_then(|s| s.top_k),
            Some(20)
        );
        for (input, expected) in [
            (pin.tag.to_owned(), installed.clone()),
            (pin.file.to_owned(), installed.clone()),
            (installed.display().to_string(), installed.clone()),
            // The spelling the user selected survives (sidecar lookup).
            (link.display().to_string(), link.clone()),
        ] {
            assert_eq!(
                resolve_model_path(&paths, &registry, &input).unwrap(),
                Some(expected),
                "{input}"
            );
            assert_eq!(model_card(&paths, &registry, &input).tag(), Some(pin.tag));
        }
        assert_eq!(
            hash_count(),
            1,
            "one hash; every spelling reuses the record"
        );
        let stat = verified_cache::StatKey::of(&fs::metadata(&installed).unwrap());
        let digest = VerifiedCache::in_home(&paths.root)
            .verified_digest(&installed, &stat)
            .expect("the verification is cached");
        assert!(digest.eq_ignore_ascii_case(retired_rtn_sha()));
        assert_eq!(
            find_local_model_path(&paths, &registry, pin.tag),
            Some(installed.clone())
        );

        // A same-named file elsewhere is not the retired artifact: no identity,
        // no verification, loaded as a bare file (even with different bytes).
        let other = paths.root.join("elsewhere");
        fs::create_dir_all(&other).unwrap();
        let lookalike = other.join(pin.file);
        fs::write(&lookalike, b"not the pinned bytes at all").unwrap();
        let input = lookalike.display().to_string();
        assert!(matches!(
            model_card(&paths, &registry, &input),
            ModelCard::Unregistered
        ));
        assert_eq!(
            resolve_model_path(&paths, &registry, &input).unwrap(),
            Some(lookalike.clone())
        );
        assert_eq!(hash_count(), 1);

        // A catalog path registered under the retired tag is held to the same
        // pin: wrong bytes are an explicit error (never a pull, never a silent
        // load); the pinned bytes serve at the path the catalog selected.
        hipfire_config::set_model_override(
            &paths.config,
            Some(pin.tag),
            Some(lookalike.clone()),
            pin.tag,
            "memory.kv_cache",
            "q8",
        )
        .unwrap();
        let error = resolve_model_path(&paths, &registry, pin.tag).unwrap_err();
        assert!(error.downcast_ref::<RetiredMissing>().is_none());
        let text = error.to_string();
        assert!(text.contains("is not the pinned"), "{text}");
        assert!(text.contains(&lookalike.display().to_string()), "{text}");
        fs::write(&lookalike, RETIRED_RTN_BYTES).unwrap();
        assert_eq!(
            resolve_model_path(&paths, &registry, pin.tag).unwrap(),
            Some(lookalike)
        );
        assert_eq!(download_count(), 0);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn retired_size_and_sha_mismatches_error_for_tag_file_and_path_and_never_pull() {
        let paths = test_paths("retired-mismatch");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = hipfire_registry::bundled().unwrap();
        let _pins = RetiredOverride::set(&retired_fixture_pins());
        let pin = RETIRED_MODELS[1];
        let installed = paths.models.join(pin.file);
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));

        let wrong_size = b"short";
        let mut wrong_sha = RETIRED_Q8_BYTES.to_vec();
        wrong_sha[0] ^= 0xff;
        for (bytes, expected) in [
            (wrong_size.to_vec(), "size mismatch"),
            (wrong_sha, "SHA-256 mismatch"),
        ] {
            fs::write(&installed, &bytes).unwrap();
            for input in [
                pin.tag.to_owned(),
                pin.file.to_owned(),
                installed.display().to_string(),
            ] {
                let error = resolve_model_path(&paths, &registry, &input).unwrap_err();
                assert!(error.downcast_ref::<RetiredMissing>().is_none(), "{input}");
                let text = error.to_string();
                assert!(text.contains(expected), "{input}: {text}");
                assert!(text.contains("is not the pinned"), "{input}: {text}");
                assert!(text.contains(&pin.refusal()), "{input}: {text}");
                let card = model_card(&paths, &registry, &input);
                assert!(card.retired().is_some());
                assert!(card.pullable().is_none());
            }
            match fs::read(paths.root.join("cache/verified-artifacts.json")) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => panic!("unexpected cache IO: {error}"),
                Ok(raw) => {
                    let cache: serde_json::Value =
                        serde_json::from_slice(&raw).expect("cache must stay well-formed");
                    assert_eq!(
                        cache["version"],
                        serde_json::json!(1),
                        "cache schema version"
                    );
                    let records = cache["records"].as_object().expect("records object");
                    assert!(
                        !records.contains_key(installed.to_string_lossy().as_ref()),
                        "failed verification creates no record"
                    );
                }
            }
        }
        assert_eq!(
            download_count(),
            0,
            "a mismatch is never repaired by a pull"
        );

        // The right bytes then verify and serve.
        fs::write(&installed, RETIRED_Q8_BYTES).unwrap();
        assert_eq!(
            resolve_model_path(&paths, &registry, pin.tag).unwrap(),
            Some(installed)
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn local_listing_marks_the_installed_retired_files_and_the_catalog_never_offers_them() {
        let paths = test_paths("retired-list");
        let registry = hipfire_registry::bundled().unwrap();
        let _pins = RetiredOverride::set(&retired_fixture_pins());
        fs::create_dir_all(paths.models.join("other")).unwrap();
        let rtn = paths.models.join(RETIRED_MODELS[0].file);
        let q8 = paths.models.join(RETIRED_MODELS[1].file);
        let lookalike = paths.models.join("other").join(RETIRED_MODELS[0].file);
        fs::write(&rtn, RETIRED_RTN_BYTES).unwrap();
        fs::write(&q8, RETIRED_Q8_BYTES).unwrap();
        fs::write(&lookalike, b"a different artifact").unwrap();

        let local = list_local_models(&paths, &registry).unwrap();
        let by_path = |path: &Path| {
            local
                .iter()
                .find(|model| model.path == fs::canonicalize(path).unwrap())
                .unwrap_or_else(|| panic!("{} is listed", path.display()))
        };
        let rtn_row = by_path(&rtn);
        assert_eq!(rtn_row.registry_tag.as_deref(), Some(RETIRED_MODELS[0].tag));
        assert!(rtn_row.retired);
        let q8_row = by_path(&q8);
        assert_eq!(q8_row.registry_tag.as_deref(), Some(RETIRED_MODELS[1].tag));
        assert!(q8_row.retired);
        let lookalike_row = by_path(&lookalike);
        assert_eq!(lookalike_row.registry_tag, None);
        assert!(!lookalike_row.retired);

        // Text and JSON marking.
        let line = local_model_line(rtn_row);
        assert!(
            line.contains("(qwen3.8:flash-next-rtn-asym) [retired]"),
            "{line}"
        );
        assert!(!local_model_line(lookalike_row).contains("retired"));
        let json = serde_json::to_value(rtn_row).unwrap();
        assert_eq!(json["retired"], true);
        assert_eq!(json["registry_tag"], RETIRED_MODELS[0].tag);
        assert_eq!(
            serde_json::to_value(lookalike_row).unwrap()["retired"],
            false
        );

        // The available catalog (what `list --remote` and the TUI offer) has neither.
        for pin in RETIRED_MODELS {
            assert!(!registry.models.contains_key(pin.tag), "{}", pin.tag);
            assert!(!registry.models.values().any(|entry| entry.file == pin.file));
        }
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_removes_an_installed_retired_file_by_tag_and_reports_a_missing_one() {
        let paths = test_paths("retired-rm");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = hipfire_registry::bundled().unwrap();
        let _pins = RetiredOverride::set(&retired_fixture_pins());
        let pin = RETIRED_MODELS[0];
        let installed = paths.models.join(pin.file);
        let error = rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: pin.tag.to_owned(),
                yes: true,
            },
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("{} is not downloaded; nothing to remove", pin.tag)
        );
        // Removal never verifies: a file that is not the pinned bytes still goes.
        fs::write(&installed, b"whatever is there").unwrap();
        rm_with_registry(
            &paths,
            &registry,
            RmArgs {
                model: pin.tag.to_owned(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!installed.exists());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// `hipfire run` reaches the daemon's `load` with the verified retired
    /// file for the tag and for its installed path, refuses a missing or
    /// mismatching file before spawning anything, and never pulls.
    #[cfg(unix)]
    #[test]
    fn run_loads_a_verified_retired_tag_and_path_and_never_pulls() {
        // Sets process env (daemon binary, offline registry): hold the shared
        // env lock so the pull tests' registry loads never see it. Declared
        // first so the EnvGuards restore before the lock releases.
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let paths = test_paths("retired-run");
        fs::create_dir_all(&paths.models).unwrap();
        let _pins = RetiredOverride::set(&retired_fixture_pins());
        let pin = RETIRED_MODELS[0];
        let installed = paths.models.join(pin.file);
        let daemon = write_task11_fake_daemon(&paths.root);
        let _bin = EnvGuard::set("HIPFIRE_DAEMON_BIN", daemon.to_str().unwrap());
        let _offline = EnvGuard::set("HIPFIRE_NO_REGISTRY_FETCH", "1");
        DOWNLOAD_INVOCATIONS.with(|count| count.set(0));
        let run = |model: &str| -> Result<()> {
            let cli = Cli::try_parse_from([
                "hipfire",
                "run",
                "--json",
                "--kv-mode",
                "q8",
                model,
                "hello",
            ])
            .unwrap();
            let Some(Commands::Run(args)) = cli.command else {
                panic!("expected run command");
            };
            // The script was just written: bounded ETXTBSY retry like the
            // other fake-daemon spawns.
            for attempt in 0..8u64 {
                match run_command(&paths, args.clone()) {
                    Err(error) if format!("{error:#}").contains("os error 26") => {
                        thread::sleep(Duration::from_millis(5 * (1 + attempt)));
                    }
                    other => return other,
                }
            }
            bail!("fake daemon spawn kept failing with ETXTBSY")
        };
        let loads = || -> Vec<serde_json::Value> {
            let raw = fs::read_to_string(paths.root.join("requests.log")).unwrap_or_default();
            raw.lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .filter(|row| row["type"] == "load")
                .collect()
        };

        // Missing: refused with the retirement line, nothing spawned or pulled.
        let error = run(pin.tag).unwrap_err();
        assert!(error.to_string().contains(&pin.refusal()), "{error:#}");
        assert!(loads().is_empty());
        // Wrong bytes: refused before the daemon starts.
        fs::write(&installed, b"short").unwrap();
        let error = run(pin.tag).unwrap_err();
        assert!(error.to_string().contains("is not the pinned"), "{error:#}");
        assert!(loads().is_empty());

        fs::write(&installed, RETIRED_RTN_BYTES).unwrap();
        for input in [pin.tag.to_owned(), installed.display().to_string()] {
            let before = loads().len();
            let _ = run(&input);
            let after = loads();
            assert_eq!(after.len(), before + 1, "{input}");
            assert_eq!(
                after[before]["model"],
                installed.display().to_string(),
                "{input}"
            );
        }
        assert_eq!(download_count(), 0);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// The shared serve request path: a request naming the retired tag or its
    /// installed path loads only the verified file; a missing, short or
    /// altered file and a lookalike path never load, and nothing is pulled.
    #[cfg(unix)]
    #[test]
    fn serve_requests_load_a_verified_retired_tag_and_installed_path_only() {
        let _pins = RetiredOverride::set(&retired_fixture_pins());
        let h = Task11HttpHarness::spawn("retired-serve");
        let pin = RETIRED_MODELS[0];
        let installed = h.paths.models.join(pin.file);
        let mut body = h.base_body("t11-stop-text", false);
        let load_count = |h: &Task11HttpHarness| {
            Task11HttpHarness::ops_of_type(&h.read_requests_log(), "load").len()
        };

        // Not installed: 404 naming the retirement, not a pull instruction.
        body["model"] = serde_json::json!(pin.tag);
        let (status, _, text) = post_status(h.port(), &body);
        assert_eq!(status, 404, "{text}");
        assert!(text.contains("is retired; the file still runs"), "{text}");
        assert!(!text.contains("hipfire pull"), "{text}");
        assert_eq!(load_count(&h), 0);

        // Short and same-size-altered files fail verification.
        let mut altered = RETIRED_RTN_BYTES.to_vec();
        altered[0] ^= 0xff;
        for bytes in [b"short".to_vec(), altered] {
            fs::write(&installed, bytes).unwrap();
            let (status, _, text) = post_status(h.port(), &body);
            assert!(status >= 400 && status != 200, "{status} {text}");
            assert!(text.contains("is not the pinned"), "{text}");
            assert_eq!(load_count(&h), 0);
        }

        // The pinned bytes load, by tag and by installed path (one load: the
        // same file stays resident).
        fs::write(&installed, RETIRED_RTN_BYTES).unwrap();
        let (status, _, text) = post_status(h.port(), &body);
        assert_eq!(status, 200, "{text}");
        let loads = Task11HttpHarness::ops_of_type(&h.read_requests_log(), "load")
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0]["model"], installed.display().to_string());
        body["model"] = serde_json::json!(installed.display().to_string());
        let (status, _, text) = post_status(h.port(), &body);
        assert_eq!(status, 200, "{text}");
        assert_eq!(load_count(&h), 1);
        assert!(h.paths.root.join("cache/verified-artifacts.json").is_file());

        // A same-named, same-bytes copy outside the models directory is just
        // a request path: refused, never given the retired identity.
        let elsewhere = h.paths.root.join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        let lookalike = elsewhere.join(pin.file);
        fs::write(&lookalike, RETIRED_RTN_BYTES).unwrap();
        body["model"] = serde_json::json!(lookalike.display().to_string());
        let (status, _, text) = post_status(h.port(), &body);
        assert_eq!(status, 404, "{text}");
        assert!(text.contains("serve.allow_request_paths"), "{text}");
        assert_eq!(load_count(&h), 1);
    }
}
