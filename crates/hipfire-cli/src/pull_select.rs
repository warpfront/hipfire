// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Interactive `hipfire pull <model>`: pick a tier of the model's registry
//! family, the PLE rows of a split base, the registry's draft/MTP sidecars and
//! the pinned expert hot set of a matching base, then confirm a summary with
//! the total download and a disk check.
//!
//! The registry entry and [`ple::plan_pull`] stay the source of truth; this
//! module only chooses the inputs the pull already takes (the tag, the `--ple`
//! selection, whether the draft/MTP sidecars are fetched and whether the
//! expert hot set is). A flag skips the matching question. A non-TTY stdin or
//! `--yes` asks nothing and takes [`PullChoice::defaults`]: the requested tag,
//! `--ple auto`, every declared sidecar and the expert hot set, which is what
//! the pull did before it asked anything.

use crate::ple::{self, PleSelection};
use crate::Paths;
use anyhow::{bail, Result};
use hipfire_config::memory::{PleFormat, PleFormatRequest};
use hipfire_registry::{
    ExpertHotsetPin, GrandfatheredMonolithic, ModelEntry, RegistryV1, Sidecar, TrunkOverlayPin,
};
use std::fs;
use std::io::{BufRead, Write};
use std::path::Path;

const GIB: f64 = (1_u64 << 30) as f64;

/// Longest tier note, in characters, before it is cut with an ellipsis.
const NOTE_CHARS: usize = 60;

/// Whether a pull asks: only with a terminal on stdin and without `--yes`.
pub(crate) fn is_interactive(stdin_is_terminal: bool, yes: bool) -> bool {
    stdin_is_terminal && !yes
}

/// One optional registry sidecar a pull fetches next to the base.
pub(crate) struct DeclaredSidecar<'a> {
    pub label: &'static str,
    pub sidecar: &'a Sidecar,
    /// A draft/MTP (speculative-decode) sidecar: the pull asks about these
    /// and `--no-draft` skips them. Every other sidecar is always fetched.
    pub spec: bool,
}

/// The entry's declared optional sidecars, in fetch order.
pub(crate) fn declared_sidecars(entry: &ModelEntry) -> impl Iterator<Item = DeclaredSidecar<'_>> {
    [
        ("TriAttention", entry.triattn.as_ref(), false),
        ("MTP", entry.mtp.as_ref(), true),
        ("DSpark", entry.dspark.as_ref(), true),
        ("DFlash", entry.dflash.as_ref(), true),
        ("XDNA", entry.xdna.as_ref(), false),
        ("Vision", entry.vision.as_ref(), false),
        ("T5", entry.t5.as_ref(), false),
        ("CLIP", entry.clip.as_ref(), false),
        ("Qwen3", entry.qwen3.as_ref(), false),
        ("VAE", entry.vae.as_ref(), false),
    ]
    .into_iter()
    .filter_map(|(label, sidecar, spec)| {
        sidecar.map(|sidecar| DeclaredSidecar {
            label,
            sidecar,
            spec,
        })
    })
}

// ---------------------------------------------------------------------------
// Registry family
// ---------------------------------------------------------------------------

/// One pullable variant of a registry model family.
#[derive(Debug)]
pub(crate) struct Tier<'a> {
    pub tag: &'a str,
    /// Short name: the root's file extension, else the tag suffix after
    /// `<root>-` (`qwen3.8:flash-next-mq3-xts` -> `mq3-xts`).
    pub label: String,
    pub entry: &'a ModelEntry,
}

/// The tiers `hipfire pull <root>` offers, the root first (the default),
/// then the rest largest first.
///
/// A tier is a tag `<root>-<suffix>` with the root's `arch_id` (which keeps
/// drafts such as `<root>-draft` out) and a download repo. Tags naming the
/// same artifact as an earlier tier (same SHA-256, else same file) are
/// dropped: they are the same weights under another name (the four Flash-Next
/// MQ4 tags are one split base). Aliases resolve to their tag before this
/// runs, so `qwen3.8:flash` offers the tiers of `qwen3.8:flash-next`.
pub(crate) fn family_tiers<'a>(registry: &'a RegistryV1, root: &str) -> Vec<Tier<'a>> {
    let Some((root_tag, root_entry)) = registry.models.get_key_value(root) else {
        return Vec::new();
    };
    let prefix = format!("{root}-");
    let members: Vec<(&'a str, &'a ModelEntry)> = registry
        .models
        .iter()
        .filter(|(tag, _)| tag.starts_with(&prefix))
        .map(|(tag, entry)| (tag.as_str(), entry))
        .collect();
    let mut others: Vec<(&'a str, &'a ModelEntry)> = members
        .into_iter()
        .filter(|(_, entry)| !entry.repo.is_empty() && entry.arch_id == root_entry.arch_id)
        .collect();
    others.sort_by(|(a_tag, a), (b_tag, b)| {
        b.size_gb
            .total_cmp(&a.size_gb)
            .then_with(|| a_tag.cmp(b_tag))
    });
    let mut tiers = vec![Tier {
        tag: root_tag,
        label: file_label(&root_entry.file),
        entry: root_entry,
    }];
    for (tag, entry) in others {
        if tiers.iter().any(|tier| same_artifact(tier.entry, entry)) {
            continue;
        }
        tiers.push(Tier {
            tag,
            label: tag[prefix.len()..].to_owned(),
            entry,
        });
    }
    tiers
}

fn file_label(file: &str) -> String {
    Path::new(file)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or(file)
        .to_owned()
}

fn same_artifact(a: &ModelEntry, b: &ModelEntry) -> bool {
    match (a.sha256.as_deref(), b.sha256.as_deref()) {
        (Some(x), Some(y)) => x.eq_ignore_ascii_case(y),
        _ => a.file == b.file,
    }
}

/// The first sentence of a registry `desc`, cut to [`NOTE_CHARS`].
fn tier_note(desc: &str) -> String {
    let sentence = desc.split(". ").next().unwrap_or(desc).trim();
    if sentence.chars().count() <= NOTE_CHARS {
        return sentence.to_owned();
    }
    let mut cut: String = sentence.chars().take(NOTE_CHARS - 1).collect();
    cut.push('…');
    cut
}

fn entry_bytes(entry: &ModelEntry) -> u64 {
    entry
        .size_bytes
        .unwrap_or_else(|| (entry.size_gb * 1e9).round() as u64)
}

// ---------------------------------------------------------------------------
// This machine
// ---------------------------------------------------------------------------

/// What the questions need to know about this machine.
#[derive(Clone, Debug)]
pub(crate) struct HostFacts {
    /// Every detected GPU arch (sorted, deduplicated); empty when none is
    /// detected.
    pub gpu_arches: Vec<String>,
    /// Largest amdgpu `mem_info_vram_total` (the carve-out on an APU).
    pub vram_bytes: Option<u64>,
    /// Installed host memory (`MemTotal`).
    pub ram_bytes: Option<u64>,
    /// Free space on the models directory's filesystem.
    pub free_disk_bytes: Option<u64>,
    /// `memory.ple_format` from the global config and environment.
    pub ple_request: PleFormatRequest,
}

impl HostFacts {
    pub(crate) fn detect(paths: &Paths) -> Self {
        Self {
            gpu_arches: ple::local_gpu_arches(),
            vram_bytes: largest_vram_bytes(Path::new("/sys/class/drm")),
            ram_bytes: hipfire_config::host_memory::host_memory()
                .map(|memory| memory.total_physical_bytes),
            free_disk_bytes: free_disk_bytes(&paths.models),
            ple_request: ple::request_from_global(paths),
        }
    }
}

/// Largest `cardN/device/mem_info_vram_total` under a DRM class directory.
fn largest_vram_bytes(drm: &Path) -> Option<u64> {
    fs::read_dir(drm)
        .ok()?
        .flatten()
        .filter(|card| {
            card.file_name().to_str().is_some_and(|name| {
                name.strip_prefix("card")
                    .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            })
        })
        .filter_map(|card| {
            rdna_compute::uma_memory::read_sysfs_u64(
                &card.path().join("device/mem_info_vram_total"),
            )
        })
        .max()
}

#[cfg(unix)]
fn free_disk_bytes(dir: &Path) -> Option<u64> {
    hipfire_runtime::ple_convert::available_bytes(dir).ok()
}

#[cfg(not(unix))]
fn free_disk_bytes(_dir: &Path) -> Option<u64> {
    None
}

fn host_line(host: &HostFacts) -> String {
    let gib = |bytes: Option<u64>| {
        bytes.map_or_else(
            || "unknown".to_owned(),
            |bytes| format!("{:.1} GiB", bytes as f64 / GIB),
        )
    };
    let arch = if host.gpu_arches.is_empty() {
        "GPU unknown".to_owned()
    } else {
        host.gpu_arches.join("+")
    };
    format!(
        "this machine: {arch}, VRAM {}, RAM {}",
        gib(host.vram_bytes),
        gib(host.ram_bytes)
    )
}

/// Whether a tier's `min_vram_gb` fits this machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fit {
    /// Fits the detected VRAM.
    Vram,
    /// Exceeds VRAM but fits VRAM plus host RAM (unified memory or spill).
    VramAndRam,
    /// Exceeds VRAM plus host RAM.
    TooLarge,
    /// No VRAM detected, or it exceeds VRAM with RAM unknown.
    Unknown,
}

impl Fit {
    fn text(self) -> &'static str {
        match self {
            Fit::Vram => "fits VRAM",
            Fit::VramAndRam => "fits VRAM+RAM",
            Fit::TooLarge => "too large",
            Fit::Unknown => "fit unknown",
        }
    }
}

pub(crate) fn tier_fit(entry: &ModelEntry, host: &HostFacts) -> Fit {
    let need = entry.min_vram_gb * GIB;
    let Some(vram) = host.vram_bytes else {
        return Fit::Unknown;
    };
    if need <= vram as f64 {
        return Fit::Vram;
    }
    match host.ram_bytes {
        Some(ram) if need <= vram.saturating_add(ram) as f64 => Fit::VramAndRam,
        Some(_) => Fit::TooLarge,
        None => Fit::Unknown,
    }
}

// ---------------------------------------------------------------------------
// The choice
// ---------------------------------------------------------------------------

/// Pull flags that answer a question up front.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PullFlags {
    /// `--ple`.
    pub ple: Option<PleSelection>,
    /// `--draft` (`Some(true)`) / `--no-draft` (`Some(false)`).
    pub draft: Option<bool>,
    /// `--force`: every file of the plan counts as a download.
    pub force: bool,
}

/// The inputs the pull runs with.
#[derive(Debug, PartialEq)]
pub(crate) struct PullChoice<'a> {
    pub tag: &'a str,
    pub entry: &'a ModelEntry,
    pub ple: PleSelection,
    /// Fetch the draft/MTP sidecars.
    pub draft: bool,
    /// Fetch the expert hot set sidecar when the base has a pinned one.
    pub expert_hotset: bool,
}

impl<'a> PullChoice<'a> {
    /// What a pull runs with when nothing is asked: the requested tag,
    /// `--ple` or `auto`, and the draft/MTP sidecars unless `--no-draft`.
    pub(crate) fn defaults(tag: &'a str, entry: &'a ModelEntry, flags: &PullFlags) -> Self {
        Self {
            tag,
            entry,
            ple: flags.ple.unwrap_or(PleSelection::Auto),
            draft: flags.draft.unwrap_or(true),
            expert_hotset: true,
        }
    }
}

/// What the questions read besides the flags. The registry outlives the
/// choice; the host facts only live through the questions.
pub(crate) struct PullContext<'r, 'c> {
    pub registry: &'r RegistryV1,
    pub host: &'c HostFacts,
    pub overlay_pins: &'c [TrunkOverlayPin],
    pub hotset_pins: &'c [ExpertHotsetPin],
    /// Monoliths a verified local copy of which serves a split entry without
    /// the base download.
    pub grandfather_pins: &'c [GrandfatheredMonolithic],
    /// Whether `file` (of the given size, when known) is already in the
    /// models directory. The pull itself still verifies the digest.
    pub present: &'c dyn Fn(&str, Option<u64>) -> bool,
}

/// A file counts as present when it exists with the expected size; the
/// summary does not hash (the pull verifies before it skips a download).
pub(crate) fn file_present(path: &Path, size: Option<u64>) -> bool {
    fs::metadata(path)
        .is_ok_and(|metadata| metadata.is_file() && size.is_none_or(|size| metadata.len() == size))
}

/// Ask the questions flags leave open, then confirm the summary. `Ok(None)`
/// means the user declined. With nothing to ask (one tier, no split rows, no
/// draft/MTP sidecars, or every question answered by a flag) it prints
/// nothing and returns the choice.
pub(crate) fn select_pull<'a>(
    ctx: &PullContext<'a, '_>,
    tag: &'a str,
    entry: &'a ModelEntry,
    flags: &PullFlags,
    prompt: &mut dyn Prompt,
) -> Result<Option<PullChoice<'a>>> {
    let mut asked = false;
    let tiers = family_tiers(ctx.registry, tag);
    let (tag, entry) = if tiers.len() > 1 {
        asked = true;
        let index = ask_tier(prompt, ctx.host, tag, &tiers)?;
        (tiers[index].tag, tiers[index].entry)
    } else {
        (tag, entry)
    };
    let ple = match flags.ple {
        Some(selection) => selection,
        None if entry.has_ple_sidecars() => {
            asked = true;
            ask_ple(prompt, ctx.host, tag, entry)?
        }
        None => PleSelection::Auto,
    };
    let draft = match flags.draft {
        Some(draft) => draft,
        None if declared_sidecars(entry).any(|sidecar| sidecar.spec) => {
            asked = true;
            ask_draft(prompt, entry)?
        }
        None => true,
    };
    let expert_hotset = if hipfire_registry::expert_hotset_in(entry, ctx.hotset_pins).is_some() {
        asked = true;
        ask_expert_hotset(prompt)?
    } else {
        true
    };
    let choice = PullChoice {
        tag,
        entry,
        ple,
        draft,
        expert_hotset,
    };
    if !asked {
        return Ok(Some(choice));
    }
    let items = plan_items(ctx, &choice, flags.force);
    Ok(confirm_summary(prompt, ctx.host, &choice, &items)?.then_some(choice))
}

fn ask_tier(
    prompt: &mut dyn Prompt,
    host: &HostFacts,
    root: &str,
    tiers: &[Tier<'_>],
) -> Result<usize> {
    prompt.say(&format!(
        "{root}: {} tiers ({})",
        tiers.len(),
        host_line(host)
    ))?;
    let label_width = tiers.iter().map(|tier| tier.label.len()).max().unwrap_or(0);
    let tag_width = tiers.iter().map(|tier| tier.tag.len()).max().unwrap_or(0);
    for (index, tier) in tiers.iter().enumerate() {
        let default = if index == 0 { "  (default)" } else { "" };
        prompt.say(&format!(
            "  {}. {:<label_width$}  {:<tag_width$}  {:>9}  {:<13}  {}{default}",
            index + 1,
            tier.label,
            tier.tag,
            format_gb(entry_bytes(tier.entry)),
            tier_fit(tier.entry, host).text(),
            tier_note(&tier.entry.desc),
        ))?;
    }
    choose(prompt, "tier", tiers.len(), 0)
}

/// The PLE selection the question preselects, and why.
///
/// It is what `--ple auto` fetches ([`ple::plan_pull`] with
/// `memory.ple_format` over every detected GPU arch: q4 on gfx1151, q8
/// elsewhere, the union on a mixed host, or the published slot when the
/// preferred one is not), except that an `auto` request picks q4 instead of q8
/// when host RAM cannot hold the base plus the q8 rows. The non-interactive
/// default stays plain `auto`.
pub(crate) fn default_ple(entry: &ModelEntry, host: &HostFacts) -> (PleSelection, &'static str) {
    let plan = ple::plan_pull(
        entry,
        PleSelection::Auto,
        host.ple_request,
        &host.gpu_arches,
    );
    let formats: Vec<PleFormat> = plan.sidecars.iter().map(|(format, _)| *format).collect();
    let selection = match formats.as_slice() {
        [PleFormat::Q4] => PleSelection::Q4,
        [] | [PleFormat::Q8] => PleSelection::Q8,
        _ => PleSelection::All,
    };
    if host.ple_request != PleFormatRequest::Auto {
        return (selection, "memory.ple_format");
    }
    if selection == PleSelection::All {
        return (selection, "each GPU's default");
    }
    if selection == PleSelection::Q4 {
        return (selection, "gfx1151");
    }
    let small_ram = match (host.ram_bytes, ple::ple_slot(entry, PleFormat::Q8)) {
        (Some(ram), Some(q8)) => q8
            .size_bytes
            .is_some_and(|rows| ram < entry_bytes(entry).saturating_add(rows)),
        _ => false,
    };
    if small_ram && ple::ple_slot(entry, PleFormat::Q4).is_some() {
        return (PleSelection::Q4, "small RAM");
    }
    (selection, "exact")
}

fn ask_ple(
    prompt: &mut dyn Prompt,
    host: &HostFacts,
    tag: &str,
    entry: &ModelEntry,
) -> Result<PleSelection> {
    let (default_selection, reason) = default_ple(entry, host);
    let mut options: Vec<(PleSelection, String)> = Vec::new();
    let mut both = Some(0_u64);
    for (format, selection, what) in [
        (PleFormat::Q4, PleSelection::Q4, "4-bit Lloyd-Max rows"),
        (PleFormat::Q8, PleSelection::Q8, "exact Q8F16 rows"),
    ] {
        let Some(sidecar) = ple::ple_slot(entry, format) else {
            both = None;
            continue;
        };
        both = both.zip(sidecar.size_bytes).map(|(sum, size)| sum + size);
        options.push((
            selection,
            format!(
                "{:<4}  {:>9}  {what}",
                format.name(),
                format_size(sidecar.size_bytes)
            ),
        ));
    }
    if options.len() == 2 {
        options.push((
            PleSelection::All,
            format!("both  {:>9}  q4 and q8", format_size(both)),
        ));
    }
    options.push((
        PleSelection::Skip,
        format!("none  {:>9}  fetched on first load", ""),
    ));
    let default = options
        .iter()
        .position(|(selection, _)| *selection == default_selection)
        .unwrap_or(0);
    prompt.say(&format!(
        "{tag}: PLE rows for the split base ({})",
        host_line(host)
    ))?;
    for (index, (_, line)) in options.iter().enumerate() {
        let note = if index == default {
            format!("  (default: {reason})")
        } else {
            String::new()
        };
        prompt.say(&format!("  {}. {line}{note}", index + 1))?;
    }
    let index = choose(prompt, "PLE rows", options.len(), default)?;
    Ok(options[index].0)
}

fn ask_draft(prompt: &mut dyn Prompt, entry: &ModelEntry) -> Result<bool> {
    prompt.say("Draft/MTP sidecars declared by the registry:")?;
    for sidecar in declared_sidecars(entry).filter(|sidecar| sidecar.spec) {
        prompt.say(&format!(
            "  {:<8} {}  ({})",
            sidecar.label,
            sidecar.sidecar.file,
            format_size(sidecar.sidecar.size_bytes)
        ))?;
    }
    match confirm(prompt, "Fetch them", true)? {
        Some(answer) => Ok(answer),
        None => bail!("pull cancelled: no answer for the draft/MTP sidecars (end of input)"),
    }
}

/// Per-sidecar yes/no for the expert hot set, default on. The size is fixed
/// text because the summary's GB rounding would print it as 0.00 GB.
fn ask_expert_hotset(prompt: &mut dyn Prompt) -> Result<bool> {
    match confirm(
        prompt,
        "Fetch Expert hot set (R9700 and 7900 XTX decode, 213 KB)",
        true,
    )? {
        Some(answer) => Ok(answer),
        None => bail!("pull cancelled: no answer for the expert hot set (end of input)"),
    }
}

/// One file of the pull plan.
#[derive(Debug, PartialEq)]
pub(crate) struct PlanItem {
    pub label: String,
    pub file: String,
    pub size: Option<u64>,
    pub download: bool,
}

/// Whether this host's GPUs apply the pinned trunk overlay, so the pull
/// fetches it. The loader applies the sidecar only on the one architecture of
/// its trunk default (`TrunkIu4Default::arch`, gfx1151 for the pinned
/// `gfx1151-zab-v1` overlays) unless `HIPFIRE_QWEN4_TRUNK_OVERLAY=1` applies it
/// anywhere (`hipfire_arch_qwen4::trunk_overlay::decide_overlay`). A host with
/// no detected GPU keeps the download: the pull cannot know where the model
/// will run. A mixed host with any gfx1151 card keeps it too.
pub(crate) fn trunk_overlay_wanted(gpu_arches: &[String]) -> bool {
    trunk_overlay_applies(
        hipfire_arch_qwen4::TrunkOverlayPolicy::from_env()
            .unwrap_or(hipfire_arch_qwen4::TrunkOverlayPolicy::Auto),
        gpu_arches,
    )
}

/// [`trunk_overlay_wanted`] under an explicit load policy.
fn trunk_overlay_applies(
    policy: hipfire_arch_qwen4::TrunkOverlayPolicy,
    gpu_arches: &[String],
) -> bool {
    if policy == hipfire_arch_qwen4::TrunkOverlayPolicy::On {
        return true;
    }
    let applying = rdna_compute::TrunkIu4Default::Gfx1151ZabV1.arch();
    gpu_arches.is_empty() || gpu_arches.iter().any(|arch| arch == applying)
}

/// Whether this host's GPUs apply the pinned HC8 sidecar, so the pull fetches
/// it. The loader applies it only on exact gfx1201 (`FeatureFlags::qwen4_hc8_enabled`).
/// A host with no detected GPU keeps the download (the pull cannot know where
/// the model will run), as does a mixed host with any gfx1201 card.
pub(crate) fn hc8_sidecar_wanted(gpu_arches: &[String]) -> bool {
    gpu_arches.is_empty() || gpu_arches.iter().any(|arch| arch == "gfx1201")
}

/// Every file the pull of `choice` touches, in pull order.
pub(crate) fn plan_items(
    ctx: &PullContext<'_, '_>,
    choice: &PullChoice<'_>,
    force: bool,
) -> Vec<PlanItem> {
    plan_items_with_ep(
        ctx,
        choice,
        force,
        hipfire_registry::EP_EXPERT_HOTSET_PINS,
        hipfire_config::memory::EP_EXPERT_HOTSET_DEFAULT_ON,
    )
}

/// [`plan_items`] with the EP2 ranking pin table and its gate supplied.
fn plan_items_with_ep(
    ctx: &PullContext<'_, '_>,
    choice: &PullChoice<'_>,
    force: bool,
    ep_pins: &[ExpertHotsetPin],
    ep_gate: bool,
) -> Vec<PlanItem> {
    let entry = choice.entry;
    let item = |label: String, file: &str, size: Option<u64>| PlanItem {
        label,
        file: file.to_owned(),
        size,
        download: force || !(ctx.present)(file, size),
    };
    let mut base = item("model".to_owned(), &entry.file, entry.size_bytes);
    // A local monolithic of the right size that serves the split entry: the
    // pull verifies its bytes before it relies on it.
    let mut served: Option<&GrandfatheredMonolithic> = None;
    if base.download && !force {
        if entry.is_flash_next_xts() {
            // The pull migrates a verified legacy spelling instead of downloading.
            base.download = !(ctx.present)(
                hipfire_registry::FLASH_NEXT_XTS_LEGACY_FILE,
                entry.size_bytes,
            );
        } else if let Some(pin) =
            hipfire_registry::grandfathered_monolithic_in(entry, ctx.grandfather_pins)
        {
            let local = [pin.canonical_file, pin.legacy_file]
                .into_iter()
                .find(|file| (ctx.present)(file, Some(pin.size_bytes)));
            if let Some(file) = local {
                base = PlanItem {
                    label: "model".to_owned(),
                    file: file.to_owned(),
                    size: Some(pin.size_bytes),
                    download: false,
                };
                served = Some(pin);
            }
        }
    }
    // The file the loader opens: the served monolithic, else the split base.
    let model_file = base.file.clone();
    let mut items = vec![base];
    let request = if choice.ple == PleSelection::Auto {
        ctx.host.ple_request
    } else {
        PleFormatRequest::Auto
    };
    let mut ple_plan = ple::plan_pull(entry, choice.ple, request, &ctx.host.gpu_arches);
    // The exact q8 rows are embedded in a served monolithic.
    if served.is_some() {
        ple::drop_embedded_exact(&mut ple_plan, &model_file);
    }
    for (format, sidecar) in ple_plan.sidecars {
        items.push(item(
            format!("ple-{}", format.name()),
            &sidecar.file,
            sidecar.size_bytes,
        ));
    }
    for sidecar in declared_sidecars(entry).filter(|sidecar| choice.draft || !sidecar.spec) {
        items.push(item(
            sidecar.label.to_owned(),
            &sidecar.sidecar.file,
            sidecar.sidecar.size_bytes,
        ));
    }
    // A served monolithic is a different artifact from the split base: its
    // overlay and hot set are looked up by its own identity.
    let identity = served.map(|pin| ModelEntry {
        file: pin.canonical_file.to_owned(),
        sha256: Some(pin.sha256.to_owned()),
        size_bytes: Some(pin.size_bytes),
        ple: std::collections::BTreeMap::new(),
        ..entry.clone()
    });
    let identity = identity.as_ref().unwrap_or(entry);
    if let Some(overlay) = hipfire_registry::trunk_overlay_in(identity, ctx.overlay_pins)
        .filter(|_| trunk_overlay_wanted(&ctx.host.gpu_arches))
    {
        let local = if served.is_some() {
            format!(
                "{model_file}{}",
                hipfire_config::memory::TRUNK_OVERLAY_SIDECAR_SUFFIX
            )
        } else {
            overlay.sidecar.file.clone()
        };
        items.push(item(
            "overlay".to_owned(),
            &local,
            overlay.sidecar.size_bytes,
        ));
    }
    // The HC8 sidecar installs beside the selected base like the overlay.
    if let Some(hc8) =
        hipfire_registry::hc8_sidecar_in(identity, hipfire_registry::HC8_SIDECAR_PINS)
            .filter(|_| hc8_sidecar_wanted(&ctx.host.gpu_arches))
    {
        let local = format!("{model_file}{}", hipfire_config::memory::HC8_SIDECAR_SUFFIX);
        items.push(item("hc8".to_owned(), &local, hc8.sidecar.size_bytes));
    }
    if choice.expert_hotset {
        if let Some(hotset) = hipfire_registry::expert_hotset_in(identity, ctx.hotset_pins) {
            // The pull installs it beside the selected base file, whatever
            // the pin's remote name is.
            let local = format!(
                "{model_file}{}",
                hipfire_config::memory::EXPERT_HOTSET_SIDECAR_SUFFIX
            );
            items.push(item(
                "Expert hot set".to_owned(),
                &local,
                hotset.sidecar.size_bytes,
            ));
        }
    }
    if choice.expert_hotset {
        if let Some(ep) = hipfire_registry::ep_expert_hotset_in(identity, ep_pins, ep_gate) {
            let local = format!(
                "{model_file}{}",
                hipfire_config::memory::EP_EXPERT_HOTSET_SIDECAR_SUFFIX
            );
            items.push(item(
                "EP expert hot set".to_owned(),
                &local,
                ep.sidecar.size_bytes,
            ));
        }
        // The EP2 hot-unit loan sidecar rides with the EP ranking; an unpublished pin is not
        // downloaded, so it is not part of the plan.
        if let Some(units) = hipfire_registry::hot_units_sidecar_in(
            identity,
            hipfire_registry::HOT_UNITS_SIDECAR_PINS,
        )
        .filter(|units| units.published)
        {
            let local = format!(
                "{model_file}{}",
                hipfire_config::memory::EP_HOT_UNITS_SIDECAR_SUFFIX
            );
            items.push(item(
                "EP hot-unit loan".to_owned(),
                &local,
                units.sidecar.size_bytes,
            ));
        }
    }
    for (name, sidecar) in &entry.heads {
        items.push(item(
            format!("head {name}"),
            &sidecar.file,
            sidecar.size_bytes,
        ));
    }
    items
}

/// Total bytes to download and the number of downloads of unknown size.
pub(crate) fn download_total(items: &[PlanItem]) -> (u64, usize) {
    items
        .iter()
        .filter(|item| item.download)
        .fold((0, 0), |(bytes, unknown), item| match item.size {
            Some(size) => (bytes + size, unknown),
            None => (bytes, unknown + 1),
        })
}

/// `Some(true)` when the free space holds the download, `None` when unknown.
pub(crate) fn disk_fits(total: u64, free: Option<u64>) -> Option<bool> {
    free.map(|free| total <= free)
}

fn confirm_summary(
    prompt: &mut dyn Prompt,
    host: &HostFacts,
    choice: &PullChoice<'_>,
    items: &[PlanItem],
) -> Result<bool> {
    prompt.say(&format!("Pull summary: {}", choice.tag))?;
    for item in items {
        let state = if item.download { "download" } else { "present" };
        prompt.say(&ple::plan_line(&item.label, &item.file, item.size, state))?;
    }
    let (total, unknown) = download_total(items);
    let unknown = if unknown > 0 {
        format!(" + {unknown} file(s) of unknown size")
    } else {
        String::new()
    };
    let fits = disk_fits(total, host.free_disk_bytes);
    let disk = match (fits, host.free_disk_bytes) {
        (Some(true), Some(free)) => format!("free disk {} (ok)", format_gb(free)),
        (Some(false), Some(free)) => format!(
            "free disk {} (NOT ENOUGH: {} short)",
            format_gb(free),
            format_gb(total - free)
        ),
        _ => "free disk unknown".to_owned(),
    };
    prompt.say(&format!(
        "Total download: {}{unknown}; {disk}",
        format_gb(total)
    ))?;
    Ok(confirm(prompt, "Proceed", fits != Some(false))?.unwrap_or(false))
}

fn format_gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

fn format_size(bytes: Option<u64>) -> String {
    bytes.map_or_else(|| "size ?".to_owned(), format_gb)
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

/// The line-oriented terminal the questions talk to.
pub(crate) trait Prompt {
    fn say(&mut self, line: &str) -> Result<()>;
    /// Print `question` and read one answer line; `None` at end of input.
    fn ask(&mut self, question: &str) -> Result<Option<String>>;
}

/// [`Prompt`] over a reader and a writer (stdin and stderr in the CLI).
pub(crate) struct LinePrompt<R, W> {
    input: R,
    output: W,
}

impl<R: BufRead, W: Write> LinePrompt<R, W> {
    pub(crate) fn new(input: R, output: W) -> Self {
        Self { input, output }
    }
}

impl<R: BufRead, W: Write> Prompt for LinePrompt<R, W> {
    fn say(&mut self, line: &str) -> Result<()> {
        writeln!(self.output, "{line}")?;
        Ok(())
    }

    fn ask(&mut self, question: &str) -> Result<Option<String>> {
        write!(self.output, "{question}")?;
        self.output.flush()?;
        let mut line = String::new();
        if self.input.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        Ok(Some(line))
    }
}

/// Numbered choice; Enter takes `default`. End of input cancels the pull.
fn choose(prompt: &mut dyn Prompt, label: &str, count: usize, default: usize) -> Result<usize> {
    loop {
        let question = format!("Select {label} [1-{count}, Enter = {}]: ", default + 1);
        let Some(line) = prompt.ask(&question)? else {
            bail!("pull cancelled: no {label} selected (end of input)");
        };
        let answer = line.trim();
        if answer.is_empty() {
            return Ok(default);
        }
        match answer.parse::<usize>() {
            Ok(number) if (1..=count).contains(&number) => return Ok(number - 1),
            _ => prompt.say("  invalid selection")?,
        }
    }
}

/// Yes/no; Enter takes `default`, `None` at end of input.
fn confirm(prompt: &mut dyn Prompt, question: &str, default: bool) -> Result<Option<bool>> {
    let hint = if default { "[Y/n]" } else { "[y/N]" };
    loop {
        let Some(line) = prompt.ask(&format!("{question}? {hint} "))? else {
            return Ok(None);
        };
        match line.trim().to_ascii_lowercase().as_str() {
            "" => return Ok(Some(default)),
            "y" | "yes" => return Ok(Some(true)),
            "n" | "no" => return Ok(Some(false)),
            _ => prompt.say("  answer y or n")?,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`plan_items`] without the EP2 ranking sidecar (the shipped gate adds it to every
    /// Flash-Next plan): the tests that assert exact file lists of the other sidecars.
    fn plan_items_no_ep(
        ctx: &PullContext<'_, '_>,
        choice: &PullChoice<'_>,
        force: bool,
    ) -> Vec<PlanItem> {
        plan_items_with_ep(
            ctx,
            choice,
            force,
            hipfire_registry::EP_EXPERT_HOTSET_PINS,
            false,
        )
    }
    use crate::{Cli, Commands};
    use clap::Parser;
    use std::io::Cursor;

    const MIB: u64 = 1 << 20;
    const GB: u64 = 1_000_000_000;

    fn registry() -> RegistryV1 {
        let value = serde_json::json!({
            "schema_version": 1,
            "generated_at": "2099-01-01T00:00:00Z",
            "models": {
                "qwen3.8:flash-next": {
                    "repo": "hipfire-models/qwen3.8-flash-next",
                    "file": "qwen3.8-flash-next-base.mq4-xts",
                    "size_gb": 70.9, "size_bytes": 70_900_000_000_u64,
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "min_vram_gb": 72.0, "arch_id": 16,
                    "desc": "Best quality. Canonical rung.",
                    "ple": {
                        "q4": {"file": "qwen3.8-flash-next-base.mq4-xts.ple-q4", "size_bytes": 28_800_000_000_u64},
                        "q8": {"file": "qwen3.8-flash-next-base.mq4-xts.ple-q8", "size_bytes": 54_400_000_000_u64}
                    },
                    "mtp": {"file": "qwen3.8-flash-next.mtp", "size_bytes": 400_000_000_u64}
                },
                "qwen3.8:flash-next-mq3-xts": {
                    "repo": "hipfire-models/qwen3.8-flash-next",
                    "file": "qwen3.8-flash-next-base.mq3-xts",
                    "size_gb": 55.0, "size_bytes": 55_000_000_000_u64,
                    "min_vram_gb": 56.0, "arch_id": 16,
                    "desc": "Balanced",
                    "ple": {"q8": {"file": "qwen3.8-flash-next-base.mq3-xts.ple-q8", "size_bytes": 54_400_000_000_u64}}
                },
                "qwen3.8:flash-next-mq2-xts": {
                    "repo": "hipfire-models/qwen3.8-flash-next",
                    "file": "qwen3.8-flash-next.mq2-xts",
                    "size_gb": 40.0, "min_vram_gb": 42.0, "arch_id": 16,
                    "desc": "Smallest"
                },
                // Same artifact as the root under another tag.
                "qwen3.8:flash-next-canon": {
                    "repo": "hipfire-models/qwen3.8-flash-next",
                    "file": "qwen3.8-flash-next-base.mq4-xts",
                    "size_gb": 70.9, "min_vram_gb": 72.0, "arch_id": 16, "desc": "dup"
                },
                // A second stock tag of the root's split base.
                "qwen3.8:flash-next-mq4": {
                    "repo": "hipfire-models/qwen3.8-flash-next",
                    "file": "qwen3.8-flash-next-base.mq4-xts",
                    "size_gb": 70.9, "min_vram_gb": 72.0, "arch_id": 16, "desc": "dup"
                },
                "qwen3.8:flash-next-draft": {
                    "repo": "hipfire-models/qwen3.8-flash-next",
                    "file": "qwen3.8-flash-next-draft.hfq",
                    "size_gb": 1.0, "min_vram_gb": 2.0, "arch_id": 20, "desc": "draft"
                },
                "qwen3.8:flash-next-local": {
                    "repo": "", "file": "qwen3.8-flash-next-local.mq4",
                    "size_gb": 10.0, "min_vram_gb": 12.0, "arch_id": 16, "desc": "local"
                },
                "qwen3.8:flash-nextgen": {
                    "repo": "x/y", "file": "nextgen.mq4",
                    "size_gb": 9.0, "min_vram_gb": 10.0, "arch_id": 16, "desc": "other family"
                },
                "solo": {
                    "repo": "x/solo", "file": "solo.mq4",
                    "size_gb": 4.0, "min_vram_gb": 6.0, "desc": "single variant"
                }
            },
            "aliases": {"qwen3.8:flash": "qwen3.8:flash-next"}
        });
        RegistryV1::parse(&value.to_string(), "pull select test").unwrap()
    }

    fn host() -> HostFacts {
        HostFacts {
            gpu_arches: vec!["gfx1201".into()],
            vram_bytes: Some(96 << 30),
            ram_bytes: Some(256 << 30),
            free_disk_bytes: Some(500 * GB),
            ple_request: PleFormatRequest::Auto,
        }
    }

    fn absent(_: &str, _: Option<u64>) -> bool {
        false
    }

    /// The chosen (tag, ple, draft); `None` when declined.
    type Picked = Option<(String, PleSelection, bool)>;

    /// Run [`select_pull`] for `input` with scripted answers; returns the
    /// chosen (tag, ple, draft) or `None` when declined, plus the transcript.
    fn run(
        input: &str,
        flags: PullFlags,
        host: &HostFacts,
        answers: &str,
    ) -> (Result<Picked>, String) {
        let (result, transcript) = run_pins(input, flags, host, answers, &[]);
        (
            result.map(|picked| picked.map(|(tag, ple, draft, _)| (tag, ple, draft))),
            transcript,
        )
    }

    /// [`run`] with expert-hotset pins; also returns the hot-set choice.
    #[allow(clippy::type_complexity)]
    fn run_pins(
        input: &str,
        flags: PullFlags,
        host: &HostFacts,
        answers: &str,
        hotset_pins: &[ExpertHotsetPin],
    ) -> (Result<Option<(String, PleSelection, bool, bool)>>, String) {
        let registry = registry();
        let (tag, entry) = registry.model(input).unwrap();
        let ctx = PullContext {
            registry: &registry,
            host,
            overlay_pins: &[],
            grandfather_pins: &[],
            hotset_pins,
            present: &absent,
        };
        let mut output = Vec::new();
        let result = {
            let mut prompt = LinePrompt::new(Cursor::new(answers.as_bytes()), &mut output);
            select_pull(&ctx, tag, entry, &flags, &mut prompt)
                .map(|choice| choice.map(|c| (c.tag.to_owned(), c.ple, c.draft, c.expert_hotset)))
        };
        (result, String::from_utf8(output).unwrap())
    }

    #[test]
    fn tiers_come_from_the_registry_family() {
        let registry = registry();
        let tiers = family_tiers(&registry, "qwen3.8:flash-next");
        let shape: Vec<(&str, &str)> = tiers
            .iter()
            .map(|tier| (tier.label.as_str(), tier.tag))
            .collect();
        // Root first; then largest first. Same-artifact tags (every stock MQ4
        // tag is the one split base), drafts (other arch), local-only entries and the
        // `flash-nextgen` prefix lookalike are not tiers.
        assert_eq!(
            shape,
            [
                ("mq4-xts", "qwen3.8:flash-next"),
                ("mq3-xts", "qwen3.8:flash-next-mq3-xts"),
                ("mq2-xts", "qwen3.8:flash-next-mq2-xts"),
            ]
        );
        // A specific variant has no sub-tiers; neither does a single model.
        assert_eq!(
            family_tiers(&registry, "qwen3.8:flash-next-mq3-xts").len(),
            1
        );
        assert_eq!(family_tiers(&registry, "solo").len(), 1);
        assert!(family_tiers(&registry, "missing").is_empty());
    }

    #[test]
    fn fit_reads_vram_then_vram_plus_ram() {
        let registry = registry();
        let entry = |tag: &str| &registry.models[tag];
        let host = host();
        assert_eq!(tier_fit(entry("qwen3.8:flash-next"), &host), Fit::Vram);
        let small = HostFacts {
            vram_bytes: Some(16 << 30),
            ram_bytes: Some(48 << 30),
            ..host.clone()
        };
        assert_eq!(
            tier_fit(entry("qwen3.8:flash-next-mq3-xts"), &small),
            Fit::VramAndRam
        );
        assert_eq!(tier_fit(entry("qwen3.8:flash-next"), &small), Fit::TooLarge);
        let unknown = HostFacts {
            vram_bytes: None,
            ..host
        };
        assert_eq!(tier_fit(entry("solo"), &unknown), Fit::Unknown);
    }

    #[test]
    fn enter_takes_every_default_and_the_summary_confirms() {
        let (result, transcript) = run(
            "qwen3.8:flash-next",
            PullFlags::default(),
            &host(),
            "\n\n\n\n",
        );
        // gfx1201 with 256 GiB RAM: q8 is the exact default.
        assert_eq!(
            result.unwrap(),
            Some(("qwen3.8:flash-next".into(), PleSelection::Q8, true))
        );
        for expected in [
            "qwen3.8:flash-next: 3 tiers (this machine: gfx1201, VRAM 96.0 GiB, RAM 256.0 GiB)",
            "1. mq4-xts",
            "fits VRAM",
            "Best quality  (default)",
            "2. q8     54.40 GB  exact Q8F16 rows  (default: exact)",
            "3. both   83.20 GB  q4 and q8",
            "4. none",
            "MTP",
            "Total download: 125.70 GB; free disk 500.00 GB (ok)",
            "Proceed? [Y/n]",
        ] {
            assert!(
                transcript.contains(expected),
                "missing {expected:?} in\n{transcript}"
            );
        }
    }

    #[test]
    fn answers_choose_the_tier_rows_and_sidecars() {
        // Tier 2 (mq3-xts publishes only q8), rows "none", no draft question
        // (no draft/MTP sidecars on that tier), then confirm.
        let (result, transcript) = run("qwen3.8:flash", PullFlags::default(), &host(), "2\n2\ny\n");
        assert_eq!(
            result.unwrap(),
            Some((
                "qwen3.8:flash-next-mq3-xts".into(),
                PleSelection::Skip,
                true
            ))
        );
        assert!(!transcript.contains("Draft/MTP"));
        assert!(!transcript.contains("both"));

        // Root, q4, decline the MTP sidecar.
        let (result, transcript) = run(
            "qwen3.8:flash-next",
            PullFlags::default(),
            &host(),
            "1\n1\nn\n\n",
        );
        assert_eq!(
            result.unwrap(),
            Some(("qwen3.8:flash-next".into(), PleSelection::Q4, false))
        );
        assert!(!transcript.contains("qwen3.8-flash-next.mtp  (0.40 GB)  download"));

        // Invalid input re-asks; the default `n` at the summary declines.
        let (result, transcript) = run(
            "qwen3.8:flash-next",
            PullFlags::default(),
            &host(),
            "9\nx\n3\nn\n",
        );
        assert_eq!(result.unwrap(), None);
        assert!(transcript.contains("invalid selection"));
    }

    #[test]
    fn gfx1151_and_small_ram_preselect_q4() {
        let halo = HostFacts {
            gpu_arches: vec!["gfx1151".into()],
            ..host()
        };
        let (result, transcript) = run(
            "qwen3.8:flash-next",
            PullFlags::default(),
            &halo,
            "\n\n\n\n",
        );
        assert_eq!(result.unwrap().unwrap().1, PleSelection::Q4);
        assert!(transcript.contains("(default: gfx1151)"));

        let small = HostFacts {
            ram_bytes: Some(64 << 30),
            ..host()
        };
        let registry = registry();
        let entry = &registry.models["qwen3.8:flash-next"];
        assert_eq!(default_ple(entry, &small), (PleSelection::Q4, "small RAM"));
        let configured = HostFacts {
            ple_request: PleFormatRequest::Q8,
            ..small
        };
        assert_eq!(
            default_ple(entry, &configured),
            (PleSelection::Q8, "memory.ple_format")
        );
        // A tier without q4 rows falls back to the published slot.
        let mq3 = &registry.models["qwen3.8:flash-next-mq3-xts"];
        assert_eq!(default_ple(mq3, &halo).0, PleSelection::Q8);
        // A mixed host (a gfx1100 next to the gfx1151) preselects both formats.
        let hipx = HostFacts {
            gpu_arches: vec!["gfx1100".into(), "gfx1151".into()],
            ..host()
        };
        assert_eq!(
            default_ple(entry, &hipx),
            (PleSelection::All, "each GPU's default")
        );
    }

    #[test]
    fn flags_skip_their_questions() {
        // --ple and --no-draft leave only the tier question (and the summary).
        let flags = PullFlags {
            ple: Some(PleSelection::All),
            draft: Some(false),
            force: false,
        };
        let (result, transcript) = run("qwen3.8:flash-next", flags, &host(), "\n\n");
        assert_eq!(
            result.unwrap(),
            Some(("qwen3.8:flash-next".into(), PleSelection::All, false))
        );
        assert!(!transcript.contains("PLE rows for"));
        assert!(!transcript.contains("Draft/MTP"));
        assert!(transcript.contains("ple-q4"));
        assert!(transcript.contains("ple-q8"));

        // A specific tier with every question answered asks nothing at all.
        let flags = PullFlags {
            ple: Some(PleSelection::Q8),
            draft: Some(true),
            force: false,
        };
        let (result, transcript) = run("qwen3.8:flash-next-mq3-xts", flags, &host(), "");
        assert_eq!(
            result.unwrap(),
            Some(("qwen3.8:flash-next-mq3-xts".into(), PleSelection::Q8, true))
        );
        assert!(transcript.is_empty());
    }

    #[test]
    fn a_single_variant_without_sidecars_asks_nothing() {
        let (result, transcript) = run("solo", PullFlags::default(), &host(), "");
        assert_eq!(
            result.unwrap(),
            Some(("solo".into(), PleSelection::Auto, true))
        );
        assert!(transcript.is_empty());
    }

    #[test]
    fn end_of_input_cancels_instead_of_guessing() {
        let (result, _) = run("qwen3.8:flash-next", PullFlags::default(), &host(), "");
        assert!(result.unwrap_err().to_string().contains("no tier selected"));
        let (result, _) = run(
            "qwen3.8:flash-next",
            PullFlags::default(),
            &host(),
            "\n\n\n",
        );
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn non_interactive_pulls_take_the_previous_defaults() {
        assert!(is_interactive(true, false));
        assert!(!is_interactive(false, false));
        assert!(!is_interactive(true, true));
        let registry = registry();
        let (tag, entry) = registry.model("qwen3.8:flash-next").unwrap();
        assert_eq!(
            PullChoice::defaults(tag, entry, &PullFlags::default()),
            PullChoice {
                tag: "qwen3.8:flash-next",
                entry,
                ple: PleSelection::Auto,
                draft: true,
                expert_hotset: true,
            }
        );
        let flags = PullFlags {
            ple: Some(PleSelection::Skip),
            draft: Some(false),
            force: true,
        };
        let choice = PullChoice::defaults(tag, entry, &flags);
        assert_eq!((choice.ple, choice.draft), (PleSelection::Skip, false));
    }

    /// The pin's remote file name; the pull installs it next to the selected
    /// entry's own file, so the plan lists [`LOCAL_HOTSET_FILE`].
    const HOTSET_FILE: &str = "qwen3.8-flash-next.mq4-xts.expert-hotset";
    const LOCAL_HOTSET_FILE: &str = "qwen3.8-flash-next-base.mq4-xts.expert-hotset";
    const HOTSET_SIZE: u64 = 217_676;
    const BASE_DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER_DIGEST: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const HOTSET_DIGEST: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    const HOTSET_PINS: &[ExpertHotsetPin] = &[ExpertHotsetPin {
        base_sha256: BASE_DIGEST,
        base_size: 70_900_000_000,
        repo: "hipfire-models/qwen3.8-flash-next",
        file: HOTSET_FILE,
        sha256: HOTSET_DIGEST,
        size_bytes: HOTSET_SIZE,
    }];

    const OTHER_BASE_PINS: &[ExpertHotsetPin] = &[ExpertHotsetPin {
        base_sha256: OTHER_DIGEST,
        base_size: 70_900_000_000,
        repo: "hipfire-models/qwen3.8-flash-next",
        file: HOTSET_FILE,
        sha256: HOTSET_DIGEST,
        size_bytes: HOTSET_SIZE,
    }];

    #[test]
    fn a_matching_expert_hotset_is_asked_and_defaults_on() {
        let (result, transcript) = run_pins(
            "qwen3.8:flash-next",
            PullFlags::default(),
            &host(),
            "\n\n\n\n\n",
            HOTSET_PINS,
        );
        let (tag, _, draft, hotset) = result.unwrap().unwrap();
        assert_eq!(
            (tag.as_str(), draft, hotset),
            ("qwen3.8:flash-next", true, true)
        );
        for expected in [
            "Fetch Expert hot set (R9700 and 7900 XTX decode, 213 KB)? [Y/n]",
            LOCAL_HOTSET_FILE,
            "Total download: 125.70 GB",
        ] {
            assert!(
                transcript.contains(expected),
                "missing {expected:?} in\n{transcript}"
            );
        }
        let summary = transcript.split("Pull summary").nth(1).unwrap();
        let line = summary
            .lines()
            .find(|line| line.contains(LOCAL_HOTSET_FILE))
            .unwrap();
        assert!(line.contains("Expert hot set") && line.ends_with("download"));
    }

    #[test]
    fn the_ep_expert_hotset_rides_the_expert_hotset_choice_behind_its_gate() {
        const EP_SIZE: u64 = 312_036;
        const EP_PINS: &[ExpertHotsetPin] = &[ExpertHotsetPin {
            base_sha256: BASE_DIGEST,
            base_size: 70_900_000_000,
            repo: "hipfire-models/qwen3.8-flash-next",
            file: "ignored-remote-name.ep.expert-hotset",
            sha256: HOTSET_DIGEST,
            size_bytes: EP_SIZE,
        }];
        const LOCAL_EP_FILE: &str = "qwen3.8-flash-next-base.mq4-xts.ep.expert-hotset";
        let registry = registry();
        let host = host();
        let (tag, entry) = registry.model("qwen3.8:flash-next").unwrap();
        let ctx = PullContext {
            registry: &registry,
            host: &host,
            overlay_pins: &[],
            grandfather_pins: &[],
            hotset_pins: HOTSET_PINS,
            present: &absent,
        };
        let mut choice = PullChoice {
            tag,
            entry,
            ple: PleSelection::Skip,
            draft: false,
            expert_hotset: true,
        };
        let baseline = plan_items(&ctx, &choice, false);
        let off = plan_items_with_ep(&ctx, &choice, false, EP_PINS, false);
        assert_eq!(off, baseline);
        assert!(off.iter().all(|item| item.label != "EP expert hot set"));

        let on = plan_items_with_ep(&ctx, &choice, false, EP_PINS, true);
        let extra: Vec<&PlanItem> = on
            .iter()
            .filter(|item| item.label == "EP expert hot set")
            .collect();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].file, LOCAL_EP_FILE);
        assert_eq!(extra[0].size, Some(EP_SIZE));
        assert_eq!(on.len(), baseline.len() + 1);

        choice.expert_hotset = false;
        let declined = plan_items_with_ep(&ctx, &choice, false, EP_PINS, true);
        assert!(declined
            .iter()
            .all(|item| item.label != "EP expert hot set"));
    }

    #[test]
    fn the_expert_hotset_is_planned_on_every_host_so_a_gfx1100_pull_fetches_it() {
        // The pin is keyed by the base artifact identity, never by GPU architecture: a 7900 XTX
        // (gfx1100) host plans the same single 217,675-byte sidecar as an R9700 (gfx1201), and a
        // host with no detected GPU or another arch plans exactly that one file as well (no
        // additional download on any arch).
        let registry = registry();
        let (tag, entry) = registry.model("qwen3.8:flash-next").unwrap();
        let mut planned: Vec<Vec<(String, Option<u64>)>> = Vec::new();
        for arches in [
            vec!["gfx1100".to_owned()],
            vec!["gfx1201".to_owned()],
            vec!["gfx1151".to_owned()],
            vec!["gfx1100".to_owned(), "gfx1151".to_owned()],
            Vec::new(),
        ] {
            let host = HostFacts {
                gpu_arches: arches.clone(),
                ..host()
            };
            let ctx = PullContext {
                registry: &registry,
                host: &host,
                overlay_pins: &[],
                grandfather_pins: &[],
                hotset_pins: HOTSET_PINS,
                present: &absent,
            };
            let choice = PullChoice {
                tag,
                entry,
                ple: PleSelection::Skip,
                draft: false,
                expert_hotset: true,
            };
            let items = plan_items(&ctx, &choice, false);
            let hot: Vec<_> = items
                .iter()
                .filter(|item| item.label.contains("hot set"))
                .map(|item| (item.file.clone(), item.size))
                .collect();
            assert_eq!(
                hot,
                vec![(LOCAL_HOTSET_FILE.to_owned(), Some(HOTSET_SIZE))],
                "{arches:?}"
            );
            planned.push(items.iter().map(|i| (i.file.clone(), i.size)).collect());
        }
        assert!(
            planned.windows(2).all(|pair| pair[0] == pair[1]),
            "plan differs by host arch"
        );
    }

    #[test]
    fn declining_the_expert_hotset_removes_it_from_the_summary_and_total() {
        let registry = registry();
        let host = host();
        let (tag, entry) = registry.model("qwen3.8:flash-next").unwrap();
        let ctx = PullContext {
            registry: &registry,
            host: &host,
            overlay_pins: &[],
            grandfather_pins: &[],
            hotset_pins: HOTSET_PINS,
            present: &absent,
        };
        let mut choice = PullChoice {
            tag,
            entry,
            ple: PleSelection::Skip,
            draft: false,
            expert_hotset: true,
        };
        let with = plan_items(&ctx, &choice, false);
        assert!(with
            .iter()
            .any(|item| item.label == "Expert hot set" && item.file == LOCAL_HOTSET_FILE));
        assert_eq!(download_total(&with), (70_900_000_000 + HOTSET_SIZE, 0));
        choice.expert_hotset = false;
        let without = plan_items(&ctx, &choice, false);
        assert!(without.iter().all(|item| item.file != LOCAL_HOTSET_FILE));
        assert_eq!(download_total(&without), (70_900_000_000, 0));

        // The present check reads the local file beside the base.
        let present = |file: &str, _: Option<u64>| file == LOCAL_HOTSET_FILE;
        let ctx = PullContext {
            present: &present,
            ..ctx
        };
        choice.expert_hotset = true;
        let items = plan_items(&ctx, &choice, false);
        let hotset = items
            .iter()
            .find(|item| item.file == LOCAL_HOTSET_FILE)
            .unwrap();
        assert!(!hotset.download);
        assert_eq!(download_total(&items), (70_900_000_000, 0));

        // Answering `n` stores the deselection; the summary omits the file.
        // Answers: tier 1, PLE none (4), no draft, no hot set, confirm.
        let (result, transcript) = run_pins(
            "qwen3.8:flash-next",
            PullFlags::default(),
            &host,
            "1\n4\nn\nn\n\n",
            HOTSET_PINS,
        );
        let (_, _, _, hotset) = result.unwrap().unwrap();
        assert!(!hotset);
        let summary = transcript.split("Pull summary").nth(1).unwrap();
        assert!(!summary.contains(LOCAL_HOTSET_FILE));
        assert!(summary.contains("Total download: 70.90 GB"));
    }

    #[test]
    fn an_absent_expert_hotset_pin_asks_and_lists_nothing() {
        // No pins, then a pin for another base identity: no question, no item.
        for pins in [&[][..], OTHER_BASE_PINS] {
            let (result, transcript) = run_pins(
                "qwen3.8:flash-next",
                PullFlags::default(),
                &host(),
                "\n\n\n\n",
                pins,
            );
            assert!(result.unwrap().unwrap().3);
            assert!(!transcript.contains("Expert hot set"));
            assert!(!transcript.contains(LOCAL_HOTSET_FILE));
        }
    }

    #[test]
    fn no_input_defaults_the_expert_hotset_on() {
        let registry = registry();
        let (tag, entry) = registry.model("qwen3.8:flash-next").unwrap();
        assert!(PullChoice::defaults(tag, entry, &PullFlags::default()).expert_hotset);
        let flags = PullFlags {
            ple: Some(PleSelection::Skip),
            draft: Some(false),
            force: true,
        };
        assert!(PullChoice::defaults(tag, entry, &flags).expert_hotset);

        // With every flag given and the tier answered, the hot set is the one
        // open question; end of input cancels instead of guessing.
        let flags = PullFlags {
            ple: Some(PleSelection::Q8),
            draft: Some(true),
            force: false,
        };
        let (result, _) = run_pins("qwen3.8:flash-next", flags, &host(), "1\n", HOTSET_PINS);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no answer for the expert hot set"));
    }

    #[test]
    fn plan_items_count_downloads_and_check_the_disk() {
        let registry = registry();
        let host = host();
        let (tag, entry) = registry.model("qwen3.8:flash-next").unwrap();
        let present = |file: &str, _: Option<u64>| file.ends_with(".ple-q4");
        let ctx = PullContext {
            registry: &registry,
            host: &host,
            overlay_pins: &[],
            grandfather_pins: &[],
            hotset_pins: &[],
            present: &present,
        };
        let choice = PullChoice {
            tag,
            entry,
            ple: PleSelection::All,
            draft: false,
            expert_hotset: true,
        };
        let items = plan_items(&ctx, &choice, false);
        let shape: Vec<(&str, bool)> = items
            .iter()
            .map(|item| (item.label.as_str(), item.download))
            .collect();
        assert_eq!(
            shape,
            [("model", true), ("ple-q4", false), ("ple-q8", true)]
        );
        assert_eq!(download_total(&items), (125_300_000_000, 0));
        // --force re-downloads what is present.
        let forced = plan_items(&ctx, &choice, true);
        assert!(forced.iter().all(|item| item.download));
        assert_eq!(disk_fits(10 * GB, Some(9 * GB)), Some(false));
        assert_eq!(disk_fits(10 * GB, Some(10 * GB)), Some(true));
        assert_eq!(disk_fits(10 * GB, None), None);
    }

    #[test]
    fn a_short_disk_defaults_the_summary_to_no() {
        let tight = HostFacts {
            free_disk_bytes: Some(MIB),
            ..host()
        };
        let (result, transcript) = run(
            "qwen3.8:flash-next",
            PullFlags::default(),
            &tight,
            "\n\n\n\n",
        );
        assert_eq!(result.unwrap(), None);
        assert!(transcript.contains("NOT ENOUGH"));
        assert!(transcript.contains("Proceed? [y/N]"));
    }

    #[test]
    fn vram_is_the_largest_card() {
        let root = std::env::temp_dir().join(format!(
            "hipfire-pull-select-drm-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for (card, vram) in [("card0", "17163091968\n"), ("card1", "34342961152\n")] {
            let device = root.join(card).join("device");
            fs::create_dir_all(&device).unwrap();
            fs::write(device.join("mem_info_vram_total"), vram).unwrap();
        }
        // Connector nodes are not cards.
        fs::create_dir_all(root.join("card1-DP-1/device")).unwrap();
        fs::write(
            root.join("card1-DP-1/device/mem_info_vram_total"),
            "999999999999\n",
        )
        .unwrap();
        assert_eq!(largest_vram_bytes(&root), Some(34_342_961_152));
        assert_eq!(largest_vram_bytes(&root.join("missing")), None);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn clap_parses_the_pull_prompt_flags() {
        let pull = |args: &[&str]| {
            let mut argv = vec!["hipfire", "pull", "qwen3.8:flash-next"];
            argv.extend_from_slice(args);
            match Cli::try_parse_from(argv).unwrap().command.unwrap() {
                Commands::Pull(pull) => (pull.ple, pull.draft_flag(), pull.yes),
                other => panic!("expected pull, got {other:?}"),
            }
        };
        assert_eq!(pull(&[]), (None, None, false));
        assert_eq!(pull(&["-y"]), (None, None, true));
        assert_eq!(pull(&["--yes"]), (None, None, true));
        assert_eq!(pull(&["--draft"]), (None, Some(true), false));
        assert_eq!(pull(&["--no-draft"]), (None, Some(false), false));
        // The last of --draft / --no-draft wins.
        assert_eq!(pull(&["--draft", "--no-draft"]), (None, Some(false), false));
        assert_eq!(pull(&["--no-draft", "--draft"]), (None, Some(true), false));
        assert_eq!(
            pull(&["--ple", "q4"]),
            (Some(PleSelection::Q4), None, false)
        );
    }

    const STOCK_TAGS: [&str; 5] = [
        "qwen3.8:flash-next",
        "qwen3.8:flash-next-mq4",
        "qwen3.8:flash-next-gptq3",
        "qwen3.8:flash-next-mq4-xts",
        "qwen3.8:flash",
    ];
    const STOCK_BASE: &str = "qwen3.8-flash-next-base.mq4-xts";
    const STOCK_BASE_SIZE: u64 = 70_888_270_360;
    const STOCK_BASE_SHA: &str = "eafaff94fb9c6bb51282942e0ea6441cbdc266b634bb003b5913cfe2ccb90f45";
    const STOCK_Q4_SIZE: u64 = 29_120_146_432;
    const STOCK_Q8_SIZE: u64 = 54_400_269_312;

    /// The four stock MQ4 tags and the `qwen3.8:flash` alias: one split base
    /// on the shared PLE pair, exactly as the bundled registry publishes them.
    fn stock_registry() -> RegistryV1 {
        let entry = serde_json::json!({
            "repo": "hipfire-models/qwen3.8-flash-next",
            "file": STOCK_BASE,
            "size_gb": 70.9, "min_vram_gb": 72.0, "arch_id": 16, "quant": "mq4",
            "desc": "Qwen3.8-Flash-Next MQ4 XTS split base",
            "sha256": STOCK_BASE_SHA, "size_bytes": STOCK_BASE_SIZE,
            "ple": {
                "q4": {
                    "file": "qwen3.8-flash-next.ple-q4",
                    "sha256": "fc78d30b9d2376db2b713732e80de2069f1cd1d6e1e40b50c716927dcc18ee1e",
                    "size_bytes": STOCK_Q4_SIZE
                },
                "q8": {
                    "file": "qwen3.8-flash-next.ple-q8",
                    "sha256": "a655edc25402d0dfbef276cb3c5ca57724fa1074976d8a9029371a3a52eca89c",
                    "size_bytes": STOCK_Q8_SIZE
                }
            }
        });
        let value = serde_json::json!({
            "schema_version": 1,
            "generated_at": "2099-01-01T00:00:00Z",
            "models": {
                "qwen3.8:flash-next": entry,
                "qwen3.8:flash-next-mq4": entry,
                "qwen3.8:flash-next-gptq3": entry,
                "qwen3.8:flash-next-mq4-xts": entry
            },
            "aliases": {"qwen3.8:flash": "qwen3.8:flash-next"}
        });
        RegistryV1::parse(&value.to_string(), "stock registry").unwrap()
    }

    fn host_with_arch(arch: &str) -> HostFacts {
        HostFacts {
            gpu_arches: if arch.is_empty() {
                Vec::new()
            } else {
                vec![arch.into()]
            },
            vram_bytes: None,
            ram_bytes: None,
            free_disk_bytes: None,
            ple_request: PleFormatRequest::Auto,
        }
    }

    #[test]
    fn stock_mq4_tags_plan_q4_on_strix_halo_and_q8_elsewhere() {
        let registry = stock_registry();
        let models = Path::new("models");
        for input in STOCK_TAGS {
            let (tag, entry) = registry.model(input).unwrap();
            assert_eq!(entry.file, STOCK_BASE, "{input}");
            // gfx1151 prefers q4; RDNA4, other arches and an unknown arch get q8.
            for (arch, format, ple_bytes) in [
                ("gfx1151", PleFormat::Q4, STOCK_Q4_SIZE),
                ("gfx1201", PleFormat::Q8, STOCK_Q8_SIZE),
                ("gfx1100", PleFormat::Q8, STOCK_Q8_SIZE),
                ("", PleFormat::Q8, STOCK_Q8_SIZE),
            ] {
                let host = host_with_arch(arch);
                let plan = ple::plan_pull(
                    entry,
                    PleSelection::Auto,
                    host.ple_request,
                    &host.gpu_arches,
                );
                assert_eq!(plan.sidecars.len(), 1, "{input} {arch}");
                let (planned, sidecar) = plan.sidecars[0];
                assert_eq!(planned, format, "{input} {arch}");
                assert_eq!(sidecar.size_bytes, Some(ple_bytes));
                // The shared file is linked under the name the loader opens.
                assert!(ple::is_shared_slot(entry, planned.name(), sidecar));
                assert_eq!(
                    hipfire_config::memory::ple_sidecar_path(&models.join(&entry.file), planned),
                    models.join(format!("{STOCK_BASE}.ple-{}", planned.name()))
                );

                let ctx = PullContext {
                    registry: &registry,
                    host: &host,
                    overlay_pins: &[],
                    hotset_pins: &[],
                    grandfather_pins: &[],
                    present: &absent,
                };
                let choice = PullChoice::defaults(tag, entry, &PullFlags::default());
                let items = plan_items_no_ep(&ctx, &choice, false);
                let files: Vec<&str> = items.iter().map(|item| item.file.as_str()).collect();
                assert_eq!(files, [STOCK_BASE, sidecar.file.as_str()], "{input} {arch}");
                assert_eq!(
                    download_total(&items),
                    (STOCK_BASE_SIZE + ple_bytes, 0),
                    "{input} {arch}"
                );
            }
            // The exact sums a pull reports.
            let q4 = plan_items_no_ep(
                &PullContext {
                    registry: &registry,
                    host: &host_with_arch("gfx1151"),
                    overlay_pins: &[],
                    hotset_pins: &[],
                    grandfather_pins: &[],
                    present: &absent,
                },
                &PullChoice::defaults(tag, entry, &PullFlags::default()),
                false,
            );
            assert_eq!(download_total(&q4), (100_008_416_792, 0));
        }
        let (tag, entry) = registry.model("qwen3.8:flash-next").unwrap();
        let rdna4 = plan_items_no_ep(
            &PullContext {
                registry: &registry,
                host: &host_with_arch("gfx1201"),
                overlay_pins: &[],
                hotset_pins: &[],
                grandfather_pins: &[],
                present: &absent,
            },
            &PullChoice::defaults(tag, entry, &PullFlags::default()),
            false,
        );
        assert_eq!(download_total(&rdna4), (125_288_539_672, 0));
        // The four stock MQ4 tags are one tier of the family (other rungs may exist).
        let tiers = family_tiers(&registry, "qwen3.8:flash-next");
        let mq4_tiers = tiers
            .iter()
            .filter(|tier| tier.entry.file == STOCK_BASE)
            .count();
        assert_eq!(mq4_tiers, 1);
    }

    #[test]
    fn stock_pull_plan_carries_the_split_base_overlay_and_hot_set() {
        let registry = stock_registry();
        let host = host_with_arch("gfx1151");
        let (tag, entry) = registry.model("qwen3.8:flash").unwrap();
        let ctx = PullContext {
            registry: &registry,
            host: &host,
            overlay_pins: hipfire_registry::TRUNK_OVERLAY_PINS,
            hotset_pins: hipfire_registry::EXPERT_HOTSET_PINS,
            grandfather_pins: hipfire_registry::GRANDFATHERED_MONOLITHICS,
            present: &absent,
        };
        let items = plan_items_no_ep(
            &ctx,
            &PullChoice::defaults(tag, entry, &PullFlags::default()),
            false,
        );
        let overlay = items.iter().find(|item| item.label == "overlay").unwrap();
        assert_eq!(overlay.file, format!("{STOCK_BASE}.trunk-overlay"));
        assert_eq!(overlay.size, Some(305_543_168));
        let hotset = items
            .iter()
            .find(|item| item.label == "Expert hot set")
            .unwrap();
        assert_eq!(hotset.file, format!("{STOCK_BASE}.expert-hotset"));
        assert_eq!(hotset.size, Some(217_675));
    }

    /// A registry whose one entry is the published symmetric-trunk base the HC8 pin binds to.
    fn symtrunk_registry() -> RegistryV1 {
        let pin = &hipfire_registry::HC8_SIDECAR_PINS[0];
        let value = serde_json::json!({
            "schema_version": 1,
            "generated_at": "2099-01-01T00:00:00Z",
            "models": {
                "symtrunk:test": {
                    "repo": "hipfire-models/qwen3.8-flash-next",
                    "file": "qwen3.8-flash-next-base.mq4-xts-symtrunk",
                    "size_gb": 70.9, "min_vram_gb": 72.0, "arch_id": 16, "quant": "mq4",
                    "desc": "Qwen3.8-Flash-Next MQ4 XTS symmetric-trunk base",
                    "sha256": pin.base_sha256, "size_bytes": pin.base_size
                }
            },
            "aliases": {}
        });
        RegistryV1::parse(&value.to_string(), "test").unwrap()
    }

    fn hc8_plan_labels(registry: &RegistryV1, tag: &str, arch: &str) -> Vec<PlanItem> {
        let host = host_with_arch(arch);
        let (tag, entry) = registry.model(tag).unwrap();
        let ctx = PullContext {
            registry,
            host: &host,
            overlay_pins: hipfire_registry::TRUNK_OVERLAY_PINS,
            hotset_pins: hipfire_registry::EXPERT_HOTSET_PINS,
            grandfather_pins: hipfire_registry::GRANDFATHERED_MONOLITHICS,
            present: &absent,
        };
        plan_items_no_ep(
            &ctx,
            &PullChoice::defaults(tag, entry, &PullFlags::default()),
            false,
        )
    }

    #[test]
    fn the_symtrunk_base_pull_plans_the_hc8_sidecar_beside_it() {
        let registry = symtrunk_registry();
        let pin = &hipfire_registry::HC8_SIDECAR_PINS[0];
        for arch in ["gfx1201", ""] {
            let items = hc8_plan_labels(&registry, "symtrunk:test", arch);
            let hc8 = items.iter().find(|item| item.label == "hc8").unwrap();
            assert_eq!(hc8.file, pin.file, "{arch:?}");
            assert_eq!(hc8.size, Some(pin.size_bytes), "{arch:?}");
        }
        // Not wanted where the loader never applies it.
        for arch in ["gfx1100", "gfx1151"] {
            let items = hc8_plan_labels(&registry, "symtrunk:test", arch);
            assert!(items.iter().all(|item| item.label != "hc8"), "{arch}");
        }
        assert!(hc8_sidecar_wanted(&["gfx1100".into(), "gfx1201".into()]));
        assert!(!hc8_sidecar_wanted(&["gfx1100".into()]));
    }

    #[test]
    fn the_stock_split_base_pull_plans_no_hc8_sidecar() {
        let registry = stock_registry();
        let items = hc8_plan_labels(&registry, "qwen3.8:flash", "gfx1201");
        assert!(items.iter().all(|item| item.label != "hc8"));
    }

    #[test]
    fn the_unpublished_hot_units_pin_is_not_in_any_pull_plan() {
        assert!(!hipfire_registry::HOT_UNITS_SIDECAR_PINS[0].published);
        let registry = symtrunk_registry();
        for arch in ["gfx1201", "", "gfx1100"] {
            let items = hc8_plan_labels(&registry, "symtrunk:test", arch);
            assert!(
                items.iter().all(|item| item.label != "EP hot-unit loan"),
                "{arch:?}"
            );
        }
        let stock = stock_registry();
        let items = hc8_plan_labels(&stock, "qwen3.8:flash", "gfx1201");
        assert!(items.iter().all(|item| item.label != "EP hot-unit loan"));
    }

    const GF_MONO: &str = "qwen3.8-flash-next.mq4-xts";
    const GF_LEGACY: &str = "qwen3.8-flash-next-gptq3.mq4";
    const GF_MONO_SIZE: u64 = 1000;
    const GF_MONO_SHA: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    /// A small-sized stand-in for the compiled grandfather pin of [`stock_registry`].
    const GF_PINS: &[GrandfatheredMonolithic] = &[GrandfatheredMonolithic {
        repo: "hipfire-models/qwen3.8-flash-next",
        split_file: STOCK_BASE,
        split_sha256: STOCK_BASE_SHA,
        split_size_bytes: STOCK_BASE_SIZE,
        canonical_file: GF_MONO,
        legacy_file: GF_LEGACY,
        sha256: GF_MONO_SHA,
        size_bytes: GF_MONO_SIZE,
    }];
    const GF_OVERLAY_PINS: &[TrunkOverlayPin] = &[
        TrunkOverlayPin {
            base_sha256: GF_MONO_SHA,
            base_size: GF_MONO_SIZE,
            repo: "hipfire-models/qwen3.8-flash-next",
            file: "qwen3.8-flash-next.mq4-xts.trunk-overlay",
            sha256: "2222222222222222222222222222222222222222222222222222222222222222",
            size_bytes: 300,
        },
        TrunkOverlayPin {
            base_sha256: STOCK_BASE_SHA,
            base_size: STOCK_BASE_SIZE,
            repo: "hipfire-models/qwen3.8-flash-next",
            file: "qwen3.8-flash-next-base.mq4-xts.trunk-overlay",
            sha256: "3333333333333333333333333333333333333333333333333333333333333333",
            size_bytes: 301,
        },
    ];
    const GF_HOTSET_PINS: &[ExpertHotsetPin] = &[
        ExpertHotsetPin {
            base_sha256: GF_MONO_SHA,
            base_size: GF_MONO_SIZE,
            repo: "hipfire-models/qwen3.8-flash-next",
            file: "qwen3.8-flash-next.mq4-xts.expert-hotset",
            sha256: "4444444444444444444444444444444444444444444444444444444444444444",
            size_bytes: 20,
        },
        ExpertHotsetPin {
            base_sha256: STOCK_BASE_SHA,
            base_size: STOCK_BASE_SIZE,
            repo: "hipfire-models/qwen3.8-flash-next",
            file: "qwen3.8-flash-next-base.mq4-xts.expert-hotset",
            sha256: "5555555555555555555555555555555555555555555555555555555555555555",
            size_bytes: 21,
        },
    ];

    fn gf_plan(
        arch: &str,
        present: &dyn Fn(&str, Option<u64>) -> bool,
        force: bool,
    ) -> Vec<PlanItem> {
        let registry = stock_registry();
        let host = host_with_arch(arch);
        let (tag, entry) = registry.model("qwen3.8:flash-next").unwrap();
        let ctx = PullContext {
            registry: &registry,
            host: &host,
            overlay_pins: GF_OVERLAY_PINS,
            hotset_pins: GF_HOTSET_PINS,
            grandfather_pins: GF_PINS,
            present,
        };
        plan_items_no_ep(
            &ctx,
            &PullChoice::defaults(tag, entry, &PullFlags::default()),
            force,
        )
    }

    #[test]
    fn a_local_grandfathered_monolithic_replaces_the_base_download_in_the_plan() {
        let on_disk = |files: &'static [&'static str]| {
            move |file: &str, size: Option<u64>| {
                files.contains(&file) && (size.is_none() || size == Some(GF_MONO_SIZE))
            }
        };
        // The canonical monolithic is present: no base, no q8; q4 is still a download;
        // overlay and hot set come from the monolithic's pins, beside the monolithic.
        let present = on_disk(&[GF_MONO]);
        let items = gf_plan("gfx1151", &present, false);
        let shape: Vec<(&str, &str, bool)> = items
            .iter()
            .map(|item| (item.label.as_str(), item.file.as_str(), item.download))
            .collect();
        assert_eq!(
            shape,
            [
                ("model", GF_MONO, false),
                ("ple-q4", "qwen3.8-flash-next.ple-q4", true),
                ("overlay", "qwen3.8-flash-next.mq4-xts.trunk-overlay", true),
                (
                    "Expert hot set",
                    "qwen3.8-flash-next.mq4-xts.expert-hotset",
                    true
                ),
            ]
        );
        assert_eq!(items[0].size, Some(GF_MONO_SIZE));
        assert_eq!(
            download_total(&items),
            (STOCK_Q4_SIZE + 300 + 20, 0),
            "the base is not counted"
        );
        // RDNA4 wants q8, which the monolithic embeds: nothing PLE to fetch.
        let items = gf_plan("gfx1201", &present, false);
        assert!(items.iter().all(|item| !item.label.starts_with("ple-")));
        assert!(!items[0].download);
        // The legacy spelling alone serves too.
        let legacy = on_disk(&[GF_LEGACY]);
        let items = gf_plan("gfx1151", &legacy, false);
        assert_eq!(
            (items[0].file.as_str(), items[0].download),
            (GF_LEGACY, false)
        );
        // A wrong-size monolithic is not counted as present: the normal split plan.
        let wrong = |file: &str, size: Option<u64>| file == GF_MONO && size != Some(GF_MONO_SIZE);
        let items = gf_plan("gfx1151", &wrong, false);
        assert_eq!(
            (items[0].file.as_str(), items[0].download),
            (STOCK_BASE, true)
        );
        assert_eq!(
            items
                .iter()
                .map(|item| item.file.as_str())
                .collect::<Vec<_>>(),
            [
                STOCK_BASE,
                "qwen3.8-flash-next.ple-q4",
                "qwen3.8-flash-next-base.mq4-xts.trunk-overlay",
                "qwen3.8-flash-next-base.mq4-xts.expert-hotset",
            ]
        );
        // --force pulls the split base whatever is on disk.
        let items = gf_plan("gfx1151", &present, true);
        assert_eq!(
            (items[0].file.as_str(), items[0].download),
            (STOCK_BASE, true)
        );
        assert!(items.iter().all(|item| item.download));
        // The split base wins over the monolithic when it is installed.
        let both = |file: &str, _: Option<u64>| file == STOCK_BASE || file == GF_MONO;
        let items = gf_plan("gfx1151", &both, false);
        assert_eq!(
            (items[0].file.as_str(), items[0].download),
            (STOCK_BASE, false)
        );
    }

    /// The file set of a stock Flash-Next pull per host: the trunk overlay is
    /// planned only where the loader applies it (gfx1151; or no detected GPU,
    /// where the pull cannot tell), and `HIPFIRE_QWEN4_TRUNK_OVERLAY=1` plans
    /// it everywhere.
    #[test]
    fn the_trunk_overlay_is_planned_only_where_the_loader_applies_it() {
        use hipfire_arch_qwen4::TrunkOverlayPolicy::{Auto, Off, On};
        let arches = |names: &[&str]| names.iter().map(|n| (*n).to_owned()).collect::<Vec<_>>();
        for (hosts, policy, planned) in [
            (arches(&["gfx1151"]), Auto, true),
            (arches(&["gfx1201"]), Auto, false),
            (arches(&["gfx1100"]), Auto, false),
            (arches(&["gfx1100", "gfx1201"]), Auto, false),
            (arches(&["gfx1100", "gfx1151"]), Auto, true),
            (arches(&["gfx906"]), Auto, false),
            (arches(&[]), Auto, true),
            (arches(&["gfx1201"]), On, true),
            (arches(&["gfx1151"]), Off, true),
        ] {
            assert_eq!(
                trunk_overlay_applies(policy, &hosts),
                planned,
                "{hosts:?} {policy:?}"
            );
        }
        // The plan itself, with the default policy: gfx1201 downloads the
        // base, the q8 PLE rows and the hot set, but no overlay.
        let absent = |_: &str, _: Option<u64>| false;
        let plan = |arch: &str| -> Vec<String> {
            let registry = stock_registry();
            let host = host_with_arch(arch);
            let (tag, entry) = registry.model("qwen3.8:flash").unwrap();
            let ctx = PullContext {
                registry: &registry,
                host: &host,
                overlay_pins: hipfire_registry::TRUNK_OVERLAY_PINS,
                hotset_pins: hipfire_registry::EXPERT_HOTSET_PINS,
                grandfather_pins: hipfire_registry::GRANDFATHERED_MONOLITHICS,
                present: &absent,
            };
            let items = plan_items_no_ep(
                &ctx,
                &PullChoice::defaults(tag, entry, &PullFlags::default()),
                false,
            );
            for item in &items {
                eprintln!("{arch}: {} {} {:?}", item.label, item.file, item.size);
            }
            items.into_iter().map(|item| item.label).collect()
        };
        assert!(plan("gfx1151").contains(&"overlay".to_owned()));
        for arch in ["gfx1201", "gfx1100"] {
            let labels = plan(arch);
            assert!(
                !labels.contains(&"overlay".to_owned()),
                "{arch}: {labels:?}"
            );
            assert!(labels.contains(&"Expert hot set".to_owned()), "{arch}");
        }
    }

    /// hipx (a gfx1100 next to a gfx1151): the summary lists the union of what
    /// each card defaults to; a local monolithic trims it to q4 (q8 is embedded).
    #[test]
    fn a_mixed_gpu_host_summary_lists_both_formats_and_a_monolithic_trims_it_to_q4() {
        let registry = stock_registry();
        let (tag, entry) = registry.model("qwen3.8:flash-next").unwrap();
        let hipx = HostFacts {
            gpu_arches: vec!["gfx1100".into(), "gfx1151".into()],
            ..host_with_arch("")
        };
        let plan = |present: &dyn Fn(&str, Option<u64>) -> bool| {
            plan_items_no_ep(
                &PullContext {
                    registry: &registry,
                    host: &hipx,
                    overlay_pins: GF_OVERLAY_PINS,
                    hotset_pins: GF_HOTSET_PINS,
                    grandfather_pins: GF_PINS,
                    present,
                },
                &PullChoice::defaults(tag, entry, &PullFlags::default()),
                false,
            )
        };
        let labels = |items: &[PlanItem]| -> Vec<String> {
            items.iter().map(|item| item.label.clone()).collect()
        };
        let items = plan(&absent);
        assert_eq!(
            labels(&items),
            ["model", "ple-q4", "ple-q8", "overlay", "Expert hot set"]
        );
        assert_eq!(
            download_total(&items),
            (
                STOCK_BASE_SIZE + STOCK_Q4_SIZE + STOCK_Q8_SIZE + 301 + 21,
                0
            )
        );
        let mono = |file: &str, size: Option<u64>| {
            file == GF_MONO && (size.is_none() || size == Some(GF_MONO_SIZE))
        };
        let items = plan(&mono);
        assert_eq!(
            labels(&items),
            ["model", "ple-q4", "overlay", "Expert hot set"]
        );
        assert_eq!(
            (items[0].file.as_str(), items[0].download),
            (GF_MONO, false)
        );
        assert_eq!(download_total(&items), (STOCK_Q4_SIZE + 300 + 20, 0));
        // The host line names every GPU.
        assert!(
            host_line(&hipx).contains("gfx1100+gfx1151"),
            "{}",
            host_line(&hipx)
        );
    }
}
