// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon bundle loader (HFQ). Owner: slice A.
//!
//! Follows `hipfire-arch-lfm2moe::carrier`: the loader-owned `Carrier`
//! (arch 19 registration) calls [`load_raon_bundle`] and wraps the bundle in
//! `LoadedModel`. This crate does not depend on the loader.

use hipfire_runtime::dflash::DflashBlockPolicy;
use hipfire_runtime::loader_api::{
    topology_refusal, LoadCtx, ModelSource, SpecLoadCfg, TopologyAxis,
};
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::spec::Speculator;
use rdna_compute::Gpu;

use crate::band::RaonBand;
use crate::bundle::{RaonBundle, PREFILL_SCRATCH_ROWS};
use crate::config::RaonConfig;
use crate::forward::RaonWork;
use crate::mesh::RaonMeshModel;
use crate::state::{RaonPrefillScratch, RaonState, RaonStateDims};
use crate::weights::load_raon_weights_band;
use hipfire_runtime::hfq::HfqFile;

/// The DFlash2 draft a load asks for, or `None` for plain AR.
///
/// The draft runs under `speculation.dflash = on` (needs its draft path; fails
/// closed without one) and under `auto` when a draft path was wired (the
/// registry pairs `raon-1-mini-preview` with its pulled DFlash2 sidecar and
/// lowers `auto`; opt out with `speculation.dflash = off`). `auto` with no
/// draft is plain AR, as Raon has no MTP. A draft under an explicit `off`, a
/// second speculator beside it (native MTP, DSpark, n-gram) and any speculator
/// this arch has no producer for are refused up front, before any allocation.
pub fn dflash_draft<'a>(
    draft_path: Option<&'a str>,
    spec: SpecLoadCfg,
) -> Result<Option<&'a str>, String> {
    let other = [
        (spec.mtp == Some(true), "native MTP (speculation.mtp)"),
        (spec.dspark == Some(true), "DSpark (speculation.dspark)"),
        (
            spec.ngram_draft == Some(true),
            "n-gram draft (speculation.ngram_draft)",
        ),
    ]
    .into_iter()
    .filter_map(|(set, name)| set.then_some(name))
    .collect::<Vec<_>>();
    if !other.is_empty() {
        return Err(format!(
            "raon1_mini: unsupported speculator(s): {}; the only speculator is a DFlash2 draft \
             (--spec dflash --model-draft <draft>)",
            other.join(", ")
        ));
    }
    match (spec.dflash, draft_path) {
        (Some(true) | None, Some(draft)) => Ok(Some(draft)),
        (Some(true), None) => Err(
            "raon1_mini: DFlash (speculation.dflash = on) needs a DFlash2 draft \
             (--model-draft / params.draft)"
                .into(),
        ),
        (Some(false), Some(_)) => Err(
            "raon1_mini: a draft model conflicts with speculation.dflash = off (drop the draft \
             or use speculation.dflash = on|auto)"
                .into(),
        ),
        (Some(false) | None, None) => Ok(None),
    }
}

/// The refusal of a `topology.pp` > 1 load that asks for speculation: the
/// DFlash2 draft verifies on one device and the mesh bands run the batched
/// prefill and band-local decode only. Pure, so admission can refuse before
/// any previous model is torn down; the load repeats it as a backstop. Every
/// speculation request [`dflash_draft`] refuses stays refused for its own
/// reason first.
pub fn pp_option_refusal(
    pp: usize,
    draft_path: Option<&str>,
    spec: SpecLoadCfg,
) -> Result<(), String> {
    if pp < 2 {
        return Ok(());
    }
    // `speculation.dflash = auto` (the registry's paired draft) on a pipeline
    // load: the draft is not run and the bands decode plain AR, as `auto`
    // without a draft does; the other speculator refusals still apply.
    if spec.dflash.is_none() {
        return dflash_draft(None, spec).map(|_| ());
    }
    if dflash_draft(draft_path, spec)?.is_some() {
        return Err(format!(
            "raon1_mini: topology.pp = {pp} does not run speculation (the DFlash2 draft verifies on \
             one device); drop --spec dflash / --model-draft or set topology.pp = 1"
        ));
    }
    Ok(())
}

/// The proposal width of a Raon DFlash2 load: an explicit
/// `speculation.dflash_block`, else the `auto` width of an unbatched load
/// (Raon has no continuous batching).
fn dflash_block_for(spec: SpecLoadCfg) -> usize {
    match spec.dflash_block {
        Some(block) if !spec.dflash_block_auto => block,
        _ => DflashBlockPolicy::Auto.resolve(false, 0),
    }
}

/// The shared DFlash2 speculator (`--spec dflash --model-draft <draft>`) on a
/// loaded Raon bundle. The draft must have been made for this model (hidden
/// width and vocabulary); its stream width and extract layers are checked by
/// the shared builder against `SpecTarget for RaonBundle`.
pub fn build_raon_dflash2_speculator(
    gpu: &mut rdna_compute::Gpu,
    draft: &str,
    bundle: &mut RaonBundle,
    max_seq: usize,
    spec: SpecLoadCfg,
) -> Result<Box<dyn Speculator>, String> {
    let draft_hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(draft))
        .map_err(|error| format!("raon1_mini: DFlash2 draft '{draft}': {error}"))?;
    let config = hipfire_runtime::dflash::DflashConfig::from_hfq(&draft_hfq).ok_or_else(|| {
        format!("raon1_mini: '{draft}' is not a DFlash draft (missing `dflash` metadata)")
    })?;
    drop(draft_hfq);
    if config.hidden != bundle.config.hidden || config.vocab_size != bundle.config.vocab {
        return Err(format!(
            "raon1_mini: DFlash2 draft '{draft}' (hidden {}, vocab {}) was not made for this \
             model (hidden {}, vocab {})",
            config.hidden, config.vocab_size, bundle.config.hidden, bundle.config.vocab
        ));
    }
    let block = dflash_block_for(spec);
    let speculator =
        hipfire_runtime::dflash2_spec::build_dflash2_speculator(gpu, draft, bundle, max_seq, block)
            .map_err(|error| format!("raon1_mini: DFlash2 speculator: {error}"))?;
    eprintln!(
        "  raon1_mini DFlash2 speculator: {draft} (block {block}, target layers {:?}, \
         {} stream(s) per layer)",
        config.target_layer_ids, config.target_streams_per_layer
    );
    Ok(speculator)
}

/// Build the Raon GPU bundle from an HFQ source.
///
/// Refused up front (before any allocation): pipeline/expert parallelism,
/// a speculation request this arch cannot serve ([`dflash_draft`]), a
/// safetensors directory (the official MXFP8
/// checkpoint is quantized offline by `hipfire quantize`, never half-loaded),
/// a non-arch-19 container, and a context length outside `1..=max_position`.
/// Admission (`hipfire_loader::admission`) refuses tp and ep above one for
/// this family first; the ep check below is a backstop for callers that
/// bypass admission and shares its refusal sentence. This is the one-device
/// load: a `topology.pp` load builds its bands with [`load_raon_mesh_model`].
/// The DFlash2 speculator, when requested, is built by the caller with
/// [`build_raon_dflash2_speculator`] on the returned bundle.
pub fn load_raon_bundle(src: ModelSource, ctx: &mut LoadCtx<'_>) -> Result<RaonBundle, String> {
    let speculative = dflash_draft(ctx.draft_path, ctx.spec)?.is_some();
    let ep = hipfire_config::topology::ep()?;
    if ep > 1 {
        return Err(topology_refusal(
            "raon1_mini",
            TopologyAxis::Expert,
            ep,
            src.is_dir(),
        ));
    }
    let hfq = match src {
        ModelSource::Hfq(hfq) => hfq,
        ModelSource::Dir(_) => {
            return Err(
                "raon1_mini: a safetensors directory cannot be served directly (the official \
                 checkpoint stores routed experts as MXFP8 and needs the symmetric MQ4V2 conversion); \
                 run `hipfire quantize` on it (Raon symmetric qt44 recipe) and load the resulting .hfq"
                    .into(),
            )
        }
    };
    if hfq.arch_id != crate::ARCH_ID {
        return Err(format!(
            "raon1_mini: HFQ arch_id {} is not {}",
            hfq.arch_id,
            crate::ARCH_ID
        ));
    }
    let config = RaonConfig::from_hfq(&hfq)?;
    if ctx.max_seq == 0 || ctx.max_seq > config.max_position {
        return Err(format!(
            "raon1_mini: max_seq {} outside 1..={}",
            ctx.max_seq, config.max_position
        ));
    }
    // Resolved before any device allocation so a malformed value fails fast.
    let forced_rows = forced_chunk_rows()?;
    let band = RaonBand::whole(&config);
    assemble_raon_bundle(
        ctx.gpu,
        &hfq,
        &config,
        &band,
        ctx.max_seq,
        forced_rows,
        speculative,
    )
}

/// Build the Raon bundle of one layer band ([`RaonBand`]) on `gpu`: the same
/// load as [`load_raon_bundle`] (inventory, weights, state, chunk admission,
/// scratch, work) scoped to the band's layers. The whole band is
/// [`load_raon_bundle`]'s bundle. `chunk_rows` forces the prefill chunk
/// (`512`, `1024`, `2048` or `4096`; `None` = `HIPFIRE_RAON_PREFILL_CHUNK_ROWS`
/// or the widest rung that fits), which bands driven together must agree on.
pub fn load_raon_band_bundle(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    max_seq: usize,
    band: &RaonBand,
    chunk_rows: Option<usize>,
) -> Result<RaonBundle, String> {
    if hfq.arch_id != crate::ARCH_ID {
        return Err(format!(
            "raon1_mini: HFQ arch_id {} is not {}",
            hfq.arch_id,
            crate::ARCH_ID
        ));
    }
    let config = RaonConfig::from_hfq(hfq)?;
    if max_seq == 0 || max_seq > config.max_position {
        return Err(format!(
            "raon1_mini: max_seq {max_seq} outside 1..={}",
            config.max_position
        ));
    }
    let forced_rows = match chunk_rows {
        Some(n) if PREFILL_CHUNK_LADDER.contains(&n) => Some(n),
        Some(n) => {
            return Err(format!(
                "raon1_mini: prefill chunk rows {n}: expected 512, 1024, 2048 or 4096"
            ))
        }
        None => forced_chunk_rows()?,
    };
    // A band is never speculative: a `topology.pp` load refuses speculation
    // ([`pp_option_refusal`]), so band slots take the non-speculative mirror rule.
    assemble_raon_bundle(gpu, hfq, &config, band, max_seq, forced_rows, false)
}

/// Rows a mesh edge must carry for a Raon pipeline load reserving `max_seq`
/// rows: the widest prefill chunk any band can admit
/// (`min(PREFILL_CHUNK_LADDER[0], max_seq)`). The caller sizes
/// `Gpus::provision_mesh_edges` with
/// [`RaonMeshModel::edge_capacity_bytes`] of this many rows before
/// [`load_raon_mesh_model`].
pub fn mesh_edge_rows(max_seq: usize) -> usize {
    PREFILL_CHUNK_LADDER[0].min(max_seq).max(1)
}

/// The rung below `rows` on [`PREFILL_CHUNK_LADDER`], if any.
fn next_chunk_rung_below(rows: usize) -> Option<usize> {
    PREFILL_CHUNK_LADDER.into_iter().find(|&rung| rung < rows)
}

/// A mesh band load that failed and left nothing loaded: the error, and the
/// narrower chunk width worth retrying when a later rank refused the width
/// the earlier ranks admitted.
struct MeshLoadFailure {
    error: String,
    narrower_may_fit: Option<usize>,
}

/// Load every band of `placement` (band `r` on `gpus.devices[r]`) at one
/// shared prefill chunk width and assemble the [`RaonMeshModel`]. On error
/// every band loaded so far is freed on its own device.
fn load_raon_mesh_bands(
    hfq: &HfqFile,
    config: &RaonConfig,
    gpus: &mut Gpus,
    placement: &BandPlacement,
    max_seq: usize,
    forced: Option<usize>,
) -> Result<RaonMeshModel, MeshLoadFailure> {
    let stages = placement.stages();
    let mut bands: Vec<RaonBundle> = Vec::with_capacity(stages);
    let mut rows = forced;
    let free_loaded = |gpus: &mut Gpus, bands: Vec<RaonBundle>| {
        for (rank, band) in bands.into_iter().enumerate() {
            let gpu = &mut gpus.devices[rank];
            if gpu.bind_thread().is_ok() {
                band.free_all(gpu);
            } else {
                std::mem::forget(band);
            }
        }
    };
    for rank in 0..stages {
        let loaded = RaonBand::new(placement.band(rank), config.n_layers).and_then(|band| {
            let gpu = &mut gpus.devices[rank];
            gpu.bind_thread().map_err(|e| e.to_string())?;
            load_raon_band_bundle(hfq, gpu, max_seq, &band, rows)
        });
        match loaded {
            Ok(band) => {
                let got = band.prefill_chunk_rows();
                if rank == 0 {
                    // Rank 0's admission (auto = the widest rung that fits
                    // it) sets the width every later band must take.
                    rows = Some(got);
                } else if rows != Some(got) {
                    // A later band's scratch allocation stepped down the
                    // ladder: the bands disagree. Retry at its width.
                    let wanted = rows.unwrap_or(got);
                    bands.push(band);
                    free_loaded(gpus, bands);
                    return Err(MeshLoadFailure {
                        error: format!(
                            "raon1_mini: pipeline band {rank} of {stages} admitted {got} prefill \
                             rows where the first band admitted {wanted}"
                        ),
                        narrower_may_fit: Some(got),
                    });
                }
                bands.push(band);
            }
            Err(error) => {
                let narrower_may_fit = (rank > 0 && error.contains("cannot be admitted"))
                    .then(|| rows.and_then(next_chunk_rung_below))
                    .flatten();
                free_loaded(gpus, bands);
                return Err(MeshLoadFailure {
                    error: format!("raon1_mini: pipeline band {rank} of {stages}: {error}"),
                    narrower_may_fit,
                });
            }
        }
    }
    match RaonMeshModel::new(config, &gpus.mesh, placement, bands) {
        Ok(model) => Ok(model),
        Err((error, bands)) => {
            free_loaded(gpus, bands);
            Err(MeshLoadFailure {
                error,
                narrower_may_fit: None,
            })
        }
    }
}

/// Build the Raon pipeline mesh: every band of `placement` on its rank's
/// device (`gpus.devices[r]`, one launch stream each, `gpus.mesh` the admitted
/// mesh, edges provisioned for [`mesh_edge_rows`] rows) at one shared prefill
/// chunk width, assembled into a [`RaonMeshModel`] driven through the generic
/// mesh executor.
///
/// The width is rank 0's admission (`HIPFIRE_RAON_PREFILL_CHUNK_ROWS` or the
/// widest rung that fits it); a later rank whose free memory cannot admit
/// that width retries the whole load one rung narrower. On failure nothing is
/// left loaded: the bands are freed on their own devices and the mesh edges
/// are drained (a failed drain leaks and says so).
pub fn load_raon_mesh_model(
    src: ModelSource,
    gpus: &mut Gpus,
    placement: &BandPlacement,
    max_seq: usize,
) -> Result<RaonMeshModel, String> {
    let hfq = match src {
        ModelSource::Hfq(hfq) => hfq,
        ModelSource::Dir(_) => {
            return Err("raon1_mini: topology.pp needs a single quantized .hfq source".into())
        }
    };
    if hfq.arch_id != crate::ARCH_ID {
        return Err(format!(
            "raon1_mini: HFQ arch_id {} is not {}",
            hfq.arch_id,
            crate::ARCH_ID
        ));
    }
    let config = RaonConfig::from_hfq(&hfq)?;
    if max_seq == 0 || max_seq > config.max_position {
        return Err(format!(
            "raon1_mini: max_seq {max_seq} outside 1..={}",
            config.max_position
        ));
    }
    if gpus.devices.len() != placement.stages() {
        return Err(format!(
            "raon1_mini: {} mesh devices for {} pipeline bands",
            gpus.devices.len(),
            placement.stages()
        ));
    }
    let env_forced = forced_chunk_rows()?;
    let mut forced = env_forced;
    let failure = loop {
        match load_raon_mesh_bands(&hfq, &config, gpus, placement, max_seq, forced) {
            Ok(model) => return Ok(model),
            Err(failure) => match failure.narrower_may_fit.filter(|_| env_forced.is_none()) {
                Some(narrower) => {
                    eprintln!(
                        "raon1_mini: a later pipeline band cannot admit the prefill chunk the \
                         first admitted; retrying the load at {narrower} rows"
                    );
                    forced = Some(narrower);
                }
                None => break failure,
            },
        }
    };
    Err(match gpus.drain_mesh_edges() {
        Ok(()) => failure.error,
        Err(drain) => format!(
            "{}; mesh edge drain failed ({drain}): resources quarantined until process restart",
            failure.error
        ),
    })
}

/// Load, state, chunk admission, scratch and work of `band` of the model
/// `config` describes (the whole model's configuration; the bundle carries
/// its band view). `speculative`: the load runs a speculator (DFlash2), so the
/// slot keeps a BF16 mirror while it covers the reserved context
/// ([`crate::state::mirror_is_q8`]); otherwise the mirror is Q8 when allowed.
fn assemble_raon_bundle(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    config: &RaonConfig,
    band: &RaonBand,
    max_seq: usize,
    forced_rows: Option<usize>,
    speculative: bool,
) -> Result<RaonBundle, String> {
    let view = config.band_view(band)?;
    // Resolved before any device allocation so a malformed value fails fast.
    let mirror_forced = crate::mla::MlaPolicy::from_env()?.mirror_q8_forced;
    let weights = load_raon_weights_band(hfq, gpu, config, band)?;
    let mut dims = view.state_dims();
    // Speculative verify rows run the tiled kernel, which is faster on BF16 than on
    // Q8 operands: such a slot keeps BF16 while it covers the reserved context.
    dims.mirror_bf16_if_covered = speculative && !mirror_forced;
    let state = match RaonState::new(gpu, &dims, max_seq) {
        Ok(s) => s,
        Err(e) => {
            weights.free_gpu(gpu);
            return Err(format!("raon1_mini: RaonState::new failed: {e}"));
        }
    };
    let rows = match admit_prefill_rows(gpu, &view, &dims, &state, forced_rows) {
        Ok(rows) => rows,
        Err(e) => {
            if let Err(fe) = state.free_gpu(gpu) {
                eprintln!("raon1_mini: state free after chunk admission failure: {fe}");
            }
            weights.free_gpu(gpu);
            return Err(e);
        }
    };
    let (scratch, rows) = match alloc_scratch_descending(gpu, config, &dims, rows, state.max_seq())
    {
        Ok(s) => s,
        Err(e) => {
            if let Err(fe) = state.free_gpu(gpu) {
                eprintln!("raon1_mini: state free after scratch failure: {fe}");
            }
            weights.free_gpu(gpu);
            return Err(format!("raon1_mini: RaonPrefillScratch::new failed: {e}"));
        }
    };
    let _ = rows;
    RaonBundle::new(gpu, view, weights, state, scratch)
}

/// Verify the admitted rung by really allocating it; step down the ladder
/// only when the allocation itself fails (a foreign allocation that leaves
/// room never changes the route).
fn alloc_scratch_descending(
    gpu: &mut Gpu,
    config: &RaonConfig,
    dims: &RaonStateDims,
    rows: usize,
    max_seq: usize,
) -> Result<(RaonPrefillScratch, usize), String> {
    let base_rows = PREFILL_SCRATCH_ROWS.min(max_seq);
    let mut rungs: Vec<usize> = vec![rows];
    for r in PREFILL_CHUNK_LADDER {
        let r = r.min(max_seq);
        if r < rows && rungs.last() != Some(&r) {
            rungs.push(r);
        }
    }
    let mut last_err = String::new();
    for r in rungs {
        // The chunk's module work buffers (MoE scratch etc.) are allocated by
        // `RaonBundle::new` after this returns, where a failure would abort the
        // load. Probe them here so the rung is verified as a whole.
        let attempt = RaonPrefillScratch::new(gpu, dims, r).and_then(|s| {
            match RaonWork::new(gpu, config, r) {
                Ok(work) => {
                    work.free(gpu);
                    Ok(s)
                }
                Err(e) => {
                    let _ = s.free_gpu(gpu);
                    Err(format!("module work buffers: {e}"))
                }
            }
        });
        match attempt {
            Ok(s) => {
                if r != rows {
                    eprintln!(
                        "raon1_mini: prefill scratch allocation for {rows} rows failed ({last_err}); fell back to {r} rows"
                    );
                }
                return Ok((s, r));
            }
            Err(e) => {
                last_err = e;
                if r <= base_rows {
                    break;
                }
            }
        }
    }
    Err(last_err)
}

/// Prefill chunk widths the loader may admit, widest first. The last rung is
/// the pre-admission policy ([`PREFILL_SCRATCH_ROWS`]) and is always taken when
/// nothing wider fits.
pub const PREFILL_CHUNK_LADDER: [usize; 4] = [4096, 2048, 1024, PREFILL_SCRATCH_ROWS];

/// `HIPFIRE_RAON_PREFILL_CHUNK_ROWS`: unset / empty / `auto` -> `None` (admit
/// the widest rung that fits); `512`, `1024`, `2048` or `4096` -> that rung.
/// Anything else is a named error, never a silent fallback. Read once, at load.
fn forced_chunk_rows() -> Result<Option<usize>, String> {
    match hipfire_config::developer_var("HIPFIRE_RAON_PREFILL_CHUNK_ROWS") {
        Err(_) => Ok(None),
        Ok(raw) => parse_chunk_rows(&raw),
    }
}

fn parse_chunk_rows(raw: &str) -> Result<Option<usize>, String> {
    let v = raw.trim().to_ascii_lowercase();
    if v.is_empty() || v == "auto" {
        return Ok(None);
    }
    match v.parse::<usize>() {
        Ok(n) if PREFILL_CHUNK_LADDER.contains(&n) => Ok(Some(n)),
        _ => Err(format!(
            "HIPFIRE_RAON_PREFILL_CHUNK_ROWS={raw:?}: expected `auto`, 512, 1024, 2048 or 4096"
        )),
    }
}

/// Result of the chunk admission.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChunkAdmission {
    /// Rows of the prefill scratch and module work buffers.
    rows: usize,
    /// Scratch + work bytes of `rows`.
    extra_bytes: usize,
    /// Mirror budget (rows) projected with `extra_bytes` charged.
    mirror_rows: usize,
    /// Mirror budget projected for the 512-row policy.
    baseline_mirror_rows: usize,
}

/// Pick the prefill chunk rows (pure: every device quantity is an argument).
///
/// * `available`: bytes the mirror, scratch and work may claim together
///   ([`RaonState::mirror_available_bytes`]).
/// * `extra(rows)`: scratch + work bytes of a `rows`-row chunk.
/// * `mirror_rows(usable)`: expanded-mirror budget (rows, at most `max_seq`)
///   that `usable` bytes back.
///
/// Rungs are `min(rung, max_seq)` (a chunk never needs more rows than the
/// reserved context). Auto takes the widest rung whose bytes fit `available`
/// and whose projected mirror budget is at least the 512-row policy's for the
/// same `max_seq` (so it never costs context the 512-row policy would have
/// served, unless the whole reserved context is still covered); the 512-row
/// rung is the floor and is always admitted, exactly as before. A forced rung
/// ignores the mirror comparison but must fit `available`, else a named error.
fn select_prefill_rows(
    max_seq: usize,
    forced: Option<usize>,
    available: usize,
    extra: &mut dyn FnMut(usize) -> Result<usize, String>,
    mirror_rows: &dyn Fn(usize) -> usize,
) -> Result<ChunkAdmission, String> {
    let base_rows = PREFILL_SCRATCH_ROWS.min(max_seq);
    let base_extra = extra(base_rows)?;
    let baseline_mirror_rows = mirror_rows(available.saturating_sub(base_extra)).min(max_seq);
    let admit = |rows: usize, e: usize| ChunkAdmission {
        rows,
        extra_bytes: e,
        mirror_rows: mirror_rows(available.saturating_sub(e)).min(max_seq),
        baseline_mirror_rows,
    };
    if let Some(f) = forced {
        let rows = f.min(max_seq);
        let e = if rows == base_rows {
            base_extra
        } else {
            extra(rows)?
        };
        if rows > base_rows && e > available {
            return Err(format!(
                "raon1_mini: HIPFIRE_RAON_PREFILL_CHUNK_ROWS={f} cannot be admitted: {rows} rows need \
                 {:.0} MiB of scratch and work buffers but only {:.0} MiB are available after the \
                 weights, the latent growth to {max_seq} rows and the admission reserves",
                e as f64 / 1048576.0,
                available as f64 / 1048576.0
            ));
        }
        return Ok(admit(rows, e));
    }
    let mut last = usize::MAX;
    for rung in PREFILL_CHUNK_LADDER {
        let rows = rung.min(max_seq);
        if rows == last {
            continue;
        }
        last = rows;
        if rows <= base_rows {
            break;
        }
        let e = extra(rows)?;
        if e > available {
            continue;
        }
        let adm = admit(rows, e);
        if adm.mirror_rows >= baseline_mirror_rows {
            return Ok(adm);
        }
    }
    Ok(admit(base_rows, base_extra))
}

/// Admit the prefill chunk rows against the device's free memory right now
/// (weights and the slot's initial state exist, scratch and work do not) and
/// log the decision once.
fn admit_prefill_rows(
    gpu: &Gpu,
    config: &RaonConfig,
    dims: &RaonStateDims,
    state: &RaonState,
    forced: Option<usize>,
) -> Result<usize, String> {
    let (available, cap) = state.mirror_available_with_input(gpu)?;
    let mut extra = |rows: usize| -> Result<usize, String> {
        Ok(RaonPrefillScratch::required_bytes(dims, rows)?
            + RaonWork::required_bytes(
                config,
                rows,
                gpu.gated_delta_vector_chunked_arch_supported(),
            )?)
    };
    let adm = select_prefill_rows(state.max_seq(), forced, available, &mut extra, &|usable| {
        state.projected_mirror_budget_rows(usable)
    })?;
    eprintln!(
        "raon1_mini: admission decision: prefill chunk rows from {cap}; requested={} admitted={} (scratch+work {:.0} MiB; \
         projected expanded K/V mirror budget {} rows of {} reserved context, 512-row policy {} rows)",
        forced.map_or("auto".to_string(), |f| f.to_string()),
        adm.rows,
        adm.extra_bytes as f64 / 1048576.0,
        adm.mirror_rows,
        state.max_seq(),
        adm.baseline_mirror_rows
    );
    Ok(adm.rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    /// 1 MiB of scratch+work per row, 1 MiB of mirror per row.
    fn pick(
        max_seq: usize,
        forced: Option<usize>,
        available_mib: usize,
    ) -> Result<ChunkAdmission, String> {
        select_prefill_rows(
            max_seq,
            forced,
            available_mib * MIB,
            &mut |rows| Ok(rows * MIB),
            &|usable| (usable / MIB).min(max_seq),
        )
    }

    #[test]
    fn foreign_allocation_does_not_flip_the_route() {
        // Planned capacity ignores the instantaneous free figure: a 14 GiB foreign
        // allocation (free 20 GiB -> 6 GiB) used to flip 4096 -> 512.
        const G: usize = 1 << 30;
        let avail = |free_now: usize| {
            rdna_compute::PlannedCapacity::new(free_now, 32 * G, 8 * G, true).bytes / MIB
        };
        let before = pick(8192, None, avail(20 * G)).unwrap();
        let during = pick(8192, None, avail(6 * G)).unwrap();
        assert_eq!(before.rows, 4096);
        assert_eq!(during.rows, before.rows);
        assert_eq!(during.mirror_rows, before.mirror_rows);
    }

    #[test]
    fn auto_takes_widest_rung_when_context_still_fits() {
        // 4096 rows leave 20000-4096 = 15904 MiB of mirror, which still covers max_seq 8192.
        assert_eq!(pick(8192, None, 20000).unwrap().rows, 4096);
    }

    #[test]
    fn auto_never_costs_mirror_rows_the_baseline_would_serve() {
        // available 10000: baseline (512) leaves 9488 of max_seq 50000; 4096 would leave 5904,
        // 2048 -> 7952, 1024 -> 8976, all below the baseline's 9488 -> 512.
        let a = pick(50000, None, 10000).unwrap();
        assert_eq!(a.rows, 512);
        assert_eq!(a.baseline_mirror_rows, 9488);
    }

    #[test]
    fn auto_degrades_to_the_rung_that_covers_the_whole_context() {
        // max_seq 3000, available 5000: rung 4096 -> cap 3000 rows leaves 2000 < 3000 (baseline
        // 4488 clipped to 3000); 2048 leaves 2952 < 3000; 1024 leaves 3976 >= 3000 -> 1024.
        let a = pick(3000, None, 5000).unwrap();
        assert_eq!(a.rows, 1024);
    }

    #[test]
    fn short_context_caps_the_chunk_at_max_seq() {
        assert_eq!(pick(300, None, 20000).unwrap().rows, 300);
        assert_eq!(pick(700, None, 20000).unwrap().rows, 700);
    }

    #[test]
    fn forced_rung_must_fit_but_ignores_the_mirror_comparison() {
        assert_eq!(pick(50000, Some(4096), 10000).unwrap().rows, 4096);
        let e = pick(50000, Some(4096), 4000).unwrap_err();
        assert!(e.contains("HIPFIRE_RAON_PREFILL_CHUNK_ROWS=4096"), "{e}");
        // The 512 rung is the baseline: never an error.
        assert_eq!(pick(50000, Some(512), 100).unwrap().rows, 512);
        // A forced rung wider than the context is capped, not refused.
        assert_eq!(pick(2000, Some(4096), 5000).unwrap().rows, 2000);
    }

    #[test]
    fn chunk_rows_env_parsing() {
        assert_eq!(parse_chunk_rows(""), Ok(None));
        assert_eq!(parse_chunk_rows(" AUTO "), Ok(None));
        for n in PREFILL_CHUNK_LADDER {
            assert_eq!(parse_chunk_rows(&n.to_string()), Ok(Some(n)));
        }
        for bad in ["0", "256", "3000", "8192", "-1", "4096rows", "max"] {
            let e = parse_chunk_rows(bad).unwrap_err();
            assert!(e.contains("HIPFIRE_RAON_PREFILL_CHUNK_ROWS"), "{e}");
        }
    }

    #[test]
    fn ladder_is_descending_and_ends_at_the_baseline() {
        assert!(PREFILL_CHUNK_LADDER.windows(2).all(|w| w[0] > w[1]));
        assert_eq!(PREFILL_CHUNK_LADDER[3], PREFILL_SCRATCH_ROWS);
        assert!(PREFILL_CHUNK_LADDER[0] <= crate::state::MAX_SCRATCH_ROWS);
    }

    #[test]
    fn real_dims_scratch_grows_with_rows() {
        let d = RaonStateDims::raon();
        let b512 = RaonPrefillScratch::required_bytes(&d, 512).unwrap();
        let b4096 = RaonPrefillScratch::required_bytes(&d, 4096).unwrap();
        assert!(b4096 > 7 * b512 && b4096 < 9 * b512);
    }
}

#[cfg(test)]
mod dflash_tests {
    use super::*;

    fn cfg(f: impl FnOnce(&mut SpecLoadCfg)) -> SpecLoadCfg {
        let mut spec = SpecLoadCfg::default();
        f(&mut spec);
        spec
    }

    #[test]
    fn no_draft_and_no_dflash_is_plain_ar() {
        assert_eq!(dflash_draft(None, SpecLoadCfg::default()), Ok(None));
        // An explicit `off` / unrequested mechanism never arms a speculator.
        let off = cfg(|s| {
            s.dflash = Some(false);
            s.mtp = Some(false);
            s.dspark = Some(false);
            s.ngram_draft = Some(false);
        });
        assert_eq!(dflash_draft(None, off), Ok(None));
    }

    #[test]
    fn dflash_on_with_a_draft_runs_it() {
        let on = cfg(|s| s.dflash = Some(true));
        assert_eq!(dflash_draft(Some("d.hfq"), on), Ok(Some("d.hfq")));
    }

    #[test]
    fn auto_with_a_wired_draft_runs_it() {
        // `dflash = auto` is `None` on the wire (`parse_tri_state`).
        assert_eq!(
            dflash_draft(Some("d.mq4"), SpecLoadCfg::default()),
            Ok(Some("d.mq4"))
        );
    }

    #[test]
    fn a_draft_under_dflash_off_is_refused() {
        let off = cfg(|s| s.dflash = Some(false));
        let error = dflash_draft(Some("d.mq4"), off).unwrap_err();
        assert!(error.contains("speculation.dflash = off"), "{error}");
    }

    #[test]
    fn dflash_on_without_a_draft_is_refused() {
        let on = cfg(|s| s.dflash = Some(true));
        let error = dflash_draft(None, on).unwrap_err();
        assert!(error.contains("needs a DFlash2 draft"), "{error}");
    }

    #[test]
    fn other_speculators_are_refused_by_name() {
        for (name, spec) in [
            ("native MTP", cfg(|s| s.mtp = Some(true))),
            ("DSpark", cfg(|s| s.dspark = Some(true))),
            ("n-gram draft", cfg(|s| s.ngram_draft = Some(true))),
            (
                "native MTP",
                cfg(|s| {
                    s.dflash = Some(true);
                    s.mtp = Some(true);
                }),
            ),
        ] {
            let error = dflash_draft(Some("d.hfq"), spec).unwrap_err();
            assert!(error.contains(name), "{name}: {error}");
            assert!(error.contains("unsupported speculator"), "{error}");
        }
    }

    #[test]
    fn block_is_the_explicit_width_else_the_unbatched_auto_width() {
        let width = |dflash_block, dflash_block_auto| {
            dflash_block_for(cfg(|s| {
                s.dflash_block = dflash_block;
                s.dflash_block_auto = dflash_block_auto;
            }))
        };
        let auto = hipfire_runtime::dflash::DFLASH_BLOCK_UNBATCHED;
        assert_eq!(width(None, false), auto);
        assert_eq!(width(None, true), auto);
        assert_eq!(width(Some(8), false), 8);
        // `auto` resolved by the daemon for a batching serve must not apply:
        // Raon never batches.
        assert_eq!(width(Some(8), true), auto);
    }

    #[test]
    fn pipeline_loads_refuse_a_draft_or_dflash_before_any_device_is_touched() {
        let on = cfg(|s| s.dflash = Some(true));
        let error = pp_option_refusal(2, Some("d.hfq"), on).unwrap_err();
        assert!(error.contains("topology.pp = 2"), "{error}");
        assert!(error.contains("does not run speculation"), "{error}");
        // The single-device refusals of `dflash_draft` stay first.
        let off = cfg(|s| s.dflash = Some(false));
        let error = pp_option_refusal(2, Some("d.hfq"), off).unwrap_err();
        assert!(error.contains("speculation.dflash = off"), "{error}");
        // `auto` with the registry's paired draft loads the bands as plain AR.
        assert_eq!(pp_option_refusal(2, Some("d.hfq"), SpecLoadCfg::default()), Ok(()));
        // Plain AR and the single-device load are untouched.
        assert_eq!(pp_option_refusal(2, None, SpecLoadCfg::default()), Ok(()));
        assert_eq!(pp_option_refusal(1, Some("d.hfq"), on), Ok(()));
    }

    #[test]
    fn mesh_edges_carry_the_widest_chunk_the_context_can_use() {
        assert_eq!(mesh_edge_rows(32768), 4096);
        assert_eq!(mesh_edge_rows(4200), 4096);
        assert_eq!(mesh_edge_rows(300), 300);
        assert_eq!(next_chunk_rung_below(4096), Some(2048));
        assert_eq!(next_chunk_rung_below(1024), Some(PREFILL_SCRATCH_ROWS));
        assert_eq!(next_chunk_rung_below(PREFILL_SCRATCH_ROWS), None);
    }
}
