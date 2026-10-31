// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon bundle loader (HFQ). Owner: slice A.
//!
//! Follows `hipfire-arch-lfm2moe::carrier`: the loader-owned `Carrier`
//! (arch 19 registration) calls [`load_raon_bundle`] and wraps the bundle in
//! `LoadedModel`. This crate does not depend on the loader.

use hipfire_runtime::dflash::DflashBlockPolicy;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::spec::Speculator;
use rdna_compute::Gpu;

use crate::bundle::{RaonBundle, PREFILL_SCRATCH_ROWS};
use crate::config::RaonConfig;
use crate::forward::RaonWork;
use crate::state::{RaonPrefillScratch, RaonState, RaonStateDims};
use crate::weights::load_raon_weights;

/// The DFlash2 draft a load asks for, or `None` for plain AR.
///
/// DFlash2 is opt-in: the draft runs only under `speculation.dflash = on` and
/// needs its draft path. A draft without `dflash = on`, `dflash = on` without
/// a draft, a second speculator beside it (native MTP, DSpark, n-gram) and
/// any speculator this arch has no producer for are refused up front, before
/// any allocation.
pub fn dflash_draft<'a>(
    draft_path: Option<&'a str>,
    spec: SpecLoadCfg,
) -> Result<Option<&'a str>, String> {
    let dflash = spec.dflash == Some(true);
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
    match (dflash, draft_path) {
        (true, Some(draft)) => Ok(Some(draft)),
        (true, None) => Err(
            "raon1_mini: DFlash (speculation.dflash = on) needs a DFlash2 draft \
             (--model-draft / params.draft)"
                .into(),
        ),
        (false, Some(_)) => Err(
            "raon1_mini: a draft model needs speculation.dflash = on (--spec dflash runs a \
             DFlash2 draft)"
                .into(),
        ),
        (false, None) => Ok(None),
    }
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
/// Tensor parallelism never reaches this carrier (it has its own route).
/// The DFlash2 speculator, when requested, is built by the caller with
/// [`build_raon_dflash2_speculator`] on the returned bundle.
pub fn load_raon_bundle(src: ModelSource, ctx: &mut LoadCtx<'_>) -> Result<RaonBundle, String> {
    dflash_draft(ctx.draft_path, ctx.spec)?;
    if ctx.pp > 1 {
        return Err(format!(
            "raon1_mini: pipeline parallelism (topology.pp = {}) is unsupported",
            ctx.pp
        ));
    }
    let ep = hipfire_config::topology::ep()?;
    if ep > 1 {
        return Err(format!(
            "raon1_mini: expert parallelism (topology.ep = {ep}) is unsupported"
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
    let weights = load_raon_weights(&hfq, ctx.gpu, &config)?;
    let dims = config.state_dims();
    let state = match RaonState::new(ctx.gpu, &dims, ctx.max_seq) {
        Ok(s) => s,
        Err(e) => {
            weights.free_gpu(ctx.gpu);
            return Err(format!("raon1_mini: RaonState::new failed: {e}"));
        }
    };
    let rows = match admit_prefill_rows(ctx.gpu, &config, &dims, &state, forced_rows) {
        Ok(rows) => rows,
        Err(e) => {
            if let Err(fe) = state.free_gpu(ctx.gpu) {
                eprintln!("raon1_mini: state free after chunk admission failure: {fe}");
            }
            weights.free_gpu(ctx.gpu);
            return Err(e);
        }
    };
    let (scratch, rows) = match alloc_scratch_descending(ctx.gpu, &dims, rows, state.max_seq()) {
        Ok(s) => s,
        Err(e) => {
            if let Err(fe) = state.free_gpu(ctx.gpu) {
                eprintln!("raon1_mini: state free after scratch failure: {fe}");
            }
            weights.free_gpu(ctx.gpu);
            return Err(format!("raon1_mini: RaonPrefillScratch::new failed: {e}"));
        }
    };
    let _ = rows;
    RaonBundle::new(ctx.gpu, config, weights, state, scratch)
}

/// Verify the admitted rung by really allocating it; step down the ladder
/// only when the allocation itself fails (a foreign allocation that leaves
/// room never changes the route).
fn alloc_scratch_descending(
    gpu: &mut Gpu,
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
        match RaonPrefillScratch::new(gpu, dims, r) {
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
            + RaonWork::required_bytes(config, rows)?)
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
    fn a_draft_without_dflash_on_is_refused() {
        for dflash in [None, Some(false)] {
            let spec = cfg(|s| s.dflash = dflash);
            let error = dflash_draft(Some("d.hfq"), spec).unwrap_err();
            assert!(error.contains("speculation.dflash = on"), "{error}");
        }
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
}
