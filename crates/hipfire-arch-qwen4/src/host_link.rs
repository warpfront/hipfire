// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Measured host-to-device link class and the policies that read it.
//!
//! Host-resident routed experts reach the GPU over the PCIe link, so the best
//! long-prefill route depends on how wide that link really is, which no arch
//! string says (a gfx1100 in a Gen4 x16 slot and in a Gen3 x4 riser are the
//! same card).  [`host_link`] therefore takes it from a fresh `hipfire calibrate`
//! record or *measures* it, once per device at load,
//! and every consumer reads the cached value: the gfx1100 prefill chunk
//! ([`crate::gpu_forward::qwen4_prefill_chunk_requested`], which also sizes the
//! expert placement reserve) and the G2 stage
//! switch ([`expert_stage_default`], read by `Qwen4ExpertStage::new` for the
//! borrowed-donor stage and the stage `min_rows` default).  One value per
//! device per process means the loader's admission and the constructor can
//! never disagree.
//!
//! Both consumers read one predicate, [`HostLink::is_narrow_discrete`].  The
//! chunk follows it on gfx1100 only, and the placement reserve is sized from
//! that chunk, so a narrow gfx1100 link cedes resident expert layers for the
//! 8192-row chunk; the stage switch the placement charges
//! ([`crate::expert_residency::expert_stage_placement_enabled`]) still ignores
//! the link.
//!
//! # Probe semantics
//!
//! * **Unified memory** ([`rdna_compute::Gpu::is_uma`], the live
//!   `hipDeviceAttributeIntegrated` query, which answers "unified" when the
//!   query fails) is classified [`HostLink::Unified`] without any copy and
//!   without reading the calibration: there is no link to measure and the
//!   discrete-link policies never apply.
//! * **Calibration first**: on a discrete device, `hipfire calibrate`'s stored
//!   record is consulted ([`hipfire_config::calibration::lookup`], keyed by
//!   the device's live PCI address, arch, VRAM size and PCIe link plus the
//!   host CPU and RAM).  A *fresh* record carrying a usable
//!   `pinned_h2d_gbps` ([`link_from_calibration`]) supplies the link, and
//!   the copy path is initialized instead of measured (below).  A missing
//!   record, a stale one (different GPU, link, RAM, CPU or probe version) or
//!   one without a pinned rate falls through to the probe below, and the load
//!   log names which source was used.
//! * **Discrete probe** ([`probe_pinned_h2d_gbps`]): one 64 MiB
//!   `hipHostMalloc` portable pinned buffer (zero initialised, so every page
//!   is resident) is copied to a 64 MiB device buffer with `hipMemcpyAsync`
//!   on a private non-blocking stream.  One untimed warm-up copy is followed
//!   by 5 copies, each bracketed by a `hipEvent` pair; the figure is
//!   `PROBE_BYTES / min(elapsed)` in GB/s (1 GB = 10^9 bytes), the *best*
//!   sample, so contention from other processes lowers it only if every
//!   sample was slowed.  The buffers, events and stream are released before
//!   [`host_link`] returns.  `hipfire calibrate` runs this same probe, so a
//!   stored figure and a load-time figure are the same quantity.
//! * **Copy-path residue**: releasing the probe's buffers does not return
//!   every byte.  The HIP runtime keeps the state its first pinned async copy
//!   on a new non-blocking stream sets up (146 MiB of VRAM on gfx1100: in
//!   hold j1760 the qwen4 placement, which reads free VRAM right after this
//!   module, saw 24376 MiB free after the probe and 24522 MiB on a
//!   calibrated load that skipped it, and kept 8 instead of 7 expert layers
//!   resident).  The probe's 64 MiB device buffer is a plain `hipMalloc`
//!   freed by `hipFree` (no allocator cache), so the residue is that runtime
//!   setup, not the probe's size, and every load allocates it later anyway.
//!   A calibrated resolution therefore runs the probe's own sequence with
//!   one [`INIT_COPY_BYTES`] copy and no timing ([`init_copy_path`]), so
//!   free VRAM after [`host_link`], and with it the placement, is the same
//!   whichever source supplied the link.
//! * **Fail closed**: any allocation, copy or timing failure, or a
//!   non-finite or non-positive figure, is [`HostLink::Unmeasured`].  An
//!   unmeasured link is never narrow, so no default that depends on a narrow
//!   link switches on; the result is cached like a measurement, so a later
//!   call cannot flip the decision between admission and construction.
//!
//! "Narrow" is a discrete link measuring below [`NARROW_LINK_GBPS`] (16 GB/s):
//! a Gen3 x16, Gen4 x8 or x4 slot or riser.  A Gen4 x16 link (about
//! 26 GB/s) and an R9700 (gfx1201) in a Gen5 x16 slot (27.3 GB/s measured
//! through the pinned-copy probe; the link itself has about 50 GB/s of
//! headroom) are above it and keep their existing behavior.  gfx1201 does
//! not even consult the measurement: its G2 default is the arch rule that
//! predates the probe ([`expert_stage_default`]), so it behaves identically
//! whatever the probe returns.  Unified-memory devices (gfx1151 and other
//! APUs) are unaffected: they are never narrow and never probed.

use hipfire_config::calibration::{self, DeviceIdentity, Lookup};
use rdna_compute::{DType, Gpu};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

/// A discrete link measuring below this many GB/s (10^9 bytes/s) is narrow.
pub const NARROW_LINK_GBPS: f64 = 16.0;

/// Bytes of one probe copy.
const PROBE_BYTES: usize = 64 << 20;
/// Untimed copies before the timed ones (first-touch, DMA queue warm-up).
const PROBE_WARMUP_COPIES: usize = 1;
/// Timed copies; the fastest sets the figure.
const PROBE_TIMED_COPIES: usize = 5;
/// Bytes of the one untimed copy that initializes the copy path on a
/// calibrated resolution ([`init_copy_path`]).
const INIT_COPY_BYTES: usize = 4 << 10;

/// What the host-to-device path of one device looks like.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HostLink {
    /// Unified-memory device (APU): device memory is host memory, there is no
    /// link, and the discrete-link policies never apply.
    Unified,
    /// Discrete device whose measured pinned H2D copy rate is `gbps` GB/s.
    Discrete { gbps: f64 },
    /// Probe failed or returned an unusable figure. Treated as not narrow.
    Unmeasured,
}

impl HostLink {
    /// Classify a probe result. `measured_gbps` is `None` when the probe
    /// failed; non-finite and non-positive figures are failures too.
    pub fn classify(unified: bool, measured_gbps: Option<f64>) -> Self {
        if unified {
            return Self::Unified;
        }
        match measured_gbps {
            Some(gbps) if gbps.is_finite() && gbps > 0.0 => Self::Discrete { gbps },
            _ => Self::Unmeasured,
        }
    }

    /// A discrete device whose measured link is below [`NARROW_LINK_GBPS`].
    /// False for unified-memory devices and for an unmeasured link.
    pub fn is_narrow_discrete(self) -> bool {
        matches!(self, Self::Discrete { gbps } if gbps < NARROW_LINK_GBPS)
    }

    /// Short text for the load log.
    pub fn describe(self) -> String {
        match self {
            Self::Unified => "unified memory (no link)".to_string(),
            Self::Discrete { gbps } => format!(
                "discrete, {gbps:.1} GB/s H2D ({})",
                if self.is_narrow_discrete() {
                    "narrow"
                } else {
                    "not narrow"
                }
            ),
            Self::Unmeasured => "unmeasured (treated as not narrow)".to_string(),
        }
    }
}

/// Default of the G2 long-prefill expert stage switch when
/// `HIPFIRE_QWEN4_EXPERT_STAGE` is unset: the arch's
/// [`rdna_compute::expert_capabilities::stage_policy`] (on for gfx1201, the card it was
/// measured on, whatever its link) and, for every other arch, on every discrete device whose
/// measured link is narrow. Off for unified memory and for an unmeasured link. `link` is only
/// evaluated where the policy reads the link, so gfx1201 never probes.
pub fn expert_stage_default(arch: &str, link: impl FnOnce() -> HostLink) -> bool {
    rdna_compute::expert_capabilities::stage_default_enabled(arch, || link().is_narrow_discrete())
}

/// [`HostLink`] of `gpu`, resolved on first use (calibration record, else the
/// pinned H2D probe) and cached per device ordinal for the life of the
/// process (see the module docs for the sources).
pub fn host_link(gpu: &Gpu) -> HostLink {
    cached(gpu.device_id, || measure(gpu))
}

static CACHE: LazyLock<Mutex<HashMap<i32, HostLink>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Cached lookup: the first value stored for `device` wins and is returned to
/// every caller, even when two probes race. The lock is not held while
/// measuring.
fn cached(device: i32, measure: impl FnOnce() -> HostLink) -> HostLink {
    let lock = || {
        CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
    if let Some(link) = lock().get(&device).copied() {
        return link;
    }
    let link = measure();
    *lock().entry(device).or_insert(link)
}

/// Resolve `gpu`'s link ([`resolve_link`]) and log which source produced it.
/// Unified devices log nothing, as before.
fn measure(gpu: &Gpu) -> HostLink {
    let (link, source) = resolve_link(
        gpu.is_uma(),
        || calibration_lookup(gpu),
        || {
            if let Err(error) = init_copy_path(gpu) {
                eprintln!(
                    "  qwen4 host link: copy-path init failed on device {} ({error})",
                    gpu.device_id
                );
            }
        },
        || match probe_pinned_h2d_gbps(gpu) {
            Ok(gbps) => Some(gbps),
            Err(error) => {
                eprintln!(
                    "  qwen4 host link: H2D probe failed on device {} ({error}); \
                     treating the link as unmeasured",
                    gpu.device_id
                );
                None
            }
        },
    );
    if let Some(suffix) = source.suffix() {
        eprintln!(
            "  qwen4 host link: device {} {} {suffix}",
            gpu.device_id,
            link.describe()
        );
    }
    link
}

/// Where a resolved [`HostLink`] came from; [`LinkSource::suffix`] is the
/// bracketed tail of the load-log line.
#[derive(Clone, Debug, PartialEq)]
enum LinkSource {
    /// Unified memory: classified without calibration or probe.
    Unified,
    /// A fresh calibration record supplied the figure.
    Calibrated { measured_at: String },
    /// The load-time probe ran; `why` is the calibration's reason for not
    /// supplying the figure.
    Measured { why: String },
}

impl LinkSource {
    fn suffix(&self) -> Option<String> {
        match self {
            Self::Unified => None,
            Self::Calibrated { measured_at } => Some(format!("[calibrated {measured_at}]")),
            Self::Measured { why } => Some(format!("[measured; {why}]")),
        }
    }
}

/// The link decision without any device: unified memory is `Unified` and
/// neither `lookup`, `init` nor `probe` runs; otherwise a fresh calibration
/// carrying a pinned rate ([`link_from_calibration`]) wins, `init` runs in
/// place of the probe (the same copy-path setup, see the module docs) and
/// `probe` does not run; otherwise `probe` supplies the figure
/// ([`HostLink::classify`]) and `init` does not run. Exactly one of `init`
/// and `probe` runs on every discrete resolution.
fn resolve_link(
    unified: bool,
    lookup: impl FnOnce() -> Lookup,
    init: impl FnOnce(),
    probe: impl FnOnce() -> Option<f64>,
) -> (HostLink, LinkSource) {
    if unified {
        return (HostLink::Unified, LinkSource::Unified);
    }
    let lookup = lookup();
    let why = match &lookup {
        Lookup::Fresh { measured_at, .. } => match link_from_calibration(&lookup) {
            Some(link @ HostLink::Discrete { .. }) => {
                init();
                return (
                    link,
                    LinkSource::Calibrated {
                        measured_at: measured_at.clone(),
                    },
                );
            }
            _ => "calibration has no usable pinned H2D rate for this device".to_string(),
        },
        Lookup::Stale(reason) => format!("calibration stale: {reason}"),
        Lookup::Missing(reason) => reason.clone(),
    };
    (
        HostLink::classify(false, probe()),
        LinkSource::Measured { why },
    )
}

/// The link a calibration record states, when it applies to this host.
/// `Fresh` is required (a stale or missing record states nothing); a unified
/// device is [`HostLink::Unified`]; a discrete one is [`HostLink::Discrete`]
/// at the stored pinned H2D rate, and `None` when that rate is absent or not
/// a finite positive figure (the caller then probes).
pub fn link_from_calibration(lookup: &Lookup) -> Option<HostLink> {
    let Lookup::Fresh { device, .. } = lookup else {
        return None;
    };
    if device.unified {
        return Some(HostLink::Unified);
    }
    match device.pinned_h2d_gbps {
        Some(gbps) if gbps.is_finite() && gbps > 0.0 => Some(HostLink::Discrete { gbps }),
        _ => None,
    }
}

/// [`calibration::lookup`] for `gpu`, built from the same live identity
/// (PCI address, arch, VRAM total) `hipfire calibrate` records.
pub(crate) fn calibration_lookup(gpu: &Gpu) -> Lookup {
    let bdf = match gpu.hip.device_pci_bus_id(gpu.device_id) {
        Ok(bdf) => bdf,
        Err(error) => return Lookup::Missing(format!("PCI address unavailable: {error}")),
    };
    let vram = gpu
        .bind_thread()
        .ok()
        .and_then(|()| gpu.hip.get_vram_info().ok())
        .map(|(_, total)| total as u64);
    calibration::lookup(&DeviceIdentity::live(&bdf, &gpu.arch, vram))
}

/// Best-of-N pinned H2D copy rate of `gpu` in GB/s, with every resource
/// released before returning. The quantity `hipfire calibrate` stores as
/// `pinned_h2d_gbps` and [`host_link`] classifies.
pub fn probe_pinned_h2d_gbps(gpu: &Gpu) -> Result<f64, String> {
    pinned_h2d_copies(gpu, PROBE_BYTES, PROBE_TIMED_COPIES)
        .map(|best_ms| gbps_from_ms(PROBE_BYTES, best_ms))
}

/// The probe's copy sequence with one [`INIT_COPY_BYTES`] copy and nothing
/// timed: sets up the same HIP runtime copy-path state the probe leaves
/// behind (the module docs' copy-path residue) without measuring anything.
fn init_copy_path(gpu: &Gpu) -> Result<(), String> {
    pinned_h2d_copies(gpu, INIT_COPY_BYTES, 0).map(|_| ())
}

/// One pinned-copy session on a private non-blocking stream: a `bytes`
/// pinned host buffer and device buffer, [`PROBE_WARMUP_COPIES`] untimed
/// copies, then `timed` event-bracketed copies; returns the fastest timed
/// copy in milliseconds (infinite when `timed` is 0). Every resource is
/// released before returning, on every path.
fn pinned_h2d_copies(gpu: &Gpu, bytes: usize, timed: usize) -> Result<f32, String> {
    let hip = &gpu.hip;
    gpu.bind_thread().map_err(|error| error.to_string())?;
    let mut host = hip
        .host_buffer(bytes)
        .map_err(|error| format!("pinned host buffer: {error}"))?;
    let device = hip
        .malloc(bytes)
        .map_err(|error| format!("device buffer: {error}"))?;
    let stream = match hip.stream_create_non_blocking() {
        Ok(stream) => stream,
        Err(error) => {
            let _ = hip.free(device);
            return Err(format!("stream: {error}"));
        }
    };
    let session = (|| -> Result<f32, String> {
        let src = host.as_bytes_mut();
        let start = hip.event_create().map_err(|error| error.to_string())?;
        let stop = match hip.event_create() {
            Ok(stop) => stop,
            Err(error) => {
                let _ = hip.event_destroy(start);
                return Err(error.to_string());
            }
        };
        let samples = (|| -> Result<f32, String> {
            for _ in 0..PROBE_WARMUP_COPIES {
                hip.memcpy_htod_async(&device, src, &stream)
                    .map_err(|error| format!("warm-up copy: {error}"))?;
            }
            let mut best_ms = f32::INFINITY;
            for _ in 0..timed {
                hip.event_record(&start, Some(&stream))
                    .map_err(|error| error.to_string())?;
                hip.memcpy_htod_async(&device, src, &stream)
                    .map_err(|error| format!("timed copy: {error}"))?;
                hip.event_record(&stop, Some(&stream))
                    .map_err(|error| error.to_string())?;
                hip.event_synchronize(&stop)
                    .map_err(|error| error.to_string())?;
                let ms = hip
                    .event_elapsed_ms(&start, &stop)
                    .map_err(|error| error.to_string())?;
                best_ms = best_ms.min(ms);
            }
            Ok(best_ms)
        })();
        // Copies may still be queued after an error: drain before the events
        // and the buffers go away.
        let drained = hip
            .stream_synchronize(&stream)
            .map_err(|error| format!("stream drain: {error}"));
        let _ = hip.event_destroy(start);
        let _ = hip.event_destroy(stop);
        let best_ms = samples?;
        drained?;
        Ok(best_ms)
    })();
    let drained = hip.stream_synchronize(&stream);
    let _ = hip.stream_destroy(stream);
    let _ = hip.free(device);
    // `host` drops last: the stream is drained, so no copy still reads it.
    drained.map_err(|error| format!("stream drain: {error}"))?;
    session
}

/// Best-of-N rate at which a GPU kernel reads host-mapped memory over the
/// link, in GB/s of source bytes (the path spilled weights take). A
/// [`PROBE_BYTES`] host-mapped tensor is copied into a device tensor by the
/// engine's `copy_f32_buffer` kernel (one untimed launch, then
/// [`PROBE_TIMED_COPIES`] timed ones, each fenced by a device synchronize).
/// Both tensors are freed on every path. Informational (`hipfire calibrate`
/// stores it): the D2 selector takes its own live samples with
/// [`zero_copy_read_samples`].
pub fn probe_zero_copy_gbps(gpu: &mut Gpu) -> Result<f64, String> {
    let elems = PROBE_BYTES / 4;
    let src = gpu
        .upload_raw_host(&vec![0u8; PROBE_BYTES], &[PROBE_BYTES])
        .map_err(|error| format!("host-mapped buffer: {error}"))?;
    let dst = match gpu.alloc_tensor(&[elems], DType::F32) {
        Ok(dst) => dst,
        Err(error) => {
            let _ = gpu.free_tensor(src);
            return Err(format!("device buffer: {error}"));
        }
    };
    let timed = (|| -> Result<f64, String> {
        let mut best_ms = f32::INFINITY;
        for launch in 0..PROBE_WARMUP_COPIES + PROBE_TIMED_COPIES {
            let start = Instant::now();
            gpu.copy_f32_buffer(&dst, &src, elems)
                .map_err(|error| format!("zero-copy read: {error}"))?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| format!("synchronize: {error}"))?;
            if launch >= PROBE_WARMUP_COPIES {
                best_ms = best_ms.min((start.elapsed().as_secs_f64() * 1.0e3) as f32);
            }
        }
        Ok(gbps_from_ms(PROBE_BYTES, best_ms))
    })();
    // A failed launch may still be queued: drain before the buffers go away.
    let _ = gpu.hip.device_synchronize();
    let _ = gpu.free_tensor(dst);
    let _ = gpu.free_tensor(src);
    timed
}

/// Individual zero-copy read wall times plus the individual launch floors,
/// for the D2 selector ([`crate::d2_default`]), which needs the sample spread
/// and the fastest samples that the best-of figure of
/// [`probe_zero_copy_gbps`] discards.
///
/// A `bytes` host-mapped tensor is read by the engine's `copy_f32_buffer`
/// kernel into a device tensor: one untimed launch, then `N` launches each
/// timed from the host launch to `device_synchronize` returning. Each read
/// sample is wall nanoseconds for the whole read. Each floor sample is the
/// same timing of a one-element read of the same pair after its own untimed
/// launch (the cost a layer's kernels pay regardless of bytes). Both tensors
/// are freed on every path. Returns `(reads, floors)`.
pub fn zero_copy_read_samples<const N: usize>(
    gpu: &mut Gpu,
    bytes: usize,
) -> Result<([u64; N], [u64; N]), String> {
    let elems = bytes / 4;
    if elems == 0 || N == 0 {
        return Err("empty zero-copy sample request".to_string());
    }
    let src = gpu
        .upload_raw_host(&vec![0u8; bytes], &[bytes])
        .map_err(|error| format!("host-mapped buffer: {error}"))?;
    let dst = match gpu.alloc_tensor(&[elems], DType::F32) {
        Ok(dst) => dst,
        Err(error) => {
            let _ = gpu.free_tensor(src);
            return Err(format!("device buffer: {error}"));
        }
    };
    let timed = (|| -> Result<([u64; N], [u64; N]), String> {
        let launch = |gpu: &mut Gpu, n: usize| -> Result<u64, String> {
            let start = Instant::now();
            gpu.copy_f32_buffer(&dst, &src, n)
                .map_err(|error| format!("zero-copy read: {error}"))?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| format!("synchronize: {error}"))?;
            Ok(start.elapsed().as_nanos() as u64)
        };
        let mut reads = [0u64; N];
        launch(gpu, elems)?;
        for sample in &mut reads {
            *sample = launch(gpu, elems)?;
        }
        let mut floors = [0u64; N];
        launch(gpu, 1)?;
        for sample in &mut floors {
            *sample = launch(gpu, 1)?;
        }
        Ok((reads, floors))
    })();
    // A failed launch may still be queued: drain before the buffers go away.
    let _ = gpu.hip.device_synchronize();
    let _ = gpu.free_tensor(dst);
    let _ = gpu.free_tensor(src);
    timed
}

/// GB/s (10^9 bytes/s) of `bytes` moved in `ms` milliseconds; 0 for a
/// non-finite or non-positive time, which [`HostLink::classify`] rejects.
fn gbps_from_ms(bytes: usize, ms: f32) -> f64 {
    if !ms.is_finite() || ms <= 0.0 {
        return 0.0;
    }
    bytes as f64 / (f64::from(ms) * 1.0e-3) / 1.0e9
}

/// One link-dependent routing decision as the engine would take it, for
/// `hipfire calibrate --show`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteChoice {
    /// Stable key of the decision.
    pub name: &'static str,
    /// The value the engine resolves to (`on`/`off` or a row count).
    pub value: String,
    /// What decided it: a developer override, the arch default or the link.
    pub why: String,
}

/// The routing decisions that read a device's [`HostLink`] (G2 long-prefill
/// expert staging, the prefill chunk, the stage `min_rows`), resolved for
/// `arch` and `link` exactly as a model load resolves them: the same
/// predicates ([`crate::expert_residency::expert_stage_switch`],
/// [`crate::gpu_forward::prefill_chunk_requested_from`],
/// [`crate::gpu_forward::expert_stage_min_rows_from`]) and the same developer
/// overrides (`HIPFIRE_QWEN4_EXPERT_STAGE`, `HIPFIRE_PREFILL_CHUNK_ROWS`,
/// `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS`). The chunk is reported before the
/// model's `max_seq` cap, which is only known at load, and for the compact
/// forward scratch plan the shipped Flash-Next artifacts take on gfx1201
/// (`Qwen4ForwardScratchMode::FnSymGfx1201`, 16384 rows); a load that keeps
/// the `Full` plan (`HIPFIRE_QWEN4_COMPACT_SCRATCH=0`) defaults to 4096 rows.
pub fn link_route_choices(arch: &str, link: HostLink) -> Vec<RouteChoice> {
    let env = |name: &str| hipfire_config::developer_var(name).ok();
    link_route_choices_from(
        arch,
        link,
        crate::gpu_forward::Qwen4ForwardScratchMode::FnSymGfx1201,
        env(crate::gpu_forward::QWEN4_EXPERT_STAGE_ENV).as_deref(),
        env(crate::gpu_forward::QWEN4_PREFILL_CHUNK_ROWS_ENV).as_deref(),
        env(crate::gpu_forward::QWEN4_EXPERT_STAGE_MIN_ROWS_ENV).as_deref(),
    )
}

/// [`link_route_choices`] with the scratch `mode` and the three
/// developer-override values supplied.
fn link_route_choices_from(
    arch: &str,
    link: HostLink,
    mode: crate::gpu_forward::Qwen4ForwardScratchMode,
    stage_env: Option<&str>,
    chunk_env: Option<&str>,
    min_rows_env: Option<&str>,
) -> Vec<RouteChoice> {
    use crate::expert_residency::expert_stage_switch;
    use crate::gpu_forward::{
        expert_stage_min_rows_from, explicit_prefill_chunk_rows, prefill_chunk_requested_from,
        qwen4_prefill_chunk_baseline_default, StageMinRowsSource, QWEN4_EXPERT_STAGE_ENV,
        QWEN4_EXPERT_STAGE_MIN_ROWS_ENV, QWEN4_PREFILL_CHUNK_ROWS_ENV,
    };
    let note = link_note(link);

    let stage = expert_stage_switch(arch, stage_env, || link);
    let stage_why = if let Some(value @ ("1" | "0")) = stage_env {
        format!("{QWEN4_EXPERT_STAGE_ENV}={value} override")
    } else if expert_stage_switch(arch, None, || HostLink::Unmeasured) {
        "arch default: on whatever the link".to_string()
    } else {
        format!("follows the link: {note}")
    };

    let chunk = prefill_chunk_requested_from(arch, chunk_env, usize::MAX, mode, || link);
    let baseline = qwen4_prefill_chunk_baseline_default(arch, mode);
    let chunk_why = if explicit_prefill_chunk_rows(chunk_env).is_some() {
        format!("{QWEN4_PREFILL_CHUNK_ROWS_ENV} override (whole 256-row tiles)")
    } else if chunk != baseline {
        format!("{note}: raises the {baseline}-row arch baseline")
    } else {
        "arch baseline".to_string()
    };

    let (rows, source) = expert_stage_min_rows_from(arch, chunk, min_rows_env, || link);
    let mut rows_why = match source {
        StageMinRowsSource::Env => {
            format!("{QWEN4_EXPERT_STAGE_MIN_ROWS_ENV} override")
        }
        StageMinRowsSource::Default => {
            let link_free =
                expert_stage_min_rows_from(arch, chunk, None, || HostLink::Unmeasured).0;
            if rows != link_free {
                format!("{note}: fixed {rows}-row floor")
            } else {
                "arch default".to_string()
            }
        }
    };
    if !stage {
        rows_why.push_str("; unused while staging is off");
    }

    vec![
        RouteChoice {
            name: "expert_stage",
            value: if stage { "on" } else { "off" }.to_string(),
            why: stage_why,
        },
        RouteChoice {
            name: "prefill_chunk_rows",
            value: chunk.to_string(),
            why: chunk_why,
        },
        RouteChoice {
            name: "expert_stage_min_rows",
            value: rows.to_string(),
            why: rows_why,
        },
    ]
}

/// One clause of a route's reason: what the link is and whether it is narrow.
fn link_note(link: HostLink) -> String {
    match link {
        HostLink::Unified => "unified memory, no link".to_string(),
        HostLink::Discrete { gbps } if link.is_narrow_discrete() => {
            format!("{gbps:.1} GB/s discrete link is narrow (< {NARROW_LINK_GBPS} GB/s)")
        }
        HostLink::Discrete { gbps } => {
            format!("{gbps:.1} GB/s discrete link is not narrow (>= {NARROW_LINK_GBPS} GB/s)")
        }
        HostLink::Unmeasured => "link unmeasured, treated as not narrow".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_forward::Qwen4ForwardScratchMode;

    #[test]
    fn classification_is_fail_closed() {
        // Unified memory never carries a discrete figure, measured or not.
        assert_eq!(HostLink::classify(true, None), HostLink::Unified);
        assert_eq!(HostLink::classify(true, Some(3.0)), HostLink::Unified);
        assert_eq!(
            HostLink::classify(false, Some(27.3)),
            HostLink::Discrete { gbps: 27.3 }
        );
        for bad in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert_eq!(
                HostLink::classify(false, bad),
                HostLink::Unmeasured,
                "{bad:?}"
            );
        }
    }

    #[test]
    fn narrow_means_a_measured_discrete_link_below_sixteen() {
        let narrow = |unified, gbps| HostLink::classify(unified, gbps).is_narrow_discrete();
        assert!(narrow(false, Some(3.1)));
        assert!(narrow(false, Some(15.99)));
        // The threshold itself and anything above are not narrow: Gen4 x16
        // (~26 GB/s) and the Gen5 x16 R9700 (27.3 GB/s through the pinned
        // copy probe) stay on their existing behavior.
        assert!(!narrow(false, Some(16.0)));
        assert!(!narrow(false, Some(27.3)));
        assert!(!narrow(false, Some(52.0)));
        // Unified and unmeasured are never narrow.
        assert!(!narrow(true, Some(3.1)));
        assert!(!narrow(true, None));
        assert!(!narrow(false, None));
    }

    #[test]
    fn stage_default_is_gfx1201_or_narrow_discrete() {
        let link = |unified, gbps| HostLink::classify(unified, gbps);
        // gfx1201 is on whatever the link, including unmeasured and unified,
        // and never evaluates the link (no probe).
        for l in [
            link(false, Some(27.3)),
            link(false, Some(8.0)),
            link(false, None),
            link(true, None),
        ] {
            assert!(expert_stage_default("gfx1201", || l), "{l:?}");
        }
        assert!(expert_stage_default("gfx1201", || unreachable!(
            "gfx1201 probed"
        )));
        // Other arches switch on only for a measured narrow discrete link.
        for arch in ["gfx1100", "gfx1030", "gfx1151", "gfx942"] {
            assert!(
                expert_stage_default(arch, || link(false, Some(12.0))),
                "{arch}"
            );
            assert!(
                !expert_stage_default(arch, || link(false, Some(27.3))),
                "{arch}"
            );
            assert!(!expert_stage_default(arch, || link(false, None)), "{arch}");
            assert!(!expert_stage_default(arch, || link(true, None)), "{arch}");
        }
    }

    #[test]
    fn gbps_arithmetic_and_bad_times() {
        // 64 MiB in 4.194304 ms is exactly 16 GB/s (decimal): the boundary.
        let ms = PROBE_BYTES as f32 / 16.0e9 * 1.0e3;
        assert!((gbps_from_ms(PROBE_BYTES, ms) - 16.0).abs() < 0.01);
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(gbps_from_ms(PROBE_BYTES, bad), 0.0, "{bad}");
        }
    }

    #[test]
    fn cache_measures_once_per_device_and_first_value_wins() {
        // Device ordinals no real GPU uses, so parallel tests cannot collide.
        let (a, b) = (-7001, -7002);
        let calls = std::cell::Cell::new(0);
        let first = cached(a, || {
            calls.set(calls.get() + 1);
            HostLink::Discrete { gbps: 9.0 }
        });
        let again = cached(a, || {
            calls.set(calls.get() + 1);
            HostLink::Discrete { gbps: 40.0 }
        });
        assert_eq!(first, HostLink::Discrete { gbps: 9.0 });
        assert_eq!(again, first);
        assert_eq!(calls.get(), 1);
        // Another device is measured on its own; an unmeasured result is
        // cached too (no retry flips a decision after admission).
        assert_eq!(cached(b, || HostLink::Unmeasured), HostLink::Unmeasured);
        assert_eq!(
            cached(b, || HostLink::Discrete { gbps: 1.0 }),
            HostLink::Unmeasured
        );
    }

    fn fresh(unified: bool, pinned: Option<f64>) -> Lookup {
        Lookup::Fresh {
            device: calibration::DeviceCalibration {
                pci_bdf: "0000:67:00.0".to_string(),
                arch: "gfx1100".to_string(),
                vram_bytes: 25 << 30,
                unified,
                link: None,
                pinned_h2d_gbps: pinned,
                zero_copy_gbps: None,
                engine: Vec::new(),
            },
            measured_at: "2026-10-18T14:03:22Z".to_string(),
        }
    }

    #[test]
    fn link_from_calibration_needs_a_fresh_usable_record() {
        assert_eq!(
            link_from_calibration(&fresh(false, Some(7.1))),
            Some(HostLink::Discrete { gbps: 7.1 })
        );
        assert_eq!(
            link_from_calibration(&fresh(true, None)),
            Some(HostLink::Unified)
        );
        for bad in [
            None,
            Some(0.0),
            Some(-3.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert_eq!(link_from_calibration(&fresh(false, bad)), None, "{bad:?}");
        }
        assert_eq!(
            link_from_calibration(&Lookup::Stale("GPU arch changed".to_string())),
            None
        );
        assert_eq!(
            link_from_calibration(&Lookup::Missing("no calibration".to_string())),
            None
        );
    }

    #[test]
    fn fresh_calibration_supplies_the_link_and_initializes_instead_of_probing() {
        let inits = std::cell::Cell::new(0);
        let (link, source) = resolve_link(
            false,
            || fresh(false, Some(7.1)),
            || inits.set(inits.get() + 1),
            || unreachable!("probed despite a fresh calibration"),
        );
        assert_eq!(link, HostLink::Discrete { gbps: 7.1 });
        assert_eq!(
            inits.get(),
            1,
            "the copy path is set up in place of the probe"
        );
        assert_eq!(
            source.suffix().as_deref(),
            Some("[calibrated 2026-10-18T14:03:22Z]")
        );
    }

    #[test]
    fn stale_missing_or_rateless_calibration_falls_back_to_the_probe() {
        let cases = [
            (
                Lookup::Stale("GPU arch changed".to_string()),
                "[measured; calibration stale: GPU arch changed]",
            ),
            (
                Lookup::Missing("no calibration in config.toml".to_string()),
                "[measured; no calibration in config.toml]",
            ),
            (
                fresh(false, None),
                "[measured; calibration has no usable pinned H2D rate for this device]",
            ),
        ];
        for (lookup, suffix) in cases {
            let probes = std::cell::Cell::new(0);
            let (link, source) = resolve_link(
                false,
                || lookup,
                || unreachable!("initialized on the probe path ({suffix})"),
                || {
                    probes.set(probes.get() + 1);
                    Some(12.5)
                },
            );
            assert_eq!(link, HostLink::Discrete { gbps: 12.5 }, "{suffix}");
            assert_eq!(source.suffix().as_deref(), Some(suffix));
            assert_eq!(probes.get(), 1, "{suffix}");
        }
        // A failed probe is still fail-closed after a fallback.
        let (link, _) = resolve_link(false, || Lookup::Missing(String::new()), || {}, || None);
        assert_eq!(link, HostLink::Unmeasured);
    }

    #[test]
    fn unified_memory_consults_neither_calibration_nor_probe() {
        let (link, source) = resolve_link(
            true,
            || unreachable!("calibration read on unified memory"),
            || unreachable!("copy path set up on unified memory"),
            || unreachable!("probed unified memory"),
        );
        assert_eq!(link, HostLink::Unified);
        assert_eq!(source.suffix(), None);
    }

    /// Hold j1760 (hipx 7900 XTX, Flash-Next mq2-xts): 5115 MiB non-expert
    /// weights, 12622 MiB charged reserve, 840 MiB per expert layer, 48 trunk
    /// layers. The calibrated link (the stored 7.12 GB/s) and the probed one
    /// (7.1 GB/s) resolve the same placement chunk, and with the same free
    /// VRAM the same resident layer count. Free VRAM is the same because both
    /// resolutions leave the copy-path state behind (the init in place of the
    /// probe); before the init, j1760's calibrated arm saw 24522 MiB free and
    /// kept 8 layers where the probed arm (24376 MiB) kept 7.
    #[test]
    fn calibrated_and_probed_links_place_identically_given_the_same_free_vram() {
        use crate::expert_residency::auto_vram_layers;
        use crate::gpu_forward::qwen4_prefill_chunk_default;
        const MIB: u64 = 1 << 20;
        let calibrated = link_from_calibration(&fresh(false, Some(7.115_572_651_289_664)))
            .expect("a fresh discrete record states its link");
        let probed = HostLink::classify(false, Some(7.1));
        let chunk = |link: HostLink| {
            qwen4_prefill_chunk_default("gfx1100", Qwen4ForwardScratchMode::Full, || link)
        };
        assert_eq!(chunk(calibrated), 8192);
        assert_eq!(chunk(calibrated), chunk(probed));
        let layers = |free_mib: u64| {
            auto_vram_layers(free_mib * MIB, 5115 * MIB, 840 * MIB, 48, 12622 * MIB)
        };
        assert_eq!(
            layers(24376),
            7,
            "both arms after the copy-path state exists"
        );
        assert_eq!(layers(24522), 8, "j1760's calibrated arm without the init");
    }

    fn route<'a>(choices: &'a [RouteChoice], name: &str) -> &'a RouteChoice {
        choices
            .iter()
            .find(|choice| choice.name == name)
            .unwrap_or_else(|| panic!("no route {name}"))
    }

    fn routes(arch: &str, link: HostLink) -> [String; 3] {
        let choices = link_route_choices_from(
            arch,
            link,
            Qwen4ForwardScratchMode::FnSymGfx1201,
            None,
            None,
            None,
        );
        [
            route(&choices, "expert_stage").value.clone(),
            route(&choices, "prefill_chunk_rows").value.clone(),
            route(&choices, "expert_stage_min_rows").value.clone(),
        ]
    }

    #[test]
    fn narrow_link_routes_match_the_engine_defaults() {
        assert_eq!(
            routes("gfx1100", HostLink::Discrete { gbps: 7.1 }),
            ["on", "8192", "768"]
        );
        assert_eq!(
            routes("gfx1100", HostLink::Discrete { gbps: 27.0 }),
            ["off", "1536", "1536"]
        );
        assert_eq!(
            routes("gfx1151", HostLink::Unified),
            ["off", "8192", "2048"]
        );
        // The arch default is on whatever the link, and the threshold never
        // reads it.
        for link in [
            HostLink::Discrete { gbps: 7.1 },
            HostLink::Discrete { gbps: 27.0 },
            HostLink::Unmeasured,
            HostLink::Unified,
        ] {
            assert_eq!(routes("gfx1201", link), ["on", "16384", "2048"], "{link:?}");
            // The Full plan keeps the 4096-row chunk.
            let full = link_route_choices_from(
                "gfx1201",
                link,
                Qwen4ForwardScratchMode::Full,
                None,
                None,
                None,
            );
            assert_eq!(route(&full, "prefill_chunk_rows").value, "4096", "{link:?}");
        }
    }

    #[test]
    fn route_reasons_name_what_decided() {
        let narrow = HostLink::Discrete { gbps: 7.1 };
        let choices = link_route_choices_from(
            "gfx1100",
            narrow,
            Qwen4ForwardScratchMode::Full,
            None,
            None,
            None,
        );
        assert!(route(&choices, "expert_stage").why.contains("narrow"));
        assert!(route(&choices, "prefill_chunk_rows")
            .why
            .contains("1536-row"));
        assert!(route(&choices, "expert_stage_min_rows").why.contains("768"));

        let off = link_route_choices_from(
            "gfx1100",
            HostLink::Discrete { gbps: 27.0 },
            Qwen4ForwardScratchMode::Full,
            None,
            None,
            None,
        );
        assert!(route(&off, "expert_stage_min_rows")
            .why
            .ends_with("unused while staging is off"));

        let arch_on = link_route_choices_from(
            "gfx1201",
            HostLink::Unmeasured,
            Qwen4ForwardScratchMode::FnSymGfx1201,
            None,
            None,
            None,
        );
        assert!(route(&arch_on, "expert_stage")
            .why
            .starts_with("arch default"));
    }

    #[test]
    fn developer_overrides_win_in_the_report_as_in_the_engine() {
        let narrow = HostLink::Discrete { gbps: 7.1 };
        let choices = link_route_choices_from(
            "gfx1100",
            narrow,
            Qwen4ForwardScratchMode::Full,
            Some("0"),
            Some("2048"),
            Some("512"),
        );
        let stage = route(&choices, "expert_stage");
        assert_eq!(stage.value, "off");
        assert!(stage.why.contains("HIPFIRE_QWEN4_EXPERT_STAGE=0"));
        let chunk = route(&choices, "prefill_chunk_rows");
        assert_eq!(chunk.value, "2048");
        assert!(chunk.why.contains("HIPFIRE_PREFILL_CHUNK_ROWS"));
        let rows = route(&choices, "expert_stage_min_rows");
        assert_eq!(rows.value, "512");
        assert!(rows.why.contains("HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS"));
        // Anything but the strict `1`/`0` is unset, as in the engine.
        let loose = link_route_choices_from(
            "gfx1100",
            narrow,
            Qwen4ForwardScratchMode::Full,
            Some("on"),
            None,
            None,
        );
        assert_eq!(route(&loose, "expert_stage").value, "on");
        assert!(route(&loose, "expert_stage").why.contains("narrow"));
        let forced = link_route_choices_from(
            "gfx1100",
            HostLink::Unified,
            Qwen4ForwardScratchMode::Full,
            Some("1"),
            None,
            None,
        );
        assert_eq!(route(&forced, "expert_stage").value, "on");
    }
}
