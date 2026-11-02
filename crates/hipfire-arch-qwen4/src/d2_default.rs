// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-load default of the D2 CPU split (decode and `<=4`-row verify of
//! host-placed routed experts on the host).
//!
//! The split is requested explicitly by `memory.offload_exec=cpu` for hot-set
//! tier layers (which also arms the whole-layer D1 splice on un-tiered layers
//! and flips the process-wide `hipfire_dispatch::cpu_exec_enabled()` mode).
//! This module is the *only* other way to request it, and it touches none of
//! that: the verdict is a plain `bool` for one load, consumed by the D2 layer
//! admission alone (`CpuSplitAdmission::enabled`). `memory.offload_exec` is not
//! rewritten, `cpu_exec_enabled()` and `passback_enabled()` keep their values,
//! D1 never arms from it, and no process-global state is latched (a second
//! model load decides for itself).
//!
//! # What it arms
//!
//! Every Flash-Next trunk layer that places routed experts in host memory and
//! the split can serve: a hot-set tier layer with at least one host-mapped
//! expert, an un-tiered aggregate layer whose two expert allocations are
//! host-mapped QT44/QT53 (the shipped placement), or an expert-slab layer whose
//! arena is host-mapped and whose present experts all have a CPU decoder (the
//! mq2 split's W3/E8P/MQ4 mix). The split reads those pinned bytes where they
//! are, so nothing is copied or allocated twice and placement is whatever the
//! loader resolved.
//!
//! # Selector
//!
//! In order, the first match wins:
//!
//! 1. `memory.offload_exec` explicitly set (any spelling, `pcie` included):
//!    the default does not apply. `cpu` keeps today's explicit behavior,
//!    `pcie`/`passback` keep D2 off.
//! 2. `developer.qwen4_d2_default = false` (`HIPFIRE_QWEN4_D2_DEFAULT=0`): off.
//! 3. `HIPFIRE_MOE_CPU_EXPERTS=0` (the developer kill-switch): off.
//! 4. No qualifying host split layer: off.
//! 5. Unified memory (`Gpu::is_uma`): off, and never probed (there is no link).
//! 6. `developer.qwen4_d2_default = true` (`HIPFIRE_QWEN4_D2_DEFAULT=1`): on,
//!    skipping only the cost comparison (lab and receipt runs).
//! 7. The decode-cost comparison below. A failed or invalid measurement, or
//!    one whose robust comparison does not clear the margin, is off.
//!
//! # Decode-cost comparison
//!
//! Per host layer, for one full-cold decode row (`top_k` distinct cold experts
//! of the layer's real resident codecs, `B` bytes in all):
//!
//! ```text
//! PCIe = B / measured_link_rate + launch_floor
//! D2   = real_cpu_arm           + CURRENT_COPY_TRANSPORT_HANDOFF_NS
//! D2 is selected only when  D2 <= 85% of PCIe  (at least 15% faster).
//! ```
//!
//! The comparison reads a **robust statistic of the CPU arm** and is
//! **optimistic about PCIe**: `D2` is the **median** of the CPU arm samples
//! (the upper median: `sorted[len / 2]`, so an even or half-slow set is never
//! read as fast) plus the handoff, `PCIe` is the **fastest** link sample's rate
//! plus the **minimum** launch-floor sample. A single preempted batch cannot
//! decide the load: the probe that read the slowest of five samples refused D2
//! on a 7,316 µs outlier among ~410 µs samples (worst-case D2 8,122 µs against
//! PCIe 3,691 µs, a ~5 tok/s load), where the median D2 was 1,216 µs. The
//! fix is a statistic, not a looser margin:
//!
//! * The CPU arm is sampled in a pass of [`PROBE_SAMPLES`] batches. When the
//!   pass is **dispersed** (its slowest sample is more than
//!   [`CPU_OUTLIER_PERCENT`] = 200% of its median) it is re-probed **once** and
//!   the two passes are pooled (10 samples, the upper median is the 6th
//!   smallest, so six of ten must be fast to read fast). A pass that is not
//!   dispersed is never re-probed and the re-probe never repeats.
//! * The **margin is unchanged**: D2 is chosen only when `D2 <= 85%` of the
//!   best-case PCIe, so a genuinely slow CPU (every sample slow, or a
//!   majority of them) still ends on PCIe. Only a minority of slow samples
//!   (fewer than three of five, fewer than five of ten once pooled) is
//!   ignored.
//! * The link keeps the optimistic fastest sample: a disturbed link sample can
//!   only be slow, and an implausibly fast one keeps PCIe (a lost D2 win, never
//!   a wrong D2).
//!
//! The load log reports the statistic used (`median CPU arm of N samples`),
//! whether the first pass was re-probed, the slowest sample, the single-slowest
//! D2 it would have been, and the spreads (max/min and max/median) of each
//! engine. There is no spread threshold. A zero or non-positive sample, byte
//! count or floor is no measurement and fails closed to PCIe.
//!
//! * **`real_cpu_arm`** is the production arm itself:
//!   `hipfire_dispatch::cpu_exec::moe_cpu_host_slot_outputs` (gate/up GEMV,
//!   BF16 boundaries, SwiGLU, down FWHT and GEMV on the pinned splice pool)
//!   over the **actual** resident expert spans the layer build would hand it
//!   ([`crate::gpu_forward::ColdLayerSource`]), a non-zero activation, one
//!   decode row, `top_k` distinct cold experts. One untimed warm batch (thread
//!   pool, scratch) is followed by [`PROBE_SAMPLES`] timed batches. Every
//!   batch takes *different* cold experts (round-robin across the host layers,
//!   so a sample is not served from the previous one's cache lines: the
//!   `(1 + 5) * top_k` experts are about 150 MB for Flash-Next, beyond a
//!   typical last-level cache). A model with fewer cold experts than that
//!   fails closed. The cost is a few milliseconds of CPU; no dense
//!   192 MiB weight is read.
//! * **The CPU codec and the GPU read bytes are separate inputs.** Each probe
//!   expert ([`ProbeExpert`]) carries the production
//!   [`hipfire_dispatch::cpu_exec::CpuMoeExpert`] (the CPU codec and its
//!   borrowed CPU span, what the arm multiplies) **and** the bytes the GPU
//!   reads for that expert over PCIe. PCIe is priced on the GPU bytes only;
//!   the CPU arm samples are normalized against that same GPU quantity
//!   (`DecodeMeasurement::batch_bytes`, one count per timed batch: `B` is the
//!   median batch's GPU bytes and the CPU samples are scaled to it), so batches
//!   whose experts differ compare as time per GPU byte. GPU traffic is never
//!   inferred from the CPU span length: where the CPU codec and the GPU read
//!   format differ (a planar CPU codec against an MQ4 GPU read) the two byte
//!   counts can differ. A tier or aggregate owner is one codec read in place,
//!   so the loader sets the GPU bytes to the CPU payload lengths
//!   ([`ProbeExpert::same_bytes`], valid only there). A host-mapped expert-slab
//!   arena holds a codec per expert (MQ4, W3, E8P) and the GPU decode reads each
//!   routed slot in place in that slot's own codec, so every slab slot carries
//!   its own GPU read bytes, resolved from the slot's plan format (an E8P slot
//!   1,305,600 B, an MQ4 slot 2,611,200 B for the mq2 shape) and never from the
//!   CPU span ranges, the arena size, the manifest recipe or one constant for
//!   the layer. A batch of mixed slots therefore compares time per GPU byte.
//! * **`measured_link_rate`** is [`PROBE_SAMPLES`] live zero-copy reads
//!   (`host_link::zero_copy_read_samples`: a GPU kernel reading
//!   [`LINK_SAMPLE_BYTES`] of host-mapped memory over the link, the path
//!   host-placed experts take), each timed launch to synchronize. The
//!   launch floor is [`PROBE_SAMPLES`] launch + synchronize timings of a
//!   one-element read of the same pair. A link sample includes one launch,
//!   so the PCIe figure counts the floor slightly twice (which can only keep
//!   PCIe).
//! * **[`CURRENT_COPY_TRANSPORT_HANDOFF_NS`]** (806.1 µs) is the cost of the
//!   transport D2 uses today around the CPU arm, deliberately conservative:
//!   its readiness term is the whole stream-sync wait, which also contains
//!   GPU prerequisites common to both routes. It is one named constant
//!   derived from the j1779 timeline; a mapped-result or CPU-wait transport
//!   that is measured cheaper lowers it (and nothing else) here.
//! * **PLE prewarm**: the probe does not wait for or pause the PLE sidecar
//!   page-cache prewarm (`ple_prewarm`; request-serving guards are acquired
//!   after load, not around this probe). A prewarm running during the probe
//!   slows and adds noise to the CPU arm: a minority of slow samples is
//!   absorbed by the median (and the dispersed-pass re-probe), a sustained
//!   slowdown resolves to PCIe (a lost D2 win on a contended load, never a
//!   wrong D2).
//!
//! No arch or card-name predicate exists anywhere in the comparison: a link
//! the GPU reads at 4–7 GB/s (a 7900 XTX) loses to a 0.4–0.6 ms CPU arm plus
//! the handoff, a link at 26–55 GB/s (an R9700) does not.
//!
//! # Calibration record
//!
//! The `hipfire calibrate` record **never** supplies or replaces any figure of
//! this comparison. The record's engine rates are dense, alone-and-best-of
//! figures with no sample spread; they cannot say what the production arm costs
//! on this model's experts or how noisy this host is right now. `calibrate
//! --show` therefore states that D2 is decided at load from the model's real
//! arm ([`calibration_verdict`]) and the decision function it prints with is
//! this module's [`decide`].
//!
//! The probe's cost is measured and logged (`probe … ms`, setup and sampling
//! included) and its device buffers are released before the load continues; the
//! load's free-VRAM sizing reads all happen before [`resolve`].

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use hipfire_config::memory::OffloadExec;
use hipfire_dispatch::cpu_exec::CpuMoeExpert;
use hipfire_dispatch::cpu_exec::{moe_cpu_host_slot_outputs, reserve_moe_cpu_host_slot_scratch};
use rdna_compute::{DType, Gpu};

use crate::config::Qwen4Config;
use crate::gpu_forward::{ColdLayerRefs, ColdLayerSource};
use crate::weights::Qwen4Weights;

/// Timed samples of each engine per probe pass. Odd, so the median is a
/// sample.
pub const PROBE_SAMPLES: usize = 5;

/// A CPU arm pass is *dispersed* (its slowest sample is an outlier of the
/// kind that decided a wrong PCIe load: 7,316 µs against a 410 µs median) when
/// `slowest > median * CPU_OUTLIER_PERCENT / 100`, strictly. A dispersed
/// first pass is re-probed once and the two passes are pooled.
pub const CPU_OUTLIER_PERCENT: u64 = 200;

/// Untimed warm batches of cold experts run before the timed ones.
pub const PROBE_WARM_BATCHES: usize = 1;

/// Bytes of host-mapped memory one link sample reads. A little above the
/// ~26 MB a Flash-Next layer reads, so the sample's rate is the layer's rate.
pub const LINK_SAMPLE_BYTES: usize = 32 << 20;

/// D2 is selected only when its predicted cost is at most this percentage of
/// PCIe's: at least 15% faster.
pub const D2_MAX_PCIE_PERCENT: u64 = 85;

/// Stream-sync API wall per host layer: the readiness wait of the CPU arm's
/// input. It includes GPU prerequisites a PCIe route also waits for, which is
/// why the reserve is conservative. j1779 timeline, 43 host layers, 16 tokens:
/// 21.400 ms/token = 497.7 µs/layer
/// (`xtx-decode50/PLAN.md` §0, `mq2-d2-slab-gates/hold-analysis/
/// d2-mq2-timeline-1-v4/timeline-summary.json:81-116`).
pub const READINESS_WAIT_NS: u64 = 497_700;

/// Cold H2D API entry → GPU copy start, per layer: 9.713 ms/token = 225.9 µs.
pub const POST_CPU_RETURN_NS: u64 = 225_900;

/// The cold-result copy itself (102,400 B/layer at ~5.6 GB/s): 18.2 µs.
pub const RESULT_COPY_NS: u64 = 18_200;

/// Copy end → next GPU kernel: 2.447 ms/token = 56.9 µs.
pub const RESUME_GAP_NS: u64 = 56_900;

/// Remaining pre-return join idle (accounting residual): 7.4 µs.
pub const JOIN_RESIDUAL_NS: u64 = 7_400;

/// Everything D2's current copy transport costs around the CPU arm, per
/// layer: readiness wait + post-CPU return + result copy + resume + residual
/// = 806.1 µs. One constant tied to today's transport; mapped-result or
/// CPU-wait transports measured cheaper replace the terms here.
pub const CURRENT_COPY_TRANSPORT_HANDOFF_NS: u64 =
    READINESS_WAIT_NS + POST_CPU_RETURN_NS + RESULT_COPY_NS + RESUME_GAP_NS + JOIN_RESIDUAL_NS;

/// Min / median / max of one set of nanosecond samples. The median is the
/// *upper* median (`sorted[len / 2]`: the middle sample of an odd count, the
/// slower of the two middle samples of an even count), so a half-slow set is
/// never read as fast.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleStats {
    pub min_ns: u64,
    pub median_ns: u64,
    pub max_ns: u64,
}

impl SampleStats {
    /// `None` when there is no sample or any sample is zero: an unusable
    /// timing is no timing.
    pub fn of(samples: &[u64]) -> Option<Self> {
        if samples.is_empty() || samples.contains(&0) {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        Some(Self {
            min_ns: sorted[0],
            median_ns: sorted[sorted.len() / 2],
            max_ns: sorted[sorted.len() - 1],
        })
    }

    /// `max / min`, for the log.
    pub fn spread(self) -> f64 {
        self.max_ns as f64 / self.min_ns as f64
    }

    /// `max / median`: how far the slowest sample sits above the statistic
    /// the decision reads, for the log.
    pub fn tail(self) -> f64 {
        self.max_ns as f64 / self.median_ns as f64
    }

    /// The slowest sample is an outlier: strictly more than
    /// [`CPU_OUTLIER_PERCENT`] of the median.
    pub fn dispersed(self) -> bool {
        u128::from(self.max_ns) * 100 > u128::from(self.median_ns) * u128::from(CPU_OUTLIER_PERCENT)
    }
}

/// Why no cost comparison could be made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unusable {
    /// There is no model arm to measure (`hipfire calibrate --show`).
    NoModelArm,
    /// Fewer distinct cold experts than the probe's batches need.
    ColdExperts { have: usize, need: usize },
    /// A layer's spans did not resolve or the arm failed.
    Arm,
    /// The link probe failed.
    Link,
    /// A sample, byte count or floor was zero.
    InvalidSample,
}

impl Unusable {
    fn describe(self) -> String {
        match self {
            Self::NoModelArm => {
                "decided at model load from the model's real CPU expert arm and live link \
                 samples; calibration rates alone cannot enable it"
                    .to_owned()
            }
            Self::ColdExperts { have, need } => format!(
                "off (the real-arm probe needs {need} distinct cold experts, the host layers \
                 have {have})"
            ),
            Self::Arm => "off (the real CPU arm could not be set up or run)".to_owned(),
            Self::Link => "off (the host-link probe failed)".to_owned(),
            Self::InvalidSample => "off (a probe sample was zero or unusable)".to_owned(),
        }
    }
}

/// The CPU arm samples of one re-probe pass (see [`DecodeMeasurement::cpu_retry`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuRetry {
    /// Wall ns of each timed batch, as [`DecodeMeasurement::cpu_ns`].
    pub cpu_ns: [u64; PROBE_SAMPLES],
    /// GPU read bytes of each timed batch, as [`DecodeMeasurement::batch_bytes`].
    pub batch_bytes: [u64; PROBE_SAMPLES],
}

/// The raw live samples of one decode-cost probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeMeasurement {
    /// Wall ns of the production CPU arm per timed batch of `top_k` distinct
    /// cold experts (first pass).
    pub cpu_ns: [u64; PROBE_SAMPLES],
    /// GPU read bytes (not CPU span lengths) of each timed batch, the `B` of
    /// that sample: what PCIe is priced on, and what that sample's CPU time
    /// is normalized against. Equal across samples for one codec; they differ
    /// when the host layers mix codecs, so every sample carries its own count.
    pub batch_bytes: [u64; PROBE_SAMPLES],
    /// A second CPU arm pass, taken once when the first was dispersed
    /// ([`SampleStats::dispersed`]); the decision pools both passes.
    pub cpu_retry: Option<CpuRetry>,
    /// Wall ns of each zero-copy read of [`Self::link_sample_bytes`].
    pub link_ns: [u64; PROBE_SAMPLES],
    /// Wall ns of each launch + synchronize of a one-element read.
    pub launch_floor_ns: [u64; PROBE_SAMPLES],
    /// Bytes each link sample read.
    pub link_sample_bytes: u64,
    /// Wall ns of the whole probe, setup and warm-up included.
    pub probe_wall_ns: u64,
}

/// The two predicted per-layer costs and what they were built from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeCosts {
    /// The decision's PCIe: best case, `B / fastest link sample rate + the
    /// minimum launch-floor sample`.
    pub pcie_ns: u64,
    /// The decision's D2: `median CPU arm sample (pooled over every pass) +
    /// CURRENT_COPY_TRANSPORT_HANDOFF_NS`.
    pub d2_ns: u64,
    /// Information only: the PCIe cost from the medians, and the D2 cost of the
    /// single slowest CPU arm sample (what the decision used before it read
    /// the median).
    pub pcie_median_ns: u64,
    pub d2_worst_ns: u64,
    /// Median (the decision's statistic) and slowest CPU arm ns, normalized to
    /// [`Self::layer_bytes`].
    pub cpu_arm_ns: u64,
    pub cpu_slowest_ns: u64,
    /// `max / min` and `max / median` of the pooled CPU samples (information
    /// only).
    pub cpu_spread: f64,
    pub cpu_tail: f64,
    /// CPU samples the median is over, and whether a dispersed first pass
    /// was re-probed.
    pub cpu_samples: usize,
    pub cpu_reprobed: bool,
    /// Median and fastest link sample ns.
    pub link_sample_ns: u64,
    pub link_fastest_ns: u64,
    /// `max / min` of the link samples (information only).
    pub link_spread: f64,
    /// Median and minimum launch floor ns.
    pub launch_floor_ns: u64,
    pub launch_floor_min_ns: u64,
    /// Median expert bytes of one timed batch (`B`).
    pub layer_bytes: u64,
    /// Smallest and largest batch bytes (equal for a single-codec layer set).
    pub layer_bytes_range: (u64, u64),
    pub link_sample_bytes: u64,
    pub probe_wall_ns: u64,
}

impl DecodeCosts {
    /// The robust D2 (median CPU arm sample + handoff) clears the best-case
    /// PCIe by at least 15%: `d2 <= 85% of pcie`, exact integer compare
    /// (equality selects D2).
    pub fn favors_d2(self) -> bool {
        u128::from(self.d2_ns) * 100 <= u128::from(self.pcie_ns) * u128::from(D2_MAX_PCIE_PERCENT)
    }

    /// Median CPU arm rate; bytes per ns is GB/s.
    pub fn cpu_gbps(self) -> f64 {
        self.layer_bytes as f64 / self.cpu_arm_ns as f64
    }

    /// Median and fastest link rates in GB/s.
    pub fn link_gbps(self) -> f64 {
        self.link_sample_bytes as f64 / self.link_sample_ns as f64
    }

    pub fn link_fastest_gbps(self) -> f64 {
        self.link_sample_bytes as f64 / self.link_fastest_ns as f64
    }

    fn detail(self) -> String {
        let us = |ns: u64| ns as f64 / 1e3;
        let mb = |bytes: u64| bytes as f64 / 1e6;
        let bytes = if self.layer_bytes_range.0 == self.layer_bytes_range.1 {
            format!("{:.1} MB", mb(self.layer_bytes))
        } else {
            format!(
                "{:.1} MB median, {:.1}-{:.1} MB across batches",
                mb(self.layer_bytes),
                mb(self.layer_bytes_range.0),
                mb(self.layer_bytes_range.1)
            )
        };
        let reprobe = if self.cpu_reprobed {
            "; first pass was dispersed, re-probed once and pooled"
        } else {
            ""
        };
        format!(
            "live samples (real CPU arm, zero-copy link; no stored rate): D2 {:.0} us (median \
             CPU arm of {} samples{reprobe}) vs best-case PCIe {:.0} us per full-cold decode \
             layer ({:.2}x; D2 needs <= {D2_MAX_PCIE_PERCENT}%) [single slowest sample would \
             give D2 {:.0} us; median PCIe {:.0} us]: CPU arm median {:.0} us / slowest \
             {:.0} us [spread x{:.2} max/min, x{:.2} max/median, {:.1} GB/s median over \
             {bytes} of experts] + {:.0} us copy-transport handoff; link fastest {:.1} GB/s / \
             median {:.1} GB/s [spread x{:.2}] + launch floor {:.0} us min / {:.0} us median; \
             real-arm probe {:.1} ms",
            us(self.d2_ns),
            self.cpu_samples,
            us(self.pcie_ns),
            self.d2_ns as f64 / self.pcie_ns as f64,
            us(self.d2_worst_ns),
            us(self.pcie_median_ns),
            us(self.cpu_arm_ns),
            us(self.cpu_slowest_ns),
            self.cpu_spread,
            self.cpu_tail,
            self.cpu_gbps(),
            us(CURRENT_COPY_TRANSPORT_HANDOFF_NS),
            self.link_fastest_gbps(),
            self.link_gbps(),
            self.link_spread,
            us(self.launch_floor_min_ns),
            us(self.launch_floor_ns),
            self.probe_wall_ns as f64 / 1e6,
        )
    }
}

/// The CPU samples of `passes` (`(wall ns, GPU bytes)` per timed batch)
/// pooled and scaled to the median batch size: sample `i` is
/// `ns[i] * median_bytes / bytes[i]`, so batches of different byte counts
/// (mixed codecs) compare as time per byte, and the identity when every batch
/// read the same bytes. `None` for no sample or a zero byte count.
fn normalized_cpu_ns(passes: &[(&[u64], &[u64])]) -> Option<(Vec<u64>, u64)> {
    let mut bytes: Vec<u64> = passes
        .iter()
        .flat_map(|(_, bytes)| bytes.iter().copied())
        .collect();
    if bytes.is_empty() || bytes.contains(&0) {
        return None;
    }
    let pooled_bytes = bytes.clone();
    bytes.sort_unstable();
    let median = bytes[bytes.len() / 2];
    let ns = passes.iter().flat_map(|(ns, _)| ns.iter().copied());
    let mut scaled = Vec::with_capacity(pooled_bytes.len());
    for (ns, bytes) in ns.zip(pooled_bytes) {
        let value = u128::from(ns) * u128::from(median) / u128::from(bytes);
        scaled.push(u64::try_from(value).ok()?);
    }
    Some((scaled, median))
}

/// Whether one CPU arm pass is dispersed ([`SampleStats::dispersed`]) after
/// normalization: the measurement should be re-probed once. A pass that cannot
/// be normalized is not dispersed (it is an invalid sample, which the decision
/// reports).
pub fn cpu_pass_dispersed(
    cpu_ns: &[u64; PROBE_SAMPLES],
    batch_bytes: &[u64; PROBE_SAMPLES],
) -> bool {
    normalized_cpu_ns(&[(cpu_ns, batch_bytes)])
        .and_then(|(scaled, _)| SampleStats::of(&scaled))
        .is_some_and(SampleStats::dispersed)
}

impl DecodeMeasurement {
    /// The pooled, normalized CPU samples of every pass and the median batch
    /// bytes.
    fn pooled_cpu(&self) -> Option<(Vec<u64>, u64)> {
        match &self.cpu_retry {
            None => normalized_cpu_ns(&[(&self.cpu_ns, &self.batch_bytes)]),
            Some(retry) => normalized_cpu_ns(&[
                (&self.cpu_ns, &self.batch_bytes),
                (&retry.cpu_ns, &retry.batch_bytes),
            ]),
        }
    }

    /// The predicted costs, or why the samples cannot decide: any zero is
    /// [`Unusable::InvalidSample`]. The decision costs are the **robust** D2
    /// (the median CPU arm sample over every pass + handoff: one outlier
    /// sample cannot move it, a uniformly slow arm moves all of it) and the
    /// optimistic PCIe (fastest link sample + minimum floor), both held to the
    /// 85% margin; the slowest sample, spreads and the other medians ride
    /// along for the log. There is no spread threshold. Pure.
    pub fn costs(&self) -> Result<DecodeCosts, Unusable> {
        if self.link_sample_bytes == 0 {
            return Err(Unusable::InvalidSample);
        }
        let (scaled, layer_bytes) = self.pooled_cpu().ok_or(Unusable::InvalidSample)?;
        let cpu = SampleStats::of(&scaled).ok_or(Unusable::InvalidSample)?;
        let link = SampleStats::of(&self.link_ns).ok_or(Unusable::InvalidSample)?;
        let floor = SampleStats::of(&self.launch_floor_ns).ok_or(Unusable::InvalidSample)?;
        let pcie = |link_ns: u64, floor_ns: u64| -> Option<u64> {
            let transfer =
                u128::from(layer_bytes) * u128::from(link_ns) / u128::from(self.link_sample_bytes);
            u64::try_from(transfer).ok()?.checked_add(floor_ns)
        };
        let d2 = |arm_ns: u64| arm_ns.checked_add(CURRENT_COPY_TRANSPORT_HANDOFF_NS);
        let invalid = Unusable::InvalidSample;
        let mut all_bytes = self.batch_bytes.to_vec();
        if let Some(retry) = &self.cpu_retry {
            all_bytes.extend_from_slice(&retry.batch_bytes);
        }
        Ok(DecodeCosts {
            pcie_ns: pcie(link.min_ns, floor.min_ns).ok_or(invalid)?,
            d2_ns: d2(cpu.median_ns).ok_or(invalid)?,
            pcie_median_ns: pcie(link.median_ns, floor.median_ns).ok_or(invalid)?,
            d2_worst_ns: d2(cpu.max_ns).ok_or(invalid)?,
            cpu_arm_ns: cpu.median_ns,
            cpu_slowest_ns: cpu.max_ns,
            cpu_spread: cpu.spread(),
            cpu_tail: cpu.tail(),
            cpu_samples: scaled.len(),
            cpu_reprobed: self.cpu_retry.is_some(),
            link_sample_ns: link.median_ns,
            link_fastest_ns: link.min_ns,
            link_spread: link.spread(),
            launch_floor_ns: floor.median_ns,
            launch_floor_min_ns: floor.min_ns,
            layer_bytes,
            layer_bytes_range: (
                *all_bytes.iter().min().unwrap_or(&layer_bytes),
                *all_bytes.iter().max().unwrap_or(&layer_bytes),
            ),
            link_sample_bytes: self.link_sample_bytes,
            probe_wall_ns: self.probe_wall_ns,
        })
    }
}

/// Why the D2 default is on or off for one load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum D2Default {
    /// `memory.offload_exec` is set: the default does not apply.
    Explicit(OffloadExec),
    /// `developer.qwen4_d2_default = false`.
    Disabled,
    /// `HIPFIRE_MOE_CPU_EXPERTS=0`.
    KillSwitch,
    /// No layer places experts in host memory where the split can serve them.
    NoHostSplitLayers,
    /// Unified-memory device: no link, never probed.
    Unified,
    /// `developer.qwen4_d2_default = true`: on without the cost comparison.
    Forced,
    /// No usable measurement (see [`Unusable`]).
    Unmeasured(Unusable),
    /// Measured: D2 is predicted at least 15% faster than PCIe.
    CostsFavorD2(DecodeCosts),
    /// Measured: D2 is not predicted at least 15% faster than PCIe.
    CostsFavorPcie(DecodeCosts),
}

impl D2Default {
    /// Whether the default requests the D2 split for this load. An explicit
    /// setting is not the default: it is honored by the existing admission.
    pub fn selects_d2(self) -> bool {
        matches!(self, Self::Forced | Self::CostsFavorD2(_))
    }

    /// Load-log text; `None` for the silent cases (nothing to decide, or the
    /// user decided).
    pub fn describe(self) -> Option<String> {
        match self {
            Self::Explicit(_) | Self::NoHostSplitLayers => None,
            Self::Disabled => Some("off (developer.qwen4_d2_default=false)".to_owned()),
            Self::KillSwitch => Some("off (HIPFIRE_MOE_CPU_EXPERTS=0)".to_owned()),
            Self::Unified => Some("off (unified memory: no link to beat)".to_owned()),
            Self::Forced => Some("FORCED ON (developer.qwen4_d2_default=true)".to_owned()),
            Self::Unmeasured(why) => Some(why.describe()),
            Self::CostsFavorD2(costs) => Some(format!(
                "ON ({}; set memory.offload_exec=pcie to opt out)",
                costs.detail()
            )),
            Self::CostsFavorPcie(costs) => Some(format!("off ({})", costs.detail())),
        }
    }
}

/// The selector truth table. `measure` is only evaluated when every earlier
/// rule passed, so an explicit setting, the typed `false`, the kill-switch, a
/// load without host split layers, a unified-memory device and a typed `true`
/// never run the probe.
pub fn decide(
    explicit: Option<OffloadExec>,
    forced: Option<bool>,
    kill_switch: bool,
    host_split_layers: usize,
    unified: bool,
    measure: impl FnOnce() -> Result<DecodeMeasurement, Unusable>,
) -> D2Default {
    if let Some(mode) = explicit {
        return D2Default::Explicit(mode);
    }
    if forced == Some(false) {
        return D2Default::Disabled;
    }
    if kill_switch {
        return D2Default::KillSwitch;
    }
    if host_split_layers == 0 {
        return D2Default::NoHostSplitLayers;
    }
    if unified {
        return D2Default::Unified;
    }
    if forced == Some(true) {
        return D2Default::Forced;
    }
    match measure().and_then(|measurement| measurement.costs()) {
        Err(why) => D2Default::Unmeasured(why),
        Ok(costs) if costs.favors_d2() => D2Default::CostsFavorD2(costs),
        Ok(costs) => D2Default::CostsFavorPcie(costs),
    }
}

/// What `hipfire calibrate --show` may say about a device's D2 default: the
/// same [`decide`] with no model arm to measure. A discrete device is
/// undecided (`("load", text)`), a unified one is off (`("off", text)`); a
/// stored record can never state `on`.
pub fn calibration_verdict(unified: bool) -> (&'static str, String) {
    let verdict = decide(None, None, false, 1, unified, || Err(Unusable::NoModelArm));
    let status = match verdict {
        D2Default::Unmeasured(Unusable::NoModelArm) => "load",
        _ => "off",
    };
    (status, verdict.describe().unwrap_or_default())
}

/// What the loader resolved for one trunk layer's routed experts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LayerHostFacts {
    /// `Some(any expert is host-mapped)` for a hot-set tier layer, `None` for
    /// an un-tiered layer.
    tier_any_host: Option<bool>,
    /// Un-tiered aggregate layer: both aggregate allocations are host-mapped.
    aggregate_host: bool,
    /// Un-tiered aggregate layer: gate/up QT44 and down QT53, the CPU-executed pair.
    aggregate_qt44_qt53: bool,
    /// `Some((arena host-mapped, every present expert has a CPU decoder))` for
    /// an expert-slab layer.
    slab: Option<(bool, bool)>,
    /// `qwen4.expert_keep` payload (compact, original-id owners cannot index it).
    pruned: bool,
}

impl LayerHostFacts {
    fn qualifies(self) -> bool {
        match (self.tier_any_host, self.slab) {
            (Some(any_host), _) => !self.pruned && any_host,
            // A slab arena keeps a pruned expert at its original id (an absent
            // zero-byte span), so pruning does not disqualify it.
            (None, Some((host, cpu_executed))) => host && cpu_executed,
            (None, None) => !self.pruned && self.aggregate_host && self.aggregate_qt44_qt53,
        }
    }
}

/// One qualifying host layer: its trunk index and the references that name its
/// cold experts (aggregate tensors or the slab arena).
struct HostSplitLayer<'w> {
    layer: usize,
    refs: ColdLayerRefs<'w>,
}

/// Trunk layers the split could serve that place experts in host memory.
/// Single device Flash-Next only: EP, pipeline bands and MTP-head layers never
/// arm here.
fn host_split_layers<'w>(
    weights: &'w Qwen4Weights,
    config: &Qwen4Config,
    single_scope: bool,
) -> Vec<HostSplitLayer<'w>> {
    if !single_scope || weights.ep_placement().is_some() || !config.profile.is_flash_next() {
        return Vec::new();
    }
    let arenas = match &weights.manifest.identity.recipe {
        crate::weights::Qwen4TierRecipe::FlashNextSlab { arenas, .. } => Some(arenas),
        _ => None,
    };
    let mut layers = Vec::new();
    for layer in &weights.layer_refs {
        let Some(moe) = layer.ffn.moe() else { continue };
        if layer.layer >= config.num_hidden_layers || moe.gate.name.starts_with("mtp.") {
            continue;
        }
        if let Some(arena_ref) = moe.expert_arena.as_ref() {
            let present: Vec<_> = arenas
                .and_then(|plan| plan.layer(layer.layer))
                .map(|plan| {
                    plan.experts
                        .iter()
                        .filter(|expert| !expert.is_absent())
                        .collect()
                })
                .unwrap_or_default();
            let facts = LayerHostFacts {
                slab: Some((
                    weights
                        .resident(arena_ref)
                        .is_ok_and(|arena| arena.buf.is_host_mapped()),
                    !present.is_empty()
                        && present.iter().all(|expert| {
                            let (gate_up, down) = expert.effective_format().dtypes();
                            hipfire_dispatch::cpu_exec::cpu_quant_for(gate_up).is_some()
                                && hipfire_dispatch::cpu_exec::cpu_quant_for(down).is_some()
                        }),
                )),
                ..LayerHostFacts::default()
            };
            if facts.qualifies() {
                layers.push(HostSplitLayer {
                    layer: layer.layer,
                    refs: ColdLayerRefs::Arena(arena_ref),
                });
            }
            continue;
        }
        let pair = weights
            .resident(&moe.experts_gate_up)
            .ok()
            .zip(weights.resident(&moe.experts_down).ok());
        let facts = LayerHostFacts {
            tier_any_host: weights
                .expert_tier_layer(layer.layer)
                .map(|view| view.any_host()),
            aggregate_host: pair
                .is_some_and(|(gu, dn)| gu.buf.is_host_mapped() && dn.buf.is_host_mapped()),
            aggregate_qt44_qt53: pair.is_some_and(|(gu, dn)| {
                gu.dtype == DType::MQ4G256V2 && dn.dtype == DType::MQ4G128V2
            }),
            slab: None,
            pruned: config.kept_expert_ids(layer.layer, false).is_some(),
        };
        if facts.qualifies() {
            layers.push(HostSplitLayer {
                layer: layer.layer,
                refs: ColdLayerRefs::Aggregate {
                    gate_up: &moe.experts_gate_up,
                    down: &moe.experts_down,
                },
            });
        }
    }
    layers
}

/// What one measurement is valid for: the device ordinal, the distinct
/// resident `(gate/up, down)` codec pairs of the host layers (sorted) and the
/// expert shape (`hidden`, `moe_intermediate`, `top_k`). Another model's shape
/// or codec set never reuses it.
type ArmKey = (i32, Vec<(DType, DType)>, usize, usize, usize);

static MEASURED: LazyLock<Mutex<BTreeMap<ArmKey, DecodeMeasurement>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// `measure()` once per distinct key that *succeeded*; the first stored value
/// wins and the lock is not held while measuring (the same discipline as
/// [`crate::host_link::host_link`]). A failed measurement is not stored, so a
/// later load measures again. Returns whether the value was cached.
fn cached_measurement(
    cache: &Mutex<BTreeMap<ArmKey, DecodeMeasurement>>,
    key: ArmKey,
    measure: impl FnOnce() -> Result<DecodeMeasurement, Unusable>,
) -> (Result<DecodeMeasurement, Unusable>, bool) {
    let lock = || {
        cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
    if let Some(cached) = lock().get(&key).copied() {
        return (Ok(cached), true);
    }
    let measured = measure();
    match measured {
        Ok(measurement) => (Ok(*lock().entry(key).or_insert(measurement)), false),
        Err(why) => (Err(why), false),
    }
}

/// The first `need` cold experts, taken expert-major across `layers` (expert
/// 0 of every layer, then expert 1 of every layer, ...) so consecutive picks
/// come from different layers' memory. `Err(have)` when fewer exist.
fn round_robin_pick<T>(
    layers: usize,
    experts_per_layer: usize,
    payload: impl Fn(usize, usize) -> Option<T>,
    need: usize,
) -> Result<Vec<T>, usize> {
    let mut picked = Vec::with_capacity(need);
    for expert in 0..experts_per_layer {
        for layer in 0..layers {
            if let Some(found) = payload(layer, expert) {
                picked.push(found);
                if picked.len() == need {
                    return Ok(picked);
                }
            }
        }
    }
    Err(picked.len())
}

/// A deterministic, bounded, non-zero activation row.
fn probe_activation(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761) >> 8;
            (h & 0xffff) as f32 / 32768.0 - 1.0 + 1.0 / 65536.0
        })
        .collect()
}

/// One expert of the arm probe: the production CPU payload **and**, as an
/// independent input, the bytes the GPU reads for the same expert over PCIe.
/// The two are different quantities (a CPU codec and span may differ from the
/// GPU read format), so GPU traffic is never inferred from the CPU span.
#[derive(Clone, Copy)]
pub struct ProbeExpert<'a> {
    /// What the CPU arm multiplies: the CPU codec and the borrowed CPU span.
    pub cpu: CpuMoeExpert<'a>,
    /// Bytes the GPU route reads for this expert (gate/up + down) over the link.
    pub gpu_read_bytes: u64,
}

impl<'a> ProbeExpert<'a> {
    /// The GPU reads exactly the CPU payload bytes: one codec, read in place
    /// (every host owner on this base: tier owners and host-mapped aggregates).
    pub fn same_bytes(cpu: CpuMoeExpert<'a>) -> Self {
        Self {
            gpu_read_bytes: (cpu.gate_up.len() + cpu.down.len()) as u64,
            cpu,
        }
    }
}

/// GPU read bytes of one batch (never the CPU span lengths).
fn gpu_batch_bytes(batch: &[ProbeExpert<'_>]) -> u64 {
    batch.iter().map(|expert| expert.gpu_read_bytes).sum()
}

/// Time the production CPU arm on `experts`: `PROBE_WARM_BATCHES +
/// PROBE_SAMPLES` consecutive batches of `top_k` experts (all distinct),
/// each one decode row with every slot cold, exactly the call
/// `CpuSplitExecutor::join` makes. The first batch is untimed. Each expert
/// carries its CPU codec and span (what is timed) and, independently, its GPU
/// read bytes: the wall ns and the GPU read bytes of every timed batch are
/// returned (the bytes differ between batches when the experts mix codecs).
pub fn sample_cpu_arm(
    dim: usize,
    mi: usize,
    top_k: usize,
    n_experts: usize,
    experts: &[ProbeExpert<'_>],
) -> Result<([u64; PROBE_SAMPLES], [u64; PROBE_SAMPLES]), Unusable> {
    if top_k == 0
        || top_k > n_experts
        || experts.len() != (PROBE_WARM_BATCHES + PROBE_SAMPLES) * top_k
    {
        return Err(Unusable::Arm);
    }
    let fail = |error: &dyn std::fmt::Debug| {
        eprintln!("  qwen4 D2 default: CPU arm probe failed: {error:?}");
        Unusable::Arm
    };
    reserve_moe_cpu_host_slot_scratch(dim, mi, top_k).map_err(|e| fail(&e))?;
    let x = probe_activation(dim);
    let ids: Vec<i32> = (0..top_k as i32).collect();
    let gpu_ids = vec![-1i32; top_k];
    let host_slots: Vec<u32> = (0..top_k as u32).collect();
    let mut out = vec![0.0f32; top_k * dim];
    let cpu: Vec<CpuMoeExpert<'_>> = experts.iter().map(|expert| expert.cpu).collect();
    let mut samples = [0u64; PROBE_SAMPLES];
    let mut bytes = [0u64; PROBE_SAMPLES];
    for (batch_index, (batch, probe)) in cpu
        .chunks_exact(top_k)
        .zip(experts.chunks_exact(top_k))
        .enumerate()
    {
        let start = Instant::now();
        moe_cpu_host_slot_outputs(
            dim,
            mi,
            1,
            top_k,
            n_experts,
            &ids,
            Some(&gpu_ids[..]),
            &host_slots,
            batch,
            &x,
            false,
            &mut out,
        )
        .map_err(|e| fail(&e))?;
        let ns = start.elapsed().as_nanos() as u64;
        std::hint::black_box(&out);
        if batch_index >= PROBE_WARM_BATCHES {
            samples[batch_index - PROBE_WARM_BATCHES] = ns;
            bytes[batch_index - PROBE_WARM_BATCHES] = gpu_batch_bytes(probe);
        }
    }
    Ok((samples, bytes))
}

/// The cold-expert sources of every host split layer, resolved exactly as the
/// layer build resolves them, and the sorted distinct `(gate/up, down)` codec
/// pairs they hold. Fails closed on a layer that does not resolve.
fn resolve_sources(
    gpu: &Gpu,
    weights: &Qwen4Weights,
    config: &Qwen4Config,
    layers: &[HostSplitLayer<'_>],
) -> Result<(Vec<ColdLayerSource>, Vec<(DType, DType)>), Unusable> {
    let mut sources = Vec::with_capacity(layers.len());
    for layer in layers {
        match ColdLayerSource::new(gpu, weights, config, layer.layer, layer.refs) {
            Ok(source) => sources.push(source),
            Err(why) => {
                eprintln!("  qwen4 D2 default: real-arm probe: {why}");
                return Err(Unusable::Arm);
            }
        }
    }
    if sources.is_empty() {
        return Err(Unusable::Arm);
    }
    let mut codecs: Vec<(DType, DType)> =
        sources.iter().flat_map(|source| source.codecs()).collect();
    codecs.sort_unstable();
    codecs.dedup();
    Ok((sources, codecs))
}

/// One CPU arm pass's `(wall ns, GPU bytes)` per timed batch.
type CpuPass = ([u64; PROBE_SAMPLES], [u64; PROBE_SAMPLES]);

/// The CPU arm samples of one probe: a first pass, and exactly one more pass
/// when the first was dispersed ([`cpu_pass_dispersed`], one slow outlier
/// such as a preempted batch), so a disturbed host is measured twice and the
/// decision pools both. A first pass that is not dispersed never re-probes;
/// the re-probe never repeats. A failed pass fails closed.
fn sample_cpu_with_reprobe(
    mut pass: impl FnMut() -> Result<CpuPass, Unusable>,
) -> Result<(CpuPass, Option<CpuRetry>), Unusable> {
    let first = pass()?;
    if !cpu_pass_dispersed(&first.0, &first.1) {
        return Ok((first, None));
    }
    let (cpu_ns, batch_bytes) = pass()?;
    Ok((
        first,
        Some(CpuRetry {
            cpu_ns,
            batch_bytes,
        }),
    ))
}

/// The live decode-cost probe of this model on this device: the real CPU arm
/// over the host layers' actual cold experts (re-probed once when the first
/// pass is dispersed), then live link samples. Fails closed on any setup
/// problem. `started` is when source resolution began, so the logged probe
/// wall covers every setup cost. The probe does not pause the PLE prewarm
/// (see the module docs): contention shows up as slower or wider samples; the
/// median CPU statistic ignores a minority of them, a majority of slow
/// samples resolves toward PCIe.
fn measure_decode(
    gpu: &mut Gpu,
    config: &Qwen4Config,
    sources: &[ColdLayerSource],
    started: Instant,
) -> Result<DecodeMeasurement, Unusable> {
    let first = sources.first().ok_or(Unusable::Arm)?;
    let top_k = config.num_experts_per_tok;
    let need = (PROBE_WARM_BATCHES + PROBE_SAMPLES) * top_k;
    let picked = round_robin_pick(
        sources.len(),
        first.experts(),
        // Each source resolves its slot's CPU payload and its GPU read bytes
        // (the CPU length for a one-codec owner, the slot's own plan format
        // for a slab).
        |layer, expert| sources[layer].cold_probe(expert),
        need,
    )
    .map_err(|have| Unusable::ColdExperts { have, need })?;
    let ((cpu_ns, batch_bytes), cpu_retry) = sample_cpu_with_reprobe(|| {
        sample_cpu_arm(
            config.hidden_size,
            config.moe_intermediate_size,
            top_k,
            config.num_experts,
            &picked,
        )
    })?;
    if cpu_retry.is_some() {
        eprintln!(
            "  qwen4 D2 default: first CPU arm pass was dispersed (slowest > {CPU_OUTLIER_PERCENT}% \
             of its median); re-probed once"
        );
    }
    drop(picked);
    let (link_ns, launch_floor_ns) =
        crate::host_link::zero_copy_read_samples::<PROBE_SAMPLES>(gpu, LINK_SAMPLE_BYTES).map_err(
            |error| {
                eprintln!(
                    "  qwen4 D2 default: link probe failed on device {}: {error}",
                    gpu.device_id
                );
                Unusable::Link
            },
        )?;
    Ok(DecodeMeasurement {
        cpu_ns,
        batch_bytes,
        cpu_retry,
        link_ns,
        launch_floor_ns,
        link_sample_bytes: LINK_SAMPLE_BYTES as u64,
        probe_wall_ns: started.elapsed().as_nanos() as u64,
    })
}

/// The measurement for this device and model shape: the cached one from an
/// earlier load of the same shape in this process, else a live probe.
fn measured_decode(
    gpu: &mut Gpu,
    weights: &Qwen4Weights,
    config: &Qwen4Config,
    layers: &[HostSplitLayer<'_>],
) -> Result<DecodeMeasurement, Unusable> {
    let started = Instant::now();
    let (sources, codecs) = resolve_sources(gpu, weights, config, layers)?;
    let key: ArmKey = (
        gpu.device_id,
        codecs,
        config.hidden_size,
        config.moe_intermediate_size,
        config.num_experts_per_tok,
    );
    let (result, cached) = cached_measurement(&MEASURED, key.clone(), || {
        measure_decode(gpu, config, &sources, started)
    });
    if cached {
        eprintln!(
            "  qwen4 D2 default: reusing this process's decode-cost measurement (device {}, \
             codecs {:?})",
            key.0, key.1
        );
    }
    result
}

/// Resolve the D2 default for one load and log it. Call once, before the MoE
/// layer runtimes are built (the layer admission reads the result), and while
/// no stage or scratch buffer competes for the probe's host memory. Returns
/// whether the default requests D2.
pub(crate) fn resolve(
    gpu: &mut Gpu,
    weights: &Qwen4Weights,
    config: &Qwen4Config,
    single_scope: bool,
) -> bool {
    let layers = host_split_layers(weights, config, single_scope);
    let unified = gpu.is_uma();
    let verdict = decide(
        hipfire_config::memory::offload_exec_explicit(),
        hipfire_config::memory::qwen4_d2_default(),
        hipfire_config::developer_var("HIPFIRE_MOE_CPU_EXPERTS")
            .ok()
            .as_deref()
            == Some("0"),
        layers.len(),
        unified,
        || measured_decode(gpu, weights, config, &layers),
    );
    if let Some(line) = verdict.describe() {
        eprintln!(
            "  qwen4 D2 default ({} host split layers): {line}",
            layers.len()
        );
    }
    verdict.selects_d2()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const FLOOR_NS: u64 = 20_000;

    /// Realistic Flash-Next layer bytes: `top_k = 10` experts of QT44 gate/up
    /// (`2 * 640` rows at `k = 2560`) plus QT53 down (`2560` rows at `k = 640`).
    fn flash_next_layer_bytes() -> u64 {
        use crate::projection::row_stride;
        let gate_up = 2 * 640 * row_stride(DType::MQ4G256V2, 2560);
        let down = 2560 * row_stride(DType::MQ4G128V2, 640);
        10 * (gate_up + down) as u64
    }

    /// Per-sample ns of the CPU arm and the link read at a steady GB/s (a
    /// GB/s is one byte per ns).
    fn arm_ns(cpu_gbps: f64) -> u64 {
        (flash_next_layer_bytes() as f64 / cpu_gbps) as u64
    }

    fn link_ns(link_gbps: f64) -> u64 {
        (LINK_SAMPLE_BYTES as f64 / link_gbps) as u64
    }

    /// A measurement over explicit CPU and link samples, one layer's bytes
    /// per batch and the steady floor.
    fn sampled(cpu_ns: [u64; PROBE_SAMPLES], link_ns: [u64; PROBE_SAMPLES]) -> DecodeMeasurement {
        DecodeMeasurement {
            cpu_ns,
            batch_bytes: [flash_next_layer_bytes(); PROBE_SAMPLES],
            cpu_retry: None,
            link_ns,
            launch_floor_ns: [FLOOR_NS; PROBE_SAMPLES],
            link_sample_bytes: LINK_SAMPLE_BYTES as u64,
            probe_wall_ns: 25_000_000,
        }
    }

    /// Steady samples (spread 1) from a CPU-arm and link rate in GB/s.
    fn from_rates(cpu_gbps: f64, link_gbps: f64) -> DecodeMeasurement {
        sampled(
            [arm_ns(cpu_gbps); PROBE_SAMPLES],
            [link_ns(link_gbps); PROBE_SAMPLES],
        )
    }

    /// `samples` scaled by per-sample multipliers.
    fn scaled(samples: [u64; PROBE_SAMPLES], by: [f64; PROBE_SAMPLES]) -> [u64; PROBE_SAMPLES] {
        let mut out = samples;
        for (sample, factor) in out.iter_mut().zip(by) {
            *sample = (*sample as f64 * factor) as u64;
        }
        out
    }

    /// Whether `decide` (unset, nothing else gating) arms D2 on `measurement`.
    fn arms(measurement: DecodeMeasurement) -> bool {
        run(None, None, false, 43, false, Ok(measurement))
            .0
            .selects_d2()
    }

    /// Run [`decide`] and report whether the probe closure ran.
    fn run(
        explicit: Option<OffloadExec>,
        forced: Option<bool>,
        kill_switch: bool,
        layers: usize,
        unified: bool,
        measured: Result<DecodeMeasurement, Unusable>,
    ) -> (D2Default, bool) {
        let probed = Cell::new(false);
        let verdict = decide(explicit, forced, kill_switch, layers, unified, || {
            probed.set(true);
            measured
        });
        (verdict, probed.get())
    }

    fn xtx() -> Result<DecodeMeasurement, Unusable> {
        Ok(from_rates(55.0, 7.1))
    }

    fn r9700() -> Result<DecodeMeasurement, Unusable> {
        Ok(from_rates(100.0, 40.0))
    }

    #[test]
    fn reserve_is_the_sum_of_the_named_j1779_terms() {
        assert_eq!(CURRENT_COPY_TRANSPORT_HANDOFF_NS, 806_100);
        assert_eq!(READINESS_WAIT_NS, 497_700);
        assert_eq!(POST_CPU_RETURN_NS, 225_900);
        assert_eq!(RESULT_COPY_NS, 18_200);
        assert_eq!(RESUME_GAP_NS, 56_900);
        assert_eq!(JOIN_RESIDUAL_NS, 7_400);
    }

    #[test]
    fn explicit_setting_wins_both_ways_and_never_probes() {
        for mode in [OffloadExec::Pcie, OffloadExec::Cpu, OffloadExec::Passback] {
            for forced in [None, Some(true), Some(false)] {
                for measured in [xtx(), r9700(), Err(Unusable::Arm)] {
                    let (verdict, probed) = run(Some(mode), forced, false, 40, false, measured);
                    assert_eq!(verdict, D2Default::Explicit(mode));
                    assert!(
                        !verdict.selects_d2(),
                        "{mode:?}: explicit is not the default"
                    );
                    assert!(!probed, "{mode:?} must not probe");
                }
            }
        }
    }

    #[test]
    fn unset_selects_d2_only_when_every_gate_passes_and_the_costs_favor_it() {
        // (kill-switch, host split layers, unified, measured) -> (selects, probed)
        let cases = [
            // The only ON row.
            (false, 40, false, xtx(), true, true),
            // Each gate alone turns it off, and the probe never runs.
            (true, 40, false, xtx(), false, false),
            (false, 0, false, xtx(), false, false),
            (false, 40, true, xtx(), false, false),
            // A measured PCIe win and no measurement are off.
            (false, 40, false, r9700(), false, true),
            (false, 40, false, Err(Unusable::Arm), false, true),
        ];
        for (kill, layers, unified, measured, selects, probed) in cases {
            let (verdict, did_probe) = run(None, None, kill, layers, unified, measured);
            assert_eq!(
                verdict.selects_d2(),
                selects,
                "kill={kill} layers={layers} unified={unified} {measured:?}: {verdict:?}"
            );
            assert_eq!(did_probe, probed, "{verdict:?}");
        }
    }

    /// The typed developer key: `false` is never, `true` skips only the cost
    /// comparison (every other gate still applies) and never probes.
    #[test]
    fn typed_key_forces_or_vetoes_but_keeps_every_other_gate() {
        let (verdict, probed) = run(None, Some(false), false, 40, false, xtx());
        assert_eq!(verdict, D2Default::Disabled);
        assert!(!verdict.selects_d2() && !probed);
        let (verdict, probed) = run(None, Some(true), false, 40, false, r9700());
        assert_eq!(verdict, D2Default::Forced);
        assert!(verdict.selects_d2() && !probed);
        assert_eq!(
            run(None, Some(true), true, 40, false, r9700()).0,
            D2Default::KillSwitch
        );
        assert_eq!(
            run(None, Some(true), false, 0, false, r9700()).0,
            D2Default::NoHostSplitLayers
        );
        assert_eq!(
            run(None, Some(true), false, 40, true, r9700()).0,
            D2Default::Unified
        );
        assert_eq!(
            run(None, Some(false), true, 40, false, r9700()).0,
            D2Default::Disabled
        );
    }

    #[test]
    fn verdict_variants_match_their_gate() {
        assert_eq!(
            run(None, None, true, 5, false, xtx()).0,
            D2Default::KillSwitch
        );
        assert_eq!(
            run(None, None, false, 0, false, xtx()).0,
            D2Default::NoHostSplitLayers
        );
        assert_eq!(run(None, None, false, 5, true, xtx()).0, D2Default::Unified);
        assert_eq!(
            run(None, None, false, 5, false, Err(Unusable::Link)).0,
            D2Default::Unmeasured(Unusable::Link)
        );
        let win = from_rates(55.0, 7.1).costs().unwrap();
        assert_eq!(
            run(None, None, false, 5, false, xtx()).0,
            D2Default::CostsFavorD2(win)
        );
        let loss = from_rates(100.0, 40.0).costs().unwrap();
        assert_eq!(
            run(None, None, false, 5, false, r9700()).0,
            D2Default::CostsFavorPcie(loss)
        );
    }

    /// The 7900 XTX receipt: a 4.2–7.1 GB/s link against any CPU arm in the
    /// 30–154 GB/s range selects D2.
    #[test]
    fn xtx_receipt_link_rates_select_d2() {
        for link in [4.2, 5.0, 5.62, 6.0, 7.0, 7.1] {
            for cpu in [30.0, 40.0, 55.0, 80.0, 105.9, 154.0] {
                let costs = from_rates(cpu, link).costs().unwrap();
                assert!(
                    costs.favors_d2(),
                    "link {link} cpu {cpu}: D2 {} us vs PCIe {} us",
                    costs.d2_ns / 1000,
                    costs.pcie_ns / 1000
                );
                assert!(arms(from_rates(cpu, link)));
            }
        }
    }

    /// The R9700 matrix: 26–55 GB/s zero-copy links against 34–154 GB/s CPU
    /// arms. The 806.1 µs reserve plus even the fastest arm exceeds 85% of
    /// what PCIe costs at every one of these links, so no cell selects D2.
    #[test]
    fn r9700_matrix_never_selects_d2() {
        for link in [26.0, 30.0, 40.0, 50.0, 55.0] {
            for cpu in [34.0, 50.0, 75.0, 100.0, 154.0] {
                let costs = from_rates(cpu, link).costs().unwrap();
                assert!(
                    !costs.favors_d2(),
                    "link {link} cpu {cpu}: D2 {} us vs PCIe {} us",
                    costs.d2_ns / 1000,
                    costs.pcie_ns / 1000
                );
                assert!(!arms(from_rates(cpu, link)));
            }
        }
    }

    /// `D2 <= 85% of PCIe` is exact at the boundary, on the decision costs:
    /// the **median** arm sample (the slowest and fastest arm samples do not
    /// move it) against the fastest link and minimum floor. Equality selects
    /// D2 and one nanosecond more does not; the result is a pure function of
    /// the samples.
    #[test]
    fn exact_fifteen_percent_boundary_uses_the_median_arm_and_the_fastest_link() {
        // layer bytes == link sample bytes: pcie = fastest link + minimum floor.
        let at = |median_arm_ns: u64| DecodeMeasurement {
            cpu_ns: [10_000, median_arm_ns, median_arm_ns, 20_000_000, 20_000_000],
            batch_bytes: [LINK_SAMPLE_BYTES as u64; PROBE_SAMPLES],
            cpu_retry: None,
            // Slower link samples and a larger median floor must not matter.
            link_ns: [1_400_000, 990_000, 1_100_000, 1_200_000, 1_300_000],
            launch_floor_ns: [30_000, 10_000, 20_000, 25_000, 40_000],
            link_sample_bytes: LINK_SAMPLE_BYTES as u64,
            probe_wall_ns: 1,
        };
        let pcie = 1_000_000;
        let d2_at_limit = pcie * D2_MAX_PCIE_PERCENT / 100;
        assert_eq!(d2_at_limit, 850_000);
        let edge = at(d2_at_limit - CURRENT_COPY_TRANSPORT_HANDOFF_NS)
            .costs()
            .unwrap();
        assert_eq!((edge.pcie_ns, edge.d2_ns), (pcie, d2_at_limit));
        assert!(edge.favors_d2(), "exactly 15% faster selects D2");
        // The slowest samples ride along for the log only; the median PCIe is
        // strictly worse than the decision's best-case PCIe.
        assert!(edge.pcie_median_ns > edge.pcie_ns);
        assert_eq!(
            edge.d2_worst_ns,
            20_000_000 + CURRENT_COPY_TRANSPORT_HANDOFF_NS
        );
        assert_eq!(
            edge,
            at(d2_at_limit - CURRENT_COPY_TRANSPORT_HANDOFF_NS)
                .costs()
                .unwrap()
        );
        let over = at(d2_at_limit - CURRENT_COPY_TRANSPORT_HANDOFF_NS + 1)
            .costs()
            .unwrap();
        assert_eq!(over.d2_ns, d2_at_limit + 1);
        assert!(!over.favors_d2(), "14.9999% faster does not");
        let under = at(d2_at_limit - CURRENT_COPY_TRANSPORT_HANDOFF_NS - 1)
            .costs()
            .unwrap();
        assert!(under.favors_d2());
    }

    /// A disturbed arm on a 7900 XTX (CPU arm 0.39 ms, link 7.1 GB/s: PCIe
    /// 3.7 ms, 85% of it 3.14 ms): a single slow outlier, a sustained 3x or
    /// 5x disturbance and wide scatter all keep the median arm + 806.1 µs
    /// under the margin, so D2 stays selected however wide the spread is (the
    /// old hard 1.2x spread reject flipped these to PCIe).
    #[test]
    fn xtx_disturbed_high_arm_samples_still_select_d2() {
        let arm = arm_ns(66.9);
        let link = link_ns(7.1);
        for by in [
            [1.0, 1.0, 1.0, 1.0, 3.0], // one outlier
            [3.0; PROBE_SAMPLES],      // sustained contention
            [5.0; PROBE_SAMPLES],      // heavy sustained contention
            [1.0, 1.5, 3.0, 4.0, 2.5], // scattered
            [0.5, 0.7, 1.0, 1.2, 4.5], // spread 9x, slowest still clears
        ] {
            let measurement = sampled(scaled([arm; PROBE_SAMPLES], by), [link; PROBE_SAMPLES]);
            let costs = measurement.costs().expect("a spread is not an error");
            assert!(costs.favors_d2(), "{by:?}: {costs:?}");
            assert!(arms(measurement), "{by:?}");
            let spread = by.iter().cloned().fold(f64::MIN, f64::max)
                / by.iter().cloned().fold(f64::MAX, f64::min);
            assert!((costs.cpu_spread - spread).abs() < 0.01, "{by:?}");
        }
        // Wide, noisy samples are still a clear win; the old rule rejected it.
        let noisy = sampled(
            scaled([arm; PROBE_SAMPLES], [1.0, 1.0, 1.0, 1.0, 3.0]),
            scaled([link; PROBE_SAMPLES], [1.0, 1.1, 1.2, 1.4, 1.3]),
        );
        assert!(noisy.costs().unwrap().cpu_spread > 1.2);
        assert!(noisy.costs().unwrap().link_spread > 1.2);
        assert!(arms(noisy));
    }

    /// Past the margin the load stays on PCIe: a sustained 7x arm disturbance
    /// (median D2 3.5 ms against 3.14 ms), a sustained 3x arm with one link
    /// sample twice as fast as the rest (best-case PCIe 1.9 ms), and an
    /// implausibly fast link sample (0.3x of the rest: best-case PCIe 1.1 ms)
    /// under a one-outlier arm all end on PCIe. One 3x arm outlier with a link
    /// sample twice as fast now selects D2: the median ignores the outlier.
    #[test]
    fn past_the_margin_noise_stays_on_pcie() {
        let arm = arm_ns(66.9);
        let link = link_ns(7.1);
        let heavy = sampled(
            scaled([arm; PROBE_SAMPLES], [7.0; PROBE_SAMPLES]),
            [link; PROBE_SAMPLES],
        );
        assert!(!heavy.costs().unwrap().favors_d2());
        assert!(!arms(heavy));
        let outlier_fast_link = sampled(
            scaled([arm; PROBE_SAMPLES], [1.0, 1.0, 1.0, 1.0, 3.0]),
            scaled([link; PROBE_SAMPLES], [0.5, 1.0, 1.0, 1.0, 1.0]),
        );
        let costs = outlier_fast_link.costs().unwrap();
        assert!(costs.favors_d2(), "{costs:?}");
        assert!(arms(outlier_fast_link));
        let implausible_link = sampled(
            scaled([arm; PROBE_SAMPLES], [1.0, 1.0, 1.0, 1.0, 3.0]),
            scaled([link; PROBE_SAMPLES], [0.3, 1.0, 1.0, 1.0, 1.0]),
        );
        assert!(!arms(implausible_link));
        // Disturbance in both engines: a sustained 3x arm and a link twice as
        // fast in one sample.
        let both = sampled(
            scaled([arm; PROBE_SAMPLES], [3.0; PROBE_SAMPLES]),
            scaled([link; PROBE_SAMPLES], [0.5, 1.0, 1.0, 1.0, 1.0]),
        );
        assert!(!arms(both));
    }

    /// The defect: hipx `xb-splitkv0-2` refused D2 because one CPU sample was
    /// 7,316 µs among ~410 µs ones (worst-case D2 8,122 µs vs PCIe 3,691 µs,
    /// so the load decoded at ~5 tok/s).
    fn reported_outlier() -> DecodeMeasurement {
        DecodeMeasurement {
            cpu_ns: [410_000, 7_316_000, 405_000, 415_000, 412_000],
            ..from_rates(66.9, 7.1)
        }
    }

    #[test]
    fn one_huge_cpu_outlier_among_steady_samples_still_selects_d2() {
        let measurement = reported_outlier();
        let costs = measurement.costs().unwrap();
        assert_eq!(costs.cpu_arm_ns, 412_000);
        assert_eq!(costs.cpu_slowest_ns, 7_316_000);
        assert_eq!(costs.d2_ns, 412_000 + CURRENT_COPY_TRANSPORT_HANDOFF_NS);
        assert!(costs.favors_d2(), "{costs:?}");
        // The slowest-sample rule this replaces refused it.
        assert!(
            u128::from(costs.d2_worst_ns) * 100
                > u128::from(costs.pcie_ns) * u128::from(D2_MAX_PCIE_PERCENT)
        );
        let (verdict, probed) = run(None, None, false, 43, false, Ok(measurement));
        assert!(probed && verdict.selects_d2(), "{verdict:?}");
        let line = verdict.describe().unwrap();
        for needle in ["median CPU arm of 5 samples", "x17.", "max/median"] {
            assert!(line.contains(needle), "{needle:?} missing from {line}");
        }
    }

    /// A uniformly slow CPU is a slow CPU, not an outlier: refused, with or
    /// without a re-probe, and a re-probe that is also slow cannot rescue a
    /// dispersed first pass.
    #[test]
    fn uniformly_slow_cpu_is_still_refused() {
        let slow = 7_316_000;
        let uniform = DecodeMeasurement {
            cpu_ns: [slow; PROBE_SAMPLES],
            ..from_rates(66.9, 7.1)
        };
        assert!(!uniform.costs().unwrap().favors_d2());
        assert!(!arms(uniform));
        assert!(!cpu_pass_dispersed(&uniform.cpu_ns, &uniform.batch_bytes));
        // Most of the first pass slow and a slow re-probe.
        let slow_both = DecodeMeasurement {
            cpu_ns: [410_000, 7_316_000, 7_000_000, 7_100_000, 7_200_000],
            cpu_retry: Some(CpuRetry {
                cpu_ns: [slow; PROBE_SAMPLES],
                batch_bytes: uniform.batch_bytes,
            }),
            ..from_rates(66.9, 7.1)
        };
        assert!(!arms(slow_both));
        // A merely-slow CPU past the margin (median D2 3.5 ms vs 3.14 ms).
        assert!(!arms(sampled(
            scaled([arm_ns(66.9); PROBE_SAMPLES], [7.0, 7.0, 7.0, 1.0, 1.0]),
            [link_ns(7.1); PROBE_SAMPLES],
        )));
    }

    /// `dispersed` is strict at exactly `CPU_OUTLIER_PERCENT` of the median.
    #[test]
    fn dispersion_threshold_is_strict_and_deterministic() {
        let bytes = [flash_next_layer_bytes(); PROBE_SAMPLES];
        let pass = |max: u64| [100_000, 100_000, 100_000, 100_000, max];
        assert!(!cpu_pass_dispersed(&pass(100_000), &bytes));
        assert!(
            !cpu_pass_dispersed(&pass(200_000), &bytes),
            "exactly 200% is not an outlier"
        );
        assert!(cpu_pass_dispersed(&pass(200_001), &bytes));
        assert!(cpu_pass_dispersed(&reported_outlier().cpu_ns, &bytes));
        // An unusable pass is not "dispersed" (the decision reports it).
        assert!(!cpu_pass_dispersed(&[0, 1, 2, 3, 4], &bytes));
        assert!(!cpu_pass_dispersed(&pass(9_000_000), &[0; PROBE_SAMPLES]));
        // Normalization is per byte: a big batch that is proportionally
        // slower is steady.
        let mixed_bytes = [1_000, 2_000, 1_000, 1_000, 1_000];
        assert!(!cpu_pass_dispersed(
            &[100, 200, 100, 100, 100],
            &mixed_bytes
        ));
    }

    /// The re-probe runs exactly once, only for a dispersed first pass, and a
    /// failing pass fails closed.
    #[test]
    fn dispersed_first_pass_reprobes_exactly_once() {
        let bytes = [flash_next_layer_bytes(); PROBE_SAMPLES];
        let steady = ([100_000u64; PROBE_SAMPLES], bytes);
        let spiky = ([100_000, 100_000, 100_000, 100_000, 900_000], bytes);
        let calls = Cell::new(0);
        let (first, retry) = sample_cpu_with_reprobe(|| {
            calls.set(calls.get() + 1);
            Ok(steady)
        })
        .unwrap();
        assert_eq!((calls.get(), first, retry), (1, steady, None));
        calls.set(0);
        let (first, retry) = sample_cpu_with_reprobe(|| {
            calls.set(calls.get() + 1);
            // The re-probe is itself dispersed: still no third pass.
            Ok(spiky)
        })
        .unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(first, spiky);
        assert_eq!(
            retry,
            Some(CpuRetry {
                cpu_ns: spiky.0,
                batch_bytes: bytes
            })
        );
        calls.set(0);
        let failed = sample_cpu_with_reprobe(|| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Ok(spiky)
            } else {
                Err(Unusable::Arm)
            }
        });
        assert_eq!((calls.get(), failed), (2, Err(Unusable::Arm)));
    }

    /// A re-probe pools with the first pass (10 samples, upper median = 6th
    /// smallest): four slow samples of ten are ignored, five are not.
    #[test]
    fn pooled_reprobe_ignores_a_minority_and_not_a_majority() {
        let fast = 410_000;
        let slow = 7_316_000;
        let pooled = |first: [u64; PROBE_SAMPLES], retry: [u64; PROBE_SAMPLES]| DecodeMeasurement {
            cpu_ns: first,
            cpu_retry: Some(CpuRetry {
                cpu_ns: retry,
                batch_bytes: [flash_next_layer_bytes(); PROBE_SAMPLES],
            }),
            ..from_rates(66.9, 7.1)
        };
        // 1 + 3 slow of ten: D2.
        let four = pooled(
            [fast, fast, slow, fast, fast],
            [slow, fast, slow, fast, slow],
        );
        let costs = four.costs().unwrap();
        assert_eq!((costs.cpu_samples, costs.cpu_reprobed), (10, true));
        assert_eq!(costs.cpu_arm_ns, fast);
        assert!(costs.favors_d2() && arms(four));
        assert!(costs.detail().contains("median CPU arm of 10 samples"));
        assert!(costs.detail().contains("re-probed once and pooled"));
        // 5 slow of ten: the upper median is slow, refused.
        let five = pooled(
            [fast, fast, slow, fast, slow],
            [slow, fast, slow, fast, slow],
        );
        assert_eq!(five.costs().unwrap().cpu_arm_ns, slow);
        assert!(!arms(five));
        // A clean re-probe after a dispersed first pass: D2 even with two
        // outliers in the first pass.
        assert!(arms(pooled(
            [fast, slow, slow, fast, fast],
            [fast; PROBE_SAMPLES]
        )));
    }

    /// An R9700 (j1406: healthy CPU 119.6 / link 54.8 GB/s, disturbed CPU
    /// 47.7 / link 55.4) stays on PCIe whether healthy, contended, scattered
    /// or with one outlier of either sign: an outlier of either sign cannot
    /// move the median and a fast arm sample cannot flip it: even a sustained
    /// arm half the healthy time plus the 806.1 µs reserve exceeds 85% of
    /// best-case PCIe.
    #[test]
    fn r9700_healthy_contended_and_outliers_stay_on_pcie() {
        let cases = [
            (119.6, 54.8),
            (47.7, 55.4),
            (119.6, 55.4),
            (47.7, 54.8),
            (154.0, 26.0),
            (34.0, 55.0),
        ];
        for (cpu, link) in cases {
            let arm = arm_ns(cpu);
            let wire = link_ns(link);
            for by in [
                [1.0; PROBE_SAMPLES],
                [1.0, 1.0, 1.0, 1.0, 4.0],  // one slow outlier
                [0.25, 1.0, 1.0, 1.0, 1.0], // one fast outlier
                [0.5; PROBE_SAMPLES],       // sustained, disturbed FAST arm
                [2.0; PROBE_SAMPLES],       // sustained contention
                [0.4, 0.8, 1.5, 3.0, 2.0],  // scattered
            ] {
                for link_by in [
                    [1.0; PROBE_SAMPLES],
                    [0.9, 1.0, 1.1, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.8],
                ] {
                    let measurement = sampled(
                        scaled([arm; PROBE_SAMPLES], by),
                        scaled([wire; PROBE_SAMPLES], link_by),
                    );
                    assert!(
                        !measurement.costs().unwrap().favors_d2(),
                        "cpu {cpu} link {link} {by:?} {link_by:?}"
                    );
                    assert!(!arms(measurement));
                }
            }
        }
    }

    /// A zero, missing or non-positive figure is no measurement and fails
    /// closed (never an infinite rate).
    #[test]
    fn invalid_samples_fail_closed_to_pcie() {
        let base = from_rates(55.0, 7.1);
        assert!(base.costs().unwrap().favors_d2());
        let steady = base.cpu_ns[0];
        let link = base.link_ns[0];
        for broken in [
            DecodeMeasurement {
                cpu_ns: [0, steady, steady, steady, steady],
                ..base
            },
            DecodeMeasurement {
                link_ns: [link, link, 0, link, link],
                ..base
            },
            DecodeMeasurement {
                launch_floor_ns: [FLOOR_NS, 0, FLOOR_NS, FLOOR_NS, FLOOR_NS],
                ..base
            },
            DecodeMeasurement {
                batch_bytes: [0; PROBE_SAMPLES],
                ..base
            },
            DecodeMeasurement {
                link_sample_bytes: 0,
                ..base
            },
        ] {
            assert_eq!(broken.costs(), Err(Unusable::InvalidSample));
            let (verdict, _) = run(None, None, false, 43, false, Ok(broken));
            assert_eq!(verdict, D2Default::Unmeasured(Unusable::InvalidSample));
            assert!(!verdict.selects_d2());
        }
    }

    #[test]
    fn sample_stats_report_min_median_max_and_spread() {
        let stats = SampleStats::of(&[110, 100, 105, 120, 100]).unwrap();
        assert_eq!(
            stats,
            SampleStats {
                min_ns: 100,
                median_ns: 105,
                max_ns: 120
            }
        );
        assert!((stats.spread() - 1.2).abs() < 1e-9);
        assert_eq!(SampleStats::of(&[1, 2, 3, 4, 0]), None);
    }

    /// The codec and byte model are inputs: batches that read different byte
    /// counts (mixed codecs) are compared per byte at the median batch, so
    /// the same time-per-byte gives the same decision and a spread of
    /// 1x, whatever mix the batches drew, and every distinct count is kept.
    #[test]
    fn batches_with_distinct_byte_counts_compare_per_byte() {
        let big = flash_next_layer_bytes();
        let small = big / 2;
        let uniform = from_rates(55.0, 7.1);
        let per_byte_ns = uniform.cpu_ns[0] as f64 / big as f64;
        let batch_bytes = [small, big, small, big, small];
        let mixed = DecodeMeasurement {
            batch_bytes,
            cpu_ns: batch_bytes.map(|bytes| (bytes as f64 * per_byte_ns) as u64),
            ..uniform
        };
        let costs = mixed.costs().unwrap();
        assert_eq!(costs.layer_bytes, small, "the median batch's bytes");
        assert_eq!(costs.layer_bytes_range, (small, big));
        assert!((costs.cpu_spread - 1.0).abs() < 0.01, "{costs:?}");
        assert!(costs.favors_d2());
        assert!(costs.detail().contains("across batches"));
        // The same experts at a faster link: PCIe costs `small` bytes, not `big`.
        let narrow_pcie = DecodeMeasurement {
            link_ns: [link_ns(300.0); PROBE_SAMPLES],
            ..mixed
        };
        assert!(!narrow_pcie.costs().unwrap().favors_d2());
        // A zero byte count in any batch is no measurement.
        let broken = DecodeMeasurement {
            batch_bytes: [small, 0, small, big, small],
            ..mixed
        };
        assert_eq!(broken.costs(), Err(Unusable::InvalidSample));
    }

    /// The log carries the decision costs, the medians, the spreads, the
    /// floor and the probe wall cost.
    #[test]
    fn log_line_names_decision_medians_spreads_floor_and_probe_cost() {
        let arm = arm_ns(66.9);
        let link = link_ns(7.1);
        let line = run(
            None,
            None,
            false,
            43,
            false,
            Ok(sampled(
                scaled([arm; PROBE_SAMPLES], [1.0, 1.0, 1.0, 1.0, 3.0]),
                [link; PROBE_SAMPLES],
            )),
        )
        .0
        .describe()
        .unwrap();
        for needle in [
            "ON (",
            "median CPU arm of 5 samples",
            "best-case PCIe",
            "single slowest sample would give D2",
            "median PCIe",
            "slowest",
            "spread x3.00 max/min, x3.00 max/median",
            "link fastest",
            "launch floor",
            "copy-transport handoff",
            "real-arm probe 25.0 ms",
            "no stored rate",
        ] {
            assert!(line.contains(needle), "{needle:?} missing from {line}");
        }
        assert!(!line.contains("re-probed"), "{line}");
    }

    /// The calibration record's stored rates can never enable D2: `--show`
    /// states the same function's answer with no model arm, which is
    /// undecided on a discrete device and off on a unified one.
    #[test]
    fn calibration_verdict_never_claims_an_actual_decision() {
        let (status, text) = calibration_verdict(false);
        assert_eq!(status, "load");
        assert!(
            text.contains("calibration rates alone cannot enable it"),
            "{text}"
        );
        assert!(!text.starts_with("ON"));
        let (status, text) = calibration_verdict(true);
        assert_eq!(status, "off");
        assert_eq!(text, "off (unified memory: no link to beat)");
    }

    /// One measurement serves exactly one `(device, codec set, shape)`; a
    /// repeat reuses the first stored value, and a failed one is not stored.
    #[test]
    fn measurement_cache_key_never_crosses_device_codecs_or_shape() {
        let cache = Mutex::new(BTreeMap::new());
        let calls = Cell::new(0);
        let measure = |cpu: f64| {
            calls.set(calls.get() + 1);
            Ok(from_rates(cpu, 7.1))
        };
        let mq4 = vec![(DType::MQ4G256V2, DType::MQ4G128V2)];
        let key: ArmKey = (0, mq4.clone(), 2560, 640, 10);
        let (first, cached) = cached_measurement(&cache, key.clone(), || measure(55.0));
        assert!(!cached);
        assert_eq!(calls.get(), 1);
        let (again, cached) = cached_measurement(&cache, key.clone(), || measure(99.0));
        assert!(cached);
        assert_eq!(again, first);
        assert_eq!(calls.get(), 1);
        let mixed = vec![
            (DType::MQ2G256V2, DType::MQ2G256V2),
            (DType::MQ4G256V2, DType::MQ4G128V2),
        ];
        for other in [
            (1, mq4.clone(), 2560, 640, 10),
            (0, vec![(DType::MQ6G256V2, DType::MQ4G128V2)], 2560, 640, 10),
            (0, mixed, 2560, 640, 10),
            (0, mq4.clone(), 2560, 512, 10),
            (0, mq4.clone(), 2560, 640, 8),
        ] {
            let (_, cached) = cached_measurement(&cache, other.clone(), || measure(60.0));
            assert!(!cached, "{other:?}");
        }
        assert_eq!(calls.get(), 6);
        let failing: ArmKey = (2, mq4, 2560, 640, 10);
        let (failed, _) = cached_measurement(&cache, failing.clone(), || Err(Unusable::Arm));
        assert_eq!(failed, Err(Unusable::Arm));
        let (retry, cached) = cached_measurement(&cache, failing, || measure(70.0));
        assert!(!cached && retry.is_ok(), "a failure is measured again");
    }

    /// Which layers count as host split layers: a tier layer needs a
    /// host-mapped expert; an un-tiered layer needs both aggregates
    /// host-mapped and the QT44/QT53 pair; a slab layer needs a host-mapped
    /// arena whose present experts all have a CPU decoder; a pruned tier or
    /// aggregate layer never counts (a slab arena keeps pruned ids).
    #[test]
    fn host_split_layer_truth_table() {
        let tier = |any_host| LayerHostFacts {
            tier_any_host: Some(any_host),
            ..LayerHostFacts::default()
        };
        let aggregate = |host, qt| LayerHostFacts {
            aggregate_host: host,
            aggregate_qt44_qt53: qt,
            ..LayerHostFacts::default()
        };
        assert!(tier(true).qualifies());
        assert!(
            !tier(false).qualifies(),
            "an all-resident tier layer has nothing to do"
        );
        assert!(aggregate(true, true).qualifies());
        assert!(
            !aggregate(false, true).qualifies(),
            "a resident aggregate stays on the GPU"
        );
        assert!(
            !aggregate(true, false).qualifies(),
            "no CPU executor for the format pair"
        );
        // A tier layer ignores the aggregate fields (its allocations are the tier owners').
        assert!(
            tier(true).qualifies()
                && LayerHostFacts {
                    aggregate_host: false,
                    aggregate_qt44_qt53: false,
                    ..tier(true)
                }
                .qualifies()
        );
        for facts in [tier(true), aggregate(true, true)] {
            assert!(
                !LayerHostFacts {
                    pruned: true,
                    ..facts
                }
                .qualifies(),
                "pruned payloads are compact: the original-id owners cannot index them"
            );
        }
        let slab = |host, cpu| LayerHostFacts {
            slab: Some((host, cpu)),
            ..LayerHostFacts::default()
        };
        assert!(slab(true, true).qualifies());
        assert!(
            !slab(false, true).qualifies(),
            "a device-resident arena has no host expert"
        );
        assert!(
            !slab(true, false).qualifies(),
            "a present expert without a CPU decoder keeps the layer on PCIe"
        );
        assert!(
            LayerHostFacts {
                pruned: true,
                ..slab(true, true)
            }
            .qualifies(),
            "a slab arena keeps pruned experts at their original ids"
        );
    }

    /// The default is a per-load `bool`: deciding it, even ON, does not change
    /// the process-wide host-exec mode, so D1, the step seam, the fusion
    /// selector and passback keep reading what they read before.
    #[test]
    fn selecting_d2_does_not_set_the_process_host_exec_mode() {
        if hipfire_config::memory::offload_exec_explicit().is_some() {
            return; // the user's explicit mode is the process mode; nothing to compare
        }
        let before = (
            hipfire_dispatch::cpu_exec_enabled(),
            hipfire_dispatch::passback_enabled(),
            hipfire_dispatch::moe_cpu_experts_enabled(),
        );
        let (verdict, _) = run(None, None, false, 40, false, xtx());
        assert!(verdict.selects_d2());
        let (forced, _) = run(None, Some(true), false, 40, false, Err(Unusable::Arm));
        assert!(forced.selects_d2());
        let after = (
            hipfire_dispatch::cpu_exec_enabled(),
            hipfire_dispatch::passback_enabled(),
            hipfire_dispatch::moe_cpu_experts_enabled(),
        );
        assert_eq!(before, after);
        assert_eq!(before, (false, false, false), "unset mode is pcie");
        assert_eq!(hipfire_config::memory::offload_exec(), OffloadExec::Pcie);
    }

    #[test]
    fn round_robin_takes_distinct_experts_across_layers_and_reports_a_shortfall() {
        // 3 layers; layer 1 has only even experts cold; 4 experts each.
        let cold = |layer: usize, expert: usize| match (layer, expert % 2) {
            (1, 1) => None,
            _ => Some((layer, expert)),
        };
        let picked = round_robin_pick(3, 4, cold, 7).unwrap();
        assert_eq!(
            picked,
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2)]
        );
        let mut unique = picked.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            picked.len(),
            "every pick is a distinct expert"
        );
        // 3 layers x 4 experts minus layer 1's two odd ones = 10 cold.
        assert_eq!(round_robin_pick(3, 4, cold, 10).unwrap().len(), 10);
        assert_eq!(round_robin_pick(3, 4, cold, 11), Err(10));
        assert_eq!(round_robin_pick::<u8>(0, 4, |_, _| None, 1), Err(0));
    }

    /// The production arm on real quantized spans of two different codecs:
    /// all three projections run (a non-zero activation through gate/up,
    /// SwiGLU and down), every timed batch is a distinct set of experts, and
    /// each batch carries the bytes of its own spans (MQ4 batches and the
    /// smaller mq2-shaped MQ2 batches differ). Any codec pair the CPU
    /// executes can be fed through the same interface. CPU only.
    #[test]
    fn real_arm_probe_times_distinct_cold_experts_with_their_own_codec_and_bytes() {
        use hipfire_dispatch::cpu_exec::cpu_quant_for;
        use hipfire_dispatch::row_bytes_for;
        let (dim, mi, top_k) = (256, 256, 4);
        let bytes_of = |gate_up: DType, down: DType| {
            (
                2 * mi * row_bytes_for(gate_up, dim).unwrap(),
                dim * row_bytes_for(down, mi).unwrap(),
            )
        };
        let mq4 = (DType::MQ4G256V2, DType::MQ4G128V2);
        let mq2 = (DType::MQ2G256V2, DType::MQ2G256V2);
        let (mq4_gu, mq4_dn) = bytes_of(mq4.0, mq4.1);
        let (mq2_gu, mq2_dn) = bytes_of(mq2.0, mq2.1);
        assert!(mq2_gu + mq2_dn < mq4_gu + mq4_dn, "mq2 experts are smaller");
        let need = (PROBE_WARM_BATCHES + PROBE_SAMPLES) * top_k;
        let fill = |len: usize, seed: usize| -> Vec<u8> {
            (0..len)
                .map(|i| (i.wrapping_mul(31).wrapping_add(seed * 17) % 251) as u8)
                .collect()
        };
        // Batch 0 (warm) and every even batch MQ4, every odd batch MQ2.
        let pool: Vec<((DType, DType), Vec<u8>, Vec<u8>)> = (0..need)
            .map(|e| {
                let codec = if (e / top_k) % 2 == 0 { mq4 } else { mq2 };
                let (gu, dn) = bytes_of(codec.0, codec.1);
                (codec, fill(gu, e), fill(dn, e + 1000))
            })
            .collect();
        let experts: Vec<ProbeExpert<'_>> = pool
            .iter()
            .map(|(codec, gate_up, down)| {
                ProbeExpert::same_bytes(CpuMoeExpert {
                    gate_up_quant: cpu_quant_for(codec.0).unwrap(),
                    gate_up,
                    down_quant: cpu_quant_for(codec.1).unwrap(),
                    down,
                })
            })
            .collect();
        let (samples, batch_bytes) = sample_cpu_arm(dim, mi, top_k, 64, &experts).unwrap();
        assert!(samples.iter().all(|&ns| ns > 0), "{samples:?}");
        let (mq4_batch, mq2_batch) = (
            (top_k * (mq4_gu + mq4_dn)) as u64,
            (top_k * (mq2_gu + mq2_dn)) as u64,
        );
        // Timed batches are 1..=5: MQ2, MQ4, MQ2, MQ4, MQ2.
        assert_eq!(
            batch_bytes,
            [mq2_batch, mq4_batch, mq2_batch, mq4_batch, mq2_batch]
        );
        // They feed the cost model as distinct byte counts.
        let measurement = DecodeMeasurement {
            cpu_ns: samples,
            batch_bytes,
            cpu_retry: None,
            link_ns: [link_ns(7.1); PROBE_SAMPLES],
            launch_floor_ns: [FLOOR_NS; PROBE_SAMPLES],
            link_sample_bytes: LINK_SAMPLE_BYTES as u64,
            probe_wall_ns: 1,
        };
        let costs = measurement.costs().unwrap();
        assert_eq!(costs.layer_bytes_range, (mq2_batch, mq4_batch));
        assert_eq!(costs.layer_bytes, mq2_batch);
        // The batch count is fixed by the expert count: a short list fails closed.
        assert_eq!(
            sample_cpu_arm(dim, mi, top_k, 64, &experts[..need - 1]),
            Err(Unusable::Arm)
        );
        // More top_k than experts addressed is not a layer.
        assert_eq!(
            sample_cpu_arm(dim, mi, top_k, top_k - 1, &experts),
            Err(Unusable::Arm)
        );
        // A span shorter than its codec needs is refused by the arm, not timed.
        let short = fill(mq4_gu - 1, 9);
        let mut broken = experts.clone();
        broken[0].cpu.gate_up = &short;
        assert_eq!(
            sample_cpu_arm(dim, mi, top_k, 64, &broken),
            Err(Unusable::Arm)
        );
        // The activation is real: non-zero, bounded.
        let x = probe_activation(2560);
        assert!(x.iter().any(|&v| v != 0.0));
        assert!(x.iter().all(|v| v.abs() <= 1.01));
    }

    /// The GPU read bytes are their own input: when they differ from the CPU
    /// span bytes (a CPU codec against another GPU read format), the batch
    /// bytes, the PCIe cost and the CPU normalization all use the GPU value,
    /// never the CPU span length. CPU only.
    #[test]
    fn pcie_is_priced_on_gpu_read_bytes_not_cpu_span_bytes() {
        use hipfire_dispatch::cpu_exec::cpu_quant_for;
        use hipfire_dispatch::row_bytes_for;
        let (dim, mi, top_k) = (256, 256, 4);
        let gate_up_len = 2 * mi * row_bytes_for(DType::MQ2G256V2, dim).unwrap();
        let down_len = dim * row_bytes_for(DType::MQ2G256V2, mi).unwrap();
        let cpu_expert_bytes = (gate_up_len + down_len) as u64;
        // The GPU reads a different format with a different size per expert:
        // 3 x the CPU payload here (any value that is not the CPU length).
        let gpu_expert_bytes = 3 * cpu_expert_bytes;
        let need = (PROBE_WARM_BATCHES + PROBE_SAMPLES) * top_k;
        let fill = |len: usize, seed: usize| -> Vec<u8> {
            (0..len)
                .map(|i| (i.wrapping_mul(29).wrapping_add(seed * 13) % 247) as u8)
                .collect()
        };
        let pool: Vec<(Vec<u8>, Vec<u8>)> = (0..need)
            .map(|e| (fill(gate_up_len, e), fill(down_len, e + 500)))
            .collect();
        let quant = cpu_quant_for(DType::MQ2G256V2).unwrap();
        let experts: Vec<ProbeExpert<'_>> = pool
            .iter()
            .map(|(gate_up, down)| ProbeExpert {
                cpu: CpuMoeExpert {
                    gate_up_quant: quant,
                    gate_up,
                    down_quant: quant,
                    down,
                },
                gpu_read_bytes: gpu_expert_bytes,
            })
            .collect();
        let (samples, batch_bytes) = sample_cpu_arm(dim, mi, top_k, 64, &experts).unwrap();
        assert_eq!(
            batch_bytes,
            [top_k as u64 * gpu_expert_bytes; PROBE_SAMPLES],
            "batch bytes are the GPU read bytes, not the CPU span bytes"
        );
        assert_ne!(batch_bytes[0], top_k as u64 * cpu_expert_bytes);
        // The same CPU experts declared with CPU-length GPU bytes.
        let same: Vec<ProbeExpert<'_>> = experts
            .iter()
            .map(|expert| ProbeExpert::same_bytes(expert.cpu))
            .collect();
        let (_, same_bytes) = sample_cpu_arm(dim, mi, top_k, 64, &same).unwrap();
        assert_eq!(same_bytes, [top_k as u64 * cpu_expert_bytes; PROBE_SAMPLES]);
        // PCIe cost scales with the GPU bytes, with identical CPU samples.
        let measure = |bytes: [u64; PROBE_SAMPLES]| DecodeMeasurement {
            cpu_ns: samples,
            batch_bytes: bytes,
            cpu_retry: None,
            link_ns: [1_000_000; PROBE_SAMPLES],
            launch_floor_ns: [FLOOR_NS; PROBE_SAMPLES],
            link_sample_bytes: LINK_SAMPLE_BYTES as u64,
            probe_wall_ns: 1,
        };
        let gpu_costs = measure(batch_bytes).costs().unwrap();
        let cpu_costs = measure(same_bytes).costs().unwrap();
        let expected = |bytes: u64| {
            (u128::from(bytes) * 1_000_000 / LINK_SAMPLE_BYTES as u128) as u64 + FLOOR_NS
        };
        assert_eq!(gpu_costs.layer_bytes, top_k as u64 * gpu_expert_bytes);
        assert_eq!(gpu_costs.pcie_ns, expected(top_k as u64 * gpu_expert_bytes));
        assert_eq!(cpu_costs.pcie_ns, expected(top_k as u64 * cpu_expert_bytes));
        assert!(gpu_costs.pcie_ns > cpu_costs.pcie_ns);
        // The CPU arm is normalized against the same GPU quantity: with the
        // identical CPU time and 3x the GPU bytes, the per-GPU-byte CPU cost
        // is the median CPU sample scaled by 1x (uniform batches).
        assert_eq!(gpu_costs.cpu_arm_ns, cpu_costs.cpu_arm_ns);
    }
}
