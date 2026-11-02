// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The ONE expert-capability table: architecture x exact on-disk quant-tag pair x route.
//!
//! Every gate that decides whether a routed-expert codec runs on an architecture (the sealed
//! slab's K1/K2 selection and buckets, the arena and tier-plan `check_executable`s, the artifact
//! load gate, dispatch's slab pair and bucket selection, the expert stage defaults) reads this
//! module. Two layers, deliberately separate:
//!
//! 1. **Raw inventory** ([`raw_inventory`]): which kernels EXIST for a cell, read from the real
//!    kernel tables (`slab_moe` K1 tables, `gemm` K2 modules). It never consults admission or a
//!    hold, so the kernel tables can call the gate without a cycle.
//! 2. **Qualification** ([`capability`]): the raw inventory minus the explicit, named
//!    [`HOLD_CELLS`]. A cell is [`Verdict::Admitted`] only when its raw inventory is present and
//!    no active hold names it. [`Verdict::NoKernel`] means no kernel exists (W2 everywhere,
//!    gfx1100 W3, every other architecture); [`Verdict::Held`] means a kernel exists and a named
//!    qualification refuses it (a registry-present, beta-refused cell).
//!
//! # Key
//!
//! The capability key is the EXACT quant-tag pair [`ExpertQt`] `{gate_up, down}` read from the
//! tensor table, never a model, product or "XTS" label, and never a generation label.
//! [`generation_label`] is a display-only map (`qt -> "v1" | "v2" | "v3"`) that nothing deciding
//! admission may read. Source names such as `MQ4G256V2` / `MQ4G128V2` for qt 44 / 53 carry a
//! codec-LAYOUT revision ("V2" = the dual-half 8-byte group header), NOT the MQ generation.
//!
//! Route prerequisite (not a table cell): qt `(44, 53)` kernels are symmetric-grid kernels
//! (`zp == -8 * sc`). The quant tag cannot tell a symmetric XTS body from an asymmetric one (the
//! retired `flash-next-rtn-asym` pin and the old `mq4-xt` dense pin carry the same qt 44 / 53);
//! the existing payload check (`artifact::routed_headers_symmetric`,
//! `Gpu::qwen4_moe_sym_check`) decides symmetry from the header bytes and stays the prerequisite.
//!
//! # Routes
//!
//! | route | meaning | raw inventory |
//! |---|---|---|
//! | [`Route::PrefillK2`] | K2 grouped GEMMs over resident arenas | the codec's K2 module |
//! | [`Route::DecodeGemv`] | K1 lane-union GEMV | the codec's K1 module |
//! | [`Route::Stage`] | whole-layer long-prefill staging of a slab arena | K1 AND K2 present ([`Raw::Composite`]) |
//! | [`Route::TierStage`] | host-mapped tier layout / tier staging (`ExpertTierPlan`) | MQ4: arch-neutral aggregate route ([`Raw::ArchNeutral`]); others K1 AND K2 |
//! | [`Route::D2CpuSplit`] | masked GPU + CPU decoder split | caller supplied ([`Raw::External`]) |
//!
//! # Beta truth table (default holds, `Holds::BETA`)
//!
//! | arch | MQ4 (44,53) | W3 (131,132) | W2 (133,134) | E8P (135,136) |
//! |---|---|---|---|---|
//! | gfx1201 | K2 K1 Stage Tier | K2 K1 Stage Tier | none | K2 K1 Stage (Tier held) |
//! | gfx1151 | K2 K1 Stage Tier | K2 K1 Stage (Tier held) | none | K2 K1 Stage (Tier held) |
//! | gfx1100 | K2 K1 Stage Tier | none | none | K2 K1 Stage (Tier held) |
//! | other | Tier only (arch-neutral) | none | none | none |
//!
//! Every registry-present / refused cell is a named row of [`HOLD_CELLS`].

use crate::dispatch::DType;
use crate::slab_admission::SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED;

/// Exact on-disk quant tags of one routed expert: the gate/up tag and the down tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertQt {
    pub gate_up: u8,
    pub down: u8,
}

/// Copied symmetric MQ4 experts: qt 44 (`MQ4G256V2`) gate/up + qt 53 (`MQ4G128V2`) down; slab
/// codec byte 0.
pub const QT_MQ4_XTS: ExpertQt = ExpertQt::new(44, 53);
/// MQ3-I4 planar (W3): qt 131 / 132; slab codec byte 1.
pub const QT_W3: ExpertQt = ExpertQt::new(131, 132);
/// MQ2-I4 planar (W2): qt 133 / 134; slab codec byte 2. No kernels anywhere.
pub const QT_W2: ExpertQt = ExpertQt::new(133, 134);
/// E8P-I4 planar: qt 135 / 136; slab codec byte 3.
pub const QT_E8P: ExpertQt = ExpertQt::new(135, 136);
/// Uniform MQ4G256 (qt 13): the v1 MQ4 tag. No qwen4 expert kernels.
pub const QT_MQ4_V1: ExpertQt = ExpertQt::new(13, 13);
/// UNCONFIRMED pre-XTS "v2" candidate: MQ4CG256 (qt 45), header differs from qt 44. No kernels.
pub const QT_MQ4C: ExpertQt = ExpertQt::new(45, 45);
/// UNCONFIRMED pre-XTS "v2" candidate: MQ4G256V2Lloyd (qt 52, needs a Lloyd sidecar). No kernels.
pub const QT_MQ4_LLOYD: ExpertQt = ExpertQt::new(52, 52);

/// Every quant pair the enumeration test walks (the registry tags that name an expert codec).
pub const REGISTRY_PAIRS: [ExpertQt; 7] = [
    QT_MQ4_V1,
    QT_MQ4C,
    QT_MQ4_LLOYD,
    QT_MQ4_XTS,
    QT_W3,
    QT_W2,
    QT_E8P,
];

impl ExpertQt {
    pub const fn new(gate_up: u8, down: u8) -> Self {
        Self { gate_up, down }
    }

    /// The slab codec byte of this pair (`0` MQ4, `1` W3, `2` W2, `3` E8P); `None` for every
    /// other pair (including crossed pairs such as `(44, 132)`).
    pub const fn slab_codec(self) -> Option<u8> {
        match (self.gate_up, self.down) {
            (44, 53) => Some(0),
            (131, 132) => Some(1),
            (133, 134) => Some(2),
            (135, 136) => Some(3),
            _ => None,
        }
    }

    /// Inverse of [`Self::slab_codec`].
    pub const fn of_slab_codec(codec: u8) -> Option<Self> {
        match codec {
            0 => Some(QT_MQ4_XTS),
            1 => Some(QT_W3),
            2 => Some(QT_W2),
            3 => Some(QT_E8P),
            _ => None,
        }
    }

    /// The quant pair of a routed `(gate_up, down)` dtype pair: the four slab pairs only
    /// (`None` for any other or crossed pair).
    pub const fn of_dtypes(gate_up: DType, down: DType) -> Option<Self> {
        match (gate_up, down) {
            (DType::MQ4G256V2, DType::MQ4G128V2) => Some(QT_MQ4_XTS),
            (DType::MQ3I4G128R256, DType::MQ3I4G128R128) => Some(QT_W3),
            (DType::MQ2I4G128R256, DType::MQ2I4G128R128) => Some(QT_W2),
            (DType::E8PI4G128R256, DType::E8PI4G128R128) => Some(QT_E8P),
            _ => None,
        }
    }
}

/// DISPLAY / REPORT ONLY map of a quant tag to its MQ generation label. Nothing that decides
/// admission reads it. Labels are tentative: qt 13 `v1`; qt 45 / 52 `v2` (UNCONFIRMED pre-XTS
/// candidates); qt 44 / 53 and the planar I4 tags (W3, W2, E8P) `v3`.
///
/// The "V2" in the SOURCE names `MQ4G256V2` / `MQ4G128V2` (qt 44 / 53) means the codec-layout
/// revision (dual fp16 half header), NOT the MQ generation.
pub const GENERATION_LABELS: [(u8, &str); 11] = [
    (13, "v1"),
    (45, "v2"),
    (52, "v2"),
    (44, "v3"),
    (53, "v3"),
    (131, "v3"),
    (132, "v3"),
    (133, "v3"),
    (134, "v3"),
    (135, "v3"),
    (136, "v3"),
];

/// Display label of quant tag `qt` ([`GENERATION_LABELS`]); `None` for an unlabelled tag.
pub fn generation_label(qt: u8) -> Option<&'static str> {
    GENERATION_LABELS
        .iter()
        .find(|(tag, _)| *tag == qt)
        .map(|(_, label)| *label)
}

/// A route of the expert capability table. `#[non_exhaustive]`: new routes (their own raw
/// inventory and holds) are added here, never folded into codec admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Route {
    /// K2 grouped GEMMs (prefill) over resident arenas.
    PrefillK2,
    /// K1 lane-union GEMV (decode).
    DecodeGemv,
    /// Whole-layer long-prefill staging of a slab arena.
    Stage,
    /// Host-mapped tier layout (`ExpertTierPlan`) and its tier staging.
    TierStage,
    /// Masked GPU + CPU decoder split (the CPU decoder inventory lives in `hipfire-dispatch`).
    D2CpuSplit,
}

impl Route {
    pub const ALL: [Route; 5] = [
        Route::PrefillK2,
        Route::DecodeGemv,
        Route::Stage,
        Route::TierStage,
        Route::D2CpuSplit,
    ];
}

/// Module and `[gate/up, down]` entry symbols of one raw kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawKernel {
    pub module: &'static str,
    pub symbols: [&'static str; 2],
}

/// What exists for a cell, before any qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Raw {
    Absent,
    /// One kernel module (K1 or K2).
    Kernel(RawKernel),
    /// Both the K1 and the K2 module of the codec exist (staged / tiered execution reuses them).
    Composite {
        k1: RawKernel,
        k2: RawKernel,
    },
    /// No arch-specific kernel: the cell is served by an arch-neutral route described by the text.
    ArchNeutral(&'static str),
    /// Inventory owned by a higher crate and supplied by the caller.
    External(&'static str),
}

impl Raw {
    pub fn is_present(self) -> bool {
        !matches!(self, Raw::Absent)
    }
}

/// A named qualification that refuses a registry-present cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hold {
    /// Tier plans refuse E8P-I4 experts on every architecture
    /// (`ExpertTierPlan::check_executable`): there is no host-mapped E8P width.
    TierPlanE8pRefused,
    /// Tier plans refuse W3 experts anywhere but gfx1201 (`ExpertTierPlan::check_executable`).
    TierPlanW3Gfx1151Refused,
    /// The gfx1151 E8P-I4 kernels exist but their admission switch
    /// ([`SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED`]) is off. Inactive while the switch is true.
    Gfx1151E8pReceiptsPending,
}

impl Hold {
    /// Whether this hold refuses its cells under `holds`.
    pub const fn active(self, holds: &Holds) -> bool {
        match self {
            Hold::TierPlanE8pRefused | Hold::TierPlanW3Gfx1151Refused => true,
            Hold::Gfx1151E8pReceiptsPending => !holds.gfx1151_e8p_landed,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Hold::TierPlanE8pRefused => "tier-plan-e8p-refused",
            Hold::TierPlanW3Gfx1151Refused => "tier-plan-w3-gfx1151-refused",
            Hold::Gfx1151E8pReceiptsPending => "gfx1151-e8p-receipts-pending",
        }
    }
}

/// One registry-present, qualification-held cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoldCell {
    pub arch: &'static str,
    pub qt: ExpertQt,
    pub route: Route,
    pub hold: Hold,
}

const fn cell(arch: &'static str, qt: ExpertQt, route: Route, hold: Hold) -> HoldCell {
    HoldCell {
        arch,
        qt,
        route,
        hold,
    }
}

/// The explicit hold list: every cell whose raw kernels exist and which beta qualification
/// refuses (the three `Gfx1151E8pReceiptsPending` rows only while that switch is false).
/// The first matching active row names the verdict.
pub const HOLD_CELLS: [HoldCell; 7] = [
    cell(
        "gfx1100",
        QT_E8P,
        Route::TierStage,
        Hold::TierPlanE8pRefused,
    ),
    cell(
        "gfx1151",
        QT_E8P,
        Route::TierStage,
        Hold::TierPlanE8pRefused,
    ),
    cell(
        "gfx1201",
        QT_E8P,
        Route::TierStage,
        Hold::TierPlanE8pRefused,
    ),
    cell(
        "gfx1151",
        QT_W3,
        Route::TierStage,
        Hold::TierPlanW3Gfx1151Refused,
    ),
    cell(
        "gfx1151",
        QT_E8P,
        Route::PrefillK2,
        Hold::Gfx1151E8pReceiptsPending,
    ),
    cell(
        "gfx1151",
        QT_E8P,
        Route::DecodeGemv,
        Hold::Gfx1151E8pReceiptsPending,
    ),
    cell(
        "gfx1151",
        QT_E8P,
        Route::Stage,
        Hold::Gfx1151E8pReceiptsPending,
    ),
];

/// Artifact-load receipts flag of the gfx1100 and gfx1201 slab route. Set because the real-layer
/// G1 receipts of BOTH gfx1201 slab kernels are in `rcpt/`, both with 0 mismatches: K1 decode
/// (`r5.k1.g1.L0-records.raw.json`, sha256 `2bb0cfa1…8239`) and K2 prefill (`g1.k2.real.json`,
/// sha256 `bb1bde6b…c226`, with the A4 identity `g1.a4.identity.json`, sha256 `2e6f9643…a979`).
/// The gfx1100 route (MQ4-only layers) is admitted by the same flag. Read by
/// [`slab_artifact_route_admitted_with`] only; it never refuses a kernel cell.
pub const SLAB_ROUTE_RECEIPTS_LANDED: bool = true;

/// gfx1151 twin of [`SLAB_ROUTE_RECEIPTS_LANDED`]. Set because the G1 receipts of the gfx1151
/// slab kernels are in `rcpt/` on the Halo card (exact gfx1151, BDF `0000:c0:00.0`), each passing
/// with 0 mismatches against the CPU twin: K1 decode `mq_i4_moe_gemv_pm_gfx1151`, synthetic plus
/// certified L0 records with the old-kernel parity (`r9.g1.k1.gfx1151.json`, sha256
/// `f5e4e524…b3c6`); K2 prefill, the W3 NT1 anchor and the NT4 entries with the down row repeats
/// 1/2/4 (`qwen4_moe_iu4_sym_w3_pm_gfx1151`, `qwen4_moe_iu4_sym_w3_nt4_pm_gfx1151`), synthetic
/// (`r9.g1.k2.synthetic.gfx1151.json`, sha256 `eb5c643a…181c`) and real layers
/// (`r9.g1.k2.real.gfx1151.json`, sha256 `ceabf444…0f73`). The A4 activations are deterministic
/// synthetic rows through the real weights. gfx1151 only; every other architecture keeps the
/// `ExpertSlabKernelsUnavailable` refusal. Read by [`slab_artifact_route_admitted_with`] only.
pub const SLAB_ROUTE_GFX1151_RECEIPTS_LANDED: bool = true;

/// The switches the holds read. [`Holds::BETA`] is the shipped state; tests flip single fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Holds {
    /// [`SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED`]: gfx1151 E8P cells are served.
    pub gfx1151_e8p_landed: bool,
    /// [`SLAB_ROUTE_RECEIPTS_LANDED`]: the gfx1100 / gfx1201 artifact-load receipts.
    pub shared_receipts_landed: bool,
    /// [`SLAB_ROUTE_GFX1151_RECEIPTS_LANDED`]: the gfx1151 artifact-load receipts.
    pub gfx1151_receipts_landed: bool,
}

impl Holds {
    pub const BETA: Holds = Holds {
        gfx1151_e8p_landed: SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED,
        shared_receipts_landed: SLAB_ROUTE_RECEIPTS_LANDED,
        gfx1151_receipts_landed: SLAB_ROUTE_GFX1151_RECEIPTS_LANDED,
    };
}

/// Qualification outcome of one cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Admitted,
    /// No kernel exists for the cell.
    NoKernel,
    /// A kernel exists and the named hold refuses it.
    Held(Hold),
}

/// One cell: its raw inventory and its verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capability {
    pub raw: Raw,
    pub verdict: Verdict,
}

impl Capability {
    pub fn admitted(self) -> bool {
        self.verdict == Verdict::Admitted
    }
}

fn k1_raw(arch: &str, codec: u8) -> Option<RawKernel> {
    let k1 = if codec == 0 {
        &crate::slab_moe::slab_arch_kernels(arch)?.mq4_k1
    } else {
        &crate::slab_moe::slab_cold_k1_metadata(arch, codec)?.k1
    };
    Some(RawKernel {
        module: k1.module,
        symbols: k1.symbols,
    })
}

fn k2_raw(arch: &str, codec: u8) -> Option<RawKernel> {
    let (module, symbols) = crate::gemm::slab_k2_raw(arch, codec)?;
    Some(RawKernel { module, symbols })
}

/// The raw kernel inventory of `(arch, qt, route)` (no external inventory). Never consults any
/// admission or hold.
pub fn raw_inventory(arch: &str, qt: ExpertQt, route: Route) -> Raw {
    raw_inventory_with_external(arch, qt, route, None)
}

/// [`raw_inventory`] with the caller-owned inventory of [`Route::D2CpuSplit`]
/// (`Some(name)` = the CPU decoder executes the pair); ignored for every other route.
pub fn raw_inventory_with_external(
    arch: &str,
    qt: ExpertQt,
    route: Route,
    external: Option<&'static str>,
) -> Raw {
    if route == Route::D2CpuSplit {
        return match (qt, external) {
            (QT_MQ4_XTS, Some(name)) => Raw::External(name),
            _ => Raw::Absent,
        };
    }
    let Some(codec) = qt.slab_codec() else {
        return Raw::Absent;
    };
    let (k1, k2) = (k1_raw(arch, codec), k2_raw(arch, codec));
    match route {
        Route::PrefillK2 => k2.map_or(Raw::Absent, Raw::Kernel),
        Route::DecodeGemv => k1.map_or(Raw::Absent, Raw::Kernel),
        Route::Stage => match (k1, k2) {
            (Some(k1), Some(k2)) => Raw::Composite { k1, k2 },
            _ => Raw::Absent,
        },
        Route::TierStage => match (codec, k1, k2) {
            (0, _, _) => Raw::ArchNeutral(
                "aggregate QT44/QT53 grouped route (MoeRouteCapability::Qt44Qt53Grouped), not arch-gated",
            ),
            (_, Some(k1), Some(k2)) => Raw::Composite { k1, k2 },
            _ => Raw::Absent,
        },
        Route::D2CpuSplit => Raw::Absent,
    }
}

/// The qualified cell for `(arch, qt, route)` under the shipped holds.
pub fn capability(arch: &str, qt: ExpertQt, route: Route) -> Capability {
    capability_with_external(arch, qt, route, None, &Holds::BETA)
}

/// [`capability`] with explicit holds.
pub fn capability_with(arch: &str, qt: ExpertQt, route: Route, holds: &Holds) -> Capability {
    capability_with_external(arch, qt, route, None, holds)
}

/// [`capability`] with explicit holds and the caller-owned inventory of [`Route::D2CpuSplit`].
pub fn capability_with_external(
    arch: &str,
    qt: ExpertQt,
    route: Route,
    external: Option<&'static str>,
    holds: &Holds,
) -> Capability {
    let raw = raw_inventory_with_external(arch, qt, route, external);
    let verdict = if !raw.is_present() {
        Verdict::NoKernel
    } else if let Some(row) = HOLD_CELLS.iter().find(|row| {
        row.arch == arch && row.qt == qt && row.route == route && row.hold.active(holds)
    }) {
        Verdict::Held(row.hold)
    } else {
        Verdict::Admitted
    };
    Capability { raw, verdict }
}

/// Whether `(arch, qt, route)` is admitted under the shipped holds.
pub fn admitted(arch: &str, qt: ExpertQt, route: Route) -> bool {
    capability(arch, qt, route).admitted()
}

/// Whether the sealed slab serves `qt` on `arch`: its decode K1 and prefill K2 are both admitted
/// (the staged arena reuses them, so [`Route::Stage`] follows).
pub fn slab_serves_with(arch: &str, qt: ExpertQt, holds: &Holds) -> bool {
    capability_with(arch, qt, Route::DecodeGemv, holds).admitted()
        && capability_with(arch, qt, Route::PrefillK2, holds).admitted()
}

/// [`slab_serves_with`] under the shipped holds.
pub fn slab_serves(arch: &str, qt: ExpertQt) -> bool {
    slab_serves_with(arch, qt, &Holds::BETA)
}

/// [`slab_serves_with`] keyed by slab codec byte; an unknown byte is refused.
pub fn slab_serves_codec_with(arch: &str, codec: u8, holds: &Holds) -> bool {
    ExpertQt::of_slab_codec(codec).is_some_and(|qt| slab_serves_with(arch, qt, holds))
}

/// [`slab_serves_codec_with`] under the shipped holds.
pub fn slab_serves_codec(arch: &str, codec: u8) -> bool {
    slab_serves_codec_with(arch, codec, &Holds::BETA)
}

/// Whether a slab artifact may load on `arch`: the architecture serves MQ4 and its artifact-load
/// receipts flag is set ([`Holds::shared_receipts_landed`] for gfx1100 / gfx1201,
/// [`Holds::gfx1151_receipts_landed`] for gfx1151). The flags gate the artifact load only, never
/// a kernel cell.
pub fn slab_artifact_route_admitted_with(arch: &str, holds: &Holds) -> bool {
    slab_serves_with(arch, QT_MQ4_XTS, holds)
        && match arch {
            "gfx1151" => holds.gfx1151_receipts_landed,
            "gfx1100" | "gfx1201" => holds.shared_receipts_landed,
            _ => false,
        }
}

/// [`slab_artifact_route_admitted_with`] under the shipped holds.
pub fn slab_artifact_route_admitted(arch: &str) -> bool {
    slab_artifact_route_admitted_with(arch, &Holds::BETA)
}

/// Explicit hold list active under `holds`: every `(arch, qt, route, hold)` whose cell is
/// registry-present and refused.
pub fn held_cells(holds: &Holds) -> Vec<HoldCell> {
    HOLD_CELLS
        .iter()
        .copied()
        .filter(|row| {
            capability_with(row.arch, row.qt, row.route, holds).verdict == Verdict::Held(row.hold)
        })
        .collect()
}

/// A 1,336,934,400-byte layer copy takes 23.5 ms (card B, gfx1201). A 2048-row forward already
/// hides it: pp2048 went from 1057 to 1377 tok/s, and the restore that was not hidden cost 18 ms
/// of that chunk. Smaller forwards keep the mapped path (K3: 1536 rows expose the DMA). This is
/// the ceiling of the default threshold: an arch whose prefill chunk is smaller (gfx1100: 1536)
/// stages its full chunk instead of never staging ([`stage_default_min_rows`]).
pub const STAGE_MIN_ROWS_DEFAULT: usize = 2048;
/// Default stage threshold on a narrow discrete host link off an [`StageDefault::Always`] arch:
/// the mapped read of a host layer costs more per forward the narrower the link, so the copy
/// pays from a shorter forward. A fixed floor, not capped at the chunk: a forward shorter than
/// this (or a chunk admitted below it) never stages and keeps the zero-copy mapped reads.
pub const STAGE_NARROW_LINK_MIN_ROWS: usize = 768;
/// gfx1100 prefill chunk rows on a measured narrow discrete host link: every chunk re-streams
/// the host experts through the narrow link, so the chunk's row count divides that cost, and G2
/// staging hides the copy behind the chunk's compute.
pub const GFX1100_NARROW_LINK_PREFILL_CHUNK_ROWS: usize = 8192;

/// When the long-prefill expert stage is on by default (`HIPFIRE_QWEN4_EXPERT_STAGE` unset).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageDefault {
    /// On whatever the link; the link is never evaluated (gfx1201, the card it was measured on).
    Always,
    /// On exactly when the measured discrete link is narrow (off for unified or unmeasured).
    OnNarrowLink,
}

/// How the stage `min_rows` default is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageMinRows {
    /// A fixed threshold whatever the link or chunk.
    Fixed(usize),
    /// `narrow_link_rows` on a narrow link, else `min(cap, chunk_rows)` (at least 2).
    NarrowLinkElseChunkCapped { narrow_link_rows: usize, cap: usize },
}

/// Per-architecture expert stage policy metadata (the old scattered `arch == "gfx1201"` /
/// `"gfx1100"` tests). This is the DEFAULT switch only. Whether a given codec may be staged on
/// the arch at all is the separate [`Route::Stage`] capability ([`stage_eligible`]), checked where
/// the stage is constructed (`Qwen4ExpertStage::new_slab`); a stage is built only when both the
/// eligibility holds and this policy (or an explicit `HIPFIRE_QWEN4_EXPERT_STAGE`) switches it on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagePolicy {
    pub default: StageDefault,
    pub min_rows: StageMinRows,
    /// Prefill chunk rows used instead of the arch baseline on a measured narrow discrete link.
    pub narrow_link_prefill_chunk_rows: Option<usize>,
    /// Whether an expert-parallel rank may build the stage at all (gfx1201 only).
    pub expert_parallel_stage: bool,
}

/// The stage policy of `arch`. gfx1201: always on, fixed 2048 rows, EP stage admitted. gfx1100:
/// narrow-link default, 8192-row chunk on a narrow link. Every other architecture, known or not:
/// narrow-link default (the borrowed-donor stage of any discrete device), no chunk override, no EP
/// stage.
pub fn stage_policy(arch: &str) -> StagePolicy {
    match arch {
        "gfx1201" => StagePolicy {
            default: StageDefault::Always,
            min_rows: StageMinRows::Fixed(STAGE_MIN_ROWS_DEFAULT),
            narrow_link_prefill_chunk_rows: None,
            expert_parallel_stage: true,
        },
        _ => StagePolicy {
            default: StageDefault::OnNarrowLink,
            min_rows: StageMinRows::NarrowLinkElseChunkCapped {
                narrow_link_rows: STAGE_NARROW_LINK_MIN_ROWS,
                cap: STAGE_MIN_ROWS_DEFAULT,
            },
            narrow_link_prefill_chunk_rows: (arch == "gfx1100")
                .then_some(GFX1100_NARROW_LINK_PREFILL_CHUNK_ROWS),
            expert_parallel_stage: false,
        },
    }
}

/// Stage ELIGIBILITY: every quant pair in `qts` has an admitted [`Route::Stage`] cell on `arch`
/// (K1 and K2 present, no hold). Independent of the default switch ([`stage_default_enabled`]):
/// it is checked where the slab stage is constructed, so a measured link or an explicit `1`
/// never stages a codec the arch has no kernels for.
pub fn stage_eligible(arch: &str, qts: impl IntoIterator<Item = ExpertQt>) -> bool {
    qts.into_iter().all(|qt| admitted(arch, qt, Route::Stage))
}

/// Default of the stage switch (policy plus the measured-link predicate only; it says nothing
/// about which codec may be staged, see [`stage_eligible`]): `narrow_link` is evaluated only when
/// the arch policy reads the link ([`StageDefault::OnNarrowLink`]), so an `Always` arch never
/// probes. Unknown architectures keep the narrow-link default: the packed (aggregate) borrowed
/// stage is arch-neutral, and a slab stage on an arch without kernels is refused at construction
/// by the Stage eligibility, as it was by the artifact load before.
pub fn stage_default_enabled(arch: &str, narrow_link: impl FnOnce() -> bool) -> bool {
    match stage_policy(arch).default {
        StageDefault::Always => true,
        StageDefault::OnNarrowLink => narrow_link(),
    }
}

/// Whether the stage policy of `arch` reads the measured link at all (false = never probe).
pub fn stage_reads_link(arch: &str) -> bool {
    stage_policy(arch).default != StageDefault::Always
}

/// The default stage `min_rows` of `arch` for a forward chunk of `chunk_rows` rows.
pub fn stage_default_min_rows(arch: &str, chunk_rows: usize, narrow_link: bool) -> usize {
    match stage_policy(arch).min_rows {
        StageMinRows::Fixed(rows) => rows,
        StageMinRows::NarrowLinkElseChunkCapped {
            narrow_link_rows,
            cap,
        } => {
            if narrow_link {
                narrow_link_rows
            } else {
                cap.min(chunk_rows.max(2))
            }
        }
    }
}

/// Prefill chunk rows of `arch` on a narrow discrete link, if the policy overrides the baseline.
pub fn narrow_link_prefill_chunk_rows(arch: &str) -> Option<usize> {
    stage_policy(arch).narrow_link_prefill_chunk_rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_registry::SUPPORTED_ARCHES;

    /// Arches walked: the registry arches plus names that must never match.
    fn arches() -> Vec<&'static str> {
        let mut arches = SUPPORTED_ARCHES.to_vec();
        arches.extend(["", "gfx1200", "gfx1201x", "gfx11000"]);
        arches
    }

    /// The beta answer of `(arch, qt, route)`, written out independently of the table.
    fn expected(arch: &str, qt: ExpertQt, route: Route, external: bool) -> bool {
        let slab = match (arch, qt) {
            ("gfx1201", QT_MQ4_XTS | QT_W3 | QT_E8P) => true,
            ("gfx1151", QT_MQ4_XTS | QT_W3 | QT_E8P) => true,
            ("gfx1100", QT_MQ4_XTS | QT_E8P) => true,
            _ => false,
        };
        match route {
            Route::PrefillK2 | Route::DecodeGemv | Route::Stage => slab,
            Route::TierStage => match qt {
                QT_MQ4_XTS => true,
                QT_W3 => arch == "gfx1201",
                _ => false,
            },
            Route::D2CpuSplit => external && qt == QT_MQ4_XTS,
        }
    }

    #[test]
    fn beta_truth_table_for_every_arch_pair_and_route() {
        // Every one of the 65,536 tag pairs x every arch x every route.
        for arch in arches() {
            for gate_up in 0..=255u8 {
                for down in 0..=255u8 {
                    let qt = ExpertQt::new(gate_up, down);
                    for route in Route::ALL {
                        for external in [false, true] {
                            let ext = external.then_some("cpu");
                            let got = capability_with_external(arch, qt, route, ext, &Holds::BETA);
                            assert_eq!(
                                got.admitted(),
                                expected(arch, qt, route, external),
                                "{arch:?} {qt:?} {route:?} external={external}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn registry_pairs_have_the_documented_verdicts() {
        use Hold::{TierPlanE8pRefused, TierPlanW3Gfx1151Refused};
        use Verdict::{Admitted, Held, NoKernel};
        let all_slab = [Route::PrefillK2, Route::DecodeGemv, Route::Stage];
        for arch in arches() {
            for qt in REGISTRY_PAIRS {
                for route in all_slab {
                    let want = match (arch, qt) {
                        ("gfx1201" | "gfx1151", QT_MQ4_XTS | QT_W3 | QT_E8P)
                        | ("gfx1100", QT_MQ4_XTS | QT_E8P) => Admitted,
                        _ => NoKernel,
                    };
                    assert_eq!(
                        capability(arch, qt, route).verdict,
                        want,
                        "{arch:?} {qt:?} {route:?}"
                    );
                }
                let tier = match (arch, qt) {
                    (_, QT_MQ4_XTS) => Admitted,
                    ("gfx1201", QT_W3) => Admitted,
                    ("gfx1151", QT_W3) => Held(TierPlanW3Gfx1151Refused),
                    ("gfx1100" | "gfx1151" | "gfx1201", QT_E8P) => Held(TierPlanE8pRefused),
                    _ => NoKernel,
                };
                assert_eq!(
                    capability(arch, qt, Route::TierStage).verdict,
                    tier,
                    "{arch:?} {qt:?} tier"
                );
            }
        }
        // W2 has no kernel anywhere; gfx1100 W3 has none either.
        for arch in arches() {
            for route in Route::ALL {
                assert_eq!(capability(arch, QT_W2, route).verdict, NoKernel);
            }
        }
        for route in [
            Route::PrefillK2,
            Route::DecodeGemv,
            Route::Stage,
            Route::TierStage,
        ] {
            assert_eq!(capability("gfx1100", QT_W3, route).verdict, NoKernel);
        }
    }

    #[test]
    fn the_hold_list_is_exactly_the_registry_present_refused_cells() {
        // Beta: exactly the four tier-plan cells.
        let beta = held_cells(&Holds::BETA);
        assert_eq!(
            beta.iter()
                .map(|row| (row.arch, row.qt, row.route, row.hold))
                .collect::<Vec<_>>(),
            vec![
                (
                    "gfx1100",
                    QT_E8P,
                    Route::TierStage,
                    Hold::TierPlanE8pRefused
                ),
                (
                    "gfx1151",
                    QT_E8P,
                    Route::TierStage,
                    Hold::TierPlanE8pRefused
                ),
                (
                    "gfx1201",
                    QT_E8P,
                    Route::TierStage,
                    Hold::TierPlanE8pRefused
                ),
                (
                    "gfx1151",
                    QT_W3,
                    Route::TierStage,
                    Hold::TierPlanW3Gfx1151Refused
                ),
            ]
        );
        // Independent derivation: every Held verdict over the whole enumeration is a listed row,
        // and every listed row's cell has raw kernels (it is registry-present).
        for arch in arches() {
            for qt in REGISTRY_PAIRS {
                for route in Route::ALL {
                    let cap = capability_with_external(arch, qt, route, Some("cpu"), &Holds::BETA);
                    if let Verdict::Held(hold) = cap.verdict {
                        assert!(
                            beta.iter()
                                .any(|row| (row.arch, row.qt, row.route, row.hold)
                                    == (arch, qt, route, hold)),
                            "{arch:?} {qt:?} {route:?} {hold:?} is held but not listed"
                        );
                        assert!(cap.raw.is_present());
                    }
                }
            }
        }
        for row in HOLD_CELLS {
            assert!(
                raw_inventory_with_external(row.arch, row.qt, row.route, Some("cpu")).is_present(),
                "{row:?}"
            );
        }
    }

    #[test]
    fn the_gfx1151_e8p_switch_holds_exactly_its_cells() {
        let off = Holds {
            gfx1151_e8p_landed: false,
            ..Holds::BETA
        };
        for route in [Route::PrefillK2, Route::DecodeGemv, Route::Stage] {
            assert_eq!(
                capability_with("gfx1151", QT_E8P, route, &off).verdict,
                Verdict::Held(Hold::Gfx1151E8pReceiptsPending)
            );
            // Other arches and codecs never read the switch.
            for (arch, qt) in [
                ("gfx1201", QT_E8P),
                ("gfx1100", QT_E8P),
                ("gfx1151", QT_W3),
                ("gfx1151", QT_MQ4_XTS),
            ] {
                assert!(
                    capability_with(arch, qt, route, &off).admitted(),
                    "{arch} {qt:?} {route:?}"
                );
            }
        }
        assert!(!slab_serves_codec_with("gfx1151", 3, &off));
        assert!(
            slab_serves_codec_with("gfx1151", 0, &off)
                && slab_serves_codec_with("gfx1151", 1, &off)
        );
        assert_eq!(held_cells(&off).len(), 4 + 3);
        assert_eq!(
            capability_with_external("gfx1151", QT_MQ4_XTS, Route::D2CpuSplit, Some("cpu"), &off)
                .verdict,
            Verdict::Admitted
        );
        assert_eq!(
            capability_with_external("gfx1151", QT_E8P, Route::D2CpuSplit, Some("cpu"), &off)
                .verdict,
            Verdict::NoKernel
        );
    }

    #[test]
    fn raw_inventory_never_reads_holds_or_admission() {
        let off = Holds {
            gfx1151_e8p_landed: false,
            shared_receipts_landed: false,
            gfx1151_receipts_landed: false,
        };
        for arch in arches() {
            for qt in REGISTRY_PAIRS {
                for route in Route::ALL {
                    assert_eq!(
                        capability_with(arch, qt, route, &off).raw,
                        capability_with(arch, qt, route, &Holds::BETA).raw
                    );
                    assert_eq!(
                        raw_inventory(arch, qt, route),
                        capability(arch, qt, route).raw
                    );
                }
            }
        }
        // The raw K2 inventory covers the trunk MQ4 / W3 modules, not only the slab K2 table.
        assert_eq!(
            raw_inventory("gfx1201", QT_MQ4_XTS, Route::PrefillK2),
            Raw::Kernel(RawKernel {
                module: "qwen4_moe_iu4_sym_pm_gfx1201",
                symbols: [
                    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
                    "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4"
                ],
            })
        );
        assert!(matches!(
            raw_inventory("gfx1151", QT_W3, Route::PrefillK2),
            Raw::Kernel(RawKernel {
                module: "qwen4_moe_iu4_sym_w3_nt4_pm_gfx1151",
                ..
            })
        ));
    }

    #[test]
    fn artifact_receipts_flags_gate_the_load_only() {
        assert!(Holds::BETA.shared_receipts_landed && Holds::BETA.gfx1151_receipts_landed);
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            assert!(slab_artifact_route_admitted(arch), "{arch}");
        }
        for arch in ["gfx906", "gfx942", "", "gfx1200"] {
            assert!(!slab_artifact_route_admitted(arch), "{arch:?}");
        }
        let no_shared = Holds {
            shared_receipts_landed: false,
            ..Holds::BETA
        };
        assert!(!slab_artifact_route_admitted_with("gfx1201", &no_shared));
        assert!(!slab_artifact_route_admitted_with("gfx1100", &no_shared));
        assert!(slab_artifact_route_admitted_with("gfx1151", &no_shared));
        let no_halo = Holds {
            gfx1151_receipts_landed: false,
            ..Holds::BETA
        };
        assert!(!slab_artifact_route_admitted_with("gfx1151", &no_halo));
        assert!(slab_artifact_route_admitted_with("gfx1201", &no_halo));
        // The flags never refuse a kernel cell.
        assert!(capability_with("gfx1201", QT_W3, Route::PrefillK2, &no_shared).admitted());
    }

    #[test]
    fn the_codec_byte_matrix_is_the_old_admission_matrix() {
        // The pre-central admission matrix, spelled out: codec ids 0 / 1 / 3 per arch.
        for arch in arches() {
            for codec in 0..=255u8 {
                let old = match arch {
                    "gfx1201" => matches!(codec, 0 | 1 | 3),
                    "gfx1151" => matches!(codec, 0 | 1 | 3),
                    "gfx1100" => matches!(codec, 0 | 3),
                    _ => false,
                };
                assert_eq!(slab_serves_codec(arch, codec), old, "{arch:?} {codec}");
            }
        }
    }

    #[test]
    fn qt_codec_and_dtype_mappings_agree() {
        for (qt, codec) in [(QT_MQ4_XTS, 0), (QT_W3, 1), (QT_W2, 2), (QT_E8P, 3)] {
            assert_eq!(qt.slab_codec(), Some(codec));
            assert_eq!(ExpertQt::of_slab_codec(codec), Some(qt));
        }
        assert_eq!(ExpertQt::of_slab_codec(4), None);
        assert_eq!(ExpertQt::new(44, 132).slab_codec(), None);
        assert_eq!(
            ExpertQt::of_dtypes(DType::MQ4G256V2, DType::MQ4G128V2),
            Some(QT_MQ4_XTS)
        );
        assert_eq!(
            ExpertQt::of_dtypes(DType::MQ3I4G128R256, DType::MQ3I4G128R128),
            Some(QT_W3)
        );
        assert_eq!(
            ExpertQt::of_dtypes(DType::MQ2I4G128R256, DType::MQ2I4G128R128),
            Some(QT_W2)
        );
        assert_eq!(
            ExpertQt::of_dtypes(DType::E8PI4G128R256, DType::E8PI4G128R128),
            Some(QT_E8P)
        );
        assert_eq!(
            ExpertQt::of_dtypes(DType::MQ4G256V2, DType::MQ3I4G128R128),
            None
        );
        assert_eq!(ExpertQt::of_dtypes(DType::MQ4G256, DType::MQ4G256), None);
    }

    #[test]
    fn generation_labels_are_display_only_and_cover_every_registry_tag() {
        for qt in REGISTRY_PAIRS {
            assert!(generation_label(qt.gate_up).is_some(), "{qt:?}");
            assert!(generation_label(qt.down).is_some(), "{qt:?}");
        }
        assert_eq!(generation_label(13), Some("v1"));
        assert_eq!(generation_label(44), Some("v3"));
        assert_eq!(generation_label(53), Some("v3"));
        assert_eq!(generation_label(255), None);
    }

    #[test]
    fn stage_policy_keeps_the_scattered_arch_defaults() {
        // gfx1201: on whatever the link (never probed), fixed 2048, no chunk override.
        assert!(stage_default_enabled("gfx1201", || unreachable!("probed")));
        assert!(!stage_reads_link("gfx1201"));
        assert_eq!(stage_default_min_rows("gfx1201", 4096, true), 2048);
        assert_eq!(stage_default_min_rows("gfx1201", 512, false), 2048);
        assert_eq!(narrow_link_prefill_chunk_rows("gfx1201"), None);
        // Every other arch, known or not: the narrow-link predicate decides.
        for arch in ["gfx1100", "gfx1151", "gfx942", "gfx906", "", "gfx1200"] {
            assert!(stage_reads_link(arch));
            assert!(stage_default_enabled(arch, || true), "{arch:?}");
            assert!(!stage_default_enabled(arch, || false), "{arch:?}");
            assert_eq!(stage_default_min_rows(arch, 8192, true), 768);
            assert_eq!(stage_default_min_rows(arch, 8192, false), 2048);
            assert_eq!(stage_default_min_rows(arch, 1536, false), 1536);
            assert_eq!(stage_default_min_rows(arch, 0, false), 2);
        }
        assert_eq!(narrow_link_prefill_chunk_rows("gfx1100"), Some(8192));
        for arch in ["gfx1151", "gfx942", ""] {
            assert_eq!(narrow_link_prefill_chunk_rows(arch), None);
        }
        // Only gfx1201 builds the stage for an expert-parallel rank.
        for arch in ["gfx1100", "gfx1151", "gfx942", "gfx906", ""] {
            assert!(!stage_policy(arch).expert_parallel_stage, "{arch:?}");
        }
        assert!(stage_policy("gfx1201").expert_parallel_stage);
    }

    #[test]
    fn stage_eligibility_is_the_stage_capability_not_the_default_switch() {
        // Eligibility follows the Stage cells: slab-served codecs only, independent of the
        // (link-dependent) default switch.
        for (arch, qt, want) in [
            ("gfx1201", QT_MQ4_XTS, true),
            ("gfx1201", QT_W3, true),
            ("gfx1201", QT_E8P, true),
            ("gfx1151", QT_W3, true),
            ("gfx1100", QT_E8P, true),
            ("gfx1100", QT_W3, false),
            ("gfx942", QT_MQ4_XTS, false),
            ("", QT_MQ4_XTS, false),
            ("gfx1201", QT_W2, false),
        ] {
            assert_eq!(stage_eligible(arch, [qt]), want, "{arch:?} {qt:?}");
            assert_eq!(
                stage_eligible(arch, [qt]),
                slab_serves(arch, qt),
                "{arch:?} {qt:?}"
            );
        }
        assert!(stage_eligible("gfx1201", [QT_MQ4_XTS, QT_W3]));
        assert!(!stage_eligible("gfx1100", [QT_MQ4_XTS, QT_W3]));
    }
}
