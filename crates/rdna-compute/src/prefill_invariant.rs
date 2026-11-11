// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batch-invariant Flash-Next prefill profile.
//!
//! Today several kernels on the qwen4 prefill path are routed by the row count
//! `n` of the forward (thresholds 8 / 512 / 2048 / `x_src_rows >= 5120`, ...),
//! so a row's bytes depend on how many other rows share its forward.  Under
//! the invariant profile every row-count ROUTING decision is taken as if `n`
//! were [`QWEN4_PREFILL_CANON_ROWS`] (the gfx1151 default prefill chunk, above
//! every threshold) while launches keep using the real `n`.  At `n == 8192`
//! the profile therefore equals the product route byte for byte.
//!
//! The profile is opt-in (`HIPFIRE_PREFILL_PM_INVARIANT`) and only engaged by
//! the qwen4 forward on gfx1151 via `Gpu::prefill_invariant_scope`.
//!
//! The same canonical-routing switch is also engaged per tile by the agent-turn
//! append route (`HIPFIRE_QWEN4_APPEND_WMMA`, on by default, exact gfx1151,
//! prefill tiles of `HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS..512` rows), which
//! additionally suppresses the symmetric IU4 MoE arm for that tile
//! (`Gpu::qwen4_append_f16_moe_scope`).

use crate::dispatch::Gpu;
use crate::gemv::WideTwinFamily;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

/// Row count every routing predicate sees under the invariant profile.
pub const QWEN4_PREFILL_CANON_ROWS: usize = 8192;

static PREFILL_PM_INVARIANT: LazyLock<bool> =
    LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_PREFILL_PM_INVARIANT", false));

/// `HIPFIRE_PREFILL_PM_INVARIANT` (developer bool, default off), read once.
pub fn prefill_pm_invariant_enabled() -> bool {
    *PREFILL_PM_INVARIANT
}

/// Lab selector for the qwen4 prefill GDN recurrence
/// (`HIPFIRE_QWEN4_GDN_PREFILL`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4GdnPrefill {
    /// Product route: WY chunked recurrence at >= 512 rows, persistent256
    /// below.
    Wy,
    /// Always the persistent256 recurrence (launch-end Q8 requant).
    Persistent,
    /// Persistent256 with the token-canonical Q8 requant (every row requants
    /// at its own position), independent of how rows split into launches.
    Canon,
}

static QWEN4_GDN_PREFILL: LazyLock<Qwen4GdnPrefill> = LazyLock::new(|| {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_GDN_PREFILL")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "persistent" => Qwen4GdnPrefill::Persistent,
        "canon" => Qwen4GdnPrefill::Canon,
        _ => Qwen4GdnPrefill::Wy,
    }
});

/// `HIPFIRE_QWEN4_GDN_PREFILL` = `wy` (default) | `persistent` | `canon`,
/// read once.
pub fn qwen4_gdn_prefill() -> Qwen4GdnPrefill {
    *QWEN4_GDN_PREFILL
}

/// Default of `HIPFIRE_QWEN4_APPEND_WMMA`: on (KLD, coherence and timing gates
/// on Strix Halo in the 0.4.3 CHANGELOG).
pub const QWEN4_APPEND_WMMA_DEFAULT: bool = true;
/// Default of `HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS`: above the measured
/// F32/append-route crossover (the route loses at 33 rows, wins at 55 and 81).
pub const QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT: usize = 48;

static QWEN4_APPEND_WMMA: LazyLock<bool> = LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_APPEND_WMMA", QWEN4_APPEND_WMMA_DEFAULT)
});

/// `HIPFIRE_QWEN4_APPEND_WMMA` (developer bool, default [`QWEN4_APPEND_WMMA_DEFAULT`]), read once.
pub fn qwen4_append_wmma_enabled() -> bool {
    *QWEN4_APPEND_WMMA
}

static QWEN4_APPEND_WMMA_MIN_ROWS: LazyLock<usize> = LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 2)
        .unwrap_or(QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT)
});

/// `HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS` (developer var, default
/// [`QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT`]; values below 2 are ignored), read once.
pub fn qwen4_append_wmma_min_rows() -> usize {
    *QWEN4_APPEND_WMMA_MIN_ROWS
}

/// Whether a qwen4 forward tile takes the agent-turn append route: an
/// enabled prefill tile of `min_rows..QWEN4_F16_WMMA_MIN_TOKENS` rows on
/// exact gfx1151 that is not already under the invariant profile. Decode,
/// MTP verify, other arches, tiles of 512+ rows and the
/// `HIPFIRE_PREFILL_PM_INVARIANT` profile (whose MoE must route the same at
/// every row count) keep their route.
pub fn qwen4_append_wmma_route(
    enabled: bool,
    prefill: bool,
    arch: &str,
    rows: usize,
    min_rows: usize,
    pm_invariant: bool,
) -> bool {
    enabled
        && prefill
        && !pm_invariant
        && arch == "gfx1151"
        && rows >= min_rows
        && rows < crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
}

/// The historic few-row limit: routes gated on `route_rows <= 8` are the
/// per-row twins of the decode GEMV (each output row runs the decode kernel's
/// lane->K map, FMA chain and reduction tree). Shared with `hipfire-config`,
/// whose `auto` MTP cap derives from the same row-exact rows.
pub const FEW_ROW_ROUTE_ROWS: usize = hipfire_config::QWEN4_FEW_ROW_ROUTE_ROWS;
/// Physical verify capacity, independent of the proven per-architecture
/// admission table. Projection twins use fixed eight-row tiles through this
/// bound; this constant alone does not qualify a route.
pub const WIDE_VERIFY_MAX_ROWS: usize = hipfire_config::QWEN4_VERIFY_MAX_ROWS;
/// gfx1151 row-exact verify rows (owned by `hipfire-config`); raised only
/// after the row-exactness oracle proves the arch at that many rows.
pub const QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1151: usize =
    hipfire_config::QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1151;
/// gfx1201 row-exact verify rows (owned by `hipfire-config`); raised only
/// after the row-exactness oracle proves the arch at that many rows.
pub const QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1201: usize =
    hipfire_config::QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1201;
/// gfx942 row-exact verify rows (owned by `hipfire-config`): the M-K-measured
/// default-route chunk == single-row decode width the drafter serves.
pub const QWEN4_ROW_EXACT_VERIFY_ROWS_GFX942: usize =
    hipfire_config::QWEN4_ROW_EXACT_VERIFY_ROWS_GFX942;

/// Parse `HIPFIRE_QWEN4_WIDE_VERIFY_ROWS`: an unsigned integer clamped to
/// [`FEW_ROW_ROUTE_ROWS`]..=[`WIDE_VERIFY_MAX_ROWS`]; unset or unparsable is
/// `None`.
fn wide_verify_rows_override_value(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .map(|n| n.clamp(FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS))
}

static QWEN4_WIDE_VERIFY_ROWS_OVERRIDE: LazyLock<Option<usize>> = LazyLock::new(|| {
    wide_verify_rows_override_value(
        hipfire_config::developer_var("HIPFIRE_QWEN4_WIDE_VERIFY_ROWS")
            .ok()
            .as_deref(),
    )
});

/// Runtime override of the verify rows (`0` = unset); see
/// [`set_qwen4_wide_verify_rows_override`].
static QWEN4_WIDE_VERIFY_ROWS_RUNTIME: AtomicUsize = AtomicUsize::new(0);

/// Oracle/test hook: set (`Some(n)`, clamped to
/// `FEW_ROW_ROUTE_ROWS..=WIDE_VERIFY_MAX_ROWS`) or clear (`None`) a runtime
/// override of [`qwen4_row_exact_verify_rows`]. It takes precedence over
/// `HIPFIRE_QWEN4_WIDE_VERIFY_ROWS` and the per-arch table, so one process
/// can run several verify widths. Not for production routes.
pub fn set_qwen4_wide_verify_rows_override(rows: Option<usize>) {
    let value = rows.map_or(0, |n| n.clamp(FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS));
    QWEN4_WIDE_VERIFY_ROWS_RUNTIME.store(value, Ordering::Relaxed);
}

/// Precedence of the verify-rows sources: runtime override (`0` = unset),
/// then the env override, then the table.
fn resolve_verify_rows(arch: &str, runtime: usize, env: Option<usize>, table: usize) -> usize {
    if arch == "gfx942" {
        // The table is authoritative here: neither lab override (RDNA
        // wide-verify oracles) widens gfx942.
        return table.clamp(FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS);
    }
    if !matches!(arch, "gfx1151" | "gfx1201" | "gfx1100") {
        return FEW_ROW_ROUTE_ROWS;
    }
    let rows = if runtime != 0 {
        runtime
    } else {
        env.unwrap_or(table)
    };
    rows.clamp(FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS)
}

/// Per-arch proven row-exact verify rows (other arches
/// [`FEW_ROW_ROUTE_ROWS`]). The runtime override
/// ([`set_qwen4_wide_verify_rows_override`]) wins, then
/// `HIPFIRE_QWEN4_WIDE_VERIFY_ROWS` (developer var, read once, clamped to
/// `FEW_ROW_ROUTE_ROWS..=WIDE_VERIFY_MAX_ROWS`), then the table; both
/// overrides apply on exact gfx1151/gfx1201/gfx1100 only (oracle runs).
/// Whether the existing wide-verify oracle override is present for `arch`:
/// the in-process runtime override ([`set_qwen4_wide_verify_rows_override`],
/// nonzero = set, including the `Some(8)` baseline) or the parsed
/// `HIPFIRE_QWEN4_WIDE_VERIFY_ROWS` developer override is set — on an exact
/// oracle arch (`gfx1151`/`gfx1201`/`gfx1100`) only. Presence, not value: an
/// unparsable env var parses to `None` and MUST NOT arm. No new knob; the
/// existing getter/setter are unchanged. This is the required real-oracle
/// consumer signal (the `qwen4_wide_verify` arms use the process runtime
/// setter), not a production-override promotion: production without either
/// override reads `false` and keeps the default route.
/// Callers combine this with their own eager-verify scope; the CPU routing
/// slice additionally requires allocated storage before admitting wide rows.
pub fn qwen4_wide_verify_override_present(arch: &str) -> bool {
    matches!(arch, "gfx1151" | "gfx1201" | "gfx1100")
        && (QWEN4_WIDE_VERIFY_ROWS_RUNTIME.load(Ordering::Relaxed) != 0
            || (*QWEN4_WIDE_VERIFY_ROWS_OVERRIDE).is_some())
}

pub fn qwen4_row_exact_verify_rows(arch: &str) -> usize {
    resolve_verify_rows(
        arch,
        QWEN4_WIDE_VERIFY_ROWS_RUNTIME.load(Ordering::Relaxed),
        *QWEN4_WIDE_VERIFY_ROWS_OVERRIDE,
        hipfire_config::qwen4_row_exact_verify_rows_table(arch),
    )
}

/// Singleton AR arithmetic latched at the beginning of an eager verify window.
/// This does not start recording or capture; `None` retains normal launch policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4VerifyReference {
    Eager,
    Recorded,
}

/// Largest `route_rows` a few-row (decode-exact) route serves given the
/// current verify rows (`0` = not in a verify forward) and whether launches
/// are eager. Unknown/unqualified architectures and scopes above the physical
/// bound stay conservative. Recording/capture never widens the historic limit.
#[inline]
fn few_row_limit_for(arch: &str, wide_verify_rows: usize, eager: bool) -> usize {
    if eager
        && matches!(arch, "gfx1151" | "gfx1201" | "gfx1100")
        && wide_verify_rows <= WIDE_VERIFY_MAX_ROWS
    {
        FEW_ROW_ROUTE_ROWS.max(wide_verify_rows)
    } else {
        FEW_ROW_ROUTE_ROWS
    }
}

/// Architecture support is not shape admission. Each launcher still validates
/// its singleton recipe, geometry and extents before binding a tile symbol.
fn row_tile_family_supported(arch: &str, family: WideTwinFamily) -> bool {
    match arch {
        "gfx1151" => true,
        // Halo Q8's decode recipe is not the gfx1201 singleton.
        "gfx1201" => family != WideTwinFamily::Q8,
        // gfx1100 uses dedicated singleton-derived recipes, never Halo aliases.
        // In particular Q8 launchers must enforce their source-proven K domain.
        "gfx1100" => true,
        _ => false,
    }
}

fn row_tile_reference_supported(arch: &str, gemv_rows: u32, pm_decode: bool) -> bool {
    match arch {
        "gfx1151" => gemv_rows == 2,
        "gfx1201" => gemv_rows == 2 && pm_decode,
        "gfx1100" => gemv_rows == 1,
        _ => false,
    }
}

fn row_tile_scope_enabled(
    rows: usize,
    wide_verify_rows: usize,
    eager: bool,
    prefill_invariant: bool,
) -> bool {
    eager
        && !prefill_invariant
        && (1..=WIDE_VERIFY_MAX_ROWS).contains(&wide_verify_rows)
        && rows != 0
        && rows <= wide_verify_rows
}

/// `(first_row, row_count)` groups a wide few-row launcher serves `rows` with:
/// groups of [`FEW_ROW_ROUTE_ROWS`] and the remainder last. Each group is the
/// already row-exact <=8-row launch, so every row stays bitwise the decode row
/// whatever `rows` is.
pub(crate) fn few_row_groups(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows)
        .step_by(FEW_ROW_ROUTE_ROWS)
        .map(move |start| (start, FEW_ROW_ROUTE_ROWS.min(rows - start)))
}

/// [`Gpu::qwen4_route_rows_cap`] of the exact EP joint prefill wave: one row under the 512-row
/// floor of every row-keyed route (F16 WMMA, symmetric IU4, INT8 shared expert, fused front,
/// BF16 streams), so a wide wave's MoE call takes the routes a sub-512-row lane's singleton takes.
pub const QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP: usize = crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS - 1;

/// The routing rows of a forward of `rows` rows ([`Gpu::route_rows`], pure): the canonical chunk
/// under the invariant profile, else `rows` clamped to `cap` when a routing-row cap is set
/// (`cap != 0`, [`Gpu::qwen4_route_rows_cap`]).
pub(crate) fn route_rows_of(invariant: bool, cap: usize, rows: usize) -> usize {
    if invariant {
        QWEN4_PREFILL_CANON_ROWS
    } else if cap != 0 {
        rows.min(cap)
    } else {
        rows
    }
}

/// `HIPFIRE_QWEN4_WIDE_GROUPED_MOE` (developer, default off, `1` on, read once): a
/// 9..=`few_row_limit` eager verify keeps the grouped, expert-unioned routed-MoE
/// route of the <=8-row call instead of the per-(row, slot) indexed route.
static QWEN4_WIDE_GROUPED_MOE: LazyLock<bool> =
    LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_QWEN4_WIDE_GROUPED_MOE", false));

/// `HIPFIRE_QWEN4_WIDE_GROUPED_DOWN` (developer, default off, `1` on, read once): on the
/// grouped routed-MoE route, a 9..=`few_row_limit` verify runs the grouped,
/// expert-unioned down (`gemm_mq4g128v2_moe_grouped_top10`) and the grouped combine
/// instead of the per-slot indexed down. Takes effect only where the routed chain is
/// grouped (`HIPFIRE_QWEN4_WIDE_GROUPED_MOE=1`).
static QWEN4_WIDE_GROUPED_DOWN: LazyLock<bool> =
    LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_QWEN4_WIDE_GROUPED_DOWN", false));

impl Gpu {
    /// See [`QWEN4_WIDE_GROUPED_DOWN`].
    #[inline]
    pub fn wide_grouped_down(&self) -> bool {
        *QWEN4_WIDE_GROUPED_DOWN
    }

    /// See [`QWEN4_WIDE_GROUPED_MOE`].
    #[inline]
    pub fn wide_grouped_moe(&self) -> bool {
        *QWEN4_WIDE_GROUPED_MOE
    }

    /// The typed numerical intent of the forward in flight (see [`PrefillRouteIntent`]).
    #[inline]
    pub fn prefill_route_intent(&self) -> PrefillRouteIntent {
        if self.ordinary_prefill_active() {
            PrefillRouteIntent::OrdinaryPrefill
        } else if self.prefill_invariant_scope {
            PrefillRouteIntent::InvariantProfile
        } else {
            PrefillRouteIntent::Product
        }
    }

    /// Row count to feed ROUTING predicates (never launch extents): the
    /// canonical prefill chunk under the invariant profile or the ordinary prefill
    /// intent ([`PrefillRouteIntent::canonical_routing`]), else `rows`
    /// (clamped to [`Gpu::qwen4_route_rows_cap`] while the exact EP wave sets it).
    /// The ordinary prefill intent wins over the cap: every call of an ordinary prefill,
    /// including a one-row tail, takes the bulk recipe.
    #[inline]
    pub fn route_rows(&self, rows: usize) -> usize {
        route_rows_of(
            self.prefill_route_intent().canonical_routing(),
            self.qwen4_route_rows_cap,
            rows,
        )
    }

    /// Largest `route_rows` a few-row (decode-exact) route serves right now:
    /// [`FEW_ROW_ROUTE_ROWS`], or inside a supported eager verify forward with
    /// a scope at most [`WIDE_VERIFY_MAX_ROWS`], `max(8, wide_verify_rows)`.
    /// Wide verify is eager-only:
    /// while a replay is recording or a graph is being captured
    /// (`replay.is_recording()` / `graphs.capture_mode`) the limit is
    /// [`FEW_ROW_ROUTE_ROWS`] even when `wide_verify_rows` is set. Under
    /// `prefill_invariant_scope` `route_rows` is already the canonical chunk,
    /// so the few-row gates stay closed there.
    #[inline]
    pub fn few_row_limit(&self) -> usize {
        few_row_limit_for(
            &self.arch,
            self.wide_verify_rows,
            !(self.replay.is_recording() || self.graphs.capture_mode),
        )
    }

    /// Whether this family's fixed-eight-row PM twin may serve an admitted
    /// eager verify. This is a lab family selector, not a capability promotion.
    /// Callers must additionally validate their architecture-specific shape.
    /// Product callers may keep narrow incumbents at <=8; direct byte oracles
    /// can exercise the same tile ABI at every logical width 1..=64.
    /// gfx1100 lab tiles use their own singleton-derived images; widening the
    /// lab scope does not promote its proven shipping table.
    #[inline]
    pub fn row_tile_twin_enabled(&self, family: WideTwinFamily, rows: usize) -> bool {
        family.enabled()
            && row_tile_family_supported(&self.arch, family)
            && (family != WideTwinFamily::Mq4 || self.qwen4_row_tile_reference_supported())
            && row_tile_scope_enabled(
                rows,
                self.wide_verify_rows,
                !(self.replay.is_recording() || self.graphs.capture_mode),
                self.prefill_route_intent().canonical_routing(),
            )
    }

    /// Whether the actual singleton MQ4 recipe matches the available full
    /// verify tile program. This does not grant width admission or change AR.
    /// Use the resolved row count: R2 on gfx1151/gfx1201, native scalar R1 on
    /// gfx1100. Halo's default LM forced-R1 shape is handled by the tile itself.
    #[inline]
    pub fn qwen4_row_tile_reference_supported(&self) -> bool {
        row_tile_reference_supported(
            &self.arch,
            self.arch_caps.gemv_rows_default(),
            self.flags.pm_decode,
        )
    }

    /// Bind the actual singleton AR reference for this verify scope; return
    /// the prior binding so the forward owner can restore it on success/error.
    #[inline]
    pub fn set_qwen4_verify_reference(
        &mut self,
        value: Option<Qwen4VerifyReference>,
    ) -> Option<Qwen4VerifyReference> {
        std::mem::replace(&mut self.qwen4_verify_reference, value)
    }

    /// The singleton arithmetic bound to this scope, not the recorder state.
    #[inline]
    pub fn qwen4_verify_reference(&self) -> Option<Qwen4VerifyReference> {
        self.qwen4_verify_reference
    }

    /// Set the verify forward's row-exact rows (`0` = not in a verify
    /// forward); returns the previous value.
    #[inline]
    pub fn set_wide_verify_rows(&mut self, rows: usize) -> usize {
        std::mem::replace(&mut self.wide_verify_rows, rows)
    }
}

/// Numerical routing intent of a forward: the typed form of the scope flags that decide
/// whether a row-keyed route is taken from the call's row count or from the canonical
/// (long-prefill) recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillRouteIntent {
    /// Decode, verify, lane steps and every unscoped forward: routes follow the real rows.
    Product,
    /// Opt-in batch-invariant Flash-Next profile (`HIPFIRE_PREFILL_PM_INVARIANT`). Takes the
    /// canonical routes but, historically, suppresses the native trunk IU4 route.
    InvariantProfile,
    /// Ordinary prefill ([`Gpu::ordinary_prefill_enter`]): canonical routes for every row
    /// count, one-row tails included, WITHOUT the invariant profile's IU4 suppression and
    /// never through the agent-turn append route's symmetric-MoE suppression.
    OrdinaryPrefill,
}

impl PrefillRouteIntent {
    /// Row-keyed routes see [`QWEN4_PREFILL_CANON_ROWS`] rows.
    #[inline]
    pub const fn canonical_routing(self) -> bool {
        !matches!(self, Self::Product)
    }

    /// The native trunk IU4 route is excluded (a row's bytes would otherwise depend on
    /// whole versus partial 128-row tiles). Ordinary prefill keeps the route and takes
    /// canonical tiles ([`Gpu::qwen4_trunk_iu4_ragged`]) instead.
    #[inline]
    pub const fn suppresses_trunk_iu4(self) -> bool {
        matches!(self, Self::InvariantProfile)
    }
}

// ---------------------------------------------------------------------------
// Canonical GDN ordinary-prefill producer policy (CHUNK-CANONICAL-DESIGN §1.1-§1.3, §2.2)
// ---------------------------------------------------------------------------

/// Stable arithmetic codes of one GDN projection field ([`crate::gdn_stream::GdnProjectionRecipe`]).
/// Equal codes mean byte-identical arithmetic for a row regardless of the call's extent or the
/// row's position inside a physical producer tile. Codes are part of the recipe identity
/// ([`GdnPrefillPlan::compute_recipe_id`]): never renumber, only append.
pub mod gdn_arith {
    /// Prepared symmetric MQ4V2 x A4 (`block_i4_128`) IU4 SET, exact gfx1201 (masked partial
    /// tiles; the fused `qkvzagdn`, zba-fold and separate SETs are one arithmetic).
    pub const IU4_G12: u32 = 1;
    /// gfx1100/gfx1151 symmetric IU4 over complete 128-row symfold tiles (partial extents run
    /// a private zero-padded full tile); includes the Z|beta|alpha fold when it qualifies.
    pub const IU4_G11_FULL_TILE: u32 = 2;
    /// gfx1100/gfx1151 static small-M F16 (MW4) beta/alpha beside IU4 QKV/Z.
    pub const MW4_F16_SMALL_M: u32 = 3;
    /// Symmetric MQ4V2 A8 (INT8 activations, MMQ SET).
    pub const A8: u32 = 4;
    /// gfx1201 native FP8 staged v2 with the symmetric fold.
    pub const FP8_V2_SYMFOLD: u32 = 5;
    /// gfx1201 native FP8 staged v2 without the symmetric fold.
    pub const FP8_V2: u32 = 6;
    /// Unprepared F16-activation WMMA projection (asymmetric MQ4V2 and legacy paths).
    pub const F16_WMMA: u32 = 7;
    /// gfx11 Q8_1 MMQ (IU4 off).
    pub const MMQ_Q8_1: u32 = 8;
    /// Output projection fed by the gfx1201 gated-norm A4 producer (BF16 or F32 scan plane).
    pub const OUT_IU4_FUSED_G12: u32 = 9;
    /// Output projection fed by the gfx11 gated-norm A4 producer.
    pub const OUT_IU4_FUSED_G11: u32 = 10;
    /// Output projection fed by the gfx1201 FP8 stream producer.
    pub const OUT_FP8_STREAM: u32 = 11;
    /// Output projection fed by the fused A8 gated-norm producer.
    pub const OUT_A8_FUSED: u32 = 12;
    /// Weight dtype outside the symmetric MQ4V2 families: code = `OTHER_BASE | dtype`.
    pub const OTHER_BASE: u32 = 0x100;
}

/// One GDN projection matrix as the plan sees it: header dtype and output rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnProjDesc {
    pub dtype: crate::DType,
    pub m: usize,
}

/// Static facts of one GDN layer: everything the numerical recipe may depend on, and nothing
/// that varies with the outer call (no row count, no free VRAM, no pointers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnLayerDesc {
    /// Model dim: `K` of the four input projections.
    pub dim: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub head_dim: usize,
    pub qkv: GdnProjDesc,
    pub z: GdnProjDesc,
    pub beta: GdnProjDesc,
    pub alpha: GdnProjDesc,
    /// The post-scan output projection (`wo`): header dtype only.
    pub out_dtype: crate::DType,
    /// The output projection accumulates into the residual (`BatchEpilogue::Residual`; always
    /// true for the MoE LA `wo`).
    pub out_residual: bool,
    /// The loader appended beta/alpha to Z (fold layout of `iu4_gdn_fold_rows` /
    /// `mq4v2_fold_betaalpha_padded_m`).
    pub z_folded: bool,
    /// MoE layer (Ornith/A3B family) rather than dense.
    pub moe: bool,
    /// Configured recurrent state quantization.
    pub state: crate::gdn_stream::GdnStateQuant,
    /// The row-independent chunk-scan admission the architecture owner resolves today (device
    /// flag and capability, `gdn_chunk_head_geometry_supported`, Q8 state with complete
    /// scale/conv/EF vectors, non-sequential/non-tree/non-tape configuration, per-family MoE
    /// requirements). `false` keeps the statically supported sequential recurrence family.
    pub scan64_admitted: bool,
}

impl Gpu {
    /// Resolve the canonical ordinary-prefill plan of one GDN layer from static facts.
    ///
    /// The plan is the bulk (aligned, long-prefill) numerical recipe of this device and
    /// configuration; it is evaluated at [`GDN_COMMIT_ROWS`] representative rows through the
    /// very predicates the producers run, so the recipe and the runtime cannot drift. While
    /// [`Gpu::ordinary_prefill_enter`] is open every actual extent `n >= 1` selects the same
    /// arithmetic ([`Gpu::prefill_rows_admit`]), including the one-row tail. Resolve once per
    /// prefix, in the execution mode (eager versus recording/capture) that prefix will use:
    /// recording and capture refuse the A4/A8/FP8 producers, which the plan then reflects.
    ///
    /// `Err` when called inside an IU4/FP8 verify scope (verify intent never shares a recipe
    /// with ordinary prefill).
    #[cfg(feature = "deltanet")]
    pub fn resolve_gdn_prefill_plan(
        &self,
        desc: &GdnLayerDesc,
    ) -> hip_bridge::HipResult<crate::gdn_stream::GdnPrefillPlan> {
        use crate::gdn_stream::{
            GdnPrefillPlan, GdnProjectionRecipe, GdnScanAlgorithm, GdnStateQuant, GdnTensorFormat,
            GDN_COMMIT_ROWS,
        };
        use crate::norm::GdnScanOut;
        if self.iu4_verify_scope || self.fp8_verify_scope || self.wide_verify_rows != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "resolve_gdn_prefill_plan: verify scope open (ordinary prefill intent only)",
            ));
        }
        if desc.dim == 0 || desc.head_dim == 0 || desc.key_heads == 0 || desc.value_heads == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "resolve_gdn_prefill_plan: empty GDN geometry",
            ));
        }
        let rows = GDN_COMMIT_ROWS;
        let k = desc.dim;
        let field = |p: GdnProjDesc, small_m_tail: bool| self.gdn_input_arith(desc, p, k, rows, small_m_tail);
        let projection_recipe = GdnProjectionRecipe {
            qkv: field(desc.qkv, false),
            z: field(desc.z, false),
            alpha: field(desc.alpha, true),
            beta: field(desc.beta, true),
            out: 0,
        };
        let (out_code, bf16_reader) = self.gdn_out_arith(desc, rows);
        let projection_recipe = GdnProjectionRecipe {
            out: out_code,
            ..projection_recipe
        };
        let (algorithm, scan_operand_format) = if desc.scan64_admitted {
            (GdnScanAlgorithm::Scan64, GdnTensorFormat::F16)
        } else {
            (GdnScanAlgorithm::Sequential, GdnTensorFormat::F32)
        };
        // The C64 scan plane: BF16 only for the gfx1201 A4 reader (and only when the scan
        // actually writes a plane); `HIPFIRE_GDN_SCAN_OUT` is a requested precision flag and
        // therefore part of the recipe.
        let scan_output_format = if algorithm == GdnScanAlgorithm::Scan64 && bf16_reader {
            match self.gdn_scan_out(GdnScanOut::Bf16)? {
                GdnScanOut::Bf16 => GdnTensorFormat::Bf16,
                GdnScanOut::F32 => GdnTensorFormat::F32,
            }
        } else {
            GdnTensorFormat::F32
        };
        let commit_rows = if desc.state == GdnStateQuant::F32 { 0 } else { GDN_COMMIT_ROWS };
        // Minimum physical producer tile: gfx1201 A4/FP8/A8 consumers mask partial tiles
        // natively (1 row); gfx11 symmetric IU4 symfold arithmetic needs a complete 128-row tile.
        let producer_tile_rows = if self.arch.starts_with("gfx11")
            && projection_recipe.qkv == gdn_arith::IU4_G11_FULL_TILE
        {
            128
        } else {
            1
        };
        let mut plan = GdnPrefillPlan {
            recipe_id: 0,
            algorithm,
            micro_rows: algorithm.micro_rows(),
            commit_rows,
            projection_recipe,
            scan_operand_format,
            scan_output_format,
            state_quant: desc.state,
            producer_tile_rows,
        };
        plan.recipe_id = plan.compute_recipe_id();
        Ok(plan)
    }

    /// Arithmetic code of one input projection field (QKV, Z, beta or alpha) evaluated at the
    /// representative bulk extent `rows`.
    #[cfg(feature = "deltanet")]
    fn gdn_input_arith(
        &self,
        desc: &GdnLayerDesc,
        p: GdnProjDesc,
        k: usize,
        rows: usize,
        small_m_tail: bool,
    ) -> u32 {
        use crate::DType;
        if p.dtype != DType::MQ4G256V2 {
            return gdn_arith::OTHER_BASE | p.dtype as u32;
        }
        if self.a8_prefill_active(rows, k) {
            return gdn_arith::A8;
        }
        let eager = !self.replay.is_recording() && !self.graphs.capture_mode && k % 256 == 0;
        match self.arch.as_str() {
            "gfx1201" => {
                if self.flags.iu4_prefill_enabled() && eager {
                    gdn_arith::IU4_G12
                } else if self.flags.gfx12_mq4v2_fp8_qkvza && eager {
                    if self.mq4v2_symmetric && self.flags.gfx12_mq4v2_fp8_v2 {
                        gdn_arith::FP8_V2_SYMFOLD
                    } else {
                        gdn_arith::FP8_V2
                    }
                } else {
                    gdn_arith::F16_WMMA
                }
            }
            "gfx1100" | "gfx1151" => {
                if self.flags.iu4_prefill_enabled() && eager {
                    let folded = desc.z_folded
                        && self.mq4v2_fold_betaalpha_active(
                            desc.qkv.m,
                            desc.z.m,
                            desc.beta.m,
                            desc.alpha.m,
                            k,
                            rows,
                        );
                    if small_m_tail && !folded && p.m < 128 {
                        gdn_arith::MW4_F16_SMALL_M
                    } else {
                        gdn_arith::IU4_G11_FULL_TILE
                    }
                } else if eager {
                    gdn_arith::MMQ_Q8_1
                } else {
                    gdn_arith::F16_WMMA
                }
            }
            _ => gdn_arith::F16_WMMA,
        }
    }

    /// `(arithmetic code, scan plane is read by the gfx1201 BF16-capable A4 reader)` of the
    /// output projection at the representative bulk extent `rows`, mirroring the reader
    /// priority of the dense/MoE output projection (A8, gfx12 A4, gfx11 A4, FP8 stream, F32).
    #[cfg(feature = "deltanet")]
    fn gdn_out_arith(&self, desc: &GdnLayerDesc, rows: usize) -> (u32, bool) {
        use crate::DType;
        let k = desc.value_heads * desc.head_dim;
        if desc.out_dtype != DType::MQ4G256V2 {
            return (gdn_arith::OTHER_BASE | desc.out_dtype as u32, false);
        }
        let shape = desc.out_residual && desc.head_dim == 128 && k > 0;
        if shape && self.flags.a8_fused_prod && self.a8_prefill_active(rows, k) {
            return (gdn_arith::OUT_A8_FUSED, false);
        }
        if shape && self.iu4_producer_quant_fused_active(rows, k) {
            return (gdn_arith::OUT_IU4_FUSED_G12, true);
        }
        if shape && self.iu4_gfx11_producer_quant_fused_active(rows, k) {
            return (gdn_arith::OUT_IU4_FUSED_G11, false);
        }
        if desc.head_dim == 128
            && self.flags.gfx12_mq4v2_fp8_resid
            && self.fp8_stream_active(rows, k)
        {
            return (gdn_arith::OUT_FP8_STREAM, false);
        }
        if self.a8_prefill_active(rows, k) {
            return (gdn_arith::A8, false);
        }
        if self.flags.iu4_prefill_enabled() && k % 256 == 0 {
            return (gdn_arith::IU4_G12 + 0x80 * (self.arch != "gfx1201") as u32, false);
        }
        (gdn_arith::F16_WMMA, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::{
        QWEN4_F16_WMMA_MIN_TOKENS, QWEN4_MOE_SYM_IU4_MIN_ROWS, QWEN4_MQ6_X4_PM_MIN_N_GFX1151,
    };

    /// The canonical row count must sit above every row threshold the profile
    /// routes on, or "taken as if n were canonical" would pick a short route.
    #[test]
    fn canonical_rows_clear_every_prefill_threshold() {
        assert_eq!(QWEN4_PREFILL_CANON_ROWS, 8192);
        assert!(QWEN4_PREFILL_CANON_ROWS >= QWEN4_F16_WMMA_MIN_TOKENS);
        assert!(QWEN4_PREFILL_CANON_ROWS >= 10 * QWEN4_F16_WMMA_MIN_TOKENS);
        assert!(QWEN4_PREFILL_CANON_ROWS >= QWEN4_MOE_SYM_IU4_MIN_ROWS);
        assert!(QWEN4_PREFILL_CANON_ROWS >= QWEN4_MQ6_X4_PM_MIN_N_GFX1151);
        // HC-down 160 x 64 tile (rows >= 2048) and the few-row arms (<= 8).
        assert!(QWEN4_PREFILL_CANON_ROWS >= 2048);
        assert!(QWEN4_PREFILL_CANON_ROWS > 8);
        assert_eq!(QWEN4_PREFILL_CANON_ROWS % 128, 0);
    }

    /// The routing-row cap of the exact EP wave: one row under every 512-row route floor, and
    /// above the few-row routes (so a capped call is a PREFILL-class call, never decode / verify).
    #[test]
    fn exact_wave_cap_is_the_last_sub_512_row_count() {
        assert_eq!(
            QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP,
            QWEN4_F16_WMMA_MIN_TOKENS - 1
        );
        assert!(QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP < QWEN4_MOE_SYM_IU4_MIN_ROWS);
        assert!(QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP > WIDE_VERIFY_MAX_ROWS);
    }

    /// Cap off is the identity at every row count (and the invariant canon still wins over a cap);
    /// cap on leaves every row count below it alone and clamps the wider ones.
    #[test]
    fn route_rows_cap_semantics() {
        for rows in [
            0usize, 1, 8, 9, 60, 149, 511, 512, 513, 1192, 2048, 4096, 100_000,
        ] {
            assert_eq!(route_rows_of(false, 0, rows), rows, "cap off, rows {rows}");
            assert_eq!(
                route_rows_of(true, 0, rows),
                QWEN4_PREFILL_CANON_ROWS,
                "invariant, rows {rows}"
            );
            assert_eq!(
                route_rows_of(true, QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP, rows),
                QWEN4_PREFILL_CANON_ROWS,
                "invariant beats the cap, rows {rows}"
            );
            let capped = route_rows_of(false, QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP, rows);
            assert_eq!(capped, rows.min(511), "cap 511, rows {rows}");
            if rows < 512 {
                assert_eq!(capped, rows, "a sub-512 lane is unchanged by the cap");
            } else {
                assert!(
                    capped < QWEN4_F16_WMMA_MIN_TOKENS,
                    "a wide wave routes sub-512"
                );
            }
        }
    }

    #[test]
    fn append_route_disabled_is_never_taken() {
        for prefill in [false, true] {
            for pm in [false, true] {
                for arch in ["gfx1151", "gfx1201", "gfx942"] {
                    for rows in [0, 1, 64, 511, 512, 8192] {
                        assert!(!qwen4_append_wmma_route(false, prefill, arch, rows, 64, pm));
                    }
                }
            }
        }
    }

    #[test]
    fn append_route_row_window_and_gates() {
        let route =
            |prefill, arch, rows, pm| qwen4_append_wmma_route(true, prefill, arch, rows, 64, pm);
        assert!(!route(true, "gfx1151", 63, false));
        assert!(route(true, "gfx1151", 64, false));
        assert!(route(true, "gfx1151", 511, false));
        assert!(!route(true, "gfx1151", 512, false));
        assert!(!route(true, "gfx1151", 8192, false));
        assert!(!route(false, "gfx1151", 256, false));
        assert!(!route(true, "gfx1151", 256, true));
        for arch in ["gfx1201", "gfx942", "gfx1100", "gfx1150"] {
            assert!(!route(true, arch, 256, false));
        }
    }

    #[test]
    fn append_route_defaults() {
        assert!(QWEN4_APPEND_WMMA_DEFAULT);
        assert!(QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT >= 2);
        assert!(QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT < QWEN4_F16_WMMA_MIN_TOKENS);
    }

    #[test]
    fn every_arch_is_at_the_few_row_limit() {
        for arch in ["gfx1151", "gfx1201", "gfx1100"] {
            assert_eq!(resolve_verify_rows(arch, 0, None, 8), 8, "{arch}");
        }
    }

    /// The native Qwen4 `auto` MTP cap is one below the runtime row-exact
    /// rows, and the `hipfire-config` effective view of `speculation.mtp_k`
    /// (which reads the shared table, not the developer override) agrees with
    /// it under the default process policy.
    #[test]
    fn native_mtp_default_cap_is_row_exact_rows_minus_one_and_matches_config_view() {
        use hipfire_config::effective::{effective, A8Prefill, ArchCtx};
        use hipfire_config::{mtp_k_default, ConfigValue, MTP_K_DEFAULT};

        let resolved = hipfire_config::resolve([]).unwrap();
        for arch in ["gfx1151", "gfx1201", "gfx1100", "gfx1150", "gfx942"] {
            let rows = resolve_verify_rows(
                arch,
                0,
                None,
                hipfire_config::qwen4_row_exact_verify_rows_table(arch),
            );
            let cap = mtp_k_default(16, rows);
            assert_eq!(cap, rows - 1, "{arch}");
            assert_eq!(cap, 7, "{arch}");
            let ctx = ArchCtx {
                gpu_arch: Some(arch.into()),
                gpu_count: 1,
                model_arch_id: Some(16),
                registry_tag: None,
                a8_prefill: A8Prefill::NotLanded,
            };
            assert_eq!(
                effective("speculation.mtp_k", &resolved, &ctx).value,
                ConfigValue::Integer(cap as i64),
                "{arch}"
            );
        }
        assert_eq!(
            mtp_k_default(5, resolve_verify_rows("gfx1151", 0, None, 8)),
            MTP_K_DEFAULT
        );
    }
}

#[cfg(test)]
mod wide_verify_tests {
    use super::*;

    const BOUNDARY_ROWS: [usize; 10] = [0, 1, 8, 9, 32, 33, 63, 64, 65, usize::MAX];

    #[test]
    fn few_row_limit_scope_recording_capture_truth_table() {
        assert_eq!(FEW_ROW_ROUTE_ROWS, 8);
        assert_eq!(WIDE_VERIFY_MAX_ROWS, 64);
        for arch in [
            "gfx1151",
            "gfx1201",
            "gfx1100",
            "gfx1150",
            "gfx942",
            "gfx1151-x",
            "",
        ] {
            for scope in BOUNDARY_ROWS {
                for recording in [false, true] {
                    for capture in [false, true] {
                        let eager = !(recording || capture);
                        let limit = few_row_limit_for(arch, scope, eager);
                        let expected = if matches!(arch, "gfx1151" | "gfx1201" | "gfx1100")
                            && eager
                            && scope <= WIDE_VERIFY_MAX_ROWS
                        {
                            scope.max(FEW_ROW_ROUTE_ROWS)
                        } else {
                            FEW_ROW_ROUTE_ROWS
                        };
                        assert_eq!(limit, expected, "{arch} scope={scope}");
                    }
                }
            }
        }
    }

    #[test]
    fn row_tile_scope_truth_table() {
        for rows in BOUNDARY_ROWS {
            for scope in BOUNDARY_ROWS {
                for recording in [false, true] {
                    for capture in [false, true] {
                        for invariant in [false, true] {
                            let eager = !(recording || capture);
                            let expected = eager
                                && !invariant
                                && (1..=64).contains(&scope)
                                && rows > 0
                                && rows <= scope;
                            assert_eq!(
                                row_tile_scope_enabled(rows, scope, eager, invariant),
                                expected,
                                "rows={rows} scope={scope} recording={recording} capture={capture} invariant={invariant}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn row_tile_family_architecture_truth_table() {
        for family in WideTwinFamily::ALL {
            assert!(row_tile_family_supported("gfx1151", family));
            assert_eq!(
                row_tile_family_supported("gfx1201", family),
                family != WideTwinFamily::Q8,
            );
            assert!(row_tile_family_supported("gfx1100", family));
            for arch in ["gfx1150", "gfx942", "gfx1151-x", "gfx1201-x", ""] {
                assert!(
                    !row_tile_family_supported(arch, family),
                    "{arch} {family:?}"
                );
            }
        }
    }

    #[test]
    fn row_tile_reference_actual_singleton_recipe_truth_table() {
        for arch in [
            "gfx1151",
            "gfx1201",
            "gfx1100",
            "gfx1150",
            "gfx942",
            "gfx1201-x",
            "",
        ] {
            for rows in [0, 1, 2, 4, 8, u32::MAX] {
                for pm_decode in [false, true] {
                    let expected = (rows == 2
                        && (arch == "gfx1151" || (arch == "gfx1201" && pm_decode)))
                        || (arch == "gfx1100" && rows == 1);
                    assert_eq!(
                        row_tile_reference_supported(arch, rows, pm_decode),
                        expected,
                        "{arch} effective_rows={rows} pm_decode={pm_decode}",
                    );
                }
            }
        }
    }

    #[test]
    fn override_parse_and_clamp() {
        assert_eq!(wide_verify_rows_override_value(None), None);
        assert_eq!(wide_verify_rows_override_value(Some("")), None);
        assert_eq!(wide_verify_rows_override_value(Some("x")), None);
        assert_eq!(wide_verify_rows_override_value(Some("-3")), None);
        assert_eq!(
            wide_verify_rows_override_value(Some("0")),
            Some(FEW_ROW_ROUTE_ROWS)
        );
        assert_eq!(
            wide_verify_rows_override_value(Some("5")),
            Some(FEW_ROW_ROUTE_ROWS)
        );
        assert_eq!(wide_verify_rows_override_value(Some(" 16 ")), Some(16));
        for rows in BOUNDARY_ROWS {
            assert_eq!(
                wide_verify_rows_override_value(Some(&rows.to_string())),
                Some(rows.clamp(FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS)),
            );
        }
        assert_eq!(
            wide_verify_rows_override_value(Some("9999")),
            Some(WIDE_VERIFY_MAX_ROWS)
        );
    }

    #[test]
    fn verify_rows_source_precedence() {
        // runtime > env > table
        for arch in ["gfx1151", "gfx1201", "gfx1100"] {
            for rows in BOUNDARY_ROWS {
                let clamped = rows.clamp(FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS);
                assert_eq!(
                    resolve_verify_rows(arch, rows, Some(24), 8),
                    if rows == 0 { 24 } else { clamped },
                );
                assert_eq!(resolve_verify_rows(arch, 0, Some(rows), 8), clamped);
                assert_eq!(resolve_verify_rows(arch, 0, None, 8), 8);
            }
        }
        // Unknown architectures cannot acquire global wide admission from
        // either lab override.
        for arch in ["gfx1150", "gfx942", "gfx1151-x", "gfx1100-x", ""] {
            for rows in BOUNDARY_ROWS {
                assert_eq!(resolve_verify_rows(arch, rows, Some(64), 8), 8);
            }
        }
    }

    #[test]
    fn runtime_override_clamps_and_clears() {
        // The runtime static is process-global: set, observe, clear.
        set_qwen4_wide_verify_rows_override(Some(4));
        assert_eq!(
            QWEN4_WIDE_VERIFY_ROWS_RUNTIME.load(Ordering::Relaxed),
            FEW_ROW_ROUTE_ROWS
        );
        set_qwen4_wide_verify_rows_override(Some(999));
        assert_eq!(
            QWEN4_WIDE_VERIFY_ROWS_RUNTIME.load(Ordering::Relaxed),
            WIDE_VERIFY_MAX_ROWS
        );
        set_qwen4_wide_verify_rows_override(None);
        assert_eq!(QWEN4_WIDE_VERIFY_ROWS_RUNTIME.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn proven_rows_table_is_within_bounds() {
        for rows in [
            QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1151,
            QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1201,
            QWEN4_ROW_EXACT_VERIFY_ROWS_GFX942,
        ] {
            assert!((FEW_ROW_ROUTE_ROWS..=WIDE_VERIFY_MAX_ROWS).contains(&rows));
        }
        assert_eq!(
            qwen4_row_exact_verify_rows("gfx942"),
            QWEN4_ROW_EXACT_VERIFY_ROWS_GFX942
        );
        for arch in ["gfx1150", "gfx1151-x", "gfx1100-x", ""] {
            assert_eq!(qwen4_row_exact_verify_rows(arch), FEW_ROW_ROUTE_ROWS);
        }
    }
}

#[cfg(test)]
mod few_row_groups_tests {
    use super::*;

    #[test]
    fn groups_are_eights_with_remainder_last() {
        assert_eq!(few_row_groups(0).count(), 0);
        assert_eq!(few_row_groups(1).collect::<Vec<_>>(), vec![(0, 1)]);
        assert_eq!(few_row_groups(8).collect::<Vec<_>>(), vec![(0, 8)]);
        assert_eq!(few_row_groups(9).collect::<Vec<_>>(), vec![(0, 8), (8, 1)]);
        assert_eq!(
            few_row_groups(WIDE_VERIFY_MAX_ROWS).collect::<Vec<_>>(),
            vec![
                (0, 8),
                (8, 8),
                (16, 8),
                (24, 8),
                (32, 8),
                (40, 8),
                (48, 8),
                (56, 8)
            ]
        );
        for rows in 1..=WIDE_VERIFY_MAX_ROWS {
            let groups: Vec<_> = few_row_groups(rows).collect();
            assert_eq!(groups.iter().map(|g| g.1).sum::<usize>(), rows);
            assert!(groups
                .iter()
                .all(|g| (1..=FEW_ROW_ROUTE_ROWS).contains(&g.1)));
            assert!(groups.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0));
        }
    }
}
