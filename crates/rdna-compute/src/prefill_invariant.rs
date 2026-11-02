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

impl Gpu {
    /// Row count to feed ROUTING predicates (never launch extents): the
    /// canonical prefill chunk under the invariant profile, else `rows`.
    #[inline]
    pub fn route_rows(&self, rows: usize) -> usize {
        if self.prefill_invariant_scope {
            QWEN4_PREFILL_CANON_ROWS
        } else {
            rows
        }
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
                self.prefill_invariant_scope,
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
        ] {
            assert!((FEW_ROW_ROUTE_ROWS..=WIDE_VERIFY_MAX_ROWS).contains(&rows));
        }
        for arch in ["gfx1150", "gfx942", "gfx1151-x", "gfx1100-x", ""] {
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
