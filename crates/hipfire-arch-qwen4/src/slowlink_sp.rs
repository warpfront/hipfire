//! SP: the pure admission, window qualification and per-layer protocol of the
//! short-prompt prefill hook (design `screens/SP/HOOK-DESIGN.md` §5.1, §6,
//! §7.1, §9 slice E).
//!
//! SP is a per-window mode of S1 (the slow-link layer-major prefill): it
//! streams the top-k* experts of a layer and runs the rest on the CPU, then
//! applies one canonical combine. Everything here is pure data and decisions:
//! no device, no stream, no allocation. The runtime (slices A-D, F1/F2) reads
//! these verdicts before any mutation.
//!
//! - [`admit_sp`]: load-time admission. Every decline is a named, sticky
//!   [`SpDecline`]; the S1 profile (resident layers, census) is never an input
//!   to it that it can change, so SP adds **zero** VRAM and moves no layer.
//! - [`qualify_sp_window`]: per-window verdict ([`SpBypass`] when the window
//!   stays on today's page-order sweep).
//! - [`SpProtocol`]: the §7.1 state machine, refusing every illegal edge, with
//!   a [`SpEventBackend`] the runtime implements over HIP events and tests
//!   implement with a fake.

use crate::slowlink_admission::{CaptureKind, WindowRoute};
use crate::slowlink_prefill::{TilePlan, WindowPlan};
use std::fmt;

/// Typed developer boolean `developer.qwen4_sp` (default off).
pub const SP_ENV: &str = "HIPFIRE_QWEN4_SP";
/// Typed developer boolean `developer.qwen4_sp_touched` (default off): the
/// touched-only SP mode ([`SpMode::TouchedOnly`]).
pub const SP_TOUCHED_ENV: &str = "HIPFIRE_QWEN4_SP_TOUCHED";
/// Lab-only: `all` (k = U, which selects [`SpMode::TouchedOnly`]), `cpu` (k = 0)
/// or `auto` (the planner).
#[cfg(feature = "lab")]
pub const SP_FORCE_K_ENV: &str = "HIPFIRE_QWEN4_SP_FORCE_K";
/// Lab-only: `1` prints one JSON line per SP layer.
#[cfg(feature = "lab")]
pub const SP_TRACE_ENV: &str = "HIPFIRE_QWEN4_SP_TRACE";

/// Fewest rows of an SP window: `CPU_SPLIT_MAX_ROWS + 1`, so D2's qualified
/// windows of up to four rows are never taken.
pub const SP_MIN_ROWS: usize = 5;
/// Most rows of an SP window by gain and by memory.
pub const SP_MAX_ROWS: usize = 5_376;
/// Smallest region fit worth arming (abandon criterion of §11): below this
/// SP is declined at load.
pub const SP_MIN_REGION_ROWS: usize = 2_048;
/// `S_cap = SP_S_CAP_FACTOR * sp_max_rows`: the CPU-row staging capacity.
pub const SP_S_CAP_FACTOR: usize = 5;
/// The only calibrated architecture.
pub const SP_ARCH: &str = "gfx1100";
/// The only calibrated expert count.
pub const SP_N_EXPERTS: usize = 512;

/// Experts' grouped-row padding slack per expert (`10N + 512 * 15`).
const GROUPED_SLACK_PER_EXPERT: usize = 15;
/// Top-k slots per token row.
const SLOTS_PER_ROW: usize = 10;

/// Rows of the grouped bound `10N + n_exp * 15`, rounded up to 16.
pub(crate) fn sp_grouped_bound(rows: usize, n_exp: usize) -> Option<usize> {
    rows.checked_mul(SLOTS_PER_ROW)?
        .checked_add(n_exp.checked_mul(GROUPED_SLACK_PER_EXPERT)?)?
        .checked_add(15)
        .map(|v| v & !15)
}

/// Pinned host bytes of the SP owners at `max_rows` (§6): `x16_host`
/// `[max_rows, hidden]` u16, `cpu_rows` `[s_cap, hidden]` u16, `sorted_host`
/// `[G_max]` i32 and `inv_host` `[max_rows * 10]` i32.
pub(crate) fn sp_pinned_bytes(hidden: usize, n_exp: usize, max_rows: usize) -> Option<u64> {
    let h = hidden as u64;
    let n = max_rows as u64;
    let s_cap = max_rows.checked_mul(SP_S_CAP_FACTOR)? as u64;
    let g_max = sp_grouped_bound(max_rows, n_exp)? as u64;
    let x16 = n.checked_mul(h)?.checked_mul(2)?;
    let cpu = s_cap.checked_mul(h)?.checked_mul(2)?;
    let sorted = g_max.checked_mul(4)?;
    let inv = n.checked_mul(SLOTS_PER_ROW as u64)?.checked_mul(4)?;
    x16.checked_add(cpu)?.checked_add(sorted)?.checked_add(inv)
}

// ── lab knobs ─────────────────────────────────────────────────────────

/// Lab placement override (`HIPFIRE_QWEN4_SP_FORCE_K`).
#[cfg(feature = "lab")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpForceK {
    /// k = U: every touched expert streams.
    All,
    /// k = 0: every expert runs on the CPU.
    Cpu,
    /// The planner decides.
    Auto,
}

/// Pure body of [`sp_force_k`]: unset is `Auto`; exact lower-case spellings
/// only, anything else (including empty) is an error naming the variable.
#[cfg(feature = "lab")]
pub fn parse_sp_force_k(raw: Option<&str>) -> Result<SpForceK, String> {
    match raw {
        None => Ok(SpForceK::Auto),
        Some("all") => Ok(SpForceK::All),
        Some("cpu") => Ok(SpForceK::Cpu),
        Some("auto") => Ok(SpForceK::Auto),
        Some(other) => Err(format!(
            "{SP_FORCE_K_ENV}={other:?} is not one of all, cpu, auto"
        )),
    }
}

/// The lab placement override of this process.
#[cfg(feature = "lab")]
pub fn sp_force_k() -> Result<SpForceK, String> {
    let raw = hipfire_config::developer_var(SP_FORCE_K_ENV).ok();
    parse_sp_force_k(raw.as_deref())
}

/// Pure body of [`sp_trace`]: only an exact `1` enables it.
#[cfg(feature = "lab")]
pub fn parse_sp_trace(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// `HIPFIRE_QWEN4_SP_TRACE=1`.
#[cfg(feature = "lab")]
pub fn sp_trace() -> bool {
    parse_sp_trace(hipfire_config::developer_var(SP_TRACE_ENV).ok().as_deref())
}

// ── mode ──────────────────────────────────────────────────────────────

/// Which SP a load armed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpMode {
    /// The hybrid SP (`developer.qwen4_sp`): stream the top-k* experts, run the
    /// rest on the CPU (E1), one canonical combine. Different numerics from
    /// stream-all S1 by design (E1 rows, K2-ized F32-tail tiles).
    Hybrid,
    /// The touched-only SP (`developer.qwen4_sp_touched`): S1's exact per-panel
    /// pipeline over a selective schedule that streams exactly the experts the
    /// layer routes to (k = U), no CPU expert. Byte-identical to stream-all S1.
    TouchedOnly,
}

impl SpMode {
    /// Stable machine label of the mode (receipt `sp.mode`).
    pub fn label(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::TouchedOnly => "touched_only",
        }
    }

    /// Whether the window runs on the SP route layout (hybrid row table, F32-tail
    /// tiles K2-ized, no cohort rows) rather than S1's own: the storage's SP flag
    /// ([`hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteStorage::set_sp`]).
    /// Touched-only keeps S1's layout, so every tile keeps S1's class.
    pub fn sp_route_layout(self) -> bool {
        self == Self::Hybrid
    }
}

/// The hybrid-SP knob of a load on `arch`: an explicit `developer.qwen4_sp`
/// (`HIPFIRE_QWEN4_SP`, `0` is the kill switch) wins, unset is **on for exact gfx1100** (the
/// only calibrated arch) and off elsewhere. When the hybrid mode declines at load, the
/// touched-only mode (its own knob) is the fallback ([`admit_sp_noted`]). Pure;
/// [`hybrid_knob`] reads the process value.
pub fn hybrid_knob_for(explicit: Option<bool>, arch: &str) -> bool {
    explicit.unwrap_or(arch == SP_ARCH)
}

/// [`hybrid_knob_for`] with the process's `developer.qwen4_sp`.
pub fn hybrid_knob(arch: &str) -> bool {
    hybrid_knob_for(hipfire_config::memory::qwen4_sp(), arch)
}

/// The touched-only knob of a load on `arch`: an explicit `developer.qwen4_sp_touched`
/// (`HIPFIRE_QWEN4_SP_TOUCHED`, `0` is the kill switch) wins, unset is **on for exact
/// gfx1100** (the only calibrated arch; byte-identical to stream-all S1) and off elsewhere.
/// Pure; [`touched_knob`] reads the process value.
pub fn touched_knob_for(explicit: Option<bool>, arch: &str) -> bool {
    explicit.unwrap_or(arch == SP_ARCH)
}

/// [`touched_knob_for`] with the process's `developer.qwen4_sp_touched`.
pub fn touched_knob(arch: &str) -> bool {
    touched_knob_for(hipfire_config::memory::qwen4_sp_touched(), arch)
}

/// The window route one window runs: the armed hybrid SP's route
/// ([`SpProfile::window_route`]) when SP runs the window (`sp_ok`: its pre-mutation
/// verdict qualified it), else the S1 profile's own `base` route. Touched-only never changes a
/// window's route, and with SP not armed every window keeps `base`.
pub fn sp_window_route(base: WindowRoute, sp: Option<&SpProfile>, sp_ok: bool) -> WindowRoute {
    match sp {
        Some(sp) if sp_ok && sp.mode == SpMode::Hybrid => sp.window_route,
        _ => base,
    }
}

/// The Gpu `qwen4_append_f16_moe_scope` of a tile scope: the agent-turn append route's own
/// request, or a hybrid SP window running the F16 route on a symmetric-IU4 profile. Either
/// one suppresses the symmetric IU4 MoE arm for that scope only.
pub fn f16_moe_scope(append: bool, sp_f16_window: bool) -> bool {
    append || sp_f16_window
}

/// The mode the knobs ask for, decided before admission and before the first
/// window (pure truth table):
///
/// | `qwen4_sp` | `qwen4_sp_touched` | placement forced k = U | mode |
/// |---|---|---|---|
/// | off | off | any | none |
/// | off | on | any | touched-only |
/// | on | off | no | hybrid |
/// | on | off | yes | touched-only |
/// | on | on | no | hybrid (hybrid wins; touched-only is its fallback when it declines) |
/// | on | on | yes | touched-only |
///
/// Both knobs are unset-on for exact gfx1100 ([`hybrid_knob_for`], [`touched_knob_for`]), so the
/// gfx1100 default is hybrid with touched-only behind it; `HIPFIRE_QWEN4_SP=0` leaves
/// touched-only, `HIPFIRE_QWEN4_SP_TOUCHED=0` leaves hybrid without a fallback, both `0` is off.
///
/// A placement forced to k = U (lab `HIPFIRE_QWEN4_SP_FORCE_K=all`) is the
/// touched-only placement by definition: streaming every touched expert with no
/// CPU expert is exactly what the touched-only mode does, with the F32-tail
/// tiles kept on S1's own path instead of K2-ized. It has no effect without a
/// knob.
pub fn sp_mode_for(hybrid_knob: bool, touched_knob: bool, force_all: bool) -> Option<SpMode> {
    match (hybrid_knob, touched_knob) {
        (false, false) => None,
        (false, true) => Some(SpMode::TouchedOnly),
        (true, _) if force_all => Some(SpMode::TouchedOnly),
        (true, _) => Some(SpMode::Hybrid),
    }
}

// ── load-time admission ───────────────────────────────────────────────

/// Everything [`admit_sp`] reads, all decided by the caller before any SP
/// allocation. Nothing here is an input of the S1 admission, so arming SP can
/// neither change `resident_layers` nor the census.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpLoadFacts<'a> {
    /// `developer.qwen4_sp` (`hipfire_config::memory::qwen4_sp`): the hybrid mode.
    pub knob: bool,
    /// `developer.qwen4_sp_touched` (`hipfire_config::memory::qwen4_sp_touched`):
    /// the touched-only mode.
    pub touched_knob: bool,
    /// The placement is forced to k = U (lab `HIPFIRE_QWEN4_SP_FORCE_K=all`;
    /// `false` outside the lab feature): see [`sp_mode_for`].
    pub force_all: bool,
    /// Pages of the S1 ring (a touched-only layer enqueues all its groups at once).
    pub pages: usize,
    /// Everything the F16 `Aggregate` window route needs is present on a load whose S1
    /// profile route is [`WindowRoute::AggregateSymIu4`]: the F16 WMMA grouped arm admits the
    /// admitted tile's sealed call (the load-time dry run, [`WindowRoute::Aggregate`]-scoped)
    /// and the F16 SP row layout fits the route region the profile charged. Read by the
    /// hybrid mode on a symmetric-IU4 profile only; `true` is the caller's claim, not a
    /// default.
    pub f16_route_ready: bool,
    /// Rows of the admitted S1 window: the touched-only mode runs the S1 route
    /// layout, so its row cap is this, not the SP-layout fit.
    pub window_rows: usize,
    /// The admitted S1 profile's window route; `None` when S1 was not admitted.
    pub s1_route: Option<WindowRoute>,
    /// `gpu.arch`.
    pub arch: &'a str,
    /// Routed expert count of the model.
    pub n_exp: usize,
    /// A CPU quant exists for the host layers' gate_up dtype.
    pub cpu_gate_up_quant: bool,
    /// A CPU quant exists for the host layers' down dtype.
    pub cpu_down_quant: bool,
    /// Workers of the splice pool; `None` (or 0) when it is unavailable.
    pub splice_workers: Option<usize>,
    /// Largest window rows whose SP layout (`with_sp_rows(..).total_bytes()`)
    /// fits `census.route_storage` (about 5,593 on the product profile). Read by
    /// the hybrid mode only.
    pub region_fit_rows: usize,
    /// Model hidden size.
    pub hidden: usize,
    /// Pinned host bytes SP may charge (the pinned budget less S1's own).
    pub pinned_budget_bytes: u64,
    /// `serve.prefix_cache` is on.
    pub prefix_cache: bool,
}

/// The frozen SP profile of a load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpProfile {
    /// The armed mode.
    pub mode: SpMode,
    /// The window route of the S1 profile (what every non-SP window runs).
    pub base_route: WindowRoute,
    /// The window route the SP-qualified windows run: the hybrid mode runs the F16
    /// `Aggregate` route (its K2-ized tails and row table are designed for it), so on a
    /// symmetric-IU4 profile it differs from [`Self::base_route`]; touched-only runs S1's own
    /// route, always `base_route`.
    pub window_route: WindowRoute,
    /// `min(SP_MAX_ROWS, region fit)` (hybrid); `min(SP_MAX_ROWS, S1 window
    /// rows)` (touched-only).
    pub sp_max_rows: usize,
    /// CPU-row staging capacity in rows: `5 * sp_max_rows` (hybrid), 0
    /// (touched-only: no CPU expert).
    pub s_cap: usize,
    /// Splice-pool workers (`SplitCost.cpu_workers`); 0 for touched-only.
    pub workers: usize,
    /// Pinned host bytes the SP owners charge at `sp_max_rows`; 0 for
    /// touched-only (it binds no owner).
    pub pinned_bytes: u64,
}

/// A sticky, named reason SP is not armed on this load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpDecline {
    /// Neither `HIPFIRE_QWEN4_SP` nor `HIPFIRE_QWEN4_SP_TOUCHED` is `1`.
    KnobOff,
    /// No admitted S1 profile.
    NoS1Profile,
    /// `sp: arch not calibrated`.
    ArchNotCalibrated { arch: String },
    /// The window route is not one the armed mode serves.
    Route { route: WindowRoute, mode: SpMode },
    /// The model's expert count is not 512.
    ExpertCount { n_exp: usize },
    /// A host layer's gate_up or down dtype has no CPU quant.
    Quants { gate_up: bool, down: bool },
    /// The splice pool is unavailable.
    SplicePool,
    /// The SP window layout does not fit the route region for enough rows. The region is the
    /// admitted S1 window's (`window_rows` = `min(16,896, max_seq)` rounded to whole tiles), so a
    /// load with a small `max_seq` has a small region and declines here.
    RegionFit {
        fit_rows: usize,
        minimum: usize,
        window_rows: usize,
    },
    /// The SP pinned host owners do not fit the pinned budget.
    PinnedFit { need: Option<u64>, budget: u64 },
    /// `serve.prefix_cache` is on.
    PrefixCache,
    /// The ring has fewer pages than the groups of a layer that touches every
    /// expert (touched-only enqueues a layer's groups all at once).
    Pages { pages: usize, need: usize },
    /// The hybrid mode needs the F16 `Aggregate` window route on a symmetric-IU4 profile and
    /// something it needs is absent (the F16 arm does not admit the tile, or the SP layout
    /// does not fit the route region).
    F16Route { route: WindowRoute },
}

impl SpDecline {
    /// Stable machine label of the reason (receipt `sp.decline`).
    pub fn label(&self) -> &'static str {
        match self {
            Self::KnobOff => "knob off",
            Self::NoS1Profile => "no S1 profile",
            Self::ArchNotCalibrated { .. } => "arch not calibrated",
            Self::Route { .. } => "route",
            Self::ExpertCount { .. } => "n_exp",
            Self::Quants { .. } => "quants",
            Self::SplicePool => "splice pool",
            Self::RegionFit { .. } => "region fit",
            Self::PinnedFit { .. } => "pinned fit",
            Self::PrefixCache => "prefix cache",
            Self::Pages { .. } => "pages",
            Self::F16Route { .. } => "f16 route",
        }
    }
}

impl fmt::Display for SpDecline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sp: {}", self.label())?;
        match self {
            Self::KnobOff => write!(f, " (neither {SP_ENV} nor {SP_TOUCHED_ENV} is 1)"),
            Self::NoS1Profile => write!(f, " (the slow-link prefill profile was not admitted)"),
            Self::ArchNotCalibrated { arch } => {
                write!(f, " ({arch:?}; calibrated on {SP_ARCH:?} only)")
            }
            Self::Route { route, mode } => match mode {
                SpMode::Hybrid => write!(
                    f,
                    " ({route:?}; the hybrid mode serves {:?} and {:?})",
                    WindowRoute::Aggregate,
                    WindowRoute::AggregateSymIu4
                ),
                SpMode::TouchedOnly => write!(
                    f,
                    " ({route:?}; touched-only serves {:?} and {:?})",
                    WindowRoute::Aggregate,
                    WindowRoute::AggregateSymIu4
                ),
            },
            Self::ExpertCount { n_exp } => {
                write!(f, " ({n_exp} experts; only {SP_N_EXPERTS} is calibrated)")
            }
            Self::Quants { gate_up, down } => write!(
                f,
                " (CPU quant for gate_up: {gate_up}, down: {down}; both are required)"
            ),
            Self::SplicePool => write!(f, " (no splice pool workers)"),
            Self::RegionFit {
                fit_rows,
                minimum,
                window_rows,
            } => write!(
                f,
                " (the route region of the {window_rows}-row S1 window fits {fit_rows} SP rows; \
                 at least {minimum} are required: load with a larger max_seq to grow the window)"
            ),
            Self::PinnedFit { need, budget } => match need {
                Some(need) => write!(
                    f,
                    " (the SP owners need {need} pinned bytes; budget {budget})"
                ),
                None => write!(f, " (the SP pinned bytes overflow; budget {budget})"),
            },
            Self::PrefixCache => write!(f, " (serve.prefix_cache is on)"),
            Self::F16Route { route } => write!(
                f,
                " (the hybrid mode runs SP windows on {:?} but the {route:?} profile cannot host \
                 it: the F16 WMMA arm or the SP layout in the charged route region is missing; \
                 HIPFIRE_QWEN4_MOE_SYM_IU4=0 loads the F16 route instead)",
                WindowRoute::Aggregate
            ),
            Self::Pages { pages, need } => write!(
                f,
                " (the ring has {pages} pages; a layer touching every expert needs {need})"
            ),
        }
    }
}

/// Load-time SP admission (§5.1). Checked in a fixed order; the first failing
/// rule is the sticky reason. Pure.
///
/// The knobs pick the mode ([`sp_mode_for`]). Both modes need an admitted S1
/// profile, gfx1100, an aggregate window route (the hybrid mode on a symmetric-IU4 profile
/// additionally needs [`SpLoadFacts::f16_route_ready`]: its windows run the F16 `Aggregate`
/// route) and at least [`SP_MIN_REGION_ROWS`] rows. The hybrid mode also needs 512 experts, CPU
/// quants, the splice pool, the pinned-host fit and `serve.prefix_cache` off.
/// The touched-only mode needs none of those (it runs no CPU arithmetic, binds
/// no owner and leaves the numerics of S1 untouched); it needs the ring to hold
/// one page per group of a layer that touches every expert.
pub(crate) fn admit_sp(facts: &SpLoadFacts<'_>) -> Result<SpProfile, SpDecline> {
    admit_sp_noted(facts).map(|(profile, _)| profile)
}

/// [`admit_sp`] that also returns the hybrid decline a touched-only fallback replaced. When the
/// knobs select the hybrid mode and it declines (a small window's region fit, no CPU quant, the
/// pinned budget, `serve.prefix_cache`, the F16 route ...) while the touched-only knob is on,
/// touched-only is admitted instead and the hybrid decline is returned beside it so the load can
/// say why; if touched-only declines too, the HYBRID decline is the verdict (it is the mode the
/// knobs asked for). A forced placement k = U never selects the hybrid mode, so it has no
/// fallback to speak of.
pub(crate) fn admit_sp_noted(
    facts: &SpLoadFacts<'_>,
) -> Result<(SpProfile, Option<SpDecline>), SpDecline> {
    let Some(mode) = sp_mode_for(facts.knob, facts.touched_knob, facts.force_all) else {
        return Err(SpDecline::KnobOff);
    };
    match admit_mode(facts, mode) {
        Ok(profile) => Ok((profile, None)),
        Err(hybrid) if mode == SpMode::Hybrid && facts.touched_knob => {
            match admit_mode(facts, SpMode::TouchedOnly) {
                Ok(profile) => Ok((profile, Some(hybrid))),
                Err(_) => Err(hybrid),
            }
        }
        Err(decline) => Err(decline),
    }
}

/// The admission rules of `mode`, in their fixed order (the mode is already chosen).
fn admit_mode(facts: &SpLoadFacts<'_>, mode: SpMode) -> Result<SpProfile, SpDecline> {
    let Some(route) = facts.s1_route else {
        return Err(SpDecline::NoS1Profile);
    };
    if facts.arch != SP_ARCH {
        return Err(SpDecline::ArchNotCalibrated {
            arch: facts.arch.to_string(),
        });
    }
    // Touched-only runs S1's own panel pipeline, which serves every aggregate lowering the
    // incumbent does: the F16 route and the symmetric IU4 route (the gfx1100 default). The
    // hybrid mode K2-izes F32-tail tiles and carries its own row table, which are designed
    // for the F16 `Aggregate` route: on a symmetric-IU4 profile its windows run that route
    // (a per-window override) when everything it needs is present.
    match (mode, route) {
        (_, WindowRoute::Aggregate) => {}
        (SpMode::TouchedOnly, WindowRoute::AggregateSymIu4) => {}
        (SpMode::Hybrid, WindowRoute::AggregateSymIu4) if facts.f16_route_ready => {}
        (SpMode::Hybrid, WindowRoute::AggregateSymIu4) => {
            return Err(SpDecline::F16Route { route });
        }
        _ => return Err(SpDecline::Route { route, mode }),
    }
    match mode {
        SpMode::Hybrid => admit_hybrid(facts, route),
        SpMode::TouchedOnly => admit_touched_only(facts, route),
    }
}

fn admit_hybrid(facts: &SpLoadFacts<'_>, base_route: WindowRoute) -> Result<SpProfile, SpDecline> {
    if facts.n_exp != SP_N_EXPERTS {
        return Err(SpDecline::ExpertCount { n_exp: facts.n_exp });
    }
    if !(facts.cpu_gate_up_quant && facts.cpu_down_quant) {
        return Err(SpDecline::Quants {
            gate_up: facts.cpu_gate_up_quant,
            down: facts.cpu_down_quant,
        });
    }
    let workers = match facts.splice_workers {
        Some(workers) if workers > 0 => workers,
        _ => return Err(SpDecline::SplicePool),
    };
    if facts.region_fit_rows < SP_MIN_REGION_ROWS {
        return Err(SpDecline::RegionFit {
            fit_rows: facts.region_fit_rows,
            minimum: SP_MIN_REGION_ROWS,
            window_rows: facts.window_rows,
        });
    }
    let sp_max_rows = SP_MAX_ROWS.min(facts.region_fit_rows);
    let need = sp_pinned_bytes(facts.hidden, facts.n_exp, sp_max_rows);
    match need {
        Some(bytes) if bytes <= facts.pinned_budget_bytes => {
            if facts.prefix_cache {
                return Err(SpDecline::PrefixCache);
            }
            Ok(SpProfile {
                mode: SpMode::Hybrid,
                base_route,
                window_route: WindowRoute::Aggregate,
                sp_max_rows,
                s_cap: sp_max_rows * SP_S_CAP_FACTOR,
                workers,
                pinned_bytes: bytes,
            })
        }
        _ => Err(SpDecline::PinnedFit {
            need,
            budget: facts.pinned_budget_bytes,
        }),
    }
}

/// Experts one S1 transport group (ring page) holds.
const TOUCHED_EXPERTS_PER_PAGE: usize = 32;

fn admit_touched_only(
    facts: &SpLoadFacts<'_>,
    base_route: WindowRoute,
) -> Result<SpProfile, SpDecline> {
    if facts.window_rows < SP_MIN_REGION_ROWS {
        return Err(SpDecline::RegionFit {
            fit_rows: facts.window_rows,
            minimum: SP_MIN_REGION_ROWS,
            window_rows: facts.window_rows,
        });
    }
    let need = facts.n_exp.div_ceil(TOUCHED_EXPERTS_PER_PAGE);
    if facts.pages < need {
        return Err(SpDecline::Pages {
            pages: facts.pages,
            need,
        });
    }
    Ok(SpProfile {
        mode: SpMode::TouchedOnly,
        base_route,
        window_route: base_route,
        sp_max_rows: SP_MAX_ROWS.min(facts.window_rows),
        s_cap: 0,
        workers: 0,
        pinned_bytes: 0,
    })
}

// ── per-window qualification ──────────────────────────────────────────

/// Why a window stays on today's page-order sweep (the named `sp_bypass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpBypass {
    /// Fewer than [`SP_MIN_ROWS`] rows.
    TooFewRows { rows: usize, minimum: usize },
    /// More than `sp_max_rows` rows.
    TooManyRows { rows: usize, maximum: usize },
    /// Interior session cuts (I8: SP is causal only without them).
    InteriorCuts { cuts: usize },
    /// A tile of one routing row (a one-row decode tile).
    OneRowTile { tile: usize },
    /// K1 rows remain after the tail tiles are K2-ized.
    K1Rows { rows: usize },
    /// All-rows or external capture already bypasses S1.
    Capture(CaptureKind),
}

impl fmt::Display for SpBypass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewRows { rows, minimum } => {
                write!(
                    f,
                    "sp_bypass: {rows} rows is below the {minimum}-row minimum"
                )
            }
            Self::TooManyRows { rows, maximum } => {
                write!(
                    f,
                    "sp_bypass: {rows} rows exceeds the {maximum}-row maximum"
                )
            }
            Self::InteriorCuts { cuts } => {
                write!(f, "sp_bypass: {cuts} interior session cut(s)")
            }
            Self::OneRowTile { tile } => write!(f, "sp_bypass: tile {tile} has one routing row"),
            Self::K1Rows { rows } => write!(f, "sp_bypass: {rows} K1 rows after K2-ization"),
            Self::Capture(kind) => write!(f, "sp_bypass: {kind:?} capture"),
        }
    }
}

/// The window rules over plain numbers: the one body both
/// [`qualify_sp_window`] and the request verdict
/// ([`crate::slowlink_admission::qualify_request_sp`]) read.
pub(crate) fn qualify_sp_shape(
    sp: &SpProfile,
    rows: usize,
    cuts: usize,
    first_one_row_tile: Option<usize>,
    k1_rows: usize,
    capture: CaptureKind,
) -> Result<(), SpBypass> {
    match capture {
        CaptureKind::None | CaptureKind::Bounded => {}
        other => return Err(SpBypass::Capture(other)),
    }
    if rows < SP_MIN_ROWS {
        return Err(SpBypass::TooFewRows {
            rows,
            minimum: SP_MIN_ROWS,
        });
    }
    if rows > sp.sp_max_rows {
        return Err(SpBypass::TooManyRows {
            rows,
            maximum: sp.sp_max_rows,
        });
    }
    if cuts != 0 {
        return Err(SpBypass::InteriorCuts { cuts });
    }
    if let Some(tile) = first_one_row_tile {
        return Err(SpBypass::OneRowTile { tile });
    }
    if k1_rows != 0 {
        return Err(SpBypass::K1Rows { rows: k1_rows });
    }
    Ok(())
}

/// Per-window SP verdict (§5.1), decided before `prompt_txn_begin`.
/// `tiles` are the window's own tiles ([`crate::slowlink_prefill::Qwen4PromptPlan::window_tiles`]).
pub(crate) fn qualify_sp_window(
    sp: &SpProfile,
    window: &WindowPlan,
    tiles: &[TilePlan],
    capture: CaptureKind,
) -> Result<(), SpBypass> {
    let one_row = tiles
        .iter()
        .find(|tile| tile.routing.route_rows == 1)
        .map(|tile| tile.index);
    qualify_sp_shape(
        sp,
        window.rows,
        window.cuts.len(),
        one_row,
        window.k1_rows,
        capture,
    )
}

// ── per-layer protocol ────────────────────────────────────────────────

/// What the protocol asks of the event layer. The runtime backs it with HIP
/// events on the compute and xfer streams; tests back it with a fake.
pub trait SpEventBackend {
    /// `x_done[layer]` (xfer, after the last tile's x16 D2H) has completed.
    fn x_done_complete(&self, layer: usize) -> bool;
    /// `rows_ready[layer]` (xfer, after the CPU-rows and inverse H2D) has
    /// completed, so `cpu_rows` and `inv_host` are free for the host.
    fn rows_ready_complete(&self, layer: usize) -> bool;
    /// `combine_done[layer]` (compute) has been recorded, so the xfer stream
    /// can wait on it.
    fn combine_done_recorded(&self, layer: usize) -> bool;
}

/// The §7.1 states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpState {
    /// Window prime, or between layers before the next `Staging(0)`.
    Idle,
    /// Tile `t` of the layer is staged.
    Staging(usize),
    Barrier,
    Planned,
    GpuEnqueued,
    CpuRunning,
    Uploaded,
    Combined,
    Poisoned,
}

/// An illegal or failed protocol step. Every error leaves the protocol
/// [`SpState::Poisoned`] (drain required) except [`Self::Poisoned`] itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpProtocolError {
    /// `op` is not legal from `state`.
    Illegal { op: &'static str, state: SpState },
    /// The protocol was already poisoned; call `fail_window` after the drain.
    Poisoned,
    /// SP was disabled for the process by an earlier failure.
    Unavailable,
    /// A layer at or below the last combined one, or a layer other than the
    /// one in flight.
    WrongLayer { expected: usize, got: usize },
    /// A tile out of order (or a changed tile count) within the layer.
    TileOrder { expected: usize, got: usize },
    /// `barrier` before every tile of the layer was staged.
    TilesIncomplete { staged: usize, tiles: usize },
    /// The exact/padded count check failed at the barrier.
    ReadbackMismatch,
    /// More groups than ring pages.
    GroupsExceedPages { groups: usize, pages: usize },
    /// More CPU slots than the staging capacity.
    CpuSlotsExceedCap { slots: usize, cap: usize },
    /// Groups retired do not equal groups enqueued.
    GroupsNotRetired { retired: usize, groups: usize },
    /// `retire_group` beyond the enqueued groups.
    TooManyRetires { groups: usize },
    /// The x16 D2H of the layer has not completed.
    XferNotDone { layer: usize },
    /// `rows_ready[layer]` has not completed: a host write to `cpu_rows` or
    /// `inv_host` would race the DMA read of the previous layer (I3).
    RowsReadyPending { layer: usize },
    /// `combine_done[layer]` has not been recorded: the xfer H2D must wait it (I5).
    CombineDoneNotRecorded { layer: usize },
    /// The CPU arm returned an error.
    CpuFailed,
    /// `upload` before the CPU rows were written and E1 returned.
    CpuNotReturned,
}

impl fmt::Display for SpProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sp protocol: {self:?}")
    }
}

impl std::error::Error for SpProtocolError {}

/// The per-layer SP state machine (§7.1). One instance per window.
///
/// Per host layer `L` the legal order is `stage_tile(L, 0..T)` →
/// `barrier` → `plan` → `enqueue_gpu` → `retire_group` ×G → `begin_cpu` →
/// `host_write_cpu_rows` → `cpu_returned(true)` → `upload` → `combine`, then
/// the next layer or `end_window`. Any other edge is refused and poisons.
#[derive(Clone, Debug)]
pub struct SpProtocol {
    state: SpState,
    /// The layer in flight (set at `Staging(0)`).
    layer: Option<usize>,
    tiles: usize,
    /// Last layer that reached `Combined` in this window.
    last_combined: Option<usize>,
    /// Last layer whose `rows_ready` was recorded (reached `Uploaded`).
    last_uploaded: Option<usize>,
    groups: usize,
    retired: usize,
    cpu_written: bool,
    cpu_returned: bool,
    /// Sticky for the process once a window failed.
    unavailable: bool,
}

impl Default for SpProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl SpProtocol {
    pub const fn new() -> Self {
        Self {
            state: SpState::Idle,
            layer: None,
            tiles: 0,
            last_combined: None,
            last_uploaded: None,
            groups: 0,
            retired: 0,
            cpu_written: false,
            cpu_returned: false,
            unavailable: false,
        }
    }

    pub fn state(&self) -> SpState {
        self.state
    }

    /// The layer in flight, if any.
    pub fn layer(&self) -> Option<usize> {
        self.layer
    }

    /// SP is disabled for the process (a window failed).
    pub fn unavailable(&self) -> bool {
        self.unavailable
    }

    fn poison<T>(&mut self, error: SpProtocolError) -> Result<T, SpProtocolError> {
        self.state = SpState::Poisoned;
        self.unavailable = true;
        Err(error)
    }

    fn illegal<T>(&mut self, op: &'static str) -> Result<T, SpProtocolError> {
        let state = self.state;
        self.poison(SpProtocolError::Illegal { op, state })
    }

    fn live(&self) -> Result<(), SpProtocolError> {
        match self.state {
            SpState::Poisoned => Err(SpProtocolError::Poisoned),
            _ => Ok(()),
        }
    }

    /// Start a window: `Idle`, no layer history. Refused while a window is in
    /// flight or SP was disabled by a failure.
    pub fn begin_window(&mut self) -> Result<(), SpProtocolError> {
        self.live()?;
        if self.unavailable {
            return Err(SpProtocolError::Unavailable);
        }
        if self.state != SpState::Idle || self.layer.is_some() {
            return self.illegal("begin_window");
        }
        *self = Self::new();
        Ok(())
    }

    /// Tile `tile` of `tiles` of host layer `layer` is prepared and its x16 D2H
    /// enqueued. Legal as tile 0 from `Idle` or `Combined` (the layer must be
    /// above the last combined one), and as tile `t + 1` of the same layer
    /// from `Staging(t)`.
    pub fn stage_tile(
        &mut self,
        layer: usize,
        tile: usize,
        tiles: usize,
    ) -> Result<(), SpProtocolError> {
        self.live()?;
        if tiles == 0 || tile >= tiles {
            return self.poison(SpProtocolError::TileOrder {
                expected: 0,
                got: tile,
            });
        }
        match self.state {
            SpState::Idle | SpState::Combined => {
                if let Some(done) = self.last_combined {
                    if layer <= done {
                        return self.poison(SpProtocolError::WrongLayer {
                            expected: done + 1,
                            got: layer,
                        });
                    }
                }
                if tile != 0 {
                    return self.poison(SpProtocolError::TileOrder {
                        expected: 0,
                        got: tile,
                    });
                }
                self.layer = Some(layer);
                self.tiles = tiles;
                self.groups = 0;
                self.retired = 0;
                self.cpu_written = false;
                self.cpu_returned = false;
                self.state = SpState::Staging(0);
                Ok(())
            }
            SpState::Staging(prev) => {
                let current = self.layer.expect("Staging has a layer");
                if layer != current {
                    return self.poison(SpProtocolError::WrongLayer {
                        expected: current,
                        got: layer,
                    });
                }
                if tiles != self.tiles || tile != prev + 1 {
                    return self.poison(SpProtocolError::TileOrder {
                        expected: prev + 1,
                        got: tile,
                    });
                }
                self.state = SpState::Staging(tile);
                Ok(())
            }
            _ => self.illegal("stage_tile"),
        }
    }

    /// `finish_route_plan_with` returned. Requires every tile staged;
    /// `exact_ok` is the exact/padded count check (false poisons).
    pub fn barrier(&mut self, exact_ok: bool) -> Result<(), SpProtocolError> {
        self.live()?;
        match self.state {
            SpState::Staging(t) => {
                if t + 1 != self.tiles {
                    let (staged, tiles) = (t + 1, self.tiles);
                    return self.poison(SpProtocolError::TilesIncomplete { staged, tiles });
                }
                if !exact_ok {
                    return self.poison(SpProtocolError::ReadbackMismatch);
                }
                self.state = SpState::Barrier;
                Ok(())
            }
            _ => self.illegal("barrier"),
        }
    }

    /// `plan_sp_layer` produced a plan of `groups` groups and `cpu_slots` CPU
    /// slots; refused above `pages` ring pages or `s_cap` staging rows.
    pub fn plan(
        &mut self,
        groups: usize,
        pages: usize,
        cpu_slots: usize,
        s_cap: usize,
    ) -> Result<(), SpProtocolError> {
        self.live()?;
        if self.state != SpState::Barrier {
            return self.illegal("plan");
        }
        if groups > pages {
            return self.poison(SpProtocolError::GroupsExceedPages { groups, pages });
        }
        if cpu_slots > s_cap {
            return self.poison(SpProtocolError::CpuSlotsExceedCap {
                slots: cpu_slots,
                cap: s_cap,
            });
        }
        self.groups = groups;
        self.retired = 0;
        self.state = SpState::Planned;
        Ok(())
    }

    /// The whole layer's groups were enqueued on the ring copy stream and the
    /// GPU walk was enqueued.
    pub fn enqueue_gpu(&mut self) -> Result<(), SpProtocolError> {
        self.live()?;
        if self.state != SpState::Planned {
            return self.illegal("enqueue_gpu");
        }
        self.state = SpState::GpuEnqueued;
        Ok(())
    }

    /// One group's GEMMs retired its page.
    pub fn retire_group(&mut self) -> Result<(), SpProtocolError> {
        self.live()?;
        if self.state != SpState::GpuEnqueued {
            return self.illegal("retire_group");
        }
        if self.retired >= self.groups {
            let groups = self.groups;
            return self.poison(SpProtocolError::TooManyRetires { groups });
        }
        self.retired += 1;
        Ok(())
    }

    /// The driver thread starts the CPU arm: every group retired, the layer's
    /// x16 D2H complete (`x_done`) and `rows_ready[L-1]` proven complete (I3).
    pub fn begin_cpu(&mut self, events: &dyn SpEventBackend) -> Result<(), SpProtocolError> {
        self.live()?;
        if self.state != SpState::GpuEnqueued {
            return self.illegal("begin_cpu");
        }
        if self.retired != self.groups {
            let (retired, groups) = (self.retired, self.groups);
            return self.poison(SpProtocolError::GroupsNotRetired { retired, groups });
        }
        let layer = self.layer.expect("GpuEnqueued has a layer");
        if !events.x_done_complete(layer) {
            return self.poison(SpProtocolError::XferNotDone { layer });
        }
        if let Some(prev) = self.last_uploaded {
            if !events.rows_ready_complete(prev) {
                return self.poison(SpProtocolError::RowsReadyPending { layer: prev });
            }
        }
        self.state = SpState::CpuRunning;
        Ok(())
    }

    /// The host is about to write `cpu_rows` / `inv_host` (I3): legal only in
    /// `CpuRunning`, and only while `rows_ready[L-1]` is complete.
    pub fn host_write_cpu_rows(
        &mut self,
        events: &dyn SpEventBackend,
    ) -> Result<(), SpProtocolError> {
        self.live()?;
        if self.state != SpState::CpuRunning || self.cpu_written {
            return self.illegal("host_write_cpu_rows");
        }
        if let Some(prev) = self.last_uploaded {
            if !events.rows_ready_complete(prev) {
                return self.poison(SpProtocolError::RowsReadyPending { layer: prev });
            }
        }
        self.cpu_written = true;
        Ok(())
    }

    /// E1 returned. `ok == false` poisons.
    pub fn cpu_returned(&mut self, ok: bool) -> Result<(), SpProtocolError> {
        self.live()?;
        if self.state != SpState::CpuRunning || !self.cpu_written || self.cpu_returned {
            return self.illegal("cpu_returned");
        }
        if !ok {
            return self.poison(SpProtocolError::CpuFailed);
        }
        self.cpu_returned = true;
        Ok(())
    }

    /// The xfer stream waited `combine_done[L-1]` (I5), enqueued the two H2D
    /// copies and recorded `rows_ready[L]`.
    pub fn upload(&mut self, events: &dyn SpEventBackend) -> Result<(), SpProtocolError> {
        self.live()?;
        if self.state != SpState::CpuRunning {
            return self.illegal("upload");
        }
        if !(self.cpu_written && self.cpu_returned) {
            return self.poison(SpProtocolError::CpuNotReturned);
        }
        if let Some(prev) = self.last_combined {
            if !events.combine_done_recorded(prev) {
                return self.poison(SpProtocolError::CombineDoneNotRecorded { layer: prev });
            }
        }
        self.last_uploaded = self.layer;
        self.state = SpState::Uploaded;
        Ok(())
    }

    /// Compute waited `rows_ready[L]`, ran the one combine, recorded
    /// `combine_done[L]` and ran the tails. Legal only from `Uploaded`.
    pub fn combine(&mut self) -> Result<(), SpProtocolError> {
        self.live()?;
        if self.state != SpState::Uploaded {
            return self.illegal("combine");
        }
        self.last_combined = self.layer;
        self.state = SpState::Combined;
        Ok(())
    }

    /// The window finished: legal from `Idle` or `Combined` only. The protocol
    /// returns to a fresh `Idle`.
    pub fn end_window(&mut self) -> Result<(), SpProtocolError> {
        self.live()?;
        match self.state {
            SpState::Idle | SpState::Combined => {
                let unavailable = self.unavailable;
                *self = Self::new();
                self.unavailable = unavailable;
                Ok(())
            }
            _ => self.illegal("end_window"),
        }
    }

    /// Any failure from outside the protocol (enqueue, event or E1 error):
    /// poison and disable SP for the process.
    pub fn fail(&mut self) {
        self.state = SpState::Poisoned;
        self.unavailable = true;
    }

    /// `fail_window` after the ring and xfer drain: leave `Poisoned`. SP stays
    /// unavailable for the process ([`Self::begin_window`] refuses).
    pub fn fail_window(&mut self) -> Result<(), SpProtocolError> {
        if self.state != SpState::Poisoned {
            return self.illegal("fail_window");
        }
        *self = Self::new();
        self.unavailable = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slowlink_prefill::TileRouting;
    use std::cell::Cell;

    // ── knob ────────────────────────────────────────────────────────

    #[test]
    fn sp_knob_names_and_typed_developer_key() {
        assert_eq!(SP_ENV, "HIPFIRE_QWEN4_SP");
        assert!(hipfire_config::is_typed_bool_developer_key(
            "developer.qwen4_sp"
        ));
        assert_eq!(
            hipfire_config::developer_key_for_env(SP_ENV).as_deref(),
            Some("developer.qwen4_sp")
        );
    }

    #[test]
    fn sp_touched_knob_names_and_typed_developer_key() {
        assert_eq!(SP_TOUCHED_ENV, "HIPFIRE_QWEN4_SP_TOUCHED");
        assert!(hipfire_config::is_typed_bool_developer_key(
            "developer.qwen4_sp_touched"
        ));
        assert_eq!(
            hipfire_config::developer_key_for_env(SP_TOUCHED_ENV).as_deref(),
            Some("developer.qwen4_sp_touched")
        );
    }

    /// Touched-only is the gfx1100 default: unset is on there and off on every other arch, an
    /// explicit value wins on every arch, and the arch gate of admission still declines an
    /// explicit `1` elsewhere by name. The hybrid knob stays default off.
    #[test]
    fn touched_only_defaults_on_for_gfx1100_only() {
        for (explicit, arch, want) in [
            (None, "gfx1100", true),
            (Some(true), "gfx1100", true),
            (Some(false), "gfx1100", false),
            (None, "gfx1151", false),
            (None, "gfx1201", false),
            (None, "gfx1030", false),
            (Some(true), "gfx1201", true),
            (Some(false), "gfx1201", false),
        ] {
            assert_eq!(
                touched_knob_for(explicit, arch),
                want,
                "{explicit:?} {arch}"
            );
        }
        // The resolved knob feeds admission: unset on gfx1100 arms touched-only with the hybrid
        // knob off; the kill switch and other arches are the named decline / knob off.
        let on = |explicit, arch: &'static str| SpLoadFacts {
            knob: false,
            touched_knob: touched_knob_for(explicit, arch),
            arch,
            ..facts()
        };
        assert_eq!(
            admit_sp(&on(None, "gfx1100")).unwrap().mode,
            SpMode::TouchedOnly
        );
        assert_eq!(
            admit_sp(&on(Some(false), "gfx1100")).unwrap_err(),
            SpDecline::KnobOff
        );
        assert_eq!(
            admit_sp(&on(None, "gfx1201")).unwrap_err(),
            SpDecline::KnobOff
        );
        assert_eq!(
            admit_sp(&on(Some(true), "gfx1201")).unwrap_err(),
            SpDecline::ArchNotCalibrated {
                arch: "gfx1201".into()
            }
        );
        // Hybrid is never defaulted: with its own knob off and touched-only killed nothing arms,
        // and with its own knob on it still wins over the default-on touched-only.
        assert_eq!(
            admit_sp(&SpLoadFacts {
                knob: true,
                ..on(None, "gfx1100")
            })
            .unwrap()
            .mode,
            SpMode::Hybrid
        );
    }

    /// Hybrid SP is the gfx1100 default: unset is on there and off elsewhere, an explicit value
    /// wins everywhere, and touched-only (also default on) is the fallback when hybrid declines.
    #[test]
    fn hybrid_defaults_on_for_gfx1100_with_touched_only_as_the_fallback() {
        for (explicit, arch, want) in [
            (None, "gfx1100", true),
            (Some(true), "gfx1100", true),
            (Some(false), "gfx1100", false),
            (None, "gfx1151", false),
            (None, "gfx1201", false),
            (Some(true), "gfx1201", true),
            (Some(false), "gfx1201", false),
        ] {
            assert_eq!(hybrid_knob_for(explicit, arch), want, "{explicit:?} {arch}");
        }
        // The two knobs resolved from (SP, TOUCHED, arch) -> (mode, hybrid decline replaced).
        let admit = |sp: Option<bool>,
                     touched: Option<bool>,
                     arch: &'static str,
                     tweak: &dyn Fn(&mut SpLoadFacts<'static>)| {
            let mut facts = SpLoadFacts {
                knob: hybrid_knob_for(sp, arch),
                touched_knob: touched_knob_for(touched, arch),
                arch,
                ..facts()
            };
            tweak(&mut facts);
            admit_sp_noted(&facts)
        };
        let none: &dyn Fn(&mut SpLoadFacts<'static>) = &|_| {};
        // Defaults on gfx1100: hybrid, no fallback needed.
        let (profile, note) = admit(None, None, "gfx1100", none).unwrap();
        assert_eq!((profile.mode, note), (SpMode::Hybrid, None));
        // SP=0 leaves touched-only; TOUCHED=0 leaves hybrid; both 0 is off.
        assert_eq!(
            admit(Some(false), None, "gfx1100", none).unwrap().0.mode,
            SpMode::TouchedOnly
        );
        assert_eq!(
            admit(None, Some(false), "gfx1100", none).unwrap().0.mode,
            SpMode::Hybrid
        );
        assert_eq!(
            admit(Some(false), Some(false), "gfx1100", none).unwrap_err(),
            SpDecline::KnobOff
        );
        // Other arches: nothing arms by default; an explicit 1 declines by arch name.
        assert_eq!(
            admit(None, None, "gfx1201", none).unwrap_err(),
            SpDecline::KnobOff
        );
        assert_eq!(
            admit(Some(true), None, "gfx1201", none).unwrap_err(),
            SpDecline::ArchNotCalibrated {
                arch: "gfx1201".into()
            }
        );
        // Every hybrid decline falls back to touched-only on gfx1100 and is reported:
        // small window (region fit), no CPU quant, no splice pool, pinned budget, prefix
        // cache, expert count, no F16 route on a symmetric profile.
        type Tweak = fn(&mut SpLoadFacts<'static>);
        let declines: [(&str, Tweak, SpDecline); 7] = [
            (
                "small window",
                |f| {
                    f.region_fit_rows = 769;
                    f.window_rows = 4_096;
                },
                SpDecline::RegionFit {
                    fit_rows: 769,
                    minimum: 2_048,
                    window_rows: 4_096,
                },
            ),
            (
                "quants",
                |f| f.cpu_down_quant = false,
                SpDecline::Quants {
                    gate_up: true,
                    down: false,
                },
            ),
            ("pool", |f| f.splice_workers = None, SpDecline::SplicePool),
            (
                "pinned",
                |f| f.pinned_budget_bytes = 1,
                SpDecline::PinnedFit {
                    need: sp_pinned_bytes(2560, 512, SP_MAX_ROWS),
                    budget: 1,
                },
            ),
            (
                "prefix cache",
                |f| f.prefix_cache = true,
                SpDecline::PrefixCache,
            ),
            (
                "experts",
                |f| f.n_exp = 256,
                SpDecline::ExpertCount { n_exp: 256 },
            ),
            (
                "f16 route",
                |f| {
                    f.s1_route = Some(WindowRoute::AggregateSymIu4);
                    f.f16_route_ready = false;
                },
                SpDecline::F16Route {
                    route: WindowRoute::AggregateSymIu4,
                },
            ),
        ];
        for (name, tweak, want) in declines {
            let (profile, note) = admit(None, None, "gfx1100", &tweak).unwrap();
            assert_eq!(profile.mode, SpMode::TouchedOnly, "{name}");
            assert_eq!(note, Some(want.clone()), "{name}");
            // TOUCHED=0 removes the fallback: the hybrid decline is the verdict.
            assert_eq!(
                admit(None, Some(false), "gfx1100", &tweak).unwrap_err(),
                want,
                "{name}"
            );
        }
        // Both modes declining reports the hybrid decline (the mode the knobs asked for).
        let both = |f: &mut SpLoadFacts<'static>| {
            f.splice_workers = None;
            f.pages = 8;
        };
        assert_eq!(
            admit(None, None, "gfx1100", &both).unwrap_err(),
            SpDecline::SplicePool
        );
        // A forced placement k = U selects touched-only directly: no hybrid decline to report.
        let forced = |f: &mut SpLoadFacts<'static>| f.force_all = true;
        let (profile, note) = admit(None, None, "gfx1100", &forced).unwrap();
        assert_eq!((profile.mode, note), (SpMode::TouchedOnly, None));
    }

    #[test]
    fn sp_min_rows_is_one_past_the_d2_split_rows() {
        assert_eq!(
            SP_MIN_ROWS,
            hipfire_dispatch::cpu_split::CPU_SPLIT_MAX_ROWS + 1
        );
    }

    #[cfg(feature = "lab")]
    #[test]
    fn sp_lab_knobs_parse() {
        assert_eq!(parse_sp_force_k(None), Ok(SpForceK::Auto));
        assert_eq!(parse_sp_force_k(Some("all")), Ok(SpForceK::All));
        assert_eq!(parse_sp_force_k(Some("cpu")), Ok(SpForceK::Cpu));
        assert_eq!(parse_sp_force_k(Some("auto")), Ok(SpForceK::Auto));
        for bad in ["", "ALL", "1", "gpu", " cpu"] {
            assert!(parse_sp_force_k(Some(bad)).is_err(), "{bad:?}");
        }
        assert!(parse_sp_trace(Some("1")));
        for off in [None, Some("0"), Some(""), Some("true"), Some("2")] {
            assert!(!parse_sp_trace(off), "{off:?}");
        }
    }

    // ── admit_sp ────────────────────────────────────────────────────

    fn facts() -> SpLoadFacts<'static> {
        SpLoadFacts {
            knob: true,
            touched_knob: false,
            force_all: false,
            pages: 18,
            f16_route_ready: true,
            window_rows: 16_896,
            s1_route: Some(WindowRoute::Aggregate),
            arch: "gfx1100",
            n_exp: 512,
            cpu_gate_up_quant: true,
            cpu_down_quant: true,
            splice_workers: Some(16),
            region_fit_rows: 5_593,
            hidden: 2560,
            pinned_budget_bytes: 1 << 30,
            prefix_cache: false,
        }
    }

    #[test]
    fn admit_sp_truth_table() {
        let ok = admit_sp(&facts()).unwrap();
        assert_eq!(ok.mode, SpMode::Hybrid);
        assert_eq!(ok.sp_max_rows, 5_376);
        assert_eq!(ok.s_cap, 26_880);
        assert_eq!(ok.workers, 16);
        // §6: about 158 MiB pinned at the cap.
        let mib = ok.pinned_bytes as f64 / (1u64 << 20) as f64;
        assert!((157.0..159.5).contains(&mib), "{mib}");

        let cases: Vec<(&str, SpLoadFacts<'static>, SpDecline)> = vec![
            (
                "knob",
                SpLoadFacts {
                    knob: false,
                    ..facts()
                },
                SpDecline::KnobOff,
            ),
            (
                "no s1",
                SpLoadFacts {
                    s1_route: None,
                    ..facts()
                },
                SpDecline::NoS1Profile,
            ),
            (
                "arch 1201",
                SpLoadFacts {
                    arch: "gfx1201",
                    ..facts()
                },
                SpDecline::ArchNotCalibrated {
                    arch: "gfx1201".into(),
                },
            ),
            (
                "arch 1151",
                SpLoadFacts {
                    arch: "gfx1151",
                    ..facts()
                },
                SpDecline::ArchNotCalibrated {
                    arch: "gfx1151".into(),
                },
            ),
            (
                "route slab",
                SpLoadFacts {
                    s1_route: Some(WindowRoute::SlabIu4),
                    ..facts()
                },
                SpDecline::Route {
                    route: WindowRoute::SlabIu4,
                    mode: SpMode::Hybrid,
                },
            ),
            (
                "route sym iu4, no F16 route",
                SpLoadFacts {
                    s1_route: Some(WindowRoute::AggregateSymIu4),
                    f16_route_ready: false,
                    ..facts()
                },
                SpDecline::F16Route {
                    route: WindowRoute::AggregateSymIu4,
                },
            ),
            (
                "n_exp 256",
                SpLoadFacts {
                    n_exp: 256,
                    ..facts()
                },
                SpDecline::ExpertCount { n_exp: 256 },
            ),
            (
                "gate_up quant",
                SpLoadFacts {
                    cpu_gate_up_quant: false,
                    ..facts()
                },
                SpDecline::Quants {
                    gate_up: false,
                    down: true,
                },
            ),
            (
                "down quant",
                SpLoadFacts {
                    cpu_down_quant: false,
                    ..facts()
                },
                SpDecline::Quants {
                    gate_up: true,
                    down: false,
                },
            ),
            (
                "pool none",
                SpLoadFacts {
                    splice_workers: None,
                    ..facts()
                },
                SpDecline::SplicePool,
            ),
            (
                "pool zero",
                SpLoadFacts {
                    splice_workers: Some(0),
                    ..facts()
                },
                SpDecline::SplicePool,
            ),
            (
                "region",
                SpLoadFacts {
                    region_fit_rows: 2_047,
                    ..facts()
                },
                SpDecline::RegionFit {
                    fit_rows: 2_047,
                    minimum: 2_048,
                    window_rows: 16_896,
                },
            ),
            (
                "prefix cache",
                SpLoadFacts {
                    prefix_cache: true,
                    ..facts()
                },
                SpDecline::PrefixCache,
            ),
        ];
        for (name, input, want) in cases {
            let got = admit_sp(&input).unwrap_err();
            assert_eq!(got, want, "{name}");
            assert!(got.to_string().starts_with("sp: "), "{name}: {got}");
        }
        // Region fit below the cap lowers sp_max_rows and s_cap; the minimum arms.
        let tight = admit_sp(&SpLoadFacts {
            region_fit_rows: 2_048,
            ..facts()
        })
        .unwrap();
        assert_eq!((tight.sp_max_rows, tight.s_cap), (2_048, 10_240));
        // A fit above the cap is clamped to it.
        let loose = admit_sp(&SpLoadFacts {
            region_fit_rows: 100_000,
            ..facts()
        })
        .unwrap();
        assert_eq!(loose.sp_max_rows, SP_MAX_ROWS);
        // Pinned fit: exact boundary.
        let need = ok.pinned_bytes;
        assert!(admit_sp(&SpLoadFacts {
            pinned_budget_bytes: need,
            ..facts()
        })
        .is_ok());
        assert_eq!(
            admit_sp(&SpLoadFacts {
                pinned_budget_bytes: need - 1,
                ..facts()
            })
            .unwrap_err(),
            SpDecline::PinnedFit {
                need: Some(need),
                budget: need - 1
            }
        );
        // Order: the knob is first; the arch precedes the route.
        assert_eq!(
            admit_sp(&SpLoadFacts {
                knob: false,
                arch: "gfx1201",
                prefix_cache: true,
                ..facts()
            })
            .unwrap_err(),
            SpDecline::KnobOff
        );
        assert_eq!(
            admit_sp(&SpLoadFacts {
                arch: "gfx1201",
                s1_route: Some(WindowRoute::SlabIu4),
                ..facts()
            })
            .unwrap_err()
            .label(),
            "arch not calibrated"
        );
    }

    #[test]
    fn sp_mode_selection_truth_table() {
        // (qwen4_sp, qwen4_sp_touched, forced k = U) -> mode
        let table = [
            (false, false, false, None),
            (false, false, true, None),
            (false, true, false, Some(SpMode::TouchedOnly)),
            (false, true, true, Some(SpMode::TouchedOnly)),
            (true, false, false, Some(SpMode::Hybrid)),
            (true, false, true, Some(SpMode::TouchedOnly)),
            // Both knobs on: the hybrid mode wins unless the placement is forced to k = U.
            (true, true, false, Some(SpMode::Hybrid)),
            (true, true, true, Some(SpMode::TouchedOnly)),
        ];
        for (hybrid, touched, force_all, want) in table {
            assert_eq!(
                sp_mode_for(hybrid, touched, force_all),
                want,
                "sp {hybrid} touched {touched} force_all {force_all}"
            );
            let admitted = admit_sp(&SpLoadFacts {
                knob: hybrid,
                touched_knob: touched,
                force_all,
                ..facts()
            });
            match want {
                None => assert_eq!(admitted.unwrap_err(), SpDecline::KnobOff),
                Some(mode) => assert_eq!(admitted.unwrap().mode, mode),
            }
        }
        assert_eq!(SpMode::Hybrid.label(), "hybrid");
        assert_eq!(SpMode::TouchedOnly.label(), "touched_only");
        // Only the hybrid mode runs the SP route layout.
        assert!(SpMode::Hybrid.sp_route_layout());
        assert!(!SpMode::TouchedOnly.sp_route_layout());
    }

    fn touched_facts() -> SpLoadFacts<'static> {
        SpLoadFacts {
            knob: false,
            touched_knob: true,
            ..facts()
        }
    }

    #[test]
    fn admit_touched_only_needs_no_cpu_pinned_or_quant_facts() {
        // Everything the hybrid mode needs for the CPU arm is absent: still armed.
        let bare = SpLoadFacts {
            n_exp: 256,
            cpu_gate_up_quant: false,
            cpu_down_quant: false,
            splice_workers: None,
            region_fit_rows: 0,
            pinned_budget_bytes: 0,
            prefix_cache: true,
            pages: 8,
            ..touched_facts()
        };
        let ok = admit_sp(&bare).unwrap();
        assert_eq!(
            ok,
            SpProfile {
                mode: SpMode::TouchedOnly,
                base_route: WindowRoute::Aggregate,
                window_route: WindowRoute::Aggregate,
                sp_max_rows: SP_MAX_ROWS,
                s_cap: 0,
                workers: 0,
                pinned_bytes: 0,
            }
        );
        // The row cap is the S1 window's, clamped to SP's.
        for (window_rows, want) in [
            (16_896, SP_MAX_ROWS),
            (3_072, 3_072),
            (SP_MIN_REGION_ROWS, 2_048),
        ] {
            let profile = admit_sp(&SpLoadFacts {
                window_rows,
                ..touched_facts()
            })
            .unwrap();
            assert_eq!(profile.sp_max_rows, want, "{window_rows}");
        }
        // The shared rules decline it by name, in the shared order.
        let cases: Vec<(&str, SpLoadFacts<'static>, SpDecline)> = vec![
            (
                "no s1",
                SpLoadFacts {
                    s1_route: None,
                    ..touched_facts()
                },
                SpDecline::NoS1Profile,
            ),
            (
                "arch",
                SpLoadFacts {
                    arch: "gfx1151",
                    ..touched_facts()
                },
                SpDecline::ArchNotCalibrated {
                    arch: "gfx1151".into(),
                },
            ),
            (
                "route slab",
                SpLoadFacts {
                    s1_route: Some(WindowRoute::SlabIu4),
                    ..touched_facts()
                },
                SpDecline::Route {
                    route: WindowRoute::SlabIu4,
                    mode: SpMode::TouchedOnly,
                },
            ),
            (
                "rows",
                SpLoadFacts {
                    window_rows: 2_047,
                    ..touched_facts()
                },
                SpDecline::RegionFit {
                    fit_rows: 2_047,
                    minimum: 2_048,
                    window_rows: 2_047,
                },
            ),
            (
                "pages",
                SpLoadFacts {
                    pages: 15,
                    ..touched_facts()
                },
                SpDecline::Pages {
                    pages: 15,
                    need: 16,
                },
            ),
        ];
        for (name, input, want) in cases {
            let got = admit_sp(&input).unwrap_err();
            assert_eq!(got, want, "{name}");
            assert!(got.to_string().starts_with("sp: "), "{name}: {got}");
        }
        // A ring of exactly the groups of an all-expert layer arms; 256 experts need 8 pages.
        assert!(admit_sp(&SpLoadFacts {
            pages: 16,
            ..touched_facts()
        })
        .is_ok());
        assert!(admit_sp(&SpLoadFacts {
            pages: 8,
            n_exp: 256,
            ..touched_facts()
        })
        .is_ok());
        // Hybrid wins with both knobs on and keeps its own CPU requirements; when it declines
        // the touched-only mode (its knob on) is the fallback, and the hybrid decline is
        // reported beside it. Without the touched-only knob the hybrid decline is the verdict.
        let both_no_pool = SpLoadFacts {
            knob: true,
            splice_workers: None,
            ..touched_facts()
        };
        let (profile, note) = admit_sp_noted(&both_no_pool).unwrap();
        assert_eq!(profile.mode, SpMode::TouchedOnly);
        assert_eq!(note, Some(SpDecline::SplicePool));
        assert_eq!(
            admit_sp(&SpLoadFacts {
                touched_knob: false,
                ..both_no_pool
            })
            .unwrap_err(),
            SpDecline::SplicePool
        );
    }

    /// Regression (card job vt): on the gfx1100 default the S1 window route is the symmetric
    /// IU4 aggregate (`A4 sidecars` in the load line). Touched-only admission demanded the F16
    /// `Aggregate` route, declined `sp: route` silently, and the request ran stream-all S1.
    /// Touched-only runs S1's own pipeline and serves both aggregate routes; the hybrid mode
    /// runs its windows on the F16 route (a per-window override) when the profile can host it
    /// and otherwise declines by name.
    #[test]
    fn both_modes_arm_on_the_default_sym_iu4_route_and_the_hybrid_declines_by_name() {
        for route in [WindowRoute::Aggregate, WindowRoute::AggregateSymIu4] {
            let touched = admit_sp(&SpLoadFacts {
                s1_route: Some(route),
                ..touched_facts()
            })
            .unwrap_or_else(|d| panic!("{route:?}: {d}"));
            assert_eq!(touched.mode, SpMode::TouchedOnly, "{route:?}");
            // Touched-only never overrides a window's route.
            assert_eq!((touched.base_route, touched.window_route), (route, route));
            // The forced placement k = U reaches the same mode on the same route.
            let forced = admit_sp(&SpLoadFacts {
                s1_route: Some(route),
                force_all: true,
                ..facts()
            })
            .unwrap();
            assert_eq!(forced.mode, SpMode::TouchedOnly, "{route:?}");
            // The hybrid mode always runs its windows on the F16 route.
            let hybrid = admit_sp(&SpLoadFacts {
                s1_route: Some(route),
                ..facts()
            })
            .unwrap_or_else(|d| panic!("{route:?}: {d}"));
            assert_eq!(hybrid.mode, SpMode::Hybrid, "{route:?}");
            assert_eq!(hybrid.base_route, route);
            assert_eq!(hybrid.window_route, WindowRoute::Aggregate);
        }
        // Everything the F16 route needs missing on the symmetric profile: a named decline.
        let declined = admit_sp(&SpLoadFacts {
            s1_route: Some(WindowRoute::AggregateSymIu4),
            f16_route_ready: false,
            ..facts()
        })
        .unwrap_err();
        assert_eq!(declined.label(), "f16 route");
        let text = declined.to_string();
        assert!(
            text.starts_with("sp: f16 route")
                && text.contains("AggregateSymIu4")
                && text.contains("HIPFIRE_QWEN4_MOE_SYM_IU4=0"),
            "{text}"
        );
        // The readiness fact is irrelevant on the F16 profile and to touched-only.
        assert!(admit_sp(&SpLoadFacts {
            f16_route_ready: false,
            ..facts()
        })
        .is_ok());
        assert!(admit_sp(&SpLoadFacts {
            s1_route: Some(WindowRoute::AggregateSymIu4),
            f16_route_ready: false,
            ..touched_facts()
        })
        .is_ok());
        // The slab route is neither mode's.
        for knobs in [facts(), touched_facts()] {
            assert_eq!(
                admit_sp(&SpLoadFacts {
                    s1_route: Some(WindowRoute::SlabIu4),
                    ..knobs
                })
                .unwrap_err()
                .label(),
                "route"
            );
        }
    }

    #[test]
    fn per_window_route_is_the_sp_route_for_sp_windows_only() {
        let sym = WindowRoute::AggregateSymIu4;
        let f16 = WindowRoute::Aggregate;
        let hybrid = admit_sp(&SpLoadFacts {
            s1_route: Some(sym),
            ..facts()
        })
        .unwrap();
        let touched = admit_sp(&SpLoadFacts {
            s1_route: Some(sym),
            ..touched_facts()
        })
        .unwrap();
        // (profile route, SP profile, window qualified) -> window route
        assert_eq!(sp_window_route(sym, Some(&hybrid), true), f16);
        assert_eq!(sp_window_route(sym, Some(&hybrid), false), sym);
        assert_eq!(sp_window_route(sym, None, true), sym);
        assert_eq!(sp_window_route(sym, Some(&touched), true), sym);
        assert_eq!(sp_window_route(f16, Some(&hybrid), true), f16);
        assert_eq!(sp_window_route(f16, None, false), f16);
        // The F16 MoE scope: the append route's or a hybrid SP window's, nothing else.
        for (append, window, want) in [
            (false, false, false),
            (true, false, true),
            (false, true, true),
            (true, true, true),
        ] {
            assert_eq!(f16_moe_scope(append, window), want);
        }
    }

    #[test]
    fn sp_pinned_bytes_matches_the_design_terms() {
        let n = 5_376u64;
        let want = n * 5_120 + 5 * n * 5_120 + (10 * n + 7_680 + 15) / 16 * 16 * 4 + n * 40;
        assert_eq!(sp_pinned_bytes(2560, 512, 5_376), Some(want));
        assert_eq!(sp_grouped_bound(1, 512), Some(7_696));
        assert_eq!(sp_pinned_bytes(usize::MAX, 512, 5_376), None);
    }

    // ── qualify_sp_window ───────────────────────────────────────────

    fn profile() -> SpProfile {
        admit_sp(&facts()).unwrap()
    }

    /// A window of `rows` rows over `tile_rows`-row tiles with the production
    /// routing (`route_rows == rows`).
    fn window_of(rows: usize, tile_rows: usize) -> (WindowPlan, Vec<TilePlan>) {
        let mut tiles = Vec::new();
        let mut at = 0;
        while at < rows {
            let n = tile_rows.min(rows - at);
            tiles.push(TilePlan {
                index: tiles.len(),
                window: 0,
                prompt_row: at,
                window_row: at,
                stream_offset: 0,
                rows: n,
                position: at,
                routing: TileRouting {
                    route_rows: n,
                    bf16_streams: true,
                },
                ends_at_cut: None,
                last_in_window: at + n == rows,
            });
            at += n;
        }
        let k1_rows = tiles.iter().filter(|t| t.rows == 1).map(|t| t.rows).sum();
        let tail_rows = tiles
            .iter()
            .filter(|t| t.rows > 1 && t.rows < 512)
            .map(|t| t.rows)
            .sum();
        let window = WindowPlan {
            index: 0,
            prompt_row: 0,
            rows,
            position: 0,
            tiles: 0..tiles.len(),
            stream_bytes: 0,
            cuts: Vec::new(),
            k1_rows,
            tail_rows,
        };
        (window, tiles)
    }

    #[test]
    fn qualify_sp_window_row_bounds() {
        let sp = profile();
        let ok = |rows| {
            let (w, t) = window_of(rows, 768);
            qualify_sp_window(&sp, &w, &t, CaptureKind::None)
        };
        assert_eq!(
            ok(4),
            Err(SpBypass::TooFewRows {
                rows: 4,
                minimum: 5
            })
        );
        assert_eq!(ok(5), Ok(()));
        assert_eq!(ok(768), Ok(()));
        // 769 rows: a 768-row tile and a one-row tail tile (tile index 1).
        assert_eq!(ok(769), Err(SpBypass::OneRowTile { tile: 1 }));
        assert_eq!(ok(1024), Ok(()));
        assert_eq!(ok(5_376), Ok(()));
        assert_eq!(
            ok(5_377),
            Err(SpBypass::TooManyRows {
                rows: 5_377,
                maximum: 5_376
            })
        );
        // A tighter region lowers the maximum.
        let tight = admit_sp(&SpLoadFacts {
            region_fit_rows: 2_048,
            ..facts()
        })
        .unwrap();
        let (w, t) = window_of(2_049, 768);
        assert_eq!(
            qualify_sp_window(&tight, &w, &t, CaptureKind::None),
            Err(SpBypass::TooManyRows {
                rows: 2_049,
                maximum: 2_048
            })
        );
    }

    #[test]
    fn qualify_sp_window_cuts_k1_and_capture() {
        let sp = profile();
        let (mut w, t) = window_of(1024, 768);
        assert_eq!(qualify_sp_window(&sp, &w, &t, CaptureKind::Bounded), Ok(()));
        assert_eq!(qualify_sp_window(&sp, &w, &t, CaptureKind::None), Ok(()));
        assert_eq!(
            qualify_sp_window(&sp, &w, &t, CaptureKind::AllRows),
            Err(SpBypass::Capture(CaptureKind::AllRows))
        );
        assert_eq!(
            qualify_sp_window(&sp, &w, &t, CaptureKind::External),
            Err(SpBypass::Capture(CaptureKind::External))
        );
        w.cuts = vec![768];
        assert_eq!(
            qualify_sp_window(&sp, &w, &t, CaptureKind::None),
            Err(SpBypass::InteriorCuts { cuts: 1 })
        );
        w.cuts.clear();
        // K1 rows that survive K2-ization are refused even with no one-row tile.
        w.k1_rows = 3;
        assert_eq!(
            qualify_sp_window(&sp, &w, &t, CaptureKind::None),
            Err(SpBypass::K1Rows { rows: 3 })
        );
        // A one-row cut segment: a tile of one routing row.
        let (w2, mut t2) = window_of(1024, 768);
        t2[0].routing.route_rows = 1;
        assert_eq!(
            qualify_sp_window(&sp, &w2, &t2, CaptureKind::None),
            Err(SpBypass::OneRowTile { tile: 0 })
        );
        // F32-tail tiles (2..512 route rows) are K2-ized, so they qualify.
        let (w3, t3) = window_of(768 + 100, 768);
        assert!(w3.tail_rows > 0 && w3.k1_rows == 0);
        assert_eq!(qualify_sp_window(&sp, &w3, &t3, CaptureKind::None), Ok(()));
    }

    // ── protocol ────────────────────────────────────────────────────

    /// Fake events: each flag is a layer-indexed completion.
    #[derive(Default)]
    struct Fake {
        x_done: Cell<bool>,
        rows_ready: Cell<bool>,
        combine_done: Cell<bool>,
    }

    impl Fake {
        fn all() -> Self {
            Self {
                x_done: Cell::new(true),
                rows_ready: Cell::new(true),
                combine_done: Cell::new(true),
            }
        }
    }

    impl SpEventBackend for Fake {
        fn x_done_complete(&self, _layer: usize) -> bool {
            self.x_done.get()
        }
        fn rows_ready_complete(&self, _layer: usize) -> bool {
            self.rows_ready.get()
        }
        fn combine_done_recorded(&self, _layer: usize) -> bool {
            self.combine_done.get()
        }
    }

    /// Drive `p` for layer `layer` (2 tiles) up to and including `upto`.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    enum Step {
        Staged,
        Barrier,
        Planned,
        GpuEnqueued,
        Retired,
        CpuRunning,
        CpuWritten,
        CpuReturned,
        Uploaded,
        Combined,
    }

    fn drive(p: &mut SpProtocol, ev: &Fake, layer: usize, upto: Step) {
        use Step::*;
        p.stage_tile(layer, 0, 2).unwrap();
        p.stage_tile(layer, 1, 2).unwrap();
        if upto == Staged {
            return;
        }
        p.barrier(true).unwrap();
        if upto == Barrier {
            return;
        }
        p.plan(3, 18, 100, 1000).unwrap();
        if upto == Planned {
            return;
        }
        p.enqueue_gpu().unwrap();
        if upto == GpuEnqueued {
            return;
        }
        for _ in 0..3 {
            p.retire_group().unwrap();
        }
        if upto == Retired {
            return;
        }
        p.begin_cpu(ev).unwrap();
        if upto == CpuRunning {
            return;
        }
        p.host_write_cpu_rows(ev).unwrap();
        if upto == CpuWritten {
            return;
        }
        p.cpu_returned(true).unwrap();
        if upto == CpuReturned {
            return;
        }
        p.upload(ev).unwrap();
        if upto == Uploaded {
            return;
        }
        p.combine().unwrap();
    }

    fn started() -> SpProtocol {
        let mut p = SpProtocol::new();
        p.begin_window().unwrap();
        p
    }

    #[test]
    fn sp_protocol_happy_path_runs_two_layers_and_ends() {
        let ev = Fake::all();
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Combined);
        assert_eq!(p.state(), SpState::Combined);
        drive(&mut p, &ev, 5, Step::Combined);
        assert_eq!(p.layer(), Some(5));
        p.end_window().unwrap();
        assert_eq!(p.state(), SpState::Idle);
        // A new window may start again.
        p.begin_window().unwrap();
        assert!(!p.unavailable());
    }

    /// Every op from every state. A cell is `Ok` only on the one legal edge;
    /// everything else is refused and poisons.
    #[test]
    fn sp_protocol_transition_table_refuses_every_illegal_edge() {
        use Step::*;
        // The state a protocol is in after `drive(.., step)`, and the ops legal there.
        // op names: stage0, stage_next, barrier, plan, enqueue, retire, begin_cpu,
        // write, returned, upload, combine, end, begin_window
        let ops = [
            "stage_same_layer_tile2",
            "stage_next_layer",
            "barrier",
            "plan",
            "enqueue_gpu",
            "retire_group",
            "begin_cpu",
            "host_write",
            "cpu_returned",
            "upload",
            "combine",
            "end_window",
            "begin_window",
        ];
        let apply = |p: &mut SpProtocol, ev: &Fake, op: &str| -> Result<(), SpProtocolError> {
            match op {
                "stage_same_layer_tile2" => p.stage_tile(3, 2, 2),
                "stage_next_layer" => p.stage_tile(4, 0, 2),
                "barrier" => p.barrier(true),
                "plan" => p.plan(3, 18, 100, 1000),
                "enqueue_gpu" => p.enqueue_gpu(),
                "retire_group" => p.retire_group(),
                "begin_cpu" => p.begin_cpu(ev),
                "host_write" => p.host_write_cpu_rows(ev),
                "cpu_returned" => p.cpu_returned(true),
                "upload" => p.upload(ev),
                "combine" => p.combine(),
                "end_window" => p.end_window(),
                "begin_window" => p.begin_window(),
                _ => unreachable!(),
            }
        };
        // (state reached, legal ops from it). Retired (GpuEnqueued with all
        // groups retired) additionally allows nothing but begin_cpu.
        let table: [(Step, &[&str]); 10] = [
            (Staged, &["barrier"]),
            (Barrier, &["plan"]),
            (Planned, &["enqueue_gpu"]),
            (GpuEnqueued, &[]), // retire_group is legal only while groups remain: tested below
            (Retired, &["begin_cpu"]),
            (CpuRunning, &["host_write"]),
            (CpuWritten, &["cpu_returned"]),
            (CpuReturned, &["upload"]),
            (Uploaded, &["combine"]),
            (Combined, &["stage_next_layer", "end_window"]),
        ];
        // `Staged` additionally legal: none of the stage ops with a *third* tile of two.
        for (step, legal) in table {
            for op in ops {
                let ev = Fake::all();
                let mut p = started();
                drive(&mut p, &ev, 3, step);
                let before = p.state();
                let got = apply(&mut p, &ev, op);
                // GpuEnqueued with 3 groups pending: retire_group is its one legal op.
                let is_legal = legal.contains(&op) || (step == GpuEnqueued && op == "retire_group");
                if is_legal {
                    assert!(got.is_ok(), "{step:?} --{op}--> must be legal, got {got:?}");
                    assert_ne!(p.state(), SpState::Poisoned, "{step:?} {op}");
                } else {
                    assert!(
                        got.is_err(),
                        "{step:?} --{op}--> must be refused (was {before:?})"
                    );
                    assert_eq!(p.state(), SpState::Poisoned, "{step:?} {op} poisons");
                    assert!(p.unavailable());
                    // Poisoned refuses every further op without changing state.
                    for again in ops {
                        let r = apply(&mut p, &ev, again);
                        assert!(r.is_err(), "poisoned must refuse {again}");
                        assert_eq!(p.state(), SpState::Poisoned);
                    }
                }
            }
        }
        // `retire_group` beyond the enqueued groups, after Retired, is refused.
        let ev = Fake::all();
        let mut p = started();
        drive(&mut p, &ev, 3, Retired);
        assert_eq!(
            p.retire_group(),
            Err(SpProtocolError::TooManyRetires { groups: 3 })
        );
        // Idle (window primed, no layer): stage 0, end_window and a restart are legal; the rest refused.
        for op in ops {
            let ev = Fake::all();
            let mut p = started();
            let got = match op {
                "stage_same_layer_tile2" => p.stage_tile(3, 1, 2),
                "stage_next_layer" => p.stage_tile(4, 0, 2),
                other => apply(&mut p, &ev, other),
            };
            let legal = matches!(op, "stage_next_layer" | "end_window" | "begin_window");
            assert_eq!(got.is_ok(), legal, "Idle --{op}-->");
        }
    }

    #[test]
    fn sp_protocol_named_hazards() {
        let ev = Fake::all();

        // Plan twice.
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Planned);
        assert_eq!(
            p.plan(3, 18, 100, 1000),
            Err(SpProtocolError::Illegal {
                op: "plan",
                state: SpState::Planned
            })
        );

        // Combine before Uploaded.
        let mut p = started();
        drive(&mut p, &ev, 3, Step::CpuReturned);
        assert_eq!(
            p.combine(),
            Err(SpProtocolError::Illegal {
                op: "combine",
                state: SpState::CpuRunning
            })
        );

        // Stage L+1 before Combined(L), from every pre-Combined state.
        for step in [
            Step::Barrier,
            Step::Planned,
            Step::GpuEnqueued,
            Step::CpuRunning,
            Step::CpuReturned,
            Step::Uploaded,
        ] {
            let mut p = started();
            drive(&mut p, &ev, 3, step);
            assert!(p.stage_tile(4, 0, 2).is_err(), "{step:?}");
            assert_eq!(p.state(), SpState::Poisoned);
        }
        // Staging layer L+1 mid-layer is a wrong layer.
        let mut p = started();
        p.stage_tile(3, 0, 2).unwrap();
        assert_eq!(
            p.stage_tile(4, 1, 2),
            Err(SpProtocolError::WrongLayer {
                expected: 3,
                got: 4
            })
        );

        // Host write to cpu_rows before rows_ready[L-1] completes.
        let ev = Fake::all();
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Combined);
        drive(&mut p, &ev, 5, Step::Retired);
        ev.rows_ready.set(false);
        assert_eq!(
            p.begin_cpu(&ev),
            Err(SpProtocolError::RowsReadyPending { layer: 3 })
        );
        // ... and the write op itself rechecks (rows_ready regresses to pending).
        let ev = Fake::all();
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Combined);
        drive(&mut p, &ev, 5, Step::CpuRunning);
        ev.rows_ready.set(false);
        assert_eq!(
            p.host_write_cpu_rows(&ev),
            Err(SpProtocolError::RowsReadyPending { layer: 3 })
        );
        // Host write outside CpuRunning (before begin_cpu).
        let ev = Fake::all();
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Retired);
        assert!(matches!(
            p.host_write_cpu_rows(&ev),
            Err(SpProtocolError::Illegal {
                op: "host_write_cpu_rows",
                ..
            })
        ));

        // The first layer has no previous rows_ready to wait for.
        let ev = Fake::all();
        ev.rows_ready.set(false);
        let mut p = started();
        drive(&mut p, &ev, 3, Step::CpuRunning);

        // x_done not complete.
        let ev = Fake::all();
        ev.x_done.set(false);
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Retired);
        assert_eq!(
            p.begin_cpu(&ev),
            Err(SpProtocolError::XferNotDone { layer: 3 })
        );

        // Upload before combine_done[L-1] is recorded (I5).
        let ev = Fake::all();
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Combined);
        drive(&mut p, &ev, 5, Step::CpuReturned);
        ev.combine_done.set(false);
        assert_eq!(
            p.upload(&ev),
            Err(SpProtocolError::CombineDoneNotRecorded { layer: 3 })
        );

        // Upload before the CPU arm returned.
        let ev = Fake::all();
        let mut p = started();
        drive(&mut p, &ev, 3, Step::CpuWritten);
        assert_eq!(p.upload(&ev), Err(SpProtocolError::CpuNotReturned));

        // A layer at or below the last combined one.
        let ev = Fake::all();
        let mut p = started();
        drive(&mut p, &ev, 5, Step::Combined);
        assert_eq!(
            p.stage_tile(5, 0, 2),
            Err(SpProtocolError::WrongLayer {
                expected: 6,
                got: 5
            })
        );
    }

    #[test]
    fn sp_protocol_refuses_bad_plans_tiles_and_failures() {
        let ev = Fake::all();
        // Barrier before every tile staged.
        let mut p = started();
        p.stage_tile(3, 0, 3).unwrap();
        p.stage_tile(3, 1, 3).unwrap();
        assert_eq!(
            p.barrier(true),
            Err(SpProtocolError::TilesIncomplete {
                staged: 2,
                tiles: 3
            })
        );
        // Tile out of order / repeated / beyond the count / changed count.
        for (second, tiles) in [(2usize, 3usize), (0, 3), (1, 4)] {
            let mut p = started();
            p.stage_tile(3, 0, 3).unwrap();
            assert!(p.stage_tile(3, second, tiles).is_err());
            assert_eq!(p.state(), SpState::Poisoned);
        }
        let mut p = started();
        assert!(p.stage_tile(3, 1, 3).is_err(), "first tile must be 0");
        let mut p = started();
        assert!(p.stage_tile(3, 3, 3).is_err(), "tile index beyond count");
        let mut p = started();
        assert!(p.stage_tile(3, 0, 0).is_err(), "zero tiles");

        // Readback mismatch at the barrier poisons.
        let mut p = started();
        p.stage_tile(3, 0, 1).unwrap();
        assert_eq!(p.barrier(false), Err(SpProtocolError::ReadbackMismatch));
        assert_eq!(p.state(), SpState::Poisoned);

        // Groups above the pages, CPU slots above the cap.
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Barrier);
        assert_eq!(
            p.plan(19, 18, 10, 100),
            Err(SpProtocolError::GroupsExceedPages {
                groups: 19,
                pages: 18
            })
        );
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Barrier);
        assert_eq!(
            p.plan(3, 18, 101, 100),
            Err(SpProtocolError::CpuSlotsExceedCap {
                slots: 101,
                cap: 100
            })
        );
        // Boundaries are accepted.
        let mut p = started();
        drive(&mut p, &ev, 3, Step::Barrier);
        p.plan(18, 18, 100, 100).unwrap();

        // begin_cpu with groups still pending.
        let mut p = started();
        drive(&mut p, &ev, 3, Step::GpuEnqueued);
        p.retire_group().unwrap();
        assert_eq!(
            p.begin_cpu(&ev),
            Err(SpProtocolError::GroupsNotRetired {
                retired: 1,
                groups: 3
            })
        );

        // E1 error poisons.
        let mut p = started();
        drive(&mut p, &ev, 3, Step::CpuWritten);
        assert_eq!(p.cpu_returned(false), Err(SpProtocolError::CpuFailed));
        assert_eq!(p.state(), SpState::Poisoned);
        // cpu_returned before the host write is illegal.
        let mut p = started();
        drive(&mut p, &ev, 3, Step::CpuRunning);
        assert!(matches!(
            p.cpu_returned(true),
            Err(SpProtocolError::Illegal {
                op: "cpu_returned",
                ..
            })
        ));
    }

    #[test]
    fn sp_protocol_poison_drain_and_sticky_unavailable() {
        let ev = Fake::all();
        let mut p = started();
        drive(&mut p, &ev, 3, Step::GpuEnqueued);
        p.fail();
        assert_eq!(p.state(), SpState::Poisoned);
        assert_eq!(p.retire_group(), Err(SpProtocolError::Poisoned));
        assert_eq!(p.end_window(), Err(SpProtocolError::Poisoned));
        assert_eq!(p.begin_window(), Err(SpProtocolError::Poisoned));
        // fail_window after the drain leaves Poisoned but SP stays disabled.
        p.fail_window().unwrap();
        assert_eq!(p.state(), SpState::Idle);
        assert!(p.unavailable());
        assert_eq!(p.begin_window(), Err(SpProtocolError::Unavailable));
        // fail_window outside Poisoned is illegal.
        let mut p = started();
        assert!(p.fail_window().is_err());
        // A window may not begin mid-window, nor end mid-layer.
        let mut p = started();
        p.stage_tile(3, 0, 1).unwrap();
        assert!(p.begin_window().is_err());
        let mut p = started();
        p.stage_tile(3, 0, 1).unwrap();
        assert!(p.end_window().is_err());
    }
}
