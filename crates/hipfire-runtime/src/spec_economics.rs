// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative-decoding economics shared by every native MTP producer: the
//! per-depth agreement table, the provisional batched-depth chooser and the
//! same-request measured AR floor ([`Floor`]).
//!
//! Pure CPU and family-free: nothing here reads the environment, a GPU or a
//! random stream.  A producer's adapter supplies its window-cost table, its
//! decay and its retirement thresholds at construction and reads its own
//! environment flags (including whether to enable the exact call log, see
//! [`Floor::with_call_log`]).  Moved from `hipfire-arch-qwen4`'s `mtp_spec`
//! with identical arithmetic.

use crate::spec::MtpWindow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-window decay of the per-depth agreement counts, and of the AR floor's
/// measured window costs and n-gram yield history (~8-window memory): the
/// Flash-Next lanes' value, and the singleton drafter's on every arch
/// without a measured longer memory.
pub const DEFAULT_DECAY: f32 = 0.875;
/// Per-depth (accepted, compared) counts a request starts from (0.8).
pub const AGREEMENT_PRIOR: (f32, f32) = (1.6, 2.0);
/// Per-window relaxation of a depth the window did not compare toward the
/// agreement of the next-shallower depth (depth 0: `AGREEMENT_PRIOR`),
/// at the prior's weight, while that agreement is below the depth's own
/// (see [`AGREEMENT_RELAX_UP`]). An unobserved depth must drift back so
/// the chooser re-explores it: a ratio frozen below the depth's threshold
/// (one rejected depth-3 draft) keeps that depth out of every later window,
/// so it is never observed again. It drifts toward the request's own
/// deepest observed acceptance rather than a fixed 0.8 because the
/// conditional acceptance is flat in depth past the first draft
/// (Flash-Next, forced K=5: Halo code 0.89-0.96 and prose 0.68-0.76 at
/// depths 2-5; forced K=5/7 on the R9700: 0.70-0.86 at depths 2-7), so a
/// fixed prior under-prices every deeper draft of a code request and
/// over-prices a prose one.
const AGREEMENT_RELAX: f32 = 0.95;
/// [`AGREEMENT_RELAX`] while the shallower depth's agreement is above
/// the uncompared depth's own (~5-window memory instead of ~20), so the
/// depths past the one a request runs catch up with its body's acceptance
/// after the early rejections a response's first windows often take
/// (the singleton retirement threshold); falling stays slow, so a depth's
/// own rejections are not forgotten.  On the deterministic CPU streams of
/// the qwen4 `chooser_*` tests (16 seeds, Halo window costs, code acceptance) it
/// emits ~1% more tokens per ms than 0.95 both ways; prose and R9700
/// streams move by less than 1%.
const AGREEMENT_RELAX_UP: f32 = 0.8;
/// Window cost of one more drafted row, in AR steps, that the chooser's
/// per-request cost fit assumes until the request has priced windows at two
/// drafted depths ([`Floor::modeled_us`]): gfx1151 Flash-Next mq4-xts,
/// 6.43 ms of verify plus ~1.3 ms of draft step per row against a 28.45 ms
/// AR step (`release-0.4.3/device-sample/VERIFY-BREAKDOWN.md` §7). The
/// R9700's row is ~0.52 AR steps (§3); the prior only prices the first
/// one-depth exploration, after which the request's own slope takes over.
const ROW_COST_PRIOR: f64 = 0.27;
/// Ridge weight of [`ROW_COST_PRIOR`] in the per-request cost fit, in
/// the fit's depth-spread units (`Σw(d - d̄)²`): two windows one depth
/// apart, so a handful of windows at two depths outweigh it.
const ROW_COST_PRIOR_WEIGHT: f64 = 0.5;
/// Draft depths tracked (the drafter's K is clamped to this).
pub const MAX_DEPTH: usize = 10;

/// Speculative windows (native or external, `k > 0`) the AR floor observes
/// before its first measured decision.  Together with the one calibration
/// token this bounds the speculation work that precedes the first decision:
/// at most one head append plus this many windows.
pub const FLOOR_PROBE_WINDOWS: usize = 3;

/// Per-window decay of the realized-cost memory a retirement must be
/// confirmed against (at least the floor's own decay): about 25 windows.
/// The 8-window `combined` memory dips below AR on any streak of rejected
/// drafts (records_parser_1k: 27.8 ms/token at window 15 against a 19.2
/// ms/token mean) and a retirement is sticky.
pub const FLOOR_RETIRE_DECAY: f32 = 0.96;

/// Routed depth at or below which native evidence is shallow: the incumbent
/// cap-3 chooser, the depth a deeper (capability-default) budget must be able
/// to fall back to before a request retires stickily.
pub const FLOOR_SHALLOW_CAP: usize = 3;

/// Hard bound on the windows one shallow re-probe may spend (native or
/// external, first-use unpriced ones included): twice
/// [`FLOOR_PROBE_WINDOWS`], so every priced window of the probe budget
/// can still be preceded by one unpriced first use.
pub const FLOOR_SHALLOW_MAX_WINDOWS: usize = 2 * FLOOR_PROBE_WINDOWS;

/// Phase of the fresh-shallow evidence a would-be retirement of a deeper
/// budget asks for ([`Floor::retire_or_probe`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ShallowState {
    /// No re-probe running or decided since the last deep window.
    #[default]
    Idle,
    /// Shallow windows are running; the verdict waits for
    /// [`FLOOR_PROBE_WINDOWS`] priced ones or the window bound.
    Probing,
    /// Shallow priced below AR: no retirement while that holds; deeper caps
    /// need positive evidence ([`Floor::update_climb_evidence`]).
    Holding,
    /// Priced shallow windows did not beat AR: the verdict retires.
    Lost,
    /// The window bound ended with no priced shallow window (every timing
    /// invalid): no win can be asserted, so the verdict retires (the
    /// invalid-AR-price rule).
    Unpriced,
}

impl ShallowState {
    fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Probing => "probing",
            Self::Holding => "holding",
            Self::Lost => "lost",
            Self::Unpriced => "unpriced",
        }
    }
}

/// Fresh shallow evidence of one re-probe (then, while holding, its decayed
/// continuation).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ShallowEvidence {
    /// Windows spent, priced or not.
    windows: usize,
    /// Windows with a valid timing.
    priced: usize,
    /// (wall us, emitted tokens) of the priced windows.
    cost: (f32, f32),
}

/// Per-depth decayed (accepted, compared) draft counts.
pub type Agreement = [(f32, f32); MAX_DEPTH];

/// A fresh request's agreement table: every depth at the prior.
pub fn fresh_agreement() -> Agreement {
    [AGREEMENT_PRIOR; MAX_DEPTH]
}

/// What the floor picks for the next window.  The two AR kinds are distinct
/// actions: neither is the interleaved route's "K0".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Ordinary target-only token, no head work (after sticky retirement: the
    /// head may lag).
    ArHeadFree,
    /// Ordinary target-only token timed for the floor, head kept synced (an
    /// AR probe of a would-be retirement).
    ArSynced,
    /// One emitted token per head-synced step.
    Interleaved,
    /// A batched window drafting this many tokens.
    Native(usize),
}

/// Which native route a measured window ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeWindow {
    Interleaved,
    Batched,
}

/// A usable wall time: finite and positive.
pub fn valid_us(us: f32) -> bool {
    us.is_finite() && us > 0.0
}

/// Add one observation to a `decay`-decayed `(sum, weight)` pair; only the
/// entry being observed decays.
fn decay_add(entry: &mut (f32, f32), sum: f32, weight: f32, decay: f32) {
    entry.0 = entry.0 * decay + sum;
    entry.1 = entry.1 * decay + weight;
}

/// Only the drafts compared before the first rejection count, so each
/// depth measures the agreement conditional on its prefix being accepted
/// (the same quantity on both routes). Compared depths decay by `decay` and
/// take the observation; depths the window did not compare relax toward
/// the next-shallower depth's agreement at the weight of
/// `AGREEMENT_PRIOR` (depth 0 toward the prior itself) instead of
/// decaying: up at `AGREEMENT_RELAX_UP`, down at `AGREEMENT_RELAX`.
/// A depth rejected once is re-explored rather than frozen below its
/// threshold, and depths no window reached are priced at the request's
/// deepest observed acceptance.
pub fn observe_agreement(agreement: &mut [(f32, f32)], window: &MtpWindow, decay: f32) {
    observe_agreement_counts(agreement, window.accepted, window.drafts_generated, decay);
}

/// [`observe_agreement`] on a window's `(accepted, drafts_generated)` alone,
/// for observations with no committed tokens to carry (an n-gram shadow).
pub fn observe_agreement_counts(
    agreement: &mut [(f32, f32)],
    accepted_drafts: usize,
    drafts: usize,
    decay: f32,
) {
    let compared = (accepted_drafts + 1).min(drafts);
    let prior_total = AGREEMENT_PRIOR.1;
    let mut shallower = AGREEMENT_PRIOR;
    for (depth, entry) in agreement.iter_mut().enumerate() {
        let (accepted, total) = entry;
        if depth < compared {
            *accepted *= decay;
            *total *= decay;
            *total += 1.0;
            *accepted += f32::from(u8::from(depth < accepted_drafts));
        } else {
            let target = shallower.0 / shallower.1.max(f32::MIN_POSITIVE);
            let relax = if target > *accepted / total.max(f32::MIN_POSITIVE) {
                AGREEMENT_RELAX_UP
            } else {
                AGREEMENT_RELAX
            };
            *accepted = *accepted * relax + prior_total * target * (1.0 - relax);
            *total = *total * relax + prior_total * (1.0 - relax);
        }
        shallower = *entry;
    }
}

/// Draft depth for the next window: the K maximizing expected emitted
/// tokens (1 + sum over depths of the accepted-prefix probability) per
/// window cost, or 0 (interleaved) when no batched window beats one
/// token per interleaved step.  `window_cost[d]` prices a window drafting `d`
/// tokens, in interleaved-route emitted tokens.
pub fn batched_depth(agreement: &[(f32, f32)], k: usize, window_cost: &[f32]) -> usize {
    let mut best = (0, 1.0f32);
    let mut prefix = 1.0f32;
    let mut expected = 1.0f32;
    let depths = agreement.iter().zip(&window_cost[1..]).take(k);
    for (i, (&(accepted, total), &cost)) in depths.enumerate() {
        let depth = i + 1;
        prefix *= accepted / total.max(f32::MIN_POSITIVE);
        expected += prefix;
        let rate = expected / cost;
        if rate > best.1 {
            best = (depth, rate);
        }
    }
    best.0
}

/// One finished native window as the floor prices it.
#[derive(Clone, Copy, Debug)]
pub struct WindowStats {
    /// Committed tokens (accepted drafts plus the bonus).
    pub emitted: usize,
    /// Drafts generated (the margin prune may stop below `cap`).
    pub drafts: usize,
    /// The depth the window was routed at (batched); 0 for interleaved.
    pub cap: usize,
}

impl WindowStats {
    pub fn of(window: &MtpWindow, cap: usize) -> Self {
        Self {
            emitted: window.committed.len(),
            drafts: window.drafts_generated,
            cap,
        }
    }
}

/// Decayed statistics of the batched windows routed at one cap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapStats<const W: usize> {
    /// Wall us.
    pub us: f32,
    /// Committed tokens.
    pub emitted: f32,
    /// Decayed windows routed at this cap.
    pub windows: f32,
    /// Windows by the depth they drafted (the margin prune stops some below
    /// the cap).
    pub drafted: [f32; W],
}

impl<const W: usize> Default for CapStats<W> {
    fn default() -> Self {
        Self {
            us: 0.0,
            emitted: 0.0,
            windows: 0.0,
            drafted: [0.0; W],
        }
    }
}

/// Decayed statistics of batched windows routed at one dynamic cap beyond
/// the static prefix: the same scalars as [`CapStats`] plus a grow-only
/// drafted-depth histogram sized at [`Floor::set_capacity`], never per window.
#[derive(Clone, Debug, Default, PartialEq)]
struct DynCapStats {
    us: f32,
    emitted: f32,
    windows: f32,
    drafted: Vec<f32>,
}

/// Expected tokens a window drafting `d` tokens emits, for every `d`: the
/// bonus plus the accepted-prefix probabilities of the per-depth agreement.
fn expected_emitted_by_depth<const W: usize>(agreement: &[(f32, f32)]) -> [f32; W] {
    let mut expected = [1.0f32; W];
    let mut prefix = 1.0f32;
    for d in 1..expected.len() {
        let (accepted, total) = agreement.get(d - 1).copied().unwrap_or(AGREEMENT_PRIOR);
        prefix *= accepted / total.max(f32::MIN_POSITIVE);
        expected[d] = expected[d - 1] + prefix;
    }
    expected
}

/// Per-process sequence number of the next call-log line.
static FLOOR_CALL_SEQ: AtomicU64 = AtomicU64::new(0);

/// Per-process source of [`Floor`] ids.
static FLOOR_ID_SEQ: AtomicU64 = AtomicU64::new(0);

fn next_floor_id() -> u64 {
    FLOOR_ID_SEQ.fetch_add(1, Ordering::Relaxed)
}

/// One call-log line: the call `op` into floor `floor_id`, number `seq` of
/// the process.  `fields` is empty or starts with a comma.
fn floor_call_line(
    prefix: &str,
    seq: u64,
    floor_id: u64,
    op: &str,
    fields: std::fmt::Arguments<'_>,
) -> String {
    format!("{prefix} {{\"seq\":{seq},\"floor\":{floor_id},\"op\":\"{op}\"{fields}}}")
}

/// Where a floor writes its call-log lines, and under which prefix.
#[derive(Clone, Copy)]
struct CallLog {
    prefix: &'static str,
    sink: fn(&str),
}

impl std::fmt::Debug for CallLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallLog")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl CallLog {
    /// Write the line of one call to the sink.
    fn emit(self, op: &str, floor_id: u64, fields: std::fmt::Arguments<'_>) {
        let seq = FLOOR_CALL_SEQ.fetch_add(1, Ordering::Relaxed);
        (self.sink)(&floor_call_line(self.prefix, seq, floor_id, op, fields));
    }
}

/// The production call-log sink: one line on stderr.
fn stderr_sink(line: &str) {
    eprintln!("{line}");
}

/// An `f32` as its exact bits: a quoted lowercase hex `u32` (`1.0` is
/// `"3f800000"`).
struct FloorBits(f32);

impl std::fmt::Display for FloorBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\"{:08x}\"", self.0.to_bits())
    }
}

/// An [`Agreement`] as `[[acc_bits,total_bits],...]`.
struct FloorAgreement<'a>(&'a [(f32, f32)]);

impl std::fmt::Display for FloorAgreement<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (depth, &(accepted, total)) in self.0.iter().enumerate() {
            if depth > 0 {
                f.write_str(",")?;
            }
            write!(f, "[{},{}]", FloorBits(accepted), FloorBits(total))?;
        }
        f.write_str("]")
    }
}

/// An [`Action`] as the log's route name: `"Ar"` (`ArHeadFree`), `"ArSample"`
/// (`ArSynced`), `"Interleaved"` or `"Batched:<depth>"` (`Native`).  The names
/// predate the enum's rename and stay so recorded logs replay.
fn action_name(action: Action) -> String {
    match action {
        Action::ArHeadFree => "Ar".to_string(),
        Action::ArSynced => "ArSample".to_string(),
        Action::Interleaved => "Interleaved".to_string(),
        Action::Native(depth) => format!("Batched:{depth}"),
    }
}

struct FloorRouteName(Action);

impl std::fmt::Display for FloorRouteName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\"{}\"", action_name(self.0))
    }
}

/// A [`NativeWindow`] as `"Interleaved"` or `"Batched"`.
struct NativeWindowName(NativeWindow);

impl std::fmt::Display for NativeWindowName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            NativeWindow::Interleaved => "\"Interleaved\"",
            NativeWindow::Batched => "\"Batched\"",
        })
    }
}

/// The floor state a caller branches on after a call:
/// `,"retired_after":..,"needs_ar_after":..`.
struct FloorAfter {
    retired: bool,
    needs_ar: bool,
}

impl std::fmt::Display for FloorAfter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            ",\"retired_after\":{},\"needs_ar_after\":{}",
            self.retired, self.needs_ar
        )
    }
}

/// The fields of one `route` call line.
struct RouteFields<'a> {
    agreement: &'a [(f32, f32)],
    k: usize,
    provisional: usize,
    ret: Action,
    after: FloorAfter,
}

impl std::fmt::Display for RouteFields<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            ",\"agreement\":{},\"k\":{},\"provisional\":{},\"ret\":{}{}",
            FloorAgreement(self.agreement),
            self.k,
            self.provisional,
            FloorRouteName(self.ret),
            self.after
        )
    }
}

/// A copy of a floor's measured state, for diagnostics and tests
/// ([`Floor::diagnostics`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloorDiagnostics<const W: usize, const E: usize> {
    pub calibrated: bool,
    /// Microseconds of one ordinary AR token; 0 when the measurement was invalid.
    pub ar_us: f32,
    /// Probe-budget windows counted (capped at [`FLOOR_PROBE_WINDOWS`]).
    pub probes: usize,
    /// Speculative windows observed (uncapped).
    pub observed: usize,
    pub retire_after: usize,
    pub retired: bool,
    /// AR probe tokens still to run.
    pub probe_left: usize,
    pub probe_tokens: usize,
    pub decay: f32,
    /// Per drafted depth: decayed (wall us, windows).
    pub batched: [(f32, f32); W],
    /// Per routed cap: decayed realized window statistics.
    pub caps: [CapStats<W>; W],
    /// The decayed least-squares sums of [`Floor::fit_line`].
    pub fit: [f64; 5],
    /// Decayed (wall us, emitted tokens) of the interleaved route.
    pub interleaved: (f32, f32),
    /// Decayed (wall us, emitted tokens) of external windows.
    pub external: (f32, f32),
    /// Per candidate count: decayed (wall us, windows) of takeover windows.
    pub external_depth: [(f32, f32); E],
    pub external_blocked: bool,
    /// Decayed realized (wall us, emitted tokens) over every valid window.
    pub combined: (f32, f32),
    /// The same with the slower retirement-confirmation memory.
    pub slow: (f32, f32),
}

/// Same-request AR floor of native MTP.
///
/// The price of AR is `ar_us`, measured on this request, on this GPU, by one
/// calibration token (the ordinary target-only forward, including a sampled
/// draw; never the drafter's first, which pays one-time costs).  Speculation is priced from this
/// request's own windows by their realized decayed wall time per emitted
/// token: a batched window under the depth (cap) it was routed at, the
/// margin prune's shorter windows included, and the interleaved route.  The
/// window-cost table only orders
/// the options while fewer than [`FLOOR_PROBE_WINDOWS`] windows have
/// been observed.  After that budget the AR verdict reads only measured
/// options: the request retires, stickily, to head-free AR as soon as no
/// measured option costs strictly less per emitted token than AR (a tie
/// retires).  An invalid AR measurement retires at the budget too: without
/// a price, a win cannot be asserted.
///
/// Depth choice (once a measured option beats AR): the candidates are the
/// measured caps plus every unmeasured cap next to a measured one, priced
/// from that neighbour ([`Self::cap_cost`]): one cap down saves a row on
/// the windows that drafted the whole cap and loses a token on those that
/// also accepted it all; one cap up adds a row on the former and gains the
/// next draft's agreement on the latter.  The row is the request's own
/// window-cost fit ([`Self::fit_line`]: a decayed least-squares line of
/// wall time on drafted depth).  The cheapest per token runs, so a request
/// climbs one cap at a time (up to `W - 1`) while the
/// next row pays.  Pricing by cap, not by drafted depth over the
/// agreement-predicted tokens, counts the windows the margin prune cuts
/// short at their real (lower) cost: on Halo prose a cap of 4 with the prune
/// beat cap 2 (forced K=4 +8.4%) while the drafted-depth model kept prose
/// requests at 2.
///
/// External (n-gram) windows are tracked separately and never feed the
/// native agreement; they count toward the shared probe budget.  A request
/// stays alive on a losing (or unmeasured) native route only while external
/// windows are measured and unblocked (cost per token below AR) AND the
/// realized per-token cost of all its windows, native and external, decayed
/// (`combined`), is strictly below AR (a tie loses).  Such a request routes
/// the cheapest measured native option within `k`, else interleaved; it
/// retires, stickily, at the first window where that stops holding.  A
/// measured native option that beats AR needs no external help.
///
/// Retire evidence (the singleton request, [`Self::with_retire_evidence`]):
/// the calibration token is one token, and under the Flash-Next expert hot
/// set one token's AR cost is a property of its routed experts, not of the
/// request (7–26% off the request's mean AR token on the R9700 traces).  A
/// verdict that would retire therefore first spends `probe_tokens` ordinary
/// AR tokens (the calibration step again, head kept synced; they are real
/// tokens, and retirement would run them anyway) and re-prices AR by their
/// mean, at most once per `retire_after` observed windows.  The verdict then
/// also needs the slow-memory (`slow`, ~25 windows) realized cost of all
/// windows to lose to that price: the 8-window memories dip on any streak of
/// rejected drafts and a retirement is sticky.
///
/// Fresh shallow evidence (singleton and lane floors, a budget above
/// [`FLOOR_SHALLOW_CAP`]): a request that explored deeper caps and whose
/// verdict would retire first needs fresh shallow native evidence, i.e.
/// [`FLOOR_PROBE_WINDOWS`] priced shallow windows (batched at a routed
/// cap ≤ 3, or interleaved) since the last deep window.  The decayed cap
/// statistics never count: only `shallow_since_deep`, which a deep window
/// zeroes.  Without it, after the AR re-price the floor spends at most
/// [`FLOOR_SHALLOW_MAX_WINDOWS`] windows on its best measured shallow
/// option (first-use unpriced windows and external windows spend the bound
/// unpriced); priced below AR it holds (no retirement, the climb waits for
/// [`FLOOR_PROBE_WINDOWS`] consecutive windows of a deeper candidate
/// pricing below the shallow one), priced at or above AR or never priced
/// it retires.  A budget of at most [`FLOOR_SHALLOW_CAP`] never probes.
#[derive(Clone, Debug)]
pub struct Floor<const W: usize, const E: usize> {
    calibrated: bool,
    /// Microseconds of one ordinary AR token; 0 when the measurement was
    /// invalid.
    ar_us: f32,
    probes: usize,
    /// Speculative windows observed (uncapped; the probe budget caps `probes`).
    observed: usize,
    /// `observed` a losing verdict needs before it retires the request.
    retire_after: usize,
    retired: bool,
    /// Per actually drafted depth in the historical measured prefix.
    batched: [(f32, f32); W],
    /// Per routed cap in the historical measured prefix.
    caps: [CapStats<W>; W],
    /// Per actually drafted depth beyond the static `W` prefix: decayed (wall
    /// us, windows) for depths `W..`, indexed `0 == W`. Empty until
    /// [`Self::set_capacity`] opts the request into a deeper capacity.
    /// Shrinkage only lowers the logical bound; storage (and observations)
    /// are kept for a later widen while the request lives.
    extra_batched: Vec<(f32, f32)>,
    /// Per routed cap beyond the static `W` prefix: decayed realized window
    /// statistics for caps `W..`, indexed `0 == W`. Empty until
    /// [`Self::set_capacity`] opts in.
    extra_caps: Vec<DynCapStats>,
    /// Admitted native depth capacity (`0` disables every batched option).
    /// Constructors leave the historical prefix (`W - 1`); only
    /// [`Self::set_capacity`] changes it.
    native_cap: usize,
    /// Whether [`Self::set_capacity`] was called (actual-bound enforcement,
    /// storage and logging). The unmeasured global-fit candidate domain
    /// additionally requires a genuinely wide admission (see
    /// [`Self::wide_fit`]); at or below the table the legacy
    /// measured/neighbour pricing is kept bit-arithmetic.
    extended: bool,
    /// Decayed least-squares sums `[w, Σd, Σus, Σd², Σd·us]` of every priced
    /// batched window's wall `us` on its drafted depth `d`, the data of
    /// [`Self::modeled_us`] (f64: the cross terms of ~1e5 us cancel).
    fit: [f64; 5],
    /// Decayed (wall us, emitted tokens).
    interleaved: (f32, f32),
    /// Decayed (wall us, emitted tokens) of external windows.
    external: (f32, f32),
    /// Per candidate count: decayed (wall us, windows) of this request's
    /// n-gram takeover windows, what [`Self::takeover_cost`] prices a take
    /// length by.
    external_depth: [(f32, f32); E],
    external_blocked: bool,
    /// Decayed realized (wall us, emitted tokens) over every valid window
    /// observed after calibration, native and external alike.
    combined: (f32, f32),
    /// Per-window decay of every decayed pair above.
    decay: f32,
    /// Realized (wall us, emitted tokens) over every valid window after
    /// calibration with the slower [`FLOOR_RETIRE_DECAY`] memory.
    slow: (f32, f32),
    /// AR probe tokens a would-be retirement takes (0 disables the probes
    /// and the `slow` confirmation).
    probe_tokens: usize,
    /// AR probe tokens still to run.
    probe_left: usize,
    /// Valid wall times of the probe tokens run so far.
    probe_samples: Vec<f32>,
    /// `observed` when the last probe set started.
    probed_at: Option<usize>,
    /// Draft budget `k` of the last routed or observed native window.
    budget_k: usize,
    /// A window routed deeper than [`FLOOR_SHALLOW_CAP`] was observed.
    deep_seen: bool,
    /// Priced shallow windows since the last deep window.
    shallow_since_deep: usize,
    shallow_state: ShallowState,
    shallow: ShallowEvidence,
    /// Consecutive windows a deeper candidate priced below the held
    /// shallow one.
    climb_streak: usize,
    /// Per-process id of this instance, the `"floor"` key of its call-log
    /// lines; kept across [`Self::rearm`].
    id: u64,
    /// The route's window-cost table, in interleaved-route emitted tokens
    /// (index = drafts of the window): the provisional depth's prior, never a
    /// measured price.
    window_cost: [f32; W],
    /// The exact call log, when the caller enabled it.
    call_log: Option<CallLog>,
}

impl<const W: usize, const E: usize> Floor<W, E> {
    /// A fresh floor that may retire after the probe budget's windows.
    pub fn new(window_cost: [f32; W]) -> Self {
        Self::with_retire_after(window_cost, FLOOR_PROBE_WINDOWS)
    }

    /// A fresh floor that may retire only after `retire_after` observed
    /// windows (at least the probe budget), decaying by
    /// [`DEFAULT_DECAY`].
    pub fn with_retire_after(window_cost: [f32; W], retire_after: usize) -> Self {
        Self {
            calibrated: false,
            ar_us: 0.0,
            probes: 0,
            observed: 0,
            retire_after: retire_after.max(FLOOR_PROBE_WINDOWS),
            retired: false,
            batched: [(0.0, 0.0); W],
            caps: [CapStats::default(); W],
            extra_batched: Vec::new(),
            extra_caps: Vec::new(),
            native_cap: W.saturating_sub(1),
            extended: false,
            fit: [0.0; 5],
            interleaved: (0.0, 0.0),
            external: (0.0, 0.0),
            external_depth: [(0.0, 0.0); E],
            external_blocked: false,
            combined: (0.0, 0.0),
            decay: DEFAULT_DECAY,
            slow: (0.0, 0.0),
            probe_tokens: 0,
            probe_left: 0,
            probe_samples: Vec::new(),
            probed_at: None,
            budget_k: 0,
            deep_seen: false,
            shallow_since_deep: 0,
            shallow_state: ShallowState::Idle,
            shallow: ShallowEvidence::default(),
            climb_streak: 0,
            id: next_floor_id(),
            window_cost,
            call_log: None,
        }
    }

    /// The same floor giving a would-be retirement `probe_tokens` fresh AR
    /// samples to re-price AR (0: none, and no realized-cost confirmation:
    /// the floor retires on its first verdict, the lanes' behavior).
    pub fn with_retire_evidence(self, probe_tokens: usize) -> Self {
        Self {
            probe_tokens,
            ..self
        }
    }

    /// The same floor decaying its measured costs by `decay`.
    pub fn with_decay(self, decay: f32) -> Self {
        Self { decay, ..self }
    }

    /// Whether the floor retired stickily to head-free AR.
    pub fn retired(&self) -> bool {
        self.retired
    }

    fn rearm_inner(&mut self) {
        let retire_after = self.retire_after;
        let decay = self.decay;
        let probe_tokens = self.probe_tokens;
        let log = self.call_log;
        self.call_log = None;
        self.reset_request(retire_after, decay, probe_tokens);
        self.call_log = log;
    }

    /// Admit a native depth capacity for this request: `0` disables every
    /// batched option (an explicit user `0` or `rows <= 1` executor width);
    /// otherwise capped at [`DERIVED_MAX_DRAFTS`]. Grow-only storage (no
    /// reallocation or observation loss when the value is unchanged);
    /// shrinkage only lowers the logical bound while keeping the rows for a
    /// later widen. Empty floors gain no rows until a first widening; legacy
    /// constructors behave as before until this is called.
    pub fn set_capacity(&mut self, max_drafts: usize) {
        let cap = max_drafts.min(DERIVED_MAX_DRAFTS);
        if cap == self.native_cap && self.extended {
            return;
        }
        if cap + 1 > self.caps.len() + self.extra_caps.len() {
            let need = cap + 1 - self.caps.len() - self.extra_caps.len();
            self.extra_caps.reserve(need);
            let extra = cap + 1 - self.batched.len() - self.extra_batched.len();
            self.extra_batched.reserve(extra);
            for _ in 0..extra {
                self.extra_batched.push((0.0, 0.0));
            }
            let span = self.batched.len() + self.extra_batched.len();
            for _ in 0..need {
                self.extra_caps.push(DynCapStats {
                    drafted: vec![0.0; span],
                    ..DynCapStats::default()
                });
            }
            for stats in &mut self.extra_caps {
                if stats.drafted.len() < span {
                    stats.drafted.resize(span, 0.0);
                }
            }
        }
        self.native_cap = cap;
        self.extended = true;
        if let Some(log) = self.call_log {
            log.emit(
                "set_capacity",
                self.id,
                format_args!(",\"capacity\":{cap}{}", self.after()),
            );
        }
    }

    /// Reset the observations and lifecycle for a new request while reusing
    /// the allocated capacity (`window_cost` included): the in-place
    /// `configure_request` path. Floor id and call-log setting survive.
    pub fn reset_request(&mut self, retire_after: usize, decay: f32, probe_tokens: usize) {
        self.calibrated = false;
        self.ar_us = 0.0;
        self.probes = 0;
        self.observed = 0;
        self.retire_after = retire_after.max(FLOOR_PROBE_WINDOWS);
        self.retired = false;
        self.batched = [(0.0, 0.0); W];
        self.caps = [CapStats::default(); W];
        for entry in &mut self.extra_batched {
            *entry = (0.0, 0.0);
        }
        for stats in &mut self.extra_caps {
            stats.us = 0.0;
            stats.emitted = 0.0;
            stats.windows = 0.0;
            for w in &mut stats.drafted {
                *w = 0.0;
            }
        }
        self.fit = [0.0; 5];
        self.interleaved = (0.0, 0.0);
        self.external = (0.0, 0.0);
        self.external_depth = [(0.0, 0.0); E];
        self.external_blocked = false;
        self.combined = (0.0, 0.0);
        self.decay = decay;
        self.slow = (0.0, 0.0);
        self.probe_tokens = probe_tokens;
        self.probe_left = 0;
        self.probe_samples.clear();
        self.probed_at = None;
        self.budget_k = 0;
        self.deep_seen = false;
        self.shallow_since_deep = 0;
        self.shallow_state = ShallowState::Idle;
        self.shallow = ShallowEvidence::default();
        self.climb_streak = 0;
        if let Some(log) = self.call_log {
            log.emit(
                "reset_request",
                self.id,
                format_args!(
                    ",\"retire_after\":{},\"decay\":{},\"probe_tokens\":{}{}",
                    self.retire_after,
                    FloorBits(self.decay),
                    self.probe_tokens,
                    self.after()
                ),
            );
        }
    }

    /// Native window cost at `depth` drafts in AR-step units, once probing
    /// finishes: the measured mean, else the request's cost fit.
    pub fn batched_cost(&self, depth: usize) -> Option<f32> {
        if self.probing() || !valid_us(self.ar_us) {
            return None;
        }
        if let Some((sum, windows)) = self.batched_entry(depth) {
            if windows > 0.0 {
                return valid_us(sum).then(|| sum / windows / self.ar_us);
            }
        }
        (self.wide_fit() && depth >= 1 && depth <= self.native_cap
            || depth >= 1 && depth < self.batched.len())
        .then(|| self.modeled_us(depth))
        .flatten()
        .map(|us| us / self.ar_us)
    }

    /// Window cost of an n-gram takeover of `d` candidates in AR-step units,
    /// once probing finishes with a valid AR price: the measured mean of this
    /// request's takeover windows of `d` candidates; else the nearest
    /// measured takeover depth `d'` (ties: more windows) moved by the fit's
    /// per-row slope (`ROW_COST_PRIOR` AR steps without a fit); else the
    /// native window cost at `d` drafts ([`Self::batched_cost`]'s measurement
    /// or fit, the fit also past the native table).
    pub fn takeover_cost(&self, d: usize) -> Option<f32> {
        if self.probing() || !valid_us(self.ar_us) {
            return None;
        }
        let measured = |depth: usize| {
            self.external_depth
                .get(depth)
                .filter(|&&(us, windows)| windows > 0.0 && valid_us(us))
                .map(|&(us, windows)| (us / windows, windows))
        };
        if let Some((mean, _)) = measured(d) {
            return Some(mean / self.ar_us);
        }
        let nearest = (1..self.external_depth.len())
            .filter_map(|depth| measured(depth).map(|(mean, windows)| (depth, mean, windows)))
            .min_by(|a, b| {
                a.0.abs_diff(d)
                    .cmp(&b.0.abs_diff(d))
                    .then(b.2.total_cmp(&a.2))
            });
        if let Some((near, mean, _)) = nearest {
            let row_us = self
                .fit_line()
                .map_or(ROW_COST_PRIOR * f64::from(self.ar_us), |(_, _, slope)| {
                    slope
                }) as f32;
            let us = mean + row_us * (d as f32 - near as f32);
            if valid_us(us) {
                return Some(us / self.ar_us);
            }
        }
        let native = match self.batched_entry(d) {
            Some((sum, windows)) if windows > 0.0 => valid_us(sum).then(|| sum / windows),
            _ => self.modeled_us(d),
        };
        native.map(|us| us / self.ar_us)
    }

    /// This request's window-cost line `(mean depth, mean us, us per
    /// drafted row)`: the decayed least-squares line of every priced batched
    /// window's wall time on its drafted depth, its slope (one more verify
    /// row and draft step on this GPU) ridge-pulled toward
    /// [`ROW_COST_PRIOR`] AR steps with weight
    /// [`ROW_COST_PRIOR_WEIGHT`] and never negative.  With windows at one
    /// depth only the slope is the prior; it becomes the request's own as
    /// soon as two depths are priced.  `None` before any batched window was
    /// priced or without a valid AR price.
    pub fn fit_line(&self) -> Option<(f64, f64, f64)> {
        let [w, sd, sus, sdd, sdus] = self.fit;
        if w <= 0.0 || !valid_us(self.ar_us) {
            return None;
        }
        let (mean_d, mean_us) = (sd / w, sus / w);
        let spread = (sdd - w * mean_d * mean_d).max(0.0);
        let covariance = sdus - w * mean_d * mean_us;
        let prior = ROW_COST_PRIOR * f64::from(self.ar_us);
        let slope = ((covariance + ROW_COST_PRIOR_WEIGHT * prior)
            / (spread + ROW_COST_PRIOR_WEIGHT))
            .max(0.0);
        Some((mean_d, mean_us, slope))
    }

    /// [`Self::fit_line`]'s wall time (us) of a batched window drafting
    /// `depth` tokens, when positive.
    fn modeled_us(&self, depth: usize) -> Option<f32> {
        let (mean_d, mean_us, slope) = self.fit_line()?;
        let us = (mean_us + slope * (depth as f64 - mean_d)) as f32;
        valid_us(us).then_some(us)
    }

    /// Admitted native depth bound (`0` disables every batched option).
    fn depth_cap(&self) -> usize {
        self.native_cap
    }

    /// Whether the request admitted a genuinely wide domain past the static
    /// prefix: only then may the unmeasured global-fit candidate domain
    /// apply. A `set_capacity` at or below the historical table still
    /// enforces the actual bound, storage and logging, but keeps the legacy
    /// measured/neighbour pricing bit-arithmetic.
    fn wide_fit(&self) -> bool {
        self.extended && self.native_cap >= self.batched.len()
    }

    /// Mutable decayed (wall us, windows) of exactly `depth` drafts, across
    /// the static prefix and the [`Self::set_capacity`] overflow. Depths past
    /// the admitted capacity read `None` and are not recorded.
    fn batched_entry_mut(&mut self, depth: usize) -> Option<&mut (f32, f32)> {
        if depth >= 1 && depth <= self.native_cap {
            if depth < self.batched.len() {
                return Some(&mut self.batched[depth]);
            }
            let at = depth.saturating_sub(self.batched.len());
            return self.extra_batched.get_mut(at);
        }
        None
    }

    /// Decayed (wall us, windows) of exactly `depth` drafts, across the
    /// static prefix and the overflow. Used by the price queries (which are
    /// fit-gated, not capacity-gated).
    fn batched_entry(&self, depth: usize) -> Option<(f32, f32)> {
        if depth < self.batched.len() {
            return Some(self.batched[depth]);
        }
        self.extra_batched
            .get(depth.saturating_sub(self.batched.len()))
            .copied()
    }

    /// One decay step of every cap's evidence (static prefix plus overflow),
    /// recording the hit at routed `cap` with drafted mix `depth`.
    /// No allocation: overflow histograms were sized at
    /// [`Self::set_capacity`].
    fn age_caps(&mut self, decay: f32, us: f32, emitted: f32, cap: usize, depth: usize) {
        for c in 0..self.caps.len() + self.extra_caps.len() {
            let hit = c == cap && c >= 1 && c <= self.native_cap;
            let add = f32::from(u8::from(hit));
            if c < self.caps.len() {
                let stats = &mut self.caps[c];
                stats.us = stats.us * decay + if hit { us } else { 0.0 };
                stats.emitted = stats.emitted * decay + if hit { emitted } else { 0.0 };
                stats.windows = stats.windows * decay + add;
                for (d, windows) in stats.drafted.iter_mut().enumerate() {
                    *windows = *windows * decay + f32::from(u8::from(hit && d == depth));
                }
            } else if let Some(stats) = self.extra_caps.get_mut(c - self.caps.len()) {
                stats.us = stats.us * decay + if hit { us } else { 0.0 };
                stats.emitted = stats.emitted * decay + if hit { emitted } else { 0.0 };
                stats.windows = stats.windows * decay + add;
                for (d, windows) in stats.drafted.iter_mut().enumerate() {
                    *windows = *windows * decay + f32::from(u8::from(hit && d == depth));
                }
            }
        }
    }

    /// Decayed (wall us, windows) of cap `cap` across the static prefix and
    /// the [`Self::set_capacity`] overflow.
    fn cap_windows(&self, cap: usize) -> (f32, f32) {
        if cap < self.caps.len() {
            (self.caps[cap].us, self.caps[cap].windows)
        } else {
            self.extra_caps
                .get(cap.saturating_sub(self.caps.len()))
                .map_or((0.0, 0.0), |stats| (stats.us, stats.windows))
        }
    }

    /// Expected emitted tokens of a window drafting exactly `depth` tokens:
    /// the bonus plus the accepted-prefix probabilities of the per-depth
    /// agreement. Missing agreement depths read as unobserved (the prior),
    /// never as zero.
    fn expected_emitted(depth: usize, agreement: &[(f32, f32)]) -> f32 {
        let mut prefix = 1.0f32;
        let mut expected = 1.0f32;
        for d in 1..=depth {
            let (accepted, total) = agreement.get(d - 1).copied().unwrap_or(AGREEMENT_PRIOR);
            prefix *= accepted / total.max(f32::MIN_POSITIVE);
            expected += prefix;
        }
        expected
    }

    /// Wall us per expected emitted token of a measured cap: the decayed
    /// mean window wall time at that cap over the expected tokens of its
    /// windows' drafted-depth mix, without heap work (the mix dot-products
    /// the static prefix and the overflow histogram in place).
    fn measured_cap_cost(
        &self,
        agreement: &[(f32, f32)],
        cap: usize,
        us: f32,
        windows: f32,
    ) -> Option<f32> {
        if windows <= 0.0 || !valid_us(us) {
            return None;
        }
        let mut emitted = 0.0f32;
        let mut mix_windows = 0.0f32;
        if cap < self.caps.len() {
            for (d, &w) in self.caps[cap].drafted.iter().enumerate() {
                emitted += w * Self::expected_emitted(d, agreement);
                mix_windows += w;
            }
        } else if let Some(stats) = self.extra_caps.get(cap.saturating_sub(self.caps.len())) {
            for (d, &w) in stats.drafted.iter().enumerate() {
                emitted += w * Self::expected_emitted(d, agreement);
                mix_windows += w;
            }
        }
        if mix_windows <= 0.0 {
            return None;
        }
        let window_us = us / windows;
        let per_token = window_us / (emitted / mix_windows);
        (valid_us(window_us) && per_token.is_finite() && per_token > 0.0).then_some(per_token)
    }

    /// Wall us per expected emitted token of an unmeasured cap in extended
    /// mode: the request's own fit priced at exactly `cap` drafts over the
    /// agreement's expected yield there. `None` without a fit: an unpriced
    /// width is never invented from the static table.
    fn unobserved_fit_cost(&self, agreement: &[(f32, f32)], cap: usize) -> Option<f32> {
        let us = self.modeled_us(cap)?;
        let emitted = Self::expected_emitted(cap, agreement);
        (emitted > 0.0).then(|| us / emitted)
    }

    /// Wall us per expected emitted token of routing `cap`: the decayed
    /// mean window wall time at that cap over the expected tokens of its
    /// windows' drafted-depth mix (each drafted depth `d` at the current
    /// agreement's `1 + Σ prefix`), so the windows the margin prune stops
    /// short count at their real, lower, cost and yield.  The neighbouring
    /// cap with more windows also counts, for the evidence it holds beyond
    /// this cap's own, shifted by one row: one cap down, its windows that
    /// drafted its whole cap draft one row (`row` us, the fit's slope) less;
    /// one cap up, one more (a window the prune stopped short is the same at
    /// either cap).  Every cap's evidence decays with every batched window
    /// ([`Self::observe_window`]), so an unmeasured cap next to the one a
    /// request runs is priced from it, and a cap measured once (a probe
    /// window during a response's poor start) is re-priced from its running
    /// neighbour instead of keeping that stale price for good.  In extended
    /// ([`Self::set_capacity`]) mode a measured cap prices from its own mix
    /// alone and an unmeasured one from the request's own fit, so feasible
    /// widths stay visible past the static prefix; legacy floors keep the
    /// neighbour-only pricing above.
    fn cap_cost(&self, agreement: &[(f32, f32)], cap: usize) -> Option<f32> {
        if cap == 0 || cap > self.depth_cap() {
            return None;
        }
        let (own_us, own_windows) = self.cap_windows(cap);
        if self.wide_fit() {
            if own_windows > 0.0 {
                return self.measured_cap_cost(agreement, cap, own_us, own_windows);
            }
            return self.unobserved_fit_cost(agreement, cap);
        }
        if cap >= self.caps.len() {
            return None;
        }
        let own = self.caps[cap];
        let (mut us, mut windows, mut mix) = (own.us, own.windows, own.drafted);
        let neighbour = [cap - 1, cap + 1]
            .into_iter()
            .filter(|&c| c >= 1 && c < self.caps.len() && self.caps[c].windows > own.windows)
            .max_by(|&a, &b| self.caps[a].windows.total_cmp(&self.caps[b].windows));
        if let (Some(c), Some((_, _, row))) = (neighbour, self.fit_line()) {
            let from = self.caps[c];
            let weight = from.windows - own.windows;
            let share = weight / from.windows;
            let mut shifted = from.drafted.map(|windows| windows * share);
            let moved = std::mem::take(&mut shifted[c]);
            shifted[cap] += moved;
            us += from.us * share + row as f32 * moved * (cap as f32 - c as f32);
            windows += weight;
            for (total, add) in mix.iter_mut().zip(shifted) {
                *total += add;
            }
        }
        if windows <= 0.0 {
            return None;
        }
        let expected = expected_emitted_by_depth::<W>(agreement);
        let emitted: f32 = mix.iter().zip(expected).map(|(w, e)| w * e).sum::<f32>() / windows;
        let window_us = us / windows;
        (valid_us(window_us) && emitted > 0.0).then(|| window_us / emitted)
    }

    /// Cheapest candidate within `max_depth` drafts, `(route, us per
    /// emitted token)`: the measured options of [`Self::best_native`] plus
    /// the unmeasured neighbours of [`Self::cap_cost`].  Ties keep the
    /// earlier (shallower) option.
    fn best_candidate(&self, agreement: &[(f32, f32)], max_depth: usize) -> Option<(Action, f32)> {
        let mut best = (self.interleaved.1 > 0.0)
            .then(|| (Action::Interleaved, self.interleaved.0 / self.interleaved.1));
        for cap in 1..self.depth_cap().min(max_depth).saturating_add(1) {
            let Some(cost) = self.cap_cost(agreement, cap) else {
                continue;
            };
            if best.is_none_or(|(_, cheapest)| cost < cheapest) {
                best = Some((Action::Native(cap), cost));
            }
        }
        best
    }

    /// Expected emitted tokens per AR step of window cost of the cheapest
    /// candidate within `max_depth` (at least 1, the interleaved route's
    /// rate), once probing finishes with a valid AR price: the native rate
    /// an n-gram takeover must match.
    pub fn candidate_rate(&self, agreement: &[(f32, f32)], max_depth: usize) -> Option<f32> {
        if self.probing() || !valid_us(self.ar_us) {
            return None;
        }
        let (_, cost) = self.best_candidate(agreement, max_depth)?;
        Some((self.ar_us / cost).max(1.0))
    }

    fn probing(&self) -> bool {
        self.probes < FLOOR_PROBE_WINDOWS
    }

    fn count_probe(&mut self) {
        self.probes = (self.probes + 1).min(FLOOR_PROBE_WINDOWS);
        self.observed = self.observed.saturating_add(1);
    }

    /// Still inside the window count a losing verdict must wait for.
    fn in_grace(&self) -> bool {
        self.observed < self.retire_after
    }

    fn observe_calibration_inner(&mut self, us: f32, agreement: &[(f32, f32)]) {
        if !self.calibrated {
            self.calibrated = true;
            self.ar_us = if valid_us(us) { us } else { 0.0 };
            self.decide(agreement);
            return;
        }
        if self.probe_left == 0 || self.retired {
            return;
        }
        self.probe_left -= 1;
        if valid_us(us) {
            self.probe_samples.push(us);
        }
        if self.probe_left == 0 {
            if !self.probe_samples.is_empty() {
                self.ar_us =
                    self.probe_samples.iter().sum::<f32>() / self.probe_samples.len() as f32;
            }
            self.decide(agreement);
        }
    }

    /// Whether the next window must be an ordinary AR token timed for the
    /// floor: the request's first, or one of the probes of a would-be
    /// retirement.
    pub fn needs_ar_sample(&self) -> bool {
        !self.calibrated || self.probe_left > 0
    }

    fn observe_native_inner(
        &mut self,
        route: NativeWindow,
        window: &MtpWindow,
        us: f32,
        k: usize,
        agreement: &[(f32, f32)],
    ) {
        let stats = WindowStats::of(window, window.drafts_generated);
        self.observe_window_inner(route, stats, us, k, agreement);
    }

    fn observe_window_inner(
        &mut self,
        route: NativeWindow,
        window: WindowStats,
        us: f32,
        k: usize,
        agreement: &[(f32, f32)],
    ) {
        if k == 0 || self.retired || !self.calibrated {
            return;
        }
        self.budget_k = k;
        // An unusable timing records no cost but still spends probe budget,
        // so the number of unmeasured windows stays bounded.
        if valid_us(us) {
            let decay = self.decay;
            let emitted = window.emitted as f32;
            decay_add(&mut self.combined, us, emitted, decay);
            decay_add(&mut self.slow, us, emitted, decay.max(FLOOR_RETIRE_DECAY));
            match route {
                NativeWindow::Interleaved => decay_add(&mut self.interleaved, us, emitted, decay),
                NativeWindow::Batched => {
                    let depth = window.drafts;
                    if depth >= 1 {
                        if let Some(entry) = self.batched_entry_mut(depth) {
                            decay_add(entry, us, 1.0, decay);
                            let (d, wus, wdecay) = (depth as f64, f64::from(us), f64::from(decay));
                            for (sum, x) in self.fit.iter_mut().zip([1.0, d, wus, d * d, d * wus]) {
                                *sum = *sum * wdecay + x;
                            }
                        }
                    }
                    // Every cap's evidence ages with every batched window, so
                    // a cap not run lately counts for less than the one that
                    // runs (see `cap_cost`).
                    let cap = window.cap.min(self.depth_cap());
                    self.age_caps(decay, us, emitted, cap, depth);
                }
            }
        }
        self.observe_shallow_epoch(route, window, us, agreement);
        self.count_probe();
        self.decide(agreement);
    }

    fn observe_external_inner(
        &mut self,
        emitted: usize,
        wall_us: f32,
        drafts: usize,
        agreement: &[(f32, f32)],
    ) -> bool {
        if !self.retired && drafts > 0 {
            if valid_us(wall_us) {
                let decay = self.decay;
                if self.calibrated {
                    decay_add(&mut self.combined, wall_us, emitted as f32, decay);
                    decay_add(
                        &mut self.slow,
                        wall_us,
                        emitted as f32,
                        decay.max(FLOOR_RETIRE_DECAY),
                    );
                }
                if emitted > 0 {
                    decay_add(&mut self.external, wall_us, emitted as f32, decay);
                }
            }
            self.spend_shallow_window(None);
            self.count_probe();
            self.decide(agreement);
        }
        self.retired || self.external_blocked
    }

    fn observe_external_copy_inner(
        &mut self,
        emitted: usize,
        wall_us: f32,
        drafts: usize,
        agreement: &[(f32, f32)],
    ) -> bool {
        let was_retired = self.retired;
        if drafts > 0 {
            if valid_us(wall_us) {
                let decay = self.decay;
                if self.calibrated && !was_retired {
                    decay_add(&mut self.combined, wall_us, emitted as f32, decay);
                    decay_add(
                        &mut self.slow,
                        wall_us,
                        emitted as f32,
                        decay.max(FLOOR_RETIRE_DECAY),
                    );
                }
                if emitted > 0 {
                    decay_add(&mut self.external, wall_us, emitted as f32, decay);
                }
                if let Some(entry) = self.external_depth.get_mut(drafts) {
                    decay_add(entry, wall_us, 1.0, decay);
                }
            }
            if was_retired {
                let external = (self.external.1 > 0.0).then(|| self.external.0 / self.external.1);
                self.external_blocked =
                    valid_us(self.ar_us) && external.is_some_and(|cost| cost >= self.ar_us);
            } else {
                self.spend_shallow_window(None);
                self.count_probe();
                self.decide(agreement);
            }
        }
        if was_retired {
            self.external_blocked
        } else {
            self.retired || self.external_blocked
        }
    }

    /// Cheapest finite native option with at most `max_depth` drafts
    /// (interleaved is always eligible): `(route, us per emitted token)`, a
    /// cap priced by [`Self::cap_cost`]. In extended ([`Self::set_capacity`])
    /// mode the finite fit prices unmeasured widths, so the floor can climb
    /// past the table end instead of retiring on a losing shallow strip;
    /// legacy floors only consult measured caps. Ties keep the earlier
    /// (shallower) option.
    fn best_native(&self, agreement: &[(f32, f32)], max_depth: usize) -> Option<(Action, f32)> {
        let mut best = (self.interleaved.1 > 0.0)
            .then(|| (Action::Interleaved, self.interleaved.0 / self.interleaved.1));
        for cap in 1..self.depth_cap().min(max_depth).saturating_add(1) {
            let (_, own_windows) = self.cap_windows(cap);
            if !self.wide_fit() && own_windows <= 0.0 {
                continue;
            }
            let Some(cost) = self.cap_cost(agreement, cap) else {
                continue;
            };
            if best.is_none_or(|(_, cheapest)| cost < cheapest) {
                best = Some((Action::Native(cap), cost));
            }
        }
        best
    }

    /// External windows are measured and their price is below AR.
    fn external_alive(&self) -> bool {
        self.external.1 > 0.0 && !self.external_blocked
    }

    /// The decayed realized per-token cost of every window (native and
    /// external) is strictly below AR.
    fn combined_wins(&self) -> bool {
        self.combined.1 > 0.0 && self.combined.0 / self.combined.1 < self.ar_us
    }

    /// The fresh AR price this verdict would retire on is not yet measured:
    /// the floor was given AR probes
    /// ([`Self::with_retire_evidence`]), has a valid price to replace, and has
    /// not taken a probe set within the last `retire_after` windows.
    fn should_probe(&self) -> bool {
        self.probe_tokens > 0
            && self.probe_left == 0
            && valid_us(self.ar_us)
            && self.probed_at.map_or(true, |at| {
                self.observed >= at.saturating_add(self.retire_after)
            })
    }

    /// A losing verdict past the grace: sticky retirement, unless AR has not
    /// been re-priced where the request is now (the next windows are AR
    /// probe tokens, re-evaluated on their mean by `observe_calibration`),
    /// or a deeper budget has no fresh shallow evidence yet (the next
    /// windows are the bounded shallow re-probe,
    /// [`Self::needs_shallow_evidence`]).  Returns the route for the next
    /// window.
    fn retire_or_probe(&mut self, agreement: &[(f32, f32)]) -> Action {
        if self.should_probe() {
            self.probe_left = self.probe_tokens;
            self.probe_samples.clear();
            self.probed_at = Some(self.observed);
            Action::ArSynced
        } else if self.needs_shallow_evidence() {
            self.shallow_state = ShallowState::Probing;
            self.shallow = ShallowEvidence::default();
            self.climb_streak = 0;
            self.shallow_route(agreement, self.budget_k)
        } else {
            self.retired = true;
            Action::ArHeadFree
        }
    }

    /// The verdict would retire a request that explored past
    /// [`FLOOR_SHALLOW_CAP`] on a budget above it, with no priced shallow
    /// window since its last deep one and no shallow verdict already taken.
    fn needs_shallow_evidence(&self) -> bool {
        self.shallow_state == ShallowState::Idle
            && self.budget_k > FLOOR_SHALLOW_CAP
            && self.deep_seen
            && self.shallow_since_deep < FLOOR_PROBE_WINDOWS
            && valid_us(self.ar_us)
    }

    /// Route of a shallow window within `k`: the cheapest measured shallow
    /// option, else the deepest shallow cap.
    fn shallow_route(&self, agreement: &[(f32, f32)], k: usize) -> Action {
        let limit = k.min(FLOOR_SHALLOW_CAP);
        match self.best_native(agreement, limit) {
            Some((route, _)) => route,
            None if limit == 0 => Action::Interleaved,
            None => Action::Native(limit),
        }
    }

    /// Fresh shallow realized wall us per emitted token.
    fn shallow_cost(&self) -> Option<f32> {
        let (us, emitted) = self.shallow.cost;
        (emitted > 0.0 && valid_us(us)).then(|| us / emitted)
    }

    /// The held shallow evidence still prices below AR.
    fn shallow_wins(&self) -> bool {
        self.shallow_state == ShallowState::Holding
            && self.shallow_cost().is_some_and(|cost| cost < self.ar_us)
    }

    /// Draft budget the route may use: a held shallow request stays at or
    /// below [`FLOOR_SHALLOW_CAP`] until a deeper candidate has the
    /// positive evidence of [`Self::update_climb_evidence`].
    fn hold_limit(&self, k: usize) -> usize {
        if self.shallow_state == ShallowState::Holding && self.climb_streak < FLOOR_PROBE_WINDOWS {
            k.min(FLOOR_SHALLOW_CAP)
        } else {
            k
        }
    }

    /// Count one finished window of a held shallow request toward a climb:
    /// the cheapest candidate within the budget is deeper than the shallow
    /// cap and prices strictly below both the cheapest shallow candidate and
    /// AR.
    fn update_climb_evidence(&mut self, agreement: &[(f32, f32)]) {
        let k = self.budget_k;
        let deeper = k > FLOOR_SHALLOW_CAP
            && match (
                self.best_candidate(agreement, FLOOR_SHALLOW_CAP),
                self.best_candidate(agreement, k),
            ) {
                (Some((_, shallow)), Some((Action::Native(cap), cost))) => {
                    cap > FLOOR_SHALLOW_CAP && cost < shallow && cost < self.ar_us
                }
                _ => false,
            };
        self.climb_streak = if deeper {
            self.climb_streak.saturating_add(1)
        } else {
            0
        };
    }

    /// Track the shallow epoch with one finished native window: a deep
    /// window zeroes the shallow count and ends a held or decided shallow
    /// verdict; a shallow one counts when priced and feeds the re-probe or
    /// the held evidence.
    fn observe_shallow_epoch(
        &mut self,
        route: NativeWindow,
        window: WindowStats,
        us: f32,
        agreement: &[(f32, f32)],
    ) {
        if route == NativeWindow::Batched && window.cap > FLOOR_SHALLOW_CAP {
            self.deep_seen = true;
            self.shallow_since_deep = 0;
            self.climb_streak = 0;
            match self.shallow_state {
                ShallowState::Probing => self.spend_shallow_window(None),
                ShallowState::Idle => {}
                _ => self.shallow_state = ShallowState::Idle,
            }
            return;
        }
        let priced = valid_us(us);
        if priced {
            self.shallow_since_deep = self.shallow_since_deep.saturating_add(1);
        }
        let emitted = window.emitted as f32;
        match self.shallow_state {
            ShallowState::Probing => self.spend_shallow_window(priced.then_some((us, emitted))),
            ShallowState::Holding => {
                if priced {
                    let decay = self.decay.max(FLOOR_RETIRE_DECAY);
                    decay_add(&mut self.shallow.cost, us, emitted, decay);
                }
                self.update_climb_evidence(agreement);
            }
            _ => {}
        }
    }

    /// Spend one window of the shallow re-probe (`evidence`: its wall us and
    /// emitted tokens when priced), and take the verdict at
    /// [`FLOOR_PROBE_WINDOWS`] priced windows or the window bound.
    fn spend_shallow_window(&mut self, evidence: Option<(f32, f32)>) {
        if self.shallow_state != ShallowState::Probing {
            return;
        }
        self.shallow.windows += 1;
        if let Some((us, emitted)) = evidence {
            self.shallow.priced += 1;
            self.shallow.cost.0 += us;
            self.shallow.cost.1 += emitted;
        }
        if self.shallow.priced < FLOOR_PROBE_WINDOWS
            && self.shallow.windows < FLOOR_SHALLOW_MAX_WINDOWS
        {
            return;
        }
        self.shallow_state = if self.shallow.priced == 0 {
            ShallowState::Unpriced
        } else if self.shallow_cost().is_some_and(|cost| cost < self.ar_us) {
            self.deep_seen = false;
            self.shallow_since_deep = self.shallow.priced;
            self.climb_streak = 0;
            ShallowState::Holding
        } else {
            ShallowState::Lost
        };
    }

    /// The slow-memory realized per-token cost of every window is strictly
    /// below AR.  Only floors given retire evidence consult it.
    fn slow_wins(&self) -> bool {
        self.probe_tokens > 0 && self.slow.1 > 0.0 && self.slow.0 / self.slow.1 < self.ar_us
    }

    /// Retire when, after the probe budget and at least `retire_after`
    /// observed windows, nothing measured beats AR, external windows are
    /// not alive on a mixture that beats AR, and (with retire evidence) the
    /// slow-memory realized cost does not beat AR either; first re-pricing
    /// AR with fresh probes ([`Self::retire_or_probe`]).
    fn decide(&mut self, agreement: &[(f32, f32)]) {
        if self.retired
            || !self.calibrated
            || self.probing()
            || self.probe_left > 0
            || self.shallow_state == ShallowState::Probing
        {
            return;
        }
        if self.ar_us <= 0.0 {
            self.retired = true;
            return;
        }
        let external = (self.external.1 > 0.0).then(|| self.external.0 / self.external.1);
        self.external_blocked = external.is_some_and(|cost| cost >= self.ar_us);
        let native_wins = self
            .best_native(agreement, usize::MAX)
            .is_some_and(|(_, cost)| cost < self.ar_us);
        if !native_wins
            && !(self.external_alive() && self.combined_wins())
            && !self.slow_wins()
            && !self.shallow_wins()
            && !self.in_grace()
        {
            self.retire_or_probe(agreement);
        }
    }

    fn route_inner(&mut self, agreement: &[(f32, f32)], k: usize, provisional: usize) -> Action {
        self.budget_k = k;
        if self.retired {
            return Action::ArHeadFree;
        }
        if self.probe_left > 0 {
            return Action::ArSynced;
        }
        if self.probing() || !self.calibrated {
            return match provisional {
                0 => Action::Interleaved,
                depth => Action::Native(depth),
            };
        }
        if self.shallow_state == ShallowState::Probing {
            return self.shallow_route(agreement, k);
        }
        let limit = self.hold_limit(k);
        let best = self.best_native(agreement, limit);
        if let Some((route, cost)) = best {
            if cost < self.ar_us {
                // The candidates include every measured option, so the
                // cheapest one costs no more than `cost`.
                return self
                    .best_candidate(agreement, limit)
                    .map_or(route, |(candidate, _)| candidate);
            }
        }
        if (self.external_alive() && self.combined_wins())
            || self.slow_wins()
            || self.shallow_wins()
        {
            // Losing native route kept alive for the external windows, the
            // realized mixture or the held shallow evidence: the cheapest
            // measured option within the limit, else interleaved.
            return best.map_or(Action::Interleaved, |(route, _)| route);
        }
        if self.in_grace() && valid_us(self.ar_us) {
            return best.map_or_else(
                || match provisional {
                    0 => Action::Interleaved,
                    depth => Action::Native(depth),
                },
                |(route, _)| route,
            );
        }
        self.retire_or_probe(agreement)
    }

    /// One `{prefix}` trace JSON line (`QWEN4_FLOOR_TRACE`, `QWEN35_FLOOR_TRACE`): the
    /// floor's state after `event`, with every candidate option in
    /// microseconds per emitted token next to `ar_us` (`capN`: routed at
    /// depth N; `modeled`: priced from a measured neighbour, `win_us` /
    /// `emit` per window, `full` / `all` the shares of windows that drafted
    /// the whole cap / also accepted it all).
    pub fn trace_line(
        &self,
        prefix: &str,
        agreement: &[(f32, f32)],
        event: &str,
        us: f32,
        emitted: usize,
        drafts: usize,
    ) -> String {
        let mut options = String::new();
        if self.interleaved.1 > 0.0 {
            options.push_str(&format!(
                "{{\"route\":\"interleaved\",\"us_per_tok\":{:.1}}}",
                self.interleaved.0 / self.interleaved.1
            ));
        }
        for cap in 1..self.depth_cap().saturating_add(1) {
            let Some(cost) = self.cap_cost(agreement, cap) else {
                continue;
            };
            if !options.is_empty() {
                options.push(',');
            }
            let (win_us, emit, drafted, modeled) = if cap < self.caps.len() {
                let stats = self.caps[cap];
                let per = |x: f32| {
                    if stats.windows > 0.0 {
                        x / stats.windows
                    } else {
                        0.0
                    }
                };
                let mean_drafted: f32 = stats
                    .drafted
                    .iter()
                    .enumerate()
                    .map(|(d, &w)| d as f32 * w)
                    .sum();
                (
                    per(stats.us),
                    per(stats.emitted),
                    per(mean_drafted),
                    stats.windows <= 0.0,
                )
            } else if let Some(stats) = self.extra_caps.get(cap.saturating_sub(self.caps.len())) {
                let per = |x: f32| {
                    if stats.windows > 0.0 {
                        x / stats.windows
                    } else {
                        0.0
                    }
                };
                let mean_drafted: f32 = stats
                    .drafted
                    .iter()
                    .enumerate()
                    .map(|(d, &w)| d as f32 * w)
                    .sum();
                (
                    per(stats.us),
                    per(stats.emitted),
                    per(mean_drafted),
                    stats.windows <= 0.0,
                )
            } else {
                continue;
            };
            options.push_str(&format!(
                "{{\"route\":\"cap{cap}\",\"modeled\":{modeled},\"win_us\":{win_us:.1},\"emit\":{emit:.3},\"drafted\":{drafted:.2},\"us_per_tok\":{cost:.1}}}",
            ));
        }
        let row_us = self.fit_line().map_or(0.0, |(_, _, slope)| slope);
        let combined = if self.combined.1 > 0.0 {
            self.combined.0 / self.combined.1
        } else {
            0.0
        };
        let slow = if self.slow.1 > 0.0 {
            self.slow.0 / self.slow.1
        } else {
            0.0
        };
        let shallow = self.shallow_cost().unwrap_or(0.0);
        let shallow_state = self.shallow_state.name();
        format!(
            "{prefix} {{\"event\":\"{event}\",\"us\":{us:.1},\"emitted\":{emitted},\"drafts\":{drafts},\"ar_us\":{:.1},\"observed\":{},\"probes\":{},\"retire_after\":{},\"retired\":{},\"probe_left\":{},\"combined_us_per_tok\":{combined:.1},\"slow_us_per_tok\":{slow:.1},\"row_us\":{row_us:.1},\"budget_k\":{},\"deep_seen\":{},\"shallow_state\":\"{shallow_state}\",\"shallow_windows\":{},\"shallow_priced\":{},\"shallow_budget\":{},\"shallow_since_deep\":{},\"shallow_us_per_tok\":{shallow:.1},\"climb_streak\":{},\"options\":[{options}]}}",
            self.ar_us,
            self.observed,
            self.probes,
            self.retire_after,
            self.retired,
            self.probe_left,
            self.budget_k,
            self.deep_seen,
            self.shallow.windows,
            self.shallow.priced,
            FLOOR_SHALLOW_MAX_WINDOWS,
            self.shallow_since_deep,
            self.climb_streak
        )
    }

    /// Enables the exact call log under `prefix` (`QWEN4_FLOOR_CALL`,
    /// `QWEN35_FLOOR_CALL`) on stderr, and writes its `install` line now:
    /// how this floor was constructed (`retire_after` as resolved, `decay`
    /// as bits, `probe_tokens`) and what built it (`owner`).  The caller
    /// decides from its own environment flag; the module never reads one.
    pub fn with_call_log(self, prefix: &'static str, owner: &'static str) -> Self {
        self.with_call_log_enabled(prefix, owner, true)
    }

    /// [`Self::with_call_log`] recording whether the floor is `enabled` for
    /// its request (an opted-out request still installs a floor it never
    /// calls).
    pub fn with_call_log_enabled(self, prefix: &'static str, owner: &str, enabled: bool) -> Self {
        self.with_call_log_to(prefix, owner, enabled, stderr_sink)
    }

    /// [`Self::with_call_log_enabled`] writing every line to `sink` instead
    /// of stderr (a test captures them to replay).
    pub fn with_call_log_to(
        self,
        prefix: &'static str,
        owner: &str,
        enabled: bool,
        sink: fn(&str),
    ) -> Self {
        let log = CallLog { prefix, sink };
        log.emit(
            "install",
            self.id,
            format_args!(
                ",\"owner\":\"{owner}\",\"enabled\":{enabled},\"retire_after\":{},\"decay\":{},\"probe_tokens\":{},\"capacity\":{},\"extended\":{}",
                self.retire_after,
                FloorBits(self.decay),
                self.probe_tokens,
                self.native_cap,
                self.extended
            ),
        );
        Self {
            call_log: Some(log),
            ..self
        }
    }

    /// Reuse path of [`Self::with_call_log_to`]: install `prefix`/`owner` on
    /// this floor without moving it, emitting the same `install` line (with
    /// the live capacity). The arch `configure_request` calls this after
    /// [`Self::clear_call_log`] + [`Self::reset_request`] instead of
    /// reconstructing the floor.
    pub fn set_call_log_enabled(&mut self, prefix: &'static str, owner: &str, enabled: bool) {
        let log = CallLog {
            prefix,
            sink: stderr_sink,
        };
        log.emit(
            "install",
            self.id,
            format_args!(
                ",\"owner\":\"{owner}\",\"enabled\":{enabled},\"retire_after\":{},\"decay\":{},\"probe_tokens\":{},\"capacity\":{},\"extended\":{}",
                self.retire_after,
                FloorBits(self.decay),
                self.probe_tokens,
                self.native_cap,
                self.extended
            ),
        );
        self.call_log = Some(log);
    }

    /// Drop the call log without a line (the trace-disabled requests of
    /// [`Self::reset_request`]).
    pub fn clear_call_log(&mut self) {
        self.call_log = None;
    }

    /// Per-process id of this instance, the `"floor"` key of its call-log
    /// lines.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Whether the floor has priced AR (its first ordinary token).
    pub fn calibrated(&self) -> bool {
        self.calibrated
    }

    /// Microseconds of one ordinary AR token as priced; 0 when the
    /// measurement was invalid or not yet taken.
    pub fn ar_us(&self) -> f32 {
        self.ar_us
    }

    /// External windows cost no less per token than AR: the caller must not
    /// take over again this request.
    pub fn external_blocked(&self) -> bool {
        self.external_blocked
    }

    /// The window-cost table's depth for the next window ([`batched_depth`]
    /// on the table this floor was built with): the provisional depth the
    /// route uses while the probe budget lasts.
    pub fn provisional_depth(&self, agreement: &[(f32, f32)], k: usize) -> usize {
        batched_depth(agreement, k, &self.window_cost)
    }

    /// A copy of the measured state (diagnostics and tests).
    pub fn diagnostics(&self) -> FloorDiagnostics<W, E> {
        FloorDiagnostics {
            calibrated: self.calibrated,
            ar_us: self.ar_us,
            probes: self.probes,
            observed: self.observed,
            retire_after: self.retire_after,
            retired: self.retired,
            probe_left: self.probe_left,
            probe_tokens: self.probe_tokens,
            decay: self.decay,
            batched: self.batched,
            caps: self.caps,
            fit: self.fit,
            interleaved: self.interleaved,
            external: self.external,
            external_depth: self.external_depth,
            external_blocked: self.external_blocked,
            combined: self.combined,
            slow: self.slow,
        }
    }

    // Test seams for drafter-level tests in the architecture crates (Qwen4
    // drives a floor into a given state through these). Production code never
    // calls them; they are hidden from the documented API.

    /// Mark the floor calibrated at an AR price of `ar_us` without the
    /// decision a real calibration token takes.
    #[doc(hidden)]
    pub fn seed_calibrated(&mut self, ar_us: f32) {
        self.calibrated = true;
        self.ar_us = ar_us;
    }

    /// Overwrite the AR price alone.
    #[doc(hidden)]
    pub fn seed_ar_us(&mut self, ar_us: f32) {
        self.ar_us = ar_us;
    }

    /// The probe budget is spent.
    #[doc(hidden)]
    pub fn seed_probes_done(&mut self) {
        self.probes = FLOOR_PROBE_WINDOWS;
    }

    /// Overwrite the sticky retirement.
    #[doc(hidden)]
    pub fn seed_retired(&mut self, retired: bool) {
        self.retired = retired;
    }

    /// Overwrite the observed-window count.
    #[doc(hidden)]
    pub fn seed_observed(&mut self, observed: usize) {
        self.observed = observed;
    }

    /// AR probe tokens still to run.
    #[doc(hidden)]
    pub fn seed_probe_left(&mut self, probe_left: usize) {
        self.probe_left = probe_left;
    }

    /// The decayed (wall us, windows) of native windows that drafted `depth`
    /// tokens.
    #[doc(hidden)]
    pub fn seed_batched(&mut self, depth: usize, entry: (f32, f32)) {
        if depth < self.batched.len() {
            self.batched[depth] = entry;
        } else if let Some(slot) = self
            .extra_batched
            .get_mut(depth.saturating_sub(self.batched.len()))
        {
            *slot = entry;
        }
    }

    /// The decayed (wall us, windows) of takeover windows of `depth`
    /// candidates.
    #[doc(hidden)]
    pub fn seed_external_depth(&mut self, depth: usize, entry: (f32, f32)) {
        self.external_depth[depth] = entry;
    }

    /// The state the floor's callers branch on, after a call.
    fn after(&self) -> FloorAfter {
        FloorAfter {
            retired: self.retired,
            needs_ar: self.needs_ar_sample(),
        }
    }

    /// Start a new probe phase with this request's original thresholds.
    pub fn rearm(&mut self) {
        self.rearm_inner();
        if let Some(log) = self.call_log {
            log.emit("rearm", self.id, format_args!("{}", self.after()));
        }
    }

    /// Set the decay the floor prices its costs with (the arch binding).
    pub fn set_decay(&mut self, decay: f32) {
        self.decay = decay;
        if let Some(log) = self.call_log {
            log.emit(
                "set_decay",
                self.id,
                format_args!(",\"decay\":{}{}", FloorBits(decay), self.after()),
            );
        }
    }

    /// One finished ordinary AR token with wall time `us`.  The request's
    /// first observed one prices AR (an invalid timing prices it 0, which
    /// retires at the probe budget); the tokens a would-be retirement asked
    /// for ([`Self::retire_or_probe`]) re-price it by their mean and
    /// re-decide.
    pub fn observe_calibration(&mut self, us: f32, agreement: &[(f32, f32)]) {
        self.observe_calibration_inner(us, agreement);
        if let Some(log) = self.call_log {
            log.emit(
                "observe_calibration",
                self.id,
                format_args!(
                    ",\"us\":{},\"agreement\":{}{}",
                    FloorBits(us),
                    FloorAgreement(agreement),
                    self.after()
                ),
            );
        }
    }

    /// One finished native window of `k > 0` budget with wall time `us`,
    /// a batched one routed at the depth it drafted (see
    /// [`Self::observe_window`]).
    ///
    /// The call log records the window as the fields the floor reads
    /// (`committed.len()`, `drafts_generated`) and `accepted`.
    pub fn observe_native(
        &mut self,
        route: NativeWindow,
        window: &MtpWindow,
        us: f32,
        k: usize,
        agreement: &[(f32, f32)],
    ) {
        self.observe_native_inner(route, window, us, k, agreement);
        if let Some(log) = self.call_log {
            log.emit(
                "observe_native",
                self.id,
                format_args!(
                    ",\"route\":{},\"committed\":{},\"accepted\":{},\"drafts_generated\":{},\"us\":{},\"k\":{},\"agreement\":{}{}",
                    NativeWindowName(route),
                    window.committed.len(),
                    window.accepted,
                    window.drafts_generated,
                    FloorBits(us),
                    k,
                    FloorAgreement(agreement),
                    self.after()
                ),
            );
        }
    }

    /// One finished native window of `k > 0` budget with wall time `us`.
    pub fn observe_window(
        &mut self,
        route: NativeWindow,
        window: WindowStats,
        us: f32,
        k: usize,
        agreement: &[(f32, f32)],
    ) {
        self.observe_window_inner(route, window, us, k, agreement);
        if let Some(log) = self.call_log {
            log.emit(
                "observe_window",
                self.id,
                format_args!(
                    ",\"route\":{},\"emitted\":{},\"drafts\":{},\"cap\":{},\"us\":{},\"k\":{},\"agreement\":{}{}",
                    NativeWindowName(route),
                    window.emitted,
                    window.drafts,
                    window.cap,
                    FloorBits(us),
                    k,
                    FloorAgreement(agreement),
                    self.after()
                ),
            );
        }
    }

    /// One finished external window (`drafts > 0` offered, `emitted` tokens
    /// committed, `wall_us` including its verify).  Returns `true` when the
    /// caller must not take over again this request: the floor retired, or
    /// external windows cost no less per token than AR.
    pub fn observe_external(
        &mut self,
        emitted: usize,
        wall_us: f32,
        drafts: usize,
        agreement: &[(f32, f32)],
    ) -> bool {
        let ret = self.observe_external_inner(emitted, wall_us, drafts, agreement);
        if let Some(log) = self.call_log {
            log.emit(
                "observe_external",
                self.id,
                format_args!(
                    ",\"emitted\":{emitted},\"wall_us\":{},\"drafts\":{drafts},\"agreement\":{},\"ret\":{ret}{}",
                    FloorBits(wall_us),
                    FloorAgreement(agreement),
                    self.after()
                ),
            );
        }
        ret
    }

    /// One finished external window of the n-gram copy lane (`drafts > 0`
    /// offered, `emitted` tokens committed, `wall_us` including its verify).
    /// Returns `true` when the caller must not take over again this request:
    /// the floor retired, or external windows cost no less per token than AR.
    ///
    /// A retired floor keeps pricing external windows (`external`,
    /// `external_depth`, `external_blocked`): retirement says native
    /// drafting loses to AR, nothing about a copy source.  It neither
    /// un-retires, nor counts probes, nor touches native routing, and it
    /// returns `external_blocked` alone, so the caller stops taking over only
    /// when takeovers themselves lose to AR.
    pub fn observe_external_copy(
        &mut self,
        emitted: usize,
        wall_us: f32,
        drafts: usize,
        agreement: &[(f32, f32)],
    ) -> bool {
        let ret = self.observe_external_copy_inner(emitted, wall_us, drafts, agreement);
        if let Some(log) = self.call_log {
            log.emit(
                "observe_external_copy",
                self.id,
                format_args!(
                    ",\"emitted\":{emitted},\"wall_us\":{},\"drafts\":{drafts},\"agreement\":{},\"ret\":{ret}{}",
                    FloorBits(wall_us),
                    FloorAgreement(agreement),
                    self.after()
                ),
            );
        }
        ret
    }

    /// Route for the next native window with `k` drafts of budget.
    /// `provisional` is the unchanged Halo-table `batched_depth(k)`, used
    /// only while the probe budget lasts.  Afterwards a window may draft
    /// only while a measured option within `k` costs strictly less per
    /// emitted token than AR; it then runs the cheapest candidate within `k`
    /// ([`Self::best_candidate`]: a measured cap, or an unmeasured cap next
    /// to a measured one priced from it).  When no
    /// measured option wins (an external-only budget, `k` below every
    /// winning measured depth, a losing measured option, no valid AR price)
    /// the request retires, stickily, and runs AR: no losing route, and no
    /// unmeasured route without a measured winner, is ever run after the
    /// budget.  Exceptions: inside the retirement grace (`retire_after`
    /// observed windows) a request with a valid AR price keeps drafting on
    /// its cheapest measured option within `k`, else the provisional depth;
    /// and a floor with retire evidence ([`Self::with_retire_evidence`])
    /// first runs [`Action::ArSynced`] probe tokens, then keeps drafting
    /// while the slow-memory realized cost beats the probed AR price.
    pub fn route(&mut self, agreement: &[(f32, f32)], k: usize, provisional: usize) -> Action {
        let ret = self.route_inner(agreement, k, provisional);
        if let Some(log) = self.call_log {
            let fields = RouteFields {
                agreement,
                k,
                provisional,
                ret,
                after: self.after(),
            };
            log.emit("route", self.id, format_args!("{fields}"));
        }
        ret
    }
}

/// One floor of a replayed call log: an `install` line and every call that
/// followed it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloorReplay {
    /// The `"floor"` id of the log (per process).
    pub floor: u64,
    /// What built the floor (the `install` line's `owner`).
    pub owner: String,
    /// Calls replayed (the `install` line excluded).
    pub calls: usize,
    /// Calls whose `ret`, `retired_after` or `needs_ar_after` differ.
    pub mismatches: usize,
    /// The first mismatch: its `seq`, op and the logged and replayed values.
    pub first_mismatch: Option<String>,
}

/// The outcome of [`replay_call_log`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayReport {
    /// One entry per `install` line, in log order.
    pub floors: Vec<FloorReplay>,
    /// Call lines of a floor whose `install` line the log does not hold (a
    /// log that begins mid-run); they cannot be replayed.
    pub orphan_calls: usize,
    /// Lines carrying the prefix that did not parse as a call.
    pub malformed: usize,
    /// The first malformed line with the reason.
    pub first_malformed: Option<String>,
}

impl ReplayReport {
    /// Calls replayed over every floor.
    pub fn calls(&self) -> usize {
        self.floors.iter().map(|floor| floor.calls).sum()
    }

    /// Calls that did not reproduce their logged result over every floor.
    pub fn mismatches(&self) -> usize {
        self.floors.iter().map(|floor| floor.mismatches).sum()
    }
}

fn json_field<'v>(
    value: &'v serde_json::Value,
    key: &str,
) -> Result<&'v serde_json::Value, String> {
    value.get(key).ok_or_else(|| format!("missing `{key}`"))
}

fn json_u64(value: &serde_json::Value, key: &str) -> Result<u64, String> {
    json_field(value, key)?
        .as_u64()
        .ok_or_else(|| format!("`{key}` is not an unsigned integer"))
}

fn json_usize(value: &serde_json::Value, key: &str) -> Result<usize, String> {
    usize::try_from(json_u64(value, key)?).map_err(|_| format!("`{key}` overflows usize"))
}

fn json_bool(value: &serde_json::Value, key: &str) -> Result<bool, String> {
    json_field(value, key)?
        .as_bool()
        .ok_or_else(|| format!("`{key}` is not a bool"))
}

fn json_str<'v>(value: &'v serde_json::Value, key: &str) -> Result<&'v str, String> {
    json_field(value, key)?
        .as_str()
        .ok_or_else(|| format!("`{key}` is not a string"))
}

/// An `f32` logged as [`FloorBits`]: a quoted hex `u32`.
fn json_bits(value: &serde_json::Value) -> Result<f32, String> {
    let digits = value.as_str().ok_or("an f32 is not a hex string")?;
    u32::from_str_radix(digits, 16)
        .map(f32::from_bits)
        .map_err(|error| format!("bad f32 bits {digits:?}: {error}"))
}

fn json_f32(value: &serde_json::Value, key: &str) -> Result<f32, String> {
    json_bits(json_field(value, key)?)
}

fn json_agreement(value: &serde_json::Value) -> Result<Vec<(f32, f32)>, String> {
    let pairs = json_field(value, "agreement")?
        .as_array()
        .ok_or("`agreement` is not an array")?;
    if pairs.len() > DERIVED_MAX_DRAFTS {
        return Err(format!(
            "`agreement` has {} depths, expected at most {DERIVED_MAX_DRAFTS}",
            pairs.len()
        ));
    }
    let mut agreement = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let pair = pair
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or("an `agreement` depth is not a pair")?;
        agreement.push((json_bits(&pair[0])?, json_bits(&pair[1])?));
    }
    Ok(agreement)
}

fn json_native_window(value: &serde_json::Value) -> Result<NativeWindow, String> {
    match json_str(value, "route")? {
        "Interleaved" => Ok(NativeWindow::Interleaved),
        "Batched" => Ok(NativeWindow::Batched),
        other => Err(format!("unknown native route {other:?}")),
    }
}

/// `Some(description)` when `logged` and `replayed` differ.
fn differs<T: PartialEq + std::fmt::Debug>(what: &str, logged: T, replayed: T) -> Option<String> {
    (logged != replayed).then(|| format!("{what} logged {logged:?}, replayed {replayed:?}"))
}

/// Re-execute one call `op` of the log on `floor`; `Ok(Some(..))` when its
/// logged result differs, `Err` when the line is not a call.
fn replay_call<const W: usize, const E: usize>(
    floor: &mut Floor<W, E>,
    op: &str,
    value: &serde_json::Value,
) -> Result<Option<String>, String> {
    let ret = match op {
        "observe_calibration" => {
            floor.observe_calibration(json_f32(value, "us")?, &json_agreement(value)?);
            None
        }
        "observe_native" => {
            let window = MtpWindow {
                committed: vec![0; json_usize(value, "committed")?],
                accepted: json_usize(value, "accepted")?,
                drafts_generated: json_usize(value, "drafts_generated")?,
            };
            floor.observe_native(
                json_native_window(value)?,
                &window,
                json_f32(value, "us")?,
                json_usize(value, "k")?,
                &json_agreement(value)?,
            );
            None
        }
        "observe_window" => {
            let window = WindowStats {
                emitted: json_usize(value, "emitted")?,
                drafts: json_usize(value, "drafts")?,
                cap: json_usize(value, "cap")?,
            };
            floor.observe_window(
                json_native_window(value)?,
                window,
                json_f32(value, "us")?,
                json_usize(value, "k")?,
                &json_agreement(value)?,
            );
            None
        }
        "observe_external" | "observe_external_copy" => {
            let (emitted, wall_us, drafts) = (
                json_usize(value, "emitted")?,
                json_f32(value, "wall_us")?,
                json_usize(value, "drafts")?,
            );
            let agreement = json_agreement(value)?;
            let ret = if op == "observe_external" {
                floor.observe_external(emitted, wall_us, drafts, &agreement)
            } else {
                floor.observe_external_copy(emitted, wall_us, drafts, &agreement)
            };
            differs("ret", json_bool(value, "ret")?, ret)
        }
        "route" => {
            let ret = floor.route(
                &json_agreement(value)?,
                json_usize(value, "k")?,
                json_usize(value, "provisional")?,
            );
            differs("ret", json_str(value, "ret")?.to_string(), action_name(ret))
        }
        "rearm" => {
            floor.rearm();
            None
        }
        "reset_request" => {
            floor.reset_request(
                json_usize(value, "retire_after")?,
                json_f32(value, "decay")?,
                json_usize(value, "probe_tokens")?,
            );
            None
        }
        "set_capacity" => {
            floor.set_capacity(json_usize(value, "capacity")?);
            None
        }
        "set_decay" => {
            floor.set_decay(json_f32(value, "decay")?);
            None
        }
        other => return Err(format!("unknown op {other:?}")),
    };
    let retired_after = json_bool(value, "retired_after")?;
    let needs_ar_after = json_bool(value, "needs_ar_after")?;
    let after = differs("retired_after", retired_after, floor.retired())
        .or_else(|| differs("needs_ar_after", needs_ar_after, floor.needs_ar_sample()));
    Ok(ret.or(after))
}

/// One parsed call-log line applied to the replay state.
fn replay_line<const W: usize, const E: usize>(
    report: &mut ReplayReport,
    live: &mut HashMap<u64, (Floor<W, E>, usize)>,
    window_cost: &[f32; W],
    value: &serde_json::Value,
) -> Result<(), String> {
    let id = json_u64(value, "floor")?;
    let op = json_str(value, "op")?;
    if op == "install" {
        let retire_after = json_usize(value, "retire_after")?;
        let mut floor = Floor::<W, E>::with_retire_after(*window_cost, retire_after)
            .with_decay(json_f32(value, "decay")?)
            .with_retire_evidence(json_usize(value, "probe_tokens")?);
        if let Ok(capacity) = json_usize(value, "capacity") {
            floor.set_capacity(capacity);
        }
        report.floors.push(FloorReplay {
            floor: id,
            owner: json_str(value, "owner")?.to_string(),
            calls: 0,
            mismatches: 0,
            first_mismatch: None,
        });
        live.insert(id, (floor, report.floors.len() - 1));
        return Ok(());
    }
    let Some((floor, slot)) = live.get_mut(&id) else {
        report.orphan_calls += 1;
        return Ok(());
    };
    let mismatch = replay_call(floor, op, value)?;
    let replayed = &mut report.floors[*slot];
    replayed.calls += 1;
    if let Some(message) = mismatch {
        replayed.mismatches += 1;
        let seq = value.get("seq").and_then(serde_json::Value::as_u64);
        replayed
            .first_mismatch
            .get_or_insert_with(|| format!("seq {seq:?} {op}: {message}"));
    }
    Ok(())
}

/// Replay a recorded call log (the lines `with_call_log` writes under
/// `prefix`, with any text before the prefix ignored) into fresh floors and
/// compare every result bit for bit.
///
/// An `install` line builds a fresh floor on `window_cost` with the logged
/// `retire_after`, `decay` and `probe_tokens`; every other op is re-executed
/// with its exact logged inputs.  The `ret`, `retired_after` and
/// `needs_ar_after` of every call must equal the logged ones.  Lines of other
/// kinds are skipped; lines that carry the prefix but do not parse count as
/// malformed.  Floor ids restart per process, so replay one process's log per
/// call.
pub fn replay_call_log<'a, const W: usize, const E: usize>(
    lines: impl Iterator<Item = &'a str>,
    prefix: &str,
    window_cost: [f32; W],
) -> ReplayReport {
    let mut report = ReplayReport::default();
    let mut live: HashMap<u64, (Floor<W, E>, usize)> = HashMap::new();
    for line in lines {
        let Some((_, rest)) = line.split_once(prefix) else {
            continue;
        };
        let json = rest.trim();
        if !json.starts_with('{') {
            continue;
        }
        let outcome = serde_json::from_str::<serde_json::Value>(json)
            .map_err(|error| format!("not JSON: {error}"))
            .and_then(|value| replay_line(&mut report, &mut live, &window_cost, &value));
        if let Err(message) = outcome {
            report.malformed += 1;
            report
                .first_malformed
                .get_or_insert_with(|| format!("{message}: {json}"));
        }
    }
    report
}

// ---------------------------------------------------------------------------
// Flash-Next derived speculative depth (replay2 policy `derived-causal`).
//
// A request-local E/T chooser over measured per-width window prices and a
// frozen calibration, ported with replay2's exact f64 arithmetic so a CPU
// oracle can demand action equality on the recorded traces. Pure CPU: the
// calibration is the generated `fn_phase1_v1` table, never read at runtime
// from a file, the environment or a GPU.
// ---------------------------------------------------------------------------

// The generated table's provenance (schema, digests, per-price sample counts)
// is read by receipts and tests, not by the lookups below.
#[allow(dead_code)]
mod fn_phase1_v1;

// Data-only measured-book payload; empty until actual receipts are exported.
#[allow(dead_code)]
mod fn_width_cost_v1;

/// Version of the frozen calibration, window prices and chooser law: the
/// generated table's `VERSION` and the `derived_k` trace events' `version`.
pub const FN_DERIVED_VERSION: &str = "fn-phase1-v1/replay2-v1";

/// Per-depth prior numerator and weight: `(s + 1.6) / (n + 2)` is 0.8 with no
/// support (replay2 `CAL_PRIOR_A`, `CAL_PRIOR_N`).
const DERIVED_PRIOR_A: f64 = 1.6;
const DERIVED_PRIOR_N: f64 = 2.0;
/// Weight with which a feature cell pools toward its depth's probability
/// (replay2 `CAL_POOL_STRENGTH`).
const DERIVED_POOL_STRENGTH: f64 = 2.0;
/// Residual cells a request keeps beyond one native cell per draft depth: the
/// n-gram feature cells a request visits. With the default [`MAX_DEPTH`] depths
/// the reservation is 128 cells.
const DERIVED_NGRAM_CELL_ALLOWANCE: usize = 118;
/// Most draft rows one derived request can price: the physical row-exact
/// verify width. Residual keys carry the depth as `u8`, so nothing below this
/// bound is capped at the incumbent floor's [`MAX_DEPTH`].
pub const DERIVED_MAX_DRAFTS: usize = 63;

/// Proposal source whose depths a derived decision prices.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DerivedSource {
    Mtp,
    Ngram,
}

/// Frozen-calibration card. The variant fixes the request-local residual
/// decay: 0.875 on the R9700, 0.97 on Halo.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DerivedCard {
    R9700,
    Halo,
}

impl DerivedCard {
    /// Per-window decay of the request-local residual counts.
    const fn decay(self) -> f64 {
        match self {
            Self::R9700 => 0.875,
            Self::Halo => 0.97,
        }
    }
}

/// Context bucket of a window's absolute position (replay2 `context_bucket`):
/// `<=2048`, `<=8192`, `<=16384`, else (replay2's label `32K+`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ContextBucket {
    Le2K,
    Le8K,
    Le16K,
    Over16K,
}

impl ContextBucket {
    /// Bucket of absolute context `position` in tokens.
    pub fn of(position: usize) -> Self {
        if position <= 2048 {
            Self::Le2K
        } else if position <= 8192 {
            Self::Le8K
        } else if position <= 16384 {
            Self::Le16K
        } else {
            Self::Over16K
        }
    }
}

/// Binned n-gram proposal length (replay2 `_cal_offered_bin`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum OfferBin {
    Missing,
    Zero,
    One,
    TwoThree,
    FourSeven,
    EightFifteen,
    SixteenThirtyOne,
    ThirtyTwoPlus,
}

/// Number of [`NativeMarginBin`] variants, in preprofile integer order.
pub const NATIVE_MARGIN_BIN_COUNT: usize = 5;

/// Binned native draft logit margin, observed at one committed draft row.
/// Preprofile schema exports these as integers 0..4 in declaration order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum NativeMarginBin {
    LtQuarter,
    LtOnePointFive,
    LtThree,
    LtSix,
    GeSix,
}

impl NativeMarginBin {
    /// Preprofile integer code of this bin: 0..4 in declaration order.
    pub const fn index(self) -> usize {
        match self {
            Self::LtQuarter => 0,
            Self::LtOnePointFive => 1,
            Self::LtThree => 2,
            Self::LtSix => 3,
            Self::GeSix => 4,
        }
    }
}

/// Bin of a native draft logit margin: `[0, 0.25)`, `[0.25, 1.5)`,
/// `[1.5, 3)`, `[3, 6)`, `[6, +inf)`. A negative or non-finite margin
/// carries no evidence (`None`); the caller retains the unconditional
/// evidence. Never invents a zero bin.
pub fn native_margin_bin(margin: f32) -> Option<NativeMarginBin> {
    if !margin.is_finite() || margin < 0.0 {
        return None;
    }
    if margin < 0.25 {
        Some(NativeMarginBin::LtQuarter)
    } else if margin < 1.5 {
        Some(NativeMarginBin::LtOnePointFive)
    } else if margin < 3.0 {
        Some(NativeMarginBin::LtThree)
    } else if margin < 6.0 {
        Some(NativeMarginBin::LtSix)
    } else {
        Some(NativeMarginBin::GeSix)
    }
}

/// Feature cell of one (window, depth). Native MTP's pre-decision cell is
/// [`DerivedFeature::Native`]: the draft margin is absent when the depth is
/// chosen, which is replay2's feature key `(('margin', 'na'),)`. A
/// [`DerivedFeature::NativeMargin`] cell is commit-only causal evidence: the
/// margin bin observed at an already-committed draft row, used to re-solve
/// the remaining depths after that row. An n-gram cell is its offer bin,
/// the matched key length and whether the previous executed n-gram window
/// accepted its first token (`None` = unknown).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DerivedFeature {
    Native,
    NativeMargin {
        observed_depth: u8,
        margin: NativeMarginBin,
    },
    Ngram {
        offered: OfferBin,
        key_len: Option<u32>,
        last_first: Option<bool>,
    },
}

/// Request-local residual cell: replay2's `(source, depth, feature_key)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ResidualKey {
    pub source: DerivedSource,
    pub depth: u8,
    pub feature: DerivedFeature,
}

/// Frozen calibration counts behind one depth: successes and reached trials of
/// the depth, and of its feature cell.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CalibrationCounts {
    pub depth_s: u64,
    pub depth_n: u64,
    pub cell_s: u64,
    pub cell_n: u64,
}

/// One draft depth a decision prices: its pre-decision residual key and the
/// calibration counts behind it.
#[derive(Clone, Copy, Debug)]
pub struct DerivedDepth {
    pub key: ResidualKey,
    pub counts: CalibrationCounts,
}

/// Measured cost of a window verified at `k` drafted rows, in microseconds;
/// `k == 0` is a plain AR step.
#[derive(Clone, Copy, Debug)]
pub struct DerivedPrice {
    pub k: u8,
    pub cost_us: f64,
}

/// What a derived decision runs: an AR step or a drafted prefix of that width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivedAction {
    Ar,
    Prefix(u8),
}

/// The chosen action with the expected emitted tokens and the cost it was
/// priced at.
#[derive(Clone, Copy, Debug)]
pub struct DerivedDecision {
    pub action: DerivedAction,
    pub expected: f64,
    pub cost_us: f64,
}

/// Which verify route a window price was measured on. A price is only valid
/// under the mode it was measured with; the two modes are never mixed, and a
/// mode without a measured profile is unsupported (`None`), never the other
/// mode's price.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DerivedCostMode {
    /// The default pool route (copy off).
    OffPool,
    /// The on-copy route.
    OnCopy,
}

/// Running expected-token yield of a drafted prefix: `expected = 1 + sum S_i`
/// with `S_i` the survival (whole-prefix acceptance) probability of depth `i`.
/// Appending a depth multiplies the survival by that depth's *conditional*
/// acceptance probability, never by a survival.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerivedYield {
    pub expected: f64,
    pub survival: f64,
}

impl DerivedYield {
    /// The width-0 (AR) yield: one target token, survival 1.
    pub fn seed() -> Self {
        Self {
            expected: 1.0,
            survival: 1.0,
        }
    }

    /// The yield one draft deeper, given the conditional probability that
    /// the new depth is accepted given every shallower depth was.
    pub fn append(self, conditional_accept: f64) -> Self {
        let survival = self.survival * conditional_accept;
        Self {
            expected: self.expected + survival,
            survival,
        }
    }

    /// This yield priced at the total window cost `cost_us`.
    pub fn objective(self, cost_us: f64) -> DerivedObjective {
        DerivedObjective {
            expected: self.expected,
            cost_us,
        }
    }
}

/// One action's expected emitted tokens and its total window cost (the draft
/// work is already inside `cost_us`). The rate `expected / cost_us` is the
/// single objective both the global chooser and the incremental marginal
/// derive from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerivedObjective {
    pub expected: f64,
    pub cost_us: f64,
}

impl DerivedObjective {
    /// Expected emitted tokens per microsecond.
    pub fn rate(self) -> f64 {
        self.expected / self.cost_us
    }

    /// The marginal of moving from `self` to the adjacent deeper `next`.
    /// `next.rate() > self.rate()` iff `delta_expected * (T / E) > delta_cost`
    /// with `T`, `E` the cost and yield of `self`. A non-positive or
    /// non-finite cost or yield makes the marginal invalid (NaN value and
    /// surplus), which never improves the rate.
    pub fn marginal_to(self, next: Self) -> DerivedMarginal {
        let delta_expected = next.expected - self.expected;
        let delta_cost_us = next.cost_us - self.cost_us;
        let valid = self.cost_us.is_finite()
            && next.cost_us.is_finite()
            && self.expected.is_finite()
            && self.cost_us > 0.0
            && next.cost_us > 0.0
            && self.expected > 0.0;
        let emitted_token_value_us = if valid {
            delta_expected * (self.cost_us / self.expected)
        } else {
            f64::NAN
        };
        DerivedMarginal {
            delta_expected,
            delta_cost_us,
            emitted_token_value_us,
            surplus_us: emitted_token_value_us - delta_cost_us,
        }
    }
}

/// The marginal of one adjacent depth: the tokens it adds, the cost it adds,
/// what those tokens are worth at the current rate (`delta_expected * T / E`)
/// and the surplus of that value over the added cost.
///
/// Stopping at the first depth whose marginal fails is a local rule: on
/// non-monotone costs a later depth can still win globally, so the global
/// optimum is [`DerivedRequest::choose`]'s enumeration, not a first failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerivedMarginal {
    pub delta_expected: f64,
    pub delta_cost_us: f64,
    pub emitted_token_value_us: f64,
    pub surplus_us: f64,
}

impl DerivedMarginal {
    /// The deeper action has a strictly higher rate. Equality does not
    /// improve, so an incremental append stops on a tie.
    pub fn improves_rate(self) -> bool {
        self.surplus_us > 0.0
    }
}

/// Request-local state of the derived chooser: the decayed residual
/// success/failure counts of every cell the request has touched, and the last
/// first-token outcome per source.
#[derive(Clone, Debug)]
pub struct DerivedRequest {
    decay: f64,
    /// `(cell, successes, failures)`, sorted by cell.
    cells: Vec<(ResidualKey, f64, f64)>,
    /// Previous executed window's first-token outcome, by [`DerivedSource`]:
    /// `Some(true)` accepted, `Some(false)` rejected, `None` unknown (nothing
    /// executed yet, or a terminal window that accepted nothing).
    last_first: [Option<bool>; 2],
}

impl DerivedRequest {
    /// An empty request for `card`; cell storage is reserved for the incumbent
    /// floor's [`MAX_DEPTH`] native depths.
    pub fn new(card: DerivedCard) -> Self {
        Self::with_max_drafts(card, MAX_DEPTH)
    }

    /// An empty request for `card` with cell storage reserved for `max_drafts`
    /// native depths (at most [`DERIVED_MAX_DRAFTS`]) plus the causal margin
    /// cells and the n-gram cells, so a window that touches up to that many
    /// depths allocates nothing.
    pub fn with_max_drafts(card: DerivedCard, max_drafts: usize) -> Self {
        let mut request = Self {
            decay: card.decay(),
            cells: Vec::new(),
            last_first: [None; 2],
        };
        request.set_capacity(max_drafts);
        request
    }

    /// Reserve residual cells for `max_drafts` admitted depths: one native
    /// cell per depth plus every causal margin cell the commit path bumps
    /// (observed depths `1..=executed` crossed with target depths
    /// `1..=executed`, times [`NATIVE_MARGIN_BIN_COUNT`] bins), plus the
    /// n-gram allowance. Grow-only (no-op when the reservation already
    /// covers it); preserves the sorted sparse history and never reallocates
    /// on in-capacity observations.
    pub fn set_capacity(&mut self, max_drafts: usize) {
        let want = max_drafts.min(DERIVED_MAX_DRAFTS);
        let need = want + want * want * NATIVE_MARGIN_BIN_COUNT + DERIVED_NGRAM_CELL_ALLOWANCE;
        if self.cells.capacity() < need {
            self.cells.reserve(need - self.cells.len());
        }
    }

    /// Per-window decay of the residual counts.
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// First-token outcome of the previous executed window of `source`.
    pub fn last_first(&self, source: DerivedSource) -> Option<bool> {
        self.last_first[source as usize]
    }

    /// Probability that draft depth `depth` is accepted given the prefix before
    /// it was: the frozen depth prior, pooled with its feature cell, then
    /// updated by the request's own decayed residual.
    pub fn probability(&self, depth: &DerivedDepth) -> f64 {
        let counts = depth.counts;
        let p_depth =
            (counts.depth_s as f64 + DERIVED_PRIOR_A) / (counts.depth_n as f64 + DERIVED_PRIOR_N);
        let p_cell = (counts.cell_s as f64 + DERIVED_POOL_STRENGTH * p_depth)
            / (counts.cell_n as f64 + DERIVED_POOL_STRENGTH);
        let (success, failure) = match self.cells.binary_search_by(|cell| cell.0.cmp(&depth.key)) {
            Ok(at) => (self.cells[at].1, self.cells[at].2),
            Err(_) => (0.0, 0.0),
        };
        (2.0 * p_cell + success) / (2.0 + success + failure)
    }

    /// Calls `visit(k, objective)` for the AR action and every drafted prefix
    /// that has a price: the expected emitted tokens of width `k` are one
    /// target token plus the survival probability at each of its depths.
    fn for_each_action(
        &self,
        depths: &[DerivedDepth],
        prices: &[DerivedPrice],
        mut visit: impl FnMut(u8, DerivedObjective),
    ) {
        let cost_of = |k: u8| prices.iter().find(|price| price.k == k).map(|p| p.cost_us);
        let mut yielded = DerivedYield::seed();
        if let Some(cost) = cost_of(0) {
            visit(0, yielded.objective(cost));
        }
        for (index, depth) in depths.iter().take(usize::from(u8::MAX)).enumerate() {
            yielded = yielded.append(self.probability(depth));
            let k = (index + 1) as u8;
            if let Some(cost) = cost_of(k) {
                visit(k, yielded.objective(cost));
            }
        }
    }

    /// The action with the highest expected tokens per microsecond among AR
    /// and the priced prefixes `1..=depths.len()`, enumerating every
    /// supported prefix (a first failed marginal is not a global optimum).
    /// `depths` are the contiguous depths `1..=W` of one source; `prices`
    /// have unique `k` in any order; prices of a width beyond `W` and
    /// non-positive costs are not actions. Rates within replay2's
    /// `math.isclose(rel_tol=1e-12, abs_tol=1e-15)` of the best tie; ties
    /// prefer `incumbent`, then the smaller `k`. `None` when no action is
    /// priced. Never mutates the request.
    pub fn choose(
        &self,
        depths: &[DerivedDepth],
        prices: &[DerivedPrice],
        incumbent: u8,
    ) -> Option<DerivedDecision> {
        let mut best: Option<f64> = None;
        self.for_each_action(depths, prices, |_, objective| {
            if objective.cost_us > 0.0 {
                let rate = objective.rate();
                if best.is_none_or(|current| rate > current) {
                    best = Some(rate);
                }
            }
        });
        let best = best?;
        let mut pick: Option<(u8, DerivedObjective)> = None;
        self.for_each_action(depths, prices, |k, objective| {
            if objective.cost_us > 0.0 && derived_rates_tied(objective.rate(), best) {
                let better =
                    pick.is_none_or(|(held, _)| (k != incumbent, k) < (held != incumbent, held));
                if better {
                    pick = Some((k, objective));
                }
            }
        });
        pick.map(|(k, objective)| DerivedDecision {
            action: if k == 0 {
                DerivedAction::Ar
            } else {
                DerivedAction::Prefix(k)
            },
            expected: objective.expected,
            cost_us: objective.cost_us,
        })
    }

    /// Adds one success or failure to `key`'s residual cell.
    fn bump(&mut self, key: ResidualKey, success: bool) {
        let at = match self.cells.binary_search_by(|cell| cell.0.cmp(&key)) {
            Ok(at) => at,
            Err(at) => {
                self.cells.insert(at, (key, 0.0, 0.0));
                at
            }
        };
        if success {
            self.cells[at].1 += 1.0;
        } else {
            self.cells[at].2 += 1.0;
        }
    }

    /// Records one consulted window at its final publish boundary. Every
    /// existing residual cell decays once first; then, when `executed > 0`,
    /// `keys[..executed]` (the pre-decision keys of depths `1..=executed`)
    /// receive the outcome: successes at depths `1..=min(accepted, executed)`
    /// and, unless `terminal`, one failure at the first unaccepted depth;
    /// deeper depths and a terminal rejection are censored. The source's
    /// first-token history becomes accepted, or unknown for a terminal window
    /// that accepted nothing, else rejected. `executed == 0` (AR, or no
    /// supported action) only ages the residuals and leaves the history.
    /// Commit-only multi-feature observations go through [`Self::observe_with_margins`]
    /// with empty margins, which is exactly this update.
    pub fn observe(&mut self, keys: &[ResidualKey], executed: u8, accepted: usize, terminal: bool) {
        self.observe_with_margins(keys, &[], executed, accepted, terminal);
    }

    /// The commit-only multi-feature observation API: [`Self::observe`] plus
    /// causal margin evidence, in one aging pass. `margins[j]` is the binned
    /// logit margin of committed draft row `j + 1` (`observed_depth = j + 1`);
    /// `None` is a row with no usable margin. A longer `margins` is ignored
    /// past `executed`, a shorter one is a `None` tail. After the base update
    /// runs verbatim, every `Some` row labels each reached target depth
    /// `t in 1..=executed` under `(Mtp, t, NativeMargin { observed_depth, margin })`
    /// with the same censoring: success iff `t <= accepted`, one failure iff
    /// `t == accepted + 1` and the window is not terminal, deeper or
    /// terminal-unobserved rejections censored. No second aging, no second
    /// `last_first` write. Empty or all-`None` margins are exactly
    /// [`Self::observe`]. Only the adapter calls this, after verify commit,
    /// for native-MTP windows; never for n-gram or AR windows.
    pub fn observe_with_margins(
        &mut self,
        keys: &[ResidualKey],
        margins: &[Option<NativeMarginBin>],
        executed: u8,
        accepted: usize,
        terminal: bool,
    ) {
        let decay = self.decay;
        for cell in &mut self.cells {
            cell.1 *= decay;
            cell.2 *= decay;
        }
        if executed == 0 {
            return;
        }
        let executed_len = usize::from(executed);
        let keys = &keys[..executed_len];
        let accepted = accepted.min(executed_len);
        for key in &keys[..accepted] {
            self.bump(*key, true);
        }
        if accepted < executed_len && !terminal {
            self.bump(keys[accepted], false);
        }
        self.last_first[keys[0].source as usize] = if accepted > 0 {
            Some(true)
        } else if terminal {
            None
        } else {
            Some(false)
        };
        for (row, bin) in margins.iter().enumerate().take(executed_len) {
            let Some(bin) = *bin else { continue };
            // `row < executed_len <= 255`, so `row + 1` is a legal depth.
            let observed_depth = (row + 1) as u8;
            for target in 1..=executed {
                let key = ResidualKey {
                    source: DerivedSource::Mtp,
                    depth: target,
                    feature: DerivedFeature::NativeMargin {
                        observed_depth,
                        margin: bin,
                    },
                };
                if usize::from(target) <= accepted {
                    self.bump(key, true);
                } else if usize::from(target) == accepted + 1 && !terminal {
                    self.bump(key, false);
                }
            }
        }
    }
}

/// `math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)`.
fn derived_rates_tied(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    (a - b).abs() <= (1e-12 * a.abs().max(b.abs())).max(1e-15)
}

/// Offer bin of an n-gram proposal length; `None` = no offer recorded.
pub fn derived_offer_bin(offered: Option<usize>) -> OfferBin {
    match offered {
        None => OfferBin::Missing,
        Some(0) => OfferBin::Zero,
        Some(1) => OfferBin::One,
        Some(2..=3) => OfferBin::TwoThree,
        Some(4..=7) => OfferBin::FourSeven,
        Some(8..=15) => OfferBin::EightFifteen,
        Some(16..=31) => OfferBin::SixteenThirtyOne,
        Some(_) => OfferBin::ThirtyTwoPlus,
    }
}

/// Frozen Flash-Next calibration of one depth: greedy totals pooled over every
/// measured arm of `card`, zeros where the table has no support.
pub fn fn_phase1_counts(
    card: DerivedCard,
    source: DerivedSource,
    depth: u8,
    feature: DerivedFeature,
) -> CalibrationCounts {
    use fn_phase1_v1::{CELL_TOTALS, DEPTH_TOTALS};
    let (depth_s, depth_n) = DEPTH_TOTALS
        .binary_search_by(|row| (row.0, row.1, row.2).cmp(&(card, source, depth)))
        .map_or((0, 0), |at| (DEPTH_TOTALS[at].3, DEPTH_TOTALS[at].4));
    let (cell_s, cell_n) = CELL_TOTALS
        .binary_search_by(|row| (row.0, row.1, row.2, row.3).cmp(&(card, source, depth, feature)))
        .map_or((0, 0), |at| (CELL_TOTALS[at].4, CELL_TOTALS[at].5));
    CalibrationCounts {
        depth_s,
        depth_n,
        cell_s,
        cell_n,
    }
}

/// Frozen median window price in microseconds at drafted width `k` under
/// verify route `mode`; the flag is `true` when the context bucket had no
/// sample and the median pools the width over every context. `None` =
/// unsupported (never interpolated, never another mode's price). Planned
/// `confirmation` is required for the frozen OffPool n-gram profile; native
/// MTP ignores it. OnCopy has no supported profile in this frozen table.
pub fn fn_phase1_window_price(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    bucket: ContextBucket,
    k: u8,
    confirmation: bool,
) -> Option<(f64, bool)> {
    use fn_phase1_v1::WINDOW_PRICES;
    if source == DerivedSource::Ngram && !confirmation {
        return None;
    }
    match mode {
        DerivedCostMode::OffPool => WINDOW_PRICES
            .binary_search_by(|row| (row.0, row.1, row.2, row.3).cmp(&(card, source, bucket, k)))
            .ok()
            .map(|at| (WINDOW_PRICES[at].4, WINDOW_PRICES[at].6)),
        DerivedCostMode::OnCopy => None,
    }
}

/// Frozen median AR step price in microseconds for `bucket` under verify
/// route `mode`; the flag is `true` when the median pools every context
/// bucket. `None` = unsupported, including every [`DerivedCostMode::OnCopy`]
/// price until that mode has a measured profile.
pub fn fn_phase1_ar_price(
    card: DerivedCard,
    mode: DerivedCostMode,
    bucket: ContextBucket,
) -> Option<(f64, bool)> {
    use fn_phase1_v1::AR_PRICES;
    match mode {
        DerivedCostMode::OffPool => AR_PRICES
            .binary_search_by(|row| (row.0, row.1).cmp(&(card, bucket)))
            .ok()
            .map(|at| (AR_PRICES[at].2, AR_PRICES[at].4)),
        DerivedCostMode::OnCopy => None,
    }
}

/// Active full-window price: an exact measured cell first. The frozen
/// OffPool profile is available only when the entire measured slot is absent,
/// never to fill holes in another arithmetic identity. Invalid or unbound
/// measured slots refuse. Totals already include head.
pub fn fn_measured_window_price(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    bucket: ContextBucket,
    k: u8,
    confirmation: bool,
) -> Option<(f64, bool)> {
    if k == 0
        || (source == DerivedSource::Ngram && mode == DerivedCostMode::OffPool && !confirmation)
    {
        return None;
    }
    let source_word = match source {
        DerivedSource::Mtp => "Mtp",
        DerivedSource::Ngram => "Ngram",
    };
    let confirmed = source == DerivedSource::Ngram && confirmation;
    let key = fn_width_key(card, mode, bucket);
    measured_book_index().active_price(
        (key.0, key.1, key.2, source_word, confirmed, usize::from(k)),
        || fn_phase1_window_price(card, mode, source, bucket, k, confirmed),
    )
}

/// Active target-only AR total at K0. Only an absent measured OffPool slot
/// uses the frozen profile; a missing cell of a measured identity refuses.
pub fn fn_measured_ar_price(
    card: DerivedCard,
    mode: DerivedCostMode,
    bucket: ContextBucket,
) -> Option<(f64, bool)> {
    let key = fn_width_key(card, mode, bucket);
    measured_book_index().active_price((key.0, key.1, key.2, "ArRouted", false, 0), || {
        fn_phase1_ar_price(card, mode, bucket)
    })
}

/// Mode-aware conditional calibration. Frozen OffPool counts are available
/// only with no measured slot. Measured native and n-gram counts must name the
/// admitted price identity, never just the same card/mode. OnCopy never borrows
/// the OffPool profile. A native margin cell pairs its measured parent with its
/// measured conditional cell; an unsupported margin cell is `None` so the
/// adapter keeps its unconditional evidence, never a fabricated 0.8.
/// Missing/ambiguous support is `None`.
pub fn fn_measured_depth_counts(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    depth: u8,
    feature: DerivedFeature,
) -> Option<CalibrationCounts> {
    let key = fn_width_key(card, mode, ContextBucket::Le2K);
    let book = measured_book_index();
    if !book.price_slot_supported(key.0, key.1) {
        return None;
    }
    let legacy = book.legacy_fallback(key.0, key.1);
    if source == DerivedSource::Ngram {
        if legacy {
            return Some(fn_phase1_counts(card, source, depth, feature));
        }
        if !matches!(feature, DerivedFeature::Ngram { .. }) {
            return None;
        }
        let (card, mode, _) = fn_width_key(card, mode, ContextBucket::Le2K);
        return fn_width_depth_counts(
            fn_width_cost_v1::DEPTH_COUNTS,
            card,
            mode,
            usize::from(depth),
            book.identities[slot_of_word(card, mode)],
        );
    }
    if !book.counts_identity_supported(
        key.0,
        key.1,
        native_counts_index().identities[slot_of_word(key.0, key.1)],
    ) {
        return None;
    }
    match feature {
        DerivedFeature::Native => match query_native_parent(card, mode, depth) {
            Ok(Some((depth_s, depth_n))) => Some(CalibrationCounts {
                depth_s,
                depth_n,
                cell_s: 0,
                cell_n: 0,
            }),
            Ok(None) if legacy => {
                let frozen = fn_phase1_counts(card, source, depth, DerivedFeature::Native);
                (frozen.depth_n > 0).then_some(frozen)
            }
            Ok(None) | Err(()) => None,
        },
        DerivedFeature::NativeMargin {
            observed_depth,
            margin,
        } => {
            if depth == 0
                || usize::from(depth) > DERIVED_MAX_DRAFTS
                || observed_depth == 0
                || usize::from(observed_depth) > DERIVED_MAX_DRAFTS
            {
                return None;
            }
            let (depth_s, depth_n) = match query_native_parent(card, mode, depth) {
                Ok(Some(parent)) => parent,
                Ok(None) if legacy => {
                    let frozen = fn_phase1_counts(card, source, depth, DerivedFeature::Native);
                    if frozen.depth_n == 0 {
                        return None;
                    }
                    (frozen.depth_s, frozen.depth_n)
                }
                Ok(None) | Err(()) => return None,
            };
            match query_native_margin(card, mode, observed_depth, margin, depth) {
                Ok(Some((cell_s, cell_n))) => Some(CalibrationCounts {
                    depth_s,
                    depth_n,
                    cell_s,
                    cell_n,
                }),
                Ok(None) | Err(()) => None,
            }
        }
        DerivedFeature::Ngram { .. } => None,
    }
}

/// Numeric card/mode bytes for index queries.
fn native_query_key(card: DerivedCard, mode: DerivedCostMode) -> (u8, u8) {
    (
        match card {
            DerivedCard::R9700 => 0,
            DerivedCard::Halo => 1,
        },
        match mode {
            DerivedCostMode::OffPool => 0,
            DerivedCostMode::OnCopy => 1,
        },
    )
}

/// Validated parent lookup through the once-built index: `Ok(Some)` measured
/// support, `Ok(None)` genuinely absent (OffPool fallback), `Err(())`
/// ambiguous/invalid (never falls back). Fallback semantics are exactly the
/// slice validators'.
fn query_native_parent(
    card: DerivedCard,
    mode: DerivedCostMode,
    depth: u8,
) -> Result<Option<(u64, u64)>, ()> {
    if depth == 0 || usize::from(depth) > DERIVED_MAX_DRAFTS {
        return Err(());
    }
    let (card, mode) = native_query_key(card, mode);
    match query_native_counts_index(
        native_counts_index(),
        card,
        mode,
        depth,
        NATIVE_MARGIN_BIN_COUNT as u8,
        depth,
    ) {
        NativeCountsOutcome::Counts(s, n) => Ok(Some((s, n))),
        // Healthy slot + missing depth = `NoSupport`, which the slice law
        // reads as `Err` (identity exists, depth unmeasured: never falls
        // back); absent slot = `Absent`, which reads as `Ok(None)` (OffPool
        // fallback). `query_native_counts_index` already separates those.
        NativeCountsOutcome::Absent => Ok(None),
        NativeCountsOutcome::NoSupport | NativeCountsOutcome::Invalid => Err(()),
    }
}

/// Validated margin-cell lookup through the once-built index: `Ok(Some)`
/// measured cell, `Ok(None)`/`Err(())` unsupported (the caller maps both to
/// `None`, keeping unconditional evidence). Mirrors
/// `fn_native_margin_counts`.
fn query_native_margin(
    card: DerivedCard,
    mode: DerivedCostMode,
    observed_depth: u8,
    margin: NativeMarginBin,
    target_depth: u8,
) -> Result<Option<(u64, u64)>, ()> {
    if observed_depth == 0
        || usize::from(observed_depth) > DERIVED_MAX_DRAFTS
        || target_depth == 0
        || usize::from(target_depth) > DERIVED_MAX_DRAFTS
    {
        return Err(());
    }
    let (card, mode) = native_query_key(card, mode);
    match query_native_counts_index(
        native_counts_index(),
        card,
        mode,
        observed_depth,
        margin.index() as u8,
        target_depth,
    ) {
        NativeCountsOutcome::Counts(s, n) => Ok(Some((s, n))),
        // Unsupported cell or unobserved row: unconditional fallback. The
        // slice validator distinguishes "observed row, missing cell" from
        // "row never observed" only to return `Ok(None)` for both; the index
        // reports both as `NoSupport`, which maps identically here.
        NativeCountsOutcome::Absent | NativeCountsOutcome::NoSupport => Ok(None),
        NativeCountsOutcome::Invalid => Err(()),
    }
}
/// [`fn_measured_load_identity`]: flat `BTreeMap<String, String>` fields
/// serialized so loader Rust and exporter Python `canonical(sort_keys,
/// separators)` are byte-identical.
pub const MEASURED_LOAD_SCHEMA: &str = "qwen4-measured-load.v1";

/// Selects the exact ONE admitted measured-load identity for `card` + `mode`:
/// the single full production+mode identity SHA present among measured
/// `WINDOW_PRICES` rows for that card/mode, with exactly one corresponding
/// `LOAD_IDENTITIES` canonical descriptor. Missing (no admitted identity or
/// no descriptor), ambiguous (more than one identity or more than one
/// descriptor for the admitted identity), or malformed (empty identity,
/// empty/invalid descriptor) is `None`: the caller stays on the real
/// incumbent, never an invented identity. No guessing across cards/modes.
/// In addition to the descriptor, admission requires an actual valid native
/// price and routed AR price in one bucket, supported execution/reference
/// routes, and exact-identity native parent support through that width.
/// This bind-time check does not parse or allocate per window.
pub fn fn_measured_load_identity(card: DerivedCard, mode: DerivedCostMode) -> Option<&'static str> {
    let key = fn_width_key(card, mode, ContextBucket::Le2K);
    measured_book_index()
        .qualified_load_descriptor(slot_of_word(key.0, key.1), native_counts_index())
}

/// Strict shape check of a `qwen4-measured-load.v1` canonical descriptor:
/// flat JSON object, all string values, exact schema marker. Full field
/// completeness is the exporter's provenance duty; this only refuses empty
/// or structurally invalid descriptors at lookup time.
fn measured_load_descriptor_valid(descriptor: &str) -> bool {
    let value: serde_json::Value = match serde_json::from_str(descriptor) {
        Ok(value) => value,
        Err(_) => return false,
    };
    match value.as_object() {
        Some(map) if !map.is_empty() => (),
        _ => return false,
    }
    let map = value.as_object().expect("checked object");
    if map.values().any(|value| value.as_str().is_none()) {
        return false;
    }
    map.get("schema").and_then(|value| value.as_str()) == Some(MEASURED_LOAD_SCHEMA)
}
/// Card/mode slot shared by the load/reference admission passes and the hot
/// load/reference lookups: `R9700/Halo x OffPool/OnCopy` is `0..4`. Unknown
/// words have no slot (index `4` reads absent), matching the slice predicates.
fn slot_of_word(card: &str, mode: &str) -> usize {
    let card = match card {
        "R9700" => 0usize,
        "Halo" => 1usize,
        _ => return 4,
    };
    let mode = match mode {
        "OffPool" => 0usize,
        "OnCopy" => 1usize,
        _ => return 4,
    };
    card * 2 + mode
}
/// Whether the measured profile behind the exact ONE admitted identity for
/// `card` + `mode` (see [`fn_measured_load_identity`]) supports the caller's
/// current AR reference mode for a `source` decision. `recorded == false` is
/// an eager AR launch (`Eager`); `true` is a retained one (`Recorded`): the
/// `arithmetic_reference` canonical string of every queried route of the
/// admitted identity must equal the matching scalar. The multi-row verify is
/// still eager HIP either way, so `execution.verify.eager` is always true and
/// both pre-window `replay_recording`/`graph_capture_mode` leaves are false;
/// only the AR leg's `eager` follows the query (`!recorded`). The queried
/// price routes are the `source`'s own `batched` route (Mtp or Ngram rows of
/// the admitted identity) plus the associated `target-only` AR route of the
/// same profile identity: a price is only valid under the reference it was
/// measured with, and a fallback/record transition must never silently reuse
/// the other reference's price. An unrelated source's `batched` route never
/// rejects a supported decision: only the queried source's route and the AR
/// route gate. Unknown, missing, ambiguous, or mismatched applicability is
/// `false`. Static string comparisons only: no formatting or JSON parsing per
/// window (the candidate scalars are compared against the stored canonical
/// route-field values).
pub fn fn_measured_reference_supported(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    recorded: bool,
) -> bool {
    let key = fn_width_key(card, mode, ContextBucket::Le2K);
    let source_num = match source {
        DerivedSource::Mtp => 0,
        DerivedSource::Ngram => 1,
    };
    // Fixed-slot decision computed once at index build from the admitted
    // identity's own `batched` source route plus its `target-only` AR route.
    measured_book_index().reference_supported(slot_of_word(key.0, key.1), source_num, recorded)
}

type FnWidthPriceRow = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    bool,
    usize,
    f64,
    usize,
    usize,
    bool,
);
type FnWidthRouteField = (&'static str, &'static str, &'static str, &'static str);
type FnWidthDepthRow = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    bool,
    usize,
    u64,
    u64,
);

fn fn_width_key(
    card: DerivedCard,
    mode: DerivedCostMode,
    bucket: ContextBucket,
) -> (&'static str, &'static str, &'static str) {
    (
        match card {
            DerivedCard::R9700 => "R9700",
            DerivedCard::Halo => "Halo",
        },
        match mode {
            DerivedCostMode::OffPool => "OffPool",
            DerivedCostMode::OnCopy => "OnCopy",
        },
        match bucket {
            ContextBucket::Le2K => "Le2K",
            ContextBucket::Le8K => "Le8K",
            ContextBucket::Le16K => "Le16K",
            ContextBucket::Over16K => "Over16K",
        },
    )
}
/// Borrowed-book price row: `(&identity, &card, &mode, &bucket, &route,
/// &source, confirmation, k, total, support, qualified, pooled)`. The
/// shipping tables use `&'static str` literals (borrowed free); the
/// `cfg(test)` oracles use short-lived slices with the same layout.
type MeasuredPriceRow<'a> = (
    &'a str,
    &'a str,
    &'a str,
    &'a str,
    &'a str,
    &'a str,
    bool,
    usize,
    f64,
    usize,
    usize,
    bool,
);
/// Opaque measured price key: `(card, mode, bucket, source, confirmation, k)`.
/// Borrowed strings throughout: the shipping tables lend `&'static str`, the
/// `cfg(test)` oracles lend short-lived slices with the same layout. Unknown
/// words stay opaque (absence reads `Ok(None)`); only the `(source, k,
/// confirmation)` shape is validated, exactly like the slice validator's
/// leading `expected_kind` match.
type MeasuredPriceKey<'a> = (&'a str, &'a str, &'a str, &'a str, bool, usize);
/// Borrowed route-field row: `(&identity, &route, &path, &scalar)`.
type MeasuredRouteField<'a> = (&'a str, &'a str, &'a str, &'a str);
/// Sorted kind rows of one [`build_measured_book_index`] build:
/// `((identity, route), kind scalar)`. Sorted once; each price cell counts its
/// `(identity, route)` run, so exactly one `kind` row equal to the expected
/// scalar passes and duplicates fail, mirroring the slice validator.
type MeasuredKindRow<'a> = ((&'a str, &'a str), &'a str);
struct MeasuredBookIndex<'a> {
    /// Sorted measured cells, keyed by full opaque
    /// [`MeasuredPriceKey`]. Every other key resolves exactly like the slice
    /// validator: `Ok(Some)` when a valid cell names it, else `Ok(None)`
    /// (OffPool falls back). Unrelated identities in other cells never
    /// poison this key: grouping is per key, not per slot.
    prices: Vec<(MeasuredPriceKey<'a>, (f64, bool))>,
    /// Sorted ambiguous keys: present but invalid cells that reject (`Err`)
    /// exactly like the slice validator's `found`/validity arms, while
    /// sibling keys still resolve. Sorted once at build; hot queries
    /// binary-search it alongside [`prices`]. No per-window allocation.
    poisoned: Vec<MeasuredPriceKey<'a>>,
    /// Per card+mode slot (same layout as [`NativeSlotHealth`] slots): the one
    /// admitted identity's canonical load descriptor. Filled once: exactly one
    /// price identity plus exactly one valid descriptor for it. Anything else
    /// (missing/empty/second identity, missing/second/empty/invalid
    /// descriptor) leaves the slot empty, which callers read as incumbent.
    loads: [Option<&'a str>; 4],
    /// Per card+mode slot: whether [`fn_measured_reference_supported`] may pass
    /// under the eager (`[slot][source][0]`) or recorded (`[slot][source][1]`)
    /// query for `Mtp/Ngram` (`[slot][0/1]`). Set once from the same validity
    /// rules as the slice predicate; malformed books leave every entry false.
    references: [[[bool; 2]; 2]; 4],
    /// Per card+mode slot identity health for load/reference admission:
    /// `Absent` = no row names this card+mode; `Healthy` = exactly one
    /// non-empty identity; `Poisoned` = two/empty identities (loads read
    /// `None`, references read `false`). Prices never consult this: per-cell
    /// ambiguity lives in the sorted [`poisoned`] key set above.
    health: [NativeSlotHealth; 4],
    /// Retained from the load admission pass for exact count/price binding.
    identities: [Option<&'a str>; 4],
}

impl<'a> MeasuredBookIndex<'a> {
    fn qualified_load_descriptor(
        &self,
        slot: usize,
        counts: &NativeCountsIndex,
    ) -> Option<&'a str> {
        let descriptor = self.load_descriptor(slot)?;
        if counts.health.get(slot) != Some(&NativeSlotHealth::Healthy)
            || self.identities.get(slot).copied().flatten()
                != counts.identities.get(slot).copied().flatten()
            || !(self.reference_supported(slot, 0, false)
                || self.reference_supported(slot, 0, true))
        {
            return None;
        }
        let supported = self.prices.iter().any(|&(key, _)| {
            slot_of_word(key.0, key.1) == slot
                && key.3 == "Mtp"
                && query_measured_price(self, key.0, key.1, key.2, "ArRouted", false, 0)
                    .is_ok_and(|price| price.is_some())
                && (1..=key.5).all(|depth| {
                    matches!(
                        query_native_counts_index(
                            counts,
                            (slot / 2) as u8,
                            (slot % 2) as u8,
                            depth as u8,
                            NATIVE_MARGIN_BIN_COUNT as u8,
                            depth as u8,
                        ),
                        NativeCountsOutcome::Counts(_, _)
                    )
                })
        });
        supported.then_some(descriptor)
    }

    /// Production price resolution, including the identity-local fallback law.
    fn active_price(
        &self,
        key: MeasuredPriceKey<'_>,
        frozen: impl FnOnce() -> Option<(f64, bool)>,
    ) -> Option<(f64, bool)> {
        if !self.price_slot_supported(key.0, key.1) {
            return None;
        }
        match query_measured_price(
            self,
            key.0,
            key.1,
            key.2,
            key.3,
            key.4,
            key.5.try_into().ok()?,
        ) {
            Ok(Some(price)) => Some(price),
            Ok(None) if self.legacy_fallback(key.0, key.1) => frozen(),
            Ok(None) | Err(()) => None,
        }
    }

    /// Legacy calibration has no generic arithmetic identity. It may only
    /// serve the wholly absent OffPool slot, never a partial or poisoned book.
    fn legacy_fallback(&self, card: &str, mode: &str) -> bool {
        mode == "OffPool"
            && self.health.get(slot_of_word(card, mode)) == Some(&NativeSlotHealth::Absent)
    }

    fn price_slot_supported(&self, card: &str, mode: &str) -> bool {
        self.legacy_fallback(card, mode) || self.load_descriptor(slot_of_word(card, mode)).is_some()
    }

    fn counts_identity_supported(&self, card: &str, mode: &str, identity: Option<&str>) -> bool {
        if self.legacy_fallback(card, mode) {
            return identity.is_none();
        }
        let slot = slot_of_word(card, mode);
        self.load_descriptor(slot).is_some()
            && identity.is_some()
            && self.identities.get(slot).copied().flatten() == identity
    }

    /// Fixed-slot load descriptor lookup: no scan, no JSON work per window.
    fn load_descriptor(&self, slot: usize) -> Option<&'a str> {
        match self.loads.get(slot) {
            Some(slot) => *slot,
            None => None,
        }
    }

    /// Fixed-slot reference decision: no slice or JSON work per window.
    fn reference_supported(&self, slot: usize, source: usize, recorded: bool) -> bool {
        match self.references.get(slot) {
            Some(slot) => *slot
                .get(source)
                .and_then(|sources| sources.get(usize::from(recorded)))
                .unwrap_or(&false),
            None => false,
        }
    }
}
/// Shape gate shared by the builder and the hot price query: the slice
/// validator's leading `expected_kind` match. `Mtp` is unconfirmed `k >= 1`
/// only, `Ngram` is `k >= 1` either way, the AR cell is unconfirmed `k == 0`
/// only; everything else is invalid before any row is consulted.
fn measured_shape_valid(source: &str, k: usize, confirmation: bool) -> bool {
    matches!(
        (source, k, confirmation),
        ("ArRouted", 0, false) | ("Mtp", 1.., false) | ("Ngram", 1.., _)
    )
}

/// Expected route `kind` scalar for a shape-valid query, mirroring the slice
/// validator's leading match: the AR cell wants `"target-only"`, drafted
/// sources want `"batched"`.
fn measured_expected_kind(source: &str) -> &'static str {
    match source {
        "ArRouted" => "\"target-only\"",
        _ => "\"batched\"",
    }
}
/// `cfg(test)` oracle mirrors of the three shipping slice predicates, reused
/// by the index-vs-old equality tests on arbitrary (non-shipping) slices.
/// They duplicate the `fn_measured_*` validation laws exactly: one non-empty
/// price identity per card+mode slot; exactly one valid canonical descriptor
/// for that identity; per queried cell the slice validity; the reference
/// route law (single `batched` source route plus a distinct single
/// `target-only` AR route with the recorded/eager scalars and full leg mask).
/// Unknown card/mode/bucket/source words never form cells, matching the
/// validators' non-matching comparisons.
fn measured_oracle_load_identity<'a>(
    prices: &[MeasuredPriceRow<'a>],
    loads: &[(&'a str, &'a str)],
    card: &str,
    mode: &str,
) -> Option<&'a str> {
    let mut identity: Option<&'a str> = None;
    for row in prices {
        if row.1 != card || row.2 != mode {
            continue;
        }
        if row.0.is_empty() {
            return None;
        }
        match identity {
            None => identity = Some(row.0),
            Some(held) if held == row.0 => {}
            Some(_) => return None,
        }
    }
    let identity = identity?;
    let mut descriptor: Option<&'a str> = None;
    let mut seen = false;
    for row in loads {
        if row.0 != identity {
            continue;
        }
        if row.1.is_empty() || !measured_load_descriptor_valid(row.1) {
            return None;
        }
        if seen {
            return None;
        }
        seen = true;
        descriptor = Some(row.1);
    }
    descriptor
}

/// Oracle mirror of the shipping reference predicate over arbitrary slices.
#[cfg(test)]
fn measured_oracle_reference_supported(
    prices: &[MeasuredPriceRow<'_>],
    routes: &[MeasuredRouteField<'_>],
    card: &str,
    mode: &str,
    source: &str,
    recorded: bool,
) -> bool {
    let source_key = match source {
        "Mtp" | "Ngram" => source,
        _ => return false,
    };
    let mut identity: Option<&str> = None;
    let mut source_seen = false;
    let mut ar_seen = false;
    for row in prices {
        if row.1 != card || row.2 != mode {
            continue;
        }
        if row.0.is_empty() {
            return false;
        }
        match identity {
            None => identity = Some(row.0),
            Some(held) if held == row.0 => {}
            Some(_) => return false,
        }
        if row.5 == source_key {
            source_seen = true;
        } else if row.5 == "ArRouted" {
            ar_seen = true;
        }
    }
    let Some(identity) = identity else {
        return false;
    };
    if !source_seen || !ar_seen {
        return false;
    }
    let (want_reference, want_ar_eager) = match recorded {
        false => ("\"Eager\"", "true"),
        true => ("\"Recorded\"", "false"),
    };
    let mut seen_pairs: Vec<(&str, &str)> = Vec::new();
    for row in prices {
        if row.0 != identity || row.1 != card || row.2 != mode {
            continue;
        }
        if row.5 == source_key || row.5 == "ArRouted" {
            if !seen_pairs.contains(&(row.4, row.5)) {
                seen_pairs.push((row.4, row.5));
            }
        }
    }
    let mut batched_route = None;
    let mut ar_route = None;
    for (route, row_source) in &seen_pairs {
        let mut kind = None;
        for field in routes {
            if field.0 == identity && field.1 == *route && field.2 == "kind" {
                if kind.replace(field.3).is_some() {
                    return false;
                }
            }
        }
        match (kind, *row_source) {
            (Some("\"batched\""), s) if s == source_key => {
                if batched_route.replace(route).is_some() {
                    return false;
                }
            }
            (Some("\"target-only\""), "ArRouted") => {
                if ar_route.replace(route).is_some() {
                    return false;
                }
            }
            _ => return false,
        }
    }
    match (batched_route, ar_route) {
        (Some(batched), Some(ar)) if batched != ar => {
            measured_routes_supported(identity, routes, batched, ar, want_reference, want_ar_eager)
        }
        _ => false,
    }
}
/// Builds the once-validated immutable measured book from the shipping
/// literals. Price semantics reuse the `fn_width_price` slice predicate
/// exactly, per opaque [`MeasuredPriceKey`]: the query validates
/// `(source, k, confirmation)` first, then binary-searches the full borrowed
/// tuple `(card, mode, bucket, source, confirmation, k)` — all string queries,
/// no numeric domain/slot health tricks, no weird unknown-bucket fallback.
/// Price rows sort once by full key; duplicate keys `Err` (as the old
/// duplicate-cell arm). Single rows replay the old checks: non-empty
/// identity/route, finite positive total, support `> 0`, qualified rows cover
/// `k`, not pooled, exactly one route `kind` row equal to the expected
/// `batched`/`target-only` scalar. The sorted kind index builds once from
/// route fields (`kind` only); duplicates are invalid and irrelevant fields
/// are ignored, mirroring the old price validator. Absence is `None`. No
/// global identity poison for prices: an unrelated identity in a different
/// cell never poisons this price; only load/ref admission uses the global
/// unique identity per card+mode slot. The load/reference `health` stays
/// slot-cached as before; `NativeCounts` is untouched. Complexity
/// `O(sort rows + sort route kinds + N log M)`, not `N^2`.
fn build_measured_book_index<'a>(
    prices: &[MeasuredPriceRow<'a>],
    routes: &[MeasuredRouteField<'a>],
    loads: &[(&'a str, &'a str)],
) -> MeasuredBookIndex<'a> {
    let mut kinds: Vec<MeasuredKindRow<'a>> = Vec::new();
    for field in routes {
        if field.2 == "kind" {
            kinds.push(((field.0, field.1), field.3));
        }
    }
    kinds.sort();
    let mut order: Vec<usize> = (0..prices.len()).collect();
    order.sort_by(|&l, &r| {
        let l = prices[l];
        let r = prices[r];
        (l.1, l.2, l.3, l.5, l.6, l.7).cmp(&(r.1, r.2, r.3, r.5, r.6, r.7))
    });
    let mut scoped: Vec<(MeasuredPriceKey<'a>, (f64, bool))> = Vec::new();
    let mut poisoned: Vec<MeasuredPriceKey<'a>> = Vec::new();
    let mut at = 0;
    while at < order.len() {
        let mut end = at + 1;
        while end < order.len() {
            let l = prices[order[at]];
            let r = prices[order[end]];
            if (l.1, l.2, l.3, l.5, l.6, l.7) != (r.1, r.2, r.3, r.5, r.6, r.7) {
                break;
            }
            end += 1;
        }
        let first = prices[order[at]];
        let key = (first.1, first.2, first.3, first.5, first.6, first.7);
        if end - at > 1 {
            poisoned.push(key);
        } else if !measured_shape_valid(first.5, first.7, first.6) {
            // Shape-invalid rows are unpriced: the slice validator ignores
            // them, so they poison nothing.
        } else if first.0.is_empty()
            || first.4.is_empty()
            || !first.8.is_finite()
            || first.8 <= 0.0
            || first.9 == 0
            || first.10 <= first.7
            || first.11
        {
            poisoned.push(key);
        } else {
            let want = measured_expected_kind(first.5);
            let pair = (first.0, first.4);
            let run = kinds.partition_point(|entry| entry.0 < pair)
                ..kinds.partition_point(|entry| entry.0 <= pair);
            let mut seen: Option<&'a str> = None;
            let mut count = 0usize;
            for entry in &kinds[run] {
                if count == 0 {
                    seen = Some(entry.1);
                }
                count += 1;
            }
            if count == 1 && seen == Some(want) {
                scoped.push((key, (first.8, first.11)));
            } else {
                poisoned.push(key);
            }
        }
        at = end;
    }
    scoped.sort_by(|l, r| l.0.cmp(&r.0));
    poisoned.sort();
    poisoned.dedup();
    let mut index = MeasuredBookIndex {
        prices: scoped,
        poisoned,
        loads: [None; 4],
        references: [[[false; 2]; 2]; 4],
        health: [NativeSlotHealth::Absent; 4],
        identities: [None; 4],
    };
    let mut identities: [Option<&'a str>; 4] = [None; 4];
    let mut identity_poisoned = [false; 4];
    // Price-identity pass (load/ref admission only): one non-empty identity
    // per card+mode slot; a second or empty identity poisons that slot's
    // load/ref decision, never prices.
    for row in prices {
        let slot = slot_of_word(row.1, row.2);
        if slot >= 4 {
            continue;
        }
        if row.0.is_empty() || identities[slot].is_some_and(|held| held != row.0) {
            identity_poisoned[slot] = true;
            continue;
        }
        identities[slot] = Some(row.0);
    }
    // Load pass: exactly one valid canonical descriptor for the admitted
    // identity. Parsed once here; hot lookups borrow the stored `&str`. A
    // descriptor failure only clears this slot's load decision: the price and
    // reference laws never consult `LOAD_IDENTITIES`.
    for slot in 0..4 {
        if identity_poisoned[slot] {
            continue;
        }
        let Some(identity) = identities[slot] else {
            continue;
        };
        let mut descriptor: Option<&'a str> = None;
        let mut seen = false;
        for row in loads {
            if row.0 != identity {
                continue;
            }
            if row.1.is_empty() || !measured_load_descriptor_valid(row.1) {
                descriptor = None;
                break;
            }
            if seen {
                descriptor = None;
                break;
            }
            seen = true;
            descriptor = Some(row.1);
        }
        index.loads[slot] = descriptor;
    }
    // Reference pass: same predicate as
    // `fn_measured_reference_supported`'s slices, evaluated once.
    for slot in 0..4 {
        if identity_poisoned[slot] || identities[slot].is_none() {
            continue;
        }
        let identity = identities[slot].expect("checked identity");
        // The queried source must own a measured price row under the admitted
        // identity; the AR leg uses the same identity's `target-only` rows.
        let mut mtp_seen = false;
        let mut ngram_seen = false;
        let mut ar_seen = false;
        for row in prices {
            if slot_of_word(row.1, row.2) != slot || row.0 != identity {
                continue;
            }
            match row.5 {
                "Mtp" => mtp_seen = true,
                "Ngram" => ngram_seen = true,
                "ArRouted" => ar_seen = true,
                _ => {}
            }
        }
        if !ar_seen {
            continue;
        }
        // Route SHAs resolve through the admitted price rows, mirroring the
        // slice predicate: the queried source's `batched` route plus the
        // identity's `target-only` AR route. Unrelated sources are ignored.
        for (source_num, source_key, seen) in [(0u8, "Mtp", mtp_seen), (1u8, "Ngram", ngram_seen)] {
            if !seen {
                continue;
            }
            let mut batched_route: Option<&str> = None;
            let mut ar_route: Option<&str> = None;
            let mut routes_ok = true;
            let mut seen_pairs: Vec<(&str, &str)> = Vec::new();
            for row in prices {
                if slot_of_word(row.1, row.2) != slot || row.0 != identity {
                    continue;
                }
                if row.5 == source_key || row.5 == "ArRouted" {
                    if !seen_pairs.contains(&(row.4, row.5)) {
                        seen_pairs.push((row.4, row.5));
                    }
                }
            }
            for (route, row_source) in &seen_pairs {
                let pair = (identity, *route);
                let run = kinds.partition_point(|entry| entry.0 < pair)
                    ..kinds.partition_point(|entry| entry.0 <= pair);
                let mut kind: Option<&str> = None;
                let mut count = 0usize;
                for entry in &kinds[run] {
                    if count == 0 {
                        kind = Some(entry.1);
                    }
                    count += 1;
                    if count > 1 {
                        break;
                    }
                }
                if count != 1 {
                    routes_ok = false;
                    break;
                }
                match (kind, *row_source) {
                    (Some("\"batched\""), s) if s == source_key => {
                        if batched_route.replace(route).is_some() {
                            routes_ok = false;
                            break;
                        }
                    }
                    (Some("\"target-only\""), "ArRouted") => {
                        if ar_route.replace(route).is_some() {
                            routes_ok = false;
                            break;
                        }
                    }
                    _ => {
                        routes_ok = false;
                        break;
                    }
                }
            }
            let (Some(batched), Some(ar)) = (batched_route, ar_route) else {
                continue;
            };
            if batched == ar || !routes_ok {
                continue;
            }
            for (flag, want_reference, want_ar_eager) in [
                (false, "\"Eager\"", "true"),
                (true, "\"Recorded\"", "false"),
            ] {
                if measured_routes_supported(
                    identity,
                    routes,
                    batched,
                    ar,
                    want_reference,
                    want_ar_eager,
                ) {
                    index.references[slot][usize::from(source_num)][usize::from(flag)] = true;
                }
            }
        }
    }
    for slot in 0..4 {
        if identity_poisoned[slot] {
            index.health[slot] = NativeSlotHealth::Poisoned;
            index.loads[slot] = None;
            index.references[slot] = [[false; 2]; 2];
        } else if identities[slot].is_some() {
            index.health[slot] = NativeSlotHealth::Healthy;
        }
    }
    index.identities = identities;
    index
}
/// Shared route-shape check for the reference pass and the `cfg(test)`
/// oracle: both `batched` and `target-only` routes of one identity carry the
/// queried reference scalar, pre-window recording/capture leaves are false,
/// the AR leg's `eager` matches the query, and the verify leg stays eager.
/// Unknown extra fields are ignored exactly like the slice predicate.
fn measured_routes_supported(
    identity: &str,
    routes: &[MeasuredRouteField<'_>],
    batched: &str,
    ar: &str,
    want_reference: &str,
    want_ar_eager: &str,
) -> bool {
    fn route_ok(
        identity: &str,
        routes: &[MeasuredRouteField<'_>],
        route: &str,
        want_reference: &str,
        want_ar_eager: &str,
    ) -> bool {
        let mut kind_ok = false;
        let mut reference_ok = false;
        // Bits 0/1: ar `replay_recording`/`graph_capture_mode` seen false;
        // bit 2: ar `eager` seen matching the query; bits 3/4/5: the verify
        // leg seen false/false/true.
        let mut legs = 0u8;
        for field in routes {
            if field.0 != identity || field.1 != route {
                continue;
            }
            match field.2 {
                "kind" => {
                    if field.3 != "\"batched\"" && field.3 != "\"target-only\"" {
                        return false;
                    }
                    kind_ok = true;
                }
                "arithmetic_reference" => {
                    if field.3 != want_reference {
                        return false;
                    }
                    reference_ok = true;
                }
                "execution.ar.replay_recording" | "execution.ar.graph_capture_mode" => {
                    if field.3 != "false" {
                        return false;
                    }
                    legs |= 1 << u8::from(field.2.ends_with("graph_capture_mode"));
                }
                "execution.ar.eager" => {
                    if field.3 != want_ar_eager {
                        return false;
                    }
                    legs |= 1 << 2;
                }
                "execution.verify.replay_recording" | "execution.verify.graph_capture_mode" => {
                    if field.3 != "false" {
                        return false;
                    }
                    legs |= 1 << (3 + u8::from(field.2.ends_with("graph_capture_mode")));
                }
                "execution.verify.eager" => {
                    if field.3 != "true" {
                        return false;
                    }
                    legs |= 1 << 5;
                }
                _ => {}
            }
        }
        kind_ok && reference_ok && legs == 0b111_111
    }
    route_ok(identity, routes, batched, want_reference, want_ar_eager)
        && route_ok(identity, routes, ar, want_reference, want_ar_eager)
}

/// Immutable validated measured book, built once from the literals at first
/// use. Activation is at most once per process; every query after that is a
/// fixed slot read or binary search: no JSON parse, no table scan, no
/// per-window allocation. Bootstrap declares the typed empty tables only,
/// so every slot reads absent until a measured export lands (no bootstrap
/// numbers are production prices).
static MEASURED_BOOK_INDEX: std::sync::LazyLock<MeasuredBookIndex<'static>> =
    std::sync::LazyLock::new(|| {
        build_measured_book_index(
            fn_width_cost_v1::WINDOW_PRICES,
            fn_width_cost_v1::ROUTE_FIELDS,
            fn_width_cost_v1::LOAD_IDENTITIES,
        )
    });

/// The process-once [`MeasuredBookIndex`] over the shipping literals.
fn measured_book_index() -> &'static MeasuredBookIndex<'static> {
    &MEASURED_BOOK_INDEX
}

/// Validated price lookup through the once-built index: `Ok(Some)` measured
/// support, `Ok(None)` genuinely absent (OffPool fallback), `Err(())`
/// ambiguous/invalid (never falls back). Semantics are exactly the slice
/// validator's: the `(source, k, confirmation)` shape is validated first
/// (`k == 0` only for the unconfirmed AR cell, `Mtp` only unconfirmed,
/// `Ngram` either, all other shapes invalid), then the full opaque tuple
/// `(card, mode, bucket, source, confirmation, k)` binary-searches the sorted
/// prices and poisoned keys. Unknown words are opaque keys like any other:
/// no row names them (`Ok(None)`); a row naming them poisons exactly its key.
fn query_measured_price(
    index: &MeasuredBookIndex<'_>,
    card: &str,
    mode: &str,
    bucket: &str,
    source: &str,
    confirmation: bool,
    k: u8,
) -> Result<Option<(f64, bool)>, ()> {
    if !measured_shape_valid(source, usize::from(k), confirmation) {
        return Err(());
    }
    let key = (card, mode, bucket, source, confirmation, usize::from(k));
    if index.poisoned.binary_search(&key).is_ok() {
        return Err(());
    }
    match index.prices.binary_search_by(|entry| entry.0.cmp(&key)) {
        Ok(at) => Ok(Some(index.prices[at].1)),
        Err(_) => Ok(None),
    }
}

/// Slice-shaped adapter of the index price path for `cfg(test)` oracle
/// equality: string card/mode/bucket/source words in, slice-validator-shaped
/// `Ok/Err` out. Unknown words stay opaque keys: no row names them
/// (`Ok(None)`); shape-invalid queries (`Mtp` confirmed, `k == 0` outside the
/// unconfirmed AR cell, unknown shapes) read `Err(())` exactly like the
/// validator's leading `expected_kind` match, which rejects before rows.
fn measured_index_price(
    index: &MeasuredBookIndex<'_>,
    card: &str,
    mode: &str,
    bucket: &str,
    source: &str,
    k: u8,
    confirmation: bool,
) -> Result<Option<(f64, bool)>, ()> {
    query_measured_price(index, card, mode, bucket, source, confirmation, k)
}

/// Slice-shaped adapter of the index load-identity path for `cfg(test)`
/// oracle equality: the admission/validity law lives in the builder, this
/// only maps unknown words to the incumbent-preserving `None`.
#[cfg(test)]
fn measured_index_load_identity<'a>(
    index: &MeasuredBookIndex<'a>,
    card: &str,
    mode: &str,
) -> Option<&'a str> {
    let slot = slot_of_word(card, mode);
    if slot >= 4 {
        return None;
    }
    index.load_descriptor(slot)
}

/// Slice-shaped adapter of the index reference path for `cfg(test)` oracle
/// equality: unknown card/mode/source words map to `false`, matching the
/// slice predicate's unknown/missing `false`.
#[cfg(test)]
fn measured_index_reference_supported(
    index: &MeasuredBookIndex<'_>,
    card: &str,
    mode: &str,
    source: &str,
    recorded: bool,
) -> bool {
    let slot = slot_of_word(card, mode);
    if slot >= 4 {
        return false;
    }
    let source = match source {
        "Mtp" => 0usize,
        "Ngram" => 1usize,
        _ => return false,
    };
    index.reference_supported(slot, source, recorded)
}

// Pure slice validator behind the once-built index: kept as the `cfg(test)`
// oracle for index-vs-old equality coverage. The shipping lookup path never
// calls this (binary search only). Distinguishes a genuinely absent cell
// (eligible for explicit old OffPool fallback) from a present but
// ambiguous/unsupported cell (never eligible).
#[cfg(test)]
fn fn_width_price(
    rows: &[MeasuredPriceRow<'_>],
    routes: &[MeasuredRouteField<'_>],
    key: (&str, &str, &str),
    source: &str,
    k: usize,
    confirmation: bool,
) -> Result<Option<(f64, bool)>, ()> {
    let expected_kind = match (source, k, confirmation) {
        ("ArRouted", 0, false) => "\"target-only\"",
        ("Mtp", 1.., false) | ("Ngram", 1.., _) => "\"batched\"",
        _ => return Err(()),
    };
    let mut found = None;
    for row in rows {
        if (row.1, row.2, row.3) != key || row.5 != source || row.6 != confirmation || row.7 != k {
            continue;
        }
        if found.is_some()
            || row.0.is_empty()
            || row.4.is_empty()
            || !row.8.is_finite()
            || row.8 <= 0.0
            || row.9 == 0
            || row.10 <= k
            || row.11
        {
            return Err(());
        }
        let mut kinds = routes
            .iter()
            .filter(|field| field.0 == row.0 && field.1 == row.4 && field.2 == "kind");
        if kinds.next().map(|field| field.3) != Some(expected_kind) || kinds.next().is_some() {
            return Err(());
        }
        found = Some((row.8, row.11));
    }
    Ok(found)
}

fn fn_width_depth_counts(
    rows: &[FnWidthDepthRow],
    card: &str,
    mode: &str,
    depth: usize,
    expected_identity: Option<&str>,
) -> Option<CalibrationCounts> {
    if depth == 0 {
        return None;
    }
    let mut identity = Some(expected_identity?);
    let mut counts = CalibrationCounts::default();
    for (at, row) in rows.iter().enumerate() {
        if row.1 != card || row.2 != mode {
            continue;
        }
        if row.0.is_empty() || identity.is_some_and(|identity| identity != row.0) {
            return None;
        }
        identity = Some(row.0);
        if row.6 != depth {
            continue;
        }
        if row.8 == 0
            || row.7 > row.8
            || rows[..at].iter().any(|prior| {
                (
                    prior.0, prior.1, prior.2, prior.3, prior.4, prior.5, prior.6,
                ) == (row.0, row.1, row.2, row.3, row.4, row.5, row.6)
            })
        {
            return None;
        }
        counts.depth_s = counts.depth_s.checked_add(row.7)?;
        counts.depth_n = counts.depth_n.checked_add(row.8)?;
    }
    (counts.depth_n > 0).then_some(counts)
}
type FnNativeDepthRow = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    usize,
    u64,
    u64,
);
type FnNativeMarginRow = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    usize,
    usize,
    usize,
    u64,
    u64,
);

// Pure slice validators behind the once-built index: kept for authored
// `cfg(test)` equality coverage of the index against the existing literal
// slices. The shipping lookup path never calls these (binary search only).
#[cfg(test)]
fn fn_native_depth_counts(
    rows: &[FnNativeDepthRow],
    card: &str,
    mode: &str,
    depth: usize,
) -> Result<Option<(u64, u64)>, ()> {
    if depth == 0 || depth > DERIVED_MAX_DRAFTS {
        return Err(());
    }
    let mut identity: Option<&str> = None;
    let mut total = (0u64, 0u64);
    for (at, row) in rows.iter().enumerate() {
        if row.1 != card || row.2 != mode {
            continue;
        }
        if row.0.is_empty() || identity.is_some_and(|held| held != row.0) {
            return Err(());
        }
        identity = Some(row.0);
        if row.5 > DERIVED_MAX_DRAFTS
            || row.7 == 0
            || row.6 > row.7
            || rows[..at].iter().any(|prior| {
                (prior.0, prior.1, prior.2, prior.3, prior.4, prior.5)
                    == (row.0, row.1, row.2, row.3, row.4, row.5)
            })
        {
            return Err(());
        }
        if row.5 != depth {
            continue;
        }
        total.0 = total.0.checked_add(row.6).ok_or(())?;
        total.1 = total.1.checked_add(row.7).ok_or(())?;
    }
    match identity {
        // No row names this identity/mode at all: genuinely absent.
        None => Ok(None),
        // The identity exists but has no measured trial at this depth.
        Some(_) if total.1 == 0 => Err(()),
        Some(_) => Ok(Some(total)),
    }
}

#[cfg(test)]
fn fn_native_margin_counts(
    rows: &[FnNativeMarginRow],
    card: &str,
    mode: &str,
    observed_depth: usize,
    margin_bin: usize,
    target_depth: usize,
) -> Result<Option<(u64, u64)>, ()> {
    if observed_depth == 0
        || observed_depth > DERIVED_MAX_DRAFTS
        || margin_bin >= NATIVE_MARGIN_BIN_COUNT
        || target_depth == 0
        || target_depth > DERIVED_MAX_DRAFTS
    {
        return Err(());
    }
    let mut identity: Option<&str> = None;
    let mut observed = false;
    let mut total = (0u64, 0u64);
    for (at, row) in rows.iter().enumerate() {
        if row.1 != card || row.2 != mode {
            continue;
        }
        if row.0.is_empty() || identity.is_some_and(|held| held != row.0) {
            return Err(());
        }
        identity = Some(row.0);
        if row.5 == 0
            || row.5 > DERIVED_MAX_DRAFTS
            || row.6 >= NATIVE_MARGIN_BIN_COUNT
            || row.7 == 0
            || row.7 > DERIVED_MAX_DRAFTS
            || row.9 == 0
            || row.8 > row.9
            || rows[..at].iter().any(|prior| {
                (
                    prior.0, prior.1, prior.2, prior.3, prior.4, prior.5, prior.6, prior.7,
                ) == (row.0, row.1, row.2, row.3, row.4, row.5, row.6, row.7)
            })
        {
            return Err(());
        }
        if row.5 == observed_depth {
            observed = true;
        }
        if (row.5, row.6, row.7) != (observed_depth, margin_bin, target_depth) {
            continue;
        }
        total.0 = total.0.checked_add(row.8).ok_or(())?;
        total.1 = total.1.checked_add(row.9).ok_or(())?;
    }
    match identity {
        // No row names this identity/mode at all: genuinely absent.
        None => Ok(None),
        // The observed row (or this exact cell) was never measured: unsupported.
        Some(_) if !observed || total.1 == 0 => Ok(None),
        Some(_) => Ok(Some(total)),
    }
}
/// One validated measured native cell, aggregated over every pooled
/// context/bucket and route variant of its identity: `(card, mode,
/// observed_depth, margin_bin, target_depth)` with `margin_bin ==
/// [`NATIVE_MARGIN_BIN_COUNT`] for the unconditional parent.
type NativeCountsEntry = ((u8, u8, u8, u8, u8), (u64, u64));

/// Health of one card+mode slot in a [`NativeCountsIndex`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeSlotHealth {
    /// No literal row names this card+mode: genuinely absent.
    Absent,
    /// Exactly one identity, all rows valid: binary-searchable.
    Healthy,
    /// Second identity, empty identity, invalid row, duplicate key, or
    /// `u64` aggregation overflow: never eligible, never falls back.
    Poisoned,
}

/// Validated, aggregated, sorted measured native counts, built once from the
/// literals (see `NATIVE_COUNTS_INDEX`): binary-search queries, no per-window
/// scans or allocations.
struct NativeCountsIndex {
    parents: Vec<NativeCountsEntry>,
    margins: Vec<NativeCountsEntry>,
    health: [NativeSlotHealth; 4],
    /// The same identity validated with the pooled count rows.
    identities: [Option<&'static str>; 4],
}

/// Card/mode slot (R9700/Halo x OffPool/OnCopy) of a literal row's strings.
fn native_counts_slot(card: &str, mode: &str) -> Option<usize> {
    let card = match card {
        "R9700" => 0,
        "Halo" => 1,
        _ => return None,
    };
    let mode = match mode {
        "OffPool" => 0,
        "OnCopy" => 1,
        _ => return None,
    };
    Some(card * 2 + mode)
}

/// Validates the measured native literals once and aggregates them into
/// sorted numeric keys for binary-search queries. Pooling law is exactly the
/// slice validators' (`fn_native_depth_counts`/`fn_native_margin_counts`):
/// one identity per card+mode; a second or empty identity, an invalid row
/// (`successes > total`, `total == 0`, out-of-range depth/bin), a duplicate
/// full key `(identity, card, mode, bucket, route, depth[, observed, bin,
/// target])`, or a `u64` aggregation overflow poisons that card+mode, which
/// then reads `Invalid` (never the OffPool fallback). Distinct bucket/route
/// rows still pool into the numeric index. The hot numeric query key is
/// unchanged; full-key tracking lives only in build-time `BTreeSet`s dropped
/// after the build.
fn build_native_counts_index(
    parents: &[FnNativeDepthRow],
    margins: &[FnNativeMarginRow],
) -> NativeCountsIndex {
    let mut index = NativeCountsIndex {
        parents: Vec::new(),
        margins: Vec::new(),
        health: [NativeSlotHealth::Absent; 4],
        identities: [None; 4],
    };
    let mut identities: [Option<&'static str>; 4] = [None; 4];
    let mut poisoned = [false; 4];
    // Build-time-only full-key sets: duplicate full keys poison the slot
    // exactly like the slice validators, while distinct bucket/route rows
    // still aggregate into the numeric index. Dropped after the build; no
    // per-window cost.
    let mut parent_keys: std::collections::BTreeSet<(&str, &str, &str, &str, &str, usize)> =
        std::collections::BTreeSet::new();
    let mut margin_keys: std::collections::BTreeSet<(
        &str,
        &str,
        &str,
        &str,
        &str,
        usize,
        usize,
        usize,
    )> = std::collections::BTreeSet::new();
    // Validation pass: identity uniqueness, row validity, and full-key
    // uniqueness per slot.
    for row in parents {
        let Some(slot) = native_counts_slot(row.1, row.2) else {
            continue;
        };
        if row.0.is_empty() || identities[slot].is_some_and(|held| held != row.0) {
            poisoned[slot] = true;
            continue;
        }
        identities[slot] = Some(row.0);
        if row.5 == 0 || row.5 > DERIVED_MAX_DRAFTS || row.7 == 0 || row.6 > row.7 {
            poisoned[slot] = true;
            continue;
        }
        if !parent_keys.insert((row.0, row.1, row.2, row.3, row.4, row.5)) {
            poisoned[slot] = true;
        }
    }
    for row in margins {
        let Some(slot) = native_counts_slot(row.1, row.2) else {
            continue;
        };
        if row.0.is_empty() || identities[slot].is_some_and(|held| held != row.0) {
            poisoned[slot] = true;
            continue;
        }
        identities[slot] = Some(row.0);
        if row.5 == 0
            || row.5 > DERIVED_MAX_DRAFTS
            || row.6 >= NATIVE_MARGIN_BIN_COUNT
            || row.7 == 0
            || row.7 > DERIVED_MAX_DRAFTS
            || row.9 == 0
            || row.8 > row.9
        {
            poisoned[slot] = true;
            continue;
        }
        if !margin_keys.insert((row.0, row.1, row.2, row.3, row.4, row.5, row.6, row.7)) {
            poisoned[slot] = true;
        }
    }
    for slot in 0..4 {
        if poisoned[slot] {
            index.health[slot] = NativeSlotHealth::Poisoned;
        } else if identities[slot].is_some() {
            index.health[slot] = NativeSlotHealth::Healthy;
        }
    }
    // Aggregation pass over healthy slots only, in sorted key order.
    let add = |poisoned: &mut [bool; 4],
               entries: &mut Vec<NativeCountsEntry>,
               key: (u8, u8, u8, u8, u8),
               s: u64,
               n: u64,
               slot: usize| {
        match entries.binary_search_by(|entry| entry.0.cmp(&key)) {
            Ok(at) => {
                let (have_s, have_n) = entries[at].1;
                match (have_s.checked_add(s), have_n.checked_add(n)) {
                    (Some(s), Some(n)) => entries[at].1 = (s, n),
                    _ => poisoned[slot] = true,
                }
            }
            Err(at) => entries.insert(at, (key, (s, n))),
        }
    };
    for row in parents {
        let Some(slot) = native_counts_slot(row.1, row.2) else {
            continue;
        };
        if poisoned[slot] || index.health[slot] != NativeSlotHealth::Healthy {
            continue;
        }
        let (card, mode) = ((slot / 2) as u8, (slot % 2) as u8);
        let key = (
            card,
            mode,
            row.5 as u8,
            NATIVE_MARGIN_BIN_COUNT as u8,
            row.5 as u8,
        );
        add(&mut poisoned, &mut index.parents, key, row.6, row.7, slot);
    }
    for row in margins {
        let Some(slot) = native_counts_slot(row.1, row.2) else {
            continue;
        };
        if poisoned[slot] || index.health[slot] != NativeSlotHealth::Healthy {
            continue;
        }
        let (card, mode) = ((slot / 2) as u8, (slot % 2) as u8);
        let key = (card, mode, row.5 as u8, row.6 as u8, row.7 as u8);
        add(&mut poisoned, &mut index.margins, key, row.8, row.9, slot);
    }
    for slot in 0..4 {
        if poisoned[slot] {
            index.health[slot] = NativeSlotHealth::Poisoned;
        }
    }
    // A poisoned slot aggregates nothing observable: drop its entries so a
    // poisoned card+mode can never read healthy counts.
    for slot in 0..4 {
        if index.health[slot] == NativeSlotHealth::Poisoned {
            let (card, mode) = ((slot / 2) as u8, (slot % 2) as u8);
            index
                .parents
                .retain(|entry| (entry.0 .0, entry.0 .1) != (card, mode));
            index
                .margins
                .retain(|entry| (entry.0 .0, entry.0 .1) != (card, mode));
        }
    }
    index.identities = identities;
    index
}

/// Query outcome of [`query_native_counts_index`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeCountsOutcome {
    /// No row names this card+mode: genuinely absent (OffPool fallback).
    Absent,
    /// Ambiguous/invalid card+mode: never eligible for fallback.
    Invalid,
    /// Healthy card+mode with aggregated support at this key.
    Counts(u64, u64),
    /// Healthy card+mode without support at this key.
    NoSupport,
}

/// Binary-search query of a [`build_native_counts_index`] index. `margin_bin
/// == [`NATIVE_MARGIN_BIN_COUNT`]` selects the unconditional parent table,
/// else the margin table at `(observed_depth, margin_bin, target_depth)`.
/// No allocation, no scan: two binary searches at most.
fn query_native_counts_index(
    index: &NativeCountsIndex,
    card: u8,
    mode: u8,
    observed_depth: u8,
    margin_bin: u8,
    target_depth: u8,
) -> NativeCountsOutcome {
    let slot = usize::from(card) * 2 + usize::from(mode);
    if slot >= index.health.len() {
        return NativeCountsOutcome::Invalid;
    }
    match index.health[slot] {
        NativeSlotHealth::Absent => NativeCountsOutcome::Absent,
        NativeSlotHealth::Poisoned => NativeCountsOutcome::Invalid,
        NativeSlotHealth::Healthy => {
            let key = (card, mode, observed_depth, margin_bin, target_depth);
            let table = if margin_bin == NATIVE_MARGIN_BIN_COUNT as u8 {
                &index.parents
            } else {
                &index.margins
            };
            match table.binary_search_by(|entry| entry.0.cmp(&key)) {
                Ok(at) => NativeCountsOutcome::Counts(table[at].1 .0, table[at].1 .1),
                Err(_) => NativeCountsOutcome::NoSupport,
            }
        }
    }
}

/// Immutable validated aggregated measured native counts, built once from the
/// literals at first use: per-card+mode identity, pooled contexts/routes,
/// sorted numeric keys. Activation is at most once per process (a `&static`
/// source read plus heap build on first query); every query after that is
/// allocation-free binary search. Poisoned/absent slots keep the existing
/// fallback semantics (`Absent` = OffPool fallback, `Invalid` = never).
static NATIVE_COUNTS_INDEX: std::sync::LazyLock<NativeCountsIndex> =
    std::sync::LazyLock::new(|| {
        build_native_counts_index(
            fn_width_cost_v1::NATIVE_DEPTH_COUNTS,
            fn_width_cost_v1::NATIVE_MARGIN_COUNTS,
        )
    });

/// The process-once [`NativeCountsIndex`] over the shipping literals.
fn native_counts_index() -> &'static NativeCountsIndex {
    &NATIVE_COUNTS_INDEX
}

#[cfg(test)]
mod tests {
    use super::*;

    /// qwen4's Flash-Next window-cost table (gfx1151): a test prior, like
    /// every table this module is given.
    const WINDOW_COST: [f32; 8] = [1.0, 1.7, 1.87, 2.25, 2.55, 2.75, 3.0, 3.6];
    /// qwen4's singleton retirement threshold and AR probe-token count.
    const SINGLETON_MIN_WINDOWS: usize = 16;
    const SINGLETON_AR_PROBE_TOKENS: usize = 6;
    const TRACE_PREFIX: &str = "QWEN4_FLOOR_TRACE";
    /// qwen4's takeover table: `NGRAM_DEPTHS + 1` slots.
    type TestFloor = Floor<8, 64>;

    fn agreement(accepted: f32, total: f32) -> Agreement {
        [(accepted, total); MAX_DEPTH]
    }

    fn window(accepted: usize, drafts: usize) -> MtpWindow {
        MtpWindow {
            committed: vec![0; accepted + 1],
            accepted,
            drafts_generated: drafts,
        }
    }

    fn calibrated(ar_us: f32, agreement: &[(f32, f32)]) -> TestFloor {
        let mut floor = TestFloor::new(WINDOW_COST);
        floor.observe_calibration(ar_us, agreement);
        floor
    }

    /// `n` batched windows routed at and drafting `depth`, every draft
    /// accepted, `us` each.
    fn batched_windows(
        floor: &mut TestFloor,
        agreement: &[(f32, f32)],
        depth: usize,
        us: f32,
        n: usize,
    ) {
        for _ in 0..n {
            floor.observe_native(
                NativeWindow::Batched,
                &window(depth, depth),
                us,
                7,
                agreement,
            );
        }
    }

    /// `n` batched windows that drafted `depth` tokens but realized three
    /// committed tokens in 60 us (20 us per token), whatever the agreement
    /// predicts.
    fn batched_windows_realized(
        floor: &mut TestFloor,
        agreement: &[(f32, f32)],
        depth: usize,
        n: usize,
    ) {
        for _ in 0..n {
            floor.observe_native(NativeWindow::Batched, &window(2, depth), 60.0, 7, agreement);
        }
    }

    #[test]
    fn floor_low_acceptance_retires_within_the_probe_budget() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            assert!(!floor.retired);
            assert_eq!(floor.route(&bad, 3, 2), Action::Native(2));
            floor.observe_native(NativeWindow::Batched, &window(0, 2), 90.0, 3, &bad);
        }
        assert!(floor.retired);
        assert_eq!(floor.probes, FLOOR_PROBE_WINDOWS);
        assert_eq!(floor.route(&bad, 3, 2), Action::ArHeadFree);
    }

    #[test]
    fn floor_call_line_encodes_route_inputs_as_exact_bits() {
        let mut agreement = agreement(0.0, 2.0);
        agreement[0] = (1.8, 2.0);
        agreement[1] = (f32::MIN_POSITIVE, 1.0e-3);
        let mut floor = calibrated(30.0, &agreement);
        let ret = floor.route(&agreement, 3, 2);
        assert_eq!(ret, Action::Native(2));
        let fields = RouteFields {
            agreement: &agreement,
            k: 3,
            provisional: 2,
            ret,
            after: floor.after(),
        };
        let line = floor_call_line("QWEN4_FLOOR_CALL", 7, 4, "route", format_args!("{fields}"));
        let json = line
            .strip_prefix("QWEN4_FLOOR_CALL ")
            .expect("fixed prefix");
        assert!(!json.contains('\n'));
        let value: serde_json::Value = serde_json::from_str(json).expect("one compact object");
        assert_eq!(value["seq"], 7);
        assert_eq!(value["floor"], 4);
        assert_eq!(value["op"], "route");
        assert_eq!(value["k"], 3);
        assert_eq!(value["provisional"], 2);
        assert_eq!(value["ret"], "Batched:2");
        assert_eq!(value["retired_after"], false);
        assert_eq!(value["needs_ar_after"], false);
        let pairs = value["agreement"].as_array().expect("agreement pairs");
        assert_eq!(pairs.len(), MAX_DEPTH);
        let decode = |hex: &serde_json::Value| {
            let digits = hex.as_str().expect("hex string");
            f32::from_bits(u32::from_str_radix(digits, 16).expect("hex digits"))
        };
        for (pair, &(accepted, total)) in pairs.iter().zip(agreement.iter()) {
            assert_eq!(decode(&pair[0]).to_bits(), accepted.to_bits());
            assert_eq!(decode(&pair[1]).to_bits(), total.to_bits());
        }
        assert_eq!(pairs[0][0], format!("{:08x}", 1.8f32.to_bits()));
        assert_eq!(pairs[1][0], "00800000");
        // Bits, not decimal renderings: signed zero and NaN survive.
        assert_eq!(FloorBits(1.0).to_string(), "\"3f800000\"");
        assert_eq!(FloorBits(-0.0).to_string(), "\"80000000\"");
        assert_eq!(FloorBits(f32::NAN).to_string(), "\"7fc00000\"");
    }

    #[test]
    fn floor_winning_cost_depth_is_unchanged() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(33.0, &good);
        batched_windows(&mut floor, &good, 3, 75.0, FLOOR_PROBE_WINDOWS);
        assert!(!floor.retired);
        // The provisional (Halo-table) depth is ignored once measured. The
        // cost fit prices the unmeasured neighbour 4 at 75 + 0.27 x 33 us
        // (83.9 us for 4.10 predicted tokens, 20.5 us per token) against
        // depth 3's measured 21.8: the chooser explores one depth up, never
        // to the table's 5 or past the budget.
        assert_eq!(floor.route(&good, 7, 5), Action::Native(4));
        assert_eq!(floor.route(&good, 3, 5), Action::Native(3));
        // `k` below the measured winning depth: no eligible measured option,
        // so the request retires to AR rather than run an unmeasured route.
        assert_eq!(floor.route(&good, 2, 5), Action::ArHeadFree);
        assert!(floor.retired);
    }

    #[test]
    fn floor_tie_with_ar_retires() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        // All at one cap: a neighbour's shifted evidence would price it below
        // its own windows.
        for _ in 1..=FLOOR_PROBE_WINDOWS {
            floor.observe_native(NativeWindow::Batched, &window(0, 2), 30.0, 3, &bad);
        }
        assert!(floor.retired);
    }

    #[test]
    fn floor_true_ar_price_changes_choices_across_cards() {
        let good = agreement(1.8, 2.0);
        // Identical measured windows (75 us for 3.44 predicted tokens, 21.8
        // us per token); only the card's measured AR price differs.
        for (ar_us, retires) in [(33.0, false), (22.0, false), (21.0, true), (10.0, true)] {
            let mut floor = calibrated(ar_us, &good);
            batched_windows(&mut floor, &good, 3, 75.0, FLOOR_PROBE_WINDOWS);
            assert_eq!(floor.retired, retires, "ar_us {ar_us}");
        }
    }

    #[test]
    fn floor_interleaved_is_priced_per_emitted_token() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_native(NativeWindow::Interleaved, &window(1, 1), 50.0, 3, &bad);
        }
        assert!(!floor.retired);
        assert_eq!(floor.route(&bad, 3, 2), Action::Interleaved);
    }

    #[test]
    fn floor_probe_budget_is_bounded_even_for_unusable_timings() {
        let good = agreement(1.8, 2.0);
        let mut floor = TestFloor::new(WINDOW_COST);
        floor.observe_native(NativeWindow::Batched, &window(2, 2), 75.0, 3, &good);
        assert_eq!(floor.probes, 0, "uncalibrated windows are not observed");
        floor.observe_calibration(33.0, &good);
        floor.observe_native(NativeWindow::Batched, &window(0, 0), 75.0, 0, &good);
        assert_eq!(floor.probes, 0, "k == 0 is not a probe");
        // Unusable timings record nothing but spend the budget, and with
        // nothing measured the request retires at the budget.
        for us in [f32::NAN, f32::INFINITY, 0.0, -1.0, f32::NAN] {
            floor.observe_native(NativeWindow::Batched, &window(2, 2), us, 3, &good);
            assert!(floor.probes <= FLOOR_PROBE_WINDOWS);
        }
        assert_eq!(floor.probes, FLOOR_PROBE_WINDOWS);
        assert_eq!(floor.batched[2], (0.0, 0.0));
        assert!(floor.retired);

        let mut floor = calibrated(33.0, &good);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_external(0, f32::NAN, 4, &good);
        }
        assert_eq!(floor.probes, FLOOR_PROBE_WINDOWS);
        assert!(floor.retired);

        let mut floor = calibrated(33.0, &good);
        batched_windows(&mut floor, &good, 2, 60.0, 2);
        floor.observe_external(4, 60.0, 4, &good);
        assert_eq!(floor.probes, FLOOR_PROBE_WINDOWS);
        batched_windows(&mut floor, &good, 2, 60.0, 10);
        assert_eq!(floor.probes, FLOOR_PROBE_WINDOWS);
    }

    #[test]
    fn floor_probe_route_is_the_provisional_depth() {
        let good = agreement(1.8, 2.0);
        let mut floor = TestFloor::new(WINDOW_COST);
        for calibration in [false, true] {
            if calibration {
                floor.observe_calibration(33.0, &good);
            }
            for provisional in 0..=7 {
                let expected = match provisional {
                    0 => Action::Interleaved,
                    depth => Action::Native(depth),
                };
                assert_eq!(floor.route(&good, 7, provisional), expected);
            }
        }
    }

    #[test]
    fn floor_retirement_is_sticky() {
        let bad = agreement(0.0, 2.0);
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(30.0, &bad);
        batched_windows(&mut floor, &bad, 2, 90.0, FLOOR_PROBE_WINDOWS);
        assert!(floor.retired);
        batched_windows(&mut floor, &good, 3, 5.0, 10);
        assert!(floor.observe_external(8, 10.0, 8, &good));
        floor.observe_calibration(1000.0, &good);
        assert!(floor.retired);
        assert_eq!(floor.route(&good, 7, 3), Action::ArHeadFree);
    }

    #[test]
    fn floor_external_costs_are_separate_from_native_agreement() {
        let good = agreement(1.8, 2.0);
        let before = good;
        let mut floor = calibrated(30.0, &good);
        batched_windows(&mut floor, &good, 3, 75.0, 2);
        // 15 us per external token beats AR; native wins too.
        assert!(!floor.observe_external(4, 60.0, 4, &good));
        assert!(!floor.retired);
        assert_eq!(good, before);
        assert_eq!(floor.interleaved, (0.0, 0.0));
        // Once external windows lose, takeover stops; native keeps running.
        let mut stopped = false;
        for _ in 0..30 {
            if floor.observe_external(1, 200.0, 4, &good) {
                stopped = true;
                break;
            }
        }
        assert!(stopped && !floor.retired);
        assert_eq!(floor.route(&good, 3, 5), Action::Native(3));
    }

    #[test]
    fn floor_good_external_keeps_a_losing_native_route_alive_while_the_mixture_beats_ar() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        // 10 us per external token beats AR.
        assert!(!floor.observe_external(4, 40.0, 4, &bad));
        assert!(!floor.observe_external(4, 40.0, 4, &bad));
        // One losing native window (90 us for one token) spends the budget;
        // the realized mixture (~20.6 us per token) still beats AR.
        floor.observe_native(NativeWindow::Batched, &window(0, 1), 90.0, 7, &bad);
        assert_eq!(floor.probes, FLOOR_PROBE_WINDOWS);
        assert!(!floor.retired);
        // The measured native route loses to AR, yet it keeps running while
        // the mixture wins.
        assert_eq!(floor.route(&bad, 7, 5), Action::Native(1));
        assert!(!floor.retired);
        // Losing native windows drag the mixture up to AR; it then retires at
        // the first window where the mixture stops beating AR, stickily.
        let mut kept_alive = 0;
        while !floor.retired {
            assert_eq!(floor.route(&bad, 7, 5), Action::Native(1));
            floor.observe_native(NativeWindow::Batched, &window(0, 1), 90.0, 7, &bad);
            kept_alive += 1;
            assert!(kept_alive < 50, "never retired");
        }
        assert!(kept_alive > 0);
        assert!(floor.combined.0 / floor.combined.1 >= 30.0);
        assert_eq!(floor.route(&bad, 7, 5), Action::ArHeadFree);
        assert!(floor.retired);
        assert!(floor.observe_external(4, 40.0, 4, &bad));
        assert_eq!(floor.route(&bad, 7, 5), Action::ArHeadFree);
    }

    #[test]
    fn floor_three_good_external_hits_do_not_retire_and_route_interleaved() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(30.0, &good);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            assert!(!floor.observe_external(4, 60.0, 4, &good));
            assert!(!floor.retired);
        }
        assert_eq!(floor.probes, FLOOR_PROBE_WINDOWS);
        // More good hits with zero native observations still keep it alive.
        assert!(!floor.observe_external(4, 60.0, 4, &good));
        assert!(!floor.retired);
        // No native option is measured: the next native window runs the
        // interleaved route rather than retiring.
        assert_eq!(floor.route(&good, 7, 5), Action::Interleaved);
        assert!(!floor.retired);
    }

    #[test]
    fn floor_good_external_with_no_native_measured_within_k_routes_interleaved() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(30.0, &good);
        assert!(!floor.observe_external(4, 40.0, 4, &good));
        assert!(!floor.observe_external(4, 40.0, 4, &good));
        // Depth 3: 150 us over 3.44 predicted tokens loses to AR, but the
        // realized mixture (~20 us per token) wins.
        floor.observe_native(NativeWindow::Batched, &window(3, 3), 150.0, 7, &good);
        assert!(!floor.retired);
        assert_eq!(floor.route(&good, 7, 5), Action::Native(3));
        // `k` below the only measured depth: nothing measured within k.
        assert_eq!(floor.route(&good, 2, 5), Action::Interleaved);
        assert!(!floor.retired);
    }

    #[test]
    fn floor_blocked_external_does_not_keep_a_losing_native_route_alive() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        // Native windows lose by prediction (one token per window) while
        // their realized cost (20 us per token) would win the mixture.
        batched_windows_realized(&mut floor, &bad, 1, 2);
        // One external window at exactly AR's price is blocked (a tie loses).
        assert!(floor.observe_external(1, 30.0, 4, &bad));
        assert!(floor.combined.0 / floor.combined.1 < 30.0);
        assert!(floor.retired);
        assert_eq!(floor.route(&bad, 7, 5), Action::ArHeadFree);
    }

    #[test]
    fn floor_combined_mixture_prices_every_valid_window_after_calibration() {
        let good = agreement(1.8, 2.0);
        let mut floor = TestFloor::new(WINDOW_COST);
        floor.observe_native(NativeWindow::Batched, &window(2, 2), 75.0, 3, &good);
        assert_eq!(
            floor.combined,
            (0.0, 0.0),
            "uncalibrated windows are not priced"
        );
        floor.observe_calibration(30.0, &good);
        floor.observe_native(NativeWindow::Batched, &window(2, 2), 75.0, 3, &good);
        assert_eq!(floor.combined, (75.0, 3.0));
        floor.observe_native(NativeWindow::Interleaved, &window(1, 1), 40.0, 3, &good);
        assert_eq!(
            floor.combined,
            (75.0 * DEFAULT_DECAY + 40.0, 3.0 * DEFAULT_DECAY + 2.0)
        );
        let before = floor.combined;
        floor.observe_native(NativeWindow::Batched, &window(1, 1), f32::NAN, 3, &good);
        assert_eq!(floor.combined, before, "invalid timings are not priced");
        floor.observe_external(4, 40.0, 4, &good);
        assert_eq!(
            floor.combined,
            (
                before.0 * DEFAULT_DECAY + 40.0,
                before.1 * DEFAULT_DECAY + 4.0
            )
        );
        assert_eq!(TestFloor::new(WINDOW_COST).combined, (0.0, 0.0));
    }

    #[test]
    fn floor_bad_external_retires_when_native_does_not_win() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        batched_windows(&mut floor, &bad, 1, 90.0, 2);
        assert!(floor.observe_external(1, 200.0, 4, &bad));
        assert!(floor.retired);
    }

    #[test]
    fn floor_blocked_external_leaves_winning_native_running() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(30.0, &good);
        batched_windows(&mut floor, &good, 3, 75.0, 2);
        assert!(floor.observe_external(1, 90.0, 2, &good));
        assert!(!floor.retired);
        assert_eq!(floor.route(&good, 3, 5), Action::Native(3));
    }

    #[test]
    fn floor_invalid_ar_price_retires_at_the_budget_not_before() {
        let good = agreement(1.8, 2.0);
        for bad_us in [f32::NAN, 0.0, -5.0] {
            let mut floor = calibrated(bad_us, &good);
            batched_windows(&mut floor, &good, 3, 75.0, FLOOR_PROBE_WINDOWS - 1);
            assert!(!floor.retired);
            batched_windows(&mut floor, &good, 3, 75.0, 1);
            assert!(floor.retired);
        }
    }

    #[test]
    fn floor_depth_is_keyed_by_drafts_actually_generated() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(33.0, &good);
        floor.observe_native(NativeWindow::Batched, &window(1, 1), 50.0, 3, &good);
        assert!(floor.batched[1].1 > 0.0);
        assert_eq!(floor.batched[3], (0.0, 0.0));
        let depth_one = floor.batched[1];
        floor.observe_native(NativeWindow::Batched, &window(2, 2), 60.0, 3, &good);
        assert_eq!(floor.batched[1], depth_one, "decay is entry-local");
    }

    #[test]
    fn floor_max_k_beyond_the_cost_table_is_clamped() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(100.0, &good);
        let top = WINDOW_COST.len() - 1;
        assert!(top < MAX_DEPTH);
        floor.observe_native(NativeWindow::Batched, &window(top, top), 100.0, 10, &good);
        floor.observe_native(NativeWindow::Batched, &window(top, top), 100.0, 10, &good);
        // A window longer than the admitted prefix has no cost entry yet: it
        // must not panic and does not create one.
        floor.observe_native(
            NativeWindow::Batched,
            &window(top + 1, top + 1),
            100.0,
            10,
            &good,
        );
        assert_eq!(floor.probes, FLOOR_PROBE_WINDOWS);
        assert_eq!(floor.batched.len(), WINDOW_COST.len());
        assert_eq!(floor.native_cap, WINDOW_COST.len() - 1);
        assert!(!floor.retired);
        // Legacy (non-extended) pricing stays at the table: within k = 10 the
        // choice is the measured top (100 us for 5.70 predicted tokens) or
        // its fitted neighbour below (100 - 0.27 x 100 = 73 us for 5.22),
        // the cheaper per token.
        assert_eq!(floor.route(&good, 10, 5), Action::Native(top - 1));
    }

    #[test]
    fn floor_extended_capacity_routes_finite_deep_optima() {
        // Full acceptance and a cheap row: window cost grows ~1 ms/row while
        // expected tokens grow a full token/row, so per-token cost strictly
        // falls with depth and the finite optimum is exactly the admitted
        // capacity (16, 32, 63). Shallow windows establish the fit; the route
        // then climbs past the historical prefix without measured deep
        // evidence, priced by the fit, never by a padded table.
        for optimum in [16usize, 32, 63] {
            let agreement = vec![(2.0, 2.0); optimum];
            let ar = 30_000.0;
            let mut floor = TestFloor::new(WINDOW_COST);
            floor.set_capacity(optimum);
            floor.observe_calibration(ar, &agreement);
            // Shallow windows with a cheap row (~1 ms/row against 30 ms AR).
            for depth in [1usize, 2, 3, 4, 5] {
                let us = 25_000.0 + 1_000.0 * depth as f32;
                for _ in 0..FLOOR_PROBE_WINDOWS {
                    floor.observe_native(
                        NativeWindow::Batched,
                        &window(depth, depth),
                        us,
                        optimum,
                        &agreement,
                    );
                }
            }
            assert!(floor.fit_line().is_some(), "optimum {optimum}");
            // No deep measurement, yet the finite optimum beats AR: the floor
            // stays alive and routes the capacity, not AR.
            assert!(!floor.retired, "optimum {optimum}");
            let route = floor.route(&agreement, optimum, 3);
            assert_eq!(route, Action::Native(optimum), "optimum {optimum}");
            // The qualified fit prices every admitted depth below AR without
            // claiming any deep measurement.
            assert!(!floor.retired, "optimum {optimum}");
        }
    }

    #[test]
    fn floor_wide_fit_wins_before_any_measured_native_cap_beats_ar() {
        let agreement = vec![(2.0, 2.0); 63];
        let mut floor = TestFloor::new(WINDOW_COST);
        floor.set_capacity(63);
        floor.observe_calibration(30_000.0, &agreement);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            // Every actually measured native window loses: 101 ms / 2
            // emitted tokens > the 30 ms AR token. Its existing fitted row
            // model nevertheless prices the admitted wide domain below AR.
            floor.observe_native(
                NativeWindow::Batched,
                &window(1, 1),
                101_000.0,
                63,
                &agreement,
            );
        }
        assert!(floor.cap_cost(&agreement, 1).unwrap() > floor.ar_us());
        assert!(floor.cap_cost(&agreement, 63).unwrap() < floor.ar_us());
        assert!(!floor.retired());
        assert_eq!(floor.route(&agreement, 63, 3), Action::Native(63));
    }

    #[test]
    fn floor_extended_capacity_is_bounded_and_zero_disables() {
        let agreement = vec![(1.9, 2.0); 63];
        let mut floor = TestFloor::new(WINDOW_COST);
        assert_eq!(floor.native_cap, WINDOW_COST.len() - 1);
        floor.set_capacity(0);
        assert_eq!(floor.native_cap, 0);
        floor.observe_calibration(30.0, &agreement);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_native(NativeWindow::Batched, &window(1, 1), 10.0, 0, &agreement);
        }
        // No batched option is eligible under capacity 0.
        assert_eq!(
            floor.best_candidate(&agreement, 63),
            floor.best_candidate(&agreement, 0)
        );
        // Clamping: past the physical executor width never admits more.
        floor.set_capacity(usize::MAX);
        assert_eq!(floor.native_cap, DERIVED_MAX_DRAFTS);
        // Re-setting the same value reallocates nothing and drops nothing.
        let batched_ptr = floor.extra_batched.as_ptr();
        let caps_ptr = floor.extra_caps.as_ptr();
        floor.observe_native(NativeWindow::Batched, &window(1, 1), 10.0, 63, &agreement);
        floor.set_capacity(DERIVED_MAX_DRAFTS);
        assert_eq!(floor.extra_batched.as_ptr(), batched_ptr);
        assert_eq!(floor.extra_caps.as_ptr(), caps_ptr);
    }
    #[test]
    fn floor_narrow_set_capacity_matches_legacy_pricing() {
        // `set_capacity(3)` on a legacy-shaped floor enforces the bound but
        // keeps the measured/neighbour pricing bit-arithmetic: same
        // diagnostics state and same route for the same feasible budget.
        let good = agreement(1.8, 2.0);
        let mut legacy = calibrated(33.0, &good);
        batched_windows(&mut legacy, &good, 3, 75.0, FLOOR_PROBE_WINDOWS);
        let mut narrow = calibrated(33.0, &good);
        narrow.set_capacity(3);
        batched_windows(&mut narrow, &good, 3, 75.0, FLOOR_PROBE_WINDOWS);
        assert_eq!(narrow.native_cap, 3);
        assert_eq!(narrow.diagnostics(), legacy.diagnostics());
        assert_eq!(narrow.route(&good, 7, 5), Action::Native(3));
        assert_eq!(narrow.route(&good, 3, 5), legacy.route(&good, 3, 5));
        assert_eq!(narrow.route(&good, 3, 5), Action::Native(3));
        assert_eq!(legacy.route(&good, 7, 5), Action::Native(4));
    }

    #[test]
    fn floor_extended_routing_requires_a_finite_fit() {
        // Extended but unmeasured: no fit (no priced windows) means no deep
        // invention; the shallow losing strip retires exactly like legacy.
        let bad = vec![(0.0, 2.0); 32];
        let mut floor = TestFloor::new(WINDOW_COST);
        floor.set_capacity(32);
        floor.observe_calibration(30.0, &bad);
        assert!(floor.fit_line().is_none());
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_native(NativeWindow::Batched, &window(0, 2), 90.0, 32, &bad);
        }
        assert!(floor.retired);
        assert_eq!(floor.route(&bad, 32, 2), Action::ArHeadFree);
        // Invalid AR with an extended capacity still retires at the budget.
        let mut floor = TestFloor::new(WINDOW_COST);
        floor.set_capacity(32);
        floor.observe_calibration(f32::NAN, &bad);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_native(NativeWindow::Batched, &window(0, 2), 90.0, 32, &bad);
        }
        assert!(floor.retired);
    }

    #[test]
    fn floor_reset_request_reuses_capacity_without_reallocation() {
        let agreement = vec![(1.9, 2.0); 32];
        let mut floor = TestFloor::new(WINDOW_COST);
        floor.set_capacity(32);
        floor.observe_calibration(30_000.0, &agreement);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_native(
                NativeWindow::Batched,
                &window(2, 2),
                60_000.0,
                32,
                &agreement,
            );
        }
        assert!(floor.diagnostics().batched[2].1 > 0.0);
        let batched_ptr = floor.extra_batched.as_ptr();
        let caps_ptr = floor.extra_caps.as_ptr();
        let (batched_cap, caps_cap) = (floor.extra_batched.capacity(), floor.extra_caps.capacity());
        let id = floor.id();
        let cap = floor.native_cap;
        floor.reset_request(FLOOR_PROBE_WINDOWS, DEFAULT_DECAY, 0);
        assert_eq!(floor.native_cap, cap);
        assert_eq!(floor.extra_batched.as_ptr(), batched_ptr);
        assert_eq!(floor.extra_caps.as_ptr(), caps_ptr);
        assert_eq!(floor.extra_batched.capacity(), batched_cap);
        assert_eq!(floor.extra_caps.capacity(), caps_cap);
        assert_eq!(floor.id(), id);
        assert_eq!(floor.diagnostics().batched[2], (0.0, 0.0));
        assert!(!floor.calibrated());
        // Capacity survives the reset: observations record without growth.
        floor.observe_calibration(30_000.0, &agreement);
        floor.observe_native(
            NativeWindow::Batched,
            &window(20, 20),
            60_000.0,
            32,
            &agreement,
        );
        assert_eq!(floor.extra_batched.as_ptr(), batched_ptr);
        assert_eq!(floor.extra_caps.as_ptr(), caps_ptr);
        assert!(floor.batched_entry(20).is_some());
    }

    /// A batched window routed at `cap` that drafted `drafts` and accepted
    /// `accepted` of them.
    fn capped(cap: usize, drafts: usize, accepted: usize) -> WindowStats {
        WindowStats {
            emitted: accepted + 1,
            drafts,
            cap,
        }
    }

    #[test]
    fn floor_prices_caps_by_their_drafted_mix_and_neighbours() {
        let good = agreement(1.6, 2.0);
        let ar = 28_600.0;
        // 0.97: the Flash-Next gfx1151 singleton decay (qwen4 `MTP_AGREEMENT_DECAY_GFX1151`).
        let mut floor = calibrated(ar, &good).with_decay(0.97);
        // Cap 3: two windows drafted all 3, one was pruned to a single draft.
        for (stats, us) in [
            (capped(3, 3, 3), 73_500.0),
            (capped(3, 3, 1), 73_500.0),
            (capped(3, 1, 0), 48_700.0),
        ] {
            floor.observe_window(NativeWindow::Batched, stats, us, 7, &good);
        }
        let caps = floor.caps[3];
        let window_us = caps.us / caps.windows;
        let (one, three) = (
            caps.drafted[1] / caps.windows,
            caps.drafted[3] / caps.windows,
        );
        assert!((one - 1.0 / 3.0).abs() < 0.05 && (three - 2.0 / 3.0).abs() < 0.05);
        // At 0.8 agreement a window drafting d emits 1 + 0.8 + ... + 0.8^d.
        let e = |d: i32| (0..=d).map(|i| 0.8f32.powi(i)).sum::<f32>();
        let cost = |cap| floor.cap_cost(&good, cap).expect("priced");
        let priced = window_us / (one * e(1) + three * e(3));
        assert!((cost(3) - priced).abs() < 1.0, "{} vs {priced}", cost(3));
        // Neighbours shift only the windows the prune let through to the
        // whole cap: one row more (up) or less (down).
        let row = floor.fit_line().expect("fit").2 as f32;
        let up = (window_us + row * three) / (one * e(1) + three * e(4));
        let down = (window_us - row * three) / (one * e(1) + three * e(2));
        assert!((cost(4) - up).abs() < 1.0, "{} vs {up}", cost(4));
        assert!((cost(2) - down).abs() < 1.0, "{} vs {down}", cost(2));
        assert_eq!(floor.cap_cost(&good, 1), None);
        assert_eq!(floor.cap_cost(&good, 5), None);
        assert!(floor
            .trace_line(TRACE_PREFIX, &good, "t", 1.0, 1, 3)
            .contains("\"route\":\"cap4\",\"modeled\":true"));
        // The row is the request's own: windows drafting 2 and 3 rows at the
        // R9700's 12.4 ms per row fit it within a few percent.
        for _ in 0..20 {
            floor.observe_window(NativeWindow::Batched, capped(2, 2, 2), 61_100.0, 7, &good);
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), 73_500.0, 7, &good);
        }
        let row = floor.modeled_us(4).expect("fit") - floor.modeled_us(3).expect("fit");
        assert!((row - 12_400.0).abs() < 0.03 * 12_400.0, "row {row}");
    }

    #[test]
    fn uncompared_depths_relax_toward_the_shallower_agreement() {
        let ratio = |(accepted, total): (f32, f32)| accepted / total;
        // Windows that accept both of two drafts never compare depth 5; it
        // drifts to the request's own acceptance, not to the 0.8 prior.
        let mut agreement = [(1.9, 2.0); MAX_DEPTH];
        for _ in 0..100 {
            observe_agreement(&mut agreement, &window(2, 2), DEFAULT_DECAY);
        }
        assert!(ratio(agreement[5]) > 0.95, "{:?}", agreement[5]);
        // Rejections at depth 0 pull every deeper depth toward depth 0's rate
        // (each depth lags the one above it by the relaxation's ~20 windows).
        let mut agreement = [AGREEMENT_PRIOR; MAX_DEPTH];
        for _ in 0..300 {
            observe_agreement(&mut agreement, &window(0, 3), DEFAULT_DECAY);
        }
        assert!(ratio(agreement[0]) < 0.01);
        assert!(ratio(agreement[4]) < 0.2, "{:?}", agreement[4]);
        // Depth 0 itself, when no draft was compared, relaxes to the prior.
        let mut agreement = [(0.0, 2.0); MAX_DEPTH];
        for _ in 0..200 {
            observe_agreement(&mut agreement, &window(0, 0), DEFAULT_DECAY);
        }
        let prior = ratio(AGREEMENT_PRIOR);
        assert!((ratio(agreement[0]) - prior).abs() < 1e-3);
        assert!((ratio(agreement[MAX_DEPTH - 1]) - prior).abs() < 0.05);
    }

    /// A window of `emitted` committed tokens that drafted `drafts`.
    fn emitted_window(emitted: usize, drafts: usize) -> MtpWindow {
        MtpWindow {
            committed: vec![0; emitted],
            accepted: emitted.saturating_sub(1),
            drafts_generated: drafts,
        }
    }

    /// Run the AR probe tokens a would-be retirement asks for, each costing
    /// `us`, asserting the route stays `ArSample` until the last.
    fn run_probes(floor: &mut TestFloor, agreement: &[(f32, f32)], n: usize, us: f32) {
        for _ in 0..n {
            assert!(floor.needs_ar_sample());
            assert_eq!(floor.route(agreement, 3, 3), Action::ArSynced);
            floor.observe_calibration(us, agreement);
        }
    }

    /// humaneval_0 on the Flash-Next hot set (trace of `a-tier-1`): the
    /// calibration token cost 19.6 ms against 24-28 ms for the AR tokens
    /// after it; the native windows that lost to the 19.6 ms price retired a
    /// request whose mixture of n-gram and native windows cost 20.5 ms per
    /// token.  Fresh AR probes price AR where the request is, and the
    /// mixture then wins.
    #[test]
    fn floor_retire_evidence_reprices_ar_before_a_sticky_retirement() {
        let poor = agreement(0.5, 2.0);
        let run = |evidence: usize| {
            let mut floor =
                TestFloor::with_retire_after(WINDOW_COST, 16).with_retire_evidence(evidence);
            floor.observe_calibration(19_600.0, &poor);
            for _ in 0..5 {
                floor.observe_native(
                    NativeWindow::Batched,
                    &emitted_window(1, 3),
                    60_000.0,
                    3,
                    &poor,
                );
            }
            for _ in 0..8 {
                floor.observe_external(4, 55_000.0, 3, &poor);
            }
            for _ in 0..4 {
                floor.observe_native(
                    NativeWindow::Batched,
                    &emitted_window(1, 3),
                    70_000.0,
                    3,
                    &poor,
                );
            }
            floor
        };
        // Without evidence the single calibration price retires it.
        let mut floor = run(0);
        assert!(floor.retired);
        assert_eq!(floor.route(&poor, 3, 3), Action::ArHeadFree);
        // With evidence the verdict first asks for the probes.
        let mut floor = run(SINGLETON_AR_PROBE_TOKENS);
        assert!(
            !floor.retired,
            "a verdict on one early token must not retire"
        );
        assert_eq!(floor.route(&poor, 3, 3), Action::ArSynced);
        assert!(floor.needs_ar_sample());
        let probes = SINGLETON_AR_PROBE_TOKENS;
        run_probes(&mut floor, &poor, probes - 1, 30_000.0);
        assert!(floor.needs_ar_sample());
        floor.observe_calibration(30_000.0, &poor);
        assert!(!floor.needs_ar_sample());
        assert_eq!(
            floor.ar_us, 30_000.0,
            "the probe mean replaces the calibration price"
        );
        assert!(
            !floor.retired,
            "the realized mixture (~27 ms/token) beats 30 ms"
        );
        assert_eq!(floor.route(&poor, 3, 3), Action::Native(3));
        // Probes ask for nothing while the verdict stays a win.
        assert!(!floor.needs_ar_sample());
    }

    /// records_parser_1k (trace of `a-tier-1`): a streak of rejected drafts
    /// at windows 12-15 pushed the 8-window cost to 27.8 ms/token against a
    /// 19.2 ms/token mean; the verdict at the first post-grace window is a
    /// coin flip around AR.  The slow realized memory keeps it alive.
    #[test]
    fn floor_retire_evidence_slow_realized_cost_survives_a_rejection_streak() {
        let poor = agreement(0.5, 2.0);
        let run = |evidence: usize| {
            let mut floor =
                TestFloor::with_retire_after(WINDOW_COST, 16).with_retire_evidence(evidence);
            floor.observe_calibration(25_000.0, &poor);
            for _ in 0..11 {
                floor.observe_native(
                    NativeWindow::Batched,
                    &emitted_window(4, 3),
                    60_000.0,
                    3,
                    &poor,
                );
            }
            for _ in 0..5 {
                floor.observe_native(
                    NativeWindow::Batched,
                    &emitted_window(1, 3),
                    62_000.0,
                    3,
                    &poor,
                );
            }
            floor
        };
        let floor = run(0);
        assert!(floor.combined.0 / floor.combined.1 > 25_000.0);
        assert!(floor.retired, "the fast memory alone retires on the streak");
        let mut floor = run(SINGLETON_AR_PROBE_TOKENS);
        assert!(floor.combined.0 / floor.combined.1 > 25_000.0);
        assert!(floor.slow.0 / floor.slow.1 < 25_000.0);
        assert!(!floor.retired && !floor.needs_ar_sample());
        assert_eq!(floor.route(&poor, 3, 3), Action::Native(3));
        assert!(!floor.retired);
    }

    /// A request whose windows really lose retires after its probes, at no
    /// more than the probe tokens' cost, and a retired request never probes.
    #[test]
    fn floor_retire_evidence_still_retires_a_losing_request_after_its_probes() {
        let poor = agreement(0.0, 2.0);
        let mut floor = TestFloor::with_retire_after(WINDOW_COST, 16).with_retire_evidence(6);
        floor.observe_calibration(25_000.0, &poor);
        for observed in 1..=15 {
            floor.observe_native(
                NativeWindow::Batched,
                &emitted_window(1, 3),
                70_000.0,
                3,
                &poor,
            );
            assert!(
                !floor.retired && !floor.needs_ar_sample(),
                "grace window {observed}"
            );
        }
        floor.observe_native(
            NativeWindow::Batched,
            &emitted_window(1, 3),
            70_000.0,
            3,
            &poor,
        );
        assert!(!floor.retired);
        assert!(floor.needs_ar_sample());
        run_probes(&mut floor, &poor, 5, 24_000.0);
        // The probes are not windows: nothing observed, nothing decayed.
        assert_eq!(floor.observed, 16);
        floor.observe_calibration(f32::NAN, &poor);
        assert!(
            floor.retired,
            "an invalid probe is skipped, the rest price AR"
        );
        assert_eq!(floor.ar_us, 24_000.0);
        assert!(!floor.needs_ar_sample());
        assert_eq!(floor.route(&poor, 3, 3), Action::ArHeadFree);
        floor.observe_calibration(24_000.0, &poor);
        assert_eq!(floor.probe_left, 0);
    }

    /// A winning request is re-probed at most once per `retire_after`
    /// windows, and only when a verdict would retire it.
    #[test]
    fn floor_retire_evidence_probes_at_most_once_per_grace() {
        let poor = agreement(0.5, 2.0);
        let mut floor = TestFloor::with_retire_after(WINDOW_COST, 16).with_retire_evidence(2);
        floor.observe_calibration(10_000.0, &poor);
        for _ in 0..16 {
            floor.observe_native(
                NativeWindow::Batched,
                &emitted_window(1, 3),
                60_000.0,
                3,
                &poor,
            );
        }
        assert!(floor.needs_ar_sample() && !floor.retired);
        let started = floor.probed_at;
        assert_eq!(started, Some(16));
        // Fresh AR now costs more than the windows (60 ms for 1 token loses
        // to nothing under 60 ms): the verdict flips to a win.
        run_probes(&mut floor, &poor, 2, 70_000.0);
        assert!(!floor.retired);
        // The same windows keep winning against 70 ms: no new probe.
        for _ in 0..15 {
            floor.observe_native(
                NativeWindow::Batched,
                &emitted_window(1, 3),
                60_000.0,
                3,
                &poor,
            );
            assert!(!floor.needs_ar_sample() && !floor.retired);
        }
        // AR gets cheap again; the verdict loses, but only once 16 windows
        // passed since the last probe does it ask for fresh evidence first.
        floor.ar_us = 10_000.0;
        floor.observe_native(
            NativeWindow::Batched,
            &emitted_window(1, 3),
            60_000.0,
            3,
            &poor,
        );
        assert_eq!(floor.observed, 32);
        assert!(floor.needs_ar_sample() && !floor.retired);
        assert_eq!(floor.probed_at, Some(32));
    }

    /// A singleton floor on a budget of 7 that ran cap 5 for its whole grace,
    /// every window losing to a 30 ms AR step, then re-priced AR: its
    /// verdict would retire, so it starts the bounded shallow re-probe.
    fn deep_only_losing_floor(bad: &[(f32, f32)]) -> TestFloor {
        let mut floor = TestFloor::with_retire_after(WINDOW_COST, SINGLETON_MIN_WINDOWS)
            .with_retire_evidence(2);
        floor.observe_calibration(30_000.0, bad);
        for _ in 0..SINGLETON_MIN_WINDOWS {
            floor.observe_window(NativeWindow::Batched, capped(5, 5, 0), 200_000.0, 7, bad);
        }
        assert!(floor.needs_ar_sample() && !floor.retired);
        for _ in 0..2 {
            floor.observe_calibration(30_000.0, bad);
        }
        assert_eq!(floor.ar_us, 30_000.0);
        assert!(!floor.retired && !floor.needs_ar_sample());
        assert_eq!(floor.shallow_state, ShallowState::Probing);
        assert!(floor.deep_seen && floor.shallow_since_deep == 0);
        floor
    }

    /// Deep-only losing evidence re-probes shallow before retiring; a
    /// shallow control that beats AR keeps the request alive on its shallow
    /// route, and the trace carries the state.
    #[test]
    fn floor_shallow_reprobe_win_prevents_retirement() {
        let bad = agreement(0.0, 2.0);
        let mut floor = deep_only_losing_floor(&bad);
        assert_eq!(floor.route(&bad, 7, 3), Action::Native(3));
        for done in 1..=FLOOR_PROBE_WINDOWS {
            assert_eq!(floor.shallow_state, ShallowState::Probing, "window {done}");
            assert!(!floor.retired);
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), 60_000.0, 7, &bad);
        }
        assert_eq!(floor.shallow_state, ShallowState::Holding);
        assert!(!floor.retired);
        assert_eq!(floor.shallow.priced, FLOOR_PROBE_WINDOWS);
        assert_eq!(floor.shallow_cost(), Some(15_000.0));
        assert!(!floor.deep_seen);
        assert_eq!(floor.shallow_since_deep, FLOOR_PROBE_WINDOWS);
        // The chooser's own cap-3 price (no accepted prefix predicted) still
        // loses; the fresh realized shallow cost keeps the request on cap 3.
        assert_eq!(floor.route(&bad, 7, 3), Action::Native(3));
        assert!(!floor.retired);
        let trace = floor.trace_line(TRACE_PREFIX, &bad, "t", 1.0, 1, 3);
        for field in [
            "\"shallow_state\":\"holding\"",
            "\"shallow_priced\":3",
            "\"shallow_windows\":3",
            "\"shallow_budget\":6",
            "\"budget_k\":7",
            "\"deep_seen\":false",
            "\"shallow_since_deep\":3",
            "\"shallow_us_per_tok\":15000.0",
            "\"climb_streak\":0",
        ] {
            assert!(trace.contains(field), "{field} in {trace}");
        }
    }

    /// Shallow windows that also lose to AR end the re-probe in a sticky
    /// retirement, after exactly the probe budget of priced windows.
    #[test]
    fn floor_shallow_reprobe_loss_retires() {
        let bad = agreement(0.0, 2.0);
        let mut floor = deep_only_losing_floor(&bad);
        for done in 1..=FLOOR_PROBE_WINDOWS {
            assert!(!floor.retired, "window {done}");
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 0), 60_000.0, 7, &bad);
        }
        assert_eq!(floor.shallow_state, ShallowState::Lost);
        assert!(floor.retired);
        assert_eq!(floor.route(&bad, 7, 3), Action::ArHeadFree);
        assert!(floor
            .trace_line(TRACE_PREFIX, &bad, "t", 1.0, 1, 3)
            .contains("\"shallow_state\":\"lost\""));
    }

    /// The lane floor (no AR probe tokens, retiring at the probe budget)
    /// takes the same shallow control before its sticky retirement.
    #[test]
    fn floor_shallow_reprobe_guards_floors_without_ar_probes() {
        let bad = agreement(0.0, 2.0);
        let deep = || {
            let mut floor = TestFloor::new(WINDOW_COST);
            floor.observe_calibration(30_000.0, &bad);
            for _ in 0..FLOOR_PROBE_WINDOWS {
                floor.observe_window(NativeWindow::Batched, capped(5, 5, 0), 200_000.0, 7, &bad);
            }
            floor
        };
        let mut floor = deep();
        assert!(!floor.retired);
        assert_eq!(floor.shallow_state, ShallowState::Probing);
        assert_eq!(floor.route(&bad, 7, 3), Action::Native(3));
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), 60_000.0, 7, &bad);
        }
        assert_eq!(floor.shallow_state, ShallowState::Holding);
        assert!(!floor.retired);
        let mut floor = deep();
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 0), 60_000.0, 7, &bad);
        }
        assert!(floor.retired);
    }

    /// First-use unpriced windows spend the hard window bound but not the
    /// priced budget: a valid control still wins, and a probe whose every
    /// timing is invalid ends, finitely, in the fail-closed retirement.
    #[test]
    fn floor_shallow_reprobe_budget_is_bounded_with_unpriced_first_uses() {
        let bad = agreement(0.0, 2.0);
        assert_eq!(FLOOR_SHALLOW_MAX_WINDOWS, 2 * FLOOR_PROBE_WINDOWS);
        // Two unpriced first uses (drafted depths 1 and 2), then a priced
        // winner at depth 3: five windows, inside the bound.
        let mut floor = deep_only_losing_floor(&bad);
        for depth in [1, 2] {
            floor.observe_window(
                NativeWindow::Batched,
                capped(3, depth, 0),
                f32::NAN,
                7,
                &bad,
            );
        }
        for _ in 0..FLOOR_PROBE_WINDOWS - 1 {
            assert_eq!(floor.shallow_state, ShallowState::Probing);
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), 60_000.0, 7, &bad);
        }
        assert_eq!(floor.shallow_state, ShallowState::Probing);
        floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), 60_000.0, 7, &bad);
        assert_eq!(floor.shallow_state, ShallowState::Holding);
        assert_eq!(floor.shallow.windows, 2 + FLOOR_PROBE_WINDOWS);
        assert_eq!(floor.shallow.priced, FLOOR_PROBE_WINDOWS);
        assert!(!floor.retired);

        // One priced winner among five unpriced windows ends at the bound
        // with that control's verdict.
        let mut floor = deep_only_losing_floor(&bad);
        floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), 60_000.0, 7, &bad);
        for spent in 2..FLOOR_SHALLOW_MAX_WINDOWS {
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), f32::NAN, 7, &bad);
            assert_eq!(floor.shallow_state, ShallowState::Probing, "window {spent}");
        }
        floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), f32::NAN, 7, &bad);
        assert_eq!(floor.shallow_state, ShallowState::Holding);
        assert_eq!(floor.shallow.windows, FLOOR_SHALLOW_MAX_WINDOWS);
        assert_eq!(floor.shallow.priced, 1);
        assert!(!floor.retired);

        // No priced control at all: bounded, then retired (an unpriced win
        // cannot be asserted).
        let mut floor = deep_only_losing_floor(&bad);
        for spent in 1..FLOOR_SHALLOW_MAX_WINDOWS {
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), f32::NAN, 7, &bad);
            assert_eq!(floor.shallow_state, ShallowState::Probing, "window {spent}");
            assert!(!floor.retired);
            assert_eq!(floor.shallow.windows, spent);
            assert_eq!(floor.route(&bad, 7, 3), Action::Native(3));
        }
        floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), f32::NAN, 7, &bad);
        assert_eq!(floor.shallow_state, ShallowState::Unpriced);
        assert!(floor.retired);
        assert_eq!(floor.shallow.priced, 0);
        assert_eq!(floor.route(&bad, 7, 3), Action::ArHeadFree);
        assert!(floor
            .trace_line(TRACE_PREFIX, &bad, "t", 1.0, 1, 3)
            .contains("\"shallow_state\":\"unpriced\""));

        // External windows spend the same bound, so a probe cannot outlive
        // it while takeovers run.
        let mut floor = deep_only_losing_floor(&bad);
        for spent in 1..FLOOR_SHALLOW_MAX_WINDOWS {
            floor.observe_external(1, 200_000.0, 4, &bad);
            assert_eq!(floor.shallow.windows, spent);
            assert_eq!(floor.shallow_state, ShallowState::Probing);
            assert!(!floor.retired);
        }
        floor.observe_external(1, 200_000.0, 4, &bad);
        assert_eq!(floor.shallow_state, ShallowState::Unpriced);
        assert!(floor.retired);
    }

    /// Only the request's current budget asks for the probe: a budget of at
    /// most 3 (an explicit K below 3, or a route `k` override back to 3)
    /// retires on its own evidence without shallow windows.
    #[test]
    fn floor_shallow_reprobe_is_skipped_for_budgets_at_or_below_three() {
        let bad = agreement(0.0, 2.0);
        for k in 1..=FLOOR_SHALLOW_CAP {
            let mut floor = TestFloor::with_retire_after(WINDOW_COST, SINGLETON_MIN_WINDOWS)
                .with_retire_evidence(2);
            floor.observe_calibration(30_000.0, &bad);
            for _ in 0..SINGLETON_MIN_WINDOWS {
                floor.observe_window(NativeWindow::Batched, capped(k, k, 0), 60_000.0, k, &bad);
            }
            assert!(floor.needs_ar_sample() && !floor.retired, "k {k}");
            for _ in 0..2 {
                floor.observe_calibration(30_000.0, &bad);
            }
            assert!(floor.retired, "k {k}");
            assert_eq!(floor.shallow_state, ShallowState::Idle, "k {k}");
            assert_eq!(floor.shallow.windows, 0, "k {k}");
        }
        // A deep request whose current route budget is 3: the override
        // starts no shallow windows even though deep windows were seen.
        let mut floor = TestFloor::with_retire_after(WINDOW_COST, SINGLETON_MIN_WINDOWS)
            .with_retire_evidence(2);
        floor.observe_calibration(30_000.0, &bad);
        for _ in 0..SINGLETON_MIN_WINDOWS {
            floor.observe_window(NativeWindow::Batched, capped(5, 5, 0), 200_000.0, 7, &bad);
        }
        assert!(floor.deep_seen && floor.needs_ar_sample());
        assert_eq!(floor.route(&bad, 3, 3), Action::ArSynced);
        for _ in 0..2 {
            floor.observe_calibration(30_000.0, &bad);
        }
        assert!(floor.retired);
        assert_eq!(floor.shallow_state, ShallowState::Idle);
    }

    /// Stale shallow samples are not fresh evidence: cap-3 windows decayed
    /// into the cap statistics before the deep exploration still leave the
    /// retirement needing its own shallow windows.
    #[test]
    fn floor_shallow_reprobe_ignores_stale_shallow_samples() {
        let bad = agreement(0.0, 2.0);
        let mut floor = TestFloor::with_retire_after(WINDOW_COST, SINGLETON_MIN_WINDOWS)
            .with_retire_evidence(2);
        floor.observe_calibration(30_000.0, &bad);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 0), 60_000.0, 7, &bad);
        }
        for _ in FLOOR_PROBE_WINDOWS..SINGLETON_MIN_WINDOWS {
            floor.observe_window(NativeWindow::Batched, capped(5, 5, 0), 200_000.0, 7, &bad);
        }
        assert!(floor.caps[3].windows > 0.0, "decayed cap-3 samples remain");
        assert!(floor.needs_ar_sample());
        for _ in 0..2 {
            floor.observe_calibration(30_000.0, &bad);
        }
        assert_eq!(floor.shallow_state, ShallowState::Probing);
        assert_eq!(floor.shallow_since_deep, 0);
        assert!(!floor.retired);
    }

    /// Positive evidence gates the climb: a held shallow request stays at
    /// cap 3 until a deeper candidate has priced below the shallow one and
    /// AR for the probe budget of consecutive windows; a deep window then
    /// ends the hold.
    #[test]
    fn floor_shallow_hold_resumes_the_climb_only_on_positive_evidence() {
        let bad = agreement(0.0, 2.0);
        let good = agreement(2.0, 2.0);
        let mut floor = deep_only_losing_floor(&bad);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), 60_000.0, 7, &bad);
        }
        assert_eq!(floor.shallow_state, ShallowState::Holding);
        assert_eq!(floor.climb_streak, 0);
        // The deep exploration's window-cost fit would only slow the
        // shallow row price's convergence; drop it to isolate the evidence
        // rule (every later window is a depth-3 one, row = the prior).
        floor.fit = [0.0; 5];
        let mut climbed = None;
        for window in 1..=40 {
            floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), 60_000.0, 7, &good);
            assert_eq!(
                floor.shallow_state,
                ShallowState::Holding,
                "window {window}"
            );
            assert!(!floor.retired);
            let route = floor.route(&good, 7, 3);
            if floor.climb_streak < FLOOR_PROBE_WINDOWS {
                assert!(
                    matches!(route, Action::Native(cap) if cap <= FLOOR_SHALLOW_CAP),
                    "window {window}: {route:?}"
                );
            } else {
                assert!(
                    matches!(route, Action::Native(cap) if cap > FLOOR_SHALLOW_CAP),
                    "window {window}: {route:?}"
                );
                climbed = Some(window);
                break;
            }
        }
        assert!(
            climbed.is_some_and(|window| window >= FLOOR_PROBE_WINDOWS),
            "{climbed:?}"
        );
        floor.observe_window(NativeWindow::Batched, capped(4, 4, 4), 70_000.0, 7, &good);
        assert_eq!(floor.shallow_state, ShallowState::Idle);
        assert!(floor.deep_seen);
        assert_eq!(floor.shallow_since_deep, 0);
        assert_eq!(floor.climb_streak, 0);
    }

    #[test]
    fn floor_retirement_is_sticky_copy_lane() {
        let bad = agreement(0.0, 2.0);
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(30.0, &bad);
        batched_windows(&mut floor, &bad, 2, 90.0, FLOOR_PROBE_WINDOWS);
        assert!(floor.retired);
        batched_windows(&mut floor, &good, 3, 5.0, 10);
        // A good takeover does not stop the lane of a retired request.
        assert!(!floor.observe_external_copy(8, 10.0, 8, &good));
        floor.observe_calibration(1000.0, &good);
        assert!(floor.retired);
        assert_eq!(floor.route(&good, 7, 3), Action::ArHeadFree);
    }

    #[test]
    fn floor_good_external_keeps_a_losing_native_route_alive_while_the_mixture_beats_ar_copy_lane()
    {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        // 10 us per external token beats AR.
        assert!(!floor.observe_external_copy(4, 40.0, 4, &bad));
        assert!(!floor.observe_external_copy(4, 40.0, 4, &bad));
        // One losing native window (90 us for one token) spends the budget;
        // the realized mixture (~20.6 us per token) still beats AR.
        floor.observe_native(NativeWindow::Batched, &window(0, 1), 90.0, 7, &bad);
        assert_eq!(floor.probes, FLOOR_PROBE_WINDOWS);
        assert!(!floor.retired);
        // The measured native route loses to AR, yet it keeps running while
        // the mixture wins.
        assert_eq!(floor.route(&bad, 7, 5), Action::Native(1));
        assert!(!floor.retired);
        // Losing native windows drag the mixture up to AR; it then retires at
        // the first window where the mixture stops beating AR, stickily.
        let mut kept_alive = 0;
        while !floor.retired {
            assert_eq!(floor.route(&bad, 7, 5), Action::Native(1));
            floor.observe_native(NativeWindow::Batched, &window(0, 1), 90.0, 7, &bad);
            kept_alive += 1;
            assert!(kept_alive < 50, "never retired");
        }
        assert!(kept_alive > 0);
        assert!(floor.combined.0 / floor.combined.1 >= 30.0);
        assert_eq!(floor.route(&bad, 7, 5), Action::ArHeadFree);
        assert!(floor.retired);
        assert!(!floor.observe_external_copy(4, 40.0, 4, &bad));
        assert_eq!(floor.route(&bad, 7, 5), Action::ArHeadFree);
    }

    #[test]
    fn floor_retired_keeps_pricing_external_windows_and_stops_only_when_they_lose() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        batched_windows(&mut floor, &bad, 1, 90.0, 2);
        // Native loses by prediction: the third window retires the request.
        batched_windows(&mut floor, &bad, 1, 90.0, 20);
        assert!(floor.retired);
        let (probes, observed) = (floor.probes, floor.observed);
        // Takeovers beating AR (10 us per token against 30) are priced and do
        // not stop the lane, the retirement and the routing stay.
        assert!(!floor.observe_external_copy(4, 40.0, 4, &bad));
        assert!(floor.retired && floor.external.1 > 0.0 && floor.external_depth[4].1 > 0.0);
        assert_eq!((floor.probes, floor.observed), (probes, observed));
        assert_eq!(floor.route(&bad, 3, 3), Action::ArHeadFree);
        // Takeovers losing to AR block the lane.
        let mut stopped = false;
        for _ in 0..30 {
            if floor.observe_external_copy(1, 200.0, 4, &bad) {
                stopped = true;
                break;
            }
        }
        assert!(stopped && floor.retired && floor.external_blocked);
    }

    // The call-log lines a test floor wrote, per test thread.
    thread_local! {
        static LOGGED: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
    }

    fn capture(line: &str) {
        LOGGED.with(|lines| lines.borrow_mut().push(line.to_string()));
    }

    fn take_logged() -> Vec<String> {
        LOGGED.with(|lines| std::mem::take(&mut *lines.borrow_mut()))
    }

    const LOG_PREFIX: &str = "QWEN4_FLOOR_CALL";

    fn logged_floor(retire_after: usize, probe_tokens: usize) -> TestFloor {
        take_logged();
        TestFloor::with_retire_after(WINDOW_COST, retire_after)
            .with_retire_evidence(probe_tokens)
            .with_decay(0.97)
            .with_call_log_to(LOG_PREFIX, "unit", true, capture)
    }

    /// One floor driven through every logged op, retirement and re-arm
    /// included; its call-log lines.
    fn scripted_log() -> (Vec<String>, TestFloor) {
        let good = agreement(1.8, 2.0);
        let bad = agreement(0.0, 2.0);
        let mut floor = logged_floor(5, 2);
        floor.route(&good, 7, 3);
        floor.observe_calibration(33.0, &good);
        floor.route(&good, 7, 3);
        floor.observe_native(NativeWindow::Batched, &window(2, 2), 75.0, 7, &good);
        floor.observe_native(NativeWindow::Interleaved, &window(1, 1), 50.0, 7, &good);
        floor.observe_window(NativeWindow::Batched, capped(3, 3, 3), 80.0, 7, &good);
        floor.observe_external(4, 60.0, 4, &good);
        floor.observe_external_copy(6, 70.0, 5, &good);
        floor.set_decay(0.9);
        floor.route(&good, 7, 3);
        for _ in 0..8 {
            floor.route(&bad, 3, 2);
            floor.observe_native(NativeWindow::Batched, &window(0, 2), 900.0, 3, &bad);
            if floor.needs_ar_sample() {
                floor.observe_calibration(40.0, &bad);
            }
        }
        floor.observe_external_copy(1, 900.0, 4, &bad);
        floor.route(&bad, 3, 2);
        floor.rearm();
        floor.route(&good, 7, 3);
        (take_logged(), floor)
    }

    #[test]
    fn call_log_records_the_install_and_every_public_call_once() {
        let (lines, floor) = scripted_log();
        assert!(lines[0].starts_with("QWEN4_FLOOR_CALL {"), "{}", lines[0]);
        let install: serde_json::Value =
            serde_json::from_str(lines[0].strip_prefix("QWEN4_FLOOR_CALL ").unwrap()).unwrap();
        assert_eq!(install["op"], "install");
        assert_eq!(install["floor"], floor.id());
        assert_eq!(install["owner"], "unit");
        assert_eq!(install["enabled"], true);
        assert_eq!(install["retire_after"], 5);
        assert_eq!(install["probe_tokens"], 2);
        assert_eq!(install["decay"], format!("{:08x}", 0.97f32.to_bits()));
        assert_eq!(install["capacity"], WINDOW_COST.len() - 1);
        assert_eq!(install["extended"], false);
        // 1 install + 3 routes + 1 calibration + 2 native + 1 window + 2
        // external + 1 set_decay, then the loop's and the tail's calls.
        let ops: Vec<String> = lines
            .iter()
            .map(|line| {
                let json = line.strip_prefix("QWEN4_FLOOR_CALL ").unwrap();
                let value: serde_json::Value = serde_json::from_str(json).unwrap();
                value["op"].as_str().unwrap().to_string()
            })
            .collect();
        assert_eq!(
            &ops[..9],
            [
                "install",
                "route",
                "observe_calibration",
                "route",
                "observe_native",
                "observe_native",
                "observe_window",
                "observe_external",
                "observe_external_copy",
            ]
        );
        assert_eq!(ops[ops.len() - 2..], ["rearm", "route"]);
        let seqs: Vec<u64> = lines
            .iter()
            .map(|line| {
                let json = line.strip_prefix("QWEN4_FLOOR_CALL ").unwrap();
                let value: serde_json::Value = serde_json::from_str(json).unwrap();
                value["seq"].as_u64().unwrap()
            })
            .collect();
        assert!(seqs.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn call_log_replays_every_call_bit_identically() {
        let (lines, _) = scripted_log();
        let report =
            replay_call_log::<8, 64>(lines.iter().map(String::as_str), LOG_PREFIX, WINDOW_COST);
        assert_eq!(report.malformed, 0, "{:?}", report.first_malformed);
        assert_eq!(report.orphan_calls, 0);
        assert_eq!(report.floors.len(), 1);
        assert_eq!(report.floors[0].owner, "unit");
        assert_eq!(report.calls(), lines.len() - 1);
        assert_eq!(
            report.mismatches(),
            0,
            "{:?}",
            report.floors[0].first_mismatch
        );
        // Text before the prefix (a log timestamp) is ignored.
        let stamped: Vec<String> = lines
            .iter()
            .map(|line| format!("[12:00:01 INFO] {line}"))
            .collect();
        let report =
            replay_call_log::<8, 64>(stamped.iter().map(String::as_str), LOG_PREFIX, WINDOW_COST);
        assert_eq!((report.calls(), report.mismatches()), (lines.len() - 1, 0));
    }

    #[test]
    fn call_log_replays_capacity_and_reset_ops() {
        take_logged();
        let good = vec![(1.8, 2.0); 16];
        let mut floor =
            TestFloor::new(WINDOW_COST).with_call_log_to(LOG_PREFIX, "unit", true, capture);
        floor.set_capacity(16);
        floor.observe_calibration(30.0, &good);
        floor.route(&good, 16, 3);
        floor.reset_request(5, DEFAULT_DECAY, 0);
        floor.route(&good, 16, 3);
        let lines = take_logged();
        let report =
            replay_call_log::<8, 64>(lines.iter().map(String::as_str), LOG_PREFIX, WINDOW_COST);
        assert_eq!(report.malformed, 0, "{:?}", report.first_malformed);
        assert_eq!((report.calls(), report.mismatches()), (lines.len() - 1, 0));
        // Short (non-legacy-length) agreements replay exactly.
        let short = vec![(1.8, 2.0); 3];
        take_logged();
        let mut floor =
            TestFloor::new(WINDOW_COST).with_call_log_to(LOG_PREFIX, "unit", true, capture);
        floor.observe_calibration(30.0, &short);
        floor.route(&short, 3, 2);
        let lines = take_logged();
        let report =
            replay_call_log::<8, 64>(lines.iter().map(String::as_str), LOG_PREFIX, WINDOW_COST);
        assert_eq!((report.calls(), report.mismatches()), (lines.len() - 1, 0));
    }

    #[test]
    fn call_log_keeps_the_recorded_route_names() {
        take_logged();
        let bad = agreement(0.0, 2.0);
        let mut floor =
            TestFloor::new(WINDOW_COST).with_call_log_to(LOG_PREFIX, "unit", true, capture);
        floor.observe_calibration(30.0, &bad);
        for _ in 0..FLOOR_PROBE_WINDOWS {
            assert_eq!(floor.route(&bad, 3, 2), Action::Native(2));
            floor.observe_native(NativeWindow::Batched, &window(0, 2), 90.0, 3, &bad);
        }
        assert_eq!(floor.route(&bad, 3, 2), Action::ArHeadFree);
        let lines = take_logged();
        let routes: Vec<&str> = lines
            .iter()
            .filter(|line| line.contains("\"op\":\"route\""))
            .map(|line| {
                line.split("\"ret\":")
                    .nth(1)
                    .unwrap()
                    .split(',')
                    .next()
                    .unwrap()
            })
            .collect();
        assert_eq!(
            routes,
            ["\"Batched:2\"", "\"Batched:2\"", "\"Batched:2\"", "\"Ar\""]
        );
        assert_eq!(action_name(Action::ArSynced), "ArSample");
        assert_eq!(action_name(Action::Interleaved), "Interleaved");
    }

    #[test]
    fn replay_reports_the_first_mismatch_of_a_tampered_floor() {
        let (mut lines, _) = scripted_log();
        let at = lines
            .iter()
            .position(|line| line.contains("\"retired_after\":false"))
            .expect("a call before retirement");
        lines[at] = lines[at].replace("\"retired_after\":false", "\"retired_after\":true");
        let report =
            replay_call_log::<8, 64>(lines.iter().map(String::as_str), LOG_PREFIX, WINDOW_COST);
        assert_eq!(report.mismatches(), 1);
        let first = report.floors[0]
            .first_mismatch
            .as_deref()
            .expect("a mismatch");
        assert!(
            first.contains("retired_after logged true, replayed false"),
            "{first}"
        );
        // A line that carries the prefix but is not a call is malformed, not skipped.
        lines.push(format!("{LOG_PREFIX} {{\"seq\":1}}"));
        let report =
            replay_call_log::<8, 64>(lines.iter().map(String::as_str), LOG_PREFIX, WINDOW_COST);
        assert_eq!(report.malformed, 1);
    }

    #[test]
    fn a_floor_without_a_call_log_writes_nothing() {
        take_logged();
        let good = agreement(1.8, 2.0);
        let mut floor = TestFloor::new(WINDOW_COST);
        floor.observe_calibration(33.0, &good);
        floor.route(&good, 7, 3);
        floor.rearm();
        assert!(take_logged().is_empty());
    }
}

/// Tests of the Flash-Next derived-K chooser and its frozen tables.
#[cfg(test)]
mod derived_k_tests {
    use super::fn_phase1_v1 as table;
    use super::*;
    use serde_json::Value;
    use std::collections::HashMap;

    fn mtp_key(depth: u8) -> ResidualKey {
        ResidualKey {
            source: DerivedSource::Mtp,
            depth,
            feature: DerivedFeature::Native,
        }
    }

    /// A depth without any calibration support: the 0.8 prior.
    fn unsupported(depth: u8) -> DerivedDepth {
        DerivedDepth {
            key: mtp_key(depth),
            counts: CalibrationCounts::default(),
        }
    }

    fn depths(n: u8) -> Vec<DerivedDepth> {
        (1..=n).map(unsupported).collect()
    }

    fn price(k: u8, cost_us: f64) -> DerivedPrice {
        DerivedPrice { k, cost_us }
    }

    /// replay2 `expected_tokens`.
    fn reference_expected(probabilities: &[f64]) -> f64 {
        let mut total = 1.0f64;
        let mut survival = 1.0f64;
        for p in probabilities {
            survival *= *p;
            total += survival;
        }
        total
    }

    /// `1 + p` of one depth as the chooser sees it: the expected tokens of a
    /// width-1 action priced on that depth alone.
    fn one_depth_expected(request: &DerivedRequest, key: ResidualKey) -> f64 {
        let depth = DerivedDepth {
            key,
            counts: CalibrationCounts::default(),
        };
        request
            .choose(&[depth], &[price(1, 1.0)], 1)
            .expect("a priced prefix")
            .expected
    }

    #[test]
    fn derived_k_offer_bin_edges() {
        let bins = [
            (None, OfferBin::Missing),
            (Some(0), OfferBin::Zero),
            (Some(1), OfferBin::One),
            (Some(2), OfferBin::TwoThree),
            (Some(3), OfferBin::TwoThree),
            (Some(4), OfferBin::FourSeven),
            (Some(7), OfferBin::FourSeven),
            (Some(8), OfferBin::EightFifteen),
            (Some(15), OfferBin::EightFifteen),
            (Some(16), OfferBin::SixteenThirtyOne),
            (Some(31), OfferBin::SixteenThirtyOne),
            (Some(32), OfferBin::ThirtyTwoPlus),
            (Some(1 << 20), OfferBin::ThirtyTwoPlus),
        ];
        for (offered, bin) in bins {
            assert_eq!(derived_offer_bin(offered), bin, "offered {offered:?}");
        }
    }

    #[test]
    fn derived_k_context_bucket_edges() {
        let buckets = [
            (0, ContextBucket::Le2K),
            (2048, ContextBucket::Le2K),
            (2049, ContextBucket::Le8K),
            (8192, ContextBucket::Le8K),
            (8193, ContextBucket::Le16K),
            (16384, ContextBucket::Le16K),
            (16385, ContextBucket::Over16K),
            (1 << 20, ContextBucket::Over16K),
        ];
        for (position, bucket) in buckets {
            assert_eq!(ContextBucket::of(position), bucket, "position {position}");
        }
    }

    #[test]
    fn derived_k_prior_is_point_eight_without_support() {
        let request = DerivedRequest::new(DerivedCard::R9700);
        let decision = request
            .choose(&depths(3), &[price(3, 1000.0)], 3)
            .expect("a priced prefix");
        assert_eq!(decision.action, DerivedAction::Prefix(3));
        assert_eq!(decision.expected, reference_expected(&[0.8, 0.8, 0.8]));
        assert_eq!(decision.cost_us, 1000.0);
    }

    #[test]
    fn derived_k_cell_pools_toward_depth() {
        let request = DerivedRequest::new(DerivedCard::R9700);
        let one = |counts: CalibrationCounts| {
            let depth = DerivedDepth {
                key: mtp_key(1),
                counts,
            };
            request
                .choose(&[depth], &[price(1, 1.0)], 1)
                .expect("a priced prefix")
                .expected
        };
        // Depth 8/10 with a 3/4 cell: the cell pools toward the depth's 0.8 at
        // weight 2.
        let p_depth = (8.0 + 1.6) / (10.0 + 2.0);
        let p_cell = (3.0 + 2.0 * p_depth) / (4.0 + 2.0);
        let counts = CalibrationCounts {
            depth_s: 8,
            depth_n: 10,
            cell_s: 3,
            cell_n: 4,
        };
        assert_eq!(one(counts), 1.0 + p_cell);
        // An empty cell collapses to the depth-only estimate.
        let empty_cell = CalibrationCounts {
            cell_s: 0,
            cell_n: 0,
            ..counts
        };
        assert_eq!(one(empty_cell), 1.0 + p_depth);
        // A cell with support but no depth support pools toward the bare prior.
        let cell_only = CalibrationCounts {
            depth_s: 0,
            depth_n: 0,
            cell_s: 2,
            cell_n: 2,
        };
        assert_eq!(one(cell_only), 1.0 + (2.0 + 2.0 * 0.8) / (2.0 + 2.0));
    }

    #[test]
    fn derived_k_loo_counts_subtract_the_held_out_request() {
        assert_eq!(loo((10, 12), (4, 5), "t"), (6, 7));
        assert_eq!(loo((3, 3), (3, 3), "t"), (0, 0));
        assert_eq!(loo((3, 5), (0, 0), "t"), (3, 5));
        // The LOO counts feed the same arithmetic as counts that never saw the
        // held-out request.
        let (s, n) = loo((10, 12), (4, 5), "t");
        let request = DerivedRequest::new(DerivedCard::Halo);
        let loo_counts = CalibrationCounts {
            depth_s: s,
            depth_n: n,
            cell_s: s,
            cell_n: n,
        };
        let direct = CalibrationCounts {
            depth_s: 6,
            depth_n: 7,
            cell_s: 6,
            cell_n: 7,
        };
        let at = |counts| {
            let depth = DerivedDepth {
                key: mtp_key(1),
                counts,
            };
            request
                .choose(&[depth], &[price(1, 1.0)], 1)
                .map(|d| d.expected)
        };
        assert_eq!(at(loo_counts), at(direct));
    }

    #[test]
    #[should_panic(expected = "contract error")]
    fn derived_k_loo_underflow_is_a_contract_error() {
        loo((3, 5), (4, 5), "depth");
    }

    #[test]
    fn derived_k_decay_all_then_add_order() {
        let mut request = DerivedRequest::new(DerivedCard::R9700);
        assert_eq!(request.decay(), 0.875);
        assert_eq!(DerivedRequest::new(DerivedCard::Halo).decay(), 0.97);
        // Window 1: success at A (depth 1).
        request.observe(&[mtp_key(1)], 1, 1, false);
        assert_eq!(
            one_depth_expected(&request, mtp_key(1)),
            1.0 + (2.0 * 0.8 + 1.0) / (2.0 + 1.0 + 0.0)
        );
        // Window 2: failure at B. A decays once; B's fresh failure is added
        // after the decay, so it is a whole 1.0.
        request.observe(&[mtp_key(2)], 1, 0, false);
        assert_eq!(
            one_depth_expected(&request, mtp_key(1)),
            1.0 + (2.0 * 0.8 + 0.875) / (2.0 + 0.875 + 0.0)
        );
        assert_eq!(
            one_depth_expected(&request, mtp_key(2)),
            1.0 + (2.0 * 0.8 + 0.0) / (2.0 + 0.0 + 1.0)
        );
        // Window 3 (AR): every cell decays again.
        request.observe(&[], 0, 0, false);
        assert_eq!(
            one_depth_expected(&request, mtp_key(2)),
            1.0 + (2.0 * 0.8 + 0.0) / (2.0 + 0.0 + 1.0 * 0.875)
        );
    }

    #[test]
    fn derived_k_terminal_censoring() {
        let keys = [mtp_key(1), mtp_key(2), mtp_key(3)];
        let accepted_depth_one = 1.0 + (2.0 * 0.8 + 1.0) / (2.0 + 1.0 + 0.0);
        let prior = 1.0 + 0.8;

        // Terminal: depth 1 succeeded; the rejection at depth 2 and everything
        // deeper are censored.
        let mut terminal = DerivedRequest::new(DerivedCard::R9700);
        terminal.observe(&keys, 3, 1, true);
        assert_eq!(one_depth_expected(&terminal, keys[0]), accepted_depth_one);
        assert_eq!(one_depth_expected(&terminal, keys[1]), prior);
        assert_eq!(one_depth_expected(&terminal, keys[2]), prior);

        // Non-terminal: depth 2 is one failure; depth 3 stays censored.
        let mut open = DerivedRequest::new(DerivedCard::R9700);
        open.observe(&keys, 3, 1, false);
        assert_eq!(one_depth_expected(&open, keys[0]), accepted_depth_one);
        assert_eq!(
            one_depth_expected(&open, keys[1]),
            1.0 + (2.0 * 0.8 + 0.0) / (2.0 + 0.0 + 1.0)
        );
        assert_eq!(one_depth_expected(&open, keys[2]), prior);

        // Accepting the whole executed prefix leaves no failure, and accepted
        // is clamped to what was executed (keys past `executed` are untouched).
        let mut clamped = DerivedRequest::new(DerivedCard::R9700);
        clamped.observe(&keys, 2, 5, false);
        assert_eq!(one_depth_expected(&clamped, keys[0]), accepted_depth_one);
        assert_eq!(one_depth_expected(&clamped, keys[1]), accepted_depth_one);
        assert_eq!(one_depth_expected(&clamped, keys[2]), prior);
        assert_eq!(clamped.last_first(DerivedSource::Mtp), Some(true));
    }

    #[test]
    fn derived_k_executed_zero_ages_and_keeps_last_first() {
        let ngram_key = ResidualKey {
            source: DerivedSource::Ngram,
            depth: 1,
            feature: DerivedFeature::Ngram {
                offered: OfferBin::FourSeven,
                key_len: Some(8),
                last_first: None,
            },
        };
        let mut request = DerivedRequest::new(DerivedCard::R9700);
        assert_eq!(request.last_first(DerivedSource::Mtp), None);
        assert_eq!(request.last_first(DerivedSource::Ngram), None);
        request.observe(&[mtp_key(1)], 1, 1, false);
        assert_eq!(request.last_first(DerivedSource::Mtp), Some(true));
        assert_eq!(request.last_first(DerivedSource::Ngram), None);
        request.observe(&[ngram_key], 1, 0, false);
        assert_eq!(request.last_first(DerivedSource::Mtp), Some(true));
        assert_eq!(request.last_first(DerivedSource::Ngram), Some(false));

        // An aging-only window (AR or no supported action) decays the residual
        // but changes no first-token history, terminal or not.
        request.observe(&[], 0, 0, false);
        request.observe(&[], 0, 7, true);
        assert_eq!(request.last_first(DerivedSource::Mtp), Some(true));
        assert_eq!(request.last_first(DerivedSource::Ngram), Some(false));
        let s = 1.0 * 0.875 * 0.875 * 0.875;
        assert_eq!(
            one_depth_expected(&request, mtp_key(1)),
            1.0 + (2.0 * 0.8 + s) / (2.0 + s + 0.0)
        );
    }

    #[test]
    fn derived_k_last_first_none_versus_false() {
        let cases = [
            (0, false, Some(false)),
            (0, true, None),
            (1, false, Some(true)),
            (1, true, Some(true)),
        ];
        for (accepted, terminal, want) in cases {
            let mut request = DerivedRequest::new(DerivedCard::Halo);
            // A known history that the window must overwrite, including with
            // "unknown".
            request.observe(&[mtp_key(1)], 1, 1, false);
            request.observe(&[mtp_key(1)], 1, accepted, terminal);
            assert_eq!(
                request.last_first(DerivedSource::Mtp),
                want,
                "accepted {accepted} terminal {terminal}"
            );
        }
    }

    #[test]
    fn derived_k_tie_tolerance_boundaries() {
        // Absolute: 1e-15 is inside the tolerance of 0, the next float up is not.
        let abs = 1e-15f64;
        let above = f64::from_bits(abs.to_bits() + 1);
        assert!(derived_rates_tied(0.0, abs));
        assert!(derived_rates_tied(abs, 0.0));
        assert!(!derived_rates_tied(0.0, above));
        // Relative: 1e-12 is 4503.6 ulps of 1.0, so 4503 ulps tie and 4504 do not.
        let step = f64::EPSILON;
        let inside = 1.0 + 4503.0 * step;
        let outside = 1.0 + 4504.0 * step;
        assert!(derived_rates_tied(1.0, inside));
        assert!(derived_rates_tied(inside, 1.0));
        assert!(!derived_rates_tied(1.0, outside));
        assert!(!derived_rates_tied(outside, 1.0));
        // The relative tolerance scales with the magnitude.
        let big = 1.0e6f64;
        assert!(derived_rates_tied(big, big * (1.0 + 0.9e-12)));
        assert!(!derived_rates_tied(big, big * (1.0 + 1.1e-12)));
        // Equal infinities tie (as math.isclose); an infinity never ties a
        // finite rate.
        assert!(derived_rates_tied(f64::INFINITY, f64::INFINITY));
        assert!(!derived_rates_tied(f64::INFINITY, 1.0e300));
        assert!(!derived_rates_tied(f64::NAN, f64::NAN));
    }

    /// Prices at which AR and every prefix of three 0.8 depths run at the same
    /// tokens per microsecond (to within a few ulps).
    fn tied_prices() -> Vec<DerivedPrice> {
        vec![
            price(0, 1000.0),
            price(1, reference_expected(&[0.8]) * 1000.0),
            price(2, reference_expected(&[0.8, 0.8]) * 1000.0),
            price(3, reference_expected(&[0.8, 0.8, 0.8]) * 1000.0),
        ]
    }

    #[test]
    fn derived_k_tie_prefers_incumbent_then_smaller_k() {
        let request = DerivedRequest::new(DerivedCard::R9700);
        let prices = tied_prices();
        let pick = |incumbent| {
            request
                .choose(&depths(3), &prices, incumbent)
                .expect("priced")
                .action
        };
        assert_eq!(pick(1), DerivedAction::Prefix(1));
        assert_eq!(pick(2), DerivedAction::Prefix(2));
        assert_eq!(pick(3), DerivedAction::Prefix(3));
        // The incumbent AR (k = 0) is itself a tied action.
        assert_eq!(pick(0), DerivedAction::Ar);
        // An incumbent outside the tied set: the smallest k, which is AR.
        assert_eq!(pick(9), DerivedAction::Ar);

        // Without AR priced the smallest tied prefix wins.
        let drafted: Vec<DerivedPrice> = prices[1..].to_vec();
        let decision = request.choose(&depths(3), &drafted, 9).expect("priced");
        assert_eq!(decision.action, DerivedAction::Prefix(1));
        assert_eq!(decision.cost_us, drafted[0].cost_us);
        assert_eq!(decision.expected, reference_expected(&[0.8]));

        // A prefix slower than the best by more than the tolerance is not tied,
        // even when it is the incumbent.
        let mut slower = tied_prices();
        slower[3].cost_us *= 1.0 + 1e-9;
        let decision = request.choose(&depths(3), &slower, 3).expect("priced");
        assert_eq!(decision.action, DerivedAction::Ar);
        // A clearly faster prefix beats the incumbent.
        let mut faster = tied_prices();
        faster[2].cost_us *= 0.5;
        let decision = request.choose(&depths(3), &faster, 0).expect("priced");
        assert_eq!(decision.action, DerivedAction::Prefix(2));
    }

    fn permutations(items: Vec<DerivedPrice>) -> Vec<Vec<DerivedPrice>> {
        if items.len() <= 1 {
            return vec![items];
        }
        let mut out = Vec::new();
        for at in 0..items.len() {
            let mut rest = items.clone();
            let head = rest.remove(at);
            for mut tail in permutations(rest) {
                tail.insert(0, head);
                out.push(tail);
            }
        }
        out
    }

    #[test]
    fn derived_k_input_price_order_does_not_change_the_result() {
        let request = DerivedRequest::new(DerivedCard::R9700);
        let supported = [
            DerivedDepth {
                key: mtp_key(1),
                counts: CalibrationCounts {
                    depth_s: 90,
                    depth_n: 100,
                    cell_s: 45,
                    cell_n: 50,
                },
            },
            DerivedDepth {
                key: mtp_key(2),
                counts: CalibrationCounts {
                    depth_s: 60,
                    depth_n: 90,
                    cell_s: 28,
                    cell_n: 45,
                },
            },
            DerivedDepth {
                key: mtp_key(3),
                counts: CalibrationCounts {
                    depth_s: 30,
                    depth_n: 60,
                    cell_s: 15,
                    cell_n: 30,
                },
            },
        ];
        let distinct = vec![
            price(0, 30000.0),
            price(1, 34000.0),
            price(2, 36000.0),
            price(3, 41000.0),
        ];
        let prior = depths(3);
        for (prices, incumbent, window_depths) in [
            (distinct, 2u8, &supported[..]),
            (tied_prices(), 2u8, &prior[..]),
            (tied_prices(), 0u8, &prior[..]),
        ] {
            let want = request
                .choose(window_depths, &prices, incumbent)
                .expect("priced");
            for order in permutations(prices) {
                let got = request
                    .choose(window_depths, &order, incumbent)
                    .expect("priced");
                assert_eq!(got.action, want.action, "order {order:?}");
                assert_eq!(got.expected, want.expected, "order {order:?}");
                assert_eq!(got.cost_us, want.cost_us, "order {order:?}");
            }
        }
    }

    #[test]
    fn derived_k_nonpositive_cost_is_ignored() {
        let request = DerivedRequest::new(DerivedCard::R9700);
        let prices = [
            price(0, 0.0),
            price(1, -5.0),
            price(2, f64::NAN),
            price(3, 1000.0),
        ];
        let decision = request.choose(&depths(3), &prices, 1).expect("one valid");
        assert_eq!(decision.action, DerivedAction::Prefix(3));
        // Only invalid prices: no action.
        assert!(request.choose(&depths(3), &prices[..3], 1).is_none());
    }

    #[test]
    fn derived_k_no_valid_action_returns_none() {
        let request = DerivedRequest::new(DerivedCard::Halo);
        assert!(request.choose(&depths(3), &[], 3).is_none());
        // A price beyond the supplied depths has no expected value: not an action.
        assert!(request.choose(&depths(3), &[price(4, 1000.0)], 3).is_none());
        assert!(request.choose(&[], &[price(1, 1000.0)], 1).is_none());
        // AR alone is an action for the chooser (whether to consult it is the
        // caller's rule).
        let ar = request
            .choose(&depths(3), &[price(0, 1000.0)], 3)
            .expect("AR priced");
        assert_eq!(ar.action, DerivedAction::Ar);
        assert_eq!(ar.expected, 1.0);
        assert_eq!(ar.cost_us, 1000.0);
    }

    /// The width the first-failed-marginal rule stops at, walking adjacent
    /// depths from AR with the shared yield/objective. `prices` must price
    /// every width `0..=window.len()`.
    fn first_fail_width(
        request: &DerivedRequest,
        window: &[DerivedDepth],
        prices: &[DerivedPrice],
    ) -> u8 {
        let cost = |k: u8| {
            prices
                .iter()
                .find(|price| price.k == k)
                .expect("contiguous prices")
                .cost_us
        };
        let mut yielded = DerivedYield::seed();
        let mut at = yielded.objective(cost(0));
        let mut width = 0u8;
        for (index, depth) in window.iter().enumerate() {
            yielded = yielded.append(request.probability(depth));
            let next = yielded.objective(cost(index as u8 + 1));
            if !at.marginal_to(next).improves_rate() {
                break;
            }
            at = next;
            width += 1;
        }
        width
    }

    #[test]
    fn derived_k_yield_recurrence_matches_the_reference_expected() {
        let probabilities = [0.9, 0.5, 0.8, 0.3, 0.77];
        let mut yielded = DerivedYield::seed();
        assert_eq!((yielded.expected, yielded.survival), (1.0, 1.0));
        let mut product = 1.0f64;
        for (n, p) in probabilities.iter().enumerate() {
            yielded = yielded.append(*p);
            product *= *p;
            assert_eq!(yielded.survival, product);
            assert_eq!(yielded.expected, reference_expected(&probabilities[..=n]));
        }
        // The chooser's expected tokens are the same recurrence.
        let request = DerivedRequest::new(DerivedCard::R9700);
        let flat: Vec<DerivedPrice> = (0..=3).map(|k| price(k, 1000.0)).collect();
        let decision = request.choose(&depths(3), &flat, 3).expect("priced");
        assert_eq!(decision.expected, reference_expected(&[0.8, 0.8, 0.8]));
    }

    #[test]
    fn derived_k_marginal_uses_conditional_probability_and_survival_once() {
        // Two depths at conditional 0.5: survival 0.5 then 0.25.
        let first = DerivedYield::seed().append(0.5);
        let second = first.append(0.5);
        assert_eq!((first.expected, first.survival), (1.5, 0.5));
        assert_eq!((second.expected, second.survival), (1.75, 0.25));
        let at = first.objective(100.0);
        let marginal = at.marginal_to(second.objective(150.0));
        // The added tokens are survival(d+1) = p * S_d, valued at T_d / E_d
        // per token: survival is applied once, not twice.
        assert_eq!(marginal.delta_expected, 0.25);
        assert_eq!(marginal.delta_cost_us, 50.0);
        assert_eq!(marginal.emitted_token_value_us, 0.25 * (100.0 / 1.5));
        assert_eq!(
            marginal.surplus_us,
            marginal.emitted_token_value_us - marginal.delta_cost_us
        );
        // The marginal agrees with the rate comparison it is derived from.
        for next_cost in [90.0, 110.0, 120.0, 150.0] {
            let next = second.objective(next_cost);
            assert_eq!(
                at.marginal_to(next).improves_rate(),
                next.rate() > at.rate(),
                "next cost {next_cost}"
            );
        }
    }

    #[test]
    fn derived_k_marginal_equality_is_strict() {
        // E 2 -> 3 at T 100: each added token is worth 50 us.
        let at = DerivedObjective {
            expected: 2.0,
            cost_us: 100.0,
        };
        let tied = at.marginal_to(DerivedObjective {
            expected: 3.0,
            cost_us: 150.0,
        });
        assert_eq!(tied.emitted_token_value_us, 50.0);
        assert_eq!(tied.surplus_us, 0.0);
        assert!(!tied.improves_rate());
        let cheaper = at.marginal_to(DerivedObjective {
            expected: 3.0,
            cost_us: 149.0,
        });
        assert!(cheaper.improves_rate());
        let dearer = at.marginal_to(DerivedObjective {
            expected: 3.0,
            cost_us: 151.0,
        });
        assert!(!dearer.improves_rate());
        // An invalid price never improves.
        for cost_us in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let next = DerivedObjective {
                expected: 3.0,
                cost_us,
            };
            assert!(!at.marginal_to(next).improves_rate(), "next cost {cost_us}");
            assert!(!next.marginal_to(at).improves_rate(), "from cost {cost_us}");
        }
    }

    #[test]
    fn derived_k_nonmonotone_costs_global_beats_first_failure() {
        // p = (0.8, 0.8); T = (1, 2, 2.1): the first marginal fails, the
        // second would pass, and K = 2 is the global rate maximum.
        let request = DerivedRequest::new(DerivedCard::R9700);
        let window = depths(2);
        let prices = [price(0, 1.0), price(1, 2.0), price(2, 2.1)];
        let decision = request.choose(&window, &prices, 0).expect("priced");
        assert_eq!(decision.action, DerivedAction::Prefix(2));
        assert_eq!(first_fail_width(&request, &window, &prices), 0);
        let one = DerivedYield::seed().append(0.8);
        let two = one.append(0.8);
        assert!(!DerivedYield::seed()
            .objective(1.0)
            .marginal_to(one.objective(2.0))
            .improves_rate());
        assert!(one
            .objective(2.0)
            .marginal_to(two.objective(2.1))
            .improves_rate());
        // Monotone prices: both rules agree.
        let monotone = [price(0, 1000.0), price(1, 1100.0), price(2, 1200.0)];
        assert_eq!(
            request
                .choose(&window, &monotone, 0)
                .expect("priced")
                .action,
            DerivedAction::Prefix(2)
        );
        assert_eq!(first_fail_width(&request, &window, &monotone), 2);
    }

    #[test]
    fn derived_k_choose_matches_the_reference_on_residual_depths() {
        let mut request = DerivedRequest::new(DerivedCard::R9700);
        let counts = CalibrationCounts {
            depth_s: 90,
            depth_n: 100,
            cell_s: 45,
            cell_n: 50,
        };
        let window: Vec<DerivedDepth> = (1..=3)
            .map(|depth| DerivedDepth {
                key: mtp_key(depth),
                counts,
            })
            .collect();
        let keys: Vec<ResidualKey> = window.iter().map(|depth| depth.key).collect();
        request.observe(&keys, 3, 2, false);
        let probabilities: Vec<f64> = window
            .iter()
            .map(|depth| request.probability(depth))
            .collect();
        let prices = [
            price(0, 30000.0),
            price(1, 34000.0),
            price(2, 36000.0),
            price(3, 41000.0),
        ];
        let mut best = (0usize, f64::MIN);
        for k in 0..=3usize {
            let rate = reference_expected(&probabilities[..k]) / prices[k].cost_us;
            if rate > best.1 {
                best = (k, rate);
            }
        }
        let decision = request.choose(&window, &prices, 3).expect("priced");
        let want = if best.0 == 0 {
            DerivedAction::Ar
        } else {
            DerivedAction::Prefix(best.0 as u8)
        };
        assert_eq!(decision.action, want);
        assert_eq!(
            decision.expected,
            reference_expected(&probabilities[..best.0])
        );
        assert_eq!(decision.cost_us, prices[best.0].cost_us);
    }

    #[test]
    fn derived_k_on_copy_never_falls_back_to_off_pool_prices() {
        for row in table::WINDOW_PRICES {
            assert!(fn_phase1_window_price(
                row.0,
                DerivedCostMode::OffPool,
                row.1,
                row.2,
                row.3,
                true
            )
            .is_some());
            assert_eq!(
                fn_phase1_window_price(row.0, DerivedCostMode::OnCopy, row.1, row.2, row.3, true),
                None,
                "{row:?}"
            );
        }
        for row in table::AR_PRICES {
            assert!(fn_phase1_ar_price(row.0, DerivedCostMode::OffPool, row.1).is_some());
            assert_eq!(
                fn_phase1_ar_price(row.0, DerivedCostMode::OnCopy, row.1),
                None,
                "{row:?}"
            );
        }
    }

    #[test]
    fn derived_k_frozen_prices_use_planned_confirmation_only_for_ngram() {
        for row in table::WINDOW_PRICES {
            let confirmed =
                fn_phase1_window_price(row.0, DerivedCostMode::OffPool, row.1, row.2, row.3, true);
            let unconfirmed =
                fn_phase1_window_price(row.0, DerivedCostMode::OffPool, row.1, row.2, row.3, false);
            match row.1 {
                DerivedSource::Ngram => assert_eq!(unconfirmed, None, "{row:?}"),
                DerivedSource::Mtp => assert_eq!(unconfirmed, confirmed, "{row:?}"),
            }
        }
    }

    #[test]
    fn derived_k_measured_prices_require_exact_supported_cells() {
        let row: FnWidthPriceRow = (
            "id", "Halo", "OnCopy", "Le2K", "route", "Ngram", false, 2, 123.0, 5, 3, false,
        );
        let routes = [("id", "route", "kind", "\"batched\"")];
        let key = ("Halo", "OnCopy", "Le2K");
        // Exercise the production resolver, not just the raw cell validator:
        // generic holes never inherit the historical same-card/mode profile.
        const LOAD: &str = r#"{"schema":"qwen4-measured-load.v1"}"#;
        let mut off_row = row;
        off_row.2 = "OffPool";
        let off_key = ("Halo", "OffPool", "Le2K", "Ngram", false, 2);
        let frozen = || {
            fn_phase1_ar_price(
                DerivedCard::Halo,
                DerivedCostMode::OffPool,
                ContextBucket::Le2K,
            )
        };
        assert!(frozen().is_some());
        let empty = build_measured_book_index(&[], &[], &[]);
        assert_eq!(empty.active_price(off_key, frozen), frozen());
        assert!(empty.counts_identity_supported("Halo", "OffPool", None));
        assert!(!empty.counts_identity_supported("Halo", "OffPool", Some("id")));
        let generic = build_measured_book_index(&[off_row], &routes, &[("id", LOAD)]);
        assert_eq!(generic.active_price(off_key, frozen), Some((123.0, false)));
        assert!(generic.counts_identity_supported("Halo", "OffPool", Some("id")));
        assert!(!generic.counts_identity_supported("Halo", "OffPool", None));
        assert!(!generic.counts_identity_supported("Halo", "OffPool", Some("other")));
        for missing in [
            ("Halo", "OffPool", "Le8K", "Ngram", false, 2),
            ("Halo", "OffPool", "Le2K", "Mtp", false, 1),
            ("Halo", "OffPool", "Le2K", "ArRouted", false, 0),
        ] {
            assert_eq!(generic.active_price(missing, frozen), None);
        }
        // Missing, duplicate, or mismatched descriptors cannot price even a
        // valid cell, and must not unlock the legacy fallback for absent cells.
        for loads in [
            vec![],
            vec![("other", LOAD)],
            vec![("id", LOAD), ("id", LOAD)],
            vec![("id", "{}")],
        ] {
            let unbound = build_measured_book_index(&[off_row], &routes, &loads);
            assert_eq!(unbound.active_price(off_key, frozen), None);
            assert_eq!(
                unbound.active_price(("Halo", "OffPool", "Le8K", "Mtp", false, 1), frozen),
                None
            );
            assert!(!unbound.counts_identity_supported("Halo", "OffPool", Some("id")));
        }
        let poisoned = build_measured_book_index(&[off_row, off_row], &routes, &[("id", LOAD)]);
        assert_eq!(poisoned.active_price(off_key, frozen), None);
        // This is the complete total, not a verify component plus head.
        assert_eq!(
            fn_width_price(&[row], &routes, key, "Ngram", 2, false),
            Ok(Some((123.0, false)))
        );
        assert_eq!(
            fn_width_price(&[row], &routes, key, "Ngram", 2, true),
            Ok(None)
        );
        assert_eq!(
            fn_width_price(&[row], &routes, key, "Ngram", 1, false),
            Ok(None)
        );
        assert_eq!(
            fn_width_price(
                &[row],
                &routes,
                ("Halo", "OffPool", "Le2K"),
                "Ngram",
                2,
                false
            ),
            Ok(None)
        );
        assert_eq!(
            fn_width_price(
                &[row],
                &routes,
                ("Halo", "OnCopy", "Le8K"),
                "Ngram",
                2,
                false
            ),
            Ok(None)
        );
        assert_eq!(
            fn_width_price(&[row], &routes, key, "unknown", 2, false),
            Err(())
        );
        assert_eq!(
            fn_width_price(&[row], &routes, key, "Mtp", 2, true),
            Err(())
        );
        assert_eq!(
            fn_width_price(&[row], &routes, key, "Ngram", 0, false),
            Err(())
        );

        let mut native = row;
        native.5 = "Mtp";
        assert_eq!(
            fn_width_price(&[native], &routes, key, "Mtp", 2, false),
            Ok(Some((123.0, false)))
        );
        let mut ar = row;
        ar.5 = "ArRouted";
        ar.7 = 0;
        let ar_routes = [("id", "route", "kind", "\"target-only\"")];
        assert_eq!(
            fn_width_price(&[ar], &ar_routes, key, "ArRouted", 0, false),
            Ok(Some((123.0, false)))
        );
        assert_eq!(
            fn_width_price(&[ar], &routes, key, "ArRouted", 0, false),
            Err(())
        );
        assert_eq!(
            fn_width_price(&[ar], &ar_routes, key, "ArRouted", 0, true),
            Err(())
        );
    }

    #[test]
    fn derived_k_measured_prices_reject_ambiguity_and_invalid_metadata() {
        let row: FnWidthPriceRow = (
            "id", "Halo", "OnCopy", "Le2K", "route", "Ngram", false, 2, 123.0, 5, 3, false,
        );
        let routes = [("id", "route", "kind", "\"batched\"")];
        let key = ("Halo", "OnCopy", "Le2K");
        assert_eq!(
            fn_width_price(&[row, row], &routes, key, "Ngram", 2, false),
            Err(())
        );
        let mut other_identity = row;
        other_identity.0 = "other";
        let identity_routes = [routes[0], ("other", "route", "kind", "\"batched\"")];
        assert_eq!(
            fn_width_price(
                &[row, other_identity],
                &identity_routes,
                key,
                "Ngram",
                2,
                false
            ),
            Err(())
        );
        let mut other_route = row;
        other_route.4 = "other-route";
        let route_fields = [routes[0], ("id", "other-route", "kind", "\"batched\"")];
        assert_eq!(
            fn_width_price(&[row, other_route], &route_fields, key, "Ngram", 2, false),
            Err(())
        );
        assert_eq!(fn_width_price(&[row], &[], key, "Ngram", 2, false), Err(()));
        assert_eq!(
            fn_width_price(&[row], &[routes[0], routes[0]], key, "Ngram", 2, false),
            Err(())
        );
        for kind in ["\"interleaved\"", "\"unknown\"", "batched"] {
            assert_eq!(
                fn_width_price(
                    &[row],
                    &[("id", "route", "kind", kind)],
                    key,
                    "Ngram",
                    2,
                    false
                ),
                Err(())
            );
        }
        for total in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut invalid = row;
            invalid.8 = total;
            assert_eq!(
                fn_width_price(&[invalid], &routes, key, "Ngram", 2, false),
                Err(())
            );
        }
        let mut unsupported = row;
        unsupported.9 = 0;
        let mut unqualified = row;
        unqualified.10 = 2;
        let mut pooled = row;
        pooled.11 = true;
        for invalid in [unsupported, unqualified, pooled] {
            assert_eq!(
                fn_width_price(&[invalid], &routes, key, "Ngram", 2, false),
                Err(())
            );
        }
    }

    #[test]
    fn derived_k_measured_depths_pool_only_their_own_unique_population() {
        let row: FnWidthDepthRow = ("id", "Halo", "OnCopy", "Le2K", "route", false, 1, 3, 5);
        let mut variant = row;
        variant.3 = "Le8K";
        variant.4 = "other-route";
        variant.5 = true;
        variant.7 = 2;
        variant.8 = 4;
        let rows = [row, variant];
        assert_eq!(
            fn_width_depth_counts(&rows, "Halo", "OnCopy", 1, Some("id")),
            Some(CalibrationCounts {
                depth_s: 5,
                depth_n: 9,
                cell_s: 0,
                cell_n: 0,
            })
        );
        assert_eq!(
            fn_width_depth_counts(&rows, "Halo", "OnCopy", 2, Some("id")),
            None
        );
        assert_eq!(
            fn_width_depth_counts(&rows, "Halo", "OffPool", 1, Some("id")),
            None
        );
        assert_eq!(
            fn_width_depth_counts(&rows, "Halo", "OnCopy", 0, Some("id")),
            None
        );
        assert_eq!(
            fn_width_depth_counts(&[row, row], "Halo", "OnCopy", 1, Some("id")),
            None
        );
        let mut other_identity = variant;
        other_identity.0 = "other";
        other_identity.6 = 2;
        assert_eq!(
            fn_width_depth_counts(&[row, other_identity], "Halo", "OnCopy", 1, Some("id")),
            None
        );
        let mut invalid = row;
        invalid.7 = 6;
        assert_eq!(
            fn_width_depth_counts(&[invalid], "Halo", "OnCopy", 1, Some("id")),
            None
        );
        let mut overflow = variant;
        overflow.8 = u64::MAX;
        assert_eq!(
            fn_width_depth_counts(&[row, overflow], "Halo", "OnCopy", 1, Some("id")),
            None
        );
        // Same card/mode is not the admitted measured population.
        assert_eq!(
            fn_width_depth_counts(&rows, "Halo", "OnCopy", 1, Some("q4-other-seal")),
            None
        );
    }

    #[test]
    fn derived_k_measured_calibration_preserves_frozen_ngram_and_strict_native() {
        for row in table::CELL_TOTALS {
            let frozen = fn_phase1_counts(row.0, row.1, row.2, row.3);
            assert_eq!(
                fn_measured_depth_counts(row.0, DerivedCostMode::OffPool, row.1, row.2, row.3),
                Some(frozen)
            );
        }
        // Bootstrapped measured native support is absent: OffPool falls back
        // only where the frozen depth itself has support, OnCopy never borrows
        // it, and an unconditional query never fabricates a 0.8 cell.
        for row in table::DEPTH_TOTALS {
            let frozen = fn_phase1_counts(row.0, row.1, row.2, DerivedFeature::Native);
            let off_pool = fn_measured_depth_counts(
                row.0,
                DerivedCostMode::OffPool,
                row.1,
                row.2,
                DerivedFeature::Native,
            );
            if row.1 == DerivedSource::Mtp {
                assert_eq!(off_pool, (frozen.depth_n > 0).then_some(frozen), "{row:?}");
            } else {
                assert_eq!(off_pool, Some(frozen), "{row:?}");
            }
            if row.1 == DerivedSource::Mtp {
                assert_eq!(
                    fn_measured_depth_counts(
                        row.0,
                        DerivedCostMode::OnCopy,
                        row.1,
                        row.2,
                        DerivedFeature::Native
                    ),
                    None,
                    "{row:?}"
                );
            }
        }
        // A frozen depth with no support (Halo Mtp depth 8) stays unsupported:
        // the active lookup returns None rather than a Some(zero) 0.8 cell.
        assert_eq!(
            fn_measured_depth_counts(
                DerivedCard::Halo,
                DerivedCostMode::OffPool,
                DerivedSource::Mtp,
                8,
                DerivedFeature::Native,
            ),
            None,
        );
        // Margin cells are unsupported until measured profiles land.
        assert_eq!(
            fn_measured_depth_counts(
                DerivedCard::Halo,
                DerivedCostMode::OffPool,
                DerivedSource::Mtp,
                1,
                DerivedFeature::NativeMargin {
                    observed_depth: 1,
                    margin: NativeMarginBin::GeSix,
                },
            ),
            None,
        );
        assert_eq!(
            fn_measured_window_price(
                DerivedCard::Halo,
                DerivedCostMode::OffPool,
                DerivedSource::Ngram,
                ContextBucket::Le2K,
                1,
                false,
            ),
            None
        );
    }
    #[test]
    fn derived_k_native_margin_bin_edges_reject_without_invention() {
        let edges = [
            (0.0, Some(NativeMarginBin::LtQuarter)),
            (0.24, Some(NativeMarginBin::LtQuarter)),
            (0.25, Some(NativeMarginBin::LtOnePointFive)),
            (1.49, Some(NativeMarginBin::LtOnePointFive)),
            (1.5, Some(NativeMarginBin::LtThree)),
            (2.99, Some(NativeMarginBin::LtThree)),
            (3.0, Some(NativeMarginBin::LtSix)),
            (5.99, Some(NativeMarginBin::LtSix)),
            (6.0, Some(NativeMarginBin::GeSix)),
            (f32::INFINITY, None),
            (f32::NEG_INFINITY, None),
            (f32::NAN, None),
            (-0.0, Some(NativeMarginBin::LtQuarter)),
            (-1.0, None),
        ];
        for (margin, want) in edges {
            assert_eq!(native_margin_bin(margin), want, "margin {margin}");
        }
        assert_eq!(
            [
                NativeMarginBin::LtQuarter,
                NativeMarginBin::LtOnePointFive,
                NativeMarginBin::LtThree,
                NativeMarginBin::LtSix,
                NativeMarginBin::GeSix,
            ]
            .map(NativeMarginBin::index),
            [0, 1, 2, 3, 4],
        );
        assert_eq!(NATIVE_MARGIN_BIN_COUNT, 5);
    }

    #[test]
    fn derived_k_native_literals_separate_and_censor() {
        // Bootstrap declares the typed empty tables only: no measured support.
        assert!(fn_width_cost_v1::NATIVE_DEPTH_COUNTS.is_empty());
        assert!(fn_width_cost_v1::NATIVE_MARGIN_COUNTS.is_empty());
        assert!(fn_width_cost_v1::LOAD_IDENTITIES.is_empty());
        assert_eq!(MEASURED_LOAD_SCHEMA, "qwen4-measured-load.v1");
        // Same identity/card/mode population law as prices: one identity pools
        // every context/bucket and route variant; a second identity rejects.
        let parent: [FnNativeDepthRow; 2] = [
            ("id", "Halo", "OffPool", "Le2K", "r1", 1, 3, 5),
            ("id", "Halo", "OffPool", "Le8K", "r2", 1, 2, 4),
        ];
        assert_eq!(
            fn_native_depth_counts(&parent, "Halo", "OffPool", 1),
            Ok(Some((5, 9)))
        );
        assert_eq!(
            fn_native_depth_counts(&parent, "Halo", "OnCopy", 1),
            Ok(None)
        );
        let mut other = parent[1];
        other.0 = "other";
        assert_eq!(
            fn_native_depth_counts(&[parent[0], other], "Halo", "OffPool", 1),
            Err(())
        );
        let mut bad = parent[0];
        bad.6 = 6;
        assert_eq!(
            fn_native_depth_counts(&[bad], "Halo", "OffPool", 1),
            Err(())
        );
        // Margin cells pool one identity's routes; an ambiguous identity
        // rejects; unobserved cells censor to Ok(None), never a zero census.
        let cells: [FnNativeMarginRow; 2] = [
            ("id", "Halo", "OffPool", "Le2K", "r1", 1, 4, 1, 7, 9),
            ("id", "Halo", "OffPool", "Le8K", "r2", 1, 4, 2, 5, 8),
        ];
        assert_eq!(
            fn_native_margin_counts(&cells, "Halo", "OffPool", 1, 4, 1),
            Ok(Some((7, 9)))
        );
        assert_eq!(
            fn_native_margin_counts(&cells, "Halo", "OffPool", 1, 4, 2),
            Ok(Some((5, 8)))
        );
        assert_eq!(
            fn_native_margin_counts(&cells, "Halo", "OffPool", 1, 0, 1),
            Ok(None)
        );
        assert_eq!(
            fn_native_margin_counts(&cells, "Halo", "OffPool", 2, 4, 1),
            Ok(None)
        );
        assert_eq!(
            fn_native_margin_counts(&cells, "Halo", "OnCopy", 1, 4, 1),
            Ok(None)
        );
        assert_eq!(
            fn_native_margin_counts(&cells, "Halo", "OffPool", 1, 5, 1),
            Err(())
        );
        assert_eq!(
            fn_native_margin_counts(&cells, "Halo", "OffPool", 0, 4, 1),
            Err(())
        );
        let mut dup = cells;
        dup[1] = cells[0];
        assert_eq!(
            fn_native_margin_counts(&dup, "Halo", "OffPool", 1, 4, 1),
            Err(())
        );
    }
    #[test]
    fn derived_k_native_index_equals_slice_validators() {
        // Focused index-vs-slice equality on genuine literal slices: the
        // once-built index returns exactly the slice validators' outcomes.
        fn parent_outcome(
            parents: &[FnNativeDepthRow],
            card: &str,
            mode: &str,
            depth: usize,
        ) -> Result<Option<(u64, u64)>, ()> {
            fn_native_depth_counts(parents, card, mode, depth)
        }
        fn parent_index(
            parents: &[FnNativeDepthRow],
            card: u8,
            mode: u8,
            depth: u8,
        ) -> Result<Option<(u64, u64)>, ()> {
            let index = build_native_counts_index(parents, &[]);
            match query_native_counts_index(
                &index,
                card,
                mode,
                depth,
                NATIVE_MARGIN_BIN_COUNT as u8,
                depth,
            ) {
                NativeCountsOutcome::Counts(s, n) => Ok(Some((s, n))),
                NativeCountsOutcome::Absent => Ok(None),
                NativeCountsOutcome::NoSupport | NativeCountsOutcome::Invalid => Err(()),
            }
        }
        // Halo = card 1, OffPool = mode 0.
        let parents: [FnNativeDepthRow; 3] = [
            ("id", "Halo", "OffPool", "Le2K", "r1", 1, 3, 5),
            ("id", "Halo", "OffPool", "Le8K", "r2", 1, 2, 4),
            ("id", "Halo", "OffPool", "Le2K", "r1", 2, 6, 6),
        ];
        for depth in [1u8, 2, 3] {
            assert_eq!(
                parent_index(&parents, 1, 0, depth),
                parent_outcome(&parents, "Halo", "OffPool", usize::from(depth)),
                "depth {depth}",
            );
        }
        // Identity present but depth unmeasured: both read `Err` (never the
        // OffPool fallback); absent card+mode: both read `Ok(None)`.
        assert_eq!(parent_index(&parents, 1, 0, 3), Err(()));
        assert_eq!(parent_outcome(&parents, "Halo", "OffPool", 3), Err(()));
        assert_eq!(parent_index(&parents, 1, 1, 1), Ok(None));
        assert_eq!(parent_outcome(&parents, "Halo", "OnCopy", 1), Ok(None));
        assert_eq!(parent_outcome(&[], "Halo", "OnCopy", 1), Ok(None));
        assert_eq!(parent_index(&[], 1, 1, 1), Ok(None));
        // Second identity poisons the slot on both paths.
        let mut split = parents;
        split[2].0 = "other";
        assert_eq!(parent_outcome(&split, "Halo", "OffPool", 1), Err(()));
        assert_eq!(parent_index(&split, 1, 0, 1), Err(()));
        // Invalid row poisons the slot on both paths.
        let mut invalid = parents;
        invalid[0].6 = 9;
        assert_eq!(parent_outcome(&invalid, "Halo", "OffPool", 1), Err(()));
        assert_eq!(parent_index(&invalid, 1, 0, 1), Err(()));
        // Margin cells: index-vs-slice equality across support, censoring,
        // and rejection.
        fn margin_index(
            cells: &[FnNativeMarginRow],
            observed: u8,
            bin: usize,
            target: u8,
        ) -> Result<Option<(u64, u64)>, ()> {
            if observed == 0
                || usize::from(observed) > DERIVED_MAX_DRAFTS
                || target == 0
                || usize::from(target) > DERIVED_MAX_DRAFTS
                || bin >= NATIVE_MARGIN_BIN_COUNT
            {
                return Err(());
            }
            let index = build_native_counts_index(&[], cells);
            match query_native_counts_index(&index, 1, 0, observed, bin as u8, target) {
                NativeCountsOutcome::Counts(s, n) => Ok(Some((s, n))),
                NativeCountsOutcome::Absent | NativeCountsOutcome::NoSupport => Ok(None),
                NativeCountsOutcome::Invalid => Err(()),
            }
        }
        let cells: [FnNativeMarginRow; 2] = [
            ("id", "Halo", "OffPool", "Le2K", "r1", 1, 4, 1, 7, 9),
            ("id", "Halo", "OffPool", "Le8K", "r2", 1, 4, 2, 5, 8),
        ];
        for (observed, bin, target) in [(1u8, 4usize, 1u8), (1, 4, 2), (1, 0, 1), (2, 4, 1)] {
            assert_eq!(
                margin_index(&cells, observed, bin, target),
                fn_native_margin_counts(
                    &cells,
                    "Halo",
                    "OffPool",
                    observed as usize,
                    bin,
                    target as usize
                ),
                "observed {observed} bin {bin} target {target}",
            );
        }
        // Out-of-range bin rejects on both paths; ambiguous identity errs.
        assert_eq!(margin_index(&cells, 1, 5, 1), Err(()));
        assert_eq!(
            fn_native_margin_counts(&cells, "Halo", "OffPool", 1, 5, 1),
            Err(())
        );
        let mut dup = cells;
        dup[1] = cells[0];
        assert_eq!(
            fn_native_margin_counts(&dup, "Halo", "OffPool", 1, 4, 1),
            Err(())
        );
        assert_eq!(margin_index(&dup, 1, 4, 1), Err(()));
        // Exact duplicate full key poisons; legit pool (distinct route)
        // aggregates the numeric cell on both paths.
        let mut exact_dup = cells;
        exact_dup[1] = ("id", "Halo", "OffPool", "Le2K", "r1", 1, 4, 1, 7, 9);
        assert_eq!(
            fn_native_margin_counts(&exact_dup, "Halo", "OffPool", 1, 4, 1),
            Err(())
        );
        assert_eq!(margin_index(&exact_dup, 1, 4, 1), Err(()));
        let pooled: [FnNativeMarginRow; 2] = [
            ("id", "Halo", "OffPool", "Le2K", "r1", 1, 4, 1, 7, 9),
            ("id", "Halo", "OffPool", "Le2K", "r2", 1, 4, 1, 1, 2),
        ];
        assert_eq!(
            fn_native_margin_counts(&pooled, "Halo", "OffPool", 1, 4, 1),
            Ok(Some((8, 11))),
        );
        assert_eq!(margin_index(&pooled, 1, 4, 1), Ok(Some((8, 11))));
        // Same for parents: exact duplicate full key poisons, distinct
        // bucket/route rows pool.
        let mut parent_dup = parents;
        parent_dup[1] = parent_dup[0];
        assert_eq!(
            fn_native_depth_counts(&parent_dup, "Halo", "OffPool", 1),
            Err(())
        );
        assert_eq!(parent_index(&parent_dup, 1, 0, 1), Err(()));
        // The shipping literal index is sorted and healthy-or-absent; the
        // bootstrap book is empty so every slot reads absent.
        let shipping = native_counts_index();
        let mut parents_sorted = shipping.parents.clone();
        parents_sorted.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(shipping.parents, parents_sorted);
        let mut margins_sorted = shipping.margins.clone();
        margins_sorted.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(shipping.margins, margins_sorted);
        for slot in 0..4 {
            assert!(
                shipping.health[slot] == NativeSlotHealth::Absent
                    || shipping.health[slot] == NativeSlotHealth::Healthy,
                "slot {slot}"
            );
        }
    }

    #[test]
    fn derived_k_observe_with_margins_ages_once_and_censors() {
        fn margin_key(depth: u8, observed: u8, bin: NativeMarginBin) -> ResidualKey {
            ResidualKey {
                source: DerivedSource::Mtp,
                depth,
                feature: DerivedFeature::NativeMargin {
                    observed_depth: observed,
                    margin: bin,
                },
            }
        }
        let keys: Vec<ResidualKey> = (1..=3).map(mtp_key).collect();
        let margins = [
            Some(NativeMarginBin::GeSix),
            Some(NativeMarginBin::LtQuarter),
            None,
        ];
        // Accepted 2 of 3, open window: base cells plus one margin success per
        // reached depth and one margin failure at the first rejection.
        let mut margined = DerivedRequest::new(DerivedCard::R9700);
        margined.observe_with_margins(&keys, &margins, 3, 2, false);
        let mut plain = DerivedRequest::new(DerivedCard::R9700);
        plain.observe(&keys, 3, 2, false);
        for depth in 1..=3u8 {
            let base = DerivedDepth {
                key: mtp_key(depth),
                counts: CalibrationCounts::default(),
            };
            assert_eq!(
                margined.probability(&base),
                plain.probability(&base),
                "base depth {depth}"
            );
        }
        let accepted_once = (2.0 * 0.8 + 1.0) / (2.0 + 1.0);
        let rejected_once = (2.0 * 0.8) / (2.0 + 1.0);
        for observed in [1u8, 2] {
            for (depth, want) in [(1u8, accepted_once), (2, accepted_once), (3, rejected_once)] {
                for bin in [NativeMarginBin::GeSix, NativeMarginBin::LtQuarter] {
                    let at = observed as usize - 1;
                    let seen = margins[at] == Some(bin);
                    let cell = DerivedDepth {
                        key: margin_key(depth, observed, bin),
                        counts: CalibrationCounts::default(),
                    };
                    let probability = margined.probability(&cell);
                    if seen {
                        assert_eq!(
                            probability, want,
                            "depth {depth} observed {observed} bin {bin:?}"
                        );
                    } else {
                        assert_eq!(
                            probability, 0.8,
                            "depth {depth} observed {observed} bin {bin:?}"
                        );
                    }
                }
            }
        }
        // Row 3 carried None, so it owns no margin cell at all.
        for depth in 1..=3u8 {
            let cell = DerivedDepth {
                key: margin_key(depth, 3, NativeMarginBin::GeSix),
                counts: CalibrationCounts::default(),
            };
            assert_eq!(margined.probability(&cell), 0.8, "depth {depth}");
        }
        // Terminal acceptance censors the rejection: reached depths succeed,
        // deeper and unobserved rejections stay uncensored priors.
        let mut terminal = DerivedRequest::new(DerivedCard::R9700);
        terminal.observe_with_margins(&keys, &[Some(NativeMarginBin::LtThree)], 2, 2, true);
        for depth in 1..=2u8 {
            let cell = DerivedDepth {
                key: margin_key(depth, 1, NativeMarginBin::LtThree),
                counts: CalibrationCounts::default(),
            };
            assert_eq!(terminal.probability(&cell), accepted_once, "depth {depth}");
        }
        let censored = DerivedDepth {
            key: margin_key(3, 1, NativeMarginBin::LtThree),
            counts: CalibrationCounts::default(),
        };
        assert_eq!(terminal.probability(&censored), 0.8);
        // Empty and all-None margins are exactly `observe`, including a single
        // aging pass and one `last_first` write.
        for probe in [vec![], vec![None, None, None]] {
            let mut probe_request = DerivedRequest::new(DerivedCard::R9700);
            probe_request.observe_with_margins(&keys, &probe, 3, 2, false);
            assert_eq!(probe_request.cells, plain.cells);
            assert_eq!(
                probe_request.last_first(DerivedSource::Mtp),
                plain.last_first(DerivedSource::Mtp)
            );
        }
        // A longer margin tail is ignored; a shorter tail is a None tail; a
        // zero-execution window only ages, as `observe` does.
        let long = [Some(NativeMarginBin::GeSix); 9];
        let mut long_request = DerivedRequest::new(DerivedCard::R9700);
        long_request.observe_with_margins(&keys, &long, 1, 1, false);
        let mut short_request = DerivedRequest::new(DerivedCard::R9700);
        short_request.observe_with_margins(&keys, &[Some(NativeMarginBin::GeSix)], 3, 1, false);
        let short_unseen = DerivedDepth {
            key: margin_key(1, 2, NativeMarginBin::GeSix),
            counts: CalibrationCounts::default(),
        };
        assert_eq!(short_request.probability(&short_unseen), 0.8);
        let mut aged = DerivedRequest::new(DerivedCard::R9700);
        aged.observe(&keys, 1, 1, false);
        let before = aged.cells.clone();
        aged.observe_with_margins(&keys, &[Some(NativeMarginBin::GeSix)], 0, 0, false);
        for (cell, was) in aged.cells.iter().zip(&before) {
            assert_eq!((cell.1, cell.2), (was.1 * 0.875, was.2 * 0.875));
        }
    }

    #[test]
    fn derived_k_physical_depths_are_not_capped_at_the_floor_maximum() {
        assert_eq!(DERIVED_MAX_DRAFTS, 63);
        assert!(DERIVED_MAX_DRAFTS > MAX_DEPTH);
        let mut request = DerivedRequest::with_max_drafts(DerivedCard::R9700, DERIVED_MAX_DRAFTS);
        let reserved = request.cells.capacity();
        assert!(
            reserved
                >= DERIVED_MAX_DRAFTS
                    + DERIVED_MAX_DRAFTS * DERIVED_MAX_DRAFTS * NATIVE_MARGIN_BIN_COUNT
                    + DERIVED_NGRAM_CELL_ALLOWANCE
        );
        // A flat price makes the deepest prefix the global best.
        let window = depths(DERIVED_MAX_DRAFTS as u8);
        let flat: Vec<DerivedPrice> = (0..=DERIVED_MAX_DRAFTS as u8)
            .map(|k| price(k, 1000.0))
            .collect();
        let decision = request.choose(&window, &flat, 10).expect("priced");
        assert_eq!(
            decision.action,
            DerivedAction::Prefix(DERIVED_MAX_DRAFTS as u8)
        );
        assert_eq!(
            decision.expected,
            reference_expected(&[0.8; DERIVED_MAX_DRAFTS])
        );
        // Every physical depth owns its residual cell; none aliases depth 10.
        let keys: Vec<ResidualKey> = (1..=DERIVED_MAX_DRAFTS as u8).map(mtp_key).collect();
        request.observe(&keys, DERIVED_MAX_DRAFTS as u8, 40, false);
        assert_eq!(request.cells.capacity(), reserved);
        let accepted_once = (1.6 + 1.0) / 3.0;
        for depth in [1u8, 10, 11, 40] {
            assert_eq!(
                request.probability(&unsupported(depth)),
                accepted_once,
                "depth {depth}"
            );
        }
        assert_eq!(request.probability(&unsupported(41)), 1.6 / 3.0);
        for depth in [42u8, 63, 64, 200] {
            assert_eq!(
                request.probability(&unsupported(depth)),
                0.8,
                "depth {depth}"
            );
        }
    }

    /// Asserts `rows` is strictly increasing by `key` (sorted, no duplicates).
    fn assert_strictly_sorted<T, K: Ord + std::fmt::Debug>(
        name: &str,
        rows: &[T],
        key: impl Fn(&T) -> K,
    ) {
        assert!(!rows.is_empty(), "{name} is empty");
        for pair in rows.windows(2) {
            let (a, b) = (key(&pair[0]), key(&pair[1]));
            assert!(a < b, "{name} not strictly sorted: {a:?} !< {b:?}");
        }
    }

    #[test]
    fn derived_k_generated_arrays_are_strictly_sorted() {
        assert_strictly_sorted("DEPTH_TOTALS", table::DEPTH_TOTALS, |r| (r.0, r.1, r.2));
        assert_strictly_sorted("CELL_TOTALS", table::CELL_TOTALS, |r| (r.0, r.1, r.2, r.3));
        assert_strictly_sorted("WINDOW_PRICES", table::WINDOW_PRICES, |r| {
            (r.0, r.1, r.2, r.3)
        });
        assert_strictly_sorted("AR_PRICES", table::AR_PRICES, |r| (r.0, r.1));
        for row in table::WINDOW_PRICES {
            assert!(row.4.is_finite() && row.4 > 0.0, "window price {row:?}");
            assert!(row.3 >= 1 && row.5 > 0, "window price {row:?}");
        }
        for row in table::AR_PRICES {
            assert!(
                row.2.is_finite() && row.2 > 0.0 && row.3 > 0,
                "AR price {row:?}"
            );
        }
        for row in table::DEPTH_TOTALS {
            assert!(row.2 >= 1 && row.3 <= row.4, "depth totals {row:?}");
        }
        for row in table::CELL_TOTALS {
            assert!(row.2 >= 1 && row.4 <= row.5, "cell totals {row:?}");
            let native = row.3 == DerivedFeature::Native;
            assert_eq!(native, row.1 == DerivedSource::Mtp, "cell totals {row:?}");
        }
    }

    #[test]
    fn derived_k_generated_table_identity() {
        assert_eq!(table::VERSION, FN_DERIVED_VERSION);
        assert_eq!(table::SCHEMA, 1);
    }

    #[test]
    fn derived_k_lookups_return_the_generated_rows() {
        for row in table::DEPTH_TOTALS {
            let counts = fn_phase1_counts(row.0, row.1, row.2, DerivedFeature::Native);
            assert_eq!((counts.depth_s, counts.depth_n), (row.3, row.4));
        }
        for row in table::CELL_TOTALS {
            let counts = fn_phase1_counts(row.0, row.1, row.2, row.3);
            assert_eq!((counts.cell_s, counts.cell_n), (row.4, row.5), "{row:?}");
        }
        for row in table::WINDOW_PRICES {
            assert_eq!(
                fn_phase1_window_price(row.0, DerivedCostMode::OffPool, row.1, row.2, row.3, true),
                Some((row.4, row.6)),
                "{row:?}"
            );
        }
        for row in table::AR_PRICES {
            assert_eq!(
                fn_phase1_ar_price(row.0, DerivedCostMode::OffPool, row.1),
                Some((row.2, row.4)),
                "{row:?}"
            );
        }
        // Absent support is zeros / unsupported, never interpolated.
        let absent = DerivedFeature::Ngram {
            offered: OfferBin::ThirtyTwoPlus,
            key_len: Some(u32::MAX),
            last_first: None,
        };
        assert_eq!(
            fn_phase1_counts(DerivedCard::R9700, DerivedSource::Ngram, u8::MAX, absent),
            CalibrationCounts::default()
        );
        assert_eq!(
            fn_phase1_window_price(
                DerivedCard::Halo,
                DerivedCostMode::OffPool,
                DerivedSource::Mtp,
                ContextBucket::Le2K,
                u8::MAX,
                true,
            ),
            None
        );
    }

    // ----- CPU oracle: the chooser against replay2's recorded decisions -----

    type Group = (DerivedCard, bool, DerivedSource);
    type Count = (i64, i64);
    type DepthCounts = HashMap<(Group, u8), Count>;
    type CellCounts = HashMap<(Group, u8, DerivedFeature), Count>;

    /// Full calibration totals and each request's own contribution.
    #[derive(Default)]
    struct OracleCalibration {
        depth: DepthCounts,
        cell: CellCounts,
        deltas: HashMap<String, (DepthCounts, CellCounts)>,
    }

    #[derive(Debug, Default)]
    struct OracleStats {
        requests: usize,
        synthetic_requests: usize,
        windows: usize,
        /// Windows replay2 skipped before choosing (`emitted <= 0`).
        skipped: usize,
        decisions: usize,
        ar: usize,
        prefix: usize,
        none: usize,
        /// Decisions whose action is not "the incumbent width".
        changed: usize,
    }

    /// Totals minus the held-out request's delta (replay2 `Calibration._loo`).
    fn loo(total: Count, delta: Count, what: &str) -> (u64, u64) {
        let (s, n) = (total.0 - delta.0, total.1 - delta.1);
        assert!(
            0 <= s && s <= n,
            "contract error: {what}: totals {total:?} minus held-out delta {delta:?}"
        );
        (s as u64, n as u64)
    }

    fn field<'a>(value: &'a Value, name: &str, at: &str) -> &'a Value {
        value
            .get(name)
            .unwrap_or_else(|| panic!("fixture {at}: missing field `{name}`"))
    }

    fn array<'a>(value: &'a Value, at: &str) -> &'a [Value] {
        value
            .as_array()
            .unwrap_or_else(|| panic!("fixture {at}: not an array: {value}"))
    }

    fn as_u64(value: &Value, at: &str) -> u64 {
        value
            .as_u64()
            .unwrap_or_else(|| panic!("fixture {at}: not an unsigned integer: {value}"))
    }

    fn as_u8(value: &Value, at: &str) -> u8 {
        u8::try_from(as_u64(value, at)).unwrap_or_else(|_| panic!("fixture {at}: > u8: {value}"))
    }

    fn as_i64(value: &Value, at: &str) -> i64 {
        value
            .as_i64()
            .unwrap_or_else(|| panic!("fixture {at}: not an integer: {value}"))
    }

    fn as_bool(value: &Value, at: &str) -> bool {
        value
            .as_bool()
            .unwrap_or_else(|| panic!("fixture {at}: not a bool: {value}"))
    }

    fn as_str<'a>(value: &'a Value, at: &str) -> &'a str {
        value
            .as_str()
            .unwrap_or_else(|| panic!("fixture {at}: not a string: {value}"))
    }

    fn parse_card(value: &Value, at: &str) -> DerivedCard {
        match as_str(value, at) {
            "R9700" => DerivedCard::R9700,
            "Halo" => DerivedCard::Halo,
            other => panic!("fixture {at}: unknown card {other:?}"),
        }
    }

    fn parse_source(value: &Value, at: &str) -> DerivedSource {
        match as_str(value, at) {
            "mtp" => DerivedSource::Mtp,
            "ngram" => DerivedSource::Ngram,
            other => panic!("fixture {at}: unknown source {other:?}"),
        }
    }

    fn parse_offer_bin(value: &Value, at: &str) -> OfferBin {
        match as_str(value, at) {
            "na" => OfferBin::Missing,
            "0" => OfferBin::Zero,
            "1" => OfferBin::One,
            "2-3" => OfferBin::TwoThree,
            "4-7" => OfferBin::FourSeven,
            "8-15" => OfferBin::EightFifteen,
            "16-31" => OfferBin::SixteenThirtyOne,
            "32+" => OfferBin::ThirtyTwoPlus,
            other => panic!("fixture {at}: unknown offer bin {other:?}"),
        }
    }

    /// replay2's `na` is JSON null.
    fn parse_opt_u32(value: &Value, at: &str) -> Option<u32> {
        if value.is_null() {
            return None;
        }
        Some(u32::try_from(as_u64(value, at)).unwrap_or_else(|_| panic!("fixture {at}: > u32")))
    }

    fn parse_opt_usize(value: &Value, at: &str) -> Option<usize> {
        if value.is_null() {
            return None;
        }
        Some(as_u64(value, at) as usize)
    }

    fn parse_opt_flag(value: &Value, at: &str) -> Option<bool> {
        match value {
            Value::Null => None,
            Value::Bool(flag) => Some(*flag),
            other => match other.as_u64() {
                Some(0) => Some(false),
                Some(1) => Some(true),
                _ => panic!("fixture {at}: not null/0/1: {other}"),
            },
        }
    }

    fn parse_feature(value: &Value, at: &str) -> DerivedFeature {
        if value.as_str() == Some("na") {
            return DerivedFeature::Native;
        }
        DerivedFeature::Ngram {
            offered: parse_offer_bin(field(value, "offered", at), at),
            key_len: parse_opt_u32(field(value, "key_len", at), at),
            last_first: parse_opt_flag(field(value, "last_first", at), at),
        }
    }

    fn parse_group(row: &[Value], at: &str) -> Group {
        (
            parse_card(&row[0], at),
            as_bool(&row[1], at),
            parse_source(&row[2], at),
        )
    }

    /// `[card, sampled, source, depth, s, n]`.
    fn parse_depth_row(row: &Value, at: &str) -> ((Group, u8), Count) {
        let row = array(row, at);
        assert_eq!(row.len(), 6, "fixture {at}: depth row {row:?}");
        (
            (parse_group(row, at), as_u8(&row[3], at)),
            (as_i64(&row[4], at), as_i64(&row[5], at)),
        )
    }

    /// `[card, sampled, source, depth, feature, s, n]`.
    fn parse_cell_row(row: &Value, at: &str) -> ((Group, u8, DerivedFeature), Count) {
        let row = array(row, at);
        assert_eq!(row.len(), 7, "fixture {at}: cell row {row:?}");
        (
            (
                parse_group(row, at),
                as_u8(&row[3], at),
                parse_feature(&row[4], at),
            ),
            (as_i64(&row[5], at), as_i64(&row[6], at)),
        )
    }

    fn insert_unique<K: std::hash::Hash + Eq + std::fmt::Debug>(
        map: &mut HashMap<K, Count>,
        key: K,
        count: Count,
        at: &str,
    ) {
        let shown = format!("{key:?}");
        assert!(
            map.insert(key, count).is_none(),
            "fixture {at}: duplicate row {shown}"
        );
    }

    fn load_calibration(root: &Value) -> OracleCalibration {
        let section = field(root, "calibration", "root");
        let mut out = OracleCalibration::default();
        for row in array(
            field(section, "depth_totals", "calibration"),
            "depth_totals",
        ) {
            let (key, count) = parse_depth_row(row, "depth_totals");
            insert_unique(&mut out.depth, key, count, "depth_totals");
        }
        for row in array(field(section, "cell_totals", "calibration"), "cell_totals") {
            let (key, count) = parse_cell_row(row, "cell_totals");
            insert_unique(&mut out.cell, key, count, "cell_totals");
        }
        let deltas = field(section, "request_deltas", "calibration");
        for entry in array(deltas, "request_deltas") {
            let id = as_str(field(entry, "id", "request_deltas"), "delta id").to_owned();
            let at = format!("request_deltas[{id}]");
            let mut depth = DepthCounts::new();
            for row in array(field(entry, "depth", &at), &at) {
                let (key, count) = parse_depth_row(row, &at);
                insert_unique(&mut depth, key, count, &at);
            }
            let mut cell = CellCounts::new();
            for row in array(field(entry, "cell", &at), &at) {
                let (key, count) = parse_cell_row(row, &at);
                insert_unique(&mut cell, key, count, &at);
            }
            assert!(
                out.deltas.insert(id.clone(), (depth, cell)).is_none(),
                "fixture: duplicate request delta id {id}"
            );
        }
        out
    }

    /// replay2 `feature_key` for FN: native has no margin at query or update;
    /// n-gram is (offer bin, key_len, executed same-source last_first).
    fn feature_for(
        source: DerivedSource,
        offered: Option<usize>,
        key_len: Option<u32>,
        last_first: Option<bool>,
    ) -> DerivedFeature {
        match source {
            DerivedSource::Mtp => DerivedFeature::Native,
            DerivedSource::Ngram => DerivedFeature::Ngram {
                offered: derived_offer_bin(offered),
                key_len,
                last_first,
            },
        }
    }

    fn keys_for(source: DerivedSource, depths: u8, feature: DerivedFeature) -> Vec<ResidualKey> {
        (1..=depths)
            .map(|depth| ResidualKey {
                source,
                depth,
                feature,
            })
            .collect()
    }

    /// `[[depth, feature], ...]` debug keys.
    fn parse_debug_keys(value: &Value, source: DerivedSource, at: &str) -> Vec<ResidualKey> {
        array(value, at)
            .iter()
            .map(|entry| {
                let entry = array(entry, at);
                assert_eq!(entry.len(), 2, "fixture {at}: debug key {entry:?}");
                ResidualKey {
                    source,
                    depth: as_u8(&entry[0], at),
                    feature: parse_feature(&entry[1], at),
                }
            })
            .collect()
    }

    fn parse_prices(value: &Value, at: &str) -> Vec<DerivedPrice> {
        array(value, at)
            .iter()
            .map(|entry| {
                let entry = array(entry, at);
                assert!(entry.len() >= 2, "fixture {at}: price {entry:?}");
                let shown = entry[1]
                    .as_f64()
                    .unwrap_or_else(|| panic!("fixture {at}: price {entry:?}"));
                // JSON floats are not guaranteed correctly rounded here; the
                // exporter adds the exact bits as a third element.
                let cost_us = match entry.get(2) {
                    Some(bits) => {
                        let exact = f64::from_bits(as_u64(bits, at));
                        assert!(
                            (exact - shown).abs() <= 1e-9 * exact.abs(),
                            "fixture {at}: price bits {exact} vs {shown}"
                        );
                        exact
                    }
                    None => shown,
                };
                DerivedPrice {
                    k: as_u8(&entry[0], at),
                    cost_us,
                }
            })
            .collect()
    }

    /// Replays one request through a fresh `DerivedRequest` and asserts the
    /// chosen action equals replay2's at every window.
    fn replay_oracle_request(
        calibration: &OracleCalibration,
        request: &Value,
        stats: &mut OracleStats,
    ) {
        let id = as_str(field(request, "id", "request"), "request id").to_owned();
        let card = parse_card(field(request, "card", &id), &id);
        let sampled = as_bool(field(request, "sampled", &id), &id);
        let delta = calibration.deltas.get(&id);
        let mut state = DerivedRequest::new(card);
        for window in array(field(request, "windows", &id), &id) {
            stats.windows += 1;
            let ordinal = as_u64(field(window, "ordinal", &id), &id);
            let source = parse_source(field(window, "source", &id), &id);
            let at = format!("request {id} window {ordinal} source {source:?}");
            let update = field(window, "update", &at);

            // replay2 skips emitted <= 0 before pricing or choosing anything.
            if as_i64(field(window, "emitted", &at), &at) <= 0 {
                assert!(update.is_null(), "{at}: skipped window carries an update");
                stats.skipped += 1;
                continue;
            }

            let width = as_u64(field(window, "width", &at), &at) as usize;
            let legal = as_u8(field(window, "legal_width", &at), &at);
            let offered = parse_opt_usize(field(window, "offered", &at), &at);
            let key_len = parse_opt_u32(field(window, "key_len", &at), &at);
            let terminal = as_bool(field(window, "terminal", &at), &at);
            let incumbent = as_u8(field(window, "incumbent", &at), &at);
            let prices = parse_prices(field(window, "prices", &at), &at);
            let expected = {
                let action = field(window, "expected_action", &at);
                if action.is_null() {
                    None
                } else {
                    Some(match as_u8(action, &at) {
                        0 => DerivedAction::Ar,
                        k => DerivedAction::Prefix(k),
                    })
                }
            };

            // Query: n-gram offered defaults to the logged width.
            let last_first = state.last_first(source);
            let query_feature = feature_for(source, offered.or(Some(width)), key_len, last_first);
            let query_keys = keys_for(source, legal, query_feature);
            let group = (card, sampled, source);
            let depths: Vec<DerivedDepth> = query_keys
                .iter()
                .map(|key| {
                    let depth_key = (group, key.depth);
                    let cell_key = (group, key.depth, key.feature);
                    let depth_total = *calibration.depth.get(&depth_key).unwrap_or_else(|| {
                        panic!("contract error: {at}: depth_totals lacks {depth_key:?}")
                    });
                    let cell_total = *calibration.cell.get(&cell_key).unwrap_or_else(|| {
                        panic!("contract error: {at}: cell_totals lacks {cell_key:?}")
                    });
                    let depth_delta = delta
                        .and_then(|d| d.0.get(&depth_key))
                        .copied()
                        .unwrap_or((0, 0));
                    let cell_delta = delta
                        .and_then(|d| d.1.get(&cell_key))
                        .copied()
                        .unwrap_or((0, 0));
                    let (depth_s, depth_n) =
                        loo(depth_total, depth_delta, &format!("{at} {depth_key:?}"));
                    let (cell_s, cell_n) =
                        loo(cell_total, cell_delta, &format!("{at} {cell_key:?}"));
                    DerivedDepth {
                        key: *key,
                        counts: CalibrationCounts {
                            depth_s,
                            depth_n,
                            cell_s,
                            cell_n,
                        },
                    }
                })
                .collect();

            let debug = window.get("debug");
            if let Some(keys) = debug.and_then(|d| d.get("query_keys")) {
                assert_eq!(
                    parse_debug_keys(keys, source, &at),
                    query_keys,
                    "{at}: reconstructed query keys differ from replay2's"
                );
            }
            if let Some(flag) = debug.and_then(|d| d.get("last_first")) {
                assert_eq!(
                    parse_opt_flag(flag, &at),
                    last_first,
                    "{at}: last_first before the window"
                );
            }

            let decision = state.choose(&depths, &prices, incumbent);
            let action = decision.map(|d| d.action);
            assert_eq!(
                action, expected,
                "{at}: action mismatch: K {legal} incumbent {incumbent} prices {prices:?} \
                 query keys {query_keys:?} decision {decision:?}"
            );
            stats.decisions += 1;
            match action {
                None => stats.none += 1,
                Some(DerivedAction::Ar) => stats.ar += 1,
                Some(DerivedAction::Prefix(_)) => stats.prefix += 1,
            }
            if action != Some(DerivedAction::Prefix(incumbent)) {
                stats.changed += 1;
            }

            // Update: the executed width is the chosen K (0 = AR / unsupported);
            // a missing offer falls back to that executed K in the update feature.
            let chosen = match action {
                Some(DerivedAction::Prefix(k)) => k,
                Some(DerivedAction::Ar) | None => 0,
            };
            if update.is_null() {
                continue;
            }
            let executed = as_u8(field(update, "width", &at), &at);
            assert_eq!(
                executed, chosen,
                "{at}: update width differs from the action"
            );
            let accepted = as_u64(field(update, "accepted", &at), &at) as usize;
            let update_feature = feature_for(
                source,
                offered.or(Some(usize::from(chosen))),
                key_len,
                last_first,
            );
            let update_keys = keys_for(source, chosen, update_feature);
            if let Some(keys) = debug.and_then(|d| d.get("update_keys")) {
                assert_eq!(
                    parse_debug_keys(keys, source, &at),
                    update_keys,
                    "{at}: reconstructed update keys differ from replay2's"
                );
            }
            state.observe(&update_keys, chosen, accepted, terminal);
        }
    }

    /// Drives every request (and any synthetic case) of a schema-1 oracle
    /// fixture through the production chooser, asserting action equality.
    fn replay_fixture(text: &str) -> OracleStats {
        let root: Value = serde_json::from_str(text).expect("oracle fixture is not JSON");
        assert_eq!(
            root["schema"].as_u64(),
            Some(1),
            "oracle fixture schema != 1"
        );
        assert_eq!(
            root["version"].as_str(),
            Some(FN_DERIVED_VERSION),
            "oracle fixture version"
        );
        assert_eq!(
            root["policy"].as_str(),
            Some("derived-causal"),
            "oracle fixture policy"
        );
        let calibration = load_calibration(&root);
        let mut stats = OracleStats::default();
        for request in array(field(&root, "requests", "root"), "requests") {
            replay_oracle_request(&calibration, request, &mut stats);
            stats.requests += 1;
        }
        if let Some(synthetic) = root.get("synthetic").and_then(Value::as_array) {
            for request in synthetic {
                replay_oracle_request(&calibration, request, &mut stats);
                stats.synthetic_requests += 1;
            }
        }
        stats
    }

    #[test]
    fn derived_k_replays_replay2_fixture() {
        let stats = replay_fixture(include_str!("../tests/fixtures/fn_derived_k_v1.json"));
        assert!(stats.requests > 0 && stats.decisions > 0, "{stats:?}");
    }
    #[test]
    fn derived_k_margin_evidence_updates_every_reached_depth_once() {
        // One committed margin observation is evidence about the whole draft
        // prefix: it must move every reached target depth, not only j + 1,
        // through exactly one extra residual bump each (single aging pass,
        // no repeated residual aging).
        let keys: Vec<ResidualKey> = (1..=4).map(mtp_key).collect();
        let margins = [None, Some(NativeMarginBin::GeSix), None, None];
        let mut request = DerivedRequest::new(DerivedCard::R9700);
        request.observe_with_margins(&keys, &margins, 4, 3, false);
        let accepted = (2.0 * 0.8 + 1.0) / (2.0 + 1.0);
        let rejected = (2.0 * 0.8) / (2.0 + 1.0);
        for (depth, want) in [(1u8, accepted), (2, accepted), (3, accepted), (4, rejected)] {
            let cell = DerivedDepth {
                key: ResidualKey {
                    source: DerivedSource::Mtp,
                    depth,
                    feature: DerivedFeature::NativeMargin {
                        observed_depth: 2,
                        margin: NativeMarginBin::GeSix,
                    },
                },
                counts: CalibrationCounts::default(),
            };
            assert_eq!(request.probability(&cell), want, "target depth {depth}");
        }
        // No future evidence: a margin observed at row 4 never labels a
        // strictly deeper target, and the base keys stay cohort-separated
        // from the margin cells.
        let future = DerivedDepth {
            key: ResidualKey {
                source: DerivedSource::Mtp,
                depth: 4,
                feature: DerivedFeature::NativeMargin {
                    observed_depth: 4,
                    margin: NativeMarginBin::LtQuarter,
                },
            },
            counts: CalibrationCounts::default(),
        };
        assert_eq!(request.probability(&future), 0.8);
        let mut base_only = DerivedRequest::new(DerivedCard::R9700);
        base_only.observe(&keys, 4, 3, false);
        for depth in 1..=4u8 {
            let base = DerivedDepth {
                key: mtp_key(depth),
                counts: CalibrationCounts::default(),
            };
            assert_eq!(
                request.probability(&base),
                base_only.probability(&base),
                "base depth {depth}"
            );
        }
    }

    #[test]
    fn derived_k_old_native_actions_stay_exact() {
        // The pre-decision path never consults margin cells: the old oracle
        // fixture's query keys are all `Native`, so this module's replay of
        // that fixture (above) exercises the unchanged baseline.
        assert_eq!(
            feature_for(DerivedSource::Mtp, Some(3), Some(9), Some(true)),
            DerivedFeature::Native,
        );
        assert_eq!(
            feature_for(DerivedSource::Mtp, None, None, None),
            DerivedFeature::Native,
        );
    }
    #[test]
    fn derived_k_load_identity_needs_one_admitted_identity_and_descriptor() {
        // Bootstrap book is empty: no admitted identity, no descriptor.
        assert_eq!(
            fn_measured_load_identity(DerivedCard::Halo, DerivedCostMode::OffPool),
            None,
        );
        assert_eq!(
            fn_measured_load_identity(DerivedCard::R9700, DerivedCostMode::OnCopy),
            None,
        );
        // Descriptor shape: flat string map with the exact schema marker.
        assert!(measured_load_descriptor_valid(
            r#"{"schema":"qwen4-measured-load.v1","model.sha256":"aa"}"#
        ));
        for bad in [
            "",
            "not json",
            "[]",
            "{}",
            r#"{"schema":"other"}"#,
            r#"{"model.sha256":"aa"}"#,
            r#"{"schema":"qwen4-measured-load.v1","max_seq":100}"#,
            r#"{"schema":"qwen4-measured-load.v1","nested":{"a":"b"}}"#,
        ] {
            assert!(!measured_load_descriptor_valid(bad), "{bad}");
        }
    }

    #[test]
    fn derived_k_load_identity_selection_uses_genuine_literal_slices() {
        // Small genuine-shape slices through the real selector helpers (not a
        // shipping book): one admitted identity + one descriptor selects; any
        // second identity, second descriptor, empty entry, or invalid JSON
        // refuses without guessing.
        fn select(
            prices: &[FnWidthPriceRow],
            identities: &[(&'static str, &'static str)],
            card: &str,
            mode: &str,
        ) -> Option<&'static str> {
            let mut identity: Option<&'static str> = None;
            for row in prices {
                if row.1 != card || row.2 != mode {
                    continue;
                }
                if row.0.is_empty() {
                    return None;
                }
                match identity {
                    None => identity = Some(row.0),
                    Some(held) if held == row.0 => {}
                    Some(_) => return None,
                }
            }
            let identity = identity?;
            let mut descriptor = None;
            let mut seen = false;
            for row in identities {
                if row.0 != identity {
                    continue;
                }
                if row.1.is_empty() || !measured_load_descriptor_valid(row.1) {
                    return None;
                }
                if seen {
                    return None;
                }
                seen = true;
                descriptor = Some(row.1);
            }
            descriptor
        }
        const GOOD: &str = r#"{"schema":"qwen4-measured-load.v1"}"#;
        let price: FnWidthPriceRow = (
            "id", "Halo", "OffPool", "Le2K", "r", "Mtp", false, 1, 10.0, 2, 2, false,
        );
        let mut other_mode = price;
        other_mode.2 = "OnCopy";
        // Same identity across buckets/routes/modes selects per card+mode.
        let mut other_bucket = price;
        other_bucket.3 = "Le8K";
        other_bucket.4 = "r2";
        assert_eq!(
            select(&[price, other_bucket], &[("id", GOOD)], "Halo", "OffPool"),
            Some(GOOD),
        );
        // Cross-card rows never leak into the selection.
        let mut other_card = price;
        other_card.1 = "R9700";
        other_card.0 = "other";
        assert_eq!(
            select(&[price, other_card], &[("id", GOOD)], "Halo", "OffPool"),
            Some(GOOD),
        );
        assert_eq!(
            select(&[other_card], &[("other", GOOD)], "R9700", "OffPool"),
            Some(GOOD)
        );
        // Ambiguity and malformation refuse.
        let mut second = price;
        second.0 = "other";
        assert_eq!(
            select(&[price, second], &[("id", GOOD)], "Halo", "OffPool"),
            None
        );
        assert_eq!(
            select(&[price], &[("id", GOOD), ("id", GOOD)], "Halo", "OffPool"),
            None
        );
        assert_eq!(select(&[price], &[], "Halo", "OffPool"), None);
        assert_eq!(select(&[], &[("id", GOOD)], "Halo", "OffPool"), None);
        assert_eq!(
            select(&[price], &[("other", GOOD)], "Halo", "OffPool"),
            None
        );
        assert_eq!(select(&[price], &[("id", "")], "Halo", "OffPool"), None);
        assert_eq!(select(&[price], &[("id", "[]")], "Halo", "OffPool"), None);
        let mut empty_id = price;
        empty_id.0 = "";
        assert_eq!(
            select(&[empty_id], &[("id", GOOD)], "Halo", "OffPool"),
            None
        );
    }
    #[test]
    fn derived_k_reference_support_matches_live_ar_policy_per_source() {
        // Bootstrap book is empty: no admitted identity, so unsupported.
        assert!(!fn_measured_reference_supported(
            DerivedCard::Halo,
            DerivedCostMode::OffPool,
            DerivedSource::Mtp,
            false,
        ));
        assert!(!fn_measured_reference_supported(
            DerivedCard::Halo,
            DerivedCostMode::OffPool,
            DerivedSource::Ngram,
            true,
        ));
        // Real flat ROUTE_FIELDS slices through the production predicate
        // shape. Live policy: `Recorded` means the AR leg is retained while
        // the multi-row verify stays eager HIP, so `execution.ar.eager` is
        // `!recorded` and `execution.verify.eager` is always true; both
        // pre-window recording/capture leaves are false.
        fn pair_supported(fields: &[FnWidthRouteField], recorded: bool) -> bool {
            let want = match recorded {
                false => "\"Eager\"",
                true => "\"Recorded\"",
            };
            let want_ar_eager = match recorded {
                false => "true",
                true => "false",
            };
            fn route_ok(
                fields: &[FnWidthRouteField],
                route: &str,
                want: &str,
                want_ar_eager: &str,
            ) -> bool {
                let mut kind_ok = false;
                let mut reference_ok = false;
                let mut legs = 0u8;
                for field in fields {
                    if field.1 != route {
                        continue;
                    }
                    match field.2 {
                        "kind" => {
                            if field.3 != "\"batched\"" && field.3 != "\"target-only\"" {
                                return false;
                            }
                            kind_ok = true;
                        }
                        "arithmetic_reference" => {
                            if field.3 != want {
                                return false;
                            }
                            reference_ok = true;
                        }
                        "execution.ar.replay_recording" | "execution.ar.graph_capture_mode" => {
                            if field.3 != "false" {
                                return false;
                            }
                            legs |= 1 << u8::from(field.2.ends_with("graph_capture_mode"));
                        }
                        "execution.ar.eager" => {
                            if field.3 != want_ar_eager {
                                return false;
                            }
                            legs |= 1 << 2;
                        }
                        "execution.verify.replay_recording"
                        | "execution.verify.graph_capture_mode" => {
                            if field.3 != "false" {
                                return false;
                            }
                            legs |= 1 << (3 + u8::from(field.2.ends_with("graph_capture_mode")));
                        }
                        "execution.verify.eager" => {
                            if field.3 != "true" {
                                return false;
                            }
                            legs |= 1 << 5;
                        }
                        _ => {}
                    }
                }
                kind_ok && reference_ok && legs == 0b111_111
            }
            let (mut batched, mut ar) = (None, None);
            for field in fields {
                if field.0 != "id" || field.2 != "kind" {
                    continue;
                }
                match field.3 {
                    "\"batched\"" => batched = Some(field.1),
                    "\"target-only\"" => ar = Some(field.1),
                    _ => return false,
                }
            }
            match (batched, ar) {
                (Some(batched), Some(ar)) if batched != ar => {
                    route_ok(fields, batched, want, want_ar_eager)
                        && route_ok(fields, ar, want, want_ar_eager)
                }
                _ => false,
            }
        }
        fn route_fields(
            route: &'static str,
            kind: &'static str,
            reference: &'static str,
            ar_eager: &'static str,
        ) -> Vec<FnWidthRouteField> {
            vec![
                ("id", route, "kind", kind),
                ("id", route, "arithmetic_reference", reference),
                ("id", route, "execution.ar.replay_recording", "false"),
                ("id", route, "execution.ar.graph_capture_mode", "false"),
                ("id", route, "execution.ar.eager", ar_eager),
                ("id", route, "execution.verify.replay_recording", "false"),
                ("id", route, "execution.verify.graph_capture_mode", "false"),
                ("id", route, "execution.verify.eager", "true"),
            ]
        }
        // Eager pair: AR leg eager, verify eager.
        let mut both = route_fields("batched", "\"batched\"", "\"Eager\"", "true");
        both.extend(route_fields("ar", "\"target-only\"", "\"Eager\"", "true"));
        assert!(pair_supported(&both, false));
        assert!(!pair_supported(&both, true));
        // Retained pair: AR leg retained (`eager=false`), verify still eager,
        // both references `Recorded`. Passes only under the recorded query.
        let mut recorded = route_fields("batched", "\"batched\"", "\"Recorded\"", "false");
        recorded.extend(route_fields(
            "ar",
            "\"target-only\"",
            "\"Recorded\"",
            "false",
        ));
        assert!(pair_supported(&recorded, true));
        assert!(!pair_supported(&recorded, false));
        // A retained AR leg mislabeled eager fails the recorded query: the
        // old `!is_recording` shape that mislabels Ready retained.
        let mut mislabeled = route_fields("batched", "\"batched\"", "\"Recorded\"", "true");
        mislabeled.extend(route_fields(
            "ar",
            "\"target-only\"",
            "\"Recorded\"",
            "true",
        ));
        assert!(!pair_supported(&mislabeled, true));
        // Singleton-unbound or missing AR route never passes.
        let mut unbound = route_fields("batched", "\"batched\"", "\"Eager\"", "true");
        unbound.extend(route_fields(
            "ar",
            "\"target-only\"",
            "\"singleton-unbound\"",
            "true",
        ));
        assert!(!pair_supported(&unbound, false));
        let batched_only = route_fields("batched", "\"batched\"", "\"Eager\"", "true");
        assert!(!pair_supported(&batched_only, false));
        // A recording/graph leg fails the pre-window gate.
        let mut recording = route_fields("batched", "\"batched\"", "\"Eager\"", "true");
        recording.extend(route_fields("ar", "\"target-only\"", "\"Eager\"", "true"));
        recording[2] = ("id", "batched", "execution.ar.replay_recording", "true");
        assert!(!pair_supported(&recording, false));
        // Mismatched references across the two routes fail together.
        let mut split = route_fields("batched", "\"batched\"", "\"Eager\"", "true");
        split.extend(route_fields(
            "ar",
            "\"target-only\"",
            "\"Recorded\"",
            "false",
        ));
        assert!(!pair_supported(&split, false));
        assert!(!pair_supported(&split, true));
        // Descriptor selection alone is not production admission: require a
        // supported execution/reference pair, real AR/native cells in the
        // same bucket, and every native parent from the exact price identity.
        const LOAD: &str = r#"{"schema":"qwen4-measured-load.v1"}"#;
        let native: MeasuredPriceRow<'_> = (
            "id", "R9700", "OffPool", "Le2K", "batched", "Mtp", false, 1, 10.0, 2, 2, false,
        );
        let ar: MeasuredPriceRow<'_> = (
            "id", "R9700", "OffPool", "Le2K", "ar", "ArRouted", false, 0, 9.0, 2, 2, false,
        );
        let parent: FnNativeDepthRow = ("id", "R9700", "OffPool", "Le2K", "batched", 1, 2, 3);
        let counts = build_native_counts_index(&[parent], &[]);
        for routes in [&both, &recorded] {
            let book = build_measured_book_index(&[native, ar], routes, &[("id", LOAD)]);
            assert_eq!(book.qualified_load_descriptor(0, &counts), Some(LOAD));
            assert_eq!(
                book.qualified_load_descriptor(0, &build_native_counts_index(&[], &[])),
                None
            );
            let mut other = parent;
            other.0 = "different-seal";
            assert_eq!(
                book.qualified_load_descriptor(0, &build_native_counts_index(&[other], &[])),
                None
            );
        }
        for routes in [&mislabeled, &unbound, &batched_only, &recording, &split] {
            let book = build_measured_book_index(&[native, ar], routes, &[("id", LOAD)]);
            assert_eq!(book.qualified_load_descriptor(0, &counts), None);
        }
        let mut wrong_bucket = ar;
        wrong_bucket.3 = "Le8K";
        let mut invalid = native;
        invalid.9 = 0;
        let mut unsupported_depth = native;
        unsupported_depth.7 = 2;
        unsupported_depth.10 = 3;
        for prices in [
            vec![],
            vec![native],
            vec![ar],
            vec![native, wrong_bucket],
            vec![invalid, ar],
            vec![native, native, ar],
            vec![unsupported_depth, ar],
        ] {
            let book = build_measured_book_index(&prices, &both, &[("id", LOAD)]);
            assert_eq!(book.qualified_load_descriptor(0, &counts), None);
        }
    }
    /// Index-vs-old oracle equality for the full measured book: valid cells,
    /// missing/ambiguous/unsupported cells, identity/route/card/mode/context/
    /// source/confirmation/width splits, bad JSON and descriptors, recorded
    /// support both ways, and the empty bootstrap book. Each row of the
    /// matrix below is a standalone slice triple `(prices, routes, loads)`
    /// checked cell-by-cell; the production shipping book check asserts the
    /// hot path reads the same values through the real public functions.
    #[test]
    fn derived_k_measured_book_index_equals_slice_oracles() {
        const GOOD: &str = r#"{"schema":"qwen4-measured-load.v1"}"#;
        fn routes_for<'a>(
            identity: &'a str,
            batched: &'a str,
            ar: &'a str,
        ) -> Vec<MeasuredRouteField<'a>> {
            let mut fields = Vec::new();
            for (route, kind) in [(batched, "\"batched\""), (ar, "\"target-only\"")] {
                fields.extend([
                    (identity, route, "kind", kind),
                    (identity, route, "arithmetic_reference", "\"Eager\""),
                    (identity, route, "execution.ar.replay_recording", "false"),
                    (identity, route, "execution.ar.graph_capture_mode", "false"),
                    (identity, route, "execution.ar.eager", "true"),
                    (
                        identity,
                        route,
                        "execution.verify.replay_recording",
                        "false",
                    ),
                    (
                        identity,
                        route,
                        "execution.verify.graph_capture_mode",
                        "false",
                    ),
                    (identity, route, "execution.verify.eager", "true"),
                ]);
            }
            fields
        }
        fn check_case<'a>(
            prices: &[MeasuredPriceRow<'a>],
            routes: &[MeasuredRouteField<'a>],
            loads: &[(&'a str, &'a str)],
            cards: &[&str],
            modes: &[&str],
        ) {
            let index = build_measured_book_index(prices, routes, loads);
            // Price matrix: every card/mode/bucket/source/k/confirmation
            // slice of the oracle law, including invalid shapes.
            for card in cards.iter().chain(["R9700", "Halo", "Unknown"].iter()) {
                for mode in modes.iter().chain(["OffPool", "OnCopy", "Unknown"].iter()) {
                    for bucket in ["Le2K", "Le8K", "Le16K", "Over16K", "Unknown"] {
                        for (source, ks, confirmations) in [
                            ("Mtp", [0u8, 1, 2].as_slice(), [false, true].as_slice()),
                            ("Ngram", [0u8, 1, 2].as_slice(), [false, true].as_slice()),
                            ("ArRouted", [0u8, 1].as_slice(), [false, true].as_slice()),
                            ("Unknown", [1u8].as_slice(), [false].as_slice()),
                        ] {
                            for k in ks {
                                for confirmation in confirmations {
                                    assert_eq!(
                                        measured_index_price(
                                            &index, card, mode, bucket, source, *k, *confirmation,
                                        ),
                                        fn_width_price(
                                            prices, routes, (card, mode, bucket), source,
                                            usize::from(*k), *confirmation,
                                        ),
                                        "price {card}/{mode}/{bucket}/{source} k={k} c={confirmation}",
                                    );
                                }
                            }
                        }
                    }
                    // Load + reference matrix per card/mode/source/recorded.
                    assert_eq!(
                        measured_index_load_identity(&index, card, mode),
                        measured_oracle_load_identity(prices, loads, card, mode),
                        "load {card}/{mode}",
                    );
                    for source in ["Mtp", "Ngram", "ArRouted", "Unknown"] {
                        for recorded in [false, true] {
                            assert_eq!(
                                measured_index_reference_supported(
                                    &index, card, mode, source, recorded
                                ),
                                measured_oracle_reference_supported(
                                    prices, routes, card, mode, source, recorded,
                                ),
                                "reference {card}/{mode}/{source} recorded={recorded}",
                            );
                        }
                    }
                }
            }
            // Sortedness: grouping sorts once at build; no per-window alloc.
            let mut sorted = index.prices.clone();
            sorted.sort_by(|left, right| left.0.cmp(&right.0));
            assert_eq!(index.prices, sorted);
            // Slot/cell law parity: a poisoned slot still resolves per-key —
            // only load/ref read `None` there — while an absent slot with no
            // rows reads `Ok(None)`; every indexed cell replays `Ok(Some)`.
            for (slot, health) in index.health.iter().enumerate() {
                let (card, mode) = match slot {
                    0 => ("R9700", "OffPool"),
                    1 => ("R9700", "OnCopy"),
                    2 => ("Halo", "OffPool"),
                    _ => ("Halo", "OnCopy"),
                };
                if *health == NativeSlotHealth::Absent {
                    for bucket in ["Le2K", "Le8K", "Le16K", "Over16K"] {
                        for (source, k, confirmation) in [
                            ("Mtp", 1u8, false),
                            ("Ngram", 1, true),
                            ("ArRouted", 0, false),
                        ] {
                            assert_eq!(
                                measured_index_price(
                                    &index,
                                    card,
                                    mode,
                                    bucket,
                                    source,
                                    k,
                                    confirmation
                                ),
                                Ok(None),
                                "absent slot {card}/{mode}",
                            );
                        }
                    }
                }
                if !matches!(health, NativeSlotHealth::Healthy) {
                    assert_eq!(measured_index_load_identity(&index, card, mode), None);
                }
            }
            for entry in &index.prices {
                let (card, mode, bucket, source, confirmation, k) = entry.0;
                assert_eq!(
                    fn_width_price(
                        prices,
                        routes,
                        (card, mode, bucket),
                        source,
                        k,
                        confirmation
                    ),
                    Ok(Some(entry.1)),
                    "indexed cell replays oracle {card}/{mode}/{bucket}/{source} k={k}",
                );
            }
        }
        // Case 0: empty bootstrap book: every slot absent, every price
        // `Ok(None)`, loads `None`, references `false`. The shipping tables
        // below assert the same through the public hot paths.
        check_case(&[], &[], &[], &["Halo"], &["OffPool"]);
        let empty_index = build_measured_book_index(&[], &[], &[]);
        assert_eq!(empty_index.health, [NativeSlotHealth::Absent; 4]);
        assert!(empty_index.prices.is_empty());
        // Case 1: one healthy slot (Halo/OffPool) with a valid measured book:
        // Mtp + Ngram batched rows, one target-only AR route, descriptor, and
        // both recorded flag variants covered by the `routes_for` Eager set.
        let prices: [MeasuredPriceRow<'_>; 4] = [
            (
                "id", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 10.0, 2, 2, false,
            ),
            (
                "id", "Halo", "OffPool", "Le8K", "ngram", "Ngram", true, 2, 20.0, 3, 3, false,
            ),
            (
                "id", "Halo", "OffPool", "Le2K", "ar", "ArRouted", false, 0, 5.0, 2, 1, false,
            ),
            (
                "id", "R9700", "OffPool", "Le2K", "other", "Mtp", false, 1, 99.0, 2, 2, false,
            ),
        ];
        let routes = routes_for("id", "mtp", "ar");
        let mut full_routes = routes.clone();
        full_routes.extend(routes_for("id", "ngram", "ar"));
        // Ngram route intentionally left without `kind`: unpriced there.
        check_case(
            &prices,
            &full_routes,
            &[("id", GOOD)],
            &["Halo", "R9700"],
            &["OffPool", "OnCopy"],
        );
        // Case 2: a second identity in one cell poisons exactly that key
        // (the duplicate opaque key reads `Err`); the sibling Mtp key and
        // the Halo/OnCopy slot still resolve. Load admission still marks
        // Halo/OffPool poisoned (`None`).
        let mut ambiguous = prices.to_vec();
        ambiguous[2].0 = "other";
        let ambiguous_index = build_measured_book_index(&ambiguous, &full_routes, &[("id", GOOD)]);
        assert_eq!(
            ambiguous_index.health[2],
            NativeSlotHealth::Poisoned,
            "second identity poisons Halo/OffPool load admission",
        );
        assert_eq!(
            measured_index_load_identity(&ambiguous_index, "Halo", "OffPool"),
            None,
        );
        assert_eq!(
            measured_index_price(
                &ambiguous_index,
                "Halo",
                "OffPool",
                "Le2K",
                "ArRouted",
                0,
                false
            ),
            Err(()),
            "duplicate opaque key rejects",
        );
        // Case 3: duplicate queried cell poisons (same slot/bucket/source/k):
        // the slice validator's `found.is_some()` arm reads `Err(())`.
        let duplicate: [MeasuredPriceRow<'_>; 2] = [
            (
                "id", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 10.0, 2, 2, false,
            ),
            (
                "id", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 11.0, 2, 2, false,
            ),
        ];
        let dup_routes = routes_for("id", "mtp", "ar");
        check_case(
            &duplicate,
            &dup_routes,
            &[("id", GOOD)],
            &["Halo"],
            &["OffPool"],
        );
        assert_eq!(
            measured_index_price(
                &build_measured_book_index(&duplicate, &dup_routes, &[("id", GOOD)]),
                "Halo",
                "OffPool",
                "Le2K",
                "Mtp",
                1,
                false,
            ),
            Err(()),
        );
        // Case 4: malformed descriptors poison loads (empty, bad JSON, wrong
        // schema, non-string value, second descriptor) while the price cells
        // keep their own validity.
        let price: [MeasuredPriceRow<'_>; 2] = [
            (
                "id", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 10.0, 2, 2, false,
            ),
            (
                "id", "Halo", "OffPool", "Le2K", "ar", "ArRouted", false, 0, 5.0, 2, 1, false,
            ),
        ];
        for bad in [
            "",
            "not json",
            "[]",
            "{}",
            r#"{"schema":"other"}"#,
            r#"{"model.sha256":"aa"}"#,
            r#"{"schema":"qwen4-measured-load.v1","max_seq":100}"#,
            r#"{"schema":"qwen4-measured-load.v1","nested":{"a":"b"}}"#,
        ] {
            check_case(&price, &dup_routes, &[("id", bad)], &["Halo"], &["OffPool"]);
        }
        check_case(
            &price,
            &dup_routes,
            &[("id", GOOD), ("id", GOOD)],
            &["Halo"],
            &["OffPool"],
        );
        check_case(&price, &dup_routes, &[], &["Halo"], &["OffPool"]);
        // Case 5: malformed cells: non-finite/non-positive totals, zero
        // support, under-qualified rows, pooled rows, empty identity/route,
        // AR shapes, and a missing/duplicate `kind` field.
        let base: MeasuredPriceRow<'_> = (
            "id", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 10.0, 2, 2, false,
        );
        let mut invalids: Vec<MeasuredPriceRow<'_>> = vec![
            (
                "", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 10.0, 2, 2, false,
            ),
            (
                "id", "Halo", "OffPool", "Le2K", "", "Mtp", false, 1, 10.0, 2, 2, false,
            ),
        ];
        for total in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut row = base;
            row.8 = total;
            invalids.push(row);
        }
        for mut row in [
            {
                let mut row = base;
                row.9 = 0;
                row
            },
            {
                let mut row = base;
                row.10 = 1;
                row
            },
            {
                let mut row = base;
                row.11 = true;
                row
            },
            {
                let mut row = base;
                row.5 = "Mtp";
                row.6 = true;
                row
            },
            {
                let mut row = base;
                row.5 = "ArRouted";
                row
            },
            {
                let mut row = base;
                row.5 = "ArRouted";
                row.6 = true;
                row
            },
            {
                let mut row = base;
                row.5 = "Unknown";
                row
            },
            {
                let mut row = base;
                row.7 = 0;
                row
            },
        ] {
            // Keep the identity visible so the slot poisons (present but
            // invalid) rather than reading absent.
            row.0 = "id";
            invalids.push(row);
        }
        for invalid in &invalids {
            check_case(
                &[*invalid],
                &dup_routes,
                &[("id", GOOD)],
                &["Halo"],
                &["OffPool"],
            );
        }
        check_case(&[base], &[], &[("id", GOOD)], &["Halo"], &["OffPool"]);
        check_case(
            &[base],
            &[
                routes_for("id", "mtp", "ar")[0],
                routes_for("id", "mtp", "ar")[0],
            ],
            &[("id", GOOD)],
            &["Halo"],
            &["OffPool"],
        );
        // Case 6: Recorded support both ways: the `Recorded` book passes only
        // the recorded query, and cross-reference mismatches fail both.
        fn recorded_routes<'a>() -> Vec<MeasuredRouteField<'a>> {
            let mut fields = Vec::new();
            for (route, kind) in [("mtp", "\"batched\""), ("ar", "\"target-only\"")] {
                fields.extend([
                    ("id", route, "kind", kind),
                    ("id", route, "arithmetic_reference", "\"Recorded\""),
                    ("id", route, "execution.ar.replay_recording", "false"),
                    ("id", route, "execution.ar.graph_capture_mode", "false"),
                    ("id", route, "execution.ar.eager", "false"),
                    ("id", route, "execution.verify.replay_recording", "false"),
                    ("id", route, "execution.verify.graph_capture_mode", "false"),
                    ("id", route, "execution.verify.eager", "true"),
                ]);
            }
            fields
        }
        let recorded_pairs: [MeasuredPriceRow<'_>; 2] = [
            (
                "id", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 10.0, 2, 2, false,
            ),
            (
                "id", "Halo", "OffPool", "Le2K", "ar", "ArRouted", false, 0, 5.0, 2, 1, false,
            ),
        ];
        check_case(
            &recorded_pairs,
            &recorded_routes(),
            &[("id", GOOD)],
            &["Halo"],
            &["OffPool"],
        );
        check_case(
            &recorded_pairs,
            &full_routes,
            &[("id", GOOD)],
            &["Halo"],
            &["OffPool"],
        );
        // Production shipping book parity: bootstrap is empty, so the hot
        // paths read absent/unsupported without claiming bootstrap prices.
        for card in [DerivedCard::R9700, DerivedCard::Halo] {
            for mode in [DerivedCostMode::OffPool, DerivedCostMode::OnCopy] {
                assert_eq!(fn_measured_load_identity(card, mode), None);
                for source in [DerivedSource::Mtp, DerivedSource::Ngram] {
                    assert!(!fn_measured_reference_supported(card, mode, source, false));
                    assert!(!fn_measured_reference_supported(card, mode, source, true));
                }
            }
        }
        // Sorted single build: the shipping index prices stay sorted.
        let shipping = measured_book_index();
        let mut sorted = shipping.prices.clone();
        sorted.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(shipping.prices, sorted);
    }

    #[test]
    fn derived_k_cross_cell_identity_does_not_poison_price_but_load_is_none() {
        const GOOD: &str = r#"{"schema":"qwen4-measured-load.v1"}"#;
        // Same opaque key would duplicate; use two cells that differ only in
        // bucket. The second identity lives in the Le8K cell, so the Le2K
        // price still resolves while load admission (global unique identity
        // per slot) refuses.
        let prices: [MeasuredPriceRow<'_>; 2] = [
            (
                "id", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 10.0, 2, 2, false,
            ),
            (
                "other", "Halo", "OffPool", "Le8K", "mtp", "Mtp", false, 1, 11.0, 2, 2, false,
            ),
        ];
        let routes = vec![
            ("id", "mtp", "kind", "\"batched\""),
            ("other", "mtp", "kind", "\"batched\""),
        ];
        let index = build_measured_book_index(&prices, &routes, &[("id", GOOD)]);
        assert_eq!(
            measured_index_price(&index, "Halo", "OffPool", "Le2K", "Mtp", 1, false),
            Ok(Some((10.0, false))),
        );
        assert_eq!(
            measured_index_price(&index, "Halo", "OffPool", "Le2K", "Mtp", 1, false),
            fn_width_price(
                &prices,
                &routes,
                ("Halo", "OffPool", "Le2K"),
                "Mtp",
                1,
                false
            ),
        );
        assert_eq!(
            measured_index_load_identity(&index, "Halo", "OffPool"),
            None
        );
        assert_eq!(
            measured_oracle_load_identity(&prices, &[("id", GOOD)], "Halo", "OffPool"),
            None,
        );
    }

    #[test]
    fn derived_k_unknown_opaque_query_matches_old_validator_ordering() {
        const GOOD: &str = r#"{"schema":"qwen4-measured-load.v1"}"#;
        let row: MeasuredPriceRow<'_> = (
            "id", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 10.0, 2, 2, false,
        );
        // Valid route set: `kind` plus the full eager-HIP leg mask is not
        // needed for prices; one `kind` row suffices.
        let routes = [("id", "mtp", "kind", "\"batched\"")];
        let index = build_measured_book_index(&[row], &routes, &[("id", GOOD)]);
        // Unknown opaque words: no row names them, so both layers read
        // absence — but the shape gate runs first. Unknown sources are
        // shape-invalid (`expected_kind` has no arm), hence `Err`; unknown
        // card/mode/bucket words are just unpriced keys (`Ok(None)`).
        for (card, mode, bucket, source, k, confirmation, shape_valid) in [
            ("Bogus", "OffPool", "Le2K", "Mtp", 1u8, false, true),
            ("Halo", "Bogus", "Le2K", "Mtp", 1, false, true),
            ("Halo", "OffPool", "Bogus", "Mtp", 1, false, true),
            ("Halo", "OffPool", "Le2K", "Bogus", 1, false, false),
            ("Halo", "OffPool", "Le2K", "Mtp", 2, true, false),
            ("Halo", "OffPool", "Le2K", "ArRouted", 1, false, false),
        ] {
            let key = (card, mode, bucket);
            let old = fn_width_price(&[row], &routes, key, source, usize::from(k), confirmation);
            let got = measured_index_price(&index, card, mode, bucket, source, k, confirmation);
            assert_eq!(got, old, "query {key:?}/{source} k={k} c={confirmation}");
            if shape_valid {
                assert_eq!(got, Ok(None));
            } else {
                assert_eq!(got, Err(()));
            }
        }
        // Shape-invalid ordering: the shape gate rejects before any row is
        // consulted, so even a poisoned/absent table agrees with the old
        // validator's leading match.
        let empty = build_measured_book_index(&[], &[], &[]);
        for (source, k, confirmation) in
            [("Mtp", 1u8, true), ("ArRouted", 0, true), ("Mtp", 0, false)]
        {
            assert_eq!(
                measured_index_price(&empty, "Halo", "OffPool", "Le2K", source, k, confirmation),
                fn_width_price(
                    &[],
                    &[],
                    ("Halo", "OffPool", "Le2K"),
                    source,
                    usize::from(k),
                    confirmation
                ),
            );
            assert_eq!(
                measured_index_price(&empty, "Halo", "OffPool", "Le2K", source, k, confirmation),
                Err(()),
            );
        }
    }

    #[test]
    fn derived_k_duplicate_cell_with_different_identity_still_errs() {
        const GOOD: &str = r#"{"schema":"qwen4-measured-load.v1"}"#;
        // Same opaque key, different identity: the old validator's
        // `found.is_some()` arm fires on the second row regardless of the
        // identity change; the per-key grouper must agree.
        let rows: [MeasuredPriceRow<'_>; 2] = [
            (
                "id", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 10.0, 2, 2, false,
            ),
            (
                "other", "Halo", "OffPool", "Le2K", "mtp", "Mtp", false, 1, 11.0, 2, 2, false,
            ),
        ];
        let routes = [
            ("id", "mtp", "kind", "\"batched\""),
            ("other", "mtp", "kind", "\"batched\""),
        ];
        let index = build_measured_book_index(&rows, &routes, &[("id", GOOD)]);
        assert_eq!(
            measured_index_price(&index, "Halo", "OffPool", "Le2K", "Mtp", 1, false),
            Err(()),
        );
        assert_eq!(
            fn_width_price(&rows, &routes, ("Halo", "OffPool", "Le2K"), "Mtp", 1, false),
            Err(()),
        );
    }

    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/derived_k_full_log_replay.rs"
    ));
}
