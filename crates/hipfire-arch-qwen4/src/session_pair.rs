// SPDX-License-Identifier: Apache-2.0
//! Fixed-state staging for a session boundary between pipeline banks.
use super::*;
use crate::ple::PleHistory;
use hipfire_runtime::session_cache::StatePart;
use rdna_compute::DType;
use serde_json::{json, Value};

pub(crate) struct PairSessionStage {
    pub(crate) fixed: Vec<GpuTensor>,
    meta: Vec<u8>,
    rows: Vec<usize>,
    pub(super) position: Option<usize>,
}

impl PairSessionStage {
    pub(crate) fn copy(
        &self,
        gpu: &mut Gpu,
        index: usize,
        src: &GpuTensor,
    ) -> Result<(), BundleError> {
        self.copy_parts(gpu, index, [src])
    }

    pub(crate) fn copy_parts<const N: usize>(
        &self,
        gpu: &mut Gpu,
        index: usize,
        src: [&GpuTensor; N],
    ) -> Result<(), BundleError> {
        let dst = self
            .fixed
            .get(index..index + N)
            .ok_or_else(|| BundleError::Forward("paired snapshot fixed index mismatch".into()))?;
        for (src, dst) in src.iter().zip(dst) {
            if src.byte_size() != dst.byte_size() || !src.byte_size().is_multiple_of(4) {
                return Err(BundleError::Forward(
                    "paired snapshot state must use equal aligned device copies".into(),
                ));
            }
        }
        // A copy kernel is stream-ordered even on the default stream;
        // copy_d2d's legacy-stream fallback would synchronize the host.
        let regions: [_; N] = std::array::from_fn(|i| rdna_compute::tensor_ops::CopyRegion {
            dst: &dst[i].buf,
            dst_offset: 0,
            src: &src[i].buf,
            src_offset: 0,
            bytes: src[i].byte_size(),
        });
        rdna_compute::tensor_ops::copy_regions(gpu, &regions).map_err(BundleError::Hip)
    }

    pub(crate) fn configure(
        &mut self,
        state: &Qwen4State,
        history: crate::ple::PleHistory,
        commits: &[(usize, usize, usize, usize, usize)],
    ) -> Result<(), BundleError> {
        let (meta, layout) = state
            .session_parts_at(self.position.expect("armed stage"), history, Some(commits))
            .map_err(BundleError::State)?;
        debug_assert_eq!(self.fixed.len(), layout.fixed.len());
        for (saved, part) in self.fixed.iter().zip(&layout.fixed) {
            debug_assert_eq!(saved.byte_size(), part.bytes);
        }
        self.meta = meta;
        #[cfg(debug_assertions)]
        {
            // The ordinary restore parser must reconstruct exactly the staged
            // boundary layout. This is host-only; never drain the HIP stream.
            let sequential = state
                .session_layout_for_meta(&self.meta)
                .map_err(BundleError::State)?;
            assert_layout_shape(
                self.fixed.iter().map(GpuTensor::byte_size),
                layout
                    .rows
                    .iter()
                    .map(|stream| (stream.row_bytes, stream.rows)),
                sequential.fixed.iter().map(|part| part.bytes),
                sequential
                    .rows
                    .iter()
                    .map(|stream| (stream.row_bytes, stream.rows)),
            );
        }
        self.rows = layout.rows.iter().map(|stream| stream.rows).collect();
        Ok(())
    }

    pub(super) fn snapshot_parts<'a>(
        &'a self,
        state: &'a Qwen4State,
    ) -> Result<SnapshotParts<'a>, StateError> {
        Ok(self.parts(state.session_layout_for_meta(&self.meta)?))
    }

    pub(super) fn parts<'a>(&'a self, mut live: StateLayout<'a>) -> SnapshotParts<'a> {
        debug_assert_eq!(self.fixed.len(), live.fixed.len());
        debug_assert_eq!(self.rows.len(), live.rows.len());
        for (saved, part) in self.fixed.iter().zip(&mut live.fixed) {
            debug_assert_eq!(saved.byte_size(), part.bytes);
            *part = StatePart {
                buf: &saved.buf,
                offset: 0,
                bytes: saved.byte_size(),
            };
        }
        for (&rows, stream) in self.rows.iter().zip(&mut live.rows) {
            debug_assert!(rows <= stream.rows);
            stream.rows = rows;
        }
        SnapshotParts {
            meta: self.meta.clone(),
            layout: live,
        }
    }
}

#[cfg(debug_assertions)]
fn assert_layout_shape(
    fixed: impl Iterator<Item = usize>,
    rows: impl Iterator<Item = (usize, usize)>,
    sequential_fixed: impl Iterator<Item = usize>,
    sequential_rows: impl Iterator<Item = (usize, usize)>,
) {
    assert!(
        fixed.eq(sequential_fixed),
        "paired snapshot fixed layout differs from sequential"
    );
    assert!(
        rows.eq(sequential_rows),
        "paired snapshot row layout differs from sequential"
    );
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::assert_layout_shape;

    #[test]
    fn staged_layout_matches_sequential_without_device_buffers() {
        let fixed = [64, 128, 32, 16];
        let rows = [(8, 8192), (8, 8192), (4, 8192), (4, 512)];
        assert_layout_shape(
            fixed.into_iter(),
            rows.into_iter(),
            fixed.into_iter(),
            rows.into_iter(),
        );
    }

    #[test]
    #[should_panic(expected = "row layout differs")]
    fn pair_end_rows_cannot_be_published_at_a_boundary() {
        assert_layout_shape(
            [64].into_iter(),
            [(8, 16384)].into_iter(),
            [64].into_iter(),
            [(8, 8192)].into_iter(),
        );
    }
}

impl Qwen4Bundle {
    pub(crate) fn prepare_pair_session_stage(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
    ) -> Result<Option<PairSessionStage>, BundleError> {
        if self.session.is_none() || self.session_next_boundary() != Some(position) {
            return Ok(None);
        }
        let (_, layout) = self.state.session_parts().map_err(BundleError::State)?;
        let mut stage = self.pair_session_stage.take().unwrap_or(PairSessionStage {
            fixed: Vec::new(),
            meta: Vec::new(),
            rows: Vec::new(),
            position: None,
        });
        if stage.fixed.is_empty() {
            for part in &layout.fixed {
                match gpu.zeros(&[part.bytes], DType::Raw) {
                    Ok(tensor) => stage.fixed.push(tensor),
                    Err(error) => {
                        for tensor in stage.fixed {
                            let _ = gpu.free_tensor(tensor);
                        }
                        return Err(BundleError::Hip(error));
                    }
                }
            }
        }
        debug_assert_eq!(stage.fixed.len(), layout.fixed.len());
        stage.position = Some(position);
        // Feedback is an unchanged fixed part on this route, but is still
        // captured: the cache stores it even on profiles without PLE.
        let hyper_feedback = self
            .state
            .hyper_feedback
            .as_ref()
            .expect("hyper_feedback is input-band state");
        stage.copy(gpu, stage.fixed.len() - 1, hyper_feedback)?;
        Ok(Some(stage))
    }

    /// Use the retained canonical prefix, without allocating another token copy.
    /// MTP calls this only after A's head append and before B's append.
    pub(crate) fn publish_pair_session_boundary(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
    ) -> Result<(), BundleError> {
        if self.session_next_boundary() != Some(position) {
            return Ok(());
        }
        let result = self
            .ep_refuse("session_at_boundary outside a replicated call")
            .and_then(|()| {
                self.with_session(|session, bundle| session.at_turn_boundary(gpu, bundle, position))
                    .unwrap_or(Ok(()))
                    .map_err(BundleError::Forward)
            });
        if let Some(stage) = &mut self.pair_session_stage {
            stage.position = None;
        }
        result
    }

    pub(crate) fn pair_boundary_is_ar(&self) -> bool {
        self.session.as_ref().and_then(SessionDriver::turn_route) == Some(SessionRoute::Ar)
    }
}

// ---------------------------------------------------------------------------
// Slow-link (S1) layer-major prompt transaction.
//
// One transaction brackets ONE admitted prompt window. Layers run the whole
// window one after another, so the global `state.position` / `ple_history`
// stay at the window ENTRY until a single commit; every layer advances only
// its own private cursor (its GDN recurrent/conv state and, for QSA layers,
// its `state.qsa[slot]` lengths). The entry state lives in the state's
// existing snapshot arena (and the head's), so nothing extra is allocated for
// it. Each session cut inside the window is staged per layer, right after the
// layer's tile that ends at the cut, into a bounded pool of fixed-size
// stages (`PairSessionStage`): only the fixed parts are copied, because the
// QSA row streams below a cut's marks are append-only (the pool kernel only
// re-pools blocks at or after the launch's `start / compress`, which is at or
// above the cut's `pooled_len`), so they are read in place when the cut is
// published.
// ---------------------------------------------------------------------------

/// Interior session cuts one window stages (the admitted boundary-snapshot
/// slots).
pub(crate) const PROMPT_TXN_MAX_CUTS: usize = crate::slowlink_admission::S1_MAX_BOUNDARY_SNAPSHOTS;
/// Widest mixer tile of a window: the wide-hidden capture buffer must hold it.
pub(crate) const PROMPT_TILE_ROWS_MAX: usize = crate::slowlink_admission::S1_TILE_ROWS;
pub(crate) const SLOWLINK_RECEIPT_SCHEMA: &str = "hipfire.qwen4.slowlink_receipt.v1";

/// The named production-primitive checks `slowlink_primitive_selftest` runs
/// (V's gate names). Each maps to a real T/D helper measured on the attached
/// forward; an unknown name is refused, never passed.
pub const SLOWLINK_PRIMITIVES: [&str; 7] = [
    "payload_table_identity",
    "repeated_epoch_same_address",
    "delayed_reader_reuse",
    "within_expert_panel_canary",
    "canonical_combine_exact",
    "k1_tail_cohort",
    "zero_row_panel",
];

/// Whether a case report honors the frozen pass/fail shape for `name`:
/// `name` equal, `status` `pass`/`fail`, `mismatches` a `u64`, `detail` present.
fn primitive_report_honored(name: &str, case: &Value) -> bool {
    case.get("name").and_then(Value::as_str) == Some(name)
        && matches!(
            case.get("status").and_then(Value::as_str),
            Some("pass" | "fail")
        )
        && case.get("mismatches").and_then(Value::as_u64).is_some()
        && case.get("detail").is_some()
}

/// The independent error report for one case: names the case, carries no
/// mismatch count (nothing was measured).
fn primitive_error_case(name: &str, error: String) -> Value {
    json!({
        "name": name,
        "status": "error",
        "mismatches": Value::Null,
        "detail": { "error": error },
    })
}

/// Invoke `run` once per requested case, in order, each independently: an
/// `Err` from one case, or a report that does not honor the frozen shape,
/// becomes that case's own `status:"error"` entry (naming that case) and the
/// remaining cases still run. Returns `{"cases":[..]}` with exactly one entry
/// per requested name. This is the production collector
/// `Qwen4Bundle::slowlink_primitive_selftest` runs its real owners through.
pub(crate) fn collect_primitive_reports(
    cases: &[&str],
    mut run: impl FnMut(&str) -> Result<Value, String>,
) -> Value {
    let out: Vec<Value> = cases
        .iter()
        .map(|&name| match run(name) {
            Err(error) => primitive_error_case(
                name,
                format!("slow-link primitive {name:?} could not run: {error}"),
            ),
            Ok(case) if primitive_report_honored(name, &case) => case,
            Ok(case) => primitive_error_case(
                name,
                format!(
                    "slow-link primitive {name:?} report does not carry \
                     name/status/mismatches/detail: {case}"
                ),
            ),
        })
        .collect();
    json!({ "cases": out })
}

/// Keys every receipt carries (`injection_consumed` is added on read).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const SLOWLINK_RECEIPT_KEYS: [&str; 21] = [
    "schema",
    "route",
    "selected_by",
    "bypass_reason",
    "entry_pos",
    "prompt_rows",
    "windows",
    "launch_cuts",
    "tile_rows",
    "retained_scratch_rows",
    "sweeps",
    "resident_layers",
    "host_layers",
    "host_expert_bytes_total",
    "submitted_bytes",
    "useful_bytes",
    "unused_prefetched_bytes",
    "reserve",
    "spans_ms",
    "overlap",
    "peak_vram_bytes",
];

/// `(full, raw, pooled, selected, position)` of one QSA slot, the commit
/// tuple `PairSessionStage::configure` takes.
pub(crate) type QsaCommit = (usize, usize, usize, usize, usize);

#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
/// Named failure points of the one-shot injection (`slowlink_inject_failure`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlowlinkFailPoint {
    BeforeDma,
    DuringPanel,
    AfterLayer,
    BeforeCommit,
}

#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
impl SlowlinkFailPoint {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "before_dma" => Self::BeforeDma,
            "during_panel" => Self::DuringPanel,
            "after_layer" => Self::AfterLayer,
            "before_commit" => Self::BeforeCommit,
            _ => return None,
        })
    }
}

/// What a request asks of its transaction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SlowlinkScope {
    /// The request's pending session boundaries are cuts (a session prompt;
    /// direct forwards never publish).
    pub(crate) cuts: bool,
    /// A window END that is a session boundary is published at commit.
    pub(crate) publish_end: bool,
    /// The MTP head is part of the transaction: it is snapshotted, restored
    /// and its cuts are published by the capture sink.
    pub(crate) head: bool,
}

impl SlowlinkScope {
    pub(crate) fn direct() -> Self {
        Self::default()
    }
    pub(crate) fn prompt() -> Self {
        Self {
            cuts: true,
            publish_end: true,
            head: false,
        }
    }
    pub(crate) fn capture() -> Self {
        Self {
            cuts: true,
            publish_end: true,
            head: true,
        }
    }
}

/// Opaque handle of the one open window transaction; all state lives in the
/// bundle.
#[must_use = "a window transaction must be committed or aborted"]
#[derive(Debug)]
pub(crate) struct PromptTxn {
    id: u64,
}

/// Pending session boundaries strictly inside `(entry, end)`, ascending.
pub(crate) fn prompt_cuts_in(entry: usize, end: usize, pending: &[usize]) -> Vec<usize> {
    debug_assert!(pending.windows(2).all(|pair| pair[0] < pair[1]));
    pending
        .iter()
        .copied()
        .filter(|&b| b > entry && b < end)
        .collect()
}

/// Largest window end `<= end` holding at most `max_cuts` strictly-inside
/// cuts: `end` itself when everything fits, else the `(max_cuts + 1)`-th
/// pending boundary after `entry` (which then ends the window as a boundary).
pub(crate) fn prompt_window_end(
    entry: usize,
    end: usize,
    pending: &[usize],
    max_cuts: usize,
) -> usize {
    pending
        .iter()
        .copied()
        .filter(|&b| b > entry && b < end)
        .nth(max_cuts)
        .unwrap_or(end)
}

/// Whether the MTP `pending` backup applies to a state sitting at `position`:
/// it was taken (`saved`) for exactly that window entry. The backup is
/// refreshed before EVERY window's transaction begins, so after a window
/// restores its entry state the matching backup is that window's own and
/// never an earlier window's (whose `pending` is the hidden of another row).
pub(crate) fn pending_backup_applies(
    saved: bool,
    saved_for: Option<usize>,
    position: usize,
) -> bool {
    saved && saved_for == Some(position)
}

/// The chunk rows [`Qwen4Bundle::attach_forward`] requests. With an admitted
/// slow-link profile it is exactly the profile's charged tile: it must fit
/// the S1 bound, the caller's capacity `max_chunk` and the PLE store's
/// `ple_rows_cap`, or the attach is refused before any allocation (never a
/// widened or narrowed chunk, never the incumbent helper). Without a profile
/// it is the incumbent request, unchanged.
pub(crate) fn attach_chunk_request(
    tile_rows: Option<usize>,
    max_chunk: usize,
    ple_rows_cap: usize,
    incumbent: impl FnOnce() -> usize,
) -> Result<usize, String> {
    let Some(tile) = tile_rows else {
        return Ok(incumbent().min(ple_rows_cap));
    };
    if tile == 0 || tile > PROMPT_TILE_ROWS_MAX {
        return Err(format!(
            "slow-link profile tile {tile} is outside 1..={PROMPT_TILE_ROWS_MAX}"
        ));
    }
    if max_chunk < tile {
        return Err(format!(
            "slow-link profile tile {tile} exceeds the caller's chunk capacity {max_chunk}"
        ));
    }
    if ple_rows_cap < tile {
        return Err(format!(
            "slow-link profile tile {tile} exceeds the PLE row store's {ple_rows_cap}-row prefetch"
        ));
    }
    Ok(tile)
}

/// The session cache's checkpoint cadence (the stride of its boundary list and
/// of the lineage split): a slow-link load's own `checkpoint_stride_rows`
/// when it has a profile, else the forward's chunk, exactly as before.
pub(crate) fn session_cadence(
    profile_stride: Option<usize>,
    forward_chunk: Option<usize>,
) -> Option<usize> {
    match profile_stride {
        Some(stride) if stride > 0 => Some(stride),
        _ => forward_chunk,
    }
}

/// A cache may use only the admitted profile's discretionary device slack,
/// never its safety margin or the free memory of an idle compute phase.
/// Without a profile the cache keeps its existing storage policy unchanged.
pub(crate) fn pressure_host_snapshot_policy(
    device_budget: Option<u64>,
    host_budget: impl FnOnce() -> u64,
) -> Option<hipfire_runtime::session_cache::PressureHostSnapshotPolicy> {
    device_budget.map(|device_budget_bytes| {
        hipfire_runtime::session_cache::PressureHostSnapshotPolicy {
            host_budget_bytes: host_budget(),
            device_budget_bytes,
        }
    })
}

/// Identity suffix of a slow-link load's session scope: its mixer tile AND
/// its checkpoint cadence. Empty for every other load, so their scopes are
/// byte-for-byte unchanged and a slow-link scope can never alias an explicit
/// chunk-1536 cadence or the incumbent chunk-8192 numerics.
pub(crate) fn slowlink_scope_tag(profile: Option<(usize, usize)>) -> String {
    profile.map_or_else(String::new, |(tile, cadence)| {
        format!("/s1tile{tile}cut{cadence}")
    })
}

/// Identity token an SP-armed load adds to its host-arithmetic identity: the
/// short-prompt hook runs the CPU experts' rows through the E1 F32/BF16
/// engine, a different prefix-state domain from every SP-off lineage.
pub(crate) const SP_HOST_IDENTITY: &str = "sp-e1v1";

/// `identity` (the D1/D2 host-arithmetic identity, empty when none is armed)
/// with [`SP_HOST_IDENTITY`] appended (`+`-joined) when the load armed SP.
/// Unchanged when SP is not armed, so every SP-off scope is byte for byte
/// what it was.
pub(crate) fn with_sp_identity(identity: String, sp_armed: bool) -> String {
    if !sp_armed {
        identity
    } else if identity.is_empty() {
        SP_HOST_IDENTITY.to_string()
    } else {
        format!("{identity}+{SP_HOST_IDENTITY}")
    }
}

/// PLE token history after `tokens` follow `entry`.
pub(crate) fn ple_history_after(entry: PleHistory, tokens: &[u32]) -> PleHistory {
    let mut history = entry;
    for &token in tokens {
        history.push(token);
    }
    history
}

/// Index of every fixed part in a stage: exactly the order of
/// `Qwen4State::session_layout` (GDN recurrent + conv per layer, then each QSA
/// layer's selection, then the PLE convolution, then the HC feedback).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StageLayout {
    pub(crate) gdn: usize,
    pub(crate) qsa: usize,
    pub(crate) ple: bool,
}

impl StageLayout {
    pub(crate) fn of(state: &Qwen4State) -> Self {
        Self {
            gdn: state.gdn.len(),
            qsa: state.qsa.len(),
            ple: state.ple_conv.is_some(),
        }
    }
    pub(crate) fn fixed_len(&self) -> usize {
        2 * self.gdn + self.qsa + usize::from(self.ple) + 1
    }
    pub(crate) fn gdn_index(&self, slot: usize) -> Option<usize> {
        (slot < self.gdn).then_some(2 * slot)
    }
    pub(crate) fn qsa_index(&self, slot: usize) -> Option<usize> {
        (slot < self.qsa).then_some(2 * self.gdn + slot)
    }
    pub(crate) fn ple_index(&self) -> Option<usize> {
        self.ple.then_some(2 * self.gdn + self.qsa)
    }
    pub(crate) fn feedback_index(&self) -> usize {
        self.fixed_len() - 1
    }
}

/// Host bookkeeping of the cuts of one window: which layers staged each cut,
/// the QSA marks the staging layers reported and the PLE token history at the
/// cut. A cut can be published only when every layer the state owns staged it.
#[derive(Debug)]
pub(crate) struct CutBook {
    cuts: Vec<usize>,
    history: Vec<PleHistory>,
    seen: Vec<Vec<bool>>,
    commits: Vec<Vec<Option<QsaCommit>>>,
    owned: usize,
    compress: usize,
    published: usize,
}

impl CutBook {
    /// `cuts`: ascending absolute positions strictly inside the window
    /// `[entry, entry + tokens.len())`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        entry: usize,
        tokens: &[u32],
        entry_history: PleHistory,
        cuts: Vec<usize>,
        layers: usize,
        owned: usize,
        qsa_slots: usize,
        compress: usize,
    ) -> Result<Self, String> {
        let end = entry + tokens.len();
        if compress == 0 {
            return Err("slow-link cut book: compress is zero".into());
        }
        if cuts.len() > PROMPT_TXN_MAX_CUTS {
            return Err(format!(
                "slow-link window has {} session cuts; at most {PROMPT_TXN_MAX_CUTS} are staged",
                cuts.len()
            ));
        }
        let mut previous = entry;
        for &cut in &cuts {
            if cut <= previous || cut >= end {
                return Err(format!(
                    "slow-link cut {cut} is not strictly inside ({previous}, {end})"
                ));
            }
            previous = cut;
        }
        let history = cuts
            .iter()
            .map(|&cut| ple_history_after(entry_history, &tokens[..cut - entry]))
            .collect();
        let count = cuts.len();
        Ok(Self {
            cuts,
            history,
            seen: vec![vec![false; layers]; count],
            commits: vec![vec![None; qsa_slots]; count],
            owned,
            compress,
            published: 0,
        })
    }

    pub(crate) fn cuts(&self) -> &[usize] {
        &self.cuts
    }

    fn visit(&mut self, cut: usize, layer: usize) -> Result<(), String> {
        let seen = self
            .seen
            .get_mut(cut)
            .ok_or_else(|| format!("slow-link cut index {cut} is outside the window"))?
            .get_mut(layer)
            .ok_or_else(|| format!("slow-link layer {layer} is outside the model"))?;
        if *seen {
            return Err(format!("slow-link layer {layer} staged cut {cut} twice"));
        }
        *seen = true;
        Ok(())
    }

    pub(crate) fn record_gdn(&mut self, cut: usize, layer: usize) -> Result<(), String> {
        self.visit(cut, layer)
    }

    /// A QSA layer staged `cut` with its private cursor `cursor`: the cursor
    /// must stand exactly at the cut.
    pub(crate) fn record_qsa(
        &mut self,
        cut: usize,
        layer: usize,
        slot: usize,
        cursor: QsaCommit,
    ) -> Result<(), String> {
        let position = *self
            .cuts
            .get(cut)
            .ok_or_else(|| format!("slow-link cut index {cut} is outside the window"))?;
        let (full, raw, pooled, _selected, at) = cursor;
        if full != position
            || raw != position
            || at != position
            || pooled != position / self.compress
        {
            return Err(format!(
                "slow-link QSA layer {layer} cursor {cursor:?} is not at cut {position}"
            ));
        }
        if slot >= self.commits[cut].len() {
            return Err(format!("slow-link QSA slot {slot} is outside the model"));
        }
        self.visit(cut, layer)?;
        self.commits[cut][slot] = Some(cursor);
        Ok(())
    }

    pub(crate) fn complete(&self, cut: usize) -> bool {
        self.seen
            .get(cut)
            .is_some_and(|seen| seen.iter().filter(|&&v| v).count() == self.owned)
            && self.commits[cut].iter().all(Option::is_some)
    }

    pub(crate) fn commits_for(&self, cut: usize) -> Result<Vec<QsaCommit>, String> {
        self.commits
            .get(cut)
            .ok_or_else(|| format!("slow-link cut index {cut} is outside the window"))?
            .iter()
            .map(|commit| commit.ok_or_else(|| format!("slow-link cut {cut} is not fully staged")))
            .collect()
    }

    pub(crate) fn history_at(&self, cut: usize) -> PleHistory {
        self.history[cut]
    }

    pub(crate) fn published(&self) -> usize {
        self.published
    }

    pub(crate) fn mark_published(&mut self) {
        self.published += 1;
    }

    /// The next unpublished cut's absolute position.
    pub(crate) fn next_cut(&self) -> Option<usize> {
        self.cuts.get(self.published).copied()
    }
}

/// The open transaction of a window.
pub(crate) struct LiveTxn {
    id: u64,
    entry: usize,
    rows: usize,
    state_ticket: Qwen4StateSnapshot,
    head_ticket: Option<MtpGpuStateSnapshot>,
    book: CutBook,
    scope: SlowlinkScope,
    ple_layer: Option<usize>,
}

/// Receipt of the most recent request, as stored.
pub(crate) enum StoredReceipt {
    Json(Value),
    Incumbent {
        reason: Option<std::borrow::Cow<'static, str>>,
        entry: usize,
        windows: Vec<(usize, usize)>,
        tile_rows: usize,
    },
}

/// Every slow-link owner field of a bundle.
#[derive(Default)]
pub(crate) struct SlowlinkOwner {
    pub(crate) profile: Option<crate::slowlink_admission::SlowlinkProfile>,
    pub(crate) scope: SlowlinkScope,
    /// Retained boundary stages (lazy, at most `PROMPT_TXN_MAX_CUTS`).
    pub(crate) stages: Vec<PairSessionStage>,
    live: Option<LiveTxn>,
    next_id: u64,
    receipt: Option<StoredReceipt>,
    receipt_fresh: bool,
    bypass: Option<std::borrow::Cow<'static, str>>,
    #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
    injection: Option<SlowlinkFailPoint>,
    #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
    injection_consumed: bool,
    /// Entry `pending` hidden of an MTP request, restored on failure.
    pending_save: Option<GpuTensor>,
    /// The entry `pending` hidden was copied into `pending_save` by THIS
    /// request's selected transaction (never before the selector's verdict).
    pending_saved: bool,
    /// The state position the backup was taken at (the entry of the window it
    /// guards): a backup is valid only for a state at exactly that position.
    pending_for: Option<usize>,
}

impl SlowlinkOwner {
    /// Every device tensor the owner holds (stage parts, the saved pending
    /// hidden), for teardown.
    pub(crate) fn into_tensors(self) -> Vec<GpuTensor> {
        let mut tensors: Vec<GpuTensor> = self
            .stages
            .into_iter()
            .flat_map(|stage| stage.fixed)
            .collect();
        tensors.extend(self.pending_save);
        tensors
    }

    /// Detach every retained boundary stage's device tensors (a failed load's
    /// unwind); the caller frees them.
    pub(crate) fn take_stage_tensors(&mut self) -> Vec<GpuTensor> {
        std::mem::take(&mut self.stages)
            .into_iter()
            .flat_map(|stage| stage.fixed)
            .collect()
    }

    /// Drop every open/armed item after a reset (the tickets died with it).
    pub(crate) fn clear_active(&mut self) {
        self.live = None;
        for stage in &mut self.stages {
            stage.position = None;
        }
    }
}

/// The incumbent-route receipt: every S1-only number is zero.
pub(crate) fn incumbent_receipt_json(
    selected_by: &str,
    bypass: Option<&str>,
    entry: usize,
    windows: &[(usize, usize)],
    tile_rows: usize,
) -> Value {
    let rows: usize = windows.iter().map(|window| window.1).sum();
    json!({
        "schema": SLOWLINK_RECEIPT_SCHEMA,
        "route": "incumbent",
        "selected_by": selected_by,
        "bypass_reason": bypass,
        "entry_pos": entry,
        "prompt_rows": rows,
        "windows": windows.iter().map(|&(start, rows)| json!({"start": start, "rows": rows})).collect::<Vec<_>>(),
        "launch_cuts": windows.iter().map(|&(start, rows)| start + rows).collect::<Vec<_>>(),
        "tile_rows": tile_rows,
        "retained_scratch_rows": tile_rows,
        "sweeps": 0u64,
        "resident_layers": 0u64,
        "host_layers": 0u64,
        "host_expert_bytes_total": 0u64,
        "submitted_bytes": 0u64,
        "useful_bytes": 0u64,
        "unused_prefetched_bytes": 0u64,
        "reserve": {"tile": 0u64, "arena": 0u64, "ring": 0u64, "lookahead": 0u64, "tables": 0u64, "total": 0u64},
        "spans_ms": {"wall": 0.0, "copy": 0.0, "compute": 0.0, "route_barrier": 0.0, "nonexpert": 0.0},
        "overlap": {
            "cross_layer_intersections": 0u64, "cross_layer_overlap_ms": 0.0,
            "copy_idle_ms": 0.0, "page_full_stalls": 0u64, "copy_starved_stalls": 0u64,
        },
        "peak_vram_bytes": 0u64,
    })
}

impl Qwen4Bundle {
    /// Hand the bundle its admitted slow-link profile (the loader, before
    /// [`Self::attach_forward`]); present on no other load.
    pub fn attach_slowlink_profile(&mut self, profile: crate::slowlink_admission::SlowlinkProfile) {
        self.slowlink.profile = Some(profile);
        if let Some(session) = self.session.as_mut() {
            if let Some(policy) =
                pressure_host_snapshot_policy(Some(profile.device_snapshot_budget_bytes), || {
                    session.cache().usage().budget_bytes
                })
            {
                session.enable_pressure_host_snapshots(policy);
            }
        }
    }

    pub fn slowlink_profile(&self) -> Option<&crate::slowlink_admission::SlowlinkProfile> {
        self.slowlink.profile.as_ref()
    }

    /// The session cache's boundary/lineage cadence of this bundle: the
    /// admitted profile's checkpoint stride, else the forward chunk.
    pub(crate) fn session_cadence_rows(&self) -> Option<usize> {
        session_cadence(
            self.slowlink
                .profile
                .map(|profile| profile.checkpoint_stride_rows),
            self.spec_chunk_rows(),
        )
    }

    /// Receipt of the most recent prefill request (`None` before any).
    pub fn slowlink_receipt(&self) -> Option<Value> {
        let mut receipt = match self.slowlink.receipt.as_ref()? {
            StoredReceipt::Json(value) => value.clone(),
            StoredReceipt::Incumbent {
                reason,
                entry,
                windows,
                tile_rows,
            } => {
                let selected_by = match crate::slowlink_admission::SlowlinkMode::from_env() {
                    crate::slowlink_admission::SlowlinkMode::Off => "env0",
                    crate::slowlink_admission::SlowlinkMode::Force => "env1",
                    crate::slowlink_admission::SlowlinkMode::Auto => "auto",
                };
                // A kill switch is "not asked", not a refusal.
                let reason = reason.as_deref().filter(|_| selected_by != "env0");
                let mut receipt =
                    incumbent_receipt_json(selected_by, reason, *entry, windows, *tile_rows);
                // The sticky SP verdict (knob on only): the incumbent ran, but
                // the receipt still names why SP is or is not armed.
                if let (Some(sp), Value::Object(map)) = (
                    self.execution
                        .as_ref()
                        .and_then(|forward| forward.slowlink.sp_receipt(&[])),
                    &mut receipt,
                ) {
                    map.insert("sp".to_string(), sp);
                }
                receipt
            }
        };
        if let Value::Object(map) = &mut receipt {
            #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
            let injection_consumed = self.slowlink.injection_consumed;
            #[cfg(not(any(test, feature = "lab", feature = "reference-parity")))]
            let injection_consumed = false;
            map.insert("injection_consumed".into(), json!(injection_consumed));
        }
        Some(receipt)
    }

    /// Arm (or clear) the one-shot failure of the NEXT layer-major prompt.
    /// Tools-only: gated `test` / `lab` / `reference-parity` like the rest of
    /// the injection API (the type, fields, parse and take branches). The
    /// one-shot failure is real rollback through the receipted transaction,
    /// not a stub; outside tools the API is unavailable and non-lab receipts
    /// truthfully report `injection_consumed: false`.
    #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
    pub fn slowlink_inject_failure(&mut self, point: Option<&str>) -> Result<(), BundleError> {
        self.slowlink.injection = match point {
            None => None,
            Some(name) => Some(SlowlinkFailPoint::parse(name).ok_or_else(|| {
                BundleError::Forward(format!(
                    "unknown slow-link failure point {name:?}; expected before_dma, during_panel, \
                     after_layer or before_commit"
                ))
            })?),
        };
        Ok(())
    }

    #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
    /// Consume the armed injection when it names `point`.
    pub(crate) fn slowlink_take_injection(&mut self, point: SlowlinkFailPoint) -> bool {
        if self.slowlink.injection == Some(point) {
            self.slowlink.injection = None;
            self.slowlink.injection_consumed = true;
            return true;
        }
        false
    }

    /// The orchestrator's complete receipt of the request (route, windows,
    /// bytes, spans); stored as given. The layer-major publication emits the
    /// `qwen4 prefill route` DEBUG event from the receipt's own borrowed
    /// scalars (`selected_by`, `entry_pos`, `prompt_rows`, `tile_rows`,
    /// `resident_layers`); the incumbent path emits at
    /// [`Self::slowlink_record_incumbent`].
    pub(crate) fn slowlink_set_receipt(&mut self, receipt: Value) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            let selected_by = receipt
                .get("selected_by")
                .and_then(Value::as_str)
                .unwrap_or("");
            let entry_pos = receipt
                .get("entry_pos")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let prompt_rows = receipt
                .get("prompt_rows")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let tile_rows = receipt
                .get("tile_rows")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let resident_layers = receipt
                .get("resident_layers")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            Self::trace_prefill_route(
                true,
                "layer-major",
                selected_by,
                None,
                entry_pos as usize,
                prompt_rows as usize,
                resident_layers as usize,
                tile_rows as usize,
            );
        }
        self.slowlink.receipt = Some(StoredReceipt::Json(receipt));
        self.slowlink.receipt_fresh = true;
    }

    /// Start a request's verdict with the incumbent route already decided
    /// for `reason` (before any S1 work).
    pub(crate) fn slowlink_note_bypass(
        &mut self,
        reason: impl Into<std::borrow::Cow<'static, str>>,
    ) {
        self.slowlink_begin_request();
        self.slowlink.bypass = Some(reason.into());
    }

    /// Production-primitive self-test on the loaded bundle's real admitted S1
    /// transport/dispatch owners. Returns
    /// `{"cases":[{"name","status","mismatches","detail"}, ..]}` for exactly
    /// the requested `cases`, in order; `status` is `"pass"`/`"fail"` with a
    /// measured `mismatches` (`u64`), or `"error"` with `mismatches: null` and
    /// `detail: {"error": <string naming that case>}` when that case's own
    /// setup/run failed or its report did not honor the frozen shape. Each
    /// named case is invoked independently (its own T/D setup, drain and
    /// restore): a failed case never stops, and is never reported on behalf
    /// of, another case, and a failed setup never carries a fabricated
    /// mismatch count. `Err` only for a global fault before any diagnostic
    /// work: an unknown/empty request, a load without an admitted profile, a
    /// bundle that cannot run it now (open window, other topology), or
    /// missing forward ownership. Never a pass by default. Tools-only.
    #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
    pub fn slowlink_primitive_selftest(
        &mut self,
        gpu: &mut Gpu,
        cases: &[&str],
    ) -> Result<Value, BundleError> {
        if cases.is_empty() {
            return Err(BundleError::Forward(
                "slow-link primitive check was asked for no cases".into(),
            ));
        }
        for &name in cases {
            if !SLOWLINK_PRIMITIVES.contains(&name) {
                return Err(BundleError::Forward(format!(
                    "unknown slow-link primitive {name:?}; expected one of {SLOWLINK_PRIMITIVES:?}"
                )));
            }
        }
        if self.slowlink.profile.is_none() {
            return Err(BundleError::Forward(
                "slow-link primitive check needs an admitted slow-link profile".into(),
            ));
        }
        self.slowlink_eligibility(SlowlinkScope::direct(), true)
            .map_err(|reason| {
                BundleError::Forward(format!("slow-link primitive check cannot run: {reason}"))
            })?;
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let report = collect_primitive_reports(cases, |name| {
            forward
                .slowlink_primitive_selftest(self, gpu, name)
                .map_err(|error| error.to_string())
        });
        self.execution = Some(forward);
        Ok(report)
    }

    fn slowlink_begin_request(&mut self) {
        self.slowlink.receipt_fresh = false;
        self.slowlink.bypass = None;
        #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
        {
            self.slowlink.injection_consumed = false;
        }
        self.slowlink.pending_saved = false;
        self.slowlink.pending_for = None;
    }

    /// Record the incumbent route's receipt (no-op when the orchestrator
    /// already set one for this request). `windows` are `(start, rows)`
    /// relative to `entry`.
    pub(crate) fn slowlink_record_incumbent(&mut self, entry: usize, windows: &[(usize, usize)]) {
        if self.slowlink.receipt_fresh {
            return;
        }
        let tile_rows = self
            .execution
            .as_ref()
            .map_or(0, |forward| forward.scratch.max_chunk);
        let prompt_rows: usize = windows.iter().map(|window| window.1).sum();
        let bypass_reason = self.slowlink.bypass.as_deref();
        let selected_by = match crate::slowlink_admission::SlowlinkMode::from_env() {
            crate::slowlink_admission::SlowlinkMode::Off => "env0",
            crate::slowlink_admission::SlowlinkMode::Force => "env1",
            crate::slowlink_admission::SlowlinkMode::Auto => "auto",
        };
        Self::trace_prefill_route(
            false,
            "incumbent",
            selected_by,
            bypass_reason,
            entry,
            prompt_rows,
            0,
            tile_rows,
        );
        self.slowlink.receipt = Some(StoredReceipt::Incumbent {
            reason: self.slowlink.bypass.take(),
            entry,
            windows: windows.to_vec(),
            tile_rows,
        });
        self.slowlink.receipt_fresh = true;
    }
    /// The DEBUG route event of one prefill request (`qwen4 prefill route`):
    /// the existing receipt's own scalars, borrowed at publication, never
    /// cloned for logging. Disabled (the default) it performs no allocation
    /// or work beyond the enabled check: every argument is a scalar or
    /// `&str`, and the one `Option<&str>` formats as empty when absent.
    fn trace_prefill_route(
        selected: bool,
        route: &str,
        selected_by: &str,
        bypass_reason: Option<&str>,
        entry_pos: usize,
        prompt_rows: usize,
        resident_layers: usize,
        tile_rows: usize,
    ) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!(
                selected,
                route,
                selected_by,
                bypass_reason = bypass_reason.unwrap_or(""),
                entry_pos,
                prompt_rows,
                resident_layers,
                tile_rows,
                "qwen4 prefill route",
            );
        }
    }

    /// Why this request cannot run layer-major, judged on host state alone
    /// (no mutation). Execution-owner checks apply only while it is attached;
    /// the orchestrator calls this with the forward taken out.
    pub(crate) fn slowlink_prompt_eligible(&self) -> Result<(), &'static str> {
        self.slowlink_eligibility(self.slowlink.scope, false)
    }

    /// [`Self::slowlink_prompt_eligible`] for an MTP capture request.
    pub(crate) fn slowlink_capture_eligible(&self) -> Result<(), &'static str> {
        if self.slowlink.profile.is_none() {
            return Err("no admitted slow-link profile");
        }
        self.slowlink_eligibility(SlowlinkScope::capture(), true)
    }

    fn slowlink_eligibility(
        &self,
        scope: SlowlinkScope,
        need_execution: bool,
    ) -> Result<(), &'static str> {
        if self.pp.is_some() {
            return Err("pipeline-parallel topology");
        }
        if self.ep.is_some() {
            return Err("expert-parallel topology");
        }
        if need_execution && self.execution.is_none() {
            return Err("forward resources are not attached");
        }
        if self.npu_trunk_planned() {
            return Err("NPU trunk planned for this request");
        }
        if self.state.row_capture_armed {
            return Err("speculative row capture armed");
        }
        // S1 lowers its own layer-major steps, which the DFlash2 layer
        // capture never hooks: the draft would read stale feature rows.
        if self.dflash.is_some() {
            return Err("DFlash2 target-layer capture armed");
        }
        if self.state.snapshot_active() {
            return Err("an outstanding state snapshot ticket");
        }
        if !self.state.scope_map().is_single() || self.state.hyper_feedback.is_none() {
            return Err("not the whole-model input band");
        }
        if self.slowlink.live.is_some() {
            return Err("a window transaction is already open");
        }
        if scope.head {
            let Some(mtp) = self.mtp.as_ref() else {
                return Err("no native MTP head attached");
            };
            if mtp.state.snapshot_active() {
                return Err("an outstanding MTP head snapshot ticket");
            }
            let (Some(_), Some(_), Some(hidden)) = (
                self.spec_logits.as_ref(),
                self.spec_top1.as_ref(),
                self.spec_hidden.as_ref(),
            ) else {
                return Err("speculative output buffers are not attached");
            };
            let width = self
                .config
                .hc_count
                .saturating_mul(self.config.hidden_size)
                .max(1);
            // The S1 forward fills the capture buffer one admitted tile at a
            // time (`slowlink_prefill` checks `n * wide` per tile), so the
            // buffer must hold the profile's tile, not the legal maximum.
            let tile_rows = self
                .slowlink
                .profile
                .as_ref()
                .map_or(PROMPT_TILE_ROWS_MAX, |profile| profile.tile_rows);
            if hidden.numel() / width < tile_rows {
                return Err("the wide-hidden capture buffer is narrower than a mixer tile");
            }
        }
        Ok(())
    }

    /// The one selector verdict of a request: run it layer-major through the
    /// orchestrator (`Ok(true)`) or leave it to the incumbent route
    /// (`Ok(false)`, decided before any mutation or S1 allocation).
    pub(crate) fn slowlink_dispatch(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        output: crate::slowlink_prefill::PromptOutput<'_>,
        scope: SlowlinkScope,
    ) -> Result<bool, BundleError> {
        self.slowlink_begin_request();
        if self.slowlink.profile.is_none() {
            self.slowlink.bypass = Some("no admitted slow-link profile".into());
            return Ok(false);
        }
        if tokens.is_empty() {
            self.slowlink.bypass = Some("empty request".into());
            return Ok(false);
        }
        if let Err(reason) = self.slowlink_eligibility(scope, true) {
            self.slowlink.bypass = Some(reason.into());
            return Ok(false);
        }
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let prior_scope = std::mem::replace(&mut forward.prefill_scope, true);
        self.slowlink.scope = scope;
        let result =
            crate::slowlink_prefill::try_run_prompt(&mut forward, self, gpu, tokens, output);
        self.slowlink.scope = SlowlinkScope::default();
        forward.prefill_scope = prior_scope;
        self.execution = Some(forward);
        // An orchestrator that returned with its window still open is
        // contained here exactly like an abort: never leave mixed positions.
        let leaked = match self.slowlink.live.take() {
            Some(live) => Some(self.slowlink_restore(gpu, live)),
            None => None,
        };
        match (result, leaked) {
            (Ok(ran), None) => {
                if !ran && !self.slowlink.receipt_fresh {
                    self.slowlink
                        .bypass
                        .get_or_insert_with(|| "selector declined the request".into());
                }
                Ok(ran)
            }
            (Ok(_), Some(outcome)) => Err(BundleError::Forward(format!(
                "slow-link prompt returned with its window transaction open ({})",
                outcome.err().unwrap_or_else(|| "state restored".into())
            ))),
            (Err(error), _) => {
                if !self.slowlink.receipt_fresh {
                    let selected = self.slowlink.bypass.take();
                    self.slowlink.receipt = Some(StoredReceipt::Json({
                        let mut value =
                            incumbent_receipt_json("auto", selected.as_deref(), 0, &[], 0);
                        if let Value::Object(map) = &mut value {
                            map.insert("status".into(), json!("error"));
                            map.insert("error".into(), json!(error.to_string()));
                        }
                        value
                    }));
                    self.slowlink.receipt_fresh = true;
                }
                Err(BundleError::Forward(error.to_string()))
            }
        }
    }

    /// MTP target capture through the selector. `Ok(Some(seed))` when the
    /// request ran layer-major (`seed` = the final row's argmax); `Ok(None)`
    /// leaves it to the incumbent per-chunk loop, with no allocation, copy or
    /// other S1 work done. `pending` (the drafter's `pending` hidden) is
    /// handed to the orchestrator through `PromptOutput::Capture::pending`:
    /// it backs it up with [`Self::slowlink_save_pending`] only AFTER the
    /// request was selected and before its first phase.
    pub(crate) fn slowlink_try_capture(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        pending: &GpuTensor,
        sink: &mut dyn FnMut(
            &mut Gpu,
            &mut Qwen4Bundle,
            crate::slowlink_prefill::PromptTile<'_>,
        ) -> Result<(), String>,
    ) -> Result<Option<u32>, BundleError> {
        let (logits, top1) = match (self.spec_logits.as_ref(), self.spec_top1.as_ref()) {
            (Some(logits), Some(top1)) => (
                logits.sub_offset(0, self.config.vocab_size),
                top1.sub_offset(0, std::mem::size_of::<i32>()),
            ),
            _ => {
                return Err(BundleError::Forward(
                    "Qwen4 spec output buffers are not attached".to_string(),
                ))
            }
        };
        let output = crate::slowlink_prefill::PromptOutput::Capture {
            sink,
            head: Some((&logits, &top1)),
            pending: Some(pending),
        };
        if !self.slowlink_dispatch(gpu, tokens, output, SlowlinkScope::capture())? {
            return Ok(None);
        }
        let bytes = std::mem::size_of::<i32>();
        if self.spec_host_top1.len() < bytes {
            return Err(BundleError::Forward(
                "Qwen4 spec host argmax capacity is too small".to_string(),
            ));
        }
        gpu.hip
            .memcpy_dtoh(&mut self.spec_host_top1[..bytes], &top1.buf)
            .map_err(BundleError::Hip)?;
        gpu.take_device_wait_error().map_err(BundleError::Hip)?;
        let b = &self.spec_host_top1[..bytes];
        Ok(Some(u32::from_ne_bytes([b[0], b[1], b[2], b[3]])))
    }

    /// Keep the entry `pending` hidden of an MTP request so a failure can
    /// put it back (the drafter owns `pending`; the copy is stream ordered).
    /// Called by the orchestrator once per window, after the request was
    /// selected and before its first phase. The backup buffer is bound at MTP
    /// attach ([`Self::slowlink_bind_pending_save`]); a request never
    /// allocates it.
    pub(crate) fn slowlink_save_pending(
        &mut self,
        gpu: &mut Gpu,
        pending: &GpuTensor,
    ) -> Result<(), BundleError> {
        let saved = self.slowlink.pending_save.as_ref().ok_or_else(|| {
            BundleError::Forward("the MTP pending backup was not bound at load".into())
        })?;
        copy_whole(gpu, pending, saved)?;
        self.slowlink.pending_saved = true;
        self.slowlink.pending_for = Some(self.state.position);
        Ok(())
    }

    /// The MTP entry `pending` backup buffer exists (bound at MTP attach).
    pub(crate) fn slowlink_pending_bound(&self) -> bool {
        self.slowlink.pending_save.is_some()
    }

    /// Bind the native-MTP entry `pending` backup (one F32 wide row,
    /// `hc_count * hidden`: the census' `pending_save_bytes`) at MTP attach,
    /// on a load that admitted a slow-link profile. Idempotent; no profile,
    /// no allocation.
    pub(crate) fn slowlink_bind_pending_save(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        if self.slowlink.profile.is_none() || self.slowlink.pending_save.is_some() {
            return Ok(());
        }
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("MTP pending backup width overflow".into()))?;
        self.slowlink.pending_save =
            Some(gpu.zeros(&[width], DType::F32).map_err(BundleError::Hip)?);
        Ok(())
    }

    /// The backup taken for the window now at the live position is available:
    /// a state restored to a window's entry restores exactly that window's
    /// `pending` (the hidden of `entry - 1`), never an earlier window's.
    pub(crate) fn slowlink_pending_restorable(&self) -> bool {
        pending_backup_applies(
            self.slowlink.pending_saved,
            self.slowlink.pending_for,
            self.state.position,
        )
    }

    /// Put the backed-up `pending` hidden back; an error when no backup was
    /// taken for the window the state now sits at (so an unselected request,
    /// or a later window, cannot restore stale bytes).
    pub(crate) fn slowlink_restore_pending(
        &mut self,
        gpu: &mut Gpu,
        pending: &GpuTensor,
    ) -> Result<(), BundleError> {
        if !self.slowlink_pending_restorable() {
            return Err(BundleError::Forward(
                "no MTP pending hidden was saved for the window the state was restored to".into(),
            ));
        }
        let saved = self
            .slowlink
            .pending_save
            .as_ref()
            .ok_or_else(|| BundleError::Forward("no saved MTP pending hidden".into()))?;
        copy_whole(gpu, saved, pending)
    }

    // ----- planning (host only) -----

    pub(crate) fn prompt_txn_max_cuts(&self) -> usize {
        PROMPT_TXN_MAX_CUTS
    }

    fn prompt_txn_pending(&self) -> Vec<usize> {
        self.session
            .as_ref()
            .map(SessionDriver::turn_boundaries)
            .unwrap_or_default()
    }

    /// Pending session boundaries strictly inside `(entry, end)`; empty for a
    /// direct (non-session) request.
    pub(crate) fn prompt_txn_cuts(&self, entry: usize, end: usize) -> Vec<usize> {
        if !self.slowlink.scope.cuts {
            return Vec::new();
        }
        prompt_cuts_in(entry, end, &self.prompt_txn_pending())
    }

    /// Largest window end `<= end` whose strictly-inside cuts fit the stages.
    pub(crate) fn prompt_txn_window_end(&self, entry: usize, end: usize) -> usize {
        if !self.slowlink.scope.cuts {
            return end;
        }
        prompt_window_end(entry, end, &self.prompt_txn_pending(), PROMPT_TXN_MAX_CUTS)
    }

    /// Bytes of ONE boundary stage (the target's fixed parts), exactly what
    /// the lazy allocation takes.
    pub(crate) fn prompt_txn_stage_bytes(&self) -> u64 {
        self.state
            .session_parts()
            .map(|(_, layout)| {
                layout
                    .fixed
                    .iter()
                    .map(|part| part.bytes as u64)
                    .sum::<u64>()
            })
            .unwrap_or(0)
    }

    /// Allocate the retained boundary stages ONCE, at the selected load
    /// ([`PROMPT_TXN_MAX_CUTS`] of them, the census' `boundary_state`): the
    /// single allocation owner of the stage parts.
    pub(crate) fn slowlink_bind_boundary_stages(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<(), BundleError> {
        self.allocate_prompt_stages(gpu, PROMPT_TXN_MAX_CUTS)
    }

    /// Detach every retained boundary stage's tensors for a failed load's
    /// unwind; the caller frees them.
    pub(crate) fn slowlink_take_boundary_stage_tensors(&mut self) -> Vec<GpuTensor> {
        self.slowlink.take_stage_tensors()
    }

    /// A window with `count` cuts finds its stages already bound (a
    /// transaction never allocates them).
    fn ensure_prompt_stages(&self, count: usize) -> Result<(), BundleError> {
        if self.slowlink.stages.len() >= count {
            return Ok(());
        }
        Err(BundleError::Forward(format!(
            "slow-link boundary stage {count} was not bound at load ({} bound)",
            self.slowlink.stages.len()
        )))
    }

    fn allocate_prompt_stages(&mut self, gpu: &mut Gpu, count: usize) -> Result<(), BundleError> {
        if self.slowlink.stages.len() >= count {
            return Ok(());
        }
        let sizes: Vec<usize> = {
            let (_, layout) = self.state.session_parts().map_err(BundleError::State)?;
            layout.fixed.iter().map(|part| part.bytes).collect()
        };
        if sizes.len() != StageLayout::of(&self.state).fixed_len() {
            return Err(BundleError::Forward(
                "slow-link stage layout differs from the session layout".into(),
            ));
        }
        while self.slowlink.stages.len() < count {
            let mut fixed = Vec::with_capacity(sizes.len());
            for &bytes in &sizes {
                match gpu.zeros(&[bytes], DType::Raw) {
                    Ok(tensor) => fixed.push(tensor),
                    Err(error) => {
                        for tensor in fixed {
                            let _ = gpu.free_tensor(tensor);
                        }
                        return Err(BundleError::Hip(error));
                    }
                }
            }
            self.slowlink.stages.push(PairSessionStage {
                fixed,
                meta: Vec::new(),
                rows: Vec::new(),
                position: None,
            });
        }
        Ok(())
    }

    // ----- transaction -----

    /// Open the transaction of one window of `tokens` at the live position.
    /// Everything that can refuse runs before the entry snapshot is taken.
    pub(crate) fn prompt_txn_begin(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
    ) -> Result<PromptTxn, BundleError> {
        let scope = self.slowlink.scope;
        self.slowlink_eligibility(scope, false).map_err(|reason| {
            BundleError::Forward(format!("slow-link window transaction refused: {reason}"))
        })?;
        if tokens.is_empty() {
            return Err(BundleError::Forward(
                "slow-link window transaction has no rows".into(),
            ));
        }
        let entry = self.state.position;
        let end = entry
            .checked_add(tokens.len())
            .filter(|&end| end <= self.state.max_seq_len)
            .ok_or_else(|| {
                BundleError::Forward("slow-link window exceeds the context capacity".into())
            })?;
        if scope.head && self.mtp_position()? != entry {
            return Err(BundleError::Forward(
                "slow-link window: the MTP head is not at the entry position".into(),
            ));
        }
        // Each window backs its entry `pending` up before it begins (the
        // orchestrator saves before EVERY begin): a window whose entry has no
        // backup would restore another window's hidden on failure.
        if scope.head
            && !pending_backup_applies(
                self.slowlink.pending_saved,
                self.slowlink.pending_for,
                entry,
            )
        {
            return Err(BundleError::Forward(
                "slow-link window: the MTP pending hidden was not saved for this window's entry"
                    .into(),
            ));
        }
        let cuts = if scope.cuts {
            prompt_cuts_in(entry, end, &self.prompt_txn_pending())
        } else {
            Vec::new()
        };
        let book = CutBook::new(
            entry,
            tokens,
            self.state.ple_history,
            cuts,
            self.config.num_hidden_layers,
            self.state.gdn.len() + self.state.qsa.len(),
            self.state.qsa.len(),
            self.config.indexer_compress_ratio,
        )
        .map_err(BundleError::Forward)?;
        self.ensure_prompt_stages(book.cuts().len())?;
        self.state
            .ensure_mapped_capacity(gpu, end)
            .map_err(BundleError::State)?;
        self.quiesce_ple()?;
        // Past this point only the snapshot arenas change; the live state is
        // untouched until the orchestrator's first layer runs.
        let state_ticket = self.state.snapshot(gpu).map_err(BundleError::State)?;
        let head_ticket = if scope.head {
            match self.mtp_snapshot(gpu) {
                Ok(ticket) => Some(ticket),
                Err(error) => {
                    let _ = self.state.commit(state_ticket, gpu);
                    return Err(error);
                }
            }
        } else {
            None
        };
        // The HC feedback is read-only on this route: it joins every stage now.
        let feedback_index = StageLayout::of(&self.state).feedback_index();
        let copied = (|| -> Result<(), BundleError> {
            let feedback = self.state.hyper_feedback.as_ref().ok_or_else(|| {
                BundleError::Forward("slow-link: hyper feedback is not owned".into())
            })?;
            for stage in &self.slowlink.stages[..book.cuts().len()] {
                stage.copy(gpu, feedback_index, feedback)?;
            }
            Ok(())
        })();
        if let Err(error) = copied {
            self.slowlink_release_tickets(gpu, state_ticket, head_ticket);
            return Err(error);
        }
        let ple_layer = self
            .config
            .ple
            .as_ref()
            .and_then(|_| self.config.ple_layer_ids.first())
            .and_then(|id| id.checked_sub(1));
        self.slowlink.next_id += 1;
        let id = self.slowlink.next_id;
        self.slowlink.live = Some(LiveTxn {
            id,
            entry,
            rows: tokens.len(),
            state_ticket,
            head_ticket,
            book,
            scope,
            ple_layer,
        });
        Ok(PromptTxn { id })
    }

    fn slowlink_release_tickets(
        &mut self,
        gpu: &mut Gpu,
        state_ticket: Qwen4StateSnapshot,
        head_ticket: Option<MtpGpuStateSnapshot>,
    ) {
        let _ = self.state.commit(state_ticket, gpu);
        if let Some(head) = head_ticket {
            if self.mtp_validate_commit(head).is_ok() {
                self.mtp_commit_validated(head);
            }
        }
    }

    /// Layer `layer` finished the tile that ends at cut `cut_index`: stage its
    /// fixed state (before its next tile advances it).
    pub(crate) fn prompt_txn_layer_cut(
        &mut self,
        gpu: &mut Gpu,
        txn: &mut PromptTxn,
        cut_index: usize,
        layer: usize,
    ) -> Result<(), BundleError> {
        let layout = StageLayout::of(&self.state);
        let SlowlinkOwner { live, stages, .. } = &mut self.slowlink;
        let live = live
            .as_mut()
            .filter(|live| live.id == txn.id)
            .ok_or_else(|| {
                BundleError::Forward("slow-link window transaction is not open".into())
            })?;
        let stage = stages
            .get(cut_index)
            .filter(|_| cut_index < live.book.cuts().len())
            .ok_or_else(|| {
                BundleError::Forward(format!(
                    "slow-link cut index {cut_index} is outside the window"
                ))
            })?;
        let missing = || {
            BundleError::Forward(format!(
                "slow-link layer {layer} is not owned by this state"
            ))
        };
        if let Some(slot) = self.state.gdn_local(layer) {
            let gdn = self.state.gdn.get(slot).ok_or_else(missing)?;
            let index = layout.gdn_index(slot).ok_or_else(missing)?;
            stage.copy_parts(gpu, index, [&gdn.recurrent, &gdn.conv])?;
            live.book
                .record_gdn(cut_index, layer)
                .map_err(BundleError::Forward)?;
        } else if let Some(slot) = self.state.qsa_local(layer) {
            let qsa = self.state.qsa.get(slot).ok_or_else(missing)?;
            let index = layout.qsa_index(slot).ok_or_else(missing)?;
            let cursor = self.state.qsa_cursor(slot).ok_or_else(missing)?;
            // Validate before the copy: a cursor that is not at the cut is a
            // schedule bug, never a snapshot to publish.
            live.book
                .record_qsa(cut_index, layer, slot, cursor)
                .map_err(BundleError::Forward)?;
            stage.copy(gpu, index, &qsa.selected_indices)?;
        } else {
            return Err(missing());
        }
        if live.ple_layer == Some(layer) {
            let ple = self.state.ple_conv.as_ref().ok_or_else(|| {
                BundleError::Forward("slow-link: PLE convolution state missing".into())
            })?;
            let index = layout.ple_index().ok_or_else(missing)?;
            stage.copy(gpu, index, ple)?;
        }
        Ok(())
    }

    /// Cut `position` is the next unpublished cut of the open window.
    pub(crate) fn slowlink_cut_due(&self, position: usize) -> bool {
        self.slowlink
            .live
            .as_ref()
            .is_some_and(|live| live.book.next_cut() == Some(position))
    }

    /// MTP route: publish the next cut now. The target part comes from the
    /// cut's stage; the head is live at the cut (the sink just appended the
    /// tile) and `pending` holds the target hidden of `cut - 1`.
    pub(crate) fn slowlink_publish_cut_mtp(
        &mut self,
        gpu: &mut Gpu,
        pending: &GpuTensor,
    ) -> Result<(), BundleError> {
        let Some(mut live) = self.slowlink.live.take() else {
            return Err(BundleError::Forward(
                "slow-link window transaction is not open".into(),
            ));
        };
        let index = live.book.published();
        let result = if !live.scope.head || index >= live.book.cuts().len() {
            Err(BundleError::Forward("slow-link: no cut is due".into()))
        } else if !live.book.complete(index) {
            Err(BundleError::Forward(format!(
                "slow-link cut {index} is not staged by every layer"
            )))
        } else {
            self.slowlink_publish_cut_at(gpu, &live, index, Some(pending))
        };
        if result.is_ok() {
            live.book.mark_published();
        }
        self.slowlink.live = Some(live);
        result
    }

    fn slowlink_publish_cut_at(
        &mut self,
        gpu: &mut Gpu,
        live: &LiveTxn,
        index: usize,
        pending: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let position = live.book.cuts()[index];
        let commits = live.book.commits_for(index).map_err(BundleError::Forward)?;
        {
            let stage = self
                .slowlink
                .stages
                .get_mut(index)
                .ok_or_else(|| BundleError::Forward("slow-link stage is missing".into()))?;
            stage.position = Some(position);
            stage.configure(&self.state, live.book.history_at(index), &commits)?;
        }
        let result = (|| -> Result<(), BundleError> {
            if let Some(pending) = pending {
                self.session_save_boundary_hidden(gpu, pending)?;
            }
            if self.session_next_boundary() != Some(position) {
                return Err(BundleError::Forward(format!(
                    "slow-link cut {position} is not the session's next boundary"
                )));
            }
            self.with_session(|session, bundle| session.at_turn_boundary(gpu, bundle, position))
                .unwrap_or(Ok(()))
                .map_err(BundleError::Forward)
        })();
        if let Some(stage) = self.slowlink.stages.get_mut(index) {
            stage.position = None;
        }
        result
    }

    /// The one global commit of the window: publish, advance, consume the
    /// entry tickets. Any failure restores the entry state first.
    pub(crate) fn prompt_txn_commit(
        &mut self,
        gpu: &mut Gpu,
        txn: PromptTxn,
        tokens: &[u32],
    ) -> Result<(), BundleError> {
        let live = match self.slowlink.live.take() {
            Some(live) if live.id == txn.id => live,
            other => {
                self.slowlink.live = other;
                return Err(BundleError::Forward(
                    "slow-link window transaction is not open".into(),
                ));
            }
        };
        if tokens.len() != live.rows {
            let cause = format!(
                "slow-link commit of {} rows closes a {}-row window",
                tokens.len(),
                live.rows
            );
            return Err(self.slowlink_fail(gpu, live, &cause));
        }
        #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
        if self.slowlink_take_injection(SlowlinkFailPoint::BeforeCommit) {
            return Err(self.slowlink_fail(gpu, live, "injected slow-link failure before commit"));
        }
        let mut live = live;
        match self.slowlink_commit_inner(gpu, &mut live, tokens) {
            Ok(()) => {
                self.state.commit_validated(live.state_ticket);
                if let Some(head) = live.head_ticket {
                    self.mtp_commit_validated(head);
                }
                Ok(())
            }
            Err(error) => Err(self.slowlink_fail(gpu, live, &error.to_string())),
        }
    }

    fn slowlink_commit_inner(
        &mut self,
        gpu: &mut Gpu,
        live: &mut LiveTxn,
        tokens: &[u32],
    ) -> Result<(), BundleError> {
        for cut in 0..live.book.cuts().len() {
            if !live.book.complete(cut) {
                return Err(BundleError::Forward(format!(
                    "slow-link cut {cut} is not staged by every layer"
                )));
            }
        }
        if live.scope.head && live.book.published() != live.book.cuts().len() {
            return Err(BundleError::Forward(
                "slow-link: the head sink left a cut unpublished".into(),
            ));
        }
        self.state
            .validate_commit(live.state_ticket)
            .map_err(BundleError::State)?;
        if let Some(head) = live.head_ticket {
            self.mtp_validate_commit(head)?;
        }
        self.state
            .commit_prompt_window(tokens)
            .map_err(BundleError::State)?;
        if !live.scope.head {
            while live.book.published() < live.book.cuts().len() {
                let index = live.book.published();
                self.slowlink_publish_cut_at(gpu, live, index, None)?;
                live.book.mark_published();
            }
        }
        let end = live.entry + live.rows;
        if live.scope.publish_end && self.session_next_boundary() == Some(end) {
            self.with_session(|session, bundle| session.at_turn_boundary(gpu, bundle, end))
                .unwrap_or(Ok(()))
                .map_err(BundleError::Forward)?;
        }
        Ok(())
    }

    /// Abort the open window: restore the entry state. A no-op `Ok` when the
    /// transaction was already consumed (a failed commit restored it).
    pub(crate) fn prompt_txn_abort(
        &mut self,
        gpu: &mut Gpu,
        txn: PromptTxn,
    ) -> Result<(), BundleError> {
        let live = match self.slowlink.live.take() {
            Some(live) if live.id == txn.id => live,
            Some(other) => {
                self.slowlink.live = Some(other);
                return Err(BundleError::Forward(
                    "slow-link abort names another window transaction".into(),
                ));
            }
            None => return Ok(()),
        };
        self.slowlink_restore(gpu, live)
            .map_err(BundleError::Forward)
    }

    fn slowlink_fail(&mut self, gpu: &mut Gpu, live: LiveTxn, cause: &str) -> BundleError {
        match self.slowlink_restore(gpu, live) {
            Ok(()) => BundleError::Forward(format!("{cause}; entry state restored")),
            Err(containment) => BundleError::Forward(format!("{cause}; {containment}")),
        }
    }

    /// Put the entry state back: device drained, target (and head) restored
    /// from the entry snapshot, derived arenas and the PLE epoch invalidated,
    /// uncommitted session captures dropped. A restore that cannot complete
    /// is contained by a full reset and reported.
    fn slowlink_restore(&mut self, gpu: &mut Gpu, live: LiveTxn) -> Result<(), String> {
        let mut failure: Option<String> = gpu
            .hip
            .device_synchronize()
            .err()
            .map(|error| format!("device drain failed: {error}"));
        if failure.is_none() {
            if let Err(error) = self.invalidate_ple_epoch() {
                failure = Some(format!("PLE epoch invalidation failed: {error}"));
            }
        }
        if failure.is_none() {
            if let Err(error) = self.state.restore(gpu, live.state_ticket) {
                failure = Some(format!("state restore failed: {error}"));
            }
        }
        if failure.is_none() {
            if let Some(head) = live.head_ticket {
                if let Err(error) = self.mtp_restore(gpu, head) {
                    failure = Some(format!("MTP head restore failed: {error}"));
                }
            }
        }
        self.state.invalidate_f16_arenas();
        for stage in &mut self.slowlink.stages {
            stage.position = None;
        }
        if let Some(session) = self.session.as_mut() {
            session.abort(gpu);
        }
        match failure {
            None => Ok(()),
            Some(reason) => {
                let reset = self.reset(gpu);
                Err(format!(
                    "{reason}; state reset to contain it ({})",
                    reset.map_or_else(|error| error.to_string(), |()| "ok".to_string())
                ))
            }
        }
    }
}

/// Stream-ordered whole-tensor device copy (no host synchronisation).
fn copy_whole(gpu: &mut Gpu, src: &GpuTensor, dst: &GpuTensor) -> Result<(), BundleError> {
    if src.byte_size() != dst.byte_size() || !src.byte_size().is_multiple_of(4) {
        return Err(BundleError::Forward(
            "slow-link pending hidden copy needs equal aligned tensors".into(),
        ));
    }
    let regions = [rdna_compute::tensor_ops::CopyRegion {
        dst: &dst.buf,
        dst_offset: 0,
        src: &src.buf,
        src_offset: 0,
        bytes: src.byte_size(),
    }];
    rdna_compute::tensor_ops::copy_regions(gpu, &regions).map_err(BundleError::Hip)
}

/// CPU models (no HIP) of the slow-link window transaction, driven by the
/// production planning structures: the session cache's own boundary list
/// (`stride_boundaries`), the incumbent segment rule
/// (`qwen4_prefill_segment_rows`), the whole-model state map of the compact
/// profile, and this file's `prompt_cuts_in` / `prompt_window_end` /
/// `StageLayout` / `CutBook` / `ple_history_after`. The model replays the
/// layer-major order (each layer runs every tile before the next layer),
/// stages cuts per layer, commits once, and is compared against the
/// sequential token-by-token state at every published cut.
#[cfg(test)]
mod slowlink_txn_tests {
    use super::*;
    use crate::config::compact_test_config;
    use crate::gpu_forward::{qwen4_prefill_segment_rows, SessionBoundaries};
    use crate::state::Qwen4StateMap;
    use crate::weights::Qwen4BandScope;
    use hipfire_runtime::session_cache::stride_boundaries;

    const END: u32 = 7;
    const TILE: usize = PROMPT_TILE_ROWS_MAX;
    /// Boundaries of the 9000-row fixture: cuts at 3, 4, 1536, 1537 and 8192
    /// plus the 1536-stride multiples between them.
    const BOUNDARIES: [usize; 9] = [3, 4, 1536, 1537, 3072, 4608, 6144, 7680, 8192];

    fn fixture_prompt() -> Vec<u32> {
        let mut prompt = vec![1u32; 9000];
        for index in [2, 3, 1536, 8191] {
            prompt[index] = END;
        }
        prompt
    }

    fn tokens(count: usize) -> Vec<u32> {
        (0..count)
            .map(|i| ((i * 7919 + 13) % 1009) as u32)
            .collect()
    }

    struct Ctx {
        gdn_slot: Vec<Option<usize>>,
        qsa_slot: Vec<Option<usize>>,
        ple_layer: Option<usize>,
        layout: StageLayout,
        compress: usize,
    }

    fn ctx() -> Ctx {
        let config = compact_test_config();
        let scope = Qwen4BandScope::single(&config);
        let map = Qwen4StateMap::new(&config, &scope).expect("single-band map");
        let layers = config.num_hidden_layers;
        Ctx {
            gdn_slot: (0..layers).map(|layer| map.gdn_local(layer)).collect(),
            qsa_slot: (0..layers).map(|layer| map.qsa_local(layer)).collect(),
            ple_layer: config.ple.as_ref().and_then(|_| {
                config
                    .ple_layer_ids
                    .first()
                    .and_then(|id| id.checked_sub(1))
            }),
            layout: StageLayout {
                gdn: map.gdn_count(),
                qsa: map.qsa_count(),
                ple: config.ple.is_some(),
            },
            compress: config.indexer_compress_ratio,
        }
    }

    fn mix(a: u64, b: u64) -> u64 {
        (a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .rotate_left(29)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9)
    }

    fn step(state: u64, layer: usize, position: usize, token: u32) -> u64 {
        mix(
            mix(state, layer as u64 + 1),
            mix(position as u64, u64::from(token)),
        )
    }

    fn marks(position: usize, compress: usize) -> QsaCommit {
        (
            position,
            position,
            position / compress,
            position.min(2048),
            position,
        )
    }

    /// Everything one session snapshot of the target carries.
    #[derive(Clone, Debug, PartialEq)]
    struct Pub {
        position: usize,
        layer: Vec<u64>,
        qsa: Vec<QsaCommit>,
        ple: u64,
        feedback: u64,
        history: PleHistory,
    }

    /// The owners of the transaction: per-layer fixed state, QSA cursors, PLE
    /// convolution, HC feedback, PLE token history, the global position and
    /// (MTP) the head position and its `pending` hidden row.
    #[derive(Clone, Debug, PartialEq)]
    struct M {
        layer: Vec<u64>,
        qsa: Vec<QsaCommit>,
        ple: u64,
        feedback: u64,
        history: PleHistory,
        position: usize,
        head_pos: usize,
        pending: Option<usize>,
    }

    impl M {
        fn new(ctx: &Ctx) -> Self {
            Self {
                layer: vec![0; ctx.gdn_slot.len()],
                qsa: vec![marks(0, ctx.compress); ctx.layout.qsa],
                ple: 0,
                feedback: 0xFEED,
                history: PleHistory::new(2),
                position: 0,
                head_pos: 0,
                pending: None,
            }
        }

        fn advance_layer(&mut self, ctx: &Ctx, layer: usize, from: usize, to: usize, all: &[u32]) {
            for position in from..to {
                self.layer[layer] = step(self.layer[layer], layer, position, all[position]);
                if ctx.ple_layer == Some(layer) {
                    self.ple = step(self.ple, 9999, position, all[position]);
                }
            }
            if let Some(slot) = ctx.qsa_slot[layer] {
                self.qsa[slot] = marks(to, ctx.compress);
            }
        }

        fn snapshot(&self) -> Pub {
            Pub {
                position: self.position,
                layer: self.layer.clone(),
                qsa: self.qsa.clone(),
                ple: self.ple,
                feedback: self.feedback,
                history: self.history,
            }
        }
    }

    /// Sequential (chunk-major, token by token) state at `to`.
    fn reference(ctx: &Ctx, entry: &M, all: &[u32], to: usize) -> M {
        let mut m = entry.clone();
        for layer in 0..ctx.gdn_slot.len() {
            m.advance_layer(ctx, layer, entry.position, to, all);
        }
        for &token in &all[entry.position..to] {
            m.history.push(token);
        }
        m.position = to;
        m.head_pos = to;
        m.pending = to.checked_sub(1);
        m
    }

    fn entry_at(ctx: &Ctx, all: &[u32], position: usize) -> M {
        reference(ctx, &M::new(ctx), all, position)
    }

    #[derive(Clone)]
    struct Stage {
        fixed: Vec<u64>,
        position: Option<usize>,
    }

    /// Rebuild the snapshot a cut publishes from its stage, exactly through
    /// the production index mapping and the book's marks and history.
    fn pub_from_stage(ctx: &Ctx, stage: &Stage, book: &CutBook, cut: usize) -> Pub {
        let layout = ctx.layout;
        let layer = (0..ctx.gdn_slot.len())
            .map(|layer| match (ctx.gdn_slot[layer], ctx.qsa_slot[layer]) {
                (Some(slot), None) => stage.fixed[layout.gdn_index(slot).unwrap()],
                (None, Some(slot)) => stage.fixed[layout.qsa_index(slot).unwrap()],
                _ => panic!("layer {layer} is neither GDN nor QSA"),
            })
            .collect();
        Pub {
            position: book.cuts()[cut],
            layer,
            qsa: book.commits_for(cut).unwrap(),
            ple: layout.ple_index().map_or(0, |index| stage.fixed[index]),
            feedback: stage.fixed[layout.feedback_index()],
            history: book.history_at(cut),
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Inject {
        None,
        BeforeDma,
        DuringPanel(usize),
        AfterLayer(usize),
        BeforeCommit,
        Sink(usize),
    }

    /// The drafter-side backup of `pending` (`slowlink_save_pending`), taken
    /// before EVERY window's begin.
    #[derive(Clone, Copy)]
    struct Saved {
        saved: bool,
        for_position: Option<usize>,
        pending: Option<usize>,
    }

    impl Saved {
        fn take(m: &M) -> Self {
            Self {
                saved: true,
                for_position: Some(m.position),
                pending: m.pending,
            }
        }

        /// `mtp_spec`'s error path, run after the transaction restored the
        /// target and head: `pending` comes back only from a backup that
        /// applies to the position the owners were restored to.
        fn restore_into(&self, m: &mut M) {
            if pending_backup_applies(self.saved, self.for_position, m.position) {
                m.pending = self.pending;
            }
        }
    }

    struct Win<'a> {
        all: &'a [u32],
        entry: usize,
        end: usize,
        /// Pending session boundaries above `entry`.
        boundaries: Vec<usize>,
        head: bool,
    }

    /// The tile ends of a window: every `TILE` rows, cut at each session cut.
    fn tile_exits(entry: usize, end: usize, cuts: &[usize]) -> Vec<usize> {
        let mut exits = Vec::new();
        let mut start = entry;
        while start < end {
            let mut exit = (start + TILE).min(end);
            if let Some(&cut) = cuts.iter().find(|&&cut| cut > start) {
                exit = exit.min(cut);
            }
            exits.push(exit);
            start = exit;
        }
        exits
    }

    /// One window transaction over the model. `Err` leaves `m` at the entry
    /// state, every stage unarmed and nothing published.
    fn run_window(
        ctx: &Ctx,
        m: &mut M,
        stages: &mut Vec<Stage>,
        w: &Win<'_>,
        inject: Inject,
        out: &mut Vec<Pub>,
    ) -> Result<(), String> {
        let entry_state = m.clone();
        let mut staged_out: Vec<Pub> = Vec::new();
        let result = (|| -> Result<(), String> {
            let tokens = &w.all[w.entry..w.end];
            let cuts = prompt_cuts_in(w.entry, w.end, &w.boundaries);
            let layers = ctx.gdn_slot.len();
            if inject == Inject::BeforeDma {
                return Err("before_dma".into());
            }
            let mut book = CutBook::new(
                w.entry,
                tokens,
                m.history,
                cuts.clone(),
                layers,
                layers,
                ctx.layout.qsa,
                ctx.compress,
            )?;
            while stages.len() < cuts.len() {
                stages.push(Stage {
                    fixed: vec![0; ctx.layout.fixed_len()],
                    position: None,
                });
            }
            for stage in &mut stages[..cuts.len()] {
                stage.fixed[ctx.layout.feedback_index()] = m.feedback;
            }
            let exits = tile_exits(w.entry, w.end, &cuts);
            for layer in 0..layers {
                let mut start = w.entry;
                for (tile, &exit) in exits.iter().enumerate() {
                    if inject == Inject::DuringPanel(layer) && tile == exits.len() / 2 {
                        return Err("during_panel".into());
                    }
                    m.advance_layer(ctx, layer, start, exit, w.all);
                    let cut = cuts.iter().position(|&cut| cut == exit);
                    if let Some(k) = cut {
                        if let Some(slot) = ctx.gdn_slot[layer] {
                            stages[k].fixed[ctx.layout.gdn_index(slot).unwrap()] = m.layer[layer];
                            book.record_gdn(k, layer)?;
                        } else if let Some(slot) = ctx.qsa_slot[layer] {
                            book.record_qsa(k, layer, slot, m.qsa[slot])?;
                            stages[k].fixed[ctx.layout.qsa_index(slot).unwrap()] = m.layer[layer];
                        }
                        if ctx.ple_layer == Some(layer) {
                            stages[k].fixed[ctx.layout.ple_index().unwrap()] = m.ple;
                        }
                    }
                    // The last layer's tile: the head appends it (MTP).
                    if layer == layers - 1 && w.head {
                        if inject == Inject::Sink(tile) {
                            return Err("sink".into());
                        }
                        m.head_pos = exit;
                        m.pending = Some(exit - 1);
                        if let Some(k) = cut {
                            assert_eq!(
                                book.next_cut(),
                                Some(exit),
                                "a published cut is the next one"
                            );
                            assert!(
                                book.complete(k),
                                "cut {exit} staged by every layer before its publication"
                            );
                            assert_eq!(m.head_pos, exit, "the head sits exactly at the cut");
                            staged_out.push(pub_from_stage(ctx, &stages[k], &book, k));
                            book.mark_published();
                        }
                    }
                    start = exit;
                }
                if inject == Inject::AfterLayer(layer) {
                    return Err("after_layer".into());
                }
            }
            if inject == Inject::BeforeCommit {
                return Err("before_commit".into());
            }
            for k in 0..cuts.len() {
                if !book.complete(k) {
                    return Err(format!("cut {k} is not fully staged"));
                }
            }
            if w.head && book.published() != cuts.len() {
                return Err("the head left a cut unpublished".into());
            }
            for slot in 0..ctx.layout.qsa {
                if m.qsa[slot] != marks(w.end, ctx.compress) {
                    return Err(format!("QSA slot {slot} cursor is not at the window end"));
                }
            }
            m.history = ple_history_after(m.history, tokens);
            m.position = w.end;
            if !w.head {
                while book.published() < cuts.len() {
                    let k = book.published();
                    staged_out.push(pub_from_stage(ctx, &stages[k], &book, k));
                    book.mark_published();
                }
            }
            if w.boundaries.contains(&w.end) {
                staged_out.push(m.snapshot());
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                out.extend(staged_out);
                Ok(())
            }
            Err(error) => {
                // The transaction restores target and head; `pending` is the
                // drafter's and comes back only through its own backup.
                let drafter_pending = m.pending;
                *m = entry_state;
                m.pending = drafter_pending;
                for stage in stages.iter_mut() {
                    stage.position = None;
                }
                Err(error)
            }
        }
    }

    /// Whole prompt as the bundle drives it: windows end where the cut budget
    /// forces (`prompt_window_end`), one transaction per window.
    fn run_prompt(
        ctx: &Ctx,
        all: &[u32],
        entry: &M,
        boundaries: &[usize],
        head: bool,
    ) -> (M, Vec<Pub>, Vec<(usize, usize)>) {
        let mut m = entry.clone();
        let mut stages = Vec::new();
        let mut out = Vec::new();
        let mut windows = Vec::new();
        let mut at = m.position;
        while at < all.len() {
            let pending: Vec<usize> = boundaries.iter().copied().filter(|&b| b > at).collect();
            let end = prompt_window_end(at, all.len(), &pending, PROMPT_TXN_MAX_CUTS);
            let w = Win {
                all,
                entry: at,
                end,
                boundaries: pending,
                head,
            };
            run_window(ctx, &mut m, &mut stages, &w, Inject::None, &mut out)
                .expect("window commits");
            windows.push((at, end));
            at = end;
        }
        (m, out, windows)
    }

    #[test]
    fn session_pair_boundary_plan_covers_3_4_1536_1537_8192() {
        let prompt = fixture_prompt();
        let boundaries = stride_boundaries(1536, &prompt, Some(END), 0, prompt.len());
        assert_eq!(boundaries, BOUNDARIES);
        assert_eq!(prompt_cuts_in(0, prompt.len(), &boundaries), BOUNDARIES);
        // Resumed entries drop the boundaries at or below them; a boundary
        // that is the window end is not an interior cut.
        assert_eq!(prompt_cuts_in(3, 3072, &boundaries), [4, 1536, 1537]);
        assert_eq!(prompt_cuts_in(1536, 1537, &boundaries), Vec::<usize>::new());
        assert_eq!(
            prompt_cuts_in(1537, prompt.len(), &boundaries),
            [3072, 4608, 6144, 7680, 8192]
        );
        assert_eq!(
            prompt_cuts_in(8192, prompt.len(), &boundaries),
            Vec::<usize>::new()
        );
        // Four staged cuts per window: the cold prompt is two windows, the
        // first one ending on the fifth boundary after its entry.
        let mut at = 0;
        let mut ends = Vec::new();
        while at < prompt.len() {
            let pending: Vec<usize> = boundaries.iter().copied().filter(|&b| b > at).collect();
            let end = prompt_window_end(at, prompt.len(), &pending, PROMPT_TXN_MAX_CUTS);
            assert!(prompt_cuts_in(at, end, &pending).len() <= PROMPT_TXN_MAX_CUTS);
            ends.push(end);
            at = end;
        }
        assert_eq!(ends, [3072, 9000]);
        // A zero-cut budget ends a window on each boundary.
        assert_eq!(prompt_window_end(0, 9000, &boundaries, 0), 3);
        // Without boundaries everything is one window.
        assert_eq!(prompt_window_end(0, 9000, &[], PROMPT_TXN_MAX_CUTS), 9000);
    }

    #[test]
    fn session_pair_cuts_are_the_incumbent_chunk_exits() {
        // The incumbent loop ends a chunk at every session boundary; the
        // layer-major cuts are the same rows, so snapshots key identically.
        let prompt = fixture_prompt();
        let boundaries = stride_boundaries(1536, &prompt, Some(END), 0, prompt.len());
        let mut pos = 0;
        let mut exits = Vec::new();
        while pos < prompt.len() {
            let next = boundaries.iter().copied().find(|&b| b > pos);
            let after = next.and_then(|n| boundaries.iter().copied().find(|&b| b > n));
            pos += qwen4_prefill_segment_rows(
                pos,
                prompt.len(),
                1536,
                SessionBoundaries { next, after },
                false,
            );
            exits.push(pos);
        }
        for boundary in &boundaries {
            assert!(
                exits.contains(boundary),
                "incumbent chunk ends at boundary {boundary}"
            );
        }
        for window in boundaries.chunks(PROMPT_TXN_MAX_CUTS) {
            assert!(window.len() <= PROMPT_TXN_MAX_CUTS);
        }
    }

    #[test]
    fn session_pair_stage_layout_is_the_session_layout_order() {
        let ctx = ctx();
        let layout = ctx.layout;
        let mut seen = vec![false; layout.fixed_len()];
        for slot in 0..layout.gdn {
            // GDN recurrent then conv, per layer.
            for part in [
                layout.gdn_index(slot).unwrap(),
                layout.gdn_index(slot).unwrap() + 1,
            ] {
                assert!(!std::mem::replace(&mut seen[part], true));
            }
        }
        for slot in 0..layout.qsa {
            assert!(!std::mem::replace(
                &mut seen[layout.qsa_index(slot).unwrap()],
                true
            ));
        }
        if let Some(index) = layout.ple_index() {
            assert!(!std::mem::replace(&mut seen[index], true));
        }
        assert!(!std::mem::replace(&mut seen[layout.feedback_index()], true));
        assert!(
            seen.iter().all(|&part| part),
            "every fixed part has exactly one index"
        );
        assert_eq!(layout.gdn_index(layout.gdn), None);
        assert_eq!(layout.qsa_index(layout.qsa), None);
        assert_eq!((ctx.layout.gdn, ctx.layout.qsa), (36, 12));
    }

    #[test]
    fn session_pair_published_cuts_equal_sequential_state_for_cold_and_resumed_entries() {
        let ctx = ctx();
        let all = tokens(9000);
        let cold = M::new(&ctx);
        for &(entry, head) in &[
            (0usize, false),
            (0, true),
            (3, false),
            (4, true),
            (1536, false),
            (1537, true),
            (8192, false),
            (8192, true),
        ] {
            let entry_m = entry_at(&ctx, &all, entry);
            let (end_m, published, windows) = run_prompt(&ctx, &all, &entry_m, &BOUNDARIES, head);
            let expected_positions: Vec<usize> =
                BOUNDARIES.iter().copied().filter(|&b| b > entry).collect();
            let got: Vec<usize> = published.iter().map(|p| p.position).collect();
            assert_eq!(
                got, expected_positions,
                "entry {entry} head {head}: every boundary once, in order"
            );
            for snapshot in &published {
                let want = reference(&ctx, &cold, &all, snapshot.position).snapshot();
                assert_eq!(
                    snapshot, &want,
                    "entry {entry} head {head}: snapshot at {}",
                    snapshot.position
                );
            }
            let want_end = reference(&ctx, &cold, &all, all.len());
            assert_eq!(
                end_m.snapshot(),
                want_end.snapshot(),
                "entry {entry} head {head}: window end state"
            );
            if head {
                assert_eq!(
                    (end_m.head_pos, end_m.pending),
                    (all.len(), Some(all.len() - 1))
                );
            }
            // Every window held at most the admitted number of cuts.
            for &(start, end) in &windows {
                let pending: Vec<usize> =
                    BOUNDARIES.iter().copied().filter(|&b| b > start).collect();
                assert!(prompt_cuts_in(start, end, &pending).len() <= PROMPT_TXN_MAX_CUTS);
            }
            assert_eq!(windows.last().map(|w| w.1), Some(all.len()));
        }
    }

    #[test]
    fn session_pair_injected_aborts_restore_every_owner_and_publish_nothing() {
        let ctx = ctx();
        let all = tokens(5000);
        let last = ctx.gdn_slot.len() - 1;
        for &(entry, head) in &[(0usize, false), (3, true), (1537, false), (1536, true)] {
            let entry_m = entry_at(&ctx, &all, entry);
            let mut injections = vec![
                Inject::BeforeDma,
                Inject::DuringPanel(0),
                Inject::DuringPanel(last),
                Inject::AfterLayer(0),
                Inject::AfterLayer(last),
                Inject::BeforeCommit,
            ];
            if head {
                injections.extend([Inject::Sink(0), Inject::Sink(1)]);
            }
            for inject in injections {
                let pending: Vec<usize> =
                    BOUNDARIES.iter().copied().filter(|&b| b > entry).collect();
                let end = prompt_window_end(entry, all.len(), &pending, PROMPT_TXN_MAX_CUTS);
                let w = Win {
                    all: &all,
                    entry,
                    end,
                    boundaries: pending,
                    head,
                };
                let mut m = entry_m.clone();
                let mut stages = Vec::new();
                let mut out = Vec::new();
                let saved = Saved::take(&m);
                let error = run_window(&ctx, &mut m, &mut stages, &w, inject, &mut out)
                    .expect_err("the injected failure surfaces");
                assert!(!error.is_empty());
                saved.restore_into(&mut m);
                assert_eq!(
                    m, entry_m,
                    "entry {entry} head {head} {inject:?}: every owner restored"
                );
                assert!(
                    out.is_empty(),
                    "{inject:?}: nothing is published by an aborted window"
                );
                assert!(stages.iter().all(|stage| stage.position.is_none()));
                // The retry is a clean window: no contamination survived.
                run_window(&ctx, &mut m, &mut stages, &w, Inject::None, &mut out)
                    .expect("retry commits");
                let cold = M::new(&ctx);
                assert_eq!(m.snapshot(), reference(&ctx, &cold, &all, end).snapshot());
            }
        }
    }

    #[test]
    fn session_pair_window_two_failure_restores_the_window_two_entry_pending() {
        // A request longer than the cut budget is two windows. The backup is
        // refreshed before each window's begin, so a failure in window 2
        // restores target, head AND `pending` to window 2's entry (the hidden
        // of row 3071), not to the request's entry.
        let ctx = ctx();
        let all = tokens(9000);
        let mut m = M::new(&ctx);
        let mut stages = Vec::new();
        let mut out = Vec::new();
        let pending1: Vec<usize> = BOUNDARIES.iter().copied().filter(|&b| b > 0).collect();
        let end1 = prompt_window_end(0, all.len(), &pending1, PROMPT_TXN_MAX_CUTS);
        assert_eq!(end1, 3072);
        let saved1 = Saved::take(&m);
        assert!(pending_backup_applies(saved1.saved, saved1.for_position, 0));
        let w1 = Win {
            all: &all,
            entry: 0,
            end: end1,
            boundaries: pending1,
            head: true,
        };
        run_window(&ctx, &mut m, &mut stages, &w1, Inject::None, &mut out)
            .expect("window 1 commits");
        assert_eq!(
            (m.position, m.head_pos, m.pending),
            (3072, 3072, Some(3071))
        );

        let pending2: Vec<usize> = BOUNDARIES.iter().copied().filter(|&b| b > 3072).collect();
        let end2 = prompt_window_end(3072, all.len(), &pending2, PROMPT_TXN_MAX_CUTS);
        assert_eq!(end2, all.len());
        let w2 = Win {
            all: &all,
            entry: 3072,
            end: end2,
            boundaries: pending2,
            head: true,
        };
        // Window 2 saves its own entry `pending` before it begins.
        let saved2 = Saved::take(&m);
        assert!(pending_backup_applies(
            saved2.saved,
            saved2.for_position,
            3072
        ));
        // The request-start backup does not apply to window 2: a begin
        // without a fresh save is refused before any mutation.
        assert!(!pending_backup_applies(
            saved1.saved,
            saved1.for_position,
            3072
        ));
        let entry2 = m.clone();
        let published_before = out.len();
        for inject in [
            Inject::AfterLayer(0),
            Inject::DuringPanel(ctx.gdn_slot.len() - 1),
            Inject::Sink(1),
            Inject::BeforeCommit,
        ] {
            let mut failed = entry2.clone();
            run_window(&ctx, &mut failed, &mut stages, &w2, inject, &mut out)
                .expect_err("window 2 fails");
            saved2.restore_into(&mut failed);
            assert_eq!(
                failed, entry2,
                "{inject:?}: target, head and pending are window 2's entry"
            );
            assert_eq!(failed.pending, Some(3071));
            assert_eq!(
                out.len(),
                published_before,
                "{inject:?}: the failed window published nothing"
            );
        }
        // With only the request-start backup the head's `pending` would stay
        // at what window 2's first tile left in it: the inconsistent state
        // the per-window refresh and the begin check exist to prevent.
        let mut stale = entry2.clone();
        run_window(
            &ctx,
            &mut stale,
            &mut stages,
            &w2,
            Inject::Sink(1),
            &mut out,
        )
        .expect_err("window 2 fails");
        saved1.restore_into(&mut stale);
        assert_ne!(stale, entry2);
        // The retry from window 2's entry is a clean window.
        run_window(&ctx, &mut m, &mut stages, &w2, Inject::None, &mut out)
            .expect("window 2 commits");
        let cold = M::new(&ctx);
        assert_eq!(
            m.snapshot(),
            reference(&ctx, &cold, &all, all.len()).snapshot()
        );
        assert_eq!((m.head_pos, m.pending), (all.len(), Some(all.len() - 1)));
    }

    #[test]
    fn session_pair_s1_checkpoint_cadence_shares_the_incumbent_boundaries_and_a_distinct_identity()
    {
        let prompt = vec![1u32; 16384];
        // A slow-link load (mixer tile 1536, checkpoint cadence 8192) keeps
        // the incumbent chunk-8192 boundary list: one interior cut at 16K,
        // the end boundary published at the window's commit.
        let cadence = session_cadence(Some(8192), Some(1536)).unwrap();
        assert_eq!(cadence, 8192);
        let s1 = stride_boundaries(cadence, &prompt, None, 0, prompt.len());
        let incumbent = stride_boundaries(
            session_cadence(None, Some(8192)).unwrap(),
            &prompt,
            None,
            0,
            prompt.len(),
        );
        assert_eq!(s1, [8192, 16384]);
        assert_eq!(s1, incumbent);
        assert_eq!(prompt_cuts_in(0, prompt.len(), &s1), [8192]);
        assert_eq!(
            prompt_window_end(0, prompt.len(), &s1, PROMPT_TXN_MAX_CUTS),
            prompt.len(),
            "one window"
        );
        // The incumbent loop (also a bypassed request on this load) stops at
        // the same session boundaries.
        let mut at = 0;
        let mut exits = Vec::new();
        while at < prompt.len() {
            at = s1.iter().copied().find(|&b| b > at).unwrap_or(prompt.len());
            exits.push(at);
        }
        assert_eq!(exits, s1);
        // The explicit chunk-1536 cadence would force ten interior cuts and
        // several windows at 16K: why the cadence is the profile's own.
        let explicit = stride_boundaries(
            session_cadence(None, Some(1536)).unwrap(),
            &prompt,
            None,
            0,
            prompt.len(),
        );
        assert_eq!(prompt_cuts_in(0, prompt.len(), &explicit).len(), 10);
        assert!(prompt_window_end(0, prompt.len(), &explicit, PROMPT_TXN_MAX_CUTS) < prompt.len());
        // Message-end cuts still union with the cadence (unchanged scheduling).
        let mut with_ends = prompt.clone();
        with_ends[2] = END;
        assert_eq!(
            stride_boundaries(8192, &with_ends, Some(END), 0, with_ends.len()),
            [3, 8192, 16384]
        );
        // No profile: the forward chunk, as before; a zero stride is not a cadence.
        assert_eq!(session_cadence(None, Some(1536)), Some(1536));
        assert_eq!(session_cadence(Some(0), Some(1536)), Some(1536));
        assert_eq!(session_cadence(None, None), None);
        // Identity: tile 1536 + cadence 8192 is neither the explicit
        // chunk-1536 scope nor the incumbent chunk-8192 scope, and another
        // cadence is a third scope.
        let base = |chunk: usize| format!("qwen4/Ar/chunk{chunk}/F");
        let s1_scope = format!("{}{}", base(1536), slowlink_scope_tag(Some((1536, 8192))));
        let explicit_scope = format!("{}{}", base(1536), slowlink_scope_tag(None));
        let incumbent_scope = format!("{}{}", base(8192), slowlink_scope_tag(None));
        let other_cadence = format!("{}{}", base(1536), slowlink_scope_tag(Some((1536, 4096))));
        let scopes = [&s1_scope, &explicit_scope, &incumbent_scope, &other_cadence];
        for (i, a) in scopes.iter().enumerate() {
            for b in &scopes[i + 1..] {
                assert_ne!(a, b);
            }
        }
        assert_eq!(slowlink_scope_tag(None), "");
    }

    #[test]
    fn session_pair_sp_identity_gives_sp_on_and_sp_off_distinct_scopes() {
        let base = |chunk: usize| format!("qwen4/Ar/chunk{chunk}/F");
        let s1 = format!("{}{}", base(1536), slowlink_scope_tag(Some((1536, 8192))));
        // `snapshot_scope` appends `/` + the host identity when it is non-empty.
        let scope = |host: String| {
            if host.is_empty() {
                s1.clone()
            } else {
                format!("{s1}/{host}")
            }
        };
        let d2 = "cpu-slab-split-v1/3".to_string();
        let off = scope(with_sp_identity(String::new(), false));
        let on = scope(with_sp_identity(String::new(), true));
        let d2_off = scope(with_sp_identity(d2.clone(), false));
        let d2_on = scope(with_sp_identity(d2.clone(), true));
        // SP off: byte for byte the scope it was.
        assert_eq!(off, s1);
        assert_eq!(d2_off, format!("{s1}/{d2}"));
        assert_eq!(with_sp_identity(d2.clone(), false), d2);
        assert_eq!(with_sp_identity(String::new(), false), "");
        // SP on: a distinct scope with and without D2, the token last.
        assert_eq!(on, format!("{s1}/sp-e1v1"));
        assert_eq!(d2_on, format!("{s1}/{d2}+sp-e1v1"));
        let scopes = [&off, &on, &d2_off, &d2_on];
        for (i, a) in scopes.iter().enumerate() {
            for b in &scopes[i + 1..] {
                assert_ne!(a, b);
            }
        }
        assert_eq!(SP_HOST_IDENTITY, "sp-e1v1");
    }

    #[test]
    fn session_pair_host_snapshot_policy_uses_only_admitted_slack() {
        for host_budget in [0, 1, 2 << 30, u64::MAX] {
            // Every incumbent/fast/UMA load lacks an S1 profile: no new policy.
            assert!(pressure_host_snapshot_policy(None, || {
                panic!("an incumbent load must not inspect the new cache policy")
            })
            .is_none());
            for device_budget in [0, 1, 156_567_700, u64::MAX] {
                let policy =
                    pressure_host_snapshot_policy(Some(device_budget), || host_budget).unwrap();
                assert_eq!(policy.device_budget_bytes, device_budget);
                assert_eq!(policy.host_budget_bytes, host_budget);
            }
        }
        // A zero device residue still permits direct bounded host capture;
        // no 512 MiB safety credit is added to the cache's device budget.
        let policy = pressure_host_snapshot_policy(Some(0), || 2 << 30).unwrap();
        assert_eq!(policy.device_budget_bytes, 0);
        assert_eq!(policy.host_budget_bytes, 2 << 30);
    }

    #[test]
    fn session_pair_attach_chunk_is_the_exact_profile_tile_or_the_unchanged_incumbent() {
        use std::cell::Cell;
        let widened = |calls: &Cell<u32>| {
            calls.set(calls.get() + 1);
            8192
        };
        // No profile: the incumbent request, capped by the PLE store, exactly as before.
        let calls = Cell::new(0);
        assert_eq!(
            attach_chunk_request(None, 16640, usize::MAX, || widened(&calls)),
            Ok(8192)
        );
        assert_eq!(
            attach_chunk_request(None, 16640, 4096, || widened(&calls)),
            Ok(4096)
        );
        assert_eq!(calls.get(), 2);
        // A profile: the charged tile for the product carrier (1536) and for a
        // caller passing the whole context (16640); the incumbent helper is
        // never consulted, so 8192 / 4096 scratch cannot be retained.
        let calls = Cell::new(0);
        assert_eq!(
            attach_chunk_request(Some(1536), 1536, usize::MAX, || widened(&calls)),
            Ok(1536)
        );
        assert_eq!(
            attach_chunk_request(Some(1536), 16640, usize::MAX, || widened(&calls)),
            Ok(1536)
        );
        assert_eq!(
            attach_chunk_request(Some(1024), 16640, 2048, || widened(&calls)),
            Ok(1024)
        );
        assert_eq!(calls.get(), 0);
        // Capacity shortfalls and a tile beyond the S1 bound refuse, never resize.
        assert!(attach_chunk_request(Some(1536), 1024, usize::MAX, || 0).is_err());
        assert!(attach_chunk_request(Some(1536), 16640, 512, || 0).is_err());
        assert!(
            attach_chunk_request(Some(PROMPT_TILE_ROWS_MAX + 1), 16640, usize::MAX, || 0).is_err()
        );
        assert!(attach_chunk_request(Some(0), 16640, usize::MAX, || 0).is_err());
    }

    #[test]
    fn session_pair_qsa_rows_below_a_cut_are_never_repooled_or_rewritten() {
        // A stage reads the K/V, raw and pooled rows below the cut's marks in
        // place when it is published, long after later tiles ran. That is
        // exact only if no later launch rewrites them: the QSA pool kernel
        // re-pools just the blocks a launch completes, starting at the block
        // holding its first position (`start / compress`), and rows are
        // written only at positions >= start. A launch starts at or after
        // the cut, so everything below the cut's marks stays final: for a cut
        // inside a compression block (3, 1537) the partial block is not in
        // the stage at all (`pooled_len = cut / compress`), for a block
        // boundary (4, 1536, 8192) the re-pooled block is the first one past
        // the stage.
        let compress = ctx().compress;
        for cut in [3usize, 4, 1536, 1537, 8192] {
            let pooled_len = cut / compress;
            for start in [
                cut,
                cut + 1,
                cut + compress - 1,
                cut + TILE,
                cut + 3 * TILE + 1,
            ] {
                assert!(
                    start / compress >= pooled_len,
                    "cut {cut}: launch at {start} re-pools only blocks >= {pooled_len}"
                );
                assert!(start >= cut, "rows are written at positions >= {start}");
            }
            // The marks the stage records for the cut.
            let mut book = CutBook::new(
                0,
                &tokens(cut + 1),
                PleHistory::new(2),
                vec![cut],
                2,
                1,
                1,
                compress,
            )
            .unwrap();
            assert!(book
                .record_qsa(0, 0, 0, (cut, cut, pooled_len, 0, cut))
                .is_ok());
        }
        // A cursor at the wrong pooled length (a partially pooled block) is
        // refused, never staged.
        let mut book = CutBook::new(
            0,
            &tokens(2000),
            PleHistory::new(2),
            vec![1537],
            2,
            1,
            1,
            compress,
        )
        .unwrap();
        assert!(book
            .record_qsa(0, 0, 0, (1537, 1537, 385, 0, 1537))
            .is_err());
        assert!(book
            .record_qsa(0, 0, 0, (1536, 1536, 384, 0, 1536))
            .is_err());
        assert!(book.record_qsa(0, 0, 0, (1537, 1537, 384, 0, 1537)).is_ok());
    }

    #[test]
    fn session_pair_cut_book_refuses_misuse() {
        let compress = ctx().compress;
        let history = PleHistory::new(2);
        // Too many cuts, a cut at the window end, a cut at the entry.
        assert!(CutBook::new(
            0,
            &tokens(100),
            history,
            vec![1, 2, 3, 4, 5],
            2,
            2,
            1,
            compress
        )
        .is_err());
        assert!(CutBook::new(0, &tokens(100), history, vec![100], 2, 2, 1, compress).is_err());
        assert!(CutBook::new(10, &tokens(100), history, vec![10], 2, 2, 1, compress).is_err());
        assert!(CutBook::new(0, &tokens(100), history, vec![5, 5], 2, 2, 1, compress).is_err());
        let mut book =
            CutBook::new(0, &tokens(100), history, vec![8, 20], 2, 2, 1, compress).unwrap();
        assert!(!book.complete(0));
        book.record_gdn(0, 0).unwrap();
        assert!(book.record_gdn(0, 0).is_err(), "a layer stages a cut once");
        assert!(!book.complete(0));
        assert!(
            book.commits_for(0).is_err(),
            "marks are missing until the QSA layer stages"
        );
        book.record_qsa(0, 1, 0, (8, 8, 2, 3, 8)).unwrap();
        assert!(book.complete(0) && !book.complete(1));
        assert_eq!(book.next_cut(), Some(8));
        book.mark_published();
        assert_eq!(book.next_cut(), Some(20));
        // PLE history at a cut is the entry history advanced by its tokens.
        let all = tokens(100);
        assert_eq!(book.history_at(1), ple_history_after(history, &all[..20]));
    }

    #[test]
    fn session_pair_mtp_tiles_pair_each_token_with_the_previous_hidden() {
        // The sink appends tile by tile: token p pairs with the target hidden
        // of p - 1 (the carried `pending` for a tile's first row; prompt
        // position 0 pairs with its own), and `pending` ends each tile on its
        // last row, so a cut published after the tile holds hidden `cut - 1`.
        for &entry in &[0usize, 3, 4, 1536, 1537, 8192] {
            let end = 9000usize;
            let pending_cuts: Vec<usize> =
                BOUNDARIES.iter().copied().filter(|&b| b > entry).collect();
            let mut at = entry;
            let mut pending = entry.checked_sub(1);
            let mut pairs = Vec::new();
            let mut held_at_cut = Vec::new();
            while at < end {
                let window_end = prompt_window_end(
                    at,
                    end,
                    &pending_cuts
                        .iter()
                        .copied()
                        .filter(|&b| b > at)
                        .collect::<Vec<_>>(),
                    PROMPT_TXN_MAX_CUTS,
                );
                let cuts = prompt_cuts_in(at, window_end, &pending_cuts);
                let mut start = at;
                for exit in tile_exits(at, window_end, &cuts) {
                    for (index, position) in (start..exit).enumerate() {
                        let hidden = if position == 0 {
                            0
                        } else if index == 0 {
                            pending.expect("a resumed tile carries pending")
                        } else {
                            position - 1
                        };
                        pairs.push((position, hidden));
                    }
                    pending = Some(exit - 1);
                    if cuts.contains(&exit) {
                        held_at_cut.push((exit, pending.unwrap()));
                    }
                    start = exit;
                }
                at = window_end;
            }
            let sequential: Vec<(usize, usize)> =
                (entry..end).map(|p| (p, p.saturating_sub(1))).collect();
            assert_eq!(
                pairs, sequential,
                "entry {entry}: bounded tiles reproduce the sequential p-1 pairing"
            );
            for (cut, held) in held_at_cut {
                assert_eq!(
                    held,
                    cut - 1,
                    "the cut's boundary hidden is the target hidden of cut - 1"
                );
            }
        }
    }

    #[test]
    fn session_pair_incumbent_receipt_has_every_required_key() {
        let receipt = incumbent_receipt_json(
            "auto",
            Some("selector declined"),
            3,
            &[(0, 1533), (1533, 4)],
            1536,
        );
        for key in SLOWLINK_RECEIPT_KEYS {
            assert!(receipt.get(key).is_some(), "receipt key {key}");
        }
        assert_eq!(receipt["route"], "incumbent");
        assert_eq!(receipt["prompt_rows"], 1537);
        assert_eq!(receipt["launch_cuts"], json!([1533, 1537]));
        assert_eq!(receipt["reserve"]["total"], 0);
        assert_eq!(receipt["bypass_reason"], "selector declined");
        assert!(incumbent_receipt_json("env0", None, 0, &[], 0)["bypass_reason"].is_null());
    }

    #[test]
    fn session_pair_bypass_reasons_borrow_static_labels_and_move_dynamic_ones() {
        use std::borrow::Cow;
        let mut owner = SlowlinkOwner::default();
        // A static label is stored borrowed: no allocation on the default-off path.
        owner.bypass = Some("no admitted slow-link profile".into());
        assert!(matches!(owner.bypass, Some(Cow::Borrowed(_))));
        // A dynamic diagnostic is moved in, never copied.
        let dynamic = format!("request end {} exceeds the context capacity {}", 9, 8);
        let address = dynamic.as_ptr();
        owner.bypass = Some(dynamic.into());
        let Some(Cow::Owned(moved)) = owner.bypass.take() else {
            panic!("a dynamic reason is stored owned");
        };
        assert_eq!(moved.as_ptr(), address, "moved, not copied");
        assert!(owner.bypass.is_none(), "the reason is taken once");
        // The stored incumbent receipt carries the same borrowed label.
        let stored = StoredReceipt::Incumbent {
            reason: Some(Cow::Borrowed("selector declined")),
            entry: 0,
            windows: Vec::new(),
            tile_rows: 1536,
        };
        let StoredReceipt::Incumbent { reason, .. } = stored else {
            unreachable!("built as an incumbent receipt");
        };
        assert!(matches!(reason, Some(Cow::Borrowed(_))));
    }

    #[test]
    fn session_pair_failure_points_parse() {
        for (name, point) in [
            ("before_dma", SlowlinkFailPoint::BeforeDma),
            ("during_panel", SlowlinkFailPoint::DuringPanel),
            ("after_layer", SlowlinkFailPoint::AfterLayer),
            ("before_commit", SlowlinkFailPoint::BeforeCommit),
        ] {
            assert_eq!(SlowlinkFailPoint::parse(name), Some(point));
        }
        assert_eq!(SlowlinkFailPoint::parse("during-panel"), None);
    }

    fn passing(name: &str) -> Value {
        json!({"name": name, "status": "pass", "mismatches": 0u64, "detail": {"rows": 1}})
    }

    fn collect(fail_at: &[usize], malformed_at: &[usize]) -> (Vec<String>, Vec<Value>) {
        let mut invoked = Vec::new();
        let report = collect_primitive_reports(&SLOWLINK_PRIMITIVES, |name| {
            let index = invoked.len();
            invoked.push(name.to_string());
            if fail_at.contains(&index) {
                Err(format!("setup of {name} failed"))
            } else if malformed_at.contains(&index) {
                Ok(json!({"name": name, "status": "pass"}))
            } else {
                Ok(passing(name))
            }
        });
        let cases = report["cases"].as_array().expect("cases array").clone();
        (invoked, cases)
    }

    fn assert_ordered_names(cases: &[Value]) {
        assert_eq!(cases.len(), SLOWLINK_PRIMITIVES.len());
        for (case, name) in cases.iter().zip(SLOWLINK_PRIMITIVES) {
            assert_eq!(case["name"], name);
        }
    }

    fn assert_error_case(case: &Value, name: &str) {
        assert_eq!(case["status"], "error");
        assert!(case["mismatches"].is_null(), "no fabricated count: {case}");
        let error = case["detail"]["error"].as_str().expect("error string");
        assert!(
            error.contains(&format!("{name:?}")),
            "error names its case: {error}"
        );
        for other in SLOWLINK_PRIMITIVES.iter().filter(|other| **other != name) {
            assert!(
                !error.contains(other),
                "error carries {other}'s context: {error}"
            );
        }
    }

    #[test]
    fn primitive_collector_runs_all_seven_in_order_when_all_pass() {
        let (invoked, cases) = collect(&[], &[]);
        assert_eq!(invoked, SLOWLINK_PRIMITIVES);
        assert_ordered_names(&cases);
        for case in &cases {
            assert_eq!(case["status"], "pass");
            assert_eq!(case["mismatches"], 0u64);
        }
    }

    #[test]
    fn primitive_collector_first_and_middle_errors_do_not_stop_or_infect_others() {
        for failing in [0usize, 3, 6] {
            let (invoked, cases) = collect(&[failing], &[]);
            assert_eq!(
                invoked, SLOWLINK_PRIMITIVES,
                "all 7 invoked, failing {failing}"
            );
            assert_ordered_names(&cases);
            for (index, case) in cases.iter().enumerate() {
                if index == failing {
                    assert_error_case(case, SLOWLINK_PRIMITIVES[index]);
                } else {
                    assert_eq!(*case, passing(SLOWLINK_PRIMITIVES[index]), "retained pass");
                }
            }
            let errors = cases
                .iter()
                .filter(|case| case["status"] == "error")
                .count();
            assert_eq!(errors, 1, "one error, not seven");
        }
    }

    #[test]
    fn primitive_collector_malformed_report_is_that_cases_error_only() {
        let (invoked, cases) = collect(&[1], &[4]);
        assert_eq!(invoked, SLOWLINK_PRIMITIVES);
        assert_ordered_names(&cases);
        assert_error_case(&cases[1], SLOWLINK_PRIMITIVES[1]);
        assert_error_case(&cases[4], SLOWLINK_PRIMITIVES[4]);
        assert!(cases[4]["detail"]["error"]
            .as_str()
            .unwrap()
            .contains("does not carry"));
        for index in [0usize, 2, 3, 5, 6] {
            assert_eq!(cases[index], passing(SLOWLINK_PRIMITIVES[index]));
        }
    }

    #[test]
    fn primitive_collector_rejects_misnamed_and_non_integer_reports() {
        let report = collect_primitive_reports(&["zero_row_panel", "k1_tail_cohort"], |name| {
            Ok(match name {
                "zero_row_panel" => passing("k1_tail_cohort"),
                _ => json!({"name": name, "status": "pass", "mismatches": -1, "detail": {}}),
            })
        });
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0]["name"], "zero_row_panel");
        assert_eq!(cases[0]["status"], "error");
        assert!(cases[0]["mismatches"].is_null());
        assert_eq!(cases[1]["name"], "k1_tail_cohort");
        assert_eq!(cases[1]["status"], "error");
        assert!(cases[1]["mismatches"].is_null());
    }

    #[test]
    fn primitive_collector_keeps_measured_fail_with_its_count() {
        let report = collect_primitive_reports(&["delayed_reader_reuse"], |name| {
            Ok(json!({"name": name, "status": "fail", "mismatches": 3u64, "detail": {}}))
        });
        assert_eq!(report["cases"][0]["status"], "fail");
        assert_eq!(report["cases"][0]["mismatches"], 3u64);
    }
}
