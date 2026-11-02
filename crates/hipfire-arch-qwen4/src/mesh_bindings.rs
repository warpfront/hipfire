// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq

//! Borrowed Qwen4 band bindings for the generic, Gpus-owned executor.

use crate::bundle::Qwen4Bundle;
use crate::gpu_forward::{Qwen4GpuForward, Qwen4MeshHead, Qwen4MeshLayout, Qwen4OutputRows};
use crate::mtp_gpu::MtpGpuStateSnapshot;
use crate::program::QWEN4_STREAMS_RESOURCE;
use crate::state::Qwen4StateSnapshot;
use crate::weights::Qwen4BandScope;
use crate::Qwen4Config;
use hipfire_dispatch::pipeline::step_graph::OpId;
use hipfire_runtime::mesh_split::{CollectiveKind, PlannedCollective};
use hipfire_runtime::multi_gpu::executor::{
    BoundaryIo, ForwardEpoch, MeshCheckpointId, MeshCheckpointKey, MeshRoute, MeshStepBindings,
};
use hipfire_runtime::multi_gpu::transport::EdgeDescriptor;
use hipfire_runtime::session_cache::{SessionPieceKey, SessionRoute};
use rdna_compute::{Gpu, GpuTensor};

type Result<T, E = String> = std::result::Result<T, E>;

/// The ticket of owner `owner` (a trunk band, or the output head at
/// `owner == bands`) for the aggregate window `key`: the window generation is
/// its family-opaque local value. Never a local `Qwen4StateSnapshot`.
pub(crate) fn window_ticket(key: &MeshCheckpointKey, owner: usize) -> MeshCheckpointId {
    MeshCheckpointId::issue(*key, owner, key.window_gen)
}

/// Pure check that `ticket` is `owner`'s ticket of the aggregate window `key`.
pub(crate) fn check_window_ticket(
    ticket: &MeshCheckpointId,
    owner: usize,
    key: &MeshCheckpointKey,
) -> Result<()> {
    if ticket.owner() != owner || ticket.key() != *key || ticket.local() != key.window_gen {
        return Err(format!(
            "Qwen4 window ticket for owner {owner} is not this window's (stale, foreign or misowned)"
        ));
    }
    Ok(())
}

/// How a window restore treats the local arena entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowRestore {
    /// Roll back and consume the entry (the local `restore`).
    Consume,
    /// Roll back and keep the entry active (the local `restore_retain`).
    Retain,
}

/// How a validated window finalizes its local arena entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowFinalize {
    /// Consume the entries (the local `commit_validated`).
    Commit,
    /// Keep them active and rollback-able (the local `retain_validated`).
    Retain,
}

/// The per-band local tickets and modes an adapter's window callbacks work
/// against. The adapter is rebuilt per call, so the owner of the tickets
/// (the mesh holder) lends them in and takes them back; local tickets never
/// leave the holder otherwise.
#[derive(Clone, Debug)]
pub(crate) struct WindowCtx {
    pub(crate) restore: WindowRestore,
    pub(crate) finalize: WindowFinalize,
    pub(crate) target: Vec<Option<Qwen4StateSnapshot>>,
    pub(crate) head: Option<MtpGpuStateSnapshot>,
}

impl WindowCtx {
    /// An empty context a checkpoint fills, one slot per band.
    pub(crate) fn checkpoint(bands: usize) -> Self {
        Self {
            restore: WindowRestore::Consume,
            finalize: WindowFinalize::Commit,
            target: vec![None; bands],
            head: None,
        }
    }

    /// A live window's tickets with the transition modes of the coming call.
    pub(crate) fn live(
        target: &[Qwen4StateSnapshot],
        head: Option<MtpGpuStateSnapshot>,
        restore: WindowRestore,
        finalize: WindowFinalize,
    ) -> Self {
        Self {
            restore,
            finalize,
            target: target.iter().copied().map(Some).collect(),
            head,
        }
    }

    /// The captured target tickets, one per band; `None` unless every band
    /// captured.
    pub(crate) fn into_target(
        self,
    ) -> Option<(Vec<Qwen4StateSnapshot>, Option<MtpGpuStateSnapshot>)> {
        let head = self.head;
        self.target
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(|target| (target, head))
    }
}

/// The checkpoint window of a session piece at prefix length `prefix`: every
/// owner's piece of one boundary shares it, and it names the current mesh,
/// load and request (the epoch the capture or restore runs under).
pub(crate) fn session_ticket_key(epoch: &ForwardEpoch, prefix: usize) -> MeshCheckpointKey {
    MeshCheckpointKey {
        mesh_epoch: epoch.mesh_epoch,
        load_gen: epoch.load_id,
        reset_gen: 0,
        request_gen: epoch.request_id,
        window_gen: prefix as u64,
    }
}

/// A freshly minted ticket for `owner`'s session piece of `prefix` tokens.
pub(crate) fn session_ticket(
    epoch: &ForwardEpoch,
    owner: usize,
    prefix: usize,
) -> MeshCheckpointId {
    MeshCheckpointId::issue(session_ticket_key(epoch, prefix), owner, prefix as u64)
}

/// One fresh ticket per owner for the aggregate record's pieces, bound to the
/// current request (never stored across requests).
pub(crate) fn session_tickets(
    epoch: &ForwardEpoch,
    pieces: &[SessionPieceKey],
) -> Vec<MeshCheckpointId> {
    pieces
        .iter()
        .enumerate()
        .map(|(owner, piece)| session_ticket(epoch, owner, piece.prefix_len()))
        .collect()
}

/// Pure check that `ticket` is `owner`'s piece of `prefix` for `epoch`.
fn check_session_ticket(
    ticket: &MeshCheckpointId,
    owner: usize,
    epoch: &ForwardEpoch,
    prefix: usize,
) -> Result<()> {
    if ticket.owner() != owner
        || ticket.key() != session_ticket_key(epoch, prefix)
        || ticket.local() != prefix as u64
    {
        return Err(format!(
            "Qwen4 session piece for owner {owner} is not the current request's piece of {prefix} tokens"
        ));
    }
    Ok(())
}

/// Per-request session context of one adapter: the prompt every band
/// captures/restores against, the aggregate record's pieces to restore, and
/// the pool keys the captures name.
struct SessionCtx<'a> {
    prompt: &'a [u32],
    route: SessionRoute,
    restore: Vec<SessionPieceKey>,
    restore_started: bool,
    captured: Vec<Option<SessionPieceKey>>,
}

/// One rank's existing owners, borrowed for submission; never owns a GPU.
pub struct Qwen4RankBand<'a> {
    pub bundle: &'a mut Qwen4Bundle,
    pub forward: &'a mut Qwen4GpuForward,
}

/// An epoch adapter. The output band may be the published root bundle; peers
/// remain in that bundle's mesh owner. Nothing is replicated or moved here.
pub struct Qwen4MeshBindings<'a> {
    ranks: Vec<Qwen4RankBand<'a>>,
    tokens: &'a [u32],
    edge_epochs: Vec<u64>,
    head: Option<Qwen4MeshHead<'a>>,
    wide: Option<&'a GpuTensor>,
    prepared: Option<ForwardEpoch>,
    ran: [bool; 8],
    layouts: [Option<Qwen4MeshLayout>; 8],
    session: Option<SessionCtx<'a>>,
    window: Option<WindowCtx>,
}

fn streams_bytes(rows: usize, branches: usize, hidden: usize, dtype_bytes: usize) -> Result<usize> {
    if rows == 0 || branches == 0 || hidden == 0 || !matches!(dtype_bytes, 2 | 4) {
        return Err("invalid Qwen4 streams geometry or dtype".into());
    }
    rows.checked_mul(branches)
        .and_then(|n| n.checked_mul(hidden))
        .and_then(|n| n.checked_mul(dtype_bytes))
        .ok_or_else(|| "Qwen4 streams payload overflows".into())
}

fn validate_scopes<'s>(
    config: &Qwen4Config,
    scopes: impl IntoIterator<Item = &'s Qwen4BandScope>,
) -> Result<()> {
    let mut end = 0;
    let mut count = 0;
    for scope in scopes {
        count += 1;
        scope.validate(config)?;
        if scope.band.start != end {
            return Err("Qwen4 rank bands must tile the trunk once in layer order".into());
        }
        end = scope.band.end;
    }
    if !(2..=8).contains(&count) {
        return Err("Qwen4 topology.pp requires 2..=8 bands".into());
    }
    if end != config.num_hidden_layers {
        return Err("Qwen4 rank bands do not cover the trunk".into());
    }
    Ok(())
}

impl<'a> Qwen4MeshBindings<'a> {
    /// Validate ownership before any submission or state mutation.
    pub fn new(ranks: Vec<Qwen4RankBand<'a>>) -> Result<Self> {
        let config = &ranks
            .first()
            .ok_or("Qwen4 mesh has no bands")?
            .bundle
            .config;
        validate_scopes(
            config,
            ranks.iter().map(|rank| rank.bundle.weights.band_scope()),
        )?;
        for rank in &ranks {
            let scope = rank.bundle.weights.band_scope();
            if &rank.bundle.config != config {
                return Err("Qwen4 mesh bands have different model configurations".into());
            }
            let map = rank.bundle.state.scope_map();
            if map.band() != scope.band
                || rank.forward.band_scope() != scope
                || map.owns_input() != scope.input
                || map.owns_output() != scope.output
                || map.gdn_count() != rank.bundle.state.gdn.len()
                || map.qsa_count() != rank.bundle.state.qsa.len()
                || rank.forward.scratch.streams.dtype != ranks[0].forward.scratch.streams.dtype
            {
                return Err("Qwen4 mesh weight and state ownership disagree".into());
            }
            // Only the output band may hold the MTP head; no band takes EP. A
            // band may hold a band-scoped lane store: the mesh forward still
            // refuses a lane scope, the lane executor drives those stores.
            if rank.bundle.ep.is_some() || (rank.bundle.mtp.is_some() && !scope.output) {
                return Err(
                    "Qwen4 topology.pp does not admit EP or an MTP head off the output band".into(),
                );
            }
        }
        Ok(Self {
            ranks,
            tokens: &[],
            edge_epochs: Vec::new(),
            head: None,
            wide: None,
            prepared: None,
            ran: [false; 8],
            layouts: [None; 8],
            session: None,
            window: None,
        })
    }

    /// Host token storage remains immutable until submission is drained.
    pub fn set_tokens(&mut self, tokens: &'a [u32]) {
        self.tokens = tokens;
        self.prepared = None;
    }

    /// Transport epochs in directed logical band order (0->1, 1->2, ...).
    pub fn set_edge_epochs(&mut self, epochs: Vec<u64>) {
        self.edge_epochs = epochs;
        self.prepared = None;
    }

    /// Prepare the route-specific precision and exact live boundary extent.
    pub fn set_rank_layout(&mut self, owner: usize, layout: Qwen4MeshLayout) -> Result<()> {
        if owner >= self.ranks.len() {
            return Err("unknown Qwen4 mesh rank".into());
        }
        self.layouts[owner] = Some(layout);
        self.prepared = None;
        Ok(())
    }

    /// `None` commits trunk state without computing a head row.
    pub fn set_head(&mut self, head: Option<Qwen4MeshHead<'a>>) {
        self.head = head;
        self.prepared = None;
    }

    /// The final pre-collapse wide hidden of this epoch's rows (F32,
    /// `hc_count * hidden` per row, exactly this epoch's sub-view); only the
    /// output band writes it. `None` captures nothing.
    pub fn set_wide_capture(&mut self, wide: Option<&'a GpuTensor>) {
        self.wide = wide;
        self.prepared = None;
    }

    /// Lend the window's local tickets and transition modes to this adapter's
    /// checkpoint, restore, repair and finalize callbacks.
    pub(crate) fn set_window(&mut self, window: WindowCtx) {
        self.window = Some(window);
    }

    /// Take the window context back (after a checkpoint, the captured tickets).
    pub(crate) fn take_window(&mut self) -> Option<WindowCtx> {
        self.window.take()
    }

    pub fn ranks(&self) -> &[Qwen4RankBand<'a>] {
        &self.ranks
    }

    pub fn ranks_mut(&mut self) -> &mut [Qwen4RankBand<'a>] {
        self.prepared = None;
        &mut self.ranks
    }

    /// Bind the prompt (absolute positions) and route every session callback of
    /// this adapter works against. Resets any earlier capture/restore state.
    pub fn set_session(&mut self, prompt: &'a [u32], route: SessionRoute) {
        self.session = Some(SessionCtx {
            prompt,
            route,
            restore: Vec::new(),
            restore_started: false,
            captured: vec![None; self.ranks.len()],
        });
    }

    /// The aggregate record's pieces, one per owner, a restore applies.
    pub fn set_session_restore(&mut self, pieces: Vec<SessionPieceKey>) -> Result<()> {
        if pieces.len() != self.ranks.len() {
            return Err("Qwen4 session record does not have one piece per band".into());
        }
        self.session
            .as_mut()
            .ok_or("Qwen4 session prompt is not bound")?
            .restore = pieces;
        Ok(())
    }

    /// Whether any owner began restoring (a failure after that point leaves
    /// mixed state; before it nothing changed).
    pub fn session_restore_started(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.restore_started)
    }

    /// The pool keys the last capture named, one slot per owner (`None` where
    /// a band had no cold piece to name), clearing them.
    pub fn take_session_captures(&mut self) -> Vec<Option<SessionPieceKey>> {
        let owners = self.ranks.len();
        self.session
            .as_mut()
            .map(|session| std::mem::replace(&mut session.captured, vec![None; owners]))
            .unwrap_or_default()
    }

    fn require_prepared(&self, epoch: &ForwardEpoch) -> Result<()> {
        if self.prepared != Some(*epoch) {
            return Err("Qwen4 mesh epoch has not been prepared".into());
        }
        Ok(())
    }
}

impl MeshStepBindings for Qwen4MeshBindings<'_> {
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        if epoch.actual_rows == 0
            || epoch.route_rows < epoch.actual_rows
            || self.tokens.len() != epoch.actual_rows
            || (epoch.route == MeshRoute::Decode && epoch.actual_rows != 1)
            || self.edge_epochs.len() + 1 != self.ranks.len()
            || self.edge_epochs.contains(&0)
        {
            return Err("invalid Qwen4 mesh rows, tokens or transport epochs".into());
        }
        let end = epoch
            .start_position
            .checked_add(epoch.actual_rows)
            .ok_or("Qwen4 mesh position overflows")?;
        for (owner, rank) in self.ranks.iter().enumerate() {
            let layout = self.layouts[owner].ok_or("Qwen4 mesh rank layout is not prepared")?;
            let dtype_bytes = if layout.bf16_streams { 2 } else { 4 };
            if layout.route_rows != epoch.route_rows
                || layout.stream_bytes
                    != streams_bytes(
                        epoch.actual_rows,
                        rank.bundle.config.hc_count,
                        rank.bundle.config.hidden_size,
                        dtype_bytes,
                    )?
                || self.layouts[0].is_some_and(|first| first.bf16_streams != layout.bf16_streams)
            {
                return Err(
                    "Qwen4 mesh rank route or live streams extent disagrees with epoch".into(),
                );
            }
            if rank.bundle.state.position != epoch.start_position
                || end > rank.bundle.state.max_seq_len
                || epoch.actual_rows > rank.forward.scratch.max_chunk
            {
                return Err("Qwen4 mesh rank position or capacity disagrees with epoch".into());
            }
        }
        let config = &self.ranks[0].bundle.config;
        if self
            .tokens
            .iter()
            .any(|&token| token as usize >= config.vocab_size)
        {
            return Err("Qwen4 mesh token is outside the embedding vocabulary".into());
        }
        if let Some(head) = &self.head {
            let rows = head.rows.count(epoch.actual_rows);
            let elements = rows
                .checked_mul(config.vocab_size)
                .ok_or("Qwen4 head shape overflows")?;
            if head.logits.dtype != rdna_compute::DType::F32 || head.logits.numel() != elements {
                return Err("Qwen4 mesh logits output has invalid dtype or capacity".into());
            }
            let top1_bytes = rows
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or("Qwen4 argmax shape overflows")?;
            if head.top1.is_some_and(|top1| {
                top1.dtype != rdna_compute::DType::Raw || top1.buf.size() < top1_bytes
            }) {
                return Err("Qwen4 mesh argmax output has invalid dtype or capacity".into());
            }
        }
        let verify = epoch.route == MeshRoute::Verify;
        if verify
            && (epoch.route_rows != epoch.actual_rows
                || self.wide.is_none()
                || self
                    .head
                    .as_ref()
                    .is_none_or(|head| head.rows != Qwen4OutputRows::All))
        {
            return Err(
                "a Qwen4 mesh verify routes its actual rows and needs all-row logits and a wide capture"
                    .into(),
            );
        }
        if let Some(wide) = self.wide {
            let elements = epoch
                .actual_rows
                .checked_mul(config.hc_count)
                .and_then(|n| n.checked_mul(config.hidden_size))
                .ok_or("Qwen4 wide capture shape overflows")?;
            if wide.dtype != rdna_compute::DType::F32 || wide.numel() < elements {
                return Err("Qwen4 wide hidden capture has invalid dtype or capacity".into());
            }
        }
        // Every band arms its row capture or none does, and only a verify of
        // at least two rows the ring covers may be armed (an armed snapshot
        // skips the GDN copies, so a one-row armed forward is unrecoverable).
        let armed = self.ranks[0].bundle.state.row_capture_armed;
        if self
            .ranks
            .iter()
            .any(|rank| rank.bundle.state.row_capture_armed != armed)
            || (armed
                && (!verify
                    || epoch.actual_rows < 2
                    || self
                        .ranks
                        .iter()
                        .any(|rank| rank.bundle.state.row_capture_rows() < epoch.actual_rows)))
        {
            return Err(
                "Qwen4 mesh bands disagree on, or cannot honor, the armed row capture".into(),
            );
        }
        self.ran.fill(false);
        self.prepared = Some(*epoch);
        Ok(())
    }

    fn run_ops(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        ops: &[OpId],
    ) -> Result<()> {
        self.require_prepared(epoch)?;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Qwen4 mesh rank")?;
        let expected = rank.forward.mesh_op_range(
            self.head.is_some(),
            self.head.as_ref().is_some_and(|head| head.top1.is_some()),
        );
        if self.ran[owner] || !ops.iter().map(|id| id.0).eq(expected.clone()) {
            return Err(
                "Qwen4 mesh band must run exactly once with its full declared op range".into(),
            );
        }
        let output = rank.bundle.weights.band_scope().output;
        let head = if output {
            self.head.as_ref().map(|head| Qwen4MeshHead {
                logits: head.logits,
                top1: head.top1,
                rows: head.rows,
            })
        } else {
            None
        };
        // Only the output band owns the final pre-collapse streams.
        let wide = if output { self.wide } else { None };
        rank.forward
            .forward_mesh_chunk(
                rank.bundle,
                gpu,
                self.tokens,
                epoch.route_rows,
                epoch.route,
                head,
                wide,
                expected,
            )
            .map_err(|error| error.to_string())?;
        self.ran[owner] = true;
        Ok(())
    }

    fn boundary(
        &self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryIo<'_>> {
        self.require_prepared(epoch)?;
        let CollectiveKind::BandXfer { src, dst } = collective.kind else {
            return Err("Qwen4 topology.pp only admits HC band transfers".into());
        };
        if src.checked_add(1) != Some(dst)
            || dst >= self.ranks.len()
            || (owner != src && owner != dst)
            || collective.operand != QWEN4_STREAMS_RESOURCE
        {
            return Err("Qwen4 HC boundary has invalid endpoints".into());
        }
        let rank = self.ranks.get(owner).ok_or("unknown Qwen4 mesh rank")?;
        let streams = &rank.forward.scratch.streams;
        let layout = self.layouts[owner].ok_or("Qwen4 mesh rank layout is not prepared")?;
        let dtype_bytes = if layout.bf16_streams { 2 } else { 4 };
        let bytes = layout.stream_bytes;
        Ok(BoundaryIo {
            view: streams,
            into: None,
            descriptor: EdgeDescriptor {
                epoch: *self
                    .edge_epochs
                    .get(src)
                    .ok_or("missing Qwen4 edge epoch")?,
                op: epoch.op_id,
                rows: epoch.actual_rows,
                dtype_bytes,
                bytes,
            },
            partial_slot: None,
        })
    }

    fn apply_rank_marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.require_prepared(epoch)?;
        let rank = self.ranks.get(owner).ok_or("unknown Qwen4 mesh rank")?;
        if !self.ran[owner]
            || rank.bundle.state.position != epoch.start_position + epoch.actual_rows
            || rank
                .bundle
                .state
                .qsa
                .iter()
                .any(|qsa| qsa.position != rank.bundle.state.position)
        {
            return Err("Qwen4 band did not apply its local position and QSA marks".into());
        }
        Ok(())
    }

    fn reset_rank(&mut self, owner: usize, gpu: &mut Gpu, _epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        self.session = None;
        self.window = None;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Qwen4 mesh rank")?;
        rank.forward.set_ple_lookahead(&[]);
        drop(rank.forward.ple_ahead.take());
        rank.bundle.reset(gpu).map_err(|error| error.to_string())?;
        // The mesh is drained: the band's uncommitted captures and live state
        // go. Published pieces (pinned by aggregate records) stay valid.
        rank.bundle.pp_session_abort(gpu);
        Ok(())
    }

    fn repair_prefix_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        keep: usize,
        tokens: &[u32],
    ) -> Result<()> {
        let window = self
            .window
            .as_ref()
            .ok_or("Qwen4 window transaction is not bound")?;
        let local = window
            .target
            .get(owner)
            .copied()
            .flatten()
            .ok_or("Qwen4 band has no window ticket to repair from")?;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Qwen4 mesh rank")?;
        let end = epoch
            .start_position
            .checked_add(tokens.len())
            .ok_or("Qwen4 repair position overflows")?;
        if epoch.route != MeshRoute::Verify
            || epoch.actual_rows != tokens.len()
            || keep == 0
            || keep >= tokens.len()
            || rank.bundle.state.position != end
        {
            return Err(format!(
                "Qwen4 band {owner} cannot keep {keep} of the {} verified rows ending at {}",
                tokens.len(),
                rank.bundle.state.position
            ));
        }
        // This band's own row-aware repair: its GDN recurrence prefix, conv
        // history and QSA marks from its own ticket and captured rows, and (on
        // the input band, the only one with a PLE history) the PLE
        // convolution input rows its own forward normalized for the block.
        let ple_normed = rank
            .forward
            .scratch
            .ple_normed_for_rollback()
            .map_err(|error| format!("Qwen4 band {owner} row repair: {error}"))?;
        rank.bundle
            .rollback_verify_rows_retain_with(gpu, local, keep, tokens, ple_normed)
            .map_err(|error| format!("Qwen4 band {owner} row repair: {error}"))
    }
    fn capture_session_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
    ) -> Result<MeshCheckpointId> {
        let session = self
            .session
            .as_mut()
            .ok_or("Qwen4 session prompt is not bound")?;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Qwen4 mesh rank")?;
        // The epoch is the last submitted tile: its rows end at the boundary.
        let position = epoch
            .start_position
            .checked_add(epoch.actual_rows)
            .ok_or("Qwen4 session boundary overflows")?;
        if rank.bundle.state.position != position {
            return Err(format!(
                "Qwen4 band {owner} is at {} but the session boundary is {position}",
                rank.bundle.state.position
            ));
        }
        let key = rank
            .bundle
            .pp_session_capture(gpu, session.prompt, position)?;
        *session
            .captured
            .get_mut(owner)
            .ok_or("unknown Qwen4 mesh rank")? = key;
        Ok(session_ticket(epoch, owner, position))
    }
    fn probe_session_rank(
        &mut self,
        owner: usize,
        epoch: &ForwardEpoch,
        piece: MeshCheckpointId,
    ) -> Result<()> {
        let session = self
            .session
            .as_ref()
            .ok_or("Qwen4 session prompt is not bound")?;
        let key = session
            .restore
            .get(owner)
            .ok_or("Qwen4 session record has no piece for this band")?;
        check_session_ticket(&piece, owner, epoch, key.prefix_len())?;
        let rank = self.ranks.get(owner).ok_or("unknown Qwen4 mesh rank")?;
        if !rank.bundle.pp_session_piece_resident(key) {
            return Err(format!(
                "Qwen4 band {owner} no longer holds its session piece (evicted)"
            ));
        }
        Ok(())
    }
    fn restore_session_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        piece: MeshCheckpointId,
    ) -> Result<()> {
        let session = self
            .session
            .as_mut()
            .ok_or("Qwen4 session prompt is not bound")?;
        let prefix = session
            .restore
            .get(owner)
            .ok_or("Qwen4 session record has no piece for this band")?
            .prefix_len();
        check_session_ticket(&piece, owner, epoch, prefix)?;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Qwen4 mesh rank")?;
        session.restore_started = true;
        // Request-local PLE work of the discarded state must not publish into
        // the restored one; the forward is borrowed here, the bundle's row
        // store epoch is invalidated by its own restore.
        rank.forward.set_ple_lookahead(&[]);
        drop(rank.forward.ple_ahead.take());
        rank.bundle
            .pp_session_begin(gpu, session.prompt, session.route, prefix)
    }
    fn checkpoint_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId> {
        let window = self
            .window
            .as_mut()
            .ok_or("Qwen4 window transaction is not bound")?;
        let bands = self.ranks.len();
        if owner < bands {
            let slot = window
                .target
                .get_mut(owner)
                .ok_or("Qwen4 window has no slot for this band")?;
            if slot.is_some() {
                return Err(format!("Qwen4 band {owner} already holds a window ticket"));
            }
            // The row capture is armed (or not) uniformly by the caller before
            // any band snapshots: an armed snapshot skips the GDN copies.
            *slot = Some(
                self.ranks[owner]
                    .bundle
                    .snapshot(gpu)
                    .map_err(|error| format!("Qwen4 band {owner} window snapshot: {error}"))?,
            );
        } else if owner == bands {
            if window.head.is_some() {
                return Err("Qwen4 window already holds a head ticket".into());
            }
            let output = &mut self.ranks[bands - 1];
            if !output.bundle.weights.band_scope().output {
                return Err("Qwen4 MTP head ticket needs the output band".into());
            }
            window.head = Some(
                output
                    .bundle
                    .mtp_snapshot(gpu)
                    .map_err(|error| format!("Qwen4 head window snapshot: {error}"))?,
            );
        } else {
            return Err("Qwen4 window owner is outside the mesh".into());
        }
        Ok(window_ticket(key, owner))
    }
    fn restore_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()> {
        check_window_ticket(&ticket, owner, key)?;
        let window = self
            .window
            .as_ref()
            .ok_or("Qwen4 window transaction is not bound")?;
        let retain = window.restore == WindowRestore::Retain;
        let bands = self.ranks.len();
        if owner < bands {
            let local = window
                .target
                .get(owner)
                .copied()
                .flatten()
                .ok_or("Qwen4 band has no window ticket to restore")?;
            let bundle = &mut *self.ranks[owner].bundle;
            let restored = if retain {
                bundle.restore_retain(gpu, local)
            } else {
                bundle.restore(gpu, local)
            };
            restored.map_err(|error| format!("Qwen4 band {owner} window restore: {error}"))
        } else if owner == bands {
            let head = window
                .head
                .ok_or("Qwen4 window has no head ticket to restore")?;
            let bundle = &mut *self.ranks[bands - 1].bundle;
            let restored = if retain {
                bundle.mtp_restore_retain(gpu, head)
            } else {
                bundle.mtp_restore(gpu, head)
            };
            restored.map_err(|error| format!("Qwen4 head window restore: {error}"))
        } else {
            Err("Qwen4 window owner is outside the mesh".into())
        }
    }
    fn validate_finalize_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()> {
        check_window_ticket(&ticket, owner, key)?;
        let window = self
            .window
            .as_ref()
            .ok_or("Qwen4 window transaction is not bound")?;
        let bands = self.ranks.len();
        if owner < bands {
            let local = window
                .target
                .get(owner)
                .copied()
                .flatten()
                .ok_or("Qwen4 band has no window ticket to validate")?;
            self.ranks[owner]
                .bundle
                .validate_commit(local)
                .map_err(|error| format!("Qwen4 band {owner} window ticket: {error}"))
        } else if owner == bands {
            let head = window
                .head
                .ok_or("Qwen4 window has no head ticket to validate")?;
            self.ranks[bands - 1]
                .bundle
                .mtp_validate_commit(head)
                .map_err(|error| format!("Qwen4 head window ticket: {error}"))
        } else {
            Err("Qwen4 window owner is outside the mesh".into())
        }
    }
    fn finalize_rank(&mut self, owner: usize, key: &MeshCheckpointKey, ticket: MeshCheckpointId) {
        // Host-only and infallible: `validate_finalize_rank` already vetted this
        // ticket and its local ticket; anything unexpected is left untouched.
        if check_window_ticket(&ticket, owner, key).is_err() {
            return;
        }
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let retain = window.finalize == WindowFinalize::Retain;
        let bands = self.ranks.len();
        if owner < bands {
            if let Some(local) = window.target.get(owner).copied().flatten() {
                let bundle = &mut *self.ranks[owner].bundle;
                if retain {
                    bundle.retain_target_validated(local);
                } else {
                    bundle.commit_validated(local);
                }
            }
        } else if owner == bands {
            if let Some(head) = window.head {
                let bundle = &mut *self.ranks[bands - 1].bundle;
                if retain {
                    bundle.mtp_retain_validated(head);
                } else {
                    bundle.mtp_commit_validated(head);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scopes(counts: &[usize]) -> Vec<Qwen4BandScope> {
        let total: usize = counts.iter().sum();
        let mut start = 0;
        counts
            .iter()
            .map(|&count| {
                let band = start..start + count;
                start += count;
                Qwen4BandScope {
                    input: band.start == 0,
                    output: band.end == total,
                    band,
                }
            })
            .collect()
    }

    #[test]
    fn streams_payload_is_exact_live_extent() {
        assert_eq!(streams_bytes(1, 4, 2560, 4).unwrap(), 40_960);
        assert_eq!(streams_bytes(1536, 4, 2560, 4).unwrap(), 62_914_560);
        assert_eq!(streams_bytes(4096, 4, 2560, 4).unwrap(), 167_772_160);
        assert_eq!(streams_bytes(1536, 4, 2560, 2).unwrap(), 31_457_280);
        assert!(streams_bytes(usize::MAX, 4, 2560, 4).is_err());
        assert!(streams_bytes(0, 4, 2560, 4).is_err());
        assert!(streams_bytes(1, 4, 2560, 1).is_err());
    }

    #[test]
    fn mesh_scope_tiles_and_pins() {
        let config = crate::config::compact_test_config();
        for counts in [&[24, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            assert!(validate_scopes(&config, &scopes(counts)).is_ok());
        }
        let mut bands = scopes(&[24, 24]);
        bands[1].band.start -= 1;
        assert!(validate_scopes(&config, &bands).is_err());
        let mut bands = scopes(&[24, 24]);
        bands[0].input = false;
        assert!(validate_scopes(&config, &bands).is_err());
        let mut bands = scopes(&[24, 24]);
        bands[1].output = false;
        assert!(validate_scopes(&config, &bands).is_err());
        let bands = scopes(&[1, 47]);
        assert!(validate_scopes(&config, &bands).is_err());
        assert!(validate_scopes(&config, &scopes(&[24, 23])).is_err());
        assert!(validate_scopes(&config, &scopes(&[48])).is_err());
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};

    fn epoch(mesh: &DeviceMesh, request: u64) -> ForwardEpoch {
        ForwardEpoch {
            mesh_epoch: mesh.epoch(),
            load_id: 7,
            request_id: request,
            op_id: 1,
            route: MeshRoute::Prefill,
            start_position: 0,
            actual_rows: 1,
            route_rows: 1,
            policy_id: 1,
        }
    }

    fn mesh() -> DeviceMesh {
        DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap()
    }

    #[test]
    fn session_tickets_are_minted_per_owner_for_the_current_request() {
        let mesh = mesh();
        let epoch = epoch(&mesh, 3);
        let a = session_ticket(&epoch, 0, 4096);
        let b = session_ticket(&epoch, 1, 4096);
        // One shared window, distinct owners.
        assert_eq!(a.key(), b.key());
        assert_eq!((a.owner(), b.owner()), (0, 1));
        assert_eq!(a.local(), 4096);
        assert!(check_session_ticket(&a, 0, &epoch, 4096).is_ok());
        assert!(check_session_ticket(&b, 1, &epoch, 4096).is_ok());
    }

    #[test]
    fn stale_foreign_and_misowned_session_tickets_are_refused() {
        let mesh = mesh();
        let other = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let now = epoch(&mesh, 3);
        let ticket = session_ticket(&now, 0, 4096);
        // Another owner's ticket, another prefix, a later request, another
        // load and another mesh generation are all refused.
        assert!(check_session_ticket(&ticket, 1, &now, 4096).is_err());
        assert!(check_session_ticket(&ticket, 0, &now, 8192).is_err());
        assert!(check_session_ticket(&ticket, 0, &epoch(&mesh, 4), 4096).is_err());
        let mut reloaded = now;
        reloaded.load_id += 1;
        assert!(check_session_ticket(&ticket, 0, &reloaded, 4096).is_err());
        assert!(check_session_ticket(&ticket, 0, &epoch(&other, 3), 4096).is_err());
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};

    fn key(mesh: &DeviceMesh, window: u64) -> MeshCheckpointKey {
        MeshCheckpointKey {
            mesh_epoch: mesh.epoch(),
            load_gen: 7,
            reset_gen: 0,
            request_gen: 3,
            window_gen: window,
        }
    }

    #[test]
    fn window_tickets_name_their_owner_and_window() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let key = key(&mesh, 5);
        let band = window_ticket(&key, 0);
        let output = window_ticket(&key, 1);
        let head = window_ticket(&key, 2);
        assert_eq!((band.owner(), output.owner(), head.owner()), (0, 1, 2));
        assert!(check_window_ticket(&band, 0, &key).is_ok());
        assert!(check_window_ticket(&head, 2, &key).is_ok());
        // Never another owner's ticket (no rank-zero broadcast, no head/trunk mix-up).
        assert!(check_window_ticket(&band, 1, &key).is_err());
        assert!(check_window_ticket(&head, 1, &key).is_err());
    }

    #[test]
    fn stale_foreign_and_reloaded_window_tickets_are_refused() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let other = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let key = key(&mesh, 5);
        let ticket = window_ticket(&key, 0);
        assert!(check_window_ticket(&ticket, 0, &self::key(&mesh, 6)).is_err());
        assert!(check_window_ticket(&ticket, 0, &self::key(&other, 5)).is_err());
        let mut reloaded = key;
        reloaded.load_gen += 1;
        assert!(check_window_ticket(&ticket, 0, &reloaded).is_err());
        let mut next_request = key;
        next_request.request_gen += 1;
        assert!(check_window_ticket(&ticket, 0, &next_request).is_err());
        // A session piece's ticket (local = prefix) is not a window ticket.
        let piece = MeshCheckpointId::issue(key, 0, 4096);
        assert!(check_window_ticket(&piece, 0, &key).is_err());
    }

    #[test]
    fn window_context_collects_one_ticket_per_band() {
        let ctx = WindowCtx::checkpoint(3);
        assert_eq!(ctx.restore, WindowRestore::Consume);
        assert_eq!(ctx.finalize, WindowFinalize::Commit);
        // Nothing captured: no complete ticket set.
        assert!(ctx.into_target().is_none());
    }
}
