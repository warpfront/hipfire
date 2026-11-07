// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative-decode draft head: ranks the vocabulary for one hidden row and
//! returns the draft token.
//!
//! Drafts only steer acceptance; the target verifies every emitted token, so a
//! draft may rank a lower-bit copy of the language head.  With re-scoring the
//! copy's best 8 are re-scored exactly against the source head, which makes
//! the draft the source head's own argmax whenever the copy ranks it there.

use super::layer_ops::{execute_argmax, hip, project_weight};
use crate::families::gemv::WeightRef;
use crate::types::DispatchError;
use hip_bridge::DeviceBuffer;
use rdna_compute::gemv::TOPK8_GROUPS;
use rdna_compute::{DType, Gpu, GpuTensor};
use smallvec::SmallVec;
use std::ffi::c_void;

/// Ranking-copy policy: an optional lower-bit copy of the language head
/// that drafts rank the vocabulary with, and whether its top 8 are
/// re-scored exactly against the source head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftHeadPolicy {
    pub copy: Option<DType>,
    pub rescore: bool,
}

impl DraftHeadPolicy {
    /// `mq2`..`mq6`, optional `r` suffix = rescore; anything else = no copy.
    pub fn parse(choice: &str) -> Self {
        let copy = match choice.trim_end_matches('r') {
            "mq6" => Some(DType::MQ6G256V2),
            "mq5" => Some(DType::MQ5G256V2),
            "mq4" => Some(DType::MQ4G256V2),
            "mq3" => Some(DType::MQ3G256V2),
            "mq2" => Some(DType::MQ2G256V2),
            _ => None,
        };
        Self {
            copy,
            rescore: choice.ends_with('r'),
        }
    }
}

/// Row layout drafts rank: `[0, front) ++ [special, vocab) ++ [front, special)`.
///
/// BPE merge order puts the frequent tokens first, so with re-scoring a draft
/// ranks only the first two ranges (`front` plus the control ids from
/// `special` on) while no recent input token fell in `[front, special)`.
#[derive(Clone, Copy, Debug)]
pub struct DraftHeadLayout {
    pub vocab: usize,
    pub hidden: usize,
    pub front: usize,
    pub special: usize,
    /// Drafts that rank the whole vocabulary after an input id in
    /// `[front, special)`.
    pub full_hold: u32,
}

/// See [`DraftHead::request_state`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftHeadRequestState {
    pub full_steps: u32,
    pub margin: f32,
}

impl Default for DraftHeadRequestState {
    fn default() -> Self {
        Self {
            full_steps: 0,
            margin: f32::INFINITY,
        }
    }
}

pub struct DraftHead {
    /// Lower-bit ranking copy, rows in the layout order (`None` = rank the
    /// source head).
    copy: Option<GpuTensor>,
    /// Top-8 re-score scratch; present only when re-scoring.
    partial: Option<GpuTensor>,
    logits: GpuTensor,
    /// Draft token id, then (with re-scoring) its f32 logit margin, the 8
    /// re-scored candidates and their exact logits.
    top1: GpuTensor,
    rotation: GpuTensor,
    /// 0 = whole vocabulary always, rows in token order.
    front: usize,
    special: usize,
    vocab: usize,
    hidden: usize,
    full_hold: u32,
    full_steps: u32,
    margin: f32,
    /// Last draft's re-scored `(token, exact logit)` candidates (re-scoring
    /// only; zeroed otherwise).
    candidates: [(u32, f32); 8],
    /// The vocabulary-shard context of the current replicated head round (`None` outside one);
    /// see [`DraftShard`].
    shard: Option<DraftShard>,
}

/// Groups (partial blocks) each EP rank reduces of a sharded draft: half of the 128.
const SHARD_GROUPS: usize = TOPK8_GROUPS / 2;

/// This rank's `(first row, rows)` of a ranked row of `ranked` rows: the split falls on the
/// partial-block boundary `64 * ceil(ranked / 128)`, so each rank owns exactly 64 of the 128
/// groups and every group's rows are its own.
fn shard_range(ranked: usize, rank: usize) -> (usize, usize) {
    let split = (SHARD_GROUPS * ranked.div_ceil(TOPK8_GROUPS)).min(ranked);
    if rank == 0 {
        (0, split)
    } else {
        (split, ranked - split)
    }
}

/// A non-owning view of rows `[lo, lo + rows)` of a ranking copy (rows are contiguous, in the
/// copy's layout order).
fn copy_rows(
    copy: &GpuTensor,
    hidden: usize,
    lo: usize,
    rows: usize,
) -> Result<GpuTensor, DispatchError> {
    let row_bytes = copy.dtype.row_bytes(hidden).ok_or_else(|| {
        DispatchError::Hip(format!("draft shard: {:?} has no row size", copy.dtype))
    })?;
    let (offset, len) = (lo * row_bytes, rows * row_bytes);
    if offset + len > copy.buf.size() {
        return Err(DispatchError::Hip(format!(
            "draft shard: rows [{lo}, {}) of the ranking copy exceed its {} bytes",
            lo + rows,
            copy.buf.size()
        )));
    }
    Ok(GpuTensor {
        buf: copy.buf.byte_view(offset, len),
        shape: vec![len],
        dtype: copy.dtype,
    })
}

/// The cross-rank context of a vocabulary-sharded draft round ([`DraftHead::set_shard`]).
///
/// The two EP ranks each rank HALF of a draft's ranked rows: rank `r` runs the MQ2 GEMV over
/// the rows of its 64 partial blocks and reduces those blocks into its half of a partial slot;
/// the halves are copied into the peer's identical slot, `exchange` orders them (publish, host
/// barrier, device wait on the peer), and both ranks then run the same exact re-score over the
/// same 1,024 pairs, so both hold today's 72-byte slot. Every draft row of the round claims its
/// own slot of `partial_rows` (the round has at most one draft per lane), so a peer's copy of
/// one draft never lands on a slot another draft's re-score still reads.
pub struct DraftShard {
    rank: usize,
    /// This rank's partial scratch: `rows` slots of [`Gpu::TOPK8_PARTIAL_BYTES`] (Raw, a view).
    partial_rows: GpuTensor,
    /// Device address of the peer's identical scratch (valid on this rank's device).
    peer_partial_rows: u64,
    /// HIP device id the peer's scratch lives on.
    peer_device: i32,
    /// One exchange: publish this rank's event, meet the peer's host, wait on the peer's event
    /// on the device. Called once per draft call (a batched class is one call), in the same
    /// order on both ranks.
    exchange: Box<dyn FnMut(&mut Gpu) -> Result<(), DispatchError> + Send>,
    /// Slots claimed so far in this round.
    used: usize,
}

impl DraftShard {
    /// Ranks a draft shard splits the head between.
    pub const RANKS: usize = 2;

    pub fn new(
        rank: usize,
        partial_rows: GpuTensor,
        peer_partial_rows: u64,
        peer_device: i32,
        exchange: Box<dyn FnMut(&mut Gpu) -> Result<(), DispatchError> + Send>,
    ) -> Result<Self, DispatchError> {
        if rank >= Self::RANKS
            || peer_partial_rows == 0
            || partial_rows.dtype != DType::Raw
            || partial_rows.byte_size() < Gpu::TOPK8_PARTIAL_BYTES
        {
            return Err(DispatchError::Hip(format!(
                "draft shard: rank {rank}, peer scratch {peer_partial_rows:#x}, local scratch \
                 {:?} x {} bytes",
                partial_rows.dtype,
                partial_rows.byte_size()
            )));
        }
        Ok(Self {
            rank,
            partial_rows,
            peer_partial_rows,
            peer_device,
            exchange,
            used: 0,
        })
    }

    /// Draft rows the partial scratch holds.
    pub fn rows(&self) -> usize {
        self.partial_rows.byte_size() / Gpu::TOPK8_PARTIAL_BYTES
    }

    /// Claim `rows` consecutive slots for one draft call; the index of the first.
    fn claim(&mut self, rows: usize) -> Result<usize, DispatchError> {
        let first = self.used;
        if first + rows > self.rows() {
            return Err(DispatchError::Hip(format!(
                "draft shard: {} draft rows in one round exceed the {} partial slots",
                first + rows,
                self.rows()
            )));
        }
        self.used += rows;
        Ok(first)
    }
}

/// Push this rank's halves of slots `first..first + rows` (pv words `[512 r, 512 r + 512)` and the
/// same words of pi) into the peer's identical slots on the active stream, then run the
/// exchange: the peer's copies into this rank's scratch are complete on the device once the
/// exchange's wait returns.
fn shard_exchange(
    shard: &mut DraftShard,
    gpu: &mut Gpu,
    first: usize,
    rows: usize,
) -> Result<(), DispatchError> {
    let half = SHARD_GROUPS * 8 * std::mem::size_of::<f32>();
    let pi_base = TOPK8_GROUPS * 8 * std::mem::size_of::<f32>();
    // SAFETY: the peer's scratch is a live device allocation of the same size as this rank's
    // (the exchange allocated both before peer access was enabled); the view is non-owning.
    let peer = unsafe {
        DeviceBuffer::from_raw(
            shard.peer_partial_rows as *mut c_void,
            shard.partial_rows.buf.size(),
        )
    };
    for slot in first..first + rows {
        for base in [0, pi_base] {
            let offset = slot * Gpu::TOPK8_PARTIAL_BYTES + base + shard.rank * half;
            let src = shard.partial_rows.buf.byte_view(offset, half);
            let dst = peer.byte_view(offset, half);
            match gpu.active_stream.as_ref() {
                Some(stream) => hip(gpu.hip.memcpy_peer_async(
                    &dst,
                    shard.peer_device,
                    &src,
                    gpu.device_id,
                    half,
                    stream,
                ))?,
                None => {
                    hip(gpu
                        .hip
                        .memcpy_peer(&dst, shard.peer_device, &src, gpu.device_id, half))?
                }
            }
        }
    }
    (shard.exchange)(gpu)
}

/// Ranked row count and the `(front, special, tail)` order the re-score
/// kernel maps ranked rows back to token ids with.
fn ranking(
    front: usize,
    special: usize,
    vocab: usize,
    full_steps: u32,
) -> (usize, (usize, usize, usize)) {
    match (front, full_steps) {
        (0, _) => (vocab, (vocab, vocab, 0)),
        (front, 0) => (front + vocab - special, (front, special, vocab - special)),
        (front, _) => (vocab, (front, special, vocab - special)),
    }
}

/// What [`DraftHead::new`] builds for a `head_dtype` head: the ranking-copy
/// format, whether the copy's top 8 are re-scored, and the layout front.
fn plan(
    head_dtype: DType,
    layout: DraftHeadLayout,
    policy: DraftHeadPolicy,
) -> (Option<DType>, bool, usize) {
    // Only a bigger head is worth a smaller copy.
    let copy_format = policy.copy.filter(|&format| {
        matches!(
            head_dtype,
            DType::Q8_0 | DType::BF16 | DType::MQ6G256V2 | DType::MQ5G256V2
        ) && format != head_dtype
    });
    // The re-score kernel reads Q8_0 or MQ6G256V2 rows at K = 2560.
    let rescore = policy.rescore
        && copy_format.is_some()
        && matches!(head_dtype, DType::Q8_0 | DType::MQ6G256V2)
        && layout.hidden == 2560;
    let front = if rescore && layout.front < layout.special {
        layout.front
    } else {
        0
    };
    (copy_format, rescore, front)
}

impl DraftHead {
    /// `(resident, load scratch)` device bytes [`Self::new`] takes for a
    /// `head_dtype` head: what the head keeps, and the most it holds on top
    /// of that while building — the row-chunked F32 requant scratch
    /// (`Gpu::REQUANT_G256_CHUNK_ROWS` rows), which goes back to the device,
    /// not the pool, before `new` returns.
    pub fn device_bytes(
        head_dtype: DType,
        layout: DraftHeadLayout,
        policy: DraftHeadPolicy,
    ) -> Option<(usize, usize)> {
        let (resident, scratch) = Self::allocation_sizes(head_dtype, layout, policy)?;
        let total = |sizes: Vec<usize>| {
            sizes
                .into_iter()
                .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        };
        Some((total(resident)?, total(scratch)?))
    }

    /// [`Self::device_bytes`] per allocation: `(resident, load scratch)`, one
    /// entry per device allocation [`Self::new`] makes, in allocation order
    /// (the requantized ranking copy, the re-score partial scratch, the F32
    /// logits, the top-8 output, the F32 rotation row) and, for the scratch,
    /// the one F32 requant chunk. Zero-byte entries are omitted.
    pub fn allocation_sizes(
        head_dtype: DType,
        layout: DraftHeadLayout,
        policy: DraftHeadPolicy,
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let (copy_format, rescore, _) = plan(head_dtype, layout, policy);
        let mut resident = Vec::with_capacity(5);
        let mut scratch = Vec::new();
        if let Some(format) = copy_format {
            let (values, copy) = Gpu::requant_g256_bytes(layout.vocab, layout.hidden, format)?;
            resident.push(copy);
            scratch.push(values);
        }
        if rescore {
            resident.push(Gpu::TOPK8_PARTIAL_BYTES);
        }
        resident.push(layout.vocab.checked_mul(std::mem::size_of::<f32>())?);
        resident.push(Gpu::TOPK8_OUT_BYTES);
        resident.push(layout.hidden.checked_mul(std::mem::size_of::<f32>())?);
        resident.retain(|&bytes| bytes > 0);
        scratch.retain(|&bytes| bytes > 0);
        Some((resident, scratch))
    }

    pub fn new(
        gpu: &mut Gpu,
        head: &GpuTensor,
        layout: DraftHeadLayout,
        policy: DraftHeadPolicy,
    ) -> Result<Self, DispatchError> {
        let (copy_format, rescore, front) = plan(head.dtype, layout, policy);
        let mut owned: Vec<GpuTensor> = Vec::with_capacity(5);
        let allocated = (|| -> hip_bridge::HipResult<()> {
            if let Some(format) = copy_format {
                // Requantized straight into the layout order: no unordered
                // copy beside the ordered one.
                let (front, special, vocab) = (front, layout.special, layout.vocab);
                let ranges: &[(usize, usize)] = if front == 0 {
                    &[(0, vocab)]
                } else {
                    &[(0, front), (special, vocab), (front, special)]
                };
                owned.push(gpu.requant_g256_rows(head, vocab, layout.hidden, format, ranges)?);
            }
            if rescore {
                owned.push(gpu.zeros(&[Gpu::TOPK8_PARTIAL_BYTES], DType::Raw)?);
            }
            owned.push(gpu.zeros(&[layout.vocab], DType::F32)?);
            owned.push(gpu.zeros(&[Gpu::TOPK8_OUT_BYTES], DType::Raw)?);
            owned.push(gpu.zeros(&[layout.hidden], DType::F32)?);
            Ok(())
        })();
        if let Err(error) = allocated {
            for tensor in owned {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(DispatchError::Hip(error.to_string()));
        }
        let mut owned = owned.into_iter();
        let mut next = || owned.next().expect("draft head allocation count");
        let copy = copy_format.map(|_| next());
        let partial = rescore.then(&mut next);
        Ok(Self {
            copy,
            partial,
            logits: next(),
            top1: next(),
            rotation: next(),
            front,
            special: layout.special,
            vocab: layout.vocab,
            hidden: layout.hidden,
            full_hold: layout.full_hold,
            full_steps: 0,
            margin: f32::INFINITY,
            candidates: [(0, f32::NEG_INFINITY); 8],
            shard: None,
        })
    }

    /// An input id in `[front, special)` ranks the whole vocabulary for
    /// `full_hold` drafts.
    pub fn observe(&mut self, token: u32) {
        Self::observe_into(
            self.front,
            self.special,
            self.full_hold,
            &mut self.full_steps,
            token,
        );
    }

    /// [`Self::observe`] on a lane's own request state (the head's resident
    /// state is left alone): the same rule on `request.full_steps`.
    pub fn observe_request(&self, request: &mut DraftHeadRequestState, token: u32) {
        Self::observe_into(
            self.front,
            self.special,
            self.full_hold,
            &mut request.full_steps,
            token,
        );
    }

    fn observe_into(
        front: usize,
        special: usize,
        full_hold: u32,
        full_steps: &mut u32,
        token: u32,
    ) {
        if (front..special).contains(&(token as usize)) {
            *full_steps = full_hold;
        }
    }

    /// Request-local draft policy: the full-vocabulary hold left and the
    /// last draft's margin. Ranking weights and scratch are not included.
    pub fn request_state(&self) -> DraftHeadRequestState {
        DraftHeadRequestState {
            full_steps: self.full_steps,
            margin: self.margin,
        }
    }

    pub fn set_request_state(&mut self, state: DraftHeadRequestState) {
        self.full_steps = state.full_steps;
        self.margin = state.margin;
    }

    /// The policy a freshly built head starts from.
    pub fn reset_request_state(&mut self) {
        self.set_request_state(DraftHeadRequestState::default());
    }

    /// Rank `hidden` (`[hidden]` F32) and return the draft token id.
    pub fn draft(
        &mut self,
        gpu: &mut Gpu,
        head: &GpuTensor,
        hidden: &GpuTensor,
    ) -> Result<u32, DispatchError> {
        let top1 = self.top1.sub_offset(0, Gpu::TOPK8_OUT_BYTES);
        self.draft_enqueue(gpu, head, hidden, &top1)?;
        let mut top = [0u8; Gpu::TOPK8_OUT_BYTES];
        let read = if self.partial.is_some() { top.len() } else { 8 };
        hip(gpu.hip.memcpy_dtoh(&mut top[..read], &top1.buf))?;
        self.draft_finish(&top[..read])
    }

    /// The device work of [`Self::draft`]: rank `hidden` (`[hidden]` F32)
    /// into `top1_out` (a Raw tensor of at least `Gpu::TOPK8_OUT_BYTES`
    /// bytes): projection, then the re-score or the argmax. Counts down the
    /// full-vocabulary hold exactly as `draft` does; the result is parsed by
    /// [`Self::draft_finish`] once the caller has downloaded `top1_out`.
    pub fn draft_enqueue(
        &mut self,
        gpu: &mut Gpu,
        head: &GpuTensor,
        hidden: &GpuTensor,
        top1_out: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let full_steps = self.full_steps;
        self.full_steps = full_steps.saturating_sub(1);
        self.rank_row(gpu, head, hidden, top1_out, full_steps)
    }

    /// [`Self::draft_enqueue`] for `requests.len()` lanes' rows in one call:
    /// `hidden_rows` holds row `i` (`[hidden]` F32, at `i * hidden`) for
    /// `requests[i]`, and row `i`'s result lands in `top1_out`'s `i`-th
    /// `Gpu::TOPK8_OUT_BYTES` slot. Each row is ranked under that lane's own
    /// full-vocabulary hold (`requests[i].full_steps`, counted down as
    /// `draft_enqueue` does); the resident request state is not touched.
    ///
    /// Without `batch` scratch every row runs exactly the launches of a
    /// one-row `draft_enqueue` (the shared logits and rotation scratch is
    /// reused row after row). With `batch` = (`logits` of at least `rows *
    /// vocab` F32, `rotation` of at least `rows * hidden` F32) and an
    /// MQ2G256V2 ranking copy, the rows that rank the same layout share one
    /// x-batched GEMV over the copy (every weight row read once per chunk: eight
    /// rows on the gfx1201 PeaceMaker module, four on the hipcc kernel): each row is rotated by the one-row rotate, the x-batched
    /// kernel is byte-identical per row to the one-row GEMV, and each row's
    /// top-8 re-score runs on its own logits row, so every slot holds the
    /// bytes a one-row `draft_enqueue` writes. Every lane's result is parsed
    /// by [`Self::draft_finish_request`].
    pub fn draft_rows_enqueue(
        &mut self,
        gpu: &mut Gpu,
        head: &GpuTensor,
        hidden_rows: &GpuTensor,
        requests: &mut [&mut DraftHeadRequestState],
        top1_out: &GpuTensor,
        batch: Option<(&GpuTensor, &GpuTensor)>,
    ) -> Result<(), DispatchError> {
        let rows = requests.len();
        let slot = Gpu::TOPK8_OUT_BYTES;
        let hidden = self.hidden;
        if hidden_rows.dtype != DType::F32
            || hidden_rows.numel() < rows * hidden
            || top1_out.byte_size() < rows * slot
        {
            return Err(DispatchError::Hip(format!(
                "draft rows: {rows} rows need {} F32 hidden elements and {} output bytes",
                rows * hidden,
                rows * slot
            )));
        }
        let holds: SmallVec<[u32; 8]> = requests.iter().map(|r| r.full_steps).collect();
        for request in requests.iter_mut() {
            request.full_steps = request.full_steps.saturating_sub(1);
        }
        let ranking_copy_mq2 = self
            .copy
            .as_ref()
            .is_some_and(|copy| copy.dtype == DType::MQ2G256V2);
        let batch = batch.filter(|(logits, rotation)| {
            ranking_copy_mq2
                && rows >= 2
                && hidden % 256 == 0
                && logits.dtype == DType::F32
                && logits.numel() >= rows * self.vocab
                && rotation.dtype == DType::F32
                && rotation.numel() >= rows * hidden
        });
        let Some((logits_rows, rotation_rows)) = batch else {
            for (row, &hold) in holds.iter().enumerate() {
                let hidden_row = hidden_rows.sub_offset(row * hidden, hidden);
                let out = top1_out.sub_offset(row * slot, slot);
                self.rank_row(gpu, head, &hidden_row, &out, hold)?;
            }
            return Ok(());
        };
        // Rows with a full-vocabulary hold rank a different layout than the
        // rest: each class is one batch.
        for full_class in [false, true] {
            let members: SmallVec<[usize; 8]> = (0..rows)
                .filter(|&row| (holds[row] > 0) == full_class)
                .collect();
            match members.len() {
                0 => {}
                1 => {
                    let row = members[0];
                    let hidden_row = hidden_rows.sub_offset(row * hidden, hidden);
                    let out = top1_out.sub_offset(row * slot, slot);
                    self.rank_row(gpu, head, &hidden_row, &out, holds[row])?;
                }
                count => {
                    if self.shard_applies() {
                        let mut shard = self.shard.take().expect("draft shard");
                        let result = self.rank_class_sharded(
                            &mut shard,
                            gpu,
                            head,
                            hidden_rows,
                            rotation_rows,
                            logits_rows,
                            top1_out,
                            &members,
                            holds[members[0]],
                        );
                        self.shard = Some(shard);
                        result?;
                        continue;
                    }
                    let (ranked, order) =
                        ranking(self.front, self.special, self.vocab, holds[members[0]]);
                    let copy = self.copy.as_ref().expect("MQ2 ranking copy");
                    for (slot_in_batch, &row) in members.iter().enumerate() {
                        let hidden_row = hidden_rows.sub_offset(row * hidden, hidden);
                        let rotated = rotation_rows.sub_offset(slot_in_batch * hidden, hidden);
                        hip(gpu.rotate_x_mq(&hidden_row, &rotated, hidden))?;
                    }
                    let x = rotation_rows.sub_offset(0, count * hidden);
                    let y = logits_rows.sub_offset(0, count * ranked);
                    hip(gpu.gemv_mq2g256v2_xbatch(copy, &x, &y, ranked, hidden, count))?;
                    for (slot_in_batch, &row) in members.iter().enumerate() {
                        let logits_row = logits_rows.sub_offset(slot_in_batch * ranked, ranked);
                        let out = top1_out.sub_offset(row * slot, slot);
                        if let Some(partial) = self.partial.as_ref() {
                            let hidden_row = hidden_rows.sub_offset(row * hidden, hidden);
                            let rotated = rotation_rows.sub_offset(slot_in_batch * hidden, hidden);
                            // An MQ6 head reads the ranking copy's FWHT-rotated input.
                            let x = if head.dtype == DType::Q8_0 {
                                &hidden_row
                            } else {
                                &rotated
                            };
                            hip(gpu.topk8_rescore_k2560(
                                &logits_row,
                                ranked,
                                order,
                                head,
                                x,
                                partial,
                                &out,
                            ))?;
                        } else {
                            execute_argmax(gpu, &logits_row, &out, 1, ranked)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// One row's ranking launches (see [`Self::draft_enqueue`]) under an
    /// explicit full-vocabulary hold.
    fn rank_row(
        &mut self,
        gpu: &mut Gpu,
        head: &GpuTensor,
        hidden: &GpuTensor,
        top1_out: &GpuTensor,
        full_steps: u32,
    ) -> Result<(), DispatchError> {
        if self.shard_applies() {
            let mut shard = self.shard.take().expect("draft shard");
            let result = self.rank_row_sharded(&mut shard, gpu, head, hidden, top1_out, full_steps);
            self.shard = Some(shard);
            return result;
        }
        let (ranked, order) = ranking(self.front, self.special, self.vocab, full_steps);
        let ranking_head = self.copy.as_ref().unwrap_or(head);
        if ranking_head.dtype == DType::MQ2G256V2 && gpu.mq2_xbatch_pm_active(ranked, self.hidden) {
            // The gfx1201 PeaceMaker x-batch module at one row: the bytes of
            // the one-row GEMV (`mq2_xbatch_oracle`), the rotated input left in
            // `self.rotation` exactly as `project_weight` leaves it.
            hip(gpu.rotate_x_mq(hidden, &self.rotation, self.hidden))?;
            hip(gpu.gemv_mq2g256v2_xbatch(
                ranking_head,
                &self.rotation,
                &self.logits,
                ranked,
                self.hidden,
                1,
            ))?;
        } else {
            project_weight(
                gpu,
                &WeightRef {
                    buf: ranking_head,
                    dtype: ranking_head.dtype,
                    m: ranked,
                    k: self.hidden,
                    row_stride: ranking_head
                        .dtype
                        .row_bytes(self.hidden)
                        .unwrap_or(self.hidden * ranking_head.dtype.size()),
                    rotation: None,
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                },
                hidden,
                &self.logits,
                1,
                Some(&self.rotation),
            )?;
        }
        if let Some(partial) = self.partial.as_ref() {
            // An MQ6 head reads the ranking copy's FWHT-rotated input.
            let x = if head.dtype == DType::Q8_0 {
                hidden
            } else {
                &self.rotation
            };
            hip(gpu.topk8_rescore_k2560(&self.logits, ranked, order, head, x, partial, top1_out))?;
        } else {
            execute_argmax(gpu, &self.logits, top1_out, 1, ranked)?;
        }
        Ok(())
    }

    /// Whether this call ranks through the vocabulary shard: a shard context is installed
    /// ([`Self::set_shard`]), the re-score path is active (`partial`) and the ranking copy is
    /// MQ2G256V2. Anything else runs the launches of the unsharded head, byte for byte.
    fn shard_applies(&self) -> bool {
        self.shard.is_some()
            && self.partial.is_some()
            && self
                .copy
                .as_ref()
                .is_some_and(|copy| copy.dtype == DType::MQ2G256V2)
    }

    /// [`Self::rank_row`] on this rank's half of the ranked rows (see [`DraftShard`]): the MQ2
    /// GEMV over rows `[lo, lo + m)` of the copy (the route the one-row call picks for the whole
    /// ranked row), the partial blocks of this rank's half of the 128 groups into the call's
    /// slot of the shard's partial scratch, one exchange of the halves, then the unchanged exact
    /// re-score over the full partial. Both ranks run it with the same arguments.
    fn rank_row_sharded(
        &self,
        shard: &mut DraftShard,
        gpu: &mut Gpu,
        head: &GpuTensor,
        hidden: &GpuTensor,
        top1_out: &GpuTensor,
        full_steps: u32,
    ) -> Result<(), DispatchError> {
        let (ranked, order) = ranking(self.front, self.special, self.vocab, full_steps);
        let copy = self.copy.as_ref().expect("MQ2 ranking copy");
        let (lo, m) = shard_range(ranked, shard.rank);
        let first = shard.claim(1)?;
        let pm = gpu.mq2_xbatch_pm_active(ranked, self.hidden);
        if m > 0 && !pm {
            let half = copy_rows(copy, self.hidden, lo, m)?;
            project_weight(
                gpu,
                &WeightRef {
                    buf: &half,
                    dtype: half.dtype,
                    m,
                    k: self.hidden,
                    row_stride: half
                        .dtype
                        .row_bytes(self.hidden)
                        .unwrap_or(self.hidden * half.dtype.size()),
                    rotation: None,
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                },
                hidden,
                &self.logits,
                1,
                Some(&self.rotation),
            )?;
        } else {
            // The gfx1201 PeaceMaker x-batch module at one row (or this rank's empty half): the
            // rotated input is left in `self.rotation` exactly as `project_weight` leaves it.
            hip(gpu.rotate_x_mq(hidden, &self.rotation, self.hidden))?;
            if m > 0 {
                let half = copy_rows(copy, self.hidden, lo, m)?;
                hip(gpu.gemv_mq2g256v2_xbatch(
                    &half,
                    &self.rotation,
                    &self.logits,
                    m,
                    self.hidden,
                    1,
                ))?;
            }
        }
        let partial = shard
            .partial_rows
            .sub_offset(first * Gpu::TOPK8_PARTIAL_BYTES, Gpu::TOPK8_PARTIAL_BYTES);
        hip(gpu.topk8_partial_blocks(
            &self.logits,
            lo,
            ranked,
            order,
            &partial,
            SHARD_GROUPS * shard.rank,
            SHARD_GROUPS,
        ))?;
        shard_exchange(shard, gpu, first, 1)?;
        // An MQ6 head reads the ranking copy's FWHT-rotated input.
        let x = if head.dtype == DType::Q8_0 {
            hidden
        } else {
            &self.rotation
        };
        hip(gpu.topk8_rescore_only(&partial, head, x, top1_out))
    }

    /// [`Self::draft_rows_enqueue`]'s batched class on this rank's half of the ranked rows: each
    /// member row is rotated as the unsharded class does, ONE x-batched MQ2 GEMV runs over the
    /// copy's rows `[lo, lo + m)` (`y` packed `[count][m]`), each member's partial blocks of this
    /// rank's groups land in its own slot of the partial scratch, ONE exchange moves every
    /// member's half, and each member's exact re-score runs on the full partial. The slots of
    /// the class are consecutive; the result slots are the bytes a one-row call writes.
    #[allow(clippy::too_many_arguments)]
    fn rank_class_sharded(
        &self,
        shard: &mut DraftShard,
        gpu: &mut Gpu,
        head: &GpuTensor,
        hidden_rows: &GpuTensor,
        rotation_rows: &GpuTensor,
        logits_rows: &GpuTensor,
        top1_out: &GpuTensor,
        members: &[usize],
        full_steps: u32,
    ) -> Result<(), DispatchError> {
        let hidden = self.hidden;
        let slot = Gpu::TOPK8_OUT_BYTES;
        let count = members.len();
        let (ranked, order) = ranking(self.front, self.special, self.vocab, full_steps);
        let copy = self.copy.as_ref().expect("MQ2 ranking copy");
        let (lo, m) = shard_range(ranked, shard.rank);
        let first = shard.claim(count)?;
        for (slot_in_batch, &row) in members.iter().enumerate() {
            let hidden_row = hidden_rows.sub_offset(row * hidden, hidden);
            let rotated = rotation_rows.sub_offset(slot_in_batch * hidden, hidden);
            hip(gpu.rotate_x_mq(&hidden_row, &rotated, hidden))?;
        }
        let y = logits_rows.sub_offset(0, count * m);
        if m > 0 {
            let half = copy_rows(copy, hidden, lo, m)?;
            let x = rotation_rows.sub_offset(0, count * hidden);
            hip(gpu.gemv_mq2g256v2_xbatch(&half, &x, &y, m, hidden, count))?;
        }
        for slot_in_batch in 0..count {
            let logits_row = y.sub_offset(slot_in_batch * m, m);
            let partial = shard.partial_rows.sub_offset(
                (first + slot_in_batch) * Gpu::TOPK8_PARTIAL_BYTES,
                Gpu::TOPK8_PARTIAL_BYTES,
            );
            hip(gpu.topk8_partial_blocks(
                &logits_row,
                lo,
                ranked,
                order,
                &partial,
                SHARD_GROUPS * shard.rank,
                SHARD_GROUPS,
            ))?;
        }
        shard_exchange(shard, gpu, first, count)?;
        for (slot_in_batch, &row) in members.iter().enumerate() {
            let partial = shard.partial_rows.sub_offset(
                (first + slot_in_batch) * Gpu::TOPK8_PARTIAL_BYTES,
                Gpu::TOPK8_PARTIAL_BYTES,
            );
            let out = top1_out.sub_offset(row * slot, slot);
            let hidden_row = hidden_rows.sub_offset(row * hidden, hidden);
            let rotated = rotation_rows.sub_offset(slot_in_batch * hidden, hidden);
            // An MQ6 head reads the ranking copy's FWHT-rotated input.
            let x = if head.dtype == DType::Q8_0 {
                &hidden_row
            } else {
                &rotated
            };
            hip(gpu.topk8_rescore_only(&partial, head, x, &out))?;
        }
        Ok(())
    }

    /// Install (or, with `None`, remove) the vocabulary-shard context of the drafts that follow
    /// ([`DraftShard`]). Installed for the span of one replicated head round on both EP ranks;
    /// without it, or without the MQ2 re-score path, every draft runs today's launches.
    pub fn set_shard(&mut self, shard: Option<DraftShard>) {
        self.shard = shard;
    }

    /// Parse a downloaded `top1_out` of [`Self::draft_enqueue`] (its first
    /// `Gpu::TOPK8_OUT_BYTES` bytes with re-scoring, 8 bytes without): the
    /// draft token, the margin and the re-scored candidates, as `draft`.
    pub fn draft_finish(&mut self, top: &[u8]) -> Result<u32, DispatchError> {
        let need = if self.partial.is_some() {
            Gpu::TOPK8_OUT_BYTES
        } else {
            8
        };
        if top.len() < need {
            return Err(DispatchError::Hip(format!(
                "draft result holds {} bytes, expected {need}",
                top.len()
            )));
        }
        let word = |i: usize| [top[4 * i], top[4 * i + 1], top[4 * i + 2], top[4 * i + 3]];
        let token = u32::from_ne_bytes(word(0));
        self.margin = if self.partial.is_some() {
            for (w, candidate) in self.candidates.iter_mut().enumerate() {
                *candidate = (
                    u32::from_ne_bytes(word(2 + w)),
                    f32::from_ne_bytes(word(10 + w)),
                );
            }
            f32::from_ne_bytes(word(1))
        } else {
            f32::INFINITY
        };
        if token as usize >= self.vocab {
            return Err(DispatchError::Hip(format!(
                "draft token {token} is outside vocab {}",
                self.vocab
            )));
        }
        Ok(token)
    }

    /// [`Self::draft_finish`] for a lane's own request state: the draft token,
    /// with the margin written to `request` (the head's resident request state
    /// keeps its margin; the re-scored candidates are the head's, as after
    /// `draft_finish`).
    pub fn draft_finish_request(
        &mut self,
        request: &mut DraftHeadRequestState,
        top: &[u8],
    ) -> Result<u32, DispatchError> {
        let resident = self.margin;
        let token = self.draft_finish(top);
        request.margin = self.margin;
        self.margin = resident;
        token
    }

    /// Exact logit margin of the last draft over its runner-up among the
    /// re-scored candidates (infinite without re-scoring).
    pub fn margin(&self) -> f32 {
        self.margin
    }

    /// The last draft's 8 re-scored candidates as `(token, exact logit)`:
    /// the support a sampled draft is drawn from. `None` without re-scoring,
    /// where [`Self::logits`] is the whole vocabulary in token order (the
    /// reordered layout is used only with re-scoring).
    pub fn rescored_candidates(&self) -> Option<&[(u32, f32); 8]> {
        self.partial.as_ref().map(|_| &self.candidates)
    }

    /// Logits of the last draft, rows in the ranking layout.
    pub fn logits(&self) -> &GpuTensor {
        &self.logits
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        let tensors = [
            Some(self.logits),
            Some(self.top1),
            Some(self.rotation),
            self.copy,
            self.partial,
        ];
        for tensor in tensors.into_iter().flatten() {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_parses_copy_tier_and_rescore_suffix() {
        let parse = DraftHeadPolicy::parse;
        assert_eq!(
            parse("mq2r"),
            DraftHeadPolicy {
                copy: Some(DType::MQ2G256V2),
                rescore: true
            }
        );
        assert_eq!(
            parse("mq6"),
            DraftHeadPolicy {
                copy: Some(DType::MQ6G256V2),
                rescore: false
            }
        );
        assert_eq!(
            parse("off"),
            DraftHeadPolicy {
                copy: None,
                rescore: false
            }
        );
        assert_eq!(
            parse("mq9r"),
            DraftHeadPolicy {
                copy: None,
                rescore: true
            }
        );
    }

    #[test]
    fn ranking_orders_front_special_tail_and_holds_full_vocab() {
        let (vocab, front, special) = (1000, 600, 900);
        assert_eq!(ranking(0, special, vocab, 3), (vocab, (vocab, vocab, 0)));
        assert_eq!(
            ranking(front, special, vocab, 0),
            (front + vocab - special, (front, special, vocab - special))
        );
        assert_eq!(
            ranking(front, special, vocab, 1),
            (vocab, (front, special, vocab - special))
        );
    }

    /// The two ranks' ranges tile the ranked rows, and the split is the partial-block
    /// boundary: rank 0 owns exactly the rows of groups 0..64, rank 1 those of groups 64..128.
    #[test]
    fn shard_ranges_split_at_the_partial_block_boundary() {
        for ranked in [1usize, 100, 128, 129, 8_192, 100_276, 248_320, 248_321] {
            let chunk = ranked.div_ceil(TOPK8_GROUPS);
            let (lo0, m0) = shard_range(ranked, 0);
            let (lo1, m1) = shard_range(ranked, 1);
            assert_eq!((lo0, lo0 + m0), (0, lo1), "ranked {ranked}");
            assert_eq!(lo1 + m1, ranked, "ranked {ranked}");
            // Group g covers rows [g * chunk, min(n, (g + 1) * chunk)): rank 0's 64 groups end at
            // the split, rank 1's begin there.
            assert_eq!(lo1, (SHARD_GROUPS * chunk).min(ranked), "ranked {ranked}");
        }
        // The production shapes.
        assert_eq!(shard_range(100_276, 0), (0, 50_176));
        assert_eq!(shard_range(100_276, 1), (50_176, 50_100));
        assert_eq!(shard_range(248_320, 0), (0, 124_160));
        assert_eq!(shard_range(248_320, 1), (124_160, 124_160));
    }
}
