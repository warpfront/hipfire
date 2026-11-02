// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Design A amendment 2026-10-25 §A9.4: the window-side half of the
//! conditional indexed-argument effect. A `KernelStatus::Conditional` node
//! (receipt `status = conditional`, `required_witnesses = [indexed_bind]`) is
//! PM4-eligible only while the owner holds an [`IndexedBindWitness`] for
//! exactly this window. Nothing here mutates the cached global facts: the
//! witness is a separate runtime value, consumed by
//! [`crate::window::PreparedWindow::attach_indexed_witness`] and re-verified by
//! `submit`.
//!
//! # Sealed positions
//!
//! [`SealedPositions`] owns the dedicated device-only positions allocation
//! (≥ 64 B, sole occupant of its allocation) and its lifecycle
//!
//! ```text
//! Retired -> HostChecked -> UploadInFlight -> SealedReady
//!         -> SubmittedReadOnly -> Retired
//! ```
//!
//! plus `Quarantined` for unknown quiescence (never writable again).
//! `HostChecked` proves the exact `[i32; 16]` bytes that will be uploaded
//! (equal to the existing physical-position producer, nonnegative, pairwise
//! distinct). The upload copies those bytes from an exclusively borrowed
//! [`PositionsStage`]; the stage refuses mutation until the H2D completion
//! fence. `SealedReady` applies the transport's acquire. From the start of the
//! upload until the final completion the positions are leased: overwrite,
//! remap, recycle and any externally queued writer are refused by this API.
//!
//! [`SealedPositions::bind_writers`] then checks, per writer node and from the
//! **bound kernarg image**, the §A9.4 predicates, and the full-program
//! no-writer/alias closure for the positions allocation, and returns the
//! witness (private fields; no public constructor).

use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};

use sha2::{Digest, Sha256};

use crate::kernel::Arg;
use crate::kernel::{ArgRole, KernelNode, KernelProgram, Lowering, WITNESS_INDEXED_BIND};
use crate::window::{AllocationIdentity, Quiescence, WindowStep};
use crate::{Binding, ResourceId};

/// Rows of the sealed positions (`batch_size == 16`).
pub const POSITION_ROWS: usize = 16;
/// Live bytes of the positions.
pub const POSITION_BYTES: u64 = (POSITION_ROWS * 4) as u64;

/// `kv_cache_write_*_batched` kernarg ABI (§A9.1): dst +0, src +8,
/// positions +16, n_kv_heads +24, head_dim +28, batch_size +32,
/// slot_descs +40, row_slot +48. Every offset is asserted against the loaded
/// object's receipt facts before the case is bound.
pub mod abi {
    pub const DST: u32 = 0;
    pub const SRC: u32 = 8;
    pub const POSITIONS: u32 = 16;
    pub const N_KV_HEADS: u32 = 24;
    pub const HEAD_DIM: u32 = 28;
    pub const BATCH: u32 = 32;
    pub const SLOT_DESCS: u32 = 40;
    pub const ROW_SLOT: u32 = 48;
    pub const HEAD_DIM_VALUE: u32 = 256;
    pub const BATCH_VALUE: u32 = 16;
    pub const BLOCK: [u32; 3] = [32, 1, 1];
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IndexBindError {
    #[error("positions allocation: {0}")]
    Allocation(String),
    #[error("positions are in state {found:?}, {op} needs {need:?}")]
    State {
        op: &'static str,
        found: PositionsState,
        need: PositionsState,
    },
    #[error("positions are {0:?}: the lease forbids {1}")]
    Leased(PositionsState, &'static str),
    #[error("host check: {0}")]
    HostCheck(String),
    #[error("stale or foreign witness: {0}")]
    Stale(String),
    #[error("upload payload differs from the host-checked bytes")]
    PayloadMismatch,
    #[error("pinned stage is in flight until the H2D completion fence")]
    StageInFlight,
    #[error("writer node {node}: {detail}")]
    Writer { node: usize, detail: String },
    #[error("closure: {0}")]
    Closure(String),
    #[error("transport: {message} ({quiescence:?})")]
    Transport {
        message: String,
        quiescence: Quiescence,
    },
}

impl IndexBindError {
    fn writer(node: usize, detail: impl Into<String>) -> Self {
        Self::Writer {
            node,
            detail: detail.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionsState {
    Retired,
    HostChecked,
    UploadInFlight,
    SealedReady,
    SubmittedReadOnly,
    Quarantined,
}

/// The trusted H2D + visibility operations of the owner's transport. `begin`
/// copies exactly `bytes` to the positions' device address; `wait` is the
/// completion fence; `acquire` is the railgun entry/system visibility acquire
/// of the existing transport/cache receipts.
pub trait PositionsTransport {
    fn begin_h2d(
        &mut self,
        device: u64,
        bytes: &[u8; POSITION_BYTES as usize],
    ) -> Result<(), TransportFailure>;
    fn wait_h2d(&mut self) -> Result<(), TransportFailure>;
    fn acquire(&mut self) -> Result<(), TransportFailure>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportFailure {
    pub message: String,
    pub quiescence: Quiescence,
}

/// The identity of the positions allocation: base, extent and mapping
/// generation (the generation bumps on remap/recycle).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PositionsIdentity {
    pub base: u64,
    pub bytes: u64,
    pub generation: u64,
}

/// Host staging for the upload. Mutation is refused while an upload that
/// reads it is in flight.
#[derive(Clone, Debug, Default)]
pub struct PositionsStage {
    inner: Arc<Mutex<StageInner>>,
}

#[derive(Debug)]
struct StageInner {
    bytes: [u8; POSITION_BYTES as usize],
    in_flight: bool,
}

impl Default for StageInner {
    fn default() -> Self {
        Self {
            bytes: [0; POSITION_BYTES as usize],
            in_flight: false,
        }
    }
}

impl PositionsStage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Write the stage; refused while an upload reads it.
    pub fn fill(&self, positions: &[i32; POSITION_ROWS]) -> Result<(), IndexBindError> {
        let mut s = lock(&self.inner);
        if s.in_flight {
            return Err(IndexBindError::StageInFlight);
        }
        for (i, p) in positions.iter().enumerate() {
            s.bytes[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
        }
        Ok(())
    }

    /// Raw overwrite of one byte (tests and owners that stage through a
    /// pinned pointer): the same in-flight refusal as [`Self::fill`].
    pub fn write_byte(&self, at: usize, value: u8) -> Result<(), IndexBindError> {
        let mut s = lock(&self.inner);
        if s.in_flight {
            return Err(IndexBindError::StageInFlight);
        }
        s.bytes[at] = value;
        Ok(())
    }

    pub fn is_in_flight(&self) -> bool {
        lock(&self.inner).in_flight
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Proof that `HostChecked` passed for one payload; consumed by
/// [`SealedPositions::begin_upload`].
#[derive(Debug)]
pub struct HostChecked {
    generation: u64,
    digest: [u8; 32],
}

/// The upload in flight; consumed by [`SealedPositions::complete_upload`].
#[derive(Debug)]
pub struct UploadToken {
    generation: u64,
    digest: [u8; 32],
    stage: PositionsStage,
}

struct Inner {
    identity: PositionsIdentity,
    state: PositionsState,
    /// Bumped by every `host_check`: the P generation witnesses bind to.
    generation: u64,
    checked: Option<([u8; POSITION_BYTES as usize], [u8; 32])>,
    positions: Option<[i32; POSITION_ROWS]>,
    /// Externally queued writers not yet completed (admission refuses).
    external_pending: u32,
    /// Bound witness identity: `(window, generation)`.
    bound: Option<(u64, u64)>,
}

/// The dedicated, device-only, immutable-while-leased positions allocation.
#[derive(Clone)]
pub struct SealedPositions {
    inner: Arc<Mutex<Inner>>,
}

impl SealedPositions {
    /// `base`/`bytes`/`generation` of an allocation that holds nothing but
    /// these positions (≥ 64 B).
    pub fn new(identity: PositionsIdentity) -> Result<Self, IndexBindError> {
        if identity.base == 0 || identity.bytes < POSITION_BYTES {
            return Err(IndexBindError::Allocation(format!(
                "needs a nonnull allocation of at least {POSITION_BYTES} bytes, got {:#x}+{}",
                identity.base, identity.bytes
            )));
        }
        if identity.base.checked_add(identity.bytes).is_none() {
            return Err(IndexBindError::Allocation("base + extent overflows".into()));
        }
        if identity.base % 4 != 0 {
            return Err(IndexBindError::Allocation(
                "base is not 4-byte aligned".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                identity,
                state: PositionsState::Retired,
                generation: 0,
                checked: None,
                positions: None,
                external_pending: 0,
                bound: None,
            })),
        })
    }

    pub fn identity(&self) -> PositionsIdentity {
        lock(&self.inner).identity
    }

    pub fn state(&self) -> PositionsState {
        lock(&self.inner).state
    }

    pub fn generation(&self) -> u64 {
        lock(&self.inner).generation
    }

    // ---------------------------------------------------------- host check

    /// `Retired -> HostChecked`. `positions` are the bytes that will be
    /// uploaded; `producer` is the existing physical-position producer's
    /// vector for this window (never an independently guessed one).
    pub fn host_check(
        &self,
        positions: &[i32],
        producer: &[i32],
    ) -> Result<HostChecked, IndexBindError> {
        let mut s = lock(&self.inner);
        if s.state != PositionsState::Retired {
            return Err(IndexBindError::State {
                op: "host_check",
                found: s.state,
                need: PositionsState::Retired,
            });
        }
        let fail = |m: String| Err(IndexBindError::HostCheck(m));
        let Ok(p): Result<[i32; POSITION_ROWS], _> = positions.try_into() else {
            return fail(format!(
                "{} positions, need {POSITION_ROWS}",
                positions.len()
            ));
        };
        if positions != producer {
            return fail("positions differ from the physical-position producer".into());
        }
        if let Some(i) = p.iter().position(|v| *v < 0) {
            return fail(format!("position {i} is negative ({})", p[i]));
        }
        let mut sorted = p;
        sorted.sort_unstable();
        if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
            return fail(format!("position {} appears twice", w[0]));
        }
        let mut bytes = [0u8; POSITION_BYTES as usize];
        for (i, v) in p.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let d = digest(&bytes);
        s.generation += 1;
        s.checked = Some((bytes, d));
        s.positions = Some(p);
        s.bound = None;
        s.state = PositionsState::HostChecked;
        Ok(HostChecked {
            generation: s.generation,
            digest: d,
        })
    }

    // -------------------------------------------------------------- upload

    /// `HostChecked -> UploadInFlight`: copy exactly the checked bytes from
    /// `stage`, which is marked in flight until [`Self::complete_upload`].
    pub fn begin_upload(
        &self,
        checked: HostChecked,
        stage: &PositionsStage,
        transport: &mut dyn PositionsTransport,
    ) -> Result<UploadToken, IndexBindError> {
        let mut s = lock(&self.inner);
        if s.state != PositionsState::HostChecked {
            return Err(IndexBindError::State {
                op: "begin_upload",
                found: s.state,
                need: PositionsState::HostChecked,
            });
        }
        if checked.generation != s.generation {
            return Err(IndexBindError::Stale(format!(
                "host check generation {} is not the current {}",
                checked.generation, s.generation
            )));
        }
        if s.external_pending != 0 {
            s.state = PositionsState::Retired;
            return Err(IndexBindError::Closure(
                "a separately queued writer targets the positions".into(),
            ));
        }
        let (bytes, d) = s.checked.expect("HostChecked holds its payload");
        if d != checked.digest {
            return Err(IndexBindError::PayloadMismatch);
        }
        let mut st = lock(&stage.inner);
        if st.in_flight {
            return Err(IndexBindError::StageInFlight);
        }
        if st.bytes != bytes || digest(&st.bytes) != d {
            s.state = PositionsState::Retired;
            return Err(IndexBindError::PayloadMismatch);
        }
        match transport.begin_h2d(s.identity.base, &st.bytes) {
            Ok(()) => {
                st.in_flight = true;
                s.state = PositionsState::UploadInFlight;
                Ok(UploadToken {
                    generation: s.generation,
                    digest: d,
                    stage: stage.clone(),
                })
            }
            Err(f) => {
                s.state = match f.quiescence {
                    Quiescence::Proven => PositionsState::Retired,
                    Quiescence::Unknown => PositionsState::Quarantined,
                };
                Err(IndexBindError::Transport {
                    message: f.message,
                    quiescence: f.quiescence,
                })
            }
        }
    }

    /// `UploadInFlight -> SealedReady`: the H2D completion fence, then the
    /// railgun entry/system visibility acquire. The stage is released only
    /// after the fence.
    pub fn complete_upload(
        &self,
        token: UploadToken,
        transport: &mut dyn PositionsTransport,
    ) -> Result<(), IndexBindError> {
        let mut s = lock(&self.inner);
        if s.state != PositionsState::UploadInFlight {
            return Err(IndexBindError::State {
                op: "complete_upload",
                found: s.state,
                need: PositionsState::UploadInFlight,
            });
        }
        if token.generation != s.generation || s.checked.map(|(_, d)| d) != Some(token.digest) {
            return Err(IndexBindError::Stale(
                "upload token belongs to another generation or payload".into(),
            ));
        }
        let outcome = transport.wait_h2d();
        if let Err(f) = outcome {
            s.state = match f.quiescence {
                Quiescence::Proven => {
                    lock(&token.stage.inner).in_flight = false;
                    PositionsState::Retired
                }
                // Unknown: the copy may still read the stage and write P.
                Quiescence::Unknown => PositionsState::Quarantined,
            };
            return Err(IndexBindError::Transport {
                message: f.message,
                quiescence: f.quiescence,
            });
        }
        lock(&token.stage.inner).in_flight = false;
        match transport.acquire() {
            Ok(()) => {
                s.state = PositionsState::SealedReady;
                Ok(())
            }
            Err(f) => {
                s.state = match f.quiescence {
                    Quiescence::Proven => PositionsState::Retired,
                    Quiescence::Unknown => PositionsState::Quarantined,
                };
                Err(IndexBindError::Transport {
                    message: f.message,
                    quiescence: f.quiescence,
                })
            }
        }
    }

    // --------------------------------------------------------------- lease

    fn leased(&self, op: &'static str) -> Result<MutexGuard<'_, Inner>, IndexBindError> {
        let s = lock(&self.inner);
        match s.state {
            PositionsState::Retired => Ok(s),
            other => Err(IndexBindError::Leased(other, op)),
        }
    }

    /// Host overwrite of the allocation (CPU mapping, another upload).
    pub fn overwrite(&self) -> Result<(), IndexBindError> {
        self.leased("overwrite").map(|_| ())
    }

    /// VMM remap / reallocation: new generation. Refused unless retired.
    pub fn remap(&self, new_base: u64, new_generation: u64) -> Result<(), IndexBindError> {
        let mut s = self.leased("remap")?;
        if new_base == 0 || new_base % 4 != 0 {
            return Err(IndexBindError::Allocation(
                "remap base is null or misaligned".into(),
            ));
        }
        if new_generation <= s.identity.generation {
            return Err(IndexBindError::Allocation(
                "a remap must raise the mapping generation".into(),
            ));
        }
        s.identity.base = new_base;
        s.identity.generation = new_generation;
        s.checked = None;
        s.positions = None;
        Ok(())
    }

    /// Allocator recycle of the backing storage. Refused unless retired.
    pub fn recycle(&self) -> Result<(), IndexBindError> {
        let mut s = self.leased("recycle")?;
        s.checked = None;
        s.positions = None;
        s.identity.generation += 1;
        Ok(())
    }

    /// A command outside the window that writes `[base, base + bytes)` was
    /// enqueued. A write overlapping the positions is refused while sealed;
    /// while retired it is recorded and blocks the next upload until
    /// [`Self::external_write_completed`].
    pub fn external_write_enqueued(&self, base: u64, bytes: u64) -> Result<(), IndexBindError> {
        let mut s = lock(&self.inner);
        let end = base.checked_add(bytes).unwrap_or(u64::MAX);
        let overlaps = base < s.identity.base + s.identity.bytes && s.identity.base < end;
        if !overlaps {
            return Ok(());
        }
        if s.state != PositionsState::Retired {
            return Err(IndexBindError::Leased(s.state, "an external write"));
        }
        s.external_pending += 1;
        Ok(())
    }

    pub fn external_write_completed(&self) {
        let mut s = lock(&self.inner);
        s.external_pending = s.external_pending.saturating_sub(1);
    }

    // ------------------------------------------------------------ bind/run

    /// `SealedReady`: prove every writer node and the full-program closure
    /// from the window's bound state and return the witness.
    pub fn bind_writers(
        &self,
        input: &BindInputs<'_>,
    ) -> Result<IndexedBindWitness, IndexBindError> {
        let mut s = lock(&self.inner);
        if s.state != PositionsState::SealedReady {
            return Err(IndexBindError::State {
                op: "bind_writers",
                found: s.state,
                need: PositionsState::SealedReady,
            });
        }
        if s.external_pending != 0 {
            return Err(IndexBindError::Closure(
                "a separately queued writer targets the positions".into(),
            ));
        }
        let positions = s.positions.expect("SealedReady holds its positions");
        let (_, payload_digest) = s.checked.expect("SealedReady holds its payload");
        let witness = bind_program(input, s.identity, s.generation, &positions, payload_digest)?;
        s.bound = Some((input.window, s.generation));
        Ok(witness)
    }

    /// `SealedReady -> SubmittedReadOnly`: the queue accepted the program this
    /// witness was bound to.
    pub(crate) fn mark_submitted(
        &self,
        witness: &IndexedBindWitness,
    ) -> Result<(), IndexBindError> {
        let mut s = lock(&self.inner);
        if s.state != PositionsState::SealedReady {
            return Err(IndexBindError::State {
                op: "submit",
                found: s.state,
                need: PositionsState::SealedReady,
            });
        }
        if s.bound != Some((witness.window, witness.p_generation))
            || witness.p_generation != s.generation
            || witness.p != s.identity
            || s.checked.map(|(_, d)| d) != Some(witness.payload_digest)
        {
            return Err(IndexBindError::Stale(
                "witness does not match the sealed generation".into(),
            ));
        }
        s.state = PositionsState::SubmittedReadOnly;
        Ok(())
    }

    /// Verify a witness against the live sealed state without transitioning:
    /// the submit pre-check.
    pub(crate) fn check_witness(&self, witness: &IndexedBindWitness) -> Result<(), IndexBindError> {
        let s = lock(&self.inner);
        if s.state != PositionsState::SealedReady {
            return Err(IndexBindError::State {
                op: "submit",
                found: s.state,
                need: PositionsState::SealedReady,
            });
        }
        if s.bound != Some((witness.window, witness.p_generation))
            || witness.p_generation != s.generation
            || witness.p != s.identity
            || s.checked.map(|(_, d)| d) != Some(witness.payload_digest)
        {
            return Err(IndexBindError::Stale(
                "witness does not match the sealed generation".into(),
            ));
        }
        Ok(())
    }

    /// The submit did not queue the work. Proven: the sealed bytes are no
    /// longer attested for another try, so the positions retire and must be
    /// checked and uploaded again. Unknown: quarantined.
    pub(crate) fn submit_failed(&self, quiescence: Quiescence) {
        let mut s = lock(&self.inner);
        if matches!(
            s.state,
            PositionsState::SealedReady | PositionsState::SubmittedReadOnly
        ) {
            s.state = match quiescence {
                Quiescence::Proven => PositionsState::Retired,
                Quiescence::Unknown => PositionsState::Quarantined,
            };
            s.bound = None;
        }
    }

    /// Final completion (or failure) of the window: `SubmittedReadOnly ->
    /// Retired` on proven quiescence; unknown quiescence quarantines the
    /// positions and their backing storage.
    pub(crate) fn finish(&self, quiescence: Quiescence) {
        let mut s = lock(&self.inner);
        if s.state == PositionsState::SubmittedReadOnly || s.state == PositionsState::SealedReady {
            s.state = match quiescence {
                Quiescence::Proven => PositionsState::Retired,
                Quiescence::Unknown => PositionsState::Quarantined,
            };
            s.bound = None;
        }
    }

    /// The window was abandoned before submit (a refused bind, a dropped
    /// [`crate::window::BoundWindow`]): nothing was queued, so `SealedReady`
    /// retires; the positions must be checked and uploaded again.
    pub fn release_unsubmitted(&self) -> Result<(), IndexBindError> {
        let mut s = lock(&self.inner);
        match s.state {
            PositionsState::SealedReady | PositionsState::HostChecked => {
                s.state = PositionsState::Retired;
                s.bound = None;
                Ok(())
            }
            other => Err(IndexBindError::State {
                op: "release_unsubmitted",
                found: other,
                need: PositionsState::SealedReady,
            }),
        }
    }
}

impl std::fmt::Debug for SealedPositions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = lock(&self.inner);
        f.debug_struct("SealedPositions")
            .field("identity", &s.identity)
            .field("state", &s.state)
            .field("generation", &s.generation)
            .finish()
    }
}

// ------------------------------------------------------------------ witness

/// The per-writer facts a bind discharged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundWriter {
    pub node: usize,
    pub symbol: String,
    pub artifact_sha256: String,
    pub receipt_sha256: String,
    pub case_digest: String,
    pub n_kv_heads: u32,
    pub row_bytes: u64,
    /// Absolute `[start, end)` of each of the 16 rows, in `bid` order.
    pub rows: Vec<(u64, u64)>,
    pub dst: AllocationIdentity,
    pub dst_mapping_generation: u64,
    pub kernarg_digest: [u8; 32],
    pub grid: [u32; 3],
}

/// Runtime-owned bind witness (§A9.2). Private fields, no public constructor:
/// only [`SealedPositions::bind_writers`] produces one, and it is consumed by
/// the window's submit.
#[derive(Debug)]
pub struct IndexedBindWitness {
    window: u64,
    shape_generation: u64,
    p_generation: u64,
    payload_digest: [u8; 32],
    p: PositionsIdentity,
    positions: [i32; POSITION_ROWS],
    writers: Vec<BoundWriter>,
    predicates: Vec<&'static str>,
}

impl IndexedBindWitness {
    pub fn window(&self) -> u64 {
        self.window
    }

    pub fn shape_generation(&self) -> u64 {
        self.shape_generation
    }

    pub fn p_generation(&self) -> u64 {
        self.p_generation
    }

    pub fn payload_digest(&self) -> [u8; 32] {
        self.payload_digest
    }

    pub fn positions(&self) -> &[i32; POSITION_ROWS] {
        &self.positions
    }

    pub fn writers(&self) -> &[BoundWriter] {
        &self.writers
    }

    pub fn predicates(&self) -> &[&'static str] {
        &self.predicates
    }

    /// `(program node index, receipt case digest, witness kind)` of every
    /// guarded case this bind discharged: the only source of the witnesses a
    /// window manifest admission accepts for its `conditional` nodes.
    pub fn discharged_cases(&self) -> Vec<(usize, &str, &'static str)> {
        self.writers
            .iter()
            .map(|w| (w.node, w.case_digest.as_str(), WITNESS_INDEXED_BIND))
            .collect()
    }

    /// Every writer's bound kernarg image and grid still equal the bound
    /// bytes (any post-bind kernarg/shape change is refused at submit).
    pub(crate) fn matches_window(
        &self,
        window: u64,
        shape_generation: u64,
        images: &[Vec<u8>],
        grids: &[[u32; 3]],
    ) -> Result<(), IndexBindError> {
        if self.window != window {
            return Err(IndexBindError::Stale("witness is another window's".into()));
        }
        if self.shape_generation != shape_generation {
            return Err(IndexBindError::Stale(format!(
                "witness shape generation {} is not the bound {shape_generation}",
                self.shape_generation
            )));
        }
        for w in &self.writers {
            let image = images
                .get(w.node)
                .ok_or_else(|| IndexBindError::writer(w.node, "no kernarg image"))?;
            if digest(image) != w.kernarg_digest {
                return Err(IndexBindError::writer(
                    w.node,
                    "kernarg image changed after bind",
                ));
            }
            if grids.get(w.node) != Some(&w.grid) {
                return Err(IndexBindError::writer(w.node, "grid changed after bind"));
            }
        }
        Ok(())
    }
}

/// The owner's per-writer extents: the intended KV tensor slice and the
/// actually mapped/writable intervals of the dst allocation, as byte offsets
/// from its base, with the mapping generation they were read at.
#[derive(Clone, Debug)]
pub struct WriterExtent {
    pub node: usize,
    pub dst_slice: Range<u64>,
    pub mapped: Vec<Range<u64>>,
    pub mapping_generation: u64,
}

/// A window's bound state, as `bind_writers` sees it.
pub struct BindInputs<'a> {
    pub window: u64,
    pub shape_generation: u64,
    pub program: &'a KernelProgram,
    pub steps: &'a [WindowStep],
    pub bindings: &'a [Binding],
    pub allocations: &'a [AllocationIdentity],
    pub images: &'a [Vec<u8>],
    pub grids: &'a [[u32; 3]],
    pub positions_resource: ResourceId,
    pub writers: &'a [WriterExtent],
}

fn le32(image: &[u8], at: u32) -> Option<u32> {
    let at = at as usize;
    Some(u32::from_le_bytes(image.get(at..at + 4)?.try_into().ok()?))
}

fn le64(image: &[u8], at: u32) -> Option<u64> {
    let at = at as usize;
    Some(u64::from_le_bytes(image.get(at..at + 8)?.try_into().ok()?))
}

fn overlaps(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

fn bind_program(
    input: &BindInputs<'_>,
    p: PositionsIdentity,
    p_generation: u64,
    positions: &[i32; POSITION_ROWS],
    payload_digest: [u8; 32],
) -> Result<IndexedBindWitness, IndexBindError> {
    let program = input.program;
    let closure = |m: String| Err(IndexBindError::Closure(m));
    if input.bindings.len() != program.resources.len()
        || input.allocations.len() != program.resources.len()
        || input.images.len() != program.nodes.len()
        || input.grids.len() != program.nodes.len()
    {
        return closure("bound state does not cover the program".into());
    }
    let pr = input.positions_resource.0 as usize;
    let Some(p_binding) = input.bindings.get(pr) else {
        return closure(format!("positions resource {pr} does not exist"));
    };
    let p_alloc = input.allocations[pr];
    if p_binding.base != p.base || p_binding.bytes != p.bytes || p_alloc.generation != p.generation
    {
        return closure(format!(
            "resource {pr} is {:#x}+{} gen {}, the sealed positions {:#x}+{} gen {}",
            p_binding.base, p_binding.bytes, p_alloc.generation, p.base, p.bytes, p.generation
        ));
    }
    let p_range = (p.base, p.base + p.bytes);

    // ---- full-program no-writer / alias closure for P ----------------------
    // Sole occupant: no other resource of the program overlaps the allocation.
    for (i, b) in input.bindings.iter().enumerate() {
        if i != pr && overlaps((b.base, b.base.saturating_add(b.bytes)), p_range) {
            return closure(format!("resource {i} aliases the positions allocation"));
        }
    }
    let mut conditional = Vec::new();
    for (index, node) in program.nodes.iter().enumerate() {
        if node.kref.is_none() {
            return closure(format!(
                "node {index} ({}) has no certified facts",
                node.symbol
            ));
        }
        if !matches!(node.lowering, Lowering::Pm4) {
            return closure(format!(
                "node {index} ({}) is not PM4-lowered: an unknown-effect node forbids the no-writer proof",
                node.symbol
            ));
        }
        for e in &node.effects {
            let b = input.bindings.get(e.resource.0 as usize).ok_or_else(|| {
                IndexBindError::Closure(format!("node {index}: unknown resource {}", e.resource.0))
            })?;
            let hits_p = e.resource == input.positions_resource
                || overlaps((b.base, b.base.saturating_add(b.bytes)), p_range);
            if hits_p && e.writes() {
                return closure(format!(
                    "node {index} ({}) writes the positions allocation (+{})",
                    node.symbol, e.arg_offset
                ));
            }
        }
        if node.indexed.is_some() {
            conditional.push(index);
        }
    }
    for (i, step) in input.steps.iter().enumerate() {
        let dst = match step {
            WindowStep::Kernel { .. } => continue,
            WindowStep::Copy(c) => c.dst.resource,
            WindowStep::Fill(f) => f.dst.resource,
        };
        let b = input.bindings.get(dst.0 as usize).ok_or_else(|| {
            IndexBindError::Closure(format!("step {i}: unknown resource {}", dst.0))
        })?;
        if dst == input.positions_resource
            || overlaps((b.base, b.base.saturating_add(b.bytes)), p_range)
        {
            return closure(format!("step {i} writes the positions allocation"));
        }
    }

    // ---- per-writer proofs ----------------------------------------------------
    let given: Vec<usize> = input.writers.iter().map(|w| w.node).collect();
    if given != conditional {
        return closure(format!(
            "writer extents cover nodes {given:?}, the conditional nodes are {conditional:?}"
        ));
    }
    let mut writers = Vec::with_capacity(conditional.len());
    for extent in input.writers {
        writers.push(bind_writer(input, extent, p, positions)?);
    }
    // K/V and successive layer writers: pairwise disjoint destination rows.
    for (i, a) in writers.iter().enumerate() {
        for b in &writers[i + 1..] {
            for ra in &a.rows {
                if b.rows.iter().any(|rb| overlaps(*ra, *rb)) {
                    return Err(IndexBindError::writer(
                        b.node,
                        format!("destination rows overlap writer node {}", a.node),
                    ));
                }
            }
        }
    }
    Ok(IndexedBindWitness {
        window: input.window,
        shape_generation: input.shape_generation,
        p_generation,
        payload_digest,
        p,
        positions: *positions,
        writers,
        predicates: vec![
            "positions.host_checked(producer,nonneg,distinct)",
            "positions.sealed_ready(h2d_fence,acquire)",
            "closure.no_writer(all_aliases)",
            "writer.guards(H>0,HD256,B16,slots_null,grid,block)",
            "writer.rows_in_slice_and_mapped(generation)",
            "writer.src_extent_and_no_overlap",
            "writer.partition(unique_rows,grid_heads_lanes)",
        ],
    })
}

fn bind_writer(
    input: &BindInputs<'_>,
    extent: &WriterExtent,
    p: PositionsIdentity,
    positions: &[i32; POSITION_ROWS],
) -> Result<BoundWriter, IndexBindError> {
    let node_index = extent.node;
    let bad = |d: String| Err(IndexBindError::writer(node_index, d));
    let node: &KernelNode = &input.program.nodes[node_index];
    let indexed = node
        .indexed
        .as_ref()
        .expect("conditional nodes carry indexed facts");
    let kref = node.kref.as_ref().expect("closure checked certified facts");
    if !indexed.witnesses.iter().any(|w| w == WITNESS_INDEXED_BIND) {
        return bad("receipt requires a witness kind this bind does not provide".into());
    }
    // Receipt <-> ABI: the actual loaded-object metadata, not source layout.
    let layout: [(u32, u32); 9] = [
        (abi::DST, 8),
        (abi::SRC, 8),
        (abi::POSITIONS, 8),
        (abi::N_KV_HEADS, 4),
        (abi::HEAD_DIM, 4),
        (abi::BATCH, 4),
        (abi::SLOT_DESCS, 8),
        (abi::ROW_SLOT, 8),
        (0, 0),
    ];
    for (off, size) in &layout[..8] {
        match indexed.args.iter().find(|a| a.offset == *off) {
            Some(a) if a.size == *size => {}
            _ => return bad(format!("receipt has no {size}-byte argument at +{off}")),
        }
    }
    let role = |off: u32| {
        indexed
            .args
            .iter()
            .find(|a| a.offset == off)
            .map(|a| &a.role)
    };
    for off in [abi::DST, abi::SRC, abi::POSITIONS] {
        if !matches!(role(off), Some(ArgRole::Pointer { .. })) {
            return bad(format!("argument +{off} is not a pointer in the receipt"));
        }
    }
    // The optional slot tables carry no access in the receipt: they are
    // guarded-null kernargs (`kernarg_eq{offset, size 8, value 0}` or
    // `ptr_null`), never pointer arguments. The guard must be recorded, and
    // the bound bytes are checked to be zero below.
    for off in [abi::SLOT_DESCS, abi::ROW_SLOT] {
        if matches!(role(off), Some(ArgRole::Pointer { classes, .. }) if !classes.is_empty()) {
            return bad(format!("argument +{off} has memory effects in the receipt"));
        }
        let guarded = indexed.guards.iter().any(|g| {
            g.arg_offset == off
                && g.size == 8
                && match g.kind.as_str() {
                    "kernarg_eq" | "scalar_eq" => g.value == "0",
                    "ptr_null" => true,
                    _ => false,
                }
        });
        if !guarded {
            return bad(format!("receipt has no null guard on +{off}"));
        }
    }
    for (off, value) in [(abi::HEAD_DIM, "256"), (abi::BATCH, "16")] {
        let guarded = indexed.guards.iter().any(|g| {
            g.arg_offset == off
                && g.size == 4
                && matches!(g.kind.as_str(), "kernarg_eq" | "scalar_eq")
                && g.value == value
        });
        if !guarded {
            return bad(format!("receipt has no `== {value}` guard on +{off}"));
        }
    }
    for off in [abi::N_KV_HEADS, abi::HEAD_DIM, abi::BATCH] {
        if !matches!(role(off), Some(ArgRole::Integer)) {
            return bad(format!(
                "argument +{off} is not a loaded integer in the receipt"
            ));
        }
    }
    let effect = indexed
        .effects
        .iter()
        .find(|e| e.base_arg_offset == abi::DST)
        .ok_or_else(|| {
            IndexBindError::writer(node_index, "no indexed effect on the dst argument")
        })?;
    if !effect
        .index_loads
        .iter()
        .any(|l| l.source_arg_offset == abi::POSITIONS && l.element == "i32")
    {
        return bad("the dst effect has no i32 index load from the positions argument".into());
    }
    if indexed.effects.iter().any(|e| {
        e.base_arg_offset != abi::DST
            || e.index_loads
                .iter()
                .any(|l| l.source_arg_offset != abi::POSITIONS)
    }) {
        return bad("an indexed effect uses an argument outside the dst/positions case".into());
    }

    let image = &input.images[node_index];
    let grid = input.grids[node_index];
    let h_raw = le32(image, abi::N_KV_HEADS)
        .ok_or_else(|| IndexBindError::writer(node_index, "short kernarg image"))?;
    let hd = le32(image, abi::HEAD_DIM)
        .ok_or_else(|| IndexBindError::writer(node_index, "short kernarg image"))?;
    let batch = le32(image, abi::BATCH)
        .ok_or_else(|| IndexBindError::writer(node_index, "short kernarg image"))?;
    let slot_descs = le64(image, abi::SLOT_DESCS)
        .ok_or_else(|| IndexBindError::writer(node_index, "short kernarg image"))?;
    let row_slot = le64(image, abi::ROW_SLOT)
        .ok_or_else(|| IndexBindError::writer(node_index, "short kernarg image"))?;
    let h = h_raw as i32;
    if h <= 0 {
        return bad(format!("n_kv_heads {h} is not positive"));
    }
    if hd != abi::HEAD_DIM_VALUE {
        return bad(format!("head_dim {hd} is not {}", abi::HEAD_DIM_VALUE));
    }
    if batch != abi::BATCH_VALUE {
        return bad(format!("batch_size {batch} is not {}", abi::BATCH_VALUE));
    }
    if slot_descs != 0 || row_slot != 0 {
        return bad("slot_descs/row_slot are not null: the guarded case does not hold".into());
    }
    let h64 = h as u64;
    if grid != [h as u32, abi::BATCH_VALUE, 1] {
        return bad(format!("grid {grid:?} is not [{h}, 16, 1]"));
    }
    if node.block != abi::BLOCK {
        return bad(format!("block {:?} is not {:?}", node.block, abi::BLOCK));
    }
    if node.dynamic_lds != 0 {
        return bad("dynamic LDS is nonzero".into());
    }
    // Signed-32 products the kernel computes, then R in checked u64.
    let row_bytes = h64
        .checked_mul(u64::from(hd) + 2)
        .ok_or_else(|| IndexBindError::writer(node_index, "R = H*258 overflows"))?;
    let src_elems = h64 * u64::from(hd) * u64::from(batch);
    if row_bytes > i32::MAX as u64 || src_elems > i32::MAX as u64 {
        return bad(format!(
            "R={row_bytes} or {src_elems} source elements exceed signed 32 bits"
        ));
    }
    let src_bytes = src_elems * 4;

    // Pointers from the bound image, each inside its program resource.
    let resource_of = |off: u32| -> Result<(ResourceId, u64, Binding), IndexBindError> {
        let value = le64(image, off)
            .ok_or_else(|| IndexBindError::writer(node_index, "short kernarg image"))?;
        let (_, arg) = node
            .args
            .iter()
            .find(|(o, a)| *o == off && matches!(a, Arg::Resource { .. }))
            .ok_or_else(|| {
                IndexBindError::writer(node_index, format!("+{off} is not a bound resource"))
            })?;
        let Arg::Resource { resource, .. } = arg else {
            unreachable!()
        };
        let b = *input
            .bindings
            .get(resource.0 as usize)
            .ok_or_else(|| IndexBindError::writer(node_index, "unknown resource"))?;
        if value < b.base || value - b.base >= b.bytes {
            return Err(IndexBindError::writer(
                node_index,
                format!(
                    "+{off} = {value:#x} lies outside its allocation {:#x}+{}",
                    b.base, b.bytes
                ),
            ));
        }
        Ok((*resource, value, b))
    };
    let (p_res, p_ptr, _) = resource_of(abi::POSITIONS)?;
    if p_res != input.positions_resource || p_ptr != p.base {
        return bad("positions argument does not point at the sealed positions' base".into());
    }
    let (dst_res, dst_ptr, dst_binding) = resource_of(abi::DST)?;
    let (src_res, src_ptr, src_binding) = resource_of(abi::SRC)?;
    if dst_res == input.positions_resource || src_res == dst_res {
        return bad("dst aliases the positions or the source resource".into());
    }
    if src_res == input.positions_resource {
        return bad("the source is the sealed positions".into());
    }
    let dst_alloc = input.allocations[dst_res.0 as usize];
    if dst_alloc.base != dst_binding.base || dst_alloc.bytes != dst_binding.bytes {
        return bad("dst allocation identity differs from its binding".into());
    }
    // Source extent: 16*H*256*4 bytes.
    let src_end = src_ptr
        .checked_add(src_bytes)
        .ok_or_else(|| IndexBindError::writer(node_index, "source extent overflows"))?;
    if src_end > src_binding.base + src_binding.bytes {
        return bad(format!(
            "source [{src_ptr:#x}, {src_end:#x}) exceeds its allocation {:#x}+{}",
            src_binding.base, src_binding.bytes
        ));
    }
    if src_ptr % 4 != 0 {
        return bad("source is not 4-byte aligned".into());
    }

    // dst slice, mapping generation and mapped/writable intervals.
    if extent.mapping_generation != dst_alloc.generation {
        return bad(format!(
            "mapping generation {} differs from the bound allocation's {}",
            extent.mapping_generation, dst_alloc.generation
        ));
    }
    // The intended tensor slice may extend past the mapped extent of a VMM
    // reservation (reserved address space is not mapped memory); mapped
    // coverage below is what makes a row writable.
    if extent.dst_slice.start >= extent.dst_slice.end {
        return bad(format!("tensor slice {:?} is empty", extent.dst_slice));
    }
    let (Some(slice_lo), Some(slice_hi)) = (
        dst_binding.base.checked_add(extent.dst_slice.start),
        dst_binding.base.checked_add(extent.dst_slice.end),
    ) else {
        return bad("tensor slice overflows the address space".into());
    };
    let slice = (slice_lo, slice_hi);
    let mut mapped: Vec<(u64, u64)> = Vec::new();
    for r in &extent.mapped {
        if r.start >= r.end || r.end > dst_binding.bytes {
            return bad(format!(
                "mapped interval {r:?} is empty or outside the allocation"
            ));
        }
        mapped.push((dst_binding.base + r.start, dst_binding.base + r.end));
    }
    mapped.sort_unstable();
    let covered = |lo: u64, hi: u64| -> bool {
        // The sorted intervals must tile [lo, hi) without a hole.
        let mut at = lo;
        for (s, e) in &mapped {
            if *e <= at {
                continue;
            }
            if *s > at {
                return false;
            }
            at = *e;
            if at >= hi {
                return true;
            }
        }
        at >= hi
    };

    let mut rows = Vec::with_capacity(POSITION_ROWS);
    for (bid, p_i) in positions.iter().enumerate() {
        if *p_i < 0 {
            return bad(format!("position {bid} is negative"));
        }
        let off = (*p_i as u64)
            .checked_mul(row_bytes)
            .ok_or_else(|| IndexBindError::writer(node_index, "P*R overflows"))?;
        let start = dst_ptr
            .checked_add(off)
            .ok_or_else(|| IndexBindError::writer(node_index, "row address overflows"))?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| IndexBindError::writer(node_index, "row end overflows"))?;
        if start < slice.0 || end > slice.1 {
            return bad(format!(
                "row {bid} (position {p_i}) [{start:#x}, {end:#x}) leaves the tensor slice [{:#x}, {:#x})",
                slice.0, slice.1
            ));
        }
        if !covered(start, end) {
            return bad(format!(
                "row {bid} (position {p_i}) [{start:#x}, {end:#x}) is not fully inside mapped/writable intervals"
            ));
        }
        if start % 4 != 0 {
            return bad(format!(
                "row {bid} start {start:#x} breaks 4-byte code-store alignment"
            ));
        }
        // Scale store alignment: `H*256 + head*2` after a 4-aligned start.
        if overlaps((start, end), (p.base, p.base + p.bytes)) {
            return bad(format!("row {bid} overlaps the sealed positions"));
        }
        if overlaps((start, end), (src_ptr, src_end)) {
            return bad(format!("row {bid} overlaps the source"));
        }
        rows.push((start, end));
    }
    // Uniqueness: rows are pairwise disjoint (re-proved on the bound rows).
    let mut sorted = rows.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0].1 > w[1].0) {
        return bad("destination rows overlap".into());
    }
    Ok(BoundWriter {
        node: node_index,
        symbol: node.symbol.clone(),
        artifact_sha256: kref.artifact_sha256.clone(),
        receipt_sha256: kref.receipt_sha256.clone(),
        case_digest: indexed.case_digest.clone(),
        n_kv_heads: h as u32,
        row_bytes,
        rows,
        dst: dst_alloc,
        dst_mapping_generation: extent.mapping_generation,
        kernarg_digest: digest(image),
        grid,
    })
}

#[cfg(test)]
pub(crate) mod tests;
