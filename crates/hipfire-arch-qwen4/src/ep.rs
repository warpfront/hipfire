// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next expert-parallel (EP) mesh: one persistent worker thread per peer
//! rank.
//!
//! Rank 0 is the [`Qwen4Bundle`] and [`Gpu`] the daemon already drives, on the
//! caller's thread. Ranks `1..N` are peer bundles owned by worker threads that
//! construct them (a `Gpu` never crosses threads). Every daemon-facing bundle
//! call is replicated: the caller runs rank 0 itself while the workers run the
//! same call on their own bundle, then every host result is compared.
//!
//! The thread/latch/poison machinery ([`RankPool`]) is generic over the peer
//! state so it is testable without a device. [`Qwen4EpMesh`] is the Qwen4
//! instantiation: peers are [`Qwen4EpPeer`] and a failure on any rank aborts
//! the row-gather barrier so no rank is left waiting on a dead peer. The mesh is
//! rank-generic over EP2..=EP8 (`attach_ep_mesh` admits `1 + peers` of 2..=8).

use crate::bundle::{BundleError, Qwen4Bundle};
use crate::ep_reserve::EpReserveMeshReceipt;
use crate::ep_trunk_split::{
    agree_layouts, split_buffer_bytes, split_census, SplitWeight, TrunkSplitMode, TrunkSplitRank,
};
use crate::gpu_forward::{
    Qwen4BalancedPlan, Qwen4PeerExpertSource, QWEN4_EP_GATHER_BUFFERS, QWEN4_EP_GATHER_MAX_ROWS,
};
use crate::weights::Qwen4EpPlacement;
use hipfire_dispatch::families::moe::{MoeRowGather, MoeTileExchange};
use hipfire_dispatch::pipeline::projection_split::split_scope;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::ep_row_gather::{EpRowGather, EpRowGatherSpec, EpSlotOwners};
use hipfire_runtime::ep_tile_exchange::{EpTileExchange, EpTileExchangeSpec};
use hipfire_runtime::rank_pool::{agree, lock, RankPool};
use hipfire_runtime::sampler::PenaltyTable;
use hipfire_runtime::serve_contract::{
    sha256_len_prefixed, ArchPolicy, CacheDomain, DeviceTopology, KvLayout, SharingNamespace,
    TemplateIdentity, TokenizerIdentity,
};
use hipfire_runtime::session_cache::{SessionCache, SessionRoute};
use rdna_compute::moe::{Qwen4EpOwnerTables, QWEN4_EP_OWNER_EXPERTS};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// ─────────────────────────────── tensor mirrors ───────────────────────────────

/// Shape and dtype of a rank-0 caller-owned device tensor; peers allocate a
/// same-shaped stand-in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorSpec {
    pub fn of(tensor: &GpuTensor) -> Self {
        Self {
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MirrorRole {
    Logits,
    Top1,
}

/// A peer's stand-ins for the rank-0 caller-owned device tensors of a call
/// (logits, top1): allocated lazily, cached by role + shape + dtype, freed at
/// teardown. Nobody reads a peer's mirror in serving; its logits equal rank
/// 0's bitwise.
#[derive(Default)]
pub struct PeerMirrors {
    entries: Vec<(MirrorRole, TensorSpec, GpuTensor)>,
    last_logits: Option<usize>,
    last_top1: Option<usize>,
    /// This peer's replica of rank 0's native MTP drafter, built on the first
    /// drafter fan-out ([`Qwen4Bundle::ep_drafter_fanout`]) and freed with the
    /// mirrors.
    drafter: Option<crate::mtp_spec::Qwen4MtpDrafter>,
}

impl PeerMirrors {
    /// This peer's drafter replica, built from `params` (rank 0's construction
    /// parameters) on first use.
    pub(crate) fn drafter(
        &mut self,
        params: crate::mtp_spec::ReplicaParams,
    ) -> &mut crate::mtp_spec::Qwen4MtpDrafter {
        self.drafter
            .get_or_insert_with(|| crate::mtp_spec::Qwen4MtpDrafter::new_replica(params))
    }

    fn ensure(
        &mut self,
        gpu: &mut Gpu,
        role: MirrorRole,
        spec: &TensorSpec,
    ) -> Result<usize, String> {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(r, s, _)| *r == role && s == spec)
        {
            return Ok(index);
        }
        let tensor = gpu
            .zeros(&spec.shape, spec.dtype)
            .map_err(|error| format!("qwen4 EP mirror {role:?} {spec:?}: {error:?}"))?;
        self.entries.push((role, spec.clone(), tensor));
        Ok(self.entries.len() - 1)
    }

    /// The logits mirror for `spec` (allocated on first use) and, when the
    /// caller passed one, the top1 mirror.
    pub(crate) fn pair(
        &mut self,
        gpu: &mut Gpu,
        logits: &TensorSpec,
        top1: Option<&TensorSpec>,
    ) -> Result<(&GpuTensor, Option<&GpuTensor>), String> {
        let logits_index = self.ensure(gpu, MirrorRole::Logits, logits)?;
        self.last_logits = Some(logits_index);
        let top1_index = match top1 {
            Some(spec) => Some(self.ensure(gpu, MirrorRole::Top1, spec)?),
            None => None,
        };
        self.last_top1 = top1_index;
        Ok((
            &self.entries[logits_index].2,
            top1_index.map(|index| &self.entries[index].2),
        ))
    }

    /// The logits mirror of the most recent logits-taking call.
    pub fn logits(&self) -> Option<&GpuTensor> {
        self.last_logits.map(|index| &self.entries[index].2)
    }

    /// The top1 mirror of the most recent call that passed a top1 tensor.
    pub fn top1(&self) -> Option<&GpuTensor> {
        self.last_top1.map(|index| &self.entries[index].2)
    }

    fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        if let Some(drafter) = self.drafter {
            <crate::mtp_spec::Qwen4MtpDrafter as hipfire_runtime::spec::MtpDrafter>::mtp_free(
                Box::new(drafter),
                gpu,
            );
        }
        let mut first = None;
        for (_, _, tensor) in self.entries {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(format!("free mirror: {error:?}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

// ─────────────────────────────── Qwen4 mesh ───────────────────────────────

/// Loads one peer rank on the worker's own thread: rank → the peer's `Gpu`
/// (on the seal's physical device for the rank) and a fully attached bundle
/// bound to rank 0's seal at that rank (`Qwen4Bundle::bind_ep_seal`).
pub type PeerLoadFn = dyn Fn(usize) -> Result<(Gpu, Qwen4Bundle), String> + Send + Sync;

/// What a peer worker owns.
pub struct Qwen4EpPeer {
    rank: usize,
    gpu: Gpu,
    bundle: Qwen4Bundle,
    mirrors: PeerMirrors,
    gather: Option<Arc<EpRowGather>>,
    tile: Option<Arc<EpTileExchange>>,
    split: Option<TrunkSplitRank>,
}

impl Qwen4EpPeer {
    /// Drain the device, then free everything this rank owns.
    fn teardown(mut self) -> Result<(), String> {
        let mut first: Option<String> = None;
        let mut note = |result: Result<(), String>| {
            if let Err(error) = result {
                first.get_or_insert(format!("rank {}: {error}", self.rank));
            }
        };
        note(sync_device(&mut self.gpu));
        note(self.mirrors.free(&mut self.gpu));
        if let Some(gather) = self.gather.take() {
            note(
                gather
                    .release(self.rank, &mut self.gpu)
                    .map_err(|error| format!("release gather events: {error}")),
            );
        }
        if let Some(tile) = self.tile.take() {
            note(
                tile.release(self.rank, &mut self.gpu)
                    .map_err(|error| format!("release tile exchange events: {error}")),
            );
        }
        if let Some(split) = self.split.take() {
            note(split.release(&mut self.gpu));
        }
        note(
            self.bundle
                .free_gpu(&mut self.gpu)
                .map_err(|error| format!("free bundle: {error}")),
        );
        first.map_or(Ok(()), Err)
    }
}

fn sync_device(gpu: &mut Gpu) -> Result<(), String> {
    gpu.bind_thread().map_err(|error| error.to_string())?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("device synchronize: {error}"))
}

/// `last_buffer` before any invocation.
const NO_BUFFER: usize = usize::MAX;

/// The row-gather hook one rank installs into its forward
/// (`Qwen4GpuForward::attach_ep`).
pub(crate) struct Qwen4EpRowGatherHook {
    rank: usize,
    trunk_layers: usize,
    gather: Arc<EpRowGather>,
    /// Device addresses of this rank's gather buffers, to check the expanded
    /// rows the stage hands over are the buffer it exchanges.
    buffer_ptrs: [usize; QWEN4_EP_GATHER_BUFFERS],
    /// Buffer index of this rank's previous invocation. The transport's hazard
    /// rule forbids the same buffer on two consecutive invocations.
    last_buffer: AtomicUsize,
    /// Balanced-ownership layers (default on under EP; `HIPFIRE_EP_BALANCED_OWNERS=0`
    /// disables), installed once after the forward attached its live tables.
    /// Never set when disabled.
    balanced: OnceLock<BalancedOwners>,
}

/// What the hook knows of the rank's balanced trunk layers.
struct BalancedOwners {
    /// Owner-assign table addresses indexed by trunk layer label.
    tables: Vec<Option<Qwen4EpOwnerTables>>,
    /// Non-owning view of the rank's slot-owner buffer (the forward owns it).
    slot_owner: Mutex<GpuTensor>,
}

/// The owner-assign tables of a balanced trunk layer; `None` for every other
/// layer, including the MTP head range (`>= trunk_layers`), whatever
/// `tables` holds.
fn balanced_layer_tables(
    tables: &[Option<Qwen4EpOwnerTables>],
    layer_idx: u16,
    trunk_layers: usize,
) -> Option<&Qwen4EpOwnerTables> {
    let layer = usize::from(layer_idx);
    if layer >= trunk_layers {
        return None;
    }
    tables.get(layer).and_then(Option::as_ref)
}

impl BalancedOwners {
    fn layer(&self, layer_idx: u16, trunk_layers: usize) -> Option<&Qwen4EpOwnerTables> {
        balanced_layer_tables(&self.tables, layer_idx, trunk_layers)
    }
}

impl Qwen4EpRowGatherHook {
    /// The static gather reads owner `e % ranks` in-kernel
    /// (`EpSlotOwners::ExpertModulo`); `seal`'s ownership must be exactly that
    /// placement over `QWEN4_EP_OWNER_EXPERTS` experts, and `rank` its rank.
    fn new(
        rank: usize,
        seal: &crate::ep_seal::Qwen4EpRankSeal,
        trunk_layers: usize,
        gather: Arc<EpRowGather>,
        buffer_ptrs: [usize; QWEN4_EP_GATHER_BUFFERS],
    ) -> Result<Self, String> {
        let owners = seal.seal.owner_ranks();
        let ranks = seal.seal.ranks();
        if rank != seal.rank
            || owners.len() != QWEN4_EP_OWNER_EXPERTS
            || owners
                .iter()
                .enumerate()
                .any(|(expert, &owner)| owner != expert % ranks)
        {
            return Err(format!(
                "qwen4 EP row gather: rank {rank} (sealed rank {}) ownership over {} experts is \
                 not the expert-modulo placement the gather kernel reads",
                seal.rank,
                owners.len()
            ));
        }
        Ok(Self {
            rank,
            trunk_layers,
            gather,
            buffer_ptrs,
            last_buffer: AtomicUsize::new(NO_BUFFER),
            balanced: OnceLock::new(),
        })
    }

    /// Install the rank's balanced layers (once, after `attach_ep`).
    fn install_balanced(&self, plan: Qwen4BalancedPlan) -> Result<(), String> {
        let mut tables: Vec<Option<Qwen4EpOwnerTables>> = vec![None; self.trunk_layers];
        for (label, layer_tables) in plan.layers {
            let slot = tables.get_mut(label).ok_or_else(|| {
                format!(
                    "qwen4 EP rank {}: balanced layer {label} is outside the {} trunk layers",
                    self.rank, self.trunk_layers
                )
            })?;
            *slot = Some(layer_tables);
        }
        self.balanced
            .set(BalancedOwners {
                tables,
                slot_owner: Mutex::new(plan.slot_owner),
            })
            .map_err(|_| {
                format!(
                    "qwen4 EP rank {}: balanced owners are already installed",
                    self.rank
                )
            })
    }
}

/// Record `buffer` as the rank's current invocation in `last`; `Err` when the
/// previous invocation used the same one.
fn note_gather_buffer(last: &AtomicUsize, rank: usize, buffer: usize) -> Result<(), String> {
    if last.swap(buffer, Ordering::Relaxed) == buffer {
        return Err(format!(
            "qwen4 EP row gather: gather buffer {buffer} of rank {rank} used by two consecutive invocations"
        ));
    }
    Ok(())
}

impl MoeRowGather for Qwen4EpRowGatherHook {
    fn gather_rows(
        &self,
        gpu: &mut Gpu,
        layer_idx: u16,
        slots: usize,
        down_expanded: &GpuTensor,
        topk_indices: &GpuTensor,
    ) -> Result<(), String> {
        // Singleton forwards select by layer parity; lane steps select by
        // invocation ordinal because a layer can invoke MoE more than once.
        // The expanded-row allocation identifies the actual transport buffer.
        let ptr = down_expanded.buf.as_ptr() as usize;
        let buffer = self
            .buffer_ptrs
            .iter()
            .position(|&buffer_ptr| buffer_ptr == ptr)
            .ok_or_else(|| {
                format!(
                    "qwen4 EP row gather: layer {layer_idx} expanded rows at {ptr:#x} are not a gather buffer of rank {}",
                    self.rank
                )
            })?;
        note_gather_buffer(&self.last_buffer, self.rank, buffer)?;
        // A balanced layer gathers from the owners its own assignment wrote
        // this call; every other layer keeps the static `e % ranks` placement.
        let balanced = self
            .balanced
            .get()
            .filter(|balanced| balanced.layer(layer_idx, self.trunk_layers).is_some());
        let layer = usize::from(layer_idx);
        let result = match balanced {
            Some(balanced) => {
                let owners = lock(&balanced.slot_owner);
                self.gather.exchange_layer(
                    gpu,
                    self.rank,
                    buffer,
                    EpSlotOwners::Explicit(&owners),
                    slots,
                    layer,
                )
            }
            None => self.gather.exchange_layer(
                gpu,
                self.rank,
                buffer,
                EpSlotOwners::ExpertModulo(topk_indices, QWEN4_EP_OWNER_EXPERTS),
                slots,
                layer,
            ),
        };
        result.map_err(|error| {
            format!(
                "qwen4 EP row gather layer {layer_idx} rank {}: {error}",
                self.rank
            )
        })
    }

    fn assigns_owners(&self, layer_idx: u16) -> bool {
        self.balanced
            .get()
            .is_some_and(|balanced| balanced.layer(layer_idx, self.trunk_layers).is_some())
    }

    fn assign_owners(
        &self,
        gpu: &mut Gpu,
        layer_idx: u16,
        slots: usize,
        topk_indices: &GpuTensor,
    ) -> Result<(), String> {
        let tables = self
            .balanced
            .get()
            .and_then(|balanced| balanced.layer(layer_idx, self.trunk_layers))
            .ok_or_else(|| {
                format!(
                    "qwen4 EP rank {}: layer {layer_idx} is not a balanced trunk layer",
                    self.rank
                )
            })?;
        gpu.qwen4_ep_owner_assign(
            topk_indices,
            slots,
            self.gather.spec().ranks,
            self.rank,
            tables,
        )
        .map_err(|error| {
            format!(
                "qwen4 EP owner assignment layer {layer_idx} rank {}: {error}",
                self.rank
            )
        })
    }
}

#[cfg(test)]
mod gather_hook_tests {
    use super::*;

    #[test]
    fn consecutive_gathers_never_reuse_a_buffer() {
        let last = AtomicUsize::new(NO_BUFFER);
        // Trunk parity, then the head's two buffers interleaved with the trunk.
        for buffer in [0, 1, 0, 1, 2, 0, 3, 1, 2, 3, 0] {
            note_gather_buffer(&last, 1, buffer).unwrap();
        }
        let error = note_gather_buffer(&last, 1, 0).unwrap_err();
        assert!(
            error.contains("buffer 0") && error.contains("rank 1"),
            "{error}"
        );
        // The same buffer twice in a row is refused wherever it occurs.
        assert!(note_gather_buffer(&last, 1, 2).is_ok());
        assert!(note_gather_buffer(&last, 1, 2).is_err());
    }

    #[test]
    fn only_balanced_trunk_layers_assign_owners_never_the_mtp_head() {
        let tables = Qwen4EpOwnerTables {
            full_gate_up: 0x1000,
            full_down: 0x2000,
            live_gate_up: 0x3000,
            live_down: 0x4000,
            slot_owner: 0x5000,
            dummy_gate_up: 0x6000,
            dummy_down: 0x7000,
        };
        // Four trunk layers, two of them balanced; the table vector is longer
        // than the trunk to prove the head range is excluded by the guard.
        let by_layer = [None, Some(tables), None, Some(tables), Some(tables)];
        let trunk_layers = 4;
        assert_eq!(
            balanced_layer_tables(&by_layer, 1, trunk_layers),
            Some(&tables)
        );
        assert_eq!(
            balanced_layer_tables(&by_layer, 3, trunk_layers),
            Some(&tables)
        );
        assert_eq!(balanced_layer_tables(&by_layer, 0, trunk_layers), None);
        assert_eq!(balanced_layer_tables(&by_layer, 2, trunk_layers), None);
        // The MTP head's layer indices start at the trunk layer count.
        assert_eq!(balanced_layer_tables(&by_layer, 4, trunk_layers), None);
        assert_eq!(balanced_layer_tables(&by_layer, 5, trunk_layers), None);
        assert_eq!(balanced_layer_tables(&[], 0, trunk_layers), None);
    }
}

/// The grouped-row capacity (padded tiles) a rank registered for the owner-routed chunk
/// MoE, `None` when the rank did not admit it.
fn tile_capacity(bundle: &Qwen4Bundle) -> Option<usize> {
    bundle
        .execution
        .as_ref()
        .and_then(|forward| forward.ep_tile_registration())
        .map(|(_, tiles)| tiles)
}

/// A rank's per-layer symmetric verdicts (`attach_ep`), `None` when the lever was not
/// admitted on this rank.
fn tile_verdicts(bundle: &Qwen4Bundle) -> Option<Vec<bool>> {
    bundle
        .execution
        .as_ref()
        .and_then(|forward| forward.ep_tile_verdicts())
}

/// The per-layer verdict vector every rank agreed on: `Some` only when every rank admitted
/// the lever, the vectors are identical, non-empty and every layer is symmetric. Any `None`,
/// mismatch or false layer refuses the lever for the whole mesh.
fn agreed_tile_verdicts(verdicts: &[Option<Vec<bool>>]) -> Option<&Vec<bool>> {
    let first = verdicts.first()?.as_ref()?;
    (!first.is_empty()
        && first.iter().all(|&symmetric| symmetric)
        && verdicts.iter().all(|layers| layers.as_ref() == Some(first)))
    .then_some(first)
}

/// The owner-routed chunk exchange one rank installs into its forward
/// (`Qwen4GpuForward::install_tile_exchange`): the shared transport, this rank and a
/// non-owning view of the rank's owner table (`e % ranks`).
pub(crate) struct Qwen4EpTileExchangeHook {
    rank: usize,
    tile: Arc<EpTileExchange>,
    owner_table: GpuTensor,
    experts: usize,
}

impl Qwen4EpTileExchangeHook {
    pub(crate) fn new(
        rank: usize,
        tile: Arc<EpTileExchange>,
        owner_table: GpuTensor,
        experts: usize,
    ) -> Self {
        Self {
            rank,
            tile,
            owner_table,
            experts,
        }
    }
}

// SAFETY: `owner_table` is a `Borrowed` alias of a device buffer the forward owns until
// `free_gpu` (read-only after upload: the hook only hands its device address to launches on
// the calling rank's own stream) and the hook never outlives that forward; HIP device
// pointers carry no thread affinity.
unsafe impl Send for Qwen4EpTileExchangeHook {}
unsafe impl Sync for Qwen4EpTileExchangeHook {}

impl MoeTileExchange for Qwen4EpTileExchangeHook {
    fn owners(&self) -> (usize, &GpuTensor, usize) {
        (self.rank, &self.owner_table, self.experts)
    }

    fn before_down(&self, gpu: &mut Gpu) -> Result<(), String> {
        self.tile
            .before_down(gpu, self.rank)
            .map_err(|error| error.to_string())
    }

    fn host_barrier(&self, what: u8, epoch: u32) -> Result<(), String> {
        self.tile
            .host_barrier(self.rank, what, epoch)
            .map_err(|error| error.to_string())
    }

    fn exchange(
        &self,
        gpu: &mut Gpu,
        layer_idx: u16,
        tile_ids: &GpuTensor,
        y_down_grouped: &GpuTensor,
        tiles: usize,
    ) -> Result<(), String> {
        self.tile
            .exchange_layer(
                gpu,
                self.rank,
                &self.owner_table,
                tile_ids,
                y_down_grouped,
                tiles,
                usize::from(layer_idx),
            )
            .map_err(|error| error.to_string())
    }
}

/// `HIPFIRE_EP_BOUNDARY` (developer, #863): `kernel` crosses every row gather
/// on a device boundary (`EpRowGatherBuilder::kernel_boundaries`) instead of
/// producer events and the host barrier; unset or `host` keeps the host
/// barrier.
pub const QWEN4_EP_BOUNDARY_ENV: &str = "HIPFIRE_EP_BOUNDARY";

fn qwen4_ep_kernel_boundaries() -> Result<bool, String> {
    match hipfire_config::developer_var(QWEN4_EP_BOUNDARY_ENV)
        .ok()
        .as_deref()
    {
        None | Some("" | "host") => Ok(false),
        Some("kernel") => Ok(true),
        Some(other) => Err(format!(
            "{QWEN4_EP_BOUNDARY_ENV}={other:?}: expected `host` or `kernel`"
        )),
    }
}

/// The attached mesh, owned by rank 0's bundle (`Qwen4Bundle::ep`).
pub struct Qwen4EpMesh {
    pool: RankPool<Qwen4EpPeer>,
    gather: Arc<EpRowGather>,
    /// Owner-routed chunk MoE transport (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`); `None`
    /// when the lever is off or was not admitted on every rank.
    tile: Option<Arc<EpTileExchange>>,
    /// Rank 0's trunk output split (`HIPFIRE_EP_TRUNK_SPLIT`); `None` when off.
    split: Option<TrunkSplitRank>,
    ranks: usize,
    /// The EP reserve-sharing plan receipt every rank agreed on at attach;
    /// `None` on an ordinary (non-reserve-sharing) load.
    reserve: Option<EpReserveMeshReceipt>,
    device_ids: Vec<i32>,
    used_bytes: Vec<u64>,
}

type PeerJob<'a, R> =
    &'a (dyn Fn(&mut Qwen4Bundle, &mut Gpu, &mut PeerMirrors) -> Result<R, String> + Sync);

impl Qwen4EpMesh {
    /// Ranks including rank 0.
    pub fn ranks(&self) -> usize {
        self.ranks
    }

    /// HIP device id of every rank (rank 0 first).
    pub fn device_ids(&self) -> &[i32] {
        &self.device_ids
    }

    /// The agreed EP reserve-sharing receipt (`None`: ordinary load). A mesh
    /// with a receipt refuses dynamic lane staging.
    pub fn reserve_receipt(&self) -> Option<&EpReserveMeshReceipt> {
        self.reserve.as_ref()
    }

    /// Device memory in use on every rank's GPU when the mesh was attached
    /// (rank 0 first): resident weights, state, scratch, expert tables.
    pub fn used_bytes(&self) -> &[u64] {
        &self.used_bytes
    }

    /// Whether an earlier failure poisoned the mesh.
    pub fn poisoned(&self) -> Option<String> {
        self.pool.poisoned()
    }

    /// Poison the mesh after the ranks diverged on a result that must agree.
    pub(crate) fn poison(&self, message: &str) {
        self.pool.poison_mesh(message);
    }

    /// [`Self::run`] that also runs on a poisoned mesh (rollback and teardown
    /// drains).
    pub(crate) fn run_forced<R: Send>(
        &self,
        local: impl FnOnce() -> Result<R, String>,
        job: PeerJob<'_, R>,
    ) -> Result<Vec<R>, String> {
        let local_split = self.split.as_ref().and_then(TrunkSplitRank::hook);
        self.pool.run_all_forced(
            || {
                let _scope = local_split.map(|hook| split_scope(hook));
                local()
            },
            &|_, peer: &mut Qwen4EpPeer| {
                let hook = peer.split.as_ref().and_then(TrunkSplitRank::hook);
                let _scope = hook.map(|hook| split_scope(hook));
                job(&mut peer.bundle, &mut peer.gpu, &mut peer.mirrors)
            },
        )
    }

    /// Run `job` on every peer while `local` runs rank 0 on the caller's
    /// thread. Results in rank order. Any rank's error or panic aborts the
    /// row-gather barrier, poisons the mesh, and is returned (the first one).
    /// Every rank's trunk split hook is installed for the call. With kernel
    /// boundaries the ranks' boundary ledgers must then match
    /// (`EpRowGather::check_step`); a refusal poisons the mesh.
    pub(crate) fn run<R: Send>(
        &self,
        local: impl FnOnce() -> Result<R, String>,
        job: PeerJob<'_, R>,
    ) -> Result<Vec<R>, String> {
        let local_split = self.split.as_ref().and_then(TrunkSplitRank::hook);
        let values = self.pool.run_all(
            || {
                let _scope = local_split.map(|hook| split_scope(hook));
                local()
            },
            &|_, peer: &mut Qwen4EpPeer| {
                let hook = peer.split.as_ref().and_then(TrunkSplitRank::hook);
                let _scope = hook.map(|hook| split_scope(hook));
                job(&mut peer.bundle, &mut peer.gpu, &mut peer.mirrors)
            },
        )?;
        if let Err(error) = self.gather.check_step() {
            let message = error.to_string();
            self.pool.poison_mesh(&message);
            return Err(message);
        }
        Ok(values)
    }

    /// Run `job` on peer `rank` alone (rank 0 is not a peer).
    pub(crate) fn run_peer<R: Send>(&self, rank: usize, job: PeerJob<'_, R>) -> Result<R, String> {
        self.pool.run_rank(rank, &|_, peer: &mut Qwen4EpPeer| {
            job(&mut peer.bundle, &mut peer.gpu, &mut peer.mirrors)
        })
    }

    /// Drain every rank, stop the workers (each frees its own bundle on its
    /// thread) and release rank 0's gather events. Always attempts every step.
    pub(crate) fn close(self, gpu: &mut Gpu) -> Result<(), String> {
        let Self {
            pool,
            gather,
            tile,
            split,
            ..
        } = self;
        let mut first: Option<String> = None;
        let drained = pool.run_all_forced(|| sync_device(gpu), &|_, peer: &mut Qwen4EpPeer| {
            sync_device(&mut peer.gpu)
        });
        if let Err(error) = drained {
            first.get_or_insert(error);
        }
        if let Err(error) = pool.close() {
            first.get_or_insert(error);
        }
        if let Err(error) = gather.release(0, gpu) {
            first.get_or_insert(format!("rank 0: release gather events: {error}"));
        }
        if let Some(tile) = tile {
            if let Err(error) = tile.release(0, gpu) {
                first.get_or_insert(format!("rank 0: release tile exchange events: {error}"));
            }
        }
        if let Some(split) = split {
            if let Err(error) = split.release(gpu) {
                first.get_or_insert(format!("rank 0: {error}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

pub(crate) fn ep_err(message: impl Into<String>) -> BundleError {
    BundleError::Forward(format!("qwen4 EP: {}", message.into()))
}

impl Qwen4Bundle {
    /// Number of ranks serving this bundle: 1 without EP.
    pub fn ranks(&self) -> usize {
        self.ep.as_ref().map_or(1, |mesh| mesh.ranks())
    }

    /// The attached EP mesh, if any.
    pub fn ep_mesh(&self) -> Option<&Qwen4EpMesh> {
        self.ep.as_deref()
    }

    /// Run `f` on rank `rank`'s bundle and `Gpu`: rank 0 is `self` with the
    /// caller's `gpu` (the mesh is detached for the call, so calls on the
    /// bundle inside `f` do not fan out); a peer runs on its own worker thread
    /// and the call blocks until `f` returns. `f` also gets that rank's
    /// [`PeerMirrors`] (always empty on rank 0).
    pub fn with_rank_mut<R: Send>(
        &mut self,
        gpu: &mut Gpu,
        rank: usize,
        f: impl FnOnce(&mut Qwen4Bundle, &mut Gpu, &PeerMirrors) -> Result<R, String> + Send,
    ) -> Result<R, String> {
        let ranks = self.ranks();
        if rank >= ranks {
            return Err(format!(
                "qwen4 EP rank {rank} out of range (ranks = {ranks})"
            ));
        }
        if rank == 0 {
            let mesh = self.ep.take();
            let result = f(self, gpu, &PeerMirrors::default());
            self.ep = mesh;
            return result;
        }
        let mesh = self
            .ep
            .as_ref()
            .ok_or_else(|| "qwen4 EP mesh is not attached".to_string())?;
        let cell = Mutex::new(Some(f));
        mesh.run_peer(rank, &|bundle, peer_gpu, mirrors| {
            let f = lock(&cell)
                .take()
                .ok_or_else(|| "qwen4 EP: rank job ran twice".to_string())?;
            f(bundle, peer_gpu, mirrors)
        })
    }

    /// Replicate the loaded single-rank bundle into an EP mesh.
    ///
    /// `self` is rank 0 on `gpu`: loaded with its forward attached and bound
    /// to its load's seal ([`Qwen4Bundle::bind_ep_seal`]), from which the
    /// rank count and every rank's physical device come. `loader` builds one
    /// peer on its worker thread (hipfire-loader supplies it). Sequence:
    ///
    /// 1. every peer loads on its own thread; it must be bound to rank 0's
    ///    seal at its own rank on the seal's device, and its config and EP
    ///    placement are checked against rank 0;
    /// 2. each rank registers its gather buffers with one
    ///    [`EpRowGatherBuilder`](hipfire_runtime::ep_row_gather::EpRowGatherBuilder)
    ///    (on its own thread), then the transport is built;
    /// 3. every rank enables P2P (after every load/allocation exists);
    /// 4. every rank reports its compact expert sources;
    /// 5. every rank builds its full (P2P) tables from all sources and installs
    ///    the row-gather hook (`Qwen4GpuForward::attach_ep`);
    /// 5b. with `HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`, every rank's tile-exchange
    ///    registration and symmetric verdicts are collected (`run_all`, every rank, any
    ///    admitted rank count); the exchange is installed only if all ranks admitted and agree;
    /// 6. per-rank memory is recorded and the mesh is stored in `self`.
    ///
    /// On `Err` the bundle must be freed, not served (rank 0's forward may
    /// already hold the hook); peers are already torn down.
    pub fn attach_ep_mesh(
        &mut self,
        gpu: &mut Gpu,
        loader: Arc<PeerLoadFn>,
    ) -> Result<(), BundleError> {
        if self.ep.is_some() {
            return Err(ep_err("the mesh is already attached"));
        }
        let seal = match &self.ep_seal {
            Some(bound) if bound.rank == 0 => bound.seal.clone(),
            Some(bound) => {
                return Err(ep_err(format!(
                    "this bundle is rank {}, not rank 0",
                    bound.rank
                )))
            }
            None => return Err(ep_err("rank 0 is not bound to an EP seal")),
        };
        let ranks = seal.ranks();
        if !(2..=8).contains(&ranks) {
            return Err(ep_err(format!("{ranks} ranks requested; supported: 2..=8")));
        }
        // The sealed planner admitted the physical devices (distinct unless
        // `HIPFIRE_EMULATE_GPUS` aliases logical ranks onto one device).
        let device_ids = seal.physical_devices().to_vec();
        if device_ids[0] != gpu.device_id {
            return Err(ep_err(format!(
                "rank 0 runs on device {}, the seal places it on {}",
                gpu.device_id, device_ids[0]
            )));
        }
        let own = seal
            .placement(0)
            .map_err(|error| ep_err(error.to_string()))?;
        let forward = self
            .execution
            .as_ref()
            .ok_or_else(|| ep_err("rank 0 has no attached forward"))?;
        if forward.ep_placement() != Some(own) {
            return Err(ep_err(format!(
                "rank 0 was loaded with placement {:?}, expected {own:?}",
                forward.ep_placement()
            )));
        }
        let trunk_layers = self.config.num_hidden_layers;
        // The exchange geometry comes from the sealed plan (ranks, stride
        // ownership, hidden size), not the config alone: `row_bytes` must agree
        // with the plan's hidden size or the load refuses before any peer
        // access. `slots` is a per-call runtime capacity the plan cannot
        // express and stays config-derived.
        let schedule = hipfire_runtime::sealed_moe::OwnerRoutedEpSchedule::derive(&seal.plans()[0])
            .map_err(ep_err)?;
        let row_bytes = self.config.hidden_size * std::mem::size_of::<f32>();
        if schedule.rank_count() != ranks || schedule.row_bytes() != row_bytes {
            return Err(ep_err(format!(
                "sealed plan geometry ({} ranks, {} hidden) disagrees with the config ({} ranks, {} hidden)",
                schedule.rank_count(),
                schedule.hidden(),
                ranks,
                self.config.hidden_size
            )));
        }
        let spec = EpRowGatherSpec {
            ranks,
            slots: QWEN4_EP_GATHER_MAX_ROWS * self.config.num_experts_per_tok,
            row_bytes,
        };
        let fingerprint = self.ep_replication_fingerprint();

        let inits: Vec<Box<dyn FnOnce(usize) -> Result<Qwen4EpPeer, String> + Send>> = (1..ranks)
            .map(|_| {
                let loader = loader.clone();
                let config = self.config.clone();
                let fingerprint = fingerprint.clone();
                let seal = seal.clone();
                Box::new(move |rank: usize| {
                    // Before the load: no early return may drop a loaded peer
                    // without its teardown.
                    let placement = seal.placement(rank).map_err(|error| error.to_string())?;
                    let (gpu, mut bundle) = loader(rank)?;
                    align_peer_session(&mut bundle, placement, &fingerprint);
                    let peer = Qwen4EpPeer {
                        rank,
                        gpu,
                        bundle,
                        mirrors: PeerMirrors::default(),
                        gather: None,
                        tile: None,
                        split: None,
                    };
                    let checked = peer
                        .check_seal(&seal)
                        .and_then(|()| peer.check_replica(&config, placement, &fingerprint));
                    match checked {
                        Ok(()) => Ok(peer),
                        Err(error) => {
                            let _ = peer.teardown();
                            Err(error)
                        }
                    }
                }) as Box<dyn FnOnce(usize) -> Result<Qwen4EpPeer, String> + Send>
            })
            .collect();
        let pool =
            RankPool::spawn("hipfire-qwen4-ep", inits, Qwen4EpPeer::teardown).map_err(ep_err)?;

        match self.ep_attach_steps(gpu, pool, spec, trunk_layers, device_ids) {
            Ok(mesh) => {
                self.ep = Some(Box::new(mesh));
                Ok(())
            }
            Err(error) => Err(ep_err(error)),
        }
    }

    /// Steps 2-6 of [`Self::attach_ep_mesh`]. Consumes the pool: on error the
    /// peers are torn down by its drop and rank 0's gather events released.
    fn ep_attach_steps(
        &mut self,
        gpu: &mut Gpu,
        pool: RankPool<Qwen4EpPeer>,
        spec: EpRowGatherSpec,
        trunk_layers: usize,
        device_ids: Vec<i32>,
    ) -> Result<Qwen4EpMesh, String> {
        let ranks = spec.ranks;
        // 2. Register every rank's gather buffers, build the transport.
        let kernel_boundaries = qwen4_ep_kernel_boundaries()?;
        let builder = Mutex::new(Some(
            EpRowGather::builder(spec).kernel_boundaries(kernel_boundaries),
        ));
        let register = |rank: usize, bundle: &Qwen4Bundle, gpu: &mut Gpu| -> Result<(), String> {
            let buffers = bundle
                .execution
                .as_ref()
                .and_then(|forward| forward.ep_gather_buffers())
                .ok_or("no EP gather buffers (the forward has no EP placement)")?;
            let refs: [&GpuTensor; QWEN4_EP_GATHER_BUFFERS] = buffers;
            lock(&builder)
                .as_mut()
                .ok_or("gather builder already consumed")?
                .register(rank, gpu, &refs)
                .map_err(|error| format!("register gather buffers: {error}"))
        };
        {
            let me = &*self;
            let gpu0 = &mut *gpu;
            pool.run_all(|| register(0, me, gpu0), &|rank, peer: &mut Qwen4EpPeer| {
                register(rank, &peer.bundle, &mut peer.gpu)
            })?;
        }
        let gather = lock(&builder)
            .take()
            .ok_or("gather builder already consumed")?
            .build()
            .map_err(|error| format!("build gather transport: {error}"))?;
        let abort = gather.clone();
        pool.set_abort(move || abort.abort());
        let tile_slot: Mutex<Option<Arc<EpTileExchange>>> = Mutex::new(None);
        let split_slot: Mutex<Option<TrunkSplitRank>> = Mutex::new(None);
        let split_transport: Mutex<Option<Arc<EpRowGather>>> = Mutex::new(None);
        let release_rank0 = |gpu: &mut Gpu, error: String| -> String {
            gather.abort();
            let _ = gather.release(0, gpu);
            if let Some(tile) = lock(&tile_slot).as_ref() {
                tile.abort();
                let _ = tile.release(0, gpu);
            }
            if let Some(transport) = lock(&split_transport).as_ref() {
                transport.abort();
            }
            if let Some(split) = lock(&split_slot).take() {
                let _ = split.release(gpu);
            }
            error
        };

        // 2b. Owner-routed chunk MoE (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`): every rank that
        // admitted it at construction registers its grouped-row buffer (allocated before
        // P2P is enabled below) with one transport.
        let capacities = {
            let me = &*self;
            pool.run_all(|| Ok(tile_capacity(me)), &|_, peer: &mut Qwen4EpPeer| {
                Ok(tile_capacity(&peer.bundle))
            })
        };
        let capacities = match capacities {
            Ok(capacities) => capacities,
            Err(error) => return Err(release_rank0(gpu, error)),
        };
        let admitted = capacities.iter().filter(|c| c.is_some()).count();
        if admitted != 0 && admitted != ranks {
            eprintln!(
                "qwen4 EP: owner-routed prefill admitted on {admitted}/{ranks} ranks only; chunks keep the full-table route"
            );
        }
        if admitted == ranks {
            let tile_spec = EpTileExchangeSpec {
                ranks,
                experts: self.config.num_experts,
                row_u4: self.config.hidden_size * std::mem::size_of::<u16>() / 16,
                max_tiles: capacities.iter().flatten().copied().min().unwrap_or(0),
            };
            let tile_builder = Mutex::new(Some(EpTileExchange::builder(tile_spec)));
            let register_tile =
                |rank: usize, bundle: &Qwen4Bundle, gpu: &mut Gpu| -> Result<(), String> {
                    let (buffer, _) = bundle
                        .execution
                        .as_ref()
                        .and_then(|forward| forward.ep_tile_registration())
                        .ok_or("no EP tile registration")?;
                    lock(&tile_builder)
                        .as_mut()
                        .ok_or("tile exchange builder already consumed")?
                        .register(rank, gpu, buffer)
                        .map_err(|error| format!("register tile exchange buffers: {error}"))
                };
            let registered = {
                let me = &*self;
                let gpu0 = &mut *gpu;
                pool.run_all(
                    || register_tile(0, me, gpu0),
                    &|rank, peer: &mut Qwen4EpPeer| {
                        register_tile(rank, &peer.bundle, &mut peer.gpu)
                    },
                )
            };
            if let Err(error) = registered {
                return Err(release_rank0(gpu, error));
            }
            let built = lock(&tile_builder)
                .take()
                .ok_or_else(|| "tile exchange builder already consumed".to_string())
                .and_then(|builder| {
                    builder
                        .build()
                        .map_err(|error| format!("build tile exchange transport: {error}"))
                });
            match built {
                Ok(tile) => {
                    let abort = tile.clone();
                    let gather_abort = gather.clone();
                    pool.set_abort(move || {
                        gather_abort.abort();
                        abort.abort();
                    });
                    *lock(&tile_slot) = Some(tile);
                }
                Err(error) => return Err(release_rank0(gpu, error.to_string())),
            }
        }

        // 2c. Trunk output split (`HIPFIRE_EP_TRUNK_SPLIT`, #862): every rank's census of
        // banded weights and its chunk rows must agree with rank 0's; each rank then registers
        // two exchange buffers (allocated before P2P is enabled below) with one band transport.
        let split_mode = match TrunkSplitMode::from_env() {
            Ok(mode) => mode,
            Err(error) => return Err(release_rank0(gpu, error)),
        };
        // The band exchange crosses on the host barrier and its buffers are not
        // in the boundary ledger, so the two are not mixed.
        if split_mode.is_some() && kernel_boundaries {
            return Err(release_rank0(
                gpu,
                format!("{QWEN4_EP_BOUNDARY_ENV}=kernel does not cover HIPFIRE_EP_TRUNK_SPLIT"),
            ));
        }
        if split_mode.is_some() {
            let layout = |bundle: &Qwen4Bundle| -> Result<(Vec<SplitWeight>, usize), String> {
                let census = split_census(bundle)?;
                let rows = bundle
                    .execution
                    .as_ref()
                    .ok_or("the forward is not attached")?
                    .prompt_chunk_rows();
                Ok((census.into_iter().map(|(weight, _)| weight).collect(), rows))
            };
            let layouts = {
                let me = &*self;
                pool.run_all(|| layout(me), &|_, peer: &mut Qwen4EpPeer| {
                    layout(&peer.bundle)
                })
            }
            .and_then(|layouts| agree_layouts(&layouts).map(|()| layouts));
            let layouts = match layouts {
                Ok(layouts) => layouts,
                Err(error) => return Err(release_rank0(gpu, error)),
            };
            let (census, chunk_rows) = &layouts[0];
            let bytes = split_buffer_bytes(census, ranks, *chunk_rows).next_multiple_of(16);
            let split_builder = Mutex::new(Some(EpRowGather::builder(EpRowGatherSpec {
                ranks,
                slots: 1,
                row_bytes: bytes,
            })));
            let register_split = |rank: usize, gpu: &mut Gpu| -> Result<TrunkSplitRank, String> {
                let split = TrunkSplitRank::allocate(rank, gpu, bytes)?;
                let registered = match lock(&split_builder).as_mut() {
                    Some(builder) => builder
                        .register(rank, gpu, &split.buffers())
                        .map_err(|error| format!("register trunk split buffers: {error}")),
                    None => Err("trunk split builder already consumed".to_string()),
                };
                match registered {
                    Ok(()) => Ok(split),
                    Err(error) => {
                        let _ = split.release(gpu);
                        Err(error)
                    }
                }
            };
            let registered = {
                let gpu0 = &mut *gpu;
                pool.run_all(
                    || {
                        *lock(&split_slot) = Some(register_split(0, gpu0)?);
                        Ok(())
                    },
                    &|rank, peer: &mut Qwen4EpPeer| {
                        peer.split = Some(register_split(rank, &mut peer.gpu)?);
                        Ok(())
                    },
                )
            };
            if let Err(error) = registered {
                return Err(release_rank0(gpu, error));
            }
            let built = lock(&split_builder)
                .take()
                .ok_or_else(|| "trunk split builder already consumed".to_string())
                .and_then(|builder| {
                    builder
                        .build()
                        .map_err(|error| format!("build trunk split transport: {error}"))
                });
            match built {
                Ok(transport) => {
                    let aborts = (gather.clone(), lock(&tile_slot).clone(), transport.clone());
                    pool.set_abort(move || {
                        aborts.0.abort();
                        if let Some(tile) = aborts.1.as_ref() {
                            tile.abort();
                        }
                        aborts.2.abort();
                    });
                    *lock(&split_transport) = Some(transport);
                }
                Err(error) => return Err(release_rank0(gpu, error)),
            }
            eprintln!(
                "qwen4 EP: trunk output split {split_mode:?} ({} weights banded over {ranks} \
                 ranks, 2 x {:.1} MiB exchange buffers per rank)",
                census.len(),
                bytes as f64 / (1 << 20) as f64
            );
        }

        // 3. P2P after every load/allocation exists.
        let tile = lock(&tile_slot).clone();
        let split_transport = lock(&split_transport).clone();
        let enable = |rank: usize, gpu: &mut Gpu| -> Result<(), String> {
            gather
                .enable_peers(rank, gpu)
                .map_err(|error| format!("enable peer access: {error}"))?;
            if let Some(tile) = tile.as_ref() {
                tile.enable_peers(rank, gpu)
                    .map_err(|error| format!("enable tile exchange peer access: {error}"))?;
            }
            if let Some(transport) = split_transport.as_ref() {
                transport
                    .enable_peers(rank, gpu)
                    .map_err(|error| format!("enable trunk split peer access: {error}"))?;
            }
            Ok(())
        };
        let enabled = {
            let gpu0 = &mut *gpu;
            pool.run_all(|| enable(0, gpu0), &|rank, peer: &mut Qwen4EpPeer| {
                peer.gather = Some(gather.clone());
                peer.tile = tile.clone();
                enable(rank, &mut peer.gpu)
            })
        };
        if let Err(error) = enabled {
            return Err(release_rank0(gpu, error));
        }

        // 3b. Every rank's split hook over its own banded weights.
        if let (Some(mode), Some(transport)) = (split_mode, split_transport.as_ref()) {
            let install = |bundle: &Qwen4Bundle, split: Option<&mut TrunkSplitRank>| {
                let census = split_census(bundle)?;
                split.ok_or("trunk split buffers missing")?.install(
                    mode,
                    &census,
                    transport.clone(),
                );
                Ok::<(), String>(())
            };
            let installed = {
                let me = &*self;
                pool.run_all(
                    || install(me, lock(&split_slot).as_mut()),
                    &|_, peer: &mut Qwen4EpPeer| install(&peer.bundle, peer.split.as_mut()),
                )
            };
            if let Err(error) = installed {
                return Err(release_rank0(gpu, error));
            }
        }

        // 4. Compact expert sources of every rank.
        let sources = {
            let me = &*self;
            pool.run_all(|| rank_sources(me), &|_, peer: &mut Qwen4EpPeer| {
                rank_sources(&peer.bundle)
            })
        };
        let sources: Vec<Vec<Qwen4PeerExpertSource>> = match sources {
            Ok(sources) => sources,
            Err(error) => return Err(release_rank0(gpu, error)),
        };

        // 5. Full tables + hooks.
        let attach = |rank: usize, bundle: &mut Qwen4Bundle, gpu: &mut Gpu| -> Result<(), String> {
            let buffers = bundle
                .execution
                .as_ref()
                .and_then(|forward| forward.ep_gather_buffers())
                .ok_or("no EP gather buffers (the forward has no EP placement)")?;
            let seal = bundle
                .ep_seal()
                .ok_or("the rank is not bound to an EP seal")?;
            let hook = Arc::new(Qwen4EpRowGatherHook::new(
                rank,
                seal,
                trunk_layers,
                gather.clone(),
                buffers.map(|buffer| buffer.buf.as_ptr() as usize),
            )?);
            let mut forward = bundle
                .execution
                .take()
                .ok_or("the forward is not attached")?;
            let installed: Arc<dyn MoeRowGather + Send + Sync> = hook.clone();
            let result = forward
                .attach_ep(gpu, bundle, &sources, installed)
                .map_err(|error| format!("attach EP tables: {error}"))
                .and_then(|()| match forward.balanced_plan() {
                    // Default on (`HIPFIRE_EP_BALANCED_OWNERS=0` disables) with a
                    // host-mapped layer on gfx1201 at 2..=4 ranks; other arches and
                    // more ranks run the exact static split (`attach_ep` builds no
                    // balanced layers).
                    Some(plan) => hook.install_balanced(plan),
                    None => Ok(()),
                });
            bundle.execution = Some(forward);
            result
        };
        let attached = {
            let me = &mut *self;
            let gpu0 = &mut *gpu;
            pool.run_all(|| attach(0, me, gpu0), &|rank, peer: &mut Qwen4EpPeer| {
                attach(rank, &mut peer.bundle, &mut peer.gpu)
            })
        };
        if let Err(error) = attached {
            return Err(release_rank0(gpu, error));
        }

        // 5b. Owner-routed chunk MoE (EP2..EP8): every rank's per-layer verdicts (symmetric IU4
        // headers on gfx1201, QT44/QT53 formats on gfx1100's F16 WMMA route; taken on the full
        // tables in `attach_ep`) must be identical and all-true on ALL ranks; only
        // then does every rank propagate them to its compact layers and install the exchange.
        // Any mismatch refuses the lever for this load: never a per-rank route divergence.
        if let Some(tile) = tile.as_ref() {
            let verdicts = {
                let me = &*self;
                pool.run_all(|| Ok(tile_verdicts(me)), &|_, peer: &mut Qwen4EpPeer| {
                    Ok(tile_verdicts(&peer.bundle))
                })
            };
            let verdicts = match verdicts {
                Ok(verdicts) => verdicts,
                Err(error) => return Err(release_rank0(gpu, error)),
            };
            let agreed = agreed_tile_verdicts(&verdicts);
            let layer_count = agreed.map_or(0, Vec::len);
            if agreed.is_some() {
                let experts = self.config.num_experts;
                let install = |rank: usize, bundle: &mut Qwen4Bundle| -> Result<(), String> {
                    let mut forward = bundle
                        .execution
                        .take()
                        .ok_or("the forward is not attached")?;
                    let result = (|| {
                        let table = forward
                            .ep_tile_owner_table_view()
                            .ok_or("the forward has no owner table")?;
                        let hook: Arc<dyn MoeTileExchange + Send + Sync> = Arc::new(
                            Qwen4EpTileExchangeHook::new(rank, tile.clone(), table, experts),
                        );
                        forward
                            .install_tile_exchange(hook)
                            .map_err(|error| format!("install tile exchange: {error}"))
                    })();
                    bundle.execution = Some(forward);
                    result
                };
                let installed = {
                    let me = &mut *self;
                    pool.run_all(|| install(0, me), &|rank, peer: &mut Qwen4EpPeer| {
                        install(rank, &mut peer.bundle)
                    })
                };
                if let Err(error) = installed {
                    return Err(release_rank0(gpu, error));
                }
                eprintln!(
                    "qwen4 EP: owner-routed prefill on ({} MoE layers admitted on every rank, \
                     owners e % {ranks})",
                    layer_count
                );
            } else {
                let counts: Vec<String> = verdicts
                    .iter()
                    .map(|layers| {
                        layers.as_ref().map_or("n/a".to_string(), |layers| {
                            format!("{}/{}", layers.iter().filter(|&&s| s).count(), layers.len())
                        })
                    })
                    .collect();
                eprintln!(
                    "qwen4 EP: owner-routed prefill refused (per-rank admitted layers {counts:?} \
                     are not identical and complete); chunks keep the full-table route"
                );
            }
        }

        // 5c. EP reserve sharing: every rank's plan receipt (source generation, logical plan
        // digest, resident/donor boundaries, prompt and exchange capacity, S1/S2 bits) must
        // be identical, and all ranks must be reserve-sharing or none. Any disagreement
        // refuses publication: no rank falls back to the ordinary path after the reduced
        // reserve was spent, and no rank may skip a restore rendezvous the others take.
        let receipts = {
            let me = &*self;
            pool.run_all(
                || Ok(rank_reserve_receipt(me)),
                &|_, peer: &mut Qwen4EpPeer| Ok(rank_reserve_receipt(&peer.bundle)),
            )
        };
        let reserve = match receipts.and_then(|receipts| agree_reserve_receipts(&receipts)) {
            Ok(reserve) => reserve,
            Err(error) => return Err(release_rank0(gpu, error)),
        };

        // 6. Per-rank device memory in use.
        let used = |gpu: &Gpu| -> Result<u64, String> {
            let (free, total) = gpu
                .device_mem_info()
                .map_err(|error| format!("device memory query: {error}"))?;
            Ok(total.saturating_sub(free) as u64)
        };
        let used_bytes = {
            let gpu0 = &*gpu;
            pool.run_all(|| used(gpu0), &|_, peer: &mut Qwen4EpPeer| used(&peer.gpu))
        };
        let used_bytes = match used_bytes {
            Ok(used_bytes) => used_bytes,
            Err(error) => return Err(release_rank0(gpu, error)),
        };
        debug_assert_eq!(used_bytes.len(), ranks);
        let split = lock(&split_slot).take();

        // 7. The mesh is serviceable: every rank prints its header/rank receipt records once
        // (`HIPFIRE_EP_RESERVE_RECEIPT=1`; host-only, a no-op otherwise). A reserve load prints
        // its plan, an ordinary load its real resident-layer count.
        {
            let me = &*self;
            let logged: Result<Vec<()>, String> = pool.run_all(
                || {
                    me.ep_receipt_log_rank();
                    Ok(())
                },
                &|_, peer: &mut Qwen4EpPeer| {
                    peer.bundle.ep_receipt_log_rank();
                    Ok(())
                },
            );
            if let Err(error) = logged {
                return Err(release_rank0(gpu, error));
            }
        }
        Ok(Qwen4EpMesh {
            pool,
            gather,
            tile,
            split,
            reserve,
            ranks,
            device_ids,
            used_bytes,
        })
    }

    /// Fan a call out to every rank (see the module docs). Without EP this is
    /// exactly `local`. The mesh is detached from `self` during rank 0's own
    /// call so nested bundle calls run locally, and restored afterwards.
    /// Host results must agree on every rank; rank 0's is returned.
    pub(crate) fn ep_fanout<R: Send + PartialEq + std::fmt::Debug>(
        &mut self,
        what: &str,
        local: impl FnOnce(&mut Self) -> Result<R, BundleError>,
        job: PeerJob<'_, R>,
    ) -> Result<R, BundleError> {
        let Some(mesh) = self.ep.take() else {
            return local(self);
        };
        let result = mesh.run(|| local(&mut *self).map_err(|error| error.to_string()), job);
        self.ep = Some(mesh);
        result
            .and_then(|values| agree(what, values))
            .map_err(BundleError::Forward)
    }

    /// Fan a native-MTP drafter call out to every rank. `lead` runs rank 0's
    /// `leader` drafter with this bundle (mesh detached, so nested calls run
    /// locally) on the caller's thread and `gpu`; `follow` runs on every peer
    /// with that peer's drafter replica (created from rank 0's construction
    /// parameters on the first call) and its own bundle. Before `follow`, the
    /// replica replays the target-less calls rank 0 ran since the last fan-out
    /// (`configure_request`, `mtp_reset`, in order) and mirrors rank 0's n-gram
    /// arming. The host results must agree on every rank; rank 0's is returned.
    /// Without a mesh this is exactly `lead`. Neither closure may poll an
    /// abort: no rank may stop alone.
    pub(crate) fn ep_drafter_fanout<R: Send + PartialEq + std::fmt::Debug>(
        &mut self,
        what: &str,
        gpu: &mut Gpu,
        leader: &mut crate::mtp_spec::Qwen4MtpDrafter,
        lead: impl FnOnce(
            &mut crate::mtp_spec::Qwen4MtpDrafter,
            &mut Qwen4Bundle,
            &mut Gpu,
        ) -> Result<R, String>,
        follow: &(dyn Fn(
            &mut crate::mtp_spec::Qwen4MtpDrafter,
            &mut Qwen4Bundle,
            &mut Gpu,
        ) -> Result<R, String>
              + Sync),
    ) -> Result<R, String> {
        let Some(mesh) = self.ep.take() else {
            return lead(leader, self, gpu);
        };
        let params = leader.replica_params();
        let ngram_active = leader.ngram_active();
        let owed = leader.pending_replay().to_vec();
        let result = mesh.run(
            || lead(&mut *leader, &mut *self, &mut *gpu),
            &|bundle, peer_gpu, mirrors| {
                let replica = mirrors.drafter(params);
                replica.replay(peer_gpu, &owed)?;
                replica.mirror_ngram_active(ngram_active);
                follow(replica, bundle, peer_gpu)
            },
        );
        self.ep = Some(mesh);
        let agreed = result.and_then(|values| agree(what, values));
        if agreed.is_ok() {
            leader.clear_replay();
        }
        agreed
    }

    /// Refuse an entry point that has no EP replication (MTP/spec, state
    /// snapshots, wave prefill) instead of running it on rank 0 alone and
    /// stalling the gather barrier.
    pub(crate) fn ep_refuse(&self, what: &str) -> Result<(), BundleError> {
        self.pp_refuse(what)?;
        if self.ep.is_some() {
            return Err(BundleError::Forward(format!(
                "{what} is unsupported under qwen4 EP"
            )));
        }
        Ok(())
    }

    /// Refuse an entry point that has no pipeline-parallel form. Entry points
    /// replicated across an EP mesh (fn-lanes) use this instead of
    /// [`Self::ep_refuse`].
    pub(crate) fn pp_refuse(&self, what: &str) -> Result<(), BundleError> {
        if self.pp.is_some() {
            return Err(BundleError::Forward(format!(
                "{what} is not supported under qwen4 topology.pp yet"
            )));
        }
        Ok(())
    }

    /// Stop and free the mesh (before the bundle itself is freed).
    pub(crate) fn ep_close(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        match self.ep.take() {
            Some(mesh) => mesh.close(gpu).map_err(ep_err),
            None => Ok(()),
        }
    }

    /// Per-rank body of a whole-prompt entry: this rank's prompt nesting is
    /// opened before `body` and closed after it, so every rank restores the
    /// borrowed reserve exactly once, at the outer completion
    /// ([`bracket_prompt`]). Nested entries inside `body` (`prefill_final_from`,
    /// `spec_advance_from`, the forward's own tiling) only deepen the nesting.
    fn ep_prompt_rank<R>(
        &mut self,
        gpu: &mut Gpu,
        body: impl FnOnce(&mut Self, &mut Gpu) -> Result<R, BundleError>,
    ) -> Result<R, BundleError> {
        bracket_prompt(
            (self, gpu),
            |(bundle, _)| bundle.share_prompt_begin(),
            |(bundle, gpu)| body(bundle, gpu),
            |(bundle, gpu), ok| bundle.share_prompt_end(gpu, ok),
        )
    }
}

/// The replicated forms of the daemon-facing bundle calls. Each runs the
/// identical call on every rank (rank 0 on the caller's thread with the caller's
/// tensors, peers on their workers with their [`PeerMirrors`]) and requires the
/// host-visible results to agree. They are reached only when the mesh is
/// attached; inside the fan-out every rank's mesh is detached, so the nested
/// public calls run locally.
impl Qwen4Bundle {
    pub(crate) fn ep_forward_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let logits_spec = TensorSpec::of(logits);
        let top1_spec = top1.map(TensorSpec::of);
        self.ep_fanout(
            "forward_token",
            |bundle| bundle.forward_token(gpu, token, logits, top1),
            &|bundle, peer_gpu, mirrors| {
                let (logits, top1) = mirrors.pair(peer_gpu, &logits_spec, top1_spec.as_ref())?;
                bundle
                    .forward_token(peer_gpu, token, logits, top1)
                    .map_err(|error| error.to_string())
            },
        )
    }

    pub(crate) fn ep_forward_token_or_argmax(
        &mut self,
        gpu: &mut Gpu,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        let logits_spec = TensorSpec::of(logits);
        self.ep_fanout(
            "forward_token_or_argmax token",
            |bundle| bundle.forward_token_or_argmax(gpu, token, logits),
            &|bundle, peer_gpu, mirrors| {
                let (logits, _) = mirrors.pair(peer_gpu, &logits_spec, None)?;
                bundle
                    .forward_token_or_argmax(peer_gpu, token, logits)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// `forward_chunk` (`final_row_only == false`) / `forward_chunk_final`.
    pub(crate) fn ep_forward_chunk(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        final_row_only: bool,
    ) -> Result<(), BundleError> {
        let logits_spec = TensorSpec::of(logits);
        let top1_spec = top1.map(TensorSpec::of);
        let run = |bundle: &mut Qwen4Bundle,
                   gpu: &mut Gpu,
                   logits: &GpuTensor,
                   top1: Option<&GpuTensor>| {
            // A direct whole-call wrapper: the forward's internal tiles (and a
            // 4096 + tail split) must not restore between forwards.
            bundle.ep_prompt_rank(gpu, |bundle, gpu| {
                if final_row_only {
                    bundle.forward_chunk_final(gpu, tokens, logits, top1)
                } else {
                    bundle.forward_chunk(gpu, tokens, logits, top1)
                }
            })
        };
        self.ep_fanout(
            "forward_chunk",
            |bundle| run(bundle, gpu, logits, top1),
            &|bundle, peer_gpu, mirrors| {
                let (logits, top1) = mirrors.pair(peer_gpu, &logits_spec, top1_spec.as_ref())?;
                run(bundle, peer_gpu, logits, top1).map_err(|error| error.to_string())
            },
        )
    }

    pub(crate) fn ep_prefill_final(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        let logits_spec = TensorSpec::of(logits);
        self.ep_fanout(
            "prefill_final",
            |bundle| {
                bundle.ep_prompt_rank(gpu, |bundle, gpu| {
                    bundle.prefill_final(gpu, prompt, reused, logits)
                })
            },
            &|bundle, peer_gpu, mirrors| {
                let (logits, _) = mirrors.pair(peer_gpu, &logits_spec, None)?;
                bundle
                    .ep_prompt_rank(peer_gpu, |bundle, gpu| {
                        bundle.prefill_final(gpu, prompt, reused, logits)
                    })
                    .map_err(|error| error.to_string())
            },
        )
    }

    pub(crate) fn ep_apply_penalty_table(
        &mut self,
        gpu: &mut Gpu,
        logits: &GpuTensor,
        table: &PenaltyTable,
    ) -> Result<(), BundleError> {
        let logits_spec = TensorSpec::of(logits);
        self.ep_fanout(
            "apply_penalty_table",
            |bundle| bundle.apply_penalty_table(gpu, logits, table),
            &|bundle, peer_gpu, mirrors| {
                let (logits, _) = mirrors.pair(peer_gpu, &logits_spec, None)?;
                bundle
                    .apply_penalty_table(peer_gpu, logits, table)
                    .map_err(|error| error.to_string())
            },
        )
    }

    pub(crate) fn ep_reset(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        self.ep_fanout(
            "reset",
            |bundle| bundle.reset(gpu),
            &|bundle, peer_gpu, _| bundle.reset(peer_gpu).map_err(|error| error.to_string()),
        )
    }

    pub(crate) fn ep_session_clear(&mut self, gpu: &mut Gpu) {
        let cleared = self.ep_fanout(
            "session_clear",
            |bundle| {
                bundle.session_clear(gpu);
                Ok(())
            },
            &|bundle, peer_gpu, _| {
                bundle.session_clear(peer_gpu);
                Ok(())
            },
        );
        if let Err(error) = cleared {
            eprintln!("  qwen4 EP: session_clear: {error}");
        }
    }

    /// `ArchModel::session_commit` (`consumed == None`) /
    /// `session_commit_live(consumed)` on every rank (host state only).
    pub(crate) fn ep_session_commit(&mut self, consumed: Option<&[u32]>) {
        let commit = |bundle: &mut Qwen4Bundle| {
            match consumed {
                Some(consumed) => ArchModel::session_commit_live(bundle, consumed),
                None => ArchModel::session_commit(bundle),
            }
            Ok(())
        };
        if let Err(error) = self.ep_fanout("session_commit", commit, &|bundle, _, _| {
            match consumed {
                Some(consumed) => ArchModel::session_commit_live(bundle, consumed),
                None => ArchModel::session_commit(bundle),
            }
            Ok(())
        }) {
            eprintln!("  qwen4 EP: session_commit: {error}");
        }
    }

    /// The reuse point every rank can serve: each rank plans over the prompt
    /// prefix `cap` allows, and `cap` drops to the lowest answer until all
    /// ranks agree (each rank's cache evicts independently under memory
    /// pressure, so they may hold different snapshots). A mesh failure plans 0;
    /// the following replicated call reports the poisoning.
    pub(crate) fn ep_session_plan(&self, prompt: &[u32], route: SessionRoute) -> usize {
        let Some(mesh) = self.ep.as_ref() else {
            return self.session_plan_capped(prompt, route, prompt.len());
        };
        let mut cap = prompt.len();
        loop {
            let plans = mesh.run(
                || Ok(self.session_plan_capped(prompt, route, cap)),
                &|bundle, _, _| Ok(bundle.session_plan_capped(prompt, route, cap)),
            );
            let Ok(plans) = plans else {
                return 0;
            };
            let lowest = plans.iter().copied().min().unwrap_or(0);
            if plans.iter().all(|&plan| plan == lowest) {
                return lowest;
            }
            cap = lowest;
        }
    }
}

/// Run a whole-prompt `body` between `begin` and `end` over one rank's state
/// `state`. `end` always runs (`ok` says whether `body` succeeded) so the
/// nesting is balanced on every path; the body's error wins over `end`'s.
/// The rank-local half of the EP reserve protocol: the outermost `end` is the
/// one that restores the borrowed donors and crosses the mesh rendezvous.
pub(crate) fn bracket_prompt<S, R, E>(
    mut state: S,
    begin: impl FnOnce(&mut S),
    body: impl FnOnce(&mut S) -> Result<R, E>,
    end: impl FnOnce(&mut S, bool) -> Result<(), E>,
) -> Result<R, E> {
    begin(&mut state);
    let result = body(&mut state);
    let finish = end(&mut state, result.is_ok());
    result.and_then(|value| finish.map(|()| value))
}

/// This rank's EP reserve-sharing plan receipt, `None` when its forward is not
/// reserve-sharing (or not attached).
fn rank_reserve_receipt(bundle: &Qwen4Bundle) -> Option<EpReserveMeshReceipt> {
    bundle
        .execution
        .as_ref()
        .and_then(|forward| forward.ep_reserve_plan_receipt())
}

/// Compare every rank's reserve receipt (rank order). All `None` is an ordinary
/// load (`Ok(None)`); all `Some` and identical is the agreed receipt. A mixed
/// set, or any differing field, is refused: the error names the ranks.
pub(crate) fn agree_reserve_receipts<T: PartialEq + Clone + std::fmt::Debug>(
    receipts: &[Option<T>],
) -> Result<Option<T>, String> {
    let Some(first) = receipts.first() else {
        return Err("EP reserve receipts: no rank reported".to_string());
    };
    for (rank, other) in receipts.iter().enumerate().skip(1) {
        if other != first {
            return Err(format!(
                "EP reserve receipt disagrees across ranks: rank 0 = {first:?}, rank {rank} = \
                 {other:?}; publication refused (no rank may serve the ordinary path after the \
                 reduced reserve was spent)"
            ));
        }
    }
    Ok(first.clone())
}

// ───────────────────────────────── fn-lanes ─────────────────────────────────

use crate::lane::{LaneSpecConfig, LaneStoreReceipt, Qwen4LaneStore};
use crate::lane_exec::{LaneDraft, Qwen4LaneExecutor};
use hipfire_runtime::slot_batch::{BatchStepPlan, RequestAdvance, RequestEpoch, StepOutput};

/// Free device bytes a rank keeps beyond its lane store when EP stages lanes.
/// The single-device loader (`batch_staging::stage_qwen4_lanes`) compares the
/// store with the free device memory directly, with no reserve, and EP stages
/// by the same rule; a loader that reserves headroom passes its own value to
/// [`ep_lane_budget`] / [`ep_lane_fit`].
pub const QWEN4_EP_LANE_RESERVE_BYTES: u64 = 0;

/// Device bytes a rank may commit to its lane store: its own `free` memory
/// minus `reserve` (saturating).
pub fn ep_lane_budget(free: u64, reserve: u64) -> u64 {
    free.saturating_sub(reserve)
}

/// Whether a lane store of `need` bytes fits rank `rank`'s `free` memory
/// after `reserve`; the refusal names the rank, the need and the free bytes.
pub fn ep_lane_fit(rank: usize, need: u64, free: u64, reserve: u64) -> Result<(), String> {
    if need <= ep_lane_budget(free, reserve) {
        return Ok(());
    }
    Err(format!(
        "rank {rank}: lane store needs {need} bytes, {free} bytes free ({reserve} bytes reserved)"
    ))
}

/// Row budget an AR lane store is staged with: the requested step row budget
/// (`serve.max_batch_tokens`), never below one row per lane. The one rule the
/// load-time reservation and the batch staging share.
pub fn ep_lane_row_budget(requested_row_budget: usize, lanes: usize) -> usize {
    requested_row_budget.max(lanes)
}

/// Device bytes one rank's lane store commits, for `lanes` lanes at
/// `row_budget` rows (`spec_k` = native-MTP lanes; EP stages AR lanes only).
/// `max_seq`, `chunk_rows` and `granularity` are the context commit the QSA
/// backend maps ([`crate::kv_backend::Qwen4ContextCommit::new`]). The single
/// source of the byte count: the loader's EP load reservation and
/// `stage_rank_lanes` (and the single-device staging check) all call it, so
/// planned and staged bytes cannot differ. `None` on overflow.
#[allow(clippy::too_many_arguments)]
pub fn ep_lane_store_device_bytes(
    config: &crate::Qwen4Config,
    format: crate::Qwen4StateFormat,
    backend: crate::Qwen4KvBackend,
    max_seq: usize,
    chunk_rows: usize,
    granularity: usize,
    lanes: usize,
    row_budget: usize,
    spec_k: Option<usize>,
) -> Option<u64> {
    let context =
        crate::kv_backend::Qwen4ContextCommit::new(backend, max_seq, chunk_rows, granularity);
    Qwen4LaneStore::device_bytes(config, format, &context, lanes, row_budget, spec_k)
}

/// A lane-entry error as the plain text every rank compares.
fn bundle_text(error: BundleError) -> String {
    match error {
        BundleError::Forward(message) => message,
        other => other.to_string(),
    }
}

/// Run `f` on `bundle`'s lane executor (`Qwen4Bundle::lane_executor`).
fn run_lane_exec<T>(
    bundle: &mut Qwen4Bundle,
    f: impl FnOnce(&mut Qwen4LaneExecutor<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let mut executor = bundle.lane_executor().map_err(bundle_text)?;
    f(&mut executor)
}

/// One rank's lane staging: the loader's budget check against this rank's own
/// free memory, then the singleton `stage_lanes`. `Ok` means this call
/// created the rank's store; `Err` means it created nothing.
fn stage_rank_lanes(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    max_lanes: usize,
    row_budget: usize,
    reserve: u64,
) -> Result<(), String> {
    let rank = bundle
        .weights
        .ep_placement()
        .map_or(0, |placement| placement.rank);
    let max_seq = bundle.state.max_seq_len;
    // The rank's attached forward plan; an EP load always keeps `Full`.
    let mode = bundle
        .forward_scratch_mode()
        .unwrap_or(crate::gpu_forward::Qwen4ForwardScratchMode::Full);
    let chunk = crate::gpu_forward::qwen4_prefill_chunk_requested(gpu, max_seq, mode);
    let need = ep_lane_store_device_bytes(
        &bundle.config,
        bundle.state_format(),
        bundle.state.qsa_backend(),
        max_seq,
        chunk,
        gpu.vmm_recommended_granularity().unwrap_or(1),
        max_lanes,
        row_budget,
        None,
    )
    .ok_or_else(|| format!("rank {rank}: lane store device bytes overflow"))?;
    let free = gpu
        .device_mem_info()
        .map(|(free, _)| free as u64)
        .map_err(|error| format!("rank {rank}: device memory query failed: {error}"))?;
    ep_lane_fit(rank, need, free, reserve)?;
    bundle
        .stage_lanes(gpu, max_lanes, row_budget, None)
        .map_err(|error| format!("rank {rank}: {}", bundle_text(error)))
}

/// Free the lane store a failed stage left on this rank (a no-op when none).
fn rollback_rank_lanes(bundle: &mut Qwen4Bundle, gpu: &mut Gpu) -> Result<(), String> {
    match bundle.lanes.take() {
        Some(store) => store.free_gpu(gpu),
        None => Ok(()),
    }
}

fn rank_lane_receipt(bundle: &Qwen4Bundle, gpu: &Gpu) -> Result<LaneStoreReceipt, String> {
    bundle
        .lanes()
        .map(|store| store.receipt(gpu))
        .ok_or_else(|| "Qwen4 lanes are not staged".to_string())
}

/// The replicated forms of the fn-lanes entries. Every rank owns its own
/// lane store, built from the same calls in the same order, so the host lane
/// tables stay identical; the device states differ per rank only in what the
/// row gather and the expert split make rank-local.
///
/// Two failure disciplines, both requiring agreed host results:
///
/// - [`Self::ep_lane_soft`] (everything but `forward_step`): no rank blocks on
///   the row-gather barrier, so a refusal or error every rank reports
///   identically is returned as an ordinary error and the mesh stays usable.
///   Ranks that disagree (one failed, or different outcomes) have diverged
///   tables or states: the mesh is poisoned.
/// - [`Self::ep_lane_strict`] (`forward_step`): ranks share the gather
///   barrier, so any rank's failure aborts it and poisons the mesh, as every
///   singleton forward does.
impl Qwen4Bundle {
    /// Run `local` (rank 0, mesh detached) and `job` (every peer) and return
    /// every rank's value in rank order without comparing them: only for
    /// calls whose per-rank values legitimately differ (staging, receipts,
    /// rollback). Without a mesh this is `local` alone. `forced` also runs on
    /// a poisoned mesh (rollback drains).
    fn ep_run_ranks<R: Send>(
        &mut self,
        forced: bool,
        local: impl FnOnce(&mut Self) -> Result<R, String>,
        job: PeerJob<'_, R>,
    ) -> Result<Vec<R>, String> {
        let Some(mesh) = self.ep.take() else {
            return local(self).map(|value| vec![value]);
        };
        let result = if forced {
            mesh.run_forced(|| local(&mut *self), job)
        } else {
            mesh.run(|| local(&mut *self), job)
        };
        self.ep = Some(mesh);
        result
    }

    /// Host-only / non-barrier lane call over [`Self::ep_fanout`]: every
    /// rank's whole outcome (value or error) is the fanned-out result, so an
    /// error all ranks agree on comes back as an ordinary `Err` and leaves
    /// the mesh usable. A mesh failure or a disagreement between ranks
    /// (already poisoned or not) leaves the mesh poisoned.
    fn ep_lane_soft<T: Send + PartialEq + std::fmt::Debug>(
        &mut self,
        what: &str,
        local: impl FnOnce(&mut Self) -> Result<T, String>,
        job: PeerJob<'_, T>,
    ) -> Result<T, String> {
        let fanned = self.ep_fanout(
            what,
            |bundle| Ok(local(bundle)),
            &|bundle, peer_gpu, mirrors| Ok(job(bundle, peer_gpu, mirrors)),
        );
        match fanned {
            Ok(outcome) => outcome,
            Err(error) => {
                let message = bundle_text(error);
                if let Some(mesh) = self.ep.as_ref() {
                    mesh.poison(&message);
                }
                Err(message)
            }
        }
    }

    /// Barrier-sharing lane call: [`Self::ep_fanout`] as is (any failure
    /// poisons the mesh).
    fn ep_lane_strict<T: Send + PartialEq + std::fmt::Debug>(
        &mut self,
        what: &str,
        local: impl FnOnce(&mut Self) -> Result<T, String>,
        job: PeerJob<'_, T>,
    ) -> Result<T, String> {
        self.ep_fanout(
            what,
            |bundle| local(bundle).map_err(BundleError::Forward),
            job,
        )
        .map_err(bundle_text)
    }

    /// Stage the AR lane store on every rank, all or nothing. Refused before
    /// any work for a native-MTP spec, or when any rank already holds a store
    /// (rollback frees only stores this call made). Each rank checks the
    /// store against its own free memory ([`ep_lane_fit`], reserve
    /// [`QWEN4_EP_LANE_RESERVE_BYTES`]); if any rank fails, every rank's new
    /// store is freed and the first failure is returned. A failed stage does
    /// not poison the mesh (the singleton route keeps running).
    pub(crate) fn ep_stage_lanes(
        &mut self,
        gpu: &mut Gpu,
        max_lanes: usize,
        row_budget: usize,
        spec: Option<LaneSpecConfig>,
    ) -> Result<(), BundleError> {
        if self.ep_reserve_active() {
            return Err(ep_err(
                "lane staging is refused on an EP reserve-sharing load: lane steps and lane \
                 prefill waves need the resident scratch bank and Full-table routes the borrowed \
                 reserve does not provide",
            ));
        }
        if spec.is_some() {
            return Err(ep_err(
                "native-MTP lanes are not supported yet; stage AR lanes (serve.batch_spec = false)",
            ));
        }
        let reserve = QWEN4_EP_LANE_RESERVE_BYTES;
        let staged = self
            .ep_run_ranks(
                false,
                |bundle| Ok(bundle.lanes.is_some()),
                &|bundle, _, _| Ok(bundle.lanes.is_some()),
            )
            .map_err(BundleError::Forward)?;
        if let Some(rank) = staged.iter().position(|&has_store| has_store) {
            return Err(ep_err(format!(
                "Qwen4 lanes are already staged (rank {rank})"
            )));
        }
        let outcomes = self.ep_run_ranks(
            false,
            |bundle| {
                Ok(stage_rank_lanes(
                    bundle, gpu, max_lanes, row_budget, reserve,
                ))
            },
            &|bundle, peer_gpu, _| {
                Ok(stage_rank_lanes(
                    bundle, peer_gpu, max_lanes, row_budget, reserve,
                ))
            },
        );
        let reason = match outcomes {
            Ok(outcomes) => match outcomes.into_iter().find_map(Result::err) {
                Some(reason) => reason,
                None => return Ok(()),
            },
            Err(pool) => pool,
        };
        let rollback = self.ep_run_ranks(
            true,
            |bundle| rollback_rank_lanes(bundle, gpu),
            &|bundle, peer_gpu, _| rollback_rank_lanes(bundle, peer_gpu),
        );
        let note = match rollback {
            Ok(_) => String::new(),
            Err(error) => format!("; rollback also failed: {error}"),
        };
        Err(ep_err(format!("{reason}{note}")))
    }

    /// Every rank's lane-store receipt in rank order (rank 0 alone without a
    /// mesh), not compared: device receipts (`mapped_bytes`) legitimately
    /// differ per rank. A rank without a store is an error naming it.
    pub fn ep_lane_receipts(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<Vec<LaneStoreReceipt>, BundleError> {
        let receipts = self
            .ep_run_ranks(
                false,
                |bundle| Ok(rank_lane_receipt(bundle, gpu)),
                &|bundle, peer_gpu, _| Ok(rank_lane_receipt(bundle, peer_gpu)),
            )
            .map_err(BundleError::Forward)?;
        receipts
            .into_iter()
            .enumerate()
            .map(|(rank, receipt)| receipt.map_err(|error| ep_err(format!("rank {rank}: {error}"))))
            .collect()
    }

    pub(crate) fn ep_admit_lane(
        &mut self,
        gpu: &mut Gpu,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
    ) -> Result<usize, BundleError> {
        self.ep_lane_soft(
            "admit_lane",
            |bundle| {
                bundle
                    .admit_lane(gpu, epoch, prompt.clone())
                    .map_err(bundle_text)
            },
            &|bundle, peer_gpu, _| {
                bundle
                    .admit_lane(peer_gpu, epoch, prompt.clone())
                    .map_err(bundle_text)
            },
        )
        .map_err(BundleError::Forward)
    }

    /// [`Self::set_lane_stage_policy`] on every rank: the policy is
    /// host-only, so ranks must agree (a store that cannot take it on every
    /// rank refuses with the common message and leaves the mesh usable).
    pub(crate) fn ep_set_lane_stage_policy(
        &mut self,
        policy: crate::gpu_forward_lanes::LaneStagePolicy,
    ) -> Result<(), BundleError> {
        self.ep_lane_soft(
            "set_lane_stage_policy",
            |bundle| bundle.set_lane_stage_policy_local(policy),
            &|bundle, _, _| bundle.set_lane_stage_policy_local(policy),
        )
        .map_err(BundleError::Forward)
    }

    pub(crate) fn ep_release_lane(&mut self, epoch: &RequestEpoch) -> Result<(), BundleError> {
        self.ep_lane_soft(
            "release_lane",
            |bundle| bundle.release_lane(epoch).map_err(bundle_text),
            &|bundle, _, _| bundle.release_lane(epoch).map_err(bundle_text),
        )
        .map_err(BundleError::Forward)
    }

    /// `poison_lane` returns nothing: a mesh failure or disagreement is
    /// logged (the mesh is poisoned and the next replicated call reports it).
    pub(crate) fn ep_poison_lane(&mut self, epoch: &RequestEpoch) {
        let poisoned = self.ep_lane_soft(
            "poison_lane",
            |bundle| {
                bundle.poison_lane(epoch);
                Ok(())
            },
            &|bundle, _, _| {
                bundle.poison_lane(epoch);
                Ok(())
            },
        );
        if let Err(error) = poisoned {
            eprintln!("  qwen4 EP: poison_lane: {error}");
        }
    }

    pub(crate) fn ep_force_lane_seed(
        &mut self,
        epoch: &RequestEpoch,
        token: u32,
    ) -> Result<(), BundleError> {
        self.ep_lane_soft(
            "force_lane_seed",
            |bundle| bundle.force_lane_seed(epoch, token).map_err(bundle_text),
            &|bundle, _, _| bundle.force_lane_seed(epoch, token).map_err(bundle_text),
        )
        .map_err(BundleError::Forward)
    }

    /// [`Self::set_lanes_sampled`] on every rank's store (host only). Returns
    /// nothing: a mesh failure or disagreement is logged (the mesh is
    /// poisoned and the next replicated call reports it).
    pub(crate) fn ep_set_lanes_sampled(&mut self, on: bool) {
        let set = self.ep_lane_soft(
            "set_lanes_sampled",
            |bundle| {
                bundle.set_lanes_sampled(on);
                Ok(())
            },
            &|bundle, _, _| {
                bundle.set_lanes_sampled(on);
                Ok(())
            },
        );
        if let Err(error) = set {
            eprintln!("  qwen4 EP: set_lanes_sampled: {error}");
        }
    }

    /// [`Self::set_lane_sampling`] on every rank: each rank's lane copy gets
    /// the same sampler and the same RNG seed, and draws from its own state
    /// in lockstep, so the ranks' host results (picks, advances) agree as
    /// greedy picks do (`ep_lane_strict` compares them at `forward_step`). A
    /// refusal every rank reports identically leaves the mesh usable.
    pub(crate) fn ep_set_lane_sampling(
        &mut self,
        epoch: &RequestEpoch,
        sampling: crate::lane::LaneSampling,
    ) -> Result<(), BundleError> {
        self.ep_lane_soft(
            "set_lane_sampling",
            |bundle| {
                bundle
                    .set_lane_sampling(epoch, sampling)
                    .map_err(bundle_text)
            },
            &|bundle, _, _| {
                bundle
                    .set_lane_sampling(epoch, sampling)
                    .map_err(bundle_text)
            },
        )
        .map_err(BundleError::Forward)
    }

    pub(crate) fn ep_abort_promotion(&mut self, epoch: &RequestEpoch) -> Result<(), BundleError> {
        self.ep_lane_soft(
            "abort_promotion",
            |bundle| bundle.abort_promotion(epoch).map_err(bundle_text),
            &|bundle, _, _| bundle.abort_promotion(epoch).map_err(bundle_text),
        )
        .map_err(BundleError::Forward)
    }

    pub(crate) fn ep_commit_promotion(
        &mut self,
        gpu: &mut Gpu,
        epoch: &RequestEpoch,
    ) -> Result<(), BundleError> {
        self.ep_lane_soft(
            "commit_promotion",
            |bundle| bundle.commit_promotion(gpu, epoch).map_err(bundle_text),
            &|bundle, peer_gpu, _| {
                bundle
                    .commit_promotion(peer_gpu, epoch)
                    .map_err(bundle_text)
            },
        )
        .map_err(BundleError::Forward)
    }

    pub(crate) fn ep_lane_draft_step(
        &mut self,
        gpu: &mut Gpu,
        requests: &[(RequestEpoch, usize)],
    ) -> Result<Vec<LaneDraft>, String> {
        self.ep_lane_soft(
            "lane draft_step",
            |bundle| run_lane_exec(bundle, |exec| exec.draft_step(gpu, requests)),
            &|bundle, peer_gpu, _| {
                run_lane_exec(bundle, |exec| exec.draft_step(peer_gpu, requests))
            },
        )
    }

    pub(crate) fn ep_lane_provision_step(
        &mut self,
        gpu: &mut Gpu,
        plan: &BatchStepPlan,
    ) -> Result<(), String> {
        self.ep_lane_soft(
            "lane provision_step",
            |bundle| run_lane_exec(bundle, |exec| exec.provision_step(gpu, plan)),
            &|bundle, peer_gpu, _| {
                run_lane_exec(bundle, |exec| exec.provision_step(peer_gpu, plan))
            },
        )
    }

    /// Best effort and void, like the singleton entry: a mesh failure or
    /// disagreement is logged and `provision_step` grows on demand.
    pub(crate) fn ep_lane_map_ahead_decode(
        &mut self,
        gpu: &mut Gpu,
        epoch: &RequestEpoch,
        remaining_new_tokens: usize,
    ) {
        let mapped = self.ep_lane_soft(
            "lane map_ahead_decode",
            |bundle| {
                run_lane_exec(bundle, |exec| {
                    exec.map_ahead_decode(gpu, epoch, remaining_new_tokens);
                    Ok(())
                })
            },
            &|bundle, peer_gpu, _| {
                run_lane_exec(bundle, |exec| {
                    exec.map_ahead_decode(peer_gpu, epoch, remaining_new_tokens);
                    Ok(())
                })
            },
        );
        if let Err(error) = mapped {
            eprintln!("  qwen4 EP: lane map_ahead_decode: {error}");
        }
    }

    pub(crate) fn ep_lane_forward_step(
        &mut self,
        gpu: &mut Gpu,
        plan: &BatchStepPlan,
    ) -> Result<StepOutput, String> {
        self.ep_lane_strict(
            "lane forward_step",
            |bundle| run_lane_exec(bundle, |exec| exec.forward_step(gpu, plan)),
            &|bundle, peer_gpu, _| run_lane_exec(bundle, |exec| exec.forward_step(peer_gpu, plan)),
        )
    }

    pub(crate) fn ep_lane_commit_step(
        &mut self,
        gpu: &mut Gpu,
        plan: &BatchStepPlan,
        output: StepOutput,
    ) -> Result<Vec<RequestAdvance>, String> {
        self.ep_lane_soft(
            "lane commit_step",
            |bundle| run_lane_exec(bundle, |exec| exec.commit_step(gpu, plan, output.clone())),
            &|bundle, peer_gpu, _| {
                run_lane_exec(bundle, |exec| {
                    exec.commit_step(peer_gpu, plan, output.clone())
                })
            },
        )
    }

    /// `abort_step` returns nothing: a mesh failure or disagreement is logged.
    pub(crate) fn ep_lane_abort_step(&mut self, plan: &BatchStepPlan) {
        let aborted = self.ep_lane_soft(
            "lane abort_step",
            |bundle| {
                run_lane_exec(bundle, |exec| {
                    exec.abort_step(plan);
                    Ok(())
                })
            },
            &|bundle, _, _| {
                run_lane_exec(bundle, |exec| {
                    exec.abort_step(plan);
                    Ok(())
                })
            },
        );
        if let Err(error) = aborted {
            eprintln!("  qwen4 EP: lane abort_step: {error}");
        }
    }
}

/// A peer loads without rank 0's tokenizer, so it has no session cache and no
/// `<|im_end|>` id. Every rank plans prefill boundaries and snapshots from the
/// same host configuration, so give the peer a cache of rank 0's budget and
/// rank 0's turn settings. Its cache domain only keys that peer's own cache.
fn align_peer_session(
    bundle: &mut Qwen4Bundle,
    placement: Qwen4EpPlacement,
    rank0: &ReplicaFingerprint,
) {
    if rank0.session_cache && bundle.session_cache().is_none() {
        let tag = format!("rank{}/{}", placement.rank, placement.ranks);
        let digest = sha256_len_prefixed(&[b"qwen4-ep-peer-session-cache", tag.as_bytes()]);
        let domain = CacheDomain {
            model_content_digest: digest.clone(),
            model_load_epoch: 1,
            sidecar_digests: Vec::new(),
            tokenizer: TokenizerIdentity {
                vocab_digest: digest.clone(),
                config_digest: digest.clone(),
            },
            template: TemplateIdentity {
                template_digest: digest,
                normalization_tag: "ep-peer".to_string(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "qwen4".to_string(),
                state_abi_tag: String::new(),
                position_attention_tag: String::new(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: Vec::new(),
                v_stride_bytes: Vec::new(),
                layout_tag: String::new(),
            },
            device: DeviceTopology {
                device_id: format!("ep-{tag}"),
                topology_id: "ep".to_string(),
                allocation_epoch: 1,
            },
            namespace: SharingNamespace {
                domain_id: "default".to_string(),
            },
        };
        bundle.attach_session_cache(SessionCache::new(
            domain,
            hipfire_config::memory::session_cache_bytes(),
        ));
    }
    bundle.set_turn_end_token(rank0.turn_end_token);
    bundle.set_turn_snapshots(rank0.turn_snapshots);
}

fn rank_sources(bundle: &Qwen4Bundle) -> Result<Vec<Qwen4PeerExpertSource>, String> {
    bundle
        .execution
        .as_ref()
        .ok_or("the forward is not attached")?
        .ep_expert_sources(bundle)
        .map_err(|error| format!("expert sources: {error}"))
}

/// Host configuration that must match on every rank for the replicated
/// session/prefill planning to stay in lockstep.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ReplicaFingerprint {
    pub(crate) session_cache: bool,
    pub(crate) chunk_rows: Option<usize>,
    pub(crate) state_format: crate::state::Qwen4StateFormat,
    pub(crate) turn_end_token: Option<u32>,
    pub(crate) turn_snapshots: bool,
    pub(crate) tail_snapshot: bool,
    pub(crate) tail_snapshot_min_gap: usize,
    pub(crate) tail_snapshot_min_prompt: usize,
    pub(crate) has_mtp: bool,
    pub(crate) has_lanes: bool,
}

impl Qwen4EpPeer {
    /// A loaded peer must be bound to rank 0's seal itself (not an equal
    /// one planned from another mesh epoch) at its own rank, on the seal's
    /// physical device for that rank.
    fn check_seal(&self, seal: &Arc<crate::ep_seal::Qwen4EpSeal>) -> Result<(), String> {
        let bound = self
            .bundle
            .ep_seal()
            .ok_or("the loaded peer is not bound to an EP seal")?;
        if !Arc::ptr_eq(&bound.seal, seal) || bound.rank != self.rank {
            return Err(format!(
                "the loaded peer is bound to rank {} of a seal of epoch {}, expected rank {} of epoch {}",
                bound.rank,
                bound.seal.mesh().epoch().as_u64(),
                self.rank,
                seal.mesh().epoch().as_u64()
            ));
        }
        let device = seal.physical_devices()[self.rank];
        if self.gpu.device_id != device {
            return Err(format!(
                "the loaded peer runs on device {}, the seal places rank {} on {device}",
                self.gpu.device_id, self.rank
            ));
        }
        Ok(())
    }

    /// A loaded peer must be the same model as rank 0, at its own placement,
    /// with the same host-side request configuration.
    fn check_replica(
        &self,
        config: &crate::config::Qwen4Config,
        placement: Qwen4EpPlacement,
        fingerprint: &ReplicaFingerprint,
    ) -> Result<(), String> {
        let forward = self
            .bundle
            .execution
            .as_ref()
            .ok_or("the loaded peer has no attached forward")?;
        if forward.ep_placement() != Some(placement) {
            return Err(format!(
                "the loaded peer has placement {:?}, expected {placement:?}",
                forward.ep_placement()
            ));
        }
        if self.bundle.config != *config {
            return Err("the loaded peer's model configuration differs from rank 0".to_string());
        }
        let theirs = self.bundle.ep_replication_fingerprint();
        if theirs != *fingerprint {
            return Err(format!(
                "the loaded peer's request configuration differs from rank 0: {theirs:?} vs {fingerprint:?}"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_specs_compare_by_shape_and_dtype() {
        let a = TensorSpec {
            shape: vec![248_320],
            dtype: DType::F32,
        };
        assert_eq!(a, a.clone());
        assert_ne!(
            a,
            TensorSpec {
                shape: vec![248_320],
                dtype: DType::F16,
            }
        );
        assert_ne!(
            a,
            TensorSpec {
                shape: vec![2, 124_160],
                dtype: DType::F32,
            }
        );
    }

    fn lane_draft(tag: u64, route: crate::lane_exec::LaneRoute) -> crate::lane_exec::LaneDraft {
        crate::lane_exec::LaneDraft {
            epoch: RequestEpoch {
                request_tag: tag,
                owner_generation: 1,
            },
            route,
            drafts: Vec::new(),
        }
    }

    #[test]
    fn lane_results_must_agree_across_ranks() {
        use crate::lane_exec::LaneRoute;
        let ar = || Ok(vec![lane_draft(1, LaneRoute::Ar)]);
        let outcomes: Vec<Result<Vec<_>, String>> = vec![ar(), ar(), ar()];
        assert_eq!(agree("lane draft_step", outcomes).unwrap(), ar());
        // A different route on one rank.
        let error = agree(
            "lane draft_step",
            vec![ar(), Ok(vec![lane_draft(1, LaneRoute::ArRetired)])],
        )
        .unwrap_err();
        assert!(
            error.contains("lane draft_step") && error.contains("rank 1"),
            "{error}"
        );
        // One rank refusing where the others succeed is a disagreement; every
        // rank refusing identically is an agreed refusal.
        let refusal: Result<Vec<_>, String> = Err("no free lane".to_string());
        assert!(agree("lane draft_step", vec![ar(), refusal.clone()]).is_err());
        assert_eq!(
            agree("lane draft_step", vec![refusal.clone(), refusal.clone()]).unwrap(),
            refusal
        );
        // Step output and advances compare field by field.
        let output = |pick: u32| -> Result<StepOutput, String> {
            Ok(StepOutput {
                step_id: 3,
                target_picks: vec![pick],
            })
        };
        assert!(agree("lane forward_step", vec![output(5), output(5)]).is_ok());
        assert!(agree("lane forward_step", vec![output(5), output(6)]).is_err());
        let advance = |position: usize| -> Result<Vec<RequestAdvance>, String> {
            Ok(vec![RequestAdvance {
                epoch: RequestEpoch {
                    request_tag: 1,
                    owner_generation: 1,
                },
                committed_ids: vec![9],
                committed_position: position,
                accepted_drafts: 0,
                verified_rows: 0,
                finish: None,
            }])
        };
        assert!(agree("lane commit_step", vec![advance(4), advance(4)]).is_ok());
        assert!(agree("lane commit_step", vec![advance(4), advance(5)]).is_err());
    }

    #[test]
    fn lane_budget_is_own_free_minus_reserve() {
        assert_eq!(QWEN4_EP_LANE_RESERVE_BYTES, 0);
        assert_eq!(ep_lane_budget(100, 0), 100);
        assert_eq!(ep_lane_budget(100, 30), 70);
        // The reserve never underflows.
        assert_eq!(ep_lane_budget(10, 30), 0);
        // The loader's rule at zero reserve: need <= free fits, one byte over does not.
        assert!(ep_lane_fit(0, 100, 100, 0).is_ok());
        assert!(ep_lane_fit(0, 70, 100, 30).is_ok());
        let error = ep_lane_fit(3, 71, 100, 30).unwrap_err();
        for part in [
            "rank 3",
            "needs 71 bytes",
            "100 bytes free",
            "30 bytes reserved",
        ] {
            assert!(error.contains(part), "{part}: {error}");
        }
        assert!(ep_lane_fit(2, 1, 0, 0).is_err());
    }

    #[test]
    fn ep_lane_reservation_bytes_are_the_stage_bytes() {
        use crate::{Qwen4KvBackend, Qwen4StateFormat};
        let config = crate::config::compact_test_config();
        let format = Qwen4StateFormat::F32;
        let (max_seq, chunk, granularity) = (32768usize, 1536usize, 2usize << 20);
        let context = crate::kv_backend::Qwen4ContextCommit::new(
            Qwen4KvBackend::Vmm,
            max_seq,
            chunk,
            granularity,
        );
        let mut previous = 0u64;
        for lanes in [2usize, 4, 8] {
            let budget = ep_lane_row_budget(4096, lanes);
            let planned = ep_lane_store_device_bytes(
                &config,
                format,
                Qwen4KvBackend::Vmm,
                max_seq,
                chunk,
                granularity,
                lanes,
                budget,
                None,
            );
            assert_eq!(
                planned,
                Qwen4LaneStore::device_bytes(&config, format, &context, lanes, budget, None)
            );
            let planned = planned.unwrap();
            assert!(planned > previous, "lanes={lanes}");
            previous = planned;
        }
        // A request below one row per lane is raised to the lane count.
        assert_eq!(ep_lane_row_budget(2, 8), 8);
        assert_eq!(ep_lane_row_budget(0, 8), 8);
        assert_eq!(ep_lane_row_budget(4096, 8), 4096);
        // Overflow is reported, never wrapped.
        assert_eq!(
            ep_lane_store_device_bytes(
                &config,
                format,
                Qwen4KvBackend::Vmm,
                max_seq,
                chunk,
                granularity,
                usize::MAX,
                usize::MAX,
                None,
            ),
            None
        );
    }
}

#[cfg(test)]
mod tile_verdict_tests {
    use super::agreed_tile_verdicts;

    #[test]
    fn agreement_needs_every_rank_identical_and_complete() {
        for ranks in [2usize, 4] {
            let all = vec![Some(vec![true; 40]); ranks];
            assert_eq!(
                agreed_tile_verdicts(&all).map(Vec::len),
                Some(40),
                "{ranks}"
            );
            // One rank that did not admit the lever refuses it for all.
            for absent in 0..ranks {
                let mut mixed = all.clone();
                mixed[absent] = None;
                assert!(
                    agreed_tile_verdicts(&mixed).is_none(),
                    "{ranks} absent {absent}"
                );
            }
            // One rank with one asymmetric layer, or a different layer count, refuses.
            for odd in 0..ranks {
                let mut asymmetric = all.clone();
                asymmetric[odd].as_mut().unwrap()[7] = false;
                assert!(
                    agreed_tile_verdicts(&asymmetric).is_none(),
                    "{ranks} asym {odd}"
                );
                let mut shorter = all.clone();
                shorter[odd].as_mut().unwrap().pop();
                assert!(
                    agreed_tile_verdicts(&shorter).is_none(),
                    "{ranks} short {odd}"
                );
            }
        }
        assert!(agreed_tile_verdicts(&[]).is_none());
        assert!(agreed_tile_verdicts(&[Some(Vec::new()), Some(Vec::new())]).is_none());
    }
}

/// P6 (EP reserve mesh entries): two-rank fake schedules over a real
/// [`RankPool`] and an abortable fence standing in for the tile hook's host
/// rendezvous. The fake follows DECISION.md §5: the first borrowed tile
/// crosses the fence before any rank writes donors; the outermost prompt
/// completion restores locally and crosses the fence before any rank reads
/// restored (Full/clean) state; an aborted/forced end recovers locally
/// without entering the fence.
#[cfg(test)]
mod reserve_schedule_tests {
    use super::{agree_reserve_receipts, bracket_prompt};
    use hipfire_runtime::rank_pool::{lock, RankPool};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    const RANKS: usize = 2;

    /// Abortable all-rank rendezvous (the existing tile hook's host barrier).
    struct Fence {
        state: Mutex<(usize, u64, bool)>,
        cv: Condvar,
    }

    impl Fence {
        fn cross(&self) -> Result<(), String> {
            let mut s = lock(&self.state);
            if s.2 {
                return Err("fence aborted".to_string());
            }
            s.0 += 1;
            if s.0 == RANKS {
                s.0 = 0;
                s.1 += 1;
                self.cv.notify_all();
                return Ok(());
            }
            let generation = s.1;
            while s.1 == generation && !s.2 {
                s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
            }
            if s.1 == generation {
                Err("fence aborted".to_string())
            } else {
                Ok(())
            }
        }

        fn abort(&self) {
            lock(&self.state).2 = true;
            self.cv.notify_all();
        }
    }

    struct Mesh {
        fence: Fence,
        /// Ranks that have begun acquire (entered the initial fence).
        entered: AtomicUsize,
        resident: Mutex<[bool; RANKS]>,
        restores: Mutex<[usize; RANKS]>,
        fences_restore: Mutex<[usize; RANKS]>,
        violations: Mutex<Vec<String>>,
    }

    impl Mesh {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                fence: Fence {
                    state: Mutex::new((0, 0, false)),
                    cv: Condvar::new(),
                },
                entered: AtomicUsize::new(0),
                resident: Mutex::new([true; RANKS]),
                restores: Mutex::new([0; RANKS]),
                fences_restore: Mutex::new([0; RANKS]),
                violations: Mutex::new(Vec::new()),
            })
        }

        /// First borrowed tile: peers must all be past their own last read of
        /// the donor weights before any rank writes them.
        fn acquire(&self, rank: usize) -> Result<(), String> {
            self.entered.fetch_add(1, Ordering::SeqCst);
            self.fence.cross()?;
            if self.entered.load(Ordering::SeqCst) != RANKS {
                lock(&self.violations).push(format!("rank {rank} wrote before all entered"));
            }
            lock(&self.resident)[rank] = false;
            Ok(())
        }

        /// `ep_reserve_require_resident`: a clean epoch is a no-op; otherwise
        /// local restore, then the global fence, then a Full/clean reader may run.
        fn require_resident(&self, rank: usize) -> Result<(), String> {
            if lock(&self.resident)[rank] {
                return Ok(());
            }
            lock(&self.resident)[rank] = true;
            lock(&self.restores)[rank] += 1;
            self.fence.cross()?;
            lock(&self.fences_restore)[rank] += 1;
            if lock(&self.resident).iter().any(|&clean| !clean) {
                lock(&self.violations).push(format!("rank {rank} read before all restored"));
            }
            Ok(())
        }

        /// Failed/aborted end: local recovery only, never the aborted fence.
        fn recover_local(&self, rank: usize) {
            let mut resident = lock(&self.resident);
            if !resident[rank] {
                resident[rank] = true;
                lock(&self.restores)[rank] += 1;
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Plan {
        /// Nested whole-prompt entries per rank (outer wrapper + inner entries).
        levels: usize,
        /// Tiles that need the borrowed bank (0: tiny tail / resident-only).
        borrowed_tiles: usize,
        delay_acquire: [u64; RANKS],
        delay_restore: [u64; RANKS],
        /// `(rank, tile)` that fails.
        fail: Option<(usize, usize)>,
        panic_rank: Option<usize>,
    }

    impl Plan {
        fn ar() -> Self {
            Self {
                levels: 3,
                borrowed_tiles: 2,
                delay_acquire: [0; RANKS],
                delay_restore: [0; RANKS],
                fail: None,
                panic_rank: None,
            }
        }
    }

    fn nested(
        mesh: &Mesh,
        rank: usize,
        depth: &mut usize,
        levels: usize,
        plan: Plan,
    ) -> Result<(), String> {
        bracket_prompt(
            depth,
            |depth| **depth += 1,
            |depth| {
                if levels > 1 {
                    return nested(mesh, rank, depth, levels - 1, plan);
                }
                for tile in 0..plan.borrowed_tiles {
                    if tile == 0 {
                        std::thread::sleep(Duration::from_millis(plan.delay_acquire[rank]));
                        mesh.acquire(rank)?;
                    }
                    if plan.panic_rank == Some(rank) {
                        panic!("rank {rank} panicked in tile {tile}");
                    }
                    if plan.fail == Some((rank, tile)) {
                        return Err(format!("rank {rank} tile {tile} failed"));
                    }
                }
                Ok(())
            },
            |depth, ok| {
                **depth -= 1;
                if **depth != 0 {
                    return Ok(());
                }
                if !ok {
                    mesh.recover_local(rank);
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(plan.delay_restore[rank]));
                mesh.require_resident(rank)
            },
        )
    }

    struct Peer {
        depth: usize,
    }

    fn pool(mesh: &Arc<Mesh>) -> RankPool<Peer> {
        let inits = (1..RANKS)
            .map(|_| {
                Box::new(|_rank: usize| Ok(Peer { depth: 0 }))
                    as Box<dyn FnOnce(usize) -> Result<Peer, String> + Send>
            })
            .collect();
        let pool = RankPool::spawn("p6", inits, |_peer: Peer| Ok(())).expect("spawn");
        let abort = mesh.clone();
        pool.set_abort(move || abort.fence.abort());
        pool
    }

    fn run(pool: &RankPool<Peer>, mesh: &Arc<Mesh>, plan: Plan) -> Result<Vec<()>, String> {
        let mut depth0 = 0usize;
        pool.run_all(
            || nested(mesh, 0, &mut depth0, plan.levels, plan),
            &|rank, peer: &mut Peer| nested(mesh, rank, &mut peer.depth, plan.levels, plan),
        )
    }

    fn assert_clean(mesh: &Mesh) {
        assert_eq!(*lock(&mesh.violations), Vec::<String>::new());
    }

    #[test]
    fn delayed_rank_before_acquire_blocks_every_writer() {
        for slow in 0..RANKS {
            let mesh = Mesh::new();
            let pool = pool(&mesh);
            let mut plan = Plan::ar();
            plan.delay_acquire[slow] = 60;
            run(&pool, &mesh, plan).expect("prompt");
            assert_clean(&mesh);
            assert_eq!(*lock(&mesh.resident), [true; RANKS]);
        }
    }

    #[test]
    fn delayed_rank_before_restore_blocks_every_clean_reader() {
        for slow in 0..RANKS {
            let mesh = Mesh::new();
            let pool = pool(&mesh);
            let mut plan = Plan::ar();
            plan.delay_restore[slow] = 60;
            run(&pool, &mesh, plan).expect("prompt");
            assert_clean(&mesh);
            // Both ranks crossed the restore fence before returning.
            assert_eq!(*lock(&mesh.fences_restore), [1; RANKS]);
        }
    }

    #[test]
    fn nested_entries_restore_once_at_the_outer_completion() {
        let mesh = Mesh::new();
        let pool = pool(&mesh);
        // AR prefill_final over spec/MTP-style nesting: five nested entries.
        let mut plan = Plan::ar();
        plan.levels = 5;
        plan.borrowed_tiles = 4;
        run(&pool, &mesh, plan).expect("prompt");
        assert_clean(&mesh);
        assert_eq!(*lock(&mesh.restores), [1; RANKS]);
        assert_eq!(*lock(&mesh.fences_restore), [1; RANKS]);
        // The next prompt is a fresh epoch: its own acquire and exactly one
        // more restore (depth balanced back to 0 on every rank).
        mesh.entered.store(0, Ordering::SeqCst);
        run(&pool, &mesh, plan).expect("second prompt");
        assert_clean(&mesh);
        assert_eq!(*lock(&mesh.restores), [2; RANKS]);
    }

    #[test]
    fn tiny_tail_and_single_token_completion_are_clean() {
        // A tail that never borrows: no acquire, no fence, resident throughout.
        let mesh = Mesh::new();
        let pool = pool(&mesh);
        let mut plan = Plan::ar();
        plan.borrowed_tiles = 0;
        run(&pool, &mesh, plan).expect("tail");
        assert_eq!(*lock(&mesh.restores), [0; RANKS]);
        assert_eq!(*lock(&mesh.resident), [true; RANKS]);
        // max_tokens = 1: one borrowed tile, restored before the entry returns
        // (not deferred to a second token).
        let mesh = Mesh::new();
        let pool = self::pool(&mesh);
        plan.borrowed_tiles = 1;
        plan.levels = 1;
        run(&pool, &mesh, plan).expect("one token");
        assert_clean(&mesh);
        assert_eq!(*lock(&mesh.resident), [true; RANKS]);
        assert_eq!(*lock(&mesh.restores), [1; RANKS]);
    }

    #[test]
    fn error_aborts_every_waiter_recovers_locally_and_stays_poisoned() {
        for failing in 0..RANKS {
            let mesh = Mesh::new();
            let pool = pool(&mesh);
            let mut plan = Plan::ar();
            plan.fail = Some((failing, 1));
            // The healthy rank would otherwise wait on the restore fence.
            let error = run(&pool, &mesh, plan).expect_err("a rank failed");
            assert!(error.contains("failed"), "{error}");
            // Forced end recovery: every rank is back to resident, but no
            // restore fence was crossed after the failure.
            assert_eq!(*lock(&mesh.resident), [true; RANKS]);
            assert_eq!(lock(&mesh.fences_restore)[failing], 0);
            // Local recovery succeeded; the mesh stays poisoned.
            assert!(pool.poisoned().is_some());
            let again = run(&pool, &mesh, Plan::ar()).expect_err("poisoned");
            assert!(again.contains("poisoned"), "{again}");
            // Forced drains still reach the workers and do not unpoison.
            pool.run_all_forced(|| Ok(()), &|_, _: &mut Peer| Ok(()))
                .expect("forced run");
            assert!(pool.poisoned().is_some());
        }
    }

    #[test]
    fn peer_panic_aborts_the_waiting_rank() {
        let mesh = Mesh::new();
        let pool = pool(&mesh);
        let mut plan = Plan::ar();
        plan.panic_rank = Some(1);
        // Rank 0 reaches the restore... never: rank 1 dies after the acquire
        // fence, so rank 0 must be released by the abort, not hang.
        let error = run(&pool, &mesh, plan).expect_err("rank 1 panicked");
        assert!(error.contains("panicked"), "{error}");
        assert!(pool.poisoned().is_some());
    }

    #[test]
    fn receipts_must_agree_or_all_be_absent() {
        #[derive(Clone, Debug, PartialEq)]
        struct R {
            digest: u64,
            resident: usize,
            chunk: usize,
        }
        let r = |digest, resident| R {
            digest,
            resident,
            chunk: 4096,
        };
        // Ordinary load: nobody reserve-shares.
        assert_eq!(agree_reserve_receipts::<R>(&[None, None]), Ok(None));
        // Identical receipts publish.
        assert_eq!(
            agree_reserve_receipts(&[Some(r(7, 33)), Some(r(7, 33))]),
            Ok(Some(r(7, 33)))
        );
        // Digest, boundary, or any-rank-absent disagreement refuses.
        for receipts in [
            [Some(r(7, 33)), Some(r(8, 33))],
            [Some(r(7, 33)), Some(r(7, 32))],
            [Some(r(7, 33)), None],
            [None, Some(r(7, 33))],
        ] {
            let error = agree_reserve_receipts(&receipts).unwrap_err();
            assert!(
                error.contains("disagrees") && error.contains("rank 1"),
                "{error}"
            );
        }
        assert!(agree_reserve_receipts::<R>(&[]).is_err());
    }
}
